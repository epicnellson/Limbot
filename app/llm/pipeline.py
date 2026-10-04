from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app import metrics
from app.config import Settings
from app.core.circuit import CircuitBreaker
from app.llm.base import LLMProvider, ProviderError, ProviderTimeoutError
from app.llm.gemini import gemini_provider
from app.llm.openai_compat import groq_provider, openai_compatible_provider
from app.llm.types import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class NoProviderAvailable(RuntimeError):
    """Every tier was unconfigured, open, or failed transiently."""

    def __init__(self, message: str, *, attempts: tuple[AttemptOutcome, ...]) -> None:
        super().__init__(message)
        self.attempts = attempts


@dataclass(frozen=True, slots=True)
class AttemptOutcome:
    provider: str
    tier: int
    result: str
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "tier": self.tier,
            "result": self.result,
            "detail": self.detail,
        }


class LLMPipeline:
    """Runs one completion against the provider tiers in order.

    The rules, in one place, because they are the whole point of the module:

    * a tier whose circuit is not accepting requests is skipped without a network call;
    * a transient provider failure trips that tier's circuit and moves to the next tier;
    * a non-transient failure, such as a rejected request or a bad key, propagates immediately,
      since every other tier would fail the same way;
    * a successful completion resets the circuit and returns.
    """

    def __init__(
        self,
        settings: Settings,
        providers: list[LLMProvider] | None = None,
        *,
        breakers: dict[str, CircuitBreaker] | None = None,
    ) -> None:
        self.settings = settings
        self.providers = providers if providers is not None else build_providers(settings)
        self.breakers: dict[str, CircuitBreaker] = breakers or {}
        for provider in self.providers:
            self.breakers.setdefault(
                provider.name,
                CircuitBreaker(
                    name=provider.name,
                    failure_threshold=settings.ai_circuit_failure_threshold,
                    recovery_seconds=settings.ai_circuit_recovery_seconds,
                    half_open_calls=settings.ai_circuit_half_open_calls,
                ),
            )

    @property
    def available(self) -> bool:
        return any(p.configured for p in self.providers)

    def describe(self) -> dict[str, Any]:
        return {
            "providers": [p.describe() for p in self.providers],
            "circuits": [breaker.snapshot().as_dict() for breaker in self.breakers.values()],
            "available": self.available,
        }

    def circuit_state(self) -> dict[str, str]:
        return {name: breaker.state.value for name, breaker in self.breakers.items()}

    @staticmethod
    def _record_fallback(from_provider: str, to_provider: str | None, reason: str) -> None:
        """Count a move to the next tier, but only when there is a tier left to move to.

        The last provider failing is not a fallback: the turn ends instead, so counting it would
        overstate how often the chain rescued a request.
        """
        if to_provider is None:
            return
        metrics.PROVIDER_FALLBACKS.labels(
            from_provider=from_provider, to_provider=to_provider, reason=reason
        ).inc()

    async def complete(self, request: LLMRequest, *, http: httpx.AsyncClient) -> LLMResponse:
        configured = [p for p in self.providers if p.configured]
        if not configured:
            metrics.LLM_CALLS.labels(provider="none", result="no_provider").inc()
            metrics.ALL_PROVIDERS_EXHAUSTED.labels(cause="unconfigured").inc()
            raise NoProviderAvailable(
                "no AI provider is configured: set GROQ_API_KEY, GEMINI_API_KEY, or enable "
                "TIER3 with TIER3_BASE_URL",
                attempts=(),
            )

        attempts: list[AttemptOutcome] = []
        last_error: ProviderError | None = None
        for index, provider in enumerate(configured):
            # The tier the loop moves to if this one cannot answer. It is None on the last
            # provider, where a failure ends the turn instead of falling back to anything.
            next_provider = configured[index + 1].name if index + 1 < len(configured) else None
            breaker = self.breakers[provider.name]
            if not breaker.allow_request():
                attempts.append(
                    AttemptOutcome(
                        provider=provider.name,
                        tier=provider.tier,
                        result="circuit_open",
                        detail=f"circuit is {breaker.state.value}",
                    )
                )
                metrics.LLM_CALLS.labels(provider=provider.name, result="circuit_open").inc()
                self._record_fallback(provider.name, next_provider, "circuit_open")
                continue

            started = time.perf_counter()
            # A hard deadline on top of whatever timeout the provider enforces internally. The
            # httpx timeout bounds a single socket operation; this bounds the whole attempt, so
            # a wedged connection can no longer hold the turn open. Cancelling the underlying
            # request on timeout also frees its pooled connection immediately.
            call = provider.complete(request, http=http)
            if request.timeout_seconds:
                call = asyncio.wait_for(call, timeout=request.timeout_seconds)
            # Annotate the base type up front. mypy infers a variable's type from its first
            # assignment, so without this the ProviderTimeoutError below would pin the inferred
            # type and the ProviderError from the provider's own `except` would be rejected as an
            # incompatible assignment. Every attribute read after this block lives on ProviderError.
            provider_error: ProviderError
            try:
                response = await call
            except TimeoutError:
                provider_error = ProviderTimeoutError(
                    f"{provider.name}/{provider.model} exceeded the "
                    f"{request.timeout_seconds}s completion deadline",
                    provider=provider.name,
                )
            except ProviderError as exc:
                provider_error = exc
            else:
                metrics.LLM_LATENCY.labels(provider=provider.name).observe(
                    time.perf_counter() - started
                )
                metrics.LLM_CALLS.labels(provider=provider.name, result="success").inc()
                breaker.record_success()
                return response

            elapsed = time.perf_counter() - started
            metrics.LLM_LATENCY.labels(provider=provider.name).observe(elapsed)
            if isinstance(provider_error, ProviderTimeoutError):
                result = "timeout"
            elif provider_error.status_code == 429:
                result = "rate_limited"
            else:
                result = "error"
            metrics.LLM_CALLS.labels(provider=provider.name, result=result).inc()
            if not provider_error.transient:
                attempts.append(
                    AttemptOutcome(
                        provider=provider.name,
                        tier=provider.tier,
                        result="fatal",
                        detail=str(provider_error),
                    )
                )
                logger.warning(
                    "provider rejected the request, no fallback attempted",
                    extra={"context": {"provider": provider.name, "error": str(provider_error)}},
                )
                raise provider_error
            breaker.record_failure()
            last_error = provider_error
            attempts.append(
                AttemptOutcome(
                    provider=provider.name,
                    tier=provider.tier,
                    result="transient_error",
                    detail=str(provider_error),
                )
            )
            logger.warning(
                "provider failed transiently, falling back",
                extra={
                    "context": {
                        "provider": provider.name,
                        "error": str(provider_error),
                        "circuit": breaker.state.value,
                    }
                },
            )
            self._record_fallback(provider.name, next_provider, result)
            continue

        metrics.ALL_PROVIDERS_EXHAUSTED.labels(cause="exhausted").inc()
        raise NoProviderAvailable(
            "every AI provider tier failed or was skipped",
            attempts=tuple(attempts),
        ) from last_error


def build_providers(settings: Settings) -> list[LLMProvider]:
    """Instantiate the tiers in fallback order, omitting unconfigured ones."""
    providers: list[LLMProvider] = []
    if settings.groq_api_key_value:
        providers.append(
            groq_provider(
                base_url=settings.groq_base_url,
                model=settings.groq_model,
                api_key=settings.groq_api_key_value,
                timeout_seconds=settings.ai_request_timeout_seconds,
            )
        )
    if settings.gemini_api_key_value:
        providers.append(
            gemini_provider(
                base_url=settings.gemini_base_url,
                model=settings.gemini_model,
                api_key=settings.gemini_api_key_value,
                timeout_seconds=settings.ai_request_timeout_seconds,
            )
        )
    if settings.tier3_enabled and settings.tier3_base_url:
        providers.append(
            openai_compatible_provider(
                base_url=settings.tier3_base_url,
                model=settings.tier3_model,
                api_key=settings.tier3_api_key_value,
                timeout_seconds=settings.ai_request_timeout_seconds,
            )
        )
    return providers


def circuit_summary(pipeline: LLMPipeline) -> dict[str, Any]:
    """Circuit state only, for the readiness probe."""
    return {breaker.name: breaker.snapshot().as_dict() for breaker in pipeline.breakers.values()}


__all__ = [
    "AttemptOutcome",
    "LLMPipeline",
    "NoProviderAvailable",
    "build_providers",
    "circuit_summary",
]
