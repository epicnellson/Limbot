from __future__ import annotations

import abc
from typing import Any, ClassVar

import httpx

from app.llm.types import LLMRequest, LLMResponse


class ProviderError(RuntimeError):
    """Base class for provider failures.

    ``transient`` is the decision the resilience pipeline acts on: transient failures trip the
    circuit and fall through to the next tier, while non-transient failures propagate, because
    a malformed request would fail the same way on every provider and retrying only adds
    latency.
    """

    transient: ClassVar[bool] = True

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retry_after = retry_after


class ProviderConfigError(ProviderError):
    """Credentials or endpoint missing or rejected. Non-transient: the tier stays broken."""

    transient: ClassVar[bool] = False


class ProviderRequestError(ProviderError):
    """The provider rejected the request itself. Non-transient."""

    transient: ClassVar[bool] = False


class ProviderRateLimitError(ProviderError):
    """Throttled by the provider."""


class ProviderServerError(ProviderError):
    """Provider side failure, 5xx."""


class ProviderTimeoutError(ProviderError):
    """Read or connect timeout."""


class ProviderTransportError(ProviderError):
    """Connection reset, DNS failure, and similar."""


class ProviderResponseError(ProviderError):
    """A 2xx response that did not contain a usable completion."""


class LLMProvider(abc.ABC):
    """One LLM endpoint. Implementations translate to and from the shared request types."""

    # Instance attributes, not ClassVar. They vary per provider instance: OpenAICompatibleProvider
    # is built once per endpoint, so Groq is tier 1 and every local runtime is tier 3 from the
    # same class (app/llm/openai_compat.py). Declaring these ClassVar while a subclass assigned them
    # per instance is what mypy rejected. Setting them here rather than leaving each subclass to
    # either a class-level default or an instance assignment also means there is one definition of
    # where an identity comes from, which is what the pipeline's tier ordering reads.
    name: str
    tier: int

    def __init__(self, model: str, *, name: str, tier: int) -> None:
        self.model = model
        self.name = name
        self.tier = tier

    @property
    def configured(self) -> bool:
        return True

    @abc.abstractmethod
    async def complete(self, request: LLMRequest, *, http: httpx.AsyncClient) -> LLMResponse:
        """Run one completion, raising a :class:`ProviderError` on failure."""

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "model": self.model,
            "configured": self.configured,
        }

    def _require_config(self) -> None:
        if not self.configured:
            raise ProviderConfigError(f"{self.name} is not configured", provider=self.name)


def retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def raise_for_status(response: httpx.Response, *, provider: str, model: str) -> None:
    """Translate an error response into the right :class:`ProviderError` subclass."""
    status = response.status_code
    if 200 <= status < 300:
        return
    body = _safe_text(response)
    detail = f"{provider}/{model} returned {status}: {body[:400]}"
    if status in (401, 403):
        raise ProviderConfigError(detail, provider=provider, status_code=status)
    if status == 429:
        raise ProviderRateLimitError(
            detail,
            provider=provider,
            status_code=status,
            retry_after=retry_after_seconds(response),
        )
    if status == 408:
        raise ProviderTimeoutError(detail, provider=provider, status_code=status)
    if status >= 500:
        raise ProviderServerError(detail, provider=provider, status_code=status)
    raise ProviderRequestError(detail, provider=provider, status_code=status)


def wrap_transport_error(exc: Exception, *, provider: str, model: str) -> ProviderError:
    if isinstance(exc, httpx.TimeoutException):
        return ProviderTimeoutError(
            f"{provider}/{model} timed out: {type(exc).__name__}",
            provider=provider,
        )
    return ProviderTransportError(
        f"{provider}/{model} transport failure: {type(exc).__name__}: {exc}",
        provider=provider,
    )


def _safe_text(response: httpx.Response) -> str:
    try:
        return response.text
    except Exception:  # pragma: no cover - defensive
        return "<unreadable body>"
