from __future__ import annotations

import httpx
import pytest
from app.config import Settings
from app.core.circuit import CircuitBreaker, CircuitState
from app.llm.base import (
    LLMProvider,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderServerError,
)
from app.llm.pipeline import LLMPipeline, NoProviderAvailable
from app.llm.types import LLMRequest, LLMResponse
from pydantic import SecretStr

from conftest import counter_value

REQUEST = LLMRequest(messages=())


class StubProvider(LLMProvider):
    """Fails a scripted number of times, then answers."""

    def __init__(
        self,
        name: str,
        tier: int,
        *,
        failures: list[Exception] | None = None,
        configured: bool = True,
    ) -> None:
        super().__init__(f"{name}-model")
        self.name = name
        self.tier = tier
        self.failures = list(failures or [])
        self._configured = configured
        self.calls = 0

    @property
    def configured(self) -> bool:
        return self._configured

    async def complete(self, request: LLMRequest, *, http: httpx.AsyncClient) -> LLMResponse:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return LLMResponse(text=f"answer from {self.name}", provider=self.name)


def make_pipeline(
    providers: list[StubProvider], settings: Settings, *, clock: object | None = None
) -> LLMPipeline:
    clock_arg: dict[str, object] = {} if clock is None else {"clock": clock}
    breakers = {
        provider.name: CircuitBreaker(
            name=provider.name,
            failure_threshold=settings.ai_circuit_failure_threshold,
            recovery_seconds=settings.ai_circuit_recovery_seconds,
            half_open_calls=settings.ai_circuit_half_open_calls,
            **clock_arg,
        )
        for provider in providers
    }
    return LLMPipeline(settings, list(providers), breakers=breakers)


async def test_first_healthy_tier_answers(settings: Settings) -> None:
    groq = StubProvider("groq", 1)
    gemini = StubProvider("gemini", 2)
    pipeline = make_pipeline([groq, gemini], settings)

    before = counter_value("limbot_llm_calls_total", provider="groq", result="success")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        response = await pipeline.complete(REQUEST, http=client)
    after = counter_value("limbot_llm_calls_total", provider="groq", result="success")

    assert response.text == "answer from groq"
    assert groq.calls == 1
    assert gemini.calls == 0
    assert after - before == 1
    assert pipeline.breakers["groq"].state is CircuitState.CLOSED


async def test_transient_failure_falls_through_to_the_next_tier(settings: Settings) -> None:
    groq = StubProvider("groq", 1, failures=[ProviderServerError("503", provider="groq")])
    gemini = StubProvider("gemini", 2)
    pipeline = make_pipeline([groq, gemini], settings)

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        response = await pipeline.complete(REQUEST, http=client)

    assert response.provider == "gemini"
    assert groq.calls == 1
    assert gemini.calls == 1
    assert pipeline.breakers["groq"].snapshot().consecutive_failures == 1
    assert pipeline.breakers["gemini"].state is CircuitState.CLOSED


async def test_rate_limited_tier_falls_through(settings: Settings) -> None:
    groq = StubProvider(
        "groq", 1, failures=[ProviderRateLimitError("429", provider="groq", status_code=429)]
    )
    tier3 = StubProvider("tier3", 3)
    pipeline = make_pipeline([groq, tier3], settings)

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        response = await pipeline.complete(REQUEST, http=client)

    assert response.provider == "tier3"
    assert pipeline.breakers["groq"].snapshot().consecutive_failures == 1


async def test_non_transient_failure_stops_the_chain(settings: Settings) -> None:
    groq = StubProvider(
        "groq", 1, failures=[ProviderRequestError("400 bad request", provider="groq")]
    )
    gemini = StubProvider("gemini", 2)
    pipeline = make_pipeline([groq, gemini], settings)

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        with pytest.raises(ProviderRequestError):
            await pipeline.complete(REQUEST, http=client)

    assert gemini.calls == 0
    assert pipeline.breakers["groq"].state is CircuitState.CLOSED


async def test_open_circuit_is_skipped_without_a_request(settings: Settings) -> None:
    fast = settings.model_copy(
        update={"ai_circuit_failure_threshold": 1, "ai_circuit_recovery_seconds": 60.0}
    )
    groq = StubProvider("groq", 1, failures=[ProviderServerError("503", provider="groq")])
    gemini = StubProvider("gemini", 2)
    pipeline = make_pipeline([groq, gemini], fast)

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        first = await pipeline.complete(REQUEST, http=client)
        second = await pipeline.complete(REQUEST, http=client)

    assert first.provider == "gemini"
    assert second.provider == "gemini"
    assert groq.calls == 1
    assert gemini.calls == 2
    assert pipeline.breakers["groq"].state is CircuitState.OPEN

    before = counter_value("limbot_circuit_rejections_total", provider="groq")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        await pipeline.complete(REQUEST, http=client)
    after = counter_value("limbot_circuit_rejections_total", provider="groq")
    assert after - before == 1


async def test_circuit_closes_again_after_a_successful_probe(settings: Settings) -> None:
    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = FakeClock()
    fast = settings.model_copy(
        update={"ai_circuit_failure_threshold": 1, "ai_circuit_recovery_seconds": 20.0}
    )
    groq = StubProvider("groq", 1, failures=[ProviderServerError("503", provider="groq")])
    gemini = StubProvider("gemini", 2)
    pipeline = make_pipeline([groq, gemini], fast, clock=clock)

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        assert (await pipeline.complete(REQUEST, http=client)).provider == "gemini"

    assert pipeline.breakers["groq"].state is CircuitState.OPEN

    clock.now = 25.0
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        response = await pipeline.complete(REQUEST, http=client)

    assert response.provider == "groq"
    assert groq.calls == 2
    assert pipeline.breakers["groq"].state is CircuitState.CLOSED


async def test_success_resets_failures_recorded_earlier(settings: Settings) -> None:
    groq = StubProvider("groq", 1, failures=[ProviderServerError("503", provider="groq")])
    pipeline = make_pipeline([groq], settings)

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        with pytest.raises(NoProviderAvailable):
            await pipeline.complete(REQUEST, http=client)
        assert pipeline.breakers["groq"].snapshot().consecutive_failures == 1
        assert (await pipeline.complete(REQUEST, http=client)).provider == "groq"

    assert pipeline.breakers["groq"].snapshot().consecutive_failures == 0


async def test_every_tier_failing_raises_with_an_attempt_trail(settings: Settings) -> None:
    groq = StubProvider("groq", 1, failures=[ProviderServerError("503", provider="groq")])
    gemini = StubProvider("gemini", 2, failures=[ProviderServerError("503", provider="gemini")])
    pipeline = make_pipeline([groq, gemini], settings)

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        with pytest.raises(NoProviderAvailable) as excinfo:
            await pipeline.complete(REQUEST, http=client)

    outcomes = [(attempt.provider, attempt.result) for attempt in excinfo.value.attempts]
    assert outcomes == [
        ("groq", "transient_error"),
        ("gemini", "transient_error"),
    ]
    assert "503" in (excinfo.value.attempts[0].detail or "")


async def test_no_configured_provider_fails_fast(settings: Settings) -> None:
    groq = StubProvider("groq", 1, configured=False)
    pipeline = make_pipeline([groq], settings)

    assert pipeline.available is False
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        with pytest.raises(NoProviderAvailable, match="no AI provider is configured"):
            await pipeline.complete(REQUEST, http=client)

    assert groq.calls == 0


async def test_a_fallback_to_the_next_tier_is_counted(settings: Settings) -> None:
    groq = StubProvider("groq", 1, failures=[ProviderServerError("503", provider="groq")])
    gemini = StubProvider("gemini", 2)
    pipeline = make_pipeline([groq, gemini], settings)

    before = counter_value(
        "limbot_provider_fallbacks_total",
        from_provider="groq",
        to_provider="gemini",
        reason="error",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        await pipeline.complete(REQUEST, http=client)
    after = counter_value(
        "limbot_provider_fallbacks_total",
        from_provider="groq",
        to_provider="gemini",
        reason="error",
    )

    assert after - before == 1


async def test_a_rate_limited_fallback_records_its_own_reason(settings: Settings) -> None:
    groq = StubProvider(
        "groq", 1, failures=[ProviderRateLimitError("429", provider="groq", status_code=429)]
    )
    gemini = StubProvider("gemini", 2)
    pipeline = make_pipeline([groq, gemini], settings)

    before = counter_value(
        "limbot_provider_fallbacks_total",
        from_provider="groq",
        to_provider="gemini",
        reason="rate_limited",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        await pipeline.complete(REQUEST, http=client)
    after = counter_value(
        "limbot_provider_fallbacks_total",
        from_provider="groq",
        to_provider="gemini",
        reason="rate_limited",
    )

    assert after - before == 1


async def test_skipping_an_open_circuit_counts_as_a_fallback(settings: Settings) -> None:
    fast = settings.model_copy(
        update={"ai_circuit_failure_threshold": 1, "ai_circuit_recovery_seconds": 60.0}
    )
    groq = StubProvider("groq", 1, failures=[ProviderServerError("503", provider="groq")])
    gemini = StubProvider("gemini", 2)
    pipeline = make_pipeline([groq, gemini], fast)

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        # The first call fails groq and opens its circuit; the second skips the now-open
        # circuit, which is the circuit_open fallback the counter is about.
        await pipeline.complete(REQUEST, http=client)

    before = counter_value(
        "limbot_provider_fallbacks_total",
        from_provider="groq",
        to_provider="gemini",
        reason="circuit_open",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        await pipeline.complete(REQUEST, http=client)
    after = counter_value(
        "limbot_provider_fallbacks_total",
        from_provider="groq",
        to_provider="gemini",
        reason="circuit_open",
    )

    assert after - before == 1


async def test_the_last_tier_failing_is_not_counted_as_a_fallback(settings: Settings) -> None:
    gemini = StubProvider("gemini", 2, failures=[ProviderServerError("503", provider="gemini")])
    pipeline = make_pipeline([gemini], settings)

    before = counter_value(
        "limbot_provider_fallbacks_total",
        from_provider="gemini",
        to_provider="none",
        reason="error",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        with pytest.raises(NoProviderAvailable):
            await pipeline.complete(REQUEST, http=client)
    after = counter_value(
        "limbot_provider_fallbacks_total",
        from_provider="gemini",
        to_provider="none",
        reason="error",
    )

    assert after - before == 0


async def test_a_healthy_first_tier_records_no_fallback(settings: Settings) -> None:
    groq = StubProvider("groq", 1)
    gemini = StubProvider("gemini", 2)
    pipeline = make_pipeline([groq, gemini], settings)

    before = counter_value(
        "limbot_provider_fallbacks_total",
        from_provider="groq",
        to_provider="gemini",
        reason="error",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        await pipeline.complete(REQUEST, http=client)
    after = counter_value(
        "limbot_provider_fallbacks_total",
        from_provider="groq",
        to_provider="gemini",
        reason="error",
    )

    assert after - before == 0


async def test_every_tier_failing_counts_one_exhausted_completion(settings: Settings) -> None:
    groq = StubProvider("groq", 1, failures=[ProviderServerError("503", provider="groq")])
    gemini = StubProvider("gemini", 2, failures=[ProviderServerError("503", provider="gemini")])
    pipeline = make_pipeline([groq, gemini], settings)

    before = counter_value("limbot_all_providers_exhausted_total", cause="exhausted")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        with pytest.raises(NoProviderAvailable):
            await pipeline.complete(REQUEST, http=client)
    after = counter_value("limbot_all_providers_exhausted_total", cause="exhausted")

    assert after - before == 1


async def test_a_tier_rescuing_the_request_is_not_counted_as_exhausted(settings: Settings) -> None:
    """The distinction the alert depends on: a saved request must stay off the exhaustion path."""
    groq = StubProvider("groq", 1, failures=[ProviderServerError("503", provider="groq")])
    gemini = StubProvider("gemini", 2)
    pipeline = make_pipeline([groq, gemini], settings)

    before = counter_value("limbot_all_providers_exhausted_total", cause="exhausted")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        response = await pipeline.complete(REQUEST, http=client)
    after = counter_value("limbot_all_providers_exhausted_total", cause="exhausted")

    assert response.provider == "gemini"
    assert after - before == 0


async def test_an_open_circuit_on_the_last_tier_still_counts_as_exhausted(
    settings: Settings,
) -> None:
    """Every tier skipped is the same student-visible outcome as every tier failing."""
    fast = settings.model_copy(
        update={"ai_circuit_failure_threshold": 1, "ai_circuit_recovery_seconds": 60.0}
    )
    gemini = StubProvider("gemini", 2, failures=[ProviderServerError("503", provider="gemini")])
    pipeline = make_pipeline([gemini], fast)

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        with pytest.raises(NoProviderAvailable):
            await pipeline.complete(REQUEST, http=client)

    before = counter_value("limbot_all_providers_exhausted_total", cause="exhausted")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        with pytest.raises(NoProviderAvailable):
            await pipeline.complete(REQUEST, http=client)
    after = counter_value("limbot_all_providers_exhausted_total", cause="exhausted")

    assert after - before == 1


async def test_no_configured_provider_is_reported_separately(settings: Settings) -> None:
    """A missing credential is a deployment mistake, not an outage, so it gets its own cause."""
    pipeline = make_pipeline([], settings)

    before = counter_value("limbot_all_providers_exhausted_total", cause="unconfigured")
    exhausted_before = counter_value("limbot_all_providers_exhausted_total", cause="exhausted")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        with pytest.raises(NoProviderAvailable):
            await pipeline.complete(REQUEST, http=client)
    after = counter_value("limbot_all_providers_exhausted_total", cause="unconfigured")
    exhausted_after = counter_value("limbot_all_providers_exhausted_total", cause="exhausted")

    assert after - before == 1
    assert exhausted_after - exhausted_before == 0


def test_build_providers_follows_the_configured_tier_order(settings: Settings) -> None:
    from app.llm.pipeline import build_providers

    with_only_gemini = settings.model_copy(update={"gemini_api_key": SecretStr("AIza-x")})
    providers = build_providers(with_only_gemini)
    assert [p.name for p in providers] == ["gemini"]
    assert [p.tier for p in providers] == [2]

    with_groq = settings.model_copy(
        update={"gemini_api_key": SecretStr("AIza-x"), "groq_api_key": SecretStr("gsk-x")}
    )
    assert [p.name for p in build_providers(with_groq)] == ["groq", "gemini"]

    with_local = with_groq.model_copy(
        update={"tier3_enabled": True, "tier3_base_url": "http://ollama:11434/v1"}
    )
    assert [p.name for p in build_providers(with_local)] == ["groq", "gemini", "tier3"]


def test_describe_exposes_providers_and_circuits(settings: Settings) -> None:
    from app.llm.pipeline import build_providers

    configured = settings.model_copy(
        update={"groq_api_key": SecretStr("gsk-x"), "gemini_api_key": SecretStr("AIza-x")}
    )
    pipeline = LLMPipeline(configured, build_providers(configured))

    described = pipeline.describe()
    assert [p["name"] for p in described["providers"]] == ["groq", "gemini"]
    assert {c["name"] for c in described["circuits"]} == {"groq", "gemini"}
    assert pipeline.circuit_state() == {"groq": "closed", "gemini": "closed"}
