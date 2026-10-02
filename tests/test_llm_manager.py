from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from app.config import Settings
from app.core.circuit import CircuitBreaker, CircuitState
from app.llm.base import (
    LLMProvider,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
)
from app.llm.llm_manager import (
    AllProvidersExhaustedException,
    LLMProviderManager,
    LLMResponse,
)
from app.llm.pipeline import LLMPipeline
from app.llm.types import LLMRequest, ToolCall, ToolSpec
from app.llm.types import LLMResponse as PipelineResponse

from conftest import counter_value


class ScriptedProvider(LLMProvider):
    """Answers, fails, or stalls on a script, and records the request it received."""

    def __init__(
        self,
        name: str,
        tier: int,
        *,
        error: Exception | None = None,
        stall: float = 0.0,
        text: str = "",
        tool_calls: tuple[ToolCall, ...] = (),
        configured: bool = True,
    ) -> None:
        super().__init__(f"{name}-model", name=name, tier=tier)
        self._error = error
        self._stall = stall
        self._text = text
        self._tool_calls = tool_calls
        self._configured = configured
        self.calls = 0
        self.seen_request: LLMRequest | None = None

    @property
    def configured(self) -> bool:
        return self._configured

    async def complete(self, request: LLMRequest, *, http: httpx.AsyncClient) -> PipelineResponse:
        self.calls += 1
        self.seen_request = request
        if self._stall:
            await asyncio.sleep(self._stall)
        if self._error is not None:
            raise self._error
        return PipelineResponse(
            text=self._text or f"answer from {self.name}",
            provider=self.name,
            tool_calls=self._tool_calls,
        )


def scripted_pipeline(settings: Settings, providers: list[ScriptedProvider]) -> LLMPipeline:
    breakers = {
        provider.name: CircuitBreaker(
            name=provider.name,
            failure_threshold=settings.ai_circuit_failure_threshold,
            recovery_seconds=settings.ai_circuit_recovery_seconds,
            half_open_calls=settings.ai_circuit_half_open_calls,
        )
        for provider in providers
    }
    return LLMPipeline(settings, list(providers), breakers=breakers)


def make_manager(
    settings: Settings, pipeline: LLMPipeline, *, timeout: float = 5.0
) -> LLMProviderManager:
    return LLMProviderManager(settings, pipeline=pipeline, timeout_seconds=timeout)


@pytest.fixture
def http() -> httpx.AsyncClient:
    # The manager never makes a real request; the transport exists only to satisfy the shared
    # client convention.
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda request: None))


async def test_completion_returns_primary_and_flat_response(
    settings: Settings, http: httpx.AsyncClient
) -> None:
    groq = ScriptedProvider("groq", 1, text="it is on monday.")
    manager = make_manager(
        settings, scripted_pipeline(settings, [groq, ScriptedProvider("gemini", 2)])
    )

    result = await manager.completion("when is the exam?", http=http)

    assert groq.calls == 1
    assert isinstance(result, LLMResponse)
    assert result.content == "it is on monday."
    assert result.provider_used == "groq"
    assert result.latency_ms >= 0
    assert result.tool_calls == []


async def test_completion_sends_the_system_instruction(
    settings: Settings, http: httpx.AsyncClient
) -> None:
    groq = ScriptedProvider("groq", 1)
    manager = make_manager(settings, scripted_pipeline(settings, [groq]))

    await manager.completion("hello", system_instruction="You are a student assistant.", http=http)

    assert groq.seen_request is not None
    roles = [message.role for message in groq.seen_request.messages]
    assert roles == ["system", "user"]
    assert groq.seen_request.messages[0].content == "You are a student assistant."
    assert groq.seen_request.messages[1].content == "hello"


async def test_no_system_message_when_instruction_is_empty(
    settings: Settings, http: httpx.AsyncClient
) -> None:
    groq = ScriptedProvider("groq", 1)
    manager = make_manager(settings, scripted_pipeline(settings, [groq]))

    await manager.completion("hello", http=http)

    assert [message.role for message in groq.seen_request.messages] == ["user"]


async def test_rate_limit_falls_back_and_counts_metrics(
    settings: Settings, http: httpx.AsyncClient
) -> None:
    groq = ScriptedProvider(
        "groq", 1, error=ProviderRateLimitError("429 too many", provider="groq", status_code=429)
    )
    gemini = ScriptedProvider("gemini", 2)
    manager = make_manager(settings, scripted_pipeline(settings, [groq, gemini]))

    before = counter_value("limbot_llm_calls_total", provider="groq", result="rate_limited")
    result = await manager.completion("help", http=http)
    after = counter_value("limbot_llm_calls_total", provider="groq", result="rate_limited")

    assert groq.calls == 1
    assert gemini.calls == 1
    assert result.provider_used == "gemini"
    assert after - before == 1


async def test_server_error_falls_through_to_tertiary(
    settings: Settings, http: httpx.AsyncClient
) -> None:
    groq = ScriptedProvider(
        "groq", 1, error=ProviderServerError("502 bad gateway", provider="groq", status_code=502)
    )
    gemini = ScriptedProvider(
        "gemini", 2, error=ProviderServerError("500 internal", provider="gemini", status_code=500)
    )
    ollama = ScriptedProvider("ollama", 3)
    manager = make_manager(settings, scripted_pipeline(settings, [groq, gemini, ollama]))

    result = await manager.completion("hello", http=http)

    assert result.provider_used == "ollama"


async def test_stalled_provider_is_cancelled_by_the_deadline(
    settings: Settings, http: httpx.AsyncClient
) -> None:
    groq = ScriptedProvider("groq", 1, stall=10.0)
    gemini = ScriptedProvider("gemini", 2)
    manager = make_manager(settings, scripted_pipeline(settings, [groq, gemini]), timeout=0.05)

    before = counter_value("limbot_llm_calls_total", provider="groq", result="timeout")
    started = asyncio.get_running_loop().time()
    result = await manager.completion("please", http=http)
    elapsed = asyncio.get_running_loop().time() - started
    after = counter_value("limbot_llm_calls_total", provider="groq", result="timeout")

    assert result.provider_used == "gemini"
    assert elapsed < 5.0
    assert after - before == 1
    assert result.latency_ms < 5_000


async def test_all_failures_raise_the_custom_exception(
    settings: Settings, http: httpx.AsyncClient
) -> None:
    groq = ScriptedProvider(
        "groq", 1, error=ProviderRateLimitError("429", provider="groq", status_code=429)
    )
    gemini = ScriptedProvider(
        "gemini", 2, error=ProviderServerError("500", provider="gemini", status_code=500)
    )
    manager = make_manager(settings, scripted_pipeline(settings, [groq, gemini]))

    with pytest.raises(AllProvidersExhaustedException) as excinfo:
        await manager.completion("hello", http=http)

    assert "failed" in str(excinfo.value)
    assert len(excinfo.value.attempts) == 2


async def test_exhaustion_carries_attempt_details(
    settings: Settings, http: httpx.AsyncClient
) -> None:
    groq = ScriptedProvider("groq", 1, error=ProviderTimeoutError("timeout", provider="groq"))
    manager = make_manager(settings, scripted_pipeline(settings, [groq]))

    with pytest.raises(AllProvidersExhaustedException) as excinfo:
        await manager.completion("hello", http=http)

    details = excinfo.value.details
    assert [entry["provider"] for entry in details] == ["groq"]
    assert [entry["result"] for entry in details] == ["transient_error"]


async def test_open_circuit_skips_the_tier(settings: Settings, http: httpx.AsyncClient) -> None:
    tight = settings.model_copy(update={"ai_circuit_failure_threshold": 1})
    groq = ScriptedProvider(
        "groq", 1, error=ProviderServerError("500", provider="groq", status_code=500)
    )
    gemini = ScriptedProvider("gemini", 2)
    manager = make_manager(tight, scripted_pipeline(tight, [groq, gemini]))

    await manager.completion("first", http=http)
    assert manager._pipeline.breakers["groq"].state is CircuitState.OPEN

    result = await manager.completion("second", http=http)

    assert groq.calls == 1  # skipped the second time
    assert gemini.calls == 2
    assert result.provider_used == "gemini"


async def test_tool_specs_are_passed_through(settings: Settings, http: httpx.AsyncClient) -> None:
    groq = ScriptedProvider("groq", 1)
    manager = make_manager(settings, scripted_pipeline(settings, [groq]))

    spec = ToolSpec(
        name="timetable",
        description="Today's classes",
        parameters={"type": "object", "properties": {}},
    )
    await manager.completion("today?", tools=[spec], http=http)

    assert [tool.name for tool in groq.seen_request.tools] == ["timetable"]


async def test_tool_dicts_are_normalised(settings: Settings, http: httpx.AsyncClient) -> None:
    groq = ScriptedProvider("groq", 1)
    manager = make_manager(settings, scripted_pipeline(settings, [groq]))

    flat = {
        "name": "grades",
        "description": "Grades",
        "parameters": {"type": "object", "properties": {}},
    }
    openai_style = {
        "type": "function",
        "function": {
            "name": "exams",
            "description": "Exams",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    await manager.completion("data?", tools=[flat, openai_style], http=http)

    assert [tool.name for tool in groq.seen_request.tools] == ["grades", "exams"]


async def test_tool_calls_are_parsed_into_the_response(
    settings: Settings, http: httpx.AsyncClient
) -> None:
    call = ToolCall(id="call_1", name="timetable", arguments={"day": "monday"})
    groq = ScriptedProvider("groq", 1, tool_calls=(call,))
    manager = make_manager(settings, scripted_pipeline(settings, [groq]))

    result = await manager.completion("today?", http=http)

    assert result.tool_calls == [
        {"id": "call_1", "name": "timetable", "arguments": {"day": "monday"}}
    ]


async def test_parse_error_tag_carries_into_tool_calls(
    settings: Settings, http: httpx.AsyncClient
) -> None:
    call = ToolCall(
        id="call_1", name="exams", arguments={}, parse_error="arguments were not valid JSON"
    )
    groq = ScriptedProvider("groq", 1, tool_calls=(call,))
    manager = make_manager(settings, scripted_pipeline(settings, [groq]))

    result = await manager.completion("exams?", http=http)

    assert result.tool_calls[0]["parse_error"].startswith("arguments were not valid")


async def test_empty_prompt_is_rejected(settings: Settings, http: httpx.AsyncClient) -> None:
    groq = ScriptedProvider("groq", 1)
    manager = make_manager(settings, scripted_pipeline(settings, [groq]))

    with pytest.raises(ValueError, match="empty"):
        await manager.completion("   ", http=http)
    assert groq.calls == 0


async def test_invalid_tool_argument_raises_type_error(
    settings: Settings, http: httpx.AsyncClient
) -> None:
    groq = ScriptedProvider("groq", 1)
    manager = make_manager(settings, scripted_pipeline(settings, [groq]))

    with pytest.raises(TypeError, match="ToolSpec or dict"):
        await manager.completion("hello", tools=["timetable"], http=http)


async def test_temperature_and_max_tokens_default_from_settings(
    settings: Settings, http: httpx.AsyncClient
) -> None:
    groq = ScriptedProvider("groq", 1)
    overridden = settings.model_copy(update={"ai_temperature": 0.7, "ai_max_tokens": 512})
    manager = make_manager(overridden, scripted_pipeline(overridden, [groq]))

    await manager.completion("hello", http=http)

    assert groq.seen_request.temperature == 0.7
    assert groq.seen_request.max_tokens == 512


async def test_unconfigured_pipeline_raises_exhausted(
    settings: Settings, http: httpx.AsyncClient
) -> None:
    groq = ScriptedProvider("groq", 1, configured=False)
    manager = make_manager(settings, scripted_pipeline(settings, [groq]))

    with pytest.raises(AllProvidersExhaustedException, match="configured"):
        await manager.completion("hello", http=http)


async def test_response_as_dict(settings: Settings, http: httpx.AsyncClient) -> None:
    groq = ScriptedProvider("groq", 1, text="yes")
    manager = make_manager(settings, scripted_pipeline(settings, [groq]))

    result = await manager.completion("hello", http=http)

    payload: dict[str, Any] = result.as_dict()
    assert payload["provider_used"] == "groq"
    assert payload["content"] == "yes"
    assert payload["latency_ms"] >= 0
