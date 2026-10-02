from __future__ import annotations

import json

import httpx
import pytest
from app.llm.base import (
    ProviderConfigError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
    ProviderTransportError,
)
from app.llm.openai_compat import groq_provider, openai_compatible_provider
from app.llm.types import ChatMessage, LLMRequest, ToolSpec

REQUEST = LLMRequest(
    messages=(ChatMessage.system("be brief"), ChatMessage.user("what is on my timetable?")),
    tools=(
        ToolSpec(
            name="get_timetable",
            description="Return the timetable for the current student.",
            parameters={
                "type": "object",
                "properties": {"day": {"type": "string", "enum": ["mon", "tue"]}},
                "required": ["day"],
                "additionalProperties": False,
            },
        ),
    ),
    max_tokens=256,
)


def client_for(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


async def test_groq_request_shape_and_plain_text_response() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "llama-3.3-70b-versatile",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Monday you have Linear Algebra.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 120, "completion_tokens": 18, "total_tokens": 138},
            },
        )

    provider = groq_provider(
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        api_key="gsk-test",
        timeout_seconds=5.0,
    )
    async with client_for(handler) as client:
        response = await provider.complete(REQUEST, http=client)

    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer gsk-test"
    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "llama-3.3-70b-versatile"
    assert payload["max_tokens"] == 256
    assert payload["stream"] is False
    assert payload["tool_choice"] == "auto"
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert messages[0] == {"role": "system", "content": "be brief"}
    assert messages[1]["role"] == "user"

    assert response.text == "Monday you have Linear Algebra."
    assert response.provider == "groq"
    assert response.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.total_tokens == 138
    assert response.has_tool_calls is False


async def test_tool_calls_are_parsed_with_idle_arguments() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": "get_timetable",
                                        "arguments": '{"day": "mon"}',
                                    },
                                },
                                {
                                    "id": "call_def",
                                    "type": "function",
                                    "function": {
                                        "name": "get_deadlines",
                                        "arguments": "not json at all",
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    provider = groq_provider(
        base_url="https://api.groq.com/openai/v1",
        model="m",
        api_key="gsk-test",
        timeout_seconds=5.0,
    )
    async with client_for(handler) as client:
        response = await provider.complete(REQUEST, http=client)

    assert response.text is None
    assert response.finish_reason == "tool_calls"
    first, second = response.tool_calls
    assert first.id == "call_abc"
    assert first.name == "get_timetable"
    assert first.arguments == {"day": "mon"}
    assert first.parse_error is None
    assert second.arguments == {}
    assert second.parse_error is not None


async def test_unconfigured_provider_raises_without_a_request() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    provider = groq_provider(base_url="", model="m", api_key=None, timeout_seconds=1.0)
    async with client_for(handler) as client:
        with pytest.raises(ProviderConfigError):
            await provider.complete(REQUEST, http=client)

    assert called is False
    assert provider.configured is False


@pytest.mark.parametrize(
    ("status", "expected", "transient"),
    [
        (401, ProviderConfigError, False),
        (403, ProviderConfigError, False),
        (400, ProviderRequestError, False),
        (404, ProviderRequestError, False),
        (408, ProviderTimeoutError, True),
        (429, ProviderRateLimitError, True),
        (500, ProviderServerError, True),
        (503, ProviderServerError, True),
    ],
)
async def test_error_status_maps_to_the_right_exception(
    status: int, expected: type[ProviderError], transient: bool
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "nope"}})

    provider = groq_provider(
        base_url="https://api.groq.com/openai/v1", model="m", api_key="k", timeout_seconds=1.0
    )
    async with client_for(handler) as client:
        with pytest.raises(expected) as excinfo:
            await provider.complete(REQUEST, http=client)

    assert excinfo.value.transient is transient
    assert excinfo.value.status_code == status
    assert excinfo.value.provider == "groq"


async def test_rate_limit_is_transient_and_keeps_retry_after() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "7"}, json={"error": "slow down"})

    provider = groq_provider(
        base_url="https://api.groq.com/openai/v1", model="m", api_key="k", timeout_seconds=1.0
    )
    async with client_for(handler) as client:
        with pytest.raises(ProviderRateLimitError) as excinfo:
            await provider.complete(REQUEST, http=client)

    assert excinfo.value.transient is True
    assert excinfo.value.retry_after == 7.0


async def test_connection_failures_become_transport_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = groq_provider(
        base_url="https://api.groq.com/openai/v1", model="m", api_key="k", timeout_seconds=1.0
    )
    async with client_for(handler) as client:
        with pytest.raises(ProviderTransportError) as excinfo:
            await provider.complete(REQUEST, http=client)

    assert excinfo.value.transient is True


async def test_timeouts_become_timeout_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    provider = groq_provider(
        base_url="https://api.groq.com/openai/v1", model="m", api_key="k", timeout_seconds=1.0
    )
    async with client_for(handler) as client:
        with pytest.raises(ProviderTimeoutError):
            await provider.complete(REQUEST, http=client)


async def test_a_200_without_choices_is_a_response_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    provider = groq_provider(
        base_url="https://api.groq.com/openai/v1", model="m", api_key="k", timeout_seconds=1.0
    )
    async with client_for(handler) as client:
        with pytest.raises(ProviderResponseError):
            await provider.complete(REQUEST, http=client)


async def test_local_tier3_omits_the_authorization_header() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "local answer"}}]})

    provider = openai_compatible_provider(
        base_url="http://ollama:11434/v1/", model="llama3.2", api_key=None, timeout_seconds=5.0
    )
    async with client_for(handler) as client:
        response = await provider.complete(REQUEST, http=client)

    assert seen["url"] == "http://ollama:11434/v1/chat/completions"
    assert seen["auth"] is None
    assert response.text == "local answer"
    assert response.provider == "tier3"
