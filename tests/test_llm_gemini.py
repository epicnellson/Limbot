from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from app.llm.base import ProviderConfigError, ProviderRequestError, ProviderResponseError
from app.llm.gemini import _upper_types, gemini_provider
from app.llm.types import ChatMessage, LLMRequest, ToolCall, ToolSpec

TOOL = ToolSpec(
    name="get_timetable",
    description="Return the timetable for the current student.",
    parameters={
        "type": "object",
        "properties": {
            "day": {"type": "string", "enum": ["mon", "tue"]},
            "include_grades": {"type": "boolean", "default": False},
        },
        "required": ["day"],
        "additionalProperties": False,
    },
)

REQUEST = LLMRequest(
    messages=(
        ChatMessage.system("be brief"),
        ChatMessage.user("what is on my timetable?"),
    ),
    tools=(TOOL,),
    max_tokens=128,
)


def client_for(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def make_provider() -> Any:
    return gemini_provider(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        model="gemini-2.0-flash",
        api_key="AIza-test",
        timeout_seconds=5.0,
    )


def test_schema_types_are_upper_cased_and_unknown_keys_dropped() -> None:
    converted = _upper_types(TOOL.parameters)

    assert converted["type"] == "OBJECT"
    assert converted["properties"]["day"]["type"] == "STRING"
    assert converted["properties"]["include_grades"]["type"] == "BOOLEAN"
    assert "additionalProperties" not in converted
    assert "default" not in converted["properties"]["include_grades"]


async def test_request_shape_uses_contents_and_system_instruction() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "Linear Algebra at 10."}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 42,
                    "candidatesTokenCount": 8,
                    "totalTokenCount": 50,
                },
            },
        )

    async with client_for(handler) as client:
        response = await make_provider().complete(REQUEST, http=client)

    assert seen["url"].endswith("/models/gemini-2.0-flash:generateContent")
    assert seen["key"] == "AIza-test"
    payload = seen["payload"]
    assert payload["systemInstruction"]["parts"] == [{"text": "be brief"}]
    assert payload["contents"] == [
        {"role": "user", "parts": [{"text": "what is on my timetable?"}]}
    ]
    assert payload["generationConfig"]["maxOutputTokens"] == 128
    declaration = payload["tools"][0]["functionDeclarations"][0]
    assert declaration["name"] == "get_timetable"
    assert declaration["parameters"]["type"] == "OBJECT"

    assert response.text == "Linear Algebra at 10."
    assert response.provider == "gemini"
    assert response.finish_reason == "STOP"
    assert response.usage is not None
    assert response.usage.input_tokens == 42
    assert response.usage.output_tokens == 8


async def test_function_call_parts_become_tool_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"functionCall": {"name": "get_timetable", "args": {"day": "mon"}}}
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ]
            },
        )

    async with client_for(handler) as client:
        response = await make_provider().complete(REQUEST, http=client)

    assert response.text is None
    call = response.tool_calls[0]
    assert call.id == "gemini_0"
    assert call.name == "get_timetable"
    assert call.arguments == {"day": "mon"}
    assert call.parse_error is None


async def test_non_object_function_args_are_flagged() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"functionCall": {"name": "get_timetable", "args": "mon"}}],
                        }
                    }
                ]
            },
        )

    async with client_for(handler) as client:
        response = await make_provider().complete(REQUEST, http=client)

    assert response.tool_calls[0].parse_error is not None


async def test_tool_round_trip_is_rendered_as_function_response_parts() -> None:
    seen: dict[str, Any] = {}
    conversation = LLMRequest(
        messages=(
            ChatMessage.system("be brief"),
            ChatMessage.user("what is on my timetable?"),
            ChatMessage.assistant(
                "", (ToolCall(id="gemini_0", name="get_timetable", arguments={"day": "mon"}),)
            ),
            ChatMessage.tool_result(
                call_id="gemini_0",
                name="get_timetable",
                content='{"entries": [{"course": "Linear Algebra"}]}',
            ),
        ),
        tools=(TOOL,),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": "Linear Algebra at 10."}]}}
                ]
            },
        )

    async with client_for(handler) as client:
        await make_provider().complete(conversation, http=client)

    contents = seen["payload"]["contents"]
    assert [entry["role"] for entry in contents] == ["user", "model", "user"]
    assert contents[1]["parts"] == [
        {"functionCall": {"name": "get_timetable", "args": {"day": "mon"}}}
    ]
    assert contents[2]["parts"] == [
        {
            "functionResponse": {
                "name": "get_timetable",
                "response": {"result": {"entries": [{"course": "Linear Algebra"}]}},
            }
        }
    ]


async def test_consecutive_tool_results_merge_into_one_user_turn() -> None:
    seen: dict[str, Any] = {}
    conversation = LLMRequest(
        messages=(
            ChatMessage.user("compare my grades with the deadline list"),
            ChatMessage.assistant(
                "",
                (
                    ToolCall(id="gemini_0", name="get_grades", arguments={}),
                    ToolCall(id="gemini_1", name="get_deadlines", arguments={}),
                ),
            ),
            ChatMessage.tool_result(call_id="gemini_0", name="get_grades", content='{"gpa": 3.4}'),
            ChatMessage.tool_result(
                call_id="gemini_1", name="get_deadlines", content="two assignments due"
            ),
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    async with client_for(handler) as client:
        await make_provider().complete(conversation, http=client)

    contents = seen["payload"]["contents"]
    assert [entry["role"] for entry in contents] == ["user", "model", "user"]
    assert len(contents[1]["parts"]) == 2
    assert len(contents[2]["parts"]) == 2
    assert contents[2]["parts"][0]["functionResponse"]["response"] == {"result": {"gpa": 3.4}}
    assert contents[2]["parts"][1]["functionResponse"]["response"] == {
        "result": {"text": "two assignments due"}
    }


async def test_blocked_prompts_are_not_retried_anywhere() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []},
        )

    async with client_for(handler) as client:
        with pytest.raises(ProviderRequestError) as excinfo:
            await make_provider().complete(REQUEST, http=client)

    assert excinfo.value.transient is False


async def test_empty_candidates_are_a_response_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": []})

    async with client_for(handler) as client:
        with pytest.raises(ProviderResponseError):
            await make_provider().complete(REQUEST, http=client)


async def test_missing_api_key_raises_before_any_request() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    provider = gemini_provider(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        model="gemini-2.0-flash",
        api_key=None,
        timeout_seconds=1.0,
    )
    async with client_for(handler) as client:
        with pytest.raises(ProviderConfigError):
            await provider.complete(REQUEST, http=client)

    assert called is False
    assert provider.configured is False
