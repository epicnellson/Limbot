from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A callable function exposed to the model, described with a JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A model requested call. ``parse_error`` is set when the arguments were not valid JSON."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    parse_error: str | None = None

    def arguments_json(self) -> str:
        return json.dumps(self.arguments)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: tuple[ToolCall, ...] = ()) -> ChatMessage:
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool_result(cls, *, call_id: str, name: str, content: str) -> ChatMessage:
        return cls(role="tool", content=content, tool_call_id=call_id, name=name)

    def to_openai(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content or None}
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments_json(),
                    },
                }
                for call in self.tool_calls
            ]
        if self.role == "tool":
            payload["tool_call_id"] = self.tool_call_id
            payload["name"] = self.name
        return payload


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def as_dict(self) -> dict[str, int | None]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    provider: str = "unknown"
    model: str = "unknown"
    finish_reason: str | None = None
    usage: Usage | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass(frozen=True, slots=True)
class LLMRequest:
    messages: tuple[ChatMessage, ...]
    tools: tuple[ToolSpec, ...] = ()
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_seconds: float | None = None

    def with_extra_messages(self, extra: tuple[ChatMessage, ...]) -> LLMRequest:
        return LLMRequest(
            messages=(*self.messages, *extra),
            tools=self.tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
        )


def parse_tool_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """Decode tool arguments, tolerating the shapes providers actually emit.

    Models occasionally return a bare string, a JSON array, or a quoted string instead of an
    object. Those are reported as a parse error so the router can hand the problem back to the
    model instead of failing the whole turn.
    """
    if raw is None:
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    if not isinstance(raw, str):
        return {}, f"arguments were {type(raw).__name__}, expected an object"
    text = raw.strip()
    if not text:
        return {}, None
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        return {}, f"arguments were not valid JSON: {exc.msg}"
    if isinstance(decoded, dict):
        return decoded, None
    return {}, f"arguments decoded to {type(decoded).__name__}, expected an object"
