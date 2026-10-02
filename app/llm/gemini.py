from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.core.http import NO_RETRIES, OutboundTransportError, send
from app.llm.base import (
    LLMProvider,
    ProviderRequestError,
    ProviderResponseError,
    raise_for_status,
    wrap_transport_error,
)
from app.llm.types import LLMRequest, LLMResponse, ToolCall, Usage

logger = logging.getLogger(__name__)

BLOCKED_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}


def _upper_types(schema: Any) -> Any:
    """Rewrite a JSON schema into the subset Gemini accepts.

    Gemini uses OpenAPI style types in upper case and rejects unknown keywords, so
    ``additionalProperties`` and friends are dropped rather than forwarded.
    """
    if isinstance(schema, dict):
        converted: dict[str, Any] = {}
        for key, value in schema.items():
            if key in {"additionalProperties", "$schema", "default", "examples"}:
                continue
            if key == "type" and isinstance(value, str):
                converted[key] = value.upper()
            else:
                converted[key] = _upper_types(value)
        return converted
    if isinstance(schema, list):
        return [_upper_types(item) for item in schema]
    return schema


class GeminiProvider(LLMProvider):
    """Google Gemini tier, used for the long context fallback."""

    name = "gemini"
    tier = 2

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_seconds: float = 45.0,
    ) -> None:
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key)

    def function_declaration(self, spec: Any) -> dict[str, Any]:
        return {
            "name": spec.name,
            "description": spec.description,
            "parameters": _upper_types(spec.parameters),
        }

    def _system_instruction(self, request: LLMRequest) -> dict[str, Any] | None:
        parts = [m.content for m in request.messages if m.role == "system" and m.content]
        if not parts:
            return None
        return {"parts": [{"text": "\n\n".join(parts)}]}

    def _contents(self, request: LLMRequest) -> list[dict[str, Any]]:
        """Convert the conversation, merging consecutive tool results into a single user turn.

        Gemini has no tool role. Assistant tool calls become ``functionCall`` parts on a model
        turn and the results become ``functionResponse`` parts on the following user turn, which
        is what the API expects for a tool round trip.
        """
        contents: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "system":
                continue
            if message.role == "user":
                contents.append({"role": "user", "parts": [{"text": message.content or "(empty)"}]})
            elif message.role == "assistant":
                parts: list[dict[str, Any]] = []
                if message.content:
                    parts.append({"text": message.content})
                parts.extend(
                    {"functionCall": {"name": call.name, "args": call.arguments}}
                    for call in message.tool_calls
                )
                if parts:
                    contents.append({"role": "model", "parts": parts})
            elif message.role == "tool":
                response = {
                    "functionResponse": {
                        "name": message.name or "tool",
                        "response": {"result": _as_result(message.content)},
                    }
                }
                if (
                    contents
                    and contents[-1]["role"] == "user"
                    and _only_function_responses(contents[-1])
                ):
                    contents[-1]["parts"].append(response)
                else:
                    contents.append({"role": "user", "parts": [response]})
        return contents

    def _payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contents": self._contents(request),
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            },
        }
        system = self._system_instruction(request)
        if system is not None:
            payload["systemInstruction"] = system
        if request.tools:
            payload["tools"] = [
                {"functionDeclarations": [self.function_declaration(s) for s in request.tools]}
            ]
        return payload

    async def complete(self, request: LLMRequest, *, http: httpx.AsyncClient) -> LLMResponse:
        self._require_config()
        url = f"{self.base_url}/models/{self.model}:generateContent"
        timeout = request.timeout_seconds or self.timeout_seconds
        try:
            response = await send(
                http,
                "POST",
                url,
                client_name=self.name,
                operation="generate_content",
                retry_policy=NO_RETRIES,
                headers={
                    "content-type": "application/json",
                    "x-goog-api-key": self.api_key or "",
                },
                json=self._payload(request),
                timeout=httpx.Timeout(timeout),
            )
        except OutboundTransportError as exc:
            raise wrap_transport_error(exc.cause, provider=self.name, model=self.model) from exc
        except httpx.HTTPError as exc:
            raise wrap_transport_error(exc, provider=self.name, model=self.model) from exc

        raise_for_status(response, provider=self.name, model=self.model)
        return self._parse(response)

    def _parse(self, response: httpx.Response) -> LLMResponse:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderResponseError(
                "gemini returned a non JSON body", provider=self.name
            ) from exc
        if not isinstance(payload, dict):
            raise ProviderResponseError(
                f"gemini returned {type(payload).__name__}, expected an object",
                provider=self.name,
            )

        feedback = payload.get("promptFeedback")
        if isinstance(feedback, dict):
            reason = feedback.get("blockReason")
            if isinstance(reason, str) and reason in BLOCKED_REASONS:
                raise ProviderRequestError(
                    f"gemini blocked the prompt: {reason}", provider=self.name
                )

        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ProviderResponseError("gemini returned no candidates", provider=self.name)
        candidate = candidates[0] if isinstance(candidates[0], dict) else {}
        content = candidate.get("content") if isinstance(candidate.get("content"), dict) else {}
        parts = content.get("parts") if isinstance(content.get("parts"), list) else []

        texts: list[str] = []
        calls: list[ToolCall] = []
        for part in parts if isinstance(parts, list) else []:
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("text"), str) and part["text"].strip():
                texts.append(part["text"])
            function_call = part.get("functionCall")
            if isinstance(function_call, dict):
                name = function_call.get("name")
                if not isinstance(name, str) or not name:
                    continue
                args = function_call.get("args")
                calls.append(
                    ToolCall(
                        id=f"gemini_{len(calls)}",
                        name=name,
                        arguments=args if isinstance(args, dict) else {},
                        parse_error=None
                        if isinstance(args, dict)
                        else "function call arguments were not an object",
                    )
                )

        finish = candidate.get("finishReason")
        return LLMResponse(
            text="\n".join(texts).strip() or None,
            tool_calls=tuple(calls),
            provider=self.name,
            model=self.model,
            finish_reason=finish if isinstance(finish, str) else None,
            usage=_parse_usage(payload.get("usageMetadata")),
        )


def gemini_provider(
    *, base_url: str, model: str, api_key: str | None, timeout_seconds: float
) -> GeminiProvider:
    return GeminiProvider(
        base_url=base_url, model=model, api_key=api_key, timeout_seconds=timeout_seconds
    )


def _only_function_responses(content: dict[str, Any]) -> bool:
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        return False
    return all(isinstance(part, dict) and "functionResponse" in part for part in parts)


def _as_result(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("{") or text.startswith("["):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return {"text": content}
        if isinstance(decoded, dict):
            return decoded
        return {"items": decoded}
    return {"text": content}


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _parse_usage(raw: Any) -> Usage | None:
    if not isinstance(raw, dict):
        return None
    return Usage(
        input_tokens=_as_int(raw.get("promptTokenCount")),
        output_tokens=_as_int(raw.get("candidatesTokenCount")),
        total_tokens=_as_int(raw.get("totalTokenCount")),
    )
