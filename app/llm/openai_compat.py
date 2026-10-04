from __future__ import annotations

import logging
from typing import Any

import httpx

from app.core.http import NO_RETRIES, OutboundTransportError, send
from app.llm.base import (
    LLMProvider,
    ProviderResponseError,
    raise_for_status,
    wrap_transport_error,
)
from app.llm.types import LLMRequest, LLMResponse, ToolCall, Usage, parse_tool_arguments

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """Any endpoint that speaks the OpenAI chat completions dialect.

    Covers Groq's hosted API and every local runtime that exposes an OpenAI compatible surface
    (Ollama, LM Studio, vLLM, llama.cpp), which is why the local fail-safe tier needs no
    bespoke adapter.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        name: str,
        api_key: str | None = None,
        tier: int = 3,
        timeout_seconds: float = 45.0,
        owns_retry: bool = True,
    ) -> None:
        super().__init__(model, name=name, tier=tier)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.owns_retry = owns_retry

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_openai() for message in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        if request.tools:
            payload["tools"] = [spec.to_openai() for spec in request.tools]
            payload["tool_choice"] = "auto"
        return payload

    async def complete(self, request: LLMRequest, *, http: httpx.AsyncClient) -> LLMResponse:
        self._require_config()
        url = f"{self.base_url}/chat/completions"
        timeout = request.timeout_seconds or self.timeout_seconds
        try:
            response = await send(
                http,
                "POST",
                url,
                client_name=self.name,
                operation="chat_completions",
                retry_policy=NO_RETRIES,
                headers=self._headers(),
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
                f"{self.name} returned a non JSON body", provider=self.name
            ) from exc
        if not isinstance(payload, dict):
            raise ProviderResponseError(
                f"{self.name} returned {type(payload).__name__}, expected an object",
                provider=self.name,
            )
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderResponseError(f"{self.name} returned no choices", provider=self.name)
        choice_raw = choices[0]
        choice: dict[str, Any] = choice_raw if isinstance(choice_raw, dict) else {}
        message_raw = choice.get("message")
        message: dict[str, Any] = message_raw if isinstance(message_raw, dict) else {}
        content = message.get("content")
        text = content if isinstance(content, str) and content.strip() else None

        raw_calls = message.get("tool_calls")
        tool_calls: list[ToolCall] = []
        if isinstance(raw_calls, list):
            for index, raw in enumerate(raw_calls):
                if not isinstance(raw, dict):
                    continue
                function_raw = raw.get("function")
                function: dict[str, Any] = function_raw if isinstance(function_raw, dict) else {}
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    continue
                arguments, parse_error = parse_tool_arguments(function.get("arguments"))
                tool_calls.append(
                    ToolCall(
                        id=str(raw.get("id") or f"call_{index}"),
                        name=name,
                        arguments=arguments,
                        parse_error=parse_error,
                    )
                )

        return LLMResponse(
            text=text,
            tool_calls=tuple(tool_calls),
            provider=self.name,
            model=str(payload.get("model") or self.model),
            finish_reason=_as_str(choice.get("finish_reason")),
            usage=_parse_usage(payload.get("usage")),
        )


def groq_provider(
    *, base_url: str, model: str, api_key: str | None, timeout_seconds: float
) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url=base_url,
        model=model,
        name="groq",
        api_key=api_key,
        tier=1,
        timeout_seconds=timeout_seconds,
        owns_retry=True,
    )


def openai_compatible_provider(
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    timeout_seconds: float,
    name: str = "tier3",
    tier: int = 3,
) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url=base_url,
        model=model,
        name=name,
        api_key=api_key,
        tier=tier,
        timeout_seconds=timeout_seconds,
        owns_retry=False,
    )


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _parse_usage(raw: Any) -> Usage | None:
    if not isinstance(raw, dict):
        return None
    return Usage(
        input_tokens=_as_int(raw.get("prompt_tokens")),
        output_tokens=_as_int(raw.get("completion_tokens")),
        total_tokens=_as_int(raw.get("total_tokens")),
    )
