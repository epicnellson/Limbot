"""A convenience facade over the provider fallback pipeline.

``LLMProviderManager`` hands you the three-tier fallback (Groq -> Gemini -> Ollama/tier3)
behind a single method --- ``completion(prompt, system_instruction, tools)`` --- with a hard
per-attempt deadline and a response that tells you who answered and how long it took.

It deliberately does not reimplement the resilience machinery: the fallback loop, the circuit
breakers, and the transient-vs-fatal classification all live in :mod:`app.llm.pipeline`, and
every provider talks to ``httpx`` through the shared client. This class is the thin, ergonomic
interface on top. See ``app.llm.types.LLMResponse`` for the richer pipeline-level response; the
``LLMResponse`` defined here is the flattened shape callers asked for (content, provider,
latency, parsed tool calls) and is not a substitute for it.

Usage::

    manager = LLMProviderManager(settings)
    result = await manager.completion(
        "when is my linear algebra exam?",
        system_instruction="You answer a university student on WhatsApp.",
        http=shared_httpx_client,
    )
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

import httpx

from app.config import Settings
from app.llm.pipeline import LLMPipeline, NoProviderAvailable
from app.llm.types import ChatMessage, LLMRequest, ToolCall, ToolSpec

logger = logging.getLogger(__name__)

# The deadline the specification asked for. Applied per provider attempt by the pipeline; the
# manager passes it down as the request timeout so one engine owns the boundary.
DEFAULT_TIMEOUT_SECONDS = 5.0


class AllProvidersExhaustedException(RuntimeError):
    """Every configured tier failed, timed out, or was blocked by its circuit breaker.

    Carries the per-attempt reasons from the pipeline so callers can decide whether to retry,
    degrade, or surface a human-facing error.
    """

    def __init__(self, message: str, *, attempts: tuple[Any, ...] = ()) -> None:
        super().__init__(message)
        self.attempts = attempts

    @property
    def details(self) -> list[dict[str, Any]]:
        from app.llm.pipeline import AttemptOutcome

        return [
            attempt.as_dict() for attempt in self.attempts if isinstance(attempt, AttemptOutcome)
        ]


class LLMResponse:
    """The flattened completion result: what was said, who said it, and how long it took."""

    __slots__ = ("content", "latency_ms", "provider_used", "tool_calls")

    def __init__(
        self,
        *,
        content: str,
        provider_used: str,
        latency_ms: float,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        self.content = content
        self.provider_used = provider_used
        self.latency_ms = latency_ms
        self.tool_calls = tool_calls

    def as_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "provider_used": self.provider_used,
            "latency_ms": round(self.latency_ms, 1),
            "tool_calls": self.tool_calls,
        }


class LLMProviderManager:
    """Run one completion across Groq, Gemini, and the local tier with fallback."""

    def __init__(
        self,
        settings: Settings,
        *,
        pipeline: LLMPipeline | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._settings = settings
        self._pipeline = pipeline if pipeline is not None else LLMPipeline(settings)
        self._timeout_seconds = timeout_seconds

    @property
    def available(self) -> bool:
        return self._pipeline.available

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    async def completion(
        self,
        prompt: str,
        system_instruction: str = "",
        tools: Sequence[ToolSpec | dict[str, Any]] | None = None,
        *,
        http: httpx.AsyncClient,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Run one completion, falling back through the configured tiers.

        Args:
            prompt: The user's question or message.
            system_instruction: Optional system prompt; omitted entirely when empty.
            tools: Tool specifications as :class:`ToolSpec` or plain dictionaries.
            http: The shared ``httpx.AsyncClient`` (one per process, owned by the lifespan).
            temperature: Override the settings default.
            max_tokens: Override the settings default.

        Returns:
            An :class:`LLMResponse` from the first tier that succeeded.

        Raises:
            AllProvidersExhaustedException: Every tier failed, timed out, or was open.

        The 5.0-second (configurable) deadline is enforced per provider attempt with
        ``asyncio.wait_for`` inside the pipeline, so a hung tier is cancelled and the next one
        is tried rather than holding the request open.
        """
        messages = self._messages(prompt, system_instruction)
        request = LLMRequest(
            messages=messages,
            tools=_coerce_tools(tools),
            timeout_seconds=self._timeout_seconds,
            temperature=temperature if temperature is not None else self._settings.ai_temperature,
            max_tokens=max_tokens or self._settings.ai_max_tokens,
        )

        started = time.perf_counter()
        try:
            response = await self._pipeline.complete(request, http=http)
        except NoProviderAvailable as exc:
            logger.warning(
                "every provider tier failed in the completion facade",
                extra={"context": {"attempts": [attempt.as_dict() for attempt in exc.attempts]}},
            )
            # With no attempts the pipeline is telling us about configuration, not failure, and
            # its message is the one worth surfacing (e.g. "no AI provider is configured").
            message = (
                str(exc) if not exc.attempts else "every AI provider tier failed or was skipped"
            )
            raise AllProvidersExhaustedException(message, attempts=exc.attempts) from exc

        return LLMResponse(
            content=response.text or "",
            provider_used=response.provider,
            latency_ms=(time.perf_counter() - started) * 1000,
            tool_calls=[_tool_call_payload(call) for call in response.tool_calls],
        )

    @staticmethod
    def _messages(prompt: str, system_instruction: str) -> tuple[ChatMessage, ...]:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        system = (ChatMessage.system(system_instruction),) if system_instruction.strip() else ()
        return (*system, ChatMessage.user(prompt))


def _coerce_tools(
    tools: Sequence[ToolSpec | dict[str, Any]] | None,
) -> tuple[ToolSpec, ...]:
    """Accept ToolSpec instances or plain dicts (flat or OpenAI-style tool objects)."""
    if not tools:
        return ()
    result: list[ToolSpec] = []
    for tool in tools:
        if isinstance(tool, ToolSpec):
            result.append(tool)
        elif isinstance(tool, dict):
            result.append(_tool_spec_from_dict(tool))
        else:
            raise TypeError(f"tool must be ToolSpec or dict, got {type(tool).__name__}")
    return tuple(result)


def _tool_spec_from_dict(tool: dict[str, Any]) -> ToolSpec:
    function = tool.get("function")
    if isinstance(function, dict):
        payload = function
    else:
        payload = tool
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("a tool dict must carry a non-empty 'name'")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"tool {name!r} must declare a 'parameters' object")
    return ToolSpec(
        name=name,
        description=str(payload.get("description", "")),
        parameters=parameters,
    )


def _tool_call_payload(call: ToolCall) -> dict[str, Any]:
    payload: dict[str, Any] = {"id": call.id, "name": call.name, "arguments": call.arguments}
    if call.parse_error:
        payload["parse_error"] = call.parse_error
    return payload


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "AllProvidersExhaustedException",
    "LLMProviderManager",
    "LLMResponse",
]
