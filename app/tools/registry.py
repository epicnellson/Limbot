from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app import metrics
from app.llm.types import ToolSpec
from app.tools.context import ToolContext
from app.tools.schema import require_valid

logger = logging.getLogger(__name__)

ToolHandler = Callable[[ToolContext, dict[str, Any]], Awaitable[dict[str, Any]]]

MAX_RESULT_CHARS = 4000
MAX_TOOL_CALLS_PER_RESPONSE = 4


class ToolFailure(RuntimeError):
    """A tool that ran correctly but has nothing to return.

    The message is written for the model, which turns it into an explanation for the student,
    so it should read like an answer rather than a stack trace.
    """


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A tool as the model sees it, plus the function that runs it."""

    spec: ToolSpec
    handler: ToolHandler
    timeout_seconds: float = 10.0

    @property
    def name(self) -> str:
        return self.spec.name


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What goes back into the conversation as the tool message."""

    name: str
    ok: bool
    payload: dict[str, Any]
    error: str | None = None

    def as_json(self) -> str:
        body: dict[str, Any] = {"ok": self.ok}
        if self.ok:
            body.update(self.payload)
        else:
            body["error"] = self.error or "the tool failed"
        text = json.dumps(body, ensure_ascii=False, default=str)
        if len(text) > MAX_RESULT_CHARS:
            truncated = body.get("result") if isinstance(body.get("result"), str) else text
            text = json.dumps(
                {"ok": self.ok, "result": f"{str(truncated)[:MAX_RESULT_CHARS]} (truncated)"},
                ensure_ascii=False,
            )
        return text


class ToolRegistry:
    """Holds the tools, validates their arguments, and bounds what they can do.

    Three rules are enforced here rather than trusted to each tool: only registered names run,
    arguments are checked against the declared schema, and every call has a timeout. The model
    chooses which tool to run and with what arguments; it never chooses who the student is.
    """

    def __init__(self, definitions: list[ToolDefinition] | None = None) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        for definition in definitions or []:
            self.register(definition)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def enabled(self) -> bool:
        return bool(self._tools)

    def register(self, definition: ToolDefinition) -> None:
        self._tools[definition.name] = definition
        logger.debug("tool registered", extra={"context": {"tool": definition.name}})

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(definition.spec for definition in self._tools.values())

    def describe(self) -> list[str]:
        return list(self._tools)

    async def dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        definition = self._tools.get(name)
        if definition is None:
            metrics.TOOL_CALLS.labels(tool=name, result="unknown").inc()
            return ToolResult(
                name=name,
                ok=False,
                payload={},
                error=(f"unknown tool {name!r}; available tools: {', '.join(sorted(self._tools))}"),
            )

        started = asyncio.get_running_loop().time()
        try:
            try:
                require_valid(definition.spec.parameters, arguments)
            except ValueError as exc:
                metrics.TOOL_CALLS.labels(tool=name, result="invalid").inc()
                return ToolResult(
                    name=name, ok=False, payload={}, error=f"invalid arguments: {exc}"
                )

            payload = await asyncio.wait_for(
                definition.handler(context, arguments), timeout=definition.timeout_seconds
            )
        except TimeoutError:
            metrics.TOOL_CALLS.labels(tool=name, result="timeout").inc()
            metrics.TOOL_LATENCY.labels(tool=name).observe(
                asyncio.get_running_loop().time() - started
            )
            logger.warning(
                "tool timed out", extra={"context": {"tool": name, "wa_id": context.wa_id}}
            )
            return ToolResult(
                name=name, ok=False, payload={}, error="the tool took too long to answer"
            )
        except ToolFailure as exc:
            metrics.TOOL_CALLS.labels(tool=name, result="empty").inc()
            metrics.TOOL_LATENCY.labels(tool=name).observe(
                asyncio.get_running_loop().time() - started
            )
            return ToolResult(name=name, ok=False, payload={}, error=str(exc))
        except Exception as exc:
            metrics.TOOL_CALLS.labels(tool=name, result="error").inc()
            metrics.TOOL_LATENCY.labels(tool=name).observe(
                asyncio.get_running_loop().time() - started
            )
            logger.warning(
                "tool raised",
                extra={
                    "context": {
                        "tool": name,
                        "wa_id": context.wa_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                },
            )
            return ToolResult(
                name=name, ok=False, payload={}, error=f"the tool failed: {type(exc).__name__}"
            )

        metrics.TOOL_LATENCY.labels(tool=name).observe(asyncio.get_running_loop().time() - started)
        metrics.TOOL_CALLS.labels(tool=name, result="success").inc()
        return ToolResult(name=name, ok=True, payload=payload)
