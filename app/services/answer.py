from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import Settings
from app.conversation.store import ConversationStore
from app.llm.pipeline import LLMPipeline, NoProviderAvailable
from app.llm.types import ChatMessage, LLMRequest, LLMResponse, ToolSpec
from app.rag.pipeline import RetrievalPipeline, RetrievalResult
from app.tools.context import ToolContext
from app.tools.registry import MAX_TOOL_CALLS_PER_RESPONSE, ToolRegistry

logger = logging.getLogger(__name__)

CONTEXT_PREAMBLE = "Course material for this question follows. Use it only if it answers the "
CONTEXT_ABSENT = (
    "No course material was found for this question. If the answer would need lecture notes, "
    "say that you could not find it rather than guessing."
)
TOOL_ERROR_PREAMBLE = (
    "The database tools did not return a usable answer. Explain the situation plainly."
)


@dataclass(frozen=True, slots=True)
class Answer:
    """One reply, with the reasoning trace kept for logs and metrics."""

    text: str
    provider: str = "none"
    model: str = "none"
    used_tools: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    tool_rounds: int = 0
    retrieved: int = 0
    degraded: bool = False
    error: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.text.strip().strip() != "" and self.error is None


class AnswerService:
    """Retrieve, then let the model call tools, then send the answer.

    The order matters. Retrieval happens first so the prompt already carries relevant material,
    which usually means the model needs no tools at all. Tools are only for the student's own
    records, and the number of rounds is capped so a confused model cannot loop: WhatsApp
    messages cost money and the student is waiting.
    """

    def __init__(
        self,
        settings: Settings,
        pipeline: LLMPipeline,
        *,
        retrieval: RetrievalPipeline | None = None,
        tools: ToolRegistry | None = None,
        conversations: ConversationStore | None = None,
    ) -> None:
        self._settings = settings
        self._pipeline = pipeline
        self._retrieval = retrieval
        self._tools = tools
        # An empty store is falsy, so this cannot be written as `conversations or ...`: doing that
        # would quietly discard a supplied store until it happened to be non-empty.
        self._conversations = (
            conversations if conversations is not None else ConversationStore(settings)
        )

    @property
    def enabled(self) -> bool:
        return self._settings.ai_enabled and self._pipeline.available

    def reset(self, wa_id: str) -> None:
        self._conversations.clear(wa_id)

    async def answer(
        self, question: str, context: ToolContext, *, http: httpx.AsyncClient
    ) -> Answer:
        if not self.enabled:
            return Answer(text="", error="no AI provider is configured")

        retrieval = await self._retrieve(question)
        messages = self._build_messages(question, context, retrieval.context)

        used_tools: list[str] = []
        tool_rounds = 0
        request = LLMRequest(
            messages=messages,
            tools=self._tool_specs(),
            temperature=self._settings.ai_temperature,
            max_tokens=self._settings.ai_max_tokens,
            timeout_seconds=self._settings.ai_request_timeout_seconds,
        )

        while True:
            try:
                response = await self._pipeline.complete(request, http=http)
            except NoProviderAvailable as exc:
                logger.warning(
                    "every provider tier failed",
                    extra={
                        "context": {
                            "wa_id": context.wa_id,
                            "attempts": [attempt.as_dict() for attempt in exc.attempts],
                        }
                    },
                )
                return Answer(
                    text="",
                    error="every AI provider tier failed",
                    used_tools=tuple(used_tools),
                    retrieved=len(retrieval.chunks),
                )

            # Once the budget is spent the tool calls are dropped and whatever the model said is
            # what the student gets, so a confused model cannot keep the turn open indefinitely.
            budget_spent = tool_rounds >= self._settings.ai_max_tool_rounds
            calls = () if budget_spent else response.tool_calls[:MAX_TOOL_CALLS_PER_RESPONSE]
            if not calls or not self._tools:
                return self._finish(
                    response,
                    question,
                    context,
                    used_tools=used_tools,
                    rounds=tool_rounds,
                    retrieved=len(retrieval.chunks),
                    sources=retrieval.sources,
                    degraded=budget_spent and bool(response.tool_calls),
                )

            tool_rounds += 1
            follow_up: list[ChatMessage] = [ChatMessage.assistant(response.text or "", calls)]
            for call in calls:
                result = await self._tools.dispatch(call.name, call.arguments, context)
                used_tools.append(call.name)
                if result.ok:
                    follow_up.append(
                        ChatMessage.tool_result(
                            call_id=call.id, name=call.name, content=result.as_json()
                        )
                    )
                else:
                    follow_up.append(
                        ChatMessage.tool_result(
                            call_id=call.id,
                            name=call.name,
                            content=f"{TOOL_ERROR_PREAMBLE} {result.error}",
                        )
                    )
            request = request.with_extra_messages(tuple(follow_up))

    def _finish(
        self,
        response: LLMResponse,
        question: str,
        context: ToolContext,
        *,
        used_tools: list[str],
        rounds: int,
        retrieved: int,
        sources: tuple[str, ...],
        degraded: bool,
    ) -> Answer:
        text = (response.text or "").strip()
        if not text:
            # A model can return tool calls with nothing to say. Fall back to what the tools
            # found, which is usually the actual answer, rather than sending an empty bubble.
            text = "I could not turn that into a reply. Try asking again in a different way."
            degraded = True
        self._conversations.record(
            context.wa_id,
            question,
            text,
            used_tools=tuple(used_tools),
            sources=sources,
        )
        return Answer(
            text=text,
            provider=response.provider,
            model=response.model,
            used_tools=tuple(dict.fromkeys(used_tools)),
            sources=sources,
            tool_rounds=rounds,
            retrieved=retrieved,
            degraded=degraded,
        )

    async def _retrieve(self, question: str) -> RetrievalResult:
        if self._retrieval is None:
            return RetrievalResult(reason="disabled")
        return await self._retrieval.retrieve(question)

    def _tool_specs(self) -> tuple[ToolSpec, ...]:
        if self._tools is None or not self._tools.enabled:
            return ()
        return self._tools.specs()

    def _build_messages(
        self, question: str, context: ToolContext, retrieved: str
    ) -> tuple[ChatMessage, ...]:
        """The system prompt, then prior exchanges, then the new question.

        The student's name goes in the system prompt rather than in front of the question, because
        the stored history is reused verbatim and decorating the question here would make the
        first message read differently from every follow up.
        """
        preamble = CONTEXT_PREAMBLE if retrieved else CONTEXT_ABSENT
        parts = [self._settings.ai_system_prompt, preamble]
        if retrieved:
            parts.append(retrieved)
        if context.display_name:
            parts.append(f"You are talking to {context.display_name}.")
        return (
            ChatMessage.system("\n\n".join(parts)),
            *self._conversations.history(context.wa_id),
            ChatMessage.user(question),
        )
