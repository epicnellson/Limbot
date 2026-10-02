from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from app import metrics
from app.config import Settings
from app.llm.types import ChatMessage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Turn:
    """One question and the answer that was sent back."""

    question: str
    answer: str
    asked_at: float
    used_tools: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()


class ConversationStore:
    """A bounded, expiring window of recent exchanges per WhatsApp number.

    Two limits, both required. The size cap bounds the prompt, and the cap on tracked senders
    bounds memory, so a bot that is flooded with numbers from one attacker cannot grow without
    limit. Entries are also dropped once they go quiet, because a student asking a follow up
    tomorrow deserves a fresh conversation rather than a stale one.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_turns = settings.conversation_max_messages // 2
        self._ttl = settings.conversation_ttl_seconds
        self._max_conversations = settings.conversation_max_conversations
        self._clock = clock
        self._turns: OrderedDict[str, list[Turn]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._turns)

    def record(
        self,
        wa_id: str,
        question: str,
        answer: str,
        *,
        used_tools: tuple[str, ...] = (),
        sources: tuple[str, ...] = (),
    ) -> None:
        now = self._clock()
        turns = self._turns.get(wa_id)
        if turns is None:
            turns = []
            self._turns[wa_id] = turns
        turns.append(
            Turn(
                question=question,
                answer=answer,
                asked_at=now,
                used_tools=used_tools,
                sources=sources,
            )
        )
        del turns[: max(0, len(turns) - self._max_turns)]
        if not turns:
            # A budget of zero or one message leaves nothing worth keeping, so the sender is
            # dropped rather than tracked with an empty window.
            self._turns.pop(wa_id, None)
            self._evict()
            return
        self._turns.move_to_end(wa_id)
        self._evict()

    def history(self, wa_id: str) -> tuple[ChatMessage, ...]:
        """Recent exchanges as chat messages, oldest first, unexpired only."""
        turns = self._live(wa_id)
        if not turns:
            return ()
        messages: list[ChatMessage] = []
        for turn in turns:
            if turn.question.strip():
                messages.append(ChatMessage.user(turn.question))
            if turn.answer.strip():
                messages.append(ChatMessage.assistant(turn.answer))
        return tuple(messages)

    def tools_used(self, wa_id: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(name for turn in self._live(wa_id) for name in turn.used_tools))

    def clear(self, wa_id: str) -> None:
        self._turns.pop(wa_id, None)

    def _live(self, wa_id: str) -> list[Turn]:
        turns = self._turns.get(wa_id)
        if not turns:
            return []
        cutoff = self._clock() - self._ttl
        unexpired = [turn for turn in turns if turn.asked_at >= cutoff]
        if not unexpired:
            self._turns.pop(wa_id, None)
            return []
        turns[:] = unexpired
        return unexpired

    def _evict(self) -> None:
        cutoff = self._clock() - self._ttl
        for wa_id in [key for key, turns in self._turns.items() if turns[-1].asked_at < cutoff]:
            self._turns.pop(wa_id, None)
        while len(self._turns) > self._max_conversations:
            evicted, _turns = self._turns.popitem(last=False)
            logger.debug("conversation evicted", extra={"context": {"wa_id": evicted}})
        metrics.CONVERSATIONS.set(len(self._turns))
