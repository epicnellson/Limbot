from __future__ import annotations

import logging
import time
from collections import deque

logger = logging.getLogger(__name__)


class MessageDeduplicator:
    """Remembers recently processed message ids so redelivered webhooks are dropped.

    Meta redelivers a webhook whenever it does not get a 2xx, and can deliver twice around
    deploys. Entries expire after ``ttl_seconds`` and the set is capped at ``max_entries``
    so a traffic spike cannot grow it without bound. Every method is synchronous and free of
    await points, which makes it safe on a single event loop without a lock.
    """

    def __init__(self, *, ttl_seconds: int, max_entries: int) -> None:
        if ttl_seconds < 1 or max_entries < 1:
            raise ValueError("ttl_seconds and max_entries must be >= 1")
        self._ttl = float(ttl_seconds)
        self._max_entries = max_entries
        self._seen: set[str] = set()
        self._expiry: deque[tuple[float, str]] = deque()
        self._duplicates = 0

    @property
    def size(self) -> int:
        return len(self._seen)

    @property
    def duplicates(self) -> int:
        return self._duplicates

    def seen(self, message_id: str) -> bool:
        """Return True when this id was already processed. Otherwise record it."""
        if not message_id:
            return False
        now = time.monotonic()
        while self._expiry and self._expiry[0][0] <= now:
            _, expired = self._expiry.popleft()
            self._seen.discard(expired)
        if message_id in self._seen:
            self._duplicates += 1
            logger.info("duplicate message dropped", extra={"context": {"message_id": message_id}})
            return True
        self._seen.add(message_id)
        self._expiry.append((now + self._ttl, message_id))
        while len(self._seen) > self._max_entries:
            _, evicted = self._expiry.popleft()
            self._seen.discard(evicted)
        return False
