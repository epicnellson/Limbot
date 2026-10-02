from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Coroutine
from typing import Any

from app import metrics

logger = logging.getLogger(__name__)

TaskResult = "completed"


class TaskManager:
    """Runs detached coroutines off the request path with a bounded number of workers.

    The webhook handler must answer Meta before doing any real work, so it hands the payload
    here and returns. Concurrency is capped to keep outbound Graph API calls inside the
    Cloud API rate limits, and :meth:`drain` lets in-flight work finish on shutdown.
    """

    def __init__(self, *, max_concurrency: int, name: str = "limbot") -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._name = name
        self._max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._tasks: set[asyncio.Task[None]] = set()
        self._inflight = 0
        self._accepting = True
        self._largest_inflight = 0

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def inflight(self) -> int:
        return self._inflight

    @property
    def largest_inflight(self) -> int:
        return self._largest_inflight

    @property
    def pending(self) -> int:
        return len(self._tasks)

    @property
    def accepting(self) -> bool:
        return self._accepting

    def submit(
        self, coro: Coroutine[Any, Any, Any], *, name: str | None = None
    ) -> asyncio.Task[None] | None:
        """Schedule ``coro`` and return its task, or ``None`` once shutdown has started."""
        if not self._accepting:
            coro.close()
            metrics.BACKGROUND_TASKS.labels(result="rejected").inc()
            return None
        task = asyncio.create_task(self._run(coro), name=name or f"{self._name}-task")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        metrics.BACKGROUND_TASKS.labels(result="submitted").inc()
        return task

    async def _run(self, coro: Coroutine[Any, Any, Any]) -> None:
        async with self._semaphore:
            self._inflight += 1
            self._largest_inflight = max(self._largest_inflight, self._inflight)
            metrics.BACKGROUND_TASKS_INFLIGHT.set(self._inflight)
            started = time.perf_counter()
            try:
                await coro
            except asyncio.CancelledError:
                metrics.BACKGROUND_TASKS.labels(result="cancelled").inc()
                raise
            except Exception:
                metrics.BACKGROUND_TASKS.labels(result="failed").inc()
                logger.exception("background task failed", extra={"context": {"task": self._name}})
            else:
                metrics.BACKGROUND_TASKS.labels(result=TaskResult).inc()
            finally:
                self._inflight -= 1
                metrics.BACKGROUND_TASKS_INFLIGHT.set(self._inflight)
                metrics.BACKGROUND_TASK_DURATION.observe(time.perf_counter() - started)

    def stop_accepting(self) -> None:
        """Refuse new work. In-flight tasks keep running."""
        self._accepting = False

    async def drain(self, grace_seconds: float) -> None:
        """Stop accepting work, wait up to ``grace_seconds``, then cancel the stragglers."""
        self.stop_accepting()
        if not self._tasks:
            return
        pending = set(self._tasks)
        logger.info("draining background tasks", extra={"context": {"pending": len(pending)}})
        _, unfinished = await asyncio.wait(pending, timeout=grace_seconds)
        if unfinished:
            logger.warning(
                "cancelling background tasks after drain timeout",
                extra={
                    "context": {
                        "cancelled": len(unfinished),
                        "grace_seconds": grace_seconds,
                    }
                },
            )
            for task in unfinished:
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*unfinished, return_exceptions=True)
