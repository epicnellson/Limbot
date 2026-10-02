from __future__ import annotations

import asyncio

import pytest
from app.core.tasks import TaskManager


async def test_concurrency_is_capped() -> None:
    manager = TaskManager(max_concurrency=2)
    release = asyncio.Event()

    async def worker() -> None:
        await release.wait()

    tasks = [manager.submit(worker()) for _ in range(6)]
    await asyncio.sleep(0.01)
    assert manager.inflight == 2
    assert manager.pending == 6

    release.set()
    await asyncio.wait([task for task in tasks if task is not None])
    assert manager.largest_inflight == 2
    assert manager.inflight == 0


async def test_submit_runs_the_coroutine() -> None:
    manager = TaskManager(max_concurrency=1)
    done = asyncio.Event()

    async def work() -> None:
        done.set()

    task = manager.submit(work(), name="unit")
    assert task is not None
    await task
    assert done.is_set()
    assert manager.pending == 0
    assert manager.inflight == 0


async def test_submit_is_refused_after_stop_accepting() -> None:
    manager = TaskManager(max_concurrency=1)
    manager.stop_accepting()
    ran = False

    async def work() -> None:
        nonlocal ran
        ran = True

    assert manager.submit(work()) is None
    await asyncio.sleep(0)
    assert ran is False


async def test_failures_are_contained() -> None:
    manager = TaskManager(max_concurrency=1)

    async def boom() -> None:
        raise ValueError("kaboom")

    task = manager.submit(boom())
    assert task is not None
    await task
    assert task.exception() is None


async def test_drain_waits_for_inflight_work() -> None:
    manager = TaskManager(max_concurrency=2)
    finished = False

    async def work() -> None:
        nonlocal finished
        await asyncio.sleep(0.05)
        finished = True

    manager.submit(work())
    await asyncio.sleep(0.01)
    await manager.drain(grace_seconds=2)
    assert finished is True
    assert manager.pending == 0


async def test_drain_cancels_work_that_overruns() -> None:
    manager = TaskManager(max_concurrency=1)
    cancelled = False

    async def work() -> None:
        nonlocal cancelled
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled = True
            raise

    manager.submit(work())
    await asyncio.sleep(0.01)
    await manager.drain(grace_seconds=0.05)
    assert cancelled is True


def test_invalid_concurrency_is_rejected() -> None:
    with pytest.raises(ValueError):
        TaskManager(max_concurrency=0)
