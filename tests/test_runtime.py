from __future__ import annotations

import logging
from typing import Any

import pytest
from app import runtime
from app.config import Settings

from conftest import _base_settings


def waking_database(failures: int) -> tuple[type, list[Any]]:
    """A Database stand-in whose first N connects are refused, as when a machine is waking."""
    created: list[Any] = []

    class WakingDatabase:
        def __init__(self, settings: Settings) -> None:
            self.attempts = 0
            self.closed = 0
            created.append(self)

        async def connect(self) -> None:
            self.attempts += 1
            if self.attempts <= failures:
                raise ConnectionRefusedError("the machine is still waking")

        async def ping(self) -> tuple[bool, float, str | None]:
            return True, 0.4, None

        async def close(self) -> None:
            self.closed += 1

    return WakingDatabase, created


async def test_a_database_that_wakes_late_is_retried_until_it_answers(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    database_type, created = waking_database(failures=2)
    monkeypatch.setattr(runtime, "Database", database_type)
    monkeypatch.setattr(runtime, "DATABASE_OPEN_RETRY_SECONDS", 0)

    with caplog.at_level(logging.WARNING, logger="app.runtime"):
        opened = await runtime._open_database(_base_settings())

    assert opened is created[0]
    assert opened.attempts == 3
    assert opened.closed == 2
    assert "retrying" in caplog.text


async def test_a_database_that_never_answers_leaves_the_tools_off(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    database_type, created = waking_database(failures=runtime.DATABASE_OPEN_ATTEMPTS)
    monkeypatch.setattr(runtime, "Database", database_type)
    monkeypatch.setattr(runtime, "DATABASE_OPEN_RETRY_SECONDS", 0)

    with caplog.at_level(logging.INFO, logger="app.runtime"):
        opened = await runtime._open_database(_base_settings())

    assert opened is None
    assert created[0].attempts == runtime.DATABASE_OPEN_ATTEMPTS
    assert "student tools are disabled" in caplog.text
