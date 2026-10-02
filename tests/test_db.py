from __future__ import annotations

from datetime import UTC, date, datetime, time
from decimal import Decimal
from typing import Any

import pytest
from app.config import Settings
from app.db.pool import SERVER_SETTINGS, Database
from app.db.students import StudentNotLinked, StudentRepository, day_number


def row(**fields: Any) -> dict[str, Any]:
    return fields


class FakePool:
    """Answers scripted rows and records every query, so SQL text can be asserted on.

    A scripted value may be a single row or a list. ``fetchrow`` and ``fetchval`` unwrap a
    one-item list, which is how a caller usually writes a single-row expectation.
    """

    def __init__(self, results: dict[str, Any] | None = None) -> None:
        self.results = results or {}
        self.queries: list[tuple[str, tuple[Any, ...]]] = []
        self.closed = False

    def _result(self, query: str, args: tuple[Any, ...]) -> Any:
        self.queries.append((query, args))
        for marker, value in self.results.items():
            if marker in query:
                return value
        return []

    async def fetchval(self, query: str, *args: Any) -> Any:
        value = self._result(query, args)
        if isinstance(value, list):
            return value[0] if value else None
        return value

    async def fetchrow(self, query: str, *args: Any) -> Any:
        value = self._result(query, args)
        if isinstance(value, list):
            return value[0] if value else None
        return value

    async def fetch(self, query: str, *args: Any) -> Any:
        value = self._result(query, args)
        if value is None:
            return []
        return value if isinstance(value, list) else [value]

    async def close(self) -> None:
        self.closed = True


class FakeAcquire:
    def __init__(self, pool: FakePool) -> None:
        self._pool = pool

    async def __aenter__(self) -> FakePool:
        return self._pool

    async def __aexit__(self, *args: object) -> None:
        return None


class FakePoolWithAcquire(FakePool):
    def acquire(self) -> FakeAcquire:
        return FakeAcquire(self)


def repo_for(
    settings: Settings, results: dict[str, Any] | None = None
) -> tuple[StudentRepository, FakePool]:
    pool = FakePoolWithAcquire(results)
    return StudentRepository(Database(settings, pool=pool)), pool


def test_the_pool_is_opened_read_only() -> None:
    assert SERVER_SETTINGS["default_transaction_read_only"] == "on"
    assert SERVER_SETTINGS["application_name"] == "limbot"


def test_an_unconfigured_database_does_not_connect(settings: Settings) -> None:
    database = Database(settings)

    assert database.configured is False
    assert database.connected is False
    assert database.describe()["read_only"] is True
    with pytest.raises(RuntimeError, match="not connected"):
        _ = database.pool


async def test_ping_reports_a_missing_pool(settings: Settings) -> None:
    reachable, _latency, error = await Database(settings).ping()
    assert reachable is False
    assert error is not None
    assert "not connected" in error


async def test_ping_reports_a_broken_pool(settings: Settings) -> None:
    class BrokenPool(FakePool):
        async def fetchval(self, query: str, *args: Any) -> Any:
            raise RuntimeError("connection refused")

    database = Database(settings, pool=BrokenPool())
    reachable, latency_ms, error = await database.ping()

    assert reachable is False
    assert latency_ms >= 0
    assert error is not None
    assert "connection refused" in error


async def test_close_is_safe_when_never_connected(settings: Settings) -> None:
    pool = FakePool()
    database = Database(settings, pool=pool)
    await database.close()
    assert pool.closed is True


def test_using_the_pool_before_connecting_fails_clearly(settings: Settings) -> None:
    with pytest.raises(RuntimeError, match="not connected"):
        _ = Database(settings).pool


async def test_an_unlinked_number_is_reported(settings: Settings) -> None:
    repo, _pool = repo_for(settings, {"FROM students WHERE wa_id": None})

    assert await repo.find_id_by_wa_id("15550009999") is None
    with pytest.raises(StudentNotLinked, match="15550009999"):
        await repo.require_id("15550009999")


async def test_a_linked_number_resolves_to_an_id(settings: Settings) -> None:
    repo, pool = repo_for(settings, {"FROM students WHERE wa_id": 7})
    assert await repo.require_id("15550001111") == 7
    assert pool.queries[0][1] == ("15550001111",)


async def test_profile_maps_decimal_gpa(settings: Settings) -> None:
    repo, _pool = repo_for(
        settings,
        {
            "FROM students": [
                row(
                    id=7,
                    full_name="Ada Lovelace",
                    email="ada@example.edu",
                    programme="BSc Computer Science",
                    year_of_study=2,
                    gpa=Decimal("3.72"),
                )
            ]
        },
    )

    profile = await repo.profile(7)

    assert profile.full_name == "Ada Lovelace"
    assert profile.gpa == 3.72
    assert profile.as_dict()["year_of_study"] == 2


async def test_profile_of_a_missing_student_raises(settings: Settings) -> None:
    repo, _pool = repo_for(settings, {"FROM students": []})
    with pytest.raises(StudentNotLinked, match="no longer exists"):
        await repo.profile(999)


async def test_courses_are_ordered_by_code(settings: Settings) -> None:
    repo, pool = repo_for(
        settings,
        {
            "FROM enrollments e": [
                row(code="CS2010", name="Data Structures", credits=15),
                row(code="MA1010", name="Linear Algebra", credits=None),
            ]
        },
    )

    courses = await repo.courses(7)

    assert [course.code for course in courses] == ["CS2010", "MA1010"]
    assert courses[0].as_dict() == {"code": "CS2010", "name": "Data Structures", "credits": 15}
    assert pool.queries[0][1] == (7,)


async def test_timetable_turns_day_numbers_into_names(settings: Settings) -> None:
    repo, pool = repo_for(
        settings,
        {
            "FROM timetable_entries": [
                row(
                    day_of_week=1,
                    start_time=time(9, 0),
                    end_time=time(10, 30),
                    room="B-104",
                    code="CS1010",
                    name="Introduction to Programming",
                )
            ]
        },
    )

    entries = await repo.timetable(7)

    assert entries[0].day == "monday"
    assert entries[0].start_time == "09:00"
    assert entries[0].end_time == "10:30"
    assert pool.queries[0][1] == (7, None)


async def test_timetable_passes_the_day_filter_as_a_parameter(settings: Settings) -> None:
    repo, pool = repo_for(settings, {"FROM timetable_entries": []})
    await repo.timetable(7, day_number("wednesday"))
    assert pool.queries[0][1] == (7, 3)


async def test_deadlines_keep_the_window_in_the_query(settings: Settings) -> None:
    repo, pool = repo_for(
        settings,
        {
            "FROM assignments a": [
                row(
                    title="Hash table assignment",
                    code="CS2010",
                    due_at=datetime(2026, 10, 1, 23, 59, tzinfo=UTC),
                    weight_pct=Decimal("15.00"),
                    status="not_started",
                    days_until_due=3,
                )
            ]
        },
    )

    deadlines = await repo.deadlines(7, within_days=30)

    assert deadlines[0].title == "Hash table assignment"
    assert deadlines[0].due_at == "2026-10-01T23:59Z"
    assert deadlines[0].weight_pct == 15.0
    assert deadlines[0].as_dict()["days_until_due"] == 3
    assert pool.queries[0][1] == (7, "30")


async def test_grades_are_capped_by_the_limit_parameter(settings: Settings) -> None:
    repo, pool = repo_for(
        settings,
        {
            "FROM submissions s": [
                row(
                    title="Linked list implementation",
                    grade=Decimal("78.00"),
                    feedback="Correct, but the recursion is hard to follow.",
                    submitted_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
                    code="CS2010",
                )
            ]
        },
    )

    grades = await repo.grades(7, limit=5)

    assert grades[0].grade == 78.0
    assert grades[0].out_of == 100.0
    assert grades[0].graded_at == "2026-09-20T12:00Z"
    assert pool.queries[0][1] == (7, 5)


async def test_exams_expose_room_and_seat(settings: Settings) -> None:
    repo, _pool = repo_for(
        settings,
        {
            "FROM exam_registrations": [
                row(
                    code="CS2010",
                    exam_date=date(2026, 10, 12),
                    starts_at=time(9, 0),
                    room="A-100",
                    seat="A-14",
                    days_until_exam=14,
                )
            ]
        },
    )

    exams = await repo.exams(7)

    assert exams[0].exam_date == "2026-10-12"
    assert exams[0].starts_at == "09:00"
    assert exams[0].seat == "A-14"
    assert exams[0].days_until_exam == 14


async def test_summary_runs_every_query_for_one_student(settings: Settings) -> None:
    repo, pool = repo_for(
        settings,
        {
            "FROM students WHERE wa_id": 7,
            "FROM students": [
                row(
                    id=7,
                    full_name="Ada Lovelace",
                    email="ada@example.edu",
                    programme="BSc Computer Science",
                    year_of_study=2,
                    gpa=Decimal("3.72"),
                )
            ],
        },
    )

    view = await repo.summary("15550001111")

    assert view.profile.full_name == "Ada Lovelace"
    assert view.courses == []
    assert view.deadlines == []
    assert len(pool.queries) == 7
    assert {args[0] for _query, args in pool.queries[1:]} == {7}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        (7, 7),
        ("monday", 1),
        ("Monday", 1),
        ("tue", 2),
        ("thursday", 4),
        ("sunday", 7),
        (0, None),
        (8, None),
        ("funday", None),
        ("", None),
        (None, None),
        (True, None),
    ],
)
def test_day_number_accepts_names_and_numbers(value: object, expected: int | None) -> None:
    assert day_number(value) == expected
