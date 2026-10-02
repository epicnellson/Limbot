from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from app.db.students import StudentRepository
from app.llm.types import ToolSpec
from app.tools.context import ToolContext
from app.tools.registry import (
    ToolDefinition,
    ToolFailure,
    ToolRegistry,
    ToolResult,
)
from app.tools.schema import SchemaError, require_valid, validate_arguments
from app.tools.student import NOT_LINKED, build_student_registry

CONTEXT = ToolContext(wa_id="15550001111", display_name="Ada")


def make_spec(**overrides: Any) -> ToolSpec:
    payload: dict[str, Any] = {
        "name": "echo",
        "description": "Echo the arguments back.",
        "parameters": {
            "type": "object",
            "properties": {
                "day": {"type": "string", "enum": ["monday", "tuesday"]},
                "within_days": {"type": "integer", "minimum": 1, "maximum": 90},
            },
            "required": ["day"],
            "additionalProperties": False,
        },
    }
    payload.update(overrides)
    return ToolSpec(**payload)


def test_valid_arguments_pass() -> None:
    schema = make_spec().parameters
    assert validate_arguments(schema, {"day": "monday", "within_days": 7}) == []
    assert require_valid(schema, {"day": "monday"}) == {"day": "monday"}


def test_a_missing_required_argument_is_reported() -> None:
    problems = validate_arguments(make_spec().parameters, {"within_days": 7})
    assert [str(problem) for problem in problems] == ["day: is required"]


def test_an_unknown_argument_is_refused() -> None:
    problems = validate_arguments(make_spec().parameters, {"day": "monday", "student_id": 1})
    assert [str(problem) for problem in problems] == ["student_id: is not an accepted argument"]


def test_an_enum_violation_lists_the_options() -> None:
    problems = validate_arguments(make_spec().parameters, {"day": "funday"})
    assert "must be one of: monday, tuesday" in str(problems[0])


def test_numeric_bounds_are_enforced() -> None:
    schema = make_spec().parameters
    assert "within_days: must be at most 90" in str(
        validate_arguments(schema, {"day": "monday", "within_days": 1000})[0]
    )
    assert "within_days: must be at least 1" in str(
        validate_arguments(schema, {"day": "monday", "within_days": 0})[0]
    )


def test_wrong_types_are_reported() -> None:
    schema = make_spec().parameters
    assert "within_days: must be of type integer" in str(
        validate_arguments(schema, {"day": "monday", "within_days": "soon"})[0]
    )
    assert "day: must be of type string" in str(validate_arguments(schema, {"day": 3})[0])


def test_booleans_are_not_integers() -> None:
    problems = validate_arguments(make_spec().parameters, {"day": "monday", "within_days": True})
    assert "must be of type integer" in str(problems[0])


def test_arrays_are_checked_item_by_item() -> None:
    schema = {
        "type": "object",
        "properties": {
            "days": {"type": "array", "items": {"type": "string", "enum": ["mon", "tue"]}}
        },
    }
    problems = validate_arguments(schema, {"days": ["mon", "wed"]})
    assert problems == [] or "days[1]" in str(problems[0])


def test_nested_objects_are_validated() -> None:
    schema = {
        "type": "object",
        "properties": {
            "filter": {
                "type": "object",
                "properties": {"course": {"type": "string"}},
                "required": ["course"],
                "additionalProperties": False,
            }
        },
    }
    problems = validate_arguments(schema, {"filter": {}})
    assert [str(problem) for problem in problems] == ["filter.course: is required"]


def test_nested_objects_reject_unknown_keys() -> None:
    schema = {
        "type": "object",
        "properties": {
            "filter": {
                "type": "object",
                "properties": {"course": {"type": "string"}},
                "additionalProperties": False,
            }
        },
    }
    problems = validate_arguments(schema, {"filter": {"course": "CS2010", "wa_id": "1"}})
    assert "filter.wa_id: is not an accepted argument" in str(problems[0])


def test_require_valid_raises_with_every_problem() -> None:
    with pytest.raises(SchemaError, match="day: is required"):
        require_valid(make_spec().parameters, {})

    with pytest.raises(SchemaError, match="arguments must be a JSON object"):
        require_valid(make_spec().parameters, ["not", "a", "dict"])  # type: ignore[arg-type]


def test_a_schema_without_properties_is_rejected() -> None:
    problems = validate_arguments({"type": "object", "properties": "nope"}, {})
    assert str(problems[0]) == "$: schema properties must be an object"


async def test_dispatch_returns_the_handler_payload() -> None:
    async def handler(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"seen": arguments, "wa_id": context.wa_id}

    registry = ToolRegistry([ToolDefinition(spec=make_spec(), handler=handler)])
    result = await registry.dispatch("echo", {"day": "monday"}, CONTEXT)

    assert result.ok is True
    assert result.payload == {"seen": {"day": "monday"}, "wa_id": "15550001111"}


async def test_dispatch_refuses_an_unknown_tool() -> None:
    registry = ToolRegistry()
    result = await registry.dispatch("drop_database", {}, CONTEXT)

    assert result.ok is False
    assert "unknown tool 'drop_database'" in (result.error or "")


async def test_dispatch_refuses_invalid_arguments() -> None:
    calls = 0

    async def handler(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {}

    registry = ToolRegistry([ToolDefinition(spec=make_spec(), handler=handler)])
    result = await registry.dispatch("echo", {"day": "funday"}, CONTEXT)

    assert result.ok is False
    assert "invalid arguments" in (result.error or "")
    assert calls == 0


async def test_dispatch_times_a_slow_tool_out() -> None:
    async def slow(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(5)
        return {}

    registry = ToolRegistry([ToolDefinition(spec=make_spec(), handler=slow, timeout_seconds=0.01)])
    result = await registry.dispatch("echo", {"day": "monday"}, CONTEXT)

    assert result.ok is False
    assert result.error == "the tool took too long to answer"


async def test_dispatch_reports_a_failing_tool_without_raising() -> None:
    async def boom(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("connection reset by peer")

    registry = ToolRegistry([ToolDefinition(spec=make_spec(), handler=boom)])
    result = await registry.dispatch("echo", {"day": "monday"}, CONTEXT)

    assert result.ok is False
    assert result.error == "the tool failed: RuntimeError"
    assert "connection reset" not in (result.error or "")


async def test_a_tool_failure_reaches_the_model_verbatim() -> None:
    async def empty(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        raise ToolFailure("no grades have been released yet")

    registry = ToolRegistry([ToolDefinition(spec=make_spec(), handler=empty)])
    result = await registry.dispatch("echo", {"day": "monday"}, CONTEXT)

    assert result.ok is False
    assert result.error == "no grades have been released yet"
    assert json.loads(result.as_json())["error"] == "no grades have been released yet"


def test_a_result_is_serialised_for_the_model() -> None:
    result = ToolResult(name="get_timetable", ok=True, payload={"entries": [{"day": "monday"}]})
    body = json.loads(result.as_json())

    assert body["ok"] is True
    assert body["entries"] == [{"day": "monday"}]


def test_a_huge_result_is_truncated() -> None:
    result = ToolResult(name="get_grades", ok=True, payload={"text": "x" * 9000})
    body = json.loads(result.as_json())

    assert body["ok"] is True
    assert "truncated" in body["result"]
    assert len(body["result"]) < 5000


def test_the_registry_lists_its_tools() -> None:
    registry = ToolRegistry()

    assert registry.enabled is False
    assert registry.specs() == ()
    assert registry.describe() == []

    async def handler(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        return {}

    registry.register(ToolDefinition(spec=make_spec(), handler=handler))
    assert registry.enabled is True
    assert registry.names == ("echo",)
    assert [spec.name for spec in registry.specs()] == ["echo"]


def test_duplicate_registration_replaces_the_handler() -> None:
    async def first(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"which": "first"}

    async def second(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"which": "second"}

    registry = ToolRegistry(
        [
            ToolDefinition(spec=make_spec(), handler=first),
            ToolDefinition(spec=make_spec(), handler=second),
        ]
    )
    assert len(registry) == 1
    assert registry.describe() == ["echo"]


class StubRepository(StudentRepository):
    """A repository that never touches Postgres, for exercising the tool wiring."""

    def __init__(self, *, linked: bool = True) -> None:
        self._linked = linked
        self.queries: list[tuple[str, int]] = []

    async def require_id(self, wa_id: str) -> int:
        self.queries.append(("require_id", 0))
        if not self._linked:
            from app.db.students import StudentNotLinked

            raise StudentNotLinked(wa_id)
        return 7

    async def profile(self, student_id: int) -> Any:
        from app.db.students import StudentProfile

        self.queries.append(("profile", student_id))
        return StudentProfile(student_id, "Ada Lovelace", "ada@example.edu", "BSc CS", 2, 3.72)

    async def courses(self, student_id: int) -> list[Any]:
        from app.db.students import Course

        self.queries.append(("courses", student_id))
        return [Course("CS2010", "Data Structures", 15)]

    async def timetable(self, student_id: int, day: int | None = None) -> list[Any]:
        from app.db.students import DAYS, TimetableEntry

        self.queries.append(("timetable", student_id))
        if day is None:
            return []
        return [
            TimetableEntry(
                day=DAYS[day - 1],
                day_number=day,
                start_time="09:00",
                end_time="10:30",
                course_code="CS2010",
                course_name="Data Structures",
                room="B-210",
            )
        ]

    async def deadlines(self, student_id: int, within_days: int = 21) -> list[Any]:
        self.queries.append(("deadlines", student_id))
        return []

    async def grades(self, student_id: int, limit: int = 10) -> list[Any]:
        self.queries.append(("grades", student_id))
        return []

    async def exams(self, student_id: int, within_days: int = 60) -> list[Any]:
        self.queries.append(("exams", student_id))
        return []


def student_registry(*, linked: bool = True) -> tuple[ToolRegistry, StubRepository]:
    repository = StubRepository(linked=linked)
    return build_student_registry(repository, timeout_seconds=0.5), repository


def test_the_six_student_tools_are_exposed() -> None:
    registry, _repository = student_registry()

    assert set(registry.names) == {
        "get_student_profile",
        "list_courses",
        "get_timetable",
        "get_upcoming_deadlines",
        "get_recent_grades",
        "get_exam_schedule",
    }


def test_no_tool_accepts_an_identity_argument() -> None:
    registry, _repository = student_registry()

    for spec in registry.specs():
        properties = set(spec.parameters.get("properties", {}))
        assert "student_id" not in properties
        assert "wa_id" not in properties
        assert spec.parameters.get("additionalProperties") is False


async def test_the_profile_tool_uses_the_verified_identity() -> None:
    registry, repository = student_registry()
    result = await registry.dispatch("get_student_profile", {}, CONTEXT)

    assert result.ok is True
    assert result.payload["student"]["full_name"] == "Ada Lovelace"
    assert repository.queries == [("require_id", 0), ("profile", 7)]


async def test_an_unlinked_number_produces_a_helpful_error() -> None:
    registry, _repository = student_registry(linked=False)
    result = await registry.dispatch("get_student_profile", {}, CONTEXT)

    assert result.ok is False
    assert result.error == NOT_LINKED


async def test_the_timetable_tool_translates_a_day_name() -> None:
    registry, repository = student_registry()
    result = await registry.dispatch("get_timetable", {"day": "wednesday"}, CONTEXT)

    assert result.ok is True
    assert result.payload["entries"][0]["start_time"] == "09:00"
    assert ("timetable", 7) in repository.queries


async def test_the_timetable_tool_accepts_a_three_letter_day() -> None:
    registry, _repository = student_registry()
    result = await registry.dispatch("get_timetable", {"day": "wed"}, CONTEXT)
    assert result.payload["entries"][0]["day"] == "wednesday"


async def test_the_timetable_tool_rejects_a_nonsense_day() -> None:
    registry, _repository = student_registry()
    result = await registry.dispatch("get_timetable", {"day": "someday"}, CONTEXT)

    assert result.ok is False
    assert "invalid arguments" in (result.error or "")
    assert "must be one of" in (result.error or "")


async def test_an_empty_timetable_explains_itself() -> None:
    registry, _repository = student_registry()
    result = await registry.dispatch("get_timetable", {}, CONTEXT)

    assert result.ok is True
    assert result.payload["count"] == 0
    assert "No classes" in result.payload["note"]


async def test_empty_collections_say_so_instead_of_returning_nothing() -> None:
    registry, _repository = student_registry()

    deadlines = await registry.dispatch("get_upcoming_deadlines", {}, CONTEXT)
    grades = await registry.dispatch("get_recent_grades", {}, CONTEXT)
    exams = await registry.dispatch("get_exam_schedule", {}, CONTEXT)

    assert deadlines.payload["count"] == 0
    assert "21 days" in deadlines.payload["note"]
    assert grades.payload["note"] == "No graded work has been released yet."
    assert exams.payload["note"] == "No exams are scheduled in the next 60 days."


async def test_window_arguments_are_honoured() -> None:
    registry, _repository = student_registry()
    result = await registry.dispatch("get_exam_schedule", {"within_days": 30}, CONTEXT)
    assert "30 days" in result.payload["note"]


async def test_out_of_range_windows_are_refused_before_the_query() -> None:
    registry, _repository = student_registry()
    result = await registry.dispatch("get_exam_schedule", {"within_days": 5000}, CONTEXT)

    assert result.ok is False
    assert "must be at most 180" in (result.error or "")
