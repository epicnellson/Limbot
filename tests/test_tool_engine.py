from __future__ import annotations

import json

import pytest
from app.tools.tool_engine import (
    check_assignment_deadlines,
    check_student_timetable,
    close_tool_db,
    execute_tool,
    init_mock_db,
)


@pytest.fixture
def db_path(tmp_path: pytest.TempPathFactory) -> str:
    path = str(tmp_path / "limbot_tools.db")
    init_mock_db(path)
    yield path
    close_tool_db(path)


def test_init_mock_db_creates_and_seeds_both_tables(db_path: str) -> None:
    timetable = check_student_timetable("S1001", db_path=db_path)
    assert timetable["count"] == 5
    assert {entry["course_code"] for entry in timetable["timetable"]} == {
        "CS101",
        "MATH201",
        "ENG110",
    }

    deadlines = check_assignment_deadlines("CS101", db_path=db_path)
    assert deadlines["count"] == 2
    assert deadlines["assignments"][0]["title"].startswith("Homework 1")


def test_init_mock_db_is_idempotent(db_path: str) -> None:
    init_mock_db(db_path)
    assert check_student_timetable("S1001", db_path=db_path)["count"] == 5
    assert check_assignment_deadlines("CS101", db_path=db_path)["count"] == 2


def test_timetable_day_filter_returns_only_that_weekday(db_path: str) -> None:
    result = check_student_timetable("S1001", "monday", db_path=db_path)
    assert result["count"] == 1
    assert result["day"] == "monday"
    assert result["timetable"][0]["venue"] == "Room A101"
    assert result["timetable"][0]["start_time"] == "09:00"


def test_timetable_accepts_day_abbreviations(db_path: str) -> None:
    full = check_student_timetable("S1001", "friday", db_path=db_path)
    short = check_student_timetable("S1001", "fri", db_path=db_path)
    assert short["day"] == "friday"
    assert [entry["course_code"] for entry in short["timetable"]] == [
        entry["course_code"] for entry in full["timetable"]
    ]


def test_timetable_for_an_unknown_student_is_empty_with_a_note(db_path: str) -> None:
    result = check_student_timetable("NOPE", db_path=db_path)
    assert result["count"] == 0
    assert result["timetable"] == []
    assert "No classes" in result["note"]


def test_timetable_rejects_a_non_weekday(db_path: str) -> None:
    with pytest.raises(ValueError, match="not a weekday name"):
        check_student_timetable("S1001", "funday", db_path=db_path)


def test_deadlines_include_how_many_days_remain(db_path: str) -> None:
    result = check_assignment_deadlines("CS101", db_path=db_path)
    assert all("days_until_due" in assignment for assignment in result["assignments"])
    assert result["assignments"][0]["days_until_due"] <= result["assignments"][1]["days_until_due"]
    assert all("requirements" in assignment for assignment in result["assignments"])


def test_deadlines_match_course_code_case_insensitively(db_path: str) -> None:
    assert check_assignment_deadlines("cs101", db_path=db_path)["count"] == 2


def test_deadlines_for_an_unknown_course_are_empty(db_path: str) -> None:
    result = check_assignment_deadlines("HIST101", db_path=db_path)
    assert result["count"] == 0
    assert "No assignments" in result["note"]


async def test_execute_tool_returns_a_json_string_for_the_model(db_path: str) -> None:
    text = await execute_tool(
        "check_student_timetable",
        {"student_id": "S1001", "day": "monday"},
        db_path=db_path,
    )
    payload = json.loads(text)
    assert payload["ok"] is True
    assert payload["tool"] == "check_student_timetable"
    assert payload["count"] == 1
    assert payload["timetable"][0]["venue"] == "Room A101"


async def test_execute_tool_serialises_deadline_rows(db_path: str) -> None:
    text = await execute_tool(
        "check_assignment_deadlines", {"course_code": "CS101"}, db_path=db_path
    )
    payload = json.loads(text)
    assert payload["ok"] is True
    assert payload["count"] == 2
    assert payload["assignments"][0]["title"]


async def test_execute_tool_reports_an_unknown_tool_without_raising(db_path: str) -> None:
    text = await execute_tool("rm -rf /", {}, db_path=db_path)
    payload = json.loads(text)
    assert payload["ok"] is False
    assert "unknown tool" in payload["error"]
    assert "check_student_timetable" in payload["error"]


async def test_execute_tool_refuses_arguments_with_the_wrong_type(db_path: str) -> None:
    text = await execute_tool("check_student_timetable", {"student_id": 1001}, db_path=db_path)
    payload = json.loads(text)
    assert payload["ok"] is False
    assert "student_id" in payload["error"]


async def test_execute_tool_refuses_unknown_arguments(db_path: str) -> None:
    text = await execute_tool(
        "check_student_timetable",
        {"student_id": "S1001", "day": "monday", "inject": "SELECT * FROM timetables"},
        db_path=db_path,
    )
    payload = json.loads(text)
    assert payload["ok"] is False
    assert "inject" in payload["error"]


async def test_execute_tool_refuses_non_object_arguments(db_path: str) -> None:
    text = await execute_tool("check_student_timetable", "S1001", db_path=db_path)  # type: ignore[arg-type]
    payload = json.loads(text)
    assert payload["ok"] is False
    assert "JSON object" in payload["error"]


async def test_execute_tool_wraps_a_failing_query_instead_of_raising(
    tmp_path: pytest.TempPathFactory,
) -> None:
    empty_path = str(tmp_path / "empty.db")
    text = await execute_tool(
        "check_student_timetable",
        {"student_id": "S1001"},
        db_path=empty_path,
    )
    payload = json.loads(text)
    assert payload["ok"] is False
    assert "failed" in payload["error"]
