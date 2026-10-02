"""A small SQLite-backed tool engine that connects model tool calls to local queries.

This module is self-contained on purpose: it owns its own SQLite database (created and seeded
by :func:`init_mock_db`) and its own argument models, so it can be lifted out and used for a
demo or a stub backend without dragging the production Postgres registry along. The tool
definitions it exposes follow the OpenAI function-calling shape, which is what the LLM pipeline
already speaks through :class:`app.llm.types.ToolSpec`.

Execution model: the sync SQLite functions run in a worker thread (``asyncio.to_thread``) so a
slow disk never blocks the event loop, and every failure is turned into a JSON string the model
can read, never an unhandled exception.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.llm.types import ToolSpec

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "limbot_tools.db"

DAY_NAMES = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
_DAY_NUMBERS = {name: index for index, name in enumerate(DAY_NAMES, start=1)}
_DAY_NUMBERS.update({name[:3]: index for index, name in enumerate(DAY_NAMES, start=1)})

_SCHEMA_TIMETABLES = """
CREATE TABLE IF NOT EXISTS timetables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    course_code TEXT NOT NULL,
    course_name TEXT NOT NULL,
    day TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    venue TEXT NOT NULL
)
"""

_SCHEMA_ASSIGNMENTS = """
CREATE TABLE IF NOT EXISTS assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code TEXT NOT NULL,
    title TEXT NOT NULL,
    due_date TEXT NOT NULL,
    weight_pct REAL NOT NULL,
    requirements TEXT
)
"""

_INSERT_TIMETABLE = """
INSERT INTO timetables (student_id, course_code, course_name, day, start_time, end_time, venue)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_ASSIGNMENT = """
INSERT INTO assignments (course_code, title, due_date, weight_pct, requirements)
VALUES (?, ?, ?, ?, ?)
"""

_SELECT_TIMETABLE = """
SELECT course_code, course_name, day, start_time, end_time, venue
  FROM timetables
 WHERE student_id = ?
   AND (? IS NULL OR LOWER(day) = LOWER(?))
"""

_SELECT_ASSIGNMENTS = """
SELECT course_code, title, due_date, weight_pct, requirements
  FROM assignments
 WHERE UPPER(course_code) = UPPER(?)
 ORDER BY due_date
"""

_SAMPLE_TIMETABLE_ROWS: tuple[tuple[str, str, str, str, str, str, str], ...] = (
    ("S1001", "CS101", "Data Structures & Algorithms", "monday", "09:00", "10:30", "Room A101"),
    ("S1001", "MATH201", "Linear Algebra", "tuesday", "11:00", "12:30", "Room B205"),
    ("S1001", "CS101", "Data Structures & Algorithms", "wednesday", "14:00", "15:30", "Lab 3"),
    ("S1001", "ENG110", "Technical Writing", "thursday", "10:00", "11:00", "Room C10"),
    ("S1001", "MATH201", "Linear Algebra", "friday", "09:00", "10:30", "Room B205"),
    ("S1002", "CS101", "Data Structures & Algorithms", "monday", "09:00", "10:30", "Room A101"),
    ("S1002", "CS201", "Operating Systems", "tuesday", "14:00", "15:30", "Room A102"),
    ("S1002", "PHY101", "Classical Mechanics", "friday", "11:00", "12:30", "Room D301"),
)

_SAMPLE_ASSIGNMENTS: tuple[tuple[str, str, int, int, str], ...] = (
    (
        "CS101",
        "Homework 1: sorting and complexity",
        5,
        10,
        "Submit a single PDF with the written analysis and the code listing.",
    ),
    (
        "CS101",
        "Project milestone 1: design document",
        12,
        20,
        "Describe the architecture, the module breakdown and the test plan.",
    ),
    ("MATH201", "Problem set 4", 3, 8, "Work the exercises from chapter 4 and show full working."),
    (
        "CS201",
        "Kernel lab 2: scheduling",
        7,
        15,
        "Implement the round-robin scheduler and run the provided benchmarks.",
    ),
    (
        "ENG110",
        "Essay: first draft",
        9,
        15,
        "A 1200-word draft on a topic from the reading list, with citations.",
    ),
)

ToolHandler = Callable[..., dict[str, Any]]


class TimetableArguments(BaseModel):
    """Arguments accepted by :func:`check_student_timetable`."""

    model_config = ConfigDict(extra="forbid")

    student_id: str = Field(description="The student identifier, e.g. 'S1001'.")
    day: (
        Literal[
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "mon",
            "tue",
            "wed",
            "thu",
            "fri",
            "sat",
            "sun",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "Weekday to filter on, as a full name or a three-letter abbreviation; omit it for "
            "the whole week."
        ),
    )


class DeadlineArguments(BaseModel):
    """Arguments accepted by :func:`check_assignment_deadlines`."""

    model_config = ConfigDict(extra="forbid")

    course_code: str = Field(
        description="The course code to read the assignment deadlines for, e.g. 'CS101'."
    )


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="check_student_timetable",
        description=(
            "Read a student's class timetable from the local SQLite database: the weekday, "
            "lecture times and venue for each course. Pass day for a single weekday or omit it "
            "for the whole week."
        ),
        parameters=TimetableArguments.model_json_schema(),
    ),
    ToolSpec(
        name="check_assignment_deadlines",
        description=(
            "Read the assignment deadlines for a course from the local SQLite database: title, "
            "due date, how many days remain and the submission requirements."
        ),
        parameters=DeadlineArguments.model_json_schema(),
    ),
)

TOOL_NAMES: tuple[str, ...] = tuple(tool.name for tool in TOOLS)

# SQLite connections are cached per path (with ``check_same_thread=False`` so the router can
# reuse one connection from its worker threads). This keeps an in-memory database alive across
# calls and avoids re-opening the file on every query. The lock inside SQLite serialises access.
_connections: dict[str, sqlite3.Connection] = {}


def _connect(db_path: str) -> sqlite3.Connection:
    connection = _connections.get(db_path)
    if connection is None:
        connection = sqlite3.connect(db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        _connections[db_path] = connection
    return connection


def close_tool_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Close a cached connection so the file (or in-memory database) can be released."""
    connection = _connections.pop(db_path, None)
    if connection is not None:
        connection.close()


def _day_number(day: str) -> int:
    key = day.strip().lower()
    number = _DAY_NUMBERS.get(key) or _DAY_NUMBERS.get(key[:3])
    if number is None:
        raise ValueError(f"{day!r} is not a weekday name such as 'monday'")
    return number


def _table_empty(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return row is None or int(row[0]) == 0


def _query(db_path: str, sql: str, params: Sequence[Any]) -> list[dict[str, Any]]:
    connection = _connect(db_path)
    rows = connection.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def init_mock_db(db_path: str = DEFAULT_DB_PATH) -> str:
    """Create the timetables/assignments tables and seed them with sample rows, if empty.

    Idempotent: existing tables are left untouched, so it is safe to call on every startup.
    Returns the resolved database path so callers can pass it straight to the router.
    """
    connection = _connect(db_path)
    with connection:
        connection.execute(_SCHEMA_TIMETABLES)
        connection.execute(_SCHEMA_ASSIGNMENTS)
        if _table_empty(connection, "timetables"):
            connection.executemany(_INSERT_TIMETABLE, _SAMPLE_TIMETABLE_ROWS)
        if _table_empty(connection, "assignments"):
            rows = [
                (code, title, (date.today() + timedelta(days=offset)).isoformat(), weight, notes)
                for code, title, offset, weight, notes in _SAMPLE_ASSIGNMENTS
            ]
            connection.executemany(_INSERT_ASSIGNMENT, rows)
    return db_path


def check_student_timetable(
    student_id: str,
    day: str | None = None,
    *,
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Return the class schedule entries for one student, venue and lecture times included.

    ``day`` accepts a full or abbreviated weekday name; anything else raises ``ValueError``.
    """
    requested_day: str | None = DAY_NAMES[_day_number(day) - 1] if day is not None else None
    rows = _query(db_path, _SELECT_TIMETABLE, [student_id, requested_day, requested_day])
    entries = [
        {
            "course_code": row["course_code"],
            "course_name": row["course_name"],
            "day": row["day"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "venue": row["venue"],
        }
        for row in rows
    ]
    entries.sort(
        key=lambda entry: (
            _DAY_NUMBERS.get(str(entry["day"]).lower(), 8),
            str(entry["start_time"]),
        )
    )

    payload: dict[str, Any] = {
        "ok": True,
        "tool": "check_student_timetable",
        "student_id": student_id,
        "day": requested_day,
        "count": len(entries),
        "timetable": entries,
    }
    if not entries:
        scope = f"on {requested_day}" if requested_day else "in the week"
        payload["note"] = f"No classes are timetabled {scope} for student {student_id}."
    return payload


def check_assignment_deadlines(
    course_code: str, *, db_path: str = DEFAULT_DB_PATH
) -> dict[str, Any]:
    """Return the assignment deadlines and requirements for one course, soonest due first."""
    rows = _query(db_path, _SELECT_ASSIGNMENTS, [course_code])
    today = date.today()
    assignments = [
        {
            "course_code": row["course_code"],
            "title": row["title"],
            "due_date": row["due_date"],
            "days_until_due": (
                (date.fromisoformat(str(row["due_date"])) - today).days
                if row["due_date"] is not None
                else None
            ),
            "weight_pct": row["weight_pct"],
            "requirements": row["requirements"],
        }
        for row in rows
        if row["course_code"] is not None
    ]

    payload: dict[str, Any] = {
        "ok": True,
        "tool": "check_assignment_deadlines",
        "course_code": course_code,
        "count": len(assignments),
        "assignments": assignments,
    }
    if not assignments:
        payload["note"] = f"No assignments are recorded for course {course_code}."
    return payload


def _error_json(tool_name: str, message: str) -> str:
    return json.dumps({"ok": False, "tool": tool_name, "error": message}, ensure_ascii=False)


def _validation_summary(exc: ValidationError) -> str:
    parts = []
    for item in exc.errors():
        location = ".".join(str(part) for part in item.get("loc", ()))
        parts.append(f"{location or '$'}: {item.get('msg', 'invalid')}")
    return "; ".join(parts)


def _timetable_handler(arguments: TimetableArguments, db_path: str) -> dict[str, Any]:
    return check_student_timetable(arguments.student_id, arguments.day, db_path=db_path)


def _deadlines_handler(arguments: DeadlineArguments, db_path: str) -> dict[str, Any]:
    return check_assignment_deadlines(arguments.course_code, db_path=db_path)


_HANDLERS: dict[str, ToolHandler] = {
    "check_student_timetable": _timetable_handler,
    "check_assignment_deadlines": _deadlines_handler,
}

_ARGUMENT_MODELS: dict[str, type[BaseModel]] = {
    "check_student_timetable": TimetableArguments,
    "check_assignment_deadlines": DeadlineArguments,
}


async def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Validate the arguments, run the tool, and return a JSON string for the LLM context.

    Never raises for an unknown tool or bad arguments: every failure becomes an ``ok: false``
    JSON object the model can read and explain to the student.
    """
    if not isinstance(arguments, dict):
        return _error_json(tool_name, "arguments must be a JSON object")

    handler = _HANDLERS.get(tool_name)
    if handler is None:
        available = ", ".join(TOOL_NAMES)
        return _error_json(tool_name, f"unknown tool {tool_name!r}; available tools: {available}")

    argument_model = _ARGUMENT_MODELS[tool_name]
    try:
        validated = argument_model.model_validate(arguments)
    except ValidationError as exc:
        return _error_json(tool_name, f"invalid arguments: {_validation_summary(exc)}")

    try:
        payload = await asyncio.to_thread(handler, validated, db_path)
    except Exception as exc:
        logger.warning(
            "tool engine query failed",
            extra={"context": {"tool": tool_name, "error": f"{type(exc).__name__}: {exc}"}},
        )
        return _error_json(tool_name, f"the tool failed: {type(exc).__name__}")
    return json.dumps(payload, ensure_ascii=False)


__all__ = [
    "DEFAULT_DB_PATH",
    "TOOLS",
    "TOOL_NAMES",
    "check_assignment_deadlines",
    "check_student_timetable",
    "close_tool_db",
    "execute_tool",
    "init_mock_db",
]
