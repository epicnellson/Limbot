from __future__ import annotations

import logging
from typing import Any

from app.db.students import DAYS, StudentNotLinked, StudentRepository, day_number
from app.llm.types import ToolSpec
from app.tools.context import ToolContext
from app.tools.registry import ToolDefinition, ToolFailure, ToolRegistry

logger = logging.getLogger(__name__)

NOT_LINKED = (
    "This WhatsApp number is not linked to a student record yet, so there is nothing to read. "
    "Ask the student to have their number added to the students table, or to tell the office "
    "which number is on their registration."
)

DAY_ENUM = list(DAYS) + [day[:3] for day in DAYS]


def _arguments(arguments: dict[str, Any], name: str, default: Any) -> Any:
    value = arguments.get(name, default)
    return default if value is None else value


def _string_list(values: list[Any]) -> list[str]:
    return [str(value) for value in values]


class StudentTools:
    """The read-only tools a student can ask for.

    Every handler resolves the student from the verified WhatsApp identity in
    :class:`ToolContext`, then runs one query. None of them accept an identifier as an argument,
    so the model cannot address a different student's rows.
    """

    def __init__(self, repository: StudentRepository) -> None:
        self._repo = repository

    async def _student_id(self, context: ToolContext) -> int:
        try:
            return await self._repo.require_id(context.wa_id)
        except StudentNotLinked as exc:
            logger.warning(
                "whatsapp number is not linked to a student record: %s",
                context.wa_id,
                extra={
                    "context": {
                        "wa_id": context.wa_id,
                        "display_name": context.display_name,
                        "hint": (
                            "store the number exactly as Meta sends it in wa_id: "
                            "E.164 digits without '+', e.g. 23279826564"
                        ),
                    }
                },
            )
            raise ToolFailure(NOT_LINKED) from exc

    async def profile(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        student_id = await self._student_id(context)
        profile = await self._repo.profile(student_id)
        return {"student": profile.as_dict()}

    async def courses(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        student_id = await self._student_id(context)
        courses = await self._repo.courses(student_id)
        return {
            "count": len(courses),
            "courses": [course.as_dict() for course in courses],
        }

    async def timetable(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        student_id = await self._student_id(context)
        raw_day = arguments.get("day")
        day = day_number(raw_day) if raw_day is not None else None
        if raw_day is not None and day is None:
            raise ToolFailure(
                f"Could not read {raw_day!r} as a weekday. Use a name such as 'monday'."
            )
        entries = await self._repo.timetable(student_id, day)
        if not entries:
            scope = f"on {DAYS[day - 1]}" if day else "in the week"
            return {"count": 0, "entries": [], "note": f"No classes are timetabled {scope}."}
        return {
            "count": len(entries),
            "entries": [entry.as_dict() for entry in entries],
        }

    async def deadlines(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        student_id = await self._student_id(context)
        within = int(_arguments(arguments, "within_days", 21))
        assignments = await self._repo.deadlines(student_id, within)
        if not assignments:
            return {
                "count": 0,
                "assignments": [],
                "note": f"Nothing is outstanding for the next {within} days.",
            }
        return {
            "count": len(assignments),
            "assignments": [assignment.as_dict() for assignment in assignments],
        }

    async def grades(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        student_id = await self._student_id(context)
        limit = int(_arguments(arguments, "limit", 10))
        grades = await self._repo.grades(student_id, limit)
        if not grades:
            return {
                "count": 0,
                "grades": [],
                "note": "No graded work has been released yet.",
            }
        average = round(sum(grade.grade for grade in grades) / len(grades), 1)
        return {
            "count": len(grades),
            "average": average,
            "grades": [grade.as_dict() for grade in grades],
        }

    async def exams(self, context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        student_id = await self._student_id(context)
        within = int(_arguments(arguments, "within_days", 60))
        exams = await self._repo.exams(student_id, within)
        if not exams:
            return {
                "count": 0,
                "exams": [],
                "note": f"No exams are scheduled in the next {within} days.",
            }
        return {"count": len(exams), "exams": [exam.as_dict() for exam in exams]}


PROFILE_TOOL = ToolSpec(
    name="get_student_profile",
    description=(
        "Read the signed-in student's own record: name, programme, year of study and GPA. "
        "Takes no arguments."
    ),
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
)

COURSES_TOOL = ToolSpec(
    name="list_courses",
    description="List the courses the signed-in student is enrolled in. Takes no arguments.",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
)

TIMETABLE_TOOL = ToolSpec(
    name="get_timetable",
    description=(
        "Read the signed-in student's weekly timetable (their class schedule). Use it for "
        "requests such as 'Timetable', 'My timetable', 'Show my schedule' or 'what is on "
        "today?'. Pass day for a single weekday, or omit it for the whole week."
    ),
    parameters={
        "type": "object",
        "properties": {
            "day": {
                "type": "string",
                "enum": DAY_ENUM,
                "description": "Weekday name, for example 'monday' or 'mon'.",
            }
        },
        "additionalProperties": False,
    },
)

DEADLINES_TOOL = ToolSpec(
    name="get_upcoming_deadlines",
    description=(
        "Read assignments the signed-in student has not submitted yet, soonest first. "
        "within_days defaults to 21."
    ),
    parameters={
        "type": "object",
        "properties": {
            "within_days": {
                "type": "integer",
                "minimum": 1,
                "maximum": 90,
                "description": "How far ahead to look, in days.",
            }
        },
        "additionalProperties": False,
    },
)

GRADES_TOOL = ToolSpec(
    name="get_recent_grades",
    description=(
        "Read the signed-in student's most recently graded work, most recent first. "
        "limit defaults to 10."
    ),
    parameters={
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 25,
                "description": "How many graded items to return.",
            }
        },
        "additionalProperties": False,
    },
)

EXAMS_TOOL = ToolSpec(
    name="get_exam_schedule",
    description=(
        "Read the signed-in student's upcoming exams with room and seat. within_days defaults "
        "to 60."
    ),
    parameters={
        "type": "object",
        "properties": {
            "within_days": {
                "type": "integer",
                "minimum": 1,
                "maximum": 180,
                "description": "How far ahead to look, in days.",
            }
        },
        "additionalProperties": False,
    },
)


def build_student_registry(
    repository: StudentRepository, *, timeout_seconds: float = 10.0
) -> ToolRegistry:
    tools = StudentTools(repository)
    definitions = [
        (PROFILE_TOOL, tools.profile),
        (COURSES_TOOL, tools.courses),
        (TIMETABLE_TOOL, tools.timetable),
        (DEADLINES_TOOL, tools.deadlines),
        (GRADES_TOOL, tools.grades),
        (EXAMS_TOOL, tools.exams),
    ]
    return ToolRegistry(
        [
            ToolDefinition(spec=spec, handler=handler, timeout_seconds=timeout_seconds)
            for spec, handler in definitions
        ]
    )
