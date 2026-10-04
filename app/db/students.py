from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from decimal import Decimal
from typing import Any

from app.db.pool import Database

logger = logging.getLogger(__name__)

DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
DAY_LOOKUP: dict[str, int] = {name: index for index, name in enumerate(DAYS, start=1)}
DAY_LOOKUP.update({name[:3]: index for index, name in enumerate(DAYS, start=1)})
# Keys are str only: day_number() validates int input against its own range before consulting this
# table, so a previous set of int keys here was unreachable. Widening the annotation to accept int
# keys would have silenced mypy while keeping dead entries; converting the keys to str would have
# quietly started accepting "3" as a weekday, which is a behaviour change, not a type fix.


class StudentNotLinked(Exception):
    """No student record matches the WhatsApp number that wrote in."""


def _money(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, Decimal)) else None


def _iso(value: Any) -> str | None:
    """Render a database timestamp compactly and unambiguously.

    Timestamps go to the model inside a prompt, so an explicit ``Z`` beats a numeric offset:
    ``2026-10-01T23:59Z`` is shorter than ``+00:00`` and cannot be misread. A naive value is
    assumed to already be UTC, which is what Postgres returns for ``timestamptz`` in practice.
    """
    if isinstance(value, datetime):
        moment: datetime = (
            value.astimezone(UTC).replace(tzinfo=None) if value.tzinfo is not None else value
        )
        return moment.isoformat(timespec="minutes") + "Z"
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.strftime("%H:%M")
    return None


def day_number(value: object) -> int | None:
    """Accept 1-7, or a weekday name in full or short form. Anything else is rejected."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1 <= value <= 7 else None
    if isinstance(value, str):
        key = value.strip().lower()
        return DAY_LOOKUP.get(key) or DAY_LOOKUP.get(key[:3])
    return None


@dataclass(frozen=True, slots=True)
class StudentProfile:
    id: int
    full_name: str
    email: str | None
    programme: str | None
    year_of_study: int | None
    gpa: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "full_name": self.full_name,
            "email": self.email,
            "programme": self.programme,
            "year_of_study": self.year_of_study,
            "gpa": self.gpa,
        }


@dataclass(frozen=True, slots=True)
class Course:
    code: str
    name: str
    credits: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "name": self.name, "credits": self.credits}


@dataclass(frozen=True, slots=True)
class TimetableEntry:
    day: str
    day_number: int
    start_time: str
    end_time: str
    course_code: str
    course_name: str
    room: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "course_code": self.course_code,
            "course_name": self.course_name,
            "room": self.room,
        }


@dataclass(frozen=True, slots=True)
class Assignment:
    title: str
    course_code: str
    due_at: str
    status: str
    days_until_due: int
    weight_pct: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "course_code": self.course_code,
            "due_at": self.due_at,
            "status": self.status,
            "days_until_due": self.days_until_due,
            "weight_pct": self.weight_pct,
        }


@dataclass(frozen=True, slots=True)
class Grade:
    title: str
    course_code: str
    grade: float
    out_of: float
    graded_at: str | None
    feedback: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "course_code": self.course_code,
            "grade": self.grade,
            "out_of": self.out_of,
            "graded_at": self.graded_at,
            "feedback": self.feedback,
        }


@dataclass(frozen=True, slots=True)
class Exam:
    course_code: str
    exam_date: str
    starts_at: str | None
    room: str | None
    seat: str | None
    days_until_exam: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "course_code": self.course_code,
            "exam_date": self.exam_date,
            "starts_at": self.starts_at,
            "room": self.room,
            "seat": self.seat,
            "days_until_exam": self.days_until_exam,
        }


@dataclass(frozen=True, slots=True)
class StudentView:
    """Everything the tools found for one student, in one object."""

    profile: StudentProfile
    courses: list[Course] = field(default_factory=list)
    timetable: list[TimetableEntry] = field(default_factory=list)
    deadlines: list[Assignment] = field(default_factory=list)
    grades: list[Grade] = field(default_factory=list)
    exams: list[Exam] = field(default_factory=list)


class StudentRepository:
    """Read-only queries over the student schema.

    Every method takes a resolved ``student_id``. Nothing here reads a number from the model:
    the tool router looks the student up once, from the verified WhatsApp identity.
    """

    def __init__(self, database: Database) -> None:
        self._db = database

    async def find_id_by_wa_id(self, wa_id: str) -> int | None:
        row = await self._db.fetchval("SELECT id FROM students WHERE wa_id = $1", wa_id)
        return int(row) if row is not None else None

    async def require_id(self, wa_id: str) -> int:
        student_id = await self.find_id_by_wa_id(wa_id)
        if student_id is None:
            raise StudentNotLinked(
                f"no student record is linked to the number that sent this message: {wa_id}"
            )
        return student_id

    async def profile(self, student_id: int) -> StudentProfile:
        row = await self._db.fetchrow(
            """
            SELECT id, full_name, email, programme, year_of_study, gpa
              FROM students
             WHERE id = $1
            """,
            student_id,
        )
        if row is None:
            raise StudentNotLinked(f"student {student_id} no longer exists")
        return StudentProfile(
            id=int(row["id"]),
            full_name=str(row["full_name"]),
            email=row["email"],
            programme=row["programme"],
            year_of_study=int(row["year_of_study"]) if row["year_of_study"] is not None else None,
            gpa=_money(row["gpa"]),
        )

    async def courses(self, student_id: int) -> list[Course]:
        rows = await self._db.fetch(
            """
            SELECT c.code, c.name, c.credits
              FROM enrollments e
              JOIN courses c ON c.id = e.course_id
             WHERE e.student_id = $1
             ORDER BY c.code
            """,
            student_id,
        )
        return [
            Course(
                code=str(row["code"]),
                name=str(row["name"]),
                credits=int(row["credits"]) if row["credits"] is not None else None,
            )
            for row in rows
        ]

    async def timetable(self, student_id: int, day: int | None = None) -> list[TimetableEntry]:
        rows = await self._db.fetch(
            """
            SELECT t.day_of_week, t.start_time, t.end_time, t.room, c.code, c.name
              FROM timetable_entries t
              JOIN enrollments e ON e.course_id = t.course_id
              JOIN courses c ON c.id = t.course_id
             WHERE e.student_id = $1
               AND ($2::smallint IS NULL OR t.day_of_week = $2)
             ORDER BY t.day_of_week, t.start_time
            """,
            student_id,
            day,
        )
        return [
            TimetableEntry(
                day=DAYS[int(row["day_of_week"]) - 1],
                day_number=int(row["day_of_week"]),
                start_time=_iso(row["start_time"]) or "",
                end_time=_iso(row["end_time"]) or "",
                course_code=str(row["code"]),
                course_name=str(row["name"]),
                room=row["room"],
            )
            for row in rows
        ]

    async def deadlines(self, student_id: int, within_days: int = 21) -> list[Assignment]:
        rows = await self._db.fetch(
            """
            SELECT a.title, a.due_at, a.weight_pct, c.code,
                   COALESCE(s.status, 'not_started') AS status,
                   (a.due_at::date - CURRENT_DATE) AS days_until_due
              FROM assignments a
              JOIN enrollments e ON e.course_id = a.course_id
              JOIN courses c ON c.id = a.course_id
              LEFT JOIN submissions s
                     ON s.assignment_id = a.id AND s.student_id = $1
             WHERE e.student_id = $1
               AND COALESCE(s.status, 'not_started') IN ('not_started', 'in_progress')
               AND a.due_at >= NOW()
               AND a.due_at <= NOW() + ($2 || ' days')::interval
             ORDER BY a.due_at
             LIMIT 25
            """,
            student_id,
            str(within_days),
        )
        return [
            Assignment(
                title=str(row["title"]),
                course_code=str(row["code"]),
                due_at=_iso(row["due_at"]) or "",
                status=str(row["status"]),
                days_until_due=int(row["days_until_due"]),
                weight_pct=_money(row["weight_pct"]),
            )
            for row in rows
        ]

    async def grades(self, student_id: int, limit: int = 10) -> list[Grade]:
        rows = await self._db.fetch(
            """
            SELECT a.title, s.grade, s.feedback, s.submitted_at, c.code
              FROM submissions s
              JOIN assignments a ON a.id = s.assignment_id
              JOIN courses c ON c.id = a.course_id
             WHERE s.student_id = $1
               AND s.status = 'graded'
               AND s.grade IS NOT NULL
             ORDER BY s.submitted_at DESC NULLS LAST, a.due_at DESC
             LIMIT $2
            """,
            student_id,
            limit,
        )
        return [
            Grade(
                title=str(row["title"]),
                course_code=str(row["code"]),
                grade=float(row["grade"]),
                out_of=100.0,
                graded_at=_iso(row["submitted_at"]),
                feedback=row["feedback"],
            )
            for row in rows
        ]

    async def exams(self, student_id: int, within_days: int = 60) -> list[Exam]:
        rows = await self._db.fetch(
            """
            SELECT c.code, x.exam_date, x.starts_at, x.room, er.seat,
                   (x.exam_date - CURRENT_DATE) AS days_until_exam
              FROM exam_registrations er
              JOIN exams x ON x.id = er.exam_id
              JOIN courses c ON c.id = x.course_id
             WHERE er.student_id = $1
               AND x.exam_date >= CURRENT_DATE
               AND x.exam_date <= CURRENT_DATE + $2
             ORDER BY x.exam_date, x.starts_at
             LIMIT 25
            """,
            student_id,
            within_days,
        )
        return [
            Exam(
                course_code=str(row["code"]),
                exam_date=_iso(row["exam_date"]) or "",
                starts_at=_iso(row["starts_at"]),
                room=row["room"],
                seat=row["seat"],
                days_until_exam=int(row["days_until_exam"]),
            )
            for row in rows
        ]

    async def summary(self, wa_id: str) -> StudentView:
        """Resolve the sender and load the whole view in one round of queries."""
        student_id = await self.require_id(wa_id)
        return StudentView(
            profile=await self.profile(student_id),
            courses=await self.courses(student_id),
            timetable=await self.timetable(student_id),
            deadlines=await self.deadlines(student_id),
            grades=await self.grades(student_id),
            exams=await self.exams(student_id),
        )
