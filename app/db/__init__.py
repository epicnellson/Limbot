"""Read-only access to the student database."""

from app.db.pool import Database
from app.db.students import (
    Assignment,
    Course,
    Exam,
    Grade,
    StudentProfile,
    StudentRepository,
    TimetableEntry,
)

__all__ = [
    "Assignment",
    "Course",
    "Database",
    "Exam",
    "Grade",
    "StudentProfile",
    "StudentRepository",
    "TimetableEntry",
]
