"""Reading the access-control tables.

Roles live in ``contacts`` and student linkage in ``students``. The bot only ever reads them:
the Postgres pool is opened read only, so granting or revoking access is an admin psql job, not
something a WhatsApp message can do.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.access import Role, audiences_for, resolve_role
from app.db.pool import Database
from app.db.students import StudentRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AccessDecision:
    """The resolved answer to "who is this, and what may they see?"."""

    role: Role
    student_id: int | None
    audiences: tuple[str, ...]


class ContactRepository:
    """Looks up a number's explicit role override, if it has one."""

    def __init__(self, database: Database) -> None:
        self._db = database
        self._missing_table_logged = False

    async def role_for(self, wa_id: str) -> tuple[Role, int | None] | None:
        """Return ``(role, student_id)`` for an explicit contacts row, or ``None``.

        A missing ``contacts`` table (an older database that predates the migration) is treated
        as "no overrides" rather than an error, so the bot keeps answering while the admin
        applies ``db/schema.sql``.
        """
        try:
            row = await self._db.fetchrow(
                "SELECT role, student_id FROM contacts WHERE wa_id = $1", wa_id
            )
        except Exception as exc:
            if not self._missing_table_logged:
                self._missing_table_logged = True
                logger.warning(
                    "contacts table could not be read; applying db/schema.sql fixes this",
                    extra={"context": {"error": f"{type(exc).__name__}: {exc}"}},
                )
            return None
        if row is None:
            return None
        role = Role(row["role"]) if row["role"] in _ROLE_VALUES else Role.BLOCKED
        student_id = int(row["student_id"]) if row["student_id"] is not None else None
        return role, student_id


class AccessResolver:
    """Combine the contacts override and the students row into one :class:`AccessDecision`."""

    def __init__(self, contacts: ContactRepository, students: StudentRepository) -> None:
        self._contacts = contacts
        self._students = students

    async def resolve(self, wa_id: str) -> AccessDecision:
        contact = await self._contacts.role_for(wa_id)
        if contact is not None:
            role, student_id = contact
            if role is Role.BLOCKED:
                return AccessDecision(role=role, student_id=None, audiences=())
            if student_id is None:
                student_id = await self._students.find_id_by_wa_id(wa_id)
            return AccessDecision(role=role, student_id=student_id, audiences=audiences_for(role))

        student_id = await self._students.find_id_by_wa_id(wa_id)
        role = resolve_role(None, is_linked_student=student_id is not None)
        return AccessDecision(role=role, student_id=student_id, audiences=audiences_for(role))


_ROLE_VALUES = frozenset(role.value for role in Role)
