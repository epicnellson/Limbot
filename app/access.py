"""Who may see what.

This module is deliberately free of database and web imports: it is the single, pure place that
turns a role into a set of knowledge audiences and answers "may this role use the database
tools?". Everything the bot exposes is gated through here, so the policy reads in one file.

The names are :class:`Role` values but the functions also accept a plain string, because the
role arrives from Postgres as text and travels through :class:`~app.tools.context.ToolContext`
as text. An unrecognised value is treated as ``blocked`` rather than ``public``: a typo in the
contacts table must fail closed, not hand a stranger the FAQ.
"""

from __future__ import annotations

from enum import StrEnum

# Knowledge audiences, ordered from least to most privileged. A chunk carries exactly one.
PUBLIC = "public"
STUDENT = "student"
STAFF = "staff"

AUDIENCES: tuple[str, ...] = (PUBLIC, STUDENT, STAFF)

# The Qdrant payload key that carries a chunk's audience. Shared so the writer (chunker) and
# the reader (vector store filter) can never disagree on the spelling.
AUDIENCE_FIELD = "audience"


class Role(StrEnum):
    """A WhatsApp number's access level, resolved server side from the verified payment."""

    PUBLIC = "public"
    STUDENT = "student"
    TESTER = "tester"
    STAFF = "staff"
    ADMIN = "admin"
    BLOCKED = "blocked"


# Ordering used by tool gating. ``blocked`` sits below ``public`` so it can never satisfy even
# the most permissive tool's ``min_role``.
_RANK: dict[Role, int] = {
    Role.BLOCKED: -1,
    Role.PUBLIC: 0,
    Role.STUDENT: 1,
    Role.TESTER: 1,
    Role.STAFF: 2,
    Role.ADMIN: 3,
}

# What each role may retrieve. Higher roles include everything lower roles see.
_ROLE_AUDIENCES: dict[Role, tuple[str, ...]] = {
    Role.BLOCKED: (),
    Role.PUBLIC: (PUBLIC,),
    Role.STUDENT: (PUBLIC, STUDENT),
    Role.TESTER: (PUBLIC, STUDENT),
    Role.STAFF: (PUBLIC, STUDENT, STAFF),
    Role.ADMIN: (PUBLIC, STUDENT, STAFF),
}

_TOOL_ROLES = frozenset({Role.STUDENT, Role.TESTER, Role.STAFF, Role.ADMIN})


def coerce_role(value: str | Role | None) -> Role:
    """Parse a role from anything, failing closed to ``BLOCKED`` for unknown input."""
    if isinstance(value, Role):
        return value
    if isinstance(value, str):
        try:
            return Role(value.strip().lower())
        except ValueError:
            return Role.BLOCKED
    return Role.BLOCKED


def audiences_for(role: str | Role) -> tuple[str, ...]:
    """The knowledge audiences a role may retrieve. Empty means no retrieval at all."""
    return _ROLE_AUDIENCES[coerce_role(role)]


def rank(role: str | Role) -> int:
    return _RANK[coerce_role(role)]


def role_at_least(role: str | Role, minimum: str | Role) -> bool:
    """True when ``role`` outranks ``minimum``, used to gate a tool on its ``min_role``."""
    return rank(role) >= rank(minimum)


def can_use_db_tools(role: str | Role) -> bool:
    """Whether a role may use the student-record tools at all."""
    return coerce_role(role) in _TOOL_ROLES


def resolve_role(contact_role: str | Role | None, *, is_linked_student: bool) -> Role:
    """Combine the contacts table with the students table into one role.

    An explicit contacts row always wins, so a ``blocked`` row overrides even a linked student
    and a ``staff`` row needs no student record. With no row, a linked student number is a
    ``student`` and anything else is ``public``.
    """
    if contact_role is not None:
        return coerce_role(contact_role)
    return Role.STUDENT if is_linked_student else Role.PUBLIC
