from __future__ import annotations

import pytest
from app.access import (
    AUDIENCES,
    PUBLIC,
    STAFF,
    STUDENT,
    Role,
    audiences_for,
    can_use_db_tools,
    coerce_role,
    rank,
    resolve_role,
    role_at_least,
)
from app.db.contacts import AccessResolver


def test_unknown_roles_fail_closed_to_blocked() -> None:
    assert coerce_role("wizard") is Role.BLOCKED
    assert coerce_role(None) is Role.BLOCKED
    assert coerce_role("") is Role.BLOCKED


def test_roles_parse_case_insensitively() -> None:
    assert coerce_role("Student") is Role.STUDENT
    assert coerce_role(" ADMIN ") is Role.ADMIN


def test_audiences_are_cumulative_and_public_only_sees_public() -> None:
    assert audiences_for(Role.PUBLIC) == (PUBLIC,)
    assert audiences_for(Role.STUDENT) == (PUBLIC, STUDENT)
    assert audiences_for(Role.TESTER) == (PUBLIC, STUDENT)
    assert audiences_for(Role.STAFF) == (PUBLIC, STUDENT, STAFF)
    assert audiences_for(Role.ADMIN) == (PUBLIC, STUDENT, STAFF)


def test_blocked_may_retrieve_nothing() -> None:
    assert audiences_for(Role.BLOCKED) == ()


def test_every_advertised_audience_is_reachable_by_some_role() -> None:
    reachable = set()
    for role in Role:
        reachable.update(audiences_for(role))
    assert reachable == set(AUDIENCES)


def test_rank_orders_blocked_below_public() -> None:
    assert rank(Role.BLOCKED) < rank(Role.PUBLIC) < rank(Role.STUDENT) <= rank(Role.STAFF)
    assert rank(Role.TESTER) == rank(Role.STUDENT)


def test_role_at_least_gates_tools() -> None:
    assert role_at_least(Role.STUDENT, PUBLIC)
    assert role_at_least(Role.TESTER, STUDENT)
    assert not role_at_least(Role.PUBLIC, STUDENT)
    assert not role_at_least(Role.BLOCKED, PUBLIC)


def test_only_linked_tiers_use_db_tools() -> None:
    assert can_use_db_tools(Role.STUDENT)
    assert can_use_db_tools(Role.ADMIN)
    assert not can_use_db_tools(Role.PUBLIC)
    assert not can_use_db_tools(Role.BLOCKED)


@pytest.mark.parametrize(
    ("contact_role", "linked", "expected"),
    [
        (None, False, Role.PUBLIC),
        (None, True, Role.STUDENT),
        ("staff", False, Role.STAFF),
        ("blocked", True, Role.BLOCKED),
        ("garbage", True, Role.BLOCKED),
    ],
)
def test_resolve_role_combines_both_tables(
    contact_role: str | None, linked: bool, expected: Role
) -> None:
    assert resolve_role(contact_role, is_linked_student=linked) is expected


class StubContacts:
    def __init__(self, value: tuple[Role, int | None] | None) -> None:
        self._value = value

    async def role_for(self, wa_id: str) -> tuple[Role, int | None] | None:
        return self._value


class StubStudents:
    def __init__(self, student_id: int | None) -> None:
        self._student_id = student_id

    async def find_id_by_wa_id(self, wa_id: str) -> int | None:
        return self._student_id


def resolver(
    contacts: tuple[Role, int | None] | None, *, student_id: int | None = None
) -> AccessResolver:
    return AccessResolver(StubContacts(contacts), StubStudents(student_id))  # type: ignore[arg-type]


async def test_an_unlisted_linked_number_is_a_student() -> None:
    decision = await resolver(None, student_id=7).resolve("15550001111")
    assert decision.role is Role.STUDENT
    assert decision.student_id == 7
    assert decision.audiences == (PUBLIC, STUDENT)


async def test_an_unlisted_unlinked_number_is_public() -> None:
    decision = await resolver(None).resolve("15550001111")
    assert decision.role is Role.PUBLIC
    assert decision.student_id is None
    assert decision.audiences == (PUBLIC,)


async def test_a_contact_row_overrides_the_students_table() -> None:
    decision = await resolver((Role.STAFF, None), student_id=7).resolve("15550001111")
    assert decision.role is Role.STAFF
    assert decision.student_id == 7
    assert decision.audiences == (PUBLIC, STUDENT, STAFF)


async def test_a_blocked_row_gets_no_audiences_or_record() -> None:
    decision = await resolver((Role.BLOCKED, 7)).resolve("15550001111")
    assert decision.role is Role.BLOCKED
    assert decision.student_id is None
    assert decision.audiences == ()
