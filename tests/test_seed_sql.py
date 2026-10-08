from __future__ import annotations

import re
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "db"

# Meta sends the number in wa_id as E.164 without '+': digits only, country code first.
DIGITS_ONLY = re.compile(r"^[0-9]{10,15}$")

SEED_VALUES = [
    "15550001111",
    "15550002222",
    "15550003333",
    "15550004444",
]


def seed_sql() -> str:
    return (DB / "seed.sql").read_text(encoding="utf-8")


def link_sql() -> str:
    return (DB / "link_student.sql").read_text(encoding="utf-8")


def test_seed_wa_ids_are_digits_only() -> None:
    seed = seed_sql()
    for value in SEED_VALUES:
        assert value in seed, f"expected placeholder {value} in seed.sql"
        assert DIGITS_ONLY.match(value), f"{value} is not digits-only E.164"


def test_seed_wa_ids_are_in_the_students_insert() -> None:
    """Every placeholder must be a student the link script can repoint to, not a stray string."""
    seed = seed_sql()
    for value in SEED_VALUES:
        # In the students INSERT the numbers appear surrounded by quotes and commas/parens.
        assert value in seed, f"{value} missing from seed.sql"


def test_seed_covers_timetable_deadlines_and_grades() -> None:
    """The sample student the link script targets must have all three kinds of data."""
    seed = seed_sql()
    assert "INSERT INTO timetable_entries" in seed
    assert "INSERT INTO assignments" in seed
    assert "INSERT INTO submissions" in seed


def test_link_script_defaults_to_a_digit_only_number() -> None:
    match = re.search(r"\\set phone ([0-9]+)", link_sql())
    assert match is not None, "link_student.sql should default the phone with `\\set phone`"
    assert DIGITS_ONLY.match(match.group(1)), f"{match.group(1)} is not digits-only E.164"


def test_link_script_repoints_a_seeded_student() -> None:
    link = link_sql()
    assert "Ada Lovelace" in link, "the docstring example must name the seeded student"
    assert "UPDATE students" in link
    assert ":'phone'" in link


def test_link_script_uses_the_variable_not_a_fixed_quoted_number() -> None:
    """hard-coding the number in more than the \\set would defeat the -v override."""
    matches = re.findall(r"(:\'phone\'|\'(?:[0-9]{10,15})\')", link_sql())
    assert ":'phone'" in matches, "the UPDATE must bind the psql variable"
