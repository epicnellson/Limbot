from __future__ import annotations

import httpx
import pytest
from app.core.http import _retry_after_seconds
from app.core.security import compute_signature, verify_signature

SECRET = "test-app-secret"
BODY = b'{"object":"whatsapp_business_account"}'


def test_valid_signature_is_accepted() -> None:
    assert verify_signature(SECRET, BODY, compute_signature(SECRET, BODY))


def test_signature_is_order_sensitive() -> None:
    assert not verify_signature(SECRET, BODY, compute_signature(SECRET, BODY + b" "))


def test_wrong_secret_is_rejected() -> None:
    assert not verify_signature(SECRET, BODY, compute_signature("other-secret", BODY))


def test_missing_header_is_rejected() -> None:
    assert not verify_signature(SECRET, BODY, None)
    assert not verify_signature(SECRET, BODY, "")


def test_header_without_sha256_prefix_is_rejected() -> None:
    digest = compute_signature(SECRET, BODY).removeprefix("sha256=")
    assert not verify_signature(SECRET, BODY, digest)


def test_empty_secret_rejects_everything() -> None:
    assert not verify_signature("", BODY, compute_signature(SECRET, BODY))


def test_signature_is_stable_across_calls() -> None:
    first = compute_signature(SECRET, BODY)
    second = compute_signature(SECRET, BODY)
    assert first == second == f"sha256={first.removeprefix('sha256=')}"


def test_retry_after_parsing_is_none_when_absent() -> None:
    assert _retry_after_seconds(httpx.Response(200)) is None


@pytest.mark.parametrize("value,expected", [("0", 0.0), ("2", 2.0), ("-5", 0.0)])
def test_retry_after_parses_seconds(value: str, expected: float) -> None:
    assert _retry_after_seconds(httpx.Response(429, headers={"retry-after": value})) == expected


def test_retry_after_treats_a_past_http_date_as_zero() -> None:
    response = httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"})
    assert _retry_after_seconds(response) == 0.0


def test_retry_after_ignores_unparseable_values() -> None:
    response = httpx.Response(429, headers={"retry-after": "soon"})
    assert _retry_after_seconds(response) is None
