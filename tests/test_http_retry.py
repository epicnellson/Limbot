from __future__ import annotations

import httpx
import pytest
from app.config import Settings
from app.core.http import OutboundTransportError, RetryPolicy, build_http_client, send


@pytest.fixture
def fast_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "outbound_max_attempts": 3,
            "outbound_retry_base_delay_seconds": 0.001,
            "outbound_retry_max_delay_seconds": 0.002,
        }
    )


async def test_throttled_request_is_retried(fast_settings: Settings) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "throttled"})
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await send(
            client,
            "POST",
            "https://graph.facebook.com/v23.0/1/messages",
            retry_policy=RetryPolicy.from_settings(fast_settings, idempotent=False),
            json={"hello": "world"},
        )

    assert response.status_code == 200
    assert attempts == 2


async def test_transport_errors_are_not_replayed_for_non_idempotent_calls(
    fast_settings: Settings,
) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("connection reset", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OutboundTransportError) as excinfo:
            await send(
                client,
                "POST",
                "https://graph.facebook.com/v23.0/1/messages",
                retry_policy=RetryPolicy.from_settings(fast_settings, idempotent=False),
                json={"hello": "world"},
            )

    assert excinfo.value.attempts == 1
    assert attempts == 1


async def test_transport_errors_are_replayed_for_idempotent_calls(
    fast_settings: Settings,
) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await send(
            client,
            "POST",
            "https://graph.facebook.com/v23.0/1/messages",
            retry_policy=RetryPolicy.from_settings(fast_settings, idempotent=True),
            json={"hello": "world"},
        )

    assert response.status_code == 200
    assert attempts == 3


async def test_client_errors_are_returned_without_retry(fast_settings: Settings) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, json={"error": {"message": "bad request", "code": 100}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await send(
            client,
            "POST",
            "https://graph.facebook.com/v23.0/1/messages",
            retry_policy=RetryPolicy.from_settings(fast_settings, idempotent=False),
            json={},
        )

    assert response.status_code == 400
    assert attempts == 1


async def test_retry_budget_is_finite(fast_settings: Settings) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, json={"error": "unavailable"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await send(
            client,
            "GET",
            "https://graph.facebook.com/v23.0/1/phone_numbers",
            retry_policy=RetryPolicy.from_settings(fast_settings, idempotent=True),
        )

    assert response.status_code == 503
    assert attempts == 3


async def test_shared_client_is_configured_from_settings(settings: Settings) -> None:
    client = build_http_client(settings, client_name="test")
    try:
        assert client.timeout.read == settings.http_timeout_seconds
        assert client.timeout.connect == settings.http_connect_timeout_seconds
    finally:
        await client.aclose()
