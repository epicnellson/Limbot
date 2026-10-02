from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager

from app.config import Settings
from app.services import message_handler
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from conftest import VERIFY_TOKEN, sign, text_message_payload

URL = "/webhooks/whatsapp"


def test_verification_handshake_returns_challenge(client: TestClient) -> None:
    response = client.get(
        URL,
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": VERIFY_TOKEN,
            "hub.challenge": "1158201444",
        },
    )
    assert response.status_code == 200
    assert response.text == "1158201444"


def test_verification_handshake_rejects_wrong_token(client: TestClient) -> None:
    response = client.get(
        URL,
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "not-the-token",
            "hub.challenge": "1158201444",
        },
    )
    assert response.status_code == 403


def test_valid_delivery_is_acknowledged_and_dispatched(
    client: TestClient, monkeypatch: object
) -> None:
    dispatched = threading.Event()
    received: list[object] = []

    async def fake_handler(payload, **kwargs):  # type: ignore[no-untyped-def]
        received.append(payload)
        dispatched.set()

    monkeypatch.setattr(message_handler, "handle_webhook", fake_handler)  # type: ignore[attr-defined]
    body = text_message_payload()

    response = client.post(URL, content=body, headers={"x-hub-signature-256": sign(body)})

    assert response.status_code == 200
    assert dispatched.wait(timeout=5), "background task never ran"
    assert received and received[0].message_count() == 1  # type: ignore[attr-defined]


def test_invalid_signature_is_rejected_before_dispatch(
    client: TestClient, monkeypatch: object
) -> None:
    called = threading.Event()

    async def fake_handler(payload, **kwargs):  # type: ignore[no-untyped-def]
        called.set()

    monkeypatch.setattr(message_handler, "handle_webhook", fake_handler)  # type: ignore[attr-defined]
    body = text_message_payload()

    response = client.post(
        URL, content=body, headers={"x-hub-signature-256": sign(body, "wrong-secret")}
    )

    assert response.status_code == 401
    assert not called.wait(timeout=0.2)


def test_missing_signature_header_is_rejected(client: TestClient) -> None:
    body = text_message_payload()
    response = client.post(URL, content=body)
    assert response.status_code == 401


def test_malformed_payload_is_rejected(client: TestClient) -> None:
    body = b'{"object": "whatsapp_business_account", "entry": [{"changes": [{}]}]}'
    response = client.post(URL, content=body, headers={"x-hub-signature-256": sign(body)})
    assert response.status_code == 422


def test_delivery_is_refused_while_draining(client: TestClient) -> None:
    client.app.state.task_manager.stop_accepting()
    body = text_message_payload()

    response = client.post(URL, content=body, headers={"x-hub-signature-256": sign(body)})

    assert response.status_code == 503


def test_health_and_readiness(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    body = client.get("/readyz").json()
    assert body["status"] in ("ready", "not_ready")
    assert "ai" in body["checks"]


def test_readiness_fails_when_the_bot_could_do_nothing(
    settings: Settings, build_client: Callable[..., AbstractContextManager[TestClient]]
) -> None:
    """Neither answering nor echoing means every message would be dropped, so this is not ready."""
    with build_client(settings) as test_client:
        response = test_client.get("/readyz")
        body = response.json()
        assert response.status_code == 503
        assert body["checks"]["ai"]["enabled"] is False
        assert body["checks"]["draining"] is False


def test_readiness_passes_when_only_echoing_is_configured(
    settings: Settings, build_client: Callable[..., AbstractContextManager[TestClient]]
) -> None:
    settings.whatsapp_echo_mode = True
    with build_client(settings) as test_client:
        assert test_client.get("/readyz").status_code == 200


def test_readiness_fails_while_draining(client: TestClient) -> None:
    client.app.state.task_manager.stop_accepting()
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_readiness_needs_a_usable_provider_when_ai_is_on(
    ai_settings: Settings, build_client: Callable[..., AbstractContextManager[TestClient]]
) -> None:
    """A bot that would drop every message must not be reported as ready."""
    with build_client(ai_settings) as test_client:
        response = test_client.get("/readyz")
        body = response.json()
        assert response.status_code == 503
        assert body["checks"]["ai"]["enabled"] is True
        assert body["checks"]["ai"]["usable_tiers"] == []


def test_readiness_succeeds_once_a_provider_tier_is_available(
    ai_settings: Settings, build_client: Callable[..., AbstractContextManager[TestClient]]
) -> None:
    ai_settings.groq_api_key = SecretStr("gsk_test")
    with build_client(ai_settings) as test_client:
        response = test_client.get("/readyz")
        body = response.json()
        assert response.status_code == 200, body
        assert body["checks"]["ai"]["usable_tiers"] == ["groq"]
        assert body["checks"]["ai"]["circuits"] == {"groq": "closed"}


def test_readiness_survives_one_open_circuit_while_a_backup_tier_is_closed(
    ai_settings: Settings, build_client: Callable[..., AbstractContextManager[TestClient]]
) -> None:
    """A tripped circuit should not take the instance down while a second tier is still closed."""
    ai_settings.groq_api_key = SecretStr("gsk_test")
    ai_settings.gemini_api_key = SecretStr("goog_test")
    with build_client(ai_settings) as test_client:
        breaker = test_client.app.state.llm_pipeline.breakers["groq"]
        for _ in range(breaker.failure_threshold):
            breaker.record_failure()
        response = test_client.get("/readyz")
        body = response.json()
        assert body["checks"]["ai"]["circuits"]["groq"] == "open"
        assert body["checks"]["ai"]["usable_tiers"] == ["gemini"]
        assert response.status_code == 200, body


def test_metrics_endpoint_exposes_application_series(client: TestClient) -> None:
    body = text_message_payload()
    client.post(URL, content=body, headers={"x-hub-signature-256": sign(body)})
    client.get("/healthz")

    exposition = client.get("/metrics").text

    assert "limbot_http_requests_total" in exposition
    assert "limbot_webhook_events_total" in exposition


def test_shutdown_drains_inflight_work(app: FastAPI, monkeypatch) -> None:
    finished = threading.Event()

    async def slow_handler(payload, **kwargs):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0.5)
        finished.set()

    monkeypatch.setattr(message_handler, "handle_webhook", slow_handler)
    body = text_message_payload()

    with TestClient(app) as test_client:
        response = test_client.post(URL, content=body, headers={"x-hub-signature-256": sign(body)})
        assert response.status_code == 200
        assert not finished.is_set()

    assert finished.is_set(), "in-flight work was not drained on shutdown"
