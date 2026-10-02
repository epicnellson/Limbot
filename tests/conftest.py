from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

import pytest
from app import metrics
from app.config import Settings, get_settings
from fastapi import FastAPI
from fastapi.testclient import TestClient

APP_SECRET = "test-app-secret"
VERIFY_TOKEN = "test-verify-token"

ENVIRONMENT_OVERRIDES = {
    "ENVIRONMENT": "local",
    "WHATSAPP_APP_SECRET": APP_SECRET,
    "WHATSAPP_VERIFY_TOKEN": VERIFY_TOKEN,
    "WHATSAPP_SIGNATURE_REQUIRED": "true",
    "WHATSAPP_ECHO_MODE": "false",
    "WHATSAPP_ACCESS_TOKEN": "",
    "WHATSAPP_PHONE_NUMBER_ID": "",
    "QDRANT_URL": "http://127.0.0.1:6333",
    "DEBUG_ENDPOINTS": "false",
    "BACKGROUND_DRAIN_TIMEOUT_SECONDS": "5",
    "LOG_LEVEL": "WARNING",
    "LOG_JSON": "false",
}


def sign(body: bytes, secret: str = APP_SECRET) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def text_message_payload(
    *,
    message_id: str = "wamid.TEST0001",
    body: str = "hello limbot",
    sender: str = "15550001111",
    timestamp: str = "1700000000",
) -> bytes:
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "102290129340398",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15550002222",
                                "phone_number_id": "1234567890",
                            },
                            "contacts": [{"profile": {"name": "Ada"}, "wa_id": sender}],
                            "messages": [
                                {
                                    "from": sender,
                                    "id": message_id,
                                    "timestamp": timestamp,
                                    "type": "text",
                                    "text": {"body": body},
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }
    return json.dumps(payload).encode("utf-8")


def counter_value(name: str, **labels: str) -> float:
    """Read a counter out of the exposition text, for delta assertions."""
    rendered = ",".join(f'{key}="{value}"' for key, value in sorted(labels.items()))
    wanted = f"{name}{{{rendered}}}"
    for line in metrics.render().decode().splitlines():
        if line.startswith(wanted):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def _base_settings(**overrides: object) -> Settings:
    """Hermetic settings. ``_env_file=None`` keeps a developer's local .env out of the tests."""
    defaults: dict[str, object] = {
        "environment": "local",
        "whatsapp_app_secret": APP_SECRET,
        "whatsapp_verify_token": VERIFY_TOKEN,
        "whatsapp_signature_required": True,
        "whatsapp_echo_mode": False,
        "qdrant_url": "http://127.0.0.1:6333",
        "qdrant_required": False,
        "ai_enabled": False,
        "rag_enabled": False,
        "log_json": False,
        "log_level": "WARNING",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)  # type: ignore[arg-type]


@pytest.fixture
def settings() -> Settings:
    return _base_settings()


@pytest.fixture
def ai_settings() -> Settings:
    """The same baseline with the AI pipeline on and no provider credentials.

    No Postgres DSN and no usable Qdrant, so the lazy imports of asyncpg and fastembed never
    happen and these tests stay hermetic.
    """
    return _base_settings(ai_enabled=True)


@pytest.fixture
def build_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[..., AbstractContextManager[TestClient]]]:
    """Build a throwaway app from explicit settings, for cases the shared fixtures cannot cover."""
    for key, value in ENVIRONMENT_OVERRIDES.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()

    # Imported here rather than at module scope: app.main builds an app when it is imported, so
    # the environment has to be in place before this line runs.
    from app.main import create_app

    @contextmanager
    def factory(settings: Settings) -> Iterator[TestClient]:
        with TestClient(create_app(settings)) as test_client:
            yield test_client

    yield factory
    get_settings.cache_clear()


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> Iterator[FastAPI]:
    for key, value in ENVIRONMENT_OVERRIDES.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()

    from app.main import create_app

    application = create_app(get_settings())
    yield application

    get_settings.cache_clear()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
