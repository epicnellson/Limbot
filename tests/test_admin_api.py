from __future__ import annotations

from app.core.embeddings import EmbeddingError
from app.rag.ingest import IngestRefused, IngestResult

from conftest import _base_settings

TOKEN = "test-admin-token"


class FakeIngestor:
    """Stands in for the real ingestor so the route can be tested without Qdrant."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str]] = []

    async def ingest(
        self, text: str, *, title: str = "", source: str = "", audience: str = "student"
    ) -> IngestResult:
        self.calls.append((text, title, source, audience))
        return IngestResult(
            document_id="abcd1234",
            title=title,
            source=source,
            chunks=3,
            characters=len(text),
            duration_ms=12.5,
            audience=audience,
        )


def admin_settings(**overrides: object) -> object:
    return _base_settings(ai_enabled=True, rag_enabled=True, admin_api_token=TOKEN, **overrides)


def test_admin_route_is_gated_by_a_token(build_client: object) -> None:
    with build_client(admin_settings()) as client:
        client.app.state.runtime.ingestor = FakeIngestor()  # type: ignore[attr-defined]

        missing = client.post("/admin/knowledge", json={"text": "hello"})
        wrong = client.post(
            "/admin/knowledge",
            json={"text": "hello"},
            headers={"Authorization": "Bearer nope"},
        )

    assert missing.status_code == 401
    assert wrong.status_code == 401


def test_admin_route_fails_closed_when_no_token_is_configured(build_client: object) -> None:
    with build_client(_base_settings(ai_enabled=True, rag_enabled=True)) as client:
        client.app.state.runtime.ingestor = FakeIngestor()  # type: ignore[attr-defined]

        response = client.post(
            "/admin/knowledge",
            json={"text": "hello"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 503


def test_admin_route_stores_the_document_for_an_audience(build_client: object) -> None:
    with build_client(admin_settings()) as client:
        fake = FakeIngestor()
        client.app.state.runtime.ingestor = fake  # type: ignore[attr-defined]

        response = client.post(
            "/admin/knowledge",
            json={"text": "Staff handbook", "title": "Handbook", "audience": "staff"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["audience"] == "staff"
        assert body["chunks"] == 3
        assert fake.calls == [("Staff handbook", "Handbook", "", "staff")]


def test_admin_route_defaults_the_audience_to_student(build_client: object) -> None:
    with build_client(admin_settings()) as client:
        fake = FakeIngestor()
        client.app.state.runtime.ingestor = fake  # type: ignore[attr-defined]

        response = client.post(
            "/admin/knowledge",
            json={"text": "hello"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 200
    assert fake.calls == [("hello", "", "", "student")]


def test_admin_route_mounts_in_production(build_client: object) -> None:
    settings = _base_settings(
        environment="production",
        ai_enabled=True,
        rag_enabled=True,
        postgres_dsn="postgresql://limbot:limbot@localhost:5432/limbot",
        whatsapp_access_token="token",
        whatsapp_phone_number_id="123",
        groq_api_key="gsk-test",
        admin_api_token=TOKEN,
    )
    with build_client(settings) as client:
        client.app.state.runtime.ingestor = FakeIngestor()  # type: ignore[attr-defined]

        response = client.post(
            "/admin/knowledge",
            json={"text": "hello"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 200


def test_admin_route_refuses_when_ingestor_is_missing(build_client: object) -> None:
    with build_client(admin_settings()) as client:
        client.app.state.runtime.ingestor = None  # type: ignore[attr-defined]

        response = client.post(
            "/admin/knowledge",
            json={"text": "hello"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 503


def test_admin_route_reports_a_refusal_as_400(build_client: object) -> None:
    with build_client(admin_settings()) as client:

        class RefusingIngestor:
            async def ingest(self, text: str, **kwargs: object) -> IngestResult:
                raise IngestRefused("unknown audience 'wizard'")

        client.app.state.runtime.ingestor = RefusingIngestor()  # type: ignore[attr-defined]

        response = client.post(
            "/admin/knowledge",
            json={"text": "hello", "audience": "wizard"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 400
    assert "wizard" in response.json()["detail"]


def test_admin_route_reports_an_embedder_failure_as_502(build_client: object) -> None:
    with build_client(admin_settings()) as client:

        class BrokenIngestor:
            async def ingest(self, text: str, **kwargs: object) -> IngestResult:
                raise EmbeddingError("model download failed")

        client.app.state.runtime.ingestor = BrokenIngestor()  # type: ignore[attr-defined]

        response = client.post(
            "/admin/knowledge",
            json={"text": "hello"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 502


def test_admin_route_rejects_an_empty_body(build_client: object) -> None:
    with build_client(admin_settings()) as client:
        client.app.state.runtime.ingestor = FakeIngestor()  # type: ignore[attr-defined]

        response = client.post(
            "/admin/knowledge",
            json={"text": ""},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 422
