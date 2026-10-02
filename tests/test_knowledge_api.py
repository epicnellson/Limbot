from __future__ import annotations

from app.core.embeddings import EmbeddingError
from app.rag.ingest import IngestRefused, IngestResult

from conftest import _base_settings


class FakeIngestor:
    """Stands in for the real knowledge ingestor so the route can be tested in isolation."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def ingest(self, text: str, *, title: str = "", source: str = "") -> IngestResult:
        self.calls.append((text, title, source))
        return IngestResult(
            document_id="abcd1234" if title else "efgh5678",
            title=title,
            source=source,
            chunks=3,
            characters=len(text),
            duration_ms=12.5,
        )


def knowledge_settings(**overrides: object) -> object:
    return _base_settings(debug_endpoints=True, ai_enabled=True, rag_enabled=True, **overrides)


async def test_knowledge_route_is_not_mounted_when_debug_is_off(build_client: object) -> None:
    with build_client(_base_settings(debug_endpoints=False, ai_enabled=True)) as client:
        response = client.post("/debug/knowledge", json={"text": "hello"})

    assert response.status_code in (404, 405)


async def test_knowledge_route_stores_the_document(build_client: object) -> None:
    with build_client(knowledge_settings()) as client:
        fake = FakeIngestor()
        client.app.state.runtime.ingestor = fake  # type: ignore[attr-defined]

        response = client.post(
            "/debug/knowledge",
            json={"text": "Bayes' theorem on the whiteboard", "title": "Lecture 3"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["document_id"] == "abcd1234"
        assert body["title"] == "Lecture 3"
        assert body["chunks"] == 3
        assert body["duration_ms"] == 12.5
        assert fake.calls == [("Bayes' theorem on the whiteboard", "Lecture 3", "")]


async def test_knowledge_route_passes_source_along(build_client: object) -> None:
    with build_client(knowledge_settings()) as client:
        fake = FakeIngestor()
        client.app.state.runtime.ingestor = fake  # type: ignore[attr-defined]

        response = client.post(
            "/debug/knowledge",
            json={"text": "hello", "source": "stats.pdf"},
        )

        assert response.status_code == 200
        assert fake.calls == [("hello", "", "stats.pdf")]


async def test_knowledge_route_refuses_when_ingestor_is_missing(build_client: object) -> None:
    with build_client(knowledge_settings().model_copy(update={"rag_enabled": False})) as client:
        response = client.post("/debug/knowledge", json={"text": "hello"})

    assert response.status_code == 503
    assert "not enabled" in response.json()["detail"]


async def test_knowledge_route_returns_400_on_refusal(build_client: object) -> None:
    with build_client(knowledge_settings()) as client:

        class RefusingIngestor:
            async def ingest(self, text: str, **kwargs: object) -> IngestResult:
                raise IngestRefused("the document is empty")

        client.app.state.runtime.ingestor = RefusingIngestor()  # type: ignore[attr-defined]

        response = client.post("/debug/knowledge", json={"text": "hello"})

    assert response.status_code == 400
    assert response.json()["detail"] == "the document is empty"


async def test_knowledge_route_returns_502_when_the_embedder_fails(build_client: object) -> None:
    with build_client(knowledge_settings()) as client:

        class BrokenIngestor:
            async def ingest(self, text: str, **kwargs: object) -> IngestResult:
                raise EmbeddingError("model download failed")

        client.app.state.runtime.ingestor = BrokenIngestor()  # type: ignore[attr-defined]

        response = client.post("/debug/knowledge", json={"text": "hello"})

    assert response.status_code == 502
    assert "embedding model failed" in response.json()["detail"]


async def test_knowledge_route_rejects_an_empty_body(build_client: object) -> None:
    with build_client(knowledge_settings()) as client:
        client.app.state.runtime.ingestor = FakeIngestor()  # type: ignore[attr-defined]

        response = client.post("/debug/knowledge", json={"text": ""})

    assert response.status_code == 422


async def test_knowledge_route_rejects_a_missing_body(build_client: object) -> None:
    with build_client(knowledge_settings()) as client:
        client.app.state.runtime.ingestor = FakeIngestor()  # type: ignore[attr-defined]

        response = client.post("/debug/knowledge", json={})

    assert response.status_code == 422
