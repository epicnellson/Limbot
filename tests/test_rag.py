from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from app.config import Settings
from app.core.embeddings import Embedder, EmbeddingError
from app.core.qdrant import VectorStore
from app.rag.pipeline import RetrievalPipeline, RetrievedChunk, format_context


class FakeBackend:
    """Deterministic 'embeddings': a unit vector picked by the first character.

    This mirrors :func:`unit`, so a stored chunk whose vector is ``unit("a")`` matches a query
    of ``"a"`` with a cosine score of exactly 1.0 and a score of 0.0 with anything else.
    """

    name = "fake-model"

    def __init__(self, dimensions: int = 4, *, fail: bool = False) -> None:
        self._dimensions = dimensions
        self.fail = fail
        self.batches: list[list[str]] = []

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: object) -> list[list[float]]:
        items = list(texts)  # type: ignore[arg-type]
        if self.fail:
            raise RuntimeError("model download failed")
        self.batches.append(items)
        return [unit(text, self._dimensions) for text in items]


def unit(text: str, dimensions: int = 4) -> list[float]:
    vector = [0.0] * dimensions
    if text:
        vector[ord(text[0]) % dimensions] = 1.0
    return vector


@asynccontextmanager
async def connected_store(settings: Settings) -> AsyncIterator[VectorStore]:
    """Connect to a real Qdrant on a collection this test owns outright.

    Qdrant outlives a single test and several tests deliberately share one collection name, so the
    collection is dropped on the way in and on the way out. Without that, a test that counts points
    or searches for hits also sees whatever the previous test upserted.
    """
    store = VectorStore(settings)
    await store.connect()
    # Captured before the test body runs: a test that simulates a broken store swaps the client out
    # from under the store, and cleanup still has to reach the real server through the original.
    client = store.client
    name = store.collection

    async def drop_collection() -> None:
        if await client.collection_exists(name):
            await client.delete_collection(name)

    await drop_collection()
    try:
        yield store
    finally:
        try:
            await drop_collection()
        finally:
            await store.close()


@pytest.fixture
def store_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "qdrant_collection": "test_knowledge",
            "embedding_dimensions": 4,
            "rag_enabled": True,
        }
    )


def pipeline_for(settings: Settings, store: VectorStore) -> RetrievalPipeline:
    return RetrievalPipeline(settings, Embedder(settings, FakeBackend()), store)


async def test_ensure_collection_creates_it_once(store_settings: Settings) -> None:
    async with connected_store(store_settings) as store:
        assert await store.ensure_collection(4) is True
        assert await store.ensure_collection(4) is False
        assert await store.count() == 0


async def test_upsert_and_search_round_trip(store_settings: Settings) -> None:
    async with connected_store(store_settings) as store:
        await store.ensure_collection(4)
        chunks = [
            {
                "text": "eigenvalues come first",
                "title": "Week 3",
                "source": "la.md",
                "ordinal": 0,
            },
            {"text": "matrices come second", "title": "Week 4", "source": "la.md", "ordinal": 1},
        ]
        written = await store.upsert_chunks(chunks, [unit("a"), unit("b")])

        assert written == 2
        assert await store.count() == 2

        hits = await store.search(unit("a"), limit=5, score_threshold=0.0)

    assert [hit.text for hit in hits] == [
        "eigenvalues come first",
        "matrices come second",
    ]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].payload["source"] == "la.md"
    assert hits[0].payload["chunk_id"] == hits[0].id


async def test_search_drops_everything_below_the_threshold(store_settings: Settings) -> None:
    async with connected_store(store_settings) as store:
        await store.ensure_collection(4)
        await store.upsert_chunks([{"text": "a", "title": "", "source": ""}], [unit("a")])

        assert await store.search(unit("b"), limit=5, score_threshold=0.5) == []
        assert len(await store.search(unit("b"), limit=5, score_threshold=0.0)) == 1


async def test_search_honours_the_limit(store_settings: Settings) -> None:
    async with connected_store(store_settings) as store:
        await store.ensure_collection(4)
        await store.upsert_chunks(
            [{"text": "one"}, {"text": "two"}, {"text": "three"}],
            [unit("a"), unit("b"), unit("c")],
        )
        hits = await store.search(unit("a"), limit=2, score_threshold=0.0)

    assert len(hits) == 2


async def test_upsert_rejects_mismatched_lengths(store_settings: Settings) -> None:
    async with connected_store(store_settings) as store:
        await store.ensure_collection(4)
        with pytest.raises(ValueError, match="2 chunks but 1 vectors"):
            await store.upsert_chunks([{"text": "a"}, {"text": "b"}], [unit("a")])


async def test_upsert_of_nothing_writes_nothing(store_settings: Settings) -> None:
    async with connected_store(store_settings) as store:
        assert await store.upsert_chunks([], []) == 0


async def test_using_the_store_before_connecting_fails_clearly(
    store_settings: Settings,
) -> None:
    store = VectorStore(store_settings)
    with pytest.raises(RuntimeError, match="not connected"):
        _ = store.client


async def test_ping_reports_an_unreachable_server(store_settings: Settings) -> None:
    store = VectorStore(store_settings)
    await store.connect()
    object.__setattr__(store, "_client", _BrokenClient())

    reachable, latency_ms, error = await store.ping()

    assert reachable is False
    assert latency_ms >= 0
    assert error is not None
    assert "Boom" in error


class _BrokenClient:
    async def get_collections(self) -> None:
        raise RuntimeError("Boom")


async def test_embedder_returns_vectors_for_a_batch(store_settings: Settings) -> None:
    backend = FakeBackend()
    embedder = Embedder(store_settings, backend)

    vectors = await embedder.embed(["alpha", "beta"])

    assert len(vectors) == 2
    assert all(len(vector) == 4 for vector in vectors)
    assert backend.batches == [["alpha", "beta"]]
    assert embedder.describe() == {"model": "fake-model", "dimensions": 4}


async def test_embedder_skips_blank_input(store_settings: Settings) -> None:
    backend = FakeBackend()
    embedder = Embedder(store_settings, backend)

    assert await embedder.embed([]) == []
    assert await embedder.embed(["  ", ""]) == []
    assert backend.batches == []


async def test_embedder_reports_backend_failures(store_settings: Settings) -> None:
    embedder = Embedder(store_settings, FakeBackend(fail=True))

    with pytest.raises(EmbeddingError, match="model download failed"):
        await embedder.embed(["alpha"])


async def test_embedder_rejects_a_dimension_mismatch(store_settings: Settings) -> None:
    class LyingBackend(FakeBackend):
        @property
        def dimensions(self) -> int:
            return 4

        def embed(self, texts: object) -> list[list[float]]:
            return [[0.5] * 8 for _ in texts]  # type: ignore[attr-defined]

    embedder = Embedder(store_settings, LyingBackend())

    with pytest.raises(EmbeddingError, match="EMBEDDING_DIMENSIONS"):
        await embedder.embed(["alpha"])


async def test_embedder_rejects_a_short_batch(store_settings: Settings) -> None:
    class ShortBackend(FakeBackend):
        def embed(self, texts: object) -> list[list[float]]:
            return super().embed(list(texts)[:1])  # type: ignore[arg-type]

    embedder = Embedder(store_settings, ShortBackend())

    with pytest.raises(EmbeddingError, match="2 input"):
        await embedder.embed(["alpha", "beta"])


async def test_retrieval_returns_context_for_a_confident_match(
    store_settings: Settings,
) -> None:
    async with connected_store(store_settings) as store:
        await store.ensure_collection(4)
        await store.upsert_chunks(
            [
                {
                    "text": "Eigenvalues are the scalars that survive a matrix.",
                    "title": "Week 3",
                    "source": "notes/week-03.md",
                }
            ],
            [unit("a")],
        )
        result = await pipeline_for(store_settings, store).retrieve("a")

    assert result.used is True
    assert result.reason == "hit"
    assert result.chunks[0].title == "Week 3"
    assert result.sources == ("notes/week-03.md",)
    assert "Eigenvalues are the scalars" in result.context
    assert "[1] Week 3" in result.context
    assert result.duration_ms > 0


async def test_retrieval_returns_nothing_when_the_threshold_is_not_met(
    store_settings: Settings,
) -> None:
    strict = store_settings.model_copy(update={"rag_score_threshold": 0.9})
    async with connected_store(strict) as store:
        await store.ensure_collection(4)
        await store.upsert_chunks([{"text": "unrelated", "title": "", "source": ""}], [unit("a")])
        result = await pipeline_for(strict, store).retrieve("b")

    assert result.used is False
    assert result.reason == "below_threshold"
    assert result.context == ""


async def test_retrieval_survives_a_broken_store(store_settings: Settings) -> None:
    async with connected_store(store_settings) as store:
        await store.ensure_collection(4)
        object.__setattr__(store, "_client", _NoCollectionsClient())
        result = await pipeline_for(store_settings, store).retrieve("alpha")

    assert result.used is False
    assert result.reason == "error"
    assert result.context == ""


class _NoCollectionsClient:
    async def query_points(self, **kwargs: object) -> None:
        raise RuntimeError("collection missing")

    async def close(self) -> None:
        return None


async def test_retrieval_is_skipped_when_rag_is_off(settings: Settings) -> None:
    off = settings.model_copy(update={"rag_enabled": False})
    async with connected_store(off) as store:
        result = await pipeline_for(off, store).retrieve("anything")
        assert pipeline_for(off, store).enabled is False

    assert result.reason == "disabled"


async def test_retrieval_ignores_an_empty_question(store_settings: Settings) -> None:
    async with connected_store(store_settings) as store:
        assert (await pipeline_for(store_settings, store).retrieve("   ")).reason == "empty_query"


def test_format_context_truncates_on_a_block_boundary() -> None:
    chunks = [
        RetrievedChunk("a" * 100, 0.9, "Week 1", "one.md"),
        RetrievedChunk("b" * 100, 0.8, "Week 2", "two.md"),
        RetrievedChunk("c" * 100, 0.7, "Week 3", "three.md"),
    ]
    context = format_context(chunks, 200)

    assert "[1] Week 1" in context
    assert "[2] Week 2" in context
    assert "(truncated)" in context
    assert "[3]" not in context


def test_format_context_of_nothing_is_empty() -> None:
    assert format_context([], 500) == ""


def test_format_context_marks_a_truncated_excerpt() -> None:
    context = format_context([RetrievedChunk("x" * 300, 0.9, "Week 1", "one.md")], 120)
    assert "(truncated)" in context


async def test_describe_reports_the_configuration(store_settings: Settings) -> None:
    async with connected_store(store_settings) as store:
        described = pipeline_for(store_settings, store).describe()

    assert described["enabled"] is True
    assert described["collection"] == "test_knowledge"
    assert described["top_k"] == store_settings.rag_top_k
    assert described["embedding"]["dimensions"] == 4
