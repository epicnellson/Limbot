from __future__ import annotations

import pytest
from app.config import Settings
from app.core.embeddings import Embedder, EmbeddingError
from app.core.qdrant import VectorStore
from app.rag.chunker import chunk_text
from app.rag.ingest import MAX_DOCUMENT_CHARS, IngestRefused, KnowledgeIngestor
from app.rag.pipeline import RetrievalPipeline

from conftest import counter_value
from test_rag import FakeBackend, connected_store

LECTURE = (
    "Bayes theorem relates a prior probability to a posterior. "
    "The prior is P(A). The likelihood is P(B|A). The posterior is P(A|B). "
    "The normalisation term is the evidence, P(B), which is the total probability of B. "
    "The theorem is only useful when P(B) is known or can be estimated. "
)


def ingestor_for(
    settings: Settings, store: VectorStore, backend: FakeBackend | None = None
) -> KnowledgeIngestor:
    return KnowledgeIngestor(settings, Embedder(settings, backend or FakeBackend()), store)


def rag_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "qdrant_collection": "test_ingest",
            "embedding_dimensions": 4,
            "rag_enabled": True,
            "knowledge_chunk_chars": 200,
            "knowledge_chunk_overlap": 40,
        }
    )


async def test_a_document_is_chunked_embedded_and_stored(settings: Settings) -> None:
    resolved = rag_settings(settings)
    backend = FakeBackend()
    async with connected_store(resolved) as store:
        result = await ingestor_for(resolved, store, backend).ingest(
            LECTURE, title="Lecture 3", source="stats.pdf"
        )

        assert result.chunks > 0
        assert result.title == "Lecture 3"
        assert result.source == "stats.pdf"
        # Ingest normalises the text before it counts or hashes it, so the trailing newline
        # in the fixture does not survive.
        assert result.characters == len(LECTURE.strip())
        assert result.duration_ms >= 0
        assert await store.count() == result.chunks
        assert backend.batches


async def test_stored_chunks_carry_their_metadata(settings: Settings) -> None:
    resolved = rag_settings(settings)
    async with connected_store(resolved) as store:
        await ingestor_for(resolved, store).ingest(LECTURE, title="Lecture 3", source="stats.pdf")

        hits = await store.search([0.0, 1.0, 0.0, 0.0], limit=5, score_threshold=None)

    assert hits
    for hit in hits:
        assert hit.payload["title"] == "Lecture 3"
        assert hit.payload["source"] == "stats.pdf"
        assert isinstance(hit.payload["ordinal"], int)
        assert hit.payload["text"].strip()
        assert hit.payload["document_id"]


async def test_the_same_document_produces_the_same_id(settings: Settings) -> None:
    resolved = rag_settings(settings)
    async with connected_store(resolved) as store:
        ingestor = ingestor_for(resolved, store)
        first = await ingestor.ingest(LECTURE, title="Lecture 3")
        second = await ingestor.ingest(LECTURE, title="Lecture 3")

    assert first.document_id == second.document_id


async def test_different_material_produces_a_different_id(settings: Settings) -> None:
    resolved = rag_settings(settings)
    async with connected_store(resolved) as store:
        ingestor = ingestor_for(resolved, store)
        first = await ingestor.ingest(LECTURE, title="Lecture 3")
        second = await ingestor.ingest(LECTURE, title="Lecture 4")

    assert first.document_id != second.document_id


async def test_the_collection_is_created_on_demand(settings: Settings) -> None:
    resolved = rag_settings(settings).model_copy(
        update={"qdrant_collection": "brand_new_collection"}
    )
    async with connected_store(resolved) as store:
        assert not await store.client.collection_exists("brand_new_collection")

        result = await ingestor_for(resolved, store).ingest(LECTURE, title="Lecture 3")

    assert result.chunks > 0


async def test_an_empty_document_is_refused(settings: Settings) -> None:
    resolved = rag_settings(settings)
    async with connected_store(resolved) as store:
        with pytest.raises(IngestRefused, match="empty"):
            await ingestor_for(resolved, store).ingest("   \n  ")


async def test_a_whitespace_only_document_is_refused(settings: Settings) -> None:
    resolved = rag_settings(settings)
    async with connected_store(resolved) as store:
        with pytest.raises(IngestRefused):
            await ingestor_for(resolved, store).ingest("\n\n\t")


async def test_an_oversized_document_is_refused_before_any_work(settings: Settings) -> None:
    resolved = rag_settings(settings)
    backend = FakeBackend()
    async with connected_store(resolved) as store:
        with pytest.raises(IngestRefused, match="limit"):
            await ingestor_for(resolved, store, backend).ingest("x" * (MAX_DOCUMENT_CHARS + 1))

    assert backend.batches == []


async def test_ingest_is_refused_when_retrieval_is_switched_off(settings: Settings) -> None:
    resolved = rag_settings(settings).model_copy(update={"rag_enabled": False})
    async with connected_store(resolved) as store:
        with pytest.raises(IngestRefused, match="disabled"):
            await ingestor_for(resolved, store).ingest(LECTURE)


async def test_a_failing_embedding_model_is_surfaced_as_an_embedding_error(
    settings: Settings,
) -> None:
    resolved = rag_settings(settings)
    async with connected_store(resolved) as store:
        with pytest.raises(EmbeddingError):
            await ingestor_for(resolved, store, FakeBackend(fail=True)).ingest(LECTURE)


async def test_nothing_is_stored_when_embedding_fails(settings: Settings) -> None:
    resolved = rag_settings(settings)
    async with connected_store(resolved) as store:
        with pytest.raises(EmbeddingError):
            await ingestor_for(resolved, store, FakeBackend(fail=True)).ingest(LECTURE)

        assert await store.count() == 0


async def test_ingestion_counts_every_chunk_it_stores(settings: Settings) -> None:
    resolved = rag_settings(settings)
    async with connected_store(resolved) as store:
        before = counter_value("limbot_rag_chunks_total", disposition="ingested")
        result = await ingestor_for(resolved, store).ingest(LECTURE, title="Lecture 3")
        after = counter_value("limbot_rag_chunks_total", disposition="ingested")

    assert after - before == result.chunks


async def test_ingested_material_can_be_retrieved_again(settings: Settings) -> None:
    """The point of the whole exercise: what is stored can be found by a later question."""
    resolved = rag_settings(settings)
    backend = FakeBackend()
    async with connected_store(resolved) as store:
        await ingestor_for(resolved, store, backend).ingest(LECTURE, title="Lecture 3")

        retrieval = RetrievalPipeline(resolved, Embedder(resolved, backend), store)
        # The fake backend maps a query onto a unit vector, so query the same leading character
        # the stored chunks begin with.
        result = await retrieval.retrieve("Bayes")

    assert result.reason == "hit"
    assert result.chunks
    assert "Bayes" in result.context
    assert result.sources == ()


async def test_a_question_far_from_the_material_retrieves_nothing(settings: Settings) -> None:
    resolved = rag_settings(settings).model_copy(update={"rag_score_threshold": 0.9})
    async with connected_store(resolved) as store:
        await ingestor_for(resolved, store).ingest(LECTURE, title="Lecture 3")

        retrieval = RetrievalPipeline(resolved, Embedder(resolved, FakeBackend()), store)
        # The fake backend buckets a string by its first character, so pick a letter no stored
        # chunk starts with to be sure this scores 0.0 against all of them.
        starts = {
            chunk.text[0]
            for chunk in chunk_text(
                LECTURE,
                max_chars=resolved.knowledge_chunk_chars,
                overlap=resolved.knowledge_chunk_overlap,
            )
        }
        query = next(letter for letter in "abcdefghijklmnopqrstuvwxyz" if letter not in starts)
        result = await retrieval.retrieve(query)

    assert result.reason == "below_threshold"
    assert result.context == ""


async def test_a_long_document_produces_several_chunks(settings: Settings) -> None:
    resolved = rag_settings(settings)
    async with connected_store(resolved) as store:
        result = await ingestor_for(resolved, store).ingest(LECTURE * 3, title="Long lecture")

    assert result.chunks > 1


async def test_describe_reports_the_settings_that_shape_chunking(settings: Settings) -> None:
    resolved = rag_settings(settings)
    async with connected_store(resolved) as store:
        described = ingestor_for(resolved, store).describe()

    assert described["enabled"] is True
    assert described["collection"] == "test_ingest"
    assert described["chunk_chars"] == 200
    assert described["chunk_overlap"] == 40
    assert described["max_document_chars"] == MAX_DOCUMENT_CHARS


async def test_ingest_refusal_does_not_create_a_collection(settings: Settings) -> None:
    resolved = rag_settings(settings).model_copy(update={"qdrant_collection": "never_created"})
    async with connected_store(resolved) as store:
        with pytest.raises(IngestRefused):
            await ingestor_for(resolved, store).ingest("  ")

        assert not await store.client.collection_exists("never_created")
