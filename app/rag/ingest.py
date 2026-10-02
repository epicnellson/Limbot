"""Loading course material into the vector store."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from app import metrics
from app.config import Settings
from app.core.embeddings import Embedder, EmbeddingError
from app.core.qdrant import VectorStore
from app.rag.chunker import chunk_document

logger = logging.getLogger(__name__)

MAX_DOCUMENT_CHARS = 200_000
MAX_BATCH = 32


@dataclass(frozen=True, slots=True)
class IngestResult:
    """What one ingest call put into the collection."""

    document_id: str
    title: str
    source: str
    chunks: int
    characters: int
    duration_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "source": self.source,
            "chunks": self.chunks,
            "characters": self.characters,
            "duration_ms": round(self.duration_ms, 1),
        }


class IngestRefused(ValueError):
    """The document cannot be ingested, with a reason worth returning to the caller."""


class KnowledgeIngestor:
    """Chunk, embed and upsert one document.

    The document id is derived from the content, not generated, so re-ingesting the same
    material produces the same ids and therefore replaces the old points instead of piling up
    near-duplicates in the collection.
    """

    def __init__(self, settings: Settings, embedder: Embedder, store: VectorStore) -> None:
        self._settings = settings
        self._embedder = embedder
        self._store = store

    @property
    def enabled(self) -> bool:
        return self._settings.vector_store_enabled

    async def ingest(self, text: str, *, title: str = "", source: str = "") -> IngestResult:
        if not self.enabled:
            raise IngestRefused("retrieval is disabled: set RAG_ENABLED and QDRANT_URL")
        body = text.strip()
        if not body:
            raise IngestRefused("the document is empty")
        if len(body) > MAX_DOCUMENT_CHARS:
            raise IngestRefused(
                f"the document is {len(body)} characters, the limit is {MAX_DOCUMENT_CHARS}"
            )

        started = time.perf_counter()
        document_id = _document_id(body, title)
        payloads = chunk_document(
            body,
            title=title or document_id,
            source=source,
            max_chars=self._settings.knowledge_chunk_chars,
            overlap=self._settings.knowledge_chunk_overlap,
        )
        if not payloads:
            raise IngestRefused("the document produced no chunks")

        await self._store.ensure_collection(self._embedder.dimensions)

        vectors: list[list[float]] = []
        for start in range(0, len(payloads), MAX_BATCH):
            batch = payloads[start : start + MAX_BATCH]
            texts = [str(chunk["text"]) for chunk in batch]
            try:
                vectors.extend(await self._embedder.embed(texts))
            except EmbeddingError:
                raise

        for payload in payloads:
            payload["document_id"] = document_id

        written = await self._store.upsert_chunks(payloads, vectors)
        elapsed = (time.perf_counter() - started) * 1000
        metrics.RAG_CHUNKS.labels(disposition="ingested").inc(written)

        logger.info(
            "document ingested",
            extra={
                "context": {
                    "document_id": document_id,
                    "title": title,
                    "source": source,
                    "chunks": written,
                    "characters": len(body),
                    "duration_ms": round(elapsed, 1),
                }
            },
        )
        return IngestResult(
            document_id=document_id,
            title=title,
            source=source,
            chunks=written,
            characters=len(body),
            duration_ms=elapsed,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "collection": self._store.collection,
            "chunk_chars": self._settings.knowledge_chunk_chars,
            "chunk_overlap": self._settings.knowledge_chunk_overlap,
            "max_document_chars": MAX_DOCUMENT_CHARS,
        }


def _document_id(text: str, title: str) -> str:
    """A stable id for the material, so re-ingesting replaces rather than duplicates."""
    digest = hashlib.sha256(f"{title.strip()}\n{text}".encode())
    return digest.hexdigest()[:32]
