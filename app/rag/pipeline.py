from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from app import metrics
from app.config import Settings
from app.core.embeddings import Embedder
from app.core.qdrant import SearchHit, VectorStore

logger = logging.getLogger(__name__)

CONTEXT_HEADER = "Course material retrieved for this question:"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    text: str
    score: float
    title: str
    source: str


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """What retrieval produced, plus whether it is worth showing to the model at all."""

    chunks: tuple[RetrievedChunk, ...] = ()
    context: str = ""
    reason: str = "disabled"
    considered: int = 0
    duration_ms: float = 0.0
    sources: tuple[str, ...] = ()

    @property
    def used(self) -> bool:
        return bool(self.chunks) and bool(self.context)


def format_context(chunks: list[RetrievedChunk], max_chars: int) -> str:
    """Numbered excerpts, truncated as a whole so the model never sees a cut-off block."""
    lines: list[str] = []
    used = 0
    for number, chunk in enumerate(chunks, start=1):
        title = f" {chunk.title}" if chunk.title else ""
        header = f"[{number}]{title}"
        remaining = max_chars - used - len(header) - 2
        if remaining <= 0:
            break
        body = chunk.text
        suffix = ""
        if len(body) > remaining:
            body = body[: max(0, remaining - 1)].rstrip() + "..."
            suffix = " (truncated)"
        line = f"{header}{suffix}\n{body}"
        lines.append(line)
        used += len(line) + 2
    if not lines:
        return ""
    return f"{CONTEXT_HEADER}\n\n" + "\n\n".join(lines)


class RetrievalPipeline:
    """Embed the question, ask Qdrant, keep only confident matches.

    The threshold is the point of this stage. A bot that pastes loosely related passages into
    the prompt makes the model answer from them, so anything below the bar is dropped and the
    model is told, in the system prompt, to say it does not know instead of inventing.
    """

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._store = store

    @property
    def enabled(self) -> bool:
        return self._settings.vector_store_enabled

    async def retrieve(
        self, question: str, *, audiences: tuple[str, ...] | None = None
    ) -> RetrievalResult:
        if not self.enabled:
            return RetrievalResult(reason="disabled")
        # An empty audience set is not "no filter": it means this caller may retrieve nothing.
        # Returning before the query keeps a blocked sender from even reaching the vector store.
        if audiences is not None and not audiences:
            return RetrievalResult(reason="no_access")
        query = question.strip()
        if not query:
            return RetrievalResult(reason="empty_query")

        started = time.perf_counter()
        try:
            vector = await self._embedder.embed_one(query)
            hits = await self._store.search(
                vector,
                limit=self._settings.rag_top_k,
                score_threshold=self._settings.rag_score_threshold,
                audiences=audiences,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            metrics.RAG_QUERIES.labels(result="error").inc()
            metrics.RAG_LATENCY.observe(elapsed)
            logger.warning(
                "retrieval failed, continuing without context",
                extra={"context": {"error": f"{type(exc).__name__}: {exc}"}},
            )
            return RetrievalResult(reason="error")

        chunks = [
            RetrievedChunk(
                text=hit.text,
                score=hit.score,
                title=_label(hit, "title"),
                source=_label(hit, "source"),
            )
            for hit in hits
            if hit.text.strip()
        ]
        context = format_context(chunks, self._settings.rag_max_context_chars)
        elapsed = time.perf_counter() - started
        metrics.RAG_LATENCY.observe(elapsed)
        metrics.RAG_QUERIES.labels(result="hit" if chunks else "miss").inc()
        metrics.RAG_CHUNKS.labels(disposition="returned").inc(len(chunks))

        logger.info(
            "retrieval complete",
            extra={
                "context": {
                    "chunks": len(chunks),
                    "considered": len(hits),
                    "audiences": list(audiences) if audiences is not None else None,
                    "top_score": round(chunks[0].score, 4) if chunks else None,
                    "duration_ms": round(elapsed * 1000, 1),
                }
            },
        )
        return RetrievalResult(
            chunks=tuple(chunks),
            context=context,
            reason="hit" if chunks else "below_threshold",
            considered=len(hits),
            duration_ms=elapsed * 1000,
            sources=tuple(dict.fromkeys(chunk.source for chunk in chunks if chunk.source)),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "collection": self._store.collection,
            "embedding": self._embedder.describe(),
            "top_k": self._settings.rag_top_k,
            "score_threshold": self._settings.rag_score_threshold,
        }


def _label(hit: SearchHit, key: str) -> str:
    value = hit.payload.get(key)
    return value if isinstance(value, str) else ""
