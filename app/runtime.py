"""Wiring the Phase 2 components together, in one place.

Startup is deliberately forgiving. A missing Postgres DSN turns the student tools off, an
unreachable Qdrant leaves retrieval reporting misses, and an unconfigured provider tier is
simply left out of the fallback chain. The bot answers as much as it can instead of refusing to
start, because a student asking a question is worse served by an error page than by a partial
answer.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings
from app.conversation.store import ConversationStore
from app.core.embeddings import Embedder
from app.core.qdrant import VectorStore
from app.db.pool import Database
from app.db.students import StudentRepository
from app.llm.pipeline import LLMPipeline
from app.rag.ingest import KnowledgeIngestor
from app.rag.pipeline import RetrievalPipeline
from app.services.answer import AnswerService
from app.tools.registry import ToolRegistry
from app.tools.student import build_student_registry

logger = logging.getLogger(__name__)

# A stopped machine behind the network proxy needs a few seconds to wake, so the first
# connection attempt can be refused or fail to resolve. The bounds keep the whole wait well
# inside the health check grace period: a boot that loses the race still starts with the tools
# off rather than late.
DATABASE_OPEN_ATTEMPTS = 4
DATABASE_OPEN_RETRY_SECONDS = 3.0


@dataclass(slots=True)
class AiRuntime:
    """Everything the request path needs, owned by the lifespan."""

    settings: Settings
    http: httpx.AsyncClient
    vector_store: VectorStore
    pipeline: LLMPipeline
    answers: AnswerService
    database: Database | None = None
    tools: ToolRegistry | None = None
    retrieval: RetrievalPipeline | None = None
    ingestor: KnowledgeIngestor | None = None

    async def close(self) -> None:
        if self.database is not None:
            await self.database.close()
        await self.vector_store.close()

    def describe(self) -> dict[str, Any]:
        return {
            "ai": {
                "enabled": self.answers.enabled,
                "system_prompt_chars": len(self.settings.ai_system_prompt),
                "max_tool_rounds": self.settings.ai_max_tool_rounds,
                **self.pipeline.describe(),
            },
            "tools": {
                "enabled": self.tools is not None,
                "names": self.tools.describe() if self.tools else [],
            },
            "retrieval": self.retrieval.describe() if self.retrieval else {"enabled": False},
            "ingest": self.ingestor.describe() if self.ingestor else {"enabled": False},
            "database": self.database.describe() if self.database else {"enabled": False},
        }


async def build_runtime(settings: Settings, http: httpx.AsyncClient) -> AiRuntime:
    vector_store = VectorStore(settings)
    pipeline = LLMPipeline(settings)
    conversations = ConversationStore(settings)

    retrieval: RetrievalPipeline | None = None
    ingestor: KnowledgeIngestor | None = None
    if settings.vector_store_enabled:
        embedder = Embedder(settings)
        retrieval = RetrievalPipeline(settings, embedder, vector_store)
        ingestor = KnowledgeIngestor(settings, embedder, vector_store)
        await _probe_vector_store(settings, embedder, vector_store)

    database: Database | None = None
    tools: ToolRegistry | None = None
    if settings.tools_enabled:
        database = await _open_database(settings)

    if database is not None:
        tools = build_student_registry(
            StudentRepository(database),
            timeout_seconds=settings.ai_tool_call_timeout_seconds,
        )

    answers = AnswerService(
        settings,
        pipeline,
        retrieval=retrieval,
        tools=tools,
        conversations=conversations,
    )
    return AiRuntime(
        settings=settings,
        http=http,
        vector_store=vector_store,
        pipeline=pipeline,
        answers=answers,
        database=database,
        tools=tools,
        retrieval=retrieval,
        ingestor=ingestor,
    )


async def _open_database(settings: Settings) -> Database | None:
    database = Database(settings)
    last_error: str | None = None
    for attempt in range(1, DATABASE_OPEN_ATTEMPTS + 1):
        try:
            await database.connect()
            reachable, latency_ms, error = await database.ping()
        except Exception as exc:
            reachable, latency_ms, error = False, 0.0, f"{type(exc).__name__}: {exc}"
        if reachable:
            logger.info(
                "postgres ready",
                extra={
                    "context": {
                        "latency_ms": round(latency_ms, 1),
                        "read_only": True,
                        "pool_size": settings.postgres_pool_max_size,
                        "attempts": attempt,
                    }
                },
            )
            return database
        last_error = error
        await database.close()
        if attempt < DATABASE_OPEN_ATTEMPTS:
            logger.warning(
                "postgres is not answering, retrying",
                extra={
                    "context": {
                        "attempt": attempt,
                        "of": DATABASE_OPEN_ATTEMPTS,
                        "error": last_error,
                    }
                },
            )
            await asyncio.sleep(DATABASE_OPEN_RETRY_SECONDS)
    logger.error(
        "postgres could not be opened, student tools are disabled",
        extra={"context": {"error": last_error, "attempts": DATABASE_OPEN_ATTEMPTS}},
    )
    return None


async def _probe_vector_store(settings: Settings, embedder: Embedder, store: VectorStore) -> None:
    try:
        await store.ensure_collection(embedder.dimensions)
    except Exception as exc:
        logger.warning(
            "qdrant is not reachable, retrieval will answer without context",
            extra={
                "context": {"url": settings.qdrant_url, "error": f"{type(exc).__name__}: {exc}"}
            },
        )
        return
    logger.info(
        "qdrant ready",
        extra={
            "context": {
                "url": settings.qdrant_url,
                "collection": store.collection,
                "dimensions": embedder.dimensions,
            }
        },
    )
