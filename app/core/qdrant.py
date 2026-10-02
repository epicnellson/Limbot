from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from app.config import Settings

logger = logging.getLogger(__name__)


class SearchHit:
    """One retrieved chunk, with the score Qdrant assigned to it."""

    __slots__ = ("id", "payload", "score")

    def __init__(self, id: Any, score: float, payload: dict[str, Any]) -> None:
        self.id = id
        self.score = score
        self.payload = payload

    @property
    def text(self) -> str:
        value = self.payload.get("text")
        return value if isinstance(value, str) else ""


class VectorStore:
    """Qdrant handle: readiness probe, collection setup, ingest, and thresholded search.

    The client is created once per process and closed on shutdown. Every method returns
    instead of raising for the failure cases the caller is expected to survive, and raises
    for the ones that mean the data is missing.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncQdrantClient | None = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("vector store is not connected")
        return self._client

    @property
    def collection(self) -> str:
        return self._settings.qdrant_collection

    async def connect(self) -> None:
        if self._client is not None:
            return
        self._client = AsyncQdrantClient(
            url=self._settings.qdrant_url,
            api_key=self._settings.qdrant_api_key_value,
            timeout=self._settings.qdrant_timeout_seconds,
        )
        logger.info(
            "qdrant client created",
            extra={"context": {"url": self._settings.qdrant_url, "collection": self.collection}},
        )

    async def close(self) -> None:
        if self._client is None:
            return
        await self._client.close()
        self._client = None

    async def ping(self) -> tuple[bool, float, str | None]:
        """Return (reachable, latency_ms, error) without letting probe errors escape."""
        if self._client is None:
            await self.connect()
        assert self._client is not None
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            async with asyncio.timeout(self._settings.qdrant_timeout_seconds):
                await self._client.get_collections()
        except Exception as exc:
            latency_ms = (loop.time() - started) * 1000
            return False, latency_ms, f"{type(exc).__name__}: {exc}"
        latency_ms = (loop.time() - started) * 1000
        return True, latency_ms, None

    async def ensure_collection(self, dimensions: int) -> bool:
        """Create the collection if it is missing. Returns True when it was created."""
        if self._client is None:
            await self.connect()
        assert self._client is not None
        name = self.collection
        if await self._client.collection_exists(name):
            logger.debug("qdrant collection already present", extra={"context": {"name": name}})
            return False
        await self._client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dimensions, distance=models.Distance.COSINE),
        )
        logger.info(
            "qdrant collection created",
            extra={"context": {"name": name, "dimensions": dimensions}},
        )
        return True

    async def upsert_chunks(self, chunks: list[dict[str, Any]], vectors: list[list[float]]) -> int:
        """Write chunk payloads and vectors. The caller supplies matching lengths."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return 0
        if self._client is None:
            await self.connect()
        assert self._client is not None
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk_id = str(uuid.uuid4())
            points.append(
                models.PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload={**chunk, "chunk_id": chunk_id},
                )
            )
        await self._client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    async def search(
        self, vector: list[float], *, limit: int, score_threshold: float | None = None
    ) -> list[SearchHit]:
        """Nearest neighbours above the threshold, best first."""
        if self._client is None:
            await self.connect()
        assert self._client is not None
        response = await self._client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=max(1, limit),
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [
            SearchHit(
                id=point.id,
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in response.points
        ]

    async def count(self) -> int:
        """Points currently in the collection, for the readiness probe."""
        if self._client is None:
            await self.connect()
        assert self._client is not None
        result = await self._client.count(collection_name=self.collection, exact=True)
        return int(result.count)
