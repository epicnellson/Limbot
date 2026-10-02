from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from app.config import Settings

logger = logging.getLogger(__name__)

EmbedFn = Callable[[Sequence[str]], list[list[float]]]


class EmbeddingError(RuntimeError):
    """The embedding model could not produce vectors."""


class EmbeddingBackend(Protocol):
    """What a text embedding model has to offer: a name, a width, and a batch embed call."""

    name: str

    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class FastEmbedBackend:
    """CPU ONNX embeddings through fastembed.

    The import is deferred so the app can start, serve webhooks and answer health checks on a
    machine where the model has not been downloaded yet. Embedding is a blocking, CPU bound
    call, so it is run in a worker thread rather than on the event loop.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self.name = settings.embedding_model

    @property
    def dimensions(self) -> int:
        return self._settings.embedding_dimensions

    def _load(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            logger.info(
                "loading embedding model",
                extra={
                    "context": {
                        "model": self.name,
                        "cache_dir": self._settings.embedding_cache_dir,
                    }
                },
            )
            self._model = TextEmbedding(
                model_name=self.name, cache_dir=self._settings.embedding_cache_dir
            )
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        vectors = [list(map(float, vector)) for vector in model.embed(list(texts))]
        return vectors


class Embedder:
    """Batching, dimension-checking wrapper around whichever backend is in use."""

    def __init__(
        self,
        settings: Settings,
        backend: EmbeddingBackend | None = None,
    ) -> None:
        self._settings = settings
        self._backend = backend or FastEmbedBackend(settings)
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._backend.name

    @property
    def dimensions(self) -> int:
        return self._backend.dimensions

    @property
    def available(self) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        return {"model": self.name, "dimensions": self.dimensions}

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch, off the event loop. Empty input short-circuits."""
        items = [text for text in texts if text.strip()]
        if not items:
            return []
        async with self._lock:
            try:
                vectors = await asyncio.to_thread(self._backend.embed, items)
            except Exception as exc:
                raise EmbeddingError(
                    f"embedding {len(items)} text(s) with {self.name} failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        self._check(vectors, len(items))
        return vectors

    async def embed_one(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        if not vectors:
            raise EmbeddingError("the embedding model returned no vector for the query")
        return vectors[0]

    def _check(self, vectors: list[list[float]], expected: int) -> None:
        if len(vectors) != expected:
            raise EmbeddingError(
                f"{self.name} returned {len(vectors)} vectors for {expected} input(s)"
            )
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise EmbeddingError(
                    f"{self.name} returned {len(vector)} dimensions, expected "
                    f"{self.dimensions}; EMBEDDING_DIMENSIONS must match the model"
                )
