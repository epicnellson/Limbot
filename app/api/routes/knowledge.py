"""Loading course material into Qdrant.

Kept under ``/debug``, so it is only mounted when ``DEBUG_ENDPOINTS`` is true and is refused
outright in production. An open endpoint that writes to the knowledge base would let anyone make
the bot cite whatever they liked, which is a far worse failure than the bot admitting it does not
know something.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.core.embeddings import EmbeddingError
from app.rag.ingest import MAX_DOCUMENT_CHARS, IngestRefused, KnowledgeIngestor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debug", tags=["debug"], include_in_schema=False)


class IngestRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_DOCUMENT_CHARS)
    title: str = Field(default="", max_length=200)
    source: str = Field(default="", max_length=200)


@router.post("/knowledge", summary="Chunk, embed and store a document")
async def ingest_document(request: Request, body: IngestRequest) -> dict[str, object]:
    """Load one document so the bot can answer from it.

    The response reports the derived document id, which is a hash of the title and text, so
    posting the same material twice replaces the previous chunks rather than duplicating them.
    """
    ingestor: KnowledgeIngestor | None = request.app.state.runtime.ingestor
    if ingestor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="retrieval is not enabled on this instance",
        )
    try:
        result = await ingestor.ingest(body.text, title=body.title, source=body.source)
    except IngestRefused as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except EmbeddingError as exc:
        logger.error("embedding failed during ingest", extra={"context": {"error": str(exc)}})
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="the embedding model failed"
        ) from exc
    return {"ok": True, **result.as_dict()}
