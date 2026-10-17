"""The authenticated content path.

Unlike ``/debug/knowledge``, this router is mounted in every environment, because posting
knowledge is a routine operator action rather than a debugging aid. It is gated by a bearer
token instead: an open writer would let anyone make the bot cite whatever they liked.

Access control itself is deliberately not reachable here. Roles live in ``contacts`` and are
written with psql, so a leaked admin token can seed course material but cannot hand out student
records.
"""

from __future__ import annotations

import logging
import secrets

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.access import AUDIENCES
from app.core.embeddings import EmbeddingError
from app.rag.ingest import MAX_DOCUMENT_CHARS, IngestRefused, KnowledgeIngestor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

_AUDIENCE_HINT = ", ".join(AUDIENCES)


class KnowledgeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_DOCUMENT_CHARS)
    title: str = Field(default="", max_length=200)
    source: str = Field(default="", max_length=200)
    audience: str = Field(default="student", max_length=32)


def _authorise(request: Request, token: str | None) -> None:
    """Reject anything that does not carry the configured admin token.

    A missing token is a 503 rather than a 200: an unconfigured admin surface must fail closed.
    The comparison is constant time so the response cannot be used to guess the token.
    """
    allowed = request.app.state.settings.admin_api_token_value
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the admin API is not configured",
        )
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not secrets.compare_digest(token.encode(), allowed.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/knowledge", summary="Publish a document to the knowledge base")
async def publish_knowledge(
    request: Request,
    body: KnowledgeRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    """Chunk, embed and store one document for a given audience.

    The audience decides who may retrieve it: ``public`` reaches everyone, ``student`` reaches
    enrolled students, testers and staff, and ``staff`` reaches staff and admins. Untagged
    material is never returned to a public sender.
    """
    token = _bearer(authorization)
    _authorise(request, token)

    ingestor: KnowledgeIngestor | None = request.app.state.runtime.ingestor
    if ingestor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="retrieval is not enabled on this instance",
        )
    try:
        result = await ingestor.ingest(
            body.text, title=body.title, source=body.source, audience=body.audience
        )
    except IngestRefused as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except EmbeddingError as exc:
        logger.error("embedding failed during admin ingest", extra={"context": {"error": str(exc)}})
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="the embedding model failed"
        ) from exc
    return {"ok": True, **result.as_dict()}


def _bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token
