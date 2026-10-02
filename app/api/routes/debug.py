from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.api.deps import SettingsDep, WhatsAppDep
from app.core.whatsapp import WhatsAppAPIError, WhatsAppNotConfigured

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debug", tags=["debug"], include_in_schema=False)


class EchoRequest(BaseModel):
    to: str = Field(min_length=8, max_length=32, description="Recipient in international format")
    message: str = Field(min_length=1, max_length=4096)
    preview_url: bool = False


@router.post("/echo", summary="Send an outbound text through the real client")
async def debug_echo(
    body: EchoRequest, settings: SettingsDep, whatsapp: WhatsAppDep
) -> dict[str, object]:
    """Verify the outbound path end to end without waiting for a real inbound message.

    The router is only mounted when ``DEBUG_ENDPOINTS`` is enabled, which is refused outright
    in production.
    """
    if not settings.whatsapp_outbound_enabled or not whatsapp.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="whatsapp outbound is not configured",
        )
    try:
        result = await whatsapp.send_text(body.to, body.message, preview_url=body.preview_url)
    except (WhatsAppAPIError, WhatsAppNotConfigured) as exc:
        logger.warning("debug echo failed", extra={"context": {"error": str(exc)}})
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="graph api rejected the request"
        ) from exc
    return {
        "ok": True,
        "message_id": result.message_id,
        "response": result.raw,
    }
