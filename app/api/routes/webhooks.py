from __future__ import annotations

import logging
import secrets

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import PlainTextResponse
from pydantic import ValidationError

from app import metrics
from app.api.deps import (
    AnswerServiceDep,
    DedupeDep,
    HttpClientDep,
    SettingsDep,
    TaskManagerDep,
    WhatsAppDep,
)
from app.core.security import SIGNATURE_HEADER, verify_signature
from app.models.whatsapp import WhatsAppWebhook
from app.services import message_handler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.get("/whatsapp", response_class=PlainTextResponse, summary="Meta subscription handshake")
async def verify_subscription(request: Request, settings: SettingsDep) -> Response:
    params = request.query_params
    mode = params.get("hub.mode", "")
    token = params.get("hub.verify_token", "")
    challenge = params.get("hub.challenge", "")
    expected = settings.whatsapp_verify_token.get_secret_value()
    if mode == "subscribe" and token and secrets.compare_digest(token.encode(), expected.encode()):
        metrics.WEBHOOK_EVENTS.labels(kind="verification", result="accepted").inc()
        logger.info("webhook subscription verified")
        return PlainTextResponse(challenge)
    metrics.WEBHOOK_EVENTS.labels(kind="verification", result="rejected").inc()
    logger.warning("webhook verification rejected")
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="verification failed")


@router.post("/whatsapp", status_code=status.HTTP_200_OK, summary="Inbound Meta events")
async def receive_webhook(
    request: Request,
    settings: SettingsDep,
    task_manager: TaskManagerDep,
    whatsapp: WhatsAppDep,
    dedupe: DedupeDep,
    answers: AnswerServiceDep,
    http: HttpClientDep,
) -> Response:
    """Acknowledge the delivery immediately and process it on a background task.

    Meta retries any non-2xx response, so every rejection here is deliberate: a bad signature
    is a 401 and an unparseable body is a 422, both of which should keep being retried only if
    the sender is misconfigured. The 503 during shutdown is what makes a redelivery happen
    after the process is back up.
    """
    body = await request.body()

    if settings.whatsapp_signature_required:
        secret = (
            settings.whatsapp_app_secret.get_secret_value() if settings.whatsapp_app_secret else ""
        )
        if not verify_signature(secret, body, request.headers.get(SIGNATURE_HEADER)):
            metrics.WEBHOOK_EVENTS.labels(kind="message", result="invalid_signature").inc()
            client = request.client
            logger.warning(
                "rejected webhook with an invalid signature",
                extra={"context": {"client": client.host if client else None, "bytes": len(body)}},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid signature"
            )

    try:
        payload = WhatsAppWebhook.model_validate_json(body)
    except ValidationError as exc:
        metrics.WEBHOOK_EVENTS.labels(kind="message", result="invalid_payload").inc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="unsupported payload"
        ) from exc

    task = task_manager.submit(
        message_handler.handle_webhook(
            payload, settings=settings, whatsapp=whatsapp, dedupe=dedupe, answers=answers, http=http
        ),
        name="whatsapp-webhook",
    )
    if task is None:
        metrics.WEBHOOK_EVENTS.labels(kind="message", result="shutting_down").inc()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="shutting down")

    metrics.WEBHOOK_EVENTS.labels(kind="message", result="accepted").inc()
    return Response(status_code=status.HTTP_200_OK)
