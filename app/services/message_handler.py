from __future__ import annotations

import logging

import httpx

from app import metrics
from app.config import Settings
from app.core.whatsapp import WhatsAppAPIError, WhatsAppCloudClient, WhatsAppNotConfigured
from app.llm.base import ProviderError
from app.models.whatsapp import (
    InboundMessage,
    WebhookChange,
    WebhookMessage,
    WhatsAppWebhook,
)
from app.services.answer import Answer, AnswerService
from app.services.dedupe import MessageDeduplicator
from app.tools.context import ToolContext

logger = logging.getLogger(__name__)

SUPPORTED_FIELDS = frozenset({"messages"})

NOTHING_TO_SAY = (
    "I did not understand that. Try asking about your courses, timetable, or deadlines."
)


async def handle_webhook(
    payload: WhatsAppWebhook,
    *,
    settings: Settings,
    whatsapp: WhatsAppCloudClient,
    dedupe: MessageDeduplicator,
    answers: AnswerService,
    http: httpx.AsyncClient,
) -> None:
    """Process one webhook delivery. Runs detached from the request that produced it."""
    for change in payload.iter_changes():
        if change.field is not None and change.field not in SUPPORTED_FIELDS:
            logger.info(
                "ignoring unsupported change field", extra={"context": {"field": change.field}}
            )
            continue
        _handle_statuses(change)
        _handle_change_errors(change)
        for message in change.value.messages:
            await _handle_message(
                message,
                change=change,
                settings=settings,
                whatsapp=whatsapp,
                dedupe=dedupe,
                answers=answers,
                http=http,
            )


def _handle_statuses(change: WebhookChange) -> None:
    for status in change.value.statuses:
        metrics.MESSAGE_STATUSES.labels(status=status.status).inc()
        logger.info(
            "delivery status",
            extra={
                "context": {
                    "message_id": status.id,
                    "status": status.status,
                    "recipient_id": status.recipient_id,
                }
            },
        )
        for error in status.errors:
            logger.warning(
                "delivery error", extra={"context": {"message_id": status.id, **error.model_dump()}}
            )


def _handle_change_errors(change: WebhookChange) -> None:
    for error in change.value.errors:
        logger.error("webhook reported an error", extra={"context": error.model_dump()})


async def _handle_message(
    message: WebhookMessage,
    *,
    change: WebhookChange,
    settings: Settings,
    whatsapp: WhatsAppCloudClient,
    dedupe: MessageDeduplicator,
    answers: AnswerService,
    http: httpx.AsyncClient,
) -> None:
    metadata = change.value.metadata
    inbound = InboundMessage(
        message_id=message.id,
        wa_id=message.from_,
        kind=message.type,
        text=message.body_text,
        profile_name=change.value.contact_name(message.from_),
        phone_number_id=metadata.phone_number_id if metadata else None,
        received_at=message.received_at,
        raw=message,
    )

    if dedupe.seen(inbound.message_id):
        metrics.WEBHOOK_DUPLICATES.inc()
        return

    for error in message.errors:
        logger.warning("inbound message error", extra={"context": error.model_dump()})

    metrics.MESSAGE_KINDS.labels(type=inbound.kind).inc()
    metrics.WHATSAPP_MESSAGES.labels(direction="inbound", result="received").inc()
    logger.info(
        "inbound message",
        extra={
            "context": {
                "message_id": inbound.message_id,
                "wa_id": inbound.wa_id,
                "kind": inbound.kind,
                "has_text": inbound.text is not None,
                "received_at": inbound.received_at.isoformat(),
            }
        },
    )

    if not whatsapp.enabled:
        # Without an access token and a phone number id nothing can be sent back, so there is no
        # point spending a completion on a reply that would be dropped.
        logger.warning(
            "dropping message: whatsapp outbound is not configured",
            extra={"context": {"wa_id": inbound.wa_id, "message_id": inbound.message_id}},
        )
        return

    await _mark_read(inbound, whatsapp=whatsapp)

    if inbound.kind != "text" or not inbound.text:
        return

    if not answers.enabled:
        logger.warning(
            "dropping message: no AI provider is configured",
            extra={"context": {"wa_id": inbound.wa_id}},
        )
        return

    if settings.whatsapp_echo_mode:
        await _echo(inbound, whatsapp=whatsapp)
        return

    context = ToolContext(
        wa_id=inbound.wa_id,
        display_name=inbound.profile_name,
        is_new_sender=inbound.profile_name is None,
    )
    answer = await _answer(inbound, context, answers=answers, http=http)
    if not answer.text.strip():
        return
    await _reply(inbound, answer, whatsapp=whatsapp)


async def _answer(
    inbound: InboundMessage,
    context: ToolContext,
    *,
    answers: AnswerService,
    http: httpx.AsyncClient,
) -> Answer:
    """Run the AI pipeline, turning any failure into a safe fallback rather than a traceback."""
    text = inbound.text or ""
    try:
        answer = await answers.answer(text, context, http=http)
    except ProviderError as exc:
        # A non-transient provider error reaches here: the request itself was rejected, so
        # retrying with a different tier would be rejected too.
        logger.error(
            "provider rejected the request",
            extra={
                "context": {
                    "wa_id": inbound.wa_id,
                    "provider": exc.provider,
                    "status_code": exc.status_code,
                    "error": str(exc),
                }
            },
        )
        return Answer(text=NOTHING_TO_SAY, error="provider rejected the request")
    except Exception as exc:
        logger.exception(
            "answer pipeline failed",
            extra={"context": {"wa_id": inbound.wa_id, "error": type(exc).__name__}},
        )
        return Answer(text=NOTHING_TO_SAY, error="answer pipeline failed")

    metrics.ANSWERS.labels(
        provider=answer.provider,
        tools="used" if answer.used_tools else "none",
    ).inc()
    logger.info(
        "answer ready",
        extra={
            "context": {
                "wa_id": inbound.wa_id,
                "provider": answer.provider,
                "model": answer.model,
                "tools": list(answer.used_tools),
                "tool_rounds": answer.tool_rounds,
                "retrieved": answer.retrieved,
                "sources": list(answer.sources),
                "degraded": answer.degraded,
                "error": answer.error,
            }
        },
    )
    if not answer.ok:
        metrics.ANSWERS.labels(provider=answer.provider or "none", tools="failed").inc()
    return answer


async def _mark_read(inbound: InboundMessage, *, whatsapp: WhatsAppCloudClient) -> None:
    try:
        await whatsapp.mark_message_read(inbound.message_id)
    except WhatsAppAPIError:
        logger.warning(
            "failed to mark message read", extra={"context": {"message_id": inbound.message_id}}
        )


async def _echo(inbound: InboundMessage, *, whatsapp: WhatsAppCloudClient) -> None:
    try:
        result = await whatsapp.send_text(inbound.wa_id, inbound.text or "")
    except (WhatsAppAPIError, WhatsAppNotConfigured):
        logger.exception("echo failed", extra={"context": {"message_id": inbound.message_id}})
        return
    logger.info(
        "echoed message",
        extra={"context": {"message_id": inbound.message_id, "outbound_id": result.message_id}},
    )


async def _reply(inbound: InboundMessage, answer: Answer, *, whatsapp: WhatsAppCloudClient) -> None:
    try:
        result = await whatsapp.send_text(inbound.wa_id, answer.text)
    except (WhatsAppAPIError, WhatsAppNotConfigured):
        logger.exception(
            "failed to send answer",
            extra={"context": {"message_id": inbound.message_id, "wa_id": inbound.wa_id}},
        )
        return
    logger.info(
        "answer sent",
        extra={
            "context": {
                "message_id": inbound.message_id,
                "outbound_id": result.message_id,
                "provider": answer.provider,
                "tools": list(answer.used_tools),
            }
        },
    )
