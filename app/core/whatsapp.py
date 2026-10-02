from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from app import metrics
from app.config import Settings
from app.core.http import OutboundTransportError, RetryPolicy, send

logger = logging.getLogger(__name__)

THROTTLE_HEADERS = ("x-app-usage", "x-business-use-case-usage", "x-fb-trace-id")


class WhatsAppNotConfigured(RuntimeError):
    """Raised when an outbound call is attempted without a token and phone number id."""


class WhatsAppAPIError(RuntimeError):
    """A Graph API call returned an error envelope."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        code: int | None = None,
        subcode: int | None = None,
        fbtrace_id: str | None = None,
        details: dict[str, Any] = field(default_factory=dict),
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.subcode = subcode
        self.fbtrace_id = fbtrace_id
        self.details = details


@dataclass(frozen=True, slots=True)
class SendResult:
    message_id: str | None
    raw: dict[str, Any]


class WhatsAppCloudClient:
    """Thin client over the Cloud API surface this bot needs, reusing the shared pool."""

    def __init__(self, http: httpx.AsyncClient, settings: Settings) -> None:
        self._http = http
        self._settings = settings
        self._policy = RetryPolicy.from_settings(settings, idempotent=False)
        self._idempotent_policy = RetryPolicy.from_settings(settings, idempotent=True)

    @property
    def enabled(self) -> bool:
        return self._settings.whatsapp_outbound_enabled

    def _headers(self) -> dict[str, str]:
        token = self._settings.whatsapp_access_token
        if token is None:
            raise WhatsAppNotConfigured("WHATSAPP_ACCESS_TOKEN is not set")
        return {
            "authorization": f"Bearer {token.get_secret_value()}",
            "content-type": "application/json",
        }

    def _url(self, segment: str) -> str:
        phone_number_id = self._settings.whatsapp_phone_number_id
        if phone_number_id is None:
            raise WhatsAppNotConfigured("WHATSAPP_PHONE_NUMBER_ID is not set")
        base = self._settings.whatsapp_graph_base_url.rstrip("/")
        version = self._settings.whatsapp_graph_version.strip("/")
        return f"{base}/{version}/{phone_number_id}/{segment}"

    async def send_text(
        self, to: str, body: str, *, preview_url: bool = False, idempotent: bool = False
    ) -> SendResult:
        """Send a text message. Not idempotent, so transport errors are not replayed."""
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"preview_url": preview_url, "body": body},
        }
        data = await self._post("messages", payload, operation="send_text", idempotent=idempotent)
        messages = data.get("messages") or []
        message_id = messages[0].get("id") if messages and isinstance(messages[0], dict) else None
        metrics.WHATSAPP_MESSAGES.labels(direction="outbound", result="sent").inc()
        return SendResult(message_id=message_id, raw=data)

    async def mark_message_read(self, message_id: str) -> None:
        """Flag an inbound message as read. Idempotent, so it is safe to retry."""
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }
        await self._post("messages", payload, operation="mark_read", idempotent=True)

    async def _post(
        self, segment: str, payload: dict[str, Any], *, operation: str, idempotent: bool
    ) -> dict[str, Any]:
        url = self._url(segment)
        policy = self._idempotent_policy if idempotent else self._policy
        try:
            response = await send(
                self._http,
                "POST",
                url,
                client_name="whatsapp",
                operation=operation,
                retry_policy=policy,
                headers=self._headers(),
                json=payload,
            )
        except OutboundTransportError as exc:
            metrics.WHATSAPP_REQUESTS.labels(operation=operation, status="transport_error").inc()
            metrics.WHATSAPP_MESSAGES.labels(direction="outbound", result="transport_error").inc()
            raise WhatsAppAPIError(
                str(exc), status_code=0, details={"attempts": exc.attempts}
            ) from exc

        metrics.WHATSAPP_REQUESTS.labels(
            operation=operation, status=str(response.status_code)
        ).inc()
        data = self._decode(response)
        if response.is_success:
            return data

        error = data.get("error") if isinstance(data.get("error"), dict) else {}
        message = str(error.get("message") or response.text or "unknown Graph API error")
        self._log_failure(operation, response, error, message)
        metrics.WHATSAPP_MESSAGES.labels(direction="outbound", result="rejected").inc()
        raise WhatsAppAPIError(
            message,
            status_code=response.status_code,
            code=error.get("code"),
            subcode=error.get("error_subcode"),
            fbtrace_id=response.headers.get("x-fb-trace-id"),
            details=error,
        )

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _log_failure(
        self,
        operation: str,
        response: httpx.Response,
        error: dict[str, Any],
        message: str,
    ) -> None:
        context = {
            "operation": operation,
            "status": response.status_code,
            "error_code": error.get("code"),
            "error_subcode": error.get("error_subcode"),
            "fbtrace_id": response.headers.get("x-fb-trace-id"),
        }
        for header in THROTTLE_HEADERS:
            value = response.headers.get(header)
            if value:
                context[header.replace("-", "_")] = value
        logger.error("whatsapp api call failed", extra={"context": {**context, "error": message}})
