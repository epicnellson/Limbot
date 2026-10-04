from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from app import metrics
from app.config import Settings

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class OutboundTransportError(RuntimeError):
    """Raised when an outbound request could not be completed within the retry budget."""

    def __init__(self, message: str, *, operation: str, attempts: int, cause: Exception) -> None:
        super().__init__(message)
        self.operation = operation
        self.attempts = attempts
        self.cause = cause


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retry budget for a single outbound call.

    ``retry_transport_errors`` is only enabled for idempotent requests. A connection reset or
    read timeout on a non-idempotent POST leaves the remote side in an unknown state, so
    replaying it can duplicate a WhatsApp message. Throttling and 5xx responses are different:
    those are explicit rejections, so replaying them is safe for any verb.
    """

    max_attempts: int
    base_delay: float
    max_delay: float
    retry_transport_errors: bool

    @classmethod
    def from_settings(cls, settings: Settings, *, idempotent: bool) -> RetryPolicy:
        return cls(
            max_attempts=max(1, settings.outbound_max_attempts),
            base_delay=settings.outbound_retry_base_delay_seconds,
            max_delay=settings.outbound_retry_max_delay_seconds,
            retry_transport_errors=idempotent,
        )


NO_RETRIES = RetryPolicy(
    max_attempts=1, base_delay=0.0, max_delay=0.0, retry_transport_errors=False
)


def build_http_client(settings: Settings, *, client_name: str = "default") -> httpx.AsyncClient:
    """Create the process-wide pooled client. One instance per app, closed on shutdown."""
    timeout = httpx.Timeout(
        settings.http_timeout_seconds,
        connect=settings.http_connect_timeout_seconds,
        read=settings.http_timeout_seconds,
        write=settings.http_timeout_seconds,
        pool=settings.http_connect_timeout_seconds,
    )
    limits = httpx.Limits(
        max_connections=settings.http_max_connections,
        max_keepalive_connections=settings.http_max_keepalive_connections,
        keepalive_expiry=settings.http_keepalive_expiry_seconds,
    )
    transport = httpx.AsyncHTTPTransport(retries=settings.http_connect_retries)
    client = httpx.AsyncClient(
        transport=transport,
        timeout=timeout,
        limits=limits,
        follow_redirects=False,
        trust_env=True,
        headers={
            "user-agent": f"{settings.app_name}/{settings.app_version}",
            "accept": "application/json",
        },
    )
    logger.debug(
        "http client created",
        extra={
            "context": {
                "client": client_name,
                "timeout_seconds": settings.http_timeout_seconds,
                "max_connections": settings.http_max_connections,
            }
        },
    )
    return client


def _backoff_delay(policy: RetryPolicy, attempt: int) -> float:
    # A float base keeps the power in float arithmetic: typeshed types `int ** int` as Any, and that
    # Any would spread through the multiplication and the min() into this return value.
    ceiling = min(policy.max_delay, policy.base_delay * 2.0 ** (attempt - 1))
    return ceiling * random.uniform(0.5, 1.0)


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        pass
    try:
        when = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def _classify_status(status_code: int) -> str:
    if status_code == 429:
        return "throttled"
    if status_code in RETRYABLE_STATUS:
        return "server_error"
    return "client_error"


async def send(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    client_name: str = "default",
    operation: str = "request",
    retry_policy: RetryPolicy | None = None,
    **kwargs: Any,
) -> httpx.Response:
    """Perform a pooled outbound request, retrying throttles and 5xx responses.

    Non-2xx responses are returned to the caller rather than raised, because the WhatsApp
    error envelope carries fields the caller needs to log and react to. This helper reads the
    whole body, so it is for buffered calls only; use ``client.stream`` for media downloads.
    """
    policy = retry_policy or NO_RETRIES
    attempt = 0
    while True:
        attempt += 1
        started = time.perf_counter()
        try:
            response = await client.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            metrics.OUTBOUND_REQUEST_DURATION.labels(
                client=client_name, operation=operation
            ).observe(time.perf_counter() - started)
            if attempt < policy.max_attempts and policy.retry_transport_errors:
                metrics.OUTBOUND_RETRIES.labels(
                    client=client_name, operation=operation, reason="transport_error"
                ).inc()
                logger.warning(
                    "outbound transport error, retrying",
                    extra={
                        "context": {
                            "client": client_name,
                            "operation": operation,
                            "attempt": attempt,
                            "error": type(exc).__name__,
                        }
                    },
                )
                await asyncio.sleep(_backoff_delay(policy, attempt))
                continue
            metrics.OUTBOUND_REQUESTS.labels(
                client=client_name, operation=operation, result="transport_error"
            ).inc()
            raise OutboundTransportError(
                f"{operation} failed after {attempt} attempt(s): {exc}",
                operation=operation,
                attempts=attempt,
                cause=exc,
            ) from exc

        elapsed = time.perf_counter() - started
        metrics.OUTBOUND_REQUEST_DURATION.labels(client=client_name, operation=operation).observe(
            elapsed
        )
        metrics.OUTBOUND_REQUESTS.labels(
            client=client_name,
            operation=operation,
            result="success" if response.is_success else _classify_status(response.status_code),
        ).inc()

        if response.status_code in RETRYABLE_STATUS and attempt < policy.max_attempts:
            reason = "throttled" if response.status_code == 429 else "server_error"
            metrics.OUTBOUND_RETRIES.labels(
                client=client_name, operation=operation, reason=reason
            ).inc()
            retry_after = _retry_after_seconds(response)
            delay = retry_after if retry_after is not None else _backoff_delay(policy, attempt)
            logger.warning(
                "outbound request rejected, retrying",
                extra={
                    "context": {
                        "client": client_name,
                        "operation": operation,
                        "attempt": attempt,
                        "status": response.status_code,
                        "delay_seconds": round(delay, 3),
                    }
                },
            )
            await response.aclose()
            await asyncio.sleep(delay)
            continue
        return response
