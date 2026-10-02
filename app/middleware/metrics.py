from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app import metrics

logger = logging.getLogger(__name__)

_SEGMENT_PATTERNS = (
    (
        re.compile(
            r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
        ),
        "{uuid}",
    ),
    (re.compile(r"^\d{6,}$"), "{id}"),
    (re.compile(r"^[0-9a-fA-F]{16,}$"), "{hash}"),
    (re.compile(r"^[a-zA-Z0-9_-]{32,}$"), "{token}"),
)

EXCLUDED_PATHS = frozenset({"/metrics"})


def _normalise_path(path: str) -> str:
    """Collapse variable path segments so unmatched routes cannot explode label cardinality."""
    if not path:
        return "/"
    segments: list[str] = []
    for segment in path.split("/"):
        for pattern, placeholder in _SEGMENT_PATTERNS:
            if pattern.match(segment):
                segment = placeholder
                break
        segments.append(segment)
    template = "/".join(segments)
    return template[:200]


def route_template(scope: Scope) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path:
        return path
    return _normalise_path(str(scope.get("path", "")))


class MetricsMiddleware:
    """Pure ASGI timing and access logging.

    Deliberately not ``BaseHTTPMiddleware``: that class buffers responses and interferes with
    the detached tasks the webhook endpoint fires off, which must outlive the request.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw_path = str(scope.get("path", ""))
        if raw_path in EXCLUDED_PATHS:
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", "GET"))
        started = time.perf_counter()
        state: dict[str, Any] = {"status": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - started
            status = int(state["status"])
            template = route_template(scope)
            metrics.HTTP_REQUESTS.labels(method=method, route=template, status=str(status)).inc()
            metrics.HTTP_REQUEST_DURATION.labels(method=method, route=template).observe(duration)
            client = scope.get("client")
            logger.info(
                "request",
                extra={
                    "context": {
                        "method": method,
                        "route": template,
                        "status": status,
                        "duration_ms": round(duration * 1000, 2),
                        "client": client[0] if client else None,
                    }
                },
            )


MetricsMiddlewareFactory = Callable[[ASGIApp], Awaitable[None]]
