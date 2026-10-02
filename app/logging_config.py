from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from app.config import Settings

LOGGER_NAME_ROOT = "limbot"
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with any ``extra={"context": {...}}`` fields merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if isinstance(context, dict):
            payload.update(context)
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key != "context":
                payload.setdefault(key, value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    default_time_format = "%Y-%m-%dT%H:%M:%S%z"
    default_fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self.default_fmt, datefmt=self.default_time_format)


def configure_logging(settings: Settings) -> None:
    """Install a single stdout handler, keeping container logs machine readable."""
    level = getattr(logging, settings.log_level, logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter() if settings.log_json else TextFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    logging.getLogger("uvicorn.error").setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(max(level, logging.WARNING))
    logging.getLogger("httpcore").setLevel(max(level, logging.WARNING))
    logging.getLogger("qdrant_client").setLevel(max(level, logging.WARNING))
    logging.getLogger("multipart").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
