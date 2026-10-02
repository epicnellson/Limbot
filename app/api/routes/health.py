from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.api.deps import SettingsDep, TaskManagerDep
from app.core.qdrant import VectorStore
from app.llm.pipeline import LLMPipeline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ops"])


@router.get("/healthz", summary="Liveness probe")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness probe")
async def readyz(
    request: Request, settings: SettingsDep, task_manager: TaskManagerDep
) -> JSONResponse:
    """Report whether this instance can serve traffic.

    The hard requirement is that the instance can do the job it was configured for. If the AI
    pipeline is on, at least one provider tier must be configured with a closed circuit,
    otherwise every student message would be dropped after a 200. If the pipeline is off, the
    only remaining job is echoing, so readiness follows ``WHATSAPP_ECHO_MODE``. Draining always
    fails the probe, which is what keeps a container from taking new webhooks while it finishes
    in-flight work.

    Qdrant and Postgres are reported but only fatal when their own flag says so. A bot without
    course material still answers timetable questions, and one without Postgres still answers
    general questions, so neither should take the instance out of rotation on its own.
    """
    runtime = request.app.state.runtime
    pipeline: LLMPipeline = runtime.pipeline
    vector_store: VectorStore = runtime.vector_store

    checks: dict[str, Any] = {
        "draining": not task_manager.accepting,
        "signature_verification": settings.whatsapp_signature_required,
        "whatsapp_outbound_configured": settings.whatsapp_outbound_enabled,
    }
    ready = True
    if not task_manager.accepting:
        ready = False

    circuits = pipeline.circuit_state()
    usable = [
        provider.name
        for provider in pipeline.providers
        if provider.configured and circuits.get(provider.name) == "closed"
    ]
    checks["ai"] = {
        "enabled": settings.ai_enabled,
        "configured_tiers": list(settings.configured_providers),
        "circuits": circuits,
        "usable_tiers": usable,
    }
    if settings.ai_enabled and not usable:
        ready = False
    elif not settings.ai_enabled and not settings.whatsapp_echo_mode:
        # Neither answering nor echoing: accepting webhooks would drop every message.
        ready = False

    reachable, latency_ms, error = await vector_store.ping()
    checks["qdrant"] = {
        "ok": reachable,
        "latency_ms": round(latency_ms, 1),
        "error": error,
        "required": settings.qdrant_required,
    }
    if not reachable and settings.qdrant_required and settings.vector_store_enabled:
        ready = False

    if runtime.database is not None:
        db_reachable, db_latency_ms, db_error = await runtime.database.ping()
        checks["postgres"] = {
            "ok": db_reachable,
            "latency_ms": round(db_latency_ms, 1),
            "error": db_error,
            "read_only": True,
        }
        if not db_reachable and settings.tools_enabled:
            ready = False
    else:
        checks["postgres"] = {"enabled": settings.tools_enabled, "ok": not settings.tools_enabled}

    body = {
        "status": "ready" if ready else "not_ready",
        "environment": settings.environment,
        "inflight_tasks": task_manager.inflight,
        "checks": checks,
    }
    return JSONResponse(
        body, status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
    )


@router.get("/", include_in_schema=False)
async def index(settings: SettingsDep) -> dict[str, Any]:
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "docs": "/docs" if settings.docs_enabled else None,
        "endpoints": {
            "webhook": "/webhooks/whatsapp",
            "health": "/healthz",
            "readiness": "/readyz",
            "metrics": "/metrics",
        },
        "build": {"version": settings.app_version, "environment": settings.environment},
    }
