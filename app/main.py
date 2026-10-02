from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from app import __version__, metrics
from app.api.routes import debug, health, knowledge, observability, webhooks
from app.config import Settings, get_settings
from app.core.http import build_http_client
from app.core.tasks import TaskManager
from app.core.whatsapp import WhatsAppCloudClient
from app.logging_config import configure_logging
from app.middleware.metrics import MetricsMiddleware
from app.runtime import build_runtime
from app.services.dedupe import MessageDeduplicator

logger = logging.getLogger("limbot.main")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)
    docs_enabled = settings.docs_enabled

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        http_client = build_http_client(settings, client_name="whatsapp")
        task_manager = TaskManager(max_concurrency=settings.background_max_concurrency)
        whatsapp = WhatsAppCloudClient(http_client, settings)
        dedupe = MessageDeduplicator(
            ttl_seconds=settings.message_dedupe_ttl_seconds,
            max_entries=settings.message_dedupe_max_entries,
        )
        runtime = await build_runtime(settings, http_client)

        app.state.http_client = http_client
        app.state.task_manager = task_manager
        app.state.whatsapp = whatsapp
        app.state.dedupe = dedupe
        app.state.runtime = runtime
        app.state.llm_pipeline = runtime.pipeline
        app.state.answers = runtime.answers
        app.state.vector_store = runtime.vector_store
        app.state.database = runtime.database
        app.state.ingestor = runtime.ingestor

        metrics.BUILD_INFO.labels(
            version=settings.app_version, environment=settings.environment
        ).set(1)

        logger.info(
            "limbot started",
            extra={
                "context": {
                    "version": settings.app_version,
                    "environment": settings.environment,
                    "background_max_concurrency": settings.background_max_concurrency,
                    "whatsapp_outbound": settings.whatsapp_outbound_enabled,
                    "whatsapp_graph_version": settings.whatsapp_graph_version,
                    "qdrant_url": settings.qdrant_url,
                    "ai_enabled": runtime.answers.enabled,
                    "providers": list(settings.configured_providers),
                    "tools": runtime.tools.describe() if runtime.tools else [],
                }
            },
        )
        if not settings.whatsapp_outbound_enabled:
            logger.warning(
                "whatsapp outbound is disabled: set WHATSAPP_ACCESS_TOKEN and "
                "WHATSAPP_PHONE_NUMBER_ID to send messages"
            )
        if not runtime.answers.enabled:
            logger.error(
                "the bot cannot answer: AI_ENABLED is false or no provider tier is configured"
            )
        if runtime.tools is None:
            logger.warning("student tools are disabled: no usable POSTGRES_DSN")

        try:
            yield
        finally:
            await task_manager.drain(settings.background_drain_timeout_seconds)
            await runtime.close()
            await http_client.aclose()
            logger.info(
                "limbot stopped",
                extra={"context": {"duplicates_dropped": dedupe.duplicates}},
            )

    app = FastAPI(
        title="Limbot",
        summary="WhatsApp Cloud API bot platform",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )
    app.state.settings = settings

    app.add_middleware(MetricsMiddleware)
    if "*" not in settings.allowed_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    app.include_router(health.router)
    app.include_router(observability.router)
    app.include_router(webhooks.router)
    if settings.debug_endpoints:
        logger.warning("debug endpoints mounted under /debug")
        app.include_router(debug.router)
        app.include_router(knowledge.router)

    @app.exception_handler(Exception)
    async def on_unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "unhandled error",
            extra={
                "context": {
                    "path": request.url.path,
                    "method": request.method,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            },
        )
        return JSONResponse({"detail": "internal server error"}, status_code=500)

    return app


app = create_app()
