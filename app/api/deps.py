from __future__ import annotations

from typing import Annotated, cast

import httpx
from fastapi import Depends, Request

from app.config import Settings
from app.core.qdrant import VectorStore
from app.core.tasks import TaskManager
from app.core.whatsapp import WhatsAppCloudClient
from app.llm.pipeline import LLMPipeline
from app.services.answer import AnswerService
from app.services.dedupe import MessageDeduplicator

# Every dependency below reads off app.state, whose attribute type is Any, so each function would
# otherwise return Any from a declared return type. cast states the intent that main.py already
# guarantees by populating state during startup. It is preferred over a type: ignore because the
# annotation stays live: if these return types change, mypy reports it here rather than staying
# silent behind a suppression.


def get_settings_dep(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_http_client(request: Request) -> httpx.AsyncClient:
    return cast(httpx.AsyncClient, request.app.state.http_client)


def get_task_manager(request: Request) -> TaskManager:
    return cast(TaskManager, request.app.state.task_manager)


def get_whatsapp_client(request: Request) -> WhatsAppCloudClient:
    return cast(WhatsAppCloudClient, request.app.state.whatsapp)


def get_dedupe(request: Request) -> MessageDeduplicator:
    return cast(MessageDeduplicator, request.app.state.dedupe)


def get_vector_store(request: Request) -> VectorStore:
    return cast(VectorStore, request.app.state.vector_store)


def get_llm_pipeline(request: Request) -> LLMPipeline:
    return cast(LLMPipeline, request.app.state.llm_pipeline)


def get_answer_service(request: Request) -> AnswerService:
    return cast(AnswerService, request.app.state.answers)


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
HttpClientDep = Annotated[httpx.AsyncClient, Depends(get_http_client)]
TaskManagerDep = Annotated[TaskManager, Depends(get_task_manager)]
WhatsAppDep = Annotated[WhatsAppCloudClient, Depends(get_whatsapp_client)]
DedupeDep = Annotated[MessageDeduplicator, Depends(get_dedupe)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
LLMPipelineDep = Annotated[LLMPipeline, Depends(get_llm_pipeline)]
AnswerServiceDep = Annotated[AnswerService, Depends(get_answer_service)]
