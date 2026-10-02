from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import Depends, Request

from app.config import Settings
from app.core.qdrant import VectorStore
from app.core.tasks import TaskManager
from app.core.whatsapp import WhatsAppCloudClient
from app.llm.pipeline import LLMPipeline
from app.services.answer import AnswerService
from app.services.dedupe import MessageDeduplicator


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_http_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client


def get_task_manager(request: Request) -> TaskManager:
    return request.app.state.task_manager


def get_whatsapp_client(request: Request) -> WhatsAppCloudClient:
    return request.app.state.whatsapp


def get_dedupe(request: Request) -> MessageDeduplicator:
    return request.app.state.dedupe


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_llm_pipeline(request: Request) -> LLMPipeline:
    return request.app.state.llm_pipeline


def get_answer_service(request: Request) -> AnswerService:
    return request.app.state.answers


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
HttpClientDep = Annotated[httpx.AsyncClient, Depends(get_http_client)]
TaskManagerDep = Annotated[TaskManager, Depends(get_task_manager)]
WhatsAppDep = Annotated[WhatsAppCloudClient, Depends(get_whatsapp_client)]
DedupeDep = Annotated[MessageDeduplicator, Depends(get_dedupe)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
LLMPipelineDep = Annotated[LLMPipeline, Depends(get_llm_pipeline)]
AnswerServiceDep = Annotated[AnswerService, Depends(get_answer_service)]
