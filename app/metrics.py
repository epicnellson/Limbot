from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    GCCollector,
    Histogram,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
)

CONTENT_TYPE = CONTENT_TYPE_LATEST

REGISTRY = CollectorRegistry(auto_describe=True)
ProcessCollector(registry=REGISTRY)
PlatformCollector(registry=REGISTRY)
GCCollector(registry=REGISTRY)

BUILD_INFO = Gauge(
    "limbot_build_info",
    "Static build and environment information, always 1.",
    ("version", "environment"),
    registry=REGISTRY,
)

HTTP_REQUESTS = Counter(
    "limbot_http_requests_total",
    "HTTP requests handled by the application.",
    ("method", "route", "status"),
    registry=REGISTRY,
)

HTTP_REQUEST_DURATION = Histogram(
    "limbot_http_request_duration_seconds",
    "Wall time spent handling an HTTP request.",
    ("method", "route"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
    registry=REGISTRY,
)

WEBHOOK_EVENTS = Counter(
    "limbot_webhook_events_total",
    "Webhook requests received, by kind and outcome.",
    ("kind", "result"),
    registry=REGISTRY,
)

WEBHOOK_DUPLICATES = Counter(
    "limbot_webhook_duplicates_total",
    "Inbound messages dropped because their id was already processed.",
    registry=REGISTRY,
)

BACKGROUND_TASKS = Counter(
    "limbot_background_tasks_total",
    "Background task lifecycle events.",
    ("result",),
    registry=REGISTRY,
)

BACKGROUND_TASKS_INFLIGHT = Gauge(
    "limbot_background_tasks_inflight",
    "Background tasks currently executing.",
    registry=REGISTRY,
)

BACKGROUND_TASK_DURATION = Histogram(
    "limbot_background_task_duration_seconds",
    "Wall time spent inside a background task.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
    registry=REGISTRY,
)

OUTBOUND_REQUESTS = Counter(
    "limbot_outbound_requests_total",
    "Outbound HTTP attempts, including retries.",
    ("client", "operation", "result"),
    registry=REGISTRY,
)

OUTBOUND_REQUEST_DURATION = Histogram(
    "limbot_outbound_request_duration_seconds",
    "Wall time of a single outbound HTTP attempt.",
    ("client", "operation"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
    registry=REGISTRY,
)

OUTBOUND_RETRIES = Counter(
    "limbot_outbound_retries_total",
    "Outbound HTTP retries, by trigger.",
    ("client", "operation", "reason"),
    registry=REGISTRY,
)

WHATSAPP_REQUESTS = Counter(
    "limbot_whatsapp_api_requests_total",
    "WhatsApp Cloud API calls, by operation and HTTP status.",
    ("operation", "status"),
    registry=REGISTRY,
)

WHATSAPP_MESSAGES = Counter(
    "limbot_whatsapp_messages_total",
    "WhatsApp messages, by direction and outcome.",
    ("direction", "result"),
    registry=REGISTRY,
)

MESSAGE_KINDS = Counter(
    "limbot_inbound_message_kinds_total",
    "Inbound message payloads, by message type.",
    ("type",),
    registry=REGISTRY,
)

MESSAGE_STATUSES = Counter(
    "limbot_message_status_events_total",
    "Delivery status updates reported by the webhook.",
    ("status",),
    registry=REGISTRY,
)

CIRCUIT_STATE = Gauge(
    "limbot_circuit_state",
    "Circuit breaker state, 1 for the active state of each provider.",
    ("provider", "state"),
    registry=REGISTRY,
)

CIRCUIT_TRANSITIONS = Counter(
    "limbot_circuit_transitions_total",
    "Circuit breaker state transitions.",
    ("provider", "to_state"),
    registry=REGISTRY,
)

CIRCUIT_FAILURES = Counter(
    "limbot_circuit_failures_total",
    "Failures recorded against a provider circuit.",
    ("provider",),
    registry=REGISTRY,
)

CIRCUIT_REJECTIONS = Counter(
    "limbot_circuit_rejections_total",
    "Calls skipped because the provider circuit was not accepting requests.",
    ("provider",),
    registry=REGISTRY,
)

LLM_CALLS = Counter(
    "limbot_llm_calls_total",
    "LLM completions by provider and outcome.",
    ("provider", "result"),
    registry=REGISTRY,
)

LLM_LATENCY = Histogram(
    "limbot_llm_call_duration_seconds",
    "Wall time of a single LLM completion attempt.",
    ("provider",),
    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 30, 60),
    registry=REGISTRY,
)

PROVIDER_FALLBACKS = Counter(
    "limbot_provider_fallbacks_total",
    "Completions that moved from one provider tier to the next, by reason.",
    ("from_provider", "to_provider", "reason"),
    registry=REGISTRY,
)

# PROVIDER_FALLBACKS deliberately stops counting one tier before the end, because a failure on
# the last tier ends the turn instead of rescuing it. That leaves the worst outcome invisible:
# nothing distinguishes "one backup saved this request" from "no tier could answer at all".
# This counter is that missing signal, and it is the one an on-call alert should page on.
ALL_PROVIDERS_EXHAUSTED = Counter(
    "limbot_all_providers_exhausted_total",
    "Completions that ended with no usable provider tier, by cause.",
    # "exhausted" means every configured tier failed or was skipped, which is an outage.
    # "unconfigured" means no tier had credentials at all, which is a deployment mistake.
    ("cause",),
    registry=REGISTRY,
)

TOOL_CALLS = Counter(
    "limbot_tool_calls_total",
    "Tool router invocations by tool and outcome.",
    ("tool", "result"),
    registry=REGISTRY,
)

TOOL_LATENCY = Histogram(
    "limbot_tool_call_duration_seconds",
    "Wall time of a tool invocation.",
    ("tool",),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    registry=REGISTRY,
)

RAG_QUERIES = Counter(
    "limbot_rag_queries_total",
    "Retrieval queries by outcome.",
    ("result",),
    registry=REGISTRY,
)

RAG_CHUNKS = Counter(
    "limbot_rag_chunks_total",
    "Chunks considered and returned by retrieval.",
    ("disposition",),
    registry=REGISTRY,
)

RAG_LATENCY = Histogram(
    "limbot_rag_query_duration_seconds",
    "Wall time of a retrieval round trip.",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
    registry=REGISTRY,
)

ANSWERS = Counter(
    "limbot_answers_total",
    "Replies produced by the AI pipeline, by provider and whether tools were needed.",
    ("provider", "tools"),
    registry=REGISTRY,
)

CONVERSATIONS = Gauge(
    "limbot_conversations_tracked",
    "Senders with a live conversation window.",
    registry=REGISTRY,
)

EVALS = Counter(
    "limbot_evals_total",
    "Synthetic evaluation cases, by case, outcome and answering provider.",
    ("case", "result", "provider"),
    registry=REGISTRY,
)


def render() -> bytes:
    """Serialize the registry using the latest Prometheus text exposition format."""
    return generate_latest(REGISTRY)
