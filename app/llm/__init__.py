"""LLM provider adapters and the tiered fallback pipeline."""

from app.llm.base import (
    LLMProvider,
    ProviderConfigError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
    ProviderTransportError,
    raise_for_status,
    wrap_transport_error,
)
from app.llm.gemini import GeminiProvider, gemini_provider
from app.llm.llm_manager import (
    DEFAULT_TIMEOUT_SECONDS,
    AllProvidersExhaustedException,
    LLMProviderManager,
)
from app.llm.openai_compat import (
    OpenAICompatibleProvider,
    groq_provider,
    openai_compatible_provider,
)
from app.llm.pipeline import (
    AttemptOutcome,
    LLMPipeline,
    NoProviderAvailable,
    build_providers,
)
from app.llm.types import (
    ChatMessage,
    LLMRequest,
    LLMResponse,
    ToolCall,
    ToolSpec,
    Usage,
    parse_tool_arguments,
)

__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "AllProvidersExhaustedException",
    "AttemptOutcome",
    "ChatMessage",
    "GeminiProvider",
    "LLMPipeline",
    "LLMProvider",
    "LLMProviderManager",
    "LLMRequest",
    "LLMResponse",
    "NoProviderAvailable",
    "OpenAICompatibleProvider",
    "ProviderConfigError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderRequestError",
    "ProviderResponseError",
    "ProviderServerError",
    "ProviderTimeoutError",
    "ProviderTransportError",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "build_providers",
    "gemini_provider",
    "groq_provider",
    "openai_compatible_provider",
    "parse_tool_arguments",
    "raise_for_status",
    "wrap_transport_error",
]
