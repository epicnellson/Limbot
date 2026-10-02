from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "dev", "staging", "production"]

DEFAULT_SYSTEM_PROMPT = (
    "You are Limbot, a WhatsApp assistant for university students.\n"
    "Rules:\n"
    "- Use the provided tools for anything about the student's own records: timetable, "
    "assignment deadlines, exam seating and grades. Never guess these.\n"
    "- Use the provided context for course material. If the answer is not in the context, "
    "say so plainly instead of inventing it.\n"
    "- If a tool reports that the student is not linked, explain how to link their number.\n"
    "- Keep replies under 120 words, plain text, no markdown tables.\n"
    "- Say which source or tool the answer came from when it matters."
)


class Settings(BaseSettings):
    """Runtime configuration, sourced from environment variables and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "limbot"
    app_version: str = "0.2.0"
    environment: Environment = "local"
    log_level: str = "INFO"
    log_json: bool = True

    host: str = "0.0.0.0"
    port: int = 8000
    enable_docs: bool | None = None
    trusted_hosts: str = "*"
    debug_endpoints: bool = False

    whatsapp_app_secret: SecretStr | None = None
    whatsapp_verify_token: SecretStr = SecretStr("change-me")
    whatsapp_access_token: SecretStr | None = None
    whatsapp_phone_number_id: str | None = None
    whatsapp_business_account_id: str | None = None
    whatsapp_graph_base_url: str = "https://graph.facebook.com"
    whatsapp_graph_version: str = "v23.0"
    whatsapp_signature_required: bool = True
    whatsapp_echo_mode: bool = False

    http_timeout_seconds: float = 15.0
    http_connect_timeout_seconds: float = 5.0
    http_max_connections: int = 50
    http_max_keepalive_connections: int = 20
    http_keepalive_expiry_seconds: float = 30.0
    http_connect_retries: int = 2
    outbound_max_attempts: int = 4
    outbound_retry_base_delay_seconds: float = 0.5
    outbound_retry_max_delay_seconds: float = 8.0

    background_max_concurrency: int = 8
    background_drain_timeout_seconds: float = 20.0
    message_dedupe_ttl_seconds: int = 900
    message_dedupe_max_entries: int = 20_000

    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_timeout_seconds: float = 5.0
    qdrant_collection: str = "limbot_knowledge"
    qdrant_required: bool = True

    ai_enabled: bool = True
    ai_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ai_request_timeout_seconds: float = 45.0
    ai_temperature: float = 0.2
    ai_max_tokens: int = 1024
    ai_max_tool_rounds: int = 3
    ai_max_tool_calls_per_response: int = 4
    ai_tool_call_timeout_seconds: float = 10.0
    ai_circuit_failure_threshold: int = 3
    ai_circuit_recovery_seconds: float = 30.0
    ai_circuit_half_open_calls: int = 1

    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"

    gemini_api_key: SecretStr | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_model: str = "gemini-2.0-flash"

    tier3_base_url: str = "http://ollama:11434/v1"
    tier3_api_key: SecretStr | None = None
    tier3_model: str = "llama3.2"
    tier3_enabled: bool = False
    tier3_owns_retry: bool = True

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    embedding_cache_dir: str = "/var/cache/limbot/embeddings"
    embedding_batch_size: int = 32

    rag_enabled: bool = True
    rag_top_k: int = 4
    rag_score_threshold: float = 0.35
    rag_max_context_chars: int = 6000
    knowledge_chunk_chars: int = 900
    knowledge_chunk_overlap: int = 150

    postgres_dsn: SecretStr | None = None
    postgres_pool_min_size: int = 1
    postgres_pool_max_size: int = 5
    postgres_command_timeout_seconds: float = 5.0

    conversation_max_messages: int = 16
    conversation_ttl_seconds: int = 1800
    conversation_max_conversations: int = 500

    @field_validator(
        "whatsapp_app_secret",
        "whatsapp_access_token",
        "qdrant_api_key",
        "whatsapp_phone_number_id",
        "whatsapp_business_account_id",
        "groq_api_key",
        "gemini_api_key",
        "tier3_api_key",
        "postgres_dsn",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        """Treat empty environment variables as unset so compose can pass through blank values."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()

    @model_validator(mode="after")
    def _validate_consistency(self) -> Settings:
        if self.whatsapp_signature_required and self.whatsapp_app_secret is None:
            raise ValueError(
                "WHATSAPP_APP_SECRET must be set when WHATSAPP_SIGNATURE_REQUIRED is true. "
                "Set the app secret from the WhatsApp developer dashboard, or disable signature "
                "verification for local-only experiments."
            )
        if self.environment == "production":
            missing = [
                name
                for name, value in (
                    ("WHATSAPP_ACCESS_TOKEN", self.whatsapp_access_token),
                    ("WHATSAPP_PHONE_NUMBER_ID", self.whatsapp_phone_number_id),
                )
                if not value
            ]
            if missing:
                raise ValueError(f"missing required production settings: {', '.join(missing)}")
            if self.whatsapp_verify_token.get_secret_value() == "change-me":
                raise ValueError("WHATSAPP_VERIFY_TOKEN must be changed in production")
            if self.debug_endpoints:
                raise ValueError("DEBUG_ENDPOINTS must be false in production")
            if self.ai_enabled and not self.configured_providers:
                raise ValueError(
                    "AI_ENABLED is true but no provider is configured: set GROQ_API_KEY, "
                    "GEMINI_API_KEY, or TIER3_ENABLED with TIER3_BASE_URL"
                )
            if self.rag_enabled and self.postgres_dsn is None:
                raise ValueError("POSTGRES_DSN must be set in production when RAG_ENABLED is true")
        if self.rag_enabled and not 0.0 <= self.rag_score_threshold <= 1.0:
            raise ValueError("RAG_SCORE_THRESHOLD must be between 0 and 1")
        if self.knowledge_chunk_overlap >= self.knowledge_chunk_chars:
            raise ValueError("KNOWLEDGE_CHUNK_OVERLAP must be smaller than KNOWLEDGE_CHUNK_CHARS")
        if self.ai_max_tool_rounds < 1:
            raise ValueError("AI_MAX_TOOL_ROUNDS must be at least 1")
        if self.ai_circuit_failure_threshold < 1:
            raise ValueError("AI_CIRCUIT_FAILURE_THRESHOLD must be at least 1")
        if self.knowledge_chunk_chars < 200:
            raise ValueError("KNOWLEDGE_CHUNK_CHARS must be at least 200")
        return self

    @property
    def docs_enabled(self) -> bool:
        if self.enable_docs is not None:
            return self.enable_docs
        return self.environment != "production"

    @property
    def allowed_hosts(self) -> list[str]:
        """Host header allowlist, comma separated. Parsed here because a list field would force
        JSON in the environment variable."""
        hosts = [item.strip() for item in self.trusted_hosts.split(",") if item.strip()]
        return hosts or ["*"]

    @property
    def whatsapp_outbound_enabled(self) -> bool:
        return bool(self.whatsapp_access_token and self.whatsapp_phone_number_id)

    @property
    def qdrant_api_key_value(self) -> str | None:
        return _secret_value(self.qdrant_api_key)

    @property
    def groq_api_key_value(self) -> str | None:
        return _secret_value(self.groq_api_key)

    @property
    def gemini_api_key_value(self) -> str | None:
        return _secret_value(self.gemini_api_key)

    @property
    def tier3_api_key_value(self) -> str | None:
        return _secret_value(self.tier3_api_key)

    @property
    def postgres_dsn_value(self) -> str | None:
        return _secret_value(self.postgres_dsn)

    @property
    def configured_providers(self) -> tuple[str, ...]:
        """Provider tiers that have credentials, in fallback order."""
        tiers: list[str] = []
        if self.groq_api_key_value:
            tiers.append("groq")
        if self.gemini_api_key_value:
            tiers.append("gemini")
        if self.tier3_enabled:
            tiers.append("tier3")
        return tuple(tiers)

    @property
    def tools_enabled(self) -> bool:
        return self.postgres_dsn_value is not None

    @property
    def vector_store_enabled(self) -> bool:
        return self.rag_enabled and bool(self.qdrant_url)


def _secret_value(secret: SecretStr | None) -> str | None:
    if secret is None:
        return None
    value = secret.get_secret_value().strip()
    return value or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
