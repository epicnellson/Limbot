from __future__ import annotations

import pytest
from app.config import Settings
from pydantic import SecretStr, ValidationError

SECRET = {"whatsapp_app_secret": "secret"}
OUTBOUND = {"whatsapp_access_token": "token", "whatsapp_phone_number_id": "123"}
AI = {"groq_api_key": "gsk-key"}
DB = {"postgres_dsn": "postgresql://limbot:limbot@postgres:5432/limbot"}


def make_settings(**overrides: object) -> Settings:
    """Build settings from explicit values only.

    ``_env_file=None`` keeps a developer's local ``.env`` out of the assertions, which
    otherwise changes results depending on what happens to be checked out.
    """
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_blank_environment_values_become_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "")
    monkeypatch.setenv("QDRANT_API_KEY", "")
    settings = make_settings(**SECRET)
    assert settings.whatsapp_access_token is None
    assert settings.qdrant_api_key_value is None
    assert settings.whatsapp_outbound_enabled is False


def test_signature_verification_requires_an_app_secret() -> None:
    with pytest.raises(ValidationError, match="WHATSAPP_APP_SECRET"):
        make_settings(whatsapp_signature_required=True)


def test_production_requires_outbound_credentials() -> None:
    with pytest.raises(ValidationError, match="WHATSAPP_ACCESS_TOKEN"):
        make_settings(**SECRET, environment="production", whatsapp_verify_token="real-token")


def test_production_refuses_placeholder_verify_token() -> None:
    with pytest.raises(ValidationError, match="WHATSAPP_VERIFY_TOKEN"):
        make_settings(**SECRET, **OUTBOUND, environment="production")


def test_production_refuses_debug_endpoints() -> None:
    with pytest.raises(ValidationError, match="DEBUG_ENDPOINTS"):
        make_settings(
            **SECRET,
            **OUTBOUND,
            environment="production",
            whatsapp_verify_token="real-token",
            debug_endpoints=True,
        )


def test_production_refuses_ai_without_a_provider() -> None:
    with pytest.raises(ValidationError, match="GROQ_API_KEY"):
        make_settings(
            **SECRET, **OUTBOUND, environment="production", whatsapp_verify_token="real-token"
        )


def test_production_refuses_rag_without_a_database() -> None:
    with pytest.raises(ValidationError, match="POSTGRES_DSN"):
        make_settings(
            **SECRET,
            **OUTBOUND,
            **AI,
            environment="production",
            whatsapp_verify_token="real-token",
        )


def test_production_accepts_a_complete_configuration() -> None:
    settings = make_settings(
        **SECRET,
        **OUTBOUND,
        **AI,
        **DB,
        environment="production",
        whatsapp_verify_token="real-token",
    )
    assert settings.docs_enabled is False
    assert settings.whatsapp_outbound_enabled is True
    assert settings.tools_enabled is True


def test_docs_are_enabled_outside_production_by_default() -> None:
    assert make_settings(**SECRET).docs_enabled is True
    assert make_settings(**SECRET, enable_docs=False).docs_enabled is False


def test_trusted_hosts_are_parsed_as_a_comma_separated_list() -> None:
    assert make_settings(**SECRET).allowed_hosts == ["*"]
    assert make_settings(**SECRET, trusted_hosts="a.test, b.test").allowed_hosts == [
        "a.test",
        "b.test",
    ]
    assert make_settings(**SECRET, trusted_hosts="").allowed_hosts == ["*"]


def test_outbound_flag_needs_both_token_and_phone_number_id() -> None:
    only_token = make_settings(**SECRET, whatsapp_access_token="token")
    both = make_settings(**SECRET, **OUTBOUND)
    assert only_token.whatsapp_outbound_enabled is False
    assert both.whatsapp_outbound_enabled is True


def test_configured_providers_are_reported_in_fallback_order() -> None:
    assert make_settings(**SECRET).configured_providers == ()
    assert make_settings(**SECRET, **AI).configured_providers == ("groq",)
    assert make_settings(**SECRET, **AI, gemini_api_key="AIza-key").configured_providers == (
        "groq",
        "gemini",
    )
    assert make_settings(
        **SECRET, tier3_enabled=True, tier3_base_url="http://ollama:11434/v1"
    ).configured_providers == ("tier3",)
    assert make_settings(**SECRET, tier3_enabled=True, tier3_base_url="").configured_providers == (
        "tier3",
    )


def test_secrets_are_only_exposed_through_value_accessors() -> None:
    settings = make_settings(
        **SECRET, **AI, **DB, gemini_api_key=SecretStr("AIza-key"), tier3_api_key="  "
    )
    assert settings.groq_api_key_value == "gsk-key"
    assert settings.gemini_api_key_value == "AIza-key"
    assert settings.postgres_dsn_value == "postgresql://limbot:limbot@postgres:5432/limbot"
    assert settings.tier3_api_key_value is None
    assert "gsk-key" not in repr(settings)


def test_tools_require_a_database() -> None:
    assert make_settings(**SECRET).tools_enabled is False
    assert make_settings(**SECRET, **DB).tools_enabled is True


def test_default_system_prompt_instructs_timetable_tool_use() -> None:
    prompt = make_settings(**SECRET).ai_system_prompt
    assert "Show my schedule" in prompt
    assert "get_timetable" not in prompt
    assert "Never guess these from memory" in prompt


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rag_score_threshold": 1.5}, "RAG_SCORE_THRESHOLD"),
        ({"knowledge_chunk_overlap": 900}, "KNOWLEDGE_CHUNK_OVERLAP"),
        ({"knowledge_chunk_chars": 50}, "KNOWLEDGE_CHUNK_CHARS"),
        ({"ai_max_tool_rounds": 0}, "AI_MAX_TOOL_ROUNDS"),
        ({"ai_circuit_failure_threshold": 0}, "AI_CIRCUIT_FAILURE_THRESHOLD"),
    ],
)
def test_phase_two_settings_are_validated(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        make_settings(**SECRET, **overrides)
