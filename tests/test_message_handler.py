from __future__ import annotations

import httpx
from app.config import Settings
from app.core.whatsapp import WhatsAppAPIError, WhatsAppCloudClient
from app.llm.base import LLMProvider, ProviderRequestError
from app.llm.pipeline import LLMPipeline
from app.llm.types import LLMRequest, LLMResponse
from app.models.whatsapp import WhatsAppWebhook
from app.services import message_handler
from app.services.answer import AnswerService
from app.services.dedupe import MessageDeduplicator

from conftest import _base_settings, counter_value, text_message_payload

SENDER = "15550001111"


class RecordingWhatsApp(WhatsAppCloudClient):
    """Captures outbound calls instead of talking to the Graph API."""

    def __init__(self, *, enabled: bool = True, fail_send: bool = False) -> None:
        self._enabled = enabled
        self._fail_send = fail_send
        self.sent: list[tuple[str, str]] = []
        self.reads: list[str] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def send_text(
        self, to: str, body: str, *, preview_url: bool = False, idempotent: bool = False
    ) -> object:
        if self._fail_send:
            raise WhatsAppAPIError("rate limited", status_code=429)
        self.sent.append((to, body))
        return type("Result", (), {"message_id": "wamid.OUT1", "raw": {}})()

    async def mark_message_read(self, message_id: str) -> None:
        self.reads.append(message_id)


class FixedProvider(LLMProvider):
    def __init__(self, text: str = "Linear algebra is at 10:00 in B204.") -> None:
        super().__init__("fixed-model", name="fixed", tier=1)
        self._text = text
        self.calls = 0

    async def complete(self, request: LLMRequest, *, http: httpx.AsyncClient) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=self._text, provider=self.name, model=self.model)


def make_answers(
    text: str = "Linear algebra is at 10:00 in B204.",
) -> tuple[AnswerService, FixedProvider]:
    settings = _base_settings(ai_enabled=True)
    provider = FixedProvider(text)
    return AnswerService(settings, LLMPipeline(settings, providers=[provider])), provider


def make_dedupe() -> MessageDeduplicator:
    return MessageDeduplicator(ttl_seconds=900, max_entries=100)


async def deliver(
    payload: WhatsAppWebhook,
    *,
    settings: Settings,
    whatsapp: RecordingWhatsApp,
    answers: AnswerService,
) -> None:
    await message_handler.handle_webhook(
        payload,
        settings=settings,
        whatsapp=whatsapp,  # type: ignore[arg-type]
        dedupe=make_dedupe(),
        answers=answers,
        http=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)),
    )


def body_for(
    text: str = "what is on today?", sender: str = SENDER, message_id: str = "wamid.IN1"
) -> WhatsAppWebhook:
    return WhatsAppWebhook.model_validate_json(
        text_message_payload(body=text, sender=sender, message_id=message_id)
    )


async def test_an_inbound_question_is_answered_and_sent_back() -> None:
    settings = _base_settings(ai_enabled=True)
    answers, provider = make_answers()
    whatsapp = RecordingWhatsApp()

    await deliver(body_for(), settings=settings, whatsapp=whatsapp, answers=answers)

    assert whatsapp.sent == [(SENDER, "Linear algebra is at 10:00 in B204.")]
    assert provider.calls == 1
    assert whatsapp.reads == ["wamid.IN1"]


async def test_the_message_is_marked_read_before_the_answer_is_produced() -> None:
    settings = _base_settings(ai_enabled=True)
    answers, _provider = make_answers()
    whatsapp = RecordingWhatsApp()

    await deliver(body_for(), settings=settings, whatsapp=whatsapp, answers=answers)

    assert whatsapp.reads == ["wamid.IN1"]


async def test_nothing_is_generated_when_whatsapp_outbound_is_not_configured() -> None:
    """No token and no phone number id means no reply can be sent, so no completion is bought."""
    settings = _base_settings(ai_enabled=True)
    answers, provider = make_answers()
    whatsapp = RecordingWhatsApp(enabled=False)

    await deliver(body_for(), settings=settings, whatsapp=whatsapp, answers=answers)

    assert whatsapp.sent == []
    assert whatsapp.reads == []
    assert provider.calls == 0


async def test_a_repeated_message_id_is_processed_only_once() -> None:
    settings = _base_settings(ai_enabled=True)
    answers, provider = make_answers()
    whatsapp = RecordingWhatsApp()
    dedupe = make_dedupe()
    payload = body_for()

    for _ in range(2):
        await message_handler.handle_webhook(
            payload,
            settings=settings,
            whatsapp=whatsapp,  # type: ignore[arg-type]
            dedupe=dedupe,
            answers=answers,
            http=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)),
        )

    assert provider.calls == 1
    assert len(whatsapp.sent) == 1
    assert dedupe.duplicates == 1


async def test_nothing_is_sent_when_no_provider_is_configured() -> None:
    settings = _base_settings(ai_enabled=True)
    answers = AnswerService(settings, LLMPipeline(settings, providers=[]))
    whatsapp = RecordingWhatsApp()

    await deliver(body_for(), settings=settings, whatsapp=whatsapp, answers=answers)

    assert whatsapp.sent == []
    assert whatsapp.reads == ["wamid.IN1"]


async def test_nothing_is_sent_when_every_tier_fails() -> None:
    settings = _base_settings(ai_enabled=True)
    answers, _provider = make_answers()
    whatsapp = RecordingWhatsApp()

    async def failing(request: LLMRequest, *, http: httpx.AsyncClient) -> LLMResponse:
        raise ProviderRequestError("400 bad request", provider="fixed")

    answers._pipeline.providers[0].complete = failing  # type: ignore[method-assign]

    await deliver(body_for(), settings=settings, whatsapp=whatsapp, answers=answers)

    assert whatsapp.sent == [("15550001111", message_handler.NOTHING_TO_SAY)]


async def test_a_broken_pipeline_still_produces_a_sendable_reply() -> None:
    settings = _base_settings(ai_enabled=True)
    answers, provider = make_answers()
    whatsapp = RecordingWhatsApp()

    async def exploding(request: LLMRequest, *, http: httpx.AsyncClient) -> LLMResponse:
        raise ZeroDivisionError("something unforeseen")

    provider.complete = exploding  # type: ignore[method-assign]

    await deliver(body_for(), settings=settings, whatsapp=whatsapp, answers=answers)

    assert whatsapp.sent == [(SENDER, message_handler.NOTHING_TO_SAY)]


async def test_a_rejected_outbound_message_does_not_break_the_handler() -> None:
    settings = _base_settings(ai_enabled=True)
    answers, _provider = make_answers()
    whatsapp = RecordingWhatsApp(fail_send=True)

    await deliver(body_for(), settings=settings, whatsapp=whatsapp, answers=answers)

    assert whatsapp.sent == []


async def test_echo_mode_skips_the_pipeline_entirely() -> None:
    settings = _base_settings(ai_enabled=True, whatsapp_echo_mode=True)
    answers, provider = make_answers()
    whatsapp = RecordingWhatsApp()

    await deliver(body_for(), settings=settings, whatsapp=whatsapp, answers=answers)

    assert whatsapp.sent == [(SENDER, "what is on today?")]
    assert provider.calls == 0


async def test_a_non_text_message_is_answered_with_nothing() -> None:
    settings = _base_settings(ai_enabled=True)
    answers, provider = make_answers()
    whatsapp = RecordingWhatsApp()
    payload = WhatsAppWebhook.model_validate_json(
        text_message_payload()
        .replace(b'"type": "text"', b'"type": "image"')
        .replace(b'"text": {"body": "what is on today?"}', b'"image": {"id": "media1"}')
    )

    await deliver(payload, settings=settings, whatsapp=whatsapp, answers=answers)

    assert whatsapp.sent == []
    assert provider.calls == 0


async def test_a_change_for_another_field_is_ignored() -> None:
    settings = _base_settings(ai_enabled=True)
    answers, provider = make_answers()
    whatsapp = RecordingWhatsApp()
    payload = WhatsAppWebhook.model_validate_json(
        text_message_payload().replace(b'"field": "messages"', b'"field": "history"')
    )

    await deliver(payload, settings=settings, whatsapp=whatsapp, answers=answers)

    assert whatsapp.sent == []
    assert provider.calls == 0


async def test_delivery_statuses_are_counted_and_do_not_trigger_the_pipeline() -> None:
    settings = _base_settings(ai_enabled=True)
    answers, provider = make_answers()
    whatsapp = RecordingWhatsApp()
    payload = WhatsAppWebhook.model_validate_json(
        text_message_payload().replace(
            b'"messages": [',
            b'"statuses": [{"id": "wamid.S1", "status": "delivered", "timestamp": "1700000001",'
            b' "recipient_id": "15550001111"}], "messages": [',
        )
    )
    before = counter_value("limbot_message_status_events_total", status="delivered")

    await deliver(payload, settings=settings, whatsapp=whatsapp, answers=answers)

    after = counter_value("limbot_message_status_events_total", status="delivered")
    assert after - before == 1
    assert provider.calls == 1


async def test_the_answers_metric_records_which_provider_answered() -> None:
    settings = _base_settings(ai_enabled=True)
    answers, _provider = make_answers()
    whatsapp = RecordingWhatsApp()
    before = counter_value("limbot_answers_total", provider="fixed", tools="none")

    await deliver(body_for(), settings=settings, whatsapp=whatsapp, answers=answers)

    after = counter_value("limbot_answers_total", provider="fixed", tools="none")
    assert after - before == 1


async def test_the_fallback_answer_carries_the_error_for_the_metrics() -> None:
    settings = _base_settings(ai_enabled=True)
    answers = AnswerService(settings, LLMPipeline(settings, providers=[]))
    whatsapp = RecordingWhatsApp()

    await deliver(body_for(), settings=settings, whatsapp=whatsapp, answers=answers)

    assert whatsapp.sent == []


async def test_a_sender_keeps_their_conversation_between_messages() -> None:
    settings = _base_settings(ai_enabled=True)
    answers, provider = make_answers()
    whatsapp = RecordingWhatsApp()

    first = body_for("first?", message_id="wamid.IN1")
    second = body_for("second?", message_id="wamid.IN2")
    await deliver(first, settings=settings, whatsapp=whatsapp, answers=answers)
    await deliver(second, settings=settings, whatsapp=whatsapp, answers=answers)

    assert provider.calls == 2
    assert [text for _sender, text in whatsapp.sent] == [
        "Linear algebra is at 10:00 in B204.",
        "Linear algebra is at 10:00 in B204.",
    ]


def test_the_fallback_text_is_short_enough_for_a_whatsapp_bubble() -> None:
    assert len(message_handler.NOTHING_TO_SAY) < 1024
