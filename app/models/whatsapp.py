from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Profile(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str | None = None


class ContactProfile(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    wa_id: str
    profile: Profile | None = None


class MessageMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    display_phone_number: str | None = None
    phone_number_id: str | None = None


class TextMessageBody(BaseModel):
    model_config = ConfigDict(extra="allow")

    body: str = ""


class MediaMessageBody(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    mime_type: str | None = None
    sha256: str | None = None
    caption: str | None = None
    filename: str | None = None


class InteractiveMessageBody(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str | None = None
    text: str | None = None
    payload: str | None = None
    title: str | None = None
    id: str | None = None


class LocationMessageBody(BaseModel):
    model_config = ConfigDict(extra="allow")

    latitude: float | None = None
    longitude: float | None = None
    name: str | None = None
    address: str | None = None


class ReactionMessageBody(BaseModel):
    model_config = ConfigDict(extra="allow")

    message_id: str | None = None
    emoji: str | None = None


class MessageContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    forwarded: bool | None = None
    frequently_forwarded: bool | None = None
    from_: str | None = Field(default=None, alias="from")
    id: str | None = None
    referred_product: dict[str, Any] | None = None


class WebhookError(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: int | None = None
    title: str | None = None
    message: str | None = None
    error_data: dict[str, Any] | None = None


class WebhookMessage(BaseModel):
    """One inbound message. Kept permissive so new Meta message types do not break parsing."""

    model_config = ConfigDict(extra="allow", populate_by_name=True, protected_namespaces=())

    from_: str = Field(alias="from")
    id: str
    timestamp: str | None = None
    type: str = "unknown"
    text: TextMessageBody | None = None
    image: MediaMessageBody | None = None
    video: MediaMessageBody | None = None
    audio: MediaMessageBody | None = None
    document: MediaMessageBody | None = None
    sticker: MediaMessageBody | None = None
    location: LocationMessageBody | None = None
    reaction: ReactionMessageBody | None = None
    interactive: InteractiveMessageBody | None = None
    button: InteractiveMessageBody | None = None
    context: MessageContext | None = None
    errors: list[WebhookError] = Field(default_factory=list)
    referral: dict[str, Any] | None = None
    system: dict[str, Any] | None = None
    identity: dict[str, Any] | None = None

    @property
    def received_at(self) -> datetime:
        return parse_timestamp(self.timestamp)

    @property
    def body_text(self) -> str | None:
        if self.text is not None:
            return self.text.body
        for media in (self.image, self.video, self.audio, self.document, self.sticker):
            if media is not None and media.caption:
                return media.caption
        button = self.button
        if button is not None:
            return button.text or button.payload
        interactive = self.interactive
        if interactive is not None:
            return interactive.title
        return None


class MessageStatus(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    status: str = "unknown"
    timestamp: str | None = None
    recipient_id: str | None = None
    conversation: dict[str, Any] | None = None
    pricing: dict[str, Any] | None = None
    errors: list[WebhookError] = Field(default_factory=list)


class ChangeValue(BaseModel):
    model_config = ConfigDict(extra="allow")

    messaging_product: str | None = None
    metadata: MessageMetadata | None = None
    contacts: list[ContactProfile] = Field(default_factory=list)
    messages: list[WebhookMessage] = Field(default_factory=list)
    statuses: list[MessageStatus] = Field(default_factory=list)
    errors: list[WebhookError] = Field(default_factory=list)

    def contact_name(self, wa_id: str) -> str | None:
        for contact in self.contacts:
            if contact.wa_id == wa_id and contact.profile is not None:
                return contact.profile.name
        return None


class WebhookChange(BaseModel):
    model_config = ConfigDict(extra="allow")

    field: str | None = None
    value: ChangeValue


class WebhookEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    changes: list[WebhookChange] = Field(default_factory=list)


class WhatsAppWebhook(BaseModel):
    """Top level ``object``/``entry`` envelope delivered to the webhook endpoint."""

    model_config = ConfigDict(extra="allow")

    object: str | None = None
    entry: list[WebhookEntry] = Field(default_factory=list)

    def iter_changes(self) -> list[WebhookChange]:
        return [change for entry in self.entry for change in entry.changes]

    def message_count(self) -> int:
        return sum(len(change.value.messages) for change in self.iter_changes())


def parse_timestamp(value: str | None) -> datetime:
    """Meta sends unix seconds as a string; fall back to now for anything unexpected."""
    if value:
        try:
            return datetime.fromtimestamp(int(value), UTC)
        except (TypeError, ValueError):
            pass
    return datetime.now(UTC)


class InboundMessage(BaseModel):
    """Normalised view of one inbound message, the contract later phases consume."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    message_id: str
    wa_id: str
    kind: str
    text: str | None = None
    profile_name: str | None = None
    phone_number_id: str | None = None
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    raw: WebhookMessage


StatusName = Literal["sent", "delivered", "read", "failed", "deleted", "unknown"]
