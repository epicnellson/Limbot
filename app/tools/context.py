from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Who is asking, and what the bot is allowed to assume about them.

    ``wa_id`` comes from the verified webhook payload, never from the model. Tools receive the
    identity here instead of an argument, so a prompt injection cannot make the bot read
    somebody else's record by asking nicely.
    """

    wa_id: str
    display_name: str | None = None
    is_new_sender: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "wa_id": self.wa_id,
            "display_name": self.display_name,
            "is_new_sender": self.is_new_sender,
        }
