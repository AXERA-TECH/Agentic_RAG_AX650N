"""Domain models for chat attachments and assembled media context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_rag.data.models import AttachmentRef, MediaType, Message, MessageRole


@dataclass
class StoredAttachment:
    """One uploaded file plus its derived text digest."""

    id: str
    kind: MediaType
    path: str
    filename: str
    mime: str
    size: int
    digest: str = ""
    user_id: str = ""
    session_id: str = ""
    created_at: str = ""

    def to_ref(self) -> AttachmentRef:
        return AttachmentRef(
            id=self.id,
            kind=self.kind,
            filename=self.filename,
            mime=self.mime,
            size=self.size,
            digest=self.digest,
        )


@dataclass
class MediaContext:
    """Media prepared for exactly one agent turn.

    ``model_parts`` contains provider-neutral OpenAI-style content parts for
    the current user message.  ``text_digest`` is the fallback and history
    representation; it is also appended to the deterministic RAG seed query.
    """

    attachments: list[StoredAttachment] = field(default_factory=list)
    model_parts: list[dict[str, Any]] = field(default_factory=list)
    text_digest: str = ""
    retrieval_query: str = ""
    estimated_tokens: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def has_media(self) -> bool:
        return bool(self.attachments)

    @property
    def has_native_visual(self) -> bool:
        return any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in self.model_parts
        )

    def to_message(self) -> Message | None:
        """Build the current-turn user message, or None when not multimodal.

        When attachments are present the content stays a *list* even if the
        model only receives a text digest.  Both engines use "current message
        is a content list" as their signal that the query has already been
        injected; collapsing to a plain string would make them append the
        query a second time.
        """
        if not self.model_parts:
            return None
        if self.attachments:
            return Message(role=MessageRole.USER, content=self.model_parts)
        if len(self.model_parts) == 1 and self.model_parts[0].get("type") == "text":
            return Message(role=MessageRole.USER, content=self.model_parts[0]["text"])
        return Message(role=MessageRole.USER, content=self.model_parts)
