"""Chat attachment and media-context services.

This package is intentionally separate from knowledge-base ingestion.  Chat
attachments are request-scoped user input; ingested media is searchable content.
"""

from agentic_rag.services.media.context import MediaContextBuilder
from agentic_rag.services.media.models import (
    MediaContext,
    StoredAttachment,
)
from agentic_rag.services.media.store import (
    AttachmentError,
    AttachmentNotFound,
    AttachmentStore,
    AttachmentTooLarge,
    UnsupportedAttachment,
)
from agentic_rag.services.media.transcribe import TranscriptionService
from agentic_rag.services.media.vision import VisionService

__all__ = [
    "AttachmentError",
    "AttachmentNotFound",
    "AttachmentStore",
    "AttachmentTooLarge",
    "MediaContext",
    "MediaContextBuilder",
    "StoredAttachment",
    "TranscriptionService",
    "UnsupportedAttachment",
    "VisionService",
]
