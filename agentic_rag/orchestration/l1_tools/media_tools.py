"""Per-request media tools.

Unlike knowledge-base ingestion, media tools operate on the attachments of the
current chat turn only.  Clients never pass local paths: each tool takes an
opaque ``attachment_id`` and refuses anything outside the request-scoped
allow-list assembled by the router.
"""

from __future__ import annotations

from typing import Any

from agentic_rag.data.models import MediaType
from agentic_rag.orchestration.l1_tools.base import BaseTool
from agentic_rag.services.media.models import StoredAttachment
from agentic_rag.services.media.store import AttachmentStore
from agentic_rag.services.media.transcribe import TranscriptionService
from agentic_rag.services.media.vision import (
    DEFAULT_FRAME_PROMPT,
    DEFAULT_IMAGE_PROMPT,
    VisionService,
)


class _ScopedMediaTool(BaseTool):
    """Shared allow-list handling for attachment-scoped tools."""

    expected_kind: MediaType

    def __init__(
        self,
        attachments: list[StoredAttachment],
        llm=None,
        store: AttachmentStore | None = None,
    ):
        self.attachments = {
            attachment.id: attachment
            for attachment in attachments
            if attachment.kind == self.expected_kind
        }
        self.llm = llm
        self.store = store
        self.vision = VisionService(llm=llm)
        self.transcriber: TranscriptionService | None = None

    @property
    def attachment_store(self) -> AttachmentStore:
        if self.store is None:
            self.store = AttachmentStore()
        return self.store

    def _resolve(self, attachment_id: str) -> StoredAttachment | str:
        attachment = self.attachments.get(attachment_id or "")
        if attachment is None:
            allowed = ", ".join(self.attachments) or "none"
            return (
                f"Attachment {attachment_id!r} is not available for this request. "
                f"Allowed {self.expected_kind.value} attachment_id values: {allowed}."
            )
        return attachment

    def _id_help(self) -> str:
        if not self.attachments:
            return "No attachments of this kind in the current request."
        return "Allowed attachment_id values: " + ", ".join(
            f"{att_id} ({att.filename})"
            for att_id, att in self.attachments.items()
        )


class ImageUnderstandTool(_ScopedMediaTool):
    """Analyze one image that belongs to the current request."""

    name = "image_understand"
    description = (
        "Analyze an uploaded image from the current request. Use this to read "
        "text, identify objects, or answer questions about an image. "
        "Only attachment IDs listed in the prompt are accepted."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "attachment_id": {
                "type": "string",
                "description": "ID of an image attachment from the current request",
            },
            "question": {
                "type": "string",
                "description": "What to ask about the image",
                "default": DEFAULT_IMAGE_PROMPT,
            },
        },
        "required": ["attachment_id"],
    }
    expected_kind = MediaType.IMAGE

    async def execute(
        self,
        attachment_id: str,
        question: str = DEFAULT_IMAGE_PROMPT,
    ) -> Any:
        attachment = self._resolve(attachment_id)
        if isinstance(attachment, str):
            return attachment
        if not self.vision.available:
            return self.vision.last_error or "Vision model is not configured."
        try:
            data_uri = self.attachment_store.read_data_uri(attachment)
        except Exception as exc:  # noqa: BLE001
            return f"Could not read the image attachment: {exc}"
        answer = await self.vision.describe_image(
            data_uri, question or DEFAULT_IMAGE_PROMPT, attachment.mime,
        )
        if not answer:
            return self.vision.last_error or "Image analysis returned no result."
        return answer


class AudioTranscribeTool(_ScopedMediaTool):
    """Transcribe one audio attachment from the current request."""

    name = "audio_transcribe"
    description = (
        "Transcribe an uploaded audio file from the current request to text. "
        "Only attachment IDs listed in the prompt are accepted."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "attachment_id": {
                "type": "string",
                "description": "ID of an audio attachment from the current request",
            },
        },
        "required": ["attachment_id"],
    }
    expected_kind = MediaType.AUDIO

    async def execute(self, attachment_id: str) -> Any:
        attachment = self._resolve(attachment_id)
        if isinstance(attachment, str):
            return attachment
        if attachment.digest.strip():
            return attachment.digest.strip()

        self.transcriber = self.transcriber or TranscriptionService()
        transcript = await self.transcriber.transcribe(attachment.path)
        if not transcript:
            return (
                self.transcriber.last_error
                or "No speech was detected in the audio attachment."
            )
        try:
            self.attachment_store.update_digest(attachment, transcript)
        except Exception:
            pass
        return transcript


class VideoAnalyzeTool(_ScopedMediaTool):
    """Describe keyframes from one video attachment in the current request."""

    name = "video_analyze"
    description = (
        "Extract keyframes from an uploaded video and describe them. Use this "
        "for visual questions about a video. Only attachment IDs listed in the "
        "prompt are accepted."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "attachment_id": {
                "type": "string",
                "description": "ID of a video attachment from the current request",
            },
            "question": {
                "type": "string",
                "description": "What to look for in the video",
                "default": DEFAULT_FRAME_PROMPT,
            },
            "max_frames": {
                "type": "integer",
                "description": "Maximum keyframes to inspect",
                "default": 4,
            },
        },
        "required": ["attachment_id"],
    }
    expected_kind = MediaType.VIDEO

    async def execute(
        self,
        attachment_id: str,
        question: str = DEFAULT_FRAME_PROMPT,
        max_frames: int = 4,
    ) -> Any:
        attachment = self._resolve(attachment_id)
        if isinstance(attachment, str):
            return attachment
        if not self.vision.available:
            return self.vision.last_error or "Vision model is not configured."

        from agentic_rag.config.settings import get_settings
        try:
            requested = int(max_frames or 4)
        except (TypeError, ValueError):
            requested = 4
        limit = max(1, min(
            requested,
            int(get_settings().media.max_video_frames),
        ))
        frames = self.vision.extract_video_frames(attachment.path, limit)
        if not frames:
            return self.vision.last_error or "Could not extract video frames."

        results = [
            f"Video attachment {attachment.filename}: "
            f"{len(frames)} keyframe(s) analyzed."
        ]
        prompt = question or DEFAULT_FRAME_PROMPT
        for i, frame in enumerate(frames, start=1):
            caption = await self.vision.describe_image(
                frame["data_uri"],
                f"{prompt}\n这是第 {i} 个关键帧（约 {frame['timestamp']} 秒）。",
                "image/jpeg",
                max_tokens=300,
            )
            results.append(
                f"Keyframe {i} (t={frame['timestamp']}s): "
                f"{caption or 'no description'}"
            )
        if len(results) == 1:
            return self.vision.last_error or "Video analysis returned no result."
        digest = "\n".join(results[1:])
        try:
            self.attachment_store.update_digest(attachment, digest)
        except Exception:
            pass
        return "\n".join(results)


def build_media_tools(
    attachments: list[StoredAttachment] | None,
    store: AttachmentStore | None = None,
    llm=None,
) -> list[BaseTool]:
    """Create the media tools needed for this request.

    Tools are instantiated per turn and scoped to the resolved attachment
    objects.  ``store`` is only needed when the caller uses a non-default
    attachment root (for example in tests).
    """
    attachments = attachments or []
    tools: list[BaseTool] = []
    kinds = {attachment.kind for attachment in attachments}
    if MediaType.IMAGE in kinds:
        tools.append(ImageUnderstandTool(attachments, llm=llm, store=store))
    if MediaType.AUDIO in kinds:
        tools.append(AudioTranscribeTool(attachments, llm=llm, store=store))
    if MediaType.VIDEO in kinds:
        tools.append(VideoAnalyzeTool(attachments, llm=llm, store=store))
    return tools
