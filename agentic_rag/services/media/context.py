"""Build the per-turn media context for one agent request.

One attachment can be represented in two ways:

* native content parts (images / video keyframes) for a vision-capable model;
* a textual digest (caption or transcript) used by text-only models, by the
  deterministic RAG seed query, by media tools and by persisted history.

Keeping both representations behind one builder means routing, tool use and
history rendering never need to know how a file was encoded.
"""

from __future__ import annotations

from typing import Any

from agentic_rag.data.models import MediaType
from agentic_rag.services.llm.base import estimate_tokens
from agentic_rag.services.media.models import MediaContext, StoredAttachment
from agentic_rag.services.media.store import AttachmentStore
from agentic_rag.services.media.transcribe import TranscriptionService
from agentic_rag.services.media.vision import (
    DEFAULT_FRAME_PROMPT,
    DEFAULT_IMAGE_PROMPT,
    VisionService,
)


def _safe_llm():
    try:
        from agentic_rag.services.llm.factory import get_llm
        return get_llm()
    except Exception:
        return None


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


class MediaContextBuilder:
    """Turn scoped attachment IDs into model parts plus a text digest."""

    def __init__(
        self,
        store: AttachmentStore | None = None,
        llm=None,
        vision: VisionService | None = None,
        transcriber: TranscriptionService | None = None,
    ):
        self._store = store
        self._llm = llm
        self._vision = vision
        self._transcriber = transcriber

    @property
    def store(self) -> AttachmentStore:
        if self._store is None:
            self._store = AttachmentStore()
        return self._store

    async def build(
        self,
        query: str,
        attachment_ids: list[str],
        *,
        need_digest: bool = False,
        user_id: str = "",
        session_id: str = "",
    ) -> MediaContext:
        """Resolve attachments and assemble the current-turn context.

        ``need_digest`` is set by the caller when the selected strategy is
        tool use (the evidence gate and media tools work on text) or when the
        active model cannot see images.  Pure DIRECT image requests therefore
        avoid an extra captioning call.
        """
        from agentic_rag.config.settings import get_settings

        settings = get_settings()
        llm = self._llm if self._llm is not None else _safe_llm()
        vision = self._vision or VisionService(llm=llm)
        transcriber = self._transcriber or TranscriptionService()

        attachments = self.store.resolve(
            attachment_ids, user_id=user_id, session_id=session_id,
        )
        supports_vision = bool(getattr(llm, "supports_vision", False))
        diagnostics: dict[str, Any] = {
            "attachment_count": len(attachments),
            "native_vision": supports_vision,
            "digest_requested": need_digest,
            "degraded": [],
        }

        manifest: list[str] = []
        digest_blocks: list[str] = []
        visual_parts: list[dict[str, Any]] = []
        prompt_blocks: list[str] = []

        for attachment in attachments:
            kind = attachment.kind
            if kind == MediaType.IMAGE:
                await self._handle_image(
                    attachment, need_digest, supports_vision, vision,
                    visual_parts, digest_blocks, prompt_blocks, diagnostics,
                )
            elif kind == MediaType.AUDIO:
                await self._handle_audio(
                    attachment, transcriber, digest_blocks, prompt_blocks,
                    diagnostics,
                )
            elif kind == MediaType.VIDEO:
                await self._handle_video(
                    attachment, need_digest, supports_vision, vision,
                    visual_parts, digest_blocks, prompt_blocks, diagnostics,
                )
            else:
                diagnostics["degraded"].append(
                    f"{attachment.id}: unsupported kind {kind.value}"
                )

            manifest.append(
                f'- {kind.value} "{attachment.filename}" '
                f"(attachment_id={attachment.id}, mime={attachment.mime})"
            )

        text_digest = _truncate(
            "\n\n".join(block for block in digest_blocks if block.strip()),
            int(settings.media.max_digest_chars),
        )

        model_parts: list[dict[str, Any]] = []
        prompt_text = (query or "").strip()
        if manifest and (need_digest or not supports_vision or prompt_blocks):
            header = "[附件上下文]\n" + "\n".join(manifest)
            if prompt_blocks:
                header += "\n\n" + "\n\n".join(prompt_blocks)
            prompt_text = (
                f"{prompt_text}\n\n{_truncate(header, int(settings.media.max_digest_chars))}"
                if prompt_text else _truncate(header, int(settings.media.max_digest_chars))
            )
        # An image/video-only turn still needs a text instruction: some
        # OpenAI-compatible servers reject (or ignore) a user message whose
        # content contains only image parts.
        if not prompt_text and visual_parts:
            prompt_text = "请描述附件内容。"
        if prompt_text:
            model_parts.append({"type": "text", "text": prompt_text})
        model_parts.extend(visual_parts)

        # A provider-neutral text representation for retrieval and history.
        retrieval_parts = [(query or "").strip(), text_digest]
        retrieval_query = "\n".join(
            part for part in retrieval_parts if part
        ).strip()

        estimated = 0
        count_tokens = getattr(llm, "count_tokens", None)
        for part in model_parts:
            if part.get("type") != "text":
                continue
            try:
                estimated += (
                    count_tokens(part["text"])
                    if callable(count_tokens)
                    else estimate_tokens(part["text"])
                )
            except Exception:
                estimated += estimate_tokens(part["text"])
        # Vision tokens are provider-specific; reserve a conservative flat
        # amount per native image so history cannot overrun the context.
        native_visuals = sum(
            1 for part in model_parts if part.get("type") == "image_url"
        )
        estimated += native_visuals * 1024

        if not supports_vision and any(
            a.kind in (MediaType.IMAGE, MediaType.VIDEO) for a in attachments
        ):
            diagnostics["degraded"].append(
                "active model has no native vision; using text digest"
            )
        if any(a.kind == MediaType.VIDEO for a in attachments) and not vision.available:
            diagnostics["degraded"].append(
                vision.last_error or "vision model unavailable for video"
            )

        return MediaContext(
            attachments=attachments,
            model_parts=model_parts,
            text_digest=text_digest,
            retrieval_query=retrieval_query,
            estimated_tokens=estimated,
            diagnostics=diagnostics,
        )

    # ── Per-kind handling ─────────────────────────────────

    async def _handle_image(
        self,
        attachment: StoredAttachment,
        need_digest: bool,
        supports_vision: bool,
        vision: VisionService,
        visual_parts: list[dict[str, Any]],
        digest_blocks: list[str],
        prompt_blocks: list[str],
        diagnostics: dict[str, Any],
    ) -> None:
        data_uri = ""
        if supports_vision:
            try:
                data_uri = self.store.read_data_uri(attachment)
            except OSError as exc:
                diagnostics["degraded"].append(
                    f"{attachment.id}: could not read image ({exc})"
                )
            else:
                visual_parts.append(_image_part(data_uri, attachment.mime))

        if need_digest or not supports_vision:
            caption = attachment.digest.strip()
            if not caption and vision.available:
                if not data_uri:
                    try:
                        data_uri = self.store.read_data_uri(attachment)
                    except OSError as exc:
                        diagnostics["degraded"].append(
                            f"{attachment.id}: could not read image ({exc})"
                        )
                if data_uri:
                    caption = await vision.describe_image(
                        data_uri, DEFAULT_IMAGE_PROMPT, attachment.mime,
                    )
                    if caption:
                        self.store.update_digest(attachment, caption)
            if not caption and (need_digest or not supports_vision):
                diagnostics["degraded"].append(
                    f"{attachment.id}: image caption unavailable"
                )
            if caption:
                digest_blocks.append(
                    f'[图片附件: {attachment.filename}]\n{caption}'
                )
                prompt_blocks.append(
                    f'[图片附件: {attachment.filename} 自动描述]\n{caption}'
                )
            else:
                prompt_blocks.append(
                    f"[图片附件: {attachment.filename}] 无法生成文字描述。"
                )

    async def _handle_audio(
        self,
        attachment: StoredAttachment,
        transcriber: TranscriptionService,
        digest_blocks: list[str],
        prompt_blocks: list[str],
        diagnostics: dict[str, Any],
    ) -> None:
        transcript = attachment.digest.strip()
        if not transcript:
            transcript = await transcriber.transcribe(attachment.path)
            if transcript:
                self.store.update_digest(attachment, transcript)
        if transcript:
            digest_blocks.append(
                f'[音频附件: {attachment.filename}]\n{transcript}'
            )
            prompt_blocks.append(
                f'[音频附件: {attachment.filename} 转写]\n{transcript}'
            )
        else:
            reason = transcriber.last_error or "未检测到语音内容"
            diagnostics["degraded"].append(
                f"{attachment.id}: {reason}"
            )
            prompt_blocks.append(
                f"[音频附件: {attachment.filename}] 转写不可用：{reason}"
            )

    async def _handle_video(
        self,
        attachment: StoredAttachment,
        need_digest: bool,
        supports_vision: bool,
        vision: VisionService,
        visual_parts: list[dict[str, Any]],
        digest_blocks: list[str],
        prompt_blocks: list[str],
        diagnostics: dict[str, Any],
    ) -> None:
        from agentic_rag.config.settings import get_settings

        frames = vision.extract_video_frames(
            attachment.path, get_settings().media.max_video_frames,
        )
        if not frames:
            reason = vision.last_error or "无法提取视频关键帧"
            diagnostics["degraded"].append(f"{attachment.id}: {reason}")
            prompt_blocks.append(
                f"[视频附件: {attachment.filename}] 处理失败：{reason}"
            )
            return

        if supports_vision:
            for frame in frames:
                visual_parts.append(
                    _image_part(frame["data_uri"], "image/jpeg")
                )

        if need_digest or not supports_vision:
            digest = attachment.digest.strip()
            if not digest:
                captions: list[str] = []
                for i, frame in enumerate(frames, start=1):
                    caption = await vision.describe_image(
                        frame["data_uri"],
                        f"视频关键帧 {i}（约 {frame['timestamp']} 秒）。"
                        + DEFAULT_FRAME_PROMPT,
                        "image/jpeg",
                        max_tokens=300,
                    )
                    captions.append(
                        f"关键帧 {i} (t={frame['timestamp']}s): "
                        f"{caption or '无描述'}"
                    )
                digest = "\n".join(captions)
                if captions:
                    self.store.update_digest(attachment, digest)
            if digest:
                digest_blocks.append(
                    f'[视频附件: {attachment.filename}]\n{digest}'
                )
                prompt_blocks.append(
                    f'[视频附件: {attachment.filename} 关键帧描述]\n{digest}'
                )
            else:
                diagnostics["degraded"].append(
                    f"{attachment.id}: video caption unavailable"
                )
                prompt_blocks.append(
                    f"[视频附件: {attachment.filename}] 无法生成视频描述。"
                )


def _image_part(data_uri: str, mime: str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": data_uri},
        "media_type": mime or "image/png",
    }
