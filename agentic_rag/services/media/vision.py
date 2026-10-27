"""Vision helpers for chat attachments.

The active chat model may or may not accept images natively.  This service
provides the second path: captions, OCR-like descriptions and video keyframe
summaries produced by a vision-capable provider.  It never reads paths from
clients; callers hand it bytes-derived data URIs or attachment paths resolved
by :class:`AttachmentStore`.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from agentic_rag.data.models import Message, MessageRole

DEFAULT_IMAGE_PROMPT = (
    "请客观描述这张图片的内容。提取其中所有可见文字、数字、表格、"
    "界面元素和关键细节；不要推测图片之外的信息。"
)

DEFAULT_FRAME_PROMPT = (
    "这是视频的一个关键帧。请简要描述画面内容，并提取可见文字。"
)


class VisionService:
    """Resolve a vision-capable LLM and run image/video understanding."""

    def __init__(self, llm=None):
        self._llm = llm
        self._resolved: Any = None
        self._resolved_done = False
        self.last_error = ""

    # ── Provider resolution ───────────────────────────────

    def _provider(self):
        """Return a vision-capable provider or None.

        Priority:
        1. an explicitly injected LLM when it supports vision;
        2. ``media.vision_provider`` from settings;
        3. the default provider;
        4. any configured provider that reports vision support.
        """
        from agentic_rag.config.settings import get_settings

        settings = get_settings()
        media = settings.media

        if not media.vision_provider and self._llm is not None:
            if getattr(self._llm, "supports_vision", False):
                return self._llm
            if self._resolved_done:
                return self._resolved

        if self._resolved_done:
            return self._resolved
        self._resolved_done = True
        self._resolved = self._resolve_from_settings(settings)

        if self._resolved is None and self._illm_supports_vision():
            self._resolved = self._llm
        if self._resolved is None:
            self.last_error = "Vision model is not configured."
        return self._resolved

    def _illm_supports_vision(self) -> bool:
        return bool(self._llm is not None
                    and getattr(self._llm, "supports_vision", False))

    @staticmethod
    def _resolve_from_settings(settings):
        from agentic_rag.services.llm.factory import LLMFactory

        media = settings.media
        names: list[str] = []
        if media.vision_provider:
            names.append(media.vision_provider)
        if settings.default_provider and settings.default_provider not in names:
            names.append(settings.default_provider)
        for name in settings.llm_providers:
            if name not in names:
                names.append(name)

        for name in names:
            config = settings.llm_providers.get(name)
            if config is None:
                continue
            model = config.model
            if name == media.vision_provider:
                model = media.vision_model or config.vision_model or config.model
            elif not media.vision_provider and name == settings.default_provider:
                model = media.vision_model or config.vision_model or config.model
            else:
                model = config.vision_model or config.model
            candidate_config = (
                config.model_copy(update={
                    "model": model,
                    # OpenAI-compatible providers send ``vision_model`` when
                    # the request contains images; keep both fields aligned
                    # with the explicit media override.
                    "vision_model": model,
                })
                if model != config.model else config
            )
            try:
                candidate = LLMFactory.create(name, candidate_config)
            except Exception:
                continue
            if getattr(candidate, "supports_vision", False):
                return candidate
        return None

    @property
    def available(self) -> bool:
        return self._provider() is not None

    # ── Image understanding ───────────────────────────────

    async def describe_image(
        self,
        data_uri: str,
        prompt: str = DEFAULT_IMAGE_PROMPT,
        mime: str = "image/png",
        *,
        max_tokens: int = 800,
    ) -> str:
        """Return a textual description, or ``""`` on failure."""
        llm = self._provider()
        if llm is None:
            return ""
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": data_uri},
                "media_type": mime or "image/png",
            },
        ]
        try:
            response = await llm.agenerate(
                [Message(role=MessageRole.USER, content=content)],
                temperature=0.2,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - degrade to text note
            self.last_error = f"Vision analysis failed: {exc}"
            return ""
        return (response.content or "").strip()

    # ── Video keyframes ───────────────────────────────────

    def extract_video_frames(
        self,
        path: str | Path,
        max_frames: int | None = None,
    ) -> list[dict[str, Any]]:
        """Extract evenly spaced JPEG keyframes as data URIs.

        Returns dictionaries with ``index``, ``timestamp`` and ``data_uri``.
        An empty list plus ``last_error`` means OpenCV is unavailable or the
        file could not be decoded.
        """
        self.last_error = ""
        try:
            import cv2
        except ImportError:
            self.last_error = (
                "OpenCV is not installed; video frames cannot be extracted."
            )
            return []

        if max_frames is None:
            from agentic_rag.config.settings import get_settings
            max_frames = get_settings().media.max_video_frames
        max_frames = max(1, int(max_frames))

        capture = cv2.VideoCapture(str(path))
        frames: list[dict[str, Any]] = []
        try:
            if not capture.isOpened():
                self.last_error = "Could not open the video file."
                return []

            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 0:
                fps = 25.0

            if total > 0:
                count = min(max_frames, total)
                if count == 1:
                    positions = [0]
                else:
                    positions = [
                        round(i * (total - 1) / (count - 1))
                        for i in range(count)
                    ]
                for position in positions:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
                    ok, frame = capture.read()
                    if not ok:
                        continue
                    encoded = self._encode_frame(cv2, frame)
                    if encoded is None:
                        continue
                    frames.append({
                        "index": int(position),
                        "timestamp": round(position / fps, 2),
                        "data_uri": encoded,
                    })
            else:
                # Some streams do not report a frame count.  Sample the first
                # N decodable frames rather than scanning the whole file.
                index = 0
                while len(frames) < max_frames:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if index % 30 == 0:
                        encoded = self._encode_frame(cv2, frame)
                        if encoded is not None:
                            frames.append({
                                "index": index,
                                "timestamp": round(index / fps, 2),
                                "data_uri": encoded,
                            })
                    index += 1
        finally:
            capture.release()

        if not frames and not self.last_error:
            self.last_error = "Could not extract frames from the video."
        return frames

    @staticmethod
    def _encode_frame(cv2, frame) -> str | None:
        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )
        if not ok:
            return None
        encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
