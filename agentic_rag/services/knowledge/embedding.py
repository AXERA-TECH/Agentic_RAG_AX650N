"""Multimodal embedding input types and adapter.

Decouples the embedding interface from plain text strings so that images
(and other non-text modalities) can be embedded directly via CLIP-like models
or at minimum not silently dropped when they lack captions.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Callable, Optional, Union


# ═══════════════════════════════════════════════════════════════
# Embedding Input
# ═══════════════════════════════════════════════════════════════

@dataclass
class EmbeddingInput:
    """Describes what should be embedded for a single content item.

    For text-only embedding models, only ``text`` is used.
    For multimodal models (CLIP, ImageBind, etc.), the media fields carry raw data
    that can be embedded directly — pixels for images/video, waveforms for audio.
    """

    text: str = ""

    # Image / visual
    image_path: str = ""
    image_bytes: Optional[bytes] = None
    image_url: str = ""

    # Video
    video_path: str = ""

    # Audio
    audio_path: str = ""
    audio_bytes: Optional[bytes] = None

    # ── Factory methods ──────────────────────────

    @classmethod
    def from_text(cls, text: str) -> "EmbeddingInput":
        return cls(text=text)

    @classmethod
    def from_image_path(cls, image_path: str, caption: str = "") -> "EmbeddingInput":
        return cls(text=caption, image_path=image_path)

    @classmethod
    def from_video_path(cls, video_path: str, caption: str = "") -> "EmbeddingInput":
        return cls(text=caption, video_path=video_path)

    @classmethod
    def from_audio_path(cls, audio_path: str, transcript: str = "") -> "EmbeddingInput":
        return cls(text=transcript, audio_path=audio_path)

    # ── Properties ───────────────────────────────

    @property
    def has_image(self) -> bool:
        """Whether this input carries image data."""
        return bool(self.image_path or self.image_bytes or self.image_url)

    @property
    def has_video(self) -> bool:
        """Whether this input carries video data."""
        return bool(self.video_path)

    @property
    def has_audio(self) -> bool:
        """Whether this input carries audio data."""
        return bool(self.audio_path or self.audio_bytes)

    @property
    def has_media(self) -> bool:
        """Whether this input carries ANY non-text media."""
        return self.has_image or self.has_video or self.has_audio

    @property
    def is_embeddable(self) -> bool:
        """An input is embeddable if it has text OR any media.

        Caption-less images, untranscribed audio, and undescribed video are
        no longer silently dropped — text-only models get a placeholder,
        multimodal models encode the raw media directly.
        """
        return bool(self.text) or self.has_media


# ═══════════════════════════════════════════════════════════════
# Embedding Function Signatures
# ═══════════════════════════════════════════════════════════════

# Legacy text-only: (text: str) -> list[float]
TextEmbeddingFunc = Callable[[str], list[float]]

# Multimodal batch: (inputs: list[EmbeddingInput]) -> list[list[float]]
MultimodalEmbeddingFunc = Callable[[list[EmbeddingInput]], list[list[float]]]

# Either signature is accepted
EmbeddingFunc = Union[TextEmbeddingFunc, MultimodalEmbeddingFunc]


# ═══════════════════════════════════════════════════════════════
# Embedding Adapter
# ═══════════════════════════════════════════════════════════════

class EmbeddingAdapter:
    """Wraps an embedding function and normalizes it to a multimodal interface.

    Detects whether the function is **text-only** ``(str) -> list[float]`` or
    **multimodal** ``(list[EmbeddingInput]) -> list[list[float]]`` and adapts
    accordingly.

    For text-only functions, images are embedded via their caption text (with a
    ``"[Image at ...]"`` fallback when no caption exists).

    For multimodal functions, inputs are passed directly so the model can encode
    visual features from pixels.
    """

    def __init__(self, func: EmbeddingFunc):
        self._func = func
        self._is_multimodal = self._detect_multimodal(func)

    # ── Public API ───────────────────────────────

    async def embed(self, inputs: list[EmbeddingInput], prompt_name: str = "document") -> list[list[float]]:
        """Embed a batch of multimodal inputs.

        Returns a list of vectors in the same order as *inputs*.
        """
        if not inputs:
            return []

        if self._is_multimodal:
            return await self._call_multimodal(inputs, prompt_name=prompt_name)
        else:
            return await self._call_text_only(inputs)

    async def embed_query(self, query: EmbeddingInput) -> list[float]:
        """Embed a single query input.  Returns one vector.

        For multimodal models this passes prompt_name="query" so the
        model can apply query-specific processing (different from document indexing).
        """
        results = await self.embed([query], prompt_name="query")
        return results[0]

    # ── Detection ────────────────────────────────

    @staticmethod
    def _detect_multimodal(func: EmbeddingFunc) -> bool:
        """Heuristic: inspect the first parameter name.

        Text-only functions typically name it ``text``.
        Multimodal functions use ``inputs``, ``items``, or ``embedding_inputs``.
        """
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            if params:
                first = params[0]
                return first in ("inputs", "items", "embedding_inputs")
        except (ValueError, TypeError):
            pass
        return False

    # ── Internal dispatch ────────────────────────

    async def _call_text_only(
        self, inputs: list[EmbeddingInput]
    ) -> list[list[float]]:
        """For text-only models: embed the ``text`` field of each input.

        Media items without text descriptions get a placeholder so they are
        not silently dropped.
        """
        texts: list[str] = []
        for inp in inputs:
            if inp.text:
                texts.append(inp.text)
            elif inp.has_image:
                texts.append(
                    f"[Image at {inp.image_path or inp.image_url or 'unknown'}]"
                )
            elif inp.has_video:
                texts.append(f"[Video at {inp.video_path or 'unknown'}]")
            elif inp.has_audio:
                texts.append(f"[Audio at {inp.audio_path or 'unknown'}]")
            else:
                texts.append("")  # edge case — will produce a near-zero embedding

        embeddings: list[list[float]] = []
        for t in texts:
            result = self._func(t)
            if asyncio.iscoroutine(result):
                emb = await result
            else:
                emb = result
            embeddings.append(emb if isinstance(emb, list) else list(emb))
        return embeddings

    async def _call_multimodal(
        self, inputs: list[EmbeddingInput], prompt_name: str = "document",
    ) -> list[list[float]]:
        """For multimodal models: pass inputs with prompt_name.

        ``prompt_name="query"`` for retrieval queries,
        ``prompt_name="document"`` for indexed documents (default).
        """
        # Check if the underlying function accepts prompt_name
        try:
            sig = inspect.signature(self._func)
            if "prompt_name" in sig.parameters:
                result = self._func(inputs, prompt_name=prompt_name)
            else:
                result = self._func(inputs)
        except (ValueError, TypeError):
            result = self._func(inputs)

        if asyncio.iscoroutine(result):
            embeddings = await result
        else:
            embeddings = result
        return [list(e) for e in embeddings]
