"""Speech-to-text for uploaded chat audio.

The knowledge-base ingestion pipeline already has an :class:`STTService`; this
wrapper only adapts it to the media context layer and keeps configuration in
one place.  Failures are reported through ``last_error`` so callers can
degrade to an explicit text note instead of breaking the whole turn.
"""

from __future__ import annotations

from pathlib import Path


class TranscriptionService:
    """Transcribe audio attachments using the configured ASR backend."""

    def __init__(self, stt=None):
        self._stt = stt
        self.last_error = ""

    def _service(self):
        if self._stt is not None:
            return self._stt
        from agentic_rag.config.settings import get_settings
        from agentic_rag.services.media.voice.stt import STTService

        voice = get_settings().voice
        self._stt = STTService(
            provider=voice.stt_provider,
            model=voice.stt_model,
            api_base=voice.stt_api_base,
            api_key=voice.stt_api_key,
            language=voice.stt_language,
            sample_rate=voice.sample_rate,
        )
        return self._stt

    async def transcribe(self, path: str | Path) -> str:
        """Return the transcript, or ``""`` when ASR is unavailable."""
        self.last_error = ""
        try:
            text = await self._service().transcribe_file(str(path))
        except ImportError as exc:
            self.last_error = f"STT dependencies unavailable: {exc}"
            return ""
        except Exception as exc:  # noqa: BLE001 - degraded media handling
            self.last_error = f"Transcription failed: {exc}"
            return ""
        return (text or "").strip()
