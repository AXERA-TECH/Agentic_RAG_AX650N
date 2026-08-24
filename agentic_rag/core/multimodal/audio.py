"""Audio transcription using Whisper."""

from pathlib import Path
from typing import Optional


class AudioProcessor:
    """Transcribe audio to text using Whisper."""

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy-load the Whisper model."""
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self.model_name)
        return self._model

    async def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file (mp3, wav, m4a, etc.)

        Returns:
            Transcribed text.
        """
        path = Path(audio_path)
        if not path.exists():
            return f"[Audio not found: {audio_path}]"

        try:
            model = self._load_model()
            result = model.transcribe(str(path))
            return result["text"]
        except ImportError:
            return "[Whisper not installed. Run: pip install openai-whisper]"
        except Exception as e:
            return f"[Transcription error: {e}]"

    async def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribe raw audio bytes."""
        import tempfile
        import wave

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                with wave.open(f, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_bytes)
                f.flush()
                return await self.transcribe(f.name)
        except Exception as e:
            return f"[Transcription error: {e}]"
