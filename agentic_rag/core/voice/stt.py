"""Speech-to-Text service.

Primary: SenseVoice-compatible local ASR server (/asr endpoint).
Fallback: OpenAI Whisper API or local Whisper model.
"""

import io
import re
import tempfile
from pathlib import Path

import numpy as np


class STTService:
    """Speech-to-Text via ASR server (SenseVoice) with Whisper fallback."""

    def __init__(
        self,
        provider: str = "sensevoice",
        model: str = "sensevoice",
        api_base: str = "http://localhost:8000",
        api_key: str = "",
        language: str = "auto",
        sample_rate: int = 16000,
    ):
        self.provider = provider          # "sensevoice" | "whisper" | "openai"
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.language = language           # auto | zh | en | ja | ko | yue
        self.sample_rate = sample_rate
        self._local_model = None

    # ── Public API ──────────────────────────────────────────

    async def transcribe_file(self, file_path: str | Path) -> str:
        """Transcribe an audio file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        if self.provider == "sensevoice":
            return await self._transcribe_sensevoice_file(path)
        elif self.provider == "openai":
            return await self._transcribe_openai(path)
        else:
            return await self._transcribe_local(path)

    async def transcribe_bytes(
        self, audio_data: bytes, language: str | None = None
    ) -> str:
        """Transcribe raw audio bytes (any format ffmpeg supports)."""
        lang = language or self.language

        if self.provider == "sensevoice":
            waveform = self._decode_to_float32(audio_data)
            if waveform is not None:
                return await self._call_asr(waveform, lang)
            return ""

        # Whisper fallback — write to temp WAV
        suffix = ".wav" if audio_data[:4] == b"RIFF" else ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_data)
            return await self.transcribe_file(f.name)

    async def transcribe_ndarray(
        self, audio: np.ndarray, language: str | None = None
    ) -> str:
        """Transcribe a numpy array (float32 or int16)."""
        lang = language or self.language

        if self.provider == "sensevoice":
            waveform = audio.astype(np.float32)
            if waveform.max() > 1.5:
                waveform = waveform / 32768.0
            return await self._call_asr(waveform, lang)

        # Whisper fallback
        if audio.dtype != np.int16:
            audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        import wave
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            with wave.open(f, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio.tobytes())
            return await self.transcribe_file(f.name)

    # ── SenseVoice ASR server ───────────────────────────────

    async def _transcribe_sensevoice_file(self, path: Path) -> str:
        """Transcribe via SenseVoice ASR server."""
        import librosa
        waveform, sr = librosa.load(str(path), sr=self.sample_rate, mono=True)
        return await self._call_asr(waveform.astype(np.float32), self.language)

    async def _call_asr(
        self, waveform: np.ndarray, language: str | None = None
    ) -> str:
        """Send audio to ASR server.

        Primary: POST /v1/audio/transcriptions (OpenAI-compatible, openai_server.py).
        Fallback: POST /asr (float32 JSON, legacy server.py).
        """
        lang = language or self.language

        # 1) OpenAI-compatible (standard)
        result = await self._try_openai_asr(waveform, lang)
        if result:
            return result

        # 2) Legacy /asr fallback
        result = await self._try_native_asr(waveform, lang)
        return result or ""

    async def _try_native_asr(
        self, waveform: np.ndarray, lang: str
    ) -> str | None:
        """Try POST /asr with float32 JSON (server.py). Returns None on 404."""
        import aiohttp

        asr_url = f"{self.api_base}/asr"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    asr_url,
                    json={
                        "audio_data": waveform.tolist(),
                        "sample_rate": self.sample_rate,
                        "language": lang,
                    },
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status == 404:
                        return None  # Signal fallback
                    if resp.status != 200:
                        text = await resp.text()
                        print(f"  [STT] ASR error {resp.status}: {text[:200]}", flush=True)
                        return ""
                    result = await resp.json()
                    text = result.get("text", "")
                    return self._clean_sensevoice(text)
        except Exception as e:
            print(f"  [STT] ASR connection error: {e}", flush=True)
            return ""
        return None

    async def _try_openai_asr(
        self, waveform: np.ndarray, lang: str
    ) -> str:
        """Try POST /v1/audio/transcriptions with WAV file upload (openai_server.py)."""
        import aiohttp
        import io
        import soundfile as sf

        transcribe_url = f"{self.api_base}/v1/audio/transcriptions"
        try:
            # Write float32 waveform to WAV in memory
            wav_buf = io.BytesIO()
            sf.write(wav_buf, waveform.astype(np.float32), self.sample_rate, format="WAV")
            wav_buf.seek(0)

            form = aiohttp.FormData()
            form.add_field("file", wav_buf.read(),
                           filename="audio.wav",
                           content_type="audio/wav")
            form.add_field("language", lang)
            form.add_field("response_format", "json")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    transcribe_url,
                    data=form,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        print(f"  [STT] OpenAI ASR error {resp.status}: {text[:200]}", flush=True)
                        return ""
                    result = await resp.json()
                    text = result.get("text", "")
                    return self._clean_sensevoice(text)
        except Exception as e:
            print(f"  [STT] OpenAI ASR error: {e}", flush=True)
            return ""

    @staticmethod
    def _clean_sensevoice(text: str) -> str:
        """Remove SenseVoice special tokens like <|zh|>, <|emotion|>."""
        text = re.sub(r'<\|[^|]*\|>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # ── Local Whisper ───────────────────────────────────────

    async def _transcribe_local(self, path: Path) -> str:
        """Transcribe using local Whisper model."""
        if self._local_model is None:
            import whisper
            self._local_model = whisper.load_model(self.model or "base")
        result = self._local_model.transcribe(str(path))
        return result["text"].strip()

    # ── OpenAI Whisper API ──────────────────────────────────

    async def _transcribe_openai(self, path: Path) -> str:
        """Transcribe using OpenAI Whisper API."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.api_key or "not-needed",
            base_url=self.api_base or "https://api.openai.com/v1",
        )
        with open(path, "rb") as audio_file:
            transcript = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
        return transcript.text.strip()

    # ── Audio decoding ──────────────────────────────────────

    @staticmethod
    def _decode_to_float32(audio_data: bytes) -> np.ndarray | None:
        """Decode audio bytes to float32 numpy array in [-1, 1]."""
        # WAV → soundfile
        if audio_data[:4] == b"RIFF":
            try:
                import soundfile as sf
                wav_buf = io.BytesIO(audio_data)
                waveform, _ = sf.read(wav_buf, dtype="float32")
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=1)
                return waveform.astype(np.float32)
            except Exception:
                pass

        # Compressed → pydub (ffmpeg)
        try:
            from pydub import AudioSegment
            import soundfile as sf

            suffix = _detect_suffix(audio_data)
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            try:
                audio = AudioSegment.from_file(tmp_path)
                audio = audio.set_channels(1).set_frame_rate(16000)
                wav_buf = io.BytesIO()
                audio.export(wav_buf, format="wav")
                wav_buf.seek(0)
                waveform, _ = sf.read(wav_buf, dtype="float32")
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=1)
                return waveform.astype(np.float32)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        except ImportError:
            pass
        except Exception:
            pass

        # Raw int16 PCM (last resort)
        try:
            arr = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            return arr / 32768.0
        except Exception:
            return None


def _detect_suffix(data: bytes) -> str:
    """Guess file extension from magic bytes."""
    if data[:4] == b"RIFF":
        return ".wav"
    if data[:3] == b"ID3":
        return ".mp3"
    if data[:4] == b"fLaC":
        return ".flac"
    if data[:4] == b"OggS":
        return ".ogg"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    return ".webm"
