"""Text-to-Speech service.

Primary: Qwen3-TTS via OpenAI-compatible /v1/audio/speech API.
  - VoiceDesign: generate voice from text description (instructions)
  - CustomVoice: use predefined speaker names
  - Base: voice cloning from reference audio
Fallback: Edge-TTS (free, good quality for Chinese).
"""

import asyncio
import io
import tempfile
import time
from pathlib import Path


class TTSService:
    """Text-to-Speech via Qwen3-TTS / Kokoro with Edge-TTS fallback."""

    def __init__(
        self,
        provider: str = "qwen",
        model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        api_base: str = "http://0.0.0.0:8091",
        api_key: str = "EMPTY",
        task_type: str = "VoiceDesign",
        instructions: str = "A clear, professional voice in Chinese",
        language: str = "Chinese",
        speaker: str = "",
        voice: str = "zh-CN-XiaoxiaoNeural",  # Edge-TTS fallback / kokoro voice
        speed: float = 1.0,                    # kokoro playback speed
        response_format: str = "wav",
        max_new_tokens: int = 0,
    ):
        self.provider = provider          # "qwen" | "kokoro" | "edge" | "openai"
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.task_type = task_type        # VoiceDesign | CustomVoice | Base (qwen)
        self.instructions = instructions  # Voice description (qwen)
        self.language = language           # Chinese | zh | en | ja (qwen/kokoro)
        self.speaker = speaker            # Speaker name (qwen CustomVoice)
        self.voice = voice                # Edge-TTS / kokoro voice name
        self.speed = speed                # kokoro speed
        self.response_format = response_format  # wav | mp3 | flac | pcm
        self.max_new_tokens = max_new_tokens

    # ── Text cleaning ───────────────────────────────────────

    @staticmethod
    def _clean_for_tts(text: str) -> str:
        """Strip Markdown, URLs, emoji, and formatting for TTS."""
        import re

        # Citations remain visible in chat, but should not be read aloud.
        text = re.split(
            r'(?:^|\n)\s*(?:📚\s*)?参考来源\s*[:：]?',
            text,
            maxsplit=1,
        )[0]
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove Markdown images/links: ![alt](url) and [text](url)
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'\[([^\]]*)\]\(.*?\)', r'\1', text)
        # Remove Markdown formatting markers
        text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)  # **bold**, *italic*
        text = re.sub(r'#{1,6}\s*', '', text)                   # headers
        text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text)          # inline code / code blocks
        text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)  # list markers
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)  # numbered lists
        # Remove reference markers like [R1], [R3]
        text = re.sub(r'\[R\d+\]', '', text)
        # Remove emoji and other non-speech symbols
        text = re.sub(r'[^一-鿿　-〿＀-￯a-zA-Z0-9\s.,!?;:，。！？；：、""\'\'（）【】《》…\-\']+', ' ', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove leading/trailing punctuation-only lines
        text = re.sub(r'^[,.!?;:，。！？；：、\s]+$', '', text, flags=re.MULTILINE)
        return text.strip()

    # ── Public API ──────────────────────────────────────────

    async def synthesize(self, text: str) -> bytes:
        """Convert text to speech, returning raw audio bytes."""
        if not text.strip():
            return b""

        # Clean text before TTS (strip Markdown/URLs/emoji)
        clean_text = self._clean_for_tts(text)
        if not clean_text:
            return b""

        if self.provider == "qwen":
            result = await self._synthesize_qwen(clean_text)
            if result:
                return result
            print(f"  [TTS] Qwen3-TTS failed, falling back to Edge-TTS", flush=True)
            return await self._synthesize_edge(clean_text)
        elif self.provider == "kokoro":
            result = await self._synthesize_kokoro(clean_text)
            if result:
                return result
            print(f"  [TTS] Kokoro failed, falling back to Edge-TTS", flush=True)
            return await self._synthesize_edge(clean_text)
        elif self.provider == "openai":
            return await self._synthesize_openai(text)
        else:
            return await self._synthesize_edge(text)

    async def synthesize_to_file(
        self, text: str, output_path: str | Path
    ) -> Path:
        """Synthesize speech and save to a file."""
        audio_bytes = await self.synthesize(text)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio_bytes)
        return path

    # ── Qwen3-TTS (VoiceDesign) ─────────────────────────────

    async def _synthesize_qwen(self, text: str) -> bytes:
        """Synthesize via Qwen3-TTS OpenAI-compatible /v1/audio/speech."""
        import aiohttp

        payload = {
            "model": self.model,
            "input": text,
            "response_format": self.response_format,
        }
        if self.task_type:
            payload["task_type"] = self.task_type
        if self.speaker:
            payload["voice"] = self.speaker
        if self.instructions:
            payload["instructions"] = self.instructions
        if self.language:
            payload["language"] = self.language
        if self.max_new_tokens > 0:
            payload["max_new_tokens"] = self.max_new_tokens

        api_url = f"{self.api_base}/v1/audio/speech"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        api_url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=120),
                    ) as resp:
                        if resp.status != 200:
                            err = await _read_error(resp)
                            print(f"  [TTS] Qwen error (attempt {attempt+1}/3): "
                                  f"{resp.status} {err[:100]}", flush=True)
                            if attempt < 2:
                                await asyncio.sleep(2 ** attempt)
                            continue
                        return await resp.read()
            except Exception as e:
                print(f"  [TTS] Qwen request error (attempt {attempt+1}/3): {e}",
                      flush=True)
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

        return b""

    # ── Edge-TTS (fallback) ─────────────────────────────────

    async def _synthesize_edge(self, text: str) -> bytes:
        """Synthesize using Edge-TTS (free, good Chinese quality)."""
        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, self.voice)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_path = f.name
            await communicate.save(temp_path)
            data = Path(temp_path).read_bytes()
            Path(temp_path).unlink(missing_ok=True)
            return data
        except ImportError:
            print(f"  [TTS] edge-tts not installed", flush=True)
            return b""
        except Exception as e:
            print(f"  [TTS] Edge-TTS error: {e}", flush=True)
            return b""

    # ── Kokoro TTS (Axera NPU) ────────────────────────────

    async def _synthesize_kokoro(self, text: str) -> bytes:
        """Synthesize via Kokoro TTS server (POST /tts → WAV bytes).

        API: POST /tts with form/JSON body
          text/sentence: text to speak
          language/lang: zh | en | ja
          voice: voice name (e.g., zf_xiaoyi, zf_xiaoxiao)
          speed: playback speed (default 1.0)
        Returns: WAV audio bytes
        """
        import aiohttp

        # Map internal language codes to Kokoro codes
        lang_map = {
            "zh": "zh", "chinese": "zh",
            "en": "en", "english": "en",
            "ja": "ja", "japanese": "ja",
            "auto": "zh",
        }
        kokoro_lang = lang_map.get(
            (self.language or "zh").lower(), "zh"
        )

        payload = {
            "text": text,
            "language": kokoro_lang,
            "voice": self.voice or "zf_xiaoyi",
            "speed": str(self.speed),
        }

        api_url = f"{self.api_base}/tts"

        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        api_url,
                        data=payload,
                        timeout=aiohttp.ClientTimeout(total=120),
                    ) as resp:
                        if resp.status != 200:
                            err_body = await resp.text()
                            print(
                                f"  [TTS] Kokoro error (attempt {attempt+1}/3): "
                                f"{resp.status} {err_body[:200]}", flush=True,
                            )
                            if attempt < 2:
                                await asyncio.sleep(2 ** attempt)
                            continue
                        return await resp.read()
            except Exception as e:
                print(
                    f"  [TTS] Kokoro request error (attempt {attempt+1}/3): {e}",
                    flush=True,
                )
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

        return b""

    # ── OpenAI TTS ──────────────────────────────────────────

    async def _synthesize_openai(self, text: str) -> bytes:
        """Synthesize using OpenAI TTS API."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.api_key or "not-needed",
            base_url=self.api_base or "https://api.openai.com/v1",
        )
        response = await client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
        )
        return response.content


async def _read_error(resp) -> str:
    """Extract error detail from a failed TTS response."""
    try:
        data = await resp.json()
        return str(data.get("detail", data.get("error", str(data))))[:200]
    except Exception:
        try:
            return (await resp.text())[:200]
        except Exception:
            return str(resp.status)
