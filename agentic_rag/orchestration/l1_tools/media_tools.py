"""Multimodal tools — image understanding, video analysis, audio transcription."""

import base64
from pathlib import Path
from typing import Any

from agentic_rag.orchestration.l1_tools.base import BaseTool


class ImageUnderstandTool(BaseTool):
    """Analyze an image using a Vision LLM."""

    name = "image_understand"
    description = "Understand and describe the content of an image. Supports file paths and base64."
    parameters_schema = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the image file or base64-encoded image data",
            },
            "question": {
                "type": "string",
                "description": "Optional question about the image",
                "default": "Describe this image in detail.",
            },
        },
        "required": ["image_path"],
    }

    def __init__(self, llm=None):
        self.llm = llm  # Must be a vision-capable LLM

    async def execute(self, image_path: str, question: str = "Describe this image in detail.") -> Any:
        if self.llm is None:
            return "Vision LLM is not configured."

        # Build multimodal content
        if image_path.startswith(("data:", "http")):
            image_url = image_path
        elif image_path.startswith(("/", "./", "../")) or Path(image_path).exists():
            # Read file and convert to base64
            path = Path(image_path)
            if not path.exists():
                return f"Image file not found: {image_path}"
            ext = path.suffix.lower()
            mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".png": "image/png", ".gif": "image/gif",
                        ".webp": "image/webp", ".bmp": "image/bmp"}
            mime = mime_map.get(ext, "image/png")
            data = base64.b64encode(path.read_bytes()).decode()
            image_url = f"data:{mime};base64,{data}"
        else:
            return f"Cannot process image: {image_path}"

        content = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

        from agentic_rag.data.models import Message
        messages = [Message(role="user", content=content)]

        try:
            response = await self.llm.agenerate(messages)
            return response.content
        except Exception as e:
            return f"Image analysis failed: {str(e)}"


class AudioTranscribeTool(BaseTool):
    """Transcribe audio using Whisper."""

    name = "audio_transcribe"
    description = "Transcribe audio content to text. Supports file paths."
    parameters_schema = {
        "type": "object",
        "properties": {
            "audio_path": {
                "type": "string",
                "description": "Path to the audio file",
            },
        },
        "required": ["audio_path"],
    }

    async def execute(self, audio_path: str) -> Any:
        try:
            from agentic_rag.core.voice.stt import STTService
            from agentic_rag.config.settings import get_settings
            s = get_settings().voice
            stt = STTService(
                provider=s.stt_provider,
                model=s.stt_model,
                api_base=s.stt_api_base,
                api_key=s.stt_api_key,
                language=s.stt_language,
                sample_rate=s.sample_rate,
            )
            text = await stt.transcribe_file(audio_path)
            return text if text else "(no speech detected)"
        except ImportError:
            return "STT service unavailable. Check voice configuration."
        except Exception as e:
            return f"Audio transcription failed: {str(e)}"


class VideoAnalyzeTool(BaseTool):
    """Extract keyframes from video and analyze them."""

    name = "video_analyze"
    description = "Extract keyframes from a video and describe each frame."
    parameters_schema = {
        "type": "object",
        "properties": {
            "video_path": {
                "type": "string",
                "description": "Path to the video file",
            },
            "max_frames": {
                "type": "integer",
                "description": "Maximum keyframes to extract",
                "default": 5,
            },
        },
        "required": ["video_path"],
    }

    def __init__(self, llm=None):
        self.llm = llm

    async def execute(self, video_path: str, max_frames: int = 5) -> Any:
        if self.llm is None:
            return "Vision LLM is not configured for video analysis."

        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            # Extract evenly spaced keyframes
            interval = max(1, total_frames // max_frames)
            frames = []
            for i in range(0, total_frames, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    _, buffer = cv2.imencode(".jpg", frame)
                    b64 = base64.b64encode(buffer).decode()
                    frames.append(f"data:image/jpeg;base64,{b64}")
                if len(frames) >= max_frames:
                    break
            cap.release()

            if not frames:
                return "Could not extract frames from video."

            # Analyze each frame
            results = [f"Video: {duration:.1f}s, {total_frames} frames, {fps:.1f} fps"]
            for i, frame_b64 in enumerate(frames):
                content = [
                    {"type": "text", "text": f"Describe keyframe {i+1} of this video briefly."},
                    {"type": "image_url", "image_url": {"url": frame_b64}},
                ]
                from agentic_rag.data.models import Message
                response = await self.llm.agenerate([Message(role="user", content=content)])
                results.append(f"Keyframe {i+1}: {response.content[:300]}")

            return "\n\n".join(results)

        except ImportError:
            return "OpenCV is not installed. Install with: pip install opencv-python"
        except Exception as e:
            return f"Video analysis failed: {str(e)}"
