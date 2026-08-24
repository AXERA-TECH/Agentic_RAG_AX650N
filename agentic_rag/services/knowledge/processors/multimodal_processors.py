"""Table, Equation, Video, and Audio modal processors."""

import base64
from pathlib import Path
from typing import Callable, Optional

from agentic_rag.services.knowledge.content_list import ContentItem, ContentType
from agentic_rag.services.knowledge.processors.base import BaseModalProcessor


class TableModalProcessor(BaseModalProcessor):
    """Interpret structured table data."""

    content_type = ContentType.TABLE
    description = "Parse and describe tabular data."

    DEFAULT_PROMPT = None  # set in __init__ from Prompts

    def __init__(self, llm_func: Optional[Callable] = None):
        from agentic_rag.config.prompts import Prompts
        self.llm_func = llm_func
        if TableModalProcessor.DEFAULT_PROMPT is None:
            TableModalProcessor.DEFAULT_PROMPT = Prompts.TABLE_ANALYSIS

    async def process(self, item: ContentItem) -> ContentItem:
        if item.type != ContentType.TABLE or not item.table_body:
            return item

        if self.llm_func:
            try:
                prompt = self.DEFAULT_PROMPT.format(
                    table_content=item.table_body[:3000]
                )
                caption = await self.llm_func(prompt)
                item.table_caption = caption
            except Exception as e:
                item.metadata["table_error"] = str(e)

        if not item.text:
            item.text = item.table_body
        return item


class EquationModalProcessor(BaseModalProcessor):
    """Parse LaTeX equations and create conceptual mappings."""

    content_type = ContentType.EQUATION
    description = "Parse LaTeX formulas and generate plain-text descriptions."

    def __init__(self, llm_func: Optional[Callable] = None):
        self.llm_func = llm_func

    async def process(self, item: ContentItem) -> ContentItem:
        if item.type != ContentType.EQUATION or not item.latex:
            return item

        # Store LaTeX as text for embedding
        item.text = f"[LaTeX Formula]: {item.latex}"

        if self.llm_func:
            try:
                from agentic_rag.config.prompts import Prompts
                prompt = Prompts.LATEX_TO_PLAIN_TEXT.format(latex=item.latex)
                description = await self.llm_func(prompt)
                item.text += f"\nDescription: {description}"
            except Exception as e:
                item.metadata["equation_error"] = str(e)

        return item


class VideoModalProcessor(BaseModalProcessor):
    """Extract keyframes from video and generate descriptions using Vision LLM."""

    content_type = ContentType.VIDEO
    description = "Extract keyframes and describe video content."

    def __init__(
        self,
        vision_func: Optional[Callable] = None,
        max_frames: int = 5,
    ):
        self.vision_func = vision_func
        self.max_frames = max_frames

    async def process(self, item: ContentItem) -> ContentItem:
        if item.type != ContentType.VIDEO or not item.video_path:
            return item

        path = Path(item.video_path)
        if not path.exists():
            item.metadata["video_error"] = f"File not found: {item.video_path}"
            return item

        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            # Extract keyframes
            interval = max(1, total_frames // self.max_frames)
            frames = []
            for i in range(0, total_frames, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    _, buffer = cv2.imencode(".jpg", frame)
                    frames.append(base64.b64encode(buffer).decode())
                if len(frames) >= self.max_frames:
                    break
            cap.release()

            # Describe frames
            descriptions = []
            for i, frame_b64 in enumerate(frames):
                if self.vision_func:
                    from agentic_rag.config.prompts import Prompts
                    desc = await self.vision_func(
                        f"data:image/jpeg;base64,{frame_b64}",
                        Prompts.VIDEO_KEYFRAME_DESCRIPTION.format(frame_num=i + 1),
                    )
                    descriptions.append(f"Frame {i+1}: {desc}")
                else:
                    descriptions.append(f"[Keyframe {i+1}]")

            item.video_caption = (
                f"Video duration: {duration:.1f}s, {total_frames} frames\n" +
                "\n".join(descriptions)
            )
            item.text = item.video_caption

        except ImportError:
            item.metadata["video_error"] = "opencv-python not installed"
        except Exception as e:
            item.metadata["video_error"] = str(e)

        return item


class AudioModalProcessor(BaseModalProcessor):
    """Transcribe audio content to text."""

    content_type = ContentType.AUDIO
    description = "Transcribe audio using Whisper."

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None

    async def process(self, item: ContentItem) -> ContentItem:
        if item.type != ContentType.AUDIO or not item.audio_path:
            return item

        path = Path(item.audio_path)
        if not path.exists():
            item.metadata["audio_error"] = f"File not found: {item.audio_path}"
            return item

        try:
            from agentic_rag.core.voice.stt import STTService
            from agentic_rag.config.settings import get_settings
            s = get_settings().voice
            stt = STTService(
                provider=s.stt_provider,
                model=s.stt_model or self.model_name,
                api_base=s.stt_api_base,
                api_key=s.stt_api_key,
                language=s.stt_language,
                sample_rate=s.sample_rate,
            )
            text = await stt.transcribe_file(str(path))
            if text:
                item.audio_transcript = text
                item.text = text
            else:
                item.metadata["audio_error"] = "No speech detected"
        except ImportError:
            item.metadata["audio_error"] = "STT service unavailable"
        except Exception as e:
            item.metadata["audio_error"] = str(e)

        return item
