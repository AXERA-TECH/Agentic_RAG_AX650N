"""Video processing — keyframe extraction and analysis."""

import base64
from pathlib import Path
from typing import Optional


class VideoProcessor:
    """Extract keyframes from video and describe them using Vision LLM."""

    def __init__(self, llm=None):
        """llm must be a vision-capable BaseLLMProvider."""
        self.llm = llm

    async def analyze(self, video_path: str | Path, max_frames: int = 5,
                       question: str = "Describe what is happening in this video frame.") -> str:
        """Extract keyframes from a video and describe each frame.

        Args:
            video_path: Path to video file.
            max_frames: Maximum number of keyframes to extract.
            question: Question to ask about each frame.

        Returns:
            Combined description of all keyframes.
        """
        path = Path(video_path)
        if not path.exists():
            return f"[Video not found: {video_path}]"

        try:
            import cv2
        except ImportError:
            return "[OpenCV not installed. Run: pip install opencv-python]"

        cap = cv2.VideoCapture(str(path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        frames_b64 = self._extract_keyframes(cap, total_frames, max_frames)
        cap.release()

        if not frames_b64:
            return "[No frames extracted from video]"

        results = [f"Video Info: {duration:.1f}s, {total_frames} frames, {fps:.1f} fps"]

        if self.llm and self.llm.supports_vision:
            from agentic_rag.data.models import Message

            for i, frame_b64 in enumerate(frames_b64):
                content = [
                    {"type": "text", "text": f"Frame {i+1}/{len(frames_b64)}: {question}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                ]
                response = await self.llm.agenerate([Message(role="user", content=content)])
                results.append(f"Frame {i+1}: {response.content[:300]}")
        else:
            results.append(f"[{len(frames_b64)} keyframes extracted but no Vision LLM configured for analysis]")

        return "\n\n".join(results)

    def _extract_keyframes(self, cap, total_frames: int, max_frames: int) -> list[str]:
        """Extract evenly-spaced keyframes as base64 strings."""
        import cv2

        frames = []
        interval = max(1, total_frames // max_frames)

        for i in range(0, total_frames, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                _, buffer = cv2.imencode(".jpg", frame)
                frames.append(base64.b64encode(buffer).decode())
            if len(frames) >= max_frames:
                break

        return frames
