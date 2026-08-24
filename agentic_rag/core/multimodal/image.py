"""Image understanding via Vision LLM."""

import base64
from pathlib import Path
from typing import Optional

from agentic_rag.data.models import Message


class ImageProcessor:
    """Process and understand images using Vision LLM."""

    MIME_MAP = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp", ".bmp": "image/bmp",
    }

    def __init__(self, llm=None):
        """llm must be a vision-capable BaseLLMProvider."""
        self.llm = llm

    async def describe(self, image: str | Path, question: str = "Describe this image in detail.") -> str:
        """Analyze an image and return a description.

        Args:
            image: Path to image file, URL, or base64 data URI.
            question: What to ask about the image.

        Returns:
            Text description of the image.
        """
        if self.llm is None:
            return "[Vision LLM not configured]"

        image_url = self._resolve_image(image)
        if image_url.startswith("ERROR:"):
            return image_url

        content = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

        response = await self.llm.agenerate([Message(role="user", content=content)])
        return response.content

    def _resolve_image(self, image: str | Path) -> str:
        """Resolve image to a base64 data URI or HTTP URL."""
        img_str = str(image)

        # Already a data URI or HTTP URL
        if img_str.startswith(("data:", "http://", "https://")):
            return img_str

        # Local file path
        path = Path(img_str)
        if not path.exists():
            return f"ERROR: Image not found: {img_str}"

        ext = path.suffix.lower()
        mime = self.MIME_MAP.get(ext, "image/png")
        data = base64.b64encode(path.read_bytes()).decode()
        return f"data:{mime};base64,{data}"

    async def compare(self, image1: str, image2: str, question: str = "Compare these two images.") -> str:
        """Compare two images."""
        url1 = self._resolve_image(image1)
        url2 = self._resolve_image(image2)

        content = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": url1}},
            {"type": "image_url", "image_url": {"url": url2}},
        ]

        response = await self.llm.agenerate([Message(role="user", content=content)])
        return response.content
