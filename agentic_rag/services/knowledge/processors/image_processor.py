"""Image modal processor — VLM captioning and visual analysis."""

import base64
from pathlib import Path
from typing import Callable, Optional

from agentic_rag.services.knowledge.content_list import ContentItem, ContentType
from agentic_rag.services.knowledge.processors.base import BaseModalProcessor


class ImageModalProcessor(BaseModalProcessor):
    """Generate captions and extract information from images using a Vision LLM.

    Supports two calling conventions (inspired by RAG-Anything):
    1. messages format — full VLM with image data + prompt
    2. Callable function — any (image_data, prompt) -> str
    """

    content_type = ContentType.IMAGE
    description = "Generate captions for images using Vision LLM"

    def __init__(
        self,
        vision_model_func: Optional[Callable] = None,
        prompt: str = "",
    ):
        """
        Args:
            vision_model_func: Async callable (image_b64, prompt) -> str.
                               If None, uses the LLM provider's vision capability.
            prompt: Custom prompt for image description.
        """
        from agentic_rag.config.prompts import Prompts
        self.vision_model_func = vision_model_func
        self.prompt = prompt or Prompts.IMAGE_CAPTION

    async def process(self, item: ContentItem) -> ContentItem:
        """Generate a caption for an image content item."""
        if item.type != ContentType.IMAGE:
            return item

        image_path = item.img_path
        if not image_path:
            return item

        # Load image as base64
        try:
            path = Path(image_path)
            if not path.exists():
                item.metadata["image_error"] = f"File not found: {image_path}"
                return item

            ext = path.suffix.lower()
            mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".png": "image/png", ".gif": "image/gif",
                        ".webp": "image/webp", ".bmp": "image/bmp"}
            mime = mime_map.get(ext, "image/png")
            image_b64 = base64.b64encode(path.read_bytes()).decode()
            data_uri = f"data:{mime};base64,{image_b64}"
        except Exception as e:
            item.metadata["image_error"] = str(e)
            return item

        # Generate caption (only if a vision function is configured)
        if self.vision_model_func:
            try:
                caption = await self.vision_model_func(data_uri, self.prompt)
                item.img_caption = caption
                item.text = caption  # Make searchable
            except Exception as e:
                item.metadata["image_error"] = f"Vision model error: {e}"
        # When no vision function is available, we leave text/caption empty.
        # The multimodal embedding model (e.g. jina-embeddings-v5-omni) will
        # encode visual features directly from the image pixels — no text
        # placeholder needed.

        return item
