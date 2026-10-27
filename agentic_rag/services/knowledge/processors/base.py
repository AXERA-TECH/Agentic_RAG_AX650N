"""Base modal processor — processes content items of a specific type."""

from abc import ABC, abstractmethod
from typing import Any

from agentic_rag.services.knowledge.content_list import ContentItem


class BaseModalProcessor(ABC):
    """Abstract base for modality-specific content processors.

    Each processor handles one ContentType:
    - ImageProcessor: VLM captioning
    - TableProcessor: structure interpretation
    - EquationProcessor: LaTeX parsing
    - VideoProcessor: keyframe + VLM
    - AudioProcessor: STT transcription
    """

    content_type: str
    description: str = ""

    @abstractmethod
    async def process(self, item: ContentItem) -> ContentItem:
        """Process a single content item. Returns the enriched item."""
        ...

    async def process_batch(self, items: list[ContentItem]) -> list[ContentItem]:
        """Process multiple items (default: sequential, override for concurrency)."""
        results = []
        for item in items:
            results.append(await self.process(item))
        return results

    def _result(self, item: ContentItem, **updates) -> ContentItem:
        """Helper to update an item with processing results."""
        for key, value in updates.items():
            if hasattr(item, key):
                setattr(item, key, value)
        return item
