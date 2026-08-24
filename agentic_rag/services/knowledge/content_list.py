"""ContentList — the universal abstraction for parsed document content.

Inspired by RAG-Anything, the content_list is the central interface between
parsing and indexing. All parsers produce a standardized list of content items,
and all downstream processors consume it.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Types of content items that can appear in a document."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"
    VIDEO = "video"
    AUDIO = "audio"
    CODE = "code"
    UNKNOWN = "unknown"


class ContentItem(BaseModel):
    """A single content element extracted from a document.

    This is the universal format — all parsers produce a list of ContentItems.
    """
    type: ContentType = ContentType.TEXT
    text: str = ""                           # Text content (or description for visual items)
    page_idx: int = 0                        # Page number (document context)
    position: Optional[dict] = None          # {"x":, "y":, "w":, "h":} bounding box

    # Type-specific fields
    img_path: Optional[str] = None           # Local image file path
    img_url: Optional[str] = None            # Public image URL
    img_caption: Optional[str] = None        # VLM-generated caption
    table_body: Optional[str] = None         # Markdown table format
    table_caption: Optional[str] = None      # Table description
    latex: Optional[str] = None              # LaTeX formula
    video_path: Optional[str] = None         # Video file path
    video_caption: Optional[str] = None      # Video description
    audio_path: Optional[str] = None         # Audio file path
    audio_transcript: Optional[str] = None   # Transcribed text

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    entities: list[str] = Field(default_factory=list)    # Extracted entities
    embedding: Optional[list[float]] = None               # Vector embedding

    @classmethod
    def from_text(cls, text: str, page_idx: int = 0, **meta) -> "ContentItem":
        return cls(type=ContentType.TEXT, text=text, page_idx=page_idx, metadata=meta)

    @classmethod
    def from_image(cls, img_path: str, caption: str = "", page_idx: int = 0, **meta) -> "ContentItem":
        return cls(type=ContentType.IMAGE, img_path=img_path, img_caption=caption,
                   page_idx=page_idx, metadata=meta)

    @classmethod
    def from_table(cls, table_body: str, caption: str = "", page_idx: int = 0, **meta) -> "ContentItem":
        return cls(type=ContentType.TABLE, table_body=table_body, table_caption=caption,
                   page_idx=page_idx, metadata=meta)

    @classmethod
    def from_equation(cls, latex: str, page_idx: int = 0, **meta) -> "ContentItem":
        return cls(type=ContentType.EQUATION, latex=latex, page_idx=page_idx, metadata=meta)

    @classmethod
    def from_video(cls, video_path: str, caption: str = "", **meta) -> "ContentItem":
        return cls(type=ContentType.VIDEO, video_path=video_path, video_caption=caption, metadata=meta)

    @classmethod
    def from_audio(cls, audio_path: str, transcript: str = "", **meta) -> "ContentItem":
        return cls(type=ContentType.AUDIO, audio_path=audio_path,
                   audio_transcript=transcript, metadata=meta)

    def to_searchable_text(self) -> str:
        """Convert to a searchable text representation for embedding."""
        parts = []
        if self.text:
            parts.append(self.text)
        if self.img_caption:
            parts.append(f"[Image]: {self.img_caption}")
        if self.table_body:
            parts.append(f"[Table]: {self.table_caption or ''}\n{self.table_body}")
        if self.latex:
            parts.append(f"[Equation]: {self.latex}")
        if self.video_caption:
            parts.append(f"[Video]: {self.video_caption}")
        if self.audio_transcript:
            parts.append(f"[Audio]: {self.audio_transcript}")
        return "\n\n".join(parts)

    def to_context_string(self) -> str:
        """Convert to a context string for LLM prompt construction."""
        prefix_map = {
            ContentType.TEXT: "",
            ContentType.IMAGE: f"[Image {self.page_idx}]: ",
            ContentType.TABLE: f"[Table {self.page_idx}]: ",
            ContentType.EQUATION: f"[Formula {self.page_idx}]: ",
            ContentType.VIDEO: f"[Video]: ",
            ContentType.AUDIO: f"[Audio]: ",
            ContentType.CODE: f"[Code {self.page_idx}]: ",
        }
        prefix = prefix_map.get(self.type, "")
        body = self.to_searchable_text()
        return f"{prefix}{body}" if prefix else body

    def to_embedding_input(self) -> "EmbeddingInput":
        """Produce the multimodal embedding input for this content item.

        This is the source of truth for what gets embedded — it replaces
        ``to_searchable_text()`` in the embedding pipeline while that method
        remains for graph entity names, display, and LLM context building.

        For IMAGE items this returns an ``EmbeddingInput`` that carries the
        image path so multimodal models (CLIP, etc.) can embed visual features
        directly instead of going through a text caption.
        """
        # Local import to avoid circular dependency
        from agentic_rag.services.knowledge.embedding import EmbeddingInput

        if self.type == ContentType.IMAGE:
            return EmbeddingInput.from_image_path(
                image_path=self.img_path or self.img_url or "",
                caption=self.img_caption or self.text or "",
            )
        elif self.type == ContentType.TABLE:
            text = (
                f"[Table]: {self.table_caption or ''}\n{self.table_body or ''}"
            ).strip()
            return EmbeddingInput.from_text(text)
        elif self.type == ContentType.EQUATION:
            text = self.text or f"[Equation]: {self.latex or ''}"
            return EmbeddingInput.from_text(text)
        elif self.type == ContentType.VIDEO:
            return EmbeddingInput.from_video_path(
                video_path=self.video_path or "",
                caption=self.video_caption or self.text or "",
            )
        elif self.type == ContentType.AUDIO:
            return EmbeddingInput.from_audio_path(
                audio_path=self.audio_path or "",
                transcript=self.audio_transcript or self.text or "",
            )
        else:
            # TEXT, CODE, UNKNOWN — use whatever text is available
            return EmbeddingInput.from_text(self.text or "")


class ContentList(BaseModel):
    """An ordered list of ContentItems representing a parsed document.

    This is THE central abstraction — parsers produce it, the pipeline enriches it,
    and the graph index consumes it.
    """
    items: list[ContentItem] = Field(default_factory=list)
    source: str = ""                           # Source file or URL
    source_type: str = ""                      # "pdf", "docx", "image", etc.
    metadata: dict[str, Any] = Field(default_factory=dict)
    document_hierarchy: Optional[dict] = None  # Section structure

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def filter_by_type(self, content_type: ContentType) -> "ContentList":
        """Get items of a specific type."""
        return ContentList(
            items=[item for item in self.items if item.type == content_type],
            source=self.source,
            source_type=self.source_type,
            metadata=self.metadata,
        )

    @property
    def text_items(self) -> list[ContentItem]:
        return [i for i in self.items if i.type == ContentType.TEXT]

    @property
    def image_items(self) -> list[ContentItem]:
        return [i for i in self.items if i.type == ContentType.IMAGE]

    @property
    def table_items(self) -> list[ContentItem]:
        return [i for i in self.items if i.type == ContentType.TABLE]

    @property
    def all_searchable_text(self) -> str:
        """All searchable text concatenated for full-document embedding."""
        return "\n---\n".join(item.to_searchable_text() for item in self.items if item.to_searchable_text())
