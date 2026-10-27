"""Text modal processor — chunking, entity extraction, and annotation."""

from agentic_rag.services.knowledge.content_list import ContentItem, ContentType
from agentic_rag.services.knowledge.processors.base import BaseModalProcessor


class TextModalProcessor(BaseModalProcessor):
    """Process text content — chunking, cleaning, entity extraction."""

    content_type = ContentType.TEXT
    description = "Clean and chunk text content."

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def process(self, item: ContentItem) -> ContentItem:
        """Clean and optionally chunk text content."""
        if item.type != ContentType.TEXT:
            return item

        # Basic cleaning
        text = item.text.strip()
        text = self._clean_text(text)

        if not text:
            return item

        item.text = text
        return item

    async def process_batch(self, items: list[ContentItem]) -> list[ContentItem]:
        """Process text items. For texts longer than chunk_size, split into chunks."""
        results = []
        for item in items:
            processed = await self.process(item)
            if processed.text and len(processed.text) > self.chunk_size:
                # Split long text into overlapping chunks
                chunks = self._chunk_text(processed.text)
                for i, chunk in enumerate(chunks):
                    chunk_item = ContentItem(
                        type=ContentType.TEXT,
                        text=chunk,
                        page_idx=processed.page_idx,
                        metadata={
                            **processed.metadata,
                            "chunk_index": i,
                            "chunk_count": len(chunks),
                            "parent_item": id(processed),
                        },
                    )
                    results.append(chunk_item)
            else:
                results.append(processed)
        return results

    def _clean_text(self, text: str) -> str:
        """Clean up text — normalize whitespace, remove artifacts."""
        import re
        # Collapse multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Collapse multiple spaces
        text = re.sub(r' {3,}', '  ', text)
        # Remove control characters except newlines
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text.strip()

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks at Chinese/English sentence boundaries.

        Overlap is sentence-aware — the last complete sentence(s) from the
        previous chunk are prepended to the next one, so no chunk starts
        with a mid-sentence fragment.
        """
        import re
        # Split on sentence-ending punctuation: Chinese + English
        # Also split on 】 (Chinese doc section headers) and ； (Chinese semicolon)
        sentence_end = r'(?<=[.!?。！？\n；;】])\s*'
        raw_parts = re.split(sentence_end, text)
        # Merge tiny fragments back
        sentences = []
        buf = ""
        for part in raw_parts:
            if not part.strip():
                continue
            if buf and len(buf) < 20:  # short fragment — merge with previous
                buf += part
            else:
                if buf.strip():
                    sentences.append(buf.strip())
                buf = part
        if buf.strip():
            sentences.append(buf.strip())

        if not sentences:
            return [text]

        chunks = []
        current_parts = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) > self.chunk_size and current_parts:
                chunks.append("".join(current_parts))

                # Prefer complete sentences, but retain a character tail when
                # no complete sentence fits the requested overlap.
                overlap_chars = 0
                overlap_parts = []
                for part in reversed(current_parts):
                    if overlap_chars + len(part) <= self.chunk_overlap:
                        overlap_parts.insert(0, part)
                        overlap_chars += len(part)
                    else:
                        break
                if not overlap_parts and self.chunk_overlap > 0:
                    overlap_parts = [current_parts[-1][-self.chunk_overlap:]]

                # Overlap plus the next sentence must still fit the configured
                # chunk size; trim only the overlap when the sentence is large.
                max_overlap = max(0, self.chunk_size - len(sent))
                overlap_text = "".join(overlap_parts)[-max_overlap:]
                current_parts = [overlap_text] if overlap_text else []
                current_len = sum(len(part) for part in current_parts)

            current_parts.append(sent)
            current_len += len(sent)

        if current_parts:
            chunks.append("".join(current_parts))

        return chunks if chunks else [text]
