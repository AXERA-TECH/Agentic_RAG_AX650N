"""RAG tools — knowledge base search, ingestion, and multi-modal query."""

import re
from typing import Any

from agentic_rag.orchestration.l1_tools.base import BaseTool
from agentic_rag.services.knowledge.content_list import ContentType


class RAGSearchTool(BaseTool):
    """Search the knowledge base with hybrid vector/FTS retrieval."""

    MIN_RELEVANCE_SCORE = 0.25
    DEFAULT_CONTEXT_CHUNKS = 3
    MAX_CONTEXT_CHUNKS = 3
    MAX_CHUNK_CHARS = 800
    name = "rag_search"
    description = "Search the knowledge base for relevant content. Supports text, images, tables, and more."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of context chunks to return (maximum 3)",
                "default": DEFAULT_CONTEXT_CHUNKS,
                "minimum": 1,
                "maximum": MAX_CONTEXT_CHUNKS,
            },
            "mode": {
                "type": "string",
                "enum": ["naive", "hybrid"],
                "description": "Retrieval mode: naive (vector only) or hybrid (vector+graph)",
                "default": "hybrid",
            },
            "modality": {
                "type": "string",
                "enum": ["all", "text", "image", "table", "equation", "video", "audio"],
                "description": "Filter by content type",
                "default": "all",
            },
        },
        "required": ["query"],
    }

    def __init__(self, pipeline=None):
        self._pipeline = pipeline

    @property
    def pipeline(self):
        if self._pipeline is None:
            from agentic_rag.services.knowledge.pipeline import get_knowledge_pipeline
            self._pipeline = get_knowledge_pipeline()
        return self._pipeline

    async def execute(self, query: str = "", top_k: int = DEFAULT_CONTEXT_CHUNKS,
                       mode: str = "hybrid", modality: str = "all", **kwargs) -> Any:
        # Small local models sometimes hallucinate extra parameters not in the
        # schema (e.g. ``context_sources``).  Silently drop them rather than
        # crash — the observation would otherwise be an opaque TypeError and
        # the model would retry with the same bad call.
        if not query:
            return "Search error: missing required parameter 'query'"

        if self.pipeline is None:
            return "Knowledge base is not configured. Please ingest documents first."

        modality_filter = None
        if modality != "all":
            modality_filter = [ContentType(modality)]
        context_k = max(1, min(int(top_k), self.MAX_CONTEXT_CHUNKS))
        # Keep the model context small while asking retrieval for a wider
        # candidate pool.  This is important for RRF agreement and for
        # deciding whether the local KB really has evidence before falling
        # back to an MCP search tool.
        retrieval_k = max(context_k, 12)

        try:
            results = await self.pipeline.retrieve(
                query=query,
                top_k=retrieval_k,
                mode=mode,
                modality_filter=modality_filter,
            )

            results = [
                result for result in results
                if result.score >= self.MIN_RELEVANCE_SCORE
            ]
            # Topical guard: no matter how strongly BM25/vector ranks a chunk,
            # it is only evidence when it actually shares topic-bearing cues
            # with the query.  RRF scores are relative ranks (1/(k+rank)), not
            # probabilities — in a single-document KB every chunk is both a
            # BM25 and a vector hit, so "corroborated" means nothing there.
            # Without this gate an off-topic query (e.g. a sports question
            # against a drug-label KB) would be answered from whatever happens
            # to be in the index.
            results = [
                result for result in results
                if self._query_overlap(
                    query, result.content_item.to_searchable_text()
                )
            ]
            # RRF scores are relative ranks, not probabilities.  Prefer
            # candidates corroborated by both lexical (BM25/FTS) and dense
            # retrieval; this is domain agnostic and avoids trusting a lone
            # semantically-near vector hit when the KB has no answer.
            has_lexical_candidates = any(
                "bm25" in str(getattr(result, "source", "")).lower()
                for result in results
            )
            corroborated = [
                result for result in results
                if "bm25" in str(getattr(result, "source", "")).lower()
                and "vector" in str(getattr(result, "source", "")).lower()
            ]
            lexical = [
                result for result in results
                if "bm25" in str(getattr(result, "source", "")).lower()
            ]
            vector_only = [
                result for result in results
                if "bm25" not in str(getattr(result, "source", "")).lower()
            ]
            if corroborated:
                results = corroborated
            elif lexical:
                results = lexical
            elif vector_only:
                results = vector_only
            elif has_lexical_candidates:
                # BM25 can rank a document solely because an incidental
                # number/date overlaps.  If no topic-bearing cue overlaps,
                # the lexical hit is not evidence for the question.
                results = []
            results = results[:context_k]
            if not results:
                return "No relevant content found in the knowledge base."

            output = []
            sources_summary = []
            source_refs = {}
            file_names = {
                item["doc_id"]: item["name"]
                for item in self.pipeline.list_files()
                if item.get("doc_id") and item.get("name")
            }
            for r in results:
                ctx = r.content_item.to_context_string()[:self.MAX_CHUNK_CHARS]
                type_label = r.content_item.type.value

                # Resolve the authoritative source name from the persisted file
                # registry. Never ask the model to invent a document title.
                doc_id = r.doc_id or (r.content_item.metadata or {}).get("source", "")
                if doc_id in file_names:
                    source_desc = file_names[doc_id]
                elif r.source_file:
                    source_desc = r.source_file.replace("\\", "/").split("/")[-1]
                elif doc_id:
                    source_desc = f"文档ID {doc_id}"
                else:
                    source_desc = "未知来源"
                source_desc = " ".join(str(source_desc).split())

                source_key = doc_id or source_desc
                if source_key not in source_refs:
                    ref_id = f"[R{len(source_refs) + 1}]"
                    source_refs[source_key] = ref_id
                    sources_summary.append(f"{ref_id} {source_desc}")
                else:
                    ref_id = source_refs[source_key]

                output.append(
                    f"{ref_id} (type: {type_label}, score: {r.score:.3f}, source: {source_desc})\n"
                    f"{ctx}"
                )

            result_text = "\n\n---\n\n".join(output)
            sources_text = "\n".join(sources_summary)
            return (
                f"{result_text}\n\n"
                f"===== SOURCES =====\n"
                f"{sources_text}\n\n"
                f"IMPORTANT: Use only these reference IDs ({', '.join(source_refs.values())}). "
                f"Source names above are exact and immutable. End with a '参考来源' section "
                f"that copies only the cited reference ID and exact source name; do not add descriptions."
            )

        except Exception as e:
            return f"Search error: {str(e)}"

    @staticmethod
    def _query_overlap(query: str, content: str) -> bool:
        """Return whether query and candidate share meaningful lexical cues."""
        q = "".join(str(query).lower().split())
        c = "".join(str(content).lower().split())
        if not q or not c:
            return False
        # Preserve complete ASCII terms (error codes, model names, etc.). Pure
        # numbers do not count by themselves when the query also has a topic:
        # an incidental date/quantity match must not turn an unrelated chunk
        # into evidence.
        ascii_terms = {
            term for term in re.findall(r"[a-z0-9][a-z0-9._/-]{1,}", q)
            if any(char.isalpha() for char in term)
        }
        if any(term in c for term in ascii_terms):
            return True
        # Character trigrams provide a lightweight tokenizer for Chinese and
        # mixed-language queries without hard-coding any domain vocabulary.
        topic_text = re.sub(r"\d+", "", q)
        q_trigrams = {
            topic_text[i:i + 3]
            for i in range(max(0, len(topic_text) - 2))
            if not topic_text[i:i + 3].isdigit()
        }
        if q_trigrams:
            return any(trigram in c for trigram in q_trigrams)
        # Numeric-only lookups (serials/IDs) still need an exact match.
        return q in c

class RAGIngestTool(BaseTool):
    """Ingest content into the knowledge base. Supports text, URLs, and multi-modal content."""

    name = "rag_ingest"
    description = "Ingest content into the knowledge base. Supports text, file paths, and multi-modal content."
    parameters_schema = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Text content to ingest, or a file path",
            },
            "source": {
                "type": "string",
                "description": "Source identifier",
                "default": "user_input",
            },
            "content_type": {
                "type": "string",
                "enum": ["text", "file", "url"],
                "description": "Type of content being ingested",
                "default": "text",
            },
        },
        "required": ["content"],
    }

    def __init__(self, pipeline=None):
        self._pipeline = pipeline

    @property
    def pipeline(self):
        if self._pipeline is None:
            from agentic_rag.services.knowledge.pipeline import get_knowledge_pipeline
            self._pipeline = get_knowledge_pipeline()
        return self._pipeline

    async def execute(self, content: str, source: str = "user_input",
                       content_type: str = "text") -> Any:
        if self.pipeline is None:
            return "Knowledge base is not configured."

        try:
            doc_id = await self.pipeline.ingest(
                source=source,
                content=content if content_type == "text" else None,
                source_type=content_type if content_type == "file" else "auto",
            )
            return (f"Content ingested successfully.\n"
                    f"Document ID: {doc_id}\n"
                    f"Source: {source}")
        except Exception as e:
            return f"Ingestion error: {str(e)}"


class RAGMultiModalIngestTool(BaseTool):
    """Ingest multi-modal content — text, images, videos, audio."""

    name = "rag_ingest_multimodal"
    description = "Ingest multi-modal content (text, images, videos, audio) into the knowledge base."
    parameters_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text content to ingest (optional)",
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of image file paths or URLs",
            },
            "videos": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of video file paths",
            },
            "audio_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of audio file paths",
            },
            "source": {
                "type": "string",
                "description": "Source identifier",
                "default": "api_multimodal",
            },
        },
        "required": [],
    }

    def __init__(self, pipeline=None):
        self._pipeline = pipeline

    @property
    def pipeline(self):
        if self._pipeline is None:
            from agentic_rag.services.knowledge.pipeline import get_knowledge_pipeline
            self._pipeline = get_knowledge_pipeline()
        return self._pipeline

    async def execute(self, text: str = "", images: list[str] = None,
                       videos: list[str] = None, audio_files: list[str] = None,
                       source: str = "api_multimodal") -> Any:
        if self.pipeline is None:
            return "Knowledge base is not configured."

        if not any([text, images, videos, audio_files]):
            return "No content provided. Specify at least one of: text, images, videos, audio_files."

        try:
            doc_id = await self.pipeline.ingest_multimodal(
                text=text,
                images=images or [],
                videos=videos or [],
                audio_files=audio_files or [],
                source=source,
            )
            counts = []
            if text: counts.append("text")
            if images: counts.append(f"{len(images)} images")
            if videos: counts.append(f"{len(videos)} videos")
            if audio_files: counts.append(f"{len(audio_files)} audio files")

            return (f"Multi-modal content ingested successfully.\n"
                    f"Document ID: {doc_id}\n"
                    f"Content: {', '.join(counts)}")
        except Exception as e:
            return f"Ingestion error: {str(e)}"
