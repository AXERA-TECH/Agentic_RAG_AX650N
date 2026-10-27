"""Knowledge Pipeline — end-to-end multi-modal document processing and retrieval.

Stages (RAG-Anything inspired):
1. Parse: Document -> ContentList (universal format)
2. Process: Content items -> modality-specific enrichment (VLM caption, STT, etc.)
3. Index: ContentList -> Knowledge Graph + Vector Store
4. Retrieve: Query -> Hybrid (Vector + Graph) -> Ranked Results
5. Generate: Results + Query -> LLM -> Answer

Usage:
    pipeline = KnowledgePipeline(llm=..., vision_func=..., embedding_func=...)
    await pipeline.ingest("path/to/doc.pdf")
    answer = await pipeline.query("What does this document say about X?")
"""

import asyncio
import inspect
import uuid
from pathlib import Path
from typing import Callable, Optional

from agentic_rag.services.knowledge.content_list import ContentItem, ContentList, ContentType
from agentic_rag.services.knowledge.document_parsers import DocumentParsingMixin
from agentic_rag.services.knowledge.file_registry import KnowledgeFileRegistry
from agentic_rag.services.knowledge.factories import (
    create_embedding_func,
    create_kg_llm_func,
    create_vector_store,
    create_vision_func,
)
from agentic_rag.services.knowledge.graph.builder import GraphBuilder
from agentic_rag.services.knowledge.graph.index import KnowledgeGraph
from agentic_rag.services.knowledge.ingest_options import IngestOptions
from agentic_rag.services.knowledge.processors.base import BaseModalProcessor
from agentic_rag.services.knowledge.processors.image_processor import ImageModalProcessor
from agentic_rag.services.knowledge.processors.multimodal_processors import (
    AudioModalProcessor,
    EquationModalProcessor,
    TableModalProcessor,
    VideoModalProcessor,
)
from agentic_rag.services.knowledge.processors.text_processor import TextModalProcessor
from agentic_rag.services.knowledge.retrieval.hybrid import HybridRetriever, RetrievalResult
from agentic_rag.utils.logging import get_logger

logger = get_logger(__name__)


class KnowledgePipeline(DocumentParsingMixin):
    """End-to-end knowledge pipeline — parse, process, index, retrieve, generate."""

    def __init__(
        self,
        llm_func: Optional[Callable] = None,
        vision_func: Optional[Callable] = None,
        embedding_func: Optional[Callable] = None,
        vector_store=None,
        enable_kg: bool = False,
        extract_entities: bool = False,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        """
        Args:
            llm_func: LLM completion function (text, tools, stop, temp, max_tokens) -> response.
            vision_func: Vision model function (image_data, prompt) -> text.
            embedding_func: Embedding function (text) -> list[float].
            vector_store: Vector store instance (BaseVectorStore).
            enable_kg: Enable knowledge graph construction.
            extract_entities: Extract fine-grained entities from text (uses LLM).
            chunk_size: Max characters per text chunk (default 800).
            chunk_overlap: Overlap characters between chunks (default 150).
        """
        self.llm_func = llm_func
        self.vision_func = vision_func
        self.embedding_func = embedding_func  # kept for backward compat
        self.vector_store = vector_store
        self.enable_kg = enable_kg
        self.extract_entities = extract_entities
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Wrap embedding function in multimodal adapter
        self._embedding_adapter: Optional["EmbeddingAdapter"] = None
        if embedding_func:
            from agentic_rag.services.knowledge.embedding import EmbeddingAdapter
            self._embedding_adapter = EmbeddingAdapter(embedding_func)

        # Internal state
        self.graph = KnowledgeGraph() if enable_kg else None
        self._content_cache: dict[str, ContentItem] = {}  # entity_id -> ContentItem
        self._documents: dict[str, ContentList] = {}       # doc_id -> ContentList

        # Process-wide ingestion defaults.  Per-request overrides are passed as
        # IngestOptions so a concurrent request cannot observe them.
        self.ingest_mode: str = "multimodal"  # "text" | "multimodal"
        # Captioning gives media a meaningful text representation for the
        # default text embedding model.  "pure" remains available for genuinely
        # multimodal embedding endpoints.
        self.mm_method: str = "caption"        # "pure" | "caption" | "both"

        # Lazily-initialized components.  Processors capture chunk settings at
        # construction time, so they are cached per (size, overlap) pair.
        self._processor_cache: dict[
            tuple[int, int], dict[ContentType, BaseModalProcessor]
        ] = {}
        self._retriever: Optional[HybridRetriever] = None
        # ``_documents`` / ``graph`` / ``_content_cache`` are shared mutable
        # state; ingestion serializes access so options cannot interleave.
        self._ingest_lock = asyncio.Lock()
        self._kg_llm_func: Optional[Callable] = None
        self._file_registry_store = KnowledgeFileRegistry(
            path_provider=lambda: self._file_registry_path,
            graph_path_provider=lambda: Path(self._graph_path),
            vector_store=vector_store,
        )

    @property
    def _graph_path(self) -> str:
        """Path to the persisted knowledge graph JSON file."""
        from agentic_rag.config.settings import get_settings
        ws = get_settings().workspace_dir
        return f"{ws}/knowledge_graph.json"

    @property
    def _file_registry_path(self) -> Path:
        """Path to the persisted source-file registry."""
        from agentic_rag.config.settings import get_settings
        return Path(get_settings().workspace_dir) / "knowledge_files.json"

    @property
    def _file_registry(self) -> dict[str, dict]:
        """Live registry entries (kept for compatibility)."""
        return self._file_registry_store.entries

    def _save_file_registry(self) -> None:
        """Persist the file registry (delegated)."""
        self._file_registry_store.save()

    def _load_file_registry(self) -> None:
        """Restore the file registry (delegated)."""
        self._file_registry_store.load()

    def _save_graph(self) -> None:
        """Persist the knowledge graph to disk."""
        if self.graph is None:
            return
        try:
            self.graph.save_json(self._graph_path)
        except Exception:
            logger.exception("Failed to save KG")

    def _load_graph(self) -> None:
        """Restore the knowledge graph from disk (if exists)."""
        if not self.enable_kg:
            return
        try:
            self.graph = KnowledgeGraph.load_json(self._graph_path)
            if self.graph.entity_count > 0:
                logger.info(
                    "KG loaded from disk: %s entities, %s relations",
                    self.graph.entity_count, self.graph.relation_count,
                )
        except Exception:
            logger.exception("Failed to load KG")

    # ── File registry ────────────────────────────

    def register_file(self, doc_id: str, name: str, size: int, content_type: str) -> None:
        """Record file metadata for later enumeration."""
        self._file_registry_store.register(doc_id, name, size, content_type)

    def list_files(self) -> list[dict]:
        """Return all registered files with their doc_ids."""
        return self._file_registry_store.list()

    def unregister_file(self, doc_id: str) -> None:
        """Remove a file from the registry."""
        self._file_registry_store.unregister(doc_id)

    def clear_file_registry(self) -> None:
        """Remove all persisted source metadata."""
        self._file_registry_store.clear()

    # ── In-memory document access (public API) ───

    @property
    def document_count(self) -> int:
        """Number of documents currently held in memory."""
        return len(self._documents)

    def get_document(self, doc_id: str) -> ContentList | None:
        """Return one in-memory document by ID, or None."""
        return self._documents.get(doc_id)

    def set_document_source(self, doc_id: str, source: str) -> None:
        """Override the recorded source name (e.g. original upload filename)."""
        document = self._documents.get(doc_id)
        if document is not None:
            document.source = source

    def remove_document(self, doc_id: str) -> None:
        """Forget one document and its cached content items."""
        self._documents.pop(doc_id, None)

    def default_ingest_options(self) -> IngestOptions:
        """Snapshot the process-wide defaults as per-request options."""
        return IngestOptions(
            ingest_mode=self.ingest_mode,
            mm_method=self.mm_method,
            enable_kg=self.enable_kg,
            extract_entities=self.extract_entities,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    async def clear_all(self) -> None:
        """Reset in-memory documents, cached content, and the knowledge graph."""
        async with self._ingest_lock:
            self._clear_all_locked()

    def _clear_all_locked(self) -> None:
        self._documents.clear()
        self._content_cache.clear()
        self.clear_file_registry()
        graph_path = Path(self._graph_path)
        if graph_path.exists():
            graph_path.unlink(missing_ok=True)
        self.graph = KnowledgeGraph()

    async def stats(self) -> dict:
        """Return knowledge-base counts for the management API."""
        vector_stats: dict = {}
        chunk_count = 0
        doc_count = self.document_count
        stats_fn = getattr(self.vector_store, "stats", None)
        if callable(stats_fn):
            try:
                result = await asyncio.to_thread(stats_fn)
                if inspect.isawaitable(result):
                    result = await result
                if isinstance(result, dict):
                    vector_stats = result
                    chunk_count = int(result.get("chunks", 0) or 0)
                    doc_count = max(
                        doc_count, int(result.get("documents", 0) or 0),
                    )
            except Exception:
                logger.exception("Vector store stats failed")
        return {
            "chunks": chunk_count,
            "documents": doc_count,
            "graph": self.graph.stats() if self.graph else {},
            "vector_store": vector_stats,
        }

    async def graph_dict(self) -> dict:
        """Return the knowledge graph in the shape the UI expects."""
        if not self.graph:
            return {
                "entities": [], "relations": [],
                "stats": {"entities": 0, "relations": 0},
            }
        data = self.graph.to_dict()
        return {
            "entities": [
                {
                    "id": entity_id,
                    "name": entity["name"][:80],
                    "type": entity["type"],
                    "content_type": entity.get("content_type", ""),
                    "text_preview": entity.get("content_text", "")[:120],
                }
                for entity_id, entity in data.get("entities", {}).items()
            ],
            "relations": [
                {
                    "source": relation["source"][:12],
                    "target": relation["target"][:12],
                    "type": relation["type"],
                    "weight": round(relation.get("weight", 1.0), 2),
                }
                for relation in data.get("relations", [])
            ],
            "stats": self.graph.stats(),
        }

    async def get_document_items(self, doc_id: str) -> dict | None:
        """Return preview items for a document, from memory or the store."""
        document = self.get_document(doc_id)
        if document:
            return {
                "doc_id": doc_id,
                "source": document.source,
                "items": [
                    {
                        "text": item.to_searchable_text() or item.text or "",
                        "type": item.type.value,
                        "page_idx": item.page_idx,
                        "image_path": item.img_path or "",
                        "video_path": item.video_path or "",
                        "audio_path": item.audio_path or "",
                        "table_body": item.table_body or "",
                    }
                    for item in document.items
                ],
            }

        getter = getattr(self.vector_store, "get_by_source", None)
        if not callable(getter):
            return None
        rows = await getter(doc_id)
        if not rows:
            return None
        return {
            "doc_id": doc_id,
            "items": [
                {
                    "text": row.get("text", ""),
                    "type": row.get("content_type", "text"),
                    "image_path": row.get("image_path", ""),
                    "video_path": row.get("video_path", ""),
                    "audio_path": row.get("audio_path", ""),
                    "table_body": row.get("table_body", ""),
                }
                for row in rows
            ],
        }

    async def delete_document(self, doc_id: str) -> int:
        """Delete vectors, full-text rows, in-memory data, and metadata."""
        deleted = await self.delete_document_vectors(doc_id)
        full_text_delete = getattr(
            self.vector_store, "delete_full_text_by_source", None,
        )
        if callable(full_text_delete):
            await full_text_delete(doc_id)
        self.remove_document(doc_id)
        self.unregister_file(doc_id)
        return deleted

    async def clear_knowledge_base(self) -> None:
        """Drop all persisted vectors plus in-memory state."""
        async with self._ingest_lock:
            clear_collection = getattr(
                self.vector_store, "clear_collection", None,
            )
            if callable(clear_collection):
                await clear_collection()
            clear_full_text = getattr(self.vector_store, "clear_full_text", None)
            if callable(clear_full_text):
                await clear_full_text()
            self._clear_all_locked()

    def _get_kg_llm(self) -> Optional[Callable]:
        """Return the configured LLM callable, creating the KG one lazily.

        A request that enables ``enable_kg`` on demand must not have to mutate
        the shared pipeline's ``llm_func``; the fallback is cached here instead.
        """
        if self.llm_func is not None:
            return self.llm_func
        if self._kg_llm_func is None:
            self._kg_llm_func = _create_kg_llm_func()
        return self._kg_llm_func

    # ── Stage 1: Ingest ──────────────────────────

    async def ingest(
        self,
        source: str | Path,
        content: str | list[dict] | ContentList | None = None,
        source_type: str = "auto",
        *,
        options: IngestOptions | None = None,
    ) -> str:
        """Ingest a document or raw content into the knowledge base.

        Args:
            source: File path, URL, or identifier.
            content: Pre-parsed content (str, list of dicts, or ContentList).
                     If None, the source is treated as a file to be parsed.
            source_type: "auto", "text", "pdf", "image", "video", "audio", "content_list".
            options: Per-request ingestion settings; the pipeline defaults are
                     used when omitted.  The pipeline is never mutated.

        Returns:
            Document ID.
        """
        options = options or self.default_ingest_options()
        llm_func = self._get_kg_llm() if options.should_extract_entities else self.llm_func
        async with self._ingest_lock:
            doc_id = str(uuid.uuid4())

            # Build ContentList from input
            if isinstance(content, ContentList):
                content_list = content
            elif isinstance(content, list):
                content_list = self._dicts_to_content_list(content, source)
            elif isinstance(content, str):
                content_list = self._text_to_content_list(content, source)
            else:
                content_list = await self._parse_file(source, source_type)

            content_list.source = str(source)
            content_list.source_type = source_type
            self._documents[doc_id] = content_list

            # Stage 2: Process (enrich each content item)
            content_list = await self._process(content_list, options, llm_func)
            # Keep previews and document lookups consistent with what is indexed.
            self._documents[doc_id] = content_list

            # Stage 3: Index (graph + vector store)
            await self._index(content_list, doc_id, options, llm_func)

            return doc_id

    async def ingest_multimodal(
        self,
        text: str = "",
        images: list[str] = None,
        tables: list[dict] = None,
        videos: list[str] = None,
        audio_files: list[str] = None,
        source: str = "api_input",
        *,
        options: IngestOptions | None = None,
    ) -> str:
        """Ingest multi-modal content directly (without document parsing).

        This is the "direct content list insertion" approach — bypasses parsing
        entirely and accepts pre-separated modality content.
        """
        options = options or self.default_ingest_options()
        llm_func = self._get_kg_llm() if options.should_extract_entities else self.llm_func
        async with self._ingest_lock:
            content_list = ContentList(source=source, source_type="multimodal")

            if text:
                content_list.items.append(ContentItem.from_text(text))
            for img in (images or []):
                content_list.items.append(ContentItem.from_image(img))
            for tbl in (tables or []):
                content_list.items.append(ContentItem.from_table(
                    tbl.get("body", ""), tbl.get("caption", "")))
            for vid in (videos or []):
                content_list.items.append(ContentItem.from_video(vid))
            for aud in (audio_files or []):
                content_list.items.append(ContentItem.from_audio(aud))

            doc_id = str(uuid.uuid4())
            self._documents[doc_id] = content_list
            content_list = await self._process(content_list, options, llm_func)
            self._documents[doc_id] = content_list
            await self._index(content_list, doc_id, options, llm_func)
            return doc_id

    # ── Stage 4: Retrieve ────────────────────────

    async def retrieve(
        self,
        query: str,
        query_image: str = "",
        top_k: int = 3,
        mode: str = "naive",
        modality_filter: list[ContentType] = None,
    ) -> list[RetrievalResult]:
        # Retrieve relevant content (naive=vector only by default).
        retriever = self._get_retriever()
        return await retriever.retrieve(
            query, mode, top_k, modality_filter, query_image=query_image,
        )

    # ── Stage 5: Generate ────────────────────────

    async def query(self, question: str, top_k: int = 3,
                     mode: str = "naive") -> dict:
        """End-to-end RAG query: retrieve -> generate answer.

        Returns:
            {"answer": str, "sources": list[RetrievalResult]}
        """
        results = await self.retrieve(question, top_k=top_k, mode=mode)

        if not results or not self.llm_func:
            return {"answer": "No relevant content found.", "sources": results}

        # Build context from results
        context_parts = []
        for i, r in enumerate(results):
            ctx = r.content_item.to_context_string()
            context_parts.append(f"[Source {i+1}] (score: {r.score:.3f}):\n{ctx}")

        context = "\n\n---\n\n".join(context_parts)

        from agentic_rag.config.prompts import Prompts
        prompt = Prompts.RAG_QUERY_ANSWER.format(context=context, question=question)

        try:
            answer = await self.llm_func(prompt)
            if hasattr(answer, 'content'):
                answer = answer.content
        except Exception as e:
            answer = f"Error generating answer: {e}"

        return {"answer": str(answer), "sources": results}

    # ── Internal Methods ─────────────────────────

    async def _process(
        self,
        content_list: ContentList,
        options: IngestOptions | None = None,
        llm_func: Optional[Callable] = None,
    ) -> ContentList:
        """Stage 2: Process all content items through modality-specific processors."""
        processors = self._get_processors(
            options or self.default_ingest_options(), llm_func,
        )

        # Group items by type for batch processing
        by_type: dict[ContentType, list[ContentItem]] = {}
        for item in content_list.items:
            by_type.setdefault(item.type, []).append(item)

        processed_items: list[ContentItem] = []
        for ctype, items in by_type.items():
            processor = processors.get(ctype)
            if processor:
                processed = await processor.process_batch(items)
            else:
                processed = items
            processed_items.extend(processed)

        # Preserve original order
        processed_items.sort(key=lambda x: content_list.items.index(x) if x in content_list.items else 0)

        return ContentList(
            items=processed_items,
            source=content_list.source,
            source_type=content_list.source_type,
            metadata=content_list.metadata,
        )

    async def _index(
        self,
        content_list: ContentList,
        doc_id: str,
        options: IngestOptions | None = None,
        llm_func: Optional[Callable] = None,
    ) -> None:
        """Stage 3: Build graph and index vectors."""
        options = options or self.default_ingest_options()
        logger.info(
            "_index: %s items, vec_store=%s, adapter=%s, kg=%s",
            len(content_list.items),
            "OK" if self.vector_store else "MISSING",
            "OK" if self._embedding_adapter else "MISSING",
            "ON" if options.enable_kg else "OFF",
        )

        # Build knowledge graph (incremental — extends existing graph)
        if options.enable_kg:
            llm_func = (
                llm_func
                if llm_func is not None
                else self._get_kg_llm() if options.should_extract_entities
                else self.llm_func
            )
            builder = GraphBuilder(
                llm_func=llm_func,
                extract_entities=(
                    options.should_extract_entities and llm_func is not None
                ),
            )
            # Pass existing graph so entities accumulate across documents
            self.graph = await builder.build(
                content_list, doc_id, existing_graph=self.graph,
            )
            logger.info(
                "KG built: %s entities, %s relations",
                self.graph.entity_count, self.graph.relation_count,
            )

            # Persist graph to disk
            self._save_graph()

            # Cache content items by entity for retrieval
            for item in content_list.items:
                searchable = item.to_searchable_text()
                if searchable:
                    self._content_cache[searchable[:100]] = item
        else:
            logger.debug("KG skipped (enable_kg=False)")

        # Index into vector store (multimodal-aware)
        if not self.vector_store:
            logger.warning("Skipping embeddings: vector_store is None (pymilvus missing?)")
            return

        # Full-text indexing is independent of embedding availability.
        full_text_adder = getattr(self.vector_store, "add_full_text", None)
        if full_text_adder:
            full_text_docs = []
            for item in content_list.items:
                searchable = item.to_searchable_text()
                if searchable:
                    full_text_docs.append(
                        self._make_doc(item, searchable, item.type.value, doc_id=doc_id)
                    )
            if full_text_docs:
                await full_text_adder("knowledge", full_text_docs)
                logger.info("Full-text indexed: %s chunks", len(full_text_docs))

        if not self._embedding_adapter:
            logger.warning("Skipping embeddings: embedding_adapter is None (openai missing?)")
            return

        if self.vector_store and self._embedding_adapter:
            docs = []
            is_multimodal_embedding = bool(
                getattr(self._embedding_adapter, "is_multimodal", False)
            )
            for item in content_list.items:
                is_media = item.type in (ContentType.IMAGE, ContentType.VIDEO, ContentType.AUDIO)

                # ── Text-only mode: skip media items ──────────
                if options.ingest_mode == "text" and is_media:
                    logger.debug("Skipping %s (text-only mode)", item.type.value)
                    continue

                # ── Multimodal mode ────────────────────────────
                if options.ingest_mode == "multimodal" and is_media:
                    if options.mm_method == "caption":
                        # VLM caption -> text embedding only
                        caption = await self._generate_caption(item)
                        if caption:
                            docs.append(self._make_doc(item, caption, "text", doc_id=doc_id))
                    elif options.mm_method == "both":
                        # Pure multimodal embedding (with text fallback for text-only APIs)
                        emb_input = item.to_embedding_input()
                        if is_multimodal_embedding or emb_input.text.strip():
                            if emb_input.is_embeddable:
                                docs.append(await self._embed_one(item, emb_input, doc_id))
                        # Caption-based text embedding
                        caption = await self._generate_caption(item)
                        if caption:
                            docs.append(self._make_doc(item, caption, "text", doc_id=doc_id))
                    else:
                        # "pure" — require a real multimodal embedding model.
                        # A filename placeholder would create a meaningless
                        # vector for text-only embedders, so skip instead.
                        emb_input = item.to_embedding_input()
                        if not emb_input.text.strip() and not is_multimodal_embedding:
                            file_path = item.img_path or item.video_path or item.audio_path
                            fname = file_path.split("/")[-1] if file_path else "unknown"
                            logger.warning(
                                "Skipping %s (no caption for text-only embedding: %s)",
                                item.type.value, fname,
                            )
                            continue
                        if emb_input.is_embeddable:
                            docs.append(await self._embed_one(item, emb_input, doc_id))
                else:
                    # Text items (or text mode fallthrough)
                    emb_input = item.to_embedding_input()
                    if emb_input.is_embeddable:
                        docs.append(await self._embed_one(item, emb_input, doc_id))

            if docs:
                docs = [d for d in docs if d and d.get("embedding")]
                if docs:
                    await self.vector_store.add("knowledge", docs)
                    logger.info(
                        "Embedded & stored: %s vectors (mode=%s/%s, doc=%s...)",
                        len(docs), options.ingest_mode, options.mm_method, doc_id[:12],
                    )
                else:
                    logger.warning("No docs to store (embeddings empty or all failed)")
            else:
                logger.warning("No embeddable items (text empty or media missing)")

    async def _embed_one(self, item: ContentItem,
                          emb_input: "EmbeddingInput",
                          doc_id: str = "") -> dict | None:
        """Embed a single item and return a Milvus doc dict."""
        try:
            from agentic_rag.services.knowledge.embedding import EmbeddingInput
            vecs = await self._embedding_adapter.embed([emb_input])
            if not vecs or not vecs[0]:
                return None
            emb = vecs[0]
            item.embedding = emb

            if emb_input.has_image:
                emb_type = "image"
            elif emb_input.has_video:
                emb_type = "video"
            elif emb_input.has_audio:
                emb_type = "audio"
            else:
                emb_type = "text"

            searchable = item.to_searchable_text()
            return self._make_doc(item, searchable, emb_type, emb, doc_id)
        except Exception:
            logger.exception("Embed failed for %s", item.type.value)
            return None

    def _make_doc(self, item: ContentItem, text: str, emb_type: str,
                  embedding: list[float] | None = None,
                  doc_id: str = "") -> dict:
        """Build a Milvus-compatible doc dict from a ContentItem.

        Includes a deterministic ``entity_id`` that matches the Knowledge Graph
        entity ID, so hybrid search can link vector results to graph nodes.
        """
        # Generate the same stable entity ID that GraphBuilder._item_to_entity uses
        from agentic_rag.services.knowledge.graph.builder import _stable_id
        id_src = (item.text or item.img_path or item.video_path or item.audio_path or "")
        eid_prefix = {
            "text": "text_c", "image": "image", "table": "table",
            "equation": "equati", "video": "video_", "audio": "audio_",
        }.get(item.type.value, "text_c")
        entity_id = _stable_id(id_src[:200], eid_prefix)

        return {
            "id": entity_id,  # use stable KG entity ID as Milvus primary key
            "text": (text or "")[:2000],
            "embedding": embedding or [],
            "metadata": {
                "doc_id": doc_id or "",
                "content_type": item.type.value,
                "embedding_input_type": emb_type,
                "image_path": item.img_path or "",
                "video_path": item.video_path or "",
                "audio_path": item.audio_path or "",
                "page_idx": item.page_idx,
                "entity_id": entity_id,  # links to KnowledgeGraph node
            },
        }

    async def _generate_caption(self, item: ContentItem) -> str:
        """Generate a VLM caption for a media item. Returns empty string on failure."""
        if not self.vision_func:
            return ""
        try:
            import base64
            from pathlib import Path
            file_path = item.img_path or item.video_path or item.audio_path
            if not file_path:
                return ""
            p = Path(file_path)
            if not p.exists():
                return ""
            ext = p.suffix.lower()
            mime_map = {".jpg":"image/jpeg", ".jpeg":"image/jpeg", ".png":"image/png",
                        ".gif":"image/gif", ".webp":"image/webp", ".bmp":"image/bmp",
                        ".mp4":"video/mp4", ".avi":"video/avi", ".mov":"video/quicktime",
                        ".mp3":"audio/mpeg", ".wav":"audio/wav"}
            mime = mime_map.get(ext, "image/png")
            b64 = base64.b64encode(p.read_bytes()).decode()
            data_uri = f"data:{mime};base64,{b64}"
            from agentic_rag.config.prompts import Prompts
            prompt = Prompts.MEDIA_CAPTION_ZH
            caption = await self.vision_func(data_uri, prompt)
            return (caption or "").strip()
        except Exception:
            logger.exception("Caption failed")
            return ""

    def _get_processors(
        self,
        options: IngestOptions | None = None,
        llm_func: Optional[Callable] = None,
    ) -> dict[ContentType, BaseModalProcessor]:
        """Return (and lazily build) processors for the given chunk settings."""
        options = options or self.default_ingest_options()
        effective_llm = llm_func if llm_func is not None else self.llm_func
        key = (options.chunk_size, options.chunk_overlap, effective_llm is not None)
        processors = self._processor_cache.get(key)
        if processors is None:
            processors = {
                ContentType.TEXT: TextModalProcessor(
                    chunk_size=options.chunk_size,
                    chunk_overlap=options.chunk_overlap,
                ),
                ContentType.IMAGE: ImageModalProcessor(vision_model_func=self.vision_func),
                ContentType.TABLE: TableModalProcessor(llm_func=effective_llm),
                ContentType.EQUATION: EquationModalProcessor(llm_func=effective_llm),
                ContentType.VIDEO: VideoModalProcessor(vision_func=self.vision_func),
                ContentType.AUDIO: AudioModalProcessor(),
            }
            self._processor_cache[key] = processors
        return processors

    def _get_retriever(self) -> HybridRetriever:
        """Lazy-init hybrid retriever."""
        if self._retriever is None:
            self._retriever = HybridRetriever(
                vector_store=self.vector_store,
                graph=self.graph,
                embedding_adapter=self._embedding_adapter,
            )
        return self._retriever

    async def delete_document_vectors(self, doc_id: str) -> int:
        """Delete all vector chunks belonging to a document ID."""
        if not self.vector_store:
            return 0
        deleter = getattr(self.vector_store, "delete_by_source", None)
        if deleter is None:
            return 0
        return await deleter(doc_id)

# ── Global instance ─────────────────────────────────

_pipeline: Optional[KnowledgePipeline] = None


# Backward-compatible aliases for the dependency factories.
_create_embedding_func = create_embedding_func
_create_vision_func = create_vision_func
_create_vector_store = create_vector_store
_create_kg_llm_func = create_kg_llm_func


def init_knowledge_pipeline(
    enable_kg: bool = False,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    extract_entities: bool = False,
) -> KnowledgePipeline:
    # Initialize the global knowledge pipeline from settings.
    # Call once at application startup.
    global _pipeline

    embedding_func = _create_embedding_func()
    vector_store = _create_vector_store()

    # Build a vision function for caption generation (used by caption/both mm methods)
    vision_func = _create_vision_func()

    # Build LLM function for semantic KG entity extraction
    llm_func = _create_kg_llm_func() if extract_entities else None

    if extract_entities:
        if llm_func:
            logger.info("KG semantic extraction: ENABLED (LLM ready)")
        else:
            logger.warning(
                "KG semantic extraction: DISABLED (LLM unavailable; check config)"
            )
    else:
        logger.info("KG semantic extraction: DISABLED (extract_entities=False)")

    if embedding_func is None:
        import warnings
        warnings.warn(
            "KnowledgePipeline: embedding function unavailable (openai not installed?). "
            "pip install openai"
        )

    _pipeline = KnowledgePipeline(
        llm_func=llm_func,
        embedding_func=embedding_func,
        vector_store=vector_store,
        vision_func=vision_func,
        enable_kg=enable_kg,
        extract_entities=extract_entities,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Restore knowledge graph from disk (survives server restarts)
    _pipeline._load_graph()
    _pipeline._load_file_registry()

    if vector_store is None:
        import warnings
        warnings.warn(
            "KnowledgePipeline: no vector store available (pymilvus not installed?). "
            "RAG search and ingestion will not work."
        )

    return _pipeline


def get_knowledge_pipeline(
    llm_func=None,
    vision_func=None,
    embedding_func=None,
    vector_store=None,
) -> KnowledgePipeline:
    # Get or create the global knowledge pipeline.
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    # Explicit deps mode (used by tests)
    if any(x is not None for x in (embedding_func, vector_store)):
        _pipeline = KnowledgePipeline(
            llm_func=llm_func,
            vision_func=vision_func,
            embedding_func=embedding_func,
            vector_store=vector_store,
        )
        return _pipeline

    # Production mode — auto-init from settings
    return init_knowledge_pipeline()
