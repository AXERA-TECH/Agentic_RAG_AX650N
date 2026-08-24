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
import json
import uuid
from pathlib import Path
from typing import Callable, Optional

from agentic_rag.services.knowledge.content_list import ContentItem, ContentList, ContentType
from agentic_rag.services.knowledge.graph.builder import GraphBuilder
from agentic_rag.services.knowledge.graph.index import KnowledgeGraph
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


class KnowledgePipeline:
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
        self._file_registry: dict[str, dict] = {}           # doc_id -> {name, size, type, time}

        # Ingestion mode — set before calling ingest/ingest_multimodal
        self.ingest_mode: str = "multimodal"  # "text" | "multimodal"
        self.mm_method: str = "pure"           # "pure" | "caption" | "both"

        # Lazy-initialized components
        self._processors: dict[ContentType, BaseModalProcessor] = {}
        self._retriever: Optional[HybridRetriever] = None

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

    def _save_file_registry(self) -> None:
        """Persist source metadata so the UI can restore it after a restart."""
        path = self._file_registry_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(self._file_registry, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(path)
        except Exception as e:
            print(f"  [Pipeline] ⚠ Failed to save file registry: {e}", flush=True)

    def _load_file_registry(self) -> None:
        """Restore source metadata from disk when the server starts."""
        path = self._file_registry_path
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
            else:
                data = {}
            if isinstance(data, dict):
                self._file_registry = {
                    str(doc_id): meta
                    for doc_id, meta in data.items()
                    if isinstance(meta, dict) and meta.get("name")
                }

            # Older versions persisted KG data but kept this registry in RAM.
            # Recover document names from those graph entities when possible.
            graph_path = Path(self._graph_path)
            if graph_path.exists():
                graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
                changed = False
                for entity in graph_data.get("entities", {}).values():
                    if entity.get("type") != "document":
                        continue
                    props = entity.get("properties", {})
                    doc_id, name = props.get("doc_id"), entity.get("name")
                    if doc_id and name and doc_id not in self._file_registry:
                        self._file_registry[doc_id] = {
                            "name": name,
                            "size": 0,
                            "type": Path(name).suffix.lower().lstrip(".") or "text",
                            "time": "",
                        }
                        changed = True
                if changed:
                    self._save_file_registry()

            # Last-resort recovery for indexes created before either metadata
            # mechanism existed.  Milvus always retains the document ID.
            if not self._file_registry and self.vector_store is not None:
                client = getattr(self.vector_store, "c", None)
                collection = getattr(self.vector_store, "col", "knowledge")
                if client is not None:
                    try:
                        client.load_collection(collection)
                        rows = client.query(
                            collection,
                            filter="source != ''",
                            output_fields=["source", "content_type"],
                            limit=10000,
                        )
                        for row in rows or []:
                            doc_id = row.get("source", "")
                            if doc_id and doc_id not in self._file_registry:
                                self._file_registry[doc_id] = {
                                    "name": doc_id,
                                    "size": 0,
                                    "type": row.get("content_type", "text"),
                                    "time": "",
                                }
                        if self._file_registry:
                            self._save_file_registry()
                    except Exception as e:
                        print(f"  [Pipeline] ⚠ Failed to recover files from vector index: {e}", flush=True)
        except Exception as e:
            print(f"  [Pipeline] ⚠ Failed to load file registry: {e}", flush=True)

    def _save_graph(self) -> None:
        """Persist the knowledge graph to disk."""
        if self.graph is None:
            return
        try:
            self.graph.save_json(self._graph_path)
        except Exception as e:
            import sys
            print(f"  [Pipeline] ⚠ Failed to save KG: {e}", flush=True)
            sys.stdout.flush()

    def _load_graph(self) -> None:
        """Restore the knowledge graph from disk (if exists)."""
        if not self.enable_kg:
            return
        try:
            self.graph = KnowledgeGraph.load_json(self._graph_path)
            if self.graph.entity_count > 0:
                import sys
                print(f"  [Pipeline] KG loaded from disk: {self.graph.entity_count} entities, "
                      f"{self.graph.relation_count} relations", flush=True)
                sys.stdout.flush()
        except Exception as e:
            import sys
            print(f"  [Pipeline] ⚠ Failed to load KG: {e}", flush=True)
            sys.stdout.flush()

    # ── File registry ────────────────────────────

    def register_file(self, doc_id: str, name: str, size: int, content_type: str) -> None:
        """Record file metadata for later enumeration."""
        self._file_registry[doc_id] = {
            "name": name,
            "size": size,
            "type": content_type,
            "time": __import__("datetime").datetime.now().isoformat(),
        }
        self._save_file_registry()

    def list_files(self) -> list[dict]:
        """Return all registered files with their doc_ids."""
        return [
            {"doc_id": did, **meta}
            for did, meta in self._file_registry.items()
        ]

    def unregister_file(self, doc_id: str) -> None:
        """Remove a file from the registry."""
        if doc_id in self._file_registry:
            self._file_registry.pop(doc_id, None)
            self._save_file_registry()

    def clear_file_registry(self) -> None:
        """Remove all persisted source metadata."""
        self._file_registry.clear()
        try:
            self._file_registry_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"  [Pipeline] ⚠ Failed to clear file registry: {e}", flush=True)

    # ── Stage 1: Ingest ──────────────────────────

    async def ingest(
        self,
        source: str | Path,
        content: str | list[dict] | ContentList | None = None,
        source_type: str = "auto",
    ) -> str:
        """Ingest a document or raw content into the knowledge base.

        Args:
            source: File path, URL, or identifier.
            content: Pre-parsed content (str, list of dicts, or ContentList).
                     If None, the source is treated as a file to be parsed.
            source_type: "auto", "text", "pdf", "image", "video", "audio", "content_list".

        Returns:
            Document ID.
        """
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
        content_list = await self._process(content_list)
        # Keep previews and document lookups consistent with what is indexed.
        self._documents[doc_id] = content_list

        # Stage 3: Index (graph + vector store)
        await self._index(content_list, doc_id)

        return doc_id

    async def ingest_multimodal(
        self,
        text: str = "",
        images: list[str] = None,
        tables: list[dict] = None,
        videos: list[str] = None,
        audio_files: list[str] = None,
        source: str = "api_input",
    ) -> str:
        """Ingest multi-modal content directly (without document parsing).

        This is the "direct content list insertion" approach — bypasses parsing
        entirely and accepts pre-separated modality content.
        """
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
        content_list = await self._process(content_list)
        self._documents[doc_id] = content_list
        await self._index(content_list, doc_id)
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

    async def _process(self, content_list: ContentList) -> ContentList:
        """Stage 2: Process all content items through modality-specific processors."""
        processors = self._get_processors()

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

    async def _index(self, content_list: ContentList, doc_id: str) -> None:
        """Stage 3: Build graph and index vectors."""
        import sys
        total_items = len(content_list.items)
        msg = (f"  [Pipeline] _index: {total_items} items in content_list, "
               f"vec_store={'OK' if self.vector_store else 'MISSING'}, "
               f"adapter={'OK' if self._embedding_adapter else 'MISSING'}, "
               f"kg={'ON' if self.enable_kg else 'OFF'}")
        print(msg, flush=True)
        sys.stdout.flush()

        # Build knowledge graph (incremental — extends existing graph)
        if self.enable_kg:
            builder = GraphBuilder(
                llm_func=self.llm_func,
                extract_entities=self.extract_entities and self.llm_func is not None,
            )
            # Pass existing graph so entities accumulate across documents
            self.graph = await builder.build(
                content_list, doc_id, existing_graph=self.graph,
            )
            print(f"  [Pipeline] KG built: {self.graph.entity_count} entities, "
                  f"{self.graph.relation_count} relations", flush=True)

            # Persist graph to disk
            self._save_graph()
            sys.stdout.flush()

            # Cache content items by entity for retrieval
            for item in content_list.items:
                searchable = item.to_searchable_text()
                if searchable:
                    self._content_cache[searchable[:100]] = item
        else:
            print(f"  [Pipeline] KG skipped (enable_kg=False)", flush=True)
            sys.stdout.flush()

        # Index into vector store (multimodal-aware)
        if not self.vector_store:
            print(f"  [Pipeline] ⚠ SKIP embed: vector_store is None (pymilvus not installed?)")
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
                print(f"  [Pipeline] ✓ full-text indexed: {len(full_text_docs)} chunks")

        if not self._embedding_adapter:
            print(f"  [Pipeline] ⚠ SKIP embed: embedding_adapter is None (openai not installed?)")
            return

        if self.vector_store and self._embedding_adapter:
            docs = []
            for item in content_list.items:
                is_media = item.type in (ContentType.IMAGE, ContentType.VIDEO, ContentType.AUDIO)

                # ── Text-only mode: skip media items ──────────
                if self.ingest_mode == "text" and is_media:
                    print(f"  [Pipeline] ⚠ Skipping {item.type.value} (text-only mode)")
                    continue

                # ── Multimodal mode ────────────────────────────
                if self.ingest_mode == "multimodal" and is_media:
                    if self.mm_method == "caption":
                        # VLM caption -> text embedding only
                        caption = await self._generate_caption(item)
                        if caption:
                            docs.append(self._make_doc(item, caption, "text", doc_id=doc_id))
                    elif self.mm_method == "both":
                        # Pure multimodal embedding (with text fallback for text-only APIs)
                        emb_input = item.to_embedding_input()
                        if not emb_input.text.strip():
                            fp = item.img_path or item.video_path or item.audio_path
                            emb_input.text = f"[{item.type.value}: {fp.split('/')[-1] if fp else 'unknown'}]"
                        if emb_input.is_embeddable:
                            docs.append(await self._embed_one(item, emb_input, doc_id))
                        # Caption-based text embedding
                        caption = await self._generate_caption(item)
                        if caption:
                            docs.append(self._make_doc(item, caption, "text", doc_id=doc_id))
                    else:
                        # "pure" — default multimodal embedding
                        emb_input = item.to_embedding_input()
                        # Auto-fallback: if media item has no text (no caption),
                        # generate a placeholder so text-only APIs can still embed it
                        if is_media and not emb_input.text.strip():
                            file_path = item.img_path or item.video_path or item.audio_path
                            fname = file_path.split("/")[-1] if file_path else "unknown"
                            emb_input.text = f"[{item.type.value}: {fname}]"
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
                    print(f"  [Pipeline] ✓ embedded & stored: {len(docs)} vectors "
                          f"(mode={self.ingest_mode}/{self.mm_method}, doc={doc_id[:12]}...)")
                else:
                    print(f"  [Pipeline] ⚠ no docs to store (embeddings empty or all failed)")
            else:
                print(f"  [Pipeline] ⚠ no embeddable items (text empty or media missing)")

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
        except Exception as e:
            print(f"  [Pipeline] ⚠ Embed failed for {item.type.value}: {e}")
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
        except Exception as e:
            print(f"  [Pipeline] ⚠ Caption failed: {e}")
            return ""

    def _get_processors(self) -> dict[ContentType, BaseModalProcessor]:
        """Lazy-init modality processors."""
        if not self._processors:
            self._processors = {
                ContentType.TEXT: TextModalProcessor(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                ),
                ContentType.IMAGE: ImageModalProcessor(vision_model_func=self.vision_func),
                ContentType.TABLE: TableModalProcessor(llm_func=self.llm_func),
                ContentType.EQUATION: EquationModalProcessor(llm_func=self.llm_func),
                ContentType.VIDEO: VideoModalProcessor(vision_func=self.vision_func),
                ContentType.AUDIO: AudioModalProcessor(),
            }
        return self._processors

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

    # ── ContentList construction helpers ─────────

    def _text_to_content_list(self, text: str, source: str) -> ContentList:
        """Convert raw text to ContentList (with basic chunking)."""
        import re
        # Split on double newlines to create paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        items = []
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if para:
                items.append(ContentItem.from_text(para, page_idx=i // 5))
        return ContentList(items=items, source=str(source), source_type="text")

    def _dicts_to_content_list(self, dicts: list[dict], source: str) -> ContentList:
        """Convert list of dicts to ContentList (direct insertion API)."""
        items = []
        for d in dicts:
            ctype = ContentType(d.get("type", "text"))
            item = ContentItem(
                type=ctype,
                text=d.get("text", ""),
                page_idx=d.get("page_idx", 0),
                img_path=d.get("img_path"),
                img_url=d.get("img_url"),
                table_body=d.get("table_body"),
                table_caption=d.get("table_caption"),
                latex=d.get("latex"),
                video_path=d.get("video_path"),
                audio_path=d.get("audio_path"),
                metadata=d.get("metadata", {}),
            )
            items.append(item)
        return ContentList(items=items, source=str(source), source_type="content_list")

    async def _parse_file(self, source: str | Path, source_type: str) -> ContentList:
        """Parse a file into ContentList (placeholder for doc parsing integration)."""
        path = Path(source)
        if not path.exists():
            return self._text_to_content_list(str(source), str(source))

        suffix = path.suffix.lower()

        # Plain text / markdown
        if suffix in (".txt", ".md", ".markdown", ".py", ".json", ".yaml", ".yml", ".csv"):
            content = path.read_text()
            return self._text_to_content_list(content, str(source))

        # Image files
        if suffix in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
            return ContentList(
                items=[ContentItem.from_image(str(path))],
                source=str(source),
                source_type="image",
            )

        # Video files
        if suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            return ContentList(
                items=[ContentItem.from_video(str(path))],
                source=str(source),
                source_type="video",
            )

        # Audio files
        if suffix in (".mp3", ".wav", ".m4a", ".ogg", ".flac"):
            return ContentList(
                items=[ContentItem.from_audio(str(path))],
                source=str(source),
                source_type="audio",
            )

        # PDF — Docling is the default parser. OCR is opt-in via OCR__MODE.
        if suffix == ".pdf":
            from agentic_rag.config.settings import get_settings
            ocr_cfg = get_settings().ocr
            if ocr_cfg.enabled and getattr(ocr_cfg, "mode", "").strip():
                ocr_result = await self._parse_pdf_with_ocr(path)
                if ocr_result is not None and len(ocr_result.items) > 0:
                    return ocr_result
                print("  [Pipeline] OCR failed, falling back to Docling", flush=True)
            else:
                print("  [Pipeline] OCR mode not configured, using Docling", flush=True)
            return await self._parse_with_docling(path)

        # Office documents — delegate to docling if available
        if suffix in (".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"):
            return await self._parse_with_docling(path)

        # Unknown — treat as text
        content = path.read_text()
        return self._text_to_content_list(content, str(source))

    async def _parse_pdf_with_ocr(self, path: Path) -> ContentList | None:
        """Parse a PDF using PaddleOCR-VL for OCR-based text extraction.

        Converts each page to an image, sends it to the OCR API, and returns
        a ContentList with per-page text. Returns None if OCR is unavailable
        or fails, so callers can fall back to other methods.
        """
        import sys
        from agentic_rag.config.settings import get_settings
        ocr_cfg = get_settings().ocr
        if not ocr_cfg.enabled or not getattr(ocr_cfg, "mode", "").strip():
            return None
        if not ocr_cfg.model.strip():
            print("  [OCR] OCR__MODEL is empty, falling back to Docling", flush=True)
            return None

        try:
            import fitz  # pymupdf
        except ImportError:
            print(f"  [OCR] pymupdf not installed, cannot render PDF pages", flush=True)
            return None

        try:
            from openai import AsyncOpenAI
        except ImportError:
            print(f"  [OCR] openai not installed, cannot call OCR API", flush=True)
            return None

        # Limit pages to avoid excessive API calls
        doc = fitz.open(str(path))
        total_pages = min(len(doc), ocr_cfg.max_pages)
        if total_pages <= 0:
            doc.close()
            return None

        print(f"  [OCR] Parsing {path.name}: {total_pages} pages via PaddleOCR-VL", flush=True)
        sys.stdout.flush()

        client = AsyncOpenAI(
            api_key=ocr_cfg.api_key,
            base_url=ocr_cfg.api_base,
            timeout=120.0,
        )

        items = []
        for page_idx in range(total_pages):
            try:
                page = doc[page_idx]
                # Render page to image at 200 DPI for good OCR quality
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")
                img_b64 = __import__("base64").b64encode(img_bytes).decode()
                data_uri = f"data:image/png;base64,{img_b64}"

                resp = await client.chat.completions.create(
                    model=ocr_cfg.model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text": "OCR:"},
                        ],
                    }],
                    temperature=0.0,
                    max_tokens=4096,
                )
                text = resp.choices[0].message.content or ""
                if text.strip():
                    items.append(ContentItem.from_text(
                        text.strip(),
                        page_idx=page_idx,
                        element_type="ocr_page",
                    ))
            except Exception as e:
                print(f"  [OCR] Page {page_idx} failed: {e}", flush=True)
                sys.stdout.flush()
                if self._is_ocr_model_unavailable(e):
                    print(
                        f"  [OCR] Model '{ocr_cfg.model}' is unavailable, "
                        "falling back to Docling",
                        flush=True,
                    )
                    doc.close()
                    return None
                # Continue with other pages — don't abort the whole document

        doc.close()

        if not items:
            print(f"  [OCR] No text extracted from {path.name}", flush=True)
            return None

        print(f"  [OCR] Extracted {len(items)} pages from {path.name}", flush=True)
        sys.stdout.flush()
        return ContentList(items=items, source=str(path), source_type="pdf")

    @staticmethod
    def _is_ocr_model_unavailable(error: Exception) -> bool:
        """Recognize OpenAI-compatible errors that mean the OCR model cannot run."""
        status_code = getattr(error, "status_code", None)
        code = str(getattr(error, "code", "") or "").lower()
        body = str(getattr(error, "body", "") or "").lower()
        message = f"{error} {body}".lower()
        markers = (
            "model_not_found",
            "model not found",
            "model does not exist",
            "unknown model",
            "no such model",
            "model unavailable",
            "model is not available",
        )
        return code == "model_not_found" or (
            status_code == 404 and "model" in message
        ) or any(marker in message for marker in markers)

    async def _parse_with_docling(self, path: Path) -> ContentList:
        """Parse PDF/Office documents with Docling."""
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = await asyncio.to_thread(converter.convert, str(path))
            doc = result.document

            items = []
            export_markdown = getattr(doc, "export_to_markdown", None)
            if callable(export_markdown):
                markdown = export_markdown().strip()
                if markdown:
                    items.append(ContentItem.from_text(
                        markdown,
                        page_idx=0,
                        element_type="docling_markdown",
                    ))
            else:
                # Compatibility with older Docling document objects.
                for i, element in enumerate(getattr(doc, "paragraphs", [])):
                    text = element.text if hasattr(element, "text") else str(element)
                    if text.strip():
                        items.append(ContentItem.from_text(
                            text.strip(),
                            page_idx=i // 10,
                            element_type="paragraph",
                        ))

            if items:
                print(f"  [Pipeline] Docling parsed {path.name}", flush=True)
                return ContentList(
                    items=items,
                    source=str(path),
                    source_type=path.suffix.lstrip("."),
                )

        except ImportError:
            pass  # Docling not available
        except Exception as e:
            print(f"  [Pipeline] Docling parse error: {e}", flush=True)

        # Fallback: try pymupdf for PDF
        if path.suffix.lower() == ".pdf":
            try:
                import fitz  # pymupdf
                items = []
                with fitz.open(str(path)) as pdf:
                    for page_idx, page in enumerate(pdf):
                        text = page.get_text().strip()
                        if text:
                            items.append(ContentItem.from_text(
                                text, page_idx=page_idx, element_type="pdf_page",
                            ))
                if items:
                    return ContentList(items=items, source=str(path), source_type="pdf")
            except ImportError:
                pass
            except Exception as e:
                print(f"  [Pipeline] pymupdf parse error: {e}", flush=True)

        # Last resort: warn and return placeholder
        print(f"  [Pipeline] ⚠ Cannot parse {path.suffix}: no parser available. "
              f"Install docling or pymupdf.", flush=True)
        return ContentList(
            items=[ContentItem.from_text(
                f"[Cannot parse {path.suffix} file: {path.name}. "
                f"Install 'docling' or 'pymupdf' for PDF/Office support.]"
            )],
            source=str(path),
            source_type=path.suffix,
        )


# ── Global instance ─────────────────────────────────

_pipeline: Optional[KnowledgePipeline] = None


def _create_embedding_func():
    """Build a multimodal embedding function using the omni embedding model.

    The AXERA jina-embeddings-v5-omni model supports an OpenAI-compatible
    /v1/embeddings endpoint with a messages-based multimodal format:

        {
          "model": "...",
          "prompt_name": "query" | "document",
          "messages": [{"role": "user", "content": [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "/abs/path"}},
            ...
          ]}]
        }

    Images/video/audio are referenced by local file path (not base64).
    """
    from pathlib import Path as _Path

    try:
        from openai import AsyncOpenAI
    except ImportError:
        return None

    from agentic_rag.config.settings import get_settings
    settings = get_settings()
    emb = settings.embedding

    client = AsyncOpenAI(
        api_key=emb.api_key or "not-needed",
        base_url=emb.api_base or "https://api.openai.com/v1",
    )

    # Detect API type: AXERA omni uses messages format, standard APIs use input
    _is_omni_api = "axera" in (emb.api_base or "").lower() or "omni" in (emb.model or "").lower()

    if _is_omni_api:
        # ── AXERA omni multimodal embedding ──────────────────────
        import base64

        def _read_as_data_uri(file_path: str, mime: str = "") -> str:
            p = _Path(file_path)
            if not p.exists():
                return ""
            raw = p.read_bytes()
            b64 = base64.b64encode(raw).decode()
            if not mime:
                ext = p.suffix.lower()
                mime_map = {
                    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".gif": "image/gif",
                    ".webp": "image/webp", ".bmp": "image/bmp",
                    ".mp4": "video/mp4", ".avi": "video/avi",
                    ".mov": "video/quicktime", ".mkv": "video/x-matroska",
                    ".mp3": "audio/mpeg", ".wav": "audio/wav",
                    ".ogg": "audio/ogg", ".flac": "audio/flac",
                }
                mime = mime_map.get(ext, "application/octet-stream")
            return f"data:{mime};base64,{b64}"

        async def embed_multimodal(inputs, prompt_name="document"):
            results = []
            for inp in inputs:
                content = []
                if inp.text:
                    content.append({"type": "text", "text": inp.text})
                if inp.image_path:
                    url = _read_as_data_uri(inp.image_path)
                    if url:
                        content.append({"type": "image_url", "image_url": {"url": url}})
                elif inp.image_url:
                    content.append({"type": "image_url", "image_url": {"url": inp.image_url}})
                if inp.video_path:
                    url = _read_as_data_uri(inp.video_path)
                    if url:
                        content.append({"type": "video_url", "video_url": {"url": url}})
                if inp.audio_path:
                    url = _read_as_data_uri(inp.audio_path)
                    if url:
                        content.append({"type": "audio_url", "audio_url": {"url": url}})
                if not content:
                    content.append({"type": "text", "text": inp.text or ""})

                resp = await client.embeddings.create(
                    model=emb.model, input="", encoding_format="float",
                    extra_body={
                        "prompt_name": prompt_name,
                        "messages": [{"role": "user", "content": content}],
                    },
                )
                results.append(list(resp.data[0].embedding))
            return results
    else:
        # ── Standard OpenAI-compatible text embedding ────────────
        async def embed_multimodal(inputs, prompt_name="document"):
            texts = [inp.text or "" for inp in inputs]
            resp = await client.embeddings.create(
                model=emb.model, input=texts, encoding_format="float",
            )
            return [list(d.embedding) for d in resp.data]

    return embed_multimodal


def _create_vision_func():
    # Build a VLM captioning function from the first configured LLM provider.
    # Returns None if no vision-capable LLM is configured.
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return None

    from agentic_rag.config.settings import get_settings
    settings = get_settings()
    providers = settings.llm_providers
    if not providers:
        return None

    provider_cfg = list(providers.values())[0]
    # LLMProviderConfig is a pydantic model — use attribute access
    api_key = getattr(provider_cfg, 'api_key', '') or 'not-needed'
    api_base = getattr(provider_cfg, 'api_base', '') or ''
    model = getattr(provider_cfg, 'model', '') or ''

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=api_base or "https://api.openai.com/v1",
    )

    async def vision_func(image_data: str, prompt: str) -> str:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data}},
        ]
        resp = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": content}],
            max_tokens=300, temperature=0.3,
        )
        return resp.choices[0].message.content or ""

    return vision_func


def _create_vector_store():
    # Build a Milvus-backed vector store from settings.
    from agentic_rag.config.settings import get_settings
    settings = get_settings()

    try:
        from pymilvus import MilvusClient
        db_path = f"{settings.workspace_dir}/milvus_lite.db"
        client = MilvusClient(db_path)
        dim = settings.embedding.dim

        if "knowledge" not in (client.list_collections() or []):
            # Explicit schema: VARCHAR id (UUID), VARCHAR text, float vector
            from pymilvus import DataType
            schema = client.create_schema(enable_dynamic_field=True)
            schema.add_field("id", DataType.VARCHAR, max_length=64, is_primary=True)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
            client.create_collection("knowledge", schema=schema, dimension=dim)
        # Ensure collection is loaded (survives restarts)
        try:
            client.load_collection("knowledge")
        except Exception:
            pass

        from agentic_rag.services.knowledge.retrieval.full_text import FullTextIndex
        full_text_index = FullTextIndex(Path(settings.workspace_dir) / "knowledge_fts.db")

        class MilvusAdapter:
            def __init__(self, c, fts, col="knowledge"):
                self.c, self.col = c, col
                self.fts = fts

            async def add(self, collection, docs):
                try:
                    self.c.load_collection(self.col)
                except Exception:
                    pass
                data = []
                for i, doc in enumerate(docs):
                    data.append({
                        "id": doc.get("id", i),
                        "vector": doc.get("embedding", []),
                        "text": doc.get("text", "")[:2000],
                        "source": doc.get("metadata", {}).get("doc_id", ""),
                        "content_type": doc.get("metadata", {}).get("content_type", "text"),
                        "image_path": doc.get("metadata", {}).get("image_path", ""),
                        "video_path": doc.get("metadata", {}).get("video_path", ""),
                        "audio_path": doc.get("metadata", {}).get("audio_path", ""),
                        "entity_id": doc.get("metadata", {}).get("entity_id", ""),
                    })
                if data:
                    # upsert: stable entity IDs mean re-ingested content
                    # updates existing vectors instead of failing with duplicate key
                    import asyncio
                    await asyncio.to_thread(self.c.upsert, self.col, data)
                    await asyncio.to_thread(self.fts.upsert, data)
                return [d.get("id", "") for d in docs]

            async def add_full_text(self, collection, docs):
                data = []
                for index, doc in enumerate(docs):
                    metadata = doc.get("metadata", {})
                    data.append({
                        "id": doc.get("id", index),
                        "text": doc.get("text", "")[:2000],
                        "source": metadata.get("doc_id", ""),
                        "content_type": metadata.get("content_type", "text"),
                        "image_path": metadata.get("image_path", ""),
                        "video_path": metadata.get("video_path", ""),
                        "audio_path": metadata.get("audio_path", ""),
                        "entity_id": metadata.get("entity_id", ""),
                    })
                if data:
                    await asyncio.to_thread(self.fts.upsert, data)
                return [item.get("id", "") for item in data]

            async def search(self, collection, query_embedding, top_k=10, filter_expr=None):
                import asyncio
                # Ensure collection is loaded (survives restarts/releases)
                try:
                    await asyncio.to_thread(self.c.load_collection, self.col)
                except Exception:
                    pass
                kw = {
                    "collection_name": self.col, "data": [query_embedding],
                    "limit": top_k,
                    "output_fields": ["text", "source", "content_type", "image_path", "video_path", "audio_path", "entity_id"],
                }
                if filter_expr:
                    kw["filter"] = filter_expr
                # Run blocking Milvus search in thread pool
                res = await asyncio.to_thread(lambda: self.c.search(**kw)[0])
                return [
                    type("D", (), {
                        "text": r["entity"].get("text", ""),
                        # Milvus returns distance (lower = better). Convert to
                        # similarity (higher = better) so all downstream code
                        # can consistently sort descending.
                        "score": 1.0 - r["distance"],
                        "entity_id": r["entity"].get("entity_id", ""),
                        "metadata": {
                            "source": r["entity"].get("source", ""),
                            "content_type": r["entity"].get("content_type", "text"),
                            "image_path": r["entity"].get("image_path", ""),
                            "video_path": r["entity"].get("video_path", ""),
                            "audio_path": r["entity"].get("audio_path", ""),
                            "entity_id": r["entity"].get("entity_id", ""),
                        },
                    })()
                    for r in res
                ]

            async def delete(self, collection, ids):
                if ids:
                    await asyncio.to_thread(self.c.delete, self.col, ids=ids)
                    await asyncio.to_thread(self.fts.delete_ids, ids)
                return len(ids)

            async def delete_by_source(self, doc_id):
                try:
                    self.c.load_collection(self.col)
                    rows = await asyncio.to_thread(
                        self.c.query,
                        self.col,
                        filter=f'source == "{doc_id}"',
                        output_fields=["id"],
                        limit=10000,
                    )
                    ids = [row["id"] for row in (rows or []) if row.get("id")]
                    if ids:
                        await asyncio.to_thread(self.c.delete, self.col, ids=ids)
                    await asyncio.to_thread(self.fts.delete_source, doc_id)
                    return len(ids)
                except Exception as e:
                    print(f"  [Milvus] ⚠ failed to delete source {doc_id}: {e}", flush=True)
                    return 0

            async def delete_full_text_by_source(self, doc_id):
                await asyncio.to_thread(self.fts.delete_source, doc_id)

            async def clear_full_text(self):
                await asyncio.to_thread(self.fts.clear)

            async def full_text_search(self, query, top_k=30, content_types=None):
                if await asyncio.to_thread(self.fts.count) == 0:
                    rows = await asyncio.to_thread(self._all_rows)
                    if rows:
                        await asyncio.to_thread(self.fts.upsert, rows)
                rows = await asyncio.to_thread(
                    self.fts.search, query, top_k, content_types,
                )
                return [self._row_to_doc(row) for row in rows]

            def _all_rows(self):
                fields = [
                    "id", "text", "source", "content_type", "image_path",
                    "video_path", "audio_path", "entity_id",
                ]
                try:
                    iterator = self.c.query_iterator(
                        collection_name=self.col,
                        batch_size=1000,
                        filter="id != ''",
                        output_fields=fields,
                    )
                    rows = []
                    while True:
                        batch = iterator.next()
                        if not batch:
                            break
                        rows.extend(batch)
                    iterator.close()
                    return rows
                except Exception:
                    return self.c.query(
                        self.col,
                        filter="id != ''",
                        output_fields=fields,
                        limit=16384,
                    )

            @staticmethod
            def _row_to_doc(row):
                return type("D", (), {
                    "text": row.get("text", ""),
                    "score": -float(row.get("rank", 0.0)),
                    "entity_id": row.get("entity_id", ""),
                    "metadata": {
                        "source": row.get("source", ""),
                        "content_type": row.get("content_type", "text"),
                        "image_path": row.get("image_path", ""),
                        "video_path": row.get("video_path", ""),
                        "audio_path": row.get("audio_path", ""),
                        "entity_id": row.get("entity_id", ""),
                    },
                })()

            async def create_collection(self, name, dim):
                if name not in (self.c.list_collections() or []):
                    from pymilvus import DataType
                    schema = self.c.create_schema(enable_dynamic_field=True)
                    schema.add_field("id", DataType.VARCHAR, max_length=64, is_primary=True)
                    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
                    self.c.create_collection(name, schema=schema, dimension=dim)

        return MilvusAdapter(client, full_text_index)

    except ImportError:
        pass  # pymilvus not available
    except Exception:
        pass

    return None


def _create_kg_llm_func():
    # Build a lightweight LLM callable for knowledge graph entity extraction.
    # Uses the first configured LLM provider. Returns None if no provider available.
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return None

    from agentic_rag.config.settings import get_settings
    settings = get_settings()
    providers = settings.llm_providers
    if not providers:
        return None

    provider_cfg = list(providers.values())[0]
    api_key = getattr(provider_cfg, "api_key", "") or "not-needed"
    api_base = getattr(provider_cfg, "api_base", "") or ""
    model = getattr(provider_cfg, "model", "") or ""

    client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def kg_llm(text: str) -> str:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": text}],
            max_tokens=1024,
            temperature=0.15,  # low temperature for structured extraction
            extra_body={
                # Disable Qwen thinking/reasoning mode for clean JSON output
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return resp.choices[0].message.content or ""

    return kg_llm


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

    import sys
    if extract_entities:
        if llm_func:
            print(f"  [Pipeline] KG semantic extraction: ENABLED (LLM ready)", flush=True)
        else:
            print(f"  [Pipeline] ⚠ KG semantic extraction: DISABLED (LLM unavailable — "
                  f"check LLM provider config)", flush=True)
    else:
        print(f"  [Pipeline] KG semantic extraction: DISABLED (extract_entities=False)", flush=True)
    sys.stdout.flush()

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
