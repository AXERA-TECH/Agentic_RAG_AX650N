"""Hybrid Retrieval — dense vector + BM25 + optional graph fusion.

Implements Stage 5 of the RAG-Anything pipeline:
- Vector similarity search (dense embeddings)
- SQLite FTS5 BM25 full-text search
- Reciprocal Rank Fusion (RRF)
- Graph traversal for contextual expansion
- Modality-aware ranking
- VLM-enhanced query (optional)
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Union

from agentic_rag.services.knowledge.content_list import ContentItem, ContentType
from agentic_rag.services.knowledge.graph.index import KnowledgeGraph


class RetrievalResult:
    """A single retrieval result with metadata."""

    def __init__(self, content_item: ContentItem, score: float,
                 entity_id: str = "", source: str = "vector",
                 doc_id: str = "", source_file: str = ""):
        self.content_item = content_item
        self.score = score
        self.entity_id = entity_id
        self.source = source                # retrieval method: "vector" or "graph"
        self.doc_id = doc_id                # document UUID from ingestion
        self.source_file = source_file      # original file path / URL

    def __repr__(self):
        text_preview = self.content_item.to_searchable_text()[:50]
        return (f"Result(score={self.score:.3f}, src={self.source}, "
                f"doc={self.source_file or self.doc_id}, text={text_preview}...)")


class HybridRetriever:
    """Combines dense, BM25, and optional graph rankings.

    Three retrieval modes (inspired by LightRAG/RAG-Anything):
    - naive: Pure vector search (no graph)
    - local: Vector search + 1-hop graph neighbors
    - global: Vector search + 2-hop graph traversal
    - hybrid (default): RRF fusion of vector + BM25 (+ graph when enabled)
    """

    def __init__(
        self,
        vector_store=None,  # MilvusStore or BaseVectorStore
        graph: Optional[KnowledgeGraph] = None,
        embedding_func=None,          # DEPRECATED: kept for backward compat
        embedding_adapter=None,       # NEW: EmbeddingAdapter instance
        top_k: int = 3,
        graph_weight: float = 0.3,
        vector_weight: float = 0.7,
    ):
        self.vector_store = vector_store
        self.graph = graph
        self.top_k = top_k
        self.graph_weight = graph_weight
        self.vector_weight = vector_weight

        # Support both old (embedding_func) and new (embedding_adapter) patterns
        if embedding_adapter is not None:
            self._adapter = embedding_adapter
        elif embedding_func is not None:
            from agentic_rag.services.knowledge.embedding import EmbeddingAdapter
            self._adapter = EmbeddingAdapter(embedding_func)
        else:
            self._adapter = None

        # Fallback embedding dimension (from config, or 1536 as last resort)
        self._fallback_dim = 1536
        try:
            from agentic_rag.config.settings import get_settings
            self._fallback_dim = get_settings().embedding.dim
        except Exception:
            pass

        # Backward compat alias
        self.embedding_func = embedding_func

    async def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 0,
        modality_filter: list[ContentType] = None,
        query_image: str = "",
    ) -> list[RetrievalResult]:
        """Retrieve relevant content items.

        Args:
            query: Search query text.
            mode: "naive", "local", "global", or "hybrid".
            top_k: Override default top_k.
            modality_filter: Only return specific content types.
            query_image: Optional image path for cross-modal visual search.

        Returns:
            Ranked list of RetrievalResults.
        """
        k = top_k or self.top_k

        # Build multimodal query input
        from agentic_rag.services.knowledge.embedding import EmbeddingInput
        query_input = EmbeddingInput.from_text(query)
        if query_image:
            query_input.image_path = query_image

        if mode == "naive":
            return await self._vector_search_with_input(query_input, k, modality_filter)

        # Default (hybrid or anything else): vector + graph fusion
        return await self._hybrid_search_with_input(query_input, k, modality_filter)

    # ── Input-aware search methods (primary) ──────

    async def _vector_search_with_input(
        self, query_input: "EmbeddingInput", k: int,
        modality_filter: list[ContentType] = None,
    ) -> list[RetrievalResult]:
        """Pure vector similarity search (multimodal-aware)."""
        if self.vector_store is None:
            return await self._fallback_search("", k, modality_filter)

        query_embedding = await self._embed(query_input)
        results = await self.vector_store.search(
            collection="knowledge",
            query_embedding=query_embedding,
            top_k=k,
        )

        retrieval_results = []
        seen_texts: set[str] = set()
        for doc in results:
            if modality_filter and not self._matches_filter(doc, modality_filter):
                continue
            item = self._to_content_item(doc)
            fingerprint = " ".join(item.to_searchable_text()[:120].lower().split())
            if fingerprint and fingerprint in seen_texts:
                continue
            seen_texts.add(fingerprint)
            retrieval_results.append(RetrievalResult(
                content_item=item,
                score=getattr(doc, 'score', 0.0),
                source="vector",
                entity_id=self._doc_source_info(doc)[2],
                doc_id=self._doc_source_info(doc)[0],
                source_file=self._doc_source_info(doc)[1],
            ))
        return retrieval_results

    async def _local_search_with_input(
        self, query_input: "EmbeddingInput", k: int,
        modality_filter: list[ContentType] = None,
    ) -> list[RetrievalResult]:
        """Vector search + 1-hop graph neighbors."""
        vector_results = await self._vector_search_with_input(
            query_input, k * 2, modality_filter,
        )
        if not self.graph:
            return vector_results[:k]

        expanded = []
        seen = set()
        for r in vector_results:
            if r.entity_id not in seen:
                expanded.append(r)
                seen.add(r.entity_id)
            neighbors = self.graph.get_neighbors(r.entity_id)
            for neighbor in neighbors:
                if neighbor.id not in seen and neighbor.content_item:
                    seen.add(neighbor.id)
                    expanded.append(RetrievalResult(
                        content_item=neighbor.content_item,
                        score=r.score * 0.7,
                        entity_id=neighbor.id,
                        source="graph",
                        doc_id=neighbor.content_item.metadata.get("source", "") or neighbor.content_item.metadata.get("doc_id", ""),
                        source_file=neighbor.content_item.metadata.get("image_path", "") or neighbor.content_item.metadata.get("video_path", "") or neighbor.content_item.metadata.get("audio_path", "") or "",
                    ))

        expanded.sort(key=lambda x: x.score, reverse=True)  # similarity: higher = better
        return expanded[:k]

    async def _global_search_with_input(
        self, query_input: "EmbeddingInput", k: int,
        modality_filter: list[ContentType] = None,
    ) -> list[RetrievalResult]:
        """Vector search + 2-hop graph traversal."""
        vector_results = await self._vector_search_with_input(
            query_input, k, modality_filter,
        )
        if not self.graph:
            return vector_results

        expanded = []
        seen = set()
        for r in vector_results:
            if r.entity_id not in seen:
                expanded.append(r)
                seen.add(r.entity_id)
            traversed = self.graph.traverse(r.entity_id, max_depth=2)
            for entity in traversed:
                if entity.id not in seen and entity.content_item:
                    seen.add(entity.id)
                    expanded.append(RetrievalResult(
                        content_item=entity.content_item,
                        score=r.score * 0.5,
                        entity_id=entity.id,
                        source="graph",
                        doc_id=entity.content_item.metadata.get("source", "") or entity.content_item.metadata.get("doc_id", ""),
                        source_file=entity.content_item.metadata.get("image_path", "") or entity.content_item.metadata.get("video_path", "") or entity.content_item.metadata.get("audio_path", "") or "",
                    ))

        expanded.sort(key=lambda x: x.score, reverse=True)  # similarity: higher = better
        return expanded[:k]

    async def _hybrid_search_with_input(
        self, query_input: "EmbeddingInput", k: int,
        modality_filter: list[ContentType] = None,
    ) -> list[RetrievalResult]:
        """Fuse independent dense, BM25, and optional graph rankings with RRF."""
        candidate_count = max(k * 6, 30)
        query_text = query_input.text if hasattr(query_input, "text") else str(query_input)
        vector_results, bm25_results = await asyncio.gather(
            self._vector_search_with_input(query_input, candidate_count, modality_filter),
            self._full_text_search(query_text, candidate_count, modality_filter),
        )
        base_results = self._rrf_fuse(
            [vector_results, bm25_results],
            candidate_count,
            weights=[1.0, 1.0],
        )

        if not self.graph:
            return self._dedupe_results(base_results)[:k]

        semantic_edges = {
            "appears_in", "describes", "inverse_appears_in", "inverse_describes",
        }
        graph_results: list[RetrievalResult] = []
        seen_graph_ids = set()
        for result in base_results[:max(k * 2, 10)]:
            if not result.entity_id:
                continue
            for neighbor in self.graph.get_neighbors(result.entity_id):
                if not neighbor.content_item or neighbor.id in seen_graph_ids:
                    continue
                is_semantic = any(
                    relation.type in semantic_edges
                    for relation in self.graph._adjacency.get(result.entity_id, [])
                    if relation.target_id == neighbor.id
                )
                if not is_semantic:
                    continue
                seen_graph_ids.add(neighbor.id)
                metadata = neighbor.content_item.metadata or {}
                graph_results.append(RetrievalResult(
                    content_item=neighbor.content_item,
                    score=0.0,
                    entity_id=neighbor.id,
                    source="graph",
                    doc_id=metadata.get("source", "") or metadata.get("doc_id", ""),
                    source_file=(metadata.get("image_path", "")
                                 or metadata.get("video_path", "")
                                 or metadata.get("audio_path", "")),
                ))

        fused = self._rrf_fuse(
            [vector_results, bm25_results, graph_results],
            candidate_count,
            weights=[1.0, 1.0, self.graph_weight],
        )
        return self._dedupe_results(fused)[:k]

    async def _full_text_search(
        self, query: str, k: int,
        modality_filter: list[ContentType] = None,
    ) -> list[RetrievalResult]:
        search = getattr(self.vector_store, "full_text_search", None)
        if search is None:
            return []
        content_types = [item.value for item in modality_filter] if modality_filter else None
        try:
            docs = await search(query=query, top_k=k, content_types=content_types)
        except Exception as exc:
            print(f"  [Hybrid] BM25 search unavailable: {exc}", flush=True)
            return []
        results = []
        for doc in docs:
            item = self._to_content_item(doc)
            doc_id, source_file, entity_id = self._doc_source_info(doc)
            results.append(RetrievalResult(
                content_item=item,
                score=getattr(doc, "score", 0.0),
                source="bm25",
                entity_id=entity_id,
                doc_id=doc_id,
                source_file=source_file,
            ))
        return results

    @classmethod
    def _rrf_fuse(
        cls, rankings: list[list[RetrievalResult]], limit: int,
        weights: list[float] | None = None, rrf_k: int = 60,
    ) -> list[RetrievalResult]:
        """Reciprocal Rank Fusion over incomparable retrieval score spaces."""
        weights = weights or [1.0] * len(rankings)
        active = [(ranking, weights[index]) for index, ranking in enumerate(rankings) if ranking]
        if not active:
            return []
        scores: dict[str, float] = {}
        selected: dict[str, RetrievalResult] = {}
        sources: dict[str, set[str]] = {}
        for ranking, weight in active:
            for rank, result in enumerate(ranking, start=1):
                key = cls._result_key(result)
                scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
                selected.setdefault(key, result)
                sources.setdefault(key, set()).add(result.source)

        max_score = sum(weight for _, weight in active) / (rrf_k + 1)
        ordered = sorted(scores, key=scores.get, reverse=True)
        fused = []
        for key in ordered[:limit]:
            result = selected[key]
            result.score = scores[key] / max_score if max_score else 0.0
            result.source = "+".join(sorted(sources[key]))
            fused.append(result)
        return fused

    @staticmethod
    def _result_key(result: RetrievalResult) -> str:
        if result.entity_id:
            return f"entity:{result.entity_id}"
        text = " ".join(result.content_item.to_searchable_text().lower().split())
        return f"text:{result.doc_id}:{text[:300]}"

    @staticmethod
    def _dedupe_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
        deduped = []
        seen = set()
        for result in results:
            text = " ".join(result.content_item.to_searchable_text().lower().split())
            fingerprint = text[:160]
            if fingerprint and fingerprint in seen:
                continue
            if fingerprint:
                seen.add(fingerprint)
            deduped.append(result)
        return deduped

    # ── Backward-compatible text-only wrappers ────

    async def _vector_search(self, query: str, k: int,
                              modality_filter: list[ContentType] = None) -> list[RetrievalResult]:
        """Pure vector similarity search (text query, backward compat)."""
        from agentic_rag.services.knowledge.embedding import EmbeddingInput
        return await self._vector_search_with_input(
            EmbeddingInput.from_text(query), k, modality_filter,
        )

    async def _local_search(self, query: str, k: int,
                             modality_filter: list[ContentType] = None) -> list[RetrievalResult]:
        from agentic_rag.services.knowledge.embedding import EmbeddingInput
        return await self._local_search_with_input(
            EmbeddingInput.from_text(query), k, modality_filter,
        )

    async def _global_search(self, query: str, k: int,
                              modality_filter: list[ContentType] = None) -> list[RetrievalResult]:
        from agentic_rag.services.knowledge.embedding import EmbeddingInput
        return await self._global_search_with_input(
            EmbeddingInput.from_text(query), k, modality_filter,
        )

    async def _hybrid_search(self, query: str, k: int,
                              modality_filter: list[ContentType] = None) -> list[RetrievalResult]:
        from agentic_rag.services.knowledge.embedding import EmbeddingInput
        return await self._hybrid_search_with_input(
            EmbeddingInput.from_text(query), k, modality_filter,
        )

    # ── Embedding ─────────────────────────────────

    async def _embed(self, query: Union[str, "EmbeddingInput"]) -> list[float]:
        """Get embedding for a query, which can be text or multimodal."""
        from agentic_rag.services.knowledge.embedding import EmbeddingInput

        if isinstance(query, str):
            query = EmbeddingInput.from_text(query)

        if self._adapter:
            return await self._adapter.embed_query(query)

        # Fallback: no embedding function configured.
        # Return a zero vector with the correct dimension so the search doesn't
        # crash, but results will be meaningless — caller should fix their config.
        import warnings
        warnings.warn(
            "HybridRetriever._embed(): no embedding function configured. "
            f"Returning zero vector (dim={self._fallback_dim}). "
            "Set embedding_func or embedding_adapter to get meaningful results.",
            RuntimeWarning,
        )
        return [0.0] * self._fallback_dim

    async def _fallback_search(self, query: str, k: int,
                                modality_filter=None) -> list[RetrievalResult]:
        """Fallback when no vector store is available."""
        return []

    def _to_content_item(self, doc: Any) -> ContentItem:
        """Convert vector store document to ContentItem.

        Reconstructs the correct ContentType from stored metadata so that
        downstream components can distinguish images, tables, etc.
        """
        if hasattr(doc, 'text'):
            text = doc.text
        elif isinstance(doc, dict):
            text = doc.get("text", "")
        else:
            text = str(doc)

        metadata = getattr(doc, 'metadata', {}) or {}
        if isinstance(metadata, str):
            metadata = {}

        # Reconstruct the correct content type from stored metadata
        content_type_str = metadata.get("content_type", "text")
        try:
            ctype = ContentType(content_type_str)
        except ValueError:
            ctype = ContentType.TEXT

        return ContentItem(
            type=ctype,
            text=text[:2000] if text else "",
            img_path=metadata.get("image_path", ""),
            video_path=metadata.get("video_path", ""),
            audio_path=metadata.get("audio_path", ""),
            metadata=metadata,
        )

    def _doc_source_info(self, doc: Any) -> tuple[str, str, str]:
        """Extract (doc_id, source_file, entity_id) from a vector-store result doc."""
        meta = getattr(doc, 'metadata', {}) or {}
        if isinstance(meta, str):
            meta = {}
        # The Milvus adapter stores doc_id as metadata.source
        doc_id = meta.get("source", "") or meta.get("doc_id", "")
        source_file = meta.get("image_path", "") or meta.get("video_path", "") or meta.get("audio_path", "") or ""
        entity_id = meta.get("entity_id", "") or getattr(doc, 'entity_id', "") or ""
        return doc_id, source_file, entity_id

    def _matches_filter(self, doc: Any, modality_filter: list[ContentType]) -> bool:
        """Check if a document matches the modality filter."""
        if not modality_filter:
            return True
        if not hasattr(doc, 'metadata'):
            return True
        doc_type = doc.metadata.get("content_type", "text")
        return any(ct.value == doc_type for ct in modality_filter)
