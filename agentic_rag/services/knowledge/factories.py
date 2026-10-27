"""Factories for the knowledge pipeline's external dependencies.

These builders translate settings into embedding/vision/vector-store/LLM
callables.  They live outside ``pipeline.py`` so the pipeline itself stays
focused on document flow and the composition root can reuse the same builders.
"""

import asyncio
from pathlib import Path
from typing import Callable, Optional

from agentic_rag.utils.logging import get_logger

logger = get_logger(__name__)


def create_embedding_func():
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


def create_vision_func():
    """Build a VLM captioning function for knowledge-base media.

    Uses the same provider resolution as chat attachments, including
    ``media.vision_provider`` / ``media.vision_model`` overrides, instead of
    arbitrarily taking the first configured provider and its text model.
    """
    try:
        from agentic_rag.services.media.vision import VisionService

        service = VisionService()
        if not service.available:
            logger.warning("No vision-capable provider; captions disabled")
            return None

        async def vision_func(image_data: str, prompt: str) -> str:
            mime = "image/png"
            if image_data.startswith("data:") and ";" in image_data:
                mime = image_data[5:].split(";", 1)[0] or mime
            return await service.describe_image(
                image_data, prompt, mime, max_tokens=300,
            )

        return vision_func
    except Exception as exc:
        logger.exception("Vision setup failed")
        return None


def create_vector_store():
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
            def __init__(self, c, fts, col="knowledge", dim=None):
                self.c, self.col = c, col
                self.fts = fts
                self.dim = dim

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
                    logger.exception("Failed to delete source %s", doc_id)
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

            def stats(self) -> dict:
                """Return chunk and distinct-document counts from Milvus."""
                rows = self._all_rows()
                sources = {
                    row.get("source", "") for row in rows if row.get("source")
                }
                return {"chunks": len(rows), "documents": len(sources)}

            async def get_by_source(
                self, doc_id: str, fields: list[str] | None = None,
                limit: int = 1000,
            ) -> list[dict]:
                """Fetch stored rows for one document (used after a restart)."""
                fields = fields or [
                    "text", "content_type", "image_path", "video_path",
                    "audio_path", "table_body",
                ]
                try:
                    await asyncio.to_thread(self.c.load_collection, self.col)
                    rows = await asyncio.to_thread(
                        self.c.query,
                        self.col,
                        filter=f'source == "{doc_id}"',
                        output_fields=fields,
                        limit=limit,
                    )
                    return [dict(row) for row in (rows or [])]
                except Exception:
                    logger.exception("Failed to read source %s", doc_id)
                    return []

            async def clear_collection(self) -> None:
                """Drop and recreate the vector collection."""
                await asyncio.to_thread(self.c.drop_collection, self.col)
                await asyncio.to_thread(self.create_collection, self.col, self.dim)
                await asyncio.to_thread(self.c.load_collection, self.col)

            async def create_collection(self, name, dim):
                if name not in (self.c.list_collections() or []):
                    from pymilvus import DataType
                    schema = self.c.create_schema(enable_dynamic_field=True)
                    schema.add_field("id", DataType.VARCHAR, max_length=64, is_primary=True)
                    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
                    self.c.create_collection(name, schema=schema, dimension=dim)

        return MilvusAdapter(client, full_text_index, dim=dim)

    except ImportError:
        pass  # pymilvus not available
    except Exception:
        pass

    return None


def create_kg_llm_func():
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


