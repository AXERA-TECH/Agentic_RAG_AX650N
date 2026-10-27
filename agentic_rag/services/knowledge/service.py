"""Application service for knowledge-base management endpoints.

Routes should only translate HTTP in/out.  Classification, upload streaming,
option validation, ingestion orchestration, and index maintenance live here so
they can be tested without FastAPI and reused by other entrypoints.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Protocol

from agentic_rag.services.knowledge.content_list import ContentType
from agentic_rag.services.knowledge.ingest_options import IngestOptions
from agentic_rag.utils.logging import get_logger

logger = get_logger(__name__)

MIME_TO_CATEGORY = {
    # Text / documents
    "text/plain": "text",
    "text/markdown": "text",
    "text/csv": "text",
    "text/html": "text",
    "application/json": "text",
    "application/pdf": "text",
    "application/x-yaml": "text",
    # Images
    "image/jpeg": "image",
    "image/png": "image",
    "image/gif": "image",
    "image/webp": "image",
    "image/bmp": "image",
    "image/svg+xml": "image",
    # Video
    "video/mp4": "video",
    "video/avi": "video",
    "video/quicktime": "video",
    "video/x-matroska": "video",
    "video/webm": "video",
    # Audio
    "audio/mpeg": "audio",
    "audio/wav": "audio",
    "audio/mp4": "audio",
    "audio/ogg": "audio",
    "audio/flac": "audio",
    "audio/x-m4a": "audio",
}

EXT_TO_CATEGORY = {
    ".txt": "text", ".md": "text", ".markdown": "text",
    ".json": "text", ".yaml": "text", ".yml": "text",
    ".csv": "text", ".py": "text", ".html": "text",
    ".pdf": "text", ".docx": "text", ".doc": "text",
    ".pptx": "text", ".ppt": "text", ".xlsx": "text", ".xls": "text",
    ".jpg": "image", ".jpeg": "image", ".png": "image",
    ".gif": "image", ".webp": "image", ".bmp": "image", ".svg": "image",
    ".mp4": "video", ".avi": "video", ".mov": "video",
    ".mkv": "video", ".webm": "video",
    ".mp3": "audio", ".wav": "audio", ".m4a": "audio",
    ".ogg": "audio", ".flac": "audio",
}

_DOCUMENT_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
}


class UploadTooLarge(Exception):
    """The upload exceeded the configured byte limit."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        super().__init__(f"File exceeds the {max_bytes} byte upload limit")


class UnsupportedUpload(Exception):
    """The upload's category cannot be ingested."""


class UploadFailed(Exception):
    """The upload could not be written to the workspace."""


class IngestionFailed(Exception):
    """The pipeline failed while ingesting the upload."""


class UploadLike(Protocol):
    filename: str | None
    content_type: str | None

    async def read(self, size: int = -1) -> bytes: ...
    async def close(self) -> None: ...


def classify_upload(filename: str | None, content_type: str | None) -> str:
    """Map a MIME type or extension to pipeline category."""
    mime = content_type or ""
    ext = Path(filename or "").suffix.lower()
    return MIME_TO_CATEGORY.get(mime) or EXT_TO_CATEGORY.get(ext) or "text"


def parse_doc_id(result: str) -> str:
    """Extract the document UUID from a tool result message."""
    for line in str(result).split("\n"):
        if "ID:" in line:
            return line.split("ID:")[-1].strip()
    return "unknown"


class KnowledgeBaseService:
    """Knowledge-base operations shared by REST routes and other callers."""

    def __init__(
        self,
        pipeline=None,
        settings=None,
        llm_factory=None,
        tool_factories: dict | None = None,
    ):
        self._pipeline_instance = pipeline
        self._settings = settings
        self._llm_factory = llm_factory
        self._tool_factories = tool_factories or {}

    # ── Lazy dependencies ────────────────────────────────────

    def _pipeline(self):
        if self._pipeline_instance is None:
            from agentic_rag.services.knowledge.pipeline import (
                get_knowledge_pipeline,
            )
            self._pipeline_instance = get_knowledge_pipeline()
        return self._pipeline_instance

    def _get_settings(self):
        if self._settings is None:
            from agentic_rag.config.settings import get_settings
            self._settings = get_settings()
        return self._settings

    def _get_llm(self):
        if self._llm_factory is not None:
            return self._llm_factory()
        from agentic_rag.services.llm.factory import get_llm
        return get_llm()

    # ── Query / inspection ───────────────────────────────────
    async def stats(self) -> dict:
        return await self._pipeline().stats()

    async def search(
        self,
        query: str,
        top_k: int = 3,
        mode: str = "hybrid",
        modality_filter: list[str] | None = None,
    ) -> dict:
        pipeline = self._pipeline()
        content_types = None
        if modality_filter:
            parsed = []
            for value in modality_filter:
                try:
                    parsed.append(ContentType(value))
                except ValueError:
                    continue
            content_types = parsed or None

        results = await pipeline.retrieve(
            query=query, top_k=top_k, mode=mode,
            modality_filter=content_types,
        )
        registry_names = {
            item["doc_id"]: item["name"]
            for item in pipeline.list_files()
            if item.get("doc_id") and item.get("name")
        }

        doc_names: dict[str, str] = {}
        for result in results:
            metadata = result.content_item.metadata or {}
            doc_id = metadata.get("source", "")
            if doc_id and doc_id not in doc_names and len(doc_id) > 20:
                document = pipeline.get_document(doc_id)
                raw = registry_names.get(doc_id) or (
                    document.source if document else doc_id[:12]
                )
                doc_names[doc_id] = raw.split("/")[-1] if "/" in raw else raw

        # Filter out very low-score embedding noise.
        filtered = [result for result in results if result.score >= 0.25]
        return {
            "query": query,
            "results": [
                {
                    "content": (
                        result.content_item.to_searchable_text()
                        or result.content_item.text or ""
                    ),
                    "content_type": result.content_item.type.value,
                    "score": result.score,
                    "source": result.source,
                    "document": doc_names.get(
                        result.content_item.metadata.get("source", ""), "",
                    ),
                    "image_path": result.content_item.img_path or "",
                    "video_path": result.content_item.video_path or "",
                    "audio_path": result.content_item.audio_path or "",
                    "table_body": result.content_item.table_body or "",
                    "page_idx": result.content_item.page_idx,
                }
                for result in filtered
            ],
        }

    async def query(self, query: str, top_k: int = 3) -> dict:
        """Retrieve context and ask the configured LLM for an answer."""
        from agentic_rag.data.models import Message

        search_factory = self._tool_factories.get("search")
        if search_factory is None:
            raise RuntimeError(
                "KnowledgeBaseService.query requires a search tool factory"
            )
        tool = search_factory(pipeline=self._pipeline())
        results = await tool.execute(query=query, top_k=top_k)
        messages = [
            Message.system(
                "Answer the user's question using the provided context. "
                "Cite sources when possible."
            ),
            Message.user(f"Context:\n{results}\n\nQuestion: {query}"),
        ]
        response = await self._get_llm().agenerate(messages)
        return {
            "answer": response.content,
            "sources": [{"content": results}],
        }

    async def graph(self) -> dict:
        return await self._pipeline().graph_dict()

    def list_files(self) -> list[dict]:
        return self._pipeline().list_files()

    async def document_items(self, doc_id: str) -> dict | None:
        return await self._pipeline().get_document_items(doc_id)

    async def delete_document(self, doc_id: str) -> dict:
        deleted = await self._pipeline().delete_document(doc_id)
        return {
            "doc_id": doc_id,
            "deleted_chunks": deleted,
            "status": "deleted",
        }

    async def clear(self) -> None:
        await self._pipeline().clear_knowledge_base()

    # ── Ingestion ────────────────────────────────────────────

    async def ingest_text(
        self,
        content: str,
        source: str = "api",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ) -> dict:
        pipeline = self._pipeline()
        options = IngestOptions(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
        doc_id = await pipeline.ingest(
            source=source,
            content=content,
            source_type="auto",
            options=options,
        )
        pipeline.register_file(
            doc_id, source or "text_ingest", len(content), "text",
        )
        return {"doc_id": doc_id, "status": "success"}

    async def ingest_upload(
        self,
        upload: UploadLike,
        *,
        source: str = "web_ui",
        ingest_mode: str = "multimodal",
        mm_method: str = "caption",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        enable_kg: bool = False,
    ) -> dict:
        """Stream an upload to disk and run it through the pipeline."""
        filename = upload.filename or "upload"
        try:
            category = classify_upload(filename, upload.content_type)
            if category not in ("text", "image", "video", "audio"):
                raise UnsupportedUpload(
                    f"Unsupported file category: {category}"
                )

            kg_enabled = (
                enable_kg if isinstance(enable_kg, bool)
                else str(enable_kg).lower() in ("true", "1", "yes", "on")
            )
            # Validate options before touching the filesystem.
            options = IngestOptions(
                ingest_mode=ingest_mode,
                mm_method=mm_method,
                enable_kg=kg_enabled,
                extract_entities=kg_enabled,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception:
            await upload.close()
            raise

        settings = self._get_settings()
        workspace = Path(settings.workspace_dir) / "uploads"
        workspace.mkdir(parents=True, exist_ok=True)
        safe_name = f"{uuid.uuid4().hex}_{filename}"
        file_path = workspace / safe_name
        max_bytes = int(
            getattr(settings.api, "max_upload_bytes", 50 * 1024 * 1024)
            or 50 * 1024 * 1024
        )
        max_bytes = max(1, max_bytes)

        file_size = 0
        try:
            with file_path.open("wb") as handle:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    file_size += len(chunk)
                    if file_size > max_bytes:
                        raise UploadTooLarge(max_bytes)
                    handle.write(chunk)
        except UploadTooLarge:
            file_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            logger.exception("Failed to save uploaded file %s", file_path)
            file_path.unlink(missing_ok=True)
            raise UploadFailed("Failed to save uploaded file") from exc
        finally:
            await upload.close()

        pipeline = self._pipeline()
        previous_doc_ids = [
            item["doc_id"]
            for item in pipeline.list_files()
            if item.get("name") == filename
        ]

        ext = Path(filename).suffix.lower()
        try:
            logger.info(
                "[Upload] %s (%s bytes), category=%s, mode=%s/%s, chunk=%s/%s, kg=%s",
                filename, file_size, category, options.ingest_mode,
                options.mm_method, options.chunk_size, options.chunk_overlap,
                options.enable_kg,
            )

            if category == "text":
                if ext in _DOCUMENT_EXTENSIONS:
                    doc_id = await pipeline.ingest(
                        source=str(file_path),
                        source_type=ext.lstrip("."),
                        options=options,
                    )
                    pipeline.set_document_source(doc_id, filename)
                else:
                    try:
                        text_content = file_path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        text_content = file_path.read_text(
                            encoding="latin-1", errors="replace",
                        )
                    doc_id = await pipeline.ingest(
                        source=source or filename,
                        content=text_content,
                        source_type="auto",
                        options=options,
                    )
            else:
                argument = {
                    "image": "images",
                    "video": "videos",
                    "audio": "audio_files",
                }[category]
                doc_id = await pipeline.ingest_multimodal(
                    **{argument: [str(file_path)]},
                    source=source or filename,
                    options=options,
                )

            pipeline.register_file(doc_id, filename, file_size, category)

            # Upsert keeps chunks whose IDs are stable.  Retire older records
            # for the same filename when boundaries changed.
            for previous_doc_id in previous_doc_ids:
                if previous_doc_id != doc_id:
                    await pipeline.delete_document(previous_doc_id)

            return {
                "doc_id": doc_id,
                "filename": filename,
                "content_type": category,
                "size": file_size,
                "status": "success",
            }
        except Exception as exc:
            file_path.unlink(missing_ok=True)
            logger.exception("Ingestion failed for %s", filename)
            raise IngestionFailed("Ingestion failed") from exc
