"""RAG endpoints — knowledge base query, ingestion, and file upload."""

import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

router = APIRouter()


def _configure_chunking(pipe, chunk_size: int, chunk_overlap: int) -> None:
    """Apply validated per-ingestion chunk settings to the shared pipeline."""
    if chunk_size < 1:
        raise HTTPException(status_code=400, detail="chunk_size must be greater than 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap must be non-negative and smaller than chunk_size",
        )
    pipe.chunk_size = int(chunk_size)
    pipe.chunk_overlap = int(chunk_overlap)
    # Processors capture these values at construction time.
    pipe._processors = {}


def _parse_doc_id(result: str) -> str:
    """Extract the UUID document ID from an ingestion result message.

    The RAGIngestTool returns a multi-line string like::

        Content ingested successfully.
        Document ID: abc123-def456-...
        Source: upload

    We only want the UUID, not the trailing "Source: ..." text.
    """
    if "ID:" in result:
        # Grab the line containing "ID:" and extract the UUID
        for line in result.split("\n"):
            if "ID:" in line:
                return line.split("ID:")[-1].strip()
    return "unknown"


@router.get("/rag/stats")
async def rag_stats():
    """Return knowledge base statistics."""
    from agentic_rag.services.knowledge.pipeline import get_knowledge_pipeline
    pipe = get_knowledge_pipeline()
    vs_stats = {}
    if pipe.vector_store and hasattr(pipe.vector_store, 'stats'):
        try:
            vs_stats = pipe.vector_store.stats()
        except Exception:
            pass

    # Count from Milvus (persistent) + pipeline (in-memory)
    chunk_count = 0
    doc_count = len(pipe._documents)
    try:
        from pymilvus import MilvusClient
        from agentic_rag.config.settings import get_settings
        ws = get_settings().workspace_dir
        mc = MilvusClient(f"{ws}/milvus_lite.db")
        if "knowledge" in (mc.list_collections() or []):
            mc.load_collection("knowledge")
            # Count chunks
            res = mc.query("knowledge", filter="id != ''", output_fields=["id", "source"])
            chunk_count = len(res) if res else 0
            # Count distinct documents (by source field)
            if res:
                sources = set(r.get("source", "") for r in res if r.get("source"))
                doc_count = max(doc_count, len(sources))
    except Exception:
        pass

    kg_stats = pipe.graph.stats() if pipe.graph else {}
    return {
        "chunks": chunk_count,
        "documents": doc_count,
        "graph": kg_stats,
        "vector_store": vs_stats,
    }


@router.get("/rag/documents")
async def rag_documents():
    """List all indexed documents with metadata (server-side, browser-agnostic)."""
    from agentic_rag.services.knowledge.pipeline import get_knowledge_pipeline
    pipe = get_knowledge_pipeline()
    return {"documents": pipe.list_files()}


@router.get("/rag/graph")
async def rag_graph():
    """Return the full knowledge graph (entities + relations) for visualization."""
    from agentic_rag.services.knowledge.pipeline import get_knowledge_pipeline
    pipe = get_knowledge_pipeline()

    if not pipe.graph:
        return {"entities": [], "relations": [], "stats": {"entities": 0, "relations": 0}}

    graph_dict = pipe.graph.to_dict()
    return {
        "entities": [
            {
                "id": eid,
                "name": ent["name"][:80],
                "type": ent["type"],
                "content_type": ent.get("content_type", ""),
                "text_preview": ent.get("content_text", "")[:120],
            }
            for eid, ent in graph_dict.get("entities", {}).items()
        ],
        "relations": [
            {
                "source": r["source"][:12],
                "target": r["target"][:12],
                "type": r["type"],
                "weight": round(r.get("weight", 1.0), 2),
            }
            for r in graph_dict.get("relations", [])
        ],
        "stats": pipe.graph.stats(),
    }


class RAGSearchRequest(BaseModel):
    """Raw RAG search request — no LLM generation."""
    query: str
    top_k: int = 3
    mode: str = "hybrid"
    modality_filter: list[str] = None


class RAGQueryRequest(BaseModel):
    """RAG query request."""
    query: str
    top_k: int = 3
    include_raw_docs: bool = False


class RAGQueryResponse(BaseModel):
    """RAG query response."""
    answer: str
    sources: list[dict] = []


class RAGIngestRequest(BaseModel):
    """Document ingestion request."""
    content: str
    source: str = "api"
    metadata: dict = {}
    chunk_size: int = 800
    chunk_overlap: int = 150


class RAGIngestResponse(BaseModel):
    """Document ingestion response."""
    doc_id: str
    status: str


@router.post("/rag/search")
async def rag_search(req: RAGSearchRequest):
    """Search the knowledge base — returns raw chunks, no LLM generation."""
    from agentic_rag.services.knowledge.pipeline import get_knowledge_pipeline
    from agentic_rag.services.knowledge.content_list import ContentType
    pipe = get_knowledge_pipeline()
    # Parse modality filter
    mf = None
    if req.modality_filter:
        mf = [ContentType(t) for t in req.modality_filter if t in ContentType.__members__.values()]
        mf = mf or None
    results = await pipe.retrieve(query=req.query, top_k=req.top_k, mode=req.mode, modality_filter=mf)
    # Resolve doc_id → display name
    doc_names = {}
    registry_names = {
        item["doc_id"]: item["name"]
        for item in pipe.list_files()
        if item.get("doc_id") and item.get("name")
    }
    for r in results:
        meta = r.content_item.metadata or {}
        did = meta.get("source", "")  # Milvus "source" field = doc_id
        if did and did not in doc_names and len(did) > 20:
            cl = pipe._documents.get(did)
            raw = registry_names.get(did) or (cl.source if cl else did[:12])
            # Show filename only, not full path
            doc_names[did] = raw.split("/")[-1] if "/" in raw else raw

    # Filter out very low-score results (embedding noise)
    MIN_SCORE = 0.25
    filtered = [r for r in results if r.score >= MIN_SCORE]

    return {
        "query": req.query,
        "results": [
            {
                "content": r.content_item.to_searchable_text() or r.content_item.text or "",
                "content_type": r.content_item.type.value,
                "score": r.score,
                "source": r.source,
                "document": doc_names.get(r.content_item.metadata.get("source", ""), ""),
                "image_path": r.content_item.img_path or "",
                "video_path": r.content_item.video_path or "",
                "audio_path": r.content_item.audio_path or "",
                "table_body": r.content_item.table_body or "",
                "page_idx": r.content_item.page_idx,
            }
            for r in filtered
        ],
    }


@router.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(req: RAGQueryRequest):
    """Query the knowledge base with RAG."""
    from agentic_rag.orchestration.l1_tools.rag_tools import RAGSearchTool
    from agentic_rag.services.llm.factory import get_llm
    from agentic_rag.data.models import Message

    llm = get_llm()
    tool = RAGSearchTool()
    results = await tool.execute(query=req.query, top_k=req.top_k)

    # Generate answer from retrieved context
    messages = [
        Message.system("Answer the user's question using the provided context. Cite sources when possible."),
        Message.user(f"Context:\n{results}\n\nQuestion: {req.query}"),
    ]
    response = await llm.agenerate(messages)

    return RAGQueryResponse(
        answer=response.content,
        sources=[{"content": results}],
    )


@router.get("/rag/file/{file_path:path}")
async def serve_kb_file(file_path: str):
    """Serve a file from the knowledge base (for image/video preview)."""
    from pathlib import Path as _Path
    from fastapi.responses import FileResponse
    p = _Path(file_path)
    if not p.is_absolute():
        p = _Path.cwd() / file_path
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    return FileResponse(str(p))


@router.get("/rag/document/{doc_id}")
async def get_document(doc_id: str):
    """Get the content of an indexed document for preview."""
    from agentic_rag.services.knowledge.pipeline import get_knowledge_pipeline
    pipe = get_knowledge_pipeline()

    # Try in-memory first
    cl = pipe._documents.get(doc_id)
    if cl:
        items = []
        for item in cl.items:
            items.append({
                "text": item.to_searchable_text() or item.text or "",
                "type": item.type.value,
                "page_idx": item.page_idx,
                "image_path": item.img_path or "",
                "video_path": item.video_path or "",
                "audio_path": item.audio_path or "",
                "table_body": item.table_body or "",
            })
        return {"doc_id": doc_id, "source": cl.source, "items": items}

    # Fallback: query Milvus by source (works after server restart)
    items = []
    try:
        from agentic_rag.config.settings import get_settings
        from pymilvus import MilvusClient
        ws = get_settings().workspace_dir
        mc = MilvusClient(f"{ws}/milvus_lite.db")
        if "knowledge" in (mc.list_collections() or []):
            mc.load_collection("knowledge")
            res = mc.query("knowledge", filter=f'source == "{doc_id}"', output_fields=["text", "content_type", "image_path", "video_path", "audio_path", "table_body"], limit=100)
            for r in res:
                items.append({
                    "text": r.get("text", ""),
                    "type": r.get("content_type", "text"),
                    "image_path": r.get("image_path", ""),
                    "video_path": r.get("video_path", ""),
                    "audio_path": r.get("audio_path", ""),
                    "table_body": r.get("table_body", ""),
                })
    except Exception as e:
        print(f"  [rag] document query error: {e}", flush=True)

    if not items:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return {"doc_id": doc_id, "items": items}


@router.delete("/rag/document/{doc_id}")
async def delete_document(doc_id: str):
    """Delete an indexed document and all its chunks from the knowledge base."""
    deleted_chunks = 0
    try:
        from agentic_rag.config.settings import get_settings
        from pymilvus import MilvusClient
        ws = get_settings().workspace_dir
        mc = MilvusClient(f"{ws}/milvus_lite.db")
        if "knowledge" in (mc.list_collections() or []):
            mc.load_collection("knowledge")
            res = mc.query("knowledge", filter=f'source == "{doc_id}"', output_fields=["id"], limit=1000)
            ids = [r["id"] for r in res]
            if ids:
                mc.delete("knowledge", ids=ids)
                deleted_chunks = len(ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {e}")

    # Also remove from in-memory pipeline
    from agentic_rag.services.knowledge.pipeline import get_knowledge_pipeline
    pipe = get_knowledge_pipeline()
    pipe._documents.pop(doc_id, None)
    full_text_delete = getattr(pipe.vector_store, "delete_full_text_by_source", None)
    if full_text_delete:
        await full_text_delete(doc_id)
    pipe.unregister_file(doc_id)

    return {"doc_id": doc_id, "deleted_chunks": deleted_chunks, "status": "deleted"}


@router.post("/rag/clear")
async def clear_knowledge_base():
    """Clear ALL documents and chunks from the knowledge base."""
    from agentic_rag.config.settings import get_settings
    from pymilvus import MilvusClient
    ws = get_settings().workspace_dir
    mc = MilvusClient(f"{ws}/milvus_lite.db")
    if "knowledge" in (mc.list_collections() or []):
        mc.drop_collection("knowledge")
    from pymilvus import DataType
    dim = get_settings().embedding.dim
    schema = mc.create_schema(enable_dynamic_field=True)
    schema.add_field("id", DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    mc.create_collection("knowledge", schema=schema, dimension=dim)
    mc.load_collection("knowledge")

    # Clear in-memory state
    from agentic_rag.services.knowledge.pipeline import get_knowledge_pipeline
    from agentic_rag.services.knowledge.graph.index import KnowledgeGraph
    pipe = get_knowledge_pipeline()
    pipe._documents.clear()
    pipe._content_cache.clear()
    full_text_clear = getattr(pipe.vector_store, "clear_full_text", None)
    if full_text_clear:
        await full_text_clear()
    pipe.clear_file_registry()
    # Reset knowledge graph
    pipe.graph = KnowledgeGraph()
    # Delete persisted graph file
    import os
    gpath = pipe._graph_path
    if os.path.exists(gpath):
        os.remove(gpath)

    return {"status": "cleared", "message": "All knowledge base data has been removed."}


@router.post("/rag/ingest", response_model=RAGIngestResponse)
async def rag_ingest(req: RAGIngestRequest):
    """Ingest a document into the knowledge base."""
    from agentic_rag.orchestration.l1_tools.rag_tools import RAGIngestTool
    from agentic_rag.services.knowledge.pipeline import get_knowledge_pipeline

    pipe = get_knowledge_pipeline()
    _configure_chunking(pipe, req.chunk_size, req.chunk_overlap)
    tool = RAGIngestTool(pipeline=pipe)
    result = await tool.execute(content=req.content, source=req.source)

    doc_id = _parse_doc_id(result)
    # Register for cross-browser document listing
    pipe.register_file(doc_id, req.source or "text_ingest", len(req.content), "text")

    return RAGIngestResponse(
        doc_id=doc_id,
        status="success",
    )


# ── File Upload ─────────────────────────────────────────────────

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
    ".jpg": "image", ".jpeg": "image", ".png": "image",
    ".gif": "image", ".webp": "image", ".bmp": "image", ".svg": "image",
    ".mp4": "video", ".avi": "video", ".mov": "video",
    ".mkv": "video", ".webm": "video",
    ".mp3": "audio", ".wav": "audio", ".m4a": "audio",
    ".ogg": "audio", ".flac": "audio",
}


@router.post("/rag/upload")
async def rag_upload(
    file: UploadFile = File(...),
    source: str = Form("web_ui"),
    ingest_mode: str = Form("multimodal"),
    mm_method: str = Form("pure"),
    chunk_size: int = Form(800),
    chunk_overlap: int = Form(150),
    enable_kg: bool = Form(False),
):
    """Upload a file for ingestion into the knowledge base.

    Supports text files (.txt, .md, .pdf, .json, .yaml, .csv),
    images (.jpg, .png, .gif, .webp), video (.mp4, .avi, .mov),
    and audio (.mp3, .wav, .m4a).
    """
    import sys
    print(f"  [Upload] received: {file.filename} ({file.content_type}), source={source}, "
          f"chunk={chunk_size}/{chunk_overlap}, kg={enable_kg}", flush=True)

    # Detect content category
    mime = file.content_type or ""
    ext = Path(file.filename or "").suffix.lower()

    category = MIME_TO_CATEGORY.get(mime) or EXT_TO_CATEGORY.get(ext)
    if category is None:
        # Fallback: try to read as text
        category = "text"

    # Save file to workspace temp directory
    from agentic_rag.config.settings import get_settings
    settings = get_settings()
    workspace = Path(settings.workspace_dir) / "uploads"
    workspace.mkdir(parents=True, exist_ok=True)

    safe_name = f"{uuid.uuid4().hex}_{file.filename or 'upload'}"
    file_path = workspace / safe_name

    try:
        content_bytes = await file.read()
        file_path.write_bytes(content_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    file_size = len(content_bytes)

    # Ingest based on category
    from agentic_rag.orchestration.l1_tools.rag_tools import RAGIngestTool, RAGMultiModalIngestTool
    from agentic_rag.services.knowledge.pipeline import get_knowledge_pipeline

    try:
        # Apply per-request pipeline settings
        pipeline = get_knowledge_pipeline()
        # Ensure bool conversion (FastAPI Form may deliver "true"/"false" strings)
        kg_enabled = enable_kg if isinstance(enable_kg, bool) else str(enable_kg).lower() in ("true", "1", "yes", "on")
        pipeline.ingest_mode = ingest_mode
        pipeline.mm_method = mm_method
        pipeline.enable_kg = kg_enabled
        pipeline.extract_entities = kg_enabled  # semantic extraction follows KG toggle
        _configure_chunking(pipeline, chunk_size, chunk_overlap)
        # Ensure LLM function is available for semantic extraction
        if pipeline.extract_entities and pipeline.llm_func is None:
            from agentic_rag.services.knowledge.pipeline import _create_kg_llm_func
            pipeline.llm_func = _create_kg_llm_func()
        previous_doc_ids = [
            item["doc_id"]
            for item in pipeline.list_files()
            if item.get("name") == (file.filename or "unknown")
        ]

        print(f"  [Upload] pipeline config: mode={ingest_mode}/{mm_method}, "
              f"chunk={pipeline.chunk_size}/{pipeline.chunk_overlap}, "
              f"kg={pipeline.enable_kg}, extract_entities={pipeline.extract_entities}, "
              f"llm={'OK' if pipeline.llm_func else 'MISSING'}", flush=True)

        if category == "text":
            # PDF / docx / office files → parse via pipeline (supports docling + fallback)
            if ext in (".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"):
                print(f"  [Upload] parsing document via pipeline: {ext}", flush=True)
                doc_id = await pipeline.ingest(source=str(file_path), source_type=ext.lstrip("."))
                # Override stored source with original filename for display
                cl = pipeline._documents.get(doc_id)
                if cl:
                    cl.source = file.filename or str(file_path)
                result = f"Content ingested successfully.\nDocument ID: {doc_id}\nSource: {source}"
            else:
                # Plain text files — pass pipeline explicitly so KG settings take effect
                try:
                    text_content = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    text_content = content_bytes.decode("latin-1", errors="replace")

                print(f"  [Upload] text content: {len(text_content)} chars, "
                      f"first 80: {text_content[:80].replace(chr(10),' ')}", flush=True)

                tool = RAGIngestTool(pipeline=pipeline)
                result = await tool.execute(
                    content=text_content,
                    source=source or file.filename or "upload",
                    content_type="text",
                )
            print(f"  [Upload] result: {str(result)[:120]}", flush=True)
            print(f"  [Upload] KG after ingest: entities={pipeline.graph.entity_count if pipeline.graph else 0}", flush=True)

        elif category == "image":
            tool = RAGMultiModalIngestTool(pipeline=pipeline)
            result = await tool.execute(
                images=[str(file_path)],
                source=source or file.filename or "upload",
            )

        elif category == "video":
            tool = RAGMultiModalIngestTool(pipeline=pipeline)
            result = await tool.execute(
                videos=[str(file_path)],
                source=source or file.filename or "upload",
            )

        elif category == "audio":
            tool = RAGMultiModalIngestTool(pipeline=pipeline)
            result = await tool.execute(
                audio_files=[str(file_path)],
                source=source or file.filename or "upload",
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file category: {category}")

        doc_id = _parse_doc_id(str(result))

        # Register file metadata for cross-browser document listing
        pipeline.register_file(doc_id, file.filename or "unknown", file_size, category)

        # Upsert replaces chunks that kept the same ID. Remove any old chunks
        # whose boundaries changed, then retire the previous document records.
        for previous_doc_id in previous_doc_ids:
            if previous_doc_id == doc_id:
                continue
            await pipeline.delete_document_vectors(previous_doc_id)
            pipeline._documents.pop(previous_doc_id, None)
            pipeline.unregister_file(previous_doc_id)

        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "content_type": category,
            "size": file_size,
            "status": "success",
        }

    except HTTPException:
        raise
    except Exception as e:
        # Cleanup temp file on failure
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
