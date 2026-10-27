"""RAG endpoints — knowledge base query, ingestion, and file upload.

Handlers stay thin: validation and HTTP mapping here, knowledge-base business
logic in :class:`KnowledgeBaseService`.
"""

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agentic_rag.services.knowledge.service import (
    IngestionFailed,
    KnowledgeBaseService,
    UnsupportedUpload,
    UploadFailed,
    UploadTooLarge,
)
from agentic_rag.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _service() -> KnowledgeBaseService:
    # The search tool lives in the orchestration layer; inject it so the
    # knowledge service itself does not depend on tool packages.
    from agentic_rag.orchestration.l1_tools.rag_tools import RAGSearchTool
    return KnowledgeBaseService(tool_factories={"search": RAGSearchTool})


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


@router.get("/rag/stats")
async def rag_stats():
    """Return knowledge base statistics."""
    return await _service().stats()


@router.get("/rag/documents")
async def rag_documents():
    """List all indexed documents with metadata (browser-agnostic)."""
    return {"documents": _service().list_files()}


@router.get("/rag/graph")
async def rag_graph():
    """Return the full knowledge graph for visualization."""
    return await _service().graph()


@router.post("/rag/search")
async def rag_search(req: RAGSearchRequest):
    """Search the knowledge base — raw chunks, no LLM generation."""
    return await _service().search(
        req.query,
        top_k=req.top_k,
        mode=req.mode,
        modality_filter=req.modality_filter,
    )


@router.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(req: RAGQueryRequest):
    """Query the knowledge base with RAG."""
    try:
        result = await _service().query(req.query, top_k=req.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return RAGQueryResponse(**result)


@router.get("/rag/file/{file_path:path}")
async def serve_kb_file(file_path: str):
    """Serve a file from the knowledge base (for image/video preview)."""
    path = Path(file_path)
    if not path.is_absolute():
        path = Path.cwd() / file_path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    return FileResponse(str(path))


@router.get("/rag/document/{doc_id}")
async def get_document(doc_id: str):
    """Get the content of an indexed document for preview."""
    document = await _service().document_items(doc_id)
    if document is None:
        raise HTTPException(
            status_code=404, detail=f"Document {doc_id} not found",
        )
    return document


@router.delete("/rag/document/{doc_id}")
async def delete_document(doc_id: str):
    """Delete an indexed document and all its chunks."""
    try:
        return await _service().delete_document(doc_id)
    except Exception as exc:
        logger.exception("Deletion failed for %s", doc_id)
        raise HTTPException(status_code=500, detail="Deletion failed") from exc


@router.post("/rag/clear")
async def clear_knowledge_base():
    """Clear ALL documents and chunks from the knowledge base."""
    try:
        await _service().clear()
    except Exception as exc:
        logger.exception("Failed to clear the knowledge base")
        raise HTTPException(
            status_code=500, detail="Failed to clear the knowledge base",
        ) from exc
    return {
        "status": "cleared",
        "message": "All knowledge base data has been removed.",
    }


@router.post("/rag/ingest", response_model=RAGIngestResponse)
async def rag_ingest(req: RAGIngestRequest):
    """Ingest a document into the knowledge base."""
    try:
        result = await _service().ingest_text(
            req.content, source=req.source,
            chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RAGIngestResponse(**result)


@router.post("/rag/upload")
async def rag_upload(
    file: UploadFile = File(...),
    source: str = Form("web_ui"),
    ingest_mode: str = Form("multimodal"),
    mm_method: str = Form("caption"),
    chunk_size: int = Form(800),
    chunk_overlap: int = Form(150),
    enable_kg: bool = Form(False),
):
    """Upload a file for ingestion into the knowledge base."""
    try:
        return await _service().ingest_upload(
            file,
            source=source,
            ingest_mode=ingest_mode,
            mm_method=mm_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            enable_kg=enable_kg,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UploadTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except UnsupportedUpload as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UploadFailed as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except IngestionFailed as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
