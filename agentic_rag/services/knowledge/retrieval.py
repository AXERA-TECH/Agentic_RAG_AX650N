"""Knowledge retrieval service — manages RAG search and ingestion."""

import uuid
from typing import Optional

from agentic_rag.data.models import Document, RetrievalResult


class KnowledgeService:
    """Knowledge base service for retrieval and ingestion.

    This is a simplified in-memory implementation. It will be upgraded to
    use LlamaIndex + Milvus in Phase 3.
    """

    def __init__(self):
        self._documents: list[Document] = []

    async def search(self, query: str, top_k: int = 3) -> RetrievalResult:
        """Search for relevant documents (simple keyword matching for now).

        In Phase 3, this will use Milvus vector similarity search.
        """
        if not self._documents:
            return RetrievalResult(documents=[], scores=[], query=query)

        # Simple keyword overlap scoring (placeholder)
        query_words = set(query.lower().split())
        scored = []
        for doc in self._documents:
            doc_words = set(doc.text.lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        return RetrievalResult(
            documents=[s[0] for s in top],
            scores=[s[1] for s in top],
            query=query,
        )

    async def ingest_text(self, content: str, source: str = "user_input",
                           metadata: Optional[dict] = None) -> str:
        """Ingest text content into the knowledge base."""
        doc_id = uuid.uuid4().hex
        doc = Document(
            id=doc_id,
            text=content,
            metadata={"source": source, **(metadata or {})},
        )
        self._documents.append(doc)
        return doc_id

    async def ingest_document(self, document: Document) -> str:
        """Ingest a pre-built document."""
        self._documents.append(document)
        return document.id

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def clear(self) -> None:
        """Clear all documents."""
        self._documents.clear()


# Global instance
_knowledge_service: Optional[KnowledgeService] = None


def get_knowledge_service() -> KnowledgeService:
    global _knowledge_service
    if _knowledge_service is None:
        _knowledge_service = KnowledgeService()
    return _knowledge_service
