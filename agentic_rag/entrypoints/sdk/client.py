"""Python SDK for Agentic RAG — async HTTP client."""

from typing import AsyncIterator, Optional

import httpx


class AgenticRAGClient:
    """Async Python client for the Agentic RAG API."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(120.0),
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Use 'async with AgenticRAGClient() as client:' context manager")
        return self._client

    # ── Chat ──────────────────────────────────────

    async def chat(self, message: str, session_id: str = "") -> dict:
        """Send a chat message and get a response."""
        resp = await self.client.post("/api/v1/chat", json={
            "message": message,
            "session_id": session_id,
            "stream": False,
        })
        resp.raise_for_status()
        return resp.json()

    async def chat_stream(self, message: str, session_id: str = "") -> AsyncIterator[dict]:
        """Send a chat message and stream the response."""
        async with self.client.stream("POST", "/api/v1/chat/stream", json={
            "message": message,
            "session_id": session_id,
            "stream": True,
        }) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    import json
                    yield json.loads(line[6:])

    # ── RAG ───────────────────────────────────────

    async def rag_query(self, query: str, top_k: int = 3) -> dict:
        """Query the knowledge base."""
        resp = await self.client.post("/api/v1/rag/query", json={
            "query": query,
            "top_k": top_k,
        })
        resp.raise_for_status()
        return resp.json()

    async def rag_ingest(self, content: str, source: str = "sdk") -> dict:
        """Ingest content into the knowledge base."""
        resp = await self.client.post("/api/v1/rag/ingest", json={
            "content": content,
            "source": source,
        })
        resp.raise_for_status()
        return resp.json()

    # ── Session ───────────────────────────────────

    async def create_session(self, user_id: str = "default") -> dict:
        """Create a new session."""
        resp = await self.client.post("/api/v1/session", params={"user_id": user_id})
        resp.raise_for_status()
        return resp.json()

    async def get_session(self, session_id: str) -> dict:
        """Get session details."""
        resp = await self.client.get(f"/api/v1/session/{session_id}")
        resp.raise_for_status()
        return resp.json()

    async def delete_session(self, session_id: str) -> dict:
        """Delete a session."""
        resp = await self.client.delete(f"/api/v1/session/{session_id}")
        resp.raise_for_status()
        return resp.json()

    # ── Health ────────────────────────────────────

    async def health(self) -> dict:
        """Check API health."""
        resp = await self.client.get("/health")
        return resp.json()

    async def ready(self) -> dict:
        """Check API readiness."""
        resp = await self.client.get("/ready")
        return resp.json()
