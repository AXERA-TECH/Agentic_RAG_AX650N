"""Web page fetching tool."""

from typing import Any

import httpx

from agentic_rag.orchestration.l1_tools.base import BaseTool


class WebFetchTool(BaseTool):
    """Fetch and extract content from a URL."""

    name = "web_fetch"
    description = "Fetch and extract text content from a web page URL."
    parameters_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch",
            },
        },
        "required": ["url"],
    }

    async def execute(self, url: str) -> Any:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, follow_redirects=True,
                                            headers={"User-Agent": "AgenticRAG/1.0"})
                response.raise_for_status()
                # Simple text extraction (strip tags)
                import re
                text = response.text
                # Remove script and style
                text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
                # Remove HTML tags
                text = re.sub(r'<[^>]+>', ' ', text)
                # Collapse whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                return text[:5000] if len(text) > 5000 else text
        except Exception as e:
            return f"Failed to fetch {url}: {str(e)}"
