"""Gateway adaptors for messaging platforms (WeChat Work, DingTalk, Feishu).

Each platform sub-package implements a webhook endpoint that:
1. Verifies the incoming request signature
2. Parses the platform-specific message format
3. Routes the query to the Agentic RAG engine
4. Returns a formatted response
"""

from agentic_rag.entrypoints.gateway.base import (
    BasePlatformAdaptor,
    PlatformMessage,
    PlatformResponse,
)
from agentic_rag.entrypoints.gateway.session import PlatformSessionMap

__all__ = [
    "BasePlatformAdaptor",
    "PlatformMessage",
    "PlatformResponse",
    "PlatformSessionMap",
]
