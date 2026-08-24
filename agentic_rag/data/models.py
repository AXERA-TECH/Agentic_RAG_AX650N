"""Core domain models for Agentic RAG."""

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Message & Conversation Models
# ──────────────────────────────────────────────

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A tool call made by the LLM."""
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallResult(BaseModel):
    """Result of a tool call."""
    call_id: str
    name: str
    result: Any
    error: Optional[str] = None


class Message(BaseModel):
    """A single message in a conversation."""
    role: MessageRole
    content: str | list[dict[str, Any]]  # text or multimodal content
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str, tool_calls: list[ToolCall] | None = None) -> "Message":
        return cls(role=MessageRole.ASSISTANT, content=content,
                   tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, content: str, tool_call_id: str) -> "Message":
        return cls(role=MessageRole.TOOL, content=content, tool_call_id=tool_call_id)


# ──────────────────────────────────────────────
# LLM Models
# ──────────────────────────────────────────────

class ToolDefinition(BaseModel):
    """Tool definition for LLM function calling."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


class LLMResponse(BaseModel):
    """Response from an LLM provider."""
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: Optional[str] = None
    usage: dict[str, int] = Field(default_factory=dict)


class LLMChunk(BaseModel):
    """Streaming chunk from an LLM provider."""
    content_delta: str = ""
    tool_call_delta: Optional[dict] = None
    stop_reason: Optional[str] = None


# ──────────────────────────────────────────────
# Multimodal Input Models
# ──────────────────────────────────────────────

class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"


class MultimodalInput(BaseModel):
    """User input that may contain multiple modalities."""
    text: Optional[str] = None
    images: list[str] = Field(default_factory=list)    # base64 or file paths
    audio: Optional[str] = None                         # base64 or file path
    video: Optional[str] = None                         # file path
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessedContent(BaseModel):
    """Content after multimodal processing."""
    text: str
    source_types: list[MediaType] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────
# Agent Models
# ──────────────────────────────────────────────

class AgentEventType(str, Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    TEXT_DELTA = "text_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_RESULT = "tool_call_result"
    ERROR = "error"
    DONE = "done"


class AgentEvent(BaseModel):
    """An event emitted during agent execution."""
    event_type: AgentEventType
    data: dict[str, Any] = Field(default_factory=dict)
    turn_id: str = ""
    timestamp: float = Field(default_factory=time.time)


class AgentInput(BaseModel):
    """Input to an agent."""
    messages: list[Message] = Field(default_factory=list)
    query: str = ""
    multimodal: Optional[MultimodalInput] = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    """Output from an agent."""
    messages: list[Message] = Field(default_factory=list)
    final_answer: str = ""
    tool_calls_made: list[ToolCallResult] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    iterations: int = 0
    diagnostics: dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────
# RAG Models
# ──────────────────────────────────────────────

class Document(BaseModel):
    """A document in the knowledge base."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """Result of a retrieval operation."""
    documents: list[Document] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    query: str = ""


# ──────────────────────────────────────────────
# Session Models
# ──────────────────────────────────────────────

class Session(BaseModel):
    """A user session."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str = "default"
    created_at: float = Field(default_factory=time.time)
    expires_at: float = Field(default_factory=lambda: time.time() + 3600)
    metadata: dict[str, Any] = Field(default_factory=dict)
