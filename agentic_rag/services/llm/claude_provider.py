"""Anthropic Claude LLM Provider implementation."""

from typing import AsyncIterator, Optional

import anthropic

from agentic_rag.data.models import LLMChunk, LLMResponse, Message, ToolCall, ToolDefinition
from agentic_rag.services.llm.base import BaseLLMProvider, log_llm_result, strip_reasoning


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider."""

    provider_name = "claude"

    def __init__(self, model_name: str, api_key: str, api_base: str,
                 max_tokens: int = 4096, temperature: float = 0.7,
                 vision_model: str = "", **kwargs):
        super().__init__(model_name, api_key, api_base, max_tokens, temperature, vision_model)
        self.client = anthropic.AsyncAnthropic(api_key=api_key, base_url=api_base or None)
        self.supports_vision = True  # Claude 3+ models support vision

    async def agenerate(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        stop: Optional[list[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        system_prompt, user_messages = self._split_system_messages(messages)
        converted = self._convert_to_claude_format(user_messages)

        kwargs: dict = {
            "model": self.vision_model or self.model_name,
            "messages": converted,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = self._convert_tools_to_claude(tools)
        if stop:
            kwargs["stop_sequences"] = stop

        response = await self.client.messages.create(**kwargs)

        tool_calls = []
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        log_llm_result(
            self.provider_name,
            kwargs["model"],
            text_content,
            tool_calls,
            native_tools_enabled=bool(tools),
        )

        return LLMResponse(
            content=strip_reasoning(text_content),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            usage={
                "prompt_tokens": response.usage.input_tokens if response.usage else 0,
                "completion_tokens": response.usage.output_tokens if response.usage else 0,
            },
        )

    async def agenerate_stream(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        stop: Optional[list[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[LLMChunk]:
        system_prompt, user_messages = self._split_system_messages(messages)
        converted = self._convert_to_claude_format(user_messages)

        kwargs: dict = {
            "model": self.vision_model or self.model_name,
            "messages": converted,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = self._convert_tools_to_claude(tools)

        content_parts: list[str] = []
        tool_deltas: list[dict] = []
        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    chunk = LLMChunk()
                    if event.delta.type == "text_delta":
                        chunk.content_delta = event.delta.text
                        content_parts.append(event.delta.text)
                    elif event.delta.type == "input_json_delta":
                        chunk.tool_call_delta = {"arguments": event.delta.partial_json}
                        tool_deltas.append(chunk.tool_call_delta)
                    yield chunk
                elif event.type == "message_stop":
                    yield LLMChunk(stop_reason="end_turn")
        log_llm_result(
            self.provider_name,
            kwargs["model"],
            "".join(content_parts),
            tool_deltas,
            streaming=True,
            native_tools_enabled=bool(tools),
        )

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        # Claude doesn't have a native embedding API; use a fallback
        raise NotImplementedError(
            "Claude does not support embeddings. Use OpenAI or a dedicated embedding provider."
        )

    def _split_system_messages(self, messages: list[Message]) -> tuple[str, list[Message]]:
        """Extract system prompt from messages, return (system_text, remaining_messages)."""
        system_parts = []
        remaining = []
        for msg in messages:
            if msg.role.value == "system":
                system_parts.append(msg.content if isinstance(msg.content, str) else "")
            else:
                remaining.append(msg)
        return "\n".join(system_parts), remaining

    def _convert_to_claude_format(self, messages: list[Message]) -> list[dict]:
        """Convert messages to Claude's content blocks format."""
        converted = []
        for msg in messages:
            if msg.role.value == "user":
                content = self._build_claude_content(msg)
                converted.append({"role": "user", "content": content})
            elif msg.role.value == "assistant":
                content = self._build_claude_content(msg)
                entry: dict = {"role": "assistant", "content": content}
                converted.append(entry)
            elif msg.role.value == "tool":
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id or "",
                        "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                    }],
                })
        return converted

    def _build_claude_content(self, msg: Message) -> list[dict] | str:
        """Build Claude content blocks from a message."""
        if isinstance(msg.content, str):
            return msg.content
        # Multimodal content
        blocks = []
        for part in msg.content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    blocks.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url":
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.get("media_type", "image/png"),
                            "data": part["image_url"].get("url", "").split(",")[-1]
                            if "base64," in part["image_url"].get("url", "") else "",
                        },
                    })
        return blocks if blocks else str(msg.content)

    def _convert_tools_to_claude(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert tools to Claude's tool format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": {
                    "type": "object",
                    "properties": t.parameters.get("properties", {}),
                    "required": t.parameters.get("required", []),
                },
            }
            for t in tools
        ]
