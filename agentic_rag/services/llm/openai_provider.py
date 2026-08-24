"""OpenAI LLM Provider implementation."""

from typing import AsyncIterator, Optional

from agentic_rag.data.models import LLMChunk, LLMResponse, Message, ToolCall, ToolDefinition
from agentic_rag.services.llm.base import (
    BaseLLMProvider,
    ReasoningStreamFilter,
    log_llm_result,
    strip_reasoning,
)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible LLM provider (GPT-4, GPT-4o, etc.)."""

    provider_name = "openai"

    def __init__(self, model_name: str, api_key: str, api_base: str,
                 max_tokens: int = 4096, temperature: float = 0.7,
                 frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
                 vision_model: str = "", **kwargs):
        super().__init__(model_name, api_key, api_base, max_tokens, temperature, vision_model)
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        from openai import AsyncOpenAI as NativeAsyncOpenAI
        from agentic_rag.config.settings import configure_langfuse_environment

        client_cls = NativeAsyncOpenAI
        if configure_langfuse_environment():
            try:
                from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI
                client_cls = LangfuseAsyncOpenAI
            except ImportError:
                print(
                    "  [Langfuse] SDK not installed; LLM tracing disabled. "
                    "Install with: pip install 'langfuse>=3.0'",
                    flush=True,
                )

        client_options = {
            "api_key": api_key,
            "base_url": api_base,
            "timeout": 120.0,
            "max_retries": 2,
        }
        # Chat completions use the Langfuse wrapper when configured.
        self.client = client_cls(**client_options)
        # Embeddings deliberately use the native client: this integration is
        # scoped to LLM generations only, as opposed to full RAG tracing.
        self.embedding_client = NativeAsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=120.0,  # 2 minutes timeout for LLM calls
            max_retries=2,
        )
        _nl = model_name.lower()
        self.supports_vision = (
            "gpt-4o" in _nl
            or "vision" in _nl
            or "vl" in _nl
            or "qwen" in _nl          # All Qwen >= 2.5 are multimodal
            or "gemini" in _nl
            or "claude" in _nl
        )

    async def agenerate(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        stop: Optional[list[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        converted = self._convert_messages(messages)
        kwargs: dict = {
            "model": self.vision_model if self._has_images(converted) else self.model_name,
            "messages": converted,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
        if stop:
            kwargs["stop"] = stop
        # Reasoning models (Qwen3 etc.) circling inside <think> blocks waste
        # significant time/tokens on the strict ReAct format — request that the
        # server skip template-based reasoning where supported (vLLM/SGLang).
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        tool_calls = []
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=self._parse_args(tc.function.arguments),
                )
                for tc in msg.tool_calls
            ]

        log_llm_result(
            self.provider_name,
            kwargs["model"],
            msg.content or "",
            tool_calls,
            native_tools_enabled=bool(tools),
        )

        return LLMResponse(
            content=strip_reasoning(msg.content or ""),
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
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
        converted = self._convert_messages(messages)
        kwargs: dict = {
            "model": self.vision_model if self._has_images(converted) else self.model_name,
            "messages": converted,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
        if stop:
            kwargs["stop"] = stop
        # See agenerate(): disable template-based reasoning where supported.
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        try:
            stream = await self.client.chat.completions.create(**kwargs)
            think_filter = ReasoningStreamFilter()
            raw_content_parts: list[str] = []
            debug_tool_calls: list[dict] = []
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                if delta.content:
                    raw_content_parts.append(delta.content)

                chunk_data = LLMChunk(
                    content_delta=think_filter.feed(delta.content or ""),
                    stop_reason=chunk.choices[0].finish_reason,
                )

                if delta.tool_calls:
                    tool_delta = {
                        "id": delta.tool_calls[0].id,
                        "name": delta.tool_calls[0].function.name,
                        "arguments": delta.tool_calls[0].function.arguments,
                    }
                    chunk_data.tool_call_delta = tool_delta
                    debug_tool_calls.append(tool_delta)

                yield chunk_data
            # Flush any text held back for partial-tag detection
            tail = think_filter.flush()
            if tail:
                yield LLMChunk(content_delta=tail)
            log_llm_result(
                self.provider_name,
                kwargs["model"],
                "".join(raw_content_parts),
                debug_tool_calls,
                streaming=True,
                native_tools_enabled=bool(tools),
            )
        except Exception as e:
            import sys
            print(f"  [LLM] ⚠ Stream error: {e}", flush=True)
            sys.stdout.flush()
            # Re-raise the exception so the caller can handle it properly
            # instead of swallowing it as a text chunk
            raise

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        response = await self.embedding_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [d.embedding for d in response.data]

    def count_tokens(self, text: str) -> int:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return super().count_tokens(text)

    @staticmethod
    def _parse_args(args: str) -> dict:
        import json
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {"raw": args}

    @staticmethod
    def _has_images(messages: list[dict]) -> bool:
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return True
        return False
