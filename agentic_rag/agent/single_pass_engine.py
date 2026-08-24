"""Direct and retrieval-first execution paths for ordinary chat requests."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import AsyncIterator

from agentic_rag.agent.react_parser import extract_final_answer
from agentic_rag.agent.request_policy import RequestStrategy
from agentic_rag.data.models import (
    AgentEvent,
    AgentEventType,
    AgentInput,
    AgentOutput,
    Message,
    ToolCallResult,
)
from agentic_rag.services.llm.base import BaseLLMProvider, ReasoningStreamFilter, strip_reasoning


ENGINE_REVISION = "single-chat-evidence-v3"


class SinglePassEngine:
    """Execute zero or one deterministic tool, then ask the LLM once."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        strategy: RequestStrategy,
        tool=None,
        fallback_tool=None,
        direct_answer: str = "",
        route_reason: str = "",
    ):
        self.llm = llm
        self.strategy = strategy
        self.tool = tool
        # Optional external search used only when the internal KB cannot
        # provide usable evidence.  Keeping this separate preserves the
        # retrieval-first behaviour for normal in-domain questions.
        self.fallback_tool = fallback_tool
        self.direct_answer = direct_answer
        self.route_reason = route_reason

    async def run(self, input: AgentInput, turn_id: str = "") -> AgentOutput:
        tool_calls: list[ToolCallResult] = []
        if self.direct_answer:
            return AgentOutput(
                final_answer=self.direct_answer,
                iterations=0,
                diagnostics=self._diagnostics(),
            )

        if self.strategy == RequestStrategy.TOOL_USE and self.tool is None and self.fallback_tool is None:
            return AgentOutput(
                final_answer=self._evidence_failure_answer(),
                iterations=0,
                diagnostics=self._diagnostics(),
            )

        evidence = ""
        evidence_strategy = self.strategy
        if self.tool is not None:
            result = await self._execute_tool(input.query)
            tool_calls.append(result)
            if not self._usable_result(result):
                if self.strategy == RequestStrategy.TOOL_USE and self.fallback_tool is not None:
                    fallback_result = await self._execute_fallback_tool(input.query)
                    tool_calls.append(fallback_result)
                    if not self._usable_result(fallback_result):
                        return AgentOutput(
                            final_answer=self._evidence_failure_answer(),
                            tool_calls_made=tool_calls,
                            iterations=0,
                            diagnostics=self._diagnostics(),
                        )
                    evidence = self._prepare_evidence(fallback_result.result)
                    evidence_strategy = RequestStrategy.TOOL_USE
                else:
                    return AgentOutput(
                        final_answer=self._evidence_failure_answer(),
                        tool_calls_made=tool_calls,
                        iterations=0,
                        diagnostics=self._diagnostics(),
                    )
            else:
                evidence = self._prepare_evidence(result.result)

        messages = self._build_messages(input, evidence)
        response = await self.llm.agenerate(messages)
        answer = self._clean_answer(response.content)
        answer = self._normalize_citations(answer, evidence, evidence_strategy)
        if not self._valid_answer(answer, evidence, evidence_strategy):
            repaired, repair_response, repair_messages = await self._repair_answer(
                input, evidence, evidence_strategy
            )
            if self._valid_answer(repaired, evidence, evidence_strategy):
                answer = repaired
                response = repair_response
                messages = repair_messages
            # A candidate can be syntactically valid yet not support the
            # question.  Retry once with external search before failing closed.
            elif (self.strategy == RequestStrategy.TOOL_USE and evidence_strategy == RequestStrategy.TOOL_USE
                    and self.fallback_tool is not None):
                fallback_result = await self._execute_fallback_tool(input.query)
                tool_calls.append(fallback_result)
                if self._usable_result(fallback_result):
                    evidence = self._prepare_evidence(fallback_result.result)
                    evidence_strategy = RequestStrategy.TOOL_USE
                    messages = self._build_messages(input, evidence)
                    response = await self.llm.agenerate(messages)
                    answer = self._clean_answer(response.content)
                    answer = self._normalize_citations(answer, evidence, evidence_strategy)
                    if not self._valid_answer(answer, evidence, evidence_strategy):
                        answer = self._answer_failure(evidence_strategy)
                else:
                    answer = self._answer_failure()
            else:
                answer = self._answer_failure(evidence_strategy)
        return AgentOutput(
            messages=messages,
            final_answer=answer,
            tool_calls_made=tool_calls,
            usage=response.usage,
            iterations=1,
            diagnostics=self._diagnostics(evidence),
        )

    async def stream(self, input: AgentInput, turn_id: str = "") -> AsyncIterator[AgentEvent]:
        turn_id = turn_id or uuid.uuid4().hex
        if self.direct_answer:
            yield AgentEvent(
                event_type=AgentEventType.TEXT_DELTA,
                data={"content": self.direct_answer},
                turn_id=turn_id,
            )
            yield AgentEvent(
                event_type=AgentEventType.DONE,
                data={
                    "final_answer": self.direct_answer,
                    "iterations": 0,
                    "diagnostics": self._diagnostics(),
                },
                turn_id=turn_id,
            )
            return

        if self.strategy == RequestStrategy.TOOL_USE and self.tool is None and self.fallback_tool is None:
            answer = self._evidence_failure_answer()
            yield AgentEvent(
                event_type=AgentEventType.DONE,
                data={
                    "final_answer": answer,
                    "iterations": 0,
                    "diagnostics": self._diagnostics(),
                },
                turn_id=turn_id,
            )
            return

        evidence = ""
        evidence_strategy = self.strategy
        tool_calls: list[ToolCallResult] = []
        if self.tool is not None:
            arguments = self._tool_arguments(input.query)
            yield AgentEvent(
                event_type=AgentEventType.TOOL_CALL_START,
                data={
                    "tool": self.tool.name,
                    "input": arguments,
                    "diagnostics": self._diagnostics(),
                },
                turn_id=turn_id,
            )
            result = await self._execute_tool(input.query, arguments)
            usable = self._usable_result(result)
            yield AgentEvent(
                event_type=AgentEventType.TOOL_CALL_RESULT,
                data={
                    "tool": self.tool.name,
                    "success": usable,
                    "result": str(result.result)[:500] if result.result else "",
                    "error": result.error,
                },
                turn_id=turn_id,
            )
            if not usable:
                if self.strategy == RequestStrategy.TOOL_USE and self.fallback_tool is not None:
                    yield AgentEvent(
                        event_type=AgentEventType.TOOL_CALL_START,
                        data={"tool": self.fallback_tool.name, "input": self._tool_arguments_for(self.fallback_tool, input.query)},
                        turn_id=turn_id,
                    )
                    fallback_result = await self._execute_fallback_tool(input.query)
                    tool_calls.append(fallback_result)
                    yield AgentEvent(
                        event_type=AgentEventType.TOOL_CALL_RESULT,
                        data={"tool": self.fallback_tool.name, "success": self._usable_result(fallback_result),
                              "result": str(fallback_result.result)[:500] if fallback_result.result else "", "error": fallback_result.error},
                        turn_id=turn_id,
                    )
                    if not self._usable_result(fallback_result):
                        answer = self._evidence_failure_answer()
                        yield AgentEvent(event_type=AgentEventType.DONE, data={"final_answer": answer, "iterations": 0}, turn_id=turn_id)
                        return
                    evidence = self._prepare_evidence(fallback_result.result)
                    evidence_strategy = RequestStrategy.TOOL_USE
                else:
                    answer = self._evidence_failure_answer()
                    yield AgentEvent(event_type=AgentEventType.DONE, data={"final_answer": answer, "iterations": 0}, turn_id=turn_id)
                    return
            else:
                evidence = self._prepare_evidence(result.result)

        messages = self._build_messages(input, evidence)
        full_content = ""
        reasoning_filter = ReasoningStreamFilter()
        try:
            async for chunk in self.llm.agenerate_stream(messages):
                if chunk.content_delta:
                    full_content += reasoning_filter.feed(chunk.content_delta)
            full_content += reasoning_filter.flush()
        except Exception as exc:
            yield AgentEvent(
                event_type=AgentEventType.ERROR,
                data={"error": f"LLM stream failed: {exc}"},
                turn_id=turn_id,
            )
            return

        answer = self._clean_answer(full_content)
        answer = self._normalize_citations(answer, evidence, evidence_strategy)
        if not self._valid_answer(answer, evidence, evidence_strategy):
            repaired, _, repair_messages = await self._repair_answer(
                input, evidence, evidence_strategy
            )
            if self._valid_answer(repaired, evidence, evidence_strategy):
                answer = repaired
                messages = repair_messages
            elif (self.strategy == RequestStrategy.TOOL_USE and evidence_strategy == RequestStrategy.TOOL_USE
                    and self.fallback_tool is not None):
                fallback_args = self._tool_arguments_for(self.fallback_tool, input.query)
                yield AgentEvent(
                    event_type=AgentEventType.TOOL_CALL_START,
                    data={"tool": self.fallback_tool.name, "input": fallback_args},
                    turn_id=turn_id,
                )
                fallback_result = await self._execute_fallback_tool(input.query)
                tool_calls.append(fallback_result)
                fallback_usable = self._usable_result(fallback_result)
                yield AgentEvent(
                    event_type=AgentEventType.TOOL_CALL_RESULT,
                    data={
                        "tool": self.fallback_tool.name,
                        "success": fallback_usable,
                        "result": str(fallback_result.result)[:500] if fallback_result.result else "",
                        "error": fallback_result.error,
                    },
                    turn_id=turn_id,
                )
                if fallback_usable:
                    evidence = self._prepare_evidence(fallback_result.result)
                    evidence_strategy = RequestStrategy.TOOL_USE
                    messages = self._build_messages(input, evidence)
                    response = await self.llm.agenerate(messages)
                    answer = self._clean_answer(response.content)
                    answer = self._normalize_citations(answer, evidence, evidence_strategy)
                    if not self._valid_answer(answer, evidence, evidence_strategy):
                        answer = self._answer_failure(evidence_strategy)
                else:
                    answer = self._evidence_failure_answer()
            else:
                answer = self._answer_failure(evidence_strategy)
        yield AgentEvent(
            event_type=AgentEventType.TEXT_DELTA,
            data={"content": answer},
            turn_id=turn_id,
        )
        yield AgentEvent(
            event_type=AgentEventType.DONE,
            data={
                "final_answer": answer,
                "iterations": 1,
                "diagnostics": self._diagnostics(evidence),
            },
            turn_id=turn_id,
        )

    async def _execute_tool(self, query: str, arguments: dict | None = None) -> ToolCallResult:
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        args = arguments or self._tool_arguments(query)
        try:
            result = await asyncio.wait_for(self.tool.execute(**args), timeout=60.0)
            return ToolCallResult(call_id=call_id, name=self.tool.name, result=result)
        except asyncio.TimeoutError:
            return ToolCallResult(
                call_id=call_id,
                name=self.tool.name,
                result=None,
                error=f"Tool '{self.tool.name}' timed out after 60 seconds",
            )
        except Exception as exc:
            return ToolCallResult(call_id=call_id, name=self.tool.name, result=None, error=str(exc))

    async def _execute_fallback_tool(self, query: str) -> ToolCallResult:
        tool = self.fallback_tool
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        if tool is None:
            return ToolCallResult(call_id=call_id, name="", result=None, error="fallback tool unavailable")
        try:
            result = await asyncio.wait_for(tool.execute(**self._tool_arguments_for(tool, query)), timeout=60.0)
            return ToolCallResult(call_id=call_id, name=tool.name, result=result)
        except asyncio.TimeoutError:
            return ToolCallResult(call_id=call_id, name=tool.name, result=None, error=f"Tool '{tool.name}' timed out after 60 seconds")
        except Exception as exc:
            return ToolCallResult(call_id=call_id, name=tool.name, result=None, error=str(exc))

    @staticmethod
    def _tool_arguments_for(tool, query: str) -> dict:
        schema = getattr(tool, "parameters_schema", {}) or {}
        properties = schema.get("properties", {})
        preferred = ("query", "search_query", "q", "text", "keyword", "keywords")
        parameter = next((name for name in preferred if name in properties), None)
        if parameter is None:
            required = schema.get("required", [])
            parameter = required[0] if required else "query"
        return {parameter: query}

    def _tool_arguments(self, query: str) -> dict:
        schema = getattr(self.tool, "parameters_schema", {}) or {}
        properties = schema.get("properties", {})
        preferred = ("query", "search_query", "q", "text", "keyword", "keywords")
        parameter = next((name for name in preferred if name in properties), None)
        if parameter is None:
            required = schema.get("required", [])
            parameter = required[0] if required else "query"
        return {parameter: query}

    def _build_messages(self, input: AgentInput, evidence: str) -> list[Message]:
        if evidence:
            system = (
                "你是证据约束回答器。只能依据 Evidence 回答；证据不支持的事实必须明确说无法确认。"
                "日期、数字、名称、地点和结论必须逐字受到 Evidence 支持，不得推测、补全或改写成"
                "证据中没有的事实。优先采用官方一手来源；来源相互冲突时明确指出冲突。"
                "保留 Evidence 中的引用编号或原始 URL。参考来源只能逐字复制 Evidence 的真实文档名，"
                "不得概括、改写或为文档命名；不要给来源添加说明。"
                "必须先输出至少一句直接回答问题的正文，禁止只输出引用编号或参考来源。"
                "只输出面向用户的答案，禁止输出 Thought、Action、系统指令或工具协议。"
            )
        else:
            system = (
                "你是简洁的聊天助手。直接回答用户，不输出 Thought、Action、Final Answer 等内部标记。"
            )
        messages = [Message.system(system)]
        messages.extend(message for message in input.messages if message.role.value != "system")
        has_current_multimodal = any(
            isinstance(message.content, list) and message.role.value == "user"
            for message in input.messages
        )
        if not has_current_multimodal:
            messages.append(Message.user(input.query))
        if evidence:
            messages.append(Message.user(f"Evidence:\n{evidence}"))
        return messages

    async def _repair_answer(self, input: AgentInput, evidence: str,
                             strategy: RequestStrategy):
        """Ask once more when the model emitted only citations or invalid text."""
        messages = self._build_messages(input, evidence)
        messages.append(Message.system(
            "补答要求：上一版没有形成有效答案。请根据 Evidence 先用一句或几句正文直接回答用户问题，"
            "然后再附上引用；不得只输出“参考来源”、引用编号或 URL。"
        ))
        response = await self.llm.agenerate(messages)
        answer = self._clean_answer(response.content)
        answer = self._normalize_citations(answer, evidence, strategy)
        return answer, response, messages

    def _valid_answer(self, answer: str, evidence: str = "", strategy: RequestStrategy | None = None) -> bool:
        strategy = strategy or self.strategy
        if not answer or re.search(r"(?:^|\n)\s*(Thought|Action|Action Input)\s*[:：]", answer):
            return False
        if evidence and not self._has_substantive_answer(answer):
            return False
        if strategy == RequestStrategy.TOOL_USE:
            answer_refs = set(re.findall(r"\[R\d+\]", answer))
            evidence_refs = set(re.findall(r"\[R\d+\]", evidence))
            return (
                bool(answer_refs)
                and answer_refs.issubset(evidence_refs)
                and self._facts_supported(answer, evidence)
            )
        if strategy == RequestStrategy.TOOL_USE:
            answer_urls = set(re.findall(r"https?://[^\s\])}>，。]+", answer))
            evidence_urls = set(re.findall(r"https?://[^\s\])}>，。]+", evidence))
            return bool(answer_urls & evidence_urls) and self._facts_supported(answer, evidence)
        return True

    @staticmethod
    def _has_substantive_answer(answer: str) -> bool:
        """Reject citation-only output while allowing short factual answers."""
        body = re.split(
            r"(?:^|\n)\s*(?:📚\s*)?参考来源\s*[:：]?",
            answer,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        body = re.sub(r"\[R\d+\]", "", body, flags=re.IGNORECASE)
        body = re.sub(r"https?://\S+", "", body)
        meaningful = re.sub(r"[^A-Za-z0-9\u3400-\u9fff]+", "", body)
        return len(meaningful) >= 2

    def _normalize_citations(self, answer: str, evidence: str, strategy: RequestStrategy | None = None) -> str:
        """Normalize weak-model citation formatting without inventing sources."""
        strategy = strategy or self.strategy
        if strategy == RequestStrategy.TOOL_USE:
            return self._normalize_mcp_citations(answer, evidence)
        if not answer:
            return answer

        evidence_refs = sorted(
            set(re.findall(r"\[R\d+\]", evidence)),
            key=lambda ref: int(re.search(r"\d+", ref).group()),
        )
        if not evidence_refs:
            return answer

        # Older prompts and some local models emit [W1] or [Source 1]. They refer
        # to the same positional evidence and can be mapped deterministically.
        def replace_alias(match: re.Match) -> str:
            return f"[R{int(match.group(1))}]"

        normalized = re.sub(r"\[W(\d+)\]", replace_alias, answer, flags=re.IGNORECASE)
        normalized = re.sub(
            r"\[(?:Source|来源)\s*(\d+)\]",
            replace_alias,
            normalized,
            flags=re.IGNORECASE,
        )
        source_map = self._rag_source_map(evidence)
        if not source_map:
            if not re.search(r"\[R\d+\]", normalized):
                normalized = f"{normalized.rstrip()}\n\n参考来源：{'、'.join(evidence_refs)}"
            return normalized

        # Source labels are data, not generated prose. Replace any model-written
        # source section with the exact names emitted by the retrieval tool.
        normalized = re.sub(
            r"\n+\s*(?:📚\s*)?参考来源\s*[:：]?\s*[\s\S]*$",
            "",
            normalized,
            flags=re.IGNORECASE,
        ).rstrip()
        cited_refs = [ref for ref in evidence_refs if ref in set(re.findall(r"\[R\d+\]", normalized))]
        if not cited_refs:
            cited_refs = evidence_refs
        source_lines = [f"- {ref} {source_map[ref]}" for ref in cited_refs if ref in source_map]
        return f"{normalized}\n\n参考来源：\n" + "\n".join(source_lines)

    @staticmethod
    def _normalize_mcp_citations(answer: str, evidence: str) -> str:
        """Attach exact retrieved URLs when a weak model omits source markup."""
        if not answer:
            return answer
        url_pattern = r"https?://[^\s\])}>，。\"']+"
        answer_urls = set(re.findall(url_pattern, answer))
        if answer_urls:
            # Do not silently bless an invented URL. Validation below will
            # reject it unless it also occurs in the retrieved evidence.
            return answer
        evidence_urls = list(dict.fromkeys(re.findall(url_pattern, evidence)))
        if not evidence_urls:
            return answer
        sources = "\n".join(
            f"- [W{index}] {url}"
            for index, url in enumerate(evidence_urls[:3], start=1)
        )
        return f"{answer.rstrip()}\n\n参考来源：\n{sources}"

    @staticmethod
    def _rag_source_map(evidence: str) -> dict[str, str]:
        """Extract exact reference-to-document mappings from tool evidence."""
        marker = "===== SOURCES ====="
        if marker not in evidence:
            return {}
        source_block = evidence.split(marker, 1)[1].split("IMPORTANT:", 1)[0]
        sources = {}
        for line in source_block.splitlines():
            match = re.match(r"^\s*(\[R\d+\])\s+(.+?)\s*$", line)
            if match:
                sources[match.group(1)] = match.group(2)
        return sources

    @classmethod
    def _facts_supported(cls, answer: str, evidence: str) -> bool:
        """Reject objective facts that are absent from the retrieved evidence.

        A matching URL alone does not ground the prose. Dates, years, percentages,
        scores and measured quantities in the answer must also occur in evidence.
        """
        answer_anchors = cls._fact_anchors(answer)
        if not answer_anchors:
            return True
        return answer_anchors.issubset(cls._fact_anchors(evidence))

    @staticmethod
    def _fact_anchors(text: str) -> set[str]:
        # URL paths and citation IDs contain incidental digits, not factual claims.
        clean = re.sub(r"https?://\S+", " ", text)
        clean = re.sub(r"\[R\d+\]", " ", clean, flags=re.IGNORECASE)
        anchors: set[str] = set()
        unit_pattern = re.compile(
            r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*"
            r"(年|月|日|号|时|点|分|秒|%|％|岁|届|名|场|次|个|支|队|"
            r"片|粒|包|毫升|升|毫克|克|千克|公斤|ml|mg|kg|km|cm|mm|美元|元)",
            re.IGNORECASE,
        )
        aliases = {"％": "%", "号": "日", "点": "时", "公斤": "kg", "千克": "kg"}
        for number, unit in unit_pattern.findall(clean):
            normalized_number = str(float(number)).rstrip("0").rstrip(".") if "." in number else str(int(number))
            normalized_unit = aliases.get(unit.lower(), unit.lower())
            anchors.add(f"{normalized_number}{normalized_unit}")

        # Normalize standalone years across Chinese and English evidence.
        # Search results commonly say "2026 FIFA ..." while the answer says
        # "2026年世界杯"; these are the same factual anchor.
        for year in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", clean):
            anchors.add(f"{year}年")

        # Canonicalize official English dates so Chinese answers remain verifiable.
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }
        month_names = "|".join(months)
        for match in re.finditer(
            rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*(\d{{4}}))?",
            clean,
            re.IGNORECASE,
        ):
            anchors.add(f"{months[match.group(1).lower()]}月")
            anchors.add(f"{int(match.group(2))}日")
            if match.group(3):
                anchors.add(f"{match.group(3)}年")

        for left, right in re.findall(r"(?<!\d)(\d+)\s*[:：]\s*(\d+)(?!\d)", clean):
            anchors.add(f"score:{int(left)}:{int(right)}")
        return anchors

    def _diagnostics(self, evidence: str = "") -> dict:
        refs = set(re.findall(r"\[R\d+\]", evidence))
        urls = set(re.findall(r"https?://[^\s\])}>，。\"']+", evidence))
        return {
            "engine_revision": ENGINE_REVISION,
            "strategy": self.strategy.value,
            "route_reason": self.route_reason,
            "selected_tool": getattr(self.tool, "name", ""),
            "evidence_source_count": len(refs or urls),
        }

    @staticmethod
    def _clean_answer(content: str) -> str:
        cleaned = strip_reasoning(content).strip()
        parsed = extract_final_answer(cleaned)
        return (parsed or cleaned).strip()

    @staticmethod
    def _usable_result(result: ToolCallResult) -> bool:
        if result.error or result.result is None:
            return False
        if isinstance(result.result, dict):
            if result.result.get("isError") or result.result.get("error"):
                return False
        text = str(result.result).strip().lower()
        if not text:
            return False
        failure_markers = (
            "no relevant content found",
            "knowledge base is not configured",
            "search error:",
            "mcp error:",
            "no response from mcp",
            "not found on any connected server",
            "not found on server",
        )
        return not any(marker in text for marker in failure_markers)

    @staticmethod
    def _prepare_evidence(result) -> str:
        if isinstance(result, (dict, list)):
            text = json.dumps(result, ensure_ascii=False)
        else:
            text = str(result)
        if len(text) <= 12000:
            return text
        return f"{text[:9000]}\n... [evidence truncated] ...\n{text[-3000:]}"

    def _evidence_failure_answer(self, strategy: RequestStrategy | None = None) -> str:
        return "检索未返回足够依据，无法可靠回答该问题。"

    def _answer_failure(self, strategy: RequestStrategy | None = None) -> str:
        return "检索已完成，但模型未能生成带有效证据引用的可靠答案。"
