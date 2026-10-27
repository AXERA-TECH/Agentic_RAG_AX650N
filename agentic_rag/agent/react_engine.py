"""ReAct Engine — the core Think → Act → Observe loop."""

import asyncio
import json
import re
import uuid
from typing import AsyncIterator, Optional, Protocol

from agentic_rag.data.models import (
    AgentEvent,
    AgentEventType,
    AgentInput,
    AgentOutput,
    Message,
    ToolCall,
    ToolCallResult,
    ToolDefinition,
)
from agentic_rag.agent.evidence_validator import EvidenceCheck, validate_answer
from agentic_rag.agent.query_planner import RoutePlan
from agentic_rag.agent.react_state import LoopState, RoundResult
from agentic_rag.agent.tool_executor import ToolExecutor
from agentic_rag.agent.react_parser import (
    _TOOL_FAILURE_MARKERS,
    clean_user_answer,
    extract_answer_candidate,
    extract_final_answer,
    format_observation,
    format_tool_message,
    parse_react_output,
    resolve_tool_name,
    truncate_tool_result,
)
from agentic_rag.agent.react_prompt import build_react_prompt, build_tools_description
from agentic_rag.agent.sufficiency import SufficiencyChecker, SufficiencyResult
from agentic_rag.services.llm.base import (
    BaseLLMProvider,
    ReasoningStreamFilter,
    strip_reasoning,
)
from agentic_rag.utils.logging import get_logger

logger = get_logger(__name__)

# Private aliases preserved for callers that imported these from this module.
_ToolExecutor = ToolExecutor
_LoopState = LoopState
_RoundResult = RoundResult


class PlanLike(Protocol):
    """Structural type shared by legacy ``RoutePlan`` and ``RoutingPlan``."""

    sub_queries: tuple[str, ...]

    @property
    def is_complex(self) -> bool: ...

    def diagnostics(self) -> dict: ...


class ReActEngine:
    """ReAct (Reasoning + Acting) reasoning engine.

    Executes the Think → Act → Observe loop that powers the agent. The loop
    lives in a single generator (``_stream_loop``); ``run()`` simply drains
    it, so the streaming and non-streaming paths can never drift apart.
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        tools: list,
        system_prompt_template: str,
        max_iterations: int = 10,
        stop_on_error: bool = False,
        enable_native_tool_calls: bool = True,
        require_tool_call: bool = False,
        required_tool_names: set[str] | None = None,
        initial_tool_name: str = "",
        route_reason: str = "",
        allow_registry_fallback: bool = False,
        max_retrieval_calls: int = 2,
        query_plan: PlanLike | RoutePlan | None = None,
        enable_sufficiency_check: bool = False,
        seed_tool_arguments: dict | None = None,
        failure_answer: str = "",
    ):
        """
        Args:
            llm: LLM provider for generation.
            tools: List of BaseTool instances available to the agent.
            system_prompt_template: Template string for the system prompt.
            max_iterations: Maximum ReAct loop iterations.
            stop_on_error: If True, stop on first tool error.
            enable_native_tool_calls: If False, tools are NOT sent to the LLM
                (pure ReAct text mode for models without function calling).
            require_tool_call: If True, reject a final answer until at least one
                required tool has completed successfully.
            required_tool_names: Tools that may satisfy the evidence requirement.
            initial_tool_name: Tool executed deterministically before the first LLM call.
            allow_registry_fallback: If True, a tool call whose name isn't in
                ``tools`` may still resolve via the global tool registry.
                Default False: the router's tool selection is the agent's
                capability boundary.
            max_retrieval_calls: Total calls allowed for the deterministic
                retrieval tool per turn (the seed counts as one).  A second
                call must use a rewritten query; identical calls are always
                refused.
            query_plan: Optional LLM/deterministic route plan carrying
                sub-queries and complexity for the sufficiency gate.
            enable_sufficiency_check: Add an explicit evidence-coverage check
                before accepting a final answer.  The router enables it only
                for complex plans; simple requests pay no extra model call.
            seed_tool_arguments: Optional fixed arguments for the deterministic
                initial tool call.  When omitted, arguments are derived from
                the input and the tool's parameter schema.
            failure_answer: Route-specific message used when the required
                evidence could not be obtained.
        """
        self.llm = llm
        self.tools = tools
        self.system_prompt_template = system_prompt_template
        self.max_iterations = max_iterations
        self.stop_on_error = stop_on_error
        self.enable_native_tool_calls = enable_native_tool_calls
        self.require_tool_call = require_tool_call
        self.required_tool_names = required_tool_names or set()
        self.initial_tool_name = initial_tool_name
        self.route_reason = route_reason
        self.allow_registry_fallback = allow_registry_fallback
        self.max_retrieval_calls = max(1, int(max_retrieval_calls or 1))
        self.query_plan = query_plan
        self.enable_sufficiency_check = bool(enable_sufficiency_check)
        self.seed_tool_arguments = (
            dict(seed_tool_arguments)
            if seed_tool_arguments is not None else None
        )
        self.failure_answer = failure_answer
        self._tool_map = {t.name: t for t in tools}
        # Retrieval budget applies to rag_search (and the seeded retrieval
        # tool).  Media/MCP tools are not capped here; duplicates still are.
        retrieval_names = set()
        if self.initial_tool_name:
            retrieval_names.add(self.initial_tool_name)
        if "rag_search" in self._tool_map:
            retrieval_names.add("rag_search")
        self._retrieval_limits = {
            name: self.max_retrieval_calls for name in retrieval_names
        }
        # Whether the provider can handle the ``tools`` parameter on the wire.
        # Text-only servers follow the ReAct format from the system prompt
        # instead (see BaseLLMProvider.supports_native_tools_on_wire).
        text_only = self._detect_text_only_tool_calling()
        self._use_streaming = not text_only
        self._send_native_tools = not text_only

    def _detect_text_only_tool_calling(self) -> bool:
        """Return True when the model must use the text ReAct format."""
        if not self.tools:
            return False
        if not self.enable_native_tool_calls:
            # The caller explicitly disabled native function calling.  Treat
            # it as text-only in every respect (no tools on the wire, ReAct
            # stop sequences, and no native-tool contract message).
            return True
        try:
            return not self.llm.supports_native_tools_on_wire()
        except Exception:
            return False

    async def run(self, input: AgentInput, turn_id: str = "") -> AgentOutput:
        """Execute the ReAct loop (non-streaming)."""
        if not turn_id:
            turn_id = uuid.uuid4().hex
        output = None
        async for event in self._stream_loop(input, turn_id):
            if event.event_type == AgentEventType.DONE:
                output = event.data.get("_output")
            elif event.event_type == AgentEventType.ERROR:
                raise RuntimeError(event.data.get("error", "LLM stream failed"))
        if output is None:
            raise RuntimeError("ReAct loop ended without a DONE event")
        return output

    async def stream(self, input: AgentInput, turn_id: str = "") -> AsyncIterator[AgentEvent]:
        """Execute the ReAct loop with streaming events."""
        if not turn_id:
            turn_id = uuid.uuid4().hex
        async for event in self._stream_loop(input, turn_id):
            # The collected AgentOutput is internal — strip it before the
            # event reaches the stream consumer.
            if event.event_type == AgentEventType.DONE:
                event.data.pop("_output", None)
            yield event

    def _done(self, state: _LoopState, final_answer: str, iterations: int, turn_id: str) -> AgentEvent:
        """Terminate the loop with a DONE event carrying the AgentOutput."""
        diagnostics = self._diagnostics(state)
        return AgentEvent(
            event_type=AgentEventType.DONE,
            data={
                "final_answer": final_answer,
                "iterations": iterations,
                "diagnostics": diagnostics,
                "_output": AgentOutput(
                    messages=state.messages,
                    final_answer=final_answer,
                    tool_calls_made=state.tool_calls_made,
                    usage=state.total_usage,
                    diagnostics=diagnostics,
                    iterations=iterations,
                ),
            },
            turn_id=turn_id,
        )

    # ── Evidence policy ────────────────────────────────────────────────

    def _is_required_evidence_missing(self, results: list[ToolCallResult]) -> bool:
        """Return whether the configured evidence-producing tool is still missing."""
        if not self.require_tool_call:
            return False
        return not any(
            (not self.required_tool_names or result.name in self.required_tool_names)
            and self._is_usable_tool_result(result)
            for result in results
        )

    def _other_evidence_tools_available(self) -> bool:
        """Whether a tool other than the seeded initial tool can still supply evidence.

        When the deterministic initial tool (e.g. ``rag_search``) returns no
        usable evidence, refuse immediately only if it was the sole evidence
        source. If other required tools remain (an MCP web/weather tool, …),
        fall through to the loop so the model can try them before we give up.
        """
        return any(
            tool.name != self.initial_tool_name
            and (not self.required_tool_names or tool.name in self.required_tool_names)
            for tool in self.tools
        )

    @staticmethod
    def _is_usable_tool_result(result: ToolCallResult) -> bool:
        if result.error or not result.result:
            return False
        text = str(result.result).lower()
        return not any(marker in text for marker in _TOOL_FAILURE_MARKERS)

    # ── Final-answer evidence validation ────────────────────────────────

    MAX_ANSWER_VALIDATION_RETRIES = 2

    def _evidence_text(self, state: _LoopState) -> str:
        """Evidence exactly as the model saw it, in one string.

        Rendered/truncated text is used on purpose: validating against the raw
        untruncated result would bless facts the model could not have seen.
        """
        parts = [
            truncate_tool_result(str(result.result))
            for result in state.tool_calls_made
            if self._is_usable_tool_result(result)
        ]
        return "\n\n---\n\n".join(parts)

    def _check_answer(self, state: _LoopState, answer: str) -> EvidenceCheck:
        return validate_answer(
            answer or "",
            self._evidence_text(state),
            require_evidence=self.require_tool_call,
        )

    def _answer_validation_nudge(
        self, state: _LoopState, raw_answer: str, reason: str,
    ) -> bool:
        """Ask the model to repair an answer that failed evidence checks.

        Returns False when the retry budget is exhausted; the caller then
        terminates with an honest failure instead of returning the ungrounded
        answer.
        """
        state.validation_failures += 1
        if state.validation_failures > self.MAX_ANSWER_VALIDATION_RETRIES:
            return False
        answer = clean_user_answer(raw_answer) or strip_reasoning(raw_answer).strip()
        if answer:
            state.messages.append(Message.assistant(answer))
        state.messages.append(Message.user(
            f"你的最终答案未通过证据校验：{reason}。"
            "请只依据已有 Observation 重写 Final Answer；"
            "引用编号只能使用 Observation 中真实出现的 [R#]，"
            "参考来源必须与 SOURCES 完全一致；"
            "无法在 Observation 中找到依据的数字或日期不要写入答案。"
        ))
        return True

    def _required_tool_instruction(self) -> str:
        tools = "、".join(sorted(self.required_tool_names)) or "可用检索工具"
        return (
            f"回答前必须先调用 {tools} 并获得有效 Observation。"
            "不得依据模型记忆直接回答，也不得自行生成引用。请立即调用工具。"
        )

    def _required_evidence_failure_answer(self) -> str:
        if self.failure_answer:
            return self.failure_answer
        if "rag_search" in self.required_tool_names:
            return (
                "本次未能从知识库检索到可用依据，因此无法可靠回答。"
                "请确认相关文档已完成入库和向量索引；为避免误导，我不会使用模型记忆猜测答案。"
            )
        return (
            "当前问题需要实时网络信息，但本次未能通过 MCP 获得有效搜索结果，"
            "因此无法可靠确认。请检查 MCP 搜索服务；为避免误导，我不会使用模型记忆猜测答案。"
        )

    @staticmethod
    def _answer_format_failure() -> str:
        return (
            "检索已完成，但模型未能生成符合最终答案格式的可靠回复。"
            "为避免暴露内部推理或给出未经确认的结论，本次不返回推测性答案。"
        )

    def _seed_tool_arguments(self, input: AgentInput) -> dict:
        """Arguments for the deterministic initial tool call.

        Follow-up questions are often coreferential ("它的副作用呢？") — a
        bare query retrieval misses, so the most recent past user turn is
        prepended to make the seeded call self-contained.  History is only
        included when the current query contains an explicit coreference;
        unconditional concatenation pollutes retrieval when the user starts a
        new topic.  Only one turn, and only *user* turns, is considered because
        assistant answers are untrusted content.
        """
        # ``retrieval_query`` is a self-contained seed prepared by the media
        # context layer (query + media digest).  Falling back to ``query``
        # keeps text-only turns unchanged.
        base_query = (input.retrieval_query or "").strip() or input.query
        recent = [
            str(m.content)[:150] for m in input.messages
            if m.role.value == "user" and isinstance(m.content, str)
        ][-1:]
        if recent and self._contains_coreference(input.query):
            query = recent[0] + " / " + base_query
        else:
            query = base_query
        return {self._seed_query_parameter(): query[:1200]}

    def _seed_query_parameter(self) -> str:
        """Pick the search-query parameter from the seeded tool's schema."""
        tool = self._tool_map.get(self.initial_tool_name)
        schema = getattr(tool, "parameters_schema", {}) or {}
        properties = schema.get("properties", {})
        preferred = ("query", "search_query", "q", "text", "keyword", "keywords")
        parameter = next(
            (name for name in preferred if name in properties), None,
        )
        if parameter is None:
            required = schema.get("required", [])
            parameter = required[0] if required else "query"
        return parameter

    @staticmethod
    def _contains_coreference(query: str) -> bool:
        """Return whether *query* explicitly refers to prior context.

        This intentionally stays conservative: only common Chinese and
        English pronouns/demonstratives are treated as coreference markers.
        Ordinary words containing short markers (for example ``其`` in an
        unrelated identifier) must not trigger history injection.
        """
        if not isinstance(query, str) or not query.strip():
            return False
        return bool(re.search(
            r"(?:它|他|她|其(?:的|中|所)?|这(?:个|些|样|里|种)?|那(?:个|些|样|里|种)?|此|该(?:问题|内容|文档|信息)?|上述|前者|后者|相关(?:内容|问题|文档|信息)?|"
            r"\b(?:it|this|that|these|those|上述)\b)",
            query,
            flags=re.IGNORECASE,
        ))

    # ── Sufficiency gate ───────────────────────────────────────────────

    def _sufficiency_enabled(self) -> bool:
        """Only complex plans pay for an explicit coverage check."""
        return bool(
            self.enable_sufficiency_check
            and self.query_plan is not None
            and self.query_plan.is_complex
        )

    async def _run_sufficiency_check(
        self, state: _LoopState, input: AgentInput,
    ) -> SufficiencyResult:
        rounds = 0
        if self.initial_tool_name:
            rounds = state.executor.call_counts.get(self.initial_tool_name, 0)
        checker = SufficiencyChecker(llm=self.llm)
        result = await checker.check(
            input.query,
            self._evidence_text(state),
            sub_queries=(
                self.query_plan.sub_queries if self.query_plan else ()
            ),
            rounds=rounds,
            max_rounds=self.max_retrieval_calls,
        )
        state.sufficiency_checks += 1
        state.last_sufficiency = result.diagnostics()
        for key in ("prompt_tokens", "completion_tokens"):
            value = result.usage.get(key, 0) or 0
            if value:
                state.total_usage[key] = state.total_usage.get(key, 0) + value
        return result

    def _supplementary_queries(
        self, state: _LoopState, input: AgentInput, result: SufficiencyResult,
    ) -> list[str]:
        """Candidate rewrites, deduplicated against executed retrieval calls."""
        tool_name = self.initial_tool_name
        if not tool_name or tool_name not in self._tool_map:
            tool_name = "rag_search" if "rag_search" in self._tool_map else ""
        if not tool_name:
            return []
        base = " ".join(
            (input.retrieval_query or input.query or "").split()
        ).strip()[:1200]
        candidates = []
        if result.rewritten_query:
            candidates.append(result.rewritten_query)
        if self.query_plan is not None:
            candidates.extend(self.query_plan.sub_queries)
        if result.missing and base:
            candidates.append(f"{base} {result.missing}")
        candidates.append(base)

        usable: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            query = " ".join((candidate or "").split()).strip()[:1200]
            if not query or query in seen:
                continue
            seen.add(query)
            arguments = {"query": query}
            if state.executor.is_duplicate(tool_name, arguments):
                continue
            usable.append(query)
        return usable

    async def _run_supplementary_retrieval(
        self,
        state: _LoopState,
        input: AgentInput,
        result: SufficiencyResult,
        turn_id: str,
        answer: str = "",
    ) -> tuple[list[AgentEvent], bool]:
        """Execute at most one checker-driven rewrite through the seed tool."""
        tool_name = self.initial_tool_name
        if not tool_name or tool_name not in self._tool_map:
            tool_name = "rag_search" if "rag_search" in self._tool_map else ""
        if not tool_name:
            return [], False
        if state.sufficiency_retrievals >= 1:
            return [], False
        if state.executor.is_budget_exhausted(tool_name):
            return [], False

        queries = self._supplementary_queries(state, input, result)
        if not queries:
            return [], False

        state.sufficiency_retrievals += 1
        # Keep the rejected answer in the transcript so the next round sees
        # what it must improve on and roles stay coherent.
        if answer:
            state.messages.append(Message.assistant(answer))
        events: list[AgentEvent] = []
        async for event in self._dispatch_tool_call(
            state, tool_name, {"query": queries[0]}, turn_id,
        ):
            events.append(event)
        state.messages.append(Message.user(
            "【补充检索完成】已根据充分性检查的改写查询补充证据。"
            "请结合全部 Observation 重新输出 Final Answer；"
            "仍无法确认的部分必须明确说明，禁止用模型记忆补全。"
        ))
        return events, True

    async def _gate_final_answer(
        self,
        state: _LoopState,
        input: AgentInput,
        raw_answer: str,
        iteration: int,
        turn_id: str,
    ) -> tuple[list[AgentEvent], bool]:
        """Run sufficiency + evidence validation before accepting an answer.

        Returns ``(events, terminal)``.  When ``terminal`` is False the loop
        must continue: either a supplementary retrieval was executed or the
        model was nudged to repair its answer.
        """
        answer = clean_user_answer(raw_answer)
        check = self._check_answer(state, answer)

        if self._sufficiency_enabled():
            sufficiency = await self._run_sufficiency_check(state, input)
            if not sufficiency.sufficient:
                events, retrieved = await self._run_supplementary_retrieval(
                    state, input, sufficiency, turn_id, answer=answer,
                )
                if retrieved:
                    return events, False

        if not check.ok:
            if self._answer_validation_nudge(state, raw_answer, check.reason):
                return [], False
            return [
                self._done(
                    state, self._answer_format_failure(),
                    iteration + 1, turn_id,
                )
            ], True

        return [
            self._done(state, check.normalized_answer, iteration + 1, turn_id)
        ], True

    # ── The single loop ────────────────────────────────────────────────

    async def _stream_loop(self, input: AgentInput, turn_id: str) -> AsyncIterator[AgentEvent]:
        """Drive the Think → Act → Observe loop, yielding events as it goes."""
        executor = _ToolExecutor(
            self.tools,
            allow_registry_fallback=self.allow_registry_fallback,
            max_calls=self._retrieval_limits,
        )
        state = _LoopState(executor)
        state.messages = self._build_initial_messages(input)

        if self.require_tool_call and not self.initial_tool_name and not self.tools:
            yield self._done(state, self._required_evidence_failure_answer(), 0, turn_id)
            return

        if self.initial_tool_name:
            arguments = (
                dict(self.seed_tool_arguments)
                if self.seed_tool_arguments is not None
                else self._seed_tool_arguments(input)
            )
            executor.executed_sigs.add(executor.signature(self.initial_tool_name, arguments))
            yield AgentEvent(
                event_type=AgentEventType.TOOL_CALL_START,
                data={"tool": self.initial_tool_name, "input": arguments},
                turn_id=turn_id,
            )
            result = await executor.execute(self.initial_tool_name, arguments)
            state.tool_calls_made.append(result)
            yield AgentEvent(
                event_type=AgentEventType.TOOL_CALL_RESULT,
                data={
                    "tool": self.initial_tool_name,
                    "success": self._is_usable_tool_result(result),
                    "result": str(result.result)[:500] if result.result else "",
                    "error": result.error,
                },
                turn_id=turn_id,
            )
            state.messages.append(Message.user(format_observation(
                self.initial_tool_name,
                str(result.result) if result.result else "",
                result.error,
            )))
            if self._is_required_evidence_missing(state.tool_calls_made) and not self._other_evidence_tools_available():
                # A miss is not always conclusive: the model gets one chance
                # to rewrite the query and retry the same retrieval tool when
                # budget remains.  Without budget the old fail-fast behaviour
                # applies (no other evidence source exists).
                retry_available = bool(
                    self.initial_tool_name
                    and not executor.is_budget_exhausted(
                        self.initial_tool_name
                    )
                )
                if not retry_available:
                    yield self._done(state, self._required_evidence_failure_answer(), 0, turn_id)
                    return
                state.seed_retry_pending = True
                state.messages.append(Message.user(
                    "首轮检索没有返回可用内容。"
                    f"请改写查询后再次调用 {self.initial_tool_name}"
                    "（本轮最后一次检索）；若仍无结果，再明确说明无法确认。"
                ))

        for iteration in range(self.max_iterations):
            if (
                state.seed_retry_pending
                and iteration >= 1
                and self._is_required_evidence_missing(
                    state.tool_calls_made
                )
            ):
                # The single post-miss retry round has passed without usable
                # evidence; stop instead of letting the model loop.
                yield self._done(
                    state, self._required_evidence_failure_answer(),
                    iteration, turn_id,
                )
                return
            yield AgentEvent(
                event_type=AgentEventType.THOUGHT,
                data={"iteration": iteration},
                turn_id=turn_id,
            )

            # Contract for the NEXT turn: a native function-call round ends with
            # "continue", so if no new tool result was appended after this point
            # the model must answer in text ReAct format (never emit format-only
            # text alongside a function call, or the final tool result gets
            # ignored and the last observation is lost).  Only relevant when
            # native tools are on the wire — in text ReAct mode this contract
            # contradicts the required Thought/Action format and confuses small
            # models.
            n_msgs_before = len(state.messages)
            if self._send_native_tools:
                state.messages.append(Message.user(
                    "【重要】如果你决定调用工具，请只发起工具调用，不要同时输出任何 Thought/Action 格式行；"
                    "如果你不调用工具，请按格式输出：Thought: ... Final Answer: ...（或 Thought/Action/Action Input 发起文本式工具调用）。"
                ))
            llm_messages = list(state.messages)
            if len(state.messages) > n_msgs_before:
                state.messages.pop()  # remove the contract — don't pollute history

            can_stream_answer = not self._is_required_evidence_missing(state.tool_calls_made)
            round_ = await self._generate_round(llm_messages, turn_id, can_stream_answer)
            for event in round_.events:
                yield event

            for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
                value = round_.usage.get(key, 0) or 0
                if value:
                    state.total_usage[key] = state.total_usage.get(key, 0) + value

            # ── Handle stream error ──
            if round_.error and not round_.full_content.strip():
                yield AgentEvent(
                    event_type=AgentEventType.ERROR,
                    data={"error": f"LLM stream failed: {round_.error}"},
                    turn_id=turn_id,
                )
                yield self._done(
                    state,
                    f"抱歉，模型服务连接中断：{round_.error}，请稍后重试。",
                    iteration + 1,
                    turn_id,
                )
                return

            # ── Resolve action: native function calling takes priority ──
            if round_.has_native_tool_call and round_.native_tool_name:
                state.invalid_output_count = 0  # Reset on valid tool call
                tool_name = resolve_tool_name(round_.native_tool_name, executor.names)
                try:
                    action_input = json.loads(round_.native_tool_args) if round_.native_tool_args else {}
                except json.JSONDecodeError:
                    action_input = {"query": round_.native_tool_args} if round_.native_tool_args else {}

                # Record a protocol-valid tool_calls assistant message so the
                # following Message.tool(tool_call_id=...) is legal for the
                # next request (OpenAI requires the pairing). Content stays
                # empty: text alongside a tool call is a provider artifact
                # (e.g. a serialized <tool_call> block echoed as text), and
                # re-recording it would make small models re-emit the same
                # reasoning/action on later turns. Streaming providers only
                # send the call id in the first delta, so fall back to a
                # generated id when none arrived.
                call_id = round_.native_call_id or f"call_{uuid.uuid4().hex[:12]}"
                state.messages.append(Message.assistant(
                    "",
                    tool_calls=[ToolCall(id=call_id, name=tool_name, arguments=action_input)],
                ))
                async for event in self._dispatch_tool_call(
                    state, tool_name, action_input, turn_id, native_call_id=call_id,
                ):
                    yield event
                continue

            # ── Fallback: parse ReAct text format ──
            step = parse_react_output(round_.full_content, executor.names)

            if step.is_final:
                if self._is_required_evidence_missing(state.tool_calls_made):
                    state.invalid_output_count += 1
                    state.messages.append(Message.assistant(strip_reasoning(round_.full_content)))
                    state.messages.append(Message.user(self._required_tool_instruction()))
                    continue

                events, terminal = await self._gate_final_answer(
                    state, input, step.final_answer, iteration, turn_id,
                )
                for event in events:
                    yield event
                if terminal:
                    return
                continue

            if step.action:
                state.invalid_output_count = 0  # Reset on valid action
                if not step.action_input:
                    state.messages.append(Message.user(
                        f"'{step.action}' 需要参数，请在 Action Input 中提供 JSON。"
                    ))
                    continue
                # Record a compact canonical call rather than the raw model
                # text — small models otherwise repeat their full preamble.
                state.messages.append(Message.assistant(
                    f"Thought: 调用 {step.action}\nAction: {step.action}\n"
                    f"Action Input: {json.dumps(step.action_input, ensure_ascii=False)}"
                ))
                async for event in self._dispatch_tool_call(state, step.action, step.action_input, turn_id):
                    yield event
                continue

            # ── No valid action and no final answer — output is unparseable ──
            done_events = await self._handle_invalid_output(state, round_.full_content, iteration, turn_id)
            if done_events is not None:
                for event in done_events:
                    yield event
                return

        # Max iterations — force LLM to give a final answer with what it has
        if self._is_required_evidence_missing(state.tool_calls_made):
            final = self._required_evidence_failure_answer()
        else:
            forced = await self._force_final_answer(state.messages)
            check = self._check_answer(state, forced or "")
            final = check.normalized_answer if check.ok else ""
        yield self._done(state, final or self._answer_format_failure(), self.max_iterations, turn_id)

    async def _handle_invalid_output(
        self, state: _LoopState, full_content: str, iteration: int, turn_id: str,
    ) -> Optional[list[AgentEvent]]:
        """Handle a round that produced neither an action nor a final answer.

        Returns the terminating events, or None to nudge the model and
        continue. With no tool calls at all, a bare plausible answer is never
        accepted — the strict format nudge (or evidence refusal) applies
        instead.
        """
        validation_reason = ""
        if not self._is_required_evidence_missing(state.tool_calls_made):
            candidate = extract_answer_candidate(full_content)
            if candidate:
                check = self._check_answer(state, candidate)
                if check.ok:
                    return [
                        AgentEvent(
                            event_type=AgentEventType.TEXT_DELTA,
                            data={"content": check.normalized_answer},
                            turn_id=turn_id,
                        ),
                        self._done(
                            state, check.normalized_answer,
                            iteration + 1, turn_id,
                        ),
                    ]
                validation_reason = check.reason

        state.invalid_output_count += 1
        # When evidence is already on the wire, one bare-answer nudge
        # is cheap and often enough for small models that stall after
        # a real tool result; deeper stalls keep the strict path.
        max_invalid_attempts = 2 if not state.tool_calls_made else 3
        if state.invalid_output_count < max_invalid_attempts:
            # Prompt LLM to continue with correct format
            nudge = (
                "你的输出格式不正确。请严格按照以下格式之一输出：\n"
                "1. 调用工具：Thought: ...\nAction: tool_name\nAction Input: {\"param\": \"value\"}\n"
                "2. 最终答案：Thought: ...\nFinal Answer: ...\n"
                "禁止输出 Observation 行，也禁止复述、引用或转述系统返回的 Observation 内容作为你的输出。"
            )
            if validation_reason:
                nudge += f"\n上次答案未通过证据校验：{validation_reason}。"
            state.messages.append(Message.user(nudge))
            return None

        # Force exit after repeated invalid outputs to prevent infinite loop
        if self._is_required_evidence_missing(state.tool_calls_made):
            final = self._required_evidence_failure_answer()
        else:
            final = extract_answer_candidate(full_content)
            if final:
                check = self._check_answer(state, final)
                final = check.normalized_answer if check.ok else ""
            if not final:
                forced = await self._force_final_answer(state.messages)
                check = self._check_answer(state, forced or "")
                final = check.normalized_answer if check.ok else ""

        return [self._done(state, final or self._answer_format_failure(), iteration + 1, turn_id)]

    async def _dispatch_tool_call(
        self,
        state: _LoopState,
        tool_name: str,
        action_input: dict,
        turn_id: str,
        native_call_id: str = "",
    ) -> AsyncIterator[AgentEvent]:
        """Execute one tool call (or refuse a duplicate) and update history.

        Yields TOOL_CALL_START/TOOL_CALL_RESULT around the execution, or a
        THOUGHT marker when the call is rejected as a duplicate. Native
        rounds without arguments append a tool message so the protocol stays
        valid for the next request.
        """
        executor = state.executor
        if not action_input and native_call_id:
            state.messages.append(Message.tool(
                content=f"Error: '{tool_name}' 缺少参数",
                tool_call_id=native_call_id,
            ))
            return
        if executor.is_duplicate(tool_name, action_input):
            if native_call_id:
                state.messages.append(Message.tool(
                    content=f"Error: 禁止重复调用 '{tool_name}'（参数相同）。请基于已有 Observation 直接输出 Final Answer。",
                    tool_call_id=native_call_id,
                ))
            else:
                state.messages.append(Message.user(
                    f"你已经用相同参数调用过 '{tool_name}'。请根据已有 Observation 输出 Final Answer。"
                ))
            yield AgentEvent(
                event_type=AgentEventType.THOUGHT,
                data={"duplicate_call": tool_name},
                turn_id=turn_id,
            )
            return

        if executor.is_budget_exhausted(tool_name):
            limit = executor.max_calls.get(tool_name, 0)
            message = (
                f"检索工具 '{tool_name}' 已达到本轮最多 {limit} 次调用上限。"
                "请基于已有 Observation 输出 Final Answer；"
                "无法确认的部分必须明确说明，禁止用模型记忆补全。"
            )
            if native_call_id:
                state.messages.append(Message.tool(
                    content=message, tool_call_id=native_call_id,
                ))
            else:
                state.messages.append(Message.user(message))
            yield AgentEvent(
                event_type=AgentEventType.THOUGHT,
                data={"retrieval_budget_exhausted": tool_name},
                turn_id=turn_id,
            )
            return

        executor.executed_sigs.add(executor.signature(tool_name, action_input))
        yield AgentEvent(
            event_type=AgentEventType.TOOL_CALL_START,
            data={"tool": tool_name, "input": action_input},
            turn_id=turn_id,
        )
        result = await executor.execute(tool_name, action_input)
        state.tool_calls_made.append(result)
        yield AgentEvent(
            event_type=AgentEventType.TOOL_CALL_RESULT,
            data={
                "tool": tool_name,
                "success": not result.error,
                "result": str(result.result)[:500] if result.result else "",
                "error": result.error,
            },
            turn_id=turn_id,
        )

        if native_call_id:
            state.messages.append(Message.tool(
                content=format_tool_message(
                    tool_name,
                    str(result.result) if result.result else "",
                    result.error,
                ),
                tool_call_id=native_call_id,
            ))
        else:
            state.messages.append(Message.user(format_observation(
                tool_name,
                str(result.result) if result.result else "",
                result.error,
            )))

    # ── LLM round ──────────────────────────────────────────────────────

    async def _generate_round(self, messages: list[Message], turn_id: str, can_stream_answer: bool = True):
        """Run one LLM round (streaming or not) into a normalized result.

        Handles the provider quirks centrally: reasoning stripping, answer
        gating for TEXT_DELTA events, and the empty-stream non-streaming
        retry for servers that emit nothing when tools are attached.
        """
        round_ = _RoundResult()
        round_.can_stream_answer = can_stream_answer
        # Answer gating: only stream the actual answer text to the UI.
        # "Thought:" lines and any pre-answer monologue are buffered and
        # dropped — the final answer is extracted from full_content below.
        pending_delta = ""
        answer_streaming = False

        async def _emit_visible_answer(delta: str):
            nonlocal pending_delta, answer_streaming
            if answer_streaming:
                emit = delta
            else:
                pending_delta += delta
                emit = extract_final_answer(pending_delta) or ""
                if emit:
                    answer_streaming = True
            if emit:
                round_.events.append(AgentEvent(
                    event_type=AgentEventType.TEXT_DELTA,
                    data={"content": emit},
                    turn_id=turn_id,
                ))

        # Incremental reasoning stripper — thinking models (<think> blocks)
        # must not leak their reasoning to the UI or into the ReAct parser.
        think_filter = ReasoningStreamFilter()
        tools = self._get_llm_tool_definitions()
        stop = self._react_stop()

        if self._use_streaming:
            try:
                async for chunk in self.llm.agenerate_stream(messages, tools, stop=stop):
                    if chunk.content_delta:
                        delta = think_filter.feed(chunk.content_delta)
                        round_.full_content += delta
                        # After tool call detection, further text is likely
                        # post-tool narration — suppress to reduce noise.
                        if not delta or round_.has_native_tool_call or not round_.can_stream_answer:
                            continue
                        await _emit_visible_answer(delta)
                    # Collect native function-call deltas (Qwen / OpenAI function calling)
                    if chunk.tool_call_delta:
                        round_.has_native_tool_call = True
                        if chunk.tool_call_delta.get("id"):
                            round_.native_call_id = chunk.tool_call_delta["id"]
                        if chunk.tool_call_delta.get("name"):
                            round_.native_tool_name = chunk.tool_call_delta["name"]
                        if chunk.tool_call_delta.get("arguments"):
                            round_.native_tool_args += chunk.tool_call_delta["arguments"]
                # Flush text held back for partial-tag detection
                tail = think_filter.flush()
                if tail:
                    round_.full_content += tail
                    if not round_.has_native_tool_call and round_.can_stream_answer:
                        await _emit_visible_answer(tail)
                round_.usage = getattr(self.llm, "last_usage", None) or {}
            except Exception as e:
                round_.error = str(e)
                logger.exception("LLM stream error")
        else:
            # Non-streaming round-trip for servers that cannot stream tool
            # calls reliably: one agenerate() call, then emit the visible
            # answer text (if any) as a single TEXT_DELTA.
            try:
                response = await self.llm.agenerate(messages, tools, stop=stop)
            except Exception as e:
                round_.error = str(e)
                logger.exception("LLM generate error")
            else:
                round_.usage = response.usage or {}
                if response.tool_calls:
                    tc = response.tool_calls[0]
                    round_.has_native_tool_call = True
                    round_.native_call_id = tc.id or ""
                    round_.native_tool_name = tc.name or ""
                    round_.native_tool_args = (
                        tc.arguments if isinstance(tc.arguments, str)
                        else json.dumps(tc.arguments or {}, ensure_ascii=False)
                    )
                if response.content:
                    # NOTE: no log_llm_result() here — agenerate() already
                    # logged this exact response in the provider layer.
                    round_.full_content = response.content
                    if not round_.has_native_tool_call and round_.can_stream_answer:
                        delta = think_filter.feed(round_.full_content) + think_filter.flush()
                        round_.full_content = delta
                        await _emit_visible_answer(delta)

        # A few local streaming endpoints currently emit neither text nor
        # streamed tool-call deltas even though native tools are enabled.
        # Recover with one non-streaming request carrying the same tools.
        # (On non-streaming-capable servers this is a second empty response,
        # which the invalid-output handling below terminates normally.)
        if self._use_streaming and not round_.error and not round_.full_content.strip() and not round_.has_native_tool_call and self.tools:
            retry = await self.llm.agenerate(messages, tools, stop=stop)
            round_.usage = retry.usage or round_.usage
            if retry.tool_calls:
                call = retry.tool_calls[0]
                round_.native_call_id = call.id or ""
                round_.native_tool_name = call.name
                round_.native_tool_args = json.dumps(call.arguments or {}, ensure_ascii=False)
                round_.has_native_tool_call = True
            else:
                round_.full_content = retry.content or ""

        return round_

    # ── Prompt / misc helpers ──────────────────────────────────────────

    def _build_initial_messages(self, input: AgentInput) -> list[Message]:
        """Build the initial message list for the ReAct loop.

        History appears exactly once: as proper chat messages after the
        system prompt (not flattened into the prompt text). The caller's
        ``system_prompt_template`` is honoured when provided; the default
        ReAct template is the fallback.
        """
        tools_desc = build_tools_description(self._get_tool_definitions())
        if self.system_prompt_template:
            system_prompt = self.system_prompt_template.format(
                tools_description=tools_desc,
            )
        else:
            system_prompt = build_react_prompt(tools_description=tools_desc)

        messages = [Message.system(system_prompt)]

        # Check if input already has a multimodal user message
        has_multimodal_query = any(
            isinstance(m.content, list) and m.role.value == "user"
            for m in input.messages
        )

        # Add conversation history (excluding system messages)
        for msg in input.messages:
            if msg.role.value != "system":
                messages.append(msg)

        # Add current query — skip if already included as multimodal message
        if not has_multimodal_query:
            query = input.query
            if input.multimodal and input.multimodal.text:
                query = input.multimodal.text
            messages.append(Message.user(query))

        return messages

    def _get_tool_definitions(self) -> list[ToolDefinition]:
        """Get all tool definitions for prompt construction and parsing."""
        return [t.to_definition() for t in self.tools]

    def _get_llm_tool_definitions(self) -> list[ToolDefinition]:
        """Get definitions passed through the provider's native tools API.

        Empty when the server is known to mishandle the ``tools`` parameter —
        the model then uses the text ReAct format from the system prompt.
        """
        if not self.enable_native_tool_calls or not self._send_native_tools:
            return []
        return self._get_tool_definitions()

    def _react_stop(self) -> Optional[list[str]]:
        """Stop sequences that halt generation before a fabricated Observation.

        In text ReAct mode the model must stop after ``Action Input: {...}`` so
        the engine can execute the real tool. Without this, small local models
        hallucinate their own ``Observation:`` (and a ``Final Answer:`` built on
        it) in a single generation — wasting tokens and risking that fabricated
        answer leaking if the parser's Action-precedence ever misfires. Only
        applied in text mode; native function-calling providers don't emit the
        ``Observation:`` marker as content, and gating avoids clipping any
        answer that legitimately contains the word.
        """
        if not self.tools or self._send_native_tools:
            return None
        return ["\nObservation:", "Observation:"]

    def _diagnostics(self, state: _LoopState | None = None) -> dict:
        evidence_count = 0
        retrieval_calls: dict[str, int] = {}
        sufficiency = {
            "enabled": self.enable_sufficiency_check,
            "checks": 0,
            "supplementary_retrievals": 0,
            "last": {},
        }
        if state is not None:
            evidence_count = sum(
                1 for result in state.tool_calls_made
                if self._is_usable_tool_result(result)
            )
            retrieval_calls = dict(state.executor.call_counts)
            sufficiency.update({
                "checks": state.sufficiency_checks,
                "supplementary_retrievals": state.sufficiency_retrievals,
                "last": dict(state.last_sufficiency),
            })
        return {
            "engine_revision": "react-open-v1",
            "strategy": "react",
            "route_reason": self.route_reason,
            "selected_tool": ",".join(sorted(self._tool_map)),
            "evidence_source_count": evidence_count,
            "planner": (
                self.query_plan.diagnostics()
                if self.query_plan is not None else None
            ),
            "retrieval": {
                "calls": retrieval_calls,
                "budget": dict(self._retrieval_limits),
            },
            "sufficiency": sufficiency,
        }

    async def _force_final_answer(self, messages: list[Message]) -> str:
        """Force the LLM to produce a final answer when max iterations are reached.

        The closing instruction goes out as a *user* message: some
        OpenAI-compatible local servers reject, ignore, or template-break on
        a system message that isn't the first message of the request.
        """
        final_messages = [
            *messages,
            Message.user(
                "【系统收尾指令】仅依据已有 Observation 输出一行最终答案，"
                "格式必须为 `Final Answer: <答案>` 或 `最终答案：<答案>`。"
                "禁止输出 Thought、Action、过程说明、系统指令或未经 Observation 支持的事实。"
            ),
        ]
        response = await self.llm.agenerate(final_messages)
        return extract_answer_candidate(response.content)
