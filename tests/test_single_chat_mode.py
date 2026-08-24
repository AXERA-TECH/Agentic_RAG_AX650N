import unittest
from unittest.mock import patch

from agentic_rag.agent.react_engine import ReActEngine
from agentic_rag.agent.request_policy import RequestPolicy, RequestStrategy
from agentic_rag.agent.router import AgentRouter
from agentic_rag.agent.single_pass_engine import SinglePassEngine
from agentic_rag.data.models import AgentInput, LLMChunk, LLMResponse, ToolDefinition
from agentic_rag.entrypoints.rest.routes import chat as chat_module
from agentic_rag.entrypoints.rest.routes.chat import ChatRequest, ChatResponse
from agentic_rag.runtime.unified_context import UnifiedContext


class _Tool:
    def __init__(self, name: str, capabilities=(), source: str = "builtin", result="[R1] 命中知识库"):
        self.name = name
        self.capabilities = frozenset(capabilities)
        self.source = source
        self.result = result
        self.calls = []
        self.description = "test tool"
        self.parameters_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    def has_capability(self, capability: str) -> bool:
        return capability in self.capabilities

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        return self.result

    def to_definition(self):
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema,
        )


class _RagFirstLLM:
    def __init__(self, final_answer="知识库答案 [R1]"):
        self.final_answer = final_answer
        self.calls = 0

    @staticmethod
    def _user_query(messages):
        for msg in messages:
            if getattr(msg, "role", None) and msg.role.value == "user":
                content = getattr(msg, "content", "")
                if isinstance(content, str) and content and not content.startswith("【重要】"):
                    return content
        return ""

    async def agenerate(self, messages, tools=None, **kwargs):
        self.calls += 1
        query = self._user_query(messages)
        if self.calls == 1:
            return LLMResponse(
                content=(
                    "Thought: 查询知识库\n"
                    "Action: rag_search\n"
                    f"Action Input: {{\"query\": \"{query}\"}}"
                )
            )
        return LLMResponse(content=f"Thought: 信息充足\nFinal Answer: {self.final_answer}")

    async def agenerate_stream(self, messages, tools=None, **kwargs):
        self.calls += 1
        query = self._user_query(messages)
        if self.calls == 1:
            yield LLMChunk(content_delta="Thought: 查询知识库\n")
            yield LLMChunk(content_delta="Action: rag_search\n")
            yield LLMChunk(content_delta=f"Action Input: {{\"query\": \"{query}\"}}")
        else:
            yield LLMChunk(content_delta=f"Thought: 信息充足\nFinal Answer: {self.final_answer}")


class _McpFirstLLM(_RagFirstLLM):
    def __init__(self, final_answer="实时答案。来源：https://example.com/source"):
        super().__init__(final_answer=final_answer)

    async def agenerate(self, messages, tools=None, **kwargs):
        self.calls += 1
        query = self._user_query(messages)
        if self.calls == 1:
            return LLMResponse(
                content=(
                    "Thought: 搜索网络\n"
                    "Action: mcp__search__query\n"
                    f"Action Input: {{\"search_query\": \"{query}\"}}"
                )
            )
        return LLMResponse(content=f"Thought: 信息充足\nFinal Answer: {self.final_answer}")

    async def agenerate_stream(self, messages, tools=None, **kwargs):
        self.calls += 1
        query = self._user_query(messages)
        if self.calls == 1:
            yield LLMChunk(content_delta="Thought: 搜索网络\n")
            yield LLMChunk(content_delta="Action: mcp__search__query\n")
            yield LLMChunk(content_delta=f"Action Input: {{\"search_query\": \"{query}\"}}")
        else:
            yield LLMChunk(content_delta=f"Thought: 信息充足\nFinal Answer: {self.final_answer}")


class _Registry:
    def __init__(self, mcp_tools=None, rag_tool=None):
        self.tools = {
            "rag_search": rag_tool or _Tool("rag_search"),
            "image_understand": _Tool("image_understand"),
            "audio_transcribe": _Tool("audio_transcribe"),
            "video_analyze": _Tool("video_analyze"),
        }
        self.mcp_tools = mcp_tools or []

    def filter(self, names):
        return [self.tools[name] for name in names if name in self.tools]

    def get(self, name):
        return self.tools.get(name)

    def get_mcp_tools(self):
        return self.mcp_tools


class _SessionRepo:
    def get(self, session_id):
        return None

    def create_with_id(self, session_id, user_id=""):
        return None

    def add_message(self, *args, **kwargs):
        return None

    def get_messages(self, *args, **kwargs):
        return []


class SingleChatModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_router_uses_react_for_knowledge(self):
        engine = await AgentRouter(object(), _Registry()).route("普通问题")

        self.assertIsInstance(engine, ReActEngine)
        self.assertEqual([tool.name for tool in engine.tools], ["rag_search"])

    async def test_media_is_a_request_attribute_not_a_mode(self):
        engine = await AgentRouter(object(), _Registry()).route("描述附件", has_media=True)

        self.assertIsInstance(engine, SinglePassEngine)
        self.assertEqual(engine.strategy, RequestStrategy.DIRECT)
        self.assertIsNone(engine.tool)

    async def test_live_queries_include_mcp_search(self):
        mcp_search = _Tool(
            "mcp__search__query",
            capabilities={"fresh_information"},
            source="mcp",
            result="World Cup result https://example.com/source",
        )
        engine = await AgentRouter(object(), _Registry(mcp_tools=[mcp_search])).route("今天的比赛结果")

        self.assertIsInstance(engine, ReActEngine)
        self.assertIn("rag_search", [tool.name for tool in engine.tools])
        self.assertIn(mcp_search.name, [tool.name for tool in engine.tools])

    async def test_react_runs_rag_tool_before_answering(self):
        rag = _Tool("rag_search")
        engine = await AgentRouter(_RagFirstLLM(), _Registry(rag_tool=rag)).route("USB 报错")
        output = await engine.run(AgentInput(query="USB 报错"))

        self.assertEqual(rag.calls, [{"query": "USB 报错"}])
        self.assertEqual([call.name for call in output.tool_calls_made], ["rag_search"])
        self.assertEqual(output.final_answer, "知识库答案 [R1]")
        self.assertEqual(output.diagnostics["strategy"], "react")
        self.assertIn("rag_search", output.diagnostics["selected_tool"])

    async def test_react_streaming_emits_tool_events(self):
        rag = _Tool("rag_search")
        engine = await AgentRouter(_RagFirstLLM(), _Registry(rag_tool=rag)).route("USB 报错")
        events = [event async for event in engine.stream(AgentInput(query="USB 报错"))]

        self.assertEqual(rag.calls, [{"query": "USB 报错"}])
        # Retrieval-first: the knowledge base is queried deterministically before
        # the first model turn, so its tool events lead the stream.
        self.assertEqual(events[0].event_type.value, "tool_call_start")
        self.assertEqual(events[0].data["tool"], "rag_search")
        self.assertEqual(events[1].event_type.value, "tool_call_result")
        self.assertEqual(events[-1].event_type.value, "done")

    async def test_react_runs_mcp_tool_when_selected(self):
        mcp_search = _Tool(
            "mcp__search__query",
            capabilities={"fresh_information"},
            source="mcp",
            result="result https://example.com/source",
        )
        engine = await AgentRouter(
            _McpFirstLLM("实时答案。来源：https://example.com/source"),
            _Registry(mcp_tools=[mcp_search]),
        ).route("2026世界杯冠军队伍")
        output = await engine.run(AgentInput(query="2026世界杯冠军队伍"))

        self.assertEqual(mcp_search.calls, [{"search_query": "2026世界杯冠军队伍"}])
        # Retrieval-first: rag_search runs before the model's turn; the MCP tool
        # the model then selects also executes and feeds the final answer.
        self.assertEqual(output.tool_calls_made[0].name, "rag_search")
        self.assertIn("mcp__search__query", [call.name for call in output.tool_calls_made])
        self.assertIn("https://example.com/source", output.final_answer)

    async def test_rag_miss_falls_through_to_mcp_web(self):
        rag = _Tool("rag_search", result="No relevant content found in the knowledge base.")
        mcp_search = _Tool(
            "mcp__search__query",
            capabilities={"fresh_information"},
            source="mcp",
            result="冠军是 X 队 https://example.com/source",
        )
        engine = await AgentRouter(
            _McpFirstLLM("实时答案。来源：https://example.com/source"),
            _Registry(mcp_tools=[mcp_search], rag_tool=rag),
        ).route("布洛芬每日用量")
        output = await engine.run(AgentInput(query="布洛芬每日用量"))

        # rag runs first and misses, then the model falls through to the MCP
        # tool instead of the engine refusing on the empty knowledge base.
        self.assertEqual(rag.calls, [{"query": "布洛芬每日用量"}])
        self.assertEqual(mcp_search.calls, [{"search_query": "布洛芬每日用量"}])
        self.assertEqual(
            [call.name for call in output.tool_calls_made],
            ["rag_search", "mcp__search__query"],
        )
        self.assertIn("https://example.com/source", output.final_answer)
        self.assertNotIn("未能从知识库", output.final_answer)

    async def test_rag_miss_without_fallback_refuses_instead_of_guessing(self):
        rag = _Tool("rag_search", result="No relevant content found in the knowledge base.")
        llm = _RagFirstLLM()
        engine = await AgentRouter(llm, _Registry(rag_tool=rag)).route("布洛芬每日用量")
        output = await engine.run(AgentInput(query="布洛芬每日用量"))

        # No fallback tool exists, so an empty knowledge base yields an honest
        # refusal — never a memory-guessed answer, and without a model turn.
        self.assertEqual(rag.calls, [{"query": "布洛芬每日用量"}])
        self.assertEqual(llm.calls, 0)
        self.assertEqual([call.name for call in output.tool_calls_made], ["rag_search"])
        self.assertIn("知识库", output.final_answer)
        self.assertIn("模型记忆", output.final_answer)
        self.assertEqual(output.iterations, 0)

    async def test_deterministic_weekday_is_computed_without_llm_or_tools(self):
        engine = await AgentRouter(object(), _Registry()).route("今天星期几")

        self.assertEqual(engine.strategy, RequestStrategy.DIRECT)
        self.assertIn("星期", engine.direct_answer)
        self.assertIsNone(engine.tool)

    async def test_greeting_does_not_search(self):
        engine = await AgentRouter(_RagFirstLLM(), _Registry()).route("你好")

        self.assertEqual(engine.strategy, RequestStrategy.DIRECT)
        self.assertIsNone(engine.tool)

    def test_policy_keeps_date_calculation_ahead_of_live_search(self):
        self.assertEqual(RequestPolicy.decide("今天星期几1").strategy, RequestStrategy.DIRECT)
        self.assertEqual(RequestPolicy.decide("2026世界杯冠军队伍").strategy, RequestStrategy.TOOL_USE)
        self.assertEqual(RequestPolicy.decide("2026世界杯举办时间").strategy, RequestStrategy.TOOL_USE)

    def test_english_words_containing_hi_are_not_treated_as_greetings(self):
        self.assertEqual(RequestPolicy.decide("history of vector databases").strategy, RequestStrategy.TOOL_USE)

    def test_request_strategy_exposes_execution_shapes_only(self):
        self.assertEqual(set(RequestStrategy), {RequestStrategy.DIRECT, RequestStrategy.TOOL_USE})

    async def test_react_engine_exposes_diagnostics(self):
        engine = await AgentRouter(_RagFirstLLM(), _Registry()).route("普通问题")
        output = await engine.run(AgentInput(query="普通问题"))

        self.assertEqual(output.diagnostics["engine_revision"], "react-open-v1")
        self.assertEqual(output.diagnostics["strategy"], "react")

    def test_rest_chat_models_do_not_expose_mode(self):
        self.assertNotIn("mode", ChatRequest.model_fields)
        self.assertNotIn("mode", ChatResponse.model_fields)

    def test_unified_context_can_be_created_by_rest_entrypoint(self):
        context = UnifiedContext.create(session_id="test-session")
        self.assertEqual(context.session_id, "test-session")

    async def test_stream_endpoint_can_create_sse_response(self):
        request = ChatRequest(message="今天星期几", session_id="test-session")
        with (
            patch.object(chat_module, "get_llm", return_value=object()),
            patch("agentic_rag.data.db.session_repo.SessionRepo", _SessionRepo),
        ):
            response = await chat_module.chat_stream(request, None)

        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
