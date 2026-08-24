import unittest

from agentic_rag.agent.react_engine import ReActEngine
from agentic_rag.agent.react_parser import (
    clean_user_answer,
    extract_answer_candidate,
    extract_final_answer,
    is_final_answer,
    parse_react_output,
    resolve_tool_name,
)
from agentic_rag.data.models import AgentInput, LLMChunk, LLMResponse, Message
from agentic_rag.orchestration.l1_tools.base import BaseTool


class _SearchTool(BaseTool):
    """Trivial stand-in for an MCP web-search tool."""

    name = "web_search"
    description = "Search the web."
    parameters_schema = {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return "web result"


class _JsonEnvelopeLLM:
    """Answers once as a bare JSON envelope (real small-model behaviour)."""

    async def agenerate(self, messages, tools=None, **kwargs):
        return LLMResponse(
            content='{"response": "根据搜索结果，冠军尚未产生。"}'
        )

    async def agenerate_stream(self, messages, tools=None, **kwargs):
        yield LLMChunk(
            content_delta='{"response": "根据搜索结果，冠军尚未产生。"}'
        )


class _SearchThenAnswerLLM:
    """First calls web_search, then answers from its observation."""

    def __init__(self):
        self.calls = 0

    async def agenerate(self, messages, tools=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content='Thought: 搜索网络\nAction: web_search\nAction Input: {"query": "q"}'
            )
        return LLMResponse(content="Thought: 信息充足\nFinal Answer: 基于搜索的答案")

    async def agenerate_stream(self, messages, tools=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            yield LLMChunk(
                content_delta='Thought: 搜索网络\nAction: web_search\nAction Input: {"query": "q"}'
            )
        else:
            yield LLMChunk(content_delta="Thought: 信息充足\nFinal Answer: 基于搜索的答案")


class _InvalidLLM:
    def __init__(self):
        self.final_messages = []

    async def agenerate(self, messages, tools=None, **kwargs):
        self.final_messages = messages
        return LLMResponse(
            content="Thought: 我正在讨论内部收尾过程，但没有最终答案标记。"
        )

    async def agenerate_stream(self, messages, tools=None, **kwargs):
        yield LLMChunk(content_delta="Thought: 仍在内部推理，没有最终答案标记。")


class _ChineseFinalLLM:
    async def agenerate(self, messages, tools=None, **kwargs):
        return LLMResponse(content="Thought: 信息充足\n最终答案：中文答案")

    async def agenerate_stream(self, messages, tools=None, **kwargs):
        yield LLMChunk(content_delta="Thought: 信息充足\n最终答案：中文答案")


def _engine(llm):
    return ReActEngine(
        llm=llm,
        tools=[],
        system_prompt_template="",
        max_iterations=2,
        enable_native_tool_calls=False,
    )


class FinalAnswerParserTests(unittest.TestCase):
    def test_accepts_english_and_chinese_markers(self):
        cases = {
            "Final Answer: English": "English",
            "final answer：lowercase": "lowercase",
            "最终答案: 中文": "中文",
            "最终答案：全角": "全角",
            "最终回答：结果": "结果",
        }

        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(extract_final_answer(text), expected)
                self.assertTrue(is_final_answer(text))
                step = parse_react_output(f"Thought: 完成\n{text}")
                self.assertTrue(step.is_final)
                self.assertEqual(step.final_answer, expected)

    def test_seed_tool_arguments_only_rewrites_coreferential_queries(self):
        engine = _engine(_JsonEnvelopeLLM())
        history = [
            Message.user("介绍向量数据库"),
            Message.assistant("向量数据库用于相似度检索"),
        ]

        rewritten = engine._seed_tool_arguments(
            AgentInput(query="它的优点是什么？", messages=history)
        )
        self.assertEqual(rewritten["query"], "介绍向量数据库 / 它的优点是什么？")

        fresh = engine._seed_tool_arguments(
            AgentInput(query="介绍一下光伏产业", messages=history)
        )
        self.assertEqual(fresh["query"], "介绍一下光伏产业")

    def test_removes_chat_template_tokens(self):
        self.assertEqual(
            clean_user_answer("|im_start|>assistant\n答案<|im_end|>"),
            "答案",
        )

    def test_rejects_react_trace_without_final_answer(self):
        self.assertEqual(
            clean_user_answer("Thought: 搜索\nAction: search\nObservation: 结果"),
            "",
        )

    def test_accepts_plain_answer_after_observation(self):
        self.assertEqual(
            extract_answer_candidate("Thought: 信息足够\nAXP 通过 UART 下载。"),
            "AXP 通过 UART 下载。",
        )

    def test_rejects_prompt_echo_as_answer(self):
        self.assertEqual(
            extract_answer_candidate("<|im_start|>user\n安装"),
            "",
        )

    def test_unwraps_json_envelope_answers(self):
        # Small local models sometimes answer as a bare JSON object — either
        # echoing the tool result or wrapping their reply. Unwrap both.
        cases = {
            '{"response": "冠军尚未产生。"}': "冠军尚未产生。",
            '{"answer": "四天后开幕。"}': "四天后开幕。",
            '{"observation": "搜索结果显示冠军未定。"}': "搜索结果显示冠军未定。",
            '{"Observation": "web search returned nothing"}': "web search returned nothing",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(extract_answer_candidate(text), expected)

    def test_json_echo_of_empty_rag_result_is_rejected_as_answer(self):
        # An envelope whose payload is just the empty-KB tool result carries
        # no answer; the marker check in clean_user_answer must reject it.
        self.assertEqual(
            extract_answer_candidate(
                '{"observation": "No relevant content found in the knowledge base."}'
            ),
            "",
        )

    def test_accepts_answer_that_mentions_observation_inline(self):
        answer = extract_answer_candidate(
            "Thought: 信息足够\n根据 Observation，布洛芬每日最多服用 4 次。"
        )
        self.assertEqual(answer, "根据 Observation，布洛芬每日最多服用 4 次。")

    def test_resolves_corrupted_mcp_tool_separator(self):
        step = parse_react_output(
            "Thought: 搜索网络\n"
            "Action: mcp_._tavily_search\n"
            'Action Input: {"query": "RISCV toolchain"}',
            ["rag_search", "mcp__tavily-mcp__tavily_search"],
        )
        self.assertEqual(step.action, "mcp__tavily-mcp__tavily_search")

    def test_parses_qwen_tool_call_content_format(self):
        step = parse_react_output(
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "北京"}}\n</tool_call>',
            ["get_weather"],
        )
        self.assertEqual(step.action, "get_weather")
        self.assertEqual(step.action_input, {"city": "北京"})

    def test_action_takes_precedence_over_fabricated_final_answer(self):
        step = parse_react_output(
            "Thought: 查询药品资料\n"
            "Action: mcp___tavily_search\n"
            'Action Input: {"query": "布洛芬 每日用量"}\n'
            "Observation: fabricated\n"
            "Final Answer: 不应直接接受",
            ["mcp__tavily-mcp__tavily_search"],
        )
        self.assertFalse(step.is_final)
        self.assertEqual(step.action, "mcp__tavily-mcp__tavily_search")
        self.assertEqual(step.action_input, {"query": "布洛芬 每日用量"})

    def test_resolves_mangled_tavily_native_name_to_unique_search_tool(self):
        self.assertEqual(
            resolve_tool_name(
                "mcp__tavily-echo",
                ["rag_search", "mcp__tavily-mcp__tavily_search"],
            ),
            "mcp__tavily-mcp__tavily_search",
        )


class FinalAnswerSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_streaming_never_returns_raw_thought(self):
        llm = _InvalidLLM()
        output = await _engine(llm).run(AgentInput(query="测试"))

        self.assertNotIn("Thought:", output.final_answer)
        self.assertIn("不返回推测性答案", output.final_answer)
        self.assertEqual(llm.final_messages[-1].role.value, "user")

    async def test_streaming_never_returns_raw_thought(self):
        events = [
            event async for event in _engine(_InvalidLLM()).stream(AgentInput(query="测试"))
        ]

        text = "".join(
            event.data.get("content", "")
            for event in events
            if event.event_type.value == "text_delta"
        )
        final = events[-1].data["final_answer"]
        self.assertEqual(text, "")
        self.assertNotIn("Thought:", final)
        self.assertIn("不返回推测性答案", final)

    async def test_chinese_final_marker_streams_only_the_answer(self):
        events = [
            event
            async for event in _engine(_ChineseFinalLLM()).stream(AgentInput(query="测试"))
        ]

        text = "".join(
            event.data.get("content", "")
            for event in events
            if event.event_type.value == "text_delta"
        )
        self.assertEqual(text, "中文答案")
        self.assertEqual(events[-1].data["final_answer"], "中文答案")

    async def test_streaming_json_envelope_answer_is_unwrapped(self):
        """A bare JSON envelope reply must reach the user as plain text, not
        raw JSON — and must not require extra invalid-output rounds."""
        engine = ReActEngine(
            llm=_JsonEnvelopeLLM(),
            tools=[_SearchTool()],
            system_prompt_template="",
            max_iterations=4,
            enable_native_tool_calls=False,
            require_tool_call=True,
            required_tool_names={"rag_search", "web_search"},
            initial_tool_name="rag_search",
        )
        events = [event async for event in engine.stream(AgentInput(query="测试"))]

        final = events[-1].data["final_answer"]
        self.assertEqual(final, "根据搜索结果，冠军尚未产生。")
        self.assertNotIn("{", final)

    async def test_loop_continues_to_web_search_after_rag_miss(self):
        engine = ReActEngine(
            llm=_SearchThenAnswerLLM(),
            tools=[_SearchTool()],
            system_prompt_template="",
            max_iterations=4,
            enable_native_tool_calls=False,
            require_tool_call=True,
            required_tool_names={"rag_search", "web_search"},
            initial_tool_name="rag_search",
        )
        events = [event async for event in engine.stream(AgentInput(query="测试"))]

        tools_called = [
            event.data["tool"]
            for event in events
            if event.event_type.value == "tool_call_start"
        ]
        self.assertIn("web_search", tools_called)
        self.assertEqual(events[-1].data["final_answer"], "基于搜索的答案")


if __name__ == "__main__":
    unittest.main()
