"""Mutable per-run state and normalized round results for the ReAct loop."""

from typing import Optional

from agentic_rag.agent.tool_executor import ToolExecutor
from agentic_rag.data.models import AgentEvent, Message, ToolCallResult


class LoopState:
    """Mutable state shared by the loop and the termination policy."""

    def __init__(self, executor: ToolExecutor):
        self.executor = executor
        self.messages: list[Message] = []
        self.tool_calls_made: list[ToolCallResult] = []
        self.invalid_output_count = 0
        self.validation_failures = 0
        self.sufficiency_checks = 0
        self.sufficiency_retrievals = 0
        self.last_sufficiency: dict = {}
        # Set when the deterministic seed missed and a rewritten retrieval is
        # the only remaining evidence path.  The loop then allows exactly one
        # model round to rewrite the query before refusing honestly.
        self.seed_retry_pending = False
        self.total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}


class RoundResult:
    """One LLM round normalized to a single shape for the loop.

    Whether the provider streamed text deltas, returned a non-streaming
    response, or issued a native function call, the loop only ever sees this
    — plus any TEXT_DELTA events the round already emitted.
    """

    def __init__(self):
        self.full_content = ""
        self.has_native_tool_call = False
        self.native_tool_name = ""
        self.native_tool_args = ""
        self.native_call_id = ""
        self.error: Optional[str] = None
        self.usage: dict[str, int] = {}
        self.events: list[AgentEvent] = []
        # Set by the loop before the round starts.
        self.can_stream_answer = True
