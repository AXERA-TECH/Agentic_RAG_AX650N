"""Hybrid routing: deterministic policy first, LLM planning only when useful.

The deterministic :class:`RequestPolicy` remains the default because it is
free, predictable and correct for the common cases (greetings, single-fact
knowledge lookups, live-information queries).  Multi-hop / comparison
requests benefit from a planner that can pick an execution strategy and
propose self-contained sub-queries.

Cost control is explicit:

* ``should_plan`` is a conservative, deterministic gate; simple questions
  never reach the planner model.
* a planner response must be valid JSON with a known strategy, otherwise the
  deterministic decision is used and the turn is marked ``complex`` so the
  sufficiency gate still runs.
* planner failures (timeout, provider error, missing ``agenerate``) never
  block the request.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from agentic_rag.agent.request_policy import (
    RequestDecision,
    RequestPolicy,
    RequestStrategy,
)
from agentic_rag.config.prompts import Prompts
from agentic_rag.data.models import Message
from agentic_rag.services.llm.base import strip_reasoning


@dataclass(frozen=True)
class RoutePlan:
    """The routing outcome plus optional multi-hop retrieval guidance."""

    decision: RequestDecision
    complexity: str = "simple"
    sub_queries: tuple[str, ...] = ()
    source: str = "rules"

    @property
    def is_complex(self) -> bool:
        return self.complexity == "complex"

    def diagnostics(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "strategy": self.decision.strategy.value,
            "complexity": self.complexity,
            "sub_queries": list(self.sub_queries),
            "preferred_capability": self.decision.preferred_capability,
            "reason": self.decision.reason,
        }


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a model response.

    Small local models often wrap JSON in code fences or prose and may emit a
    reasoning block first.  The decoder is used on every ``{`` position so
    nested objects (for example MCP tool arguments) are parsed correctly.
    """
    cleaned = strip_reasoning(text or "")
    candidates = [
        match.group(1)
        for match in re.finditer(
            r"```(?:json)?\s*(\{.*?\})\s*```", cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ]
    candidates.append(cleaned)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start():])
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                return value
    return None


class QueryPlanner:
    """Route complex requests through an LLM, with a deterministic fallback."""

    # Explicit comparison / multi-hop signals.  Deliberately conservative:
    # "维护步骤" and other ordinary KB questions must stay on the free path.
    _COMPLEX_PATTERNS = (
        re.compile(r"\b(?:compare|comparison|versus|vs\.?)\b"),
        re.compile(r"\b(?:and then|then also|as well as|respectively)\b"),
        re.compile(r"分别|对比|比较|异同|差异|优缺点|各自"),
        re.compile(r"并且|同时|另外|此外|还需|还要|并说明|并给出|并列出"),
        re.compile(r"先.{0,20}再(?:检索|查询|分析|说明|比较)"),
        re.compile(r"多角度|全面分析|综合分析|交叉验证|多步骤"),
        re.compile(r"first\b.{0,80}\bthen\b", re.DOTALL),
    )
    _ENUM_RE = re.compile(r"(?:^|\n)\s*(?:\d+[.、)]|[一二三四五六七八九十]+[、.])")
    _QUESTION_RE = re.compile(r"[?？]")
    _COMPLEX_HINTS = (
        "深入研究", "综合调研", "调研报告", "多步骤", "交叉验证",
        "research report", "deep research", "investigate and compare",
    )

    def __init__(self, llm=None, timeout_seconds: float | None = None):
        self.llm = llm
        self.last_error = ""
        if timeout_seconds is None:
            try:
                from agentic_rag.config.settings import get_settings
                timeout_seconds = float(
                    get_settings().agent.planner_timeout_seconds
                )
            except Exception:
                timeout_seconds = 12.0
        self.timeout_seconds = max(1.0, float(timeout_seconds))

    @classmethod
    def should_plan(cls, query: str, has_media: bool = False) -> bool:
        """Return whether a request is complex enough to justify planning.

        This gate is intentionally cheap and deterministic.  Requests that
        the deterministic policy already answers directly (media-only
        description, greeting, simple rewrite, …) are never sent to the
        planner.
        """
        text = (query or "").strip()
        if not text:
            return False
        lowered = text.lower()
        if has_media:
            decision = RequestPolicy.decide(text, has_media=True)
            if decision.strategy == RequestStrategy.DIRECT:
                return False
        if any(hint in lowered for hint in cls._COMPLEX_HINTS):
            return True
        if len(cls._QUESTION_RE.findall(text)) >= 2:
            return True
        if cls._ENUM_RE.search(text):
            return True
        return any(pattern.search(lowered) for pattern in cls._COMPLEX_PATTERNS)

    async def plan(
        self,
        query: str,
        *,
        has_media: bool = False,
        tool_names: Iterable[str] = (),
    ) -> RoutePlan:
        """Return a routing plan, falling back to deterministic routing."""
        deterministic = RequestPolicy.decide(query, has_media=has_media)
        if not self.should_plan(query, has_media=has_media):
            return RoutePlan(deterministic)

        fallback = RoutePlan(
            deterministic, complexity="complex", source="rules",
        )
        try:
            from agentic_rag.config.settings import get_settings
            if not get_settings().agent.planner_enabled:
                return RoutePlan(
                    deterministic, complexity="complex",
                    source="rules-planner-disabled",
                )
        except Exception:
            pass

        agenerate = getattr(self.llm, "agenerate", None)
        if not callable(agenerate):
            return fallback

        names = [str(name) for name in tool_names if name][:30]
        messages = [
            Message.system(Prompts.QUERY_PLANNER_SYSTEM),
            Message.user(Prompts.query_planner_user(
                query, has_media=has_media, tool_names=names,
            )),
        ]
        try:
            response = await asyncio.wait_for(
                agenerate(messages), timeout=self.timeout_seconds,
            )
        except Exception as exc:  # timeout, provider error, bad signature
            self.last_error = str(exc)
            return fallback

        plan = self._parse_plan(
            getattr(response, "content", "") or "",
            deterministic,
            has_media=has_media,
        )
        if plan is None:
            self.last_error = "invalid planner output"
            return fallback
        return plan

    @classmethod
    def _parse_plan(
        cls,
        text: str,
        deterministic: RequestDecision,
        *,
        has_media: bool,
    ) -> RoutePlan | None:
        data = extract_json_object(text)
        if not isinstance(data, dict):
            return None

        strategy_raw = str(data.get("strategy") or "").strip().lower()
        if strategy_raw not in {"direct", "tool_use"}:
            return None
        strategy = RequestStrategy(strategy_raw)

        preferred = str(data.get("preferred_capability") or "").strip()
        if preferred not in {"", "fresh_information"}:
            preferred = ""
        # The deterministic policy encodes the evidence requirement.  The
        # planner may add retrieval (upgrade DIRECT → TOOL_USE), but it must
        # never downgrade a request that deterministically requires evidence
        # into a model-memory answer.
        forced_tool_use = (
            deterministic.strategy == RequestStrategy.TOOL_USE
            and strategy == RequestStrategy.DIRECT
        )
        if deterministic.strategy == RequestStrategy.TOOL_USE:
            strategy = RequestStrategy.TOOL_USE
        if deterministic.preferred_capability:
            preferred = deterministic.preferred_capability

        sub_queries = cls._clean_sub_queries(data.get("sub_queries"))
        if strategy == RequestStrategy.DIRECT:
            sub_queries = ()
            complexity = "simple"
        else:
            raw_complexity = str(data.get("complexity") or "").strip().lower()
            if raw_complexity not in {"simple", "complex"}:
                raw_complexity = "complex"
            complexity = (
                "complex"
                if (
                    forced_tool_use
                    or raw_complexity == "complex"
                    or len(sub_queries) > 1
                )
                else "simple"
            )

        reason = str(data.get("reason") or "").strip()[:200] or "llm planning"
        decision = RequestDecision(
            strategy=strategy,
            reason=f"llm planner: {reason}",
            preferred_capability=preferred,
        )
        return RoutePlan(
            decision=decision,
            complexity=complexity,
            sub_queries=sub_queries,
            source="llm",
        )

    @staticmethod
    def _clean_sub_queries(value: Any) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            query = " ".join(item.split()).strip()
            if not query or len(query) > 300 or query in cleaned:
                continue
            cleaned.append(query)
            if len(cleaned) >= 3:
                break
        return tuple(cleaned)
