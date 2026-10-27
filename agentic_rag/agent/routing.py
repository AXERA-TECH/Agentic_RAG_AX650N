"""Simplified three-route routing for the single public chat endpoint.

The router answers three questions, in order:

1. Does answering require external evidence?  (``DIRECT`` vs ``SEARCH``)
2. If so, which evidence sources are needed?  (``kb`` / ``web``)
3. Does the request need a multi-step tool loop?  (``simple`` / ``complex``)

Rules are intentionally conservative: only strong, unambiguous signals lock a
route.  Everything else falls back to the configured routing profile, and only
complex requests may consult the LLM planner.  The planner can refine a route
but it cannot downgrade a locked evidence requirement.

``RequestPolicy`` / ``QueryPlanner`` remain in place as the legacy API used by
the media-context layer and older tests; this module is the routing source for
``AgentRouter``.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from agentic_rag.agent.query_planner import QueryPlanner, extract_json_object
from agentic_rag.config.prompts import Prompts
from agentic_rag.data.models import Message


class Route(str, Enum):
    """Execution shape, not an intent taxonomy."""

    DIRECT = "direct"
    SEARCH = "search"
    ACTION = "action"


@dataclass(frozen=True)
class RouteDecision:
    """One route decision with enough data for the engine factory.

    ``first_tool`` is a tool *name* so this object stays serializable and
    free of runtime dependencies.  ``ToolCatalog`` resolves it to an actual
    tool after checking availability.
    """

    route: Route
    evidence_sources: frozenset[str] = field(default_factory=frozenset)
    complexity: str = "simple"
    direct_answer: str = ""
    reason: str = ""
    decided_by: str = "rules"
    locked: bool = False
    allow_model_fallback: bool = False
    first_tool: str = ""
    action_tool: str = ""
    failure_answer: str = ""

    @property
    def evidence_required(self) -> bool:
        return bool(self.evidence_sources)

    def evolve(self, **changes: Any) -> "RouteDecision":
        return replace(self, **changes)


@dataclass(frozen=True)
class RoutingPlan:
    """A route decision plus optional multi-hop guidance.

    The property names mirror the legacy ``RoutePlan`` so engines can accept
    either object without a runtime dependency on this module.
    """

    decision: RouteDecision
    sub_queries: tuple[str, ...] = ()
    planner_error: str = ""

    @property
    def source(self) -> str:
        return self.decision.decided_by

    @property
    def is_complex(self) -> bool:
        return self.decision.complexity == "complex"

    def diagnostics(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "route": self.decision.route.value,
            "evidence_sources": sorted(self.decision.evidence_sources),
            "first_tool": self.decision.first_tool,
            "action_tool": self.decision.action_tool,
            "complexity": self.decision.complexity,
            "sub_queries": list(self.sub_queries),
            "locked": self.decision.locked,
            "allow_model_fallback": self.decision.allow_model_fallback,
            "reason": self.decision.reason,
            "planner_error": self.planner_error,
        }


class RuleRouter:
    """Strong deterministic rules first; ambiguity is resolved by profile."""

    timezone = ZoneInfo("Asia/Shanghai")

    # Explicit knowledge-base intent, or domain words that almost always
    # require a document lookup.  General chat words ("如何", "为什么") are
    # deliberately not included: they are handled by the profile default or
    # by the planner.
    _KB_RE = re.compile(
        r"知识库|"
        r"文档(?:里|中|内|内容|的)|"
        r"资料(?:里|中|内)|"
        r"手册|说明书|制度|条款|"
        r"技术规范|标准规范|规范(?:要求|中|里)|"
        r"维护|检修|保养|故障|报错|安装|配置|参数|型号|"
        r"步骤|流程|操作(?:说明|步骤|规程)|剂量|"
        r"价格|报价|费用|"
        r"\b(?:documentation|manual|specification|troubleshoot|"
        r"knowledge base|procedure|policy)\b",
        re.IGNORECASE,
    )

    # Freshness is only strong when a time word is paired with a volatile
    # topic.  A bare "今天" is not enough to force a web search.
    _WEB_PATTERNS = (
        re.compile(
            r"(?:最新|实时|当前|现在|目前|今天|今日|本周|本月|今年)"
            r".{0,16}(?:新闻|天气|气温|比分|赛况|比赛|结果|排名|冠军|"
            r"决赛|半决赛|价格|报价|汇率|股价|政策|版本|发布|选举|"
            r"总统|现任|航班|状态)"
        ),
        re.compile(
            r"(?:19|20)\d{2}.{0,24}(?:世界杯|奥运|赛事|冠军|决赛|大选|"
            r"发布|财报|演唱会)"
        ),
        re.compile(
            r"\b(?:latest|current|today|now|this week|this month)\b"
            r".{0,40}\b(?:news|weather|score|ranking|price|release|"
            r"version|election)\b",
            re.IGNORECASE,
        ),
    )

    _DIRECT_EXACT = {"你好", "您好", "hello", "hi", "谢谢", "再见", "thanks", "bye"}
    _DIRECT_HINTS = (
        "你是谁", "你是什么", "你能做什么", "自我介绍",
        "写一首", "写一段", "写一个", "写个", "生成一个", "生成一段",
        "翻译", "润色", "改写", "续写", "总结以下", "计算",
    )
    _MEDIA_HINTS = (
        "描述", "识别", "看图", "图中", "图上", "图里", "图片里", "图片中",
        "图片上", "这张图", "这幅图", "这张照片", "照片", "截图", "画面",
        "这是什么", "写了什么", "说了什么", "转写", "听写", "转录",
        "音频里", "音频中", "视频里", "视频中", "这段音频", "这段视频",
        "提取文字", "翻译", "分析这张", "分析一下", "看看这", "看一下",
        "describe", "transcribe", "what is in this image", "read the text",
    )

    @classmethod
    def decide(
        cls,
        query: str,
        *,
        has_media: bool = False,
        profile: str = "balanced",
    ) -> RouteDecision:
        text = (query or "").strip()
        lowered = text.lower()
        profile = cls._normalize_profile(profile)

        local_answer = cls._local_date_answer(text)
        if local_answer:
            return RouteDecision(
                route=Route.DIRECT,
                direct_answer=local_answer,
                reason="deterministic local date calculation",
                decided_by="rules",
                locked=True,
            )

        kb = bool(cls._KB_RE.search(text))
        web = any(pattern.search(lowered) for pattern in cls._WEB_PATTERNS)
        complex_ = cls._is_complex(text) or (kb and web)

        if kb and web:
            return RouteDecision(
                route=Route.SEARCH,
                evidence_sources=frozenset({"kb", "web"}),
                complexity="complex",
                reason="request needs both internal documents and fresh web data",
                decided_by="rules",
                locked=True,
            )
        if web:
            return RouteDecision(
                route=Route.SEARCH,
                evidence_sources=frozenset({"web"}),
                complexity="complex" if complex_ else "simple",
                reason="fresh information signal",
                decided_by="rules",
                locked=True,
            )
        if kb:
            return RouteDecision(
                route=Route.SEARCH,
                evidence_sources=frozenset({"kb"}),
                complexity="complex" if complex_ else "simple",
                reason="knowledge-base signal",
                decided_by="rules",
                locked=True,
            )

        if has_media and not lowered:
            return RouteDecision(
                route=Route.DIRECT,
                reason="media without a text request",
                decided_by="rules",
                locked=True,
            )
        if has_media and cls._is_media_only(lowered):
            return RouteDecision(
                route=Route.DIRECT,
                reason="media-only request can be handled by the current model",
                decided_by="rules",
                locked=True,
            )
        if cls._is_direct(lowered):
            return RouteDecision(
                route=Route.DIRECT,
                reason="no external evidence required",
                decided_by="rules",
                locked=True,
            )
        if has_media:
            # The attachment is the evidence; no retrieval signal was found.
            return RouteDecision(
                route=Route.DIRECT,
                reason="media turn without a retrieval signal",
                decided_by="rules",
                locked=False,
            )

        return cls._profile_default(
            profile,
            complexity="complex" if complex_ else "simple",
        )

    @classmethod
    def _profile_default(cls, profile: str, *, complexity: str) -> RouteDecision:
        if profile == "general":
            return RouteDecision(
                route=Route.DIRECT,
                reason="routing profile allows a model-memory answer",
                decided_by="profile",
                locked=False,
            )
        return RouteDecision(
            route=Route.SEARCH,
            evidence_sources=frozenset({"kb"}),
            complexity="complex" if complexity == "complex" else "simple",
            reason=(
                "strict routing profile requires the knowledge base"
                if profile == "strict_kb"
                else "routing profile defaults to the knowledge base"
            ),
            decided_by="profile",
            # Strict mode has no profile-based escape hatch: the planner may
            # add sub-queries, but it cannot drop the KB requirement.
            locked=(profile == "strict_kb"),
            allow_model_fallback=(profile == "balanced"),
        )

    @staticmethod
    def _normalize_profile(profile: str) -> str:
        value = (profile or "").strip().lower()
        return value if value in {"strict_kb", "balanced", "general"} else "balanced"

    @classmethod
    def _is_complex(cls, query: str) -> bool:
        if not query:
            return False
        return QueryPlanner.should_plan(query, has_media=False)

    @classmethod
    def _is_direct(cls, lowered: str) -> bool:
        if not lowered:
            return True
        if lowered in cls._DIRECT_EXACT:
            return True
        if re.match(r"^(你好|您好|hello\b|hi\b)", lowered):
            return True
        if any(hint in lowered for hint in cls._DIRECT_HINTS):
            return True
        return re.fullmatch(r"[\d\s+\-*/().=]+", lowered) is not None

    @classmethod
    def _is_media_only(cls, lowered: str) -> bool:
        if not any(hint in lowered for hint in cls._MEDIA_HINTS):
            return False
        return not (
            cls._KB_RE.search(lowered)
            or any(p.search(lowered) for p in cls._WEB_PATTERNS)
            or cls._is_complex(lowered)
        )

    @classmethod
    def _local_date_answer(cls, query: str) -> str:
        if not re.search(r"(星期几|周几|几号|日期)", query):
            return ""
        offsets = {"昨天": -1, "今天": 0, "今日": 0, "明天": 1}
        matched = next((word for word in offsets if word in query), None)
        if matched is None and "当前" not in query:
            return ""
        offset = offsets.get(matched, 0)
        target = datetime.now(cls.timezone) + timedelta(days=offset)
        weekdays = (
            "星期一", "星期二", "星期三", "星期四",
            "星期五", "星期六", "星期日",
        )
        return (
            f"{target.year}年{target.month}月{target.day}日，"
            f"{weekdays[target.weekday()]}。"
        )


class RoutingPlanner:
    """LLM refinement for complex requests only.

    The planner receives the rule decision, the available evidence sources and
    the registered tool names.  It may add sub-queries, choose between the
    available sources, or select an action tool.  It cannot remove a locked
    evidence requirement or invent a tool.
    """

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

    @staticmethod
    def should_plan(decision: RouteDecision) -> bool:
        return (
            decision.route == Route.SEARCH
            and decision.complexity == "complex"
        )

    async def refine(
        self,
        query: str,
        decision: RouteDecision,
        *,
        available_sources: Iterable[str] = (),
        tool_names: Iterable[str] = (),
        has_media: bool = False,
    ) -> RoutingPlan:
        names = [str(name) for name in tool_names if name][:30]
        source_names = sorted(set(available_sources))
        fallback = RoutingPlan(decision)
        try:
            from agentic_rag.config.settings import get_settings
            if not getattr(get_settings().agent, "planner_enabled", True):
                return RoutingPlan(decision, planner_error="planner disabled")
        except Exception:
            pass

        agenerate = getattr(self.llm, "agenerate", None)
        if not callable(agenerate):
            return fallback

        messages = [
            Message.system(Prompts.ROUTE_PLANNER_SYSTEM),
            Message.user(Prompts.route_planner_user(
                query,
                has_media=has_media,
                available_sources=source_names,
                tool_names=names,
            )),
        ]
        try:
            response = await asyncio.wait_for(
                agenerate(messages), timeout=self.timeout_seconds,
            )
        except Exception as exc:
            self.last_error = str(exc)
            return RoutingPlan(decision, planner_error=self.last_error)

        parsed = self._parse(getattr(response, "content", "") or "")
        if parsed is None:
            self.last_error = "invalid planner output"
            return RoutingPlan(decision, planner_error=self.last_error)
        return self._merge(decision, parsed, names)

    # ── Parsing / guardrails ────────────────────────────────────────────

    @classmethod
    def _parse(cls, text: str) -> dict[str, Any] | None:
        data = extract_json_object(text)
        if not isinstance(data, dict):
            return None

        raw_route = str(data.get("route") or "").strip().lower()
        if raw_route not in {route.value for route in Route}:
            # Backward compatibility with the legacy planner schema.  This
            # keeps older prompts/tests and cached model outputs usable.
            strategy = str(data.get("strategy") or "").strip().lower()
            if strategy == "direct":
                raw_route = Route.DIRECT.value
            elif strategy == "tool_use":
                raw_route = Route.SEARCH.value
            else:
                return None

        raw_sources = data.get("evidence_sources", data.get("source", []))
        if isinstance(raw_sources, str):
            raw_sources = [raw_sources]
        sources = {
            str(value).strip().lower()
            for value in (raw_sources or [])
            if str(value).strip().lower() in {"kb", "web"}
        }
        preferred = str(
            data.get("preferred_capability") or ""
        ).strip().lower()
        if not sources and preferred == "fresh_information":
            sources = {"web"}

        raw_complexity = str(
            data.get("complexity") or ""
        ).strip().lower()
        if raw_complexity not in {"simple", "complex"}:
            raw_complexity = ""

        sub_queries = QueryPlanner._clean_sub_queries(
            data.get("sub_queries")
        )
        reason = str(data.get("reason") or "").strip()[:200]
        tool = str(data.get("tool") or data.get("action_tool") or "").strip()

        return {
            "route": Route(raw_route),
            "evidence_sources": frozenset(sources),
            "complexity": raw_complexity,
            "sub_queries": sub_queries,
            "reason": reason,
            "tool": tool,
        }

    @classmethod
    def _merge(
        cls,
        deterministic: RouteDecision,
        parsed: dict[str, Any],
        tool_names: list[str],
    ) -> RoutingPlan:
        route = parsed["route"]
        sources = parsed["evidence_sources"]
        complexity = parsed["complexity"]
        sub_queries = parsed["sub_queries"]
        reason = parsed["reason"]
        action_tool = parsed["tool"]

        if route == Route.ACTION:
            if action_tool not in tool_names:
                return RoutingPlan(
                    deterministic,
                    planner_error=f"unknown action tool: {action_tool or '(empty)'}",
                )
            sources = frozenset()
            complexity = "complex"
            sub_queries = ()
        elif route == Route.DIRECT:
            sources = frozenset()
            complexity = "simple"
            sub_queries = ()
        else:
            if not sources:
                sources = deterministic.evidence_sources
            if not sources:
                sources = frozenset({"kb"})
            if len(sub_queries) > 1 and not complexity:
                complexity = "complex"

        if not complexity:
            complexity = deterministic.complexity

        # A locked deterministic decision encodes an evidence requirement.
        # The planner may add sub-queries or choose an action, but it must not
        # turn a required retrieval into a model-memory answer.
        if deterministic.locked:
            route = deterministic.route
            sources = deterministic.evidence_sources
            action_tool = deterministic.action_tool
            complexity = deterministic.complexity
            if route == Route.DIRECT:
                sub_queries = ()

        decision = deterministic.evolve(
            route=route,
            evidence_sources=sources,
            complexity=complexity,
            reason=f"llm planner: {reason}" if reason else "llm planner",
            decided_by="llm",
            first_tool="",
            action_tool=action_tool if route == Route.ACTION else "",
            direct_answer="",
        )
        return RoutingPlan(
            decision=decision,
            sub_queries=sub_queries if route == Route.SEARCH else (),
        )
