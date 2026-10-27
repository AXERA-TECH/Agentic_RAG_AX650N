"""Agent Router — selects the smallest execution path for one chat endpoint.

Routing is deliberately split from execution:

* :class:`~agentic_rag.agent.routing.RuleRouter` decides DIRECT / SEARCH /
  ACTION and, for SEARCH, which evidence sources are required.
* :class:`_ToolCatalog` resolves those sources against the tools that are
  actually registered and picks the first tool to execute.
* The engine factory maps the decision to SinglePass (simple requests) or
  ReAct (complex, multi-tool, media, or action requests).

The old behaviour of always seeding ``rag_search`` is gone.  A live query with
no KB signal starts with the fresh-information MCP tool; a KB query starts
with RAG; a request requiring both starts with RAG and may fall through to the
web tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_rag.agent.react_engine import ReActEngine
from agentic_rag.agent.react_prompt import SYSTEM_PROMPT
from agentic_rag.agent.request_policy import RequestStrategy
from agentic_rag.agent.routing import (
    Route,
    RouteDecision,
    RoutingPlan,
    RoutingPlanner,
    RuleRouter,
)
from agentic_rag.agent.single_pass_engine import SinglePassEngine
from agentic_rag.agent.tool_selector import ToolSelector
from agentic_rag.data.models import AttachmentRef
from agentic_rag.orchestration.l1_tools.media_tools import build_media_tools
from agentic_rag.services.llm.base import BaseLLMProvider
from agentic_rag.services.media.models import StoredAttachment
from agentic_rag.services.media.store import AttachmentStore


_KB_FAILURE = (
    "当前问题需要知识库检索，但知识库检索工具不可用，因此无法可靠回答。"
)
_WEB_FAILURE = (
    "当前问题需要实时信息，但未配置可用的联网检索工具，因此无法可靠回答。"
)
_BOTH_FAILURE = (
    "当前问题需要知识库和联网检索，但两类工具均不可用，因此无法可靠回答。"
)
_SEARCH_FAILURE = "当前问题需要检索，但没有可用的检索工具，因此无法可靠回答。"
_ACTION_FAILURE = "未能找到可执行该操作的工具，因此本次无法完成请求。"


def _has_capability(tool, capability: str) -> bool:
    checker = getattr(tool, "has_capability", None)
    if callable(checker):
        try:
            return bool(checker(capability))
        except Exception:
            return False
    return capability in (getattr(tool, "capabilities", ()) or ())


def _failure_message(sources) -> str:
    requested = set(sources)
    if requested == {"kb"}:
        return _KB_FAILURE
    if requested == {"web"}:
        return _WEB_FAILURE
    if requested == {"kb", "web"}:
        return _BOTH_FAILURE
    return _SEARCH_FAILURE


def _dedupe(tools) -> list:
    seen = set()
    result = []
    for tool in tools:
        if tool is None:
            continue
        key = getattr(tool, "name", id(tool))
        if key in seen:
            continue
        seen.add(key)
        result.append(tool)
    return result


@dataclass
class _ToolCatalog:
    """The tools available for this request, indexed by evidence role."""

    rag: Any | None = None
    mcp_tools: list | None = None
    fresh_tools: list | None = None
    media_tools: list | None = None

    def __post_init__(self):
        self.mcp_tools = list(self.mcp_tools or [])
        self.fresh_tools = list(self.fresh_tools or [])
        self.media_tools = list(self.media_tools or [])

    @classmethod
    def build(cls, registry, attachments, store, llm) -> "_ToolCatalog":
        rag = registry.get("rag_search")
        mcp_tools = list(registry.get_mcp_tools() or [])
        return cls(
            rag=rag,
            mcp_tools=mcp_tools,
            fresh_tools=[
                tool for tool in mcp_tools
                if _has_capability(tool, "fresh_information")
            ],
            media_tools=build_media_tools(
                attachments, store=store, llm=llm,
            ),
        )

    # ── Source availability ─────────────────────────────────────────────

    @property
    def available_sources(self) -> set[str]:
        sources: set[str] = set()
        if self.rag is not None:
            sources.add("kb")
        if self.fresh_tools:
            sources.add("web")
        return sources

    def normalize_sources(self, sources) -> frozenset[str]:
        available = self.available_sources
        return frozenset(
            source for source in sources or () if source in available
        )

    def default_sources(self, profile: str) -> frozenset[str]:
        if profile == "general":
            return frozenset()
        if self.rag is not None:
            return frozenset({"kb"})
        if self.fresh_tools:
            return frozenset({"web"})
        return frozenset()

    # ── Decision resolution ─────────────────────────────────────────────

    def resolve(
        self, decision: RouteDecision, query: str, profile: str,
    ) -> RouteDecision:
        if decision.route == Route.DIRECT:
            return decision.evolve(
                evidence_sources=frozenset(),
                first_tool="",
                action_tool="",
                failure_answer="",
            )

        if decision.route == Route.ACTION:
            tool = self.tool_by_name(decision.action_tool)
            if tool is not None and tool in self.mcp_tools:
                return decision.evolve(
                    evidence_sources=frozenset(),
                    first_tool="",
                    action_tool=tool.name,
                    failure_answer=decision.failure_answer or _ACTION_FAILURE,
                )
            return decision.evolve(
                route=Route.DIRECT,
                evidence_sources=frozenset(),
                first_tool="",
                action_tool="",
                complexity="simple",
                direct_answer=_ACTION_FAILURE,
                failure_answer=_ACTION_FAILURE,
            )

        sources = self.normalize_sources(decision.evidence_sources)
        if not sources and not decision.locked:
            if profile == "general" and decision.decided_by == "profile":
                return decision.evolve(
                    route=Route.DIRECT,
                    evidence_sources=frozenset(),
                    first_tool="",
                    action_tool="",
                    complexity="simple",
                )
            sources = self.default_sources(profile)

        if not sources:
            return decision.evolve(
                evidence_sources=frozenset(decision.evidence_sources),
                first_tool="",
                action_tool="",
                failure_answer=(
                    decision.failure_answer
                    or _failure_message(decision.evidence_sources)
                ),
                allow_model_fallback=False,
            )

        first = self.first_tool(sources, query)
        failure = decision.failure_answer
        if first is None:
            failure = failure or _failure_message(sources)
        return decision.evolve(
            evidence_sources=sources,
            first_tool=getattr(first, "name", "") if first else "",
            action_tool="",
            failure_answer=failure,
        )

    def first_tool(self, sources, query: str):
        source_set = set(sources)
        if "kb" in source_set and self.rag is not None:
            return self.rag
        if "web" in source_set and self.fresh_tools:
            return self._pick_fresh(query)
        return None

    def fallback_tool_for(self, decision: RouteDecision, query: str):
        """External search only for ambiguous, profile-default KB requests."""
        if decision.route != Route.SEARCH:
            return None
        if set(decision.evidence_sources) != {"kb"}:
            return None
        if not decision.allow_model_fallback or not self.fresh_tools:
            return None
        return self._pick_fresh(query)

    def _pick_fresh(self, query: str):
        if not self.fresh_tools:
            return None
        if len(self.fresh_tools) == 1:
            return self.fresh_tools[0]
        selected = ToolSelector.select(
            self.fresh_tools,
            query,
            preferred_capability="fresh_information",
            max_tools=1,
        )
        return selected[0] if selected else self.fresh_tools[0]

    # ── Engine-facing helpers ───────────────────────────────────────────

    def kind(self, tool) -> str:
        if tool is None:
            return ""
        if tool is self.rag or getattr(tool, "name", "") == "rag_search":
            return "kb"
        if tool in self.fresh_tools:
            return "web"
        return "web" if _has_capability(tool, "fresh_information") else ""

    def tool_by_name(self, name: str):
        if not name:
            return None
        for tool in (self.rag, *self.mcp_tools, *self.media_tools):
            if tool is not None and getattr(tool, "name", "") == name:
                return tool
        return None

    def evidence_tool_names(self, decision: RouteDecision) -> set[str]:
        names: set[str] = set()
        if "kb" in decision.evidence_sources and self.rag is not None:
            names.add(self.rag.name)
        if "web" in decision.evidence_sources:
            names.update(tool.name for tool in self.fresh_tools)
        # Media tools are request-scoped evidence for a turn with attachments;
        # a usable image/audio/video observation can satisfy the evidence gate
        # even when the text requested colors it as a KB lookup.
        names.update(tool.name for tool in self.media_tools)
        if (
            not names
            and decision.route == Route.ACTION
            and decision.action_tool
        ):
            names.add(decision.action_tool)
        return names

    def tools_for(
        self,
        decision: RouteDecision,
        query: str,
        plan: RoutingPlan,
        *,
        max_mcp_tools: int,
        selection_enabled: bool,
    ) -> list:
        if decision.route == Route.ACTION:
            selected = self._select_mcp(
                self.mcp_tools, query, plan,
                preferred="",
                max_tools=max_mcp_tools,
                enabled=selection_enabled,
            )
            action = self.tool_by_name(decision.action_tool)
            if action is not None and action not in selected:
                selected = [action, *selected][:max_mcp_tools]
            return _dedupe([*selected, *self.media_tools])

        tools = []
        first = self.tool_by_name(decision.first_tool)
        if first is not None:
            tools.append(first)
        if "kb" in decision.evidence_sources and self.rag is not None:
            tools.append(self.rag)
        if "web" in decision.evidence_sources and self.fresh_tools:
            tools.extend(self._select_mcp(
                self.fresh_tools, query, plan,
                preferred="fresh_information",
                max_tools=max_mcp_tools,
                enabled=selection_enabled,
            ))
        tools.extend(self.media_tools)
        return _dedupe(tools)

    def _select_mcp(
        self, tools, query: str, plan: RoutingPlan,
        *, preferred: str, max_tools: int, enabled: bool,
    ) -> list:
        ordered = self._order_mcp_tools(tools, preferred)
        if not enabled or len(ordered) <= max_tools:
            return ordered
        selection_query = query
        if plan.sub_queries:
            selection_query = f"{query} " + " ".join(plan.sub_queries)
        return ToolSelector.select(
            ordered,
            selection_query,
            preferred_capability=preferred,
            max_tools=max_tools,
        )

    @staticmethod
    def _order_mcp_tools(tools, preferred_capability: str):
        """Put capability-matched MCP tools first without dropping the rest."""
        if not preferred_capability:
            return list(tools)
        matched = [
            tool for tool in tools
            if _has_capability(tool, preferred_capability)
        ]
        if not matched:
            return list(tools)
        matched_ids = {id(tool) for tool in matched}
        return matched + [
            tool for tool in tools if id(tool) not in matched_ids
        ]

    @property
    def planner_tool_names(self) -> list[str]:
        names = []
        for tool in (self.rag, *self.mcp_tools, *self.media_tools):
            if tool is not None and tool.name not in names:
                names.append(tool.name)
        return names


class AgentRouter:
    """Create one chat agent without exposing user-selectable agent modes."""

    def __init__(self, llm: BaseLLMProvider, tool_registry):
        self.llm = llm
        self.tool_registry = tool_registry
        # Ask the active provider instance, not settings.default_provider:
        # callers may pass a different provider per request.
        try:
            supports = getattr(llm, "supports_native_tools_on_wire", None)
            self._native_tool_calls = bool(supports()) if callable(supports) else True
        except Exception:
            self._native_tool_calls = True

    async def route(
        self,
        query: str,
        has_media: bool = False,
        attachments: list[StoredAttachment] | None = None,
        attachment_store: AttachmentStore | None = None,
    ):
        """Build the minimal engine required for this query.

        Args:
            query: User's query text.
            has_media: Whether the turn carries attachments.  Kept for
                compatibility; resolved ``attachments`` also imply media.
            attachments: Request-scoped attachment records used to build the
                media tools and decide DIRECT vs SEARCH.
            attachment_store: Store backing those records (tests may use a
                non-default root).

        Returns:
            A direct, one-tool, or ReAct engine.
        """
        attachments = attachments or []
        # ``AgentInput.attachments`` carries serializable refs; media tools
        # need the resolved path + MIME from the store.
        if attachments and isinstance(attachments[0], AttachmentRef):
            store = attachment_store or AttachmentStore()
            attachments = store.resolve([ref.id for ref in attachments])
            attachment_store = store
        has_media = has_media or bool(attachments)

        catalog = _ToolCatalog.build(
            self.tool_registry, attachments, attachment_store, self.llm,
        )
        profile = self._routing_profile()
        decision = RuleRouter.decide(
            query, has_media=has_media, profile=profile,
        )
        decision = catalog.resolve(decision, query, profile)
        plan = RoutingPlan(decision)

        if RoutingPlanner.should_plan(decision):
            planner = RoutingPlanner(llm=self.llm)
            plan = await planner.refine(
                query,
                decision,
                available_sources=catalog.available_sources,
                tool_names=catalog.planner_tool_names,
                has_media=has_media,
            )
            decision = catalog.resolve(plan.decision, query, profile)
            plan = RoutingPlan(
                decision, plan.sub_queries, plan.planner_error,
            )

        return self._build_engine(
            decision, plan, catalog, query, has_media,
        )

    # ── Settings ────────────────────────────────────────────────────────

    @staticmethod
    def _agent_settings():
        try:
            from agentic_rag.config.settings import get_settings
            return get_settings().agent
        except Exception:
            return None

    def _routing_profile(self) -> str:
        settings = self._agent_settings()
        return str(
            getattr(settings, "routing_profile", "balanced") or "balanced"
        )

    # ── Engine factory ──────────────────────────────────────────────────

    def _build_engine(
        self,
        decision: RouteDecision,
        plan: RoutingPlan,
        catalog: _ToolCatalog,
        query: str,
        has_media: bool,
    ):
        if decision.route == Route.DIRECT:
            return SinglePassEngine(
                llm=self.llm,
                strategy=RequestStrategy.DIRECT,
                direct_answer=decision.direct_answer,
                route_reason=decision.reason,
                failure_answer=decision.failure_answer,
                query_plan=plan,
                enable_native_tool_calls=self._native_tool_calls,
            )

        settings = self._agent_settings()
        max_retrieval_calls = int(
            getattr(settings, "max_retrieval_rounds", 2) or 2
        )
        max_mcp_tools = max(
            1, int(getattr(settings, "max_mcp_tools", 5) or 5),
        )
        tool_selection_enabled = bool(
            getattr(settings, "tool_selection_enabled", True)
        )
        sufficiency_enabled = bool(
            getattr(settings, "sufficiency_check_enabled", True)
        )
        single_pass_enabled = bool(
            getattr(settings, "simple_kb_single_pass", True)
        )

        if decision.route == Route.ACTION:
            return self._build_react(
                decision, plan, catalog, query, has_media,
                max_retrieval_calls=max_retrieval_calls,
                max_mcp_tools=max_mcp_tools,
                selection_enabled=tool_selection_enabled,
                sufficiency_enabled=False,
                action=True,
            )

        first = catalog.tool_by_name(decision.first_tool)
        use_single_pass = (
            single_pass_enabled
            and decision.complexity != "complex"
            and not has_media
        )
        if use_single_pass:
            fallback = catalog.fallback_tool_for(decision, query)
            return SinglePassEngine(
                llm=self.llm,
                strategy=RequestStrategy.TOOL_USE,
                tool=first,
                fallback_tool=fallback,
                route_reason=decision.reason,
                evidence_kind=catalog.kind(first),
                allow_model_fallback=decision.allow_model_fallback,
                failure_answer=decision.failure_answer,
                query_plan=plan,
                enable_native_tool_calls=self._native_tool_calls,
            )

        return self._build_react(
            decision, plan, catalog, query, has_media,
            max_retrieval_calls=max_retrieval_calls,
            max_mcp_tools=max_mcp_tools,
            selection_enabled=tool_selection_enabled,
            sufficiency_enabled=sufficiency_enabled,
            action=False,
        )

    def _build_react(
        self,
        decision: RouteDecision,
        plan: RoutingPlan,
        catalog: _ToolCatalog,
        query: str,
        has_media: bool,
        *,
        max_retrieval_calls: int,
        max_mcp_tools: int,
        selection_enabled: bool,
        sufficiency_enabled: bool,
        action: bool,
    ) -> ReActEngine:
        tools = catalog.tools_for(
            decision,
            query,
            plan,
            max_mcp_tools=max_mcp_tools,
            selection_enabled=selection_enabled,
        )
        if action:
            required_names = {
                tool.name for tool in tools
                if tool in catalog.mcp_tools
            }
        else:
            required_names = catalog.evidence_tool_names(decision)

        return ReActEngine(
            llm=self.llm,
            tools=tools,
            system_prompt_template=SYSTEM_PROMPT,
            max_iterations=6,
            enable_native_tool_calls=self._native_tool_calls,
            # Only the route's first evidence tool is seeded.  For an ACTION
            # request the model chooses the tool itself.
            initial_tool_name="" if action else decision.first_tool,
            # Never answer an evidence/action request from model memory alone.
            require_tool_call=True,
            required_tool_names=required_names,
            route_reason=decision.reason,
            max_retrieval_calls=max_retrieval_calls,
            query_plan=plan,
            enable_sufficiency_check=(
                sufficiency_enabled and plan.is_complex and not action
            ),
            failure_answer=decision.failure_answer,
        )
