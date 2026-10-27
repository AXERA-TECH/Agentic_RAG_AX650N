"""Select a small, query-relevant subset of MCP tools.

Every MCP server can contribute many tools, and dumping all of them into one
prompt both crowds the context and gives small models too many choices.  The
selector keeps tool discovery per-request: it ranks MCP tools against the
current turn (including planner sub-queries) and only applies a cap when the
server actually exposes more tools than the configured limit.

Selection is lexical and capability-based — no extra model call — so it is
cheap enough to run on every request.
"""

from __future__ import annotations

import json
import re

_ASCII_TERM_RE = re.compile(r"[a-z0-9_]{2,}")
_CJK_RUN_RE = re.compile(r"[一-鿿]+")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "what", "how",
    "query", "search", "tool", "mcp", "server", "result", "returns",
    "什么", "怎么", "怎样", "如何", "可以", "这个", "那个", "我们", "你们",
    "他们", "一个", "一下", "现在", "今天", "以及", "还有", "进行", "使用",
}
_SEARCH_HINTS = (
    "search", "web", "internet", "browser", "browse", "news",
    "搜索", "网页", "网络", "新闻", "检索",
)


class ToolSelector:
    """Rank and cap MCP tools without an additional LLM round-trip."""

    @classmethod
    def select(
        cls,
        tools: list,
        query: str,
        *,
        preferred_capability: str = "",
        max_tools: int = 5,
    ) -> list:
        if not tools:
            return []
        limit = max(1, int(max_tools))
        if len(tools) <= limit:
            return list(tools)

        terms = cls._terms(query)
        scored = [
            (cls._score(tool, terms, preferred_capability), index, tool)
            for index, tool in enumerate(tools)
        ]
        # Highest score first; stable original order for ties.
        scored.sort(key=lambda item: (-item[0], item[1]))
        ranked = [tool for _, _, tool in scored]

        # Search/fresh-information tools are the deterministic fallback for
        # live queries and must survive the cap.
        anchors = [tool for tool in tools if cls._is_search_tool(tool)]
        required: list = []
        if preferred_capability:
            required.extend(
                tool for tool in tools
                if cls._has_capability(tool, preferred_capability)
            )
        if not any(
            cls._is_search_tool(tool)
            for tool in (*required, *ranked[:limit])
        ):
            required.extend(anchors[:1])

        selected: list = []
        seen: set[int] = set()
        for tool in (*required, *ranked):
            if id(tool) in seen:
                continue
            selected.append(tool)
            seen.add(id(tool))
            if len(selected) >= limit:
                break

        # A long capability list can crowd out the search anchor; keep one
        # anchor when the cap would otherwise hide every search tool.
        if not any(cls._is_search_tool(tool) for tool in selected) and anchors:
            selected[-1] = anchors[0]
        return selected

    # ── Internals ─────────────────────────────────────────

    @classmethod
    def _score(cls, tool, terms: set[str], preferred_capability: str) -> int:
        name = str(getattr(tool, "name", "")).replace("__", " ").lower()
        description = str(getattr(tool, "description", "") or "").lower()
        schema = ""
        try:
            schema = json.dumps(
                getattr(tool, "parameters_schema", {}) or {},
                ensure_ascii=False,
            ).lower()
        except (TypeError, ValueError):
            pass
        metadata = f"{name} {description} {schema}"
        score = 0
        for term in terms:
            if term in name:
                score += 4
            elif term in metadata:
                score += 1
        if preferred_capability and cls._has_capability(
            tool, preferred_capability,
        ):
            score += 20
        if cls._is_search_tool(tool):
            score += 2
        return score

    @staticmethod
    def _has_capability(tool, capability: str) -> bool:
        capability_check = getattr(tool, "has_capability", None)
        if callable(capability_check):
            try:
                return bool(capability_check(capability))
            except Exception:
                return False
        return capability in (getattr(tool, "capabilities", ()) or ())

    @staticmethod
    def _is_search_tool(tool) -> bool:
        metadata = (
            f"{getattr(tool, 'name', '')} "
            f"{getattr(tool, 'description', '') or ''}"
        ).lower()
        return any(hint in metadata for hint in _SEARCH_HINTS)

    @classmethod
    def _terms(cls, query: str) -> set[str]:
        text = (query or "").lower()
        terms = {
            term for term in _ASCII_TERM_RE.findall(text)
            if term not in _STOPWORDS
        }
        for run in _CJK_RUN_RE.findall(text):
            if len(run) <= 2:
                candidates = [run]
            else:
                candidates = [
                    run[index:index + 2]
                    for index in range(len(run) - 1)
                ]
            terms.update(
                term for term in candidates
                if term not in _STOPWORDS
            )
        return terms
