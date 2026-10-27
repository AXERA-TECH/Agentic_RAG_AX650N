"""Explicit retrieval-sufficiency check before an answer is accepted.

Answer validation answers "is the answer grounded in the evidence?".  It does
not answer "did we retrieve enough evidence for the whole question?".  This
module adds that second, complementary gate.

The check is only enabled for complex plans (see ``QueryPlanner``), so simple
deterministic requests pay no extra model call.  Invalid or failed checks fail
open (the existing answer validator remains the safety net) but the engine
records the failure in diagnostics.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from agentic_rag.agent.query_planner import extract_json_object
from agentic_rag.config.prompts import Prompts
from agentic_rag.data.models import Message


@dataclass(frozen=True)
class SufficiencyResult:
    sufficient: bool
    reason: str = ""
    missing: str = ""
    rewritten_query: str = ""
    checked: bool = True
    usage: dict[str, int] = field(default_factory=dict)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "checked": self.checked,
            "reason": self.reason,
            "missing": self.missing,
            "rewritten_query": self.rewritten_query,
        }


class SufficiencyChecker:
    """Ask the model whether existing Observations cover the request."""

    MAX_EVIDENCE_CHARS = 6000

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

    async def check(
        self,
        query: str,
        evidence: str,
        *,
        sub_queries: tuple[str, ...] = (),
        rounds: int = 0,
        max_rounds: int = 2,
    ) -> SufficiencyResult:
        evidence = (evidence or "").strip()
        if not evidence:
            return SufficiencyResult(
                False,
                reason="no usable evidence",
                missing="没有任何可用的检索结果",
                checked=True,
            )

        agenerate = getattr(self.llm, "agenerate", None)
        if not callable(agenerate):
            return SufficiencyResult(
                True, reason="checker llm unavailable", checked=False,
            )

        prompt = Prompts.sufficiency_check(
            query,
            evidence[: self.MAX_EVIDENCE_CHARS],
            sub_queries=sub_queries,
            rounds=rounds,
            max_rounds=max_rounds,
        )
        messages = [
            Message.system(Prompts.SUFFICIENCY_SYSTEM),
            Message.user(prompt),
        ]
        try:
            response = await asyncio.wait_for(
                agenerate(messages), timeout=self.timeout_seconds,
            )
        except Exception as exc:
            self.last_error = str(exc)
            return SufficiencyResult(
                True, reason=f"sufficiency check failed: {exc}", checked=False,
            )

        data = extract_json_object(getattr(response, "content", "") or "")
        if not isinstance(data, dict):
            self.last_error = "invalid sufficiency output"
            return SufficiencyResult(
                True, reason="invalid sufficiency output", checked=False,
            )

        sufficient = data.get("sufficient")
        if isinstance(sufficient, str):
            sufficient = sufficient.strip().lower() in {"true", "yes", "1"}
        if not isinstance(sufficient, bool):
            self.last_error = "missing sufficient field"
            return SufficiencyResult(
                True, reason="invalid sufficiency field", checked=False,
            )

        missing = str(data.get("missing") or "").strip()[:300]
        rewritten = str(data.get("rewritten_query") or "").strip()[:300]
        usage = getattr(response, "usage", None)
        return SufficiencyResult(
            sufficient=sufficient,
            reason=str(data.get("reason") or "").strip()[:200],
            missing="" if sufficient else missing,
            rewritten_query="" if sufficient else rewritten,
            checked=True,
            usage=dict(usage) if isinstance(usage, dict) else {},
        )
