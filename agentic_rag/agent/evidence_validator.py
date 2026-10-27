"""Shared final-answer validation for evidence-grounded engines.

Both the production ReAct path and ``SinglePassEngine`` used to carry their
own citation rules.  The strict checks (citations exist in the evidence,
source names come from the tool output, numbers/dates are grounded) live here
so the two engines cannot drift apart.

The validator is deliberately deterministic and dependency-free: it never
calls an LLM, so it can run on the final answer inside the agent loop without
adding latency or another failure mode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

REF_RE = re.compile(r"\[R(\d+)\]", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s\])}>，。\"']+")
_REACT_TRACE_RE = re.compile(
    r"(?:^|\n)\s*(Thought|Action|Action Input)\s*[:：]",
    re.IGNORECASE,
)
_SOURCE_SECTION_RE = re.compile(
    r"\n+\s*(?:📚\s*)?参考来源\s*[:：]?\s*[\s\S]*$",
    re.IGNORECASE,
)
_SOURCE_LINE_RE = re.compile(r"^\s*(\[R\d+\])\s+(.+?)\s*$", re.IGNORECASE)
_SOURCE_MARKER = "===== SOURCES ====="
# Older prompts and some local models emit [W1], [Source 1] or [来源 1].
_ALIAS_RE = re.compile(r"\[(?:W|Source|来源)\s*(\d+)\]", re.IGNORECASE)


@dataclass(frozen=True)
class EvidenceCheck:
    """Result of validating one final answer against tool evidence."""

    ok: bool
    reason: str = ""
    normalized_answer: str = ""


def _canonical_ref(ref: str) -> str:
    match = REF_RE.search(ref)
    return f"[R{int(match.group(1))}]" if match else ref.upper()


def _canonical_refs(text: str) -> set[str]:
    return {f"[R{int(match.group(1))}]" for match in REF_RE.finditer(text or "")}


def _ref_sort_key(ref: str) -> int:
    match = re.search(r"\d+", ref)
    return int(match.group()) if match else 0


def has_substantive_answer(answer: str) -> bool:
    """Reject citation-only output while allowing short factual answers."""
    body = re.split(
        r"(?:^|\n)\s*(?:📚\s*)?参考来源\s*[:：]?",
        answer,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    body = re.sub(r"\[R\d+\]", "", body, flags=re.IGNORECASE)
    body = re.sub(r"https?://\S+", "", body)
    meaningful = re.sub(r"[^A-Za-z0-9㐀-鿿]+", "", body)
    return len(meaningful) >= 2


def rag_source_map(evidence: str) -> dict[str, str]:
    """Extract the exact reference-to-document mapping from RAG tool output.

    A turn can contain more than one RAG observation (different queries), so
    every ``===== SOURCES =====`` block is scanned.  If two blocks reuse the
    same positional reference, the first mapping wins; the answer's reference
    is still valid, only its display name becomes ambiguous.
    """
    sources: dict[str, str] = {}
    for block in evidence.split(_SOURCE_MARKER)[1:]:
        block = block.split("IMPORTANT:", 1)[0]
        for line in block.splitlines():
            match = _SOURCE_LINE_RE.match(line)
            if match:
                sources.setdefault(
                    _canonical_ref(match.group(1)), match.group(2).strip(),
                )
    return sources


def _urls(text: str) -> set[str]:
    """URLs compared without trailing sentence punctuation."""
    return {url.rstrip(".,;:!?") for url in URL_RE.findall(text or "")}


def normalize_citations(answer: str, evidence: str) -> str:
    """Normalize citation aliases and rebuild the source section from evidence.

    Only aliases for references that already exist in the evidence are
    rewritten.  A fabricated ``[R9]`` is left untouched on purpose so
    :func:`validate_answer` can reject the answer instead of silently
    stripping the bad citation.
    """
    if not answer:
        return answer

    evidence_refs = _canonical_refs(evidence)
    if not evidence_refs:
        return answer

    normalized = _ALIAS_RE.sub(
        lambda match: f"[R{int(match.group(1))}]", answer,
    )
    normalized = REF_RE.sub(
        lambda match: f"[R{int(match.group(1))}]", normalized,
    )

    cited_refs = _canonical_refs(normalized)
    if cited_refs - evidence_refs:
        # Keep the invalid reference visible for validation.
        return normalized

    source_map = rag_source_map(evidence)
    if not source_map:
        return normalized

    body = _SOURCE_SECTION_RE.sub("", normalized).rstrip()
    if not cited_refs:
        # A source section without any reference cannot be verified.
        return body

    lines = [
        f"- {ref} {source_map[ref]}"
        for ref in sorted(source_map, key=_ref_sort_key)
        if ref in cited_refs
    ]
    if not lines:
        return normalized
    return f"{body}\n\n参考来源：\n" + "\n".join(lines)


def fact_anchors(text: str) -> set[str]:
    """Extract objective anchors (numbers, dates, scores) from text.

    URLs and citation IDs contain incidental digits, not factual claims, so
    they are removed before extraction.
    """
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
        normalized_number = (
            str(float(number)).rstrip("0").rstrip(".")
            if "." in number else str(int(number))
        )
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


def facts_supported(answer: str, evidence: str) -> bool:
    """Whether every objective anchor in the answer also occurs in evidence."""
    answer_anchors = fact_anchors(answer)
    if not answer_anchors:
        return True
    return answer_anchors.issubset(fact_anchors(evidence))


def validate_answer(
    answer: str,
    evidence: str,
    *,
    require_evidence: bool = False,
    require_citation: bool = False,
) -> EvidenceCheck:
    """Validate one final answer against the evidence the model actually saw.

    Args:
        answer: User-facing answer extracted from the ReAct output.
        evidence: Rendered tool observations (already truncated the same way
            the model saw them).
        require_evidence: When True, an empty evidence string is a failure.
        require_citation: When True, answers backed by RAG references must
            cite at least one of them.
    """
    answer = answer or ""
    evidence = evidence or ""
    normalized = normalize_citations(answer, evidence)

    if not normalized.strip():
        return EvidenceCheck(False, "最终答案为空", normalized)

    if _REACT_TRACE_RE.search(normalized):
        return EvidenceCheck(
            False, "最终答案包含 Thought/Action 过程行", normalized,
        )

    evidence_refs = _canonical_refs(evidence)
    answer_refs = _canonical_refs(normalized)
    invalid_refs = answer_refs - evidence_refs
    if invalid_refs:
        refs = "、".join(sorted(invalid_refs, key=_ref_sort_key))
        return EvidenceCheck(
            False, f"引用了检索结果中不存在的来源编号：{refs}", normalized,
        )

    if require_citation and evidence_refs and not answer_refs:
        return EvidenceCheck(
            False, "答案没有引用任何检索来源", normalized,
        )

    if require_evidence and not evidence.strip():
        return EvidenceCheck(False, "没有可用的工具证据", normalized)

    if evidence.strip() and not has_substantive_answer(normalized):
        return EvidenceCheck(
            False, "答案只有来源标记，缺少实质内容", normalized,
        )

    if evidence.strip():
        answer_urls = _urls(normalized)
        if answer_urls:
            invalid_urls = answer_urls - _urls(evidence)
            if invalid_urls:
                urls = "、".join(sorted(invalid_urls))
                return EvidenceCheck(
                    False, f"引用了检索结果中不存在的链接：{urls}",
                    normalized,
                )

        unsupported = fact_anchors(normalized) - fact_anchors(evidence)
        if unsupported:
            anchors = "、".join(sorted(unsupported))
            return EvidenceCheck(
                False,
                f"答案中的数字或日期在检索结果中找不到依据：{anchors}",
                normalized,
            )

    return EvidenceCheck(True, "", normalized)
