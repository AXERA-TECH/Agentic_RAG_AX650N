"""Deterministic request policy for the single public chat endpoint."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from zoneinfo import ZoneInfo


class RequestStrategy(str, Enum):
    DIRECT = "direct"
    TOOL_USE = "tool_use"


@dataclass(frozen=True)
class RequestDecision:
    strategy: RequestStrategy
    reason: str
    direct_answer: str = ""


class RequestPolicy:
    """Classify a request into the smallest execution path that can answer it."""

    timezone = ZoneInfo("Asia/Shanghai")

    _DIRECT_EXACT = {"你好", "您好", "hello", "hi", "谢谢", "再见", "thanks", "bye"}
    _DIRECT_HINTS = (
        "你是谁", "你是什么", "你能做什么", "自我介绍",
        "写一首", "写一段", "改写", "润色", "翻译",
    )
    _AGENT_HINTS = (
        "深入研究", "综合调研", "调研报告", "多步骤", "交叉验证",
        "research report", "deep research", "investigate and compare",
    )
    _LIVE_HINTS = (
        "最新", "目前", "现在", "今日", "今天", "实时", "冠军",
        "四强", "4强", "决赛", "半决赛", "排名", "比分", "现任",
        "latest", "current", "today", "winner", "semifinal",
        "semi-final", "ranking", "score",
    )

    @classmethod
    def decide(cls, query: str, has_media: bool = False) -> RequestDecision:
        text = query.strip()
        lowered = text.lower()

        local_answer = cls._local_date_answer(text)
        if local_answer:
            return RequestDecision(RequestStrategy.DIRECT, "deterministic date calculation", local_answer)

        if has_media:
            return RequestDecision(RequestStrategy.DIRECT, "multimodal input handled by the LLM")

        if cls._is_direct_request(lowered):
            return RequestDecision(RequestStrategy.DIRECT, "no external evidence required")

        if any(hint in lowered for hint in cls._AGENT_HINTS):
            return RequestDecision(RequestStrategy.TOOL_USE, "explicit multi-step research request")

        now = datetime.now(cls.timezone)
        # ``\b`` does not separate digits from Chinese characters because both
        # are Unicode word characters. Use numeric boundaries instead so queries
        # such as "2026世界杯举办时间" are recognized correctly.
        years = [int(year) for year in re.findall(r"(?<!\d)20\d{2}(?!\d)", lowered)]
        if any(hint in lowered for hint in cls._LIVE_HINTS) or any(year >= now.year for year in years):
            return RequestDecision(RequestStrategy.TOOL_USE, "fresh information required")

        return RequestDecision(RequestStrategy.TOOL_USE, "knowledge-base lookup")

    @classmethod
    def _is_direct_request(cls, lowered: str) -> bool:
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
    def _local_date_answer(cls, query: str) -> str:
        if not re.search(r"(星期几|周几|几号|日期)", query):
            return ""
        offsets = {"昨天": -1, "今天": 0, "今日": 0, "明天": 1}
        matched = next((word for word in offsets if word in query), None)
        if matched is None and "当前" not in query:
            return ""
        offset = offsets.get(matched, 0)
        target = datetime.now(cls.timezone) + timedelta(days=offset)
        weekdays = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")
        return f"{target.year}年{target.month}月{target.day}日，{weekdays[target.weekday()]}。"
