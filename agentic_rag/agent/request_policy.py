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
    # Optional semantic capability the router should prefer when ordering
    # tools (e.g. "fresh_information" for live queries).  Empty means no
    # preference; it never removes tools from the available set.
    preferred_capability: str = ""


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
    # Media requests are only DIRECT when they can be answered from the media
    # itself (describe, OCR, translate, transcribe, …).  As soon as the text
    # asks for knowledge or an action, the request goes through the agent so
    # RAG, MCP and the evidence gate stay in the loop.
    _MEDIA_HINTS = (
        "描述", "识别", "看图", "图中", "图上", "图里", "图片里", "图片中",
        "图片上", "这张图", "这幅图", "这张照片", "照片", "截图", "画面",
        "这是什么", "写了什么", "说了什么", "转写", "听写", "转录",
        "音频里", "音频中", "视频里", "视频中", "这段音频", "这段视频",
        "提取文字", "翻译", "分析这张", "分析一下", "看看这", "看一下",
        "describe", "transcribe", "what is in this image", "read the text",
    )
    _KNOWLEDGE_HINTS = (
        "知识库", "文档", "资料", "检索", "搜索", "查找", "查询", "报告",
        "维护", "参数", "型号", "步骤", "如何", "怎么", "怎样", "为什么",
        "多少", "剂量", "价格", "对比", "区别", "说明", "规范", "流程",
        "政策", "条款", "配置", "安装", "故障", "报错", "原因", "方法",
        "document", "manual", "spec", "troubleshoot", "compare", "price",
        "policy", "procedure", "how to", "why",
    )

    @classmethod
    def decide(cls, query: str, has_media: bool = False) -> RequestDecision:
        text = query.strip()
        lowered = text.lower()

        local_answer = cls._local_date_answer(text)
        if local_answer:
            return RequestDecision(RequestStrategy.DIRECT, "deterministic date calculation", local_answer)

        if has_media:
            if not lowered:
                return RequestDecision(
                    RequestStrategy.DIRECT, "media without a text request"
                )
            if cls._is_media_only_request(lowered):
                return RequestDecision(
                    RequestStrategy.DIRECT,
                    "media-only request can be handled by the current model",
                )
            # A generic direct request (e.g. "写一首诗" with a picture) does
            # not become a retrieval task just because media is present.
            if (cls._is_direct_request(lowered)
                    and not any(h in lowered for h in cls._KNOWLEDGE_HINTS)):
                return RequestDecision(
                    RequestStrategy.DIRECT,
                    "no external evidence required beyond the attached media",
                )

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
            return RequestDecision(
                RequestStrategy.TOOL_USE,
                "fresh information required",
                preferred_capability="fresh_information",
            )

        if has_media:
            return RequestDecision(
                RequestStrategy.TOOL_USE,
                "media plus knowledge/task request",
            )
        return RequestDecision(RequestStrategy.TOOL_USE, "knowledge-base lookup")

    @classmethod
    def _is_media_only_request(cls, lowered: str) -> bool:
        """Whether the text asks only for media understanding.

        Explicit media intent with no knowledge, live or multi-step signal is
        the DIRECT case.  ``has_media`` alone is deliberately not enough: a
        picture attached to "它的维护步骤是什么" must still search the KB.
        """
        if not any(hint in lowered for hint in cls._MEDIA_HINTS):
            return False
        if any(hint in lowered for hint in cls._KNOWLEDGE_HINTS):
            return False
        if any(hint in lowered for hint in cls._AGENT_HINTS):
            return False
        if any(hint in lowered for hint in cls._LIVE_HINTS):
            return False
        return True

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
