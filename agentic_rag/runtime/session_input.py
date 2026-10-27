"""Shared session-input construction for every chat entrypoint.

``/chat``, ``/chat/stream``, ``/chat/voice``, the messaging gateways and the
WebSocket capability must assemble history identically.  This module is the
single place that:

1. ensures the SQLite session exists,
2. loads summary + recent history through :class:`ContextBuilder`,
3. resolves attachment IDs through :class:`MediaContextBuilder` and appends
   exactly one current-turn message (native parts and/or a text digest).

It deliberately does **not** persist the current user message.  Persistence
happens once, in ``ChatOrchestrator``, after the turn completes, so a message
is never both "history" and "current query" in the same request.
"""

from __future__ import annotations

import asyncio

from agentic_rag.agent.routing import Route, RuleRouter
from agentic_rag.data.models import AgentInput
from agentic_rag.services.media.context import MediaContextBuilder
from agentic_rag.services.media.store import AttachmentStore


def _safe_llm():
    """Return the configured LLM, or None when the provider is unavailable."""
    try:
        from agentic_rag.services.llm.factory import get_llm
        return get_llm()
    except Exception:
        return None


async def build_chat_input(
    sid: str,
    query: str,
    attachment_ids: list[str] | None = None,
    repo=None,
    llm=None,
    user_id: str = "web_ui",
    *,
    attachment_store: AttachmentStore | None = None,
    media_builder: MediaContextBuilder | None = None,
) -> AgentInput:
    """Load history and build the request input for one user turn."""
    from agentic_rag.data.db.session_repo import SessionRepo
    from agentic_rag.services.llm.base import estimate_tokens
    from agentic_rag.services.memory.context_builder import (
        get_context_builder,
        history_budget_for,
    )

    repo = repo or SessionRepo()
    # Session lookups are synchronous SQLite calls; run them in a worker
    # thread so a slow disk does not stall the event loop.
    if not await asyncio.to_thread(repo.get, sid):
        try:
            await asyncio.to_thread(repo.create_with_id, sid, user_id=user_id)
        except Exception:
            # A concurrent request may have created it between the check and
            # the insert; only re-raise when the session is still missing.
            if not await asyncio.to_thread(repo.get, sid):
                raise

    llm = llm or _safe_llm()
    attachment_ids = [att for att in (attachment_ids or []) if att]

    has_media = bool(attachment_ids)
    profile = "balanced"
    try:
        from agentic_rag.config.settings import get_settings
        profile = str(
            getattr(get_settings().agent, "routing_profile", "balanced")
            or "balanced"
        )
    except Exception:
        pass
    decision = RuleRouter.decide(query, has_media=has_media, profile=profile)
    supports_vision = bool(getattr(llm, "supports_vision", False))
    # Tool use works on textual observations, and a text-only model needs the
    # digest regardless of strategy.  Pure DIRECT requests to a vision model
    # skip the extra captioning call and use the image natively.  A complex
    # media request may need a retrieval seed even when its route is direct.
    complex_media = has_media and decision.complexity == "complex"
    need_digest = (
        decision.route != Route.DIRECT
        or not supports_vision
        or complex_media
    )

    media_context = None
    if has_media:
        builder = media_builder or MediaContextBuilder(
            store=attachment_store, llm=llm,
        )
        media_context = await builder.build(
            query,
            attachment_ids,
            need_digest=need_digest,
            user_id=user_id,
            session_id=sid,
        )

    # The current query (and any media representation) is not part of history,
    # but it consumes the same context window, so subtract it from the budget.
    count_tokens = getattr(llm, "count_tokens", None)
    try:
        query_tokens = (
            count_tokens(query)
            if callable(count_tokens)
            else estimate_tokens(query)
        )
    except Exception:
        query_tokens = estimate_tokens(query)
    current_tokens = (
        media_context.estimated_tokens if media_context is not None
        else query_tokens
    )
    budget = max(256, history_budget_for(llm) - current_tokens)

    bundle = await get_context_builder().build(sid, budget, llm=llm)
    history = bundle.to_messages()

    input_data = AgentInput(query=query, messages=history)
    if media_context is not None:
        current_message = media_context.to_message()
        if current_message is not None:
            input_data.messages = history + [current_message]
        input_data.attachments = [
            attachment.to_ref() for attachment in media_context.attachments
        ]
        input_data.retrieval_query = (
            media_context.retrieval_query or query
        )
        input_data.parameters["media"] = media_context.diagnostics
        input_data.parameters["strategy"] = (
            "direct" if decision.route == Route.DIRECT else "tool_use"
        )
    else:
        input_data.retrieval_query = query

    input_data.parameters["session_id"] = sid
    input_data.parameters["sid"] = sid
    input_data.parameters["user_id"] = user_id
    input_data.parameters["context_report"] = bundle.report
    return input_data
