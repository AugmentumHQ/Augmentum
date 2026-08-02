"""Memory integration helpers for route handlers."""

from __future__ import annotations

import asyncio
import contextlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.security.untrusted import ensure_policy_in_system, wrap_untrusted
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from starlette.datastructures import State

    from augmentum.memory.store import MemoryStore
    from augmentum.models.base import InternalChatRequest

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Message accumulator — batches messages before LLM extraction
# ---------------------------------------------------------------------------

@dataclass
class _MessagePair:
    user: str
    assistant: str


@dataclass
class _SessionBuffer:
    """Accumulates message pairs for a session until batch threshold is met."""
    pairs: list[_MessagePair] = field(default_factory=list)
    mode: str = "passthrough"
    scope: str | None = None
    chat_model: str = ""  # model the user is chatting with (fallback for extraction)
    flush_handle: asyncio.TimerHandle | None = None


# Global buffer: (user_id, session_id) → _SessionBuffer. The tuple key
# prevents two tenants with the same session_id from cross-mixing pending
# pairs before extraction lands them in the DB.
_extraction_buffers: dict[tuple[str, str], _SessionBuffer] = {}

# Track total stored memories per user to trigger reflection periodically
_user_stored_counts: dict[str, int] = {}
_REFLECTION_EVERY_N = 50  # trigger reflection every N stored memories

# Seconds of inactivity before flushing a partial buffer
_FLUSH_DELAY_SECONDS = 60.0


def _get_buffer(
    user_id: str, session_id: str, mode: str, scope: str | None,
) -> _SessionBuffer:
    key = (user_id, session_id)
    buf = _extraction_buffers.get(key)
    if buf is None:
        buf = _SessionBuffer(mode=mode, scope=scope)
        _extraction_buffers[key] = buf
    else:
        buf.mode = mode
        buf.scope = scope
    return buf


def schedule_extraction(
    app_state: State,
    request: InternalChatRequest,
    assistant_content: str,
    session_id: str,
    user_id: str = "default",
    mode: str = "passthrough",
    source_message_id: str | None = None,
) -> None:
    """Schedule async memory extraction if memory is enabled.

    Accumulates message pairs and runs LLM extraction in batches
    (every N messages, configured via memory_extraction_batch_size).
    Explicit "remember X" instructions are always extracted immediately.

    Note: source_message_id is the frontend tree node ID of the user message.
    Currently not passed by routes (frontend doesn't send it). The dream
    context retrieval falls back to fuzzy-matching evidence text against
    session trees. Future: frontend could send message_id in the request
    body to enable precise context window retrieval.
    """
    if not settings.memory_enabled:
        log.info("extraction_skip_disabled")
        return

    # Narrative mode has its own memory system — skip to avoid poisoning
    # analytical/passthrough/agentic memories with in-character roleplay content.
    if mode == "narrative":
        log.debug("extraction_skip_narrative")
        return

    store = getattr(app_state, "memory_store", None)
    if store is None:
        log.info("extraction_skip_no_store")
        return

    # Get the last user message
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        return

    scope = mode if settings.memory_scope_by_mode else None

    # Step 1: Always extract explicit instructions immediately ("remember X").
    # Runs in every non-narrative mode — the user opted in by phrasing.
    from augmentum.memory.extractor import _extract_explicit_only

    explicit_facts = _extract_explicit_only(user_message)
    if explicit_facts:
        _fire_explicit_extraction(
            app_state, store, explicit_facts, session_id, user_id, scope,
        )

    # Step 2: Batch LLM extraction — only for modes in the capture allowlist.
    # Coder / builder / voice work isn't getting-to-know-you chat; mining their
    # stream for user facts produces too many false positives (project specs
    # read as user preferences, in-progress build state read as durable facts).
    # Explicit "remember X" still works in those modes via Step 1.
    if mode not in settings.memory_capture_modes:
        log.debug("extraction_skip_mode_not_in_capture_list", mode=mode)
        return

    # Step 3: Accumulate into session buffer
    # NOTE: Per-message pre-filtering is deliberately removed. With batch-of-10,
    # individual messages that seem boring alone may reveal meaningful facts in
    # the full conversation context. The validation gate in batch_extract_and_store
    # handles quality filtering after extraction instead.
    buf = _get_buffer(user_id, session_id, mode, scope)
    buf.pairs.append(_MessagePair(user=user_message, assistant=assistant_content))
    # Track the chat model for extraction fallback
    if request.model:
        buf.chat_model = request.model

    batch_size = max(1, settings.memory_extraction_batch_size)

    if len(buf.pairs) < batch_size:
        log.info(
            "extraction_buffered",
            session_id=session_id,
            buffered=len(buf.pairs),
            threshold=batch_size,
        )
        # Schedule a delayed flush so partial buffers don't sit forever
        _schedule_delayed_flush(buf, app_state, store, session_id, user_id, mode, scope)
        return

    # Step 3: Batch threshold reached — cancel any pending timer and flush
    if buf.flush_handle is not None:
        buf.flush_handle.cancel()
        buf.flush_handle = None

    pairs = buf.pairs[:]
    chat_model = buf.chat_model
    buf.pairs.clear()

    # Clean up empty buffer to prevent unbounded growth of _extraction_buffers
    _extraction_buffers.pop((user_id, session_id), None)

    _fire_batch_extraction(
        app_state, store, pairs, session_id, user_id, mode, scope,
        chat_model=chat_model,
        source_message_id=source_message_id,
    )


def _schedule_delayed_flush(
    buf: _SessionBuffer, app_state: State, store: MemoryStore,
    session_id: str, user_id: str, mode: str, scope: str | None,
) -> None:
    """Schedule a delayed flush for a partial buffer.

    If the user stops sending messages before the batch threshold is reached,
    this ensures accumulated pairs still get extracted after a timeout.
    """
    if buf.flush_handle is not None:
        buf.flush_handle.cancel()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()

    key = (user_id, session_id)

    def _do_flush() -> None:
        buf.flush_handle = None
        if not buf.pairs:
            _extraction_buffers.pop(key, None)
            return
        pairs = buf.pairs[:]
        buf.pairs.clear()
        _extraction_buffers.pop(key, None)  # cleanup after flush
        log.debug("extraction_delayed_flush", session_id=session_id, flushed=len(pairs))
        _fire_batch_extraction(app_state, store, pairs, session_id, user_id, mode, scope,
                               chat_model=buf.chat_model)

    buf.flush_handle = loop.call_later(_FLUSH_DELAY_SECONDS, _do_flush)


def _fire_explicit_extraction(
    app_state: State,
    store: MemoryStore,
    facts: list,
    session_id: str,
    user_id: str,
    scope: str | None,
) -> None:
    """Immediately store explicit 'remember X' facts — no LLM needed."""
    async def _store_explicit() -> None:
        stored = 0
        stored_details: list[dict] = []
        for fact in facts:
            fact.source_context["session_id"] = session_id
            memory_id = await store.store_fact(
                fact, user_id=user_id, session_id=session_id,
                is_explicit=True, scope=scope,
            )
            stored += 1
            stored_details.append({
                "id": memory_id,
                "content": fact.content,
                "evidence": fact.evidence,
                "type": fact.type.value if hasattr(fact.type, "value") else str(fact.type),
                "confidence": fact.confidence,
            })
        if stored:
            log.info("explicit_memory_stored", count=stored, session_id=session_id)
            profile_mgr = getattr(app_state, "core_profile_manager", None)
            if profile_mgr:
                profile_mgr.notify_extraction(user_id)
            if not settings.memory_auto_approve:
                from augmentum.memory.notifications import queue_notification
                for detail in stored_details:
                    try:
                        await queue_notification(
                            store._conn,
                            memory_id=detail["id"],
                            content=detail["content"],
                            evidence=detail.get("evidence", ""),
                            tier="active",
                            confidence=detail.get("confidence", 1.0),
                            memory_type=detail.get("type", "fact"),
                            user_id=user_id,
                        )
                    except Exception:
                        log.warning("explicit_memory_notification_failed", memory_id=detail["id"], exc_info=True)
            # Explicit memories are always approved — notify dream scheduler
            dream_scheduler = getattr(app_state, "dream_scheduler", None)
            if dream_scheduler:
                for fact in facts:
                    dream_scheduler.notify_approval(fact.content[:32], user_id=user_id)

    def _on_error(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception():
            log.warning("explicit_extraction_failed", error=str(task.exception()))

    task = asyncio.create_task(_store_explicit())
    task.add_done_callback(_on_error)


def _fire_batch_extraction(
    app_state: State,
    store: MemoryStore,
    pairs: list[_MessagePair],
    session_id: str,
    user_id: str,
    mode: str,
    scope: str | None,
    chat_model: str = "",
    source_message_id: str | None = None,
) -> None:
    """Fire batched LLM extraction for accumulated message pairs."""
    from augmentum.memory.extractor import batch_extract_and_store

    async def _extract_and_notify() -> None:
        # Resolve backend inside the async task via role-based chain
        backend = None
        model = None
        if settings.memory_llm_extraction_enabled:
            registry = getattr(app_state, "provider_registry", None)
            if registry:
                try:
                    backend, model = await registry.resolve_model_for_role(
                        "utility",
                        override=settings.memory_llm_extraction_model,
                        settings=settings,
                    )
                except (ValueError, KeyError):
                    backend = None

        log.info(
            "batch_extraction_starting",
            session_id=session_id,
            pairs=len(pairs),
            backend_available=backend is not None,
            model=model or "(none)",
        )
        result = await batch_extract_and_store(
            session_id=session_id,
            user_id=user_id,
            pairs=[(p.user, p.assistant) for p in pairs],
            store=store,
            scope=scope,
            backend=backend,
            model=model,
            mode=mode,
        )
        # Handle both old (int) and new (int, list) return formats
        if isinstance(result, tuple):
            stored, stored_details = result
        else:
            stored, stored_details = result, []

        log.info(
            "batch_extraction_result",
            session_id=session_id,
            stored=stored,
            details=len(stored_details),
        )

        # Record source_message_id for dream context retrieval
        if source_message_id and stored_details:
            for detail in stored_details:
                memory_id = detail.get("id")
                if memory_id:
                    try:
                        await store._conn.execute(
                            "UPDATE memories SET source_message_id = ? WHERE id = ?",
                            (source_message_id, memory_id),
                        )
                    except Exception as exc:
                        # Dream context retrieval has fuzzy fallback;
                        # surface the failure for repeat-pattern visibility.
                        log.debug(
                            "memory_source_message_id_update_failed",
                            memory_id=memory_id,
                            error=str(exc),
                        )
            try:
                await store._conn.commit()
            except Exception as exc:
                log.debug("memory_integration_commit_failed", error=str(exc))

        if stored > 0:
            profile_mgr = getattr(app_state, "core_profile_manager", None)
            if profile_mgr:
                profile_mgr.notify_extraction(user_id)

            # Auto-approve: promote all to ACTIVE silently, skip notification UI
            if settings.memory_auto_approve:
                for detail in stored_details:
                    if detail.get("confidence", 1.0) < 0.7:
                        try:
                            await store.update_tier(detail["id"], "active", user_id=user_id)
                        except Exception:
                            log.warning("memory_auto_approve_failed", memory_id=detail.get("id"), exc_info=True)
                log.info("memory_auto_approved", count=stored)
            else:
                # Queue notifications for the frontend
                from augmentum.memory.notifications import queue_notification
                for detail in stored_details:
                    tier = "provisional" if detail.get("confidence", 1.0) < 0.7 else "active"
                    try:
                        await queue_notification(
                            store._conn,
                            memory_id=detail["id"],
                            content=detail["content"],
                            evidence=detail.get("evidence", ""),
                            tier=tier,
                            confidence=detail.get("confidence", 0.8),
                            memory_type=detail.get("type", "fact"),
                            user_id=user_id,
                        )
                    except Exception:
                        log.warning("memory_notification_queue_failed", memory_id=detail.get("id"), exc_info=True)

            # Notify dream scheduler regardless of approval mode. The scheduler's
            # `_approved_since` gate exists to ensure dream cycles only fire when
            # there's something memorable to reflect on — and the LLM extractor's
            # confidence filtering is what actually establishes that. Whether the
            # user has auto-approve on or prefers to manually review notifications
            # is a UX preference, not a signal about whether the material is
            # worth dreaming about. Without this, users who leave auto-approve
            # off and don't actively use the approval UI never trigger a single
            # dream cycle no matter how long they use Augmentum.
            dream_scheduler = getattr(app_state, "dream_scheduler", None)
            if dream_scheduler:
                for detail in stored_details:
                    dream_scheduler.notify_approval(detail["id"], user_id=user_id)

            # Trigger reflection periodically (every N stored memories)
            prev = _user_stored_counts.get(user_id, 0)
            _user_stored_counts[user_id] = prev + stored
            if _user_stored_counts[user_id] >= _REFLECTION_EVERY_N:
                _user_stored_counts[user_id] = 0
                try:
                    ids = await trigger_reflection(app_state, user_id=user_id)
                    if ids:
                        log.info("reflection_triggered", user_id=user_id, count=len(ids))
                except Exception:
                    log.debug("reflection_trigger_failed", exc_info=True)

    def _on_error(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception():
            log.warning("batch_extraction_failed", error=str(task.exception()))

    task = asyncio.create_task(_extract_and_notify())
    task.add_done_callback(_on_error)


def _build_document_context(
    chunks: list,
    sufficiency: str,
) -> str:
    """Build structured document context for system message injection."""
    if not chunks and sufficiency == "none":
        return ""

    lines = ["<reference_material>"]
    for sc in chunks:
        c = sc.chunk
        label = "high" if sc.tier == "high" else "moderate"
        source = f"[Document: {c.get('filename', 'unknown')}"
        if c.get("page_num"):
            source += f" p.{c['page_num']}"
        source += f"] (relevance: {label})"
        lines.append(f"{source}\n{c.get('content', '')}")
    lines.append("</reference_material>")

    if sufficiency == "sufficient":
        lines.append(
            "Ground your response in the reference material above. "
            "Cite specific details when possible."
        )
    elif sufficiency == "partial":
        lines.append(
            "The reference material above may not fully address the query. "
            "Use it where relevant but indicate when you are drawing on "
            "general knowledge rather than the provided material."
        )

    return "\n\n".join(lines)


async def _resolve_analyzer_backend(app_state):
    """Resolve LLM backend for query analysis. Returns None if unavailable."""
    registry = getattr(app_state, "provider_registry", None)
    if not registry:
        return None

    # Per-feature RAG analysis override takes precedence; falls back to role chain
    rag_override = settings.document_rag_query_analysis_model or settings.memory_llm_extraction_model or ""
    try:
        backend, _ = await registry.resolve_model_for_role(
            "utility",
            override=rag_override,
            settings=settings,
        )
        return backend
    except (ValueError, KeyError, AttributeError):
        return None


async def _build_topic_map_lazy(app_state, doc_store) -> None:
    """Build topic coverage map on first use, cache on app_state."""
    from augmentum.documents.topic_coverage import build_topic_map

    conn = doc_store._conn  # noqa: SLF001
    rows = await conn.execute("SELECT id, filename FROM documents")
    docs = []
    for row in await rows.fetchall():
        chunks_rows = await conn.execute(
            "SELECT content FROM document_chunks "
            "WHERE document_id = ? AND chunk_index >= 0 ORDER BY chunk_index",
            (row["id"],),
        )
        content = "\n".join(r["content"] for r in await chunks_rows.fetchall())
        docs.append({"id": row["id"], "filename": row["filename"], "content": content})

    if docs:
        build_topic_map(docs)
        app_state._topic_map_built = True
        log.debug("topic_map_built_lazy", documents=len(docs))


async def recall_and_inject(
    request: InternalChatRequest,
    app_state: State,
    user_id: str = "default",
    limit: int | None = None,
    min_score: float | None = None,
    mode: str = "passthrough",
    scope: str | None = None,
    session_id: str | None = None,
    memory_query: str = "",
) -> None:
    """Recall relevant memories and inject them into the request's system prompt.

    Modifies the request in-place by prepending a ``[background]`` block
    to the system message (or creating one if absent).  The block ends
    with a directive telling the model to only reference entries when
    the user's current message directly concerns them — preventing the
    unprompted "you mentioned X" behaviour.  Analytical mode is skipped
    because UARF uses MemoryRecallTool on-demand.
    """
    if not settings.memory_enabled:
        return

    store: MemoryStore | None = getattr(app_state, "memory_store", None)
    if store is None:
        return

    # Use config defaults if not overridden
    effective_limit = limit if limit is not None else settings.memory_recall_limit
    effective_min_score = min_score if min_score is not None else settings.memory_recall_min_score

    # Derive scope from mode when not explicitly provided
    effective_scope = scope if scope is not None else (mode if settings.memory_scope_by_mode else None)

    # Analytical mode: gated by memory_inject_analytical (default off)
    if mode == "analytical":
        if not settings.memory_inject_analytical:
            return
        await _set_analytical_memory_hint(
            request, store, user_id, effective_limit, effective_min_score, effective_scope,
        )
        return

    # Agentic mode: gated by memory_inject_agentic (default off)
    if mode == "agentic" and not settings.memory_inject_agentic:
        return

    # Narrative mode: skip cross-session memory injection when disabled
    # (can poison narratives across different cards/personas)
    if mode == "narrative" and not settings.narrative_cross_session_memory:
        return

    # Use explicit memory query if provided, otherwise derive from last user message
    query = memory_query.strip() if memory_query else ""
    if not query:
        for msg in reversed(request.messages):
            if msg.role == "user":
                query = msg.content
                break
    if not query:
        return

    # Injection-only floor: stricter than the plain recall floor so weak
    # topical matches don't auto-appear in the system prompt every turn.
    # The tool/hint paths keep using ``effective_min_score`` directly.
    inject_floor = effective_min_score
    try:
        configured_inject_floor = float(settings.memory_inject_min_score)
        if configured_inject_floor > inject_floor:
            inject_floor = configured_inject_floor
    except (TypeError, ValueError, AttributeError):
        pass

    # HyDE: optional query expansion via a small LLM. Generates a
    # hypothetical answer and adds its embedding as a third RRF leg in
    # store.recall. Gated by setting; backend resolution failure is a
    # silent fall-through to the plain recall path.
    hyde_text = ""
    if settings.memory_hyde_enabled:
        registry = getattr(app_state, "provider_registry", None)
        if registry is not None:
            try:
                hyde_backend, hyde_model = await registry.resolve_model_for_role(
                    "utility",
                    override=settings.memory_hyde_model,
                    settings=settings,
                )
            except (ValueError, KeyError):
                hyde_backend = None
                hyde_model = ""
            if hyde_backend is not None:
                from augmentum.memory.hyde import expand_query
                try:
                    hyde_text = await expand_query(query, hyde_backend, hyde_model)
                except Exception:
                    log.debug("hyde_expand_failed", exc_info=True)

    try:
        memories = await store.recall(
            query, user_id=user_id, limit=effective_limit,
            min_score=inject_floor, scope=effective_scope,
            hyde_text=hyde_text,
        )
    except Exception:
        log.warning("memory_recall_failed", exc_info=True)
        return

    # Build compact summary instead of raw verbatim injection. Query is
    # passed so long memories (reflections, archive entries) reduce to
    # query-relevant sentences instead of bleeding the prompt budget on
    # paragraphs the model doesn't need this turn.
    memory_block = (
        _build_user_summary(memories, max_chars=settings.memory_summary_max_chars, query=query)
        if memories else ""
    )

    # Prepend core profile if available
    if settings.memory_core_profile_enabled:
        profile_mgr = getattr(app_state, "core_profile_manager", None)
        if profile_mgr and mode != "analytical":
            try:
                core_text = await profile_mgr.get_profile(user_id)
                if core_text:
                    memory_block = (
                        core_text + "\n" + memory_block if memory_block else core_text
                    )
            except Exception:
                log.debug("core_profile_injection_failed", exc_info=True)

    # Document RAG — v2 pipeline with query analysis + scoring
    doc_block = ""
    doc_results: list = []
    if settings.document_rag_enabled:
        doc_store = getattr(app_state, "document_store", None)
        if doc_store:
            try:
                bindings = await _get_session_doc_bindings(app_state, session_id, user_id)

                if bindings:
                    full_ids = [b["document_id"] for b in bindings if b["inject_mode"] == "full"]
                    search_ids = [b["document_id"] for b in bindings if b["inject_mode"] == "search"]

                    doc_lines: list[str] = []

                    # Full injection (unchanged — concatenate entire document)
                    for doc_id in full_ids:
                        full_content = await doc_store.get_full_content(doc_id, user_id=user_id)
                        if full_content:
                            doc_lines.append(
                                f"[Document: {full_content['filename']}]\n{full_content['content']}"
                            )

                    # Search injection — v2 pipeline
                    if search_ids and query:
                        from augmentum.documents.dedup import deduplicate
                        from augmentum.documents.query_analyzer import QueryAnalyzer
                        from augmentum.documents.scoring import (
                            apply_budget,
                            cliff_detect,
                            determine_sufficiency,
                            score_gate,
                        )
                        from augmentum.documents.topic_coverage import (
                            check_topic_coverage,
                        )

                        # Resolve search-mode document names for analyzer context
                        search_doc_names: list[str] = []
                        for b in bindings:
                            if b["inject_mode"] == "search":
                                doc_meta = await doc_store.get_document(b["document_id"], user_id=user_id)
                                if doc_meta:
                                    search_doc_names.append(doc_meta.get("filename", "unknown"))

                        # Lazy-build topic coverage map (cached on app_state)
                        if not getattr(app_state, "_topic_map_built", False):
                            try:
                                await _build_topic_map_lazy(app_state, doc_store)
                            except Exception:
                                log.debug("topic_map_build_failed", exc_info=True)

                        # Topic coverage: soft signal for uncovered queries
                        topic_coverage_score = 1.0
                        try:
                            coverage = check_topic_coverage(query)
                            topic_coverage_score = coverage["best_match_score"]
                        except Exception:
                            log.debug("topic_coverage_check_failed", query=query[:80], exc_info=True)

                        # Step 1: Query analysis
                        analyzer = QueryAnalyzer(
                            backend=await _resolve_analyzer_backend(app_state),
                        )
                        analysis = await analyzer.analyze(
                            query,
                            doc_names=search_doc_names,
                            has_full_docs=bool(full_ids),
                        )
                        log.debug("rag_query_analysis",
                                  strategy=analysis.strategy,
                                  queries=analysis.queries,
                                  reason=analysis.reason)

                        if analysis.strategy != "skip":
                            # Step 2: Search (single or multi-query)
                            search_queries = analysis.queries or [query]
                            _search_limit = settings.document_rag_recall_limit

                            all_result_lists: list = []  # populated only for multi-query

                            if len(search_queries) == 1:
                                raw_results = await doc_store.search(
                                    search_queries[0],
                                    user_id=user_id,
                                    limit=_search_limit,
                                    document_id=search_ids[0] if len(search_ids) == 1 else None,
                                )
                                if len(search_ids) > 1:
                                    raw_results = [
                                        r for r in raw_results
                                        if r.get("doc_id") in search_ids
                                    ]
                            else:
                                # Concurrent sub-query search
                                import asyncio as _aio
                                tasks = [
                                    doc_store.search(sq, user_id=user_id, limit=_search_limit)
                                    for sq in search_queries
                                ]
                                all_result_lists = await _aio.gather(*tasks)
                                # Max-score merge by chunk_id
                                merged_map: dict[str, dict] = {}
                                for sq_results in all_result_lists:
                                    for r in sq_results:
                                        cid = r["chunk_id"]
                                        if cid not in merged_map or r.get("score", 0) > merged_map[cid].get("score", 0):
                                            merged_map[cid] = r
                                # EXPERIMENTAL: guarantee top-1 from each sub-query
                                if settings.document_rag_min_representation:
                                    for sq_results in all_result_lists:
                                        if sq_results:
                                            top = sq_results[0]
                                            cid = top["chunk_id"]
                                            if cid not in merged_map:
                                                merged_map[cid] = top

                                raw_results = sorted(
                                    merged_map.values(),
                                    key=lambda r: r.get("score", 0),
                                    reverse=True,
                                )
                                raw_results = [
                                    r for r in raw_results
                                    if r.get("doc_id") in search_ids
                                ]

                            # Detect if both vec and FTS contributed
                            dual_source = getattr(doc_store, "_last_search_dual_source", True)

                            # Topic coverage dampening: reduce scores for poorly-covered topics
                            # so the score gate filters them more aggressively
                            if raw_results and topic_coverage_score < 0.3:
                                dampen = 0.3 + topic_coverage_score
                                for r in raw_results:
                                    r["score"] = r.get("score", 0) * dampen

                            if raw_results:
                                # Step 3: Score gate
                                scored = score_gate(
                                    raw_results,
                                    reranker_enabled=settings.reranker_enabled,
                                    dual_source=dual_source,
                                    query=query,
                                )
                                # Step 4: Cliff detection
                                clipped = cliff_detect(
                                    scored,
                                    cliff_ratio=settings.document_rag_cliff_ratio,
                                    max_results=settings.document_rag_recall_limit,
                                )
                                # Step 5: Deduplication
                                deduped = deduplicate(clipped)
                                # Step 6: Budget
                                budgeted = apply_budget(
                                    deduped,
                                    max_tokens=settings.document_rag_max_context_tokens,
                                )
                                # Step 7: Sufficiency + context assembly
                                # EXPERIMENTAL: pass sub-query results for decompose-aware check
                                _sq_results = (
                                    all_result_lists
                                    if (analysis.strategy == "decompose"
                                        and settings.document_rag_decompose_sufficiency
                                        and len(search_queries) > 1)
                                    else None
                                )
                                sufficiency = determine_sufficiency(
                                    budgeted,
                                    strategy=analysis.strategy,
                                    sub_query_results=_sq_results,
                                )
                                doc_block = _build_document_context(budgeted, sufficiency)
                                doc_results = [sc.chunk for sc in budgeted]

                                log.debug("rag_v2_pipeline",
                                          strategy=analysis.strategy,
                                          raw=len(raw_results),
                                          scored=len(scored),
                                          clipped=len(clipped),
                                          deduped=len(deduped),
                                          budgeted=len(budgeted),
                                          sufficiency=sufficiency)

                            # Preview fallback — when search returned nothing
                            # but docs are explicitly session-bound
                            if not doc_results and analysis.strategy != "skip":
                                preview_lines: list[str] = []
                                for doc_id in search_ids:
                                    full_content = await doc_store.get_full_content(doc_id, user_id=user_id)
                                    if full_content:
                                        preview = full_content["content"][:4000]
                                        if len(full_content["content"]) > 4000:
                                            preview += "\n[... truncated ...]"
                                        preview_lines.append(
                                            f"[Document: {full_content['filename']}]\n{preview}"
                                        )
                                if preview_lines:
                                    doc_block = (
                                        "<reference_material>\n"
                                        + "\n\n".join(preview_lines)
                                        + "\n</reference_material>\n\n"
                                        "The reference material above may not fully address the query. "
                                        "Use it where relevant but indicate when you are drawing on "
                                        "general knowledge rather than the provided material."
                                    )

                    # Wrap full-mode docs in reference tags
                    if doc_lines and not doc_block:
                        doc_block = (
                            "<reference_material>\n"
                            + "\n\n".join(doc_lines)
                            + "\n</reference_material>\n\n"
                            "Ground your response in the reference material above. "
                            "Cite specific details when possible."
                        )
                    elif doc_lines and doc_block:
                        full_section = (
                            "<reference_material>\n"
                            + "\n\n".join(doc_lines)
                            + "\n</reference_material>"
                        )
                        doc_block = full_section + "\n\n" + doc_block

            except Exception:
                log.debug("document_recall_failed", exc_info=True)

    # Knowledge-pack injection is intentionally NOT here. Packs are
    # encyclopedic reference corpora (Wikipedia, Python docs, medwiki),
    # orthogonal to memory recall. They have per-mode toggles and a
    # condensing path of their own — see augmentum/knowledge/injection.py.
    # The route layer calls inject_pack_context() separately so the per-
    # mode early returns above don't silently disable pack RAG.

    # Combine blocks — documents first, then memory
    # NOTE: This reverses the prior ordering (memory first, then docs).
    # Anthropic research shows long documents at top with query at bottom
    # improves quality by up to 30%.
    #
    # Build Plan Phase 1.1: each block is wrapped in untrusted-content
    # markers (``augmentum/security/untrusted.py``) before it enters the
    # prompt. The wrapper marks document chunks and memory recall as
    # data, not instructions, so a poisoned memory or document chunk
    # cannot smuggle commands into Becca's behavior. The policy
    # preamble explaining the marker convention is added once per turn
    # via ``ensure_policy_in_system`` — idempotent across the other
    # injection paths (knowledge/injection.py) that also wrap content.
    parts: list[str] = []
    if doc_block:
        parts.append(wrap_untrusted("documents/rag", doc_block))
    if memory_block:
        parts.append(wrap_untrusted("memory/active", memory_block))
    combined = "\n\n".join(parts)
    if not combined:
        return

    # Find or create system message
    from augmentum.models.base import Message

    system_msg = None
    for msg in request.messages:
        if msg.role == "system":
            system_msg = msg
            break

    if system_msg:
        system_msg.content = combined + "\n\n" + system_msg.content
    else:
        request.messages.insert(0, Message(role="system", content=combined))

    # Add the prompt-safety policy preamble. Idempotent — knowledge
    # pack injection (knowledge/injection.py) also calls this; whoever
    # runs first wins, subsequent calls are no-ops.
    ensure_policy_in_system(request)

    log.debug("context_injected",
              memories=len(memories) if memories else 0,
              documents=len(doc_results) if doc_block else 0,
              query=query[:60])


async def _get_session_doc_bindings(
    app_state: State,
    session_id: str | None,
    user_id: str,
) -> list[dict]:
    """Look up documents bound to a session via session_documents table.

    Returns list of {document_id, inject_mode} for bindings that belong to
    *user_id*. Rows with a NULL user_id (legacy bindings written before
    multi-tenant scoping landed) are intentionally excluded — re-bind via
    the UI to populate user_id, or run a one-time backfill from sessions.
    """
    if not session_id or not user_id:
        return []

    try:
        from augmentum.state.backends.sqlite import SQLiteBackend
        sm = getattr(app_state, "state_manager", None)
        if not sm or not isinstance(sm.backend, SQLiteBackend):
            return []
        conn = sm.backend.conn
        cursor = await conn.execute(
            "SELECT document_id, inject_mode FROM session_documents "
            "WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


async def _set_analytical_memory_hint(
    request: InternalChatRequest,
    store: MemoryStore,
    user_id: str,
    limit: int,
    min_score: float,
    scope: str | None,
) -> None:
    """Check for relevant memories and set a lightweight hint on the request.

    Does NOT modify ``request.messages`` — only sets ``request.memory_hint``
    so the UARF engine can thread the signal into its context.
    """
    query = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            query = msg.content
            break
    if not query:
        return

    try:
        memories = await store.recall(
            query, user_id=user_id, limit=limit,
            min_score=min_score, scope=scope,
        )
    except Exception:
        log.debug("memory_hint_recall_failed", exc_info=True)
        return

    if not memories:
        return

    hint = _build_memory_hint(memories)
    if hint:
        request.memory_hint = hint
        log.debug("memory_hint_set", count=len(memories), query=query[:60])


def _build_memory_hint(memories: list) -> str:
    """Build a short hint string describing available memories by type.

    Returns an empty string if the list is empty.
    """
    if not memories:
        return ""

    from collections import Counter

    type_counts: Counter[str] = Counter()
    for mem in memories:
        mtype = getattr(mem, "memory_type", "fact")
        # Normalise MemoryType enum values to plain strings
        type_counts[str(mtype).split(".")[-1]] += 1

    parts = [f"{count} {mtype}{'s' if count != 1 else ''}" for mtype, count in type_counts.items()]
    total = sum(type_counts.values())
    breakdown = ", ".join(parts)
    return (
        f"[memory_available]\n"
        f"{total} relevant user memor{'ies' if total != 1 else 'y'} found ({breakdown}). "
        f"Use the memory_recall tool to retrieve details if needed."
    )


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\w+")
_SENTENCE_TRIM_FLOOR = 200  # only trim memories longer than this
_TRIM_MAX_SENTENCES = 3

# Tiny stopword set — generic English function words. Kept small so
# domain-specific tokens (cat, API, AI) survive. Length-based
# filters dropped meaningful short words; stopword-based filtering
# preserves them.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "does",
    "for", "from", "has", "have", "he", "her", "his", "i", "in", "is",
    "it", "its", "me", "my", "of", "on", "or", "our", "she", "that",
    "the", "their", "them", "they", "this", "to", "was", "we", "were",
    "what", "when", "where", "which", "who", "will", "with", "you",
    "your", "yours", "im", "youre", "ive", "weve", "theyre",
})


def _content_words(text: str) -> set[str]:
    """Tokenize and strip stopwords. Generic across domains."""
    return {
        w.lower()
        for w in _WORD_RE.findall(text)
        if len(w) >= 2 and w.lower() not in _STOPWORDS
    }


def _trim_to_relevant_sentences(content: str, query: str) -> str:
    """Return at most ``_TRIM_MAX_SENTENCES`` sentences ordered by query overlap.

    Generic word-overlap scoring — fast, no embedding cost, works for any
    domain. Short content (≤ floor) and queryless calls return content
    unchanged. Borrowed in shape from ArcRift's "surgical sentence
    trimming" but scored on word overlap instead of per-sentence
    embeddings (the per-sentence embedding pass is heavier and we don't
    have that index).
    """
    if not content or len(content) <= _SENTENCE_TRIM_FLOOR:
        return content
    if not query:
        return content[:_SENTENCE_TRIM_FLOOR]
    sentences = _SENTENCE_SPLIT_RE.split(content.strip())
    if len(sentences) <= _TRIM_MAX_SENTENCES:
        return content
    query_words = _content_words(query)
    if not query_words:
        return content[:_SENTENCE_TRIM_FLOOR]
    scored: list[tuple[int, int, str]] = []
    for idx, sent in enumerate(sentences):
        sent_words = _content_words(sent)
        overlap = len(query_words & sent_words)
        if overlap > 0:
            scored.append((overlap, idx, sent))
    if not scored:
        # No sentence matched the query — keep first 200 chars as a
        # generic preview rather than fabricating relevance.
        return content[:_SENTENCE_TRIM_FLOOR]
    scored.sort(key=lambda t: (-t[0], t[1]))
    keep_indices = {idx for _, idx, _ in scored[:_TRIM_MAX_SENTENCES]}
    # Reassemble in original order so prose still reads naturally.
    return " ".join(s for i, s in enumerate(sentences) if i in keep_indices)


def _build_user_summary(memories: list, max_chars: int = 300, query: str = "") -> str:
    """Build an internal-knowledge block from recalled memories.

    Preserves the caller-provided order — ``MemoryStore.recall`` already
    ranks by composite relevance * strength * importance * tier * surprise,
    so re-sorting here by ``importance * confidence`` would discard the
    query-specific signal and keep surfacing the same top-N memories every
    turn regardless of what the user actually asked about.

    When ``query`` is supplied, memories longer than the sentence-trim
    floor get reduced to their query-relevant sentences — saves prompt
    budget for the long-form reflections / archive entries the system
    sometimes surfaces, without affecting the typical short-form memory.

    Framed as internal knowledge (not context-to-surface) so the model uses
    it to shape HOW it communicates rather than quoting entries back at the
    user. ``max_chars`` caps the bullet section; the preamble is fixed.
    """
    if not memories:
        return ""

    bullets: list[str] = []
    chars_used = 0
    for mem in memories:
        content = _trim_to_relevant_sentences(mem.content, query)
        line = f"- {content}"
        if chars_used + len(line) + 1 > max_chars:
            break
        bullets.append(line)
        chars_used += len(line) + 1  # +1 for newline

    if not bullets:
        return ""  # No memories fit the budget

    preamble = (
        "[internal_knowledge]\n"
        "This is knowledge you have collected about the user over a long "
        "period of time. It may or may not be relevant to the current "
        "conversation or query.\n"
        "Use it as a guide for how to communicate with the user — shaping "
        "your tone, style, and the assumptions you make. It is not content "
        "to surface directly. You saw these notes; the user has not, and "
        "would be confused if shown them unprompted."
    )
    return preamble + "\n\n" + "\n".join(bullets)


def format_memory_summary(memories: list) -> str:
    """Format memories into a summary string for narrative context builder."""
    if not memories:
        return ""
    lines = ["Context from previous sessions:"]
    for mem in memories:
        conf = f" (confidence: {mem.confidence})" if mem.confidence < 1.0 else ""
        lines.append(f"- {mem.content}{conf}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 2: PROVISIONAL cleanup trigger
# ---------------------------------------------------------------------------

async def cleanup_provisional_memories(app_state: State) -> int:
    """Clean up expired PROVISIONAL memories. Call periodically."""
    store = getattr(app_state, "memory_store", None)
    if store is None:
        return 0
    return await store.cleanup_provisional()


# ---------------------------------------------------------------------------
# Phase 4: Reflection trigger
# ---------------------------------------------------------------------------

async def trigger_reflection(app_state: State, user_id: str = "default") -> list[str]:
    """Generate reflective insights from accumulated memories.

    Returns list of stored reflection memory IDs.
    Call periodically (every 50 messages) or on-demand.
    """
    store = getattr(app_state, "memory_store", None)
    if store is None:
        return []

    if not settings.memory_enabled:
        return []

    registry = getattr(app_state, "provider_registry", None)
    if not registry:
        return []

    # Resolve backend
    try:
        backend, model = await registry.resolve_model_for_role(
            "utility",
            override=settings.memory_llm_extraction_model,
            settings=settings,
        )
    except (ValueError, KeyError):
        return []

    if not backend:
        return []

    from augmentum.memory.reflection import generate_reflections

    return await generate_reflections(store, backend, model, user_id=user_id)


# ---------------------------------------------------------------------------
# Dream context injection
# ---------------------------------------------------------------------------


async def apply_dream_injection_to_request(
    internal_req: InternalChatRequest,
    app_state,
    user_id: str,
) -> None:
    """Route-facing wrapper that injects dream context into a chat request.

    Handles the dict round-trip and the possible prepend of a new leading
    system message — both of which the inline route version got subtly
    wrong, offsetting the user's turn by one slot when no system message
    existed. The in-place path preserves ``images`` / ``tool_calls`` /
    ``thinking`` / ``tool_call_id`` on unchanged Message objects; only
    ``content`` is updated when it actually changed.
    """
    from augmentum.models.base import Message

    msg_dicts = [
        {"role": m.role, "content": m.content} for m in internal_req.messages
    ]
    original_len = len(msg_dicts)

    await resolve_and_inject_dream_context(msg_dicts, app_state, user_id)

    # inject_dream_context has exactly two behaviours: modify messages[0]
    # in place (when a system message already exists), or insert a fresh
    # system message at index 0 (when one didn't). It never touches
    # messages[1:], so we only need to propagate the head back.
    if len(msg_dicts) == original_len + 1:
        # New leading system message inserted.
        internal_req.messages.insert(
            0, Message(role=msg_dicts[0]["role"], content=msg_dicts[0]["content"]),
        )
    elif len(msg_dicts) == original_len:
        # Either unchanged, or leading content was extended with the dream block.
        if msg_dicts and internal_req.messages:
            new_content = msg_dicts[0]["content"]
            if internal_req.messages[0].content != new_content:
                internal_req.messages[0].content = new_content
    else:
        # Contract violation — the low-level helper should never change size
        # by more than one. Log and leave the request untouched.
        log.error(
            "dream_inject_size_mismatch",
            before=original_len, after=len(msg_dicts),
        )


async def resolve_and_inject_dream_context(
    messages: list[dict],
    app_state,
    user_id: str,
) -> None:
    """Fetch the caller's dream portrait + recall entries and inject into ``messages``.

    This is the single correct way to surface dream context in chat
    prompts. It encapsulates three policy gates that the route-level
    inline version historically got wrong:

    * **Persona**: the engine stores all dream data with
      ``persona_id="default"`` (single-profile design). Callers that
      passed ``user_id`` positionally matched it against ``persona_id``
      and always got empty results.
    * **Per-user opt-in**: the dream subsystem is a process singleton
      but data is user-scoped. A caller whose own ``ui.dreamEnabled``
      is off should not see dream context in their prompts even if
      another tenant has the subsystem booted.
    * **Per-user recall toggle**: ``ui.dreamRecallEnabled`` /
      ``ui.dreamRecallLimit`` are UI-exposed but were historically
      bypassed by a read of ``settings.dream_recall_enabled`` (global
      env var). The per-user value wins; the global acts as a fallback
      default and as an admin kill-switch for recall specifically.

    No-ops quietly when the subsystem isn't running, when the user
    hasn't opted in, or when there's nothing to inject.
    """
    portrait_mgr = getattr(app_state, "dream_portrait_manager", None)
    dream_journal = getattr(app_state, "dream_journal", None)
    if portrait_mgr is None and dream_journal is None:
        return

    # Opt-in check — caller must have dreamEnabled=true (or inherit the
    # install default). Fail closed on lookup errors: without a verified
    # opt-in we'd rather omit context than leak.
    store = getattr(app_state, "settings_store", None)
    if store is None:
        # No way to verify — preserve legacy unscoped behaviour.
        user_dream_on = True
    else:
        try:
            user_dream_on = (
                await store.get_user_or_global(user_id, "ui.dreamEnabled")
            ) == "true"
        except Exception:
            log.warning("dream_inject_opt_in_lookup_failed", user_id=user_id, exc_info=True)
            return

    if not user_dream_on:
        return

    # Portrait — the "evolved self" block. Present whenever there is one.
    portrait = None
    if portrait_mgr is not None:
        try:
            portrait = await portrait_mgr.get_current("default", user_id=user_id)
        except Exception:
            log.warning("dream_inject_portrait_fetch_failed", exc_info=True)

    # Entries — the "recent reflections" block. Gated by per-user recall
    # toggle on top of the dream-enabled gate above.
    dream_entries: list = []
    if dream_journal is not None:
        recall_enabled = settings.dream_recall_enabled
        recall_limit = settings.dream_recall_limit
        recall_min_sim = settings.dream_recall_min_similarity
        if store is not None:
            try:
                rval = await store.get_user_or_global(user_id, "ui.dreamRecallEnabled")
                if rval is not None:
                    recall_enabled = rval == "true"
                rlim = await store.get_user_or_global(user_id, "ui.dreamRecallLimit")
                if rlim:
                    with contextlib.suppress(TypeError, ValueError):
                        recall_limit = int(rlim)
                rsim = await store.get_user_or_global(user_id, "ui.dreamRecallMinSimilarity")
                if rsim:
                    with contextlib.suppress(TypeError, ValueError):
                        recall_min_sim = float(rsim)
            except Exception:
                log.warning("dream_inject_recall_settings_failed", exc_info=True)

        if recall_enabled:
            # Strategy: prefer semantic recall, fall back to chronological
            # ONLY when semantic recall is unavailable (no vec extension,
            # no user message text, or vec query errored). Critically, an
            # empty result from semantic recall is a SUCCESS — it means
            # nothing in the journal is similar enough to the current
            # message to be worth injecting. Falling back to chronological
            # in that case would defeat the whole purpose of the threshold
            # gate (it'd inject unrelated entries and prime the model
            # toward off-topic recall). Track availability explicitly so
            # the empty-vs-unavailable distinction is unambiguous.
            query_text = ""
            for msg in reversed(messages):
                if msg.get("role") == "user" and msg.get("content"):
                    query_text = str(msg["content"])
                    break

            semantic_attempted = False
            if query_text and getattr(dream_journal, "_vec_enabled", False):
                try:
                    similar = await dream_journal.find_similar_entries(
                        query_text, "default",
                        user_id=user_id, limit=recall_limit,
                        min_similarity=recall_min_sim,
                    )
                    semantic_attempted = True
                    dream_entries = similar  # may be empty — that's fine
                except Exception:
                    log.warning("dream_inject_semantic_recall_failed", exc_info=True)
                    # Errored — fall through to chronological as safety net

            if not semantic_attempted:
                try:
                    entries, _ = await dream_journal.list_entries(
                        "default", limit=recall_limit, user_id=user_id,
                    )
                    dream_entries = [e for e in entries if e.expires_at is None]
                except Exception:
                    log.warning("dream_inject_entries_fetch_failed", exc_info=True)

    if portrait is None and not dream_entries:
        return

    await inject_dream_context(messages, portrait, dream_entries=dream_entries)


async def inject_dream_context(
    messages: list[dict],
    portrait: object | None,
    dream_entries: list | None = None,
) -> None:
    """Inject dream portrait and relevant dream entries into system message.

    Called server-side alongside memory injection. Adds <evolved_self> block
    for the portrait and <recent_reflections> for topically relevant dreams.

    Prefer :func:`resolve_and_inject_dream_context` from route handlers — it
    wraps this with the opt-in + persona_id + recall-toggle policy gates.
    This low-level helper is kept separate so tests can exercise the string-
    assembly logic without the settings-store plumbing.

    Args:
        messages: The request messages list (modified in-place)
        portrait: DreamPortrait object (or None if no dreams yet)
        dream_entries: List of relevant DreamEntry objects (or empty/None)
    """
    if portrait is None and not dream_entries:
        return

    parts = []

    if portrait is not None:
        voice = getattr(portrait, "voice_notes", "") or ""
        threads = getattr(portrait, "active_threads", "") or ""
        impressions_text = getattr(portrait, "impressions", "") or ""

        if voice or threads or impressions_text:
            parts.append(
                "<evolved_self>\n"
                f"<voice>\n{voice}\n</voice>\n"
                f"<active_threads>\n{threads}\n</active_threads>\n"
                f"<impressions>\n{impressions_text}\n</impressions>\n"
                "Use this material to shape your voice, tone, and depth — not to\n"
                "introduce subjects. Only reference these topics if the user brings\n"
                "them up first. Do not announce that this context exists, do not\n"
                "list it back, and do not steer the conversation toward it.\n"
                "</evolved_self>"
            )

    if dream_entries:
        reflections = "\n".join(
            f"- {getattr(e, 'content', str(e))}" for e in dream_entries
        )
        parts.append(
            "<recent_reflections>\n"
            "Background context only — these are private reflections you've had,\n"
            "shown for relevance to what the user just said. Do NOT bring up these\n"
            f"topics unprompted. Reference them only if the user does first.\n\n{reflections}\n"
            "</recent_reflections>"
        )

    if not parts:
        return

    dream_block = "\n\n".join(parts)

    # Find or create system message
    system_msg = None
    for msg in messages:
        if msg.get("role") == "system":
            system_msg = msg
            break

    if system_msg:
        system_msg["content"] = dream_block + "\n\n" + system_msg["content"]
    else:
        messages.insert(0, {"role": "system", "content": dream_block})
