"""Knowledge-pack context injection.

Independent of the memory subsystem: pack RAG fires whenever a session has
a pack bound and the per-mode toggle is on. Memory recall and pack recall
are orthogonal concepts — packs are encyclopedic reference corpora
(Wikipedia, Python docs, medical wikis) attached explicitly by the user;
memory is global cross-session recall about the user.

Pipeline per request:
  1. Per-mode toggle gate (narrative defaults off; chat-style modes on).
  2. Resolve session→pack bindings; bail if none.
  3. Maybe-condense the latest user message into a self-contained query
     (skipped on first turn, on long self-contained questions, and on
     cache hits).
  4. Hybrid search via PackManager: per-pack vector + FTS5 + ZIM keyword,
     RRF merge, optional cross-encoder rerank.
  5. Format results into a <reference_material> block, prepend to the
     system message.

Every skip path emits an observable log line; every failure degrades to
no-op rather than raising. Operators can answer "why didn't my pack
respond?" by filtering on ``knowledge_pack_*`` in structured logs.
"""
from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.models.base import InternalChatRequest, Message
from augmentum.security.untrusted import ensure_policy_in_system, wrap_untrusted
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from starlette.datastructures import State

    from augmentum.knowledge.packs import PackResult

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-mode configuration
# ---------------------------------------------------------------------------

def _pack_mode_enabled(mode: str) -> bool:
    """Whether pack injection is enabled for the given request mode."""
    flag_map = {
        "passthrough": "knowledge_packs_passthrough",
        "analytical": "knowledge_packs_analytical",
        "agentic": "knowledge_packs_agentic",
        "narrative": "knowledge_packs_narrative",
    }
    flag = flag_map.get(mode)
    if not flag:
        # Unknown mode (e.g. coder) — packs default off. Coder uses its own
        # repo/digest indexing; pack noise hurts its plan/act loop.
        return False
    return bool(getattr(settings, flag, False))


def _per_mode_limit(mode: str) -> int:
    """Result cap for the given mode, falling back to ``knowledge_max_results``."""
    key_map = {
        "passthrough": "knowledge_max_results_passthrough",
        "analytical": "knowledge_max_results_analytical",
        "agentic": "knowledge_max_results_agentic",
        "narrative": "knowledge_max_results_narrative",
    }
    key = key_map.get(mode)
    if key:
        value = int(getattr(settings, key, 0) or 0)
        if value > 0:
            return value
    return int(getattr(settings, "knowledge_max_results", 5) or 5)


def _per_mode_max_chars(mode: str) -> int:
    """Reference block char budget. Chat-mode is tighter; UARF/agentic widen."""
    if mode == "passthrough":
        return 1500
    return 2500


async def _search_packs(
    *,
    pack_mgr,
    app_state,
    user_id: str,
    query: str,
    pack_ids: list[str],
    limit: int,
) -> list:
    """Run the hybrid pack search, optionally fanning out to peers.

    When fabric is disabled or no requested pack is peer-only, this is
    exactly the original single ``pack_mgr.search`` call -- zero
    behaviour change. When fabric is enabled AND one or more requested
    packs are only available on a peer, we partition the request:
    locally-held packs run on local hybrid search; peer-held packs
    proxy via :meth:`RoutingDirector.fanout_knowledge_search`, which
    POSTs a signed search request to each owning peer in parallel.

    Peer results come back as dicts (the JSON form of PackResult);
    they're translated into PackResult-compatible objects so the rest
    of the pipeline (rerank, formatting) doesn't care about the source.
    """
    director = getattr(app_state, "fabric_director", None)
    if director is None:
        # Default-off path: identical to pre-fabric behaviour.
        return await pack_mgr.search(
            query=query,
            pack_ids=pack_ids,
            limit=limit,
            rerank=settings.reranker_enabled,
        )

    # Identify which packs we actually have locally so the director
    # can partition. The PackManager already tracks both augpack and
    # zim caches; we use the public listing to avoid touching its
    # private attrs.
    local_installed = {p["pack_id"] for p in pack_mgr.installed()}

    # All packs we need are local? Stay on the simple path.
    if all(pid in local_installed for pid in pack_ids):
        return await pack_mgr.search(
            query=query,
            pack_ids=pack_ids,
            limit=limit,
            rerank=settings.reranker_enabled,
        )

    async def _local_search(local_pack_ids: list[str]):
        if not local_pack_ids:
            return []
        # Don't apply rerank inside the partial leg — we re-rerank
        # across the union at the end.
        return await pack_mgr.search(
            query=query, pack_ids=local_pack_ids, limit=limit, rerank=False,
        )

    raw = await director.fanout_knowledge_search(
        query=query, requested_pack_ids=pack_ids,
        local_pack_ids=local_installed,
        user_id=user_id, limit=limit,
        local_search_fn=_local_search,
    )

    # Translate dict-shaped peer results into PackResult-compatible
    # objects for downstream code that does ``r.content``, ``r.score``
    # etc. Local results in ``raw`` are already PackResult instances.
    from augmentum.knowledge.packs import PackResult

    normalised = []
    for item in raw:
        if isinstance(item, PackResult):
            normalised.append(item)
        elif isinstance(item, dict):
            try:
                normalised.append(PackResult(
                    content=str(item.get("content", "") or ""),
                    title=str(item.get("title", "") or ""),
                    section=str(item.get("section", "") or ""),
                    url=str(item.get("url", "") or ""),
                    pack_id=str(item.get("pack_id", "") or ""),
                    source=str(item.get("source", "peer") or "peer"),
                    score=float(item.get("score", 0.0) or 0.0),
                ))
            except Exception:
                log.debug("knowledge_pack_peer_result_skipped", item_keys=list(item.keys()))

    # Sort by score descending and trim to the global limit. We do not
    # re-run cross-encoder rerank across the union here because peer
    # results were already RRF'd on their side; doing it again would
    # need shipping the cross-encoder cost cross-fabric or running it
    # locally on imported chunks (which we can do, but only when
    # ``settings.reranker_enabled``). Phase 6.b enhancement.
    normalised.sort(key=lambda r: r.score, reverse=True)
    return normalised[:limit]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def inject_pack_context(
    request: InternalChatRequest,
    app_state: State,
    *,
    user_id: str,
    session_id: str,
    mode: str,
) -> dict | None:
    """Search session-bound packs and prepend a <reference_material> block.

    Side effects:
        Modifies ``request.messages`` in place — prepends the pack block
        onto the existing system message, or inserts a new system message
        if none is present.

    Returns:
        A small metadata dict the route layer surfaces to the UI when the
        feature was *attempted* (pack bound, mode enabled, query present).
        ``None`` when the path was a true no-op (no bindings, mode off,
        no session) — those should not produce a UI chip.

        Shape::
            {
                "packs_searched": [pack_id, ...],
                "results_found": int,        # post-search, post-rerank
                "results_injected": int,     # made it into the block
                "results_dropped_oversized": int,
                "block_chars": int,
                "budget_chars": int,
                "outcome": "injected" | "no_results" | "all_dropped" | "search_failed",
            }

    Failure mode:
        Never raises. Every error path logs and returns without modifying
        the request. Pack injection is best-effort context — the chat must
        still work if retrieval fails.

    Invariants:
        - No-op (returns None) when no packs are bound to the session.
        - No-op (returns None) when the per-mode toggle is off.
        - No-op (returns None) when ``session_id`` or ``user_id`` is missing.
    """
    # TEMP: bumped to INFO during commissioning — once we confirm the
    # call is reaching this site reliably, drop back to debug.
    log.info(
        "knowledge_pack_evaluating",
        mode=mode, session_id=session_id, user_id=user_id,
    )

    if not _pack_mode_enabled(mode):
        log.info("knowledge_pack_skipped", reason="mode_disabled", mode=mode)
        return None

    if not session_id:
        log.debug("knowledge_pack_skipped", reason="no_session_id", mode=mode)
        return None

    pack_mgr = getattr(app_state, "pack_manager", None)
    if pack_mgr is None:
        log.info("knowledge_pack_skipped", reason="no_pack_manager", mode=mode)
        return None

    try:
        pack_ids = await _get_session_pack_ids(app_state, session_id, user_id)
    except Exception:
        log.warning("knowledge_pack_session_lookup_failed", session_id=session_id, exc_info=True)
        return None

    if not pack_ids:
        # Silent for un-bound chats would be too quiet — when the user binds a
        # pack and the lookup still misses, the most common cause is a user_id
        # mismatch (e.g. auth produced "default" but the binding was written
        # under the real account). Surface that at INFO so it's visible
        # without raising the global log level.
        bind_count = await _count_session_bindings(app_state, session_id)
        if bind_count > 0:
            log.warning(
                "knowledge_pack_user_id_mismatch",
                session_id=session_id,
                user_id_used=user_id,
                rows_for_session=bind_count,
                mode=mode,
            )
        else:
            log.debug(
                "knowledge_pack_skipped",
                reason="no_bindings", mode=mode, session_id=session_id,
            )
        return None

    raw_query = _last_user_message(request.messages)
    if not raw_query:
        log.info("knowledge_pack_skipped", reason="no_user_message", mode=mode)
        return None

    query = await _maybe_condense(
        request.messages, raw_query, app_state, session_id,
    )

    limit = _per_mode_limit(mode)
    budget = _per_mode_max_chars(mode)
    try:
        results = await _search_packs(
            pack_mgr=pack_mgr,
            app_state=app_state,
            user_id=user_id,
            query=query,
            pack_ids=pack_ids,
            limit=limit,
        )
    except Exception:
        log.warning(
            "knowledge_pack_search_failed",
            session_id=session_id,
            packs=len(pack_ids),
            exc_info=True,
        )
        return {
            "packs_searched": pack_ids,
            "results_found": 0,
            "results_injected": 0,
            "results_dropped_oversized": 0,
            "block_chars": 0,
            "budget_chars": budget,
            "outcome": "search_failed",
        }

    if not results:
        # TEMP at INFO during commissioning — useful signal for verifying
        # the search legs are firing. Move back to debug once stable.
        log.info(
            "knowledge_pack_no_results",
            query=query[:80],
            pack_ids=pack_ids,
            condensed=(query != raw_query),
            mode=mode,
        )
        return {
            "packs_searched": pack_ids,
            "results_found": 0,
            "results_injected": 0,
            "results_dropped_oversized": 0,
            "block_chars": 0,
            "budget_chars": budget,
            "outcome": "no_results",
        }

    block, kept, dropped = _build_reference_block(results, max_chars=budget)
    # Surface top-K source identifiers to the UI so the chat chip can
    # link directly to the article in Browse. ZIM packs get a "zim:"
    # URL the Browse panel knows how to render via /api/zim. Augpack
    # results don't have a browseable URL today (chunks have no native
    # standalone view), so they surface as plain references the chip
    # renders as non-clickable text.
    sources_payload = [
        {
            "pack_id": r.pack_id,
            "title": r.title,
            "section": r.section,
            "url": (
                f"zim:{r.pack_id}/{r.url}"
                if r.source == "zim" and r.url
                else (r.url or "")
            ),
            "is_browseable": bool(r.source == "zim" and r.url),
        }
        for r in results[: max(kept, 5)]
    ]
    if not block:
        # All results too large for the budget. This is the silent-failure
        # path the no-silent-truncation-in-authoritative-context rule
        # specifically warns about: retrieval succeeded, the model
        # gets nothing, and absent this WARN nobody can tell the difference
        # between "pack worked" and "pack contributed zero." Common with
        # ZIM packs whose articles are 10K-100K chars vs a 1500-char chat
        # budget. Operators should see the sizes; UI surfaces the same to
        # users via the returned dict's "all_dropped" outcome.
        sizes = sorted((len(r.content) for r in results), reverse=True)[:5]
        log.warning(
            "knowledge_pack_block_drop_all",
            packs=pack_ids,
            results_found=len(results),
            budget_chars=budget,
            top_result_sizes=sizes,
            mode=mode,
        )
        return {
            "packs_searched": pack_ids,
            "results_found": len(results),
            "results_injected": 0,
            "results_dropped_oversized": len(results),
            "block_chars": 0,
            "budget_chars": budget,
            "outcome": "all_dropped",
            # Surface the candidates so the user can click into the source
            # even though it didn't fit the chat budget — this is exactly
            # the case where the user most needs the original article.
            "top_sources": sources_payload,
        }
    _prepend_to_system(request, block)

    log.info(
        "knowledge_pack_injected",
        results=kept,
        dropped=dropped,
        packs=len({r.pack_id for r in results[:kept]}),
        condensed=(query != raw_query),
        mode=mode,
    )
    return {
        "packs_searched": pack_ids,
        "results_found": len(results),
        "results_injected": kept,
        "results_dropped_oversized": dropped,
        "block_chars": len(block),
        "budget_chars": budget,
        "outcome": "injected",
        "top_sources": sources_payload,
    }


# ---------------------------------------------------------------------------
# Session→pack lookup (was a private helper of memory.integration)
# ---------------------------------------------------------------------------

async def _get_session_pack_ids(
    app_state: State, session_id: str, user_id: str,
) -> list[str]:
    """Fetch pack IDs bound to a session for the calling user."""
    from augmentum.state.backends.sqlite import SQLiteBackend

    sm = getattr(app_state, "state_manager", None)
    if not sm or not isinstance(sm.backend, SQLiteBackend):
        return []
    conn = sm.backend.conn
    sql = "SELECT pack_id FROM session_knowledge_packs WHERE session_id = ?"
    params: list = [session_id]
    if user_id:
        sql += " AND user_id = ?"
        params.append(user_id)
    cursor = await conn.execute(sql, params)
    rows = await cursor.fetchall()
    return [row[0] for row in rows]


async def _count_session_bindings(
    app_state: State, session_id: str,
) -> int:
    """Count pack bindings for a session ignoring user_id.

    Used only for diagnostics — when ``_get_session_pack_ids`` returns []
    but bindings exist for the session under a different user_id, that's
    a strong signal of an auth/user_id resolution bug elsewhere.
    """
    from augmentum.state.backends.sqlite import SQLiteBackend

    sm = getattr(app_state, "state_manager", None)
    if not sm or not isinstance(sm.backend, SQLiteBackend):
        return 0
    conn = sm.backend.conn
    try:
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM session_knowledge_packs WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Query condensing
# ---------------------------------------------------------------------------

# Bounded LRU. Keyed by (session_id, history_hash) — same session, same
# preceding history, same final user message → same condensed query, even
# across regen attempts. Process-local; resets on restart.
_CONDENSE_CACHE: OrderedDict[tuple[str, str], str] = OrderedDict()
_CONDENSE_CACHE_MAX = 256

# Pronouns and discourse markers that signal "this query depends on prior
# context." Frozenset for O(1) membership.
_ANAPHORIC = frozenset({
    "it", "its", "they", "them", "their", "those", "these",
    "that", "this", "him", "her", "his", "hers", "he", "she",
})
_DISCOURSE_OPENERS = frozenset({"and", "but", "also", "so", "or"})

_CONDENSE_SYSTEM_PROMPT = (
    "Rewrite the user's latest message as a self-contained search query "
    "for a reference corpus. Use the prior turns only to resolve pronouns "
    "and supply topical context. Do not add facts that are not in the "
    "conversation. If the message is already self-contained, return it "
    "unchanged. Reply with only the rewritten query — no preamble, no "
    "explanation, no quotes."
)


def _needs_condensing(query: str) -> bool:
    """Cheap heuristic: should we pay for an LLM rewrite on this query?

    Returns True for:
        - Short queries (≤ 6 words) — likely missing topic
        - Queries beginning with an anaphoric pronoun
        - Queries beginning with a discourse marker (and/but/also/so/or)
        - "what about X" / "how about X" style follow-ups

    Returns False for long queries (> 6 words) that don't open with an
    anaphor — these are usually self-contained enough to embed directly.
    """
    words = query.lower().strip().split()
    if not words:
        return False
    if len(words) <= 6:
        return True
    if words[0] in _ANAPHORIC:
        return True
    if words[0] in _DISCOURSE_OPENERS:
        return True
    bigram = " ".join(words[:2])
    if bigram in ("what about", "how about", "tell me"):
        return True
    return False


def _history_hash(messages: list[Message]) -> str:
    """Stable hash of recent conversation state for the condense cache."""
    # Last 5 messages is enough context to disambiguate; longer windows
    # would hurt cache hit rate without changing what condensing produces.
    recent = messages[-5:]
    blob = "\n".join(f"{m.role}:{m.content[:200]}" for m in recent)
    return hashlib.sha1(blob.encode("utf-8", errors="replace")).hexdigest()


def _format_history_for_condense(messages: list[Message], max_turns: int = 4) -> str:
    """Format the last few turns as plain text for the condense prompt."""
    lines = []
    for msg in messages[-(max_turns + 1):-1]:  # exclude the current user msg
        if msg.role not in ("user", "assistant"):
            continue
        prefix = "User" if msg.role == "user" else "Assistant"
        # Truncate each turn — condensing only needs topical anchors, not
        # full prior responses.
        text = (msg.content or "")[:400]
        lines.append(f"{prefix}: {text}")
    return "\n".join(lines)


async def _maybe_condense(
    messages: list[Message],
    raw_query: str,
    app_state: State,
    session_id: str,
) -> str:
    """Rewrite ``raw_query`` as a standalone search query when needed.

    Skip-paths return ``raw_query`` unchanged:
        - Condensing globally disabled
        - First turn (no history to condense against)
        - Long self-contained query (heuristic)
        - Cache hit (same session, same recent history)

    Failure-paths return ``raw_query``:
        - Model resolution failed
        - Backend call timed out (500ms cap)
        - Backend call raised
        - Backend returned empty or absurdly long output
    """
    if not settings.knowledge_query_condense_enabled:
        return raw_query

    convo_turns = sum(1 for m in messages if m.role in ("user", "assistant"))
    if convo_turns <= 1:
        return raw_query  # First turn: nothing to condense against.

    if not _needs_condensing(raw_query):
        return raw_query

    cache_key = (session_id, _history_hash(messages))
    cached = _CONDENSE_CACHE.get(cache_key)
    if cached is not None:
        _CONDENSE_CACHE.move_to_end(cache_key)
        return cached

    registry = getattr(app_state, "provider_registry", None)
    if registry is None:
        log.debug("knowledge_condense_skipped", reason="no_registry")
        return raw_query

    try:
        backend, model = await registry.resolve_model_for_role(
            role="utility",
            override=settings.knowledge_query_condense_model,
            settings=settings,
        )
    except Exception:
        log.warning("knowledge_condense_model_resolve_failed", exc_info=True)
        return raw_query

    if not backend or not model:
        return raw_query

    rewritten = await _call_condense(backend, model, messages, raw_query)
    if not rewritten or len(rewritten) > 400:
        # Empty or runaway output — embedding the raw query is safer than
        # embedding model nonsense. (400 chars ≈ 80 words; legitimate
        # search queries are universally below this.)
        return raw_query

    _CONDENSE_CACHE[cache_key] = rewritten
    if len(_CONDENSE_CACHE) > _CONDENSE_CACHE_MAX:
        _CONDENSE_CACHE.popitem(last=False)
    return rewritten


async def _call_condense(
    backend, model: str, messages: list[Message], raw_query: str,
) -> str:
    """Single non-streaming call. Reasoning disabled, output capped, 500ms hard timeout."""
    history = _format_history_for_condense(messages, max_turns=4)
    user_content = (
        f"Conversation so far:\n{history}\n\n"
        f"Latest message: {raw_query}\n\n"
        "Rewrite the latest message as a self-contained search query."
    )

    req = InternalChatRequest(
        model=model,
        messages=[
            Message(role="system", content=_CONDENSE_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ],
        max_tokens=80,        # search queries fit in <30 words; 80 tokens is generous
        temperature=0.2,      # near-deterministic, slight room for paraphrase
        think=False,          # CRITICAL: never burn reasoning budget on a 10-word rewrite
        stream=False,
        stop=["\n\n"],
    )

    try:
        result = await asyncio.wait_for(backend.chat(req), timeout=0.5)
    except TimeoutError:
        log.warning("knowledge_condense_timeout", model=model)
        return ""
    except Exception:
        log.warning("knowledge_condense_failed", model=model, exc_info=True)
        return ""

    text = (result.message.content or "").strip()
    # Strip preambles models occasionally emit despite the system prompt.
    for prefix in ("Rewritten query:", "Query:", "Search query:", "Standalone query:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    # Strip surrounding quotes — some models wrap the query.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()
    return text


# ---------------------------------------------------------------------------
# Reference block formatting
# ---------------------------------------------------------------------------

def _last_user_message(messages: list[Message]) -> str:
    """Return the content of the most recent user message, or empty string."""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content or ""
    return ""


def _build_reference_block(
    results: list[PackResult], *, max_chars: int,
) -> tuple[str, int, int]:
    """Format pack results as a <reference_material> block.

    Mirrors the directive-bearing shape used by document/web-search recall
    so pack content participates in the same grounding contract. Drops
    results that would overflow ``max_chars``.

    Returns:
        ``(block, kept_count, dropped_count)`` — block is empty string if
        every result was too large to fit. The caller uses the counts to
        surface diagnostics (loud failure on all-dropped, UI chip with
        "5 of 8 sources used", etc.).
    """
    if not results:
        return ("", 0, 0)

    lines: list[str] = []
    chars_used = 0
    dropped = 0
    for r in results:
        section_hint = f" ({r.section})" if r.section else ""
        entry = (
            f"[Source: {r.pack_id} — {r.title}]{section_hint}\n"
            f"{r.content}"
        )
        if chars_used + len(entry) + 2 > max_chars:
            dropped += 1
            continue
        lines.append(entry)
        chars_used += len(entry) + 2  # +2 for "\n\n" join

    if not lines:
        return ("", 0, dropped)

    if dropped:
        log.info("knowledge_pack_block_truncated", dropped_count=dropped, kept=len(lines))

    header = "<reference_material>"
    footer = "</reference_material>"
    directive = (
        "Ground your response in the reference material above. "
        "Cite sources inline as [Source: pack — title] when you use them."
    )
    block = header + "\n" + "\n\n".join(lines) + "\n" + footer + "\n\n" + directive
    return (block, len(lines), dropped)


def _prepend_to_system(request: InternalChatRequest, block: str) -> None:
    """Prepend ``block`` onto the existing system message, or insert one.

    Build Plan Phase 1.1: the block is wrapped in untrusted-content
    markers via ``augmentum/security/untrusted.py`` before injection.
    Pack content is retrieved from offline corpora (Wikipedia, MDWiki,
    etc.) curated by the operator, but those corpora are still external
    content that may contain prompt-injection attempts (legitimately —
    e.g. a Wikipedia article quoting an attacker prompt for analysis).
    The wrapper marks them as data; the policy preamble explains the
    marker to the model. Idempotent with memory/integration.py.
    """
    wrapped = wrap_untrusted("knowledge/pack", block)
    for msg in request.messages:
        if msg.role == "system":
            msg.content = wrapped + "\n\n" + (msg.content or "")
            ensure_policy_in_system(request)
            return
    request.messages.insert(0, Message(role="system", content=wrapped))
    ensure_policy_in_system(request)
