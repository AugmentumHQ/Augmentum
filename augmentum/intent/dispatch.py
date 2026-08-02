"""Intent dispatch — run the matched action and shape the result.

Bridges the matcher result (``IntentMatch``) to the action handler
and packages whatever the handler returns into a form the WS / chat
pipeline can consume.

Two consumers:
  * Voice WS — calls ``dispatch_for_voice`` after STT, before the
    LLM. Short-circuit results bypass the LLM entirely.
  * Text chat handlers — call ``dispatch_for_text`` before mode
    routing. Same return shape, different WS emitter.

If no intent matches, returns ``None`` and the caller falls through
to the existing pipeline. This module is intentionally additive —
removing the import would not break the chat path.
"""

from __future__ import annotations

import time as _time
from typing import Any

from augmentum.intent.action import (
    ActionResult,
    IntentMatch,
    ReferentCache,
    SessionContext,
)
from augmentum.intent.matcher import match_intent
from augmentum.intent.registry import REGISTRY
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Eviction tuning — kept module-level so the matching sweeper test in
# tests/test_smoke_intent.py can monkey-patch without reaching into the
# function body. Values picked for the running deployment shape:
#
#   * REFERENT_TTL_SECONDS — 24h matches the typical "I came back the
#     next day and asked about that thing" pattern. Shorter and the
#     follow-up referent ("show me the image again") goes cold; longer
#     and stale caches accumulate forever for users who never come back.
#   * REFERENT_SWEEP_INTERVAL_S — 60s upper-bounds how often the sweep
#     walks the store. Below that, the lazy gate skips. At ~hundreds of
#     concurrent sessions the walk is microseconds; the gate is just
#     insurance against pathologic high-frequency lookups.
REFERENT_TTL_SECONDS: float = 24 * 3600.0
REFERENT_SWEEP_INTERVAL_S: float = 60.0

# Parked-intent freshness — a clarifying question's parked verb is
# fillable for this long. Past it, the user has moved on; a stale park
# hijacking an unrelated turn is worse than re-deriving from scratch.
PENDING_INTENT_TTL_S: float = 150.0


def get_fresh_pending_intent(refs) -> dict | None:
    """Return the parked intent iff still fresh; clears stale parks.

    Shape (ReferentCache.pending_intent): ``{action_id, args, missing,
    question, asked_at}`` — written by clarify paths, consumed by the
    architect router's confidence stack so the user's ANSWER fills the
    slot instead of re-deriving (and possibly re-gambling) the intent.
    """
    pi = getattr(refs, "pending_intent", None)
    if not pi:
        return None
    import time as _t
    if _t.time() - float(pi.get("asked_at", 0) or 0) > PENDING_INTENT_TTL_S:
        refs.pending_intent = None
        return None
    return pi


def park_clarify(
    refs,
    *,
    action_id: str,
    args: dict | None,
    clarify: dict,
    question: str,
) -> None:
    """Park a handler-level clarify (``ActionResult.clarify``).

    Dispatcher-level missing-required-args clarifies park inline; this
    is for asks raised INSIDE a handler (weather.today's home-location
    ladder ending in "what city should I use?") which the dispatcher
    can't see into. All three dispatch layers (bare intent, architect,
    native-loop registry execution) call this so the parked shape stays
    identical regardless of which path ran the verb.
    """
    if refs is None or not isinstance(clarify, dict):
        return
    import time as _t
    merged = dict(args or {})
    overrides = clarify.get("args")
    if isinstance(overrides, dict):
        merged.update(overrides)
    missing = [str(m) for m in (clarify.get("missing") or [])]
    refs.pending_intent = {
        "action_id": action_id,
        "args": merged,
        "missing": missing,
        "question": (question or "")[:300],
        "asked_at": _t.time(),
    }
    log.info("intent_clarify_parked", action=action_id, missing=missing)


def _evict_stale_referents(store: dict, now: float) -> int:
    """Drop ReferentCache entries idle past the TTL.

    Returns the number of evictions. Caller passes ``now`` so tests can
    pin the clock. Pending surface events on an evicted cache are NOT
    re-queued — they were waiting for a voice WebSocket that's clearly
    gone (24h of inactivity), so re-emit would land nowhere. We log the
    count when there were any pending so operators can see the cost of
    a too-aggressive TTL.
    """
    to_drop: list[tuple] = []
    pending_lost = 0
    for key, cache in store.items():
        touched = getattr(cache, "last_touched", 0.0)
        if touched and now - touched <= REFERENT_TTL_SECONDS:
            continue
        if not touched:
            # Never-touched cache — shouldn't happen in production
            # (get_referent_cache always touches). Defensive: leave it
            # alone for one sweep so a race against a brand-new cache
            # doesn't drop it before its first use.
            continue
        to_drop.append(key)
        pending_lost += len(getattr(cache, "pending_surface_events", []) or [])
    for key in to_drop:
        store.pop(key, None)
    if to_drop:
        log.info(
            "referent_cache_evicted",
            count=len(to_drop),
            pending_lost=pending_lost,
            remaining=len(store),
        )
    return len(to_drop)


def _maybe_sweep_referents(app_state: Any, store: dict, now: float) -> None:
    """Run the eviction sweep at most once per ``REFERENT_SWEEP_INTERVAL_S``.

    The "last sweep at" timestamp is stored on app_state as a side
    attribute so the sweep gate survives across requests without
    growing a dedicated holder object.
    """
    last_sweep = getattr(app_state, "_intent_referents_last_sweep", 0.0)
    if now - last_sweep < REFERENT_SWEEP_INTERVAL_S:
        return
    try:
        _evict_stale_referents(store, now)
    finally:
        # Narrowed to setattr-failure exception types only: production
        # paths always allow attribute assignment on app.state; tests
        # sometimes pass frozen mocks or namespace objects that reject
        # it. Broader catches here would mask real bugs.
        try:
            app_state._intent_referents_last_sweep = now
        except (AttributeError, TypeError):
            pass


# Per-session referent cache singleton. Lives on app.state.intent_referents
# (dict keyed by (user_id, session_id) → ReferentCache). Lazy-created so
# unit tests / standalone matcher use can skip it.
def get_referent_cache(
    app_state: Any,
    user_id: str,
    session_id: str,
) -> ReferentCache:
    """Return the per-session ReferentCache, creating one if needed.

    Following the project's handler-cache convention: keyed by
    ``(user_id, session_id)`` so the same SQLite session_id under
    different users stays isolated.

    Also rate-limited-sweeps stale per-session caches. The sweep is
    lazy (runs only when this function is called) and gated to at most
    once per ``REFERENT_SWEEP_INTERVAL_S`` so a tight voice loop
    doesn't pay the walk cost on every turn.
    """
    if app_state is None:
        return ReferentCache()
    store = getattr(app_state, "intent_referents", None)
    if store is None:
        store = {}
        try:
            app_state.intent_referents = store
        except Exception:
            return ReferentCache()

    now = _time.monotonic()
    _maybe_sweep_referents(app_state, store, now)

    key = (user_id or "", session_id or "")
    cache = store.get(key)
    if cache is None:
        cache = ReferentCache()
        store[key] = cache
    cache.last_touched = now
    return cache


async def dispatch(
    text: str,
    *,
    session: SessionContext,
    fast_path_only: bool = False,
) -> tuple[IntentMatch, ActionResult] | None:
    """Match the transcript and run the handler.

    Returns the match + result if an action fired, or None if no
    intent matched (caller falls through to UARF / mode handler).

    ``fast_path_only`` restricts matching to conversation-control verbs
    (``fanout.fast_path``) — used by the voice route's pre-router pass
    so 'stop' / 'scratch that' fire before the address router.
    """
    match = match_intent(text, mode=session.mode, fast_path_only=fast_path_only)
    if match is None:
        return None

    action = REGISTRY.get(match.action_id)
    if action is None:
        log.warning("intent_dispatch_missing_action", id=match.action_id)
        return None

    try:
        result = await action.handler(text, session, match.args)
    except Exception as exc:
        log.warning(
            "intent_handler_error",
            action=match.action_id, error=str(exc),
        )
        return None

    if result is None:
        # Handler chose to opt out at runtime (e.g., no last image to
        # show). Fall through to the LLM rather than swallow the turn.
        return None

    if result.clarify:
        park_clarify(
            getattr(session, "referents", None),
            action_id=match.action_id,
            args=match.args,
            clarify=result.clarify,
            question=result.speak,
        )
    else:
        # Results ring — verb outcomes decay to digests, not to nothing.
        from augmentum.companion_runtime import ring as _ring
        _ring.record_action_result(
            getattr(session, "referents", None),
            action_id=match.action_id, args=match.args, result=result,
        )

    log.info(
        "intent_dispatch_ok",
        action=match.action_id,
        tier=match.tier,
        short_circuit=result.short_circuit,
        has_addendum=bool(result.prompt_addendum),
        has_surface_emit=result.surface_emit is not None,
    )
    return match, result


def serialize_action_event(
    match: IntentMatch,
    result: ActionResult,
) -> dict[str, Any]:
    """Build the WS payload for an intent action.

    Schema (versioned via ``v``):
      {"type": "intent_action", "v": 1, "action": str,
       "speak": str, "toast": str,
       "surface": {channel: str, payload: dict} | None}
    """
    payload: dict[str, Any] = {
        "type": "intent_action",
        "v": 1,
        "action": match.action_id,
        "tier": match.tier,
        "short_circuit": result.short_circuit,
    }
    if result.speak:
        payload["speak"] = result.speak
    if result.toast:
        payload["toast"] = result.toast
    if result.surface_emit is not None:
        payload["surface"] = result.surface_emit
    return payload
