"""Chat-mode resolution via the companion runtime's dispatcher.

The legacy :class:`augmentum.classifier.router.RequestClassifier` decides
mode from request shape: model prefix, X-Augmentum-Mode header, narrative
patterns, complexity heuristics, session continuity. The companion
runtime's dispatcher (:mod:`augmentum.companion_runtime.dispatch`) decides
which *subagent* should handle an intent using a richer feature set —
persona-kernel affinity, runtime state/role/focus, lexical similarity,
recency, the user-mode-hint as one feature among many.

This module bridges the two. When the flag is on and the runtime is
ready, the dispatcher runs on the incoming chat turn and — if it picks
a chat-compatible subagent with reasonable confidence — its decision
becomes the mode. When the flag is off, when the runtime isn't up,
when dispatch abstains, or when dispatch picks a non-chat subagent
(``build`` / ``bug_finder``), the legacy classifier is the source of
truth.

This is the *first* place dispatch decisions reach production traffic.
Before this, dispatch was only reachable via the explicit
``/api/companion/intent`` route, and chat ran in parallel to it. Now
the chat path consults her.

**Streaming is preserved.** Dispatch chooses the mode; the existing
streaming mode handlers still produce the response. Per-subagent
streaming responders are a future move (a separate PR that adds a
``becca_direct`` stream path).

Gated by ``companion_dispatch_routes_chat`` (default False). When
off, this module is a transparent wrapper around the classifier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.classifier.router import (
    MODE_MAP,
    ClassificationResult,
    Mode,
    RequestClassifier,
)
from augmentum.companion_runtime.user_flags import resolve_bool
from augmentum.config import settings
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.base import InternalChatRequest

log = get_logger(__name__)


# ── Subagent name → chat-compatible mode ──────────────────────────────
#
# Five of the seven subagents map cleanly to chat modes. Two
# (``build`` and ``bug_finder``) are autonomous-only — they orchestrate
# multi-turn workflows from the tick loop, not single chat turns —
# so a dispatch decision that picks one of them for a chat request
# is treated as "no chat-compatible winner" and falls through to the
# classifier.

_SUBAGENT_TO_MODE: dict[str, Mode] = {
    "passthrough": Mode.PASSTHROUGH,
    "narrative": Mode.NARRATIVE,
    "agentic": Mode.AGENTIC,
    "coder": Mode.CODER,
    "analytical": Mode.ANALYTICAL,
    # The seam — when dispatch picks becca_direct, the chat path
    # routes through her own prompt composer instead of a legacy
    # mode handler. Subagent registration is gated by
    # ``companion_becca_direct_enabled``, so this mapping only
    # produces a winner when the flag is on.
    "becca_direct": Mode.BECCA_DIRECT,
}


# Minimum dispatch winning utility to override the classifier. Picked
# conservative: dispatch must have meaningful confidence, not just be
# numerically ahead. Below this, the classifier (which has narrative
# detection + complexity analysis baked in) is the safer source of
# truth. Tunable via ``companion_dispatch_chat_min_utility``.
DEFAULT_MIN_UTILITY: float = 0.45


async def _read_default_mode(app_state, user_id: str) -> str | None:
    """Read the user's pinned default mode from user_settings.

    Written by the offer substrate's ``mode_switch`` accept handler.
    Returns ``None`` for anonymous users or when the settings store
    isn't attached (test contexts).
    """
    if not user_id:
        return None
    store = getattr(app_state, "settings_store", None)
    if store is None:
        return None
    try:
        return await store.get_user(user_id, "default_mode")
    except Exception as exc:
        log.warning("default_mode_lookup_failed", error=str(exc)[:160])
        return None


# ── Public API ────────────────────────────────────────────────────────


async def resolve_chat_mode(
    app_state,
    request: InternalChatRequest,
    *,
    classifier: RequestClassifier,
    mode_override: str | None = None,
    session_mode: str | None = None,
    user_id: str = "",
    session_id: str = "",
) -> ClassificationResult:
    """Decide the chat mode for an incoming request.

    Priority (highest first):

    1. **Explicit mode override** (``mode_override`` from header / prefix).
       Honored unconditionally; user explicit intent always wins.
    2. **Companion dispatcher** when ``companion_dispatch_routes_chat``
       is on AND the runtime is up AND dispatch enabled AND the winner
       is a chat-compatible subagent above ``min_utility``.
    3. **Legacy classifier** for everything else.

    Always returns a :class:`ClassificationResult`. Never raises.

    Bus emission: when dispatch is consulted, ``dispatch.routed_chat``
    fires with *both* the dispatch winner and the classifier's
    independent decision. This is the data that lets us evaluate
    whether dispatch should be the default (i.e. compare its picks
    against the classifier's over real traffic).
    """
    # 1. Explicit override always wins — same as the classifier's own
    #    priority chain. We resolve it here so the dispatch path
    #    doesn't even run for an override turn.
    if mode_override:
        mode = MODE_MAP.get(mode_override.lower())
        if mode:
            return ClassificationResult(
                mode=mode,
                confidence=1.0,
                reason=f"explicit header override: {mode_override}",
            )

    # User-pinned default (from user_settings, written by mode_switch
    # offer accept handlers). Read here so classify() can honor it
    # over content heuristics — see priority 3 in RequestClassifier.
    default_mode = await _read_default_mode(app_state, user_id)

    # 2. Companion dispatcher. Multiple gates here — any one being false
    #    means classifier wins. We compute the classifier's decision
    #    first regardless, because (a) we need it for telemetry, (b) the
    #    fallthrough path uses it.
    classifier_result = classifier.classify(
        request,
        mode_override=mode_override,
        session_mode=session_mode,
        default_mode=default_mode,
    )

    # Per-user gates (multi-tenant fix): each user's own dispatch choice
    # decides whether their chat routes through the companion. Resolves
    # user override → install-wide → default, so existing installs are
    # unchanged.
    _store = getattr(app_state, "settings_store", None)
    if not await resolve_bool(
        _store, user_id or "", "companion_dispatch_routes_chat", False,
    ):
        return classifier_result
    runtime = getattr(app_state, "companion_runtime", None)
    if runtime is None or not getattr(runtime, "_started", False):
        return classifier_result
    if not await resolve_bool(
        _store, user_id or "", "companion_dispatch_enabled", False,
    ):
        # Routing chat through dispatch requires dispatch to be on.
        # This is by design — chat is the production surface, not the
        # place to enable dispatch for the first time.
        return classifier_result

    # Build an Intent from the request. We use the last user turn as
    # the canonical text; the dispatcher's feature extractors care
    # about that string + the user_mode_hint metadata.
    user_text = _last_user_text(request)
    if not user_text:
        return classifier_result

    try:
        from augmentum.companion_runtime import dispatch
        from augmentum.companion_runtime.runtime import Intent

        intent = Intent(
            text=user_text,
            user_id=user_id or "",
            source="user_chat",
            device_id="",
            explicit_mode="",
            metadata={
                "session_id": session_id,
                "classifier_mode": classifier_result.mode.value,
                "classifier_confidence": classifier_result.confidence,
            },
        )
        decision = await dispatch.decide(
            intent,
            runtime=runtime,
            user_mode_hint=mode_override or "",
        )
    except Exception:
        log.debug("chat_router_dispatch_failed", exc_info=True)
        return classifier_result

    # Telemetry: always emit both decisions so we can evaluate dispatch
    # picks against the classifier's over real traffic. Best-effort —
    # bus failures don't propagate.
    try:
        await runtime.bus.publish_topic(
            "dispatch.routed_chat",
            {
                "session_id": session_id,
                "classifier_mode": classifier_result.mode.value,
                "classifier_confidence": classifier_result.confidence,
                "dispatch_winner": decision.winner.name if decision.winner else None,
                "dispatch_utility": decision.winner.utility if decision.winner else 0.0,
                "dispatch_used_tiebreaker": decision.used_tiebreaker,
                "abstained": decision.abstained,
            },
            source_companion_id=runtime.companion_id,
        )
    except Exception:
        log.warning("chat_router_emit_failed", exc_info=True)

    # Decide whether dispatch overrides. Multiple gates:
    if decision.abstained or decision.winner is None:
        return classifier_result
    winner_name = decision.winner.name
    if winner_name not in _SUBAGENT_TO_MODE:
        # ``build`` / ``bug_finder`` / future autonomous-only subagents
        # — these aren't chat-appropriate even when they score highest.
        log.debug(
            "chat_router_winner_not_chat_compatible",
            winner=winner_name,
        )
        return classifier_result
    min_utility = float(
        getattr(settings, "companion_dispatch_chat_min_utility", DEFAULT_MIN_UTILITY)
    )
    if decision.winner.utility < min_utility:
        log.debug(
            "chat_router_dispatch_below_threshold",
            winner=winner_name,
            utility=decision.winner.utility,
            threshold=min_utility,
        )
        return classifier_result

    # Dispatch wins. Build the result with full provenance.
    dispatched_mode = _SUBAGENT_TO_MODE[winner_name]
    log.info(
        "chat_router_dispatch_wins",
        winner=winner_name,
        dispatched_mode=dispatched_mode.value,
        utility=decision.winner.utility,
        classifier_would_have=classifier_result.mode.value,
        classifier_confidence=classifier_result.confidence,
    )
    return ClassificationResult(
        mode=dispatched_mode,
        confidence=min(0.99, decision.winner.utility),
        reason=(
            f"companion dispatch: {winner_name} "
            f"(utility={decision.winner.utility:.3f}; "
            f"classifier would have picked {classifier_result.mode.value} "
            f"@ {classifier_result.confidence:.2f})"
        ),
        metadata={
            "source": "companion_dispatch",
            "winner": winner_name,
            "utility": decision.winner.utility,
            "used_tiebreaker": decision.used_tiebreaker,
            "classifier_alt": classifier_result.mode.value,
            "classifier_alt_confidence": classifier_result.confidence,
        },
    )


def _last_user_text(request: InternalChatRequest) -> str:
    """Extract the most recent user turn for dispatch's feature inputs.

    Mirrors :func:`augmentum.proxy.streaming._last_user_text` but lives
    here too to avoid the streaming module being an import dependency
    of every chat route.
    """
    for msg in reversed(request.messages or []):
        if getattr(msg, "role", "") == "user":
            return getattr(msg, "content", "") or ""
    return ""


__all__ = ["resolve_chat_mode", "DEFAULT_MIN_UTILITY"]
