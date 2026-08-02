"""Architect dispatch — match + filter by surface + infer args + handle.

Single entry point for "the architect runs a primitive on the user's
behalf". Voice/chat/cast all funnel through this when
``architect_dispatch_enabled`` is True. Falls back to the existing
intent dispatcher when disabled or when no architect action fires.

This module is intentionally thin — it does NOT re-implement matching
(that stays in ``augmentum.intent.matcher``), the registry, or the
handler contract. Three new things on top of the bare intent path:

  1. **Surface filter** — Action.surfaces gates whether a match is
     dispatched on this client surface. Out-of-surface actions return
     None so the caller falls through.
  2. **Inference layer** — Action.arg_inferrer fills missing args from
     observation history. Runs BETWEEN matcher hit and handler call.
  3. **Required-args validation** — after inference, if required_args
     are still missing, return a clarifying ActionResult rather than
     calling the handler with incomplete state.

The result shape is a superset of ActionResult — adds the IntentMatch
so the caller can serialize the WS event with action_id + tier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from augmentum.architect.inference import infer_args
from augmentum.config import settings
from augmentum.intent.action import (
    ActionResult,
    IntentMatch,
    SessionContext,
)
from augmentum.intent.dispatch import get_referent_cache
from augmentum.intent.matcher import match_intent
from augmentum.intent.registry import REGISTRY
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ArchitectResult:
    """Wrap an ActionResult with the originating match metadata.

    ``surface`` echoes the client surface the architect dispatched on,
    so the caller can route the WS emit (voice route vs chat route)
    correctly. ``inferred_args`` carries any args the inferrer filled
    in — useful for surfacing "we picked X" toast/speech to the user.
    """

    match: IntentMatch
    action_result: ActionResult
    surface: str
    inferred_args: dict[str, Any]

    @property
    def short_circuit(self) -> bool:
        return self.action_result.short_circuit

    @property
    def speak(self) -> str:
        return self.action_result.speak

    @property
    def toast(self) -> str:
        return self.action_result.toast

    @property
    def surface_emit(self) -> dict[str, Any] | None:
        return self.action_result.surface_emit

    @property
    def prompt_addendum(self) -> str:
        return self.action_result.prompt_addendum


async def dispatch_architect_command(
    text: str,
    *,
    surface: str,
    session: SessionContext,
    app_state: Any = None,
) -> ArchitectResult | None:
    """Run the architect command pipeline.

    Returns:
      * ArchitectResult when an architect-eligible action matched AND
        the handler returned a non-None result.
      * None in every other case (architect disabled, no match, action
        not exposed on this surface, handler opted out). Callers MUST
        fall through to the existing dispatch / LLM path on None.

    The function is None-safe on app_state — handlers that need the
    runtime (companion runtime, DB conn) read it from session.app_state
    or via getattr on app_state. Tests can omit it entirely.
    """
    enabled = getattr(settings, "architect_dispatch_enabled", False)
    # Entry log — fires once per call so we can correlate "architect saw
    # X" with "architect did Y" without grep-juggling. Cheap; produces
    # one line per voice/chat turn.
    log.info(
        "architect_dispatch_entry",
        enabled=enabled,
        surface=surface,
        text_preview=text[:80] if text else "",
    )

    if not enabled:
        return None

    if not text or not surface:
        return None

    # Match — uses the existing intent matcher (tier 1 regex today;
    # tier 2 embedding lands later and slots in here transparently).
    match = match_intent(text, mode=session.mode)
    if match is None:
        log.info("architect_dispatch_no_match", text_preview=text[:80])
        return None

    action = REGISTRY.get(match.action_id)
    if action is None:
        log.warning("architect_dispatch_missing_action", id=match.action_id)
        return None

    # Surface filter — an Action might be registered with surfaces=['voice']
    # meaning it's only callable from a voice command. Out-of-surface match
    # falls through so the existing intent dispatcher (or LLM) can pick it up.
    if not action.surfaces_for(surface):
        log.debug(
            "architect_dispatch_surface_filtered",
            action=match.action_id, surface=surface,
            action_surfaces=action.surfaces,
        )
        return None

    # Bind the persistent per-session ReferentCache. SessionContext was
    # constructed with a fresh ReferentCache via default_factory; swap
    # in the persistent one from app.state so any anchor writes survive
    # past this call. Handlers + the dispatch-anchor write below both
    # need this binding.
    if app_state is not None:
        session.referents = get_referent_cache(
            app_state, session.user_id, session.session_id,
        )
        # Continuity: rehydrate the working set (active note, trail)
        # for fresh caches — voice mints a new session id per connect,
        # which previously orphaned "the note" on every reconnect.
        try:
            from augmentum.companion_runtime.working_state import (
                hydrate_working_state,
            )
            await hydrate_working_state(
                app_state, session.user_id, session.referents,
            )
        except Exception:  # noqa: BLE001
            log.debug("working_state_hydrate_failed", exc_info=True)

    # Inference — fill missing args from observation history. Failures
    # inside the inferrer are caught and logged; partial args flow on.
    runtime = getattr(app_state, "companion_runtime", None) if app_state else None
    filled_args = await infer_args(action, dict(match.args), session, runtime)

    # Translation — reshape raw user-derived args into well-formed tool
    # input. The image primitive expands "a dog" into a scene-rich
    # prompt here, BEFORE the surface event fires. Optional — most
    # primitives don't need it (grove just needs the query verbatim).
    # Failures degrade to the untransformed args so a slow / errored
    # translator never blocks a dispatch.
    if action.arg_transformer is not None:
        try:
            transformed = await action.arg_transformer(
                dict(filled_args), session, runtime,
            )
            if isinstance(transformed, dict):
                filled_args = transformed
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            log.warning(
                "architect_arg_transformer_failed",
                action_id=action.id, error=str(exc)[:200],
            )

    # Required-args validation post-inference. If a required arg is still
    # missing, return a clarifying result rather than calling the handler.
    missing = [a for a in action.required_args if a not in filled_args or filled_args[a] in (None, "", [])]
    if missing:
        log.info(
            "architect_dispatch_missing_required",
            action=match.action_id, missing=missing,
        )
        question = f"I need to know the {missing[0]} for that — can you tell me?"
        # Park so the answer fills the slot on the next turn.
        import time as _t
        refs = getattr(session, "referents", None)
        if refs is not None:
            refs.pending_intent = {
                "action_id": match.action_id,
                "args": dict(filled_args),
                "missing": list(missing),
                "question": question,
                "asked_at": _t.time(),
            }
        clarifier = ActionResult(
            short_circuit=True,
            speak=question,
        )
        return ArchitectResult(
            match=match,
            action_result=clarifier,
            surface=surface,
            inferred_args=filled_args,
        )

    # Hand off to the existing Action handler. Handler signature is
    # unchanged so this is interchangeable with the bare intent path.
    _pi_before = getattr(getattr(session, "referents", None), "pending_intent", None)
    try:
        result = await action.handler(text, session, filled_args)
    except Exception as exc:  # noqa: BLE001 — log and fall through
        log.warning(
            "architect_handler_error",
            action=match.action_id, error=str(exc)[:200],
        )
        return None

    if result is None:
        # Handler opted out at runtime — fall through to LLM.
        return None

    if result.clarify:
        # Handler-level ask ("what city should I use?") — park so the
        # answer fills the slot. The new dict fails the _pi_before
        # identity check below, so the dispatch-resolves-park clear
        # leaves it alone.
        from augmentum.intent.dispatch import park_clarify
        park_clarify(
            getattr(session, "referents", None),
            action_id=match.action_id,
            args=filled_args,
            clarify=result.clarify,
            question=result.speak,
        )
    else:
        # Results ring — the dispatch survives a few turns as a digest
        # so "when does that start?" has a referent next turn.
        from augmentum.companion_runtime import ring as _ring
        _ring.record_action_result(
            getattr(session, "referents", None),
            action_id=match.action_id, args=filled_args, result=result,
        )

    log.info(
        "architect_dispatch_ok",
        action=match.action_id,
        tier=match.tier,
        surface=surface,
        short_circuit=result.short_circuit,
        inferred_keys=sorted(set(filled_args) - set(match.args)),
    )

    # Record the dispatch as an observation so Becca remembers she did
    # this. Three sinks:
    #   1. Generic ReferentCache anchors (last_dispatch_*) so "the
    #      thing you just did" resolves on a follow-up turn.
    #   2. Runtime bus emit on ``surface.companion.architect_dispatch``
    #      so the observer's recent deque + journal pick it up via the
    #      existing surface.* prefix subscription.
    # Per-primitive anchors (last_played_track, last_image_prompt) are
    # set by the primitive handlers themselves, since only they know
    # their domain-specific subject.
    import time as _time
    refs = getattr(session, "referents", None)
    if refs is not None:
        refs.last_dispatch_action = match.action_id
        refs.last_dispatch_args = dict(filled_args)
        refs.last_dispatch_summary = (result.speak or result.toast or "")[:200]
        refs.last_dispatch_ts = _time.time()
        # A successful dispatch resolves any parked clarification —
        # either it WAS the fill, or the user moved on — UNLESS this
        # very handler just parked a fresh one (identity check).
        if refs.pending_intent is _pi_before:
            refs.pending_intent = None
        # Continuity write-through (active note, dispatch anchors).
        try:
            from augmentum.companion_runtime.working_state import (
                save_working_state,
            )
            await save_working_state(app_state, session.user_id, refs)
        except Exception:  # noqa: BLE001
            log.debug("working_state_save_failed", exc_info=True)

    if runtime is not None:
        try:
            bus = getattr(runtime, "bus", None)
            if bus is not None:
                await bus.publish_topic(
                    "surface.companion.architect_dispatch",
                    {
                        "user_id": session.user_id,
                        "session_id": session.session_id,
                        "surface": surface,
                        "action_id": match.action_id,
                        "tier": match.tier,
                        "args": _scrub_payload(filled_args),
                        "outcome": "short_circuit" if result.short_circuit else "augmented",
                        "spoken": result.speak[:200] if result.speak else "",
                    },
                )
        except Exception as exc:  # noqa: BLE001 — observability never breaks dispatch
            log.warning("architect_dispatch_bus_emit_failed", error=str(exc)[:200])

    return ArchitectResult(
        match=match,
        action_result=result,
        surface=surface,
        inferred_args=filled_args,
    )


def _scrub_payload(args: dict[str, Any]) -> dict[str, Any]:
    """Strip large/sensitive fields from the args before broadcast.

    Bus payloads land in the recent deque (50 items) and the journal —
    keep them small. Drops embedding blobs, long content bodies, etc.
    Keeps short string/int/float/bool/list-of-primitives values.
    """
    out: dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, (bool, int, float)):
            out[k] = v
        elif isinstance(v, str):
            out[k] = v[:200]
        elif isinstance(v, (list, tuple)):
            # Keep small lists of primitives; drop nested structures.
            primitives = [x for x in v if isinstance(x, (str, int, float, bool))]
            if primitives and len(primitives) == len(v):
                out[k] = primitives[:10]
        # Drop dicts / blobs silently.
    return out
