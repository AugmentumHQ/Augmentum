"""Tool adapter — exposes Action primitives as UARF Tools.

The Action registry is the canonical store of Becca's primitives (one
verb per action, with arg schema + handler). To let the LLM *invoke*
those primitives mid-turn, each tier-3 action needs to be reachable
through the existing tool framework (``augmentum.tools.base.Tool`` +
``augmentum.tools.registry.ToolRegistry``). This module bridges the
two — wrapping each Action in a ``Tool`` subclass and providing a
single ``register_action_tools(...)`` helper that ``server.py`` calls
during startup.

When the LLM emits a tool call for an action (e.g. ``note.create``),
the tool runner finds the wrapper here, which invokes the action's
handler against a freshly-built SessionContext. The action's
``ActionResult`` is folded into a ``ToolResult``:

  * ``speak`` becomes the tool ``output`` (what the LLM sees + can
    compose its spoken reply around)
  * ``toast`` is included in the output text so the model knows what
    confirmation surfaced
  * ``surface_emit`` is stashed in ``metadata.intent_action`` so a
    future side-channel emitter (Phase 9 v2) can push the WS event;
    today the DB write is enough — the user reads the LLM's
    confirmation and can surface artifacts via direct phrasing.
"""

from __future__ import annotations

from typing import Any

from augmentum.intent.action import (
    Action,
    SessionContext,
)
from augmentum.intent.registry import REGISTRY
from augmentum.tools.base import SurfaceExposure, Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Stakes classes whose actions must never auto-fire from a model-issued tool
# call. Only the two genuinely-irreversible tiers: ``disruptive``/``costly`` are
# recoverable, and ``personal`` gates on speaker verification elsewhere, so
# gating those too would be heavy-handed (a media-interrupt confirm prompt every
# time). Tunable in one place as the write-verbs from the Android action head
# (send/call/pay/set_alarm) come online.
_CONSENT_REQUIRED_STAKES: frozenset[str] = frozenset({
    "irrevocable", "safety_critical",
})


class ActionTool(Tool):
    """Tool wrapper around a registered :class:`Action`.

    The wrapper carries the Action by reference, so registry hot-swaps
    (rare) are reflected without re-registration. ``input_schema`` is
    derived from the action's ``arg_schema``; properties not present
    map to a permissive ``{}`` schema, which still lets the LLM call
    with no args.
    """

    # Flag for augmentum.tools.chain — these tools write user-scoped
    # data (notes, memory, etc.) so chain.py must inject ``_user_id``
    # into kwargs before calling execute().
    needs_user_context = True

    def __init__(self, action: Action, app_state: Any) -> None:
        self._action = action
        self._app_state = app_state

    @property
    def name(self) -> str:
        return self._action.id

    @property
    def description(self) -> str:
        # Examples are appended so the model gets concrete phrasings it
        # can pattern on; the LLM doesn't see ``examples`` separately.
        # Keep terse — large tool descriptions push out other context.
        base = (self._action.summary or "").strip()
        if not self._action.examples:
            return base
        # Cap examples list to 3 to keep the tool description compact.
        ex = ", ".join(repr(e) for e in self._action.examples[:3])
        return f"{base} e.g. {ex}".strip()

    @property
    def category(self) -> ToolCategory:
        # Action primitives don't slot neatly into the existing tool
        # categories (which were designed for the UARF phase gating).
        # ``EXECUTE`` is the closest semantic match — these are verbs
        # the model invokes to make something happen, not searches or
        # fetches.
        return ToolCategory.EXECUTE

    @property
    def surfaces(self) -> SurfaceExposure:
        # Same reach as the historical default (chat + coder) EXCEPT
        # reasoning flows: action verbs are conversational surface
        # actions (open a panel, set an alarm, play media) — inside a
        # flow step they're noise at best and a surprise side effect at
        # worst. This one declaration keeps all ~80 registry verbs out
        # of the flow editor's tools grid and the flow executor's
        # category expansion. Explicit per-step tool_names pins still
        # resolve them.
        return SurfaceExposure(flow=False)

    @property
    def input_schema(self) -> dict:
        props = self._action.arg_schema or {}
        schema: dict[str, Any] = {
            "type": "object",
            "properties": props,
        }
        if self._action.required_args:
            schema["required"] = list(self._action.required_args)
        return schema

    @property
    def cacheable(self) -> bool:
        # Action primitives have side effects (DB writes, surface
        # emits). Caching them would be wrong — every call is fresh.
        return False

    async def execute(self, **kwargs) -> ToolResult:
        # TWO injection conventions reach this adapter and both must
        # work: chain.py passes ``_user_id`` top-level; passthrough's
        # ``_execute_tool`` (and therefore the shared native loop)
        # nests it inside ``_context``. Reading only the first ran
        # every loop-executed registry verb as ANONYMOUS — handlers
        # correctly refused with "signed-out" lines the moment voice
        # routed a real verb through the loop (2026-06-11).
        ctx_meta = kwargs.pop("_context", {}) or {}
        user_id = (
            kwargs.pop("_user_id", "")
            or str(ctx_meta.get("user_id") or "")
        )
        session_id = (
            ctx_meta.get("session_id")
            or kwargs.pop("_session_id", "")
            or ""
        )
        # Filter out remaining underscore-prefixed framework kwargs
        # before forwarding the rest to the handler as action args.
        action_args = {
            k: v for k, v in kwargs.items() if not k.startswith("_")
        }
        # Consent gate. The architect router tier-gates high-stakes intent on
        # the voice path, but a tool the MODEL calls directly reaches this
        # adapter with no such check — so without this, a future send/call/pay
        # verb would fire the instant the model decided to call it. Refuse
        # irreversible real-world actions here and hand the decision back to the
        # model to confirm with the user; the confirmed execution routes through
        # the architect's confirm-then-act path, which carries real consent.
        # Read/disruptive/costly verbs are unaffected (recoverable or gated by
        # speaker verification elsewhere). No high-stakes verb is registered
        # today, so this is a fail-safe foundation, not a behavior change.
        if self._action.stakes in _CONSENT_REQUIRED_STAKES:
            log.info(
                "action_tool_consent_gated",
                action=self._action.id, stakes=self._action.stakes,
                user_id=user_id,
            )
            return ToolResult(
                success=False,
                output=(
                    f"'{self._action.id}' has an irreversible real-world effect "
                    f"({self._action.stakes}) and must not be performed "
                    "automatically. Tell the user exactly what you're about to "
                    "do and ask them to confirm out loud or on screen first. "
                    "Do NOT claim it is done."
                ),
            )

        session = SessionContext(
            user_id=user_id,
            session_id=session_id,
            mode="passthrough",
            app_state=self._app_state,
        )
        try:
            result = await self._action.handler("", session, action_args)
        except Exception as exc:
            log.warning(
                "action_tool_handler_error",
                action=self._action.id, error=str(exc),
            )
            return ToolResult(
                success=False,
                error=f"Action {self._action.id} failed: {exc}",
            )

        if result is None:
            # The handler opted out at runtime (no active note, no
            # search hits, etc.). Treat as a no-op so the model knows
            # to try a different approach.
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"{self._action.id} could not run "
                    "(missing referent or precondition)"
                ),
            )

        # Compose the text the LLM sees from speak + toast — speak is
        # what Becca would say aloud, toast is the visible chip. Either
        # makes a fine summary for tool-call chaining.
        bits: list[str] = []
        if result.speak:
            bits.append(result.speak)
        if result.toast and result.toast not in (result.speak or ""):
            bits.append(f"[{result.toast}]")
        if result.prompt_addendum:
            # Soft-augmentation: include the addendum so the LLM can
            # reason against the recall hits / referent data.
            bits.append(result.prompt_addendum)
        output = "\n".join(b for b in bits if b) or "Done."

        metadata: dict[str, Any] = {}
        if result.surface_emit:
            ws_payload = {
                "type": "intent_action",
                "v": 1,
                "action": self._action.id,
                "tier": 3,
                "short_circuit": result.short_circuit,
                "surface": result.surface_emit,
            }
            if result.speak:
                ws_payload["speak"] = result.speak
            if result.toast:
                ws_payload["toast"] = result.toast

            # Stash on the per-session referent cache so the voice
            # route can drain + emit at the next turn boundary. The
            # chain layer doesn't have a WS handle, so this queue is
            # the bridge.
            if user_id:
                from augmentum.intent.dispatch import get_referent_cache
                refs = get_referent_cache(
                    self._app_state, user_id, session_id,
                )
                refs.pending_surface_events.append(ws_payload)
            # Keep the raw form in metadata too for any future
            # in-stream emitter (Phase 9.5+) that wants it.
            metadata["intent_action"] = ws_payload
        return ToolResult(
            success=True,
            output=output,
            metadata=metadata,
        )


def register_action_tools(registry: Any, app_state: Any) -> int:
    """Register every tier-3-eligible action in REGISTRY as a Tool.

    Idempotent — if the tool registry already has a tool with the same
    name, the existing one wins (handles in-place restarts).

    Returns the count of new tools registered.
    """
    if registry is None:
        log.warning("action_tools_skip_no_registry")
        return 0
    added = 0
    for action in REGISTRY.all():
        if not action.fanout.tier3:
            continue
        if registry.get(action.name if hasattr(action, "name") else action.id):
            # Already registered (e.g., container hot-reload).
            continue
        if registry.get(action.id):
            continue
        try:
            registry.register(ActionTool(action, app_state))
            added += 1
        except Exception as exc:
            log.warning(
                "action_tool_register_failed",
                id=action.id, error=str(exc),
            )
    log.info("action_tools_registered", count=added)
    return added
