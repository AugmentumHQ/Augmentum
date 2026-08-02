"""BeccaObserver — listens to the live system and keeps Becca aware.

She is not the primary chat responder. Chat flows through whichever
mode the user picked (narrative, utility, coder, etc.) exactly as if
she weren't there. This observer is how she keeps up: it subscribes
to the runtime bus, updates a small aggregate of "what's happening
right now" that other subsystems (initiative scorer, direct-address
path) can read, and writes a journal entry only when something is
genuinely worth remembering.

Sprint A scope:
- Subscribe to chat.turn_*, mode.changed, tool.*, focus.transition
- Maintain ``runtime.observed_state``: last_chat_mode, last_chat_at,
  last_tool, last_tool_at, last_mode_change, recent (deque)
- Journal mode changes (real shifts in user activity)
- Do NOT journal every chat turn — that's noise; the aggregate covers it

Lifecycle mirrors :class:`TickLoop` — start/stop owned by the runtime.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


_OBSERVED_TOPICS = frozenset({
    "chat.turn_started",
    "chat.turn_completed",
    "chat.moment_observed",      # Synapse Layer §1 — scored salient moment
    "mode.changed",
    "tool.invoked",
    "tool.completed",
    "focus.transition",
    # BeccaVoice's voice-turn completion (PTT Stage 3). Mirrors
    # chat.turn_completed semantics so initiative scoring + "what's
    # she been doing" reads see voice turns alongside chat turns.
    "voice.completed",
    "voice.turn_ended",          # Synapse Layer §3 — content-bearing voice moment
})

# Topic prefixes that match (in addition to the exact set above).
# Piece 8' — silent surfaces emit. Topics like `surface.browse.viewed`,
# `surface.media.played`, `surface.image.generated`, `surface.file.imported`
# all land in the recent deque so initiative + future consumers see
# cross-surface activity, not just chat. We don't add custom _handle
# branches for them: the recent deque captures them generically.
_OBSERVED_PREFIXES = ("surface.",)

_RECENT_MAX = 50


class BeccaObserver:
    """Per-runtime observer task. Owned by the runtime."""

    def __init__(self, runtime: CompanionRuntime) -> None:
        self.runtime = runtime
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._subscription = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop_event.clear()
        # Initialize the observed-state aggregate on the runtime so
        # other consumers (initiative, direct-address) have a stable
        # surface to read.
        self.runtime.observed_state = {
            "last_chat_mode": None,
            "last_chat_at": 0.0,
            "last_tool": None,
            "last_tool_at": 0.0,
            "last_mode_change": None,   # {"from": str|None, "to": str, "at": float}
            "recent": deque(maxlen=_RECENT_MAX),
        }
        self._subscription = await self.runtime.bus.subscribe(
            "**",
            slice_key="becca_observer",
        )
        self._task = asyncio.create_task(self._run(), name="becca_observer")
        log.info("becca_observer_started", companion_id=self.runtime.companion_id)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_event.set()
        if self._subscription is not None:
            try:
                await self.runtime.bus.unsubscribe(self._subscription)
            except Exception:
                log.debug("becca_observer_unsubscribe_failed", exc_info=True)
            self._subscription = None
        try:
            await asyncio.wait_for(self._task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._task.cancel()
        self._task = None
        log.info("becca_observer_stopped", companion_id=self.runtime.companion_id)

    async def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    event = await asyncio.wait_for(
                        self._subscription.queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                if event is None:
                    continue
                if (
                    event.topic not in _OBSERVED_TOPICS
                    and not event.topic.startswith(_OBSERVED_PREFIXES)
                ):
                    continue
                try:
                    await self._handle(event)
                except Exception:
                    log.warning(
                        "becca_observer_handle_failed",
                        topic=event.topic,
                        exc_info=True,
                    )
        except asyncio.CancelledError:
            pass

    async def _handle(self, event) -> None:
        topic = event.topic
        payload = event.payload or {}
        now = time.time()
        state = self.runtime.observed_state

        # Always record into the recent buffer first — cheap, useful for
        # context queries from the direct-address path later.
        state["recent"].append({
            "topic": topic,
            "payload": payload,
            "t": event.t,
        })

        if topic == "chat.turn_started":
            state["last_chat_mode"] = payload.get("mode")
            state["last_chat_at"] = now
        elif topic == "chat.turn_completed":
            state["last_chat_mode"] = payload.get("mode")
            state["last_chat_at"] = now
        elif topic == "mode.changed":
            state["last_mode_change"] = {
                "from": payload.get("from"),
                "to": payload.get("to"),
                "at": now,
            }
            # NOTE: We do NOT journal mode transitions. They're
            # operational signal (the recent[] deque + last_mode_change
            # already capture them); writing one journal entry per
            # mode shift produced 165+ templated noticings of the form
            # "user shifted from unknown to passthrough mode" — clogging
            # the notes pip with no interior content. Mode shifts now
            # live in observed_state only; the journal stays prose.
        elif topic == "tool.invoked":
            state["last_tool"] = payload.get("name")
            state["last_tool_at"] = now
        elif topic == "tool.completed":
            # Only update if it was the same tool (defensive — out-of-order
            # delivery is unlikely on an in-process bus but cheap to handle).
            if payload.get("name") == state["last_tool"]:
                state["last_tool_at"] = now
        elif topic == "focus.transition":
            # Her own focus moved. The state-machine already wrote this
            # to its own ledger; we mirror only for the aggregate view.
            state["recent"][-1]["payload"] = {
                **payload,
                "axis": "focus",
            }
        elif topic == "voice.completed":
            # A voice turn just finished. Surface it in the aggregate
            # view as a chat-equivalent so initiative scoring doesn't
            # treat the user as silent when they've been speaking. The
            # ``mode`` slot stays "voice" so consumers can distinguish.
            state["last_chat_mode"] = "voice"
            state["last_chat_at"] = now
        elif topic == "chat.moment_observed":
            # Synapse Layer §1 — a scored salient moment from the
            # chat path. Propagation policy (§5) decides what we do:
            #
            #   full           → journal the moment with affect tag
            #   affect_only    → journal a placeholder + affect tag
            #                    (content stripped at scorer)
            #   factual_only   → never gets here (scorer returns None)
            #   private        → never gets here (filtered upstream)
            #
            # The propagation lives on the event itself, not in the
            # payload — read it from event.propagation.
            await self._handle_chat_moment(event, payload)
        elif topic == "voice.turn_ended":
            # Synapse Layer §3 — *the kept thing*. A voice turn just
            # finished and emit_voice_turn_ended cleared the salience
            # bar. Journal it with source='voice_turn' so retrospection
            # can distinguish chat-derived moments from voice-derived
            # ones (voice carries different texture — short turns,
            # higher affect density, full presence).
            await self._handle_voice_turn(event, payload)

    async def _handle_chat_moment(self, event, payload: dict) -> None:
        """Journal a salient chat moment.

        Called from :meth:`_handle` for ``chat.moment_observed``. The
        scoring + threshold + propagation gating already happened
        upstream in :func:`emit_chat_turn_completed`; by the time we
        get here, the moment has cleared the bar. We just persist it.

        Surfacing policy: an affect-only moment is for Becca's interior
        (the scorer deliberately blanked the content), and a raw chat
        extract is fine to journal but not fine to push to the user
        pip — it reads as a quote, not a note. The pip only sees these
        once :mod:`augmentum.companion_runtime.bus` has run the LLM
        rewrite step. Until then, ``surfaceable_default=False`` keeps
        the interior record from also becoming a user-facing note.
        """
        from augmentum.companion_runtime.bus import PROP_AFFECT_ONLY, PROP_FULL

        propagation = getattr(event, "propagation", PROP_FULL)
        if propagation not in (PROP_FULL, PROP_AFFECT_ONLY):
            # Containment defense-in-depth: the scorer should not have
            # produced a moment for factual_only / private propagation,
            # but if a future change to the upstream ever did, we still
            # refuse to journal here.
            return

        moment_text = payload.get("moment") or ""
        affect = payload.get("user_affect") or "unclear"
        session_id = payload.get("session_id") or ""
        user_id = payload.get("user_id") or None
        salience_score = float(payload.get("salience") or 0.0)
        # When the bus has rewritten the moment via the utility tier
        # (companion_salience_llm_enabled + salience >= rewrite
        # threshold), it sets ``rewritten=True`` on the payload. Only
        # rewritten moments are surfaceable as user-facing notes;
        # affect-only stays interior regardless.
        rewritten = bool(payload.get("rewritten"))

        if not moment_text:
            return

        surfaceable = propagation == PROP_FULL and rewritten

        try:
            await self.runtime.memory.journal(
                content=moment_text,
                entry_type="conversation_moment",
                user_id=user_id,
                affect_tag=affect,
                content_refs=[{"kind": "chat_session", "id": session_id}] if session_id else None,
                confidence_numeric=min(0.95, 0.5 + salience_score * 0.5),
                source="observer_salience",
                surfaceable_default=surfaceable,
            )
        except Exception:
            log.warning("becca_observer_moment_journal_failed", exc_info=True)

        # Synapse Layer §2 — feed observed user affect. Containment is
        # already honored upstream (factual_only never emits a moment).
        # An update here is safe; the tracker handles unknown tags by
        # mapping to "unclear" / neutral.
        await self._echo_user_affect(user_id=user_id, tag=affect, source="chat")

    async def _handle_voice_turn(self, event, payload: dict) -> None:
        """Journal a salient voice turn.

        Voice turns are the channel where Becca is most herself —
        composes her own prompt, streams her own words, expresses
        through her own pipeline. Until Synapse Layer §3, those
        turns evaporated when the WebSocket closed; now they land
        as ``conversation_moment`` entries with ``source='voice_turn'``
        so the consolidation pipeline (Synapse §4) and the dream
        engine can both draw on her own voice as raw material.
        """
        from augmentum.companion_runtime.bus import (
            PROP_AFFECT_ONLY, PROP_FULL,
        )

        propagation = getattr(event, "propagation", PROP_FULL)
        if propagation not in (PROP_FULL, PROP_AFFECT_ONLY):
            return  # containment defense-in-depth

        moment_text = payload.get("moment") or ""
        affect = payload.get("user_affect") or "engaged"
        session_id = payload.get("session_id") or ""
        invocation_id = payload.get("invocation_id") or ""
        user_id = payload.get("user_id") or None
        salience_score = float(payload.get("salience") or 0.0)
        assistant_excerpt = payload.get("assistant_excerpt") or ""
        rewritten = bool(payload.get("rewritten"))

        if not moment_text:
            return

        # Voice turns ground in TWO refs: the voice session (for
        # device/place recall) and the invocation_id (for tracing
        # back to the exact turn). The resolver can rehydrate either.
        refs: list[dict] = []
        if session_id:
            refs.append({"kind": "voice_session", "id": session_id})
        if invocation_id:
            refs.append({"kind": "voice_invocation", "id": invocation_id})

        # Compose the journal content. Excerpt of her own words
        # gives the consolidation pipeline her actual voice to learn
        # from, not just a third-person summary.
        if assistant_excerpt:
            entry = f"{moment_text}\n\n— and I said: {assistant_excerpt}"
        else:
            entry = moment_text

        surfaceable = propagation == PROP_FULL and rewritten

        try:
            await self.runtime.memory.journal(
                content=entry,
                entry_type="conversation_moment",
                user_id=user_id,
                affect_tag=affect,
                content_refs=refs or None,
                confidence_numeric=min(0.95, 0.6 + salience_score * 0.4),
                source="voice_turn",
                surfaceable_default=surfaceable,
            )
        except Exception:
            log.warning("becca_observer_voice_journal_failed", exc_info=True)

        # Synapse Layer §2 — feed observed user affect from voice. Voice
        # turns get extra weight: an in-person conversation is a
        # higher-signal read on his state than text, and the affect
        # tag here may be Becca's *published* affect (read from her
        # PAD at speak time) which she's the authority on.
        await self._echo_user_affect(user_id=user_id, tag=affect, source="voice")

    async def _echo_user_affect(
        self,
        *,
        user_id: str | None,
        tag: str,
        source: str,
    ) -> None:
        """Update the runtime's user-affect tracker + emit a bus event.

        Synapse Layer §2 wiring. The bus event lets downstream
        consumers (presence widget, future avatar emotion bridge,
        Observatory) react to the read without polling. Best-effort —
        failures don't propagate up to the journal write path.
        """
        if not user_id:
            return
        tracker = getattr(self.runtime, "user_affect", None)
        if tracker is None:
            return
        try:
            observation = tracker.update(user_id, tag)
        except Exception:
            log.debug("user_affect_update_failed", exc_info=True)
            return
        try:
            await self.runtime.bus.publish_topic(
                "user.affect_observed",
                {
                    "user_id": user_id,
                    "tag": observation.tag,
                    "valence": observation.valence,
                    "arousal": observation.arousal,
                    "dominance": observation.dominance,
                    "source": source,
                    "sample_count": observation.sample_count,
                },
                source_companion_id=self.runtime.companion_id,
            )
        except Exception:
            log.warning("user_affect_observed_publish_failed", exc_info=True)
