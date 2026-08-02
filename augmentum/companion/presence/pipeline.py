"""PresencePipeline — orchestrator + state machine.

Owns the in-memory state for one (user_id, session_id) conversation.
Receives events from the various producers (VAD, Smart Turn, ASR, LLM,
TTS, audio bus) and applies transitions through the data-driven table
in state.py.

Concurrency model:
  Events arrive from multiple coroutines (WS message handler, VAD
  callback, LLM streaming iterator, TTS playback callback). To prevent
  interleaved transitions from corrupting state, every event handler
  acquires an asyncio.Lock before reading + writing state. Critical
  sections are tight (single transition + buffer update) so contention
  stays low.

Side-effect policy:
  Per decision D3, verbs invoked during GENERATING_SPECULATIVE are
  buffered into context.pending_actions instead of firing immediately.
  On TURN_COMMITTED they fire in order; on CANCEL_SPECULATIVE they're
  dropped. Phase 1 sets up the buffer/flush/drop primitives; the verb
  dispatcher integration arrives in Phase 4 alongside speculative
  generation.

Invalid transitions:
  When an event arrives in a state with no defined transition (e.g.
  CHUNK_QUEUE_EMPTY while LISTENING), we log a warning at INFO level
  and remain in the current state. We do NOT raise — production
  pipelines shouldn't crash a session on an unexpected event order;
  log + recover is the right policy.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from augmentum.companion.presence.state import (
    PendingAction,
    PresenceContext,
    PresenceEvent,
    PresenceState,
    StateTransition,
    next_state,
)
from augmentum.utils.bg_tasks import track
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


Listener = Callable[[StateTransition], Awaitable[None] | None]


class PresencePipeline:
    """Stateful conversation orchestrator for the companion presence path.

    One instance per (user_id, session_id). Lifecycle:
      pipeline = PresencePipeline(session_id=..., user_id=...)
      await pipeline.on_speech_detected()      # IDLE -> LISTENING
      await pipeline.on_turn_committed()       # LISTENING -> GENERATING
      ...
      await pipeline.close()                   # release resources
    """

    def __init__(self, *, session_id: str, user_id: str) -> None:
        if not session_id:
            raise ValueError("PresencePipeline requires session_id")
        if not user_id:
            raise ValueError("PresencePipeline requires user_id")
        now = time.monotonic()
        self._state: PresenceState = PresenceState.IDLE
        self._context = PresenceContext(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            state_entered_at=now,
        )
        self._lock = asyncio.Lock()
        self._listeners: list[Listener] = []
        self._closed = False

    # ── Public state inspection ──────────────────────────────────

    @property
    def state(self) -> PresenceState:
        """Current state. Safe to read without lock — single-word atomic."""
        return self._state

    @property
    def context(self) -> PresenceContext:
        """Per-conversation mutable context. Callers should NOT mutate."""
        return self._context

    # ── Event handlers (one per event type) ──────────────────────
    #
    # Each handler:
    #   1. Acquires the lock
    #   2. Updates context payload-specific fields (e.g. buffers a token)
    #   3. Resolves + applies the transition via _transition
    #   4. Releases the lock
    #
    # Returns the post-transition state for caller convenience (most
    # callers don't need it but tests do).

    async def on_speech_detected(self) -> PresenceState:
        async with self._lock:
            return await self._transition(PresenceEvent.SPEECH_DETECTED)

    async def on_turn_likely(self, confidence: float) -> PresenceState:
        async with self._lock:
            # Confidence is informational — the transition is the same
            # for any "turn_likely" signal. Log for diagnostics.
            log.debug(
                "presence_turn_likely",
                session_id=self._context.session_id,
                confidence=round(float(confidence), 3),
                state=self._state.value,
            )
            return await self._transition(PresenceEvent.TURN_LIKELY)

    async def on_turn_committed(self) -> PresenceState:
        async with self._lock:
            # On commit, fire any pending actions queued speculatively.
            # We return the list for the caller (typically the companion
            # runtime) to actually invoke against the verb dispatcher;
            # the pipeline itself doesn't import the dispatcher (would
            # be a layering inversion).
            committed_actions = self._context.commit_pending_actions()
            self._last_committed_actions = committed_actions  # for caller probe
            return await self._transition(PresenceEvent.TURN_COMMITTED)

    async def on_speech_continued(self) -> PresenceState:
        async with self._lock:
            # Speech continued mid-speculative-generation → cancel.
            # The orchestrator's CANCEL_SPECULATIVE entry handler drops
            # pending actions (the LLM may have queued some by now).
            if self._state is PresenceState.GENERATING_SPECULATIVE:
                dropped = self._context.drop_pending_actions()
                if dropped:
                    log.info(
                        "presence_dropped_speculative_actions",
                        session_id=self._context.session_id,
                        dropped_count=dropped,
                    )
            return await self._transition(PresenceEvent.SPEECH_CONTINUED)

    async def on_llm_token(self, token: str) -> PresenceState:
        async with self._lock:
            self._context.llm_token_buffer += token
            return await self._transition(PresenceEvent.LLM_TOKEN)

    async def on_first_chunk_ready(self) -> PresenceState:
        async with self._lock:
            return await self._transition(PresenceEvent.FIRST_CHUNK_READY)

    async def on_chunk_queue_empty(self) -> PresenceState:
        async with self._lock:
            # Turn naturally ended — reset per-turn buffers. Recovery
            # flags (was_interrupted / mid_phrase) are preserved here
            # since by definition we weren't interrupted on this path.
            self._context.reset_turn_buffers()
            return await self._transition(PresenceEvent.CHUNK_QUEUE_EMPTY)

    async def on_interrupt_vad(self, *, mid_phrase: str = "") -> PresenceState:
        async with self._lock:
            # Capture what Becca was mid-saying for next-turn recovery.
            # The mid_phrase is the concatenation of played chunks + the
            # currently-fading chunk; the caller computes this from the
            # audio bus state and passes it in.
            self._context.was_interrupted = True
            self._context.mid_phrase = mid_phrase
            return await self._transition(PresenceEvent.INTERRUPT_VAD)

    async def on_user_backchannel_detected(self) -> PresenceState:
        async with self._lock:
            # User said "mhm" / "yeah" / etc. while Becca was speaking —
            # we keep speaking. This is the explicit self-loop. Logging
            # so we can analyze how often this fires.
            log.debug(
                "presence_user_backchannel",
                session_id=self._context.session_id,
            )
            return await self._transition(
                PresenceEvent.USER_BACKCHANNEL_DETECTED,
            )

    async def on_beat_complete(self) -> PresenceState:
        async with self._lock:
            return await self._transition(PresenceEvent.BEAT_COMPLETE)

    async def on_cleanup_complete(self) -> PresenceState:
        async with self._lock:
            # Speculative cancel cleanup finished — reset buffers and
            # return to listening. Recovery flags stay clear here
            # (we weren't interrupted, just cancelled mid-thought).
            self._context.reset_turn_buffers()
            return await self._transition(PresenceEvent.CLEANUP_COMPLETE)

    async def on_error(self, exc: Exception) -> PresenceState:
        async with self._lock:
            log.warning(
                "presence_error",
                session_id=self._context.session_id,
                state=self._state.value,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return await self._transition(PresenceEvent.ERROR_OCCURRED)

    async def on_recovered(self) -> PresenceState:
        async with self._lock:
            self._context.reset_turn_buffers()
            return await self._transition(PresenceEvent.RECOVERED)

    # ── Speculative-action buffer (called by companion runtime in P4) ──

    async def queue_pending_action(
        self, verb_id: str, args: dict[str, Any] | None = None,
    ) -> None:
        """Queue a side-effect verb for deferred dispatch.

        Called by the companion runtime when the LLM (during speculative
        generation) invokes a verb. The verb's side effects (memory writes,
        growth events, observation logs) are deferred until TURN_COMMITTED;
        if the user keeps talking (CANCEL_SPECULATIVE), the queue is dropped.

        Outside GENERATING_SPECULATIVE this is a no-op WITH a warning —
        the verb dispatcher shouldn't be invoking deferred-action queuing
        from any other state.
        """
        async with self._lock:
            if self._state is not PresenceState.GENERATING_SPECULATIVE:
                log.warning(
                    "presence_queue_action_wrong_state",
                    session_id=self._context.session_id,
                    state=self._state.value,
                    verb_id=verb_id,
                )
                return
            self._context.pending_actions.append(
                PendingAction(
                    verb_id=verb_id,
                    args=dict(args or {}),
                    queued_at=time.monotonic(),
                ),
            )

    @property
    def last_committed_actions(self) -> list[PendingAction]:
        """Actions committed on the most recent TURN_COMMITTED transition.

        Cleared next time TURN_COMMITTED fires. The companion runtime
        polls this after awaiting on_turn_committed() to know which
        verbs to actually dispatch.
        """
        return list(getattr(self, "_last_committed_actions", []))

    # ── Listener subscription ────────────────────────────────────

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Subscribe to state transitions. Returns an unsubscribe callable.

        Listeners may be sync or async; the orchestrator awaits async
        listeners but does NOT hold the lock while awaiting (so a slow
        listener doesn't block subsequent transitions). Tradeoff: a
        listener observing transition N may briefly see state(N+1) when
        it reads pipeline.state — listeners that need consistency should
        use the StateTransition's from_state / to_state fields, not
        re-read the live pipeline state.
        """
        self._listeners.append(listener)

        def _unsubscribe() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return _unsubscribe

    # ── Lifecycle ────────────────────────────────────────────────

    async def close(self) -> None:
        """Release resources. Safe to call multiple times."""
        self._closed = True
        self._listeners.clear()

    # ── Internals ────────────────────────────────────────────────

    async def _transition(self, event: PresenceEvent) -> PresenceState:
        """Resolve + apply a transition. Caller holds self._lock.

        Returns the new state. Invalid transitions log + remain in
        current state (no exception).
        """
        if self._closed:
            log.warning(
                "presence_transition_after_close",
                session_id=self._context.session_id,
                presence_event=event.value,
            )
            return self._state

        from_state = self._state
        to_state = next_state(from_state, event)
        if to_state is None:
            log.info(
                "presence_invalid_transition",
                session_id=self._context.session_id,
                from_state=from_state.value,
                presence_event=event.value,
            )
            return self._state

        now = time.monotonic()
        self._state = to_state
        self._context.state_entered_at = now
        self._context.transition_count += 1
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            event=event,
            timestamp=now,
            session_id=self._context.session_id,
            user_id=self._context.user_id,
        )
        # Drop the lock before notifying listeners so a slow listener
        # doesn't block subsequent transitions. The listener loop is
        # outside the locked section.
        listeners_snapshot = list(self._listeners)
        # Schedule listener dispatch but don't await it inside the lock.
        # We snapshot the list because subscribe/unsubscribe may run
        # concurrently with the dispatch.
        track(_dispatch_listeners(listeners_snapshot, transition),
              name="presence_dispatch_listeners")
        return self._state


async def _dispatch_listeners(
    listeners: list[Listener], transition: StateTransition,
) -> None:
    """Fan out a transition to listeners outside the orchestrator lock.

    Listeners are best-effort: a listener exception is logged but does
    NOT propagate (one buggy listener shouldn't sink the others or the
    pipeline). Async listeners are awaited; sync are called directly.
    """
    for listener in listeners:
        try:
            result = listener(transition)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            log.warning(
                "presence_listener_error",
                session_id=transition.session_id,
                presence_event=transition.event.value,
                error=str(exc),
                error_type=type(exc).__name__,
            )
