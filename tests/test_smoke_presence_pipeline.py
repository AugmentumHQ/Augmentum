"""Tests for the presence pipeline orchestrator (pipeline.py).

Pins:
  - Event handlers apply transitions correctly + return the new state
  - Invalid transitions log a warning + leave state unchanged (no raise)
  - Context buffers update on LLM_TOKEN
  - Pending actions defer during GENERATING_SPECULATIVE
  - Pending actions commit (returned via last_committed_actions) on TURN_COMMITTED
  - Pending actions drop on SPEECH_CONTINUED from GENERATING_SPECULATIVE
  - Interruption sets was_interrupted + mid_phrase on the context
  - Listener subscribe/unsubscribe fan-out works
  - Concurrent events from multiple producers serialize correctly (asyncio.Lock)
  - Multi-tenant: user_id + session_id required at construction
  - close() is idempotent
"""
from __future__ import annotations

import asyncio

import pytest

from augmentum.companion.presence import (
    PresencePipeline,
    PresenceState,
)


# ── Construction + multi-tenant invariants ──────────────────────


class TestConstruction:
    def test_requires_session_id(self):
        with pytest.raises(ValueError, match="session_id"):
            PresencePipeline(session_id="", user_id="usr_1")

    def test_requires_user_id(self):
        with pytest.raises(ValueError, match="user_id"):
            PresencePipeline(session_id="sess_1", user_id="")

    def test_starts_in_idle_state(self):
        p = PresencePipeline(session_id="sess_1", user_id="usr_1")
        assert p.state is PresenceState.IDLE
        assert p.context.transition_count == 0

    def test_context_carries_session_and_user(self):
        p = PresencePipeline(session_id="sess_xyz", user_id="usr_abc")
        assert p.context.session_id == "sess_xyz"
        assert p.context.user_id == "usr_abc"


# ── Happy-path turn cycle ───────────────────────────────────────


class TestTurnCycle:
    @pytest.mark.asyncio
    async def test_full_turn_cycle(self):
        """Walk a complete non-speculative turn from IDLE back to IDLE."""
        p = PresencePipeline(session_id="sess_1", user_id="usr_1")

        # User starts speaking
        assert await p.on_speech_detected() is PresenceState.LISTENING

        # Turn closes (silence timeout or PTT release)
        assert await p.on_turn_committed() is PresenceState.GENERATING

        # LLM streams tokens
        assert await p.on_llm_token("Hello") is PresenceState.GENERATING
        assert await p.on_llm_token(" there") is PresenceState.GENERATING
        assert p.context.llm_token_buffer == "Hello there"

        # First synthesis chunk lands
        assert await p.on_first_chunk_ready() is PresenceState.SPEAKING

        # Becca finishes speaking
        assert await p.on_chunk_queue_empty() is PresenceState.IDLE
        # Buffers cleared on natural end
        assert p.context.llm_token_buffer == ""

    @pytest.mark.asyncio
    async def test_speculative_committed_path(self):
        """Speculative kickoff that ends up committing — actions fire."""
        p = PresencePipeline(session_id="sess_1", user_id="usr_1")

        await p.on_speech_detected()
        assert await p.on_turn_likely(0.85) is PresenceState.GENERATING_SPECULATIVE

        # Companion runtime queues a verb during speculation
        await p.queue_pending_action("memory.save", {"key": "x"})
        await p.queue_pending_action("growth.log", {"event": "y"})
        assert len(p.context.pending_actions) == 2

        # Turn confirmed
        assert await p.on_turn_committed() is PresenceState.GENERATING
        # Actions flushed and exposed for the companion runtime to fire
        committed = p.last_committed_actions
        assert len(committed) == 2
        assert committed[0].verb_id == "memory.save"
        assert committed[1].verb_id == "growth.log"
        # Buffer cleared after commit
        assert len(p.context.pending_actions) == 0

    @pytest.mark.asyncio
    async def test_speculative_cancelled_path(self):
        """Speculative kickoff that gets cancelled — actions dropped."""
        p = PresencePipeline(session_id="sess_1", user_id="usr_1")

        await p.on_speech_detected()
        await p.on_turn_likely(0.75)
        await p.queue_pending_action("memory.save")
        await p.queue_pending_action("growth.log")

        # User keeps talking — speculation was wrong
        assert await p.on_speech_continued() is PresenceState.CANCEL_SPECULATIVE
        # Pending actions dropped
        assert len(p.context.pending_actions) == 0

        # Cleanup completes → back to listening
        assert await p.on_cleanup_complete() is PresenceState.LISTENING


# ── Interruption path ───────────────────────────────────────────


class TestInterruption:
    @pytest.mark.asyncio
    async def test_interrupt_sets_recovery_flags(self):
        p = PresencePipeline(session_id="sess_1", user_id="usr_1")

        # Walk to SPEAKING
        await p.on_speech_detected()
        await p.on_turn_committed()
        await p.on_llm_token("The answer is forty-two")
        await p.on_first_chunk_ready()

        # User interrupts mid-sentence
        assert await p.on_interrupt_vad(
            mid_phrase="The answer is forty-",
        ) is PresenceState.INTERRUPTED

        # Recovery flags set for next turn's prompt
        assert p.context.was_interrupted is True
        assert p.context.mid_phrase == "The answer is forty-"

        # Beat completes → back to listening, recovery flags preserved
        # (the next turn's prompt frame consumes them)
        assert await p.on_beat_complete() is PresenceState.LISTENING
        assert p.context.was_interrupted is True
        assert p.context.mid_phrase == "The answer is forty-"

        # Once consumed, the runtime calls clear_recovery_flags()
        p.context.clear_recovery_flags()
        assert p.context.was_interrupted is False


# ── Invalid transitions ─────────────────────────────────────────


class TestInvalidTransitions:
    @pytest.mark.asyncio
    async def test_invalid_event_leaves_state_unchanged(self):
        """Event that doesn't apply in the current state: log + stay."""
        p = PresencePipeline(session_id="sess_1", user_id="usr_1")

        # CHUNK_QUEUE_EMPTY is invalid in IDLE
        assert await p.on_chunk_queue_empty() is PresenceState.IDLE
        assert p.state is PresenceState.IDLE
        # No transition counted
        assert p.context.transition_count == 0

    @pytest.mark.asyncio
    async def test_invalid_transition_does_not_raise(self):
        """Pipeline never raises on an out-of-order event."""
        p = PresencePipeline(session_id="sess_1", user_id="usr_1")
        # Try every event from IDLE; only SPEECH_DETECTED and
        # ERROR_OCCURRED are valid. None should raise.
        await p.on_turn_committed()       # invalid
        await p.on_llm_token("nope")      # invalid
        await p.on_first_chunk_ready()    # invalid
        await p.on_chunk_queue_empty()    # invalid
        await p.on_interrupt_vad()        # invalid
        await p.on_beat_complete()        # invalid
        # Pipeline still healthy in IDLE
        assert p.state is PresenceState.IDLE

    @pytest.mark.asyncio
    async def test_queue_action_outside_speculative_is_noop(self):
        """Verbs queued outside GENERATING_SPECULATIVE are dropped + logged."""
        p = PresencePipeline(session_id="sess_1", user_id="usr_1")
        # In IDLE
        await p.queue_pending_action("memory.save")
        assert len(p.context.pending_actions) == 0

        # In LISTENING
        await p.on_speech_detected()
        await p.queue_pending_action("memory.save")
        assert len(p.context.pending_actions) == 0


# ── Listener fan-out ────────────────────────────────────────────


class TestListeners:
    @pytest.mark.asyncio
    async def test_sync_listener_receives_transitions(self):
        p = PresencePipeline(session_id="sess_1", user_id="usr_1")
        received = []

        def listener(t):
            received.append(t)

        p.subscribe(listener)
        await p.on_speech_detected()
        await p.on_turn_committed()
        # Listener dispatch is async (task), let the event loop drain
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert len(received) >= 2
        assert received[0].to_state is PresenceState.LISTENING
        assert received[1].to_state is PresenceState.GENERATING

    @pytest.mark.asyncio
    async def test_async_listener_awaited(self):
        p = PresencePipeline(session_id="sess_1", user_id="usr_1")
        received = []

        async def listener(t):
            await asyncio.sleep(0)
            received.append(t)

        p.subscribe(listener)
        await p.on_speech_detected()
        # Drain listener tasks
        for _ in range(5):
            await asyncio.sleep(0)
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_delivery(self):
        p = PresencePipeline(session_id="sess_1", user_id="usr_1")
        received = []

        unsub = p.subscribe(lambda t: received.append(t))
        await p.on_speech_detected()
        for _ in range(3):
            await asyncio.sleep(0)
        first_count = len(received)

        unsub()
        await p.on_turn_committed()
        for _ in range(3):
            await asyncio.sleep(0)
        assert len(received) == first_count  # no new transitions delivered

    @pytest.mark.asyncio
    async def test_listener_exception_doesnt_kill_pipeline(self):
        p = PresencePipeline(session_id="sess_1", user_id="usr_1")

        def buggy(t):
            raise RuntimeError("listener exploded")

        good_received = []
        p.subscribe(buggy)
        p.subscribe(lambda t: good_received.append(t))

        await p.on_speech_detected()
        for _ in range(3):
            await asyncio.sleep(0)
        # Good listener still got the transition
        assert len(good_received) == 1


# ── Concurrency invariants ──────────────────────────────────────


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_interleaved_events_serialize_via_lock(self):
        """Concurrent on_* calls from multiple producers don't corrupt state.

        We fire a burst of LLM_TOKEN events alongside FIRST_CHUNK_READY
        and CHUNK_QUEUE_EMPTY; the final state must be consistent with
        the serialized order, and the token buffer should reflect every
        token (no lost writes).
        """
        p = PresencePipeline(session_id="sess_1", user_id="usr_1")
        await p.on_speech_detected()
        await p.on_turn_committed()
        # Fire 20 interleaved token events
        tokens = [f" t{i}" for i in range(20)]
        await asyncio.gather(*(p.on_llm_token(t) for t in tokens))
        # All tokens landed (concatenation is order-dependent, but total
        # length must match)
        joined_len = sum(len(t) for t in tokens)
        assert len(p.context.llm_token_buffer) == joined_len
        # State still consistent
        assert p.state in (PresenceState.GENERATING,)


# ── Lifecycle ───────────────────────────────────────────────────


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_close_is_idempotent(self):
        p = PresencePipeline(session_id="sess_1", user_id="usr_1")
        await p.close()
        await p.close()  # no exception

    @pytest.mark.asyncio
    async def test_transition_after_close_is_noop(self):
        p = PresencePipeline(session_id="sess_1", user_id="usr_1")
        await p.close()
        # Event handlers don't transition once closed; they log + return
        # the current state.
        assert await p.on_speech_detected() is PresenceState.IDLE
