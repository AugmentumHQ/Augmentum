"""Tests for the presence state model (state.py).

Pins:
  - Every (state, event) pair in VALID_TRANSITIONS resolves to a defined state
  - Invalid pairs return None from next_state (no exception)
  - PendingAction queue/commit/drop primitives work
  - PresenceContext recovery flag lifecycle (set on interrupt, cleared on next-turn consumption)
  - StateTransition is frozen (frozen=True dataclass invariant)
  - ERROR_OCCURRED is reachable from every non-ERROR state
"""
from __future__ import annotations

import time

import pytest

from augmentum.companion.presence.state import (
    PendingAction,
    PresenceContext,
    PresenceEvent,
    PresenceState,
    StateTransition,
    VALID_TRANSITIONS,
    is_valid_transition,
    next_state,
)


# ── Transition table integrity ───────────────────────────────────


class TestTransitionTable:
    def test_every_table_entry_resolves(self):
        """Every (state, event) pair in the table has a defined to_state."""
        for (from_state, event), to_state in VALID_TRANSITIONS.items():
            assert isinstance(from_state, PresenceState)
            assert isinstance(event, PresenceEvent)
            assert isinstance(to_state, PresenceState)

    def test_next_state_matches_table(self):
        """next_state() returns the table value for valid pairs."""
        for (from_state, event), expected in VALID_TRANSITIONS.items():
            assert next_state(from_state, event) == expected

    def test_invalid_pair_returns_none(self):
        """next_state() returns None for any pair not in the table."""
        # IDLE doesn't accept LLM_TOKEN
        assert next_state(PresenceState.IDLE, PresenceEvent.LLM_TOKEN) is None
        # IDLE doesn't accept CHUNK_QUEUE_EMPTY
        assert next_state(
            PresenceState.IDLE, PresenceEvent.CHUNK_QUEUE_EMPTY,
        ) is None
        # SPEAKING doesn't accept TURN_LIKELY
        assert next_state(
            PresenceState.SPEAKING, PresenceEvent.TURN_LIKELY,
        ) is None

    def test_is_valid_transition_matches_next_state(self):
        """is_valid_transition is equivalent to (next_state is not None)."""
        for from_state in PresenceState:
            for event in PresenceEvent:
                assert is_valid_transition(from_state, event) == (
                    next_state(from_state, event) is not None
                )

    def test_error_reachable_from_every_non_error_state(self):
        """ERROR_OCCURRED transitions every non-ERROR state to ERROR."""
        for state in PresenceState:
            if state is PresenceState.ERROR:
                # ERROR state doesn't accept ERROR_OCCURRED again
                continue
            assert next_state(
                state, PresenceEvent.ERROR_OCCURRED,
            ) is PresenceState.ERROR

    def test_error_only_recovers_via_recovered_event(self):
        """ERROR state transitions only on RECOVERED."""
        for event in PresenceEvent:
            result = next_state(PresenceState.ERROR, event)
            if event is PresenceEvent.RECOVERED:
                assert result is PresenceState.IDLE
            else:
                assert result is None


class TestSpecificTransitions:
    """Tier-1 transitions from the design doc — must hold exactly."""

    def test_idle_to_listening_on_speech(self):
        assert next_state(
            PresenceState.IDLE, PresenceEvent.SPEECH_DETECTED,
        ) is PresenceState.LISTENING

    def test_listening_to_speculative_on_turn_likely(self):
        assert next_state(
            PresenceState.LISTENING, PresenceEvent.TURN_LIKELY,
        ) is PresenceState.GENERATING_SPECULATIVE

    def test_listening_to_generating_on_committed(self):
        assert next_state(
            PresenceState.LISTENING, PresenceEvent.TURN_COMMITTED,
        ) is PresenceState.GENERATING

    def test_speculative_to_cancel_on_continued_speech(self):
        assert next_state(
            PresenceState.GENERATING_SPECULATIVE,
            PresenceEvent.SPEECH_CONTINUED,
        ) is PresenceState.CANCEL_SPECULATIVE

    def test_speculative_to_generating_on_commit(self):
        assert next_state(
            PresenceState.GENERATING_SPECULATIVE,
            PresenceEvent.TURN_COMMITTED,
        ) is PresenceState.GENERATING

    def test_generating_to_speaking_on_first_chunk(self):
        assert next_state(
            PresenceState.GENERATING,
            PresenceEvent.FIRST_CHUNK_READY,
        ) is PresenceState.SPEAKING

    def test_speaking_to_idle_on_queue_empty(self):
        assert next_state(
            PresenceState.SPEAKING,
            PresenceEvent.CHUNK_QUEUE_EMPTY,
        ) is PresenceState.IDLE

    def test_speaking_to_interrupted_on_interrupt(self):
        assert next_state(
            PresenceState.SPEAKING,
            PresenceEvent.INTERRUPT_VAD,
        ) is PresenceState.INTERRUPTED

    def test_interrupted_to_listening_after_beat(self):
        assert next_state(
            PresenceState.INTERRUPTED,
            PresenceEvent.BEAT_COMPLETE,
        ) is PresenceState.LISTENING

    def test_speaking_user_backchannel_self_loop(self):
        """User backchannel during speaking → keep speaking."""
        assert next_state(
            PresenceState.SPEAKING,
            PresenceEvent.USER_BACKCHANNEL_DETECTED,
        ) is PresenceState.SPEAKING

    def test_listening_speech_continued_self_loop(self):
        assert next_state(
            PresenceState.LISTENING,
            PresenceEvent.SPEECH_CONTINUED,
        ) is PresenceState.LISTENING


# ── Context behaviors ────────────────────────────────────────────


class TestPresenceContext:
    def _ctx(self) -> PresenceContext:
        return PresenceContext(session_id="sess_1", user_id="usr_1")

    def test_pending_actions_commit_and_clear(self):
        ctx = self._ctx()
        ctx.pending_actions.append(PendingAction(verb_id="memory.save"))
        ctx.pending_actions.append(PendingAction(verb_id="growth.log"))
        committed = ctx.commit_pending_actions()
        assert len(committed) == 2
        assert committed[0].verb_id == "memory.save"
        assert committed[1].verb_id == "growth.log"
        assert len(ctx.pending_actions) == 0

    def test_pending_actions_drop_returns_count(self):
        ctx = self._ctx()
        ctx.pending_actions.append(PendingAction(verb_id="memory.save"))
        ctx.pending_actions.append(PendingAction(verb_id="growth.log"))
        ctx.pending_actions.append(PendingAction(verb_id="offer.suggest"))
        dropped = ctx.drop_pending_actions()
        assert dropped == 3
        assert len(ctx.pending_actions) == 0

    def test_reset_turn_buffers_preserves_recovery_flags(self):
        """Turn buffers reset on chunk_queue_empty must NOT clear was_interrupted /
        mid_phrase — those belong to the NEXT turn's prompt frame."""
        ctx = self._ctx()
        ctx.partial_transcript = "what do you"
        ctx.llm_token_buffer = "I was saying"
        ctx.was_interrupted = True
        ctx.mid_phrase = "the answer is"
        ctx.reset_turn_buffers()
        assert ctx.partial_transcript == ""
        assert ctx.llm_token_buffer == ""
        assert ctx.was_interrupted is True  # preserved
        assert ctx.mid_phrase == "the answer is"  # preserved

    def test_clear_recovery_flags(self):
        ctx = self._ctx()
        ctx.was_interrupted = True
        ctx.mid_phrase = "I was saying"
        ctx.clear_recovery_flags()
        assert ctx.was_interrupted is False
        assert ctx.mid_phrase == ""


class TestStateTransitionImmutable:
    def test_state_transition_is_frozen(self):
        t = StateTransition(
            from_state=PresenceState.IDLE,
            to_state=PresenceState.LISTENING,
            event=PresenceEvent.SPEECH_DETECTED,
            timestamp=time.monotonic(),
            session_id="sess_1",
            user_id="usr_1",
        )
        with pytest.raises(AttributeError):
            t.to_state = PresenceState.ERROR  # type: ignore[misc]


# ── Enum serialization ───────────────────────────────────────────


class TestEnumSerialization:
    """str-subclass enums must serialize to JSON cleanly without a custom encoder."""

    def test_state_values_are_strings(self):
        import json
        encoded = json.dumps({"state": PresenceState.SPEAKING.value})
        assert json.loads(encoded) == {"state": "speaking"}

    def test_event_values_are_strings(self):
        import json
        encoded = json.dumps({"event": PresenceEvent.INTERRUPT_VAD.value})
        assert json.loads(encoded) == {"event": "interrupt_vad"}

    def test_state_enum_str_value_match(self):
        """PresenceState members are str-subclass and equal their .value."""
        assert PresenceState.IDLE == "idle"
        assert PresenceState.LISTENING == "listening"
        assert PresenceState.SPEAKING == "speaking"
