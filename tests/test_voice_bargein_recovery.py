"""Tests for false-barge-in recovery and explicit-capture addressing.

Pins the four failure modes from the 2026-06-10 voice session where six
utterances produced zero audible replies:

1. Barge-in confirmation measured wall-clock since speech_start, but
   ``vad.is_speaking`` stays True through TRAILING — a ~100ms blip plus
   trailing silence confirmed an interrupt that the segment-end check
   then discarded (``voiced_ms`` is the fix).
2. The interrupt rollback raced the turn's ``finally`` block
   (``is_speaking`` already False when speech_discard arrived), so the
   stale ``interrupted`` flag silently drained every TTS sentence while
   the full text was committed to history — she "remembered" a reply
   the user never heard (``_replay_undelivered_tts`` is the fix).
3. ``goal=idle`` from the voice router hard-vetoed addressing even for
   a deliberate PTT "Good morning." (prompt narrowed + explicit-capture
   promotion is the fix; prompt text pinned here).
4. Unaddressed/empty exits sent no terminal WS message, leaving the
   widget in its thinking pulse forever (covered by route-level sends;
   the VoiceSession provenance fields are pinned here).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import augmentum.proxy.voice_routes as voice_routes
from augmentum.architect.voice_router import _build_system_prompt
from augmentum.voice.pipeline import VoiceSession
from augmentum.voice.vad import VadProcessor, VadState

# ── VAD voiced-span measurement ───────────────────────────────────────


class TestVoicedMs:
    def test_idle_reports_zero(self):
        vad = VadProcessor()
        assert vad.voiced_ms == 0.0

    def test_speech_reports_voiced_span(self):
        vad = VadProcessor()
        vad._state = VadState.SPEECH
        vad._speech_start_time = 100.0
        vad._last_speech_time = 100.1
        assert abs(vad.voiced_ms - 100.0) < 1e-6

    def test_trailing_freezes_voiced_span(self):
        """TRAILING keeps is_speaking True but must not grow voiced_ms —
        this asymmetry is the whole point: a blip's trailing silence
        used to count toward barge-in confirmation."""
        vad = VadProcessor()
        vad._state = VadState.TRAILING
        vad._speech_start_time = 100.0
        vad._last_speech_time = 100.1  # voiced span ended at +100ms
        assert vad.is_speaking is True
        assert abs(vad.voiced_ms - 100.0) < 1e-6

    def test_blip_stays_under_bargein_threshold(self):
        """A 100ms voiced blip never reaches the 250ms confirmation
        floor, no matter how long its trailing silence runs — aligned
        with the segment-end discard check (min_speech_ms=250)."""
        vad = VadProcessor()
        vad._state = VadState.TRAILING
        vad._speech_start_time = 50.0
        vad._last_speech_time = 50.1
        assert vad.voiced_ms < vad.min_speech_ms

    def test_negative_clock_skew_clamped(self):
        vad = VadProcessor()
        vad._state = VadState.SPEECH
        vad._speech_start_time = 100.0
        vad._last_speech_time = 99.0
        assert vad.voiced_ms == 0.0


# ── VoiceSession recovery + provenance state ──────────────────────────


class TestVoiceSessionFields:
    def test_recovery_defaults(self):
        s = VoiceSession(session_id="t")
        assert s.tts_started is False
        assert s.bargein_pending is False
        assert s.undelivered_tts == []
        assert s.tts_params == {}

    def test_provenance_defaults(self):
        s = VoiceSession(session_id="t")
        assert s.capture_source == ""
        assert s.last_utterance_explicit is False

    def test_undelivered_lists_are_independent(self):
        a = VoiceSession(session_id="a")
        b = VoiceSession(session_id="b")
        a.undelivered_tts.append("hello")
        assert b.undelivered_tts == []


# ── Router prompt: greetings converse, idle narrowed ──────────────────


class TestRouterPromptGoals:
    def test_converse_includes_greetings(self):
        prompt = _build_system_prompt("Becca")
        converse_line = next(
            line for line in prompt.splitlines() if '"converse"' in line
        )
        assert "greeting" in converse_line
        assert "good morning" in converse_line

    def test_idle_is_scoped_to_completed_acknowledgments(self):
        prompt = _build_system_prompt("Becca")
        idle_line = next(
            line for line in prompt.splitlines() if '"idle"' in line
        )
        assert "just completed" in idle_line
        # The old wording ("short acknowledgment") swallowed greetings.
        assert "short acknowledgment" not in idle_line


# ── Replay of drained TTS after a false barge-in ──────────────────────


class _FakeWS:
    """Captures _send_json payloads (it serializes via send_text)."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    def types(self) -> list[str]:
        return [m.get("type") for m in self.sent]


def _replay_session(**overrides: Any) -> VoiceSession:
    s = VoiceSession(session_id="replay-test")
    s.interrupted = True
    s.bargein_pending = False  # caller clears it before invoking replay
    s.undelivered_tts = ["First sentence.", "Second sentence."]
    s.tts_params = {
        "voice": "af_heart", "speed": 1.0, "provider": None, "format": "mp3",
    }
    for key, value in overrides.items():
        setattr(s, key, value)
    return s


class TestReplayUndeliveredTts:
    def test_streams_drained_sentences_in_order(self, monkeypatch):
        ws = _FakeWS()
        session = _replay_session()
        streamed: list[str] = []

        async def fake_stream(sentence, websocket, conn, **kwargs):
            streamed.append(sentence)
            return True

        monkeypatch.setattr(voice_routes, "_state_conn", lambda app_state: object())
        monkeypatch.setattr(voice_routes, "stream_tts_sentence", fake_stream)

        replayed = asyncio.run(
            voice_routes._replay_undelivered_tts(ws, session, object())
        )

        assert replayed is True
        assert streamed == ["First sentence.", "Second sentence."]
        assert session.undelivered_tts == []
        assert session.interrupted is False
        assert session.is_speaking is False
        assert session.tts_ended_at > 0.0
        assert ws.types() == ["tts_start", "tts_end", "tts_start", "tts_end"]

    def test_nothing_pending_is_a_noop(self, monkeypatch):
        ws = _FakeWS()
        session = _replay_session(undelivered_tts=[])
        monkeypatch.setattr(voice_routes, "_state_conn", lambda app_state: object())

        replayed = asyncio.run(
            voice_routes._replay_undelivered_tts(ws, session, object())
        )

        assert replayed is False
        assert ws.sent == []

    def test_reinterrupt_mid_replay_keeps_tail(self, monkeypatch):
        """A real barge-in during replay must stop streaming and stash
        the remaining sentences for the next recovery pass."""
        ws = _FakeWS()
        session = _replay_session()
        streamed: list[str] = []

        async def fake_stream(sentence, websocket, conn, **kwargs):
            streamed.append(sentence)
            session.interrupted = True  # user starts speaking
            return True

        monkeypatch.setattr(voice_routes, "_state_conn", lambda app_state: object())
        monkeypatch.setattr(voice_routes, "stream_tts_sentence", fake_stream)

        replayed = asyncio.run(
            voice_routes._replay_undelivered_tts(ws, session, object())
        )

        assert replayed is True
        assert streamed == ["First sentence."]
        assert session.undelivered_tts == ["Second sentence."]

    def test_stream_failure_stops_without_raising(self, monkeypatch):
        ws = _FakeWS()
        session = _replay_session()

        async def fake_stream(sentence, websocket, conn, **kwargs):
            raise RuntimeError("tts backend down")

        monkeypatch.setattr(voice_routes, "_state_conn", lambda app_state: object())
        monkeypatch.setattr(voice_routes, "stream_tts_sentence", fake_stream)

        replayed = asyncio.run(
            voice_routes._replay_undelivered_tts(ws, session, object())
        )

        assert replayed is False
        assert session.is_speaking is False
        # The failed (unstreamed) tail is stashed for the next pass —
        # not silently dropped.
        assert session.undelivered_tts == ["First sentence.", "Second sentence."]

    def test_no_db_bails_cleanly(self, monkeypatch):
        ws = _FakeWS()
        session = _replay_session()
        monkeypatch.setattr(voice_routes, "_state_conn", lambda app_state: None)

        replayed = asyncio.run(
            voice_routes._replay_undelivered_tts(ws, session, object())
        )

        assert replayed is False
        assert ws.sent == []


# ── Honest history: heard-only rewrite (audio-only surfaces) ──────────


def _session_with_reply(content: str = "Alpha. Beta. Gamma.") -> VoiceSession:
    s = VoiceSession(session_id="rewrite-test")
    s.add_user_message("say something")
    s.add_assistant_message(content)
    return s


class TestApplyPendingHeardRewrite:
    def test_rewrites_to_heard_portion(self):
        session = _session_with_reply()
        session.pending_heard_rewrite = "Alpha."

        voice_routes._apply_pending_heard_rewrite(session)

        assert session.messages[-1]["content"] == "Alpha."
        assert session.pending_heard_rewrite is None

    def test_empty_heard_drops_the_message(self):
        """Zero sentences streamed = she never said it — an audio-only
        history must not carry a reply the user has no way of knowing
        about."""
        session = _session_with_reply()
        session.pending_heard_rewrite = ""

        voice_routes._apply_pending_heard_rewrite(session)

        assert session.messages[-1]["role"] == "user"
        assert session.pending_heard_rewrite is None

    def test_none_pending_is_a_noop(self):
        session = _session_with_reply()
        voice_routes._apply_pending_heard_rewrite(session)
        assert session.messages[-1]["content"] == "Alpha. Beta. Gamma."

    def test_non_assistant_tail_is_untouched(self):
        """If the last message isn't the interrupted reply (defensive —
        shouldn't happen given the vindication ordering), don't rewrite
        the wrong message."""
        session = _session_with_reply()
        session.add_user_message("follow-up")
        session.pending_heard_rewrite = "Alpha."

        voice_routes._apply_pending_heard_rewrite(session)

        assert session.messages[-1]["content"] == "follow-up"
        assert session.pending_heard_rewrite is None  # consumed either way


class TestReplayHeardRewriteReconciliation:
    def test_full_replay_discards_pending_rewrite(self, monkeypatch):
        """Everything drained got delivered — the committed full text is
        accurate, so the deferred shrink must not apply."""
        ws = _FakeWS()
        session = _replay_session(pending_heard_rewrite="Heard part.")

        async def fake_stream(sentence, websocket, conn, **kwargs):
            return True

        monkeypatch.setattr(voice_routes, "_state_conn", lambda app_state: object())
        monkeypatch.setattr(voice_routes, "stream_tts_sentence", fake_stream)

        asyncio.run(voice_routes._replay_undelivered_tts(ws, session, object()))

        assert session.pending_heard_rewrite is None

    def test_partial_replay_extends_pending_rewrite(self, monkeypatch):
        """Re-interrupted after one sentence — the heard boundary moved
        forward by exactly what was delivered."""
        ws = _FakeWS()
        session = _replay_session(pending_heard_rewrite="Heard part.")

        async def fake_stream(sentence, websocket, conn, **kwargs):
            session.interrupted = True
            return True

        monkeypatch.setattr(voice_routes, "_state_conn", lambda app_state: object())
        monkeypatch.setattr(voice_routes, "stream_tts_sentence", fake_stream)

        asyncio.run(voice_routes._replay_undelivered_tts(ws, session, object()))

        assert session.pending_heard_rewrite == "Heard part. First sentence."
        assert session.undelivered_tts == ["Second sentence."]

    def test_empty_drain_discards_pending_rewrite(self, monkeypatch):
        """Nothing was drained (all sentences streamed before the flag
        rose) — full text was delivered, deferred shrink is moot."""
        ws = _FakeWS()
        session = _replay_session(
            undelivered_tts=[], pending_heard_rewrite="Heard part.",
        )
        monkeypatch.setattr(voice_routes, "_state_conn", lambda app_state: object())

        replayed = asyncio.run(
            voice_routes._replay_undelivered_tts(ws, session, object())
        )

        assert replayed is False
        assert session.pending_heard_rewrite is None
