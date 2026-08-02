"""STT capture continuity — 2026-06-11 hard-cutoff / lost-head fixes.

Three live-fire failure modes from always-listening voice:

1. Smart-turn veto confidence was compared as raw completion
   probability — prob=0.03 ("97% sure the user is still talking")
   read as a 3%-confidence veto and was OVERRIDDEN, hard-cutting the
   user mid-thought. Confidence is threshold − prob.
2. The veto-deadline safety valve force-finalized even while VAD
   showed active speech (user resumed — the case the veto protected).
3. A veto continuation's next speech_start cleared the transcript
   parts + PCM buffer, destroying the pre-pause segment.

Plus: the VAD prefix ring grew 700→1500ms, made safe by clamping to
post-TTS audio so her own voice can't ride into the user's transcript.
"""

from __future__ import annotations

from augmentum.proxy.voice_routes import (
    _smart_turn_veto_confidence,
    _trim_prefix_to_post_tts,
)
from augmentum.voice.vad import FRAME_BYTES


class TestVetoConfidence:
    def test_near_zero_prob_is_max_confidence_veto(self):
        # prob=0.03: model near-certain the turn is INCOMPLETE.
        assert _smart_turn_veto_confidence(0.03, 0.5) >= 0.45

    def test_borderline_prob_is_low_confidence_veto(self):
        # prob=0.45 against threshold 0.5: coin-flip veto.
        assert abs(_smart_turn_veto_confidence(0.45, 0.5) - 0.05) < 1e-9

    def test_logged_failure_cases_now_honored(self):
        # The 2026-06-11 live-fire overrides: 0.034 / 0.064 must clear
        # the default 0.3 floor (veto honored); 0.313 stays below it
        # (defer to VAD — acceptable borderline).
        min_conf = 0.3
        assert _smart_turn_veto_confidence(0.034, 0.5) >= min_conf
        assert _smart_turn_veto_confidence(0.064, 0.5) >= min_conf
        assert _smart_turn_veto_confidence(0.313, 0.5) < min_conf

    def test_never_negative(self):
        assert _smart_turn_veto_confidence(0.9, 0.5) == 0.0


class TestPrefixTrim:
    def test_no_tts_no_trim(self):
        buf = b"x" * (10 * FRAME_BYTES)
        assert _trim_prefix_to_post_tts(buf, None, 100.0) is buf
        assert _trim_prefix_to_post_tts(buf, 0.0, 100.0) is buf

    def test_long_elapsed_no_trim(self):
        # TTS ended 10s ago — the whole ring is post-TTS.
        buf = b"x" * (40 * FRAME_BYTES)  # ~1.3s
        assert _trim_prefix_to_post_tts(buf, 90.0, 100.0) is buf

    def test_recent_tts_trims_to_elapsed(self):
        # TTS ended ~320ms ago (~10 frames); a 40-frame ring must
        # shrink to roughly the last 10 frames so her tail isn't
        # transcribed. Float jitter on elapsed can shave one frame.
        buf = bytes(range(256)) * (40 * FRAME_BYTES // 256)
        out = _trim_prefix_to_post_tts(buf, 100.0 - 0.320, 100.0)
        assert len(out) in (9 * FRAME_BYTES, 10 * FRAME_BYTES)
        assert buf.endswith(out)

    def test_active_tts_negative_elapsed_no_trim(self):
        # Barge-in: tts_ended_at can sit in the future relative to a
        # stale monotonic read — never raise, never trim to nothing.
        buf = b"x" * (5 * FRAME_BYTES)
        assert _trim_prefix_to_post_tts(buf, 200.0, 100.0) is buf

    def test_empty_prefix_passthrough(self):
        assert _trim_prefix_to_post_tts(b"", 90.0, 100.0) == b""
