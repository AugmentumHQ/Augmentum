"""Time grounding layer (8.8) in the companion prompt composer.

Chat modes get date/time from ModeHandler._ensure_datetime; every
companion path (voice, becca_direct, native loop) bypasses mode
handlers and composes via prompt_compose — which had no clock at all
until 2026-06-11.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from augmentum.companion_runtime.prompt_compose import (
    _format_exchange_gap,
    _time_grounding_block,
)


class TestFormatExchangeGap:
    def test_silent_in_live_conversation(self):
        assert _format_exchange_gap(45.0) == ""
        assert _format_exchange_gap(1799.0) == ""

    def test_about_an_hour(self):
        assert "about an hour" in _format_exchange_gap(3600.0)

    def test_hours(self):
        line = _format_exchange_gap(5 * 3600.0)
        assert "about 5 hours" in line

    def test_about_a_day(self):
        assert "about a day" in _format_exchange_gap(30 * 3600.0)

    def test_days(self):
        assert "about 3 days" in _format_exchange_gap(3 * 86400.0)


class TestTimeGroundingBlock:
    def _pair(self, user_id="u1"):
        intent = SimpleNamespace(user_id=user_id, metadata={})
        runtime = SimpleNamespace()
        return intent, runtime

    def test_contains_current_time_block(self):
        intent, runtime = self._pair()
        block = _time_grounding_block(intent, runtime)
        assert "<current_time>" in block
        assert "Current date:" in block
        assert "date-stamped" in block

    def test_first_turn_has_no_gap_line(self):
        intent, runtime = self._pair()
        block = _time_grounding_block(intent, runtime)
        assert "since your last exchange" not in block

    def test_gap_line_after_long_silence(self):
        intent, runtime = self._pair()
        _time_grounding_block(intent, runtime)
        # Rewind the recorded timestamp by 3 hours.
        runtime._last_turn_ts_by_user["u1"] = time.time() - 3 * 3600.0
        block = _time_grounding_block(intent, runtime)
        assert "since your last exchange" in block

    def test_gap_tracking_is_per_user(self):
        intent_a, runtime = self._pair("alice")
        _time_grounding_block(intent_a, runtime)
        runtime._last_turn_ts_by_user["alice"] = time.time() - 3 * 3600.0
        # Bob's first turn must not inherit Alice's gap.
        intent_b = SimpleNamespace(user_id="bob", metadata={})
        block_b = _time_grounding_block(intent_b, runtime)
        assert "since your last exchange" not in block_b
        # Alice still gets hers.
        block_a = _time_grounding_block(intent_a, runtime)
        assert "since your last exchange" in block_a

    def test_quick_followup_no_gap_line(self):
        intent, runtime = self._pair()
        _time_grounding_block(intent, runtime)
        block = _time_grounding_block(intent, runtime)  # seconds later
        assert "since your last exchange" not in block
