"""Tests pinning curator.step's attempt-debounce.

The bug: prior to this fix, curator.step's debounce was gated only on
_last_curator_at (set on successful note write). In steady state most
candidates are already journaled, low-value, or safety-blocked, so no
note gets written, _last_curator_at stays stale, and every 5s tick in
'present' state re-runs the expensive gather_feeds + SearXNG recommender
pipeline. Logs from 2026-06-10 showed all 6 search engines suspended
within minutes as a result.

The fix: separate _last_curator_attempt_at (set unconditionally before
the expensive work) from _last_curator_at (set on write). Either active
window prevents re-entry.
"""
from __future__ import annotations

import re
from pathlib import Path

_CURATOR = Path("augmentum/companion_runtime/curator.py")


def _src() -> str:
    return _CURATOR.read_text(encoding="utf-8")


class TestAttemptDebounceConstant:
    """A separate attempt-interval constant must exist + be shorter than
    the write-debounce interval. Both must be in source so the tunables
    are auditable in one place."""

    def test_attempt_constant_declared(self):
        src = _src()
        assert "_STEP_ATTEMPT_DEBOUNCE_SECONDS" in src

    def test_attempt_constant_is_seconds_float(self):
        src = _src()
        # Match the literal assignment so a future refactor that swaps to
        # a different name fails loudly.
        m = re.search(
            r"_STEP_ATTEMPT_DEBOUNCE_SECONDS\s*:\s*float\s*=\s*(\d+(?:\.\d+)?)",
            src,
        )
        assert m is not None, "_STEP_ATTEMPT_DEBOUNCE_SECONDS must be a float literal"
        value = float(m.group(1))
        # 10 min is the design value. Allow 1 min - 30 min as the
        # reasonable range — a future tweak inside that band shouldn't
        # break the test, but a swing to 5s would.
        assert 60.0 <= value <= 1800.0, (
            f"_STEP_ATTEMPT_DEBOUNCE_SECONDS={value} is outside the "
            "sane range (60s-30min). Sub-60s lets the recommender "
            "re-fire faster than upstream engines tolerate."
        )

    def test_attempt_interval_strictly_shorter_than_write(self):
        """Attempt-debounce must be shorter than write-debounce, otherwise
        the attempt check never fires."""
        src = _src()
        write_m = re.search(
            r"_STEP_DEBOUNCE_SECONDS\s*:\s*float\s*=\s*(\d+(?:\.\d+)?)",
            src,
        )
        attempt_m = re.search(
            r"_STEP_ATTEMPT_DEBOUNCE_SECONDS\s*:\s*float\s*=\s*(\d+(?:\.\d+)?)",
            src,
        )
        assert write_m is not None
        assert attempt_m is not None
        assert float(attempt_m.group(1)) < float(write_m.group(1))


class TestDebounceWiring:
    """The actual step() gate must check both attempt + write windows
    and set the attempt timestamp BEFORE the expensive work begins."""

    def test_attempt_check_present(self):
        src = _src()
        assert "_last_curator_attempt_at" in src
        # The check must compare against the attempt interval, not just
        # the write interval.
        assert "now - last_attempt < attempt_interval" in src or (
            "now - last_attempt" in src and "attempt_interval" in src
        )

    def test_attempt_timestamp_set_before_expensive_work(self):
        """The attempt timestamp must be set IMMEDIATELY after the gate,
        not after generate_recommendations returns. Otherwise a slow
        SearXNG call leaves the next tick free to re-enter while the
        first call's still in flight."""
        src = _src()
        # Find the step function body
        step_idx = src.find("async def step(runtime")
        assert step_idx > 0
        body = src[step_idx:step_idx + 3000]
        # The attempt set must appear before any generate_recommendations
        # call. Tolerate intervening comments / log statements.
        set_idx = body.find("runtime._last_curator_attempt_at = now")
        gen_idx = body.find("generate_recommendations")
        assert set_idx > 0, "Attempt timestamp not set inside step()"
        # gen_idx may be -1 if the call is in a helper. If both are in
        # the same body, set must come first.
        if gen_idx > 0:
            assert set_idx < gen_idx, (
                "Attempt timestamp must be set BEFORE generate_recommendations "
                "to prevent re-entry during in-flight SearXNG calls"
            )

    def test_write_timestamp_still_set_on_success(self):
        """The original _last_curator_at semantics must be preserved
        — set on successful note write. Regression guard against
        someone "simplifying" the fix by dropping the write tracker."""
        src = _src()
        # Count occurrences — should be set in at least two places
        # (For-You write path + legacy topic write path).
        assert src.count("runtime._last_curator_at = now") >= 2


class TestSettingsOverride:
    """The attempt interval must be overridable via settings so prod
    deployments can tune without code changes."""

    def test_settings_override_supported(self):
        src = _src()
        assert "companion_curator_attempt_interval_s" in src

    def test_write_setting_still_supported(self):
        """The existing write-debounce setting must keep working —
        regression guard."""
        src = _src()
        assert "companion_curator_interval_s" in src
