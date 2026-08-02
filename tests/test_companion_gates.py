"""Smoke tests for companion_runtime.gates.

Slice 0 ships four pure gate functions that protect the user's
attention and the primary engine's KV cache from autonomous companion
activity. These tests pin the contract: fail-open semantics, timestamp
comparisons, settings drift handling.

Each test is fast (<5 ms) because the gates are pure-Python — no I/O,
no DB, no LLM. The point is that they STAY that way: if anyone makes
``is_primary_busy`` do a network call, these tests will keep running
fast and the regression will show up in profiling.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

# ── is_primary_busy ──────────────────────────────────────────────────


def test_is_primary_busy_none_runtime():
    """Fail-open: missing runtime returns False, not True."""
    from augmentum.companion_runtime.gates import is_primary_busy
    assert is_primary_busy(None) is False


def test_is_primary_busy_no_app_state():
    """Runtime without app_state binding → not busy (fail-open)."""
    from augmentum.companion_runtime.gates import is_primary_busy
    rt = SimpleNamespace(_app_state=None)
    assert is_primary_busy(rt) is False


def test_is_primary_busy_no_llama_manager():
    """app_state without llama_manager → not busy."""
    from augmentum.companion_runtime.gates import is_primary_busy
    rt = SimpleNamespace(_app_state=SimpleNamespace())
    assert is_primary_busy(rt) is False


def test_is_primary_busy_idle_manager():
    """Manager with is_busy=False → not busy."""
    from augmentum.companion_runtime.gates import is_primary_busy
    mgr = SimpleNamespace(is_busy=False)
    rt = SimpleNamespace(_app_state=SimpleNamespace(llama_manager=mgr))
    assert is_primary_busy(rt) is False


def test_is_primary_busy_in_flight():
    """Manager with in-flight request → busy."""
    from augmentum.companion_runtime.gates import is_primary_busy
    mgr = SimpleNamespace(is_busy=True)
    rt = SimpleNamespace(_app_state=SimpleNamespace(llama_manager=mgr))
    assert is_primary_busy(rt) is True


def test_is_primary_busy_manager_raises():
    """Fail-open: if the property raises, treat as not busy."""
    from augmentum.companion_runtime.gates import is_primary_busy

    class Mgr:
        @property
        def is_busy(self):
            raise RuntimeError("synthetic failure")

    rt = SimpleNamespace(_app_state=SimpleNamespace(llama_manager=Mgr()))
    assert is_primary_busy(rt) is False


# ── is_user_recently_active ──────────────────────────────────────────


def test_is_user_recently_active_default():
    """Default 0.0 cooldown → not active."""
    from augmentum.companion_runtime.gates import is_user_recently_active
    rt = SimpleNamespace(user_cooldown_until=0.0)
    assert is_user_recently_active(rt) is False


def test_is_user_recently_active_in_window():
    """Cooldown_until in the future → active."""
    from augmentum.companion_runtime.gates import is_user_recently_active
    rt = SimpleNamespace(user_cooldown_until=time.time() + 30)
    assert is_user_recently_active(rt) is True


def test_is_user_recently_active_expired():
    """Cooldown_until in the past → not active."""
    from augmentum.companion_runtime.gates import is_user_recently_active
    rt = SimpleNamespace(user_cooldown_until=time.time() - 30)
    assert is_user_recently_active(rt) is False


def test_is_user_recently_active_none_safe():
    """None runtime + missing attribute → fail-open False."""
    from augmentum.companion_runtime.gates import is_user_recently_active
    assert is_user_recently_active(None) is False
    assert is_user_recently_active(SimpleNamespace()) is False


# ── is_heavy_quiet ───────────────────────────────────────────────────


def test_is_heavy_quiet_default():
    from augmentum.companion_runtime.gates import is_heavy_quiet
    rt = SimpleNamespace(heavy_quiet_until=0.0)
    assert is_heavy_quiet(rt) is False


def test_is_heavy_quiet_in_window():
    from augmentum.companion_runtime.gates import is_heavy_quiet
    rt = SimpleNamespace(heavy_quiet_until=time.time() + 600)
    assert is_heavy_quiet(rt) is True


# ── is_hushed_now ────────────────────────────────────────────────────


def _set_hush(value: str):
    """Helper: set companion_journal_hushed_until on the live settings.
    Tests must restore the original via try/finally to avoid bleed."""
    from augmentum.config import settings
    object.__setattr__(settings, "companion_journal_hushed_until", value)


def test_is_hushed_now_empty_setting():
    """Empty string → not hushed."""
    from augmentum.companion_runtime.gates import is_hushed_now
    from augmentum.config import settings
    orig = getattr(settings, "companion_journal_hushed_until", "")
    try:
        _set_hush("")
        assert is_hushed_now() is False
    finally:
        _set_hush(orig)


def test_is_hushed_now_future_iso():
    """ISO timestamp in the future → hushed."""
    from augmentum.companion_runtime.gates import is_hushed_now
    from augmentum.config import settings
    orig = getattr(settings, "companion_journal_hushed_until", "")
    future = (datetime.now(UTC) + timedelta(hours=1)).strftime(
        "%Y-%m-%d %H:%M:%S",
    )
    try:
        _set_hush(future)
        assert is_hushed_now() is True
    finally:
        _set_hush(orig)


def test_is_hushed_now_past_iso():
    """ISO timestamp in the past → not hushed."""
    from augmentum.companion_runtime.gates import is_hushed_now
    from augmentum.config import settings
    orig = getattr(settings, "companion_journal_hushed_until", "")
    past = (datetime.now(UTC) - timedelta(hours=1)).strftime(
        "%Y-%m-%d %H:%M:%S",
    )
    try:
        _set_hush(past)
        assert is_hushed_now() is False
    finally:
        _set_hush(past)
        _set_hush(orig)


def test_is_hushed_now_garbage_fails_open():
    """Malformed timestamp → not hushed (don't muzzle Becca forever)."""
    from augmentum.companion_runtime.gates import is_hushed_now
    from augmentum.config import settings
    orig = getattr(settings, "companion_journal_hushed_until", "")
    try:
        _set_hush("not-a-timestamp")
        assert is_hushed_now() is False
    finally:
        _set_hush(orig)


def test_is_hushed_now_iso_t_z_format():
    """ISO with 'T' separator and 'Z' suffix should parse."""
    from augmentum.companion_runtime.gates import is_hushed_now
    from augmentum.config import settings
    orig = getattr(settings, "companion_journal_hushed_until", "")
    future = (datetime.now(UTC) + timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ",
    )
    try:
        _set_hush(future)
        assert is_hushed_now() is True
    finally:
        _set_hush(orig)
