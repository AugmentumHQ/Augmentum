"""Deadline-countdown kind — schedule resolution (offset boundaries) + the
runner's days-remaining message and day-of completion."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from augmentum.companion_runtime.standing_tasks import (
    _compute_next_run_at,
    _kind_deadline,
)
from augmentum.tools.schedule_deadline import _DEFAULT_OFFSETS, _normalize_offsets


def _today():
    return datetime.now(timezone.utc).date()


# ── Schedule resolution (deadline mode in _compute_next_run_at) ──────────

def test_next_run_picks_soonest_future_offset():
    today = _today()
    target = today + timedelta(days=100)
    res = _compute_next_run_at(
        params={"target_date": target.isoformat(),
                "offsets_days": [30, 14, 7, 1], "local_time": "09:00"},
        interval_seconds=86400, user_timezone="UTC", jitter_seed="",
    )
    # Soonest boundary = target − 30 days, at 09:00.
    assert res == f"{(today + timedelta(days=70)).isoformat()} 09:00:00"


def test_next_run_consumes_past_offsets():
    today = _today()
    target = today + timedelta(days=5)
    res = _compute_next_run_at(
        params={"target_date": target.isoformat(),
                "offsets_days": [30, 14, 7, 1], "local_time": "09:00"},
        interval_seconds=86400, user_timezone="UTC", jitter_seed="",
    )
    # 30/14/7-day boundaries are past; the 1-day-before (today+4) is next.
    assert res == f"{(today + timedelta(days=4)).isoformat()} 09:00:00"


def test_next_run_defaults_offsets_when_missing():
    today = _today()
    target = today + timedelta(days=100)
    res = _compute_next_run_at(
        params={"target_date": target.isoformat(), "local_time": "09:00"},
        interval_seconds=86400, user_timezone="UTC", jitter_seed="",
    )
    # Default cadence includes 30 → soonest is target−30.
    assert res == f"{(today + timedelta(days=70)).isoformat()} 09:00:00"


def test_next_run_default_fire_time_when_no_local_time():
    today = _today()
    target = today + timedelta(days=100)
    res = _compute_next_run_at(
        params={"target_date": target.isoformat(), "offsets_days": [30]},
        interval_seconds=86400, user_timezone="UTC", jitter_seed="",
    )
    assert res == f"{(today + timedelta(days=70)).isoformat()} 09:00:00"


def test_next_run_exhausted_returns_day_of():
    today = _today()
    target = today + timedelta(days=100)
    # All offsets already past → fall back to the target moment itself.
    res = _compute_next_run_at(
        params={"target_date": target.isoformat(),
                "offsets_days": [200, 150], "local_time": "09:00"},
        interval_seconds=86400, user_timezone="UTC", jitter_seed="",
    )
    assert res == f"{target.isoformat()} 09:00:00"


# ── Runner ──────────────────────────────────────────────────────────────

class _FakeStore:
    async def get_user_or_global(self, uid, key):
        return "UTC"


class _FakeApp:
    settings_store = _FakeStore()


class _FakeRuntime:
    _app_state = _FakeApp()
    companion_id = "c1"


async def test_runner_future_no_completion():
    target = (_today() + timedelta(days=10)).isoformat()
    res = await _kind_deadline(_FakeRuntime(), user_id="u",
                               params={"target_date": target, "title": "Taxes due"})
    assert res["noteworthy"] is True
    assert "Taxes due" in res["summary"]
    assert "10 days left" in res["summary"]
    assert "params_update" not in res["details"]  # not completed yet


async def test_runner_day_of_completes():
    res = await _kind_deadline(_FakeRuntime(), user_id="u",
                               params={"target_date": _today().isoformat(), "title": "Taxes"})
    assert "today" in res["summary"]
    assert res["details"]["params_update"]["one_shot"] is True


async def test_runner_past_completes():
    past = (_today() - timedelta(days=3)).isoformat()
    res = await _kind_deadline(_FakeRuntime(), user_id="u",
                               params={"target_date": past, "title": "X"})
    assert "ago" in res["summary"]
    assert res["details"]["params_update"]["one_shot"] is True


async def test_runner_checklist_and_note_in_content():
    target = (_today() + timedelta(days=5)).isoformat()
    res = await _kind_deadline(_FakeRuntime(), user_id="u", params={
        "target_date": target, "title": "Grant",
        "checklist": ["draft budget", "get letters"], "note": "Submit via portal",
    })
    content = res["details"]["content"]
    assert "draft budget" in content
    assert "get letters" in content
    assert "Submit via portal" in content


async def test_runner_bad_or_missing_date_raises():
    with pytest.raises(ValueError):
        await _kind_deadline(_FakeRuntime(), user_id="u",
                             params={"target_date": "not-a-date", "title": "X"})
    with pytest.raises(ValueError):
        await _kind_deadline(_FakeRuntime(), user_id="u", params={"title": "X"})


# ── Tool offset normalization ───────────────────────────────────────────

def test_normalize_offsets():
    assert _normalize_offsets([30, 14, 7, 1]) == [30, 14, 7, 1]
    assert _normalize_offsets(["30", "14", "day-of"]) == [30, 14, 0]
    assert _normalize_offsets([]) == _DEFAULT_OFFSETS
    assert _normalize_offsets("junk") == _DEFAULT_OFFSETS
    # negatives, absurd values, and junk dropped; survivors sorted desc.
    assert _normalize_offsets([-5, 10_000_000, "x", 7]) == [7]
