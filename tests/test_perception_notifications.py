"""The notification acquisition stream end-to-end — store, fuser, and the live
pass that wires them.

Three layers, each tested where it lives:
  * **store** (`acquisition/notifications.py`) against real in-memory SQLite +
    migrations — dedup on re-post, recency window, retention prune, user scoping;
  * **fuser** (`fusers/notifications.py`) as a pure function over synthetic
    observation lists — the echo-vs-aware proof: one ping → nothing, a *pattern*
    → one insight with its evidence chain;
  * **live** (`live.py`) — with the acquire gate flipped on, a full
    `evaluate_user` pass loads the stored notifications, fuses, judges, and
    delivers onto the real initiative queue.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from augmentum.companion_runtime.perception.acquisition.notifications import (
    NotificationObservation,
    prune_notifications,
    recent_notifications,
    record_notifications,
)
from augmentum.companion_runtime.perception.fusers.notifications import (
    fuse_notifications,
)
from augmentum.companion_runtime.perception.fusion import FusionContext


async def _boot_backend():
    from augmentum.state.backends.sqlite import SQLiteBackend
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    return backend


def _obs(**kw) -> NotificationObservation:
    base = {"source_pkg": "com.whatsapp", "source_app": "WhatsApp",
            "category": "msg", "title": "Jordan", "body": "you around?",
            "person": "Jordan", "is_message": True, "posted_at": 1000.0,
            "dedup_key": ""}
    base.update(kw)
    # give each a distinct dedup_key by default so they don't collapse
    if not base["dedup_key"]:
        base["dedup_key"] = f"{base['source_pkg']}|{base['title']}|{base['body']}"
    return NotificationObservation(**base)


# --- the store -------------------------------------------------------------

@pytest.mark.asyncio
async def test_record_and_recent_round_trip():
    backend = await _boot_backend()
    n = await record_notifications(
        backend, user_id="u1",
        observations=[_obs(dedup_key="a"), _obs(dedup_key="b", body="ping 2")],
        now=1000.0,
    )
    assert n == 2
    rows = await recent_notifications(backend, user_id="u1", now=1000.0)
    assert len(rows) == 2
    assert {r.body for r in rows} == {"you around?", "ping 2"}
    assert all(r.is_message for r in rows)


@pytest.mark.asyncio
async def test_dedup_on_reposted_notification():
    backend = await _boot_backend()
    # Android re-posts the same notification (typing dots, delivery tick) — same
    # dedup_key. The second insert must be ignored, not counted as new pressure.
    await record_notifications(backend, user_id="u1",
                               observations=[_obs(dedup_key="k1")], now=1000.0)
    n2 = await record_notifications(backend, user_id="u1",
                                    observations=[_obs(dedup_key="k1", body="updated")],
                                    now=1001.0)
    assert n2 == 0
    rows = await recent_notifications(backend, user_id="u1", now=1001.0)
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_recent_window_and_user_scope():
    backend = await _boot_backend()
    await record_notifications(backend, user_id="u1", observations=[
        _obs(dedup_key="fresh", posted_at=10_000.0),
        _obs(dedup_key="stale", posted_at=100.0),
    ], now=10_000.0)
    # u2 must never see u1's rows.
    await record_notifications(backend, user_id="u2",
                               observations=[_obs(dedup_key="other")], now=10_000.0)
    rows = await recent_notifications(backend, user_id="u1",
                                      since_s=3600.0, now=10_000.0)
    assert [r.dedup_key for r in rows] == ["fresh"]
    assert await recent_notifications(backend, user_id="u1", now=10_000.0) != []
    u2 = await recent_notifications(backend, user_id="u2", now=10_000.0)
    assert len(u2) == 1 and u2[0].dedup_key == "other"


@pytest.mark.asyncio
async def test_prune_retention():
    backend = await _boot_backend()
    now = 10 * 86400.0
    await record_notifications(backend, user_id="u1", observations=[
        _obs(dedup_key="old", posted_at=1.0),               # ~10 days old
        _obs(dedup_key="new", posted_at=now - 3600.0),      # 1h old
    ], now=now)
    deleted = await prune_notifications(backend, user_id="u1",
                                        retention_days=7, now=now)
    assert deleted == 1
    rows = await recent_notifications(backend, user_id="u1",
                                      since_s=now, now=now)
    assert [r.dedup_key for r in rows] == ["new"]


@pytest.mark.asyncio
async def test_from_wire_normalizes_and_skips_empty():
    assert NotificationObservation.from_wire({}) is None
    assert NotificationObservation.from_wire({"title": "", "body": ""}) is None
    obs = NotificationObservation.from_wire({
        "package": "com.slack", "app": "Slack", "category": "msg",
        "title": "Eva", "text": "standup?", "notif_key": "kkk",
    })
    assert obs is not None
    assert obs.source_pkg == "com.slack" and obs.is_message is True
    assert obs.person == "Eva" and obs.dedup_key == "kkk"


# --- the fuser (pure: the echo-vs-aware proof) -----------------------------

def _ctx(notifs, now=1000.0):
    return FusionContext(user_id="u1", now=now, signals={"notifications": notifs})


def test_single_ping_is_not_an_insight():
    # One message from one person is an echo, not a pattern — emit nothing.
    out = fuse_notifications(_ctx([_obs()]))
    assert out == []


def test_two_from_one_person_fuses_to_pressure():
    out = fuse_notifications(_ctx([
        _obs(dedup_key="1", body="you around?"),
        _obs(dedup_key="2", body="call me when free"),
    ]))
    assert len(out) == 1
    ins = out[0]
    assert ins.kind == "social.pressure" and ins.shape == "social.pressure"
    assert "Jordan" in ins.summary and "2 unread" in ins.summary
    assert ins.time_critical is False              # pull-worthy, not an interrupt
    assert any("2 unread" in e for e in ins.evidence)
    assert ins.base_score > 0


def test_pressure_value_scales_with_count():
    two = fuse_notifications(_ctx([_obs(dedup_key=str(i)) for i in range(2)]))[0]
    five = fuse_notifications(_ctx([_obs(dedup_key=str(i)) for i in range(5)]))[0]
    assert five.value > two.value


def test_two_people_each_pressing_yield_two_insights():
    out = fuse_notifications(_ctx([
        _obs(dedup_key="j1", person="Jordan", title="Jordan"),
        _obs(dedup_key="j2", person="Jordan", title="Jordan"),
        _obs(dedup_key="e1", person="Eva", title="Eva"),
        _obs(dedup_key="e2", person="Eva", title="Eva"),
    ]))
    assert {i.summary.split(" ")[0] for i in out} == {"Jordan", "Eva"}


def test_repeated_calls_are_time_critical():
    out = fuse_notifications(_ctx([
        _obs(dedup_key="c1", category="call", is_message=False, body=""),
        _obs(dedup_key="c2", category="call", is_message=False, body=""),
    ]))
    assert len(out) == 1
    assert out[0].kind == "comms.missed_call" and out[0].time_critical is True
    assert "tried to call you 2 times" in out[0].summary


def test_stale_notifications_dropped_by_window():
    # posted long before now, beyond the fresh window → no live pattern.
    out = fuse_notifications(_ctx([
        _obs(dedup_key="1", posted_at=1.0),
        _obs(dedup_key="2", posted_at=2.0),
    ], now=10_000_000.0))
    assert out == []


def test_empty_signal_bag_is_noop():
    assert fuse_notifications(FusionContext(user_id="u1", now=1.0)) == []
    assert fuse_notifications(_ctx([])) == []


# --- live: the full pass with the acquire gate on --------------------------

def _fake_runtime(backend):
    rt = MagicMock()
    rt.backend = backend
    rt.companion_id = "becca"
    rt.owner_user_id = "u1"
    rt.app_state = None
    published: list[tuple[str, dict]] = []

    async def _publish(topic, payload, **kw):
        published.append((topic, payload))

    rt.bus.publish_topic = _publish
    rt._published = published
    return rt


@pytest.mark.asyncio
async def test_live_pass_acquires_fuses_and_delivers(monkeypatch):
    from augmentum.companion_runtime.perception import fusion as _fusion
    from augmentum.companion_runtime.perception.budget import BUDGET
    from augmentum.companion_runtime.perception.live import evaluate_user
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_perception_enabled", True, raising=False)
    monkeypatch.setattr(
        settings, "companion_perception_acquire_notifications", True, raising=False,
    )
    _fusion.clear_fusers()
    BUDGET.reset()

    backend = await _boot_backend()
    # A missed-call pattern lands in the store (time-critical → eligible to speak).
    await record_notifications(backend, user_id="u1", observations=[
        NotificationObservation.from_wire({
            "package": "com.android.dialer", "app": "Phone", "category": "call",
            "title": "Mom", "person": "Mom", "notif_key": "c1", "posted_at": 1000.0,
        }),
        NotificationObservation.from_wire({
            "package": "com.android.dialer", "app": "Phone", "category": "call",
            "title": "Mom", "person": "Mom", "notif_key": "c2", "posted_at": 1001.0,
        }),
    ], now=1001.0)

    rt = _fake_runtime(backend)
    try:
        out = await evaluate_user(rt, user_id="u1", now=1002.0)
    finally:
        _fusion.clear_fusers()

    # The fuser fired (registered by the live pass), the gate delivered, and a
    # 'perceived' row landed on the real initiative queue.
    assert sum(out.values()) >= 1
    cur = await backend.conn.execute(
        "SELECT kind, payload FROM companion_initiative_queue WHERE user_id='u1'",
    )
    rows = await cur.fetchall()
    await cur.close()
    assert rows and rows[0][0] == "perceived"
    assert "Mom" in rows[0][1]


@pytest.mark.asyncio
async def test_live_pass_inert_when_acquire_gate_off(monkeypatch):
    from augmentum.companion_runtime.perception import fusion as _fusion
    from augmentum.companion_runtime.perception.budget import BUDGET
    from augmentum.companion_runtime.perception.live import evaluate_user
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_perception_enabled", True, raising=False)
    monkeypatch.setattr(
        settings, "companion_perception_acquire_notifications", False, raising=False,
    )
    _fusion.clear_fusers()
    BUDGET.reset()

    backend = await _boot_backend()
    await record_notifications(backend, user_id="u1", observations=[
        _obs(dedup_key="1"), _obs(dedup_key="2"),
    ], now=1000.0)
    rt = _fake_runtime(backend)
    try:
        await evaluate_user(rt, user_id="u1", now=1001.0)
    finally:
        _fusion.clear_fusers()
    # Acquire gate off → signals never loaded → nothing fused → empty queue.
    cur = await backend.conn.execute(
        "SELECT COUNT(*) FROM companion_initiative_queue WHERE user_id='u1'",
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == 0
