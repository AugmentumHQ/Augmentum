"""Sprint 4 multi-tenant-scoping + autonomy-lifecycle fixes — regression
pins (audit 2026-06-17).

Pins the new primitives + the cross-tenant guarantees. The SQL-scoping
conversions (initiative/revisit/dream) are exercised for regressions by
their existing suites; here we pin the helper, the bus owner-filter, the
LRU caps, the per-(user,verb) auto-pause, and the standing-task row shape.
"""
from __future__ import annotations

import asyncio

import pytest


# ── owner_clause helper ───────────────────────────────────────────────

def test_owner_clause_shapes():
    from augmentum.companion_runtime.scoping import owner_clause, owner_clause_nullable

    frag, p = owner_clause("usr_a")
    assert frag == "AND (? = '' OR user_id = ?)"
    assert p == ("usr_a", "usr_a")

    nfrag, np = owner_clause_nullable("usr_a")
    assert "user_id IS NULL" in nfrag
    assert np == ("usr_a", "usr_a")

    # Empty owner = pass-through (the first branch is true at SQL time).
    frag0, p0 = owner_clause("")
    assert p0 == ("", "")


# ── PresenceEvent carries + serializes the owner ──────────────────────

def test_presence_event_owner_round_trips():
    import json

    from augmentum.companion_runtime.bus import PresenceEvent

    ev = PresenceEvent(topic="chat.turn_completed", payload={"x": 1}, owner_user_id="usr_a")
    data = json.loads(ev.to_json())
    assert data["owner_user_id"] == "usr_a"
    # Default is global.
    ev2 = PresenceEvent(topic="state.transition", payload={})
    assert json.loads(ev2.to_json())["owner_user_id"] == ""


# ── ws_fanout owner filter (the cross-tenant leak fix) ────────────────

class _FakeWS:
    def __init__(self):
        self.sent: list[str] = []
        self.application_state = None
        self.client_state = None

    async def accept(self):
        pass

    async def send_text(self, txt: str):
        self.sent.append(txt)

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_ws_fanout_filters_other_users():
    import json

    from augmentum.companion_runtime.bus import PresenceBus, ws_fanout

    bus = PresenceBus()
    ws = _FakeWS()

    # Run the pump; feed events; then close the subscription so it exits.
    task = asyncio.create_task(ws_fanout(ws, bus, owner_user_id="usr_a"))
    await asyncio.sleep(0.02)  # let it subscribe
    await bus.publish_topic("chat.turn_completed", {"m": 1}, owner_user_id="usr_a")   # mine
    await bus.publish_topic("chat.turn_completed", {"m": 2}, owner_user_id="usr_b")   # NOT mine
    await bus.publish_topic("state.transition", {"m": 3})                              # global
    await bus.publish_topic("voice.turn_ended", {"m": 4, "user_id": "usr_b"})          # payload-sniff: not mine
    await asyncio.sleep(0.05)
    # Close: unsubscribe by cancelling the pump.
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass

    seen = [json.loads(s)["payload"]["m"] for s in ws.sent]
    assert 1 in seen          # owner's event delivered
    assert 3 in seen          # global delivered
    assert 2 not in seen      # other user's native-owner event dropped
    assert 4 not in seen      # other user's payload-sniffed event dropped


# ── presence_context LRU caps ─────────────────────────────────────────

def test_attention_store_lru_evicts_oldest_user(monkeypatch):
    from augmentum.companion_runtime import presence_context as pc

    monkeypatch.setattr(pc, "_MAX_TRACKED_USERS", 3)
    store = pc.AttentionStore()
    for i in range(5):
        store.note(f"u{i}", "page", url=f"http://x/{i}")
    # Only the 3 most-recent users survive.
    assert len(store._slots) == 3
    assert "u0" not in store._slots
    assert "u4" in store._slots


def test_loaded_context_store_lru_evicts_oldest_user(monkeypatch):
    from augmentum.companion_runtime import presence_context as pc

    monkeypatch.setattr(pc, "_MAX_TRACKED_USERS", 2)
    store = pc.LoadedContextStore()
    for i in range(4):
        store.load(f"u{i}", "page", label="L", content="body")
    assert len(store._items) == 2
    assert "u3" in store._items
    assert "u0" not in store._items


# ── event_bus auto-pause keyed per (user, verb) ───────────────────────

def test_verb_pause_is_per_user():
    from augmentum.companion_runtime.event_bus import VerbDispatcher

    d = VerbDispatcher.__new__(VerbDispatcher)
    d._paused = set()
    d._error_counts = {}

    d._paused.add(("usr_a", "weather"))
    assert d.is_paused("weather", user_id="usr_a")
    assert not d.is_paused("weather", user_id="usr_b")  # other user unaffected

    # resume with a user clears just that pair...
    d._paused.add(("usr_b", "weather"))
    d.resume("weather", user_id="usr_a")
    assert not d.is_paused("weather", user_id="usr_a")
    assert d.is_paused("weather", user_id="usr_b")

    # ...resume without a user clears every tenant's pause for the verb.
    d.resume("weather")
    assert not d.is_paused("weather", user_id="usr_b")


# ── standing_tasks row carries the budget counter ─────────────────────

def test_standing_task_row_has_budget_counter():
    from augmentum.companion_runtime.standing_tasks import _row_to_task

    # 14-column row (with the new budget counter at index 13).
    row = (1, "u", "becca", "t", "feed_digest", "{}", 3600,
           None, None, None, None, 1, 2, 7)
    task = _row_to_task(row)
    assert task.consecutive_error_count == 2
    assert task.consecutive_budget_timeout_count == 7

    # Legacy 13-column row (pre-migration) defaults the budget counter.
    legacy = (1, "u", "becca", "t", "feed_digest", "{}", 3600,
              None, None, None, None, 1, 2)
    t2 = _row_to_task(legacy)
    assert t2.consecutive_budget_timeout_count == 0
