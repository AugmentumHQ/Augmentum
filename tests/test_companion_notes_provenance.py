"""Notes v2 Phase 1 — provenance substrate.

Spec: docs/superpowers/specs/2026-06-10-notes-v2-useful-first-design.md §Move 3.

Covers:
* Client filter in the topical aggregator — cast_receiver events never
  form attention threads (the 2026-06-08 incident's structural fix)
* Legacy events without a ``client`` field count as "web"
* Thread.clients carries the distinct sources, dominant first
* ``companion_attention_sources`` parsing (blank falls back to the
  default rather than allow-all)
* Wondering origin record shape ("from browsing (web) · 3 visits" fuel)
* origin_json survives the safe_journal → companion_journal round-trip
  and decodes through the notes-API helper
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest


def _evt(url: str, *, user_id: str = "u1", client: str | None = None,
         t: float | None = None) -> dict:
    payload: dict = {"url": url, "user_id": user_id}
    if client is not None:
        payload["client"] = client
    return {
        "topic": "surface.browse.opened",
        "payload": payload,
        "t": t if t is not None else time.time(),
    }


# ── Aggregator client filter ─────────────────────────────────────────


def test_cast_receiver_events_form_no_threads():
    """The incident gate: a shared TV logged in as you must not write
    your attention stream."""
    from augmentum.companion_runtime.perception.topical import aggregate_threads
    now = time.time()
    events = [
        _evt(f"https://example.com/{i}", client="cast_receiver", t=now - i * 60)
        for i in range(4)
    ]
    threads = aggregate_threads(
        events, user_id="u1", min_events=3, now=now,
        allowed_clients=frozenset({"web", "android"}),
    )
    assert threads == []


def test_missing_client_counts_as_web():
    """Events written before the emit sites carried ``client`` keep
    forming threads under the default filter."""
    from augmentum.companion_runtime.perception.topical import aggregate_threads
    now = time.time()
    events = [
        _evt(f"https://example.com/{i}", t=now - i * 60) for i in range(3)
    ]
    threads = aggregate_threads(
        events, user_id="u1", min_events=3, now=now,
        allowed_clients=frozenset({"web", "android"}),
    )
    assert len(threads) == 1
    assert threads[0].clients == ("web",)


def test_mixed_clients_only_allowed_events_counted():
    """3 web + 2 cast_receiver on one domain → thread of exactly the
    3 web events; the TV's contribution is invisible."""
    from augmentum.companion_runtime.perception.topical import aggregate_threads
    now = time.time()
    events = [
        _evt(f"https://example.com/w{i}", client="web", t=now - i * 60)
        for i in range(3)
    ] + [
        _evt(f"https://example.com/c{i}", client="cast_receiver", t=now - i * 30)
        for i in range(2)
    ]
    threads = aggregate_threads(
        events, user_id="u1", min_events=3, now=now,
        allowed_clients=frozenset({"web", "android"}),
    )
    assert len(threads) == 1
    assert threads[0].event_count == 3
    assert threads[0].clients == ("web",)


def test_no_filter_keeps_all_and_ranks_clients_dominant_first():
    from augmentum.companion_runtime.perception.topical import aggregate_threads
    now = time.time()
    events = [
        _evt(f"https://example.com/a{i}", client="android", t=now - i * 60)
        for i in range(3)
    ] + [
        _evt("https://example.com/w0", client="web", t=now - 10),
    ]
    threads = aggregate_threads(
        events, user_id="u1", min_events=3, now=now, allowed_clients=None,
    )
    assert len(threads) == 1
    assert threads[0].event_count == 4
    assert threads[0].clients[0] == "android"
    assert set(threads[0].clients) == {"android", "web"}


# ── Setting parsing ──────────────────────────────────────────────────


def test_allowed_attention_clients_default(monkeypatch):
    from augmentum.companion_runtime.wondering import allowed_attention_clients
    from augmentum.config import settings
    monkeypatch.setattr(
        settings, "companion_attention_sources", "web,android", raising=False,
    )
    assert allowed_attention_clients() == frozenset({"web", "android"})


def test_allowed_attention_clients_custom_and_normalized(monkeypatch):
    from augmentum.companion_runtime.wondering import allowed_attention_clients
    from augmentum.config import settings
    monkeypatch.setattr(
        settings, "companion_attention_sources", " Web , CAST_RECEIVER ",
        raising=False,
    )
    assert allowed_attention_clients() == frozenset({"web", "cast_receiver"})


def test_allowed_attention_clients_blank_falls_back_not_allow_all(monkeypatch):
    """Clearing the field must not silently widen the privacy boundary."""
    from augmentum.companion_runtime.wondering import allowed_attention_clients
    from augmentum.config import settings
    monkeypatch.setattr(
        settings, "companion_attention_sources", "", raising=False,
    )
    assert allowed_attention_clients() == frozenset({"web", "android"})


# ── Wondering origin record ──────────────────────────────────────────


def test_thread_origin_shape():
    from augmentum.companion_runtime.perception.topical import Thread
    from augmentum.companion_runtime.wondering import _thread_origin

    now = time.time()
    thread = Thread(
        topic="example.com",
        event_ids=("1", "2", "3"),
        domains=("example.com",),
        keywords=("rust", "async"),
        first_seen=now - 600,
        last_seen=now,
        event_count=3,
        clients=("web",),
    )
    origin = _thread_origin(thread)
    assert origin["source"] == "attention"
    assert origin["client"] == "web"
    assert origin["signal_count"] == 3
    assert origin["detail"] == "browse: example.com x3"
    # Window is "YYYY-MM-DDTHH:MM/HH:MM" when both ends share a day
    assert "/" in origin["window"]
    assert origin["window"].startswith("20")


def test_thread_origin_tolerates_missing_clients():
    """Threads built by older callers (no clients field) still produce
    a well-formed origin."""
    from augmentum.companion_runtime.perception.topical import Thread
    from augmentum.companion_runtime.wondering import _thread_origin

    now = time.time()
    thread = Thread(
        topic="example.com", event_ids=("1",), domains=("example.com",),
        keywords=(), first_seen=now, last_seen=now, event_count=1,
    )
    origin = _thread_origin(thread)
    assert origin["client"] == "web"


# ── DB round-trip ────────────────────────────────────────────────────


async def _boot_backend():
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        ("usr_test", "tester", "x"),
    )
    await backend.conn.commit()
    return backend


@pytest.mark.asyncio
async def test_safe_journal_persists_origin_json():
    from augmentum.companion_runtime.memory import CompanionMemory

    backend = await _boot_backend()
    memory = CompanionMemory(backend, "becca")
    origin = {
        "source": "attention", "client": "web", "signal_count": 3,
        "window": "2026-06-10T06:34/06:40",
        "detail": "browse: example.com x3",
    }
    journal_id = await memory.safe_journal(
        "A few returns to example.com today, mostly rust and async.",
        source="autonomous",
        user_id="usr_test",
        entry_type="wondering",
        affect_tag="curious",
        embed=False,
        origin=origin,
    )
    assert journal_id

    cur = await backend.conn.execute(
        "SELECT origin_json, COALESCE(quarantined, 0) FROM companion_journal "
        "WHERE id = ?",
        (journal_id,),
    )
    row = await cur.fetchone()
    await cur.close()
    assert row is not None
    assert int(row[1]) == 0
    assert json.loads(row[0]) == origin


@pytest.mark.asyncio
async def test_quarantined_write_keeps_origin_for_forensics():
    from augmentum.companion_runtime.memory import CompanionMemory

    backend = await _boot_backend()
    memory = CompanionMemory(backend, "becca")
    origin = {"source": "attention", "client": "cast_receiver",
              "signal_count": 4, "window": "", "detail": "browse: bad.example x4"}
    # Too short → structural quarantine; the origin must still land.
    journal_id = await memory.safe_journal(
        "x", source="autonomous", user_id="usr_test",
        entry_type="wondering", embed=False, origin=origin,
    )
    cur = await backend.conn.execute(
        "SELECT origin_json, COALESCE(quarantined, 0) FROM companion_journal "
        "WHERE id = ?",
        (journal_id,),
    )
    row = await cur.fetchone()
    await cur.close()
    assert row is not None
    assert int(row[1]) == 1
    assert json.loads(row[0])["client"] == "cast_receiver"


def test_decode_origin_contract():
    """API helper: empty dict for NULL / garbage, dict passthrough."""
    from augmentum.proxy.companion_routes import _decode_origin

    assert _decode_origin(None) == {}
    assert _decode_origin("") == {}
    assert _decode_origin("not json") == {}
    assert _decode_origin('["list"]') == {}
    assert _decode_origin('{"source": "attention"}') == {"source": "attention"}


# ── Wondering writer passes origin through (unit, mocked memory) ─────


@pytest.mark.asyncio
async def test_wondering_write_includes_origin(monkeypatch):
    """maybe_write_wondering's safe_journal call carries the origin
    built from the selected thread."""
    from augmentum.companion_runtime import wondering
    from augmentum.companion_runtime.perception.topical import Thread
    from augmentum.config import settings

    monkeypatch.setattr(
        settings, "companion_topical_aggregator_enabled", True, raising=False,
    )

    now = time.time()
    thread = Thread(
        topic="example.com", event_ids=("1", "2", "3"),
        domains=("example.com",), keywords=("rust",),
        first_seen=now - 300, last_seen=now, event_count=3,
        clients=("android",),
    )

    runtime = MagicMock()
    runtime.companion_id = "becca"

    captured: dict = {}

    async def fake_safe_journal(content, **kwargs):
        captured.update(kwargs)
        return 42

    runtime.memory.safe_journal = fake_safe_journal

    async def _none(*a, **k):
        return None

    async def _zero(*a, **k):
        return 0

    async def _false(*a, **k):
        return False

    monkeypatch.setattr(wondering, "_count_wonderings_today", _zero)
    monkeypatch.setattr(wondering, "_is_topic_muted", _false)
    monkeypatch.setattr(wondering, "_existing_wondering_for_topic", _none)
    monkeypatch.setattr(wondering, "_curiosity_elevated", _false)

    from augmentum.companion_runtime import gates, presence_mode
    monkeypatch.setattr(presence_mode, "autonomy_allowed", lambda: True)
    monkeypatch.setattr(gates, "is_hushed_now", lambda: False)
    monkeypatch.setattr(gates, "is_user_recently_active", lambda rt: False)

    journal_id = await wondering.maybe_write_wondering(
        runtime, user_id="usr_test", threads=[thread], now=now,
    )
    assert journal_id == 42
    origin = captured.get("origin")
    assert origin is not None
    assert origin["source"] == "attention"
    assert origin["client"] == "android"
    assert origin["signal_count"] == 3
