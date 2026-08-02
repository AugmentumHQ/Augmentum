"""Tests for the trusted_receivers + receiver_cast_events stores.

Pins:
  - upsert_on_connect creates a row on first connect (per device_id),
    rebinds the same row on reconnect, refuses revoked devices
  - update_label / revoke / list_for_user are user-scoped
  - cross-user reads return None / empty list (multi-tenant invariant)
  - cast event store records starts, closes by surface_id, closes by
    registration_id (disconnect path), lists recent + active correctly
  - ReceiverRegistry binds + revokes via the trusted store on ready
  - cast_routes /trusted-receivers + /trusted-receivers/{id}/revoke
    enforce auth + user-scoping; revoke closes live WS for that device
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from augmentum.cast.cast_events import (
    END_REASON_DISCONNECTED,
    END_REASON_USER_STOP,
    CastEventStore,
)
from augmentum.cast.receiver_protocol import ReceiverEvent
from augmentum.cast.receiver_registry import ReceiverRegistry
from augmentum.cast.trusted_receivers import TrustedReceiverStore


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


async def _drain_pending_tasks() -> None:
    """Yield to the event loop until every non-current task is done.

    record_event() schedules ``_bind_trusted`` as a fire-and-forget
    background task. In production a slow DB doesn't block the event
    handler; in tests we want to assert on its result, so we drain.
    """
    me = asyncio.current_task()
    for _ in range(40):  # safety cap — should resolve in 1-2 iterations
        tasks = [t for t in asyncio.all_tasks() if t is not me and not t.done()]
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def db():
    """In-memory SQLite with migrations applied + a seeded user."""
    from augmentum.auth.session_manager import SessionManager
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    _run(backend.connect())
    sm = SessionManager(backend._conn)
    alice = _run(sm.create_user("alice", "pw_for_alice_pls", role="user"))
    bob = _run(sm.create_user("bob", "pw_for_bob_pls", role="user"))
    yield backend, alice, bob
    _run(backend.close())


@pytest.fixture
def trusted_store(db):
    backend, _alice, _bob = db
    return TrustedReceiverStore(backend._conn)


@pytest.fixture
def event_store(db):
    backend, _alice, _bob = db
    return CastEventStore(backend._conn)


# ── TrustedReceiverStore ─────────────────────────────────────────


def test_upsert_creates_row_on_first_connect(trusted_store, db):
    _, alice, _bob = db
    record = _run(trusted_store.upsert_on_connect(
        user_id=alice.id, device_id="dev-1",
        platform="android-tv", info={"label": "Living Room TV"},
    ))
    assert record is not None
    assert record.user_id == alice.id
    assert record.device_id == "dev-1"
    assert record.platform == "android-tv"
    assert record.label == "Living Room TV"
    assert not record.is_revoked


def test_upsert_returns_same_row_on_reconnect(trusted_store, db):
    _, alice, _bob = db
    first = _run(trusted_store.upsert_on_connect(
        user_id=alice.id, device_id="dev-1", platform="android-tv",
    ))
    second = _run(trusted_store.upsert_on_connect(
        user_id=alice.id, device_id="dev-1", platform="android-tv-v2",
        info={"label": "ignored_because_first_set_label"},
    ))
    assert second is not None
    assert second.id == first.id
    # Reconnect refreshes platform/info but preserves the user-chosen label.
    assert second.platform == "android-tv-v2"
    assert second.label == first.label


def test_upsert_rejects_revoked_device(trusted_store, db):
    _, alice, _bob = db
    first = _run(trusted_store.upsert_on_connect(
        user_id=alice.id, device_id="dev-x", platform="android-tv",
    ))
    assert _run(trusted_store.revoke(first.id, user_id=alice.id)) is True
    again = _run(trusted_store.upsert_on_connect(
        user_id=alice.id, device_id="dev-x", platform="android-tv",
    ))
    assert again is None, "revoked devices must not auto-reconnect"


def test_get_by_device_scoped_per_user(trusted_store, db):
    _, alice, bob = db
    _run(trusted_store.upsert_on_connect(
        user_id=alice.id, device_id="dev-shared", platform="android-tv",
    ))
    assert _run(trusted_store.get_by_device("dev-shared", user_id=alice.id)) is not None
    # Bob never paired this device — must not see Alice's row.
    assert _run(trusted_store.get_by_device("dev-shared", user_id=bob.id)) is None


def test_list_excludes_revoked_by_default(trusted_store, db):
    _, alice, _bob = db
    keep = _run(trusted_store.upsert_on_connect(
        user_id=alice.id, device_id="keep", platform="x",
    ))
    drop = _run(trusted_store.upsert_on_connect(
        user_id=alice.id, device_id="drop", platform="x",
    ))
    _run(trusted_store.revoke(drop.id, user_id=alice.id))

    active_only = _run(trusted_store.list_for_user(user_id=alice.id))
    assert [r.id for r in active_only] == [keep.id]

    everything = _run(trusted_store.list_for_user(
        user_id=alice.id, include_revoked=True,
    ))
    assert {r.id for r in everything} == {keep.id, drop.id}


def test_update_label_user_scoped(trusted_store, db):
    _, alice, bob = db
    r = _run(trusted_store.upsert_on_connect(
        user_id=alice.id, device_id="dev-1", platform="x", label="OldName",
    ))
    # Bob can't rename Alice's row.
    assert _run(trusted_store.update_label(
        r.id, user_id=bob.id, label="HACKED",
    )) is False
    # But Alice can.
    assert _run(trusted_store.update_label(
        r.id, user_id=alice.id, label="Family Room",
    )) is True
    fresh = _run(trusted_store.get(r.id, user_id=alice.id))
    assert fresh.label == "Family Room"


def test_revoke_user_scoped(trusted_store, db):
    _, alice, bob = db
    r = _run(trusted_store.upsert_on_connect(
        user_id=alice.id, device_id="dev-1", platform="x",
    ))
    # Bob can't revoke Alice's receiver.
    assert _run(trusted_store.revoke(r.id, user_id=bob.id)) is False
    fresh = _run(trusted_store.get(r.id, user_id=alice.id))
    assert not fresh.is_revoked
    # Alice can.
    assert _run(trusted_store.revoke(r.id, user_id=alice.id)) is True


# ── CastEventStore ───────────────────────────────────────────────


def test_record_start_returns_id_and_persists(event_store, db):
    _, alice, _ = db
    eid = _run(event_store.record_start(
        user_id=alice.id, trusted_id="tr_a", registration_id="rcv_1",
        surface_id="sf_1", surface_kind="vrm.avatar",
        surface_url="/ui/cast-vrm/", slot="main",
    ))
    assert eid.startswith("cev_")
    rows = _run(event_store.list_recent(user_id=alice.id))
    assert len(rows) == 1
    assert rows[0].surface_id == "sf_1"
    assert rows[0].is_active


def test_mark_end_by_surface_closes_active_event(event_store, db):
    _, alice, _ = db
    _run(event_store.record_start(
        user_id=alice.id, surface_id="sf_x", surface_kind="html.generic",
    ))
    updated = _run(event_store.mark_end_by_surface(
        user_id=alice.id, surface_id="sf_x", reason=END_REASON_USER_STOP,
    ))
    assert updated is True
    rows = _run(event_store.list_recent(user_id=alice.id))
    assert len(rows) == 1
    assert not rows[0].is_active
    assert rows[0].end_reason == END_REASON_USER_STOP


def test_mark_end_by_registration_closes_all_active(event_store, db):
    _, alice, _ = db
    for sid in ("sf_a", "sf_b"):
        _run(event_store.record_start(
            user_id=alice.id, registration_id="rcv_dead",
            surface_id=sid, surface_kind="html.generic",
        ))
    _run(event_store.record_start(
        user_id=alice.id, registration_id="rcv_other",
        surface_id="sf_other", surface_kind="html.generic",
    ))
    n_closed = _run(event_store.mark_end_by_registration(
        user_id=alice.id, registration_id="rcv_dead",
        reason=END_REASON_DISCONNECTED,
    ))
    assert n_closed == 2
    rows = _run(event_store.list_recent(user_id=alice.id))
    closed_ids = {r.surface_id for r in rows if not r.is_active}
    assert closed_ids == {"sf_a", "sf_b"}


def test_list_active_for_trusted_scoped(event_store, db):
    _, alice, bob = db
    _run(event_store.record_start(
        user_id=alice.id, trusted_id="tr_alice", surface_id="sf_a",
        surface_kind="x",
    ))
    _run(event_store.record_start(
        user_id=bob.id, trusted_id="tr_alice", surface_id="sf_b",
        surface_kind="x",
    ))
    alice_active = _run(event_store.list_active_for_trusted(
        "tr_alice", user_id=alice.id,
    ))
    assert [e.surface_id for e in alice_active] == ["sf_a"]
    bob_active = _run(event_store.list_active_for_trusted(
        "tr_alice", user_id=bob.id,
    ))
    assert [e.surface_id for e in bob_active] == ["sf_b"]


# ── ReceiverRegistry integration ─────────────────────────────────


class _FakeWS:
    """Minimal stand-in for fastapi.WebSocket for registry tests."""
    def __init__(self) -> None:
        self.closed = False
        self.close_code: int | None = None
        self.sent: list[Any] = []

    async def send_json(self, data: Any) -> None:
        self.sent.append(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = True
        self.close_code = code


def test_registry_binds_trusted_on_ready(trusted_store, event_store, db):
    _, alice, _ = db
    registry = ReceiverRegistry(
        trusted_store=trusted_store, event_store=event_store,
    )

    async def _exercise():
        ws = _FakeWS()
        rec = registry.attach(ws=ws, user_id=alice.id)
        registry.record_event(rec.registration_id, ReceiverEvent(
            event="ready",
            data={
                "platform": "android-tv",
                "device_id": "dev-binding",
                "label": "Family Room",
            },
        ))
        # _bind_trusted runs as a background task — let it complete.
        await _drain_pending_tasks()
        return rec

    rec = _run(_exercise())
    refreshed = registry.get(rec.registration_id)
    assert refreshed.trusted_id.startswith("tr_")
    stored = _run(trusted_store.get(refreshed.trusted_id, user_id=alice.id))
    assert stored is not None
    assert stored.device_id == "dev-binding"


def test_registry_kicks_revoked_device(trusted_store, event_store, db):
    _, alice, _ = db
    seeded = _run(trusted_store.upsert_on_connect(
        user_id=alice.id, device_id="dev-bad", platform="android-tv",
    ))
    _run(trusted_store.revoke(seeded.id, user_id=alice.id))

    registry = ReceiverRegistry(
        trusted_store=trusted_store, event_store=event_store,
    )

    async def _exercise():
        ws = _FakeWS()
        rec = registry.attach(ws=ws, user_id=alice.id)
        registry.record_event(rec.registration_id, ReceiverEvent(
            event="ready",
            data={"platform": "android-tv", "device_id": "dev-bad"},
        ))
        await _drain_pending_tasks()
        return ws, rec

    ws, rec = _run(_exercise())
    assert ws.closed, "revoked device must be hung up"
    assert ws.close_code == 4003
    assert registry.get(rec.registration_id) is None
    # The receiver must get a structured ``revoked`` cmd before the
    # close so it can render a terminal placeholder instead of looping
    # the QR re-pair. Without this, a revoked TV ping-pongs between
    # pair/start and 4003 with no on-screen explanation.
    revoke_cmds = [m for m in ws.sent if m.get("cmd") == "revoked"]
    assert len(revoke_cmds) == 1, ws.sent
    assert revoke_cmds[0]["args"]["trusted_id"] == seeded.id
    assert revoke_cmds[0]["args"]["reason"] == "revoked"


def test_registry_detach_closes_active_events(trusted_store, event_store, db):
    _, alice, _ = db
    registry = ReceiverRegistry(
        trusted_store=trusted_store, event_store=event_store,
    )

    async def _exercise():
        ws = _FakeWS()
        rec = registry.attach(ws=ws, user_id=alice.id)
        await event_store.record_start(
            user_id=alice.id, registration_id=rec.registration_id,
            surface_id="sf_live", surface_kind="vrm.avatar",
        )
        assert registry.detach(rec.registration_id) is True
        await _drain_pending_tasks()
        return rec

    _run(_exercise())
    rows = _run(event_store.list_recent(user_id=alice.id))
    assert len(rows) == 1
    assert not rows[0].is_active
    assert rows[0].end_reason == END_REASON_DISCONNECTED


# ── Route-level smoke ────────────────────────────────────────────


def _make_app_with_user_scope(
    backend, trusted_store, event_store, registry, *, user,
):
    """Stand up a FastAPI app with the cast router + a stub
    auth-middleware that pins request.scope['user'] to ``user``."""
    from augmentum.proxy.cast_routes import router as cast_router

    app = FastAPI()
    app.state.trusted_receiver_store = trusted_store
    app.state.cast_event_store = event_store
    app.state.receiver_registry = registry

    @app.middleware("http")
    async def _stub_auth(request, call_next):
        request.scope["user"] = user
        return await call_next(request)

    app.include_router(cast_router)
    return app


def test_list_trusted_receivers_endpoint_enriches_with_connection(
    trusted_store, event_store, db,
):
    backend, alice, _bob = db
    registry = ReceiverRegistry(
        trusted_store=trusted_store, event_store=event_store,
    )

    async def _setup():
        ws = _FakeWS()
        rec = registry.attach(ws=ws, user_id=alice.id)
        registry.record_event(rec.registration_id, ReceiverEvent(
            event="ready",
            data={"platform": "android-tv", "device_id": "dev-route"},
        ))
        await _drain_pending_tasks()
        return rec

    rec = _run(_setup())

    app = _make_app_with_user_scope(
        backend, trusted_store, event_store, registry, user=alice,
    )
    client = TestClient(app)
    r = client.get("/api/cast/trusted-receivers")
    assert r.status_code == 200
    body = r.json()
    assert len(body["receivers"]) == 1
    row = body["receivers"][0]
    assert row["connected"] is True
    assert row["registration_id"] == rec.registration_id


def test_revoke_endpoint_closes_live_ws_and_marks_revoked(
    trusted_store, event_store, db,
):
    backend, alice, _bob = db
    registry = ReceiverRegistry(
        trusted_store=trusted_store, event_store=event_store,
    )
    holder: dict[str, Any] = {}

    async def _setup():
        ws = _FakeWS()
        rec = registry.attach(ws=ws, user_id=alice.id)
        registry.record_event(rec.registration_id, ReceiverEvent(
            event="ready",
            data={"platform": "android-tv", "device_id": "dev-revoke"},
        ))
        await _drain_pending_tasks()
        holder["ws"] = ws
        holder["rec"] = rec

    _run(_setup())
    ws = holder["ws"]
    rec = holder["rec"]
    trusted_id = registry.get(rec.registration_id).trusted_id

    app = _make_app_with_user_scope(
        backend, trusted_store, event_store, registry, user=alice,
    )
    client = TestClient(app)
    r = client.post(f"/api/cast/trusted-receivers/{trusted_id}/revoke")
    assert r.status_code == 200
    body = r.json()
    assert body["revoked"] is True
    assert body["closed_connections"] >= 1
    assert ws.closed
    fresh = _run(trusted_store.get(trusted_id, user_id=alice.id))
    assert fresh.is_revoked


def test_revoke_endpoint_rejects_cross_user(trusted_store, event_store, db):
    backend, alice, bob = db
    r = _run(trusted_store.upsert_on_connect(
        user_id=alice.id, device_id="dev-cross", platform="x",
    ))
    registry = ReceiverRegistry(
        trusted_store=trusted_store, event_store=event_store,
    )
    app = _make_app_with_user_scope(
        backend, trusted_store, event_store, registry, user=bob,
    )
    client = TestClient(app)
    resp = client.post(f"/api/cast/trusted-receivers/{r.id}/revoke")
    assert resp.status_code == 404
    # And Alice's row remains intact.
    refreshed = _run(trusted_store.get(r.id, user_id=alice.id))
    assert not refreshed.is_revoked
