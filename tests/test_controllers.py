"""Tests for the AXF controller framework (Phase E).

Covers:
* Defaults catalog completeness (all systems have d-pad + at least one face button)
* ControllerStore CRUD + user isolation + invalid pad_routing rejection
* ControllerService resolution: defaults-only, override merge, reset
* Route shapes: profiles list, resolve, GET/PUT/DELETE remap, master toggle
"""

from __future__ import annotations

import asyncio

import aiosqlite
import pytest
from fastapi.testclient import TestClient

_SCHEMA_SQL = """
CREATE TABLE users (id TEXT PRIMARY KEY);

CREATE TABLE controller_remaps (
    user_id         TEXT NOT NULL REFERENCES users(id),
    system_id       TEXT NOT NULL,
    bindings_json   TEXT NOT NULL DEFAULT '{}',
    pad_routing     TEXT NOT NULL DEFAULT 'index',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, system_id)
);
"""


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


async def _mkdb():
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA_SQL)
    await conn.execute("INSERT INTO users (id) VALUES ('usr_test')")
    await conn.execute("INSERT INTO users (id) VALUES ('usr_other')")
    await conn.commit()
    return conn


# ── Defaults catalog ─────────────────────────────────────────────────


def test_defaults_catalog_is_non_empty():
    from augmentum.controllers import SYSTEM_PROFILES, list_systems

    profiles = list_systems()
    assert profiles == list(SYSTEM_PROFILES)
    assert len(profiles) >= 12, "expected ~12 retro systems shipped"


def test_each_profile_has_dpad_and_action_button():
    """Sanity: every retro system needs a directional + at least one face."""
    from augmentum.controllers import list_systems

    for prof in list_systems():
        # The d-pad helper namespaces ids by prefix; just check that
        # SOME action contains "_up" / "_down" / "_left" / "_right".
        ids = set(prof.actions.keys())
        for direction in ("_up", "_down", "_left", "_right"):
            matching = [a for a in ids if a.endswith(direction)]
            assert matching, f"{prof.id}: no {direction} action"
        non_dpad = [
            a for a in ids
            if not any(a.endswith(d) for d in ("_up", "_down", "_left", "_right"))
        ]
        assert non_dpad, f"{prof.id}: no face/system buttons"


def test_get_system_profile_known_and_unknown():
    from augmentum.controllers import get_system_profile

    nes = get_system_profile("nes")
    assert nes is not None and nes.label.startswith("Nintendo")
    assert get_system_profile("not-a-system") is None


# ── Store ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_store_upsert_and_get():
    from augmentum.controllers import ControllerStore

    conn = await _mkdb()
    store = ControllerStore(conn)
    rec = await store.upsert(
        user_id="usr_test", system_id="nes",
        bindings={"nes_a": {"keyboard": "KeyB"}},
    )
    assert rec.user_id == "usr_test"
    assert rec.system_id == "nes"
    assert rec.bindings["nes_a"]["keyboard"] == "KeyB"
    fetched = await store.get(user_id="usr_test", system_id="nes")
    assert fetched is not None and fetched.bindings == rec.bindings


@pytest.mark.asyncio
async def test_store_user_isolation():
    from augmentum.controllers import ControllerStore

    conn = await _mkdb()
    store = ControllerStore(conn)
    await store.upsert(user_id="usr_test", system_id="nes",
                       bindings={"nes_a": {"keyboard": "KeyB"}})
    other = await store.get(user_id="usr_other", system_id="nes")
    assert other is None


@pytest.mark.asyncio
async def test_store_rejects_invalid_pad_routing():
    from augmentum.controllers import ControllerStore

    conn = await _mkdb()
    store = ControllerStore(conn)
    with pytest.raises(ValueError):
        await store.upsert(
            user_id="usr_test", system_id="nes",
            pad_routing="bogus",
        )


@pytest.mark.asyncio
async def test_store_delete_returns_truthiness():
    from augmentum.controllers import ControllerStore

    conn = await _mkdb()
    store = ControllerStore(conn)
    await store.upsert(user_id="usr_test", system_id="nes")
    assert await store.delete(user_id="usr_test", system_id="nes") is True
    assert await store.delete(user_id="usr_test", system_id="nes") is False


# ── Service ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_no_override_returns_defaults():
    from augmentum.controllers import (
        ControllerService,
        ControllerStore,
        get_system_profile,
    )

    conn = await _mkdb()
    svc = ControllerService(store=ControllerStore(conn))
    layout = await svc.resolve(user_id="usr_test", system_id="nes")
    assert layout is not None
    nes_default = get_system_profile("nes")
    # Every action's resolved binding equals the default
    for aid, default in nes_default.actions.items():
        resolved = layout.actions[aid]
        assert resolved["keyboard"] == default.keyboard
        assert resolved["gamepad_button"] == default.gamepad_button


@pytest.mark.asyncio
async def test_resolve_override_wins_per_action():
    from augmentum.controllers import ControllerService, ControllerStore

    conn = await _mkdb()
    store = ControllerStore(conn)
    svc = ControllerService(store=store)
    await store.upsert(
        user_id="usr_test", system_id="nes",
        bindings={"nes_a": {"keyboard": "KeyJ", "gamepad_button": 7}},
    )
    layout = await svc.resolve(user_id="usr_test", system_id="nes")
    a = layout.actions["nes_a"]
    assert a["keyboard"] == "KeyJ"
    assert a["gamepad_button"] == 7
    # Other actions still inherit defaults
    b = layout.actions["nes_b"]
    assert b["keyboard"] == "KeyX"


@pytest.mark.asyncio
async def test_resolve_unknown_system_returns_none():
    from augmentum.controllers import ControllerService, ControllerStore

    conn = await _mkdb()
    svc = ControllerService(store=ControllerStore(conn))
    assert await svc.resolve(user_id="usr_test", system_id="bogus") is None


@pytest.mark.asyncio
async def test_service_rejects_unknown_actions_in_override():
    from augmentum.controllers import ControllerService, ControllerStore

    conn = await _mkdb()
    svc = ControllerService(store=ControllerStore(conn))
    with pytest.raises(ValueError):
        await svc.update_remap(
            user_id="usr_test", system_id="nes",
            bindings={"not_an_action": {"keyboard": "KeyZ"}},
        )


@pytest.mark.asyncio
async def test_reset_returns_to_defaults():
    from augmentum.controllers import ControllerService, ControllerStore

    conn = await _mkdb()
    store = ControllerStore(conn)
    svc = ControllerService(store=store)
    await svc.update_remap(
        user_id="usr_test", system_id="nes",
        bindings={"nes_a": {"keyboard": "KeyJ"}},
    )
    assert (await svc.resolve(
        user_id="usr_test", system_id="nes",
    )).actions["nes_a"]["keyboard"] == "KeyJ"
    await svc.reset_remap(user_id="usr_test", system_id="nes")
    layout = await svc.resolve(user_id="usr_test", system_id="nes")
    assert layout.actions["nes_a"]["keyboard"] == "KeyZ"  # canonical default


# ── Routes ──────────────────────────────────────────────────────────


@pytest.fixture
def controllers_client(app, monkeypatch):
    from augmentum.config import settings as config_settings
    from augmentum.controllers import ControllerService, ControllerStore

    monkeypatch.setattr(config_settings, "controller_remap_enabled", True)
    conn = _run(aiosqlite.connect(":memory:"))
    _run(conn.executescript(_SCHEMA_SQL))
    _run(conn.execute("INSERT INTO users (id) VALUES ('usr_test')"))
    _run(conn.execute("INSERT INTO users (id) VALUES ('usr_other')"))
    _run(conn.commit())
    store = ControllerStore(conn)
    svc = ControllerService(store=store)
    app.state.controller_store = store
    app.state.controller_service = svc

    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    yield tc, svc, conn
    _run(conn.close())


def test_route_profiles_list(controllers_client):
    client, *_ = controllers_client
    r = client.get("/api/controllers/profiles")
    assert r.status_code == 200
    ids = {p["id"] for p in r.json()["profiles"]}
    assert "nes" in ids and "snes" in ids and "psx" in ids


def test_route_resolve(controllers_client):
    client, *_ = controllers_client
    r = client.get("/api/controllers/nes")
    assert r.status_code == 200
    layout = r.json()["layout"]
    assert layout["system_id"] == "nes"
    assert "nes_a" in layout["actions"]


def test_route_resolve_unknown_system_404(controllers_client):
    client, *_ = controllers_client
    r = client.get("/api/controllers/no-system")
    assert r.status_code == 404


def test_route_get_remap_empty(controllers_client):
    client, *_ = controllers_client
    r = client.get("/api/controllers/nes/remap")
    assert r.status_code == 200
    body = r.json()
    assert body["bindings"] == {}
    assert body["pad_routing"] == "index"


def test_route_put_remap_persists(controllers_client):
    client, *_ = controllers_client
    r = client.put("/api/controllers/nes/remap", json={
        "bindings": {"nes_a": {"keyboard": "KeyJ", "gamepad_button": 7}},
        "pad_routing": "firstpress",
    })
    assert r.status_code == 200
    remap = r.json()["remap"]
    assert remap["pad_routing"] == "firstpress"
    # Resolved layout reflects the override
    rr = client.get("/api/controllers/nes").json()["layout"]
    assert rr["actions"]["nes_a"]["keyboard"] == "KeyJ"


def test_route_put_remap_rejects_unknown_action(controllers_client):
    client, *_ = controllers_client
    r = client.put("/api/controllers/nes/remap", json={
        "bindings": {"bogus_action": {"keyboard": "KeyZ"}},
    })
    assert r.status_code == 400


def test_route_delete_remap_resets(controllers_client):
    client, *_ = controllers_client
    client.put("/api/controllers/nes/remap", json={
        "bindings": {"nes_a": {"keyboard": "KeyJ"}},
    })
    r = client.delete("/api/controllers/nes/remap")
    assert r.status_code == 200
    layout = client.get("/api/controllers/nes").json()["layout"]
    assert layout["actions"]["nes_a"]["keyboard"] == "KeyZ"  # canonical


def test_route_master_toggle_blocks_writes(app, monkeypatch):
    """When controller_remap_enabled=false, PUT/DELETE return 503;
    GETs still serve the catalog (useful for the UI to render hints)."""
    from augmentum.config import settings as config_settings
    from augmentum.controllers import ControllerService, ControllerStore

    monkeypatch.setattr(config_settings, "controller_remap_enabled", False)
    conn = _run(aiosqlite.connect(":memory:"))
    _run(conn.executescript(_SCHEMA_SQL))
    _run(conn.execute("INSERT INTO users (id) VALUES ('usr_test')"))
    _run(conn.commit())
    app.state.controller_store = ControllerStore(conn)
    app.state.controller_service = ControllerService(
        store=app.state.controller_store,
    )
    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})

    # Reads still work
    assert tc.get("/api/controllers/profiles").status_code == 200
    assert tc.get("/api/controllers/nes").status_code == 200
    # Writes blocked
    assert tc.put("/api/controllers/nes/remap", json={"bindings": {}}).status_code == 503
    assert tc.delete("/api/controllers/nes/remap").status_code == 503
    _run(conn.close())
