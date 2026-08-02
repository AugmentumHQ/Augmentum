"""Live reshape integration — the exact path POST /api/selfedit/reshape drives.

Proves the config/Adaptation surface works end to end through the real wiring
(register_default_surfaces + run_reshape_request + build_store_recorder): an ask
to set a per-user value applies instantly, the mechanical read-back oracle earns
VERIFIED, it auto-promotes, lands in the never-pruned archive, and reverts
exactly. This is a real self-edit running with no container/token/model.
"""

from __future__ import annotations

import pathlib

import aiosqlite
import pytest

from augmentum.selfedit import store
from augmentum.selfedit.surfaces import base as sbase
from augmentum.selfedit.surfaces.base import ReshapeChange
from augmentum.selfedit.surfaces.engine import (
    STATUS_PROMOTED,
    ReshapeRequest,
    run_reshape_request,
)
from augmentum.selfedit.surfaces.live import build_store_recorder, register_default_surfaces

_MIGRATION = (
    pathlib.Path(__file__).resolve().parent.parent
    / "augmentum" / "state" / "migrations" / "288_self_edit_attempts.sql"
)


class _FakeSettingsStore:
    """Stand-in for SettingsStore's per-user get/set (a plain dict)."""
    def __init__(self):
        self.data: dict[tuple[str, str], object] = {}

    async def get_user(self, user_id: str, key: str):
        return self.data.get((user_id, key))

    async def set_user(self, user_id: str, key: str, value):
        self.data[(user_id, key)] = value


async def _db():
    from augmentum.selfedit.growth_db import _ensure_columns
    conn = await aiosqlite.connect(":memory:")
    await conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY, description TEXT)")
    await conn.executescript(_MIGRATION.read_text())
    await _ensure_columns(conn)  # mirror the live growth-DB open (post-288 columns)
    await conn.commit()
    return conn


@pytest.fixture(autouse=True)
def _clean_registry():
    sbase.clear_surfaces()
    yield
    sbase.clear_surfaces()


async def test_config_reshape_verifies_applies_and_archives():
    fs = _FakeSettingsStore()
    ledger: dict = {}
    register_default_surfaces(fs, revert_ledger=ledger)
    conn = await _db()
    try:
        change = ReshapeChange(surface="config", change_class="adaptation",
                               payload={"key": "ui_density", "value": "compact"},
                               intent="make it denser", actor="u1")

        async def classify(_req, _surfaces):
            return change

        on_start, on_finish = build_store_recorder(conn)
        res = await run_reshape_request(
            ReshapeRequest(ask="make it denser", actor="u1", surface_hint="config"),
            classify=classify, on_start=on_start, on_finish=on_finish,
        )
        # verified-by-construction → auto-promoted
        assert res.mapped is True and res.status == STATUS_PROMOTED
        assert res.reshape.auto_promotable is True and res.reshape.kept is True
        # the value actually landed, scoped to the user
        assert fs.data[("u1", "ui_density")] == "compact"
        # recorded in the never-pruned archive as promoted
        row = await store.get_attempt(conn, attempt_id=res.attempt_id, user_id="u1")
        assert row is not None and row["status"] == "promoted" and row["surface"] == "config"
        # revert restores the prior value exactly
        token = res.reshape.revert_token
        adapter = sbase.get_surface("config")
        assert await adapter.revert(token) is True
        assert fs.data[("u1", "ui_density")] is None  # prior was unset
    finally:
        await conn.close()


async def test_reshape_unmapped_surface_is_handled():
    register_default_surfaces(_FakeSettingsStore())
    conn = await _db()
    try:
        async def classify(_req, _surfaces):
            return None  # classifier couldn't map the ask
        on_start, on_finish = build_store_recorder(conn)
        res = await run_reshape_request(
            ReshapeRequest(ask="do something impossible", actor="u1"),
            classify=classify, on_start=on_start, on_finish=on_finish,
        )
        assert res.mapped is False and res.status == "unmapped"
    finally:
        await conn.close()


async def test_reshape_empty_actor_refused():
    fs = _FakeSettingsStore()
    register_default_surfaces(fs)
    conn = await _db()
    try:
        # actor empty → the config adapter refuses (never writes the shared row)
        change = ReshapeChange(surface="config", change_class="adaptation",
                               payload={"key": "k", "value": "v"}, actor="")

        async def classify(_req, _surfaces):
            return change

        on_start, on_finish = build_store_recorder(conn)
        res = await run_reshape_request(
            ReshapeRequest(ask="x", actor=""), classify=classify,
            on_start=on_start, on_finish=on_finish,
        )
        assert res.reshape.applied is False
        assert fs.data == {}  # nothing written
    finally:
        await conn.close()
