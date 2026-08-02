"""Companion working-set persistence — continuity across restarts.

Pins the 2026-06-11 continuity slice: the ReferentCache working set
(active note, trail, dispatch anchors) write-throughs to the EXISTING
per-user settings store and rehydrates fresh caches — no new table.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from augmentum.companion_runtime.working_state import (
    hydrate_working_state,
    save_working_state,
)
from augmentum.intent.action import ReferentCache


class _FakeSettingsStore:
    def __init__(self):
        self.data = {}

    async def get_user(self, user_id, key):
        return self.data.get((user_id, key))

    async def set_user(self, user_id, key, value):
        self.data[(user_id, key)] = value


def _state():
    return SimpleNamespace(settings_store=_FakeSettingsStore())


@pytest.mark.asyncio
async def test_save_then_hydrate_fresh_cache():
    state = _state()
    refs = ReferentCache()
    refs.active_note_id = "n42"
    refs.active_note_title = "Trip ideas"
    refs.trail = [{"kind": "page", "label": "Kyoto guide", "ref": "https://x", "ts": 1.0}]
    refs.last_dispatch_action = "note.append"
    await save_working_state(state, "u1", refs)

    # Simulate restart / new voice session id: brand-new cache.
    fresh = ReferentCache()
    fresh.trail = []
    await hydrate_working_state(state, "u1", fresh)
    assert fresh.active_note_id == "n42"
    assert fresh.active_note_title == "Trip ideas"
    assert fresh.trail[0]["label"] == "Kyoto guide"
    assert fresh.last_dispatch_action == "note.append"


@pytest.mark.asyncio
async def test_hydrate_never_overwrites_live_state():
    state = _state()
    stale = ReferentCache()
    stale.active_note_id = "old_note"
    await save_working_state(state, "u1", stale)

    live = ReferentCache()
    live.active_note_id = "current_note"
    live.trail = [{"kind": "search", "label": "fresh", "ref": "", "ts": 2.0}]
    await hydrate_working_state(state, "u1", live)
    # In-memory state wins over the snapshot.
    assert live.active_note_id == "current_note"
    assert live.trail[0]["label"] == "fresh"


@pytest.mark.asyncio
async def test_hydrate_is_idempotent_per_cache():
    state = _state()
    refs = ReferentCache()
    refs.active_note_id = "n1"
    await save_working_state(state, "u1", refs)

    fresh = ReferentCache()
    fresh.trail = []
    await hydrate_working_state(state, "u1", fresh)
    # Mutate the store after first hydrate — second call must not re-read.
    other = ReferentCache()
    other.active_note_id = "n2"
    await save_working_state(state, "u1", other)
    fresh.active_note_id = ""
    await hydrate_working_state(state, "u1", fresh)
    assert fresh.active_note_id == ""  # second hydrate was a no-op


@pytest.mark.asyncio
async def test_per_user_isolation():
    state = _state()
    a = ReferentCache()
    a.active_note_id = "alice_note"
    await save_working_state(state, "alice", a)

    b = ReferentCache()
    b.trail = []
    await hydrate_working_state(state, "bob", b)
    assert not b.active_note_id


@pytest.mark.asyncio
async def test_soft_fail_on_missing_store_and_user():
    refs = ReferentCache()
    await save_working_state(None, "u1", refs)
    await hydrate_working_state(None, "u1", refs)
    await save_working_state(SimpleNamespace(), "", refs)
    await hydrate_working_state(SimpleNamespace(settings_store=None), "u1", ReferentCache())
    # No exceptions = pass.


@pytest.mark.asyncio
async def test_corrupt_blob_soft_fails():
    state = _state()
    state.settings_store.data[("u1", "companion.working_state")] = "{not json"
    fresh = ReferentCache()
    await hydrate_working_state(state, "u1", fresh)
    assert not fresh.active_note_id
