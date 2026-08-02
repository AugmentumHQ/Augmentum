"""Per-user resolution of image panel settings.

Regression: the image_generation tool / narrative scene-gen read the
process-global ``image_active_settings`` mirror, which only the anon/startup
path writes. For an authenticated user that ignored their panel selection and
fell back to the install default — the "uses the last-installed model, not the
one I selected" bug. ``resolve_active_settings`` must read per-user.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from augmentum.image.active_settings import resolve_active_settings


class _Store:
    def __init__(self, per_user: dict[str, str] | None = None):
        self._per_user = per_user or {}

    async def get_user(self, uid: str, key: str) -> str:
        return self._per_user.get(uid, "")


def _app(*, by_user=None, store=None, global_settings=None):
    return SimpleNamespace(
        _image_active_settings_by_user=by_user if by_user is not None else {},
        settings_store=store,
        image_active_settings=global_settings if global_settings is not None else {},
    )


@pytest.mark.asyncio
async def test_authenticated_selection_beats_global_last_installed():
    # The core bug: global mirror points at the last-installed model, the user
    # selected a different one in the panel. The user's choice must win.
    app = _app(
        by_user={"u1": {"model": "my-selected-sdxl"}},
        global_settings={"model": "last-installed-flux"},
    )
    ui = await resolve_active_settings(app, "u1")
    assert ui["model"] == "my-selected-sdxl"


@pytest.mark.asyncio
async def test_authenticated_falls_back_to_persisted_row_after_restart():
    # In-memory cache empty (post-restart) → load THIS user's persisted row,
    # never the global mirror.
    store = _Store({"u1": json.dumps({"model": "persisted-pick"})})
    app = _app(by_user={}, store=store, global_settings={"model": "last-installed"})
    ui = await resolve_active_settings(app, "u1")
    assert ui["model"] == "persisted-pick"


@pytest.mark.asyncio
async def test_authenticated_with_nothing_saved_is_empty_not_global():
    # A tenant who never touched the panel must NOT inherit the global mirror.
    store = _Store({})
    app = _app(by_user={}, store=store, global_settings={"model": "someone-elses"})
    ui = await resolve_active_settings(app, "u1")
    assert ui == {}


@pytest.mark.asyncio
async def test_anonymous_uses_global_mirror():
    app = _app(global_settings={"model": "single-user-pick"})
    ui = await resolve_active_settings(app, "")
    assert ui["model"] == "single-user-pick"


@pytest.mark.asyncio
async def test_corrupt_persisted_row_is_safe():
    store = _Store({"u1": "{not json"})
    app = _app(by_user={}, store=store)
    assert await resolve_active_settings(app, "u1") == {}


@pytest.mark.asyncio
async def test_none_app_state_is_safe():
    assert await resolve_active_settings(None, "u1") == {}
