"""Per-user state must not cross between owner and borrower.

An admin-shared media server conveys the CREDENTIAL and the CATALOG.
It must never convey the owner's viewing state to a borrower, nor push
the borrower's playback into the owner's upstream account.

These tests pin both directions plus the subtle bit: stripped fields are
OMITTED, not zeroed, so the borrower's own progress survives a resync.
"""

from __future__ import annotations

import pytest

from augmentum.media.providers.base import CatalogItem
from augmentum.media.store import MediaServer

OWNER = "u_admin"
BORROWER = "u_bench"


def _server(**over) -> MediaServer:
    base = dict(
        id="srv1", user_id=OWNER, provider="emby", name="Emby",
        base_url="http://emby", access_token="OWNER_TOKEN",
        status="ok", status_detail="", last_sync_at=None,
        item_count=0, created_at="", updated_at="", scope="shared",
    )
    base.update(over)
    return MediaServer(**base)


def _item(**over) -> CatalogItem:
    """A catalog item as Emby's parser produces it — owner UserData baked in."""
    base = dict(
        external_id="item1", name="The Matrix", kind="video",
        mime_type="video/mp4", size_bytes=1, duration_ms=1000,
        # Channel 2: these came from the OWNER's UserData on the
        # catalog fetch, before fetch_progress is even consulted.
        progress_pct=73.5,
        stream_path="/Videos/item1/stream",
        extra={
            "current_time_s": 900.0,
            "is_finished": True,
            "unplayed_count": 4,
            "is_favorite": True,
            "play_count": 9,
            "library_view_id": "lib1",
            "year": 1999,
        },
    )
    base.update(over)
    return CatalogItem(**base)


# --- ownership predicate -------------------------------------------------

def test_is_borrowed_by_distinguishes_owner_from_borrower():
    srv = _server()
    assert srv.is_borrowed_by(BORROWER) is True
    assert srv.is_borrowed_by(OWNER) is False


def test_is_borrowed_by_treats_empty_user_as_not_borrowed():
    """Internal/test callers with no viewer context keep pre-share behavior."""
    assert _server().is_borrowed_by("") is False


def test_private_server_synced_by_its_owner_is_never_borrowed():
    assert _server(scope="private").is_borrowed_by(OWNER) is False


# --- read direction: owner state must not reach the borrower -------------

def test_strip_removes_every_owner_user_state_field():
    from augmentum.media.sync import _strip_owner_user_state

    item = _item()
    _strip_owner_user_state(item)

    assert item.progress_pct == 0.0
    for leaked in (
        "current_time_s", "is_finished", "unplayed_count",
        "is_favorite", "play_count",
    ):
        assert leaked not in item.extra, f"{leaked} leaked from owner"


def test_strip_preserves_item_facts():
    """Only USER state goes. What the ITEM *is* must survive."""
    from augmentum.media.sync import _strip_owner_user_state

    item = _item()
    _strip_owner_user_state(item)

    assert item.name == "The Matrix"
    assert item.extra["year"] == 1999
    assert item.extra["library_view_id"] == "lib1"
    assert item.stream_path == "/Videos/item1/stream"


@pytest.mark.asyncio
async def test_borrowed_sync_omits_user_state_keys_from_payload(monkeypatch):
    """The load-bearing detail: OMIT, don't zero.

    register()'s preserve-merge only restores a stored value when the key
    is absent from the incoming payload. Writing 0.0 would clobber the
    borrower's own progress on every sync.
    """
    from augmentum.media import sync as sync_mod

    captured: dict = {}

    async def fake_register(**kwargs):
        captured.update(kwargs)
        return "fi_x"

    monkeypatch.setattr(sync_mod, "register_file", fake_register)

    await sync_mod._index_item(
        server=_server(), item=_item(), target_user_id=BORROWER,
        strip_user_state=True,
    )

    meta = captured["source_metadata"]
    for key in ("progress_pct", "current_time_s", "is_finished",
                "unplayed_count"):
        assert key not in meta, f"{key} present — would clobber borrower state"
    nested = meta.get("extra") or {}
    assert "is_favorite" not in nested
    assert "play_count" not in nested
    # Item facts still written.
    assert meta["server_id"] == "srv1"
    assert meta["year"] == 1999


@pytest.mark.asyncio
async def test_owner_sync_still_writes_their_own_progress(monkeypatch):
    """Regression guard: the fix must not break the normal private case."""
    from augmentum.media import sync as sync_mod

    captured: dict = {}

    async def fake_register(**kwargs):
        captured.update(kwargs)
        return "fi_x"

    monkeypatch.setattr(sync_mod, "register_file", fake_register)

    await sync_mod._index_item(
        server=_server(scope="private"), item=_item(),
        target_user_id=OWNER, strip_user_state=False,
    )

    meta = captured["source_metadata"]
    assert meta["progress_pct"] == 73.5
    assert meta["current_time_s"] == 900.0
    assert meta["is_finished"] is True


def test_owner_user_state_key_list_covers_known_provider_fields():
    """Tripwire: a new per-user provider field must be added to the list.

    These are the UserData-derived names the Emby/Jellyfin/Komga parsers
    actually emit. If a parser starts emitting another, add it here and
    to _OWNER_USER_STATE_EXTRA_KEYS together.
    """
    from augmentum.media.sync import _OWNER_USER_STATE_EXTRA_KEYS

    for known in (
        "current_time_s", "is_finished", "unplayed_count",
        "is_favorite", "play_count", "current_page",
    ):
        assert known in _OWNER_USER_STATE_EXTRA_KEYS
