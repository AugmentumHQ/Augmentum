"""``GET /api/media/gutenberg-text/{file_id}`` — status-code matrix.

Calls the route handler directly with a fabricated Request so we don't
have to spin up the full FastAPI app + auth middleware. The handler
only reads two things off the request: ``scope['user'].id`` and
``app.state.file_index``, both easy to fake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.responses import JSONResponse, PlainTextResponse

from augmentum.proxy.media_routes import get_gutenberg_text


@dataclass
class _FakeEntry:
    id: str
    user_id: str
    source_metadata: dict[str, Any] = field(default_factory=dict)


class _FakeIndex:
    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], _FakeEntry] = {}

    def seed(self, entry: _FakeEntry) -> None:
        self._rows[(entry.id, entry.user_id)] = entry

    async def get(self, file_id: str, *, user_id: str):
        return self._rows.get((file_id, user_id))

    async def update_source_metadata(
        self, file_id: str, metadata: dict, *, user_id: str,
    ) -> bool:
        key = (file_id, user_id)
        if key not in self._rows:
            return False
        self._rows[key].source_metadata = metadata
        return True


def _mk_request(user_id: str, idx: _FakeIndex):
    """Build a minimal Request-compatible object for the route handler."""
    user = SimpleNamespace(id=user_id) if user_id else None
    app = SimpleNamespace(state=SimpleNamespace(file_index=idx))
    return SimpleNamespace(
        app=app,
        scope={"user": user},
    )


@pytest.mark.asyncio
async def test_serves_fetched_plaintext(tmp_path):
    blob = tmp_path / "fi_ok.txt"
    blob.write_text("Call me Ishmael.\n", encoding="utf-8")

    idx = _FakeIndex()
    idx.seed(_FakeEntry(
        id="fi_ok", user_id="u1",
        source_metadata={"gutenberg_status": "fetched", "gutenberg_path": str(blob)},
    ))
    resp = await get_gutenberg_text("fi_ok", _mk_request("u1", idx))
    assert isinstance(resp, PlainTextResponse)
    assert resp.status_code == 200
    assert resp.body.decode("utf-8").strip() == "Call me Ishmael."


@pytest.mark.asyncio
async def test_missing_file_id_returns_404():
    idx = _FakeIndex()
    resp = await get_gutenberg_text("fi_missing", _mk_request("u1", idx))
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_other_user_cannot_read(tmp_path):
    blob = tmp_path / "fi_private.txt"
    blob.write_text("secret content", encoding="utf-8")
    idx = _FakeIndex()
    idx.seed(_FakeEntry(
        id="fi_private", user_id="owner",
        source_metadata={"gutenberg_status": "fetched", "gutenberg_path": str(blob)},
    ))
    # Attacker knows the file_id but their user_id doesn't match.
    resp = await get_gutenberg_text("fi_private", _mk_request("attacker", idx))
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_no_auth_returns_401():
    idx = _FakeIndex()
    resp = await get_gutenberg_text("fi_x", _mk_request("", idx))
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_pending_status_returns_202():
    idx = _FakeIndex()
    idx.seed(_FakeEntry(
        id="fi_pending", user_id="u1",
        source_metadata={"gutenberg_status": "fetching"},
    ))
    resp = await get_gutenberg_text("fi_pending", _mk_request("u1", idx))
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_unavailable_status_returns_410():
    idx = _FakeIndex()
    idx.seed(_FakeEntry(
        id="fi_unavail", user_id="u1",
        source_metadata={
            "gutenberg_status": "unavailable",
            "gutenberg_error": "not a Gutenberg URL",
        },
    ))
    resp = await get_gutenberg_text("fi_unavail", _mk_request("u1", idx))
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 410


@pytest.mark.asyncio
async def test_blob_missing_on_disk_marks_missing(tmp_path):
    # Metadata says fetched but the blob was deleted from disk. The
    # handler should 410 and flip status to 'missing' so a future job
    # can re-fetch instead of repeatedly claiming it has the text.
    idx = _FakeIndex()
    idx.seed(_FakeEntry(
        id="fi_gone", user_id="u1",
        source_metadata={
            "gutenberg_status": "fetched",
            "gutenberg_path": str(tmp_path / "not-there.txt"),
        },
    ))
    resp = await get_gutenberg_text("fi_gone", _mk_request("u1", idx))
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 410
    assert idx._rows[("fi_gone", "u1")].source_metadata["gutenberg_status"] == "missing"
