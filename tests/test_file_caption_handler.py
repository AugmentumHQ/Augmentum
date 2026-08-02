"""Tests for the file_caption job handler.

The handler is the load-bearing piece of Piece 2: it takes a file_id
+ user_id and writes a caption back to file_index.description via the
VisionRouter. These tests pin its idempotency contract, mime filter,
and graceful-degradation behavior — the substrate must NEVER crash
when vision is disabled, files are missing, or descriptions already
exist.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_app(
    *,
    vision_router=None,
    file_index=None,
):
    """Build a minimal app.state stand-in for the handler factory."""
    return SimpleNamespace(state=SimpleNamespace(
        file_index=file_index,
        vision_router=vision_router,
    ))


def _make_ctx(payload: dict):
    """Build a minimal JobContext stand-in.

    The real JobContext has more surface but the handler only uses
    ``payload`` and ``run_in_thread``. Anything else means the handler
    has gained a dependency that needs to be documented + tested.
    """
    ctx = SimpleNamespace(payload=payload)
    # run_in_thread is only awaited when we actually read file bytes;
    # tests that exercise that path provide their own AsyncMock.
    ctx.run_in_thread = AsyncMock()
    return ctx


# ── Payload validation ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_payload_keys_skip():
    from augmentum.jobs.handlers.file_caption import make_file_caption_handler
    handler = make_file_caption_handler(_make_app())
    result = await handler(_make_ctx({}))
    assert result["status"] == "skipped"


@pytest.mark.asyncio
async def test_missing_file_id_skips():
    from augmentum.jobs.handlers.file_caption import make_file_caption_handler
    handler = make_file_caption_handler(_make_app())
    result = await handler(_make_ctx({"user_id": "usr_x"}))
    assert result["status"] == "skipped"


# ── Service availability ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_file_index_skips():
    """If app.state.file_index isn't wired, handler skips not crashes."""
    from augmentum.jobs.handlers.file_caption import make_file_caption_handler
    app = _make_app(file_index=None, vision_router=MagicMock())
    handler = make_file_caption_handler(app)
    result = await handler(_make_ctx({"file_id": "fi_x", "user_id": "u_x"}))
    assert result["status"] == "skipped"


@pytest.mark.asyncio
async def test_no_vision_router_skips():
    """If vision_router is None, handler skips. Resilience to
    vision_provider_enabled=False."""
    from augmentum.jobs.handlers.file_caption import make_file_caption_handler
    fi = MagicMock()
    app = _make_app(file_index=fi, vision_router=None)
    handler = make_file_caption_handler(app)
    result = await handler(_make_ctx({"file_id": "fi_x", "user_id": "u_x"}))
    assert result["status"] == "skipped"


@pytest.mark.asyncio
async def test_router_without_providers_skips():
    """Router exists but has no providers → skip with explicit reason."""
    from augmentum.jobs.handlers.file_caption import make_file_caption_handler
    router = MagicMock()
    router.has_any_provider = False
    fi = MagicMock()
    app = _make_app(file_index=fi, vision_router=router)
    handler = make_file_caption_handler(app)
    result = await handler(_make_ctx({"file_id": "fi_x", "user_id": "u_x"}))
    assert result["status"] == "skipped"
    assert "provider" in result["reason"].lower()


@pytest.mark.asyncio
async def test_router_unavailable_skips():
    """Router has providers configured but none ready → skip."""
    from augmentum.jobs.handlers.file_caption import make_file_caption_handler
    router = MagicMock()
    router.has_any_provider = True
    router.is_available = AsyncMock(return_value=False)
    fi = MagicMock()
    app = _make_app(file_index=fi, vision_router=router)
    handler = make_file_caption_handler(app)
    result = await handler(_make_ctx({"file_id": "fi_x", "user_id": "u_x"}))
    assert result["status"] == "skipped"


# ── Row-level filters ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_row_not_found_skips():
    from augmentum.jobs.handlers.file_caption import make_file_caption_handler
    router = MagicMock()
    router.has_any_provider = True
    router.is_available = AsyncMock(return_value=True)
    fi = MagicMock()
    fi.get = AsyncMock(return_value=None)
    app = _make_app(file_index=fi, vision_router=router)
    handler = make_file_caption_handler(app)
    result = await handler(_make_ctx({"file_id": "fi_x", "user_id": "u_x"}))
    assert result["status"] == "skipped"
    assert "not found" in result["reason"]


@pytest.mark.asyncio
async def test_non_image_mime_skips():
    """PDF row → captioner skips (no useful description from VL on PDF bytes)."""
    from augmentum.jobs.handlers.file_caption import make_file_caption_handler
    router = MagicMock()
    router.has_any_provider = True
    router.is_available = AsyncMock(return_value=True)

    row = SimpleNamespace(
        mime_type="application/pdf",
        description="",
        real_path="/tmp/x.pdf",
    )
    fi = MagicMock()
    fi.get = AsyncMock(return_value=row)
    app = _make_app(file_index=fi, vision_router=router)
    handler = make_file_caption_handler(app)
    result = await handler(_make_ctx({"file_id": "fi_x", "user_id": "u_x"}))
    assert result["status"] == "skipped"
    assert "mime" in result["reason"].lower()


@pytest.mark.asyncio
async def test_existing_description_skips():
    """Idempotency: row with description already set → skip without
    calling vision. This is what makes the auto-enqueue at register()
    + the backfill loop's enqueue safe even when they fire on the same
    row."""
    from augmentum.jobs.handlers.file_caption import make_file_caption_handler
    router = MagicMock()
    router.has_any_provider = True
    router.is_available = AsyncMock(return_value=True)
    router.caption = AsyncMock(return_value="should-not-be-called")

    row = SimpleNamespace(
        mime_type="image/png",
        description="already captioned",
        real_path="/tmp/x.png",
    )
    fi = MagicMock()
    fi.get = AsyncMock(return_value=row)
    fi.update_enrichment = AsyncMock()
    app = _make_app(file_index=fi, vision_router=router)
    handler = make_file_caption_handler(app)
    result = await handler(_make_ctx({"file_id": "fi_x", "user_id": "u_x"}))
    assert result["status"] == "skipped"
    assert "already set" in result["reason"]
    router.caption.assert_not_awaited()
    fi.update_enrichment.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_file_bytes_skips():
    """real_path unset or file missing → skip without crashing."""
    from augmentum.jobs.handlers.file_caption import make_file_caption_handler
    router = MagicMock()
    router.has_any_provider = True
    router.is_available = AsyncMock(return_value=True)

    row = SimpleNamespace(
        mime_type="image/jpeg",
        description="",
        real_path="",
    )
    fi = MagicMock()
    fi.get = AsyncMock(return_value=row)
    app = _make_app(file_index=fi, vision_router=router)
    handler = make_file_caption_handler(app)
    result = await handler(_make_ctx({"file_id": "fi_x", "user_id": "u_x"}))
    assert result["status"] == "skipped"
    assert "disk" in result["reason"].lower()


# ── Happy path ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_happy_path_writes_caption(tmp_path):
    """Image row with empty description + working vision → caption
    written via update_enrichment."""
    from augmentum.jobs.handlers.file_caption import make_file_caption_handler

    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fake-png-bytes")

    router = MagicMock()
    router.has_any_provider = True
    router.is_available = AsyncMock(return_value=True)
    router.caption = AsyncMock(return_value="A test image of nothing.")

    row = SimpleNamespace(
        mime_type="image/png",
        description="",
        real_path=str(img),
    )
    fi = MagicMock()
    fi.get = AsyncMock(return_value=row)
    fi.update_enrichment = AsyncMock(return_value=True)

    app = _make_app(file_index=fi, vision_router=router)
    handler = make_file_caption_handler(app)

    ctx = _make_ctx({"file_id": "fi_happy", "user_id": "u_happy"})
    # Replace run_in_thread so file read actually returns the test bytes.
    async def run_in_thread(fn):
        return fn()
    ctx.run_in_thread = run_in_thread

    result = await handler(ctx)
    assert result["status"] == "ok"
    assert result["caption_chars"] > 0
    router.caption.assert_awaited_once()
    fi.update_enrichment.assert_awaited_once_with(
        "fi_happy", user_id="u_happy",
        description="A test image of nothing.",
    )


@pytest.mark.asyncio
async def test_empty_caption_skips(tmp_path):
    """Vision returned empty string → skip without writing."""
    from augmentum.jobs.handlers.file_caption import make_file_caption_handler

    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    router = MagicMock()
    router.has_any_provider = True
    router.is_available = AsyncMock(return_value=True)
    router.caption = AsyncMock(return_value="")

    row = SimpleNamespace(
        mime_type="image/png",
        description="",
        real_path=str(img),
    )
    fi = MagicMock()
    fi.get = AsyncMock(return_value=row)
    fi.update_enrichment = AsyncMock()
    app = _make_app(file_index=fi, vision_router=router)
    handler = make_file_caption_handler(app)

    ctx = _make_ctx({"file_id": "fi_empty", "user_id": "u_x"})
    async def run_in_thread(fn):
        return fn()
    ctx.run_in_thread = run_in_thread

    result = await handler(ctx)
    assert result["status"] == "skipped"
    fi.update_enrichment.assert_not_awaited()
