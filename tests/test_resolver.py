"""Tests for the Reference Resolver (Piece 6).

The resolver merges 4 retrieval legs via RRF and returns ranked
moments. These tests exercise:

* RRF math correctness — pure unit test of ``_rrf_merge``
* Snippet truncation
* Full end-to-end with mocked file_index + memory
* Kinds filter (file-only, journal-only, both)
* Graceful degradation when a leg raises
* Embedding failure → FTS-only fallback works
* Empty query short-circuit
* Tool input validation
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


# ── RRF unit tests ────────────────────────────────────────────────────


def test_rrf_merge_single_leg():
    """Single-leg result: ranking preserved, score = 1/(k+rank)."""
    from augmentum.resolver.core import _rrf_merge, _RRF_K

    legs = {
        "file_vec": [
            ("file:1", {"id": "1", "name": "A", "description": "alpha"}),
            ("file:2", {"id": "2", "name": "B", "description": "beta"}),
        ],
    }
    out = _rrf_merge(legs)
    assert [m.id for m in out] == ["1", "2"]
    assert out[0].score == pytest.approx(1.0 / (_RRF_K + 1))
    assert out[1].score == pytest.approx(1.0 / (_RRF_K + 2))


def test_rrf_merge_two_legs_promote_shared():
    """Item appearing in two legs ranks above item in only one leg."""
    from augmentum.resolver.core import _rrf_merge

    legs = {
        "file_vec": [
            ("file:1", {"id": "1", "name": "Shared", "description": "x"}),
            ("file:2", {"id": "2", "name": "VecOnly", "description": "y"}),
        ],
        "file_fts": [
            ("file:3", {"id": "3", "name": "FtsOnly", "description": "z"}),
            ("file:1", {"id": "1", "name": "Shared", "description": "x"}),
        ],
    }
    out = _rrf_merge(legs)
    # Shared (file:1) appears in both → highest RRF
    assert out[0].id == "1"
    # legs tracking populated
    assert set(out[0].legs) == {"file_vec", "file_fts"}


def test_rrf_merge_cross_kind_no_collision():
    """File id 42 and journal id 42 must not merge."""
    from augmentum.resolver.core import _rrf_merge

    legs = {
        "file_vec": [
            ("file:42", {"id": "42", "name": "F"}),
        ],
        "journal_vec": [
            ("journal:42", {"id": 42, "content": "J"}),
        ],
    }
    out = _rrf_merge(legs)
    assert len(out) == 2
    kinds = {m.kind for m in out}
    assert kinds == {"file", "journal"}


def test_snippet_truncation():
    """Long content gets truncated with ellipsis; short content passes through."""
    from augmentum.resolver.core import _snippet

    short = _snippet("a brief thought")
    assert short == "a brief thought"

    long = _snippet("x" * 500)
    assert len(long) <= 240
    assert long.endswith("…")


def test_snippet_strips_newlines():
    """Multi-line content is collapsed to single-line for card rendering."""
    from augmentum.resolver.core import _snippet

    out = _snippet("first\nsecond\n\nthird")
    assert "\n" not in out
    assert "first" in out and "third" in out


# ── resolve_moments end-to-end ────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_empty_query_returns_empty():
    from augmentum.resolver import resolve_moments
    out = await resolve_moments("", user_id="u1")
    assert out == []


@pytest.mark.asyncio
async def test_resolve_no_services_returns_empty():
    """Both file_index and memory None → empty list, no crash."""
    from augmentum.resolver import resolve_moments

    # The embed call will succeed but every leg returns empty.
    # We don't mock embedding here; it'll either succeed (real model)
    # or fall through to FTS-only and return empty.
    out = await resolve_moments(
        "anything", user_id="u1",
        file_index=None, memory=None,
    )
    assert out == []


@pytest.mark.asyncio
async def test_resolve_file_only_kind_skips_journal():
    """Passing kinds=('file',) must not call any journal methods."""
    from augmentum.resolver import resolve_moments

    fi = MagicMock()
    fake_entry = MagicMock()
    fake_entry.__dict__ = {
        "id": "fi_1", "name": "Manga", "description": "five sisters",
        "created_at": "2026-05-19",
    }
    fi.search_by_embedding = AsyncMock(return_value=[fake_entry])
    fi.search = AsyncMock(return_value=[fake_entry])

    mem = MagicMock()
    mem.search_journal_fts = AsyncMock(return_value=[])
    mem.search_journal_by_embedding = AsyncMock(return_value=[])

    out = await resolve_moments(
        "manga five sisters",
        user_id="u1",
        file_index=fi,
        memory=mem,
        kinds=("file",),
    )

    # Journal methods MUST NOT be called
    mem.search_journal_fts.assert_not_called()
    mem.search_journal_by_embedding.assert_not_called()

    # Got at least the file leg's results
    assert any(m.kind == "file" for m in out)


@pytest.mark.asyncio
async def test_resolve_journal_only_kind_skips_file():
    from augmentum.resolver import resolve_moments

    fi = MagicMock()
    fi.search_by_embedding = AsyncMock(return_value=[])
    fi.search = AsyncMock(return_value=[])

    mem = MagicMock()
    mem.search_journal_fts = AsyncMock(return_value=[{
        "id": 7, "content": "thinking about Alex",
        "entry_type": "noticing", "created_at": "2026-05-19",
        "content_refs": [], "place_ref": "",
    }])
    mem.search_journal_by_embedding = AsyncMock(return_value=[])

    out = await resolve_moments(
        "alex",
        user_id="u1",
        file_index=fi,
        memory=mem,
        kinds=("journal",),
    )

    fi.search_by_embedding.assert_not_called()
    fi.search.assert_not_called()
    assert any(m.kind == "journal" for m in out)


@pytest.mark.asyncio
async def test_resolve_leg_exception_degrades_gracefully():
    """One leg raises → the others still produce results."""
    from augmentum.resolver import resolve_moments

    fi = MagicMock()
    fi.search_by_embedding = AsyncMock(side_effect=RuntimeError("vec borked"))
    fake_entry = MagicMock()
    fake_entry.__dict__ = {"id": "fi_2", "name": "x", "description": ""}
    fi.search = AsyncMock(return_value=[fake_entry])

    out = await resolve_moments(
        "x",
        user_id="u1",
        file_index=fi,
        memory=None,
        kinds=("file",),
    )
    # FTS leg still produced a result
    assert len(out) == 1
    assert out[0].id == "fi_2"


@pytest.mark.asyncio
async def test_resolve_limit_clamp():
    """``limit`` caps the returned moments even with many candidates."""
    from augmentum.resolver import resolve_moments

    fi = MagicMock()
    # 20 unique entries from each leg
    def _mk(prefix, n):
        out = []
        for i in range(n):
            e = MagicMock()
            e.__dict__ = {"id": f"{prefix}{i}", "name": f"n{i}", "description": "d"}
            out.append(e)
        return out

    fi.search_by_embedding = AsyncMock(return_value=_mk("v", 20))
    fi.search = AsyncMock(return_value=_mk("f", 20))

    out = await resolve_moments(
        "anything",
        user_id="u1",
        file_index=fi,
        memory=None,
        kinds=("file",),
        limit=5,
    )
    assert len(out) <= 5


# ── Tool wrapper ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_rejects_empty_query():
    from augmentum.tools.resolve_moments import ResolveMomentsTool

    tool = ResolveMomentsTool()
    result = await tool.execute(query="")
    assert not result.success
    assert "query" in result.error.lower()


@pytest.mark.asyncio
async def test_tool_requires_user_id():
    """Missing both _user_id and _context['user_id'] → error."""
    from augmentum.tools.resolve_moments import ResolveMomentsTool

    tool = ResolveMomentsTool()
    result = await tool.execute(query="anything")
    assert not result.success
    assert "user_id" in result.error.lower()


@pytest.mark.asyncio
async def test_tool_accepts_user_id_via_context():
    """user_id via _context dict works (artifact/image tool pattern)."""
    from augmentum.tools.resolve_moments import ResolveMomentsTool

    tool = ResolveMomentsTool(file_index=None, memory=None)
    result = await tool.execute(
        query="x",
        _context={"user_id": "u_test"},
    )
    # Doesn't error on user_id; degrades to empty since no services
    assert result.success is True
    assert result.metadata.get("count") == 0


@pytest.mark.asyncio
async def test_tool_accepts_user_id_via_underscore_kwarg():
    """user_id via _user_id top-level (FlowTool pattern) works."""
    from augmentum.tools.resolve_moments import ResolveMomentsTool

    tool = ResolveMomentsTool(file_index=None, memory=None)
    result = await tool.execute(query="x", _user_id="u_test")
    assert result.success is True


@pytest.mark.asyncio
async def test_tool_clamps_limit():
    """Models that pass limit=10000 don't trigger unbounded retrieval."""
    from augmentum.tools.resolve_moments import ResolveMomentsTool

    fi = MagicMock()
    captured_limit = {}

    def _capture(*args, **kwargs):
        captured_limit["v"] = kwargs.get("limit")
        return []

    fi.search_by_embedding = AsyncMock(side_effect=_capture)
    fi.search = AsyncMock(side_effect=_capture)

    tool = ResolveMomentsTool(file_index=fi, memory=None)
    await tool.execute(
        query="anything",
        _user_id="u",
        limit=10000,
        kinds=["file"],
    )
    # search() got an over-fetched limit derived from the clamped 50
    # (per-leg = max(limit*3, 20) = 150). We just check it's bounded.
    assert captured_limit["v"] is not None
    assert captured_limit["v"] <= 200


@pytest.mark.asyncio
async def test_tool_metadata_includes_moments():
    """Success result must put structured moments in metadata for UI."""
    from augmentum.tools.resolve_moments import ResolveMomentsTool

    fake_entry = MagicMock()
    fake_entry.__dict__ = {
        "id": "fi_42",
        "name": "Quintessential Quintuplets",
        "description": "manga about five sisters",
        "created_at": "2026-05-19",
    }
    fi = MagicMock()
    fi.search_by_embedding = AsyncMock(return_value=[fake_entry])
    fi.search = AsyncMock(return_value=[fake_entry])

    tool = ResolveMomentsTool(file_index=fi, memory=None)
    result = await tool.execute(
        query="manga quintuplets",
        _user_id="u",
        kinds=["file"],
    )
    assert result.success
    assert result.metadata["count"] >= 1
    moments = result.metadata["moments"]
    assert moments[0]["id"] == "fi_42"
    assert moments[0]["kind"] == "file"
    # Score is the RRF sum, not a leg-level similarity
    assert moments[0]["score"] > 0
