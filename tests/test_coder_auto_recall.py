"""Tests for coder auto-recall — the read path that wakes the durable archive.

Auto-recall surfaces the top semantically-relevant PAST turns into each
turn's context (instead of waiting for the model to call the ``recall``
tool). These tests pin two contracts:

* ``_render_recalled_block`` — HISTORICAL framing, budget truncation,
  empty-in/empty-out.
* ``_build_recalled_block`` — settings-gated, dedup against the
  <prior_turns> ring, k-capped, and best-effort (any miss → "").

``search_similar`` is monkeypatched so no DB / embedder is needed.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from augmentum.coder.turn_archive_embed import RecallHit
from augmentum.modes.coder.turn_context import (
    _build_recalled_block,
    _render_recalled_block,
)


def _hit(turn_index: int, *, goal: str = "", summary: str = "",
         outcome: str = "done", event_time: int = 0, distance: float = 0.1) -> RecallHit:
    return RecallHit(
        archive_id=f"a{turn_index}",
        turn_index=turn_index,
        user_goal=goal,
        outcome=outcome,
        summary=summary,
        event_time=event_time,
        distance=distance,
    )


def _fake_handler(*, conn: object = object(), user_id: str = "u1",
                  workspace_id: str = "ws1", recent: list[int] | None = None):
    """Minimal CoderHandler stand-in for _build_recalled_block."""
    summaries = [{"turn_idx": i} for i in (recent or [])]
    return SimpleNamespace(
        _state_manager=SimpleNamespace(backend=SimpleNamespace(conn=conn)),
        _user_id=user_id,
        _workspace_id=workspace_id,
        _state=SimpleNamespace(turn_summaries=summaries),
    )


def _patch_search(monkeypatch, hits, *, capture: dict | None = None):
    async def fake_search(conn, *, user_id, workspace_id, query, k=5,
                          similarity_threshold=0.0):
        if capture is not None:
            capture.update(
                user_id=user_id, workspace_id=workspace_id, query=query,
                k=k, similarity_threshold=similarity_threshold,
            )
        return list(hits)

    monkeypatch.setattr(
        "augmentum.coder.turn_archive_embed.search_similar", fake_search,
    )


# ---------------------------------------------------------------------------
# _render_recalled_block
# ---------------------------------------------------------------------------

def test_render_empty_returns_empty():
    assert _render_recalled_block([]) == ""


def test_render_frames_historical_and_lists_turns():
    block = _render_recalled_block([
        _hit(14, goal="fix auth 401 on refresh"),
        _hit(9, goal="add timezone handling"),
    ])
    assert block.startswith("<recalled_context>")
    assert block.endswith("</recalled_context>")
    assert "HISTORICAL" in block
    assert "Turn 14" in block and "fix auth 401" in block
    assert "Turn 9" in block


def test_render_falls_back_to_summary_when_no_goal():
    block = _render_recalled_block([_hit(3, summary="discovered pytest is the runner")])
    assert "pytest is the runner" in block


def test_render_budget_truncates_but_stays_valid():
    long = "x" * 400
    hits = [_hit(i, goal=long) for i in range(20)]
    block = _render_recalled_block(hits)
    # Some hits dropped by the budget, but the block is still well-formed.
    assert block.endswith("</recalled_context>")
    assert len(block) <= 1200  # budget (800) + framing/header slack
    assert block.count("• Turn") < 20


def test_render_clips_overlong_detail():
    block = _render_recalled_block([_hit(1, goal="g" * 500)])
    assert "…" in block


def test_render_timestamps_are_absolute_and_byte_stable():
    """Recall timestamps must be absolute ('Jul 2 14:32'), never relative.

    Relative phrasing ('7m ago') re-renders differently every minute, so
    an unchanged hit would still mutate the prompt prefix and invalidate
    the llama-server slot cache (live regression, 2026-07-02). The model
    computes recency against the <current_time> block that rides at the
    end of the runtime carrier.
    """
    import time

    hit = _hit(7, goal="wire the websocket", event_time=int(time.time()) - 420)
    first = _render_recalled_block([hit])
    second = _render_recalled_block([hit])
    assert "ago" not in first
    assert first == second  # byte-identical re-render = cache-stable
    # Absolute stamp shape: "<Mon> <day> <HH:MM>", e.g. "Jul 2 14:32".
    assert re.search(r"· [A-Z][a-z]{2} \d{1,2} \d{2}:\d{2} ·", first)


# ---------------------------------------------------------------------------
# _build_recalled_block — gating
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_disabled_setting_returns_empty(monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "coder_auto_recall_enabled", False)
    _patch_search(monkeypatch, [_hit(1, goal="x")])
    out = await _build_recalled_block(_fake_handler(), user_goal="g", user_query="q")
    assert out == ""


@pytest.mark.asyncio
async def test_build_archive_disabled_returns_empty(monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "coder_auto_recall_enabled", True)
    monkeypatch.setattr(settings, "coder_archive_enabled", False)
    _patch_search(monkeypatch, [_hit(1, goal="x")])
    out = await _build_recalled_block(_fake_handler(), user_goal="g", user_query="q")
    assert out == ""


@pytest.mark.asyncio
async def test_build_empty_query_returns_empty(monkeypatch):
    _patch_search(monkeypatch, [_hit(1, goal="x")])
    out = await _build_recalled_block(_fake_handler(), user_goal="", user_query="")
    assert out == ""


@pytest.mark.asyncio
async def test_build_no_conn_returns_empty(monkeypatch):
    _patch_search(monkeypatch, [_hit(1, goal="x")])
    handler = _fake_handler(conn=None)
    out = await _build_recalled_block(handler, user_goal="g", user_query="q")
    assert out == ""


# ---------------------------------------------------------------------------
# _build_recalled_block — behavior
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_renders_when_hits(monkeypatch):
    _patch_search(monkeypatch, [_hit(7, goal="wire the websocket")])
    out = await _build_recalled_block(_fake_handler(), user_goal="fix ws", user_query="q")
    assert "Turn 7" in out and "wire the websocket" in out


@pytest.mark.asyncio
async def test_build_dedups_against_prior_turns_ring(monkeypatch):
    # Turn 5 is already shown verbatim in <prior_turns>; recall must skip it.
    _patch_search(monkeypatch, [_hit(5, goal="recent shown"), _hit(2, goal="older relevant")])
    handler = _fake_handler(recent=[5, 6])
    out = await _build_recalled_block(handler, user_goal="g", user_query="q")
    assert "Turn 2" in out
    assert "Turn 5" not in out


@pytest.mark.asyncio
async def test_build_caps_at_k(monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "coder_auto_recall_k", 2)
    _patch_search(monkeypatch, [_hit(i, goal=f"goal {i}") for i in range(10)])
    out = await _build_recalled_block(_fake_handler(), user_goal="g", user_query="q")
    assert out.count("• Turn") == 2


@pytest.mark.asyncio
async def test_build_no_hits_returns_empty(monkeypatch):
    _patch_search(monkeypatch, [])
    out = await _build_recalled_block(_fake_handler(), user_goal="g", user_query="q")
    assert out == ""


@pytest.mark.asyncio
async def test_build_query_prefers_goal_over_query(monkeypatch):
    cap: dict = {}
    _patch_search(monkeypatch, [_hit(1, goal="x")], capture=cap)
    await _build_recalled_block(_fake_handler(), user_goal="the goal", user_query="the query")
    assert cap["query"] == "the goal"


@pytest.mark.asyncio
async def test_build_swallows_search_error(monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("embedder down")
    monkeypatch.setattr("augmentum.coder.turn_archive_embed.search_similar", boom)
    out = await _build_recalled_block(_fake_handler(), user_goal="g", user_query="q")
    assert out == ""
