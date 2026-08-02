"""Loaded-context handoff — the widget "Read this …" button (2026-06-13).

The perception bridge only carries index/digest fidelity (page title,
file name). This is the opt-in deep channel: the user hands the companion
the FULL content of what they're looking at. It lands in LoadedContextStore
(ephemeral, per-user, latest-per-kind); the prompt carries a digest while
the full body waits behind context_peek('loaded') — so the prompt budget
pays only for the digest, not the whole article.
"""

from __future__ import annotations

import asyncio

from augmentum.companion_runtime.presence_context import (
    LOADED,
    LoadedContextStore,
    now_context,
    prompt_lines,
)
from augmentum.tools.context_peek import _SLOTS, ContextPeekTool


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── LoadedContextStore ────────────────────────────────────────────────


def test_load_stores_and_caps():
    s = LoadedContextStore()
    n = s.load("u", "page", label="X", content="a" * 50_000)
    assert n == 16_000  # _LOADED_MAX_CHARS
    got = s.get("u", "page")
    assert got and got["label"] == "X" and len(got["content"]) == 16_000


def test_latest_picks_most_recent_kind():
    s = LoadedContextStore()
    s.load("u", "page", label="P", content="page body")
    s.load("u", "chat", label="C", content="chat body")
    latest = s.get_latest("u")
    assert latest["kind"] == "chat"  # loaded second


def test_load_rejects_empty_identity():
    s = LoadedContextStore()
    assert s.load("", "page", label="X", content="y") == 0
    assert s.load("u", "", label="X", content="y") == 0
    assert s.get_latest("u") is None


def test_clear_and_reset():
    s = LoadedContextStore()
    s.load("u", "page", label="X", content="y")
    s.clear("u", "page")
    assert s.get("u", "page") is None
    s.load("u", "page", label="X", content="y")
    s.reset()
    assert s.get_latest("u") is None


def test_stale_entries_drop(monkeypatch):
    import augmentum.companion_runtime.presence_context as pc
    s = LoadedContextStore()
    s.load("u", "page", label="X", content="y")
    # Jump past the freshness window.
    real = pc.time.time()
    monkeypatch.setattr(pc.time, "time", lambda: real + pc.LOADED_FRESH_S + 1)
    assert s.get("u", "page") is None
    assert s.get_latest("u") is None


# ── now_context + prompt rendering ────────────────────────────────────


def test_snapshot_includes_loaded():
    LOADED.reset()
    LOADED.load("u", "page", label="Rust ownership", content="The borrow checker " * 30)
    snap = _run(now_context(None, "u"))
    assert (snap.get("loaded") or {}).get("label") == "Rust ownership"
    LOADED.reset()


def test_degraded_prompt_lines_render_loaded():
    LOADED.reset()
    LOADED.load("u", "chat", label="our planning thread", content="we decided to ship")
    snap = _run(now_context(None, "u"))
    lines = prompt_lines(snap)
    assert any("handed you the full chat" in line for line in lines)
    assert any("we decided to ship" in line for line in lines)
    LOADED.reset()


# ── context_peek('loaded') ────────────────────────────────────────────


def test_loaded_is_a_peek_slot():
    assert "loaded" in _SLOTS


def test_peek_loaded_returns_full_body():
    LOADED.reset()
    LOADED.load("u", "page", label="Doc", content="alpha bravo charlie " * 50)
    res = _run(ContextPeekTool(app_state=None).execute(slot="loaded", _user_id="u"))
    assert res.success
    assert "alpha bravo charlie" in res.output
    # peek strips trailing whitespace, so chars matches the stripped body
    assert res.metadata["chars"] == len(("alpha bravo charlie " * 50).strip())
    LOADED.reset()


def test_peek_loaded_caps_at_4k():
    LOADED.reset()
    LOADED.load("u", "page", label="Big", content="z" * 10_000)
    res = _run(ContextPeekTool(app_state=None).execute(slot="loaded", _user_id="u"))
    assert res.success
    assert "truncated" in res.output
    LOADED.reset()


def test_peek_loaded_empty_is_graceful():
    LOADED.reset()
    res = _run(ContextPeekTool(app_state=None).execute(slot="loaded", _user_id="nobody"))
    assert res.success
    assert "haven't handed you anything" in res.output
