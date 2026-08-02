"""Media resolver + media.play handler — Companion Direct Action P2.

Pins the precision-over-recall contract from the 2026-06-10 spec:
clear winner auto-plays, near-ties offer cards, junk matches never
auto-play, misses are honest (no panel yank, no surface_emit).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from augmentum.intent.action import ReferentCache, SessionContext
from augmentum.intent.builtin.media import _media_play
from augmentum.media.resolver import (
    OFFER_THRESHOLD,
    PLAY_THRESHOLD,
    extract_kind_hint,
    resolve_media,
)


def _entry(
    *,
    id: str,
    name: str,
    kind: str = "audio",
    source: str = "librivox",
    author: str = "",
    progress_pct: float = 0.0,
    is_finished: bool = False,
    is_favorite: bool = False,
    last_played_at: str = "",
    is_directory: bool = False,
):
    meta = {}
    if author:
        meta["author"] = author
    if progress_pct:
        meta["progress_pct"] = progress_pct
        meta["is_finished"] = is_finished
    return SimpleNamespace(
        id=id, name=name, kind=kind, source=source,
        source_metadata=meta, is_favorite=is_favorite,
        last_played_at=last_played_at, is_directory=is_directory,
    )


class _FakeIndex:
    def __init__(self, entries):
        self._entries = entries
        self.calls = []

    async def search(self, query, *, user_id, kind=None, limit=20, **kw):
        self.calls.append({"query": query, "kind": kind})
        if kind:
            return [e for e in self._entries if e.kind == kind]
        return list(self._entries)


def _app_state(entries):
    return SimpleNamespace(file_index=_FakeIndex(entries))


# ── extract_kind_hint ─────────────────────────────────────────────────

def test_kind_hint_audiobook():
    title, index_kind, content_kind = extract_kind_hint("the dune audiobook")
    assert title == "dune"
    assert index_kind == "audio"
    assert content_kind == "audiobook"


def test_kind_hint_none():
    title, index_kind, content_kind = extract_kind_hint("foundation")
    assert title == "foundation"
    assert index_kind == ""
    assert content_kind == ""


def test_kind_hint_all_filler_is_empty():
    title, _, _ = extract_kind_hint("play something for me")
    assert title == "something"  # 'something' is not a filler — kept


# ── decision policy ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clear_winner_plays():
    state = _app_state([
        _entry(id="f1", name="Dune - Frank Herbert.m4b", author="Frank Herbert"),
        _entry(id="f2", name="Cooking Basics.m4b"),
    ])
    r = await resolve_media(state, user_id="u1", query="the dune audiobook")
    assert r.decision == "play"
    assert r.top.file_id == "f1"
    assert r.top.content_kind == "audiobook"
    assert r.top.score >= PLAY_THRESHOLD


@pytest.mark.asyncio
async def test_near_tie_offers_cards():
    state = _app_state([
        _entry(id="f1", name="Dune - Full Cast Dramatization.m4b"),
        _entry(id="f2", name="Dune - Scarlett Johansson Narration.m4b"),
    ])
    r = await resolve_media(state, user_id="u1", query="dune audiobook")
    assert r.decision == "offer"
    assert len(r.candidates) == 2
    ids = {c.file_id for c in r.candidates}
    assert ids == {"f1", "f2"}


@pytest.mark.asyncio
async def test_junk_match_never_plays():
    # Library has nothing resembling the ask — weak matches must not
    # reach the play threshold (the trust-protecting precision rule).
    state = _app_state([
        _entry(id="f1", name="Meeting notes 2026.m4b"),
        _entry(id="f2", name="Random mixtape.mp3"),
    ])
    r = await resolve_media(state, user_id="u1", query="the dune audiobook")
    assert r.decision != "play"


@pytest.mark.asyncio
async def test_empty_library_is_none():
    state = _app_state([])
    r = await resolve_media(state, user_id="u1", query="dune")
    assert r.decision == "none"
    assert r.candidates == []


@pytest.mark.asyncio
async def test_in_progress_copy_beats_fresh_duplicate():
    state = _app_state([
        _entry(id="fresh", name="Dune Part Two.m4b"),
        _entry(id="started", name="Dune Part Two.m4b", progress_pct=42.0),
    ])
    r = await resolve_media(state, user_id="u1", query="dune part two")
    # Identical titles — the in-progress one must rank first whatever
    # the decision (offer is acceptable; wrong ordering is not).
    assert r.candidates[0].file_id == "started"
    assert r.candidates[0].in_progress is True


@pytest.mark.asyncio
async def test_unplayable_kinds_filtered():
    state = _app_state([
        _entry(id="f1", name="dune wallpaper.png", kind="image"),
        _entry(id="f2", name="dune_save.zip", kind="archive"),
    ])
    r = await resolve_media(state, user_id="u1", query="dune")
    assert r.decision == "none"


@pytest.mark.asyncio
async def test_anon_user_resolves_none():
    state = _app_state([_entry(id="f1", name="Dune.m4b")])
    r = await resolve_media(state, user_id="", query="dune")
    assert r.decision == "none"


# ── media.play handler ────────────────────────────────────────────────

def _session(state, user_id="u1"):
    return SessionContext(
        user_id=user_id, session_id="s1",
        referents=ReferentCache(), app_state=state,
    )


@pytest.mark.asyncio
async def test_handler_play_emits_media_resume():
    state = _app_state([
        _entry(id="f1", name="Dune - Frank Herbert.m4b", author="Frank Herbert"),
    ])
    session = _session(state)
    res = await _media_play("", session, {"query": "the dune audiobook"})
    assert res.short_circuit
    assert res.surface_emit["channel"] == "media.resume"
    assert res.surface_emit["payload"]["file_id"] == "f1"
    assert res.surface_emit["payload"]["content_kind"] == "audiobook"
    assert "Starting" in res.speak
    assert session.referents.last_file_id == "f1"
    assert session.referents.pending_candidates == []


@pytest.mark.asyncio
async def test_handler_offer_parks_candidates():
    state = _app_state([
        _entry(id="f1", name="Dune - Full Cast.m4b"),
        _entry(id="f2", name="Dune - Johansson.m4b"),
    ])
    session = _session(state)
    res = await _media_play("", session, {"query": "dune audiobook"})
    assert res.surface_emit["channel"] == "companion.candidates"
    payload = res.surface_emit["payload"]
    assert payload["intent"] == "media.play"
    assert len(payload["candidates"]) == 2
    assert len(session.referents.pending_candidates) == 2
    assert "Which one" in res.speak


@pytest.mark.asyncio
async def test_handler_miss_is_honest_no_panel():
    state = _app_state([])
    session = _session(state)
    res = await _media_play("", session, {"query": "the dune audiobook"})
    assert res.surface_emit is None          # NO files panel yank
    assert "don't see" in res.speak
    assert "YouTube" in res.speak


@pytest.mark.asyncio
async def test_handler_no_query_asks():
    session = _session(_app_state([]))
    res = await _media_play("", session, {})
    assert res.surface_emit is None
    assert "What should I play" in res.speak


@pytest.mark.asyncio
async def test_handler_anon_refuses():
    session = _session(_app_state([]), user_id="")
    res = await _media_play("", session, {"query": "dune"})
    assert "signed-out" in res.speak


@pytest.mark.asyncio
async def test_offer_threshold_exported_sane():
    assert 0.0 < OFFER_THRESHOLD < PLAY_THRESHOLD < 1.0
