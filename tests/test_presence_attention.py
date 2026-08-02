"""Attention store + presence snapshot tests.

Pins the generalized perception layer (2026-06-10): client surfaces
report attention via /api/architect/observe topics, the store keeps
latest-per-slot, and ``now_context`` reads store-first with the
history-table fallback intact.
"""

from __future__ import annotations

import time
from datetime import UTC

import pytest

from augmentum.companion_runtime import presence_context as pc
from augmentum.companion_runtime.presence_context import (
    ATTENTION,
    now_context,
    observe_attention,
    prompt_lines,
)

UID = "user-presence-test"


@pytest.fixture(autouse=True)
def _clean_store():
    ATTENTION.reset()
    yield
    ATTENTION.reset()


# ---------------------------------------------------------------------------
# AttentionStore basics
# ---------------------------------------------------------------------------


def test_note_and_get_roundtrip():
    ATTENTION.note(UID, "page", label="Quantum computing", url="https://x.test/qc")
    got = ATTENTION.get(UID, "page")
    assert got is not None
    assert got["label"] == "Quantum computing"
    assert got["url"] == "https://x.test/qc"
    assert got["age_s"] < 5


def test_unknown_slot_and_empty_user_are_noops():
    ATTENTION.note(UID, "nonsense_slot", label="x")
    ATTENTION.note("", "page", label="x")
    assert ATTENTION.get(UID, "nonsense_slot") is None
    assert ATTENTION.get("", "page") is None


def test_ttl_expiry_clears_entry():
    ATTENTION.note(UID, "page", label="Old article")
    # Backdate past the page TTL.
    ATTENTION._slots[UID]["page"]["ts"] = time.time() - (pc.PAGE_FRESH_S + 10)
    assert ATTENTION.get(UID, "page") is None
    # Expiry also evicts so the dict doesn't accumulate stale entries.
    assert "page" not in ATTENTION._slots.get(UID, {})


def test_per_user_isolation():
    ATTENTION.note(UID, "page", label="Mine")
    ATTENTION.note("other-user", "page", label="Theirs")
    assert ATTENTION.get(UID, "page")["label"] == "Mine"
    assert ATTENTION.get("other-user", "page")["label"] == "Theirs"


def test_latest_wins_per_slot():
    ATTENTION.note(UID, "playing", label="First Song", kind="audio")
    ATTENTION.note(UID, "playing", label="Second Song", kind="audio")
    assert ATTENTION.get(UID, "playing")["label"] == "Second Song"


# ---------------------------------------------------------------------------
# Topic mapper — observe events → slots
# ---------------------------------------------------------------------------


def test_page_opened_and_closed():
    observe_attention(UID, "surface.browse.page_opened",
                      {"title": "Dune (novel)", "url": "https://w.test/dune"})
    assert ATTENTION.get(UID, "page")["label"] == "Dune (novel)"
    observe_attention(UID, "surface.browse.page_closed", {})
    assert ATTENTION.get(UID, "page") is None


def test_page_excerpt_carried_in_store():
    # The excerpt is stored at full fidelity; RENDERING now follows the
    # ring lifecycle (full on first sight, digest while warm, peek when
    # cold) — pinned in test_results_ring.py. prompt_lines is the
    # index-only tier and must NOT push the excerpt.
    observe_attention(UID, "surface.browse.page_opened", {
        "title": "ICE policy shift",
        "url": "https://news.test/a",
        "excerpt": "WASHINGTON (AP) — The administration announced " + "x" * 2000,
    })
    page = ATTENTION.get(UID, "page")
    assert page["excerpt"].startswith("WASHINGTON (AP)")
    assert len(page["excerpt"]) <= 1500
    lines = prompt_lines({"page": page})
    joined = "\n".join(lines)
    assert "ICE policy shift" in joined
    assert "How the page starts:" not in joined


def test_media_playback_started():
    observe_attention(UID, "surface.media.playback_started",
                      {"label": "Blade Runner 2049", "kind": "video", "ref": "f123"})
    got = ATTENTION.get(UID, "playing")
    assert got["label"] == "Blade Runner 2049"
    assert got["kind"] == "video"
    assert got["ref"] == "f123"


def test_station_playing_maps_to_playing_slot():
    observe_attention(UID, "surface.audio.station_playing",
                      {"label": "Smooth Jazz 24/7", "kind": "radio · jazz"})
    assert ATTENTION.get(UID, "playing")["label"] == "Smooth Jazz 24/7"


def test_comic_and_readalong_map_to_reading():
    observe_attention(UID, "surface.comic.opened",
                      {"label": "Saga #1", "ref": "f9"})
    got = ATTENTION.get(UID, "reading")
    assert got["label"] == "Saga #1"
    assert got["kind"] == "comic"

    observe_attention(UID, "surface.media.reading_started",
                      {"label": "Dune", "kind": "book", "ref": "f10"})
    assert ATTENTION.get(UID, "reading")["kind"] == "book"


def test_mode_changed():
    observe_attention(UID, "surface.attention.mode_changed", {"mode": "coder"})
    assert ATTENTION.get(UID, "mode")["label"] == "coder"


def test_narrative_scene_active_and_closed():
    observe_attention(UID, "surface.narrative.scene_active",
                      {"label": "Mira the Cartographer", "ref": "char-7"})
    got = ATTENTION.get(UID, "scene")
    assert got["label"] == "Mira the Cartographer"
    assert got["ref"] == "char-7"
    observe_attention(UID, "surface.narrative.scene_closed", {})
    assert ATTENTION.get(UID, "scene") is None


def test_coder_file_opened_and_closed():
    observe_attention(UID, "surface.coder.file_opened",
                      {"label": "server.py", "path": "augmentum/proxy/server.py",
                       "ref": "ws-1"})
    got = ATTENTION.get(UID, "working")
    assert got["label"] == "server.py"
    assert got["path"] == "augmentum/proxy/server.py"
    observe_attention(UID, "surface.coder.closed", {})
    assert ATTENTION.get(UID, "working") is None


def test_new_topics_pass_route_allow_list():
    from augmentum.proxy.architect_routes import _is_allowed_topic
    assert _is_allowed_topic("surface.narrative.scene_active")
    assert _is_allowed_topic("surface.narrative.scene_closed")
    assert _is_allowed_topic("surface.coder.file_opened")
    assert not _is_allowed_topic("surface.evil.injection")


def test_audio_silence_clears_playing_only():
    observe_attention(UID, "surface.media.playback_started",
                      {"label": "Song", "kind": "audio"})
    observe_attention(UID, "surface.browse.page_opened",
                      {"title": "Page", "url": "https://x.test"})
    observe_attention(UID, "surface.audio.kind_changed", {"kinds": []})
    assert ATTENTION.get(UID, "playing") is None
    assert ATTENTION.get(UID, "page") is not None


def test_audio_nonempty_kinds_does_not_clear():
    observe_attention(UID, "surface.media.playback_started",
                      {"label": "Song", "kind": "audio"})
    observe_attention(UID, "surface.audio.kind_changed", {"kinds": ["music"]})
    assert ATTENTION.get(UID, "playing") is not None


def test_mapper_never_raises_on_garbage():
    observe_attention(UID, "surface.browse.page_opened", {"title": object()})
    observe_attention(UID, "surface.audio.kind_changed", {"kinds": "not-a-list"})
    # No assertion — surviving is the test.


# ---------------------------------------------------------------------------
# now_context — store-first, fallback intact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_now_context_store_first_no_conn():
    observe_attention(UID, "surface.browse.page_opened",
                      {"title": "Dune (novel)", "url": "https://w.test/dune"})
    observe_attention(UID, "surface.attention.mode_changed", {"mode": "passthrough"})
    snap = await now_context(None, UID)
    assert snap["page"]["label"] == "Dune (novel)"
    assert snap["mode"]["label"] == "passthrough"
    assert snap["playing"] is None
    assert snap["reading"] is None


@pytest.mark.asyncio
async def test_now_context_empty_user():
    snap = await now_context(None, "")
    assert snap == {"page": None, "playing": None, "reading": None,
                    "scene": None, "working": None, "mode": None,
                    "loaded": None}


@pytest.mark.asyncio
async def test_now_context_store_beats_table(monkeypatch):
    """When the store has a page, the browse_history fallback is skipped."""
    async def _boom(*a, **k):
        raise AssertionError("fallback should not run when store is fresh")

    import augmentum.architect.inference as inference
    monkeypatch.setattr(inference, "query_browse_history", _boom)
    observe_attention(UID, "surface.browse.page_opened",
                      {"title": "Fresh", "url": "https://x.test"})
    snap = await now_context(object(), UID)
    assert snap["page"]["label"] == "Fresh"


@pytest.mark.asyncio
async def test_now_context_table_fallback(monkeypatch):
    """Empty store + fresh browse_history row → fallback populates page."""
    from datetime import datetime

    async def _rows(conn, user_id, limit=1):
        return [{
            "domain": "example.test",
            "url": "https://example.test/a",
            "last_visited": datetime.now(UTC).isoformat(),
        }]

    async def _no_plays(conn, user_id, limit=1, favourites_first=False):
        return []

    import augmentum.architect.inference as inference
    monkeypatch.setattr(inference, "query_browse_history", _rows)
    monkeypatch.setattr(inference, "query_play_history", _no_plays)
    snap = await now_context(object(), UID)
    assert snap["page"]["label"] == "example.test"
    assert snap["playing"] is None


# ---------------------------------------------------------------------------
# prompt_lines rendering
# ---------------------------------------------------------------------------


def test_prompt_lines_all_slots():
    lines = prompt_lines({
        "page": {"label": "Dune (novel)", "url": "https://w.test", "age_s": 30},
        "reading": {"label": "Saga #1", "kind": "comic", "age_s": 60},
        "playing": {"label": "Smooth Jazz", "kind": "radio", "age_s": 30},
        "mode": {"label": "coder", "age_s": 10},
    })
    joined = "\n".join(lines)
    assert "Dune (novel)" in joined and "this page" in joined
    assert "Saga #1" in joined and "comic" in joined
    assert "Smooth Jazz" in joined and "right now" in joined
    assert "coder area" in joined.replace("coder area", "coder area") or "the coder" in joined


def test_prompt_lines_empty_snapshot():
    assert prompt_lines({"page": None, "playing": None, "reading": None,
                         "scene": None, "working": None, "mode": None}) == []


def test_prompt_lines_scene_and_working():
    lines = prompt_lines({
        "scene": {"label": "Mira the Cartographer", "ref": "char-7", "age_s": 30},
        "working": {"label": "server.py", "path": "augmentum/proxy/server.py",
                    "age_s": 10},
    })
    joined = "\n".join(lines)
    assert "Mira the Cartographer" in joined and "this scene" in joined
    # The scene line warns her off acting inside the story — narrative
    # mode owns the story; she only grounds deixis.
    assert "don't take actions inside it" in joined
    assert "server.py" in joined and "this file" in joined
    assert "augmentum/proxy/server.py" in joined


def test_prompt_lines_working_path_same_as_label_not_doubled():
    lines = prompt_lines({"working": {"label": "notes.md", "path": "notes.md",
                                      "age_s": 5}})
    assert len(lines) == 1
    assert lines[0].count("notes.md") == 1


def test_prompt_lines_legacy_title_key_still_renders():
    # Pre-rename snapshots used page.title — keep tolerating it.
    lines = prompt_lines({"page": {"title": "Old Shape", "age_s": 5}})
    assert any("Old Shape" in ln for ln in lines)


# ── Co-author note context (active_note_lines) ────────────────────────

class _FakeNotesStore:
    def __init__(self, notes):
        self._notes = notes

    async def get(self, note_id, *, user_id=""):
        return self._notes.get(note_id)


class _FakeAppState:
    """Minimal app.state stand-in for get_referent_cache + notes_store."""

    def __init__(self, notes):
        self.notes_store = _FakeNotesStore(notes)


@pytest.mark.asyncio
async def test_active_note_lines_no_note():
    from augmentum.companion_runtime.presence_context import active_note_lines

    state = _FakeAppState({})
    assert await active_note_lines(state, "u1", "s1") == []
    # Missing app_state / user are graceful no-ops, never raises.
    assert await active_note_lines(None, "u1", "s1") == []
    assert await active_note_lines(state, "", "s1") == []


@pytest.mark.asyncio
async def test_active_note_lines_renders_title_and_tail():
    from augmentum.companion_runtime.presence_context import active_note_lines
    from augmentum.intent.dispatch import get_referent_cache

    note = {"id": "n1", "title": "Garden plan", "content": "dig the bed\nplant basil"}
    state = _FakeAppState({"n1": note})
    refs = get_referent_cache(state, "u1", "s1")
    refs.active_note_id = "n1"

    lines = await active_note_lines(state, "u1", "s1")
    assert len(lines) == 2
    assert "Garden plan" in lines[0]
    assert "plant basil" in lines[1]


@pytest.mark.asyncio
async def test_active_note_lines_tail_is_bounded():
    from augmentum.companion_runtime.presence_context import active_note_lines
    from augmentum.intent.dispatch import get_referent_cache

    long_body = "x" * 2000 + " THE END"
    state = _FakeAppState({"n2": {"id": "n2", "title": "Long", "content": long_body}})
    refs = get_referent_cache(state, "u2", "s2")
    refs.active_note_id = "n2"

    lines = await active_note_lines(state, "u2", "s2")
    assert "THE END" in lines[1]          # tail, not head
    assert len(lines[1]) < 700            # bounded
    assert "…" in lines[1]                # truncation is visible
