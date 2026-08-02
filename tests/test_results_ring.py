"""Results ring + perception contract tests.

Pins the 2026-06-12 design: full -> digest -> pull lifecycle, same-slot
supersede, touch-to-refresh via deterministic overlap, compose-time
re-inflation, indexical digests, the blind line, and context_peek as
the pull door.
"""

from __future__ import annotations

import pytest

from augmentum.companion_runtime import ring
from augmentum.companion_runtime.presence_context import (
    ATTENTION,
    perception_lines,
)
from augmentum.intent.action import ActionResult, ReferentCache
from augmentum.intent.dispatch import get_referent_cache

UID = "user-ring-test"


@pytest.fixture(autouse=True)
def _clean():
    ATTENTION.reset()
    yield
    ATTENTION.reset()


# ---------------------------------------------------------------------------
# Ring mechanics
# ---------------------------------------------------------------------------


def test_record_and_alive_roundtrip():
    refs = ReferentCache()
    ring.bump_turn(refs)
    ring.record(refs, kind="tool", slot="tool:web_search",
                label="web_search: rust borrow checker",
                digest="result gathered", detail="full text here")
    entries = ring.alive(refs)
    assert len(entries) == 1
    assert entries[0]["detail"] == "full text here"


def test_same_slot_supersedes():
    refs = ReferentCache()
    ring.bump_turn(refs)
    ring.record(refs, slot="presence:page", kind="presence",
                label="Article A", digest="open")
    ring.record(refs, slot="presence:page", kind="presence",
                label="Article B", digest="open")
    entries = ring.alive(refs)
    assert len(entries) == 1
    assert entries[0]["label"] == "Article B"


def test_cap_evicts_oldest():
    refs = ReferentCache()
    ring.bump_turn(refs)
    for i in range(6):
        ring.record(refs, kind="tool", label=f"entry {i}", digest="d")
    entries = ring.alive(refs)
    assert len(entries) == ring.RING_CAP
    assert entries[0]["label"] == "entry 2"


def test_decay_after_keep_turns():
    refs = ReferentCache()
    ring.bump_turn(refs)
    ring.record(refs, kind="tool", label="old result", digest="d")
    for _ in range(ring.DEFAULT_KEEP_TURNS + 1):
        ring.bump_turn(refs)
    assert ring.alive(refs) == []
    # Eviction is physical, not just filtered.
    assert refs.results_ring == []


def test_touch_refreshes_decay_clock():
    refs = ReferentCache()
    ring.bump_turn(refs)
    ring.record(refs, kind="tool",
                label="web_search: woodworking guides", digest="d")
    ring.bump_turn(refs)
    ring.bump_turn(refs)
    matched = ring.touch_and_match(refs, "open the woodworking one")
    assert len(matched) == 1
    # Two more turns would have killed it without the touch.
    ring.bump_turn(refs)
    ring.bump_turn(refs)
    assert len(ring.alive(refs)) == 1


def test_no_touch_on_stopword_overlap():
    refs = ReferentCache()
    ring.bump_turn(refs)
    ring.record(refs, kind="tool", label="web_search: the best thing",
                digest="result gathered")
    matched = ring.touch_and_match(refs, "what do you think about it")
    assert matched == []


def test_record_action_result_skips_clarifies():
    refs = ReferentCache()
    ring.bump_turn(refs)
    ring.record_action_result(
        refs, action_id="weather.today", args={},
        result=ActionResult(speak="What city should I use?",
                            clarify={"missing": ["location"]}),
    )
    assert ring.alive(refs) == []
    ring.record_action_result(
        refs, action_id="weather.today", args={"location": "Springfield"},
        result=ActionResult(speak="72 and clear in Springfield today."),
    )
    entries = ring.alive(refs)
    assert len(entries) == 1
    assert "Springfield" in entries[0]["label"]
    # Digest defaults to the spoken line — indexical by construction.
    assert entries[0]["digest"].startswith("72 and clear")


def test_working_state_persists_ring_and_clock():
    import asyncio
    import json

    class _Store:
        def __init__(self):
            self.data = {}
        async def get_user(self, uid, key):
            return self.data.get((uid, key))
        async def set_user(self, uid, key, val):
            self.data[(uid, key)] = val

    class _State:
        def __init__(self):
            self.settings_store = _Store()

    async def _run():
        from augmentum.companion_runtime.working_state import (
            hydrate_working_state,
            save_working_state,
        )
        state = _State()
        refs = ReferentCache()
        refs.turn_seq = 7
        ring.record(refs, kind="tool", label="saved entry", digest="d")
        await save_working_state(state, "u1", refs)
        raw = state.settings_store.data[("u1", "companion.working_state")]
        assert json.loads(raw)["turn_seq"] == 7

        fresh = ReferentCache()
        await hydrate_working_state(state, "u1", fresh)
        assert fresh.turn_seq == 7
        assert fresh.results_ring[0]["label"] == "saved entry"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Perception contract (perception_lines)
# ---------------------------------------------------------------------------


class _FakeNotesStore:
    def __init__(self, notes):
        self._notes = notes

    async def get(self, note_id, *, user_id=""):
        return self._notes.get(note_id)


class _FakeAppState:
    def __init__(self, notes=None):
        self.notes_store = _FakeNotesStore(notes or {})


@pytest.mark.asyncio
async def test_page_full_on_first_sight_then_digest():
    state = _FakeAppState()
    from augmentum.companion_runtime.presence_context import observe_attention
    observe_attention(UID, "surface.browse.page_opened", {
        "title": "Rust borrow checker", "url": "https://x.test/rust",
        "excerpt": "Ownership is Rust's most unique feature " + "x" * 100,
    })
    # Turn 1 — first sight: full excerpt rides in.
    lines = await perception_lines(state, None, UID, "s1", scoring_text="hi")
    joined = "\n".join(lines)
    assert "How the page starts: Ownership" in joined
    assert joined.rstrip().endswith("rather than guess.")

    # Several untouched turns later: digest tier — pointer, not text.
    for text in ("how was your day", "nice weather huh",
                 "going for dinner soon", "very hungry"):
        lines = await perception_lines(state, None, UID, "s1", scoring_text=text)
    joined = "\n".join(lines)
    assert "How the page starts" not in joined
    assert "peek page" in joined
    # The index line never decays while the source is fresh.
    assert "Rust borrow checker" in joined


@pytest.mark.asyncio
async def test_page_reinflates_on_reference():
    state = _FakeAppState()
    from augmentum.companion_runtime.presence_context import observe_attention
    observe_attention(UID, "surface.browse.page_opened", {
        "title": "Sourdough starter guide", "url": "https://x.test/bread",
        "excerpt": "Feed the starter twice daily with equal parts flour",
    })
    await perception_lines(state, None, UID, "s2", scoring_text="hello")
    for text in ("unrelated chatter", "more chatter"):
        await perception_lines(state, None, UID, "s2", scoring_text=text)
    # Now the user references it — server-side re-inflation, no model
    # judgment involved.
    lines = await perception_lines(
        state, None, UID, "s2",
        scoring_text="wait how often do I feed the sourdough starter?",
    )
    joined = "\n".join(lines)
    assert "How the page starts: Feed the starter" in joined


@pytest.mark.asyncio
async def test_note_full_when_changed_digest_when_idle():
    notes = {"n1": {"id": "n1", "title": "Garden plan",
                    "content": "dig the bed\nplant basil"}}
    state = _FakeAppState(notes)
    refs = get_referent_cache(state, UID, "s3")
    refs.active_note_id = "n1"

    # Fresh content — full tail.
    lines = await perception_lines(state, None, UID, "s3", scoring_text="hey")
    joined = "\n".join(lines)
    assert "plant basil" in joined

    # Unchanged for several turns — decays to count + peek pointer.
    for text in ("totally unrelated", "still unrelated",
                 "nothing to do with it", "weather chat"):
        lines = await perception_lines(state, None, UID, "s3", scoring_text=text)
    joined = "\n".join(lines)
    assert "plant basil" not in joined
    assert "peek note" in joined

    # The user EDITS the note — full tail re-earns its place.
    notes["n1"]["content"] = "dig the bed\nplant basil\nadd tomatoes"
    lines = await perception_lines(state, None, UID, "s3", scoring_text="ok")
    assert any("add tomatoes" in ln for ln in lines)


@pytest.mark.asyncio
async def test_tool_entries_render_with_age_and_reinflate():
    state = _FakeAppState()
    refs = get_referent_cache(state, UID, "s4")
    # Simulate: turn 1 happened, a search ran after compose.
    await perception_lines(state, None, UID, "s4", scoring_text="search stuff")
    ring.record(refs, kind="tool", slot="tool:web_search",
                label="web_search: beginner woodworking guides",
                digest="result gathered — details available",
                detail="1. The Wood Whisperer 2. Rex Krueger 3. Steve Ramsey")
    # Next turn: digest line with age.
    lines = await perception_lines(state, None, UID, "s4", scoring_text="cool")
    joined = "\n".join(lines)
    assert "recently looked at" in joined
    assert "[1 turn ago] web_search: beginner woodworking guides" in joined
    assert "Rex Krueger" not in joined  # detail NOT pushed unprompted
    # Referencing turn: detail re-inflates.
    lines = await perception_lines(
        state, None, UID, "s4",
        scoring_text="open the second woodworking guide",
    )
    joined = "\n".join(lines)
    assert "Rex Krueger" in joined


@pytest.mark.asyncio
async def test_detail_budget_governor_priority():
    """Everything-at-once turn: re-inflations outrank the note tail
    outrank the page excerpt; squeezed pieces keep their pointer."""
    from augmentum.companion_runtime.presence_context import observe_attention
    notes = {"n1": {"id": "n1", "title": "Plan",
                    "content": "NOTE-TAIL " * 40}}       # ~400 chars
    state = _FakeAppState(notes)
    refs = get_referent_cache(state, UID, "s10")
    refs.active_note_id = "n1"
    observe_attention(UID, "surface.browse.page_opened", {
        "title": "Big article", "url": "https://x.test/big",
        "excerpt": "PAGE-TEXT " * 60,                     # ~600 chars
    })
    ring.bump_turn(refs)
    ring.record(refs, kind="tool", slot="tool:web_search",
                label="web_search: kestrel migration",
                digest="gathered", detail="KESTREL-DETAIL " * 30)  # ~450
    # The referencing turn qualifies all three; budget fits only the
    # re-inflation + the note tail.
    lines = await perception_lines(
        state, None, UID, "s10",
        scoring_text="tell me about the kestrel migration result",
        detail_budget_chars=900,
    )
    joined = "\n".join(lines)
    assert "KESTREL-DETAIL" in joined            # priority 1 survives
    assert "NOTE-TAIL" in joined                 # priority 2 fits
    assert "PAGE-TEXT" not in joined             # priority 3 squeezed
    assert "peek page" in joined                 # ...but keeps its pointer
    assert "Big article" in joined               # index never drops


@pytest.mark.asyncio
async def test_blind_line_when_nothing_perceived():
    state = _FakeAppState()
    lines = await perception_lines(state, None, UID, "s5", scoring_text="hi")
    assert len(lines) == 1
    assert "don't" in lines[0] and "guess" in lines[0]


# ---------------------------------------------------------------------------
# context_peek — the pull door
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_peek_page_returns_full_excerpt_and_rewarms():
    from augmentum.companion_runtime.presence_context import observe_attention
    from augmentum.tools.context_peek import ContextPeekTool
    state = _FakeAppState()
    observe_attention(UID, "surface.browse.page_opened", {
        "title": "Quantum tunneling", "url": "https://x.test/qt",
        "excerpt": "Particles cross barriers they classically couldn't",
    })
    tool = ContextPeekTool(state)
    res = await tool.execute(slot="page", _context={
        "user_id": UID, "session_id": "s6",
    })
    assert res.success
    assert "Particles cross barriers" in res.output
    # The peek re-warmed the ring entry.
    refs = get_referent_cache(state, UID, "s6")
    slots = {e.get("slot") for e in refs.results_ring}
    assert "presence:page" in slots


@pytest.mark.asyncio
async def test_peek_note_full_content():
    from augmentum.tools.context_peek import ContextPeekTool
    notes = {"n9": {"id": "n9", "title": "Big note",
                    "content": "alpha\n" * 50 + "OMEGA LINE"}}
    state = _FakeAppState(notes)
    refs = get_referent_cache(state, UID, "s7")
    refs.active_note_id = "n9"
    tool = ContextPeekTool(state)
    res = await tool.execute(slot="note", _context={
        "user_id": UID, "session_id": "s7",
    })
    assert res.success
    assert "OMEGA LINE" in res.output


@pytest.mark.asyncio
async def test_peek_recent_returns_details():
    from augmentum.tools.context_peek import ContextPeekTool
    state = _FakeAppState()
    refs = get_referent_cache(state, UID, "s8")
    ring.bump_turn(refs)
    ring.record(refs, kind="tool", label="web_search: cats",
                digest="gathered", detail="cats are obligate carnivores")
    tool = ContextPeekTool(state)
    res = await tool.execute(slot="recent", _context={
        "user_id": UID, "session_id": "s8",
    })
    assert res.success
    assert "obligate carnivores" in res.output


@pytest.mark.asyncio
async def test_peek_honest_on_empty_slots():
    from augmentum.tools.context_peek import ContextPeekTool
    state = _FakeAppState()
    tool = ContextPeekTool(state)
    for slot in ("page", "note", "working"):
        res = await tool.execute(slot=slot, _context={
            "user_id": UID, "session_id": "s9",
        })
        assert res.success
        assert "No " in res.output or "no " in res.output

    bad = await tool.execute(slot="everything", _context={
        "user_id": UID, "session_id": "s9",
    })
    assert not bad.success
