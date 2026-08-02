"""Unit tests for the Cardsmith scratchpad + reference index.

Covers:
- ScratchEntry serialization round-trip
- add_to_scratchpad dedup by path
- Active zone cap demotion (oldest active → indexed)
- Total cap LRU eviction (non-consumed only)
- mark_consumed transitions
- build_reference_index extracts terms from title/aliases/sections/infobox
- Title variations (parens, leading articles)
- Proper-noun extraction from infobox values
- recall_for_turn skips active entries
- recall_for_turn surfaces indexed/consumed when terms match
- render_scratchpad_block produces well-formed XML for all zones
- ContentDoc → ScratchEntry conversion preserves field richness
"""

from __future__ import annotations

import time

import pytest

from augmentum.knowledge.content_extractor import ContentDoc, Link
from augmentum.modes.narrative.cardsmith.scratchpad import (
    ZONE_ACTIVE,
    ZONE_CONSUMED,
    ZONE_INDEXED,
    ScratchEntry,
    add_to_scratchpad,
    build_reference_index,
    deserialize_scratchpad,
    mark_consumed,
    recall_for_turn,
    render_scratchpad_block,
    serialize_scratchpad,
)

# ── ScratchEntry serialization ────────────────────────────────────────────

def test_scratch_entry_round_trips_through_dict():
    entry = ScratchEntry(
        url="https://example.com/wiki/X",
        path="/wiki/X",
        title="X",
        summary="Summary text",
        sections={"History": "Founded by..."},
        infobox={"Type": "Kingdom"},
        aliases=["X-Land"],
        source_kind="fandom",
        zone=ZONE_ACTIVE,
    )
    d = entry.to_dict()
    back = ScratchEntry.from_dict(d)
    assert back == entry


def test_scratch_entry_from_content_doc_copies_fields():
    doc = ContentDoc(
        url="https://example.com/wiki/Sapin",
        source_kind="fandom",
        title="Sapin Kingdom",
        summary="The largest kingdom of Dicathen.",
        sections={"History": "Founded...", "Geography": "Mountains..."},
        infobox={"Capital": "Etistin"},
        aliases=["Sapin"],
        extracted_links=[
            Link(title="Etistin", path="/wiki/Etistin", is_internal=True),
            Link(title="Reynolds Family", path="/wiki/Reynolds_Family", is_internal=True),
        ],
    )
    entry = ScratchEntry.from_content_doc(doc, path="/wiki/Sapin_Kingdom")
    assert entry.url == doc.url
    assert entry.path == "/wiki/Sapin_Kingdom"
    assert entry.title == doc.title
    assert entry.sections == doc.sections
    assert entry.aliases == doc.aliases
    # Regression guard — extracted_links must round-trip from doc to entry
    # so the system prompt can render real paths for fetch_targets[].
    assert len(entry.extracted_links) == 2
    assert entry.extracted_links[0]["title"] == "Etistin"
    assert entry.extracted_links[0]["path"] == "/wiki/Etistin"
    assert entry.zone == ZONE_ACTIVE


def test_render_active_doc_shows_extracted_links():
    """Without surfacing real links, the model hallucinates fetch paths.
    Audit log on dendro.fandom.com surfaced this — model invented Luminous,
    Stockholm, QuSense, Princess (none exist) when scratchpad lacked links.
    """
    sp = [
        ScratchEntry(
            url="https://example.com/wiki/Home",
            path="https://example.com/wiki/Home",
            title="Home Page",
            summary="The wiki main page.",
            extracted_links=[
                {"title": "Sapin Kingdom", "path": "/wiki/Sapin_Kingdom", "is_internal": True},
                {"title": "Mana", "path": "/wiki/Mana", "is_internal": True},
            ],
            zone=ZONE_ACTIVE,
        ),
    ]
    block = render_scratchpad_block(sp)
    assert "<links" in block
    assert "/wiki/Sapin_Kingdom" in block
    assert "/wiki/Mana" in block
    assert "Sapin Kingdom" in block


def test_serialize_deserialize_preserves_state():
    sp = [
        ScratchEntry(url="u1", path="/wiki/A", title="A", summary="a"),
        ScratchEntry(url="u2", path="/wiki/B", title="B", summary="b", zone=ZONE_INDEXED),
    ]
    serialized = serialize_scratchpad(sp)
    restored = deserialize_scratchpad(serialized)
    assert len(restored) == 2
    assert restored[0].path == "/wiki/A"
    assert restored[1].zone == ZONE_INDEXED


# ── add_to_scratchpad ─────────────────────────────────────────────────────

def test_add_to_scratchpad_dedups_by_path():
    sp = []
    e1 = ScratchEntry(url="u1", path="/wiki/A", title="A", summary="x")
    e2 = ScratchEntry(url="u1-different", path="/wiki/A", title="Same path", summary="y")
    assert add_to_scratchpad(sp, e1) is True
    assert add_to_scratchpad(sp, e2) is False
    assert len(sp) == 1


def test_active_cap_demotes_oldest_to_indexed(monkeypatch):
    from augmentum.modes.narrative.cardsmith import scratchpad as sp_mod
    monkeypatch.setattr(sp_mod, "_ACTIVE_CAP", 3)

    sp = []
    for i in range(5):
        e = ScratchEntry(
            url=f"u{i}", path=f"/wiki/{i}", title=f"T{i}", summary="s",
            fetched_at=time.time() + i,  # younger entries have larger ts
        )
        add_to_scratchpad(sp, e)

    actives = [e for e in sp if e.zone == ZONE_ACTIVE]
    indexed = [e for e in sp if e.zone == ZONE_INDEXED]
    assert len(actives) == 3
    assert len(indexed) == 2
    # The 3 newest should be active
    active_paths = {e.path for e in actives}
    assert active_paths == {"/wiki/2", "/wiki/3", "/wiki/4"}


def test_total_cap_evicts_oldest_non_consumed(monkeypatch):
    from augmentum.modes.narrative.cardsmith import scratchpad as sp_mod
    monkeypatch.setattr(sp_mod, "_TOTAL_CAP", 3)
    monkeypatch.setattr(sp_mod, "_ACTIVE_CAP", 2)

    sp = []
    for i in range(5):
        e = ScratchEntry(
            url=f"u{i}", path=f"/wiki/{i}", title=f"T{i}", summary="s",
            fetched_at=time.time() + i,
        )
        add_to_scratchpad(sp, e)

    # Cap at 3 — only the 3 newest survive
    assert len(sp) == 3
    paths = {e.path for e in sp}
    assert paths == {"/wiki/2", "/wiki/3", "/wiki/4"}


def test_consumed_entries_protected_from_eviction(monkeypatch):
    from augmentum.modes.narrative.cardsmith import scratchpad as sp_mod
    monkeypatch.setattr(sp_mod, "_TOTAL_CAP", 2)
    monkeypatch.setattr(sp_mod, "_ACTIVE_CAP", 1)

    sp = []
    consumed = ScratchEntry(
        url="u_c", path="/wiki/Consumed", title="C", summary="c",
        zone=ZONE_CONSUMED, consumed_by="lorebook[0]",
        fetched_at=time.time(),
    )
    sp.append(consumed)
    for i in range(3):
        e = ScratchEntry(
            url=f"u{i}", path=f"/wiki/{i}", title=f"T{i}", summary="s",
            fetched_at=time.time() + i + 1,
        )
        add_to_scratchpad(sp, e)

    # Consumed entry should still be in scratchpad (immune to LRU)
    paths = {e.path for e in sp}
    assert "/wiki/Consumed" in paths


# ── mark_consumed ─────────────────────────────────────────────────────────

def test_mark_consumed_transitions_zones():
    sp = [
        ScratchEntry(url="u1", path="/wiki/A", title="A", summary="a", zone=ZONE_ACTIVE),
        ScratchEntry(url="u2", path="/wiki/B", title="B", summary="b", zone=ZONE_INDEXED),
        ScratchEntry(url="u3", path="/wiki/C", title="C", summary="c", zone=ZONE_ACTIVE),
    ]
    n = mark_consumed(sp, ["/wiki/A", "/wiki/B"], consumer="lorebook[0]")
    assert n == 2
    by_path = {e.path: e for e in sp}
    assert by_path["/wiki/A"].zone == ZONE_CONSUMED
    assert by_path["/wiki/A"].consumed_by == "lorebook[0]"
    assert by_path["/wiki/B"].zone == ZONE_CONSUMED
    assert by_path["/wiki/C"].zone == ZONE_ACTIVE  # untouched


def test_mark_consumed_idempotent_on_already_consumed():
    sp = [
        ScratchEntry(url="u1", path="/wiki/A", title="A", summary="a", zone=ZONE_CONSUMED, consumed_by="prev"),
    ]
    n = mark_consumed(sp, ["/wiki/A"], consumer="next")
    assert n == 0  # already consumed, not re-marked
    assert sp[0].consumed_by == "prev"


# ── Reference index ───────────────────────────────────────────────────────

def test_index_extracts_title_aliases_sections():
    sp = [
        ScratchEntry(
            url="u1", path="/wiki/Sapin",
            title="Sapin Kingdom",
            summary="...",
            aliases=["Sapin", "Sapin Kingdom of the South"],
            sections={"History": "...", "Government": "..."},
        ),
    ]
    index = build_reference_index(sp)
    keys = set(index.keys())
    # Title (lowercased)
    assert "sapin kingdom" in keys
    # Aliases
    assert "sapin" in keys
    assert "sapin kingdom of the south" in keys
    # Section headings
    assert "history" in keys
    assert "government" in keys


def test_index_title_variations_parens_and_articles():
    sp = [
        ScratchEntry(url="u1", path="/wiki/X", title="Sapin (Kingdom)", summary=""),
        ScratchEntry(url="u2", path="/wiki/Y", title="The Lance Order", summary=""),
    ]
    index = build_reference_index(sp)
    # Parenthetical stripped
    assert "sapin" in index
    # Leading article stripped
    assert "lance order" in index


def test_index_extracts_proper_nouns_from_infobox_values():
    sp = [
        ScratchEntry(
            url="u1", path="/wiki/Tessia",
            title="Tessia Eralith",
            summary="",
            infobox={"Family": "Virion Eralith and Merial Eralith"},
        ),
    ]
    index = build_reference_index(sp)
    # Multi-word proper nouns extracted from infobox
    assert "virion eralith" in index or "merial eralith" in index


def test_index_short_terms_filtered_out():
    sp = [
        ScratchEntry(
            url="u1", path="/wiki/A",
            title="X",  # 1 char — should be skipped
            summary="",
            aliases=["A"],  # also too short
        ),
    ]
    index = build_reference_index(sp)
    assert len(index) == 0  # nothing extracted


# ── recall_for_turn ───────────────────────────────────────────────────────

def test_recall_skips_active_zone():
    sp = [
        ScratchEntry(url="u1", path="/wiki/A", title="Tessia", summary="...", zone=ZONE_ACTIVE),
    ]
    index = build_reference_index(sp)
    recalled = recall_for_turn("Tell me about Tessia", "", index, sp)
    assert recalled == []  # active is in baseline prompt; recall doesn't duplicate


def test_recall_surfaces_indexed_zone():
    sp = [
        ScratchEntry(url="u1", path="/wiki/A", title="Tessia", summary="...", zone=ZONE_INDEXED),
    ]
    index = build_reference_index(sp)
    recalled = recall_for_turn("Tell me about Tessia", "", index, sp)
    assert len(recalled) == 1
    assert recalled[0].title == "Tessia"


def test_recall_surfaces_consumed_zone():
    sp = [
        ScratchEntry(
            url="u1", path="/wiki/A", title="Mana", summary="...",
            zone=ZONE_CONSUMED, consumed_by="lorebook[2]",
        ),
    ]
    index = build_reference_index(sp)
    recalled = recall_for_turn("how does mana work?", "", index, sp)
    assert len(recalled) == 1
    assert recalled[0].consumed_by == "lorebook[2]"


def test_recall_caps_at_max():
    sp = [
        ScratchEntry(url=f"u{i}", path=f"/wiki/{i}", title=f"Topic{i}", summary="x", zone=ZONE_INDEXED)
        for i in range(10)
    ]
    index = build_reference_index(sp)
    text = " ".join(f"Topic{i}" for i in range(10))
    recalled = recall_for_turn(text, "", index, sp, max_recalls=3)
    assert len(recalled) == 3


def test_recall_no_match_returns_empty():
    sp = [
        ScratchEntry(url="u1", path="/wiki/A", title="Tessia", summary="...", zone=ZONE_INDEXED),
    ]
    index = build_reference_index(sp)
    recalled = recall_for_turn("completely unrelated topic", "", index, sp)
    assert recalled == []


def test_recall_uses_conversation_tail_too():
    sp = [
        ScratchEntry(url="u1", path="/wiki/A", title="Sapin", summary="...", zone=ZONE_INDEXED),
    ]
    index = build_reference_index(sp)
    # User message doesn't mention Sapin, but earlier turn did
    recalled = recall_for_turn(
        "tell me more",
        "Sapin is the largest kingdom in Dicathen.",
        index,
        sp,
    )
    assert len(recalled) == 1


# ── render_scratchpad_block ───────────────────────────────────────────────

def test_render_empty_scratchpad_returns_empty_string():
    assert render_scratchpad_block([]) == ""


def test_render_includes_all_three_zones():
    sp = [
        ScratchEntry(url="u1", path="/wiki/A", title="A", summary="active doc", zone=ZONE_ACTIVE),
        ScratchEntry(url="u2", path="/wiki/B", title="B", summary="indexed doc", zone=ZONE_INDEXED),
        ScratchEntry(
            url="u3", path="/wiki/C", title="C", summary="consumed doc",
            zone=ZONE_CONSUMED, consumed_by="lorebook[0]",
        ),
    ]
    block = render_scratchpad_block(sp)
    assert 'active="1"' in block
    assert 'indexed="1"' in block
    assert 'consumed="1"' in block
    assert "/wiki/A" in block
    assert "/wiki/B" in block
    assert "/wiki/C" in block
    assert "lorebook[0]" in block


def test_render_includes_recalled_block_when_provided():
    sp = [
        ScratchEntry(url="u1", path="/wiki/X", title="X", summary="hidden", zone=ZONE_INDEXED),
    ]
    block = render_scratchpad_block(sp, recalled=sp)
    assert "<recalled>" in block
    assert "</recalled>" in block
    assert "hidden" in block  # full content surfaces in recall


def test_render_active_doc_shows_full_sections():
    sp = [
        ScratchEntry(
            url="u1", path="/wiki/A", title="A",
            summary="The summary.",
            sections={"History": "Long history.", "Politics": "Complex politics."},
            zone=ZONE_ACTIVE,
        ),
    ]
    block = render_scratchpad_block(sp)
    assert "Long history" in block
    assert "Complex politics" in block


def test_render_includes_recently_fetched_hint():
    """Entries fetched within the recent window surface in a hint block
    so the model doesn't re-request them via fetch_targets[]."""
    sp = [
        ScratchEntry(
            url="u1", path="/wiki/Sapin", title="Sapin Kingdom", summary="...",
            zone=ZONE_INDEXED, fetched_at=time.time() - 10.0,
        ),
        ScratchEntry(
            url="u2", path="/wiki/Mana", title="Mana", summary="...",
            zone=ZONE_INDEXED, fetched_at=time.time() - 30.0,
        ),
    ]
    block = render_scratchpad_block(sp)
    assert "<recently_fetched" in block
    assert "/wiki/Sapin" in block
    assert "/wiki/Mana" in block


def test_render_omits_recently_fetched_for_old_entries():
    sp = [
        ScratchEntry(
            url="u1", path="/wiki/Old", title="Old Doc", summary="...",
            zone=ZONE_INDEXED, fetched_at=time.time() - 600.0,  # 10 min ago
        ),
    ]
    block = render_scratchpad_block(sp)
    assert "<recently_fetched" not in block


def test_render_indexed_doc_only_shows_digest():
    sp = [
        ScratchEntry(
            url="u1", path="/wiki/A", title="A",
            summary="A short summary.",
            sections={"Hidden": "Should not appear in indexed digest."},
            zone=ZONE_INDEXED,
        ),
    ]
    block = render_scratchpad_block(sp)
    assert "A short summary" in block
    assert "Should not appear" not in block


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
