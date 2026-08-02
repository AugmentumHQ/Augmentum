"""Tests for the narrative recall data layer.

Verifies the lookup verbs the engine will eventually expose to the
LLM as tools. Specifically pins:

* user-scoping (the seed user only sees their own data)
* graceful empty results (no exceptions when entity/thread missing)
* truncation + ``total_available`` reporting
* exact-or-alias entity match — NO fuzzy fallback so wrong-name
  recalls fail loudly instead of silently returning the wrong card
* superseded fact exclusion — recalling stale truths after the engine
  has overwritten them is exactly the failure mode the substrate
  exists to prevent

Run: python -m pytest tests/test_narrative_recall.py -v
"""
from __future__ import annotations

import pytest

from augmentum.modes.narrative.recall import (
    list_entities,
    recall_entity,
    recall_facts,
    recall_plot_thread,
)
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.narrative_persistence import NarrativePersistence
from augmentum.state.narrative_state import (
    Entity,
    EntityState,
    EntityType,
    Fact,
    PlotStatus,
    PlotThread,
)

_UID = "u-test"
_OTHER_UID = "u-other"
_SID = "s-recall"


@pytest.fixture
async def backend():
    be = SQLiteBackend(":memory:")
    await be.connect()
    yield be
    await be.close()


@pytest.fixture
async def persist(backend):
    return NarrativePersistence(backend.conn)


async def _seed(persist: NarrativePersistence) -> None:
    """Seed a small worldbuilding scene for the recall tests."""
    # Two characters, one with an alias and rich state
    elena = Entity(
        id="e-elena", session_id=_SID, name="Elena", aliases=["the Witness"],
        entity_type=EntityType.CHARACTER,
        state=EntityState(
            location="south wing",
            emotional_state="grieving",
            physical_state="kneeling",
            inventory=["bloodstained letter", "iron key"],
            relationships={"Duke Aric": "blames him", "King": "her late liege"},
        ),
    )
    duke = Entity(
        id="e-duke", session_id=_SID, name="Duke Aric",
        entity_type=EntityType.CHARACTER,
        state=EntityState(location="great hall", emotional_state="calm"),
    )
    castle = Entity(
        id="e-castle", session_id=_SID, name="Castle Mire",
        entity_type=EntityType.LOCATION,
        state=EntityState(custom={"weather": "rain"}),
    )
    await persist._save_entities(
        _SID,
        {e.id: e for e in [elena, duke, castle]},
        user_id=_UID,
    )

    facts = [
        Fact(id="f-1", session_id=_SID, content="The king was assassinated at the harvest festival.",
             established_at=10, domain="event", tags=["assassination", "king"]),
        Fact(id="f-2", session_id=_SID, content="Elena witnessed the murder from the south wing.",
             established_at=11, domain="event", tags=["witness", "elena"]),
        Fact(id="f-3", session_id=_SID, content="The duke's cousin Roderick wielded the dagger.",
             established_at=12, domain="event", confidence=0.55,
             tags=["assassination", "roderick"]),
        Fact(id="f-4", session_id=_SID, content="Elena has flat blonde hair.",
             established_at=2, domain="appearance", tags=["elena", "appearance"],
             superseded_by="f-5"),
        Fact(id="f-5", session_id=_SID, content="Elena has curly auburn hair after the curse.",
             established_at=20, domain="appearance", tags=["elena", "appearance"]),
    ]
    await persist._save_facts(_SID, facts, user_id=_UID)

    threads = [
        PlotThread(id="p-1", session_id=_SID, title="The Regicide Investigation",
                   description="Discover who killed the king and why.",
                   status=PlotStatus.ACTIVE, established_at=10),
        PlotThread(id="p-2", session_id=_SID, title="Elena's Inheritance",
                   description="Old family debt comes due.",
                   status=PlotStatus.ACTIVE, established_at=5),
        PlotThread(id="p-3", session_id=_SID, title="The Lost Crown of Mire",
                   description="A subplot that was dropped.",
                   status=PlotStatus.ABANDONED, established_at=3,
                   resolved_at=8),
    ]
    await persist._save_plot_threads(_SID, threads, user_id=_UID)


# ---------------------------------------------------------------------------
# recall_entity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_entity_exact_name(persist):
    await _seed(persist)
    r = await recall_entity(persist, _SID, user_id=_UID, name="Elena")
    assert r.total_available == 1
    assert "Elena" in r.summary
    assert "south wing" in r.summary
    assert "grieving" in r.summary
    assert "bloodstained letter" in r.summary


@pytest.mark.asyncio
async def test_recall_entity_alias_match(persist):
    await _seed(persist)
    r = await recall_entity(persist, _SID, user_id=_UID, name="the witness")
    assert r.total_available == 1
    assert "Elena" in r.summary


@pytest.mark.asyncio
async def test_recall_entity_not_found_lists_known(persist):
    await _seed(persist)
    r = await recall_entity(persist, _SID, user_id=_UID, name="Mythical Stranger")
    assert r.total_available == 3  # 3 entities exist; just not this one
    assert r.items == []
    # Failure surface must name the misses + suggest what IS known
    assert "Mythical Stranger" in r.summary
    assert "Elena" in r.summary
    assert "Duke Aric" in r.summary


@pytest.mark.asyncio
async def test_recall_entity_user_scoped(persist):
    """A second user's empty session must not see _UID's entities."""
    await _seed(persist)
    r = await recall_entity(persist, _SID, user_id=_OTHER_UID, name="Elena")
    assert r.total_available == 0


# ---------------------------------------------------------------------------
# list_entities
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_entities_filtered_by_type(persist):
    await _seed(persist)
    chars = await list_entities(
        persist, _SID, user_id=_UID, entity_type=EntityType.CHARACTER,
    )
    assert chars.total_available == 2
    locs = await list_entities(
        persist, _SID, user_id=_UID, entity_type=EntityType.LOCATION,
    )
    assert locs.total_available == 1
    assert "Castle Mire" in locs.summary
    items = await list_entities(
        persist, _SID, user_id=_UID, entity_type=EntityType.ITEM,
    )
    assert items.total_available == 0


# ---------------------------------------------------------------------------
# recall_facts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_facts_substring_match(persist):
    await _seed(persist)
    r = await recall_facts(persist, _SID, user_id=_UID, query="assassinat")
    assert r.total_available >= 1
    assert "harvest festival" in r.summary or "Roderick" in r.summary


@pytest.mark.asyncio
async def test_recall_facts_excludes_superseded(persist):
    """The blonde-hair fact (f-4) is superseded by curly-auburn (f-5).
    A query for hair must NOT surface the stale truth."""
    await _seed(persist)
    r = await recall_facts(persist, _SID, user_id=_UID, query="hair")
    assert "blonde" not in r.summary
    assert "curly auburn" in r.summary


@pytest.mark.asyncio
async def test_recall_facts_empty_query_returns_recent(persist):
    """Empty query returns the most recently established facts."""
    await _seed(persist)
    r = await recall_facts(persist, _SID, user_id=_UID, query="", limit=2)
    assert len(r.items) == 2
    # f-5 (established_at=20) and f-3 (12) are the two most recent non-superseded.
    ids = {item["id"] for item in r.items}
    assert ids == {"f-5", "f-3"}


@pytest.mark.asyncio
async def test_recall_facts_no_match_returns_clean_summary(persist):
    await _seed(persist)
    r = await recall_facts(persist, _SID, user_id=_UID, query="xyzzy")
    assert r.items == []
    assert "xyzzy" in r.summary


@pytest.mark.asyncio
async def test_recall_facts_caps_limit(persist):
    """Even a 'limit=10000' call can't flood the model — capped at 10."""
    await _seed(persist)
    r = await recall_facts(persist, _SID, user_id=_UID, query="", limit=10_000)
    assert len(r.items) <= 10


# ---------------------------------------------------------------------------
# recall_plot_thread
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_plot_thread_by_id(persist):
    await _seed(persist)
    r = await recall_plot_thread(persist, _SID, user_id=_UID, query="p-1")
    assert r.total_available == 1
    assert "Regicide Investigation" in r.summary


@pytest.mark.asyncio
async def test_recall_plot_thread_by_title_substring(persist):
    await _seed(persist)
    r = await recall_plot_thread(persist, _SID, user_id=_UID, query="regicide")
    assert r.total_available == 1
    assert "Regicide Investigation" in r.summary


@pytest.mark.asyncio
async def test_recall_plot_thread_includes_resolved(persist):
    """Resolved/abandoned threads stay recallable so the model can
    answer 'what happened with the lost crown' callbacks."""
    await _seed(persist)
    r = await recall_plot_thread(persist, _SID, user_id=_UID, query="lost crown")
    assert r.total_available == 1
    assert "abandoned" in r.summary.lower() or "ABANDONED" in r.summary


@pytest.mark.asyncio
async def test_recall_plot_thread_not_found_lists_active(persist):
    await _seed(persist)
    r = await recall_plot_thread(persist, _SID, user_id=_UID, query="dragons")
    assert r.items == []
    # Failure surface must show what active threads ARE in scope
    assert "Regicide Investigation" in r.summary or "Inheritance" in r.summary
