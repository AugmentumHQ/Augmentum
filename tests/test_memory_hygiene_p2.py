"""Memory hygiene — wiring program Phase 2.

memory.forget (recall-then-confirm; deletion never wins a tie),
memory.tier (promote/demote), memory.save update-don't-duplicate,
recall staleness composition, and the ring's warn-before-decay line.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from augmentum.intent.action import SessionContext
from augmentum.memory.models import Memory, MemoryType
from augmentum.memory.store import MemoryStore

UID = "user-p2"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeMemoryStore(MemoryStore):
    """Subclasses the real store so notes.py's isinstance gate passes;
    skips __init__ entirely (no conn, no embedder)."""

    def __init__(self, hits=()):
        self._hits = list(hits)
        self.forgotten: list[str] = []
        self.superseded: list[tuple] = []
        self.stored: list[str] = []
        self.tier_calls: list[tuple] = []

    async def recall(self, q, *, user_id, limit=10, scope=None):
        return list(self._hits)[:limit]

    async def forget(self, memory_id, *, user_id):
        self.forgotten.append(memory_id)
        return True

    async def supersede(self, old_id, new_content, *, user_id, **kwargs):
        self.superseded.append((old_id, new_content))
        return "new-id"

    async def store(self, content=None, **kwargs):
        self.stored.append(content or kwargs.get("content", ""))
        return "mem-id"

    async def update_tier(self, memory_id, tier, *, user_id, source="system"):
        self.tier_calls.append((memory_id, tier, source))
        return True


def _mem(mid="m1", content="user's lucky number is 17", days_old=0):
    created = (datetime.now(UTC) - timedelta(days=days_old)).isoformat()
    return Memory(
        id=mid, user_id=UID, content=content,
        memory_type=MemoryType.FACT, created_at=created,
    )


def _ctx(store):
    return SessionContext(
        user_id=UID, session_id="s1",
        app_state=SimpleNamespace(memory_store=store),
    )


# ---------------------------------------------------------------------------
# memory.forget — recall-then-confirm
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_forget_without_query_clarifies():
    from augmentum.intent.builtin.memory_admin import _memory_forget
    res = await _memory_forget("forget", _ctx(FakeMemoryStore()), {})
    assert res.clarify is not None and "query" in res.clarify["missing"]


@pytest.mark.asyncio
async def test_forget_speaks_the_fact_and_parks_confirm():
    from augmentum.intent.builtin.memory_admin import _memory_forget
    store = FakeMemoryStore(hits=[_mem()])
    res = await _memory_forget(
        "forget my lucky number", _ctx(store), {"query": "lucky number"},
    )
    assert "lucky number is 17" in res.speak
    assert res.clarify is not None
    assert res.clarify["missing"] == ["confirm"]
    assert res.clarify["args"]["memory_id"] == "m1"
    assert store.forgotten == []  # nothing deleted before assent


@pytest.mark.asyncio
async def test_forget_assent_deletes():
    from augmentum.intent.builtin.memory_admin import _memory_forget
    store = FakeMemoryStore()
    res = await _memory_forget(
        "yes", _ctx(store), {"memory_id": "m1", "confirm": "yes please"},
    )
    assert store.forgotten == ["m1"]
    assert "Forgotten" in res.speak


@pytest.mark.asyncio
async def test_forget_denial_keeps():
    from augmentum.intent.builtin.memory_admin import _memory_forget
    store = FakeMemoryStore()
    res = await _memory_forget(
        "no", _ctx(store), {"memory_id": "m1", "confirm": "no, keep it"},
    )
    assert store.forgotten == []
    assert "Kept" in res.speak


@pytest.mark.asyncio
async def test_forget_ambiguous_answer_keeps():
    # Deletion never wins a tie — anything not clearly assent keeps.
    from augmentum.intent.builtin.memory_admin import _memory_forget
    store = FakeMemoryStore()
    res = await _memory_forget(
        "hmm", _ctx(store), {"memory_id": "m1", "confirm": "hmm maybe"},
    )
    assert store.forgotten == []
    assert "Kept" in res.speak


@pytest.mark.asyncio
async def test_forget_no_hits_is_honest():
    from augmentum.intent.builtin.memory_admin import _memory_forget
    res = await _memory_forget(
        "forget x", _ctx(FakeMemoryStore()), {"query": "the moon landing"},
    )
    assert "don't have anything" in res.speak
    assert res.clarify is None


# ---------------------------------------------------------------------------
# memory.tier
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tier_promote_maps_to_core_with_manual_source():
    from augmentum.intent.builtin.memory_admin import _memory_tier
    store = FakeMemoryStore(hits=[_mem()])
    res = await _memory_tier(
        "keep that long-term", _ctx(store),
        {"query": "lucky number", "level": "long_term"},
    )
    assert store.tier_calls == [("m1", "core", "manual")]
    assert "long-term" in res.speak


@pytest.mark.asyncio
async def test_tier_archive_and_missing_level_clarifies():
    from augmentum.intent.builtin.memory_admin import _memory_tier
    store = FakeMemoryStore(hits=[_mem()])
    await _memory_tier(
        "archive it", _ctx(store), {"query": "lucky", "level": "archive"},
    )
    assert store.tier_calls[-1][1] == "archive"
    res = await _memory_tier("hm", _ctx(store), {"query": "lucky"})
    assert res.clarify is not None and "level" in res.clarify["missing"]


@pytest.mark.asyncio
async def test_tier_no_hit_offers_save_instead():
    from augmentum.intent.builtin.memory_admin import _memory_tier
    res = await _memory_tier(
        "keep it", _ctx(FakeMemoryStore()),
        {"query": "anniversary", "level": "long_term"},
    )
    assert "remember it first" in res.speak


# ---------------------------------------------------------------------------
# memory.save — update-don't-duplicate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_near_duplicate_supersedes():
    from augmentum.intent.builtin.notes import _save_to_memory
    store = FakeMemoryStore(hits=[_mem(content="my favorite color is teal")])
    res = await _save_to_memory(
        "remember", _ctx(store),
        {"content": "my favorite color is teal"},
    )
    assert store.superseded and store.superseded[0][0] == "m1"
    assert store.stored == []
    assert "updated" in res.speak.lower()


@pytest.mark.asyncio
async def test_save_distinct_fact_stores_normally():
    from augmentum.intent.builtin.notes import _save_to_memory
    store = FakeMemoryStore(hits=[_mem(content="my favorite color is teal")])
    res = await _save_to_memory(
        "remember", _ctx(store),
        {"content": "my dentist appointment is on Tuesday morning"},
    )
    assert store.superseded == []
    assert store.stored == ["my dentist appointment is on Tuesday morning"]
    assert "remember" in res.speak.lower()


def test_near_duplicate_threshold_behavior():
    from augmentum.intent.builtin.memory_admin import near_duplicate
    assert near_duplicate("my favorite color is teal",
                          "my favorite color is teal")
    # Same-subject NEW VALUE must NOT merge here (extractor's lane).
    assert not near_duplicate("my favorite color is teal",
                              "my favorite color is blue")
    assert not near_duplicate("", "anything")


# ---------------------------------------------------------------------------
# memory.recall — staleness composition
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recall_addendum_carries_age_and_hedge():
    from augmentum.intent.builtin.notes import _recall
    store = FakeMemoryStore(hits=[_mem(days_old=90)])
    res = await _recall(
        "what's my lucky number", _ctx(store), {"query": "lucky number"},
    )
    assert res.prompt_addendum
    assert "(saved 3 months ago)" in res.prompt_addendum
    assert "may be stale" in res.prompt_addendum


def test_age_phrase_buckets():
    from augmentum.intent.builtin.memory_admin import age_phrase
    now = datetime.now(UTC)
    assert age_phrase(now.isoformat()) == "today"
    assert age_phrase((now - timedelta(days=5)).isoformat()) == "5 days ago"
    assert age_phrase((now - timedelta(days=21)).isoformat()) == "3 weeks ago"
    assert age_phrase((now - timedelta(days=400)).isoformat()) == "over a year ago"
    assert age_phrase("garbage") == ""


# ---------------------------------------------------------------------------
# Ring — warn-before-decay
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ring_entry_warns_on_last_alive_turn():
    from augmentum.companion_runtime import ring
    from augmentum.companion_runtime.presence_context import perception_lines
    from augmentum.intent.dispatch import get_referent_cache

    app_state = SimpleNamespace()
    uid = "user-p2-fade"
    refs = get_referent_cache(app_state, uid, "s-fade")
    ring.bump_turn(refs)
    ring.record(
        refs, kind="tool", slot="tool:web_search",
        label="searched llama releases", digest="result gathered",
        detail="release notes detail",
    )
    joined = ""
    for _ in range(3):  # default keep_turns=3 → third turn after birth
        lines = await perception_lines(app_state, None, uid, "s-fade")
        joined = "\n".join(lines)
    assert "searched llama releases" in joined
    assert "about to fade" in joined


@pytest.mark.asyncio
async def test_ring_entry_no_warning_while_fresh():
    from augmentum.companion_runtime import ring
    from augmentum.companion_runtime.presence_context import perception_lines
    from augmentum.intent.dispatch import get_referent_cache

    app_state = SimpleNamespace()
    uid = "user-p2-fresh"
    refs = get_referent_cache(app_state, uid, "s-fresh")
    ring.bump_turn(refs)
    ring.record(
        refs, kind="tool", slot="tool:web_search",
        label="searched something", digest="result gathered", detail="d",
    )
    lines = await perception_lines(app_state, None, uid, "s-fresh")
    joined = "\n".join(lines)
    assert "searched something" in joined
    assert "about to fade" not in joined


# ---------------------------------------------------------------------------
# Registration pins
# ---------------------------------------------------------------------------

def test_phase2_verbs_registered_and_core_bucketed():
    import augmentum.intent  # noqa: F401
    from augmentum.intent.manifest import VOICE_TOOLS_CORE
    from augmentum.intent.registry import REGISTRY
    ids = {a.id for a in REGISTRY.all()}
    for verb in ("memory.forget", "memory.tier"):
        assert verb in ids, f"{verb} not registered"
        assert verb in VOICE_TOOLS_CORE
