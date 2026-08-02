"""Subtractive-memory Slice 3: the read-back surface.

The topic-less "what do you remember about me?" reads back the EARNED
standing set (CORE + ACTIVE) on demand — the subtractive pull (recite
when asked, not every turn). Excludes the unproven PROVISIONAL pile and
the tucked-away ARCHIVE. Topical recall ("what did I say about X")
still flows through hybrid search unchanged.

See docs/superpowers/specs/2026-06-20-memory-subtractive-design.md.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from augmentum.intent.action import SessionContext
from augmentum.memory.models import Memory, MemoryTier, MemoryType

UID = "user-s3"


class FakeStore:
    """Implements only the read paths the recall verb touches."""

    def __init__(self, *, by_tier=None, recall_hits=()):
        self._by_tier = by_tier or {}
        self._recall_hits = list(recall_hits)
        self.recall_calls: list[str] = []
        self.list_all_calls: list[str] = []

    async def list_all(self, *, user_id, tier=None, limit=50, **kwargs):
        self.list_all_calls.append(tier)
        return list(self._by_tier.get(tier, []))[:limit]

    async def recall(self, query, *, user_id, limit=5, **kwargs):
        self.recall_calls.append(query)
        return list(self._recall_hits)[:limit]


def _mem(mid, content, tier=MemoryTier.ACTIVE):
    return Memory(
        id=mid, user_id=UID, content=content,
        memory_type=MemoryType.FACT, tier=tier,
        created_at="2026-06-01T00:00:00+00:00",
    )


def _ctx(store):
    return SessionContext(
        user_id=UID, session_id="s1",
        app_state=SimpleNamespace(memory_store=store),
    )


@pytest.mark.asyncio
async def test_empty_query_reads_back_standing_set():
    from augmentum.intent.builtin.notes import _recall
    store = FakeStore(by_tier={
        "core": [_mem("c1", "name is Matt", MemoryTier.CORE)],
        "active": [_mem("a1", "likes isekai manga", MemoryTier.ACTIVE)],
    })
    res = await _recall("what do you remember about me", _ctx(store), {})
    assert res is not None and res.prompt_addendum
    assert "name is Matt" in res.prompt_addendum
    assert "likes isekai manga" in res.prompt_addendum
    # CORE labelled distinctly from ACTIVE.
    assert "holding close" in res.prompt_addendum
    assert "remember" in res.prompt_addendum
    # Standing set is a list_all read, NOT a hybrid recall on "me".
    assert store.recall_calls == []
    assert "core" in store.list_all_calls


@pytest.mark.asyncio
async def test_generic_self_query_routes_to_standing_set():
    """A small model passing query='me' still gets the standing set."""
    from augmentum.intent.builtin.notes import _recall
    store = FakeStore(by_tier={"active": [_mem("a1", "has a cat named Moo")]})
    res = await _recall("about me", _ctx(store), {"query": "me"})
    assert res.prompt_addendum and "Moo" in res.prompt_addendum
    assert store.recall_calls == []  # did NOT hybrid-search "me"


@pytest.mark.asyncio
async def test_empty_standing_set_is_honest():
    from augmentum.intent.builtin.notes import _recall
    store = FakeStore(by_tier={})
    res = await _recall("what do you know about me", _ctx(store), {})
    assert res.short_circuit is True
    assert "not much yet" in res.speak.lower()


@pytest.mark.asyncio
async def test_provisional_and_archive_excluded():
    """Only CORE + ACTIVE are read back — unproven/archived never surface."""
    from augmentum.intent.builtin.notes import _recall
    store = FakeStore(by_tier={
        "core": [_mem("c1", "earned core fact", MemoryTier.CORE)],
        "active": [_mem("a1", "earned active fact", MemoryTier.ACTIVE)],
        "provisional": [_mem("p1", "unproven trivia", MemoryTier.PROVISIONAL)],
        "archive": [_mem("z1", "old archived fact", MemoryTier.ARCHIVE)],
    })
    res = await _recall("what do you remember about me", _ctx(store), {})
    assert "earned core fact" in res.prompt_addendum
    assert "earned active fact" in res.prompt_addendum
    assert "unproven trivia" not in res.prompt_addendum
    assert "old archived fact" not in res.prompt_addendum
    # Never even queried the excluded tiers.
    assert "provisional" not in store.list_all_calls
    assert "archive" not in store.list_all_calls


@pytest.mark.asyncio
async def test_topical_query_still_uses_hybrid_recall():
    """Regression: a real topic flows through recall(), not standing set."""
    from augmentum.intent.builtin.notes import _recall
    store = FakeStore(recall_hits=[_mem("a1", "the book was Dune")])
    res = await _recall(
        "what did I say about that book", _ctx(store),
        {"query": "that book"},
    )
    assert res.prompt_addendum and "Dune" in res.prompt_addendum
    assert store.recall_calls == ["that book"]
    assert store.list_all_calls == []  # standing set NOT consulted


def test_recall_verb_query_is_optional():
    """The general ask requires query to be omittable from the schema."""
    import augmentum.intent  # noqa: F401
    from augmentum.intent.registry import REGISTRY
    action = next(a for a in REGISTRY.all() if a.id == "memory.recall")
    assert "query" not in (action.required_args or [])
