"""memory_recall must search the *logged-in* user's memory.

Pins the 2026-06-18 fix: ``MemoryStore.recall`` defaults ``user_id`` to
``"default"`` when omitted, and ``MemoryRecallTool.execute`` never
extracted or passed it — so every recall hit the wrong bucket and
returned "No relevant memories found" for real users (the same
silent-empty class as the media-recommendation bug). These tests guard
that the tool threads the caller's user_id and refuses without one.
"""

from __future__ import annotations

import pytest

from augmentum.tools.memory_recall import MemoryRecallTool


class _CapturingStore:
    """Records the user_id recall() was called with; returns nothing."""

    def __init__(self) -> None:
        self.seen_user_id: str | None = None

    async def recall(self, query, user_id="default", **kwargs):  # noqa: ARG002
        self.seen_user_id = user_id
        return []


@pytest.mark.asyncio
async def test_memory_recall_threads_logged_in_user():
    store = _CapturingStore()
    tool = MemoryRecallTool(store)
    result = await tool.execute(
        query="what do I like", _context={"user_id": "usr_real"},
    )
    assert result.success
    assert store.seen_user_id == "usr_real"  # NOT "default"


@pytest.mark.asyncio
async def test_memory_recall_accepts_top_level_user_id():
    store = _CapturingStore()
    tool = MemoryRecallTool(store)
    await tool.execute(query="x", _user_id="usr_chain")
    assert store.seen_user_id == "usr_chain"


@pytest.mark.asyncio
async def test_memory_recall_refuses_without_user_context():
    store = _CapturingStore()
    tool = MemoryRecallTool(store)
    result = await tool.execute(query="x")
    assert not result.success
    assert store.seen_user_id is None  # never reached the store
