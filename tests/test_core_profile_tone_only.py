"""Subtractive-memory Slice 1: the always-on Layer-3 profile carries only
EARNED (CORE-tier) facts when companion_profile_tone_only is on, instead of the
top-50 life-story dump that produced the echo chamber.

See docs/superpowers/specs/2026-06-20-memory-subtractive-design.md.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.memory.core_profile import CoreProfileManager
from augmentum.memory.models import Memory, MemoryTier, MemoryType


def _mem(mid: str, content: str, tier: MemoryTier, importance: float) -> Memory:
    return Memory(
        id=mid,
        user_id="userA",
        content=content,
        memory_type=MemoryType.FACT,
        importance=importance,
        tier=tier,
        created_at="2026-06-19T12:00:00+00:00",
        updated_at="2026-06-19T12:00:00+00:00",
    )


def _manager(memories: list[Memory]) -> CoreProfileManager:
    store = MagicMock()
    store.list_all = AsyncMock(return_value=memories)
    # app_state=None → _synthesize_profile falls back to the deterministic
    # bullet list, so we can assert exactly which facts landed in the profile.
    mgr = CoreProfileManager(store=store, app_state=None)
    mgr._persist = AsyncMock()
    mgr._flush_state = AsyncMock()
    return mgr


CORE_FACT = "User's name is Matt"
CORE_FACT_2 = "User has a cat named Moo"
TRIVIA = "User mentioned reading Isekai manga once"


@pytest.mark.asyncio
async def test_tone_only_uses_core_tier_only(monkeypatch):
    monkeypatch.setattr(
        "augmentum.memory.core_profile.settings.companion_profile_tone_only",
        True, raising=False,
    )
    mems = [
        _mem("c1", CORE_FACT, MemoryTier.CORE, 0.9),
        _mem("c2", CORE_FACT_2, MemoryTier.CORE, 0.8),
        _mem("a1", TRIVIA, MemoryTier.ACTIVE, 0.5),
        _mem("a2", "User said the weather was nice", MemoryTier.ACTIVE, 0.4),
    ]
    mgr = _manager(mems)
    await mgr._rebuild_locked("userA")
    profile = mgr._cache["userA"]

    assert CORE_FACT in profile
    assert CORE_FACT_2 in profile
    # The ACTIVE-tier trivia must NOT reach the always-on profile.
    assert TRIVIA not in profile


@pytest.mark.asyncio
async def test_tone_only_bridges_when_no_core(monkeypatch):
    """No earned CORE facts yet → thin bridge (top few), never blank."""
    monkeypatch.setattr(
        "augmentum.memory.core_profile.settings.companion_profile_tone_only",
        True, raising=False,
    )
    mems = [
        _mem(f"a{i}", f"fact number {i}", MemoryTier.ACTIVE, 0.6 - i * 0.05)
        for i in range(8)
    ]
    mgr = _manager(mems)
    await mgr._rebuild_locked("userA")
    profile = mgr._cache["userA"]

    assert profile  # not blank
    # Bridge is capped at the top 3 — the 5th-onward fact must be excluded.
    assert "fact number 0" in profile
    assert "fact number 7" not in profile


@pytest.mark.asyncio
async def test_legacy_mode_keeps_full_dump(monkeypatch):
    """tone_only=False restores the broad life-story (revert path)."""
    monkeypatch.setattr(
        "augmentum.memory.core_profile.settings.companion_profile_tone_only",
        False, raising=False,
    )
    mems = [
        _mem("c1", CORE_FACT, MemoryTier.CORE, 0.9),
        _mem("a1", TRIVIA, MemoryTier.ACTIVE, 0.5),
    ]
    mgr = _manager(mems)
    await mgr._rebuild_locked("userA")
    profile = mgr._cache["userA"]

    # Legacy mode injects ACTIVE facts too.
    assert CORE_FACT in profile
    assert TRIVIA in profile
