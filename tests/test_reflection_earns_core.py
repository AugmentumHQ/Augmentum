"""Earned Understanding P1: reflections EARN core, they aren't handed it.

A reflection is an LLM synthesis over already-stored memories, so it has
evidentiary basis (lands ACTIVE, not the PROVISIONAL quarantine) — but a
machine-made abstraction must climb to always-on CORE via the same
corroboration ladder as everything else. The legacy behavior force-promoted
it to CORE on write, letting an unverified pattern outrank user-confirmed
facts in the always-injected set.

See docs/superpowers/specs/2026-06-20-earned-understanding-design.md.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import augmentum.memory.reflection as refl
from augmentum.memory.models import Memory, MemoryTier, MemoryType


class _FakeStore:
    def __init__(self):
        self.stored: list[dict] = []
        self.tier_calls: list[tuple] = []

    async def store(self, **kwargs):
        self.stored.append(kwargs)
        return "mem-1"

    async def update_tier(self, memory_id, tier, *, user_id, **kwargs):
        self.tier_calls.append((memory_id, tier))
        return True


class _FakeBackend:
    async def chat(self, req):
        return SimpleNamespace(
            message=SimpleNamespace(
                content='{"reflection": "Values reliability and offline capability"}'
            )
        )


async def _run(force_core: bool, monkeypatch) -> _FakeStore:
    store = _FakeStore()
    mems = [
        Memory(id=f"m{i}", user_id="u1", content=f"verified fact {i}",
               memory_type=MemoryType.FACT,
               created_at="2026-06-01T00:00:00+00:00")
        for i in range(3)
    ]
    monkeypatch.setattr(refl, "_count_reflections", AsyncMock(return_value=0))
    monkeypatch.setattr(refl, "_fetch_eligible_memories", AsyncMock(return_value=mems))
    monkeypatch.setattr(refl, "_cluster_memories", lambda eligible: [mems])
    monkeypatch.setattr(
        "augmentum.config.settings.memory_reflection_force_core",
        force_core, raising=False,
    )
    ids = await refl.generate_reflections(store, _FakeBackend(), "model", user_id="u1")
    assert ids == ["mem-1"]
    return store


@pytest.mark.asyncio
async def test_reflection_lands_active_and_earns_core(monkeypatch):
    """Default: stored once, NEVER force-promoted to CORE on write."""
    store = await _run(force_core=False, monkeypatch=monkeypatch)
    assert len(store.stored) == 1
    # No forced tier bump — it lands at the store's default (ACTIVE) and must
    # earn CORE through the corroboration ladder like everything else.
    assert store.tier_calls == []


@pytest.mark.asyncio
async def test_reflection_force_core_legacy_escape_hatch(monkeypatch):
    """Legacy: force_core=True still promotes straight to CORE on write."""
    store = await _run(force_core=True, monkeypatch=monkeypatch)
    assert store.tier_calls == [("mem-1", MemoryTier.CORE)]
