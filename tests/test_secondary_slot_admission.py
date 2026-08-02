"""Admission control + Slot B resource visibility for the secondary slot.

Two correctness properties of the professional implementation:

1. ``ResourceLedger.check_engine_fit`` is the admission gate — it blocks a
   load that the accumulated profile says won't fit current free VRAM, but
   stays conservative (allows) when footprint or free VRAM is unknown.
2. ``ResourceLedger.collect`` surfaces the SECOND resident engine's loaded
   model, so its VRAM is attributed instead of landing in
   ``unattributed_vram_mb``.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.resource.ledger import ModelProfile, ResourceLedger, ResourceSnapshot


def _snap(free_mb: int) -> ResourceSnapshot:
    return ResourceSnapshot(timestamp=datetime(2026, 6, 10), gpu_free_mb=free_mb)


async def _fit(ledger, model, *, free_mb, profile_vram=0, size_bytes=0):
    ledger.collect = AsyncMock(return_value=_snap(free_mb))
    prof = (
        ModelProfile(model_name=model, subsystem="llm", backend="engine", vram_mb=profile_vram)
        if profile_vram
        else None
    )
    ledger.get_model_profile = AsyncMock(return_value=prof)
    return await ledger.check_engine_fit(model, size_bytes=size_bytes)


@pytest.mark.asyncio
async def test_fit_blocks_when_profile_exceeds_free():
    ledger = ResourceLedger(db=None)
    ok, reason, needed, free = await _fit(ledger, "Big", free_mb=6000, profile_vram=18000)
    assert ok is False
    assert needed == 18000 and free == 6000
    assert "GB" in reason


@pytest.mark.asyncio
async def test_fit_allows_when_within_free():
    ledger = ResourceLedger(db=None)
    ok, _, needed, free = await _fit(ledger, "Small", free_mb=12000, profile_vram=4000)
    assert ok is True
    assert needed == 4000 and free == 12000


@pytest.mark.asyncio
async def test_fit_allows_unknown_profile_and_no_size():
    """First-ever load (no profile, no size) is never blocked — conservative."""
    ledger = ResourceLedger(db=None)
    ok, reason, needed, _ = await _fit(ledger, "New", free_mb=6000)
    assert ok is True
    assert needed == 0 and reason == ""


@pytest.mark.asyncio
async def test_fit_uses_size_bytes_fallback_when_no_profile():
    ledger = ResourceLedger(db=None)
    # 20 GB file, 6 GB free → blocked via size fallback.
    ok, _, needed, _ = await _fit(
        ledger, "NoProfile", free_mb=6000, size_bytes=20 * 1024 * 1024 * 1024,
    )
    assert ok is False
    assert needed == 20 * 1024  # MB


@pytest.mark.asyncio
async def test_fit_allows_when_free_unknown():
    """nvidia-smi unavailable (free=0) → don't block on missing data."""
    ledger = ResourceLedger(db=None)
    ok, _, _, free = await _fit(ledger, "X", free_mb=0, profile_vram=18000)
    assert ok is True
    assert free == 0


@pytest.mark.asyncio
async def test_headroom_is_respected():
    """needed just under raw free but over the 90% headroom → blocked."""
    ledger = ResourceLedger(db=None)
    # free=10000, headroom = 9000. needed 9500 < 10000 but >= 9000 → block.
    ok, _, _, _ = await _fit(ledger, "Edge", free_mb=10000, profile_vram=9500)
    assert ok is False


@pytest.mark.asyncio
async def test_collect_surfaces_both_engine_slots():
    """Primary + Slot B both appear in the snapshot models list."""
    ledger = ResourceLedger(db=None)

    def _mgr(model_id):
        m = MagicMock()
        m.status.return_value = {
            "state": "ready",
            "model_id": model_id,
            "pid": 0,
            "actual_memory": {"vram_total_mib": 4000, "ram_total_mib": 0},
        }
        return m

    ledger.set_llama_manager(_mgr("Primary-A"))
    ledger.set_secondary_slot(SimpleNamespace(manager=_mgr("SlotB-B")))

    with patch("augmentum.resource.ledger._probe_gpu", return_value=("GPU", 24000, 8000, 16000)), \
         patch("augmentum.resource.ledger._probe_gpu_processes", return_value=[]), \
         patch("augmentum.resource.ledger._probe_ram", return_value=(32000, 8000, 24000)):
        snap = await ledger.collect(force=True)

    names = {m.name for m in snap.models}
    assert "Primary-A" in names
    assert "SlotB-B" in names
    engine_models = [m for m in snap.models if m.backend == "engine"]
    assert len(engine_models) == 2


@pytest.mark.asyncio
async def test_two_engines_per_model_vram_bounded_by_device_total():
    """Regression for the 17.4 + 4.4 = 21.8 GB > 12.8 GB device-total bug.

    A model WITH reliable actual_memory keeps its real number; a model
    WITHOUT it gets only the UNCLAIMED remainder of device VRAM — never its
    (inflated) load-plan estimate — so the attributed sum can't exceed what
    the GPU is actually using."""
    ledger = ResourceLedger(db=None)

    reliable = MagicMock()
    reliable.status.return_value = {
        "state": "ready", "model_id": "gemma", "pid": 0,
        "actual_memory": {"vram_total_mib": 4400, "ram_total_mib": 1100},
    }
    unknown = MagicMock()
    unknown.status.return_value = {
        "state": "ready", "model_id": "nemotron", "pid": 0,
        # No actual_memory; a wildly inflated plan estimate that must NOT be
        # used verbatim (this was the 17.4 GB the panel showed).
        "load_plan": {"memory": {"estimated_vram_mb": 17400}},
    }
    ledger.set_llama_manager(reliable)
    ledger.set_secondary_slot(SimpleNamespace(manager=unknown))

    with patch("augmentum.resource.ledger._probe_gpu", return_value=("GPU", 24000, 12800, 11200)), \
         patch("augmentum.resource.ledger._probe_gpu_processes", return_value=[]), \
         patch("augmentum.resource.ledger._probe_ram", return_value=(64000, 6000, 58000)):
        snap = await ledger.collect(force=True)

    by_name = {m.name: m.vram_mb for m in snap.models if m.backend == "engine"}
    assert by_name["gemma"] == 4400                 # reliable, unchanged
    assert by_name["nemotron"] == 12800 - 4400      # remainder, NOT 17400
    assert by_name["gemma"] + by_name["nemotron"] <= 12800  # never exceeds used


@pytest.mark.asyncio
async def test_single_engine_no_actual_memory_gets_device_used():
    """Single engine, no per-process data → it owns all used VRAM (the
    original, correct single-engine behavior is preserved)."""
    ledger = ResourceLedger(db=None)
    only = MagicMock()
    only.status.return_value = {
        "state": "ready", "model_id": "solo", "pid": 0,
        "load_plan": {"memory": {"estimated_vram_mb": 9999}},
    }
    ledger.set_llama_manager(only)

    with patch("augmentum.resource.ledger._probe_gpu", return_value=("GPU", 24000, 7000, 17000)), \
         patch("augmentum.resource.ledger._probe_gpu_processes", return_value=[]), \
         patch("augmentum.resource.ledger._probe_ram", return_value=(64000, 6000, 58000)):
        snap = await ledger.collect(force=True)

    by_name = {m.name: m.vram_mb for m in snap.models if m.backend == "engine"}
    assert by_name["solo"] == 7000  # all device-used VRAM, not the 9999 estimate
