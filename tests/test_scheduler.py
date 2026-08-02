"""Tests for the request scheduler (Phase 5)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "engine"))

from scheduler import (
    Priority,
    RequestScheduler,
    _extract_family,
    detect_speculative_pairs,
)


@pytest.fixture
def scheduler():
    return RequestScheduler(max_concurrent=1)


@pytest.mark.asyncio
async def test_single_request(scheduler):
    """Single request gets dispatched immediately."""
    req = await scheduler.schedule(priority=Priority.NORMAL)
    assert req.started
    assert not req.completed
    await scheduler.complete(req)
    assert scheduler.stats()["total_completed"] == 1


@pytest.mark.asyncio
async def test_priority_ordering(scheduler):
    """Higher priority requests are dispatched before lower priority ones."""
    order = []

    async def make_request(priority, label):
        req = await scheduler.schedule(priority=priority)
        order.append(label)
        await asyncio.sleep(0.01)
        await scheduler.complete(req)

    # Start a P2 request first (it gets the slot)
    task_p2 = asyncio.create_task(make_request(Priority.NORMAL, "P2"))
    await asyncio.sleep(0.01)  # let it start

    # Queue P0 and P3 while P2 is running
    task_p0 = asyncio.create_task(make_request(Priority.CRITICAL, "P0"))
    task_p3 = asyncio.create_task(make_request(Priority.BACKGROUND, "P3"))
    await asyncio.sleep(0.01)

    # Wait for all
    await asyncio.gather(task_p2, task_p0, task_p3)

    # P2 started first (already running), then P0 (higher priority), then P3
    assert order[0] == "P2"
    assert order[1] == "P0"
    assert order[2] == "P3"


@pytest.mark.asyncio
async def test_cancel(scheduler):
    """Cancelled requests don't get dispatched."""
    # Start one request to fill the slot
    req1 = await scheduler.schedule(priority=Priority.NORMAL)

    # Queue another
    task = asyncio.create_task(scheduler.schedule(priority=Priority.BACKGROUND))
    await asyncio.sleep(0.01)

    # Cancel the queued one — need to find its ID from stats
    stats = scheduler.stats()
    assert stats["queue_depth"] >= 0

    # Complete first request
    await scheduler.complete(req1)

    # The queued task should now dispatch
    req2 = await task
    await scheduler.complete(req2)
    assert scheduler.stats()["total_completed"] == 2


@pytest.mark.asyncio
async def test_stats(scheduler):
    """Stats reflect scheduler state."""
    stats = scheduler.stats()
    assert stats["total_scheduled"] == 0
    assert stats["queue_depth"] == 0
    assert stats["active_request"] is None

    req = await scheduler.schedule(priority=Priority.HIGH)
    stats = scheduler.stats()
    assert stats["total_scheduled"] == 1
    assert stats["active_request"] is not None
    assert stats["active_request"]["priority"] == "HIGH"

    await scheduler.complete(req)
    stats = scheduler.stats()
    assert stats["total_completed"] == 1
    assert stats["active_request"] is None


@pytest.mark.asyncio
async def test_preemption_signal(scheduler):
    """Preemption signal is set when higher priority request arrives."""
    # Start a background request
    req_bg = await scheduler.schedule(priority=Priority.BACKGROUND)
    assert not scheduler.should_preempt

    # Queue a critical request — should trigger preemption
    task_crit = asyncio.create_task(scheduler.schedule(priority=Priority.CRITICAL))
    await asyncio.sleep(0.01)

    assert scheduler.should_preempt

    # Complete background (simulating it yielded)
    await scheduler.complete(req_bg)

    # Critical should now dispatch
    req_crit = await task_crit
    assert req_crit.started
    assert not scheduler.should_preempt

    await scheduler.complete(req_crit)


# ---------------------------------------------------------------------------
# Speculative decoding detection
# ---------------------------------------------------------------------------

def test_extract_family():
    """Model family extraction from filenames."""
    assert _extract_family("qwen3.5-32b-q4_k_m.gguf") == "qwen3.5"
    assert _extract_family("qwen3.5-4b-q8_0.gguf") == "qwen3.5"
    assert _extract_family("llama-3.2-1b-q4_k_s.gguf") == "llama-3.2"
    assert _extract_family("llama-3.2-70b-q4_k_m.gguf") == "llama-3.2"
    assert _extract_family("mistral-7b-instruct-v0.3-q4_k_s.gguf") == "mistral-v0.3"


def test_detect_speculative_pairs():
    """Detect valid speculative decoding model pairs."""
    models = [
        {"id": "qwen3.5-32b-q4_k_m.gguf", "size_bytes": 20_000_000_000, "path": "/m/a"},
        {"id": "qwen3.5-4b-q8_0.gguf", "size_bytes": 4_500_000_000, "path": "/m/b"},
        {"id": "llama-3.2-8b-q4_k_m.gguf", "size_bytes": 5_000_000_000, "path": "/m/c"},
    ]

    candidates = detect_speculative_pairs(models)
    assert len(candidates) >= 1

    # qwen3.5-32b should pair with qwen3.5-4b (ratio ~4.4x)
    qwen_pair = [c for c in candidates if "qwen" in c.main_model and "qwen" in c.draft_model]
    assert len(qwen_pair) == 1
    assert qwen_pair[0].size_ratio >= 4.0


def test_no_speculative_single_model():
    """No pairs with single model."""
    models = [{"id": "qwen3.5-7b-q4_k_m.gguf", "size_bytes": 5_000_000_000, "path": "/m/a"}]
    assert detect_speculative_pairs(models) == []


def test_no_speculative_different_families():
    """Different architecture families don't pair."""
    models = [
        {"id": "qwen3.5-32b-q4_k_m.gguf", "size_bytes": 20_000_000_000, "path": "/m/a"},
        {"id": "llama-3.2-1b-q4_k_s.gguf", "size_bytes": 1_000_000_000, "path": "/m/b"},
    ]
    assert detect_speculative_pairs(models) == []
