"""Sprint 3 async-discipline fixes — regression pins (audit 2026-06-17).

Covers the new primitives:
  * bg_tasks.track now reaps + logs (GC-safe fire-and-forget)
  * EmbeddingService.aembed_one offloads to a worker thread
  * state._age_seconds bridges wall-clock → monotonic for cooldown restore

The async-conversions (identity drift, consolidation drift) are exercised
for regressions by test_companion_identity_api / test_synapse_consolidation;
the relationship json_patch merge by test_companion_identity_api.
"""
from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta

import pytest

# ── bg_tasks.track: GC-safe + reaps ───────────────────────────────────

@pytest.mark.asyncio
async def test_track_removes_ref_after_success():
    from augmentum.utils import bg_tasks

    async def ok():
        return 1

    t = bg_tasks.track(ok(), name="ok")
    await t
    await asyncio.sleep(0)  # let the done-callback (_reap) run
    assert t.done()
    assert bg_tasks.in_flight_count() == 0


@pytest.mark.asyncio
async def test_track_handles_exception_without_crashing_caller():
    """A raising tracked task must not propagate to the spawner, and the
    reaper must retrieve the exception + drop the ref."""
    from augmentum.utils import bg_tasks

    async def boom():
        raise ValueError("kaboom")

    t = bg_tasks.track(boom(), name="boom")
    await asyncio.sleep(0)  # run the task
    await asyncio.sleep(0)  # run the done-callback
    assert t.done()
    assert bg_tasks.in_flight_count() == 0


# ── aembed_one offloads off the event loop ────────────────────────────

@pytest.mark.asyncio
async def test_aembed_one_runs_on_worker_thread(monkeypatch):
    from augmentum.memory.embeddings import EmbeddingService

    main_thread = threading.current_thread().name
    seen: dict[str, str] = {}

    def fake_embed_one(text: str):
        seen["thread"] = threading.current_thread().name
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(EmbeddingService, "embed_one", staticmethod(fake_embed_one))
    result = await EmbeddingService.aembed_one("hello")
    assert result == [0.1, 0.2, 0.3]
    # to_thread → ran on a worker thread, not the event-loop thread.
    assert seen["thread"] != main_thread


# ── state cooldown restore: wall-clock → monotonic bridge ─────────────

def test_age_seconds_parses_past_timestamp():
    from augmentum.companion_runtime.state import _age_seconds

    past = (datetime.now(UTC) - timedelta(hours=1)).strftime(
        "%Y-%m-%d %H:%M:%S",
    )
    age = _age_seconds(past)
    assert 3500 < age < 3700  # ~1 hour


def test_age_seconds_safe_defaults():
    from augmentum.companion_runtime.state import _age_seconds

    assert _age_seconds("") == 0.0
    assert _age_seconds(None) == 0.0
    assert _age_seconds("not-a-timestamp") == 0.0
    # Future timestamp (clock skew) clamps to 0, never negative.
    future = (datetime.now(UTC) + timedelta(hours=1)).strftime(
        "%Y-%m-%d %H:%M:%S",
    )
    assert _age_seconds(future) == 0.0
