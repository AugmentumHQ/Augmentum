"""Tests for Sprint A — ResourceSnapshot inventory + disk + active jobs.

Invariants pinned here:

  - Disk probe caches per-dir within ``_DISK_PROBE_TTL_S``; force-refreshes
    only when ``invalidate_disk()`` was called after the cache write.
  - Inventory mtime + action-clock cache: unchanged inputs reuse cached
    entries WITHOUT re-running enumeration. ``invalidate_inventory(m)``
    bumps the clock so the next collect re-enumerates ``m`` regardless of
    mtime.
  - ``inventory_etag`` advances when mtimes change OR a modality clock bumps,
    not otherwise. (UI clients use this as a render trigger.)
  - ``_can_fit`` is conservative — unknown size or unknown VRAM returns
    True; otherwise size_mb < 90% of free_vram_mb.
  - The module-level ``invalidate(app_state, modality)`` helper is a no-op
    when no ledger is wired (degrades gracefully in tests + minimal setups).
  - JobsStore's new ``list_active()`` populates on ``claim_next_pending``,
    drops on ``mark_completed`` / ``mark_failed`` / ``mark_cancelled``.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import aiosqlite
import pytest

from augmentum.resource.ledger import (
    DiskDestination,
    InventoryEntry,
    JobStatus,
    ResourceLedger,
    ResourceSnapshot,
    _can_fit,
    _dir_mtimes,
    _probe_disk_batch,
    invalidate as invalidate_helper,
)


# ── _can_fit ──────────────────────────────────────────────────────


def test_can_fit_unknown_size_returns_true():
    """Unknown size = no opinion. Prefer to surface the model + let
    the user try, rather than grey out something they could load.
    """
    assert _can_fit(0, 8000) is True


def test_can_fit_unknown_vram_returns_true():
    """Same logic the other way."""
    assert _can_fit(7 * 1024 * 1024 * 1024, 0) is True


def test_can_fit_small_model_big_vram():
    """A 4GB model on a 24GB VRAM peer."""
    assert _can_fit(4 * 1024 * 1024 * 1024, 24000) is True


def test_can_fit_big_model_small_vram():
    """A 13GB model on an 8GB VRAM peer."""
    assert _can_fit(13 * 1024 * 1024 * 1024, 8000) is False


def test_can_fit_uses_90_percent_headroom():
    """We leave ~10% headroom for inference workspace, so a model that
    exactly fills VRAM is marked NOT capable.
    """
    # 8 GB model file, 8 GB VRAM free → must fail.
    assert _can_fit(8000 * 1024 * 1024, 8000) is False


# ── invalidate() helper ───────────────────────────────────────────


def test_invalidate_helper_no_ledger_is_noop():
    """Action sites can safely call this without wiring a ledger
    (tests, bare-bones deployments). Just exits.
    """
    state = SimpleNamespace()  # no resource_ledger
    invalidate_helper(state, "llm")  # should not raise


def test_invalidate_helper_bumps_clock():
    """When a ledger is wired the modality clock advances by 1."""
    ledger = ResourceLedger()
    state = SimpleNamespace(resource_ledger=ledger)
    invalidate_helper(state, "llm")
    assert ledger._inventory_clock["llm"] == 1
    invalidate_helper(state, "llm")
    assert ledger._inventory_clock["llm"] == 2


def test_invalidate_helper_optional_disk_flag():
    """Disk default off; opt-in via disk=True for filesystem-mutating
    actions only.
    """
    ledger = ResourceLedger()
    state = SimpleNamespace(resource_ledger=ledger)
    initial_disk_ts = ledger._disk_invalidated_at
    invalidate_helper(state, "tts")  # provider CRUD — no disk impact
    assert ledger._disk_invalidated_at == initial_disk_ts  # unchanged
    invalidate_helper(state, "llm", disk=True)  # download completion
    assert ledger._disk_invalidated_at > initial_disk_ts


# ── Disk probe ────────────────────────────────────────────────────


def test_probe_disk_batch_returns_results_for_each_dir(tmp_path):
    """statfs runs in a worker thread; result tuples carry through."""
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    results = _probe_disk_batch([("llm", str(d1)), ("image", str(d2))])
    assert len(results) == 2
    modalities = {r[0] for r in results}
    assert modalities == {"llm", "image"}
    # Each row: (modality, dir, free_bytes, total_bytes, error="")
    for modality, path, free, total, err in results:
        assert err == ""
        assert free > 0
        assert total > 0


def test_probe_disk_batch_handles_missing_dir():
    """Non-existent dir returns an OSError captured in the error field."""
    results = _probe_disk_batch([("llm", "/path/that/does/not/exist/nope")])
    assert len(results) == 1
    modality, path, free, total, err = results[0]
    assert err != ""
    assert free == 0
    assert total == 0


@pytest.mark.asyncio
async def test_ledger_disk_cache_hits_within_ttl(tmp_path, monkeypatch):
    """Two collects of disk destinations back to back should hit the
    cache on the second call — no second statfs.
    """
    from augmentum.config import settings

    # Point one of the watched dirs at our temp path.
    monkeypatch.setattr(settings, "engine_model_dir", str(tmp_path))
    monkeypatch.setattr(settings, "llamacpp_model_dir", "")
    monkeypatch.setattr(settings, "image_model_dir", "")
    monkeypatch.setattr(settings, "knowledge_packs_dir", "")

    ledger = ResourceLedger()
    probe_calls = {"count": 0}
    orig_probe = _probe_disk_batch

    def counting_probe(targets):
        probe_calls["count"] += 1
        return orig_probe(targets)

    monkeypatch.setattr("augmentum.resource.ledger._probe_disk_batch", counting_probe)

    r1 = await ledger._probe_disk_destinations()
    r2 = await ledger._probe_disk_destinations()
    assert len(r1) == 1
    assert len(r2) == 1
    # First call did the syscall; second hit the cache.
    assert probe_calls["count"] == 1


@pytest.mark.asyncio
async def test_ledger_disk_invalidate_forces_refresh(tmp_path, monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "engine_model_dir", str(tmp_path))
    monkeypatch.setattr(settings, "llamacpp_model_dir", "")
    monkeypatch.setattr(settings, "image_model_dir", "")
    monkeypatch.setattr(settings, "knowledge_packs_dir", "")

    ledger = ResourceLedger()
    probe_calls = {"count": 0}
    orig_probe = _probe_disk_batch

    def counting_probe(targets):
        probe_calls["count"] += 1
        return orig_probe(targets)

    monkeypatch.setattr("augmentum.resource.ledger._probe_disk_batch", counting_probe)

    await ledger._probe_disk_destinations()
    ledger.invalidate_disk()
    await ledger._probe_disk_destinations()
    assert probe_calls["count"] == 2  # cache was bypassed


# ── Inventory mtime cache ─────────────────────────────────────────


def test_dir_mtimes_skips_missing_dirs(tmp_path):
    out = _dir_mtimes([str(tmp_path), "/missing/path/here", ""])
    assert str(tmp_path) in out
    assert "/missing/path/here" not in out
    assert "" not in out


@pytest.mark.asyncio
async def test_inventory_cache_hits_when_mtimes_unchanged(tmp_path, monkeypatch):
    """A fresh enum + a second call to the same modality enumerator
    shouldn't re-glob — return cached entries.
    """
    from augmentum.config import settings
    monkeypatch.setattr(settings, "engine_model_dir", str(tmp_path))
    monkeypatch.setattr(settings, "llamacpp_model_dir", "")

    ledger = ResourceLedger()
    # Provide a stub model manager so list_local_gguf is callable.
    fake_mm = MagicMock()
    fake_mm.list_local_gguf = MagicMock(return_value=[
        {"name": "qwen-7b.gguf", "size": 4_000_000_000, "path": str(tmp_path / "qwen-7b.gguf")},
    ])
    fake_mm.list_all_models = MagicMock()
    async def _list_all():
        return []
    fake_mm.list_all_models = _list_all
    ledger.set_model_manager(fake_mm)

    entries1, _ = await ledger._enum_llm_inventory(loaded_names=set(), gpu_free_mb=24000)
    glob_count_after_first = fake_mm.list_local_gguf.call_count
    entries2, _ = await ledger._enum_llm_inventory(loaded_names=set(), gpu_free_mb=24000)
    glob_count_after_second = fake_mm.list_local_gguf.call_count
    assert len(entries1) == 1
    assert len(entries2) == 1
    # Cache hit — list_local_gguf wasn't called again.
    assert glob_count_after_second == glob_count_after_first


@pytest.mark.asyncio
async def test_inventory_clock_invalidates_cache(tmp_path, monkeypatch):
    """invalidate_inventory('llm') forces re-enumeration even when
    mtimes are unchanged.
    """
    from augmentum.config import settings
    monkeypatch.setattr(settings, "engine_model_dir", str(tmp_path))
    monkeypatch.setattr(settings, "llamacpp_model_dir", "")

    ledger = ResourceLedger()
    fake_mm = MagicMock()
    fake_mm.list_local_gguf = MagicMock(return_value=[])
    async def _list_all():
        return []
    fake_mm.list_all_models = _list_all
    ledger.set_model_manager(fake_mm)

    await ledger._enum_llm_inventory(loaded_names=set(), gpu_free_mb=24000)
    ledger.invalidate_inventory("llm")
    await ledger._enum_llm_inventory(loaded_names=set(), gpu_free_mb=24000)
    assert fake_mm.list_local_gguf.call_count == 2  # cache was bypassed


@pytest.mark.asyncio
async def test_inventory_refreshes_loaded_and_capable_on_cache_hit(tmp_path, monkeypatch):
    """Even when the cache hits, the ``loaded`` + ``capable`` fields must
    reflect the CURRENT loaded model set + GPU free, because those are
    transient.
    """
    from augmentum.config import settings
    monkeypatch.setattr(settings, "engine_model_dir", str(tmp_path))
    monkeypatch.setattr(settings, "llamacpp_model_dir", "")

    ledger = ResourceLedger()
    fake_mm = MagicMock()
    fake_mm.list_local_gguf = MagicMock(return_value=[
        {"name": "qwen-7b.gguf", "size": 4_000_000_000, "path": str(tmp_path / "qwen-7b.gguf")},
    ])
    async def _list_all():
        return []
    fake_mm.list_all_models = _list_all
    ledger.set_model_manager(fake_mm)

    # First collect: not loaded, capable on big GPU.
    entries, _ = await ledger._enum_llm_inventory(loaded_names=set(), gpu_free_mb=24000)
    assert entries[0].loaded is False
    assert entries[0].capable is True

    # Second collect (cache hit), but now the model is loaded + GPU is tiny.
    entries2, _ = await ledger._enum_llm_inventory(
        loaded_names={"qwen-7b.gguf"}, gpu_free_mb=1000,
    )
    assert entries2[0].loaded is True
    assert entries2[0].capable is False  # 4GB > 90% of 1GB


# ── Inventory etag ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inventory_etag_advances_on_clock_bump(tmp_path, monkeypatch):
    """Etag is the UI's "did the inventory change?" hook. Bumping a
    modality clock MUST advance the etag.
    """
    from augmentum.config import settings
    monkeypatch.setattr(settings, "engine_model_dir", str(tmp_path))
    monkeypatch.setattr(settings, "llamacpp_model_dir", "")
    monkeypatch.setattr(settings, "image_model_dir", "")
    monkeypatch.setattr(settings, "knowledge_packs_dir", "")

    ledger = ResourceLedger()
    fake_mm = MagicMock()
    fake_mm.list_local_gguf = MagicMock(return_value=[])
    async def _list_all():
        return []
    fake_mm.list_all_models = _list_all
    ledger.set_model_manager(fake_mm)

    _, etag1 = await ledger._probe_inventory([], 0)
    ledger.invalidate_inventory("llm")
    _, etag2 = await ledger._probe_inventory([], 0)
    assert etag1 != etag2


# ── JobsStore list_active ─────────────────────────────────────────


_JOBS_SCHEMA = """
CREATE TABLE IF NOT EXISTS background_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    job_type TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    priority INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    status TEXT NOT NULL DEFAULT 'pending',
    progress REAL NOT NULL DEFAULT 0.0,
    stage TEXT NOT NULL DEFAULT '',
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    started_at INTEGER,
    completed_at INTEGER,
    error TEXT,
    result TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
"""


@pytest.mark.asyncio
async def test_jobs_store_list_active_initially_empty():
    """No jobs claimed → empty list."""
    from augmentum.state.jobs_store import JobsStore
    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.executescript(_JOBS_SCHEMA)
        await conn.commit()
        store = JobsStore(conn)
        assert store.list_active() == []
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_jobs_store_list_active_populates_on_claim():
    """A claimed job appears in the active list with the right shape."""
    from augmentum.state.jobs_store import JobsStore
    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.executescript(_JOBS_SCHEMA)
        await conn.commit()
        store = JobsStore(conn)
        await store.create(
            user_id="user-1", job_type="gguf_download",
            payload={"name": "qwen-7b"},
        )
        claimed = await store.claim_next_pending()
        assert claimed is not None
        active = store.list_active()
        assert len(active) == 1
        assert active[0]["job_type"] == "gguf_download"
        assert active[0]["user_id"] == "user-1"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_jobs_store_list_active_drops_on_terminal():
    """mark_completed drops the job from the active registry."""
    from augmentum.state.jobs_store import JobsStore
    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.executescript(_JOBS_SCHEMA)
        await conn.commit()
        store = JobsStore(conn)
        await store.create(user_id="user-1", job_type="gguf_download")
        claimed = await store.claim_next_pending()
        assert store.list_active() != []
        await store.mark_completed(claimed["id"], result={"ok": True})
        assert store.list_active() == []
    finally:
        await conn.close()


# ── Resource ledger _probe_active_jobs integration ─────────────────


@pytest.mark.asyncio
async def test_probe_active_jobs_returns_empty_without_jobs_store():
    """Graceful degrade — no store set means active_jobs=[]."""
    ledger = ResourceLedger()
    assert ledger._probe_active_jobs() == []


@pytest.mark.asyncio
async def test_probe_active_jobs_reads_from_jobs_store():
    """When a JobsStore is wired, in-flight jobs surface as JobStatus."""
    from augmentum.state.jobs_store import JobsStore
    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.executescript(_JOBS_SCHEMA)
        await conn.commit()
        store = JobsStore(conn)
        await store.create(
            user_id="user-1", job_type="gguf_download",
            payload={"name": "qwen-7b"},
        )
        await store.claim_next_pending()

        ledger = ResourceLedger()
        ledger.set_jobs_store(store)
        active = ledger._probe_active_jobs()
        assert len(active) == 1
        job = active[0]
        assert isinstance(job, JobStatus)
        assert job.kind == "gguf_download"
        assert job.target_id == "qwen-7b"
    finally:
        await conn.close()


# ── DB write budget (the user-stated invariant) ────────────────────


@pytest.mark.asyncio
async def test_inventory_collect_writes_zero_db_rows_when_idle(tmp_path, monkeypatch):
    """Inventory enumeration on an idle system MUST NOT write to the
    DB. The audio + knowledge enumerators may issue SELECTs but never
    UPDATEs. This is the contract that lets us run the resource
    ledger at heartbeat cadence over fabric without the per-peer DB
    storming.
    """
    from augmentum.config import settings
    monkeypatch.setattr(settings, "engine_model_dir", str(tmp_path))
    monkeypatch.setattr(settings, "llamacpp_model_dir", "")
    monkeypatch.setattr(settings, "image_model_dir", "")
    monkeypatch.setattr(settings, "knowledge_packs_dir", "")

    conn = await aiosqlite.connect(":memory:")
    try:
        # Need the schema both inventory enumerators query.
        await conn.execute(
            "CREATE TABLE audio_providers (id TEXT PRIMARY KEY, kind TEXT, name TEXT, base_url TEXT)"
        )
        await conn.execute(
            "CREATE TABLE knowledge_packs (id TEXT PRIMARY KEY, name TEXT,"
            " pack_format TEXT, install_path TEXT, size_bytes INTEGER)"
        )
        await conn.commit()

        ledger = ResourceLedger(db=conn)
        fake_mm = MagicMock()
        fake_mm.list_local_gguf = MagicMock(return_value=[])
        async def _list_all():
            return []
        fake_mm.list_all_models = _list_all
        ledger.set_model_manager(fake_mm)

        # Track writes by counting commits — sentinel for "wrote SOMETHING".
        commit_count = {"n": 0}
        orig_commit = conn.commit

        async def counting_commit():
            commit_count["n"] += 1
            return await orig_commit()

        conn.commit = counting_commit  # type: ignore[assignment]

        await ledger._probe_inventory([], 0)
        await ledger._probe_inventory([], 0)
        await ledger._probe_inventory([], 0)
        # Zero commits across three inventory collects.
        assert commit_count["n"] == 0
    finally:
        await conn.close()
