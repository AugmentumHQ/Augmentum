"""``gguf_download`` job handler — multi-part, sparse-allocated, sidecar resume.

Three layers of coverage:

  * Plan/sidecar unit tests (no HTTP): verify part-size clamping, sidecar
    round-trip, and legacy ``.part`` discard.
  * End-to-end with ``httpx.MockTransport``: verify the whole multi-part
    pipeline assembles bytes correctly, honors Range, cleans sidecars,
    and renames atomically.
  * Failure-mode tests: server returning 200 OK to a Range request, and
    resume from partial sidecars + ``.part``.

Tests stub the JobContext store directly rather than spinning up a
JobsStore + JobRunner; the handler's interaction with the store is just
``update_progress`` / ``is_cancel_requested``, so a small fake covers it.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from augmentum.jobs.context import JobCancelled, JobContext, JobRetryable
from augmentum.jobs.handlers import gguf_download as gd

# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #

class _FakeStore:
    """Minimal stand-in for JobsStore: records progress, optional cancel flag."""

    def __init__(self) -> None:
        self.progress_log: list[tuple[float, str]] = []
        self.cancelled = False

    async def update_progress(self, job_id: str, *, progress: float, stage: str) -> None:
        self.progress_log.append((progress, stage))

    async def is_cancel_requested(self, job_id: str) -> bool:
        return self.cancelled


def _ctx(payload: dict) -> tuple[JobContext, _FakeStore]:
    store = _FakeStore()
    return JobContext(
        job_id="job-test",
        user_id="u1",
        job_type="gguf_download",
        payload=payload,
        store=store,
    ), store


def _app_with_manager() -> SimpleNamespace:
    """Build an app double whose model_manager exposes only what the handler needs."""
    manager = MagicMock()
    manager._hf_resolve_url = lambda repo, fn: f"https://hf.test/{repo}/resolve/main/{fn}"
    manager._hf_headers = lambda: {"User-Agent": "test"}

    async def resolve_size(repo, fn):
        return manager._resolved_size

    manager.resolve_hf_file_size = resolve_size
    manager._resolved_size = 0
    return SimpleNamespace(state=SimpleNamespace(model_manager=manager))


def _ranged_transport(file_bytes: bytes, *, support_range: bool = True) -> httpx.MockTransport:
    """Mock HF/CDN: respond to Range requests with 206 partial content."""
    def handler(request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("range", "")
        m = re.match(r"bytes=(\d+)-(\d*)", rng) if rng else None
        if m and support_range:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else len(file_bytes) - 1
            chunk = file_bytes[start:end + 1]
            return httpx.Response(
                206, content=chunk,
                headers={
                    "content-length": str(len(chunk)),
                    "content-range": f"bytes {start}-{end}/{len(file_bytes)}",
                },
            )
        return httpx.Response(
            200, content=file_bytes,
            headers={"content-length": str(len(file_bytes))},
        )
    return httpx.MockTransport(handler)


def _patch_client(monkeypatch, transport: httpx.MockTransport) -> None:
    def factory(timeout):
        return httpx.AsyncClient(transport=transport, follow_redirects=True, timeout=timeout)
    monkeypatch.setattr(gd, "_build_http_client", factory)


def _patch_settings(monkeypatch, **overrides) -> None:
    """Override gguf_download settings without touching the global config."""
    for name, value in overrides.items():
        monkeypatch.setattr(gd.settings, name, value)


# --------------------------------------------------------------------------- #
# Plan / sidecar units
# --------------------------------------------------------------------------- #

def test_build_plan_clamps_to_min_part_for_small_files(tmp_path, monkeypatch):
    _patch_settings(
        monkeypatch,
        gguf_download_max_parts=8,
        gguf_download_min_part_mb=100,
        gguf_download_max_part_mb=1000,
    )
    parts = gd._build_plan(50 * 1024 * 1024, str(tmp_path / "x.part"))
    # 50 MB total < 100 MB min → single part of size = total.
    assert len(parts) == 1
    assert parts[0].offset == 0
    assert parts[0].size == 50 * 1024 * 1024


def test_build_plan_splits_large_files(tmp_path, monkeypatch):
    _patch_settings(
        monkeypatch,
        gguf_download_max_parts=8,
        gguf_download_min_part_mb=100,
        gguf_download_max_part_mb=1024,
    )
    total = 8 * 1024 * 1024 * 1024  # 8 GiB
    parts = gd._build_plan(total, str(tmp_path / "big.part"))
    # 8 GiB / 8 = 1 GiB per part, within [100 MiB, 1024 MiB] → 8 equal parts.
    assert len(parts) == 8
    assert all(p.size == 1024 * 1024 * 1024 for p in parts)
    # Coverage: parts tile [0, total) exactly.
    assert parts[0].offset == 0
    assert parts[-1].offset + parts[-1].size == total
    # Offsets are contiguous.
    for prev, curr in zip(parts, parts[1:], strict=False):
        assert curr.offset == prev.offset + prev.size


def test_sidecar_roundtrip(tmp_path):
    part_path = str(tmp_path / "model.gguf.part")
    p = gd._Part(n=3, offset=300, size=100, completed=42)
    gd._write_sidecar(part_path, p)
    loaded = gd._load_sidecars(part_path)
    assert 3 in loaded
    assert loaded[3].offset == 300
    assert loaded[3].size == 100
    assert loaded[3].completed == 42


def test_legacy_part_without_sidecars_is_discarded(tmp_path):
    """Old single-stream .part with no sidecars must be cleared so the multi-part
    plan starts fresh."""
    part_path = str(tmp_path / "legacy.gguf.part")
    Path(part_path).write_bytes(b"old single-stream bytes")
    gd._cleanup_stale_state(part_path)
    assert not os.path.exists(part_path)


def test_part_with_sidecars_is_preserved(tmp_path):
    """When .part + sidecars are both present we're resuming — leave them."""
    part_path = str(tmp_path / "resuming.gguf.part")
    Path(part_path).write_bytes(b"\x00" * 100)
    gd._write_sidecar(part_path, gd._Part(n=0, offset=0, size=100, completed=50))
    gd._cleanup_stale_state(part_path)
    assert os.path.exists(part_path), "should preserve .part when sidecars exist"


def test_orphan_sidecars_without_part_are_cleaned(tmp_path):
    """If a previous run renamed .part → final but failed to delete sidecars,
    those orphans must be removed before a new download — otherwise the
    handler thinks the file is already complete and renames a zero-filled
    fresh .part over the user's destination."""
    part_path = str(tmp_path / "orphan.gguf.part")
    # Sidecars saying the download is complete, but no .part exists.
    gd._write_sidecar(part_path, gd._Part(n=0, offset=0, size=100, completed=100))
    gd._write_sidecar(part_path, gd._Part(n=1, offset=100, size=100, completed=100))
    assert not os.path.exists(part_path)

    gd._cleanup_stale_state(part_path)

    assert gd._load_sidecars(part_path) == {}, "orphan sidecars must be cleared"


def test_sidecar_glob_handles_brackets_in_filename(tmp_path):
    """HF GGUF filenames occasionally contain `[brackets]`. Without
    glob.escape, glob would interpret them as a character class and miss
    real sidecars."""
    part_path = str(tmp_path / "model[Q4].gguf.part")
    gd._write_sidecar(part_path, gd._Part(n=0, offset=0, size=10, completed=5))
    loaded = gd._load_sidecars(part_path)
    assert 0 in loaded
    assert loaded[0].completed == 5


# --------------------------------------------------------------------------- #
# End-to-end: handler runs, file assembles correctly
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_handler_assembles_multipart_file(tmp_path, monkeypatch):
    """8 MB file, 1 MB parts → 8 ranged GETs assemble into a final file
    matching the source bytes exactly. Sidecars cleaned up on success."""
    _patch_settings(
        monkeypatch,
        gguf_download_max_parts=8,
        gguf_download_min_part_mb=1,        # tiny for the test
        gguf_download_max_part_mb=2,
        gguf_download_part_max_retries=2,
        gguf_download_stall_threshold_s=5.0,
    )
    file_bytes = bytes((i & 0xFF) for i in range(8 * 1024 * 1024))
    _patch_client(monkeypatch, _ranged_transport(file_bytes))

    app = _app_with_manager()
    app.state.model_manager._resolved_size = len(file_bytes)
    handler = gd.make_gguf_download_handler(app)

    model_dir = tmp_path / "models"
    ctx, store = _ctx({
        "repo_id": "test/repo", "filename": "model.gguf",
        "model_dir": str(model_dir), "backend": "llamacpp",
        "total_size": len(file_bytes),
    })
    result = await handler(ctx)

    final = model_dir / "model.gguf"
    assert final.read_bytes() == file_bytes
    assert result["size"] == len(file_bytes)
    assert result["parts"] >= 4, "expected at least 4 parts at 1MB part size"

    # Cleanup: no .part, no sidecars left behind.
    assert not (model_dir / "model.gguf.part").exists()
    leftover = list(model_dir.glob("model.gguf.part.*"))
    assert leftover == [], f"sidecars not cleaned: {leftover}"

    # Final progress entry should be 1.0.
    assert store.progress_log[-1][0] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_handler_short_circuits_when_dest_exists(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "already.gguf").write_bytes(b"existing")

    # No HTTP allowed — if the handler tries to fetch we'd get an error.
    transport = httpx.MockTransport(
        lambda r: pytest.fail("no HTTP expected when dest exists"),
    )
    _patch_client(monkeypatch, transport)

    app = _app_with_manager()
    handler = gd.make_gguf_download_handler(app)
    ctx, _ = _ctx({
        "repo_id": "x/y", "filename": "already.gguf",
        "model_dir": str(model_dir), "backend": "llamacpp",
    })
    result = await handler(ctx)
    assert result["skipped"] == "exists"
    assert result["size"] == len(b"existing")


@pytest.mark.asyncio
async def test_handler_aborts_on_no_range_support(tmp_path, monkeypatch):
    """If the server returns 200 OK to a Range request, multi-part can't
    proceed safely — the handler must raise JobRetryable, not silently
    restart from byte 0 (the legacy behavior)."""
    _patch_settings(
        monkeypatch,
        gguf_download_max_parts=2,
        gguf_download_min_part_mb=1,
        gguf_download_max_part_mb=2,
        gguf_download_part_max_retries=1,
        gguf_download_stall_threshold_s=5.0,
    )
    file_bytes = b"x" * (4 * 1024 * 1024)
    _patch_client(
        monkeypatch,
        _ranged_transport(file_bytes, support_range=False),
    )

    app = _app_with_manager()
    app.state.model_manager._resolved_size = len(file_bytes)
    handler = gd.make_gguf_download_handler(app)
    ctx, _ = _ctx({
        "repo_id": "test/repo", "filename": "noerange.gguf",
        "model_dir": str(tmp_path / "out"), "backend": "llamacpp",
        "total_size": len(file_bytes),
    })

    with pytest.raises(JobRetryable, match="Range"):
        await handler(ctx)


@pytest.mark.asyncio
async def test_handler_resumes_from_existing_sidecars(tmp_path, monkeypatch):
    """Pre-seed a half-done .part + sidecars; the handler should request
    only the missing ranges and produce the full file."""
    _patch_settings(
        monkeypatch,
        gguf_download_max_parts=2,
        gguf_download_min_part_mb=1,
        gguf_download_max_part_mb=2,
        gguf_download_part_max_retries=2,
        gguf_download_stall_threshold_s=5.0,
    )
    # 4 MB file, 2 MB parts → exactly 2 parts.
    file_bytes = bytes((i & 0xFF) for i in range(4 * 1024 * 1024))
    part_size = 2 * 1024 * 1024

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    part_path = model_dir / "resume.gguf.part"

    # Pre-allocate the .part file at full size, copy ONLY part 0's bytes in.
    part_path.write_bytes(b"\x00" * len(file_bytes))
    with open(part_path, "r+b") as fh:
        fh.seek(0)
        fh.write(file_bytes[:part_size])

    # Sidecars: part 0 done, part 1 not started.
    gd._write_sidecar(
        str(part_path), gd._Part(n=0, offset=0, size=part_size, completed=part_size),
    )
    gd._write_sidecar(
        str(part_path), gd._Part(n=1, offset=part_size, size=part_size, completed=0),
    )

    # Track which Range requests the server saw — part 0 should NOT be re-fetched.
    seen_ranges: list[str] = []

    def tracking_handler(request: httpx.Request) -> httpx.Response:
        seen_ranges.append(request.headers.get("range", ""))
        rng = request.headers.get("range", "")
        m = re.match(r"bytes=(\d+)-(\d*)", rng)
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else len(file_bytes) - 1
        return httpx.Response(
            206, content=file_bytes[start:end + 1],
            headers={
                "content-length": str(end - start + 1),
                "content-range": f"bytes {start}-{end}/{len(file_bytes)}",
            },
        )

    _patch_client(monkeypatch, httpx.MockTransport(tracking_handler))

    app = _app_with_manager()
    app.state.model_manager._resolved_size = len(file_bytes)
    handler = gd.make_gguf_download_handler(app)
    ctx, _ = _ctx({
        "repo_id": "test/repo", "filename": "resume.gguf",
        "model_dir": str(model_dir), "backend": "llamacpp",
        "total_size": len(file_bytes),
    })
    await handler(ctx)

    assert (model_dir / "resume.gguf").read_bytes() == file_bytes
    # Only part 1 should have been requested. Part 0's range starts at 0.
    assert all(not r.startswith("bytes=0-") for r in seen_ranges), (
        f"part 0 was re-fetched on resume: {seen_ranges}"
    )


@pytest.mark.asyncio
async def test_handler_writes_sidecars_during_download(tmp_path, monkeypatch):
    """Sidecars must reflect committed progress so a crash mid-flight
    can resume from the right offset."""
    _patch_settings(
        monkeypatch,
        gguf_download_max_parts=1,
        gguf_download_min_part_mb=1,
        gguf_download_max_part_mb=2,
        gguf_download_part_max_retries=1,
        gguf_download_stall_threshold_s=5.0,
    )
    file_bytes = b"a" * (1024 * 1024)
    _patch_client(monkeypatch, _ranged_transport(file_bytes))

    model_dir = tmp_path / "models"
    app = _app_with_manager()
    app.state.model_manager._resolved_size = len(file_bytes)
    handler = gd.make_gguf_download_handler(app)
    ctx, _ = _ctx({
        "repo_id": "test/repo", "filename": "ok.gguf",
        "model_dir": str(model_dir), "backend": "llamacpp",
        "total_size": len(file_bytes),
    })
    await handler(ctx)

    # Final state: dest exists, sidecars cleaned. We can't peek at sidecars
    # mid-flight without yielding the loop, but verify cleanup happened.
    assert (model_dir / "ok.gguf").exists()
    assert list(model_dir.glob("*.part*")) == []


@pytest.mark.asyncio
async def test_orphan_sidecars_dont_short_circuit_a_fresh_download(tmp_path, monkeypatch):
    """Repro for the orphan-sidecar bug: if a previous run left sidecars
    saying the download was complete but no .part file, a new download
    must NOT short-circuit and rename a zero-filled fresh .part. The
    handler must download the file from scratch."""
    _patch_settings(
        monkeypatch,
        gguf_download_max_parts=2,
        gguf_download_min_part_mb=1,
        gguf_download_max_part_mb=2,
        gguf_download_part_max_retries=2,
        gguf_download_stall_threshold_s=5.0,
    )
    file_bytes = bytes((i & 0xFF) for i in range(2 * 1024 * 1024))
    _patch_client(monkeypatch, _ranged_transport(file_bytes))

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    part_path = model_dir / "fresh.gguf.part"
    # Seed orphan sidecars marked complete; .part deliberately absent.
    gd._write_sidecar(
        str(part_path), gd._Part(n=0, offset=0, size=len(file_bytes), completed=len(file_bytes)),
    )
    assert not part_path.exists()

    app = _app_with_manager()
    app.state.model_manager._resolved_size = len(file_bytes)
    handler = gd.make_gguf_download_handler(app)
    ctx, _ = _ctx({
        "repo_id": "test/repo", "filename": "fresh.gguf",
        "model_dir": str(model_dir), "backend": "llamacpp",
        "total_size": len(file_bytes),
    })
    await handler(ctx)

    final = model_dir / "fresh.gguf"
    assert final.read_bytes() == file_bytes, (
        "orphan sidecars caused a zero-byte rename — the bug we just fixed"
    )


@pytest.mark.asyncio
async def test_disk_full_fails_immediately_no_retries(tmp_path, monkeypatch):
    """ENOSPC is non-transient — retrying just burns bandwidth on bytes
    that won't fit. The handler must raise immediately."""
    _patch_settings(
        monkeypatch,
        gguf_download_max_parts=1,
        gguf_download_min_part_mb=1,
        gguf_download_max_part_mb=2,
        gguf_download_part_max_retries=6,   # plenty if we DID retry
        gguf_download_stall_threshold_s=5.0,
    )
    file_bytes = b"x" * (1024 * 1024)
    _patch_client(monkeypatch, _ranged_transport(file_bytes))

    # Force every fh.write to raise ENOSPC.
    real_to_thread = asyncio.to_thread
    write_calls = 0

    async def fake_to_thread(fn, *args, **kwargs):
        nonlocal write_calls
        # Identify file-write calls (they pass bytes as the first arg).
        if args and isinstance(args[0], bytes | bytearray | memoryview):
            write_calls += 1
            raise OSError(28, "No space left on device")  # 28 = errno.ENOSPC
        return await real_to_thread(fn, *args, **kwargs)

    # Patch on the gguf_download module so its `asyncio.to_thread` is intercepted.
    monkeypatch.setattr(gd.asyncio, "to_thread", fake_to_thread)

    app = _app_with_manager()
    app.state.model_manager._resolved_size = len(file_bytes)
    handler = gd.make_gguf_download_handler(app)
    ctx, _ = _ctx({
        "repo_id": "test/repo", "filename": "full.gguf",
        "model_dir": str(tmp_path / "models"), "backend": "llamacpp",
        "total_size": len(file_bytes),
    })

    with pytest.raises(RuntimeError, match="disk space"):
        await handler(ctx)

    # Critical assertion: we failed on the FIRST write attempt, not after
    # 6 retries downloading the same bytes 6 times.
    assert write_calls == 1, f"expected immediate fail, but saw {write_calls} write attempts"


@pytest.mark.asyncio
async def test_cancel_interrupts_active_stream(tmp_path, monkeypatch):
    """Regression: pressing Cancel must stop an IN-FLIGHT download.

    The only cancel checks used to be (a) before the download starts,
    (b) between part *retries*, and (c) a detached heartbeat whose
    JobCancelled couldn't cancel the download tasks — so a part streamed
    to completion regardless of Cancel ("I press cancel but it keeps
    going"). The active chunk loop now checks the flag on the sidecar
    cadence. Here we drive that cadence to every chunk and have the mock
    server flip the cancel flag the moment it starts serving bytes, so
    the abort happens genuinely mid-stream. We assert the handler raises
    JobCancelled before finalizing AND preserves the .part + sidecars so
    the download can be resumed/retried.
    """
    _patch_settings(
        monkeypatch,
        gguf_download_max_parts=1,
        gguf_download_min_part_mb=1,
        gguf_download_max_part_mb=100,   # one part spanning the whole file
        gguf_download_part_max_retries=2,
        gguf_download_stall_threshold_s=5.0,
    )
    # Check cancel on every committed chunk, and keep chunks small so the
    # stream spans several of them (bytes genuinely remain at abort time).
    monkeypatch.setattr(gd, "_SIDECAR_PERSIST_S", 0.0)
    monkeypatch.setattr(gd, "_CHUNK_BYTES", 64 * 1024)

    file_bytes = b"z" * (256 * 1024)   # 4 chunks at 64 KiB

    model_dir = tmp_path / "models"
    app = _app_with_manager()
    app.state.model_manager._resolved_size = len(file_bytes)
    handler = gd.make_gguf_download_handler(app)
    ctx, store = _ctx({
        "repo_id": "test/repo", "filename": "cancelled.gguf",
        "model_dir": str(model_dir), "backend": "engine",
        "total_size": len(file_bytes),
    })

    # The transport runs INSIDE download_part_once (after the pre-stream
    # cancel checks), so flipping the flag here means the FIRST in-loop
    # check — after the first chunk is written — is the one that trips.
    def cancel_on_serve(request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("range", "")
        m = re.match(r"bytes=(\d+)-(\d*)", rng)
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else len(file_bytes) - 1
        chunk = file_bytes[start:end + 1]
        store.cancelled = True
        return httpx.Response(
            206, content=chunk,
            headers={
                "content-length": str(len(chunk)),
                "content-range": f"bytes {start}-{end}/{len(file_bytes)}",
            },
        )

    _patch_client(monkeypatch, httpx.MockTransport(cancel_on_serve))

    with pytest.raises(JobCancelled):
        await handler(ctx)

    # Never finalized...
    assert not (model_dir / "cancelled.gguf").exists()
    # ...but the partial + sidecars survive so Retry can resume.
    assert (model_dir / "cancelled.gguf.part").exists()
    assert list(model_dir.glob("cancelled.gguf.part.*.json")), (
        "sidecars should be kept for resume after cancel"
    )


@pytest.mark.asyncio
async def test_stalls_dont_count_against_retry_budget(tmp_path, monkeypatch):
    """Mirror Ollama: stalls (no bytes for stall_threshold seconds) get free
    retries so a flaky connection doesn't burn the whole budget. We intercept
    ``asyncio.wait_for`` to simulate four consecutive stalls, then let the
    fifth chunk-read succeed. With max_retries=2, the part can only complete
    if stall-retries are free."""
    _patch_settings(
        monkeypatch,
        gguf_download_max_parts=1,
        gguf_download_min_part_mb=1,
        gguf_download_max_part_mb=2,
        gguf_download_part_max_retries=2,   # would fail-out fast if stalls counted
        gguf_download_stall_threshold_s=5.0,
    )
    file_bytes = b"y" * (256 * 1024)
    _patch_client(monkeypatch, _ranged_transport(file_bytes))

    real_wait_for = asyncio.wait_for
    timeouts_raised = 0

    async def fake_wait_for(aw, timeout):
        nonlocal timeouts_raised
        if timeouts_raised < 4:
            timeouts_raised += 1
            # Close the coroutine so we don't leak a "coroutine was never awaited" warning.
            if asyncio.iscoroutine(aw):
                aw.close()
            raise TimeoutError("simulated stall")
        return await real_wait_for(aw, timeout)

    monkeypatch.setattr(gd.asyncio, "wait_for", fake_wait_for)

    app = _app_with_manager()
    app.state.model_manager._resolved_size = len(file_bytes)
    handler = gd.make_gguf_download_handler(app)
    ctx, _ = _ctx({
        "repo_id": "test/repo", "filename": "stalled.gguf",
        "model_dir": str(tmp_path / "models"), "backend": "llamacpp",
        "total_size": len(file_bytes),
    })
    await handler(ctx)

    assert timeouts_raised == 4, "wait_for stub didn't fire as expected"
    assert (tmp_path / "models" / "stalled.gguf").read_bytes() == file_bytes
