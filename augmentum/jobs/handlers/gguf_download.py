"""``gguf_download`` job handler.

Multi-part background download of one *or more* GGUF files from
HuggingFace Hub. Modeled on Ollama's ``server/download.go``: each file's
sparse-allocated destination is split into N parts, downloaded
concurrently with ranged GET requests, with per-part JSON sidecars as the
source of truth for resume. When the payload carries ``files: [...]``,
all listed files download in parallel — matches llama.cpp's
``std::async``-per-shard behavior so a 4-shard model isn't 4× slower than
a single-shard one (the job runner is single-worker, so without
in-handler shard parallelism the shards would queue serially).

Survives client disconnect, server restart, per-part network failures,
and per-part stalls (no bytes for `gguf_download_stall_threshold_s`). On
restart the runner re-queues the job; we glob existing sidecars per file
and resume each incomplete part from its persisted offset+completed.

Idempotent at file granularity: any file already present at the final
destination is skipped; the rest still download.

Layout under ``model_dir/`` while a download is in flight (per file):

    foo.gguf.part           # sparse, pre-allocated to total_size
    foo.gguf.part.0.json    # per-part: {n, offset, size, completed}
    foo.gguf.part.1.json
    ...

On finalize we ``os.replace(.part, foo.gguf)`` and remove all sidecars.

Payload shape (single file, backwards-compatible):

    {
        "repo_id":    "bartowski/Qwen_Qwen3.5-4B-GGUF",
        "filename":   "Qwen3.5-4B-Q4_K_M.gguf",
        "model_dir":  "/models/host",
        "backend":    "llamacpp" | "engine",
        "total_size": 12345678         # optional; refreshed from HEAD
    }

Payload shape (bundle — multi-shard, mmproj sibling, etc.):

    {
        "repo_id":    "bartowski/Foo-GGUF",
        "files": [
            {"filename": "Foo-Q4-00001-of-00004.gguf", "total_size": 12345},
            {"filename": "Foo-Q4-00002-of-00004.gguf", "total_size": 12345},
            ...
            {"filename": "mmproj-F16.gguf", "total_size": 5000},
        ],
        "model_dir":  "/models/host",
        "backend":    "engine",
    }

Result (single file): ``{path, size, backend, parts, filename}``.
Result (bundle):     ``{files: [...], total_size, total_parts, backend}``.
A bundle of length 1 also carries the single-file fields at the top
level so callers that expected the old shape keep working.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import glob
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from augmentum.config import settings
from augmentum.jobs.context import JobCancelled, JobContext, JobRetryable
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# 4 MB read chunks. Per-chunk overhead (aiter_bytes resume, progress_lock,
# threadpool fh.write hop) used to dominate at 256 KB on a 60 MB/s pipe —
# ~2,000 chunks/sec across 8 parts. 4 MB drops that to ~120/sec. Larger is
# fine on HF's CDN and progress UI is rate-limited to 0.5s anyway, so the
# coarser progress granularity is invisible to the user.
_CHUNK_BYTES = 1024 * 1024 * 4
_PROGRESS_EMIT_S = 0.5
_SIDECAR_PERSIST_S = 1.0
_HEARTBEAT_S = 1.0
_MEGABYTE = 1024 * 1024
# Bytes written between POSIX_FADV_DONTNEED hints. Without these the kernel
# keeps every downloaded page in the page cache, eventually tripping the
# global dirty-page limit. Once that hits, every other writer — including
# SQLite's WAL fsync on /data — synchronously blocks until pages flush.
# Symptom seen 2026-05-30: 25s COMMIT + 24s SELECT during a 17 GB download
# even though the DB lives on a different physical disk (separate sd[c-e]).
# 64 MiB per fadvise call: small enough that dirty bytes stay bounded across
# 8 parallel parts; large enough that the syscall overhead is negligible.
_FADVISE_DROP_INTERVAL_BYTES = 64 * _MEGABYTE


def _drop_page_cache(fd: int, offset: int, length: int) -> None:
    """Hint the kernel that we won't reread this range; drop clean pages.

    Linux-only via os.posix_fadvise. Filesystems that don't support the
    advice (9p over Hyper-V, tmpfs, etc.) raise OSError — that's fine,
    we just skip the hint on those mounts. Caller passes (0, 0) for the
    "whole file" form, which is the cheapest call.
    """
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    with contextlib.suppress(OSError):
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)


def _mark_sequential(fd: int) -> None:
    """Hint sequential access so the kernel keeps the readahead window
    small and is more willing to reclaim already-streamed pages."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_SEQUENTIAL"):
        return
    with contextlib.suppress(OSError):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)


def _build_http_client(timeout: httpx.Timeout) -> httpx.AsyncClient:
    """Factory split out so tests can inject a MockTransport-backed client."""
    return httpx.AsyncClient(follow_redirects=True, timeout=timeout)


class _PartStalled(Exception):
    """No bytes received within the stall threshold."""


@dataclass
class _Part:
    n: int
    offset: int
    size: int
    completed: int = 0

    @property
    def done(self) -> bool:
        return self.completed >= self.size

    @property
    def cursor(self) -> int:
        return self.offset + self.completed

    @property
    def end_inclusive(self) -> int:
        return self.offset + self.size - 1


# --------------------------------------------------------------------------- #
# Sidecar / plan helpers (sync; called via ctx.run_in_thread or asyncio.to_thread)
# --------------------------------------------------------------------------- #

def _sidecar_path(part_path: str, n: int) -> str:
    return f"{part_path}.{n}.json"


def _write_sidecar(part_path: str, p: _Part) -> None:
    payload = {"n": p.n, "offset": p.offset, "size": p.size, "completed": p.completed}
    final = _sidecar_path(part_path, p.n)
    tmp = final + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, final)


def _load_sidecars(part_path: str) -> dict[int, _Part]:
    # ``glob.escape`` so filenames containing brackets or other glob metacharacters
    # (occasionally seen in HF GGUF filenames for quant variants) match literally.
    pattern = f"{glob.escape(part_path)}.*.json"
    out: dict[int, _Part] = {}
    for entry in glob.glob(pattern):
        try:
            with open(entry) as fh:
                data = json.load(fh)
            p = _Part(
                n=int(data["n"]),
                offset=int(data["offset"]),
                size=int(data["size"]),
                completed=int(data.get("completed", 0)),
            )
            out[p.n] = p
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            log.warning("gguf_sidecar_unreadable", path=entry, exc_info=True)
            with contextlib.suppress(OSError):
                os.remove(entry)
    return out


def _build_plan(total: int, part_path: str) -> list[_Part]:
    """Resume from on-disk sidecars or build a fresh part plan."""
    existing = _load_sidecars(part_path)
    if existing:
        ordered = sorted(existing.values(), key=lambda p: p.n)
        coverage_ok = (
            len(ordered) > 0
            and ordered[0].offset == 0
            and ordered[-1].offset + ordered[-1].size == total
            and all(
                ordered[i + 1].offset == ordered[i].offset + ordered[i].size
                for i in range(len(ordered) - 1)
            )
        )
        if coverage_ok:
            return ordered
        log.warning(
            "gguf_sidecars_inconsistent_rebuilding",
            part_path=part_path, sidecars=len(ordered), total=total,
        )
        for p in ordered:
            with contextlib.suppress(OSError):
                os.remove(_sidecar_path(part_path, p.n))

    max_parts = max(1, int(settings.gguf_download_max_parts))
    min_size = max(_MEGABYTE, int(settings.gguf_download_min_part_mb) * _MEGABYTE)
    max_size = max(min_size, int(settings.gguf_download_max_part_mb) * _MEGABYTE)
    raw = total // max_parts if max_parts > 0 else total
    part_size = max(min_size, min(max_size, raw or total))
    if part_size <= 0:
        part_size = total or 1

    parts: list[_Part] = []
    n = 0
    offset = 0
    while offset < total:
        size = min(part_size, total - offset)
        parts.append(_Part(n=n, offset=offset, size=size, completed=0))
        offset += size
        n += 1
    return parts


def _preallocate_sparse(path: str, total: int) -> None:
    """Open-or-create the destination at ``total`` bytes (sparse on POSIX)."""
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        os.ftruncate(fd, total)
    finally:
        os.close(fd)


def _cleanup_stale_state(part_path: str) -> None:
    """Three pre-flight states we have to neutralize before building a plan:

    1. Legacy single-stream ``.part`` (no sidecars): bytes were appended from
       offset 0 with no per-part metadata. Can't be integrated into a multi-part
       plan — discard and start fresh.
    2. Orphan sidecars (``.part`` missing but sidecars present): residue from a
       prior successful download whose finalize-rename succeeded but whose
       sidecar cleanup didn't. If we read these we'd treat the download as
       already complete and rename a freshly-allocated zero-filled ``.part``
       over the user's destination. Delete the sidecars.
    3. Both present: a real resume — leave it alone, the planner will validate.
    """
    sidecars = _load_sidecars(part_path)
    part_exists = os.path.exists(part_path)

    if part_exists and not sidecars:
        log.info("gguf_legacy_part_discarded", path=part_path)
        with contextlib.suppress(OSError):
            os.remove(part_path)
        return

    if sidecars and not part_exists:
        log.info(
            "gguf_orphan_sidecars_cleaned",
            path=part_path, count=len(sidecars),
        )
        for n in sidecars:
            with contextlib.suppress(OSError):
                os.remove(_sidecar_path(part_path, n))


def _count_resumed_bytes(part_path: str) -> int:
    """Sum ``completed`` across all per-part sidecars for one file.

    Returns 0 if no sidecars exist (fresh download). Used by the handler
    to seed the overall progress denominator before the first emit, so a
    resumed download doesn't appear to leap from 0 to N-GB in the SSE
    stream's first inter-event delta.
    """
    sidecars = _load_sidecars(part_path)
    return sum(p.completed for p in sidecars.values())


def _format_bytes(n: int) -> str:
    if n <= 0:
        return "0B"
    units = ("B", "KB", "MB", "GB", "TB")
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    if i == 0:
        return f"{int(f)}{units[i]}"
    return f"{f:.1f}{units[i]}"


# --------------------------------------------------------------------------- #
# Handler
# --------------------------------------------------------------------------- #

def make_gguf_download_handler(app):
    """Build the handler bound to ``app.state`` services."""

    async def handler(ctx: JobContext) -> dict[str, Any]:
        manager = getattr(app.state, "model_manager", None)
        if manager is None:
            raise RuntimeError("gguf_download: model_manager not initialized")

        repo_id = str(ctx.payload.get("repo_id") or "").strip()
        model_dir = str(ctx.payload.get("model_dir") or "").strip()
        backend = str(ctx.payload.get("backend") or "llamacpp").strip()

        # Normalize single-file vs bundle payload into one list. Single-file
        # callers still pass ``filename`` (+ optional ``total_size``); bundle
        # callers pass ``files: [{filename, total_size?}, ...]``.
        raw_files = ctx.payload.get("files")
        if raw_files:
            if not isinstance(raw_files, list) or not raw_files:
                raise RuntimeError(
                    f"gguf_download: 'files' must be a non-empty list, got: {raw_files!r}"
                )
            files_payload: list[dict[str, Any]] = []
            for entry in raw_files:
                if not isinstance(entry, dict):
                    raise RuntimeError(
                        f"gguf_download: 'files' entries must be dicts, got: {entry!r}"
                    )
                fn = str(entry.get("filename") or "").strip()
                if not fn:
                    raise RuntimeError(
                        f"gguf_download: 'files' entry missing 'filename': {entry!r}"
                    )
                files_payload.append({
                    "filename": fn,
                    "total_size": int(entry.get("total_size") or 0),
                })
        else:
            single_filename = str(ctx.payload.get("filename") or "").strip()
            if not single_filename:
                raise RuntimeError(
                    f"gguf_download: malformed payload — need 'filename' or 'files' "
                    f"(repo_id={repo_id!r}, model_dir={model_dir!r})"
                )
            files_payload = [{
                "filename": single_filename,
                "total_size": int(ctx.payload.get("total_size") or 0),
            }]

        if not repo_id or not model_dir:
            raise RuntimeError(
                f"gguf_download: malformed payload "
                f"(repo_id={repo_id!r}, model_dir={model_dir!r})"
            )

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        await ctx.update_progress(0.0, stage="resolving")
        await ctx.check_cancel()

        # Build per-file specs. Files already present on disk skip the HEAD
        # probe and the download phase entirely; their result rows still flow
        # back so callers can list everything that's now available.
        file_specs: list[dict[str, Any]] = []
        for entry in files_payload:
            filename = entry["filename"]
            dest_path = os.path.join(model_dir, filename)
            Path(dest_path).parent.mkdir(parents=True, exist_ok=True)

            if os.path.exists(dest_path):
                file_specs.append({
                    "filename": filename,
                    "dest_path": dest_path,
                    "part_path": dest_path + ".part",
                    "total_size": os.path.getsize(dest_path),
                    "skipped": True,
                })
                continue

            total_size = int(entry["total_size"])
            if total_size <= 0:
                try:
                    total_size = await manager.resolve_hf_file_size(repo_id, filename)
                except Exception as exc:
                    raise JobRetryable(
                        f"resolve_hf_file_size failed for {filename!r}: {exc}"
                    ) from exc
            if total_size <= 0:
                raise JobRetryable(
                    f"total_size could not be resolved for {filename!r} "
                    "(HEAD/GET both returned 0)"
                )

            file_specs.append({
                "filename": filename,
                "dest_path": dest_path,
                "part_path": dest_path + ".part",
                "total_size": total_size,
                "skipped": False,
            })

        to_download = [s for s in file_specs if not s["skipped"]]
        overall_total = sum(int(s["total_size"]) for s in to_download)

        # Pre-scan sidecars so the very first progress emit reflects bytes
        # already on disk from a prior interrupted run. Without this, a
        # resume emits {completed: 0} (initial) → {completed: resumed_total}
        # (after the per-file bump) in two close-together SSE events, which
        # the client's speed tracker reads as an N-GB-per-second burst and
        # bakes into the EMA. Reading sidecars is cheap (tiny JSON files),
        # so we eat the IO upfront in exchange for accurate speed/ETA from
        # the very first sample. The per-file _download_one_file no longer
        # bumps progress for resumed bytes since they're already counted.
        resumed_per_file = await asyncio.gather(*(
            asyncio.to_thread(_count_resumed_bytes, s["part_path"])
            for s in to_download
        ))
        overall_completed = sum(resumed_per_file)
        last_emit = 0.0
        progress_lock = asyncio.Lock()

        async def emit_progress(*, force: bool = False) -> None:
            nonlocal last_emit
            now = time.monotonic()
            if not force and (now - last_emit) < _PROGRESS_EMIT_S:
                return
            last_emit = now
            progress = (overall_completed / overall_total) if overall_total > 0 else 1.0
            stage = (
                f"downloading {_format_bytes(overall_completed)} / "
                f"{_format_bytes(overall_total)}"
            )
            if len(to_download) > 1:
                stage += f" ({len(to_download)} files)"
            await ctx.update_progress(progress, stage=stage)

        async def progress_chunk(n_bytes: int) -> None:
            nonlocal overall_completed
            async with progress_lock:
                overall_completed += n_bytes
            await emit_progress()

        # Short-circuit: every requested file is already on disk. Still run
        # the post-finalize hooks so a UI that just kicked off an "install"
        # gets the model map invalidation it expects.
        if not to_download:
            await ctx.update_progress(1.0, stage="exists")
            return await _finalize_results(file_specs, repo_id, backend, ctx, app)

        base_headers = manager._hf_headers()
        timeout = httpx.Timeout(connect=20.0, read=None, write=30.0, pool=20.0)
        max_parts = max(1, int(settings.gguf_download_max_parts))
        max_retries = max(1, int(settings.gguf_download_part_max_retries))
        stall_s = max(5.0, float(settings.gguf_download_stall_threshold_s))

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(_HEARTBEAT_S)
                await ctx.check_cancel()
                await emit_progress()

        async def _download_one_file(
            client: httpx.AsyncClient, spec: dict[str, Any],
        ) -> dict[str, Any]:
            filename = spec["filename"]
            dest_path = spec["dest_path"]
            part_path = spec["part_path"]
            total_size = int(spec["total_size"])

            await ctx.run_in_thread(_cleanup_stale_state, part_path)
            parts = await ctx.run_in_thread(_build_plan, total_size, part_path)
            await ctx.run_in_thread(_preallocate_sparse, part_path, total_size)
            for p in parts:
                await ctx.run_in_thread(_write_sidecar, part_path, p)
            # Resumed bytes are seeded into overall_completed by the
            # handler's pre-scan. Don't bump progress here — that would
            # double-count and re-introduce the SSE spike this fixed.

            url = manager._hf_resolve_url(repo_id, filename)
            sem = asyncio.Semaphore(max_parts)

            async def download_part_once(part: _Part) -> None:
                if part.done:
                    return
                headers = dict(base_headers)
                headers["Range"] = f"bytes={part.cursor}-{part.end_inclusive}"
                sidecar_last = time.monotonic()

                async with client.stream("GET", url, headers=headers) as resp:
                    # Multi-part needs Range. A 200 OK means the server
                    # ignored Range and is sending the whole file — we
                    # can't carve it into our slice.
                    if resp.status_code == 200:
                        raise JobRetryable(
                            "server returned 200 OK to a Range request — "
                            "multi-part download requires HTTP Range support"
                        )
                    resp.raise_for_status()

                    with open(part_path, "r+b") as fh:
                        fh.seek(part.cursor)
                        fd = fh.fileno()
                        _mark_sequential(fd)
                        bytes_since_fadvise = 0
                        chunk_iter = resp.aiter_bytes(_CHUNK_BYTES)
                        while True:
                            try:
                                chunk = await asyncio.wait_for(
                                    chunk_iter.__anext__(), timeout=stall_s,
                                )
                            except StopAsyncIteration:
                                break
                            except TimeoutError as exc:
                                raise _PartStalled(
                                    f"part {part.n} of {filename!r} stalled "
                                    f">{stall_s:.0f}s"
                                ) from exc

                            if not chunk:
                                continue
                            await asyncio.to_thread(fh.write, chunk)
                            part.completed += len(chunk)
                            bytes_since_fadvise += len(chunk)
                            await progress_chunk(len(chunk))

                            if bytes_since_fadvise >= _FADVISE_DROP_INTERVAL_BYTES:
                                await asyncio.to_thread(_drop_page_cache, fd, 0, 0)
                                bytes_since_fadvise = 0

                            now = time.monotonic()
                            if (now - sidecar_last) >= _SIDECAR_PERSIST_S or part.done:
                                sidecar_last = now
                                await asyncio.to_thread(_write_sidecar, part_path, part)
                                # Honor cancellation promptly, mid-stream. This
                                # is the ONLY cancel check that runs while a part
                                # is actively downloading: the per-retry check
                                # (run_part_with_retries) only fires between
                                # attempts, and the heartbeat's JobCancelled is
                                # raised in a detached task that can't cancel
                                # these download tasks. Without this a large part
                                # streams to completion and the user's Cancel
                                # appears to do nothing. Throttled to the sidecar
                                # cadence (~1s/part) so it costs one flag read.
                                await ctx.check_cancel()

                        if bytes_since_fadvise > 0:
                            await asyncio.to_thread(_drop_page_cache, fd, 0, 0)

            async def run_part_with_retries(part: _Part) -> None:
                async with sem:
                    last_exc: Exception | None = None
                    attempt = 0
                    while attempt < max_retries:
                        await ctx.check_cancel()
                        try:
                            await download_part_once(part)
                            await asyncio.to_thread(_write_sidecar, part_path, part)
                            return
                        except (JobCancelled, JobRetryable):
                            raise
                        except OSError as exc:
                            if exc.errno == errno.ENOSPC:
                                raise RuntimeError(
                                    f"out of disk space writing part {part.n} "
                                    f"of {filename!r}: {exc}"
                                ) from exc
                            last_exc = exc
                        except _PartStalled as exc:
                            last_exc = exc
                            # Stalls don't count against the retry budget —
                            # flaky network is the most common cause and the
                            # user shouldn't lose all attempts to it
                            # (mirrors Ollama's `try--` on errPartStalled).
                            attempt -= 1
                        except httpx.HTTPError as exc:
                            last_exc = exc

                        with contextlib.suppress(OSError):
                            await asyncio.to_thread(_write_sidecar, part_path, part)

                        attempt += 1
                        if attempt >= max_retries:
                            break
                        backoff = (2 ** max(attempt - 1, 0)) * (0.5 + random.random())
                        # Per-part retry inside an exponential-backoff loop.
                        # The final give-up raises JobRetryable below, which IS
                        # surfaced at warning level. The retry-in-progress is
                        # informational — info is the right floor.
                        log.info(
                            "gguf_part_retry",
                            job_id=ctx.job_id, file=filename, part=part.n,
                            attempt=attempt, backoff_s=round(backoff, 2),
                            error=str(last_exc),
                        )
                        await asyncio.sleep(backoff)
                    raise JobRetryable(
                        f"part {part.n} of {filename!r} failed after "
                        f"{max_retries} attempts: {last_exc}"
                    ) from last_exc

            part_tasks = [
                asyncio.create_task(run_part_with_retries(p))
                for p in parts if not p.done
            ]
            try:
                await asyncio.gather(*part_tasks)
            except BaseException:
                for t in part_tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*part_tasks, return_exceptions=True)
                raise

            # Finalize per file: clean sidecars BEFORE rename. If rename
            # fails, sidecars are already gone — but the full .part still
            # exists. The next run would discard it as legacy single-stream
            # and re-download. Wasteful but safe. Doing it the other way
            # round risks orphan sidecars silently corrupting a re-pull.
            for p in parts:
                with contextlib.suppress(OSError):
                    os.remove(_sidecar_path(part_path, p.n))
            try:
                await asyncio.to_thread(os.replace, part_path, dest_path)
            except OSError as exc:
                raise RuntimeError(
                    f"finalize move failed for {filename!r}: {exc}"
                ) from exc

            final_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
            return {
                "filename": filename,
                "path": dest_path,
                "size": final_size,
                "parts": len(parts),
            }

        await emit_progress(force=True)

        # Per-file results in input order. ``existing`` is already-on-disk
        # files; ``downloaded`` is the parallel-download result.
        existing = [
            {
                "filename": s["filename"], "path": s["dest_path"],
                "size": int(s["total_size"]), "parts": 0, "skipped": "exists",
            }
            for s in file_specs if s["skipped"]
        ]

        try:
            async with _build_http_client(timeout) as client:
                file_tasks = [
                    asyncio.create_task(_download_one_file(client, s))
                    for s in to_download
                ]
                hb = asyncio.create_task(heartbeat())
                try:
                    downloaded = await asyncio.gather(*file_tasks)
                except BaseException:
                    # First failure cancels the rest so we don't burn
                    # bandwidth on shards we'll throw away.
                    for t in file_tasks:
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(*file_tasks, return_exceptions=True)
                    raise
                finally:
                    hb.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await hb
        except JobCancelled:
            log.info(
                "gguf_download_cancelled_keeping_state",
                job_id=ctx.job_id,
                files=[s["filename"] for s in to_download],
                bytes_downloaded=overall_completed,
            )
            raise

        # Stitch results back in input order: walk file_specs and pick from
        # ``existing`` or ``downloaded`` to match.
        downloaded_by_name = {r["filename"]: r for r in downloaded}
        existing_by_name = {r["filename"]: r for r in existing}
        results = [
            downloaded_by_name.get(s["filename"]) or existing_by_name[s["filename"]]
            for s in file_specs
        ]

        await emit_progress(force=True)
        await ctx.update_progress(1.0, stage="finalizing")
        return await _finalize_results(file_specs, repo_id, backend, ctx, app, results=results)

    return handler


async def _finalize_results(
    file_specs: list[dict[str, Any]],
    repo_id: str,
    backend: str,
    ctx: JobContext,
    app,
    *,
    results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the post-download hooks (model-map invalidation, engine rescan)
    and shape the result dict. Shared between the all-skipped fast path and
    the normal multi-file finalize so neither forgets to invalidate caches.
    """
    if results is None:
        results = [
            {
                "filename": s["filename"], "path": s["dest_path"],
                "size": int(s["total_size"]), "parts": 0, "skipped": "exists",
            }
            for s in file_specs
        ]

    try:
        registry = getattr(app.state, "provider_registry", None)
        if registry is not None:
            registry.invalidate_model_map()
    except Exception:
        log.warning("gguf_download_invalidate_failed", exc_info=True)

    try:
        from augmentum.proxy import system_events
        system_events.publish("models.installed", {
            "backend": backend,
            "repo_id": repo_id,
            "filenames": [r.get("filename", "") for r in results],
        })
    except Exception:
        log.debug("gguf_download_publish_failed", exc_info=True)

    # Inventory + disk caches in the resource ledger — a fresh GGUF
    # just landed on disk; the panel should show it on the next collect
    # without waiting for the 15s disk TTL or the dir mtime tick.
    try:
        from augmentum.resource.ledger import invalidate as _invalidate_resource
        _invalidate_resource(app.state, "llm", disk=True)
    except Exception:
        log.debug("gguf_download_inventory_invalidate_failed", exc_info=True)

    if backend == "engine":
        try:
            llama_mgr = getattr(app.state, "llama_manager", None)
            if llama_mgr is not None:
                await llama_mgr.scan_and_cache_profiles()
        except Exception:
            log.warning("gguf_download_engine_scan_failed", exc_info=True)

    # Auto-import known-good sampling for each freshly-downloaded model so a
    # new model lands with sane, family-aware defaults the user can see and
    # edit in the model library (Qwen3 → 0.6/0.95/20, Gemma-4 → 1.0/0.95/64,
    # …). Seeds the install-wide (global) layer; per-user library edits
    # override it later. Best-effort: the family fallback in resolve_sampling
    # already applies these at chat time even if this seed's key doesn't match
    # the runtime model id, so a miss here is harmless. Never seed over an
    # existing global profile (don't stomp a prior manual install default).
    try:
        store = getattr(app.state, "settings_store", None)
        if store is not None:
            from augmentum.models.sampling_profiles import (
                load_overrides,
                recommended_for,
                save_overrides,
            )
            for r in results:
                fn = r.get("filename", "") or ""
                if not fn.lower().endswith(".gguf"):
                    continue  # skip mmproj/sidecars that aren't a chat model
                # The chat path keys sampling off the model id, usually the
                # filename without its .gguf suffix — seed under that form.
                model_id = fn[:-5] if fn.lower().endswith(".gguf") else fn
                existing = await load_overrides(model_id, store)  # global layer
                if existing.to_request_kwargs():
                    continue  # already seeded / manually set — leave it
                await save_overrides(model_id, recommended_for(model_id), store)
    except Exception:
        log.warning("gguf_download_sampling_seed_failed", exc_info=True)

    total_size = sum(int(r.get("size", 0)) for r in results)
    total_parts = sum(int(r.get("parts", 0)) for r in results)

    log.info(
        "gguf_download_complete",
        job_id=ctx.job_id, source=repo_id,
        files=[r["filename"] for r in results],
        total_size=total_size, total_parts=total_parts,
    )

    bundle_result: dict[str, Any] = {
        "files": results,
        "total_size": total_size,
        "total_parts": total_parts,
        "backend": backend,
    }
    if len(results) == 1:
        # Single-file callers (chip downloads, individual quant rows)
        # expect path/size/parts at the top level — keep that contract.
        first = results[0]
        bundle_result.update({
            "path": first.get("path", ""),
            "size": int(first.get("size", 0)),
            "parts": int(first.get("parts", 0)),
            "filename": first.get("filename", ""),
        })
        if first.get("skipped"):
            bundle_result["skipped"] = first["skipped"]
    return bundle_result
