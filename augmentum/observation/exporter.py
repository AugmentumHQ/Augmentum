"""Per-model lookup-cache exporter.

The L0 store is tokenizer-agnostic (text in / text out). The actual
binary cache that ``llama-server --lookup-cache-static`` consumes is
tokenizer-specific — it stores token IDs against the model's vocab.
This module bridges the two by:

  1. Pulling the top-K observations for a user.
  2. Writing them as a plain-text corpus (one "prefix continuation"
     line per observation, weighted by repetition).
  3. Shelling out to the bundled ``llama-lookup-create`` binary to
     produce the binary cache against the currently-loaded model.
  4. Atomically renaming the result into place.

Atomic rename matters: ``llama-server --lookup-cache-static`` reads
the file at startup. If we write the cache in-place and llama-server
swaps in mid-write, it gets a truncated file and either crashes or
silently disables drafting. The temp-then-rename pattern means the
file llama-server sees is always complete.

Per-model regeneration is lazy: the cache file path is keyed by
``(user_id, model_stem, llama_server_version)`` so a model swap OR
a llama-server upgrade both invalidate cleanly. The orchestrator
(``observation_routes`` admin endpoint, future LlamaServerManager
hook) decides when to call this.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from augmentum.observation.store import ObservationStore
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Default path layout — overridable per-call so tests don't pollute /data.
# /data is the conventional augmentum data dir inside the container per
# Dockerfile.gpu's mkdir -p /data/...
_DEFAULT_CACHE_ROOT = Path("/data/lookup_cache")

# Default location of the bundled tool. The Dockerfile.gpu COPY puts
# it at this path; tests or out-of-container runs override via the
# llama_lookup_create_bin kwarg.
_DEFAULT_LLAMA_LOOKUP_CREATE = "/usr/local/bin/llama-lookup-create"

# Per-corpus-line repetition cap. An observation seen 50 times
# shouldn't repeat 50× in the corpus — llama-lookup-create n-gram
# extraction already weights by frequency, so 8× is enough to make
# the pattern dominant without bloating the corpus file.
_CORPUS_REPETITION_CAP = 8

# Time budget for the llama-lookup-create subprocess. The tool runs
# n-gram extraction over the corpus + tokenizes — for our 50k-line
# corpus it's well under a minute. Cap at 5min so a wedged subprocess
# can't pin the rebuild path forever.
_LLAMA_LOOKUP_CREATE_TIMEOUT_S = 300.0


@dataclass(slots=True, frozen=True)
class ExportResult:
    """What ``export_lookup_cache`` returns to its caller.

    The orchestrator surfaces these fields to the admin endpoint
    response so the operator can see what happened without grepping
    container logs.
    """

    cache_path: Path
    corpus_path: Path
    observations_used: int
    corpus_bytes: int
    cache_bytes: int
    duration_seconds: float


def cache_path_for(
    user_id: str,
    model_path: str,
    *,
    cache_root: Path | str = _DEFAULT_CACHE_ROOT,
) -> Path:
    """Compute the deterministic on-disk path for this (user, model)
    pair's cache file.

    Caller-owned helper so the LlamaServerManager hook can probe for
    cache existence without invoking the full exporter (the args-build
    path runs every model start and shouldn't kick off subprocess work).
    """
    root = Path(cache_root)
    safe_user = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
    model_stem = Path(model_path).stem
    return root / safe_user / f"{model_stem}.bin"


async def export_lookup_cache(
    store: ObservationStore,
    *,
    user_id: str,
    model_path: str,
    cache_root: Path | str = _DEFAULT_CACHE_ROOT,
    max_entries: int = 50_000,
    llama_lookup_create_bin: str = _DEFAULT_LLAMA_LOOKUP_CREATE,
) -> ExportResult:
    """Build the lookup cache for ``user_id`` against ``model_path``.

    Raises:
        FileNotFoundError — model_path doesn't exist (caller should
            handle; usually means the operator pointed at a stale path).
        RuntimeError — llama-lookup-create exited non-zero or timed out;
            the message includes stderr tail for diagnosis.
    """
    if not user_id:
        raise ValueError("export_lookup_cache requires a non-empty user_id")

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    cache_path = cache_path_for(user_id, model_path, cache_root=cache_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    observations = await store.top_k(user_id=user_id, k=max_entries)
    if not observations:
        # Empty corpus would produce an empty cache that llama-server
        # treats as broken. Raise cleanly so the caller can return a
        # meaningful "no observations yet" response to the operator.
        raise RuntimeError(
            f"observation store empty for user {user_id!r}; "
            "seed before exporting"
        )

    # Write corpus to a temp file alongside the final cache dir so
    # cleanup is bounded to that subtree.
    started = asyncio.get_event_loop().time()
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(cache_path.parent),
        prefix=".corpus_",
        suffix=".txt",
    ) as corpus_handle:
        corpus_path = Path(corpus_handle.name)
        for obs in observations:
            # Repetition encodes weight without ballooning. Lookup-create
            # is frequency-sensitive so the right amplitude here drives
            # which n-grams survive its internal pruning.
            rep = min(_CORPUS_REPETITION_CAP, max(1, obs.observation_count))
            line = f"{obs.prefix_text} {obs.continuation}\n"
            for _ in range(rep):
                corpus_handle.write(line)

    corpus_bytes = corpus_path.stat().st_size

    # Build the cache to a temp path, then atomic rename. If
    # llama-server happens to start while we're mid-build it sees the
    # PREVIOUS cache file (or no file), not a half-written one.
    tmp_cache_path = cache_path.with_suffix(".bin.partial")
    try:
        await _run_llama_lookup_create(
            llama_lookup_create_bin,
            model_path=str(model_path_obj),
            corpus_path=corpus_path,
            output_path=tmp_cache_path,
        )
        # os.replace is atomic within a filesystem; cache_path and
        # tmp_cache_path are in the same dir so this is guaranteed.
        os.replace(str(tmp_cache_path), str(cache_path))
    finally:
        # Drop the corpus file regardless of success. The cache is the
        # durable artifact; keeping the corpus around just clutters
        # /data/lookup_cache.
        try:
            corpus_path.unlink()
        except OSError:
            log.debug("export_corpus_cleanup_failed", path=str(corpus_path))
        # Best-effort cleanup of the partial cache if the rename didn't
        # happen (subprocess failure path).
        if tmp_cache_path.exists():
            try:
                tmp_cache_path.unlink()
            except OSError:
                log.debug(
                    "export_partial_cleanup_failed",
                    path=str(tmp_cache_path),
                )

    duration = asyncio.get_event_loop().time() - started
    cache_bytes = cache_path.stat().st_size if cache_path.exists() else 0
    log.info(
        "observation_cache_exported",
        user_id=user_id,
        model_stem=Path(model_path).stem,
        observations=len(observations),
        corpus_bytes=corpus_bytes,
        cache_bytes=cache_bytes,
        duration_s=round(duration, 2),
    )
    return ExportResult(
        cache_path=cache_path,
        corpus_path=corpus_path,
        observations_used=len(observations),
        corpus_bytes=corpus_bytes,
        cache_bytes=cache_bytes,
        duration_seconds=duration,
    )


async def _run_llama_lookup_create(
    binary: str,
    *,
    model_path: str,
    corpus_path: Path,
    output_path: Path,
) -> None:
    """Invoke the bundled llama-lookup-create subprocess.

    Wraps the CLI shape llama.cpp ships:
        llama-lookup-create -m MODEL -f CORPUS --lookup-cache-static OUTPUT

    The output flag is ``--lookup-cache-static`` (alias ``-lcs``), the same
    flag llama-server reads on the consumer side. There is no ``-o`` — the
    lookup-create example reuses common args and treats ``-lcs`` as the
    write target. Get this wrong and the binary exits ``invalid argument: -o``.

    Raises RuntimeError on non-zero exit or timeout — the message
    carries the stderr tail so the operator sees what went wrong
    without a separate log dive.
    """
    if not Path(binary).exists() and shutil.which(binary) is None:
        raise RuntimeError(
            f"llama-lookup-create binary not found at {binary!r}; "
            "ensure Dockerfile.gpu COPY includes it"
        )

    proc = await asyncio.create_subprocess_exec(
        binary,
        "-m", model_path,
        "-f", str(corpus_path),
        "--lookup-cache-static", str(output_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_LLAMA_LOOKUP_CREATE_TIMEOUT_S,
        )
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise RuntimeError(
            f"llama-lookup-create timed out after "
            f"{_LLAMA_LOOKUP_CREATE_TIMEOUT_S}s"
        ) from exc

    if proc.returncode != 0:
        tail = (stderr.decode("utf-8", errors="replace") or "")[-800:]
        raise RuntimeError(
            f"llama-lookup-create exited {proc.returncode}: {tail}"
        )
