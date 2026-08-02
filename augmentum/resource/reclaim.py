"""Manual memory reclamation — the governor's actuator layer, run by hand.

Spec: ``docs/superpowers/specs/2026-07-25-resource-governance-design.md`` §7.1.

Augmentum's memory problem is not a spike, it is a **ratchet**. The 2026-07-25
incident accumulated over hours: nothing shrank because nothing was under
pressure, so a pressure-triggered governor would have arrived too late by
construction (§5.5.1). This module is the other half — reclamation, which runs
because someone asked, not because the box is already in trouble.

It is deliberately the *manual* version first. The actuators here are exactly
the ones an automatic governor would fire; putting a human on the trigger lets
us watch them work, and gives Phase 4's dry-run mode a control to compare
against, before anything evicts on its own.

Three honesty rules, each earned from a specific failure mode:

* **Report measured deltas, never declared ones.** ``free()`` returns memory to
  the allocator's arena, not to the kernel (§5.5.1 H2), so a component that
  says it freed 4 GB may have returned nothing. Every number this module
  reports is a working-set reading taken before and after.
* **Name what cannot be reclaimed, and why.** ``mlock``'d weights are
  structurally unreturnable while the process lives (H3). Hiding that reads as
  "reclaim did nothing"; showing it teaches that such memory has to be refused
  at *admission* instead, which is what ``ledger.check_ram_fit`` now does.
* **Never touch what is in use.** A reclaim that interrupts the user is a
  worse outcome than the memory it recovered.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any

from augmentum.resource import hostmem
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

#: A slot must be idle at least this long to be offered as a candidate.
#: Short enough to be useful mid-session, long enough that a user pausing to
#: read a reply does not get their model unloaded out from under them.
DEFAULT_MIN_IDLE_S = 120.0


@dataclass
class Candidate:
    """One reclaimable (or explicitly non-reclaimable) memory holder."""

    key: str
    """Stable identifier the execute call passes back to select this item."""

    label: str
    """Human-facing name, e.g. ``Slot B — qwen3-8b``."""

    kind: str
    """``slot`` | ``allocator`` | ``cuda_cache``."""

    mib: int
    """Best available size estimate. 0 when genuinely unknown — see ``est``."""

    est: bool = False
    """True when ``mib`` is an estimate rather than a measurement. The UI must
    not present an estimate as a promise; the measured delta comes after."""

    reclaimable: bool = True
    """False means it is shown but cannot be selected."""

    reason: str = ""
    """Why it cannot be reclaimed, or what reclaiming costs. Always populated
    for non-reclaimable entries — an unexplained refusal is a bug report."""

    restore_s: float = 0.0
    """Rough cost to bring this back, in seconds. This is the eviction currency
    the automatic governor will rank by (§5.5 cost-to-restore), surfaced here so
    the human trigger prices the same thing the machine eventually will."""

    detail: dict[str, Any] = field(default_factory=dict)


def _proc_rss_mib(pid: int | None) -> int:
    """RSS of a pid in MiB, or 0 if unavailable."""
    if not pid:
        return 0
    try:
        import psutil

        return int(psutil.Process(pid).memory_info().rss // (1024 * 1024))
    except Exception:
        return 0


def _proc_locked_mib(pid: int | None) -> int:
    """Locked (unswappable, unreclaimable) memory of a pid, in MiB.

    Read from ``VmLck`` rather than inferred from the command line: a process
    may carry ``--mlock`` and still have failed to lock anything if it hit
    ``RLIMIT_MEMLOCK``, and the kernel's number is the one that matters.
    """
    if not pid:
        return 0
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmLck:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        return 0
    return 0


def trim_allocator() -> int:
    """Return freed pages from our own heap to the kernel. Measured, in MiB.

    CPython (and glibc beneath it) keep freed pages in their own arenas rather
    than handing them back, so a model teardown can look complete while the
    container's working set never moves. ``malloc_trim`` is what actually
    returns them.

    This already existed, wired to exactly one teardown path in
    ``image/vram.py``. Every unload path needs it, so it lives here now.
    """
    before = _proc_rss_mib(_own_pid())
    # Three passes: generational GC can only collect a cycle once the objects
    # keeping it alive are themselves collected.
    for _ in range(3):
        gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        # Windows/macOS (no libc.so.6) or musl (no malloc_trim symbol). The
        # gc.collect() passes above still did their part.
        pass
    after = _proc_rss_mib(_own_pid())
    return max(0, before - after)


def drop_file_cache(path: str) -> int:
    """Evict one file's pages from the kernel page cache. Returns MiB advised.

    **Why this exists, and why it is not optional on WSL2.**

    llama.cpp mmaps the whole GGUF, so every model load leaves the entire
    file resident as page cache. That cache is normally harmless — it is
    reclaimable, and Linux drops it under pressure. On WSL2 with GPU-PV it
    is not harmless: a CUDA device allocation needs a host-memory backing
    allocation, and that path does **not** trigger page-cache reclaim. It
    simply fails. The driver then reports ``cudaMalloc failed: out of
    memory``, naming VRAM that is completely free.

    Observed 2026-07-26: after a few model swaps the WSL VM sat at 88 GB of
    page cache and 2 GB free, and a 17.8 GB model could not claim its
    15.7 GiB weights buffer on a 24 GB card with 23 GB of VRAM idle. The
    identical load succeeded seconds after the cache was dropped. The user
    reads this as "the weights are never released from RAM", which is
    exactly right.

    ``POSIX_FADV_DONTNEED`` is the targeted fix: it evicts the pages of the
    file we just stopped using, needs no privileges (unlike
    ``/proc/sys/vm/drop_caches``, which is read-only inside a container),
    and touches nothing else. Clean pages go immediately; anything still
    mapped by a live process stays, which is the correct behaviour — we
    only ever call this once the owning llama-server has exited.
    """
    if not path:
        return 0
    try:
        import os

        size = os.path.getsize(path)
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        return int(size // (1024 * 1024))
    except (OSError, AttributeError):
        # Missing file (already swapped out), or a platform without
        # posix_fadvise (macOS/Windows). Neither is worth failing a teardown.
        return 0


def _own_pid() -> int | None:
    try:
        import os

        return os.getpid()
    except Exception:
        return None


def _slot_entries(app_state: Any) -> list[tuple[str, str, Any]]:
    """(key, display prefix, manager) for each engine slot that may be resident."""
    out: list[tuple[str, str, Any]] = []
    primary = getattr(app_state, "llama_manager", None)
    if primary is not None:
        out.append(("slot_a", "Slot A", primary))
    for attr, key, name in (
        ("secondary_slot", "slot_b", "Slot B"),
        ("classifier_slot", "slot_c", "Slot C"),
    ):
        holder = getattr(app_state, attr, None)
        mgr = getattr(holder, "manager", None) if holder is not None else None
        if mgr is not None:
            out.append((key, name, mgr))
    return out


def _slot_candidate(key: str, name: str, mgr: Any, min_idle_s: float) -> Candidate | None:
    """Classify one slot. Returns None when the slot holds nothing."""
    model_id = getattr(mgr, "model_id", "") or ""
    proc = getattr(mgr, "process", None)
    pid = getattr(proc, "pid", None) if proc is not None else None
    if not model_id or pid is None:
        return None

    label = f"{name} — {model_id}"
    rss = _proc_rss_mib(pid)
    locked = _proc_locked_mib(pid)
    last = float(getattr(mgr, "_last_request_time", 0.0) or 0.0)
    idle_s = (time.monotonic() - last) if last > 0 else -1.0

    cand = Candidate(
        key=key,
        label=label,
        kind="slot",
        mib=rss,
        reclaimable=True,
        # Reloading means re-reading the weights from disk and re-warming;
        # scale off resident size rather than pretending to a precision we
        # do not have. Deliberately coarse — it ranks, it does not promise.
        restore_s=round(max(2.0, rss / 512.0), 1),
        detail={"pid": pid, "model_id": model_id, "idle_s": round(idle_s, 1),
                "locked_mib": locked},
    )

    # Pinned: something holds a live reference (an in-flight request, a
    # session bound to this slot). Never reclaim.
    try:
        is_pinned = bool(mgr.is_pinned(model_id))
    except Exception:
        # Treat an unanswerable pin check as pinned. Guessing "free" here
        # unloads a model out from under a live request; guessing "busy"
        # costs nothing but a skipped reclaim.
        is_pinned = True
    if is_pinned:
        cand.reclaimable = False
        cand.reason = "pinned — in active use"
        return cand

    if idle_s < 0:
        cand.reclaimable = False
        cand.reason = "no completed request yet — may be mid-load"
        return cand

    if idle_s < min_idle_s:
        cand.reclaimable = False
        cand.reason = f"active {int(idle_s)}s ago (needs {int(min_idle_s)}s idle)"
        return cand

    if locked > 0:
        # H3 made visible. The memory is real and it is ours, but no amount of
        # trimming returns it — only stopping the process does. Say so plainly
        # rather than letting the user read a small delta as a broken button.
        cand.reason = (
            f"{locked / 1024:.1f} GB is mlocked — only a full unload returns it"
        )
    return cand


async def preview(app_state: Any, *, min_idle_s: float = DEFAULT_MIN_IDLE_S) -> dict:
    """What a reclaim would do, without doing it."""
    info = hostmem.memory_info()
    cands = [
        c
        for key, name, mgr in _slot_entries(app_state)
        if (c := _slot_candidate(key, name, mgr, min_idle_s)) is not None
    ]

    # The allocator is always offered: it is free, instant, and reversible in
    # the only sense that matters (the pages come back when we need them).
    cands.append(
        Candidate(
            key="allocator",
            label="Python/glibc allocator slack",
            kind="allocator",
            mib=0,
            est=True,
            reason="size is not knowable in advance — measured after the run",
            restore_s=0.0,
        )
    )

    reclaimable = [c for c in cands if c.reclaimable]
    blocked = [c for c in cands if not c.reclaimable]
    return {
        "memory": info._asdict(),
        "candidates": [c.__dict__ for c in reclaimable],
        "blocked": [c.__dict__ for c in blocked],
        # Sum of the knowable parts only. Flagged so the UI can render it as
        # "up to", never as a guarantee.
        "estimated_mib": sum(c.mib for c in reclaimable if not c.est),
        "estimate_is_partial": any(c.est for c in reclaimable),
    }


async def run(
    app_state: Any,
    *,
    keys: list[str] | None = None,
    min_idle_s: float = DEFAULT_MIN_IDLE_S,
) -> dict:
    """Reclaim the selected candidates and report the **measured** delta.

    ``keys=None`` means "everything currently reclaimable". Anything not
    reclaimable at execute time is skipped and reported, even if the caller
    asked for it — the preview may be seconds stale, and a slot that became
    busy in between must win.
    """
    plan = await preview(app_state, min_idle_s=min_idle_s)
    wanted = set(keys) if keys else {c["key"] for c in plan["candidates"]}

    before = hostmem.memory_info()
    freed: list[dict] = []
    skipped: list[dict] = []

    for c in plan["blocked"]:
        if c["key"] in wanted:
            skipped.append({"key": c["key"], "label": c["label"], "reason": c["reason"]})

    for c in plan["candidates"]:
        if c["key"] not in wanted or c["kind"] != "slot":
            continue
        # Re-check immediately before acting: preview → click → execute is
        # easily 10s of seconds, and the user may have started a generation.
        entry = next((e for e in _slot_entries(app_state) if e[0] == c["key"]), None)
        if entry is None:
            continue
        fresh = _slot_candidate(entry[0], entry[1], entry[2], min_idle_s)
        if fresh is None or not fresh.reclaimable:
            skipped.append({
                "key": c["key"], "label": c["label"],
                "reason": fresh.reason if fresh else "no longer resident",
            })
            continue
        pid = c["detail"].get("pid")
        rss_before = _proc_rss_mib(pid)
        try:
            await entry[2].stop()
        except Exception as exc:
            log.warning("reclaim_slot_stop_failed", key=c["key"], error=str(exc))
            skipped.append({"key": c["key"], "label": c["label"],
                            "reason": f"unload failed: {exc}"})
            continue
        freed.append({"key": c["key"], "label": c["label"], "measured_mib": rss_before})
        log.info("reclaim_slot_unloaded", key=c["key"], label=c["label"],
                 rss_mib=rss_before)

    trimmed = 0
    if "allocator" in wanted:
        trimmed = trim_allocator()
        freed.append({
            "key": "allocator",
            "label": "Python/glibc allocator slack",
            "measured_mib": trimmed,
        })

    after = hostmem.memory_info()
    # The container-level delta is the number that actually matters, and it is
    # the one that catches H2: if a component "freed" 4 GB and this reads 0,
    # the memory went back to an arena, not to the kernel.
    delta = max(0, before.used_mib - after.used_mib)
    log.info(
        "reclaim_run_complete",
        working_set_before_mib=before.used_mib,
        working_set_after_mib=after.used_mib,
        measured_freed_mib=delta,
        allocator_trim_mib=trimmed,
        items=len(freed),
        skipped=len(skipped),
    )
    return {
        "ok": True,
        "freed": freed,
        "skipped": skipped,
        "measured_freed_mib": delta,
        "before": before._asdict(),
        "after": after._asdict(),
    }
