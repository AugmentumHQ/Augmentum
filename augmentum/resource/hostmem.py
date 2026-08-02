"""Container-aware host memory accounting.

``psutil.virtual_memory()`` reports the **kernel's** view of RAM. Inside a
container that is the host (or, on WSL2, the whole WSL VM) — *not* the
cgroup limit this process is actually allowed to use. Every consumer that
sized itself as "a fraction of system RAM" therefore sized against memory
it could never have, and every consumer did so independently.

That is the arithmetic behind the 2026-07-25 incident: ``--cache-ram`` was
auto-sized to 25% of a 94 GiB WSL VM (23.6 GiB) while the augmentum
container's real share was a fraction of that. See
``docs/superpowers/specs/2026-07-25-resource-governance-design.md`` §4.1/B1.

This module is the single place that answers "how much memory do we
actually have," reading the cgroup limit when one exists and falling back
to psutil otherwise. Prefer :func:`memory_info` over ``psutil.virtual_memory``
anywhere the answer feeds a *sizing* decision.

Accounting rules, both load-bearing:

* **Working set is ``usage - inactive_file``**, matching kubelet. Reclaimable
  page cache is not "used". On this project's own box, 17 GB of a 26 GB
  footprint was page cache the kernel would have returned on demand; counting
  it as pressure would evict models for nothing.
* **cgroup v1 and v2 spell everything differently**, including their
  "unlimited" sentinels: v2 uses the literal string ``max``, v1 uses a huge
  integer (``PAGE_COUNTER_MAX`` scaled by page size). Both are handled below.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_CG_V2_ROOT = Path("/sys/fs/cgroup")
_CG_V1_ROOT = Path("/sys/fs/cgroup/memory")

# cgroup v1 writes a huge sentinel rather than a keyword when unlimited.
# The exact value depends on page size, so treat anything absurd as "no
# limit" rather than matching a specific constant.
_V1_UNLIMITED_FLOOR = 2**62

_MIB = 1024 * 1024


class MemoryInfo(NamedTuple):
    """A container-aware memory snapshot, all values in MiB."""

    total_mib: int
    """Memory this process may use: the cgroup limit if one is set,
    otherwise the kernel's total."""

    available_mib: int
    """Memory obtainable without reclaim pain. Under a cgroup limit this is
    ``limit - working_set``, so it shrinks as *we* grow rather than tracking
    a host-wide number we do not control."""

    used_mib: int
    """Working set: ``usage - inactive_file``. Excludes reclaimable cache."""

    source: str
    """``cgroup_v2`` | ``cgroup_v1`` | ``psutil`` — surfaced in telemetry so
    a mis-detection is visible rather than silently wrong."""

    limited: bool
    """True when an actual cgroup ceiling applies. When False, callers doing
    admission control should be conservative: nothing is bounding them."""


def _read_int(path: Path) -> int | None:
    """Read a single integer from a cgroup file, tolerating both sentinels."""
    try:
        raw = path.read_text().strip()
    except (OSError, ValueError):
        return None
    if not raw or raw == "max":  # v2 "unlimited"
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value <= 0 or value >= _V1_UNLIMITED_FLOOR:  # v1 "unlimited"
        return None
    return value


def _read_keyed(path: Path) -> dict[str, int]:
    """Parse a ``key value`` cgroup stat file (memory.stat)."""
    out: dict[str, int] = {}
    try:
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    out[parts[0]] = int(parts[1])
                except ValueError:
                    continue
    except OSError:
        return {}
    return out


class _CgroupView(NamedTuple):
    """Raw cgroup readings in bytes. ``limit`` is None when unbounded."""

    limit: int | None
    working: int
    version: str


def _probe_cgroup() -> _CgroupView | None:
    """Read this process's cgroup limit AND working set.

    **Usage is discovered independently of the limit.** An unlimited cgroup
    still accounts our memory accurately, and that reading is far more
    useful than the host's — it is *our* number. Conflating the two meant an
    explicitly-configured limit got paired with host-wide usage, reporting
    zero available memory on a mostly-idle container.
    """
    # cgroup v2: unified hierarchy, files at the root.
    current = _read_int(_CG_V2_ROOT / "memory.current")
    if current is not None:
        stat = _read_keyed(_CG_V2_ROOT / "memory.stat")
        working = max(0, current - stat.get("inactive_file", 0))
        return _CgroupView(_read_int(_CG_V2_ROOT / "memory.max"), working, "cgroup_v2")

    # cgroup v1: per-controller hierarchy. Docker Desktop on WSL2 is here,
    # which also means no `memory.high` — v1 has no throttle rung, only a
    # hard limit that OOM-kills.
    current = _read_int(_CG_V1_ROOT / "memory.usage_in_bytes")
    if current is not None:
        stat = _read_keyed(_CG_V1_ROOT / "memory.stat")
        inactive = stat.get("total_inactive_file", stat.get("inactive_file", 0))
        working = max(0, current - inactive)
        limit = _read_int(_CG_V1_ROOT / "memory.limit_in_bytes")
        return _CgroupView(limit, working, "cgroup_v1")

    return None


def _from_psutil() -> MemoryInfo:
    try:
        import psutil

        mem = psutil.virtual_memory()
        return MemoryInfo(
            total_mib=int(mem.total // _MIB),
            available_mib=int(mem.available // _MIB),
            used_mib=int((mem.total - mem.available) // _MIB),
            source="psutil",
            limited=False,
        )
    except Exception as exc:  # pragma: no cover - psutil absent/broken
        log.warning("hostmem_psutil_failed", error=str(exc))
        # A wrong-but-conservative default beats a crash on a sizing path.
        return MemoryInfo(
            total_mib=4096,
            available_mib=2048,
            used_mib=2048,
            source="fallback",
            limited=False,
        )


def _self_working_set_mib() -> int:
    """RSS of this process plus its children, in MiB.

    Used only when a memory ceiling applies but no cgroup accounts us
    against it. Children matter here: the llama-server slots are
    subprocesses, and they are the largest consumers by far.

    Sums RSS, so shared pages are counted more than once. That
    over-estimates, which is the safe direction for an admission gate —
    it refuses slightly early rather than slightly late.
    """
    try:
        import psutil

        proc = psutil.Process()
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return int(total // _MIB)
    except Exception:
        return 0


def memory_info() -> MemoryInfo:
    """Return a container-aware memory snapshot.

    Never raises: every sizing path in the codebase calls this, and a
    memory probe that throws would take down model loading entirely.
    """
    try:
        cg = _probe_cgroup()
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("hostmem_cgroup_probe_failed", error=str(exc))
        cg = None

    # An explicit limit always wins: the cgroup may be unreadable or
    # unbounded even when the operator has set a real ceiling in compose.
    limit_mib: int | None = None
    source = cg.version if cg else "psutil"
    override = os.environ.get("AUGMENTUM_MEMORY_LIMIT_MIB", "").strip()
    if override:
        try:
            limit_mib = max(256, int(override))
            source = "env_override"
        except ValueError:
            log.warning("hostmem_bad_override", value=override)
    if limit_mib is None and cg and cg.limit is not None:
        limit_mib = cg.limit // _MIB

    if limit_mib is None:
        # No ceiling anywhere. Report the kernel's view but keep our own
        # working set when the cgroup gave us one — it is the honest
        # number for "how much have WE taken".
        host = _from_psutil()
        if cg is not None:
            return MemoryInfo(
                total_mib=host.total_mib,
                available_mib=host.available_mib,
                used_mib=cg.working // _MIB,
                source=cg.version,
                limited=False,
            )
        return host

    # When no cgroup accounts us against the ceiling (bare metal, macOS,
    # Windows dev), the host's "used" is the wrong number — it includes
    # every other process on the machine. Measure our own process tree
    # instead, which is what the ceiling actually governs.
    used_mib = (cg.working // _MIB) if cg is not None else _self_working_set_mib()
    used_mib = max(0, min(used_mib, limit_mib))
    return MemoryInfo(
        total_mib=limit_mib,
        available_mib=max(0, limit_mib - used_mib),
        used_mib=used_mib,
        source=source,
        limited=True,
    )


def total_mib() -> int:
    """Memory this process may use, in MiB. Container-aware."""
    return memory_info().total_mib


def available_mib() -> int:
    """Memory obtainable without reclaim pain, in MiB. Container-aware."""
    return memory_info().available_mib


def budget_mib(fraction: float, *, floor_mib: int = 0, ceiling_mib: int = 0) -> int:
    """Size a cache as a fraction of memory we *actually* have.

    This is the safe replacement for the ``int(psutil.virtual_memory().total
    * X)`` pattern. The fraction applies to ``available`` rather than
    ``total`` whenever a real limit is known, because under a container
    ceiling the memory already spoken for is not ours to hand out again.
    """
    info = memory_info()
    base = info.available_mib if info.limited else info.total_mib
    value = int(max(0.0, fraction) * base)
    if floor_mib:
        value = max(floor_mib, value)
    if ceiling_mib:
        value = min(ceiling_mib, value)
    return max(0, value)
