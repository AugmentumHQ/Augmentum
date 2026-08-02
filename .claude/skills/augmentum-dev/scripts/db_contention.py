#!/usr/bin/env python3
"""Augmentum DB-contention scanner — live writer-lock contention stats.

Counts "database is locked" errors and slow ``BEGIN IMMEDIATE`` events
in the augmentum container's recent log window. Surfaces the top callers
so the next investigator immediately knows which subsystem is contending
without needing to spelunk logs themselves.

Why this exists
---------------
The lock-contention diagnosis on 2026-05-26 walked through ~10 grep
passes to figure out that the resource ledger's BEGIN IMMEDIATE was
losing the race to chat-traffic writes. The signal was in the logs the
whole time; nothing surfaced it as a tracked metric. With this scanner
in the audit, you can:

  * See at a glance whether contention is hitting today
  * Compare contention now vs. recent runs (via audit history trend)
  * Catch a regression where a new subsystem starts hammering writes

Designed to be informational, not score-affecting. Lock contention is
expected under load — the resource ledger's failures are by-design
best-effort telemetry. This scanner makes the noise visible, doesn't
treat it as a defect.

Gracefully skips when Docker isn't running, the augmentum service isn't
up, or ``docker compose`` isn't on PATH — output reports ``skipped``
with the reason so the audit pipeline stays green in CI environments
without a live container.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Windows consoles default to a non-UTF-8 codepage; ✓ / → / … would
# raise UnicodeEncodeError. Make stdout/stderr lenient.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass


def _find_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "augmentum" / "proxy").is_dir() and (parent / "ui").is_dir():
            return parent
    print("ERROR: Cannot find Augmentum project root.", file=sys.stderr)
    sys.exit(2)


ROOT = _find_root()

_COLOR = os.environ.get("TERM") or os.name != "nt"
def _red(s: str) -> str:    return f"\033[91m{s}\033[0m" if _COLOR else s
def _yellow(s: str) -> str: return f"\033[93m{s}\033[0m" if _COLOR else s
def _green(s: str) -> str:  return f"\033[92m{s}\033[0m" if _COLOR else s
def _cyan(s: str) -> str:   return f"\033[96m{s}\033[0m" if _COLOR else s
def _dim(s: str) -> str:    return f"\033[2m{s}\033[0m" if _COLOR else s
def _bold(s: str) -> str:   return f"\033[1m{s}\033[0m" if _COLOR else s


# Default look-back window. Long enough to catch a chat session worth of
# activity; short enough that a one-off contention spike from hours ago
# doesn't pollute today's number.
DEFAULT_WINDOW = "1h"

# The two patterns we actually care about. ``database is locked`` is the
# hard-failure marker (lock acquisition timed out). ``slow_db_op`` with
# a BEGIN op is the soft signal (BEGIN waited > slow_ms but eventually
# acquired) — useful for trending contention before it becomes failures.
# Matches ANSI CSI escapes — color/style codes that structlog emits even
# when ``docker compose logs --no-color`` is set. ``--no-color`` only
# suppresses docker's own framing colour, NOT the inner log payload.
# Pre-stripping makes every downstream regex color-tolerant.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_LOCKED_RE = re.compile(r"OperationalError:\s+database is locked")
_SLOW_BEGIN_RE = re.compile(r"slow_db_op.*sql=BEGIN", re.IGNORECASE)
# Caller hop closest to user code in the trace. The first ``augmentum/<sub>``
# path on the trace line that isn't ``state/backends/sqlite.py`` is the
# subsystem actually doing the write. The trace is space-arrow-space
# separated so split on " <- " and skip frames in the sqlite layer.
_TRACE_RE = re.compile(r"caller=(.+?)\s+elapsed_ms")
_ELAPSED_RE = re.compile(r"elapsed_ms=([0-9.]+)")
_AUG_FRAME_RE = re.compile(r"augmentum/[a-z_/]+\.py:\d+")


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _fetch_logs(window: str) -> tuple[str, str | None]:
    """Run ``docker compose logs augmentum --since=<window>``.

    Returns ``(text, error)``. On success ``error`` is None and ``text``
    is the captured stdout. On skip (docker missing, service down,
    compose command failed) ``error`` is a short human-readable reason.
    """
    if not _docker_available():
        return "", "docker not on PATH"
    try:
        # ``--no-color`` strips the ANSI escapes augmentum's structlog
        # config emits, so our regexes don't have to be color-aware.
        # cwd=ROOT because ``docker compose`` reads the compose file from
        # the working directory.
        # encoding=utf-8 + errors=replace required: structlog's output
        # carries box-drawing chars + emoji that cp1252 (Windows default
        # for subprocess.run with text=True) can't decode, raising
        # UnicodeDecodeError in the reader thread and silently truncating
        # captured output.
        result = subprocess.run(
            ["docker", "compose", "logs", "augmentum",
             f"--since={window}", "--no-color"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return "", "docker compose logs timeout (30s)"
    except FileNotFoundError:
        return "", "docker compose not invokable"
    if result.returncode != 0:
        # Most common: service not running. Surface the stderr tail so
        # the user can debug without rerunning manually.
        stderr_tail = (result.stderr or "").strip().splitlines()[-1:] or [""]
        return "", f"docker compose logs failed: {stderr_tail[0][:120]}"
    return result.stdout or "", None


def _extract_subsystem_caller(trace: str) -> str:
    """Pick the closest-to-user-code frame from a caller trace.

    The trace is structlog's caller chain, e.g.::

      augmentum/state/backends/sqlite.py:48:_augmentum_caller_trace
        <- augmentum/state/backends/sqlite.py:146:_maybe_log
        <- ...
        <- augmentum/resource/ledger.py:1538:_persist_and_clear
        <- augmentum/main.py:14:main

    The frame we want for grouping is the first one NOT in
    ``state/backends/sqlite.py`` — that's the subsystem doing the
    actual call. Fall back to the whole trace if nothing matches.
    """
    for match in _AUG_FRAME_RE.finditer(trace):
        path = match.group(0)
        if "state/backends/sqlite.py" not in path:
            return path
    return trace[:80]


def _parse_slow_begin(line: str) -> tuple[str, float] | None:
    """Extract ``(subsystem_caller, elapsed_ms)`` from a slow_db_op line."""
    t = _TRACE_RE.search(line)
    e = _ELAPSED_RE.search(line)
    if not t or not e:
        return None
    try:
        return _extract_subsystem_caller(t.group(1)), float(e.group(1))
    except ValueError:
        return None


def scan(window: str = DEFAULT_WINDOW) -> dict:
    text, err = _fetch_logs(window)
    if err is not None:
        return {"skipped": True, "reason": err, "window": window}

    # Strip ANSI escapes so regexes (which assume plain text) match
    # cleanly. Done once per scan; SHA on 1.5MB of logs takes ~10ms.
    text = _ANSI_RE.sub("", text)

    locked_count = 0
    slow_begin_count = 0
    callers: Counter[str] = Counter()
    elapsed_samples: list[float] = []

    for line in text.splitlines():
        if _LOCKED_RE.search(line):
            locked_count += 1
            continue
        if _SLOW_BEGIN_RE.search(line):
            slow_begin_count += 1
            parsed = _parse_slow_begin(line)
            if parsed is not None:
                caller, elapsed = parsed
                callers[caller] += 1
                elapsed_samples.append(elapsed)

    elapsed_samples.sort()
    median = elapsed_samples[len(elapsed_samples) // 2] if elapsed_samples else 0.0
    p95_idx = max(0, int(len(elapsed_samples) * 0.95) - 1)
    p95 = elapsed_samples[p95_idx] if elapsed_samples else 0.0
    return {
        "skipped": False,
        "window": window,
        "locked_count": locked_count,
        "slow_begin_count": slow_begin_count,
        "median_ms": round(median, 1),
        "p95_ms": round(p95, 1),
        "top_callers": callers.most_common(5),
    }


def _print_report(stats: dict, verbose: bool) -> int:
    print(_bold("Augmentum DB-contention scan"))
    print(_dim(f"  window: {stats['window']}"))
    print()
    if stats.get("skipped"):
        print(_dim(f"  Skipped — {stats['reason']}."))
        print(_dim("  Live contention metrics require the augmentum container to be running."))
        print()
        # Empty-state still prints the closing line so audit.py's parser
        # doesn't false-fail on "no metric line".
        print(_green("  0 errors, 0 warnings"))
        return 0

    locked = stats["locked_count"]
    slow_begin = stats["slow_begin_count"]
    color = _red if locked > 50 else _yellow if locked > 0 else _green
    print(color(f"  database is locked errors: {locked}"))
    color = _yellow if slow_begin > 0 else _green
    print(color(f"  slow BEGIN events:         {slow_begin}"))
    if stats["median_ms"] or stats["p95_ms"]:
        print(_dim(f"  BEGIN wait: median {stats['median_ms']:.0f}ms / p95 {stats['p95_ms']:.0f}ms"))
    print()

    if stats["top_callers"]:
        print(_bold("  Top contending callers"))
        for caller, count in stats["top_callers"]:
            print(f"    {count:>4}  {caller}")
        print()

    if not locked and not slow_begin:
        print(_green("  No DB-contention events in window — writer lock looks healthy."))
        print()
    elif verbose and stats["top_callers"]:
        # Verbose hint about what these usually mean — the comment in
        # ledger.py is too far away for an investigator to find quickly.
        print(_dim("  Note: ledger.py:1538 (_persist_and_clear) is BEST-EFFORT telemetry"))
        print(_dim("  with an intentional 3s busy_timeout — its failures are expected"))
        print(_dim("  under chat load and don't impact user requests."))
        print()

    # Informational only — never emit errors. Warnings track presence of
    # contention so the audit history captures the trend.
    warnings = 1 if (locked or slow_begin) else 0
    if warnings:
        print(_yellow(f"  0 errors, {warnings} warning(s)"))
    else:
        print(_green("  0 errors, 0 warnings"))
    return 0


def main() -> int:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    window = DEFAULT_WINDOW
    for arg in sys.argv[1:]:
        if arg.startswith("--window="):
            window = arg.split("=", 1)[1]
    stats = scan(window)
    return _print_report(stats, verbose)


if __name__ == "__main__":
    sys.exit(main())
