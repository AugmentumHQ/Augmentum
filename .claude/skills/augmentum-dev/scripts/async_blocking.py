#!/usr/bin/env python3
"""Static event-loop blocking-call detector.

Augmentum is a single-event-loop FastAPI app. A synchronous blocking call made
*directly* inside an ``async def`` freezes the loop for every concurrent user
until it returns — the exact class behind the 2026-06-13 event-loop-stall
incident (sync ``EmbeddingService`` on the roster path, sqlite/subprocess on
request paths). There's a runtime watchdog that logs ``event_loop_stall lag_s=``
after the fact, and ``db_contention.py`` scrapes Docker logs for slow BEGINs —
but nothing caught these *before* they shipped. This scanner does, statically.

What it flags (only calls lexically DIRECT in an async body — never inside a
nested ``def``/``lambda``, since those are the standard ``to_thread`` offload
idiom and flagging them is almost always a false positive):

  ERROR  — unambiguous loop blockers:
           time.sleep, requests.*, subprocess.run/call/check_*/Popen,
           urllib.request.urlopen, os.system, httpx top-level sync helpers
  WARNING — heuristic project blockers: synchronous embedding calls
           (.embed / .embed_query / .get_embedding / …) — wrap in
           ``ctx.run_in_thread`` / ``asyncio.to_thread``.

A reference passed to an offloader (``asyncio.to_thread(time.sleep, 5)``) is a
Name, not a Call, so it is correctly NOT flagged.

Reviewed false positives go in ``async_blocking_suppressions.json``
(``path`` / ``path:line`` / dir-prefix entries), same convention as the other
scanners. Suppress false positives, not real findings.

Usage:
    python async_blocking.py
    python async_blocking.py --verbose   # also list suppressed findings
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

from _common import (
    ROOT,
    bold,
    cyan,
    dim,
    green,
    is_suppressed,
    load_suppressions,
    red,
    rel,
    yellow,
)

# Only the server runs on the shared loop. Tests/scripts/ui are out of scope.
SCAN_DIR = ROOT / "augmentum"

SUPPRESSIONS_FILE = "async_blocking_suppressions.json"

# --- blocking signatures ----------------------------------------------------
# Matched on the trailing "module.func" of the call's dotted path, so both
# `import time; time.sleep()` and `import subprocess as sp; sp.run()` resolve.
_ERROR_PAIRS: dict[str, str] = {
    "time.sleep": "await asyncio.sleep(...)",
    "subprocess.run": "await asyncio.create_subprocess_exec(...) / ctx.run_in_thread(...)",
    "subprocess.call": "await asyncio.create_subprocess_exec(...)",
    "subprocess.check_call": "await asyncio.create_subprocess_exec(...)",
    "subprocess.check_output": "await asyncio.create_subprocess_exec(...) and read stdout",
    "os.system": "await asyncio.create_subprocess_exec(...)",
    "request.urlopen": "httpx.AsyncClient",            # urllib.request.urlopen
    "requests.get": "httpx.AsyncClient",
    "requests.post": "httpx.AsyncClient",
    "requests.put": "httpx.AsyncClient",
    "requests.patch": "httpx.AsyncClient",
    "requests.delete": "httpx.AsyncClient",
    "requests.head": "httpx.AsyncClient",
    "requests.request": "httpx.AsyncClient",
}
# Bare-name aliases brought in via `from time import sleep`, etc.
_ERROR_BARE: dict[str, str] = {
    "sleep": "await asyncio.sleep(...)",          # from time import sleep
    "urlopen": "httpx.AsyncClient",               # from urllib.request import urlopen
}
# httpx top-level helpers are synchronous (httpx.get != AsyncClient.get).
_HTTPX_SYNC = {"httpx.get", "httpx.post", "httpx.put", "httpx.patch",
               "httpx.delete", "httpx.head", "httpx.request", "httpx.stream"}

# Spawn-and-return calls: briefly block on fork/exec but don't wait for the
# child, so they're a softer signal than the waiting subprocess.* family.
_WARN_PAIRS: dict[str, str] = {
    "subprocess.Popen": "asyncio.create_subprocess_exec(...) (fork/exec briefly blocks the loop)",
}

# Heuristic (WARNING): synchronous embedding — the incident pattern.
_EMBED_METHODS = {
    "embed", "embed_text", "embed_query", "embed_documents",
    "embed_sync", "get_embedding", "get_embeddings", "encode_text",
}


def _dotted(func: ast.AST) -> str:
    """Dotted path of a call target, e.g. ``self.embedder.embed`` or ``time.sleep``."""
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def _last_two(dotted: str) -> str:
    return ".".join(dotted.split(".")[-2:])


def _direct_calls(node: ast.AST):
    """Yield Call nodes inside an async function body WITHOUT descending into
    nested function / lambda scopes (those run later, often offloaded)."""
    for stmt in getattr(node, "body", []):
        yield from _walk(stmt)


def _walk(n: ast.AST):
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return
    if isinstance(n, ast.Call):
        yield n
    for child in ast.iter_child_nodes(n):
        yield from _walk(child)


class _Finding:
    __slots__ = ("path", "line", "func", "fix", "severity")

    def __init__(self, path: str, line: int, func: str, fix: str, severity: str):
        self.path = path
        self.line = line
        self.func = func
        self.fix = fix
        self.severity = severity


def _classify(call: ast.Call) -> tuple[str, str, str] | None:
    """Return (display_name, fix_hint, severity) or None if not blocking."""
    dotted = _dotted(call.func)
    if not dotted:
        return None
    pair = _last_two(dotted)
    if pair in _ERROR_PAIRS:
        return dotted, _ERROR_PAIRS[pair], "error"
    if dotted in _HTTPX_SYNC:
        return dotted, "httpx.AsyncClient (top-level httpx.* is synchronous)", "error"
    if isinstance(call.func, ast.Name) and call.func.id in _ERROR_BARE:
        return call.func.id, _ERROR_BARE[call.func.id], "error"
    if pair in _WARN_PAIRS:
        return dotted, _WARN_PAIRS[pair], "warning"
    # Heuristic embedding: an attribute call whose method name is an embed verb.
    if isinstance(call.func, ast.Attribute) and call.func.attr in _EMBED_METHODS:
        return dotted, "ctx.run_in_thread(...) / asyncio.to_thread(...)", "warning"
    return None


def scan() -> list[_Finding]:
    findings: list[_Finding] = []
    for py in sorted(SCAN_DIR.rglob("*.py")):
        try:
            src = py.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(py))
        except (OSError, SyntaxError):
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.AsyncFunctionDef):
                continue
            # A blocking call is never awaited; `await sleep(...)` (where sleep
            # is a coroutine fn) must NOT be flagged. Collect awaited call nodes.
            awaited = {
                id(n.value)
                for n in ast.walk(fn)
                if isinstance(n, ast.Await) and isinstance(n.value, ast.Call)
            }
            for call in _direct_calls(fn):
                if id(call) in awaited:
                    continue
                hit = _classify(call)
                if hit is None:
                    continue
                name, fix, severity = hit
                findings.append(
                    _Finding(rel(py), call.lineno, name, fix, severity)
                )
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verbose", action="store_true",
                    help="also list suppressed findings")
    args = ap.parse_args(argv)

    supp = load_suppressions(SUPPRESSIONS_FILE, ("findings",))["findings"]
    raw = scan()

    shown: list[_Finding] = []
    suppressed = 0
    for f in raw:
        if is_suppressed(supp, f.path, f.line):
            suppressed += 1
            if args.verbose:
                print(dim(f"  suppressed  {f.path}:{f.line}  {f.func}"))
            continue
        shown.append(f)

    errors = [f for f in shown if f.severity == "error"]
    warnings = [f for f in shown if f.severity == "warning"]

    print(bold(cyan("Async event-loop blocking calls")))
    if errors:
        print(red(f"\n  Loop blockers ({len(errors)}):"))
        for f in errors:
            # Path printed as augmentum/...py:line so audit.py's hotspot
            # extractor links it across scanners.
            print(f"    {f.path}:{f.line}  {red(f.func)}()  → {f.fix}")
    if warnings:
        print(yellow(f"\n  Sync-embedding (heuristic) ({len(warnings)}):"))
        for f in warnings:
            print(f"    {f.path}:{f.line}  {yellow(f.func)}()  → {f.fix}")

    if suppressed:
        print(dim(f"\n  Suppressions applied: {suppressed}"))

    print()
    if errors:
        print(red(f"  {len(errors)} error(s), {len(warnings)} warning(s)"))
        return 1
    print(green(f"  0 errors, {len(warnings)} warning(s)") if warnings
          else green("  0 errors, 0 warnings — no blocking calls on the loop"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
