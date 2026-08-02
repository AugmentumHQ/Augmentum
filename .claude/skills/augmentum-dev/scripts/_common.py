#!/usr/bin/env python3
"""Shared helpers for the augmentum-dev scanner scripts.

Importing this module is enough to make stdout/stderr UTF-8-safe on a Windows
console (the ✓/→/… we print would otherwise raise UnicodeEncodeError mid-report
and the audit logs "parser found no metrics"). The other helpers — project-root
resolution, ANSI colors, and path:line suppression loading — are opt-in.

Scanners that predate this module keep their own copies; new code should use
these. Run scripts directly with no special flags and they just work.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

# --- console safety (applied on import) ------------------------------------
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


# --- project root ----------------------------------------------------------
def find_root() -> Path:
    """Walk up from this file to the Augmentum repo root (has augmentum/ + ui/)."""
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "augmentum" / "proxy").is_dir() and (parent / "ui").is_dir():
            return parent
    print("ERROR: cannot find Augmentum project root.", file=sys.stderr)
    sys.exit(2)


ROOT = find_root()


def rel(path: Path) -> str:
    """Forward-slash repo-relative path (or the str path if outside the repo)."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# --- ANSI colors (no-op when not a TTY-ish environment) --------------------
_COLOR = bool(os.environ.get("TERM")) or os.name != "nt"


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _COLOR else s


def red(s: str) -> str:    return _c("91", s)
def yellow(s: str) -> str: return _c("93", s)
def green(s: str) -> str:  return _c("92", s)
def cyan(s: str) -> str:   return _c("96", s)
def bold(s: str) -> str:   return _c("1", s)
def dim(s: str) -> str:    return _c("2", s)


# --- path:line suppressions ------------------------------------------------
def load_suppressions(filename: str, keys: tuple[str, ...]) -> dict[str, list[str]]:
    """Load (creating an empty skeleton if missing) a suppressions JSON that
    maps each key in ``keys`` to a list of ``path`` / ``path:line`` / dir-prefix
    entries. ``filename`` is resolved next to the calling scanner's directory —
    pass an absolute Path-derived str, or just the bare name and we'll look in
    this scripts/ dir."""
    path = Path(filename)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / filename
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {k: list(data.get(k, [])) for k in keys}
        except (json.JSONDecodeError, KeyError):
            pass
    skeleton = {"_comment": "path / path:line / dir-prefix entries reviewed and accepted; fix new findings, don't suppress them.", **{k: [] for k in keys}}
    path.write_text(json.dumps(skeleton, indent=2) + "\n", encoding="utf-8")
    return {k: [] for k in keys}


def is_suppressed(entries: list[str], rel_path: str, line_no: int | None = None) -> bool:
    """An entry matches if it equals the path, equals ``path:line``, or is a
    directory prefix of the path."""
    targets = {rel_path}
    if line_no is not None:
        targets.add(f"{rel_path}:{line_no}")
    for e in entries:
        if e in targets or rel_path == e or rel_path.startswith(e.rstrip("/") + "/"):
            return True
    return False
