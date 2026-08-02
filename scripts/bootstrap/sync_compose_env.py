"""Reconcile the compose overlay set into a single, portable source of truth.

`.augmentum.conf` is the human-editable overlay list (space-separated). Docker
Compose, however, reads the overlay set from `.env`'s `COMPOSE_FILE` (split by
`COMPOSE_PATH_SEPARATOR`). Historically those two were written independently by
setup and drifted (e.g. `compose.ocr.yaml` present in one, absent in the other),
and `.env` used `;` with no separator pinned — which works on Windows but breaks
raw `docker compose` on Linux (POSIX default separator is `:`, so a `;`-joined
list is read as one filename).

This module makes `.env` a *generated view* of `.augmentum.conf`:

  - `COMPOSE_FILE` = the `.augmentum.conf` overlays joined with `:`
  - `COMPOSE_PATH_SEPARATOR=:` pinned (safe on Windows AND Linux for the relative
    overlay filenames, which contain no `:`)

Run it on every `start` (the shell shims call this) so the raw-compose path and
the script path can never diverge. Idempotent; it rewrites only the two managed
lines and preserves every other line (including secrets) byte-for-byte. It never
prints secret values.

Usage:  python scripts/bootstrap/sync_compose_env.py [--check]
  --check : exit non-zero (and print a diff summary) if `.env` is out of sync,
            without writing. For a pre-push / CI drift guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Pinned separator — works cross-OS because overlay filenames are relative and
# colon-free. Do not change without auditing Windows path handling.
SEP = ":"

REPO = Path(__file__).resolve().parents[2]
CONF = REPO / ".augmentum.conf"
ENV = REPO / ".env"


def _read_overlays() -> list[str]:
    if not CONF.exists():
        print(f"[sync] {CONF.name} not found — nothing to reconcile", file=sys.stderr)
        raise SystemExit(2)
    # Space/newline separated; tolerate blank lines and stray whitespace.
    raw = CONF.read_text(encoding="utf-8")
    overlays = [tok for tok in raw.split() if tok.strip()]
    if not overlays:
        print(f"[sync] {CONF.name} is empty", file=sys.stderr)
        raise SystemExit(2)
    return overlays


def _desired(overlays: list[str]) -> dict[str, str]:
    return {
        "COMPOSE_FILE": SEP.join(overlays),
        "COMPOSE_PATH_SEPARATOR": SEP,
    }


def _split_line(line: str) -> tuple[str, str] | None:
    """Return (KEY, value) for a `KEY=...` env line, else None (blank/comment)."""
    stripped = line.lstrip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, _, val = stripped.partition("=")
    return key.strip(), val.rstrip("\n")


def _reconcile(lines: list[str], desired: dict[str, str]) -> tuple[list[str], bool]:
    """Replace managed keys in-place; append any missing. Returns (lines, changed)."""
    seen: set[str] = set()
    out: list[str] = []
    changed = False
    for line in lines:
        parsed = _split_line(line)
        if parsed and parsed[0] in desired:
            key = parsed[0]
            seen.add(key)
            want = f"{key}={desired[key]}"
            nl = "\n" if line.endswith("\n") else ""
            if parsed[1] != desired[key]:
                changed = True
            out.append(want + (nl or "\n"))
        else:
            out.append(line)
    for key, val in desired.items():
        if key not in seen:
            if out and not out[-1].endswith("\n"):
                out[-1] = out[-1] + "\n"
            out.append(f"{key}={val}\n")
            changed = True
    return out, changed


def main(argv: list[str]) -> int:
    check = "--check" in argv[1:]
    overlays = _read_overlays()
    desired = _desired(overlays)

    lines = ENV.read_text(encoding="utf-8").splitlines(keepends=True) if ENV.exists() else []
    new_lines, changed = _reconcile(lines, desired)

    if check:
        if changed:
            print("[sync] .env is OUT OF SYNC with .augmentum.conf:", file=sys.stderr)
            print(f"       want COMPOSE_FILE = {len(overlays)} overlays (sep '{SEP}')", file=sys.stderr)
            return 1
        print(f"[sync] .env in sync ({len(overlays)} overlays).")
        return 0

    if not changed:
        print(f"[sync] .env already current ({len(overlays)} overlays).")
        return 0

    # Atomic-ish write: temp then replace, so a crash can't leave a half file.
    tmp = ENV.with_suffix(".env.tmp")
    tmp.write_text("".join(new_lines), encoding="utf-8")
    tmp.replace(ENV)
    print(f"[sync] .env reconciled: COMPOSE_FILE = {len(overlays)} overlays, "
          f"COMPOSE_PATH_SEPARATOR='{SEP}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
