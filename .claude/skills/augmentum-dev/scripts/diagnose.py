#!/usr/bin/env python3
"""Run codebase-model queries with diagnostic clustering.

Usage:
    python diagnose.py                       # all queries, with diagnosis
    python diagnose.py orphaned_endpoints    # one query
    python diagnose.py --list                # list available queries

Each query reports its row count + severity, then runs its DIAGNOSE
SQL (if defined) to break the count down by subsystem / recency / etc.
This is the "tell me WHERE the debt lives" layer that pure counts
miss.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_SKILL_DIR = _THIS.parent.parent
sys.path.insert(0, str(_SKILL_DIR))

from model import find_project_root, open_model, refresh  # noqa: E402
from queries import ALL_QUERIES  # noqa: E402


def _run_one(db, mod) -> int:
    """Execute one query module. Returns its row count."""
    rows = list(db.execute(mod.QUERY))
    count = len(rows)
    severity = mod.severity(count) if hasattr(mod, "severity") else "info"
    marker = {"ok": "[ok]", "warn": "[warn]", "error": "[error]"}.get(severity, "[--]")
    print(f"{marker} {mod.NAME}: {count}")
    if hasattr(mod, "DESCRIPTION"):
        print(f"      {mod.DESCRIPTION}")
    if count == 0:
        return 0
    if hasattr(mod, "DIAGNOSE"):
        diag_rows = list(db.execute(mod.DIAGNOSE))
        if diag_rows:
            print(f"      diagnosis ({len(diag_rows)} group(s)):")
            for r in diag_rows:
                _print_diag_row(r)
    return count


def _print_diag_row(row):
    """Render one diagnose row in a human-readable line.

    Heuristic — show ALL columns except long sample lists, which we
    truncate. Works for any DIAGNOSE shape because we don't hard-code
    column names.
    """
    parts = []
    sample_text = ""
    # sqlite3.Row's iteration yields VALUES, not keys — .keys() is the
    # only way to walk column names, so the SIM118 hint doesn't apply.
    for key in row.keys():  # noqa: SIM118
        val = row[key]
        if key.startswith("sample"):
            txt = str(val) if val else ""
            if len(txt) > 110:
                txt = txt[:107] + "..."
            sample_text = f"        e.g. {txt}"
            continue
        parts.append(f"{key}={val}")
    print("        " + "  ".join(parts))
    if sample_text:
        print(sample_text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("query_name", nargs="?", help="run only this query")
    parser.add_argument("--list", action="store_true", help="list query names + descriptions")
    args = parser.parse_args(argv)

    if args.list:
        for mod in ALL_QUERIES:
            desc = getattr(mod, "DESCRIPTION", "")
            print(f"  {mod.NAME:<32}  {desc}")
        return 0

    project_root = find_project_root(Path.cwd())
    db = open_model(project_root)
    refresh(db, project_root)

    if args.query_name:
        for mod in ALL_QUERIES:
            if args.query_name == mod.NAME:
                _run_one(db, mod)
                return 0
        print(f"unknown query: {args.query_name}", file=sys.stderr)
        print("available:", ", ".join(m.NAME for m in ALL_QUERIES), file=sys.stderr)
        return 2

    for mod in ALL_QUERIES:
        _run_one(db, mod)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
