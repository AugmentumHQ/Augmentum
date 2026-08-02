#!/usr/bin/env python3
"""Emit paste-ready stub doc entries for undocumented items.

For every gap the coverage engine finds (a subsystem with no Map row, a
mode with no Handler row, a provider with no card), print a stub line
ready to paste into the doc — so the residual human step is filling a
one-line description, not authoring a row from scratch.

Usage:
    scaffold_doc_row.py                 # stubs for every spec with a gap
    scaffold_doc_row.py subsystems      # just one spec
    scaffold_doc_row.py --list          # list registered coverage specs
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_SKILL_DIR = _THIS.parent.parent  # scripts/ -> augmentum-dev/
sys.path.insert(0, str(_SKILL_DIR))

import _common  # noqa: E402,F401 — side effect: UTF-8-safe stdout on Windows
from doc_coverage import SPECS, evaluate, scaffold_for, spec_by_name  # noqa: E402
from model import find_project_root  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = find_project_root(Path.cwd())

    if "--list" in argv:
        print("Registered coverage specs:")
        for spec in SPECS:
            print(f"  {spec.name:<16} {spec.describe}")
        return 0

    names = [a for a in argv if not a.startswith("-")]
    if names:
        specs = [spec_by_name(n) for n in names]
        if any(s is None for s in specs):
            unknown = [n for n, s in zip(names, specs) if s is None]
            print(f"unknown spec(s): {', '.join(unknown)}  "
                  f"(see --list)")
            return 1
    else:
        specs = list(SPECS)

    any_gap = False
    for spec in specs:
        res = evaluate(spec, root)
        if not res.missing:
            continue
        any_gap = True
        print(f"\n# {spec.name}: {len(res.missing)} undocumented "
              f"→ paste into {spec.fix_location}")
        for line in scaffold_for(spec, root):
            print(line)

    if not any_gap:
        print("No coverage gaps — every tracked list is fully documented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
