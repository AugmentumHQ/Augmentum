#!/usr/bin/env python3
"""Thin CLI shim → augmentum.contracts.probe, wired with augmentum-dev's baseline.

The harness itself lives in augmentum/contracts/probe.py (a shipped module) so
the self-edit gate and coder gate share the exact same probing + diagnosis. This
shim just bootstraps sys.path (repo + project venv), pins the skill's baseline
file, and delegates. All flags (--out, --format, --quiet, --baseline,
--update-baseline, --timeout, --verbose) are handled by probe.main.

    python contract_test.py                    # probe, delta vs skill baseline
    python contract_test.py --update-baseline  # record current breaks as baseline
    python contract_test.py --format=json --out=FILE
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "augmentum" / "proxy").is_dir() and (parent / "ui").is_dir():
            return parent
    print("ERROR: cannot find Augmentum project root.", file=sys.stderr)
    sys.exit(2)


ROOT = _find_root()


def _bootstrap_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    venv = ROOT / ".venv"
    candidates = [venv / "Lib" / "site-packages"]
    lib = venv / "lib"
    if lib.is_dir():
        for child in lib.iterdir():
            if child.name.startswith("python") and (child / "site-packages").is_dir():
                candidates.append(child / "site-packages")
    for sp in candidates:
        if sp.is_dir() and str(sp) not in sys.path:
            sys.path.insert(0, str(sp))
            break


_bootstrap_path()

_BASELINE = ROOT / ".claude" / "skills" / "augmentum-dev" / "references" / "contracts_baseline.json"


if __name__ == "__main__":
    from augmentum.contracts.probe import main

    sys.exit(main(default_baseline=str(_BASELINE)))
