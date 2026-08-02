"""CLI wrapper for the Hybrid Coder eval bench."""
# ruff: noqa: E402,I001

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.coder_evals.bench import main


if __name__ == "__main__":
    raise SystemExit(main())
