"""Doc-coverage engine — the autonomy spine's Tier-2 mechanism.

Tier 1 (facts + refresh_docs) auto-heals *countable* claims. Tier 2
(this package) auto-*detects* when a code-derived SET drifts from the
SET the docs declare — subsystems, dispatch modes, provider cards, and
anything else you register as a ``CoverageSpec``. It cannot safely
auto-write prose (a wrong description is worse than a gap), so its job
is to surface the gap on every audit and hand the human a paste-ready
stub, reducing intervention to filling one cell.

Public API:
    from doc_coverage import SPECS, evaluate, scaffold_for, CoverageResult
"""

from __future__ import annotations

from .engine import CoverageResult, CoverageSpec, evaluate, scaffold_for
from .specs import SPECS, spec_by_name

__all__ = [
    "CoverageResult",
    "CoverageSpec",
    "SPECS",
    "evaluate",
    "scaffold_for",
    "spec_by_name",
]
