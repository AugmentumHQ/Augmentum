"""Contract-test substrate — auto-derived endpoint regression detection.

The shared engine behind three consumers (the augmentum-dev skill, the
self-edit gate, and the coder completion gate): discover a surface's
callable entrypoints, probe them, and turn any break into a complete,
deduped diagnostic dossier the fixer can act on.

This package holds the *oracle-agnostic* core — route discovery
(``discover``) and failure diagnosis (``diagnose``). How a callable app
is obtained (in-process mock harness vs a real workspace container) is
the consumer's job, not the core's.

Milestone 1 wiring lives in the augmentum-dev skill
(``scripts/contract_test.py``): in-process GET probing of Augmentum's
own FastAPI routes.
"""

from __future__ import annotations

from augmentum.contracts.discover import RouteSpec, discover_routes
from augmentum.contracts.diagnose import Dossier, ProbeResult, build_dossier, classify

__all__ = [
    "RouteSpec",
    "discover_routes",
    "ProbeResult",
    "Dossier",
    "classify",
    "build_dossier",
]
