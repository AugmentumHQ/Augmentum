"""CompanionRuntime — the top-level kernel for the Companion (Becca).

See the canonical design package at:
- docs/superpowers/specs/2026-05-14-companion-runtime-README.md
- docs/superpowers/specs/2026-05-14-companion-runtime-design-v2.md

Sprint 1 ships the substrate (this package skeleton + identity + state +
memory facade + presence bus + kernel composition). Behind
``companion_runtime_enabled`` flag; inert until flipped on.
"""

from __future__ import annotations

__all__: list[str] = []
