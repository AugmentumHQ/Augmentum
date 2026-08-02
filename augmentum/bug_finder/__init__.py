"""Bug finder — autonomous closed-loop bug detection + fixing.

A *workspace kind*, not a mode. Coder owns workspace creation; bug
finder receives an existing ``workspace_id`` and drives the pipeline
against it. See ``docs/superpowers/specs/2026-05-10-bug-finder-mode-
design.md`` for the design rationale (notes on the reshape from
standalone mode → coder-integrated workspace kind are folded in).
"""

from __future__ import annotations
