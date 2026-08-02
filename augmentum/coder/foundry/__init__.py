"""Game foundry — the closed-loop generative-game pipeline.

Assembles existing Augmentum capabilities into a ring: Blender assets +
coder-generated game code -> deploy -> game_agent autonomous play ->
progress.py score -> defect relay -> regenerate. See
``docs/superpowers/specs/2026-08-01-blender-foundry-mvp-build-plan.md``.
"""
from __future__ import annotations

from augmentum.coder.foundry.contract import (
    GameBuildSpec,
    contract_prompt,
    semantic_inputs_from,
    validate_generated_game,
)

__all__ = [
    "GameBuildSpec",
    "contract_prompt",
    "semantic_inputs_from",
    "validate_generated_game",
]
