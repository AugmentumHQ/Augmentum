"""Cardsmith — co-design pipeline for new character cards.

Phase 1 ships the Single card type with the AI-Describe lane. Wiki, Ensemble,
and World/RPG land in subsequent phases. Public surface:

    from augmentum.modes.narrative.cardsmith import (
        CardsmithSession,
        get_or_create_session,
        get_session,
        drop_session,
        get_prompt,
        parse_field_emissions,
        build_character_payload,
    )
"""

from __future__ import annotations

from .output_mapper import build_character_payload
from .parser import FieldEmission, parse_field_emissions
from .prompts import (
    DEFAULT_ENSEMBLE_PROMPT,
    DEFAULT_SINGLE_PROMPT,
    DEFAULT_WORLD_RPG_PROMPT,
    get_prompt,
)
from .state import (
    CardsmithSession,
    drop_session,
    get_or_create_session,
    get_session,
)

__all__ = [
    "CardsmithSession",
    "DEFAULT_ENSEMBLE_PROMPT",
    "DEFAULT_SINGLE_PROMPT",
    "DEFAULT_WORLD_RPG_PROMPT",
    "FieldEmission",
    "build_character_payload",
    "drop_session",
    "get_or_create_session",
    "get_prompt",
    "get_session",
    "parse_field_emissions",
]
