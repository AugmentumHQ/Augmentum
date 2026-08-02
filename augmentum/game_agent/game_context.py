"""Static per-game context blocks for the fast system prompt.

Every fast turn's system prompt carries these.  They are:
- fully known at session start (no inference needed)
- static for the entire session (never change)
- small enough to fit in the stable KV-prefix region

For unknown game profiles the block is omitted entirely so the prompt
stays terse on unsupported titles.
"""

from __future__ import annotations

# game_profile -> one short block: title · platform · genre, then
# one line each for controls and key mechanics.
# Keep each block under ~200 chars so it doesn't crowd the prompt.
_GAME_CONTEXT: dict[str, str] = {
    # ── Game-specific profiles ─────────────────────────────────────
    "pokemon_emerald": (
        "GAME: Pokémon Emerald · GBA · JRPG\n"
        "CONTROLS: confirm=A (advance/select), cancel=B (back/cancel), "
        "menu=START (pause menu), nav_*=D-pad (walk / move cursor)\n"
        "MECHANICS: turn-based battles (FIGHT/BAG/POKÉMON/RUN); dialog boxes "
        "swallow movement (press confirm to advance); long mandatory intro "
        "before overworld; navigate_to only works once walk_grid loads"
    ),
    "pokemon_rs": (
        "GAME: Pokémon Ruby/Sapphire · GBA · JRPG\n"
        "CONTROLS: confirm=A, cancel=B, menu=START, nav_*=D-pad\n"
        "MECHANICS: turn-based battles; dialog swallows movement; "
        "navigate_to only works once walk_grid loads"
    ),
    "pokemon_rby": (
        "GAME: Pokémon Red/Blue/Yellow · GB · JRPG\n"
        "CONTROLS: confirm=A, cancel=B, menu=START, nav_*=D-pad\n"
        "MECHANICS: turn-based battles; dialog swallows movement"
    ),
    "pokemon_gsc": (
        "GAME: Pokémon Gold/Silver/Crystal · GBC · JRPG\n"
        "CONTROLS: confirm=A, cancel=B, menu=START, nav_*=D-pad\n"
        "MECHANICS: turn-based battles; dialog swallows movement; "
        "day/night cycle; two regions"
    ),
    "zelda_links_awakening_dx": (
        "GAME: Zelda: Link's Awakening DX · GBC · Action-adventure\n"
        "CONTROLS: confirm=A (sword/action), cancel=B (item), "
        "menu=START (inventory), nav_*=D-pad (move)\n"
        "MECHANICS: real-time overworld combat; dialog signed by character "
        "name; navigate_to available once walk_grid loads"
    ),
    # ── Generic platform profiles ──────────────────────────────────
    "generic_gba": (
        "GAME: Generic GBA title · GBA · unknown genre\n"
        "CONTROLS: confirm=A, cancel=B, menu=START, select=SELECT, "
        "nav_*=D-pad"
    ),
    "generic_nes": (
        "GAME: Generic NES title · NES · unknown genre\n"
        "CONTROLS: confirm=A, cancel=B, menu=START, select=SELECT, "
        "nav_*=D-pad"
    ),
    "generic_snes": (
        "GAME: Generic SNES title · SNES · unknown genre\n"
        "CONTROLS: confirm=A, cancel=B, menu=START, select=SELECT, "
        "nav_*=D-pad, shoulder_l/L, shoulder_r/R"
    ),
    "generic_nds": (
        "GAME: Generic NDS title · NDS · unknown genre\n"
        "CONTROLS: confirm=A, cancel=B, menu=START, select=SELECT, "
        "nav_*=D-pad, shoulder_l/L, shoulder_r/R"
    ),
    "generic_genesis": (
        "GAME: Generic Genesis title · Genesis · unknown genre\n"
        "CONTROLS: confirm=A, cancel=B, menu=START, nav_*=D-pad, "
        "shoulder_l/C, shoulder_r/?, back=SELECT"
    ),
    "generic_gg": (
        "GAME: Generic Game Gear title · Game Gear · unknown genre\n"
        "CONTROLS: confirm=1, cancel=2, nav_*=D-pad, menu=START"
    ),
    "generic_sms": (
        "GAME: Generic Master System title · SMS · unknown genre\n"
        "CONTROLS: confirm=1, cancel=2, nav_*=D-pad, menu=START"
    ),
    "generic_pce": (
        "GAME: Generic PC Engine title · PCE · unknown genre\n"
        "CONTROLS: confirm=I, cancel=II, nav_*=D-pad, menu=SELECT, start=RUN"
    ),
    "generic_psx": (
        "GAME: Generic PlayStation title · PSX · unknown genre\n"
        "CONTROLS: confirm=Cross, cancel=Circle, menu=START, select=SELECT, "
        "nav_*=D-pad, shoulder_l/L1, shoulder_r/R1"
    ),
}


def game_context_for(game_profile: str | None) -> str:
    """Return the static context block for ``game_profile``, or '' if none."""

    if not game_profile:
        return ""
    return _GAME_CONTEXT.get(game_profile, "")


__all__ = ["game_context_for"]
