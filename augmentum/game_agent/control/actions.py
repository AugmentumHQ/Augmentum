"""Layer 1: canonical universal action vocabulary.

What the slow-path agent emits in its plan. These names describe
INTENT, not hardware. The same string means the same thing on a GBA,
a PSX, an MMO, and a desktop captured by xdotool.

Action names are ``[a-z_]+`` and stable across schema versions; adding
a new universal action is a deliberate vocabulary expansion that
deserves discussion. Game-specific actions that don't belong in the
universal namespace go in their game profile's ``actions`` dict
directly with arbitrary identifiers.

Categories
----------
The actions group into rough semantic clusters; the agent prompt
renders them grouped so the model finds the right verb quickly.
"""

from __future__ import annotations

# ── Navigation (cardinal-direction intent) ─────────────────────────────
#
# Tile-grid games (Pokémon, Zelda 2D) interpret these as "step one
# tile". Continuous-movement games (Mario, action RPGs) interpret as
# "hold this direction for ``duration_ms``". The agent treats them as
# a verb; per-game profiles document the actual behavior.
NAVIGATION_ACTIONS: frozenset[str] = frozenset({
    "nav_up", "nav_down", "nav_left", "nav_right",
})


# ── UI / dialog ───────────────────────────────────────────────────────
#
# Universal across every menu-driven game from JRPGs to MMOs. Carrying
# these as a stable layer means a journal entry about "confirmed past
# Brock's intro dialog" stays meaningful even if you later replay the
# same fight on a different console with a different button mapping.
UI_ACTIONS: frozenset[str] = frozenset({
    "confirm",   # A/Cross/Enter/Z — advance dialog, accept menu choice
    "cancel",    # B/Circle/Esc/X — back out, close menu, run away
    "menu",      # Start/Options/Menu — open primary menu
    "pause",     # often == menu, but games sometimes split them
    "back",      # Back/Share — secondary back / system-level back
})


# ── Interaction (gameplay verbs) ──────────────────────────────────────
#
# Genre-spanning verbs. Not every game has every one; the game
# profile only binds the ones that exist. ``special`` is the most
# elastic — it might be a magic attack in a JRPG, a grenade in a
# shooter, a build menu in a sandbox.
INTERACTION_ACTIONS: frozenset[str] = frozenset({
    "interact",       # use / talk to / pick up
    "attack",         # primary attack / use selected item
    "defend",         # block / guard
    "special",        # secondary / sub / heavy attack
    "use_item",       # quick-slot item use
})


# ── System / inventory ────────────────────────────────────────────────
#
# Less universally mapped — many games fold these into ``menu``. The
# split exists so a game profile CAN expose them when the game has a
# dedicated button (e.g., NDS uses different stylus areas for Bag vs
# Pokémon list).
SYSTEM_ACTIONS: frozenset[str] = frozenset({
    "inventory",
    "map",
    "talk",
    "weapon_swap",
})


# ── The full universal vocabulary ─────────────────────────────────────
#
# A game profile MAY declare actions outside this set (e.g., "register"
# for Pokémon's registered key item slot, or "dig" for Luanti). Those
# count as "game-specific actions" -- they're rendered in the prompt
# alongside universal ones, and the resolver handles them identically.
# But preferring the universal set keeps prompts and journals portable.
UNIVERSAL_ACTIONS: frozenset[str] = (
    NAVIGATION_ACTIONS
    | UI_ACTIONS
    | INTERACTION_ACTIONS
    | SYSTEM_ACTIONS
)


# Default human-readable descriptions, used as fallback hints when a
# game profile doesn't override. Game profiles SHOULD override with
# game-specific text ("confirm = advance Pokémon dialog") because
# the in-game effect is more useful than the generic verb.
ACTION_DESCRIPTIONS: dict[str, str] = {
    # Navigation
    "nav_up":     "Move/walk/face north (or up on a 2D plane).",
    "nav_down":   "Move/walk/face south (or down on a 2D plane).",
    "nav_left":   "Move/walk/face west (or left on a 2D plane).",
    "nav_right":  "Move/walk/face east (or right on a 2D plane).",
    # UI
    "confirm":    "Accept current choice, advance dialog, pick up item.",
    "cancel":     "Back out, close menu, refuse, flee.",
    "menu":       "Open the main menu / inventory / status screen.",
    "pause":      "Pause the game without opening a menu (when distinct).",
    "back":       "System-level back / dismiss / share.",
    # Interaction
    "interact":   "Use / talk to / pick up the thing in front of you.",
    "attack":     "Primary attack or use selected weapon/item.",
    "defend":     "Block / guard / parry.",
    "special":    "Secondary / sub / heavy / magic action.",
    "use_item":   "Activate the quick-slot or registered item.",
    # System
    "inventory":  "Open the inventory / bag.",
    "map":        "Open the map / world view.",
    "talk":       "Initiate dialog with the nearest NPC.",
    "weapon_swap": "Cycle to next weapon or hotbar slot.",
}


def is_universal_action(name: str) -> bool:
    """True iff ``name`` is in the canonical universal vocabulary.

    Use when:
    - Validating profile JSON to flag suspicious deviations.
    - Splitting a prompt-render into "universal" vs "game-specific"
      sections so the model sees the portable ones first.
    """

    return name in UNIVERSAL_ACTIONS


__all__ = [
    "ACTION_DESCRIPTIONS",
    "INTERACTION_ACTIONS",
    "NAVIGATION_ACTIONS",
    "SYSTEM_ACTIONS",
    "UI_ACTIONS",
    "UNIVERSAL_ACTIONS",
    "is_universal_action",
]
