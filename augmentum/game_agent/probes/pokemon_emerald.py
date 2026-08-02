"""Pokémon Emerald RAM probe preset (Gen-3, GBA).

Why Emerald needs its own preset (vs. reusing pokemon_rs)
--------------------------------------------------------
Ruby/Sapphire keep the SaveBlocks at *fixed* EWRAM addresses. Emerald
moved to runtime ``Alloc()``-ed SaveBlocks whose live base is stored in
a pointer, ``gSaveBlock1Ptr = 0x03005D8C`` (IWRAM), and additionally
randomises the SaveBlock location by up to ``SAVEBLOCK_MOVE_RANGE``
bytes as an anti-cheat measure. So the player's tile position cannot be
read from any static address — it must be **pointer-dereferenced**:
read the 4-byte pointer at ``gSaveBlock1Ptr``, then read ``pos`` at the
resulting EWRAM address. This is exactly the game-agnostic pointer-deref
primitive (:class:`Probe.pointer_at`), which the bridge resolves through
the console memory map.

Party data, by contrast, lives at *fixed* EWRAM symbols in Emerald
(``gPlayerParty``/``gPlayerPartyCount``), so those are direct reads.
This preset therefore exercises BOTH addressing paths.

Memory regions (agnostic addressing)
------------------------------------
Addresses are absolute GBA **bus** addresses. The bridge resolves each
through the core's memory map:

* ``gSaveBlock1Ptr`` (0x03005D8C) is in **IWRAM** (0x03000000) — exposed
  by the EmulatorJS nightly mgba core today, so the *pointer read*
  already works.
* The dereferenced SaveBlock and ``gPlayerParty`` live in **EWRAM**
  (0x02000000). The final reads light up once the core exposes EWRAM
  (a standard mGBA core maps SYSTEM_RAM→EWRAM; the nightly build maps it
  to IWRAM — see the memory-map export work). Until then these probes
  resolve to "no covering region" and are skipped, honestly, rather than
  reading the wrong bytes.

ROM revision
------------
Addresses target the **US v1.0/1.1 (BPEE)** build. Other regions/revs
shift the SaveBlock backing store and party symbols; ``variant`` warns
the bridge. Verify against a live EWRAM read before trusting values on a
non-US ROM.

Sources
-------
* https://github.com/pret/pokeemerald (definitive decompilation)
* https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Emerald/RAM_map
"""

from __future__ import annotations

from typing import Any

from augmentum.game_agent.probes.pokemon_rby import Probe, probe_entry

# ── Fixed globals (Emerald US, BPEE) ───────────────────────────────────
#
# gSaveBlock1Ptr : 0x03005D8C  (IWRAM) — u32 pointer to SaveBlock1 (EWRAM)
# gPlayerParty   : 0x020244EC  (EWRAM) — 100 bytes × 6 slots
# gPlayerPartyCount : 0x020244E9 (EWRAM) — u8
# gStringVar4    : 0x02021FC4  (EWRAM) — 1000-byte expanded-string buffer;
#                  msgbox dialogue is composed here before printing, so
#                  reading it = "what is the game saying right now".
# gDisplayedStringBattle : 0x02022E2C (EWRAM) — 300-byte battle message.
# All three verified against pret/pokeemerald symbols (2026-07).

_GSAVEBLOCK1_PTR       = 0x03005D8C
_GPLAYER_PARTY         = 0x020244EC
_GPLAYER_PARTY_COUNT   = 0x020244E9
_GSTRING_VAR4          = 0x02021FC4
_GDISPLAYED_STRING_BATTLE = 0x02022E2C

# gBackupMapLayout (IWRAM) — the live collision map:
#   struct BackupMapLayout { s32 width; s32 height; u16 *map; }
# ``map`` points into EWRAM; each u16 cell carries collision in bits
# 10-11 (0 = passable). Width/height include the 7-tile border pad on
# each side (+1), so grid index = (player_y + 7) * width + (player_x
# + 7). Verified against pret/pokeemerald sym + MAPGRID_COLLISION_MASK.
_GBACKUP_MAP_LAYOUT    = 0x03005DC0

# gMain.callback2 — THE screen discriminator. gMain lives at 0x030022C0
# (IWRAM); callback2 is the second MainCallback field (+0x04). Its value
# is the ROM address of the active screen's main callback, so mapping
# known callback addresses to names gives the agent ground truth for
# "which screen am I on" — the single fact it most often gets wrong from
# pixels alone. Addresses from pret's pokeemerald.sym (US BPEE); the
# bridge masks the Thumb bit before lookup. Unknown values render as
# hex — still a usable "screen changed" signal during transitions.
_GMAIN_CALLBACK2       = 0x030022C4
# Both the one-frame Init/loader callbacks AND the persistent runner
# callbacks each screen settles into (live-verified: the Birch intro
# sits in CB2_MainMenu 0x0802F6B0, which the Init-only table missed).
_SCREEN_CALLBACKS = {
    0x0802F6DC: "main_menu",
    0x0802F6B0: "main_menu",          # CB2_MainMenu runner (incl. Birch intro)
    0x08036760: "battle_starting",
    0x08038420: "battle",
    0x08085E5C: "overworld",
    0x08085E50: "overworld",          # CB2_OverworldBasic
    0x08085F58: "whiteout",
    0x08085FCC: "loading_map",        # CB2_LoadMap
    0x080860C8: "loading_map",        # CB2_ReturnToField
    0x08086230: "loading_save",
    0x080AA7A4: "title_screen",
    0x080B0AF8: "battle_ending",
    0x080BA4DC: "options_menu",
    0x080E2E04: "naming_screen",
    0x080E4F58: "naming_screen",      # CB2_NamingScreen runner
    0x081AAD5C: "bag_menu",           # CB2_BagMenuRun
    0x081B01E0: "party_menu",
    0x081B3828: "pokemon_summary",
}

# Offsets into SaveBlock1 (pret/pokeemerald ``struct SaveBlock1``):
#   struct Coords16 pos;         @ 0x00  (s16 x @ 0x00, s16 y @ 0x02)
#   struct WarpData location;    @ 0x04  (u8 mapGroup @ 0x04, u8 mapNum @ 0x05)
_SB1_POS_X             = 0x0000
_SB1_POS_Y             = 0x0002
_SB1_MAP_GROUP         = 0x0004
_SB1_MAP_NUM           = 0x0005

# Per-Pokémon (100-byte) unencrypted battle-stat shortcuts. Same layout
# as Ruby/Sapphire (shared Gen-3 struct). Species/moves/EVs live in the
# XOR-encrypted substructure and are not exposed here.
_POKE_LEVEL            = 0x0054   # u8
_POKE_HP_CURRENT       = 0x0056   # u16le
_POKE_HP_MAX           = 0x0058   # u16le
_PARTY_SLOT_STRIDE     = 100      # 0x64


def _party_slot_addr(slot_index: int, field_offset: int) -> int:
    """Absolute EWRAM address for a fixed-address party-slot field."""

    return _GPLAYER_PARTY + (slot_index * _PARTY_SLOT_STRIDE) + field_offset


PROBES: tuple[Probe, ...] = (
    # ── Player position + map (pointer-dereferenced through SaveBlock1) ─
    #
    # address is unused for deref probes (pass 0); the bridge reads the
    # u32 at pointer_at, adds pointer_offset, then reads length/type.
    Probe(
        name="player_x",
        address=0,
        length=2,
        type="s16le",
        pointer_at=_GSAVEBLOCK1_PTR,
        pointer_offset=_SB1_POS_X,
        description="Player tile X on the current map (via gSaveBlock1Ptr).",
    ),
    Probe(
        name="player_y",
        address=0,
        length=2,
        type="s16le",
        pointer_at=_GSAVEBLOCK1_PTR,
        pointer_offset=_SB1_POS_Y,
        description="Player tile Y on the current map (via gSaveBlock1Ptr).",
    ),
    Probe(
        name="map_group",
        address=0,
        length=1,
        type="u8",
        pointer_at=_GSAVEBLOCK1_PTR,
        pointer_offset=_SB1_MAP_GROUP,
        description="Map group id (pair with map_num to name a location).",
    ),
    Probe(
        name="map_num",
        address=0,
        length=1,
        type="u8",
        pointer_at=_GSAVEBLOCK1_PTR,
        pointer_offset=_SB1_MAP_NUM,
        description="Map number within group. (map_group, map_num) names the map.",
    ),
    # ── Party header (fixed EWRAM address) ─────────────────────────────
    Probe(
        name="party_count",
        address=_GPLAYER_PARTY_COUNT,
        length=1,
        type="u8",
        description="Number of Pokémon currently in party (0-6).",
    ),
    # ── Lead Pokémon (slot 0, fixed EWRAM address) ─────────────────────
    Probe(
        name="p1_level",
        address=_party_slot_addr(0, _POKE_LEVEL),
        length=1,
        type="u8",
        description="Lead Pokémon current level.",
    ),
    Probe(
        name="p1_hp_current",
        address=_party_slot_addr(0, _POKE_HP_CURRENT),
        length=2,
        type="u16le",
        description="Lead Pokémon current HP.",
    ),
    Probe(
        name="p1_hp_max",
        address=_party_slot_addr(0, _POKE_HP_MAX),
        length=2,
        type="u16le",
        description="Lead Pokémon max HP.",
    ),
    # ── Screen discriminator ────────────────────────────────────────────
    Probe(
        name="screen",
        address=_GMAIN_CALLBACK2,
        length=4,
        type="u32le",
        value_labels=_SCREEN_CALLBACKS,
        description=(
            "Which screen the game is on RIGHT NOW (overworld / battle / "
            "naming_screen / menu / title...). Trust this over the frame."
        ),
    ),
    # ── Live dialogue ("translate game text into world lore") ──────────
    #
    # The game's own words are the strongest tutorial signal there is —
    # Emerald's intro literally explains the controls. dialog_text is
    # the expanded msgbox buffer (gStringVar4); battle_text is the
    # battle-message buffer. Both decode through the bridge's gen3
    # charmap into readable prose; the orchestrator accumulates changed
    # lines as DIALOGUE_LORE for the planner.
    Probe(
        name="dialog_text",
        address=_GSTRING_VAR4,
        length=160,
        type="text",
        charmap="gen3",
        description="Current overworld dialog/message text, decoded.",
    ),
    Probe(
        name="battle_text",
        address=_GDISPLAYED_STRING_BATTLE,
        length=120,
        type="text",
        charmap="gen3",
        description="Current battle message text, decoded.",
    ),
    # ── Walkability window (feeds the navigate_to quickaction) ─────────
    #
    # Hidden: the raw rows never enter the prompt; the navigation
    # compiler BFS-es over them and the planner overlay gets a rendered
    # view. 15×15 centred on the player covers a full screen + margin.
    Probe(
        name="walk_grid",
        address=0,
        length=0,
        type="grid",
        hidden=True,
        grid={
            "header_at": _GBACKUP_MAP_LAYOUT,
            "width_offset": 0,
            "height_offset": 4,
            "map_ptr_offset": 8,
            "cell_bytes": 2,
            "collision_shift": 10,
            "collision_mask": 3,
            "anchor_x": "player_x",
            "anchor_y": "player_y",
            "border": 7,
            "window": 15,
        },
        description=(
            "Collision map window around the player ('.' walkable, "
            "'#' blocked), map-tile coordinates matching player_x/y."
        ),
    ),
)


def to_dict() -> dict[str, Any]:
    """Return the preset as a JSON-serialisable dict (bridge wire shape).

    ``memory_region``/``region_base_address`` are retained only as the
    legacy single-region fallback; the bridge prefers bus-address
    resolution through the console memory map, which is what makes the
    pointer-deref position probes work across IWRAM (pointer) + EWRAM
    (target).
    """

    return {
        "name": "pokemon_emerald",
        "system": "gba",
        "variant": "us_bpee",
        "memory_region": "system_ram",
        "region_base_address": 0x02000000,
        "probes": [probe_entry(p) for p in PROBES],
    }
