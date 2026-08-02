"""Pokémon Ruby / Sapphire RAM probe preset (Gen-3, GBA).

Memory map references
---------------------
The addresses below target the **US v1.0** ROM revisions of Ruby and
Sapphire. The two games share the same engine; the addresses below
are byte-identical between them. Sources:

* https://github.com/pret/pokeruby (definitive decompilation)
* https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Ruby_and_Sapphire/RAM_map

Why Ruby/Sapphire and not Emerald
---------------------------------
In Ruby/Sapphire the SaveBlock structures are at fixed addresses;
Emerald moved to runtime ``Alloc()``-allocated SaveBlocks with the
pointer kept at ``gSaveBlock1Ptr = 0x03005D8C``. Pointer-chasing
probes require a different bridge primitive than the static address
table this module emits, so Emerald gets its own preset once that
plumbing exists. Adding partial Emerald support here would silently
read garbage on Emerald ROMs.

ROM revision caveats
--------------------
GameFAQ rom dumps differ across regions and minor revisions:

* US v1.0 (Sapphire 1.0): the addresses below.
* US v1.1 / European builds / Japanese: SaveBlock locations shift
  by ~0x100 bytes on some builds. We expose ``"variant": "us_v1_0"``
  in the dict so the bridge can refuse to read or warn loudly when a
  future version-detect probe fires on a mismatched ROM.

Memory region
-------------
All addresses below live in **EWRAM** (``0x02000000-0x0203FFFF``),
which mGBA exposes as libretro's ``RETRO_MEMORY_SYSTEM_RAM`` for
GBA cores. The browser shim subtracts ``region_base_address`` from
each probe's absolute address to get a region-relative offset.

Decoder vocabulary
------------------
Inherits the decoder names defined in :mod:`pokemon_rby`:
``u8`` / ``u16be`` / ``u16le`` / ``s16le`` / ``raw``. Pokémon Gen-3
stores multi-byte fields little-endian (ARM7TDMI), unlike Gen-1's
big-endian Game Boy storage; battle stats (HP, level, max-HP) use
``u16le`` here.
"""

from __future__ import annotations

from typing import Any

from augmentum.game_agent.probes.pokemon_rby import Probe, probe_entry

# ── Fixed SaveBlock / global addresses (Ruby/Sapphire US v1.0) ─────────
#
# SaveBlock1 (player + world): 0x02025734, 0x9C0 bytes
# SaveBlock2 (config + player meta): 0x02024EA4, 0xF24 bytes
# gPlayerParty (party Pokémon): 0x02024284, 100 bytes × 6 slots
# gPlayerPartyCount: 0x02024029, u8
#
# These come from the pret/pokeruby symbol map for the US v1.0 build.

_SAVEBLOCK1_BASE        = 0x02025734
_GPLAYER_PARTY          = 0x02024284
_GPLAYER_PARTY_COUNT    = 0x02024029

# Offsets into SaveBlock1. See ``struct SaveBlock1`` in
# pret/pokeruby/include/global.h.
_SB1_POS_X              = 0x0000   # s16le, tile coordinate
_SB1_POS_Y              = 0x0002   # s16le, tile coordinate
_SB1_MAP_GROUP          = 0x0004   # u8
_SB1_MAP_NUM            = 0x0005   # u8

# Offsets within one PartyPokemon (100-byte) entry. See
# ``struct Pokemon`` in pokeruby/include/pokemon.h. Battle stats
# (level/HP/maxHP) are unencrypted "shortcut" fields the game
# maintains; the *encrypted* substructure (species, moves, EVs)
# requires XOR with personality+OTID and isn't exposed here.
_POKE_LEVEL             = 0x0054   # u8
_POKE_HP_CURRENT        = 0x0056   # u16le
_POKE_HP_MAX            = 0x0058   # u16le

_PARTY_SLOT_STRIDE      = 100      # 0x64 bytes per slot, 6 slots


def _party_slot_addr(slot_index: int, field_offset: int) -> int:
    """Compose an absolute EWRAM address for a party-slot field.

    @example: ``_party_slot_addr(0, _POKE_HP_CURRENT)`` returns the
    current HP of the lead Pokémon at 0x020242DA.
    """

    return _GPLAYER_PARTY + (slot_index * _PARTY_SLOT_STRIDE) + field_offset


PROBES: tuple[Probe, ...] = (
    # ── Player position + map ─────────────────────────────────────────
    Probe(
        name="player_x",
        address=_SAVEBLOCK1_BASE + _SB1_POS_X,
        length=2,
        type="u16le",  # technically s16le, but tile coords are >=0 in normal play
        description="Player tile X on the current map.",
    ),
    Probe(
        name="player_y",
        address=_SAVEBLOCK1_BASE + _SB1_POS_Y,
        length=2,
        type="u16le",
        description="Player tile Y on the current map.",
    ),
    Probe(
        name="map_group",
        address=_SAVEBLOCK1_BASE + _SB1_MAP_GROUP,
        length=1,
        type="u8",
        description="Map group id (region grouping; pair with map_num).",
    ),
    Probe(
        name="map_num",
        address=_SAVEBLOCK1_BASE + _SB1_MAP_NUM,
        length=1,
        type="u8",
        description="Map number within group. (map_group, map_num) uniquely names a location.",
    ),
    # ── Party header ──────────────────────────────────────────────────
    Probe(
        name="party_count",
        address=_GPLAYER_PARTY_COUNT,
        length=1,
        type="u8",
        description="Number of Pokémon currently in party (0-6).",
    ),
    # ── Lead Pokémon (slot 0) ─────────────────────────────────────────
    #
    # Species / moves require decrypting the data box with the PID+OTID
    # XOR key — not exposed in this preset. Level + HP are unencrypted
    # battle-stat shortcuts; safe to read directly.
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
)


def to_dict() -> dict[str, Any]:
    """Return the preset as a JSON-serialisable dict.

    The wire shape mirrors :func:`pokemon_rby.to_dict` so the browser
    bridge ingests it identically. The ``variant`` field warns
    operators that addresses target US v1.0 specifically; mismatched
    ROMs will read plausible-looking garbage from adjacent fields.

    Before:
    - ``PROBES[0]`` -> ``Probe(name="player_x", address=0x02025734, ...)``
    After:
    - ``to_dict()["probes"][0]`` ->
      ``{"name": "player_x", "address": 33625396, "length": 2, "type": "u16le", ...}``
    """

    return {
        "name": "pokemon_rs",
        "system": "gba",
        "variant": "us_v1_0",
        "memory_region": "system_ram",
        "region_base_address": 0x02000000,
        "probes": [probe_entry(p) for p in PROBES],
    }
