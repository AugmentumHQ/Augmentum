"""Pokémon Gold / Silver / Crystal RAM probe preset (Gen-2, GB/GBC).

All three Gen-2 mainline games share one engine (pret/pokecrystal is the
authoritative decompilation; Gold and Silver share virtually identical
RAM layouts with Crystal aside from a handful of unused fields and the
PokéGear / Mobile Adapter holes). The addresses below are taken from the
pret/pokecrystal symbol map and are stable across the three games for
every field we probe.

Memory map references
---------------------
* https://github.com/pret/pokecrystal (definitive, includes wram.asm)
* https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Crystal/RAM_map
* https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Gold_and_Silver/RAM_map

Memory region
-------------
All probed addresses live in WRAM (``0xC000-0xDFFF``). The browser bridge
reads from libretro's ``RETRO_MEMORY_SYSTEM_RAM`` region; on Game Boy /
Game Boy Color that region equals WRAM, so the offset is
``address - 0xC000``.

Decoder vocabulary
------------------
Inherits the decoder names defined in :mod:`pokemon_rby`:
``u8`` / ``u16be`` / ``u16le`` / ``bcd3`` / ``bitfield8`` / ``raw``.
Gen-2 keeps Gen-1's big-endian Pokémon stat fields (the engine still
runs on the LR35902 / SM83, same as Gen-1).

Scope of v1
-----------
The top-level WRAM globals (position, map, party header, money, badges,
battle state) are shipped directly. The per-PartyMon 48-byte struct's
deeper offsets (level, HP, status) are intentionally deferred to a v2
revision after live verification on a running Crystal ROM. The lead
species ``p1_species`` is exposed (top of slot 0 == ``wPartyMons``);
deeper stat reads need offset confirmation against a save file we can
inspect alongside the running game.

ROM revision caveats
--------------------
Gold and Silver share addresses across regions (US/EU/JP); Crystal moved
a handful of unused / PokéGear-related fields but none of the ones below.
Tagging ``"variant": "gsc_common"`` lets a future version-detect probe
flag a mismatched build.
"""

from __future__ import annotations

from typing import Any

from augmentum.game_agent.probes.pokemon_rby import Probe, probe_entry

# ── Top-level WRAM globals (pokecrystal/wram.asm) ─────────────────────
#
# All addresses below are taken from the pret/pokecrystal symbol map.
# Values are decimal-equivalent to the hex shown in the trailing comment
# so JSON serialisation (which emits plain ints) is auditable against
# the decomp.

_WMAP_GROUP        = 0xDCB5   # current map's MapGroup id
_WMAP_NUMBER       = 0xDCB6   # current map's MapNumber id (within group)
_WYCOORD           = 0xDCB7   # player Y in tiles on the current map
_WXCOORD           = 0xDCB8   # player X in tiles on the current map

_WPARTY_COUNT      = 0xDCD7   # 0..6 Pokémon currently in the party
_WPARTY_SPECIES    = 0xDCD8   # 6 bytes, terminated with 0xFF after last
_WPARTY_MONS       = 0xDCDF   # first PartyMon slot starts here

_WBATTLE_MODE      = 0xD22D   # 0 = overworld, 1 = wild, 2 = trainer
_WENEMY_SPECIES    = 0xD0EC   # opponent species id while in battle

_WMONEY            = 0xD84E   # 3-byte BCD, 6 digits, 0..999999
_WJOHTO_BADGES     = 0xD857   # 1 byte bitfield, LSB = Zephyr
_WKANTO_BADGES     = 0xD858   # 1 byte bitfield, LSB = Boulder

# Per-PartyMon offset for the species byte (slot 0 == ``wPartyMons``).
# Used only to label the lead Pokémon at slot 0; deeper stats deferred
# until we can verify offsets live against a known save.
_POKE_SPECIES_OFFSET = 0x00


_JOHTO_BADGE_LABELS = (
    "zephyr",     # Falkner
    "hive",       # Bugsy
    "plain",      # Whitney
    "fog",        # Morty
    "storm",      # Chuck
    "mineral",    # Jasmine
    "glacier",    # Pryce
    "rising",     # Clair
)


_KANTO_BADGE_LABELS = (
    "boulder",    # Brock
    "cascade",    # Misty
    "thunder",    # Lt. Surge
    "rainbow",    # Erika
    "soul",       # Koga
    "marsh",      # Sabrina
    "volcano",    # Blaine
    "earth",      # Blue
)


PROBES: tuple[Probe, ...] = (
    # ── World / location ──
    Probe(
        name="map_group",
        address=_WMAP_GROUP,
        length=1,
        type="u8",
        description="MapGroup id of the player's current map.",
    ),
    Probe(
        name="map_number",
        address=_WMAP_NUMBER,
        length=1,
        type="u8",
        description="MapNumber within MapGroup; together they identify the map.",
    ),
    Probe(
        name="player_y",
        address=_WYCOORD,
        length=1,
        type="u8",
        description="Player Y position in tiles on the current map.",
    ),
    Probe(
        name="player_x",
        address=_WXCOORD,
        length=1,
        type="u8",
        description="Player X position in tiles on the current map.",
    ),
    # ── Party header ──
    Probe(
        name="party_count",
        address=_WPARTY_COUNT,
        length=1,
        type="u8",
        description="Number of Pokémon currently in the party (0..6).",
    ),
    Probe(
        name="party_species",
        address=_WPARTY_SPECIES,
        length=6,
        type="raw",
        description="Up to 6 species ids; 0xFF padding after the last mon.",
    ),
    Probe(
        name="p1_species",
        address=_WPARTY_MONS + _POKE_SPECIES_OFFSET,
        length=1,
        type="u8",
        description="Lead Pokémon species id (matches party_species[0]).",
    ),
    # ── Battle state ──
    Probe(
        name="battle_mode",
        address=_WBATTLE_MODE,
        length=1,
        type="u8",
        description="0 = overworld, 1 = wild battle, 2 = trainer battle.",
    ),
    Probe(
        name="opponent_species",
        address=_WENEMY_SPECIES,
        length=1,
        type="u8",
        description="Opponent species id; meaningful only when battle_mode != 0.",
    ),
    # ── Progression ──
    Probe(
        name="money",
        address=_WMONEY,
        length=3,
        type="bcd3",
        description="Wallet contents, 0..999999 as 6 BCD digits.",
    ),
    Probe(
        name="johto_badges",
        address=_WJOHTO_BADGES,
        length=1,
        type="bitfield8",
        labels=_JOHTO_BADGE_LABELS,
        description="Johto gym badges earned (LSB = Zephyr).",
    ),
    Probe(
        name="kanto_badges",
        address=_WKANTO_BADGES,
        length=1,
        type="bitfield8",
        labels=_KANTO_BADGE_LABELS,
        description="Kanto gym badges earned (LSB = Boulder). Crystal/GS only.",
    ),
)


def to_dict() -> dict[str, Any]:
    """Return the preset as a JSON-serialisable dict.

    Used by the browser bridge: the shim fetches this on session start
    and applies it on every libretro tick. The wire shape matches the
    decoder schema in the JS shim.

    Before:
    - ``PROBES[0]`` -> ``Probe(name="map_group", address=0xDCB5, length=1, type="u8", ...)``

    After:
    - ``to_dict()["probes"][0]`` ->
      ``{"name": "map_group", "address": 56501, "length": 1, "type": "u8", "description": "..."}``
    """

    return {
        "name": "pokemon_gsc",
        "system": "gbc",
        "memory_region": "system_ram",
        "region_base_address": 0xC000,
        "variant": "gsc_common",
        "probes": [probe_entry(p) for p in PROBES],
    }
