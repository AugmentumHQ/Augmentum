"""Pokemon Red / Blue / Yellow RAM probe preset.

Memory map references
---------------------
The addresses below are stable across all three games (RBY share a
common engine; the meaningful Yellow-only deltas are sound and event
flags, not the fields we probe). Source for the addresses:

* https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map
* https://github.com/pret/pokered/blob/master/wram.asm (definitive)

All probed addresses live in WRAM (0xC000-0xDFFF on the Game Boy
memory map). The browser bridge reads from libretro's
``RETRO_MEMORY_SYSTEM_RAM`` region; on Game Boy this region is the
WRAM, so the offset within the region equals ``address - 0xC000``.

Decoder vocabulary
------------------
``u8``         Unsigned single byte.
``u16be``      Big-endian 16-bit unsigned integer. Pokemon stats and
               HP are stored big-endian; this is intrinsic to the GB
               game engine, not a quirk of the emulator.
``bcd3``       3-byte Binary-Coded Decimal. Used for money, which can
               hold 0-999,999 in 6 nibbles across 3 bytes.
``bitfield8``  Single byte where each bit is a labelled flag. Badges
               and event flags are stored this way.

The JS bridge ships its own decoder for each ``type`` value; if you
add a new decoder name here you must also add it on the JS side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

DecoderType = Literal[
    "u8", "u16be", "u16le", "s16le", "s16be", "u32le", "bcd3", "bitfield8", "raw",
    "text",
    "grid",
]


@dataclass(frozen=True)
class Probe:
    """One memory location to read each tick.

    ``address`` is the absolute **console-bus** address (GB 0xC000-range,
    GBA 0x02.. / 0x03.., …). The bridge resolves it through the core's
    memory map to a (region, offset) before reading — no per-preset
    region assumption. ``length`` is in bytes and must match the decoder.

    **Pointer dereference** (``pointer_at`` set): instead of reading
    ``address`` directly, the bridge reads a ``pointer_size``-byte
    little-endian pointer at bus address ``pointer_at``, adds
    ``pointer_offset``, and reads ``length``/``type`` at the resulting
    bus address. This is the game-agnostic primitive for dynamically
    relocated structures (e.g. Emerald's ``gSaveBlock1Ptr``). When
    ``pointer_at`` is set, ``address`` is unused (pass 0).
    """

    name: str
    address: int
    length: int
    type: DecoderType
    description: str = ""
    # Only used by ``bitfield8``: ordered labels for each bit, LSB
    # first. The decoder emits ``{label: bool}`` for every entry.
    labels: tuple[str, ...] | None = None
    # Pointer-deref (all None/0 for a plain direct read).
    pointer_at: int | None = None
    pointer_size: int = 4
    pointer_offset: int = 0
    # Only used by ``text``: which character table decodes the buffer.
    # The JS bridge owns the tables ("gen3", "ascii"); adding a new
    # charmap there unlocks it here.
    charmap: str | None = None
    # Optional value→name map for integer decoders (u8/u16/u32): the
    # bridge emits the NAME instead of the raw number (unknown values
    # fall back to hex). Function-pointer values get their Thumb bit
    # masked before lookup. This is how a raw state word (e.g. Gen-3's
    # gMain.callback2) becomes a legible ``screen: "overworld"`` field.
    value_labels: dict[int, str] | None = None
    # Only used by ``grid``: a collision-map window around the player.
    # The bridge reads a {width, height, map-pointer} header at
    # ``header_at``, then samples a ``window``×``window`` cell block
    # centred on the (anchor_x, anchor_y) probes' values (+``border``
    # padding offset into the stored grid), decoding each cell's
    # collision bits via ``collision_shift``/``collision_mask``. Emits
    # {"x0","y0","rows"} with '.'=walkable '#'=blocked '?'=outside, in
    # the same coordinate space as the anchor probes.
    grid: dict[str, Any] | None = None
    # Hidden probes feed the world blackboard + quickaction compilers
    # but are excluded from prompt overlays/deltas (structural data the
    # model shouldn't burn tokens reading raw).
    hidden: bool = False


# Address constants. Centralised so a careful reader can spot-check
# them against pret/pokered without hunting through the probe list.
# All values are decimal-equivalent to hex in the comment to make
# JSON serialisation (which renders these as plain ints) auditable.

_WPLAYER_NAME      = 0xD158   # 11 bytes terminated with 0x50
_WPARTY_COUNT      = 0xD163   # 1 byte
_WPARTY_SPECIES    = 0xD164   # 6 bytes; 0xFF after the last party member
_WPARTY_MONS       = 0xD16B   # First party slot starts here; 44 bytes each
_WMONEY            = 0xD347   # 3 bytes BCD, 6 digits
_WBADGES           = 0xD356   # 1 byte, bitfield
_WCURRENT_MAP      = 0xD35E   # 1 byte map id
_WYCOORD           = 0xD361   # 1 byte (tile)
_WXCOORD           = 0xD362   # 1 byte (tile)
_WIS_IN_BATTLE     = 0xD057   # 0 = no, 1 = wild, 2 = trainer
_WCUR_OPPONENT     = 0xCFD8   # 1 byte, species id when in battle
_WTEXT_BOX_ID      = 0xD125   # 1 byte; non-zero when a textbox is open
_WJOYINPUT_LAST    = 0xFF8B   # 1 byte; last frame joypad bits (debug)


# Per-Pokemon offsets within the 44-byte party slot. Used to derive
# explicit addresses below; not exported.
_OFFSET_SPECIES    = 0x00
_OFFSET_HP_CURRENT = 0x01     # u16be
_OFFSET_LEVEL_RAW  = 0x03     # u8  (encounter level; resets after evolution)
_OFFSET_STATUS     = 0x04     # bitfield
_OFFSET_TYPE_1     = 0x05
_OFFSET_TYPE_2     = 0x06
_OFFSET_CATCH_RATE = 0x07
_OFFSET_MOVES      = 0x08     # 4 bytes
_OFFSET_LEVEL_CUR  = 0x21     # u8  (current displayed level; canonical)
_OFFSET_HP_MAX     = 0x22     # u16be

_PARTY_SLOT_STRIDE = 0x2C     # 44 bytes


def _slot_addr(slot_index: int, field_offset: int) -> int:
    """Compose an absolute WRAM address for a party-slot field.

    @example: ``_slot_addr(0, _OFFSET_HP_CURRENT)`` returns the
    current HP of the lead Pokemon at 0xD16C.
    """

    return _WPARTY_MONS + (slot_index * _PARTY_SLOT_STRIDE) + field_offset


# Badge labels in obtain order (gym leader order). Aligns with the
# in-game badge case rendering.
_BADGE_LABELS = (
    "boulder",   # Brock
    "cascade",   # Misty
    "thunder",   # Lt. Surge
    "rainbow",   # Erika
    "soul",      # Koga
    "marsh",     # Sabrina
    "volcano",   # Blaine
    "earth",     # Giovanni
)


PROBES: tuple[Probe, ...] = (
    # ── World / location ──
    Probe(
        name="current_map",
        address=_WCURRENT_MAP,
        length=1,
        type="u8",
        description="Current map id (0-247). 0 == Pallet Town interior.",
    ),
    Probe(
        name="y_tile",
        address=_WYCOORD,
        length=1,
        type="u8",
        description="Y position in tiles on the current map.",
    ),
    Probe(
        name="x_tile",
        address=_WXCOORD,
        length=1,
        type="u8",
        description="X position in tiles on the current map.",
    ),
    Probe(
        name="textbox_open",
        address=_WTEXT_BOX_ID,
        length=1,
        type="u8",
        description="Non-zero when a textbox is on screen (advance with A).",
    ),
    # ── Party header ──
    Probe(
        name="party_count",
        address=_WPARTY_COUNT,
        length=1,
        type="u8",
        description="Number of Pokemon currently in party (0-6).",
    ),
    Probe(
        name="party_species",
        address=_WPARTY_SPECIES,
        length=6,
        type="raw",
        description="Up to 6 species ids; 0xFF padding after last party member.",
    ),
    # ── Lead Pokemon (slot 0) ──
    Probe(
        name="p1_species",
        address=_slot_addr(0, _OFFSET_SPECIES),
        length=1,
        type="u8",
        description="Lead Pokemon species id (internal id, not Pokedex id).",
    ),
    Probe(
        name="p1_hp_current",
        address=_slot_addr(0, _OFFSET_HP_CURRENT),
        length=2,
        type="u16be",
        description="Lead Pokemon current HP.",
    ),
    Probe(
        name="p1_hp_max",
        address=_slot_addr(0, _OFFSET_HP_MAX),
        length=2,
        type="u16be",
        description="Lead Pokemon max HP.",
    ),
    Probe(
        name="p1_level",
        address=_slot_addr(0, _OFFSET_LEVEL_CUR),
        length=1,
        type="u8",
        description="Lead Pokemon current displayed level.",
    ),
    Probe(
        name="p1_status",
        address=_slot_addr(0, _OFFSET_STATUS),
        length=1,
        type="bitfield8",
        labels=(
            "sleep_count_0",
            "sleep_count_1",
            "sleep_count_2",
            "poisoned",
            "burned",
            "frozen",
            "paralysed",
            "_reserved",
        ),
        description="Status condition bitfield (sleep is the 3 LSBs).",
    ),
    # ── Battle state ──
    Probe(
        name="in_battle",
        address=_WIS_IN_BATTLE,
        length=1,
        type="u8",
        description="0 = overworld, 1 = wild battle, 2 = trainer battle.",
    ),
    Probe(
        name="opponent_species",
        address=_WCUR_OPPONENT,
        length=1,
        type="u8",
        description="Opponent species id; valid only while in_battle != 0.",
    ),
    # ── Progression ──
    Probe(
        name="badges",
        address=_WBADGES,
        length=1,
        type="bitfield8",
        labels=_BADGE_LABELS,
        description="Earned gym badges as a bitfield (LSB == Boulder).",
    ),
    Probe(
        name="money",
        address=_WMONEY,
        length=3,
        type="bcd3",
        description="Wallet contents, 0-999999 stored as 6 BCD digits.",
    ),
)


def to_dict() -> dict[str, Any]:
    """Return the preset as a JSON-serialisable dict.

    Used by the browser bridge: the shim fetches this and applies it
    on every libretro tick. The wire shape matches the JS bridge's
    expected schema in ``ui/scripts/agent/emulatorjs-bridge.js``.

    Before:
    - ``PROBES[0]`` -> ``Probe(name="current_map", address=0xD35E, length=1, type="u8", ...)``

    After:
    - ``to_dict()["probes"][0]`` ->
      ``{"name": "current_map", "address": 54110, "length": 1, "type": "u8", "description": "..."}``
    """

    return {
        "name": "pokemon_rby",
        "system": "gb",
        "memory_region": "system_ram",
        "region_base_address": 0xC000,
        "probes": [probe_entry(p) for p in PROBES],
    }


def probe_entry(p: Probe) -> dict[str, Any]:
    """Serialise one :class:`Probe` to the bridge wire schema.

    Shared by every preset's ``to_dict`` so the pointer-deref block is
    emitted identically. A direct probe carries ``address``; a deref
    probe additionally carries a ``pointer`` block and the bridge reads
    ``address`` only if ``pointer`` is absent.
    """

    entry: dict[str, Any] = {
        "name": p.name,
        "address": p.address,
        "length": p.length,
        "type": p.type,
    }
    if p.description:
        entry["description"] = p.description
    if p.labels is not None:
        entry["labels"] = list(p.labels)
    if p.pointer_at is not None:
        entry["pointer"] = {
            "at": p.pointer_at,
            "size": p.pointer_size,
            "offset": p.pointer_offset,
        }
    if p.charmap:
        entry["charmap"] = p.charmap
    if p.value_labels:
        # JSON object keys are strings; the JS side looks up by decimal
        # string after masking the Thumb bit.
        entry["value_labels"] = {str(k): v for k, v in p.value_labels.items()}
    if p.grid is not None:
        entry["grid"] = dict(p.grid)
    if p.hidden:
        entry["hidden"] = True
    return entry
