"""RAM probe preset structural tests.

These do not depend on a real emulator; they verify that the Python
probe table is internally consistent and that ``to_dict()`` produces
a wire format the JS bridge can consume without surprise.
"""

from __future__ import annotations

from augmentum.game_agent.probes.pokemon_rby import PROBES, to_dict

_VALID_DECODER_TYPES = {
    "u8", "u16le", "u16be", "s16le", "s16be", "u32le", "bcd3", "bitfield8", "raw",
    "text",
    "grid",
}


def test_pokemon_rby_probes_have_valid_decoders() -> None:
    """@example: every probe declares a decoder name the JS shim implements."""

    for probe in PROBES:
        assert probe.type in _VALID_DECODER_TYPES, (
            f"probe {probe.name!r} declares unknown decoder type {probe.type!r}"
        )


def test_pokemon_rby_probe_lengths_match_types() -> None:
    """@example: declared byte length agrees with the decoder's expected width.

    ROOT CAUSE:
      An earlier version had ``p1_hp_max`` typed ``u16be`` with
      ``length=1`` -- the decoder would have silently truncated the
      hi byte and reported half-correct HP. Asserting the length
      catches every such mismatch at import time.
    """

    expected_length = {
        "u8": 1,
        "u16le": 2,
        "u16be": 2,
        "bcd3": 3,
        "bitfield8": 1,
    }
    for probe in PROBES:
        if probe.type in expected_length:
            assert probe.length == expected_length[probe.type], (
                f"probe {probe.name!r}: type {probe.type!r} expects "
                f"length {expected_length[probe.type]}, got {probe.length}"
            )


def test_pokemon_rby_addresses_live_in_wram() -> None:
    """@example: every probed address is inside the Game Boy WRAM range.

    WRAM is 0xC000 - 0xDFFF on Game Boy; HRAM (0xFF80-0xFFFE) is
    intentionally allowed too because some debug-style probes (like
    last-joypad-input) live there.
    """

    for probe in PROBES:
        a = probe.address
        in_wram = 0xC000 <= a <= 0xDFFF
        in_hram = 0xFF80 <= a <= 0xFFFE
        assert in_wram or in_hram, (
            f"probe {probe.name!r} at {a:#x} is outside WRAM/HRAM"
        )


def test_pokemon_rby_bitfield_labels_match_byte_width() -> None:
    """@example: a bitfield8 probe never declares more than 8 labels."""

    for probe in PROBES:
        if probe.type == "bitfield8":
            assert probe.labels is not None
            assert len(probe.labels) <= 8, (
                f"probe {probe.name!r} declares {len(probe.labels)} labels "
                "for a single-byte bitfield"
            )


def test_to_dict_roundtrips_essential_fields() -> None:
    """@example: to_dict produces the wire shape the JS bridge expects."""

    blob = to_dict()
    assert blob["name"] == "pokemon_rby"
    assert blob["system"] == "gb"
    assert blob["memory_region"] == "system_ram"
    assert blob["region_base_address"] == 0xC000
    assert isinstance(blob["probes"], list)
    assert len(blob["probes"]) == len(PROBES)
    sample = next(p for p in blob["probes"] if p["name"] == "current_map")
    assert sample["address"] == 0xD35E
    assert sample["length"] == 1
    assert sample["type"] == "u8"


def test_to_dict_emits_labels_only_when_present() -> None:
    """@example: probes without labels do not pollute the JSON shape."""

    blob = to_dict()
    by_name = {p["name"]: p for p in blob["probes"]}
    # current_map has no labels:
    assert "labels" not in by_name["current_map"]
    # badges has labels:
    assert "labels" in by_name["badges"]
    assert by_name["badges"]["labels"][0] == "boulder"


# ── Pokémon Ruby/Sapphire (Gen-3, GBA) ─────────────────────────────────

from augmentum.game_agent.probes.pokemon_rs import (  # noqa: E402
    PROBES as RS_PROBES,
)
from augmentum.game_agent.probes.pokemon_rs import (  # noqa: E402
    to_dict as rs_to_dict,
)


def test_pokemon_rs_probes_have_valid_decoders() -> None:
    """@example: every Gen-3 probe declares a decoder name the JS shim implements."""

    for probe in RS_PROBES:
        assert probe.type in _VALID_DECODER_TYPES, (
            f"probe {probe.name!r} declares unknown decoder type {probe.type!r}"
        )


def test_pokemon_rs_probe_lengths_match_types() -> None:
    """@example: declared byte length agrees with the Gen-3 decoder width.

    ROOT CAUSE:
      Gen-3 stores multi-byte fields little-endian (ARM7TDMI); a
      mistyped u16le with length=1 would silently drop the high byte
      and report half-correct HP. This guards the address table at
      import time before any session sees a bad reading.
    """

    expected_length = {
        "u8": 1,
        "u16le": 2,
        "u16be": 2,
        "bcd3": 3,
        "bitfield8": 1,
    }
    for probe in RS_PROBES:
        if probe.type in expected_length:
            assert probe.length == expected_length[probe.type], (
                f"probe {probe.name!r}: type {probe.type!r} expects "
                f"length {expected_length[probe.type]}, got {probe.length}"
            )


def test_pokemon_rs_addresses_live_in_gba_ewram() -> None:
    """@example: every Gen-3 probe sits inside GBA EWRAM (0x02000000-0x0203FFFF).

    ROOT CAUSE:
      mGBA exposes ``RETRO_MEMORY_SYSTEM_RAM`` as EWRAM only; an IWRAM
      address would silently read out-of-bounds garbage. The bridge
      validates ``offset + length <= region_size`` per tick, but
      catching it here surfaces the bug at the address-table level
      instead of after a session is already running.
    """

    for probe in RS_PROBES:
        a = probe.address
        in_ewram = 0x02000000 <= a < 0x02040000
        assert in_ewram, (
            f"probe {probe.name!r} at {a:#x} is outside GBA EWRAM"
        )


def test_pokemon_rs_to_dict_roundtrips_essential_fields() -> None:
    """@example: to_dict produces the wire shape the JS bridge expects."""

    blob = rs_to_dict()
    assert blob["name"] == "pokemon_rs"
    assert blob["system"] == "gba"
    assert blob["variant"] == "us_v1_0"
    assert blob["memory_region"] == "system_ram"
    assert blob["region_base_address"] == 0x02000000
    assert isinstance(blob["probes"], list)
    assert len(blob["probes"]) == len(RS_PROBES)
    sample = next(p for p in blob["probes"] if p["name"] == "player_x")
    assert sample["address"] == 0x02025734
    assert sample["length"] == 2
    assert sample["type"] == "u16le"


def test_pokemon_rs_party_slot_arithmetic() -> None:
    """@example: lead-Pokémon offsets compose correctly into absolute EWRAM addrs.

    The slot stride is 100 bytes (0x64), starting at gPlayerParty
    (0x02024284). Lead Pokémon current HP lives at offset 0x56 within
    the slot, so its absolute address is 0x020242DA.
    """

    blob = rs_to_dict()
    by_name = {p["name"]: p for p in blob["probes"]}
    assert by_name["p1_hp_current"]["address"] == 0x020242DA
    assert by_name["p1_hp_max"]["address"] == 0x020242DC
    assert by_name["p1_level"]["address"] == 0x020242D8


# ── Pokémon Gold/Silver/Crystal (Gen-2) ───────────────────────────────


from augmentum.game_agent.probes.pokemon_gsc import (  # noqa: E402
    PROBES as GSC_PROBES,
)
from augmentum.game_agent.probes.pokemon_gsc import (  # noqa: E402
    to_dict as gsc_to_dict,
)


def test_pokemon_gsc_probes_have_valid_decoders() -> None:
    """@example: every Gen-2 probe declares a decoder name the JS shim implements."""

    for probe in GSC_PROBES:
        assert probe.type in _VALID_DECODER_TYPES, (
            f"probe {probe.name!r} declares unknown decoder type {probe.type!r}"
        )


def test_pokemon_gsc_addresses_live_in_gbc_wram() -> None:
    """@example: every Gen-2 probed address is inside the GB/GBC WRAM range.

    Gen-2 still uses LR35902-class memory layout; WRAM occupies
    0xC000-0xDFFF identically to Gen-1.
    """

    for probe in GSC_PROBES:
        assert 0xC000 <= probe.address <= 0xDFFF, (
            f"probe {probe.name!r} address 0x{probe.address:04X} not in WRAM"
        )


def test_pokemon_gsc_to_dict_metadata() -> None:
    """@example: the wire blob declares the gbc system and the gsc_common variant."""

    blob = gsc_to_dict()
    assert blob["name"] == "pokemon_gsc"
    assert blob["system"] == "gbc"
    assert blob["memory_region"] == "system_ram"
    assert blob["region_base_address"] == 0xC000
    assert blob["variant"] == "gsc_common"
    names = {p["name"] for p in blob["probes"]}
    # Anchor probes any LLM-driven agent needs as a minimum overlay.
    assert {
        "map_group", "map_number", "player_x", "player_y",
        "party_count", "battle_mode", "money",
        "johto_badges", "kanto_badges",
    }.issubset(names)


def test_pokemon_gsc_badge_bitfields_have_eight_labels() -> None:
    """@example: bitfield8 probes carry exactly 8 labels (LSB first)."""

    blob = gsc_to_dict()
    badges = [p for p in blob["probes"] if p["type"] == "bitfield8"]
    assert badges, "expected at least one bitfield8 probe in GSC"
    for entry in badges:
        assert len(entry["labels"]) == 8, (
            f"bitfield8 probe {entry['name']!r} expects 8 labels, "
            f"got {len(entry['labels'])}"
        )


# ── Zelda: Link's Awakening DX ────────────────────────────────────────


from augmentum.game_agent.probes.zelda_links_awakening_dx import (  # noqa: E402
    PROBES as LA_PROBES,
)
from augmentum.game_agent.probes.zelda_links_awakening_dx import (  # noqa: E402
    to_dict as la_to_dict,
)


def test_zelda_la_probes_have_valid_decoders() -> None:
    """@example: every Link's Awakening probe uses a known decoder."""

    for probe in LA_PROBES:
        assert probe.type in _VALID_DECODER_TYPES


def test_zelda_la_addresses_live_in_wram() -> None:
    """@example: every Link's Awakening probe address sits inside GB WRAM."""

    for probe in LA_PROBES:
        assert 0xC000 <= probe.address <= 0xDFFF, (
            f"LA probe {probe.name!r} address 0x{probe.address:04X} not in WRAM"
        )


def test_zelda_la_to_dict_carries_us_v1_0_variant() -> None:
    """@example: blob declares us_v1_0 so a future version-detect probe can refuse mismatches."""

    blob = la_to_dict()
    assert blob["name"] == "zelda_links_awakening_dx"
    assert blob["system"] == "gbc"
    assert blob["variant"] == "us_v1_0"
    names = {p["name"] for p in blob["probes"]}
    assert {
        "current_health", "max_health", "rupees",
        "item_slot_a", "item_slot_b",
    }.issubset(names)


def test_zelda_la_health_stored_as_hearts_times_eight() -> None:
    """@example: current_health and max_health are byte-wide, hearts*8 encoded.

    Asserts the wire blob carries u8 width for both, matching the
    decoder shape the JS shim expects (one byte, divisor of 8 applied
    client-side if hearts-rounded display is desired).
    """

    blob = la_to_dict()
    by_name = {p["name"]: p for p in blob["probes"]}
    assert by_name["current_health"]["type"] == "u8"
    assert by_name["max_health"]["type"] == "u8"
    assert by_name["current_health"]["length"] == 1
    assert by_name["max_health"]["length"] == 1


# ── Pokémon Emerald (Gen-3, GBA — pointer-deref SaveBlock) ─────────────


from augmentum.game_agent.probes.pokemon_emerald import (  # noqa: E402
    PROBES as EMERALD_PROBES,
)
from augmentum.game_agent.probes.pokemon_emerald import (  # noqa: E402
    to_dict as emerald_to_dict,
)


def test_pokemon_emerald_probes_have_valid_decoders() -> None:
    """@example: every Emerald probe uses a decoder the JS shim implements."""

    for probe in EMERALD_PROBES:
        assert probe.type in _VALID_DECODER_TYPES, (
            f"probe {probe.name!r} declares unknown decoder type {probe.type!r}"
        )


def test_pokemon_emerald_position_probes_use_pointer_deref() -> None:
    """@example: player position derefs gSaveBlock1Ptr (0x03005D8C, IWRAM).

    Emerald relocates its SaveBlock at runtime, so position CANNOT be a
    fixed address. The wire blob must carry a ``pointer`` block whose
    ``at`` is the IWRAM pointer and whose ``offset`` is the field's
    offset within SaveBlock1.
    """

    blob = emerald_to_dict()
    by_name = {p["name"]: p for p in blob["probes"]}
    for field, off in (("player_x", 0x00), ("player_y", 0x02),
                       ("map_group", 0x04), ("map_num", 0x05)):
        entry = by_name[field]
        assert "pointer" in entry, f"{field} must be a pointer-deref probe"
        assert entry["pointer"]["at"] == 0x03005D8C, f"{field} derefs gSaveBlock1Ptr"
        assert entry["pointer"]["size"] == 4
        assert entry["pointer"]["offset"] == off


def test_pokemon_emerald_party_probes_are_fixed_ewram() -> None:
    """@example: party data is direct (no pointer) at fixed EWRAM symbols."""

    blob = emerald_to_dict()
    by_name = {p["name"]: p for p in blob["probes"]}
    for field in ("party_count", "p1_level", "p1_hp_current", "p1_hp_max"):
        entry = by_name[field]
        assert "pointer" not in entry, f"{field} is a fixed-address read"
        assert 0x02000000 <= entry["address"] < 0x02040000, (
            f"{field} at {entry['address']:#x} is outside GBA EWRAM"
        )


def test_pokemon_emerald_to_dict_metadata() -> None:
    """@example: blob declares the gba system + us_bpee variant."""

    blob = emerald_to_dict()
    assert blob["name"] == "pokemon_emerald"
    assert blob["system"] == "gba"
    assert blob["variant"] == "us_bpee"
    # Position uses signed 16-bit tile coords.
    by_name = {p["name"]: p for p in blob["probes"]}
    assert by_name["player_x"]["type"] == "s16le"
    assert by_name["player_x"]["length"] == 2
