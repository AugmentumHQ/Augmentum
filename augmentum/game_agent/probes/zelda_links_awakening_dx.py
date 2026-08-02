"""Zelda: Link's Awakening DX RAM probe preset (GBC).

Memory map references
---------------------
Link's Awakening DX has no public decompilation on the level of the
pret/pokecrystal series, so the addresses below come from the
community-maintained RAM map at:

* https://datacrystal.tcrf.net/wiki/The_Legend_of_Zelda:_Link%27s_Awakening_DX/RAM_map

Targets the **US v1.0** ROM revision (sha1 ``0xCE08…``). EU and Japanese
builds may shift a handful of bytes; tag the preset's ``"variant"`` so a
future version-detect probe can refuse to read on a mismatched ROM.

Encoding gotchas
----------------
* **Health** is stored as ``hearts * 8`` in both current and max fields.
  A "3-hearts-full" Link reads as ``24`` here, not ``3``. The slow-path
  agent can divide by 8 or compare current/max directly without caring
  about the unit.
* **Rupees** are little-endian 2-byte unsigned, 0..9999. Standard Game
  Boy storage convention for the engine (the cap is 999 on-screen but
  the field can hold higher values during cutscenes).
* **Item slots** point into an item-id enum (0x00 = no item, 0x01 =
  sword, 0x02 = bombs, 0x03 = power bracelet, etc.). The mapping is
  static and worth surfacing to the prompt as a label table in a future
  revision; for v1 we ship the raw byte so the agent can still
  distinguish "equipped item changed".

Scope of v1
-----------
Top-level inventory + health globals only. Link's screen-space
position and current-room index are intentionally deferred until we can
verify the live values against a running ROM; the community map is less
unanimous on those addresses than on the inventory page.
"""

from __future__ import annotations

from typing import Any

from augmentum.game_agent.probes.pokemon_rby import Probe, probe_entry

# ── Top-level WRAM globals (Link's Awakening DX, US v1.0) ─────────────
#
# Source: datacrystal community map. Addresses are decimal-equivalent to
# the hex in the trailing comment so the JSON shim's plain-int output is
# auditable against the upstream map.

_WITEM_SLOT_A      = 0xDB00   # currently A-bound item id
_WITEM_SLOT_B      = 0xDB01   # currently B-bound item id
_WCURRENT_HEALTH   = 0xDB5A   # u8, value = hearts * 8
_WMAX_HEALTH       = 0xDB5B   # u8, value = max hearts * 8
_WRUPEE_COUNT      = 0xDB5D   # u16le, 0..9999


PROBES: tuple[Probe, ...] = (
    Probe(
        name="current_health",
        address=_WCURRENT_HEALTH,
        length=1,
        type="u8",
        description="Link's current health, stored as hearts*8 (24 == 3 hearts).",
    ),
    Probe(
        name="max_health",
        address=_WMAX_HEALTH,
        length=1,
        type="u8",
        description="Link's maximum health, stored as hearts*8.",
    ),
    Probe(
        name="rupees",
        address=_WRUPEE_COUNT,
        length=2,
        type="u16le",
        description="Rupee count, little-endian 2 bytes, capped 9999 visually.",
    ),
    Probe(
        name="item_slot_a",
        address=_WITEM_SLOT_A,
        length=1,
        type="u8",
        description="Item id currently bound to the A button (0 = none).",
    ),
    Probe(
        name="item_slot_b",
        address=_WITEM_SLOT_B,
        length=1,
        type="u8",
        description="Item id currently bound to the B button (0 = none).",
    ),
)


def to_dict() -> dict[str, Any]:
    """Return the preset as a JSON-serialisable dict.

    Before:
    - ``PROBES[0]`` -> ``Probe(name="current_health", address=0xDB5A, length=1, type="u8", ...)``

    After:
    - ``to_dict()["probes"][0]`` ->
      ``{"name": "current_health", "address": 56154, "length": 1, "type": "u8", "description": "..."}``
    """

    return {
        "name": "zelda_links_awakening_dx",
        "system": "gbc",
        "memory_region": "system_ram",
        "region_base_address": 0xC000,
        "variant": "us_v1_0",
        "probes": [probe_entry(p) for p in PROBES],
    }
