"""Per-game RAM probe presets for emulator surfaces.

A probe preset is a declarative table of memory locations the bridge
should read every tick and emit as structured log events. The table
is the source of truth for what the slow-path agent *can know* about
the running game without doing vision.

Each preset is a single Python module under this package with two
public exports:

* ``PROBES`` -- a tuple of :class:`Probe` definitions.
* ``to_dict()`` -- a JSON-serialisable view of the same table, so the
  browser shim (which does the actual reading) can ingest it
  identically. The decoder names in the JSON view map 1:1 to the
  small decoder library shipped with the JS shim.

Presets live next to their game family rather than next to the
surface adapter because the same emulator core (e.g. ``gambatte``)
runs many different games with disjoint memory layouts.
"""

from __future__ import annotations

from augmentum.game_agent.probes.pokemon_emerald import (
    PROBES as POKEMON_EMERALD_PROBES,
)
from augmentum.game_agent.probes.pokemon_emerald import (
    to_dict as pokemon_emerald_to_dict,
)
from augmentum.game_agent.probes.pokemon_gsc import (
    PROBES as POKEMON_GSC_PROBES,
)
from augmentum.game_agent.probes.pokemon_gsc import (
    to_dict as pokemon_gsc_to_dict,
)
from augmentum.game_agent.probes.pokemon_rby import (
    PROBES as POKEMON_RBY_PROBES,
)
from augmentum.game_agent.probes.pokemon_rby import (
    to_dict as pokemon_rby_to_dict,
)
from augmentum.game_agent.probes.pokemon_rs import (
    PROBES as POKEMON_RS_PROBES,
)
from augmentum.game_agent.probes.pokemon_rs import (
    to_dict as pokemon_rs_to_dict,
)
from augmentum.game_agent.probes.zelda_links_awakening_dx import (
    PROBES as ZELDA_LINKS_AWAKENING_DX_PROBES,
)
from augmentum.game_agent.probes.zelda_links_awakening_dx import (
    to_dict as zelda_links_awakening_dx_to_dict,
)

_PRESET_PROBES = {
    "pokemon_emerald": POKEMON_EMERALD_PROBES,
    "pokemon_gsc": POKEMON_GSC_PROBES,
    "pokemon_rby": POKEMON_RBY_PROBES,
    "pokemon_rs": POKEMON_RS_PROBES,
    "zelda_links_awakening_dx": ZELDA_LINKS_AWAKENING_DX_PROBES,
}


def hidden_probe_names(game_profile: str | None) -> frozenset[str]:
    """Names of probes that feed the blackboard but never the prompt.

    Keyed by the Phase-G ``game_profile`` (presets and profiles share
    ids). Unknown profiles hide nothing.
    """

    probes = _PRESET_PROBES.get(game_profile or "")
    if not probes:
        return frozenset()
    return frozenset(p.name for p in probes if getattr(p, "hidden", False))


__all__ = [
    "POKEMON_EMERALD_PROBES",
    "POKEMON_GSC_PROBES",
    "POKEMON_RBY_PROBES",
    "POKEMON_RS_PROBES",
    "ZELDA_LINKS_AWAKENING_DX_PROBES",
    "hidden_probe_names",
    "pokemon_emerald_to_dict",
    "pokemon_gsc_to_dict",
    "pokemon_rby_to_dict",
    "pokemon_rs_to_dict",
    "zelda_links_awakening_dx_to_dict",
]
