"""Single source of truth for cast-surface library rails.

Owns the rail definitions that drive:
  - server-side data fetch (`cast_library_home` builds sections from these)
  - the per-receiver prefs schema (which slugs are valid `rails_visible` keys)
  - the controller's prefs sheet labels (served via `/api/cast/rails/catalog`)

Before this module existed, the three sites duplicated the rail list with
no enforced coupling — adding/removing a rail required three coordinated
edits and a `music` ghost lived in the prefs schema + UI for months
with no backing section. Defining everything once here closes that gap.

Adding a rail:
  1. Append a `RailSpec` to `RAIL_CATALOG` below
  2. The prefs schema picks it up automatically (KNOWN_RAILS is derived)
  3. The prefs sheet picks it up automatically (catalog endpoint is derived)
  4. The home-screen render picks it up automatically
     (cast_routes.py builds `_CAST_LIBRARY_SECTIONS` from this module)

Removing a rail: delete its entry. Old stored prefs containing the
removed slug are dropped silently by `coerce_prefs` (unknown keys),
so retired rails don't leave dead toggles in user data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RailSpec:
    """Canonical definition of one cast library rail.

    Fields with server-only meaning (`query`, `home_limit`) stay on the
    server. `catalog_ui_meta()` serialises only the user-facing fields
    (slug + title + hint) for the controller's prefs sheet.
    """
    slug: str
    title: str          # Short label shown in rail headers + the prefs toggle
    hint: str           # One-line description shown beneath the toggle
    home_limit: int     # Items fetched for the cast-home rail preview
    # file_index query that powers the section. dict (not a static
    # dict literal) so each spec carries its own copy without aliasing.
    query: dict[str, Any] = field(default_factory=dict)


# Display order matches definition order — the home-screen and prefs
# sheet both iterate in this order. Group resume/recently-added at the
# top (highest engagement) then library kinds.
RAIL_CATALOG: tuple[RailSpec, ...] = (
    RailSpec(
        slug="resume",
        title="Continue",
        hint="Continue watching / listening",
        home_limit=12,
        query={"media_status": "in_progress", "sort": "progress"},
    ),
    RailSpec(
        slug="recently_added",
        title="Recently added",
        hint="Newest items across libraries",
        home_limit=20,
        query={
            "sort": "added",
            "exclude_kinds": ["doc"],
            "exclude_entity_kinds": ["season", "episode"],
        },
    ),
    RailSpec(
        slug="audiobooks",
        title="Audiobooks",
        hint="Audio library",
        home_limit=20,
        query={"kind": "audio"},
    ),
    RailSpec(
        slug="movies",
        title="Movies",
        hint="Standalone films",
        home_limit=20,
        query={"kind": "video", "entity_kinds": ["movie"]},
    ),
    RailSpec(
        slug="shows",
        title="Shows",
        hint="TV series (collapsed to series cards)",
        home_limit=20,
        # `series` only — episodes belong inside a series via the
        # drill-in (/api/cast/library/episodes/{file_id}), not at the
        # catalog root.
        query={"kind": "video", "entity_kinds": ["series"]},
    ),
    RailSpec(
        slug="music_videos",
        title="Music videos",
        hint="Music video catalog",
        home_limit=20,
        query={"kind": "video", "entity_kinds": ["music_video"]},
    ),
    RailSpec(
        slug="comics",
        title="Comics",
        hint="Comic series (collapsed)",
        home_limit=20,
        query={"kind": "comic"},
    ),
    RailSpec(
        slug="gallery",
        title="Gallery",
        hint="Your image library",
        home_limit=24,
        query={"kind": "image"},
    ),
    RailSpec(
        slug="games",
        title="Games",
        hint="Playable titles — tap to launch on the picked TV",
        home_limit=12,
        # Games don't live in file_index — they come from TitleService.
        # The cast_routes section handler branches on slug == "games"
        # rather than running this query through list_recent. The query
        # dict is kept non-empty so the spec round-trips cleanly through
        # section_specs() but its contents are unused.
        query={"_source": "title_service"},
    ),
)


# Derived: tuple of slugs in display order. Imported by
# `receiver_prefs.KNOWN_RAILS` so the prefs schema accepts exactly
# the slugs the data layer can actually render.
KNOWN_RAILS: tuple[str, ...] = tuple(spec.slug for spec in RAIL_CATALOG)


def section_specs() -> dict[str, dict[str, Any]]:
    """Server-side view of the catalog — dict shape historically used
    by `cast_routes._CAST_LIBRARY_SECTIONS`. Returns a fresh dict on
    each call so mutating the query (the home builder narrows `limit`)
    doesn't bleed back into the canonical definitions.
    """
    return {
        spec.slug: {
            "title": spec.title,
            "query": dict(spec.query),
            "home_limit": spec.home_limit,
        }
        for spec in RAIL_CATALOG
    }


def catalog_ui_meta() -> list[dict[str, str]]:
    """User-facing view of the catalog — fed to the controller's prefs
    sheet via `/api/cast/rails/catalog`. Excludes server-only fields
    (`query`, `home_limit`) so internal data-fetch knobs don't leak to
    clients. Display order matches `RAIL_CATALOG`.
    """
    return [
        {"slug": spec.slug, "title": spec.title, "hint": spec.hint}
        for spec in RAIL_CATALOG
    ]
