"""Pins the single-source-of-truth invariant for cast library rails.

Before `rail_catalog.py` existed, the rail list was duplicated across
three sites with no enforced coupling — the `music` slug lived in the
prefs schema + controller UI for months without a backing section.
These tests catch any future re-introduction of that drift.
"""

from __future__ import annotations

from augmentum.cast.rail_catalog import (
    KNOWN_RAILS,
    RAIL_CATALOG,
    catalog_ui_meta,
    section_specs,
)


def test_known_rails_derived_from_catalog():
    """KNOWN_RAILS must be exactly the catalog slugs in display order.

    The prefs schema imports KNOWN_RAILS — if the derivation breaks,
    `rails_visible` would accept slugs the data layer can't render
    (or reject ones it can).
    """
    assert tuple(spec.slug for spec in RAIL_CATALOG) == KNOWN_RAILS


def test_receiver_prefs_uses_catalog_known_rails():
    """receiver_prefs re-exports KNOWN_RAILS from rail_catalog. If
    someone re-introduces a hardcoded list there, this fails."""
    from augmentum.cast import rail_catalog, receiver_prefs
    assert receiver_prefs.KNOWN_RAILS is rail_catalog.KNOWN_RAILS


def test_section_specs_alignment():
    """section_specs() must produce one entry per catalog rail with
    the title + query + home_limit copied through. cast_routes.py
    consumes this directly as _CAST_LIBRARY_SECTIONS."""
    specs = section_specs()
    assert set(specs.keys()) == set(KNOWN_RAILS)
    for slug in KNOWN_RAILS:
        spec = next(s for s in RAIL_CATALOG if s.slug == slug)
        assert specs[slug]["title"] == spec.title
        assert specs[slug]["home_limit"] == spec.home_limit
        # Query dict must be a copy — mutating one shouldn't bleed
        # into the canonical definition.
        assert specs[slug]["query"] == spec.query
        assert specs[slug]["query"] is not spec.query


def test_ui_meta_excludes_server_only_fields():
    """catalog_ui_meta() serialises only slug + title + hint — the
    server-only `query` and `home_limit` must not leak to clients."""
    meta = catalog_ui_meta()
    assert len(meta) == len(RAIL_CATALOG)
    for entry in meta:
        assert set(entry.keys()) == {"slug", "title", "hint"}


def test_no_music_ghost():
    """Explicit guard against the bug this refactor fixed: the prefs
    schema accepted `music` but no section ever existed for it."""
    assert "music" not in KNOWN_RAILS
    assert "music" not in section_specs()


def test_rails_visible_round_trip_for_server_filter():
    """The library_home route's per-receiver filter derives hidden
    slugs from `with_defaults(stored).rails_visible` — anything with
    explicit `False` is dropped. This test pins that round-trip so a
    future refactor of receiver_prefs can't silently change the shape
    the route depends on.
    """
    from augmentum.cast.receiver_prefs import coerce_prefs, with_defaults
    # User toggled Movies off, left Comics + Shows alone.
    stored = coerce_prefs({"rails_visible": {"movies": False, "comics": True}})
    merged = with_defaults(stored)
    rv = merged["rails_visible"]
    # Stored False survives the round-trip.
    assert rv["movies"] is False
    # Explicit True survives.
    assert rv["comics"] is True
    # Unspecified rails default to True (so newly-added rails appear).
    assert rv["shows"] is True
    # The route's filter expression: hidden = explicit-False entries.
    hidden = {slug for slug, visible in rv.items() if visible is False}
    assert hidden == {"movies"}


def test_coerce_prefs_drops_ghost_rails():
    """Stored prefs from before the refactor may carry rails_visible
    keys (like `music`) that are no longer in KNOWN_RAILS. coerce_prefs
    must drop them silently so old data doesn't break the bag."""
    from augmentum.cast.receiver_prefs import coerce_prefs
    cleaned = coerce_prefs({
        "rails_visible": {
            "movies": False,
            "music": True,          # legacy ghost
            "not_a_real_rail": True,
        },
    })
    rv = cleaned["rails_visible"]
    assert "movies" in rv
    assert "music" not in rv
    assert "not_a_real_rail" not in rv
