"""Router catalog — the live closed-world vocabulary for surface_emit synthesis.

Load-bearing:
  - parse_router_catalog extracts the real channels + navigate targets and does
    NOT leak nested object keys;
  - validate_emit_target rejects a dead channel/surface (the "verified but does
    nothing" failure) and is tolerant when the catalog can't be read;
  - the parser still understands the REAL router file (guard against a refactor
    that silently empties the catalog and disables the gate);
  - synthesis actually rejects a dead-target spec end to end.
"""

from __future__ import annotations

from augmentum.selfedit.capabilities import (
    CapabilitySpec,
    load_router_catalog,
    parse_router_catalog,
    validate_emit_target,
)
from augmentum.selfedit.capabilities.router_catalog import (
    RouterCatalog,
    describe_for_prompt,
    validate_declared_args,
)
from augmentum.selfedit.capabilities.synthesize import synthesize_capability_spec

# A trimmed-but-faithful slice of intent-action-router.js: a couple of switch
# arms plus a _NAV_TARGETS block with a nested object key that must NOT be
# mistaken for a top-level surface.
_ROUTER_SNIPPET = """\
const _NAV_TARGETS = {
  browse:       () => import('./browse.js').then(m => m.openBrowsePanel?.()),
  notes:        async () => {
    const m = await import('./browse.js');
    document.dispatchEvent(new CustomEvent('augmentum:switch-browse-tab',
      { detail: { tab: 'notes' } }));
  },
  settings:     () => import('./settings.js').then(m => m.openSettings?.()),
};

function routeIntentAction(channel, payload) {
  switch (channel) {
    case 'navigate.open_surface': { return true; }
    case 'palette.run': return true;
    case 'timer.set': return true;
  }
}
"""


def _cat() -> RouterCatalog:
    return parse_router_catalog(_ROUTER_SNIPPET, source="<snippet>")


# --- parsing ---------------------------------------------------------------

def test_parse_extracts_channels_and_nav_surfaces():
    cat = _cat()
    assert cat.available
    assert {"navigate.open_surface", "palette.run", "timer.set"} <= cat.channels
    assert cat.nav_surfaces == {"browse", "notes", "settings"}


def test_parse_does_not_leak_nested_object_keys():
    # 'detail' and 'tab' live inside a nested object literal — never surfaces.
    cat = _cat()
    assert "detail" not in cat.nav_surfaces
    assert "tab" not in cat.nav_surfaces


def test_empty_text_is_unavailable():
    cat = parse_router_catalog("// nothing here")
    assert not cat.available
    assert cat.channels == frozenset()


# --- the dead-target gate --------------------------------------------------

def test_validate_accepts_real_channel_and_surface():
    cat = _cat()
    assert validate_emit_target("navigate.open_surface", {"surface": "browse"}, catalog=cat) == ""
    assert validate_emit_target("timer.set", {"duration_s": 60}, catalog=cat) == ""


def test_validate_rejects_unknown_channel():
    problem = validate_emit_target("totally.madeup", {}, catalog=_cat())
    assert "not handled by the frontend router" in problem


def test_validate_rejects_dead_nav_surface():
    # the exact bug we had baked in: navigate.open_surface -> "workshop"
    problem = validate_emit_target("navigate.open_surface", {"surface": "workshop"}, catalog=_cat())
    assert "workshop" in problem and "no-op" in problem


def test_validate_rejects_palette_run_as_unsynthesizable():
    # palette command ids are a per-user ephemeral runtime catalog; a permanent
    # synthesized verb can't target them. app.act already does this dynamically.
    cat = _cat()
    problem = validate_emit_target("palette.run", {"command_id": "x"}, catalog=cat)
    assert "app.act" in problem and "not synthesizable" in problem


def test_validate_declared_args_flags_dead_arg():
    # navigate.open_surface only reads `surface`; a declared `city` would be dropped
    problem = validate_declared_args("navigate.open_surface", ["surface", "city"])
    assert "city" in problem and "silently dropped" in problem


def test_validate_declared_args_accepts_consumed_arg():
    assert validate_declared_args("navigate.open_surface", ["surface"]) == ""
    assert validate_declared_args("navigate.back", []) == ""


def test_validate_declared_args_tolerant_for_uncurated_channel():
    # a channel we haven't curated consumed-keys for isn't blocked
    assert validate_declared_args("browse.open_url", ["url", "anything"]) == ""


def test_validate_is_tolerant_when_catalog_unavailable():
    empty = RouterCatalog(channels=frozenset(), nav_surfaces=frozenset())
    # can't read the source -> don't block (build-time oracle is the backstop)
    assert validate_emit_target("anything.at.all", {"surface": "nope"}, catalog=empty) == ""


def test_describe_for_prompt_lists_real_targets_or_is_empty():
    assert "navigate.open_surface" in describe_for_prompt(_cat())
    assert describe_for_prompt(RouterCatalog(frozenset(), frozenset())) == ""


# --- guard against a real-file refactor that empties the catalog -----------

def test_real_router_file_still_parses():
    cat = load_router_catalog()
    # If this fails, ui/scripts/intent-action-router.js moved or its _NAV_TARGETS
    # / case syntax changed shape — the synthesis gate silently disables until the
    # parser is updated. Treat it like the stress harness: fix the parser.
    assert cat.available, "router catalog parsed empty — parser is out of date"
    assert {"navigate.open_surface", "palette.run", "timer.set"} <= cat.channels
    assert {"browse", "settings"} <= cat.nav_surfaces
    assert "workshop" not in cat.nav_surfaces  # the dead target stays dead


# --- end to end: synthesis rejects a dead target ---------------------------

async def test_synthesize_rejects_dead_surface_target():
    import json

    dead = json.dumps({
        "id": "navigate.open_workshop",
        "summary": "Open the Workshop.",
        "examples": ["open the workshop"],
        "behavior": "surface_emit",
        "channel": "navigate.open_surface",
        "payload": {"surface": "workshop"},   # not a real navigate target
        "stakes": "trivial_reversible",
    })

    # model returns the dead spec twice (synth + repair); both rejected.
    async def mi(_prompt: str) -> str:
        return dead

    spec, errs = await synthesize_capability_spec("open the workshop", model_invoke=mi)
    assert spec is None
    assert any("workshop" in e for e in errs)


def test_validate_spec_unchanged_by_target_gate():
    # structural validation stays target-agnostic: a dead surface is structurally
    # valid (the live gate, not validate_spec, is what rejects it).
    from augmentum.selfedit.capabilities import validate_spec

    s = CapabilitySpec(
        id="navigate.open_workshop", summary="x", examples=["x"],
        behavior="surface_emit", channel="navigate.open_surface",
        payload={"surface": "workshop"},
    )
    assert validate_spec(s) == []
