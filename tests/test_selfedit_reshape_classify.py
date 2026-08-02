"""Tests for the model-backed reshape classifier + catalog (NL → catalog-validated
change).

Model call is injected (canned JSON), so this is deterministic. Load-bearing:
  - a clear ask maps to the right catalog entry (enum normalized to canonical);
  - an out-of-catalog key/value is REJECTED → None (the safety allowlist; never a
    wild guess);
  - explicit ``{"unmapped": true}`` and low confidence → None;
  - a free-text field accepts any value;
  - end-to-end through the engine: NL ask → classify → config reshape → VERIFIED.
"""

from __future__ import annotations

from augmentum.selfedit import verifier as V
from augmentum.selfedit.surfaces import (
    STATUS_PROMOTED,
    ReshapeRequest,
    build_config_surface,
    build_model_classifier,
    clear_schemas,
    clear_surfaces,
    example_adaptation_schema,
    register_schema,
    register_surface,
    run_reshape_request,
    validate,
)


def _invoke_returning(text: str):
    async def invoke(_prompt: str) -> str:
        return text
    return invoke


def _setup_catalog():
    clear_schemas()
    register_schema(example_adaptation_schema())  # config: theme/density/accent


async def _classify(ask: str, model_text: str, *, min_confidence: float = 0.0):
    _setup_catalog()
    classifier = build_model_classifier(_invoke_returning(model_text),
                                        min_confidence=min_confidence)
    return await classifier(ReshapeRequest(ask=ask, actor="u1"), ["config"])


# --- catalog validation ----------------------------------------------------

def test_validate_enum_normalizes_and_rejects():
    schemas = [example_adaptation_schema()]
    ok, canon, _ = validate(schemas, "config", "theme", "DARK")
    assert ok and canon == "dark"                       # case-insensitive → canonical
    bad, _, reason = validate(schemas, "config", "theme", "neon")[:3]
    assert bad is False and "not allowed" in reason
    unknown, _, why = validate(schemas, "config", "made_up_key", "x")[:3]
    assert unknown is False and "not adaptable" in why  # safety allowlist


# --- classifier ------------------------------------------------------------

async def test_clear_ask_maps_and_normalizes():
    change = await _classify("make it dark", '{"surface":"config","key":"theme","value":"Dark"}')
    assert change is not None
    assert change.payload == {"key": "theme", "value": "dark"}   # normalized
    assert change.actor == "u1"
    assert change.intent == "set theme=dark"            # surfaced for see-it/keep-it


async def test_out_of_catalog_value_is_rejected():
    change = await _classify("neon theme", '{"surface":"config","key":"theme","value":"neon"}')
    assert change is None                                # disallowed enum → unmapped, not guessed


async def test_out_of_catalog_key_is_rejected():
    change = await _classify("hack the db", '{"surface":"config","key":"db_password","value":"x"}')
    assert change is None                                # safety allowlist holds


async def test_unmapped_and_low_confidence():
    assert await _classify("what's the weather", '{"unmapped": true}') is None
    low = await _classify("maybe darker?",
                          '{"surface":"config","key":"theme","value":"dark","confidence":0.3}',
                          min_confidence=0.7)
    assert low is None                                   # below floor → ask, don't act


async def test_free_text_field_accepts_any_value():
    change = await _classify("accent teal", '{"surface":"config","key":"accent","value":"teal"}')
    assert change is not None and change.payload == {"key": "accent", "value": "teal"}


async def test_garbage_model_output_is_unmapped_not_crash():
    assert await _classify("do a thing", "sorry I can't help") is None


# --- end to end: NL ask → classify → live config reshape -------------------

async def test_nl_ask_drives_verified_config_reshape():
    _setup_catalog()
    clear_surfaces()
    store: dict = {}

    async def read(uid, key):
        return store.get((uid, key))

    async def write(uid, key, val):
        store[(uid, key)] = val

    register_surface(build_config_surface(read=read, write=write))
    classifier = build_model_classifier(
        _invoke_returning('{"surface":"config","key":"density","value":"compact"}'))

    res = await run_reshape_request(ReshapeRequest(ask="make the panels denser", actor="u1"),
                                    classify=classifier)
    assert res.mapped and res.status == STATUS_PROMOTED
    assert res.reshape.verdict.tier == V.TIER_VERIFIED
    assert store[("u1", "density")] == "compact"
