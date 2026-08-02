"""Tests for the surface-agnostic reshape layer.

The point of this layer is that ONE orchestration works for every surface. The
load-bearing tests:
  - config (Adaptation) earns the VERIFIED tier via a mechanical read-back oracle
    → keeps + auto-promotable (why "it rearranged itself for me" is instant+safe);
  - a verify FAILURE auto-reverts (a broken change never lingers);
  - a non-confirming oracle → kept but needs_human (the surface-agnostic
    "see it, keep it" path) — proving keep/hold/revert keys on oracle-tier, not
    surface.
All store effects are injected (a dict); no DB, no model, no RNG, no clock.
"""

from __future__ import annotations

from augmentum.selfedit import verifier as V
from augmentum.selfedit.surfaces import (
    CLASS_ADAPTATION,
    CLASS_BUILD,
    CaptureArtifact,
    ReshapeChange,
    ReshapeOutcome,
    SurfaceAdapter,
    build_config_surface,
    clear_surfaces,
    get_surface,
    register_surface,
    registered_surfaces,
    reshape,
)


class FakeStore:
    def __init__(self):
        self.data: dict = {}

    async def read(self, uid, key):
        return self.data.get((uid, key))

    async def write(self, uid, key, val):
        self.data[(uid, key)] = val


def _config(store: FakeStore) -> SurfaceAdapter:
    return build_config_surface(read=store.read, write=store.write)


def _change(key="theme", value="dark", actor="u1", cls=CLASS_ADAPTATION):
    return ReshapeChange(surface="config", change_class=cls,
                         payload={"key": key, "value": value}, actor=actor,
                         intent=f"set {key}")


# --- registry --------------------------------------------------------------

def test_registry_register_get_clear():
    clear_surfaces()
    store = FakeStore()
    register_surface(_config(store))
    assert get_surface("config") is not None
    assert "config" in registered_surfaces()
    clear_surfaces()
    assert get_surface("config") is None


# --- config surface --------------------------------------------------------

async def test_config_apply_verify_is_verified():
    store = FakeStore()
    res = await reshape(_change("theme", "dark"), adapter=_config(store))
    assert res.applied and res.kept
    assert res.auto_promotable is True and res.needs_human is False
    assert res.verdict.tier == V.TIER_VERIFIED          # mechanical read-back confirmed intent
    assert store.data[("u1", "theme")] == "dark"
    assert "theme" in res.capture.summary


async def test_config_refuses_empty_actor():
    store = FakeStore()
    res = await reshape(_change(actor=""), adapter=_config(store))
    assert res.applied is False
    assert "actor" in res.detail


async def test_config_revert_restores_prior():
    store = FakeStore()
    store.data[("u1", "density")] = "comfy"
    surface = _config(store)
    outcome = await surface.apply(_change("density", "cozy"))
    assert store.data[("u1", "density")] == "cozy"
    assert await surface.revert(outcome.revert_token) is True
    assert store.data[("u1", "density")] == "comfy"     # prior restored


async def test_reshape_reverts_on_verify_failure():
    store = FakeStore()

    async def write_noop(_uid, _key, _val):
        return None  # write that never persists → read-back != intended → FAIL

    surface = build_config_surface(read=store.read, write=write_noop)
    res = await reshape(_change("theme", "dark"), adapter=surface)
    assert res.applied is True and res.kept is False     # applied then auto-reverted
    assert res.verdict.tier == V.TIER_FAILED


# --- agnostic routing guards ----------------------------------------------

async def test_reshape_unknown_surface():
    clear_surfaces()
    res = await reshape(ReshapeChange(surface="nope", change_class=CLASS_ADAPTATION,
                                      payload={"key": "x", "value": 1}, actor="u1"))
    assert res.applied is False and "no adapter" in res.detail


async def test_reshape_rejects_unhandled_change_class():
    store = FakeStore()
    res = await reshape(_change(cls=CLASS_BUILD), adapter=_config(store))
    assert res.applied is False and "does not handle" in res.detail


# --- the surface-agnostic "see it, keep it" path --------------------------

def _taste_adapter(applied_box: dict) -> SurfaceAdapter:
    """A fake surface whose oracle only proves liveness (confirms_intent=False) —
    the taste/visual case (a VR room recolor): runs fine, but only the user can
    say it's good."""
    async def apply(_change):
        applied_box["applied"] = True
        return ReshapeOutcome(True, revert_token="tok")

    async def revert(_token):
        applied_box["reverted"] = True
        return True

    def make_verifier(_change):
        async def run(_ctx):
            return V.VerifierResult("liveness", V.ORACLE_MECHANICAL, V.PASS,
                                    confirms_intent=False, required=True)
        return V.Verifier("liveness", V.ORACLE_MECHANICAL, run,
                          intent_classes=("*",), confirms_intent=False)

    async def capture(_change):
        return CaptureArtifact(kind="screenshot", ref="/tmp/x.png", summary="snap")

    return SurfaceAdapter(name="taste", change_classes=(CLASS_ADAPTATION,),
                          apply=apply, revert=revert, make_verifier=make_verifier,
                          capture=capture)


async def test_nonconfirming_oracle_keeps_but_needs_human():
    box: dict = {}
    res = await reshape(_change(), adapter=_taste_adapter(box))
    assert res.applied and res.kept                      # stays applied — "see it"
    assert res.needs_human is True                       # ...but awaiting the user's pick
    assert res.auto_promotable is False
    assert res.verdict.tier == V.TIER_HUMAN_REQUIRED
    assert box.get("reverted") is None                   # NOT reverted (it didn't fail)
    assert res.capture.kind == "screenshot"
