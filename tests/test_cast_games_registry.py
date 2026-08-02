"""Tests for the universal cast pipeline — CastProfileRegistry +
CastClassifier + same-origin strategy.

Pins:
  - models: round-trip dataclass ↔ JSON ↔ SQLite row
  - registry: get/upsert/override/delete are user-scoped
  - registry: input_chain coercion drops unknown adapters
  - registry: cross-user reads return None (multi-tenant invariant)
  - same-origin strategy: can_handle + prepare for both URL shapes
  - classifier: default profile when no registry hit
  - classifier: registry hit wins
  - classifier: recent failed_at promotes to next strategy
"""
from __future__ import annotations

import asyncio
import time

import pytest

from augmentum.cast.games.classifier import DEMOTION_WINDOW_S, CastClassifier
from augmentum.cast.games.models import (
    CLASSIFIED_DEFAULT,
    CLASSIFIED_MANUAL,
    STRATEGY_PROXY,
    STRATEGY_SHIM,
    CastProfile,
    HostCapabilities,
    KeymapProfile,
    _coerce_input_chain,
    _coerce_strategy,
)
from augmentum.cast.games.registry import CastProfileRegistry
from augmentum.cast.games.strategies.base import (
    CastStrategy,
    StrategyRegistry,
)
from augmentum.cast.games.strategies.same_origin import SameOriginStrategy


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def db():
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    _run(backend.connect())
    yield backend
    _run(backend.close())


@pytest.fixture
def registry(db):
    return CastProfileRegistry(db._conn)


@pytest.fixture
def stub_strategies():
    """A StrategyRegistry preloaded with two stub strategies (shim +
    proxy) so classifier tests don't depend on the global singleton."""
    reg = StrategyRegistry()

    class _Shim(CastStrategy):
        id = STRATEGY_SHIM
        cost_rank = 1

        async def can_handle(self, title, host):
            return True

        async def prepare(self, title, profile):
            from augmentum.cast.games.models import PreparedCast
            return PreparedCast(
                title_id=str(title.get("id") or ""),
                strategy=self.id,
                surface_url=f"/ui/play/?title_id={title.get('id', '')}",
                input_chain=profile.input_chain,
            )

    class _Proxy(CastStrategy):
        id = STRATEGY_PROXY
        cost_rank = 2

        async def can_handle(self, title, host):
            return bool(title.get("embed_url"))

        async def prepare(self, title, profile):
            from augmentum.cast.games.models import PreparedCast
            return PreparedCast(
                title_id=str(title.get("id") or ""),
                strategy=self.id,
                surface_url="/api/cast/game-proxy/sess/index.html",
                input_chain=profile.input_chain,
            )

    reg.register(_Shim())
    reg.register(_Proxy())
    return reg


# ── models / coercion ────────────────────────────────────────────


def test_coerce_input_chain_drops_unknown():
    chain = _coerce_input_chain(["gamepad_api", "frobnicate", "keyboard"])
    assert chain == ("gamepad_api", "keyboard")


def test_coerce_input_chain_dedupes():
    chain = _coerce_input_chain(["keyboard", "keyboard", "keyboard"])
    assert chain == ("keyboard",)


def test_coerce_input_chain_empty_falls_back():
    assert _coerce_input_chain([]) == ("gamepad_api",)
    assert _coerce_input_chain(None) == ("gamepad_api",)
    assert _coerce_input_chain(["nonexistent_only"]) == ("gamepad_api",)


def test_coerce_strategy_unknown_falls_back_to_shim():
    assert _coerce_strategy("not_a_real_strategy") == STRATEGY_SHIM
    assert _coerce_strategy(None) == STRATEGY_SHIM
    assert _coerce_strategy("PROXY") == STRATEGY_PROXY  # case-insensitive


def test_keymap_to_dict_omits_empty_sections():
    km = KeymapProfile(keyboard={"buttons": {"0": "KeyZ"}})
    out = km.to_dict()
    assert "keyboard" in out
    assert "touch" not in out
    assert "pointer" not in out


def test_keymap_from_dict_handles_none():
    km = KeymapProfile.from_dict(None)
    assert km.is_empty()


def test_profile_to_dict_omits_empty_keymap():
    p = CastProfile(title_id="t1")
    out = p.to_dict()
    assert "keymap" not in out


# ── registry CRUD ────────────────────────────────────────────────


def test_get_missing_returns_none(registry):
    assert _run(registry.get("does-not-exist", user_id="alice")) is None


def test_upsert_then_get_roundtrips(registry):
    p = CastProfile(
        title_id="game-1",
        strategy=STRATEGY_PROXY,
        embed_url="https://js13kgames.com/foo/index.html",
        input_chain=("gamepad_api", "keyboard"),
        keymap=KeymapProfile(keyboard={"buttons": {"0": "Space"}}),
        quirks={"sw_disable": True},
        notes="manual override",
    )
    _run(registry.upsert(p, user_id="alice"))
    got = _run(registry.get("game-1", user_id="alice"))
    assert got is not None
    assert got.title_id == "game-1"
    assert got.strategy == STRATEGY_PROXY
    assert got.embed_url == "https://js13kgames.com/foo/index.html"
    assert got.input_chain == ("gamepad_api", "keyboard")
    assert got.keymap and got.keymap.keyboard == {"buttons": {"0": "Space"}}
    assert got.quirks == {"sw_disable": True}
    assert got.notes == "manual override"
    # classified_at is auto-stamped if missing
    assert got.classified_at > 0


def test_upsert_normalises_input_chain(registry):
    p = CastProfile(
        title_id="game-2",
        input_chain=("gamepad_api", "fake_adapter", "keyboard"),
    )
    _run(registry.upsert(p, user_id="alice"))
    got = _run(registry.get("game-2", user_id="alice"))
    assert got.input_chain == ("gamepad_api", "keyboard")  # unknown dropped


def test_upsert_replaces_on_conflict(registry):
    p1 = CastProfile(title_id="game-3", strategy=STRATEGY_SHIM, notes="v1")
    _run(registry.upsert(p1, user_id="alice"))
    p2 = CastProfile(title_id="game-3", strategy=STRATEGY_PROXY, notes="v2")
    _run(registry.upsert(p2, user_id="alice"))
    got = _run(registry.get("game-3", user_id="alice"))
    assert got.strategy == STRATEGY_PROXY
    assert got.notes == "v2"


def test_get_is_user_scoped(registry):
    p = CastProfile(title_id="game-x", notes="alice's profile")
    _run(registry.upsert(p, user_id="alice"))
    assert _run(registry.get("game-x", user_id="alice")) is not None
    assert _run(registry.get("game-x", user_id="bob")) is None


def test_list_for_user_is_isolated(registry):
    _run(registry.upsert(CastProfile(title_id="a1"), user_id="alice"))
    _run(registry.upsert(CastProfile(title_id="a2"), user_id="alice"))
    _run(registry.upsert(CastProfile(title_id="b1"), user_id="bob"))
    alice_list = _run(registry.list_for_user(user_id="alice"))
    bob_list = _run(registry.list_for_user(user_id="bob"))
    assert {p.title_id for p in alice_list} == {"a1", "a2"}
    assert {p.title_id for p in bob_list} == {"b1"}


def test_delete_returns_true_on_hit_false_on_miss(registry):
    _run(registry.upsert(CastProfile(title_id="z"), user_id="alice"))
    assert _run(registry.delete("z", user_id="alice")) is True
    assert _run(registry.delete("z", user_id="alice")) is False


def test_override_marks_classified_by_manual(registry):
    # First write a probe-classified entry
    _run(registry.upsert(
        CastProfile(
            title_id="g",
            strategy=STRATEGY_SHIM,
            classified_by="probe",
            classified_at=time.time() - 100,
        ),
        user_id="alice",
    ))
    # Override should flip provenance + bump classified_at
    out = _run(registry.override(
        "g", user_id="alice",
        strategy=STRATEGY_PROXY,
        notes="user said so",
    ))
    assert out.classified_by == CLASSIFIED_MANUAL
    assert out.strategy == STRATEGY_PROXY
    assert out.notes == "user said so"


def test_override_on_missing_creates_default_base(registry):
    out = _run(registry.override(
        "new-title", user_id="alice",
        input_chain=("keyboard",),
    ))
    assert out.title_id == "new-title"
    assert out.classified_by == CLASSIFIED_MANUAL
    assert out.input_chain == ("keyboard",)


def test_mark_failed_stamps_timestamp(registry):
    _run(registry.upsert(CastProfile(title_id="fg"), user_id="alice"))
    _run(registry.mark_failed("fg", user_id="alice"))
    got = _run(registry.get("fg", user_id="alice"))
    assert got.failed_at > 0


# ── same-origin strategy ─────────────────────────────────────────


def test_same_origin_can_handle_emulator_rom():
    strat = SameOriginStrategy()
    title = {"id": "rom-1", "kind": "emulator_rom"}
    assert _run(strat.can_handle(title, HostCapabilities())) is True


def test_same_origin_can_handle_js13k_with_embed():
    strat = SameOriginStrategy()
    title = {
        "id": "js13k-1",
        "kind": "js13k_game",
        "metadata": {"embed_url": "https://example.com/g/"},
    }
    assert _run(strat.can_handle(title, HostCapabilities())) is True


def test_same_origin_can_handle_rejects_when_no_url_no_id():
    strat = SameOriginStrategy()
    title = {"kind": "js13k_game"}  # no id, no embed
    assert _run(strat.can_handle(title, HostCapabilities())) is False


def test_same_origin_prepare_emulator_url():
    strat = SameOriginStrategy()
    title = {"id": "rom-abc", "kind": "emulator_rom"}
    profile = CastProfile(title_id="rom-abc")
    prep = _run(strat.prepare(title, profile))
    assert prep.surface_url == "/ui/play/?title_id=rom-abc&kiosk=1"
    assert prep.input_chain == ("gamepad_api",)
    assert prep.strategy == STRATEGY_SHIM


def test_same_origin_prepare_js13k_url_quotes():
    strat = SameOriginStrategy()
    title = {
        "id": "js13k-1",
        "kind": "js13k_game",
        "display_name": "Foo & Bar",
        "metadata": {"embed_url": "https://example.com/g/?level=1"},
    }
    profile = CastProfile(
        title_id="js13k-1",
        input_chain=("gamepad_api", "keyboard"),
    )
    prep = _run(strat.prepare(title, profile))
    assert "/ui/play-web/" in prep.surface_url
    assert "embed_url=https%3A%2F%2Fexample.com%2Fg%2F%3Flevel%3D1" in prep.surface_url
    assert "title=Foo%20%26%20Bar" in prep.surface_url
    assert prep.input_chain == ("gamepad_api", "keyboard")


# ── classifier ───────────────────────────────────────────────────


def test_classify_default_when_no_registry_hit(registry, stub_strategies):
    cls = CastClassifier(profile_registry=registry, strategies=stub_strategies)
    title = {"id": "t1", "kind": "emulator_rom"}
    result = _run(cls.classify(title))
    assert result.source == "default"
    assert result.strategy.id == STRATEGY_SHIM   # cheapest
    assert result.profile.classified_by == CLASSIFIED_DEFAULT


def test_classify_registry_hit_wins(registry, stub_strategies):
    p = CastProfile(
        title_id="t2",
        strategy=STRATEGY_PROXY,
        input_chain=("keyboard",),
    )
    _run(registry.upsert(p, user_id="alice"))
    cls = CastClassifier(profile_registry=registry, strategies=stub_strategies)
    title = {"id": "t2", "embed_url": "https://example.com"}
    result = _run(cls.classify(title, user_id="alice"))
    assert result.source == "registry"
    assert result.strategy.id == STRATEGY_PROXY
    assert result.profile.input_chain == ("keyboard",)


def test_classify_recent_failure_promotes(registry, stub_strategies):
    p = CastProfile(
        title_id="t3",
        strategy=STRATEGY_SHIM,
        failed_at=time.time() - 10,  # within demotion window
    )
    _run(registry.upsert(p, user_id="alice"))
    cls = CastClassifier(profile_registry=registry, strategies=stub_strategies)
    title = {"id": "t3", "embed_url": "https://example.com"}
    result = _run(cls.classify(title, user_id="alice"))
    assert result.source == "registry_promoted"
    assert result.strategy.id == STRATEGY_PROXY


def test_classify_old_failure_does_not_promote(registry, stub_strategies):
    p = CastProfile(
        title_id="t4",
        strategy=STRATEGY_SHIM,
        failed_at=time.time() - DEMOTION_WINDOW_S - 100,
    )
    _run(registry.upsert(p, user_id="alice"))
    cls = CastClassifier(profile_registry=registry, strategies=stub_strategies)
    title = {"id": "t4", "embed_url": "https://example.com"}
    result = _run(cls.classify(title, user_id="alice"))
    assert result.source == "registry"  # old failure ignored
    assert result.strategy.id == STRATEGY_SHIM


def test_classify_picks_cheapest_qualifying(registry, stub_strategies):
    """When both strategies can_handle, cheapest wins."""
    cls = CastClassifier(profile_registry=registry, strategies=stub_strategies)
    title = {"id": "t5", "embed_url": "https://example.com"}  # both can handle
    result = _run(cls.classify(title))
    assert result.strategy.id == STRATEGY_SHIM  # cost_rank=1


def test_classify_falls_back_when_named_strategy_unknown(registry, stub_strategies):
    """A persisted profile naming a never-registered strategy should
    still resolve to *some* strategy, not crash."""
    p = CastProfile(title_id="t6", strategy="containerized")  # not registered
    _run(registry.upsert(p, user_id="alice"))
    cls = CastClassifier(profile_registry=registry, strategies=stub_strategies)
    title = {"id": "t6", "embed_url": "https://example.com"}
    result = _run(cls.classify(title, user_id="alice"))
    assert result.strategy.id in (STRATEGY_SHIM, STRATEGY_PROXY)


def test_classifier_raises_with_no_strategies(registry):
    empty = StrategyRegistry()
    cls = CastClassifier(profile_registry=registry, strategies=empty)
    with pytest.raises(RuntimeError):
        _run(cls.classify({"id": "x"}))


# ── StrategyRegistry mechanics ───────────────────────────────────


def test_strategy_registry_cheapest_first():
    reg = StrategyRegistry()

    class _Cheap(CastStrategy):
        id = "cheap"
        cost_rank = 1
        async def can_handle(self, t, h): return True
        async def prepare(self, t, p): raise NotImplementedError

    class _Expensive(CastStrategy):
        id = "expensive"
        cost_rank = 5
        async def can_handle(self, t, h): return True
        async def prepare(self, t, p): raise NotImplementedError

    # Register in reverse cost order
    reg.register(_Expensive())
    reg.register(_Cheap())

    ranked = [s.id for s in reg.cheapest_first()]
    assert ranked == ["cheap", "expensive"]


def test_strategy_registry_register_requires_id():
    reg = StrategyRegistry()

    class _Bad(CastStrategy):
        id = ""
        cost_rank = 1
        async def can_handle(self, t, h): return True
        async def prepare(self, t, p): raise NotImplementedError

    with pytest.raises(ValueError):
        reg.register(_Bad())


def test_default_strategy_registry_has_shim():
    """The package's auto-registered global registry must contain the
    shim strategy (so the classifier picks it up out of the box)."""
    from augmentum.cast.games.strategies import strategy_registry
    assert strategy_registry.has(STRATEGY_SHIM)
