"""Tests for the Phase-4 telemetry-driven demotion loop.

Pins:
  - demoter accumulates per-tick window_ms until the judging window
  - healthy cast (input reaching, no unreachable iframe) is NOT demoted
  - cross-origin shim cast (frames + unreachable iframe) IS demoted
  - demotion before the user actually pushes input is withheld
  - demotion is latched (recorded at most once per surface)
  - a proxy cast that's still failing is logged, not thrashed
  - no-row title gets a telemetry-classified profile carrying the
    failing strategy so the classifier has an anchor to promote from
  - END-TO-END: demoter mark → classifier promotes shim → proxy
"""
from __future__ import annotations

import asyncio

import pytest

from augmentum.cast.games.classifier import CastClassifier
from augmentum.cast.games.models import (
    CLASSIFIED_TELEMETRY,
    STRATEGY_PROXY,
    STRATEGY_SHIM,
    CastProfile,
    HostCapabilities,
)
from augmentum.cast.games.registry import CastProfileRegistry
from augmentum.cast.games.strategies.base import CastStrategy, StrategyRegistry
from augmentum.cast.games.telemetry import TelemetryDemoter


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
def demoter(registry):
    # Fixed clock + small thresholds keep the tests deterministic and
    # short: 3 frames + 1s window is enough to "judge" in-test.
    return TelemetryDemoter(
        profile_registry=registry,
        min_frames=3,
        min_window_ms=1000.0,
        now=lambda: 1_000_000.0,
    )


def _tick(*, title_id="game1", strategy="shim", frames=2, unreachable=0,
          window_ms=600.0):
    return {
        "type": "augmentum.input_telemetry",
        "title_id": title_id,
        "strategy": strategy,
        "frames_received": frames,
        "dispatches": frames,
        "reachable_targets": 0 if unreachable else 1,
        "unreachable_targets": unreachable,
        "window_ms": window_ms,
    }


# ── Accumulation + windowing ─────────────────────────────────────


def test_accumulates_until_window_elapses(demoter):
    # One short tick — under the 1000ms window — should just accumulate.
    verdict = _run(demoter.on_telemetry("u1", _tick(window_ms=600, frames=2)))
    assert verdict == "accumulating"


def test_skips_when_no_title(demoter):
    verdict = _run(demoter.on_telemetry("u1", _tick(title_id="")))
    assert verdict == "skipped"


# ── Healthy vs failing ───────────────────────────────────────────


def test_healthy_same_origin_not_demoted(demoter, registry):
    # Window elapses with frames but NO unreachable iframe → reaching.
    _run(demoter.on_telemetry("u1", _tick(frames=2, unreachable=0, window_ms=600)))
    verdict = _run(demoter.on_telemetry(
        "u1", _tick(frames=5, unreachable=0, window_ms=600)))
    assert verdict == "healthy"
    assert _run(registry.get("game1", user_id="u1")) is None


def test_cross_origin_shim_is_demoted(demoter, registry):
    # Frames climbing + an unreachable cross-origin iframe on the cheap
    # shim → demote.
    _run(demoter.on_telemetry("u1", _tick(frames=2, unreachable=1, window_ms=600)))
    verdict = _run(demoter.on_telemetry(
        "u1", _tick(frames=5, unreachable=1, window_ms=600)))
    assert verdict == "demoted"
    profile = _run(registry.get("game1", user_id="u1"))
    assert profile is not None
    assert profile.failed_at > 0
    assert profile.classified_by == CLASSIFIED_TELEMETRY
    assert profile.strategy == STRATEGY_SHIM


def test_no_demotion_before_user_pushes_input(demoter, registry):
    # Window elapses but frames stay under min_frames (user idle) +
    # unreachable iframe present → withhold judgement, treat as healthy.
    verdict = _run(demoter.on_telemetry(
        "u1", _tick(frames=1, unreachable=1, window_ms=1200)))
    assert verdict == "healthy"
    assert _run(registry.get("game1", user_id="u1")) is None


def test_demotion_is_latched(demoter, registry):
    # First elapsed window demotes; subsequent windows don't re-write.
    _run(demoter.on_telemetry("u1", _tick(frames=5, unreachable=1, window_ms=1200)))
    first = _run(registry.get("game1", user_id="u1"))
    assert first is not None and first.failed_at > 0
    verdict = _run(demoter.on_telemetry(
        "u1", _tick(frames=9, unreachable=1, window_ms=1200)))
    assert verdict == "skipped"


def test_failing_proxy_is_logged_not_thrashed(demoter, registry):
    # A proxy cast still showing an unreachable iframe has nowhere cheaper
    # to escalate (containerized is Phase 5) → skip, no profile churn.
    verdict = _run(demoter.on_telemetry(
        "u1", _tick(strategy=STRATEGY_PROXY, frames=9, unreachable=1,
                    window_ms=1200)))
    assert verdict == "skipped"
    assert _run(registry.get("game1", user_id="u1")) is None


def test_existing_profile_gets_marked_failed(demoter, registry):
    # When a row already exists, demotion stamps failed_at in place
    # (doesn't overwrite the user's strategy/embed_url).
    _run(registry.upsert(CastProfile(
        title_id="game1", strategy=STRATEGY_SHIM, embed_url="https://e.com/g",
    ), user_id="u1"))
    _run(demoter.on_telemetry("u1", _tick(frames=5, unreachable=1, window_ms=1200)))
    profile = _run(registry.get("game1", user_id="u1"))
    assert profile.failed_at > 0
    assert profile.embed_url == "https://e.com/g"  # preserved


# ── End-to-end: demotion actually changes the next classify ──────


def test_end_to_end_demotion_promotes_shim_to_proxy(registry):
    """The whole point: a demoted shim cast classifies as proxy next.

    Uses the real clock (not the fixed-clock fixture) so the stamped
    ``failed_at`` lands inside the classifier's demotion window — the
    classifier reads ``time.time()`` internally and the two must agree.
    """
    demoter = TelemetryDemoter(
        profile_registry=registry, min_frames=3, min_window_ms=1000.0,
    )

    class _Shim(CastStrategy):
        id = STRATEGY_SHIM
        cost_rank = 1

        async def can_handle(self, title, host):
            return True

        async def prepare(self, title, profile):  # pragma: no cover
            raise NotImplementedError

    class _Proxy(CastStrategy):
        id = STRATEGY_PROXY
        cost_rank = 2

        async def can_handle(self, title, host):
            return True

        async def prepare(self, title, profile):  # pragma: no cover
            raise NotImplementedError

    strat_reg = StrategyRegistry()
    strat_reg.register(_Shim())
    strat_reg.register(_Proxy())
    classifier = CastClassifier(
        profile_registry=registry, strategies=strat_reg,
    )
    title = {"id": "game1", "embed_url": "https://e.com/g"}

    # Before demotion: no row → cheapest qualifying = shim.
    before = _run(classifier.classify(title, HostCapabilities(), user_id="u1"))
    assert before.strategy.id == STRATEGY_SHIM
    assert before.source == "default"

    # Telemetry demotes the shim cast.
    verdict = _run(demoter.on_telemetry(
        "u1", _tick(frames=5, unreachable=1, window_ms=1200)))
    assert verdict == "demoted"

    # After demotion: the recent failed_at promotes shim → proxy.
    after = _run(classifier.classify(title, HostCapabilities(), user_id="u1"))
    assert after.strategy.id == STRATEGY_PROXY
    assert after.source == "registry_promoted"
