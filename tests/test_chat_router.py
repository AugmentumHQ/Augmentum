"""Tests for the companion-dispatch → chat-mode bridge.

The fallthrough discipline is the load-bearing thing here: chat must
work identically when the runtime is off, missing, abstaining, or
picking a non-chat subagent. That's the "companion is optional" promise
made structural.

Coverage:

- Flag off → classifier wins regardless of runtime state
- Runtime missing → classifier wins (even with flag on)
- Runtime not started → classifier wins
- companion_dispatch_enabled off → classifier wins (chat is not the
  place to enable dispatch for the first time)
- Dispatch abstains → classifier wins
- Dispatch picks non-chat subagent (build/bug_finder) → classifier wins
- Dispatch picks chat subagent below min_utility → classifier wins
- Dispatch picks chat subagent above min_utility → dispatch wins
- Explicit mode_override always wins (header)
- Telemetry: dispatch.routed_chat event carries both decisions
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_request(text: str = "hello"):
    """Minimal InternalChatRequest stand-in. The router only touches
    request.messages, so we can use a duck-typed object."""
    @dataclass
    class _Msg:
        role: str
        content: str

    @dataclass
    class _Req:
        messages: list

    return _Req(messages=[_Msg(role="user", content=text)])


def _make_classifier(mode_str: str = "passthrough", confidence: float = 0.9):
    """RequestClassifier stand-in returning a fixed ClassificationResult."""
    from augmentum.classifier.router import ClassificationResult, Mode

    class _Classifier:
        def classify(self, *args, **kwargs):
            return ClassificationResult(
                mode=Mode(mode_str),
                confidence=confidence,
                reason=f"test classifier picked {mode_str}",
            )

    return _Classifier()


async def _make_runtime(*, started: bool = True):
    """Real bus + minimal runtime shim."""
    from augmentum.companion_runtime.bus import PresenceBus

    class _FakeRuntime:
        bus = PresenceBus()
        companion_id = "becca"
        _started = started

    return _FakeRuntime()


def _make_app_state(runtime=None):
    class _AppState:
        companion_runtime = runtime

    return _AppState()


def _patch_dispatch(monkeypatch, *, winner_name: str | None,
                     utility: float = 0.7, abstained: bool = False,
                     raise_exc: bool = False):
    """Replace dispatch.decide with a controllable stub."""
    from augmentum.companion_runtime import dispatch

    @dataclass
    class _Candidate:
        name: str
        utility: float

        @property
        def subagent(self):
            return None

    @dataclass
    class _Decision:
        winner: _Candidate | None
        ranked: list
        used_tiebreaker: bool = False
        tiebreaker_rationale: str = ""
        decision_ms: float = 1.0
        abstained: bool = False

    async def _fake_decide(*args, **kwargs):
        if raise_exc:
            raise RuntimeError("dispatch boom")
        if winner_name is None or abstained:
            return _Decision(
                winner=None, ranked=[], abstained=True,
            )
        winner = _Candidate(name=winner_name, utility=utility)
        return _Decision(winner=winner, ranked=[winner])

    monkeypatch.setattr(dispatch, "decide", _fake_decide)


# ── Flag off ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_flag_off_classifier_wins(monkeypatch):
    """The default state: dispatch routing is OFF, classifier picks."""
    from augmentum.companion_runtime.chat_router import resolve_chat_mode
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_dispatch_routes_chat", False)

    classifier = _make_classifier("narrative", 0.85)
    runtime = await _make_runtime()
    app_state = _make_app_state(runtime)

    result = await resolve_chat_mode(
        app_state, _make_request(),
        classifier=classifier, mode_override=None,
    )
    assert result.mode.value == "narrative"
    assert "companion dispatch" not in result.reason


# ── Runtime not available ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_runtime_classifier_wins(monkeypatch):
    """Flag ON but runtime missing — companion is optional."""
    from augmentum.companion_runtime.chat_router import resolve_chat_mode
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_dispatch_routes_chat", True)
    monkeypatch.setattr(_settings, "companion_dispatch_enabled", True)

    classifier = _make_classifier("coder", 0.7)
    app_state = _make_app_state(runtime=None)

    result = await resolve_chat_mode(
        app_state, _make_request(),
        classifier=classifier, mode_override=None,
    )
    assert result.mode.value == "coder"


@pytest.mark.asyncio
async def test_runtime_not_started_classifier_wins(monkeypatch):
    """Runtime instance present but not yet started — fall through."""
    from augmentum.companion_runtime.chat_router import resolve_chat_mode
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_dispatch_routes_chat", True)
    monkeypatch.setattr(_settings, "companion_dispatch_enabled", True)

    classifier = _make_classifier("agentic", 0.6)
    runtime = await _make_runtime(started=False)
    app_state = _make_app_state(runtime)

    result = await resolve_chat_mode(
        app_state, _make_request(),
        classifier=classifier, mode_override=None,
    )
    assert result.mode.value == "agentic"


# ── Dispatch flag off ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_disabled_classifier_wins(monkeypatch):
    """Chat-routing flag on but companion_dispatch_enabled is off.
    Chat is not the place to enable dispatch for the first time."""
    from augmentum.companion_runtime.chat_router import resolve_chat_mode
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_dispatch_routes_chat", True)
    monkeypatch.setattr(_settings, "companion_dispatch_enabled", False)

    classifier = _make_classifier("passthrough", 0.95)
    runtime = await _make_runtime()
    app_state = _make_app_state(runtime)

    result = await resolve_chat_mode(
        app_state, _make_request(),
        classifier=classifier, mode_override=None,
    )
    assert result.mode.value == "passthrough"


# ── Dispatch abstains / errors ────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_abstains_classifier_wins(monkeypatch):
    from augmentum.companion_runtime.chat_router import resolve_chat_mode
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_dispatch_routes_chat", True)
    monkeypatch.setattr(_settings, "companion_dispatch_enabled", True)
    _patch_dispatch(monkeypatch, winner_name=None, abstained=True)

    classifier = _make_classifier("analytical", 0.7)
    runtime = await _make_runtime()
    app_state = _make_app_state(runtime)

    result = await resolve_chat_mode(
        app_state, _make_request(),
        classifier=classifier, mode_override=None,
    )
    assert result.mode.value == "analytical"


@pytest.mark.asyncio
async def test_dispatch_raises_classifier_wins(monkeypatch):
    """Any dispatch exception falls through cleanly — never breaks chat."""
    from augmentum.companion_runtime.chat_router import resolve_chat_mode
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_dispatch_routes_chat", True)
    monkeypatch.setattr(_settings, "companion_dispatch_enabled", True)
    _patch_dispatch(monkeypatch, winner_name="passthrough", raise_exc=True)

    classifier = _make_classifier("narrative", 0.8)
    runtime = await _make_runtime()
    app_state = _make_app_state(runtime)

    result = await resolve_chat_mode(
        app_state, _make_request(),
        classifier=classifier, mode_override=None,
    )
    assert result.mode.value == "narrative"


# ── Non-chat subagent winners fall through ───────────────────────────


@pytest.mark.asyncio
async def test_winner_build_falls_through(monkeypatch):
    """build/bug_finder are autonomous-only — never chat winners."""
    from augmentum.companion_runtime.chat_router import resolve_chat_mode
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_dispatch_routes_chat", True)
    monkeypatch.setattr(_settings, "companion_dispatch_enabled", True)
    _patch_dispatch(monkeypatch, winner_name="build", utility=0.95)

    classifier = _make_classifier("passthrough", 0.8)
    runtime = await _make_runtime()
    app_state = _make_app_state(runtime)

    result = await resolve_chat_mode(
        app_state, _make_request("compile and ship"),
        classifier=classifier, mode_override=None,
    )
    assert result.mode.value == "passthrough"


@pytest.mark.asyncio
async def test_winner_bug_finder_falls_through(monkeypatch):
    from augmentum.companion_runtime.chat_router import resolve_chat_mode
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_dispatch_routes_chat", True)
    monkeypatch.setattr(_settings, "companion_dispatch_enabled", True)
    _patch_dispatch(monkeypatch, winner_name="bug_finder", utility=0.95)

    classifier = _make_classifier("coder", 0.7)
    runtime = await _make_runtime()
    app_state = _make_app_state(runtime)

    result = await resolve_chat_mode(
        app_state, _make_request("find the bug"),
        classifier=classifier, mode_override=None,
    )
    assert result.mode.value == "coder"


# ── Below threshold falls through ────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_below_min_utility_falls_through(monkeypatch):
    from augmentum.companion_runtime.chat_router import resolve_chat_mode
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_dispatch_routes_chat", True)
    monkeypatch.setattr(_settings, "companion_dispatch_enabled", True)
    monkeypatch.setattr(_settings, "companion_dispatch_chat_min_utility", 0.6)
    _patch_dispatch(monkeypatch, winner_name="passthrough", utility=0.3)

    classifier = _make_classifier("narrative", 0.85)
    runtime = await _make_runtime()
    app_state = _make_app_state(runtime)

    result = await resolve_chat_mode(
        app_state, _make_request(),
        classifier=classifier, mode_override=None,
    )
    # Dispatch's pick was below threshold; classifier wins
    assert result.mode.value == "narrative"


# ── Dispatch wins ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_overrides_classifier_above_threshold(monkeypatch):
    """The interesting case — dispatch is the source of truth."""
    from augmentum.companion_runtime.chat_router import resolve_chat_mode
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_dispatch_routes_chat", True)
    monkeypatch.setattr(_settings, "companion_dispatch_enabled", True)
    monkeypatch.setattr(_settings, "companion_dispatch_chat_min_utility", 0.4)
    _patch_dispatch(monkeypatch, winner_name="coder", utility=0.78)

    # Classifier would have picked passthrough; dispatch picks coder.
    classifier = _make_classifier("passthrough", 0.85)
    runtime = await _make_runtime()
    app_state = _make_app_state(runtime)

    result = await resolve_chat_mode(
        app_state, _make_request("refactor this function to be cleaner"),
        classifier=classifier, mode_override=None,
    )
    assert result.mode.value == "coder"
    assert result.metadata.get("source") == "companion_dispatch"
    assert result.metadata.get("classifier_alt") == "passthrough"


# ── Explicit override beats everything ───────────────────────────────


@pytest.mark.asyncio
async def test_explicit_header_override_beats_dispatch(monkeypatch):
    """User explicit intent always wins — same as the classifier's
    own priority chain."""
    from augmentum.companion_runtime.chat_router import resolve_chat_mode
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_dispatch_routes_chat", True)
    monkeypatch.setattr(_settings, "companion_dispatch_enabled", True)
    _patch_dispatch(monkeypatch, winner_name="coder", utility=0.99)

    classifier = _make_classifier("passthrough", 0.7)
    runtime = await _make_runtime()
    app_state = _make_app_state(runtime)

    result = await resolve_chat_mode(
        app_state, _make_request(),
        classifier=classifier, mode_override="narrative",
    )
    assert result.mode.value == "narrative"
    assert "explicit header override" in result.reason


# ── Telemetry ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_emits_routed_chat_event(monkeypatch):
    """Every dispatch consultation emits the comparison event so we
    can A/B telemetry classifier vs dispatch picks over real traffic."""
    from augmentum.companion_runtime.chat_router import resolve_chat_mode
    from augmentum.config import settings as _settings

    monkeypatch.setattr(_settings, "companion_dispatch_routes_chat", True)
    monkeypatch.setattr(_settings, "companion_dispatch_enabled", True)
    monkeypatch.setattr(_settings, "companion_dispatch_chat_min_utility", 0.4)
    _patch_dispatch(monkeypatch, winner_name="coder", utility=0.8)

    classifier = _make_classifier("passthrough", 0.85)
    runtime = await _make_runtime()
    app_state = _make_app_state(runtime)

    sub = await runtime.bus.subscribe("dispatch.routed_chat", slice_key="t")
    captured: list[dict] = []

    async def _drain():
        try:
            ev = await asyncio.wait_for(sub.queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            return
        if ev is not None:
            captured.append({"topic": ev.topic, "payload": ev.payload})

    drain_task = asyncio.create_task(_drain())

    try:
        await resolve_chat_mode(
            app_state, _make_request("test"),
            classifier=classifier, mode_override=None,
            session_id="s_test",
        )
        await drain_task
        assert len(captured) == 1
        payload = captured[0]["payload"]
        assert payload["classifier_mode"] == "passthrough"
        assert payload["dispatch_winner"] == "coder"
        assert payload["session_id"] == "s_test"
    finally:
        await runtime.bus.unsubscribe(sub)
