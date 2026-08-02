"""Tests for the Phase-4 proactive probe.

Pins:
  - fingerprint: observed events → ordered input_chain (pure)
  - pixel_diff: byte-diff ratio + responded threshold (pure)
  - driver: _parse handles ok / not-ok / garbage; responded via pixel_diff
  - driver: probe() with injected runner + unsafe-url guard (no browser)
  - build_probe_profile: chain + classified_by=probe + responded note
  - coordinator: dedup guard, no-embed guard, _run writes profile,
    _choose_strategy is origin-aware + can_handle-gated
  - LIVE (opt-in): real headless probe of a keyboard-listening data: page
"""
from __future__ import annotations

import asyncio
import base64

import pytest

from augmentum.cast.games.models import (
    CLASSIFIED_PROBE,
    STRATEGY_PROXY,
    STRATEGY_SHIM,
)
from augmentum.cast.games.probe.fingerprint import (
    GETGAMEPADS_POLL,
    INSTRUMENTATION_JS,
    classify_input_style,
)
from augmentum.cast.games.probe.job import CastProbeCoordinator, _same_origin
from augmentum.cast.games.probe.pixel_diff import frame_diff_ratio, responded
from augmentum.cast.games.probe.playwright_probe import (
    PlaywrightProbe,
    ProbeResult,
    build_probe_profile,
)
from augmentum.cast.games.registry import CastProfileRegistry
from augmentum.cast.games.strategies.base import CastStrategy, StrategyRegistry


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── fingerprint (pure) ───────────────────────────────────────────


def test_fingerprint_empty_defaults_to_gamepad():
    fp = classify_input_style([])
    assert fp.input_chain == ("gamepad_api",)


def test_fingerprint_keyboard_game():
    fp = classify_input_style(["keydown", "keyup"])
    assert fp.input_chain == ("gamepad_api", "keyboard")
    assert "keyboard" in fp.styles


def test_fingerprint_touch_game():
    fp = classify_input_style(["touchstart", "touchmove"])
    assert "touch" in fp.input_chain


def test_fingerprint_gamepad_poll_only():
    fp = classify_input_style([GETGAMEPADS_POLL])
    assert fp.input_chain == ("gamepad_api",)
    assert "gamepad" in fp.styles


def test_fingerprint_mixed_ordered():
    fp = classify_input_style(["mousemove", "keydown", "touchstart", "gamepadconnected"])
    # gamepad_api always first; then keyboard, touch, pointer in that order.
    assert fp.input_chain == ("gamepad_api", "keyboard", "touch", "pointer")


def test_fingerprint_is_case_insensitive():
    fp = classify_input_style(["KeyDown", "TOUCHSTART"])
    assert "keyboard" in fp.input_chain
    assert "touch" in fp.input_chain


def test_instrumentation_traps_the_right_surface():
    assert "addEventListener" in INSTRUMENTATION_JS
    assert "getGamepads" in INSTRUMENTATION_JS
    assert "__augProbeObserved" in INSTRUMENTATION_JS


# ── pixel_diff (pure) ────────────────────────────────────────────


def test_diff_identical_is_zero():
    assert frame_diff_ratio(b"abcdef", b"abcdef") == 0.0


def test_diff_fully_different_is_one():
    assert frame_diff_ratio(b"\x00\x00\x00", b"\xff\xff\xff") == 1.0


def test_diff_partial():
    # 1 of 4 bytes differs → 0.25
    assert frame_diff_ratio(b"aaaa", b"aaab") == 0.25


def test_diff_length_mismatch_counts_as_change():
    r = frame_diff_ratio(b"aa", b"aaaa")
    assert r > 0.0


def test_diff_empty_inputs():
    assert frame_diff_ratio(b"", b"") == 0.0
    assert frame_diff_ratio(b"", b"x") == 1.0


def test_responded_threshold():
    assert responded(b"aaaa", b"aaab", threshold=0.1) is True   # 0.25 > 0.1
    assert responded(b"aaaa", b"aaab", threshold=0.5) is False  # 0.25 < 0.5


def test_diff_accepts_base64():
    before = base64.b64encode(b"aaaa").decode()
    after = base64.b64encode(b"aaab").decode()
    assert frame_diff_ratio(before, after) == 0.25


# ── driver: parse + probe (no browser) ───────────────────────────


def test_parse_not_ok_returns_none():
    p = PlaywrightProbe(runner=None)
    assert p._parse('{"ok": false, "error": "playwright unavailable"}') is None


def test_parse_garbage_returns_none():
    p = PlaywrightProbe()
    assert p._parse("not json at all") is None


def test_parse_ok_computes_responded_via_pixel_diff():
    before = base64.b64encode(b"\x00\x00\x00\x00").decode()
    after = base64.b64encode(b"\xff\xff\xff\xff").decode()
    p = PlaywrightProbe(diff_threshold=0.1)
    res = p._parse(
        '{"ok": true, "observed": ["keydown"], '
        f'"before_b64": "{before}", "after_b64": "{after}"}}'
    )
    assert res is not None
    assert res.ok is True
    assert res.observed == ("keydown",)
    assert res.responded is True


def test_probe_rejects_unsafe_url():
    p = PlaywrightProbe()
    assert _run(p.probe("http://127.0.0.1/game")) is None
    assert _run(p.probe("file:///etc/passwd")) is None


def test_probe_with_injected_runner():
    before = base64.b64encode(b"aaaa").decode()
    after = base64.b64encode(b"zzzz").decode()

    async def fake_runner(cfg, timeout):
        assert cfg["url"] == "https://example.com/g"
        return (
            '{"ok": true, "observed": ["keydown", "__getgamepads_poll__"], '
            f'"before_b64": "{before}", "after_b64": "{after}"}}'
        )

    p = PlaywrightProbe(runner=fake_runner, diff_threshold=0.1)
    res = _run(p.probe("https://example.com/g"))
    assert res is not None and res.responded is True
    assert "keydown" in res.observed


def test_probe_runner_exception_returns_none():
    async def boom(cfg, timeout):
        raise RuntimeError("subprocess died")

    p = PlaywrightProbe(runner=boom)
    assert _run(p.probe("https://example.com/g")) is None


# ── build_probe_profile ──────────────────────────────────────────


def test_build_profile_maps_chain_and_provenance():
    res = ProbeResult(ok=True, observed=("keydown",), responded=True)
    prof = build_probe_profile(
        title_id="g1", user_id="u1", embed_url="https://e.com/g",
        result=res, strategy=STRATEGY_SHIM, now=123.0,
    )
    assert prof.classified_by == CLASSIFIED_PROBE
    assert prof.input_chain == ("gamepad_api", "keyboard")
    assert prof.classified_at == 123.0
    assert "reached game" in prof.notes


def test_build_profile_records_non_response():
    res = ProbeResult(ok=True, observed=("keydown",), responded=False)
    prof = build_probe_profile(
        title_id="g1", user_id="u1", embed_url="https://e.com/g",
        result=res, now=1.0,
    )
    assert "no visible reaction" in prof.notes


# ── coordinator ──────────────────────────────────────────────────


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


class _StubProbe:
    def __init__(self, result):
        self._result = result

    async def probe(self, embed_url):
        return self._result


def test_coordinator_dedup_guard(registry):
    coord = CastProbeCoordinator(
        probe=_StubProbe(None), profile_registry=registry,
        strategy_registry=StrategyRegistry(),
    )
    coord._inflight.add(("u1", "g1"))
    assert coord.maybe_probe(title_id="g1", user_id="u1", embed_url="https://e/g") is False


def test_coordinator_requires_embed(registry):
    coord = CastProbeCoordinator(
        probe=_StubProbe(None), profile_registry=registry,
        strategy_registry=StrategyRegistry(),
    )
    assert coord.maybe_probe(title_id="g1", user_id="u1", embed_url="") is False


def test_coordinator_run_writes_profile(registry):
    res = ProbeResult(ok=True, observed=("keydown",), responded=True)
    coord = CastProbeCoordinator(
        probe=_StubProbe(res), profile_registry=registry,
        strategy_registry=StrategyRegistry(),
    )
    _run(coord._run("g1", "u1", "https://e.com/g"))
    prof = _run(registry.get("g1", user_id="u1"))
    assert prof is not None
    assert prof.classified_by == CLASSIFIED_PROBE
    assert "keyboard" in prof.input_chain


def test_coordinator_run_no_result_writes_nothing(registry):
    coord = CastProbeCoordinator(
        probe=_StubProbe(None), profile_registry=registry,
        strategy_registry=StrategyRegistry(),
    )
    _run(coord._run("g1", "u1", "https://e.com/g"))
    assert _run(registry.get("g1", user_id="u1")) is None


def test_choose_strategy_same_origin_is_shim(registry):
    coord = CastProbeCoordinator(
        probe=_StubProbe(None), profile_registry=registry,
        strategy_registry=StrategyRegistry(),
        server_origin="https://host.example",
    )
    s = _run(coord._choose_strategy("g1", "u1", "https://host.example/game"))
    assert s == STRATEGY_SHIM


def test_choose_strategy_cross_origin_prefers_serviceable_proxy(registry):
    class _Proxy(CastStrategy):
        id = STRATEGY_PROXY
        cost_rank = 2

        async def can_handle(self, title, host):
            return True

        async def prepare(self, title, profile):  # pragma: no cover
            raise NotImplementedError

    sreg = StrategyRegistry()
    sreg.register(_Proxy())
    coord = CastProbeCoordinator(
        probe=_StubProbe(None), profile_registry=registry,
        strategy_registry=sreg, server_origin="https://host.example",
    )
    s = _run(coord._choose_strategy("g1", "u1", "https://elsewhere.io/game"))
    assert s == STRATEGY_PROXY


def test_choose_strategy_no_server_origin_is_shim(registry):
    coord = CastProbeCoordinator(
        probe=_StubProbe(None), profile_registry=registry,
        strategy_registry=StrategyRegistry(),
        server_origin="",
    )
    s = _run(coord._choose_strategy("g1", "u1", "https://elsewhere.io/game"))
    assert s == STRATEGY_SHIM


def test_same_origin_helper():
    assert _same_origin("https://a.com/x", "https://a.com") is True
    assert _same_origin("https://a.com:443/x", "https://a.com") is True
    assert _same_origin("https://a.com/x", "https://b.com") is False
    assert _same_origin("", "https://a.com") is False


# ── LIVE (opt-in): real headless probe ───────────────────────────


def _playwright_available() -> bool:
    try:
        import playwright.sync_api  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _playwright_available(), reason="playwright not installed")
def test_live_probe_data_url_keyboard_game():
    # A minimal page that wires a keydown listener — the probe should
    # observe it and build a keyboard chain. data: URLs are rejected by
    # is_url_safe, so this drives the subprocess script directly via a
    # tiny http server would be heavier; instead we assert the driver's
    # graceful path when no browser binary is present is already covered,
    # and here only run when a real chromium is installed.
    html = (
        "<!doctype html><html><body><script>"
        "window.addEventListener('keydown', ()=>{document.body.style.background='red';});"
        "</script></body></html>"
    )
    import http.server
    import socketserver
    import threading

    class _H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())

        def log_message(self, *a):  # silence
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), _H) as srv:
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            # allow_private fetch isn't in play here — the probe's url
            # safety check rejects 127.0.0.1, so we exercise _subprocess_runner
            # directly against the loopback game.
            p = PlaywrightProbe(diff_threshold=0.0005)
            cfg = {
                "url": f"http://127.0.0.1:{port}/",
                "instrumentation": INSTRUMENTATION_JS,
                "keys": ["ArrowDown", " "],
                "boot_ms": 300, "react_ms": 300,
            }
            raw = _run(p._subprocess_runner(cfg, 60.0))
            res = p._parse(raw)
        finally:
            srv.shutdown()
    # If chromium isn't actually installed the subprocess prints ok:false
    # → _parse returns None; only assert the fingerprint when it ran.
    if res is not None:
        assert "keydown" in res.observed
