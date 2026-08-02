"""Unit tests for the in-container ``agent-bridge.py`` daemon.

The script lives under ``services/game-stream/scripts/`` because it is
a deployment artifact shipped into the streamed-emulator image, not a
Python package importable through the normal ``augmentum.*`` namespace.
We load it via :mod:`importlib` with the ``evdev`` and ``websockets``
modules mocked so the suite runs on hosts that don't have the kernel
headers / system libs those packages need (Windows dev machines,
CI runners without ``apt install python3-evdev``).
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

# ── evdev shim ──────────────────────────────────────────────────────
#
# python-evdev compiles a C extension against linux/input.h. The bridge
# only uses a handful of code constants and ``UInput`` / ``AbsInfo``,
# so we synthesise just enough surface for the import to succeed and
# for our recording fake to drop in.

_BTN_BASE = 0x130
_ABS_BASE = 0x00


class _FakeECodes:
    EV_KEY = 0x01
    EV_ABS = 0x03
    BUS_USB = 0x03
    BTN_SOUTH = _BTN_BASE + 0
    BTN_EAST = _BTN_BASE + 1
    BTN_NORTH = _BTN_BASE + 3
    BTN_WEST = _BTN_BASE + 4
    BTN_TL = _BTN_BASE + 6
    BTN_TR = _BTN_BASE + 7
    BTN_SELECT = _BTN_BASE + 10
    BTN_START = _BTN_BASE + 11
    BTN_MODE = _BTN_BASE + 12
    BTN_THUMBL = _BTN_BASE + 13
    BTN_THUMBR = _BTN_BASE + 14
    ABS_X = _ABS_BASE + 0
    ABS_Y = _ABS_BASE + 1
    ABS_Z = _ABS_BASE + 2
    ABS_RX = _ABS_BASE + 3
    ABS_RY = _ABS_BASE + 4
    ABS_RZ = _ABS_BASE + 5
    ABS_HAT0X = _ABS_BASE + 16
    ABS_HAT0Y = _ABS_BASE + 17


class _FakeAbsInfo:
    def __init__(self, *, value: int, min: int, max: int,
                 fuzz: int, flat: int, resolution: int) -> None:
        self.value = value
        self.min = min
        self.max = max
        self.fuzz = fuzz
        self.flat = flat
        self.resolution = resolution


class RecordingUInput:
    """Fake :class:`evdev.UInput` that records all writes for assertion.

    The order of (kind, code, value) tuples is the same order the bridge
    pushed them, so a test can assert "press, sleep, release" without
    caring about syn() counts.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.events: list[tuple[int, int, int]] = []
        self.syn_count = 0
        # Mimic the real .device.path attribute the bridge logs at start.
        self.device = types.SimpleNamespace(path="/dev/input/event99")

    def write(self, kind: int, code: int, value: int) -> None:
        self.events.append((kind, code, value))

    def syn(self) -> None:
        self.syn_count += 1

    def close(self) -> None:
        pass


def _install_evdev_shim() -> None:
    """Install a stub ``evdev`` module so the bridge can import. The
    test then re-monkeypatches the bridge module's UInput reference per
    test so we get a fresh recorder."""

    mod = types.ModuleType("evdev")
    mod.UInput = RecordingUInput  # type: ignore[attr-defined]
    mod.AbsInfo = _FakeAbsInfo  # type: ignore[attr-defined]
    mod.ecodes = _FakeECodes  # type: ignore[attr-defined]
    sys.modules["evdev"] = mod


def _install_websockets_shim() -> None:
    """Stub the ``websockets`` package surface the bridge imports.

    Only the symbols touched at import time matter -- the bridge does
    not exercise the WS path in these unit tests.
    """

    if "websockets" in sys.modules:
        return
    root = types.ModuleType("websockets")
    sys.modules["websockets"] = root

    asyncio_mod = types.ModuleType("websockets.asyncio")
    sys.modules["websockets.asyncio"] = asyncio_mod
    root.asyncio = asyncio_mod  # type: ignore[attr-defined]

    client_mod = types.ModuleType("websockets.asyncio.client")
    sys.modules["websockets.asyncio.client"] = client_mod
    asyncio_mod.client = client_mod  # type: ignore[attr-defined]

    class _ClientConnection:  # placeholder type for the bridge's typing
        pass

    client_mod.ClientConnection = _ClientConnection  # type: ignore[attr-defined]
    client_mod.connect = lambda *a, **kw: None  # type: ignore[attr-defined]

    exc_mod = types.ModuleType("websockets.exceptions")
    sys.modules["websockets.exceptions"] = exc_mod

    class _ConnectionClosed(Exception):
        pass

    exc_mod.ConnectionClosed = _ConnectionClosed  # type: ignore[attr-defined]
    root.exceptions = exc_mod  # type: ignore[attr-defined]


def _load_bridge_module() -> Any:
    """Load services/game-stream/scripts/agent-bridge.py as a module."""

    _install_evdev_shim()
    _install_websockets_shim()

    # tests/ -> game_agent/ -> augmentum/ (inner) -> augmentum/ (repo root)
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "services" / "game-stream" / "scripts" / "agent-bridge.py"
    assert script_path.exists(), f"bridge script missing at {script_path}"

    spec = importlib.util.spec_from_file_location("agent_bridge", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def bridge() -> Any:
    """One bridge module per test so module-level state doesn't leak."""

    return _load_bridge_module()


# ── _press: button semantics ────────────────────────────────────────


@pytest.mark.asyncio
async def test_press_button_a_emits_down_then_up(bridge: Any) -> None:
    """@example: ``button_a`` -> KEY_DOWN(BTN_SOUTH), KEY_UP(BTN_SOUTH)."""

    ui = RecordingUInput()
    dispatched = await bridge._press(ui, "button_a", 32)
    assert dispatched is True
    assert ui.events == [
        (bridge.ecodes.EV_KEY, bridge.ecodes.BTN_SOUTH, 1),
        (bridge.ecodes.EV_KEY, bridge.ecodes.BTN_SOUTH, 0),
    ]
    # One syn after press, one after release.
    assert ui.syn_count == 2


@pytest.mark.asyncio
async def test_press_button_b_uses_btn_east(bridge: Any) -> None:
    """@example: ``button_b`` maps to BTN_EAST (XInput B)."""

    ui = RecordingUInput()
    await bridge._press(ui, "button_b", 16)
    codes_pressed = {e[1] for e in ui.events}
    assert bridge.ecodes.BTN_EAST in codes_pressed


# ── _press: d-pad via HAT axis ──────────────────────────────────────


@pytest.mark.asyncio
async def test_press_dpad_up_deflects_then_neutrals(bridge: Any) -> None:
    """@example: ``dpad_up`` -> ABS_HAT0Y=-1, then ABS_HAT0Y=0."""

    ui = RecordingUInput()
    dispatched = await bridge._press(ui, "dpad_up", 50)
    assert dispatched is True
    assert ui.events == [
        (bridge.ecodes.EV_ABS, bridge.ecodes.ABS_HAT0Y, -1),
        (bridge.ecodes.EV_ABS, bridge.ecodes.ABS_HAT0Y, 0),
    ]


@pytest.mark.asyncio
async def test_press_dpad_right_uses_hat0x_positive(bridge: Any) -> None:
    """@example: ``dpad_right`` -> ABS_HAT0X=+1."""

    ui = RecordingUInput()
    await bridge._press(ui, "dpad_right", 50)
    assert ui.events[0] == (bridge.ecodes.EV_ABS, bridge.ecodes.ABS_HAT0X, 1)


# ── _press: triggers ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_press_trigger_l_full_deflection(bridge: Any) -> None:
    """@example: ``trigger_l`` -> ABS_Z=32767 (pressed), then ABS_Z=0."""

    ui = RecordingUInput()
    await bridge._press(ui, "trigger_l", 50)
    assert ui.events == [
        (bridge.ecodes.EV_ABS, bridge.ecodes.ABS_Z, 32767),
        (bridge.ecodes.EV_ABS, bridge.ecodes.ABS_Z, 0),
    ]


# ── _press: analog sticks ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_press_stick_l_left_negative_abs_x(bridge: Any) -> None:
    """@example: ``stick_l_left`` -> ABS_X=-32767, then ABS_X=0."""

    ui = RecordingUInput()
    await bridge._press(ui, "stick_l_left", 50)
    assert ui.events == [
        (bridge.ecodes.EV_ABS, bridge.ecodes.ABS_X, -32767),
        (bridge.ecodes.EV_ABS, bridge.ecodes.ABS_X, 0),
    ]


# ── _press: edge cases ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_press_unknown_semantic_returns_false(bridge: Any) -> None:
    """@example: an unmapped semantic is reported but never dispatched."""

    ui = RecordingUInput()
    dispatched = await bridge._press(ui, "not_a_real_button", 50)
    assert dispatched is False
    assert ui.events == []


@pytest.mark.asyncio
async def test_press_clamps_excessive_duration(bridge: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """@example: a duration of 30s is clamped to 5s before sleeping.

    ROOT CAUSE:
    An LLM-emitted plan can carry nonsense like duration_ms=30000.
    Without clamping the bridge pins the action worker for half a
    minute per button press, starving the agent's planning loop.
    The fix clamps duration to [16, 5000] ms.
    """

    captured: list[float] = []

    async def _fake_sleep(s: float) -> None:
        captured.append(s)

    monkeypatch.setattr(bridge.asyncio, "sleep", _fake_sleep)

    ui = RecordingUInput()
    await bridge._press(ui, "button_a", 30_000)
    assert captured == [5.0]


@pytest.mark.asyncio
async def test_press_floors_tiny_duration(bridge: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """@example: a duration of 1ms is floored to 16ms so emulators see the press."""

    captured: list[float] = []

    async def _fake_sleep(s: float) -> None:
        captured.append(s)

    monkeypatch.setattr(bridge.asyncio, "sleep", _fake_sleep)

    ui = RecordingUInput()
    await bridge._press(ui, "button_a", 1)
    assert captured == [0.016]


def test_button_map_covers_default_semantic_vocabulary(bridge: Any) -> None:
    """@example: every semantic in the default emulator vocab is mappable.

    Cross-checks the bridge's button/dpad/trigger maps against the
    list :mod:`augmentum.game_agent.surfaces.emulator` advertises as
    the default vocabulary. If they drift, the bridge silently drops
    inputs the agent thinks it can emit.
    """

    # Copied from the docstring of the deprecated EmulatorAdapter to
    # keep the test independent of that module continuing to exist.
    default_semantics = {
        "button_a", "button_b", "button_x", "button_y",
        "dpad_up", "dpad_down", "dpad_left", "dpad_right",
        "shoulder_l", "shoulder_r",
        "trigger_l", "trigger_r",
        "start", "select",
    }

    mapped = (
        set(bridge._BUTTON_MAP)
        | set(bridge._DPAD_MAP)
        | set(bridge._TRIGGER_MAP)
        | set(bridge._STICK_MAP)
    )
    missing = default_semantics - mapped
    assert not missing, f"bridge cannot dispatch: {missing}"


# ── _read_loop: input_ack contract ──────────────────────────────────


class _StubWS:
    """Minimal WS surface for the read-loop test.

    Implements ``__aiter__`` (yields preset frames), ``send`` (records
    outbound payloads), and nothing else. The bridge only calls those
    two on the connection object.
    """

    def __init__(self, inbound: list[str]) -> None:
        self._inbound = list(inbound)
        self.sent: list[str] = []

    def __aiter__(self) -> _StubWS:
        return self

    async def __anext__(self) -> str:
        if not self._inbound:
            raise StopAsyncIteration
        return self._inbound.pop(0)

    async def send(self, payload: str) -> None:
        self.sent.append(payload)


@pytest.mark.asyncio
async def test_read_loop_acks_each_dispatched_action(bridge: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """@example: every action frame with a request_id gets an input_ack reply.

    The BridgedAdapter on the augmentum side waits on these acks before
    returning from ``resolver.apply()``; missing acks surface as
    ``input_ack_timeout`` warnings and stall the action worker.
    """

    import json as _json

    async def _no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr(bridge.asyncio, "sleep", _no_sleep)

    inbound = [
        _json.dumps({
            "action": "button_a", "duration_ms": 32, "request_id": "act-1",
        }),
        _json.dumps({
            "action": "dpad_up", "duration_ms": 50, "request_id": "act-2",
        }),
    ]
    ws = _StubWS(inbound)
    ui = RecordingUInput()
    stop = asyncio.Event()
    await bridge._read_loop(ws, ui, stop)

    decoded = [_json.loads(s) for s in ws.sent]
    assert [d["kind"] for d in decoded] == ["event", "event"]
    assert [d["data"]["event"] for d in decoded] == ["input_ack", "input_ack"]
    assert [d["data"]["request_id"] for d in decoded] == ["act-1", "act-2"]
    assert all(d["data"]["dispatched"] is True for d in decoded)


@pytest.mark.asyncio
async def test_read_loop_acks_unknown_semantic_with_dispatched_false(
    bridge: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@example: unknown semantic still ACKs so BridgedAdapter doesn't time out."""

    import json as _json

    async def _no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr(bridge.asyncio, "sleep", _no_sleep)

    inbound = [
        _json.dumps({
            "action": "wiggle_ears", "duration_ms": 50, "request_id": "act-7",
        }),
    ]
    ws = _StubWS(inbound)
    ui = RecordingUInput()
    stop = asyncio.Event()
    await bridge._read_loop(ws, ui, stop)

    assert len(ws.sent) == 1
    decoded = _json.loads(ws.sent[0])
    assert decoded["data"]["event"] == "input_ack"
    assert decoded["data"]["request_id"] == "act-7"
    assert decoded["data"]["dispatched"] is False
