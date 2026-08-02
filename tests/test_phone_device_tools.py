"""Phone-as-capability-provider: presence + the bluetooth round-trip.

Covers the cert-free server↔phone seam end to end (minus the APK):

  * NotificationHub device tagging + targeted send (a phone command
    reaches ONLY the phone's connection, never a web tab)
  * DeviceCommandBus request/response correlation, plus the
    not_connected / timeout / no_hub degrade paths
  * presence: an attached "android" connection surfaces as a device
    slot the companion prompt can see
  * device.bluetooth_list verb: gated off, signed-out, the three
    unreachable paths, and a real device list → spoken summary + data
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from augmentum.notifications.device_bus import DeviceCommandBus, get_device_bus
from augmentum.notifications.hub import NotificationHub


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(text)


class _StubHub:
    """Minimal hub for bus tests — records sends, returns a fixed count."""

    def __init__(self, delivered: int = 1) -> None:
        self._delivered = delivered
        self.sent: list[dict] = []

    async def send_to_device(self, *, user_id, device_type, payload):
        self.sent.append(payload)
        return self._delivered


# ── NotificationHub: device tagging + targeted send ──────────────────


@pytest.mark.asyncio
async def test_hub_targets_only_the_phone():
    hub = NotificationHub()
    ws_phone, ws_web = _FakeWS(), _FakeWS()
    a_phone = await hub.attach(ws=ws_phone, user_id="u1")
    await hub.attach(ws=ws_web, user_id="u1")  # untagged web tab
    await hub.set_device_type(a_phone.connection_id, "Android")  # case-folds

    assert hub.device_types("u1") == {"android"}
    n = await hub.send_to_device(
        user_id="u1", device_type="android", payload={"type": "device_command"},
    )
    assert n == 1
    assert ws_phone.sent and not ws_web.sent


@pytest.mark.asyncio
async def test_hub_no_device_no_send():
    hub = NotificationHub()
    await hub.attach(ws=_FakeWS(), user_id="u1")  # untagged
    assert hub.device_types("u1") == set()
    n = await hub.send_to_device(
        user_id="u1", device_type="android", payload={"x": 1},
    )
    assert n == 0


@pytest.mark.asyncio
async def test_set_device_type_missing_connection_noop():
    hub = NotificationHub()
    await hub.set_device_type("notif-conn-999", "android")  # must not raise
    assert hub.device_types("u1") == set()


# ── DeviceCommandBus: correlation + degrade paths ────────────────────


@pytest.mark.asyncio
async def test_bus_request_resolves():
    hub = _StubHub(delivered=1)
    app_state = SimpleNamespace(notification_hub=hub)
    bus = get_device_bus(app_state)

    async def _answer():
        for _ in range(1000):
            if hub.sent:
                break
            await asyncio.sleep(0)
        rid = hub.sent[-1]["request_id"]
        bus.resolve(rid, {"ok": True, "devices": [{"name": "Car"}]})

    task = asyncio.create_task(_answer())
    result = await bus.request(user_id="u1", action="bluetooth_list", timeout=2.0)
    await task
    assert result["ok"] is True
    assert result["devices"][0]["name"] == "Car"
    assert bus.pending_count() == 0  # slot reclaimed


@pytest.mark.asyncio
async def test_bus_not_connected():
    hub = _StubHub(delivered=0)
    bus = DeviceCommandBus(SimpleNamespace(notification_hub=hub))
    result = await bus.request(user_id="u1", action="bluetooth_list", timeout=1.0)
    assert result == {"ok": False, "error": "not_connected"}


@pytest.mark.asyncio
async def test_bus_no_hub():
    bus = DeviceCommandBus(SimpleNamespace())
    result = await bus.request(user_id="u1", action="bluetooth_list", timeout=1.0)
    assert result == {"ok": False, "error": "no_hub"}


@pytest.mark.asyncio
async def test_bus_timeout():
    hub = _StubHub(delivered=1)  # delivered, but nobody resolves
    bus = DeviceCommandBus(SimpleNamespace(notification_hub=hub))
    result = await bus.request(user_id="u1", action="bluetooth_list", timeout=0.1)
    assert result == {"ok": False, "error": "timeout"}
    assert bus.pending_count() == 0


@pytest.mark.asyncio
async def test_bus_resolve_unknown_request_is_dropped():
    bus = DeviceCommandBus(SimpleNamespace())
    assert bus.resolve("devcmd-nope", {"ok": True}) is False


# ── Presence: android connection → device slot ───────────────────────


@pytest.mark.asyncio
async def test_presence_device_from_hub():
    from augmentum.companion_runtime.presence_context import now_context

    hub = NotificationHub()
    att = await hub.attach(ws=_FakeWS(), user_id="u1")
    await hub.set_device_type(att.connection_id, "android")
    app_state = SimpleNamespace(notification_hub=hub, device_registry=None)

    snap = await now_context(None, "u1", app_state=app_state)
    assert snap["device"] and snap["device"]["kind"] == "android"


@pytest.mark.asyncio
async def test_presence_no_phone_no_device():
    from augmentum.companion_runtime.presence_context import now_context

    hub = NotificationHub()
    await hub.attach(ws=_FakeWS(), user_id="u1")  # web tab, untagged
    app_state = SimpleNamespace(notification_hub=hub, device_registry=None)
    snap = await now_context(None, "u1", app_state=app_state)
    assert snap["device"] is None


def test_prompt_lines_render_device():
    from augmentum.companion_runtime.presence_context import prompt_lines

    lines = prompt_lines({"device": {"kind": "android", "label": "Android phone"}})
    assert any("Android phone" in line for line in lines)


# ── device.bluetooth_list verb ───────────────────────────────────────


class _StubBus:
    def __init__(self, result) -> None:
        self._result = result
        self.calls: list[str] = []

    async def request(self, *, user_id, action, params=None,
                      device_type="android", timeout=8.0):
        self.calls.append(action)
        return self._result


def _session(result, user_id="u1"):
    app_state = SimpleNamespace(device_command_bus=_StubBus(result))
    return SimpleNamespace(user_id=user_id, app_state=app_state)


def _enable(monkeypatch, on=True):
    from augmentum.config import settings
    monkeypatch.setattr(
        settings, "companion_device_tools_enabled", on, raising=False,
    )


@pytest.mark.asyncio
async def test_verb_gated_off(monkeypatch):
    from augmentum.intent.builtin.device import _bluetooth_list_handler
    _enable(monkeypatch, on=False)
    res = await _bluetooth_list_handler("", _session({"ok": True}), {})
    assert "turned off" in res.speak.lower()


@pytest.mark.asyncio
async def test_verb_signed_out(monkeypatch):
    from augmentum.intent.builtin.device import _bluetooth_list_handler
    _enable(monkeypatch, on=True)
    res = await _bluetooth_list_handler("", _session({"ok": True}, user_id=""), {})
    assert "signed-out" in res.speak.lower()


@pytest.mark.asyncio
async def test_verb_not_connected(monkeypatch):
    from augmentum.intent.builtin.device import _bluetooth_list_handler
    _enable(monkeypatch, on=True)
    res = await _bluetooth_list_handler(
        "", _session({"ok": False, "error": "not_connected"}), {},
    )
    assert "isn't connected" in res.speak


@pytest.mark.asyncio
async def test_verb_timeout(monkeypatch):
    from augmentum.intent.builtin.device import _bluetooth_list_handler
    _enable(monkeypatch, on=True)
    res = await _bluetooth_list_handler(
        "", _session({"ok": False, "error": "timeout"}), {},
    )
    assert "didn't answer" in res.speak


@pytest.mark.asyncio
async def test_verb_lists_connected_device(monkeypatch):
    from augmentum.intent.builtin.device import _bluetooth_list_handler
    _enable(monkeypatch, on=True)
    result = {
        "ok": True,
        "devices": [
            {"name": "Pixel Buds", "connected": True, "address": "AA:BB"},
            {"name": "Civic", "connected": False},
        ],
    }
    res = await _bluetooth_list_handler("", _session(result), {})
    assert "Pixel Buds" in res.speak
    assert "1 more paired" in res.speak
    assert "[bluetooth devices]" in res.prompt_addendum
    assert "Pixel Buds" in res.prompt_addendum


@pytest.mark.asyncio
async def test_verb_no_devices(monkeypatch):
    from augmentum.intent.builtin.device import _bluetooth_list_handler
    _enable(monkeypatch, on=True)
    res = await _bluetooth_list_handler("", _session({"ok": True, "devices": []}), {})
    assert "isn't paired" in res.speak


# ── device.* action verbs (set_alarm/set_timer/dial/text/contact/app) ─
#
# These assert two things the design hinges on: (1) the natural-language
# slot is normalized to the right phone primitive BEFORE it leaves the
# server (the cross-model robustness contract), and (2) the spoken line
# stays honest — never "I called/sent/saved", only "I opened …".


class _CapBus:
    """Bus stub that captures the action + params each verb sends."""

    def __init__(self, result) -> None:
        self._result = result
        self.last_action: str | None = None
        self.last_params: dict | None = None

    async def request(self, *, user_id, action, params=None,
                      device_type="android", timeout=8.0):
        self.last_action = action
        self.last_params = params or {}
        return self._result


def _cap_session(result, user_id="u1"):
    bus = _CapBus(result)
    app_state = SimpleNamespace(device_command_bus=bus)
    return SimpleNamespace(user_id=user_id, app_state=app_state), bus


@pytest.mark.asyncio
async def test_set_alarm_absolute_normalizes_and_speaks(monkeypatch):
    from augmentum.intent.builtin.device import _set_alarm_handler
    _enable(monkeypatch, on=True)
    sess, bus = _cap_session({"ok": True})
    res = await _set_alarm_handler("", sess, {"time": "7:30am", "label": "gym"})
    assert bus.last_action == "set_alarm"
    assert bus.last_params == {"hour": 7, "minute": 30, "label": "gym"}
    assert "7:30 AM" in res.speak and "gym" in res.speak


@pytest.mark.asyncio
async def test_set_alarm_relative_sends_seconds_and_uses_phone_clock(monkeypatch):
    from augmentum.intent.builtin.device import _set_alarm_handler
    _enable(monkeypatch, on=True)
    # Relative time: server sends in_seconds; phone resolves wall-clock
    # against its own timezone and echoes scheduled_for, which we speak.
    sess, bus = _cap_session({"ok": True, "scheduled_for": "5:30 PM"})
    res = await _set_alarm_handler("", sess, {"time": "in 2 hours"})
    assert bus.last_params == {"in_seconds": 7200}
    assert "5:30 PM" in res.speak


@pytest.mark.asyncio
async def test_set_alarm_unparseable_time_asks(monkeypatch):
    from augmentum.intent.builtin.device import _set_alarm_handler
    _enable(monkeypatch, on=True)
    sess, bus = _cap_session({"ok": True})
    res = await _set_alarm_handler("", sess, {"time": "whenever"})
    assert bus.last_action is None  # never reached the phone
    assert "when" in res.speak.lower()


@pytest.mark.asyncio
async def test_set_timer_normalizes_duration(monkeypatch):
    from augmentum.intent.builtin.device import _set_timer_handler
    _enable(monkeypatch, on=True)
    sess, bus = _cap_session({"ok": True})
    res = await _set_timer_handler("", sess, {"duration": "an hour and a half"})
    assert bus.last_params == {"seconds": 5400}
    assert "1 hour 30 minutes" in res.speak


@pytest.mark.asyncio
async def test_set_timer_no_duration_asks(monkeypatch):
    from augmentum.intent.builtin.device import _set_timer_handler
    _enable(monkeypatch, on=True)
    sess, bus = _cap_session({"ok": True})
    res = await _set_timer_handler("", sess, {"duration": ""})
    assert bus.last_action is None
    assert "how long" in res.speak.lower()


@pytest.mark.asyncio
async def test_dial_cleans_number_and_stays_honest(monkeypatch):
    from augmentum.intent.builtin.device import _dial_handler
    _enable(monkeypatch, on=True)
    sess, bus = _cap_session({"ok": True})
    res = await _dial_handler("", sess, {"number": "(800) 555-0142", "name": "Mom"})
    assert bus.last_params == {"number": "8005550142"}
    assert "dialer" in res.speak.lower() and "Mom" in res.speak
    assert "won't place the call" in res.speak.lower()


@pytest.mark.asyncio
async def test_dial_no_number_degrades(monkeypatch):
    from augmentum.intent.builtin.device import _dial_handler
    _enable(monkeypatch, on=True)
    sess, bus = _cap_session({"ok": True})
    res = await _dial_handler("", sess, {"name": "Mom"})
    assert bus.last_action is None
    assert "number" in res.speak.lower()


@pytest.mark.asyncio
async def test_compose_text_drafts_body_and_stays_honest(monkeypatch):
    from augmentum.intent.builtin.device import _compose_text_handler
    _enable(monkeypatch, on=True)
    sess, bus = _cap_session({"ok": True})
    res = await _compose_text_handler(
        "", sess, {"number": "5550142", "body": "running late"},
    )
    assert bus.last_params == {"number": "5550142", "body": "running late"}
    assert "drafted" in res.speak.lower()
    assert "won't send" in res.speak.lower()


@pytest.mark.asyncio
async def test_add_contact_opens_editor(monkeypatch):
    from augmentum.intent.builtin.device import _add_contact_handler
    _enable(monkeypatch, on=True)
    sess, bus = _cap_session({"ok": True})
    res = await _add_contact_handler(
        "", sess, {"name": "Dr. Lee", "email": "lee@clinic.com"},
    )
    assert bus.last_params == {"name": "Dr. Lee", "email": "lee@clinic.com"}
    assert "Dr. Lee" in res.speak and "save" in res.speak.lower()


@pytest.mark.asyncio
async def test_launch_app_speaks_resolved_label(monkeypatch):
    from augmentum.intent.builtin.device import _launch_app_handler
    _enable(monkeypatch, on=True)
    # The phone returns the REAL label it matched; we speak that, not the
    # user's (possibly misheard) query.
    sess, bus = _cap_session({"ok": True, "label": "Spotify"})
    res = await _launch_app_handler("", sess, {"app": "spotofy"})
    assert bus.last_params == {"query": "spotofy"}
    assert res.speak == "Opening Spotify."


@pytest.mark.asyncio
async def test_launch_app_no_match(monkeypatch):
    from augmentum.intent.builtin.device import _launch_app_handler
    _enable(monkeypatch, on=True)
    sess, bus = _cap_session({"ok": True, "error": "no_match"})
    res = await _launch_app_handler("", sess, {"app": "nonesuch"})
    assert "couldn't find" in res.speak.lower()


@pytest.mark.asyncio
async def test_action_verb_bus_error_degrades(monkeypatch):
    from augmentum.intent.builtin.device import _set_timer_handler
    _enable(monkeypatch, on=True)
    sess, _ = _cap_session({"ok": False, "error": "not_connected"})
    res = await _set_timer_handler("", sess, {"duration": "5 minutes"})
    assert "isn't connected" in res.speak


@pytest.mark.asyncio
async def test_action_verb_gated_off(monkeypatch):
    from augmentum.intent.builtin.device import _set_alarm_handler
    _enable(monkeypatch, on=False)
    sess, bus = _cap_session({"ok": True})
    res = await _set_alarm_handler("", sess, {"time": "7am"})
    assert bus.last_action is None
    assert "turned off" in res.speak.lower()
