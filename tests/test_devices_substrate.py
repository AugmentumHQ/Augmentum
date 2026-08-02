"""Pure-Python unit tests for the device substrate.

Covers capability resolution, device IDs, sessions, events, and the
discovery dedup logic. SQLite + registry integration lives in
``test_devices_registry.py``.
"""

from __future__ import annotations

import asyncio

import pytest

from augmentum.devices.capabilities import (
    get_capability,
    has_capability,
    list_capabilities,
    resolve_action,
)
from augmentum.devices.capability import ActionSchema, Capability
from augmentum.devices.device import Device, DiscoveredDevice, make_device_id
from augmentum.devices.discovery.coordinator import (
    merge_discovered_with_saved,
)
from augmentum.devices.events import EventBus
from augmentum.devices.host_resolver import PublicHostResolver
from augmentum.devices.invocation import (
    Event,
    InvocationResult,
    PairResult,
)
from augmentum.devices.sessions import CapabilitySession, SessionRuntime

# ----------------------------------------------------------------------------
# Capabilities
# ----------------------------------------------------------------------------


class TestCapabilityCatalog:
    def test_catalog_loads(self):
        caps = list_capabilities()
        assert len(caps) >= 12
        ids = {c.id for c in caps}
        assert "media.audio_play@1" in ids
        assert "media.video_play@1" in ids
        assert "lighting.set_color@1" in ids
        assert "surface.follow_state@1" in ids
        assert "display.comic_read@1" in ids
        assert "input.gamepad_bridge@1" in ids

    def test_get_capability_returns_typed(self):
        cap = get_capability("media.audio_play@1")
        assert cap is not None
        assert cap.label == "Audio Playback"
        assert any(a.name == "play" for a in cap.actions)

    def test_unknown_capability_returns_none(self):
        assert get_capability("nope.does_not_exist@1") is None
        assert not has_capability("nope.does_not_exist@1")

    def test_resolve_action_local(self):
        resolved = resolve_action("media.audio_play@1", "play")
        assert resolved is not None
        cap, action = resolved
        assert cap.id == "media.audio_play@1"
        assert action.name == "play"
        assert action.is_stateful is True

    def test_resolve_action_via_extends_chain(self):
        # video_play extends audio_play; even if we removed an action from
        # video_play locally, the chain should find it on audio_play. Today
        # video_play declares its own copy of `play`, so resolution is local.
        # The contract we want: if a child capability's actions tuple is
        # missing an action, the resolver finds it on the parent.
        child = Capability(
            id="testing.child@1",
            label="Child",
            description="",
            actions=(
                ActionSchema(name="extra", description="", args_schema={}),
            ),
            extends="media.audio_play@1",
        )
        # Walk should still find `play` from audio_play via extends.
        from augmentum.devices.capabilities import _ALL  # noqa: SLF001
        _ALL[child.id] = child
        try:
            resolved = resolve_action(child.id, "play")
            assert resolved is not None
            assert resolved[0].id == "media.audio_play@1"
            assert resolved[1].name == "play"
        finally:
            _ALL.pop(child.id, None)

    def test_resolve_unknown_action_returns_none(self):
        assert resolve_action("media.audio_play@1", "fly_to_the_moon") is None


# ----------------------------------------------------------------------------
# Device IDs and supports()
# ----------------------------------------------------------------------------


class TestDevice:
    def test_make_device_id_deterministic(self):
        a = make_device_id(driver="dlna", native_id="UDN:abc", user_id="user_1")
        b = make_device_id(driver="dlna", native_id="UDN:abc", user_id="user_1")
        assert a == b
        assert a.startswith("dev_")

    def test_make_device_id_user_scoped(self):
        a = make_device_id(driver="dlna", native_id="UDN:abc", user_id="user_1")
        b = make_device_id(driver="dlna", native_id="UDN:abc", user_id="user_2")
        assert a != b  # same physical device, different users → different IDs

    def test_supports_direct(self):
        d = Device(
            id="dev_x",
            user_id="u",
            driver="dlna",
            native_id="x",
            label="TV",
            capabilities=["media.video_play@1"],
        )
        assert d.supports("media.video_play@1")
        assert not d.supports("media.audio_play@1")

    def test_supports_via_binding(self):
        d = Device(
            id="dev_x",
            user_id="u",
            driver="dlna",
            native_id="x",
            label="TV",
            capabilities=["media.video_play@1"],
            bindings=[{"driver": "cast", "native_id": "y", "capabilities": ["display.image_show@1"]}],
        )
        assert d.supports("display.image_show@1")
        assert d.driver_for("display.image_show@1") == "cast"
        assert d.driver_for("media.video_play@1") == "dlna"


# ----------------------------------------------------------------------------
# Sessions
# ----------------------------------------------------------------------------


class TestSessionRuntime:
    def test_put_and_get_user_scoped(self):
        rt = SessionRuntime()
        s = CapabilitySession(
            user_id="user_a", device_id="dev_1", driver="dlna",
            capability_id="media.video_play@1", title="Movie",
        )
        rt.put(s)
        assert rt.get(s.id, user_id="user_a") is not None
        assert rt.get(s.id, user_id="user_b") is None  # cross-user denied

    def test_list_filters_by_user(self):
        rt = SessionRuntime()
        rt.put(CapabilitySession(user_id="u1", device_id="d", driver="x", capability_id="c"))
        rt.put(CapabilitySession(user_id="u2", device_id="d", driver="x", capability_id="c"))
        assert len(rt.list(user_id="u1")) == 1
        assert len(rt.list(user_id="u2")) == 1

    def test_remove_all_for_device(self):
        rt = SessionRuntime()
        rt.put(CapabilitySession(user_id="u", device_id="d1", driver="x", capability_id="c"))
        rt.put(CapabilitySession(user_id="u", device_id="d1", driver="x", capability_id="c2"))
        rt.put(CapabilitySession(user_id="u", device_id="d2", driver="x", capability_id="c"))
        n = rt.remove_all_for_device(user_id="u", device_id="d1")
        assert n == 2
        assert len(rt.list_for_device(user_id="u", device_id="d1")) == 0
        assert len(rt.list_for_device(user_id="u", device_id="d2")) == 1

    def test_update_state_merges(self):
        rt = SessionRuntime()
        s = CapabilitySession(
            user_id="u", device_id="d", driver="x", capability_id="c",
            state={"a": 1, "b": 2},
        )
        rt.put(s)
        s.update_state({"b": 99, "c": 3})
        assert s.state == {"a": 1, "b": 99, "c": 3}


# ----------------------------------------------------------------------------
# Event bus
# ----------------------------------------------------------------------------


class TestEventBus:
    @pytest.mark.asyncio
    async def test_user_scoped_subscribe(self):
        bus = EventBus()
        sub_a = bus.subscribe(user_id="alice")
        sub_b = bus.subscribe(user_id="bob")

        await bus.publish(Event(
            device_id="d1", capability_id="c", type="t", user_id="alice",
        ))
        await bus.publish(Event(
            device_id="d2", capability_id="c", type="t", user_id="bob",
        ))

        async def _take_one(it):
            async for e in it:
                return e
            return None

        e_a = await asyncio.wait_for(_take_one(sub_a), timeout=0.5)
        e_b = await asyncio.wait_for(_take_one(sub_b), timeout=0.5)
        assert e_a is not None and e_a.device_id == "d1"
        assert e_b is not None and e_b.device_id == "d2"

    @pytest.mark.asyncio
    async def test_filter_by_device(self):
        bus = EventBus()
        sub = bus.subscribe(user_id="u", device_id="dev_target")
        await bus.publish(Event(device_id="dev_other", capability_id="c", type="t", user_id="u"))
        await bus.publish(Event(device_id="dev_target", capability_id="c", type="t", user_id="u"))

        async def _take_one(it):
            async for e in it:
                return e
            return None

        e = await asyncio.wait_for(_take_one(sub), timeout=0.5)
        assert e is not None and e.device_id == "dev_target"

    @pytest.mark.asyncio
    async def test_bounded_queue_drops_oldest(self):
        bus = EventBus(default_queue_size=3)
        sub = bus.subscribe(user_id="u", queue_size=3)
        # Publish 5 events; queue holds 3 most recent
        for i in range(5):
            await bus.publish(Event(
                device_id=f"d{i}", capability_id="c", type="t", user_id="u",
            ))
        received: list[str] = []

        async def _drain(it):
            try:
                while True:
                    e = await asyncio.wait_for(it.__anext__(), timeout=0.1)
                    received.append(e.device_id)
            except (StopAsyncIteration, TimeoutError):
                return

        await _drain(sub)
        # We expect d2, d3, d4 (oldest two dropped)
        assert received == ["d2", "d3", "d4"]


# ----------------------------------------------------------------------------
# Public host resolution
# ----------------------------------------------------------------------------


class TestPublicHostResolver:
    def test_configured_host_wins_for_tv_urls(self):
        resolver = PublicHostResolver(configured="192.168.1.10:6443")
        request = type("Req", (), {
            "headers": {"host": "localhost:6100"},
            "url": type("Url", (), {"scheme": "https"})(),
            "scope": {},
        })()

        assert resolver.public_url("/api/cast/blob/tok", request=request) == (
            "https://192.168.1.10:6443/api/cast/blob/tok"
        )

    def test_learns_non_loopback_host_from_scope(self):
        resolver = PublicHostResolver()
        resolver.observe_scope({
            "type": "http",
            "headers": [(b"host", b"192.168.1.20:6100")],
        })

        assert resolver.public_url("/api/cast/blob/tok", scheme="http") == (
            "http://192.168.1.20:6100/api/cast/blob/tok"
        )


# ----------------------------------------------------------------------------
# Discovery dedup
# ----------------------------------------------------------------------------


class TestDiscoveryDedup:
    def test_exact_native_id_match_marks_online(self):
        saved = [Device(
            id="dev_1", user_id="u", driver="dlna", native_id="UDN:abc",
            label="TV", capabilities=["media.video_play@1"],
        )]
        discovered = [DiscoveredDevice(
            driver="dlna", native_id="UDN:abc", label="TV",
            capabilities=["media.video_play@1"],
        )]
        new, online, _heal = merge_discovered_with_saved(discovered, saved)
        assert new == []
        assert online == ["dev_1"]

    def test_truly_new_when_no_match(self):
        saved = [Device(
            id="dev_1", user_id="u", driver="dlna", native_id="UDN:a",
            label="A", capabilities=["media.video_play@1"],
        )]
        discovered = [DiscoveredDevice(
            driver="dlna", native_id="UDN:b", label="B",
            capabilities=["media.video_play@1"],
        )]
        new, online, _heal = merge_discovered_with_saved(discovered, saved)
        assert len(new) == 1
        assert new[0].native_id == "UDN:b"
        assert online == []

    def test_fingerprint_match_across_drivers(self):
        # Same physical TV, found via DLNA previously (saved), now showing
        # up via Cast in this sweep — should NOT be flagged as new.
        saved = [Device(
            id="dev_1", user_id="u", driver="dlna", native_id="UDN:dlna_id",
            label="Living Room", capabilities=["media.video_play@1"],
            address={"host": "192.168.1.42"},
            metadata={"manufacturer": "sony", "model_name": "bravia-x900"},
        )]
        discovered = [DiscoveredDevice(
            driver="cast", native_id="cast_uuid_xyz", label="Living Room TV",
            capabilities=["media.video_play@1"],
            address={"host": "192.168.1.42"},
            metadata={"manufacturer": "sony", "model_name": "bravia-x900"},
        )]
        new, online, _heal = merge_discovered_with_saved(discovered, saved)
        assert new == []
        assert online == ["dev_1"]

    def test_binding_native_id_matches(self):
        saved = [Device(
            id="dev_1", user_id="u", driver="dlna", native_id="UDN:dlna",
            label="Living Room", capabilities=["media.video_play@1"],
            bindings=[{"driver": "cast", "native_id": "cast_uuid"}],
        )]
        discovered = [DiscoveredDevice(
            driver="cast", native_id="cast_uuid", label="Living Room TV",
            capabilities=["media.video_play@1"],
        )]
        new, online, _heal = merge_discovered_with_saved(discovered, saved)
        assert new == []
        assert online == ["dev_1"]


# ----------------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------------


class TestInvocationResult:
    def test_success_serializes(self):
        r = InvocationResult.success(state={"a": 1}, session_id="sess_x")
        d = r.to_dict()
        assert d["ok"] is True
        assert d["state"] == {"a": 1}
        assert d["session_id"] == "sess_x"

    def test_failure_serializes(self):
        r = InvocationResult.failure("nope", code="device_not_found")
        d = r.to_dict()
        assert d["ok"] is False
        assert d["code"] == "device_not_found"
        assert d["message"] == "nope"

    def test_pair_result_serializes(self):
        r = PairResult(
            state="pending",
            requires_user_action=True,
            instructions="Press the link button",
        )
        d = r.to_dict()
        assert d["state"] == "pending"
        assert d["requires_user_action"] is True
        assert "link button" in d["instructions"]
