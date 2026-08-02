"""DeviceRegistry integration tests.

Wires a mock driver into the registry over an in-memory SQLite DB and
exercises the full path: save → discover → invoke → snapshot → session
lifecycle. User scoping is verified at every join point.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiosqlite
import pytest
import pytest_asyncio

from augmentum.devices.device import Device, DiscoveredDevice
from augmentum.devices.events import EventBus
from augmentum.devices.invocation import (
    Event,
    InvocationContext,
    InvocationResult,
    PairResult,
)
from augmentum.devices.registry import DeviceRegistry
from augmentum.devices.sessions import SessionRuntime
from augmentum.devices.store import (
    DevicePairingStore,
    DevicePlayHistoryStore,
    DeviceStore,
)

_MIGRATIONS_DIR = Path(__file__).parent.parent / "augmentum" / "state" / "migrations"

_BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT
);
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
INSERT INTO users (id, username) VALUES ('user_alice', 'alice');
INSERT INTO users (id, username) VALUES ('user_bob',   'bob');
"""


async def _apply_migration(conn: aiosqlite.Connection, version: int) -> None:
    for path in _MIGRATIONS_DIR.glob("*.sql"):
        try:
            v = int(path.stem.split("_")[0])
        except (ValueError, IndexError):
            continue
        if v == version:
            await conn.executescript(path.read_text(encoding="utf-8"))
            await conn.commit()
            return
    raise FileNotFoundError(f"Migration {version:03d} not found")


# ----------------------------------------------------------------------------
# Mock driver — exercises the protocol without real hardware
# ----------------------------------------------------------------------------


class MockDriver:
    id: str = "mock"
    label: str = "Mock"
    description: str = "Test driver"
    capabilities: tuple[str, ...] = (
        "media.video_play@1",
        "media.audio_play@1",
        "lighting.set_state@1",
    )
    discovery_modes: tuple[str, ...] = ("manual_only",)
    requires_pairing: bool = False
    supports_passive_discovery: bool = False

    def __init__(self) -> None:
        self.invocations: list[tuple[str, str, str, dict]] = []
        self.next_invoke_result: InvocationResult | None = None
        self.discovery_results: list[DiscoveredDevice] = []
        self.probe_result: DiscoveredDevice | None = None
        self.start_called: int = 0
        self.stop_called: int = 0

    async def start(self, ctx: Any) -> None:
        self.start_called += 1

    async def stop(self) -> None:
        self.stop_called += 1

    async def discover(
        self,
        *,
        timeout_s: float = 3.0,
        user_id: str = "",  # noqa: ARG002
    ) -> list[DiscoveredDevice]:
        return list(self.discovery_results)

    async def probe(
        self,
        *,
        host: str,
        port: int | None = None,
        hint: dict | None = None,
    ) -> DiscoveredDevice | None:
        return self.probe_result

    async def invoke(
        self,
        device: Device,
        capability: str,
        action: str,
        args: dict,
        ctx: InvocationContext,
    ) -> InvocationResult:
        self.invocations.append((device.id, capability, action, dict(args)))
        if self.next_invoke_result is not None:
            r, self.next_invoke_result = self.next_invoke_result, None
            return r
        return InvocationResult.success(state={"echoed": dict(args)})

    async def snapshot(
        self,
        device: Device,
        capability: str,
        ctx: InvocationContext,
    ) -> dict | None:
        return {"polled": True, "device_id": device.id}

    async def subscribe(
        self,
        device: Device,
        capability: str,
        ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        async def _empty() -> AsyncIterator[Event]:
            if False:
                yield  # noqa: B901
        return _empty()

    async def pair_start(self, device: Device, ctx: InvocationContext) -> PairResult:
        return PairResult(state="active")

    async def pair_complete(
        self,
        device: Device,
        code: str,
        ctx: InvocationContext,
    ) -> PairResult:
        return PairResult(state="active")


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


@pytest_asyncio.fixture
async def registry():
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.executescript(_BOOTSTRAP_SQL)
    await conn.commit()
    await _apply_migration(conn, 129)
    await _apply_migration(conn, 130)
    await _apply_migration(conn, 131)

    reg = DeviceRegistry(
        device_store=DeviceStore(conn),
        pairing_store=DevicePairingStore(conn),
        history_store=DevicePlayHistoryStore(conn),
        sessions=SessionRuntime(),
        bus=EventBus(),
    )
    driver = MockDriver()
    reg.register_driver(driver)
    await reg.start()

    yield reg, driver

    await reg.stop()
    await conn.close()


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


class TestRegistryLifecycle:
    @pytest.mark.asyncio
    async def test_start_invokes_drivers(self, registry):
        _, driver = registry
        assert driver.start_called == 1

    @pytest.mark.asyncio
    async def test_lists_drivers(self, registry):
        reg, _ = registry
        drivers = reg.list_drivers()
        assert len(drivers) == 1
        assert drivers[0].id == "mock"


class TestRegistryPersistence:
    @pytest.mark.asyncio
    async def test_save_expands_capabilities_via_extends(self, registry):
        # video_play extends audio_play — saving with [video_play] expands.
        reg, _ = registry
        device = await reg.save(
            user_id="user_alice",
            driver="mock",
            native_id="tv_1",
            label="Living Room",
            capabilities=["media.video_play@1"],
            address={"host": "192.168.1.10"},
            status="online",
        )
        assert "media.video_play@1" in device.capabilities
        assert "media.audio_play@1" in device.capabilities  # expanded

    @pytest.mark.asyncio
    async def test_user_scoped_listing(self, registry):
        reg, _ = registry
        await reg.save(
            user_id="user_alice", driver="mock", native_id="a",
            label="A", capabilities=["lighting.set_state@1"],
        )
        await reg.save(
            user_id="user_bob", driver="mock", native_id="b",
            label="B", capabilities=["lighting.set_state@1"],
        )
        alice_devices = await reg.list(user_id="user_alice")
        bob_devices = await reg.list(user_id="user_bob")
        assert len(alice_devices) == 1
        assert alice_devices[0].label == "A"
        assert len(bob_devices) == 1
        assert bob_devices[0].label == "B"

    @pytest.mark.asyncio
    async def test_get_user_scoped(self, registry):
        reg, _ = registry
        d = await reg.save(
            user_id="user_alice", driver="mock", native_id="a",
            label="A", capabilities=["lighting.set_state@1"],
        )
        # Same device id, wrong user → no result
        assert await reg.get(d.id, user_id="user_bob") is None
        assert await reg.get(d.id, user_id="user_alice") is not None

    @pytest.mark.asyncio
    async def test_delete_user_scoped(self, registry):
        reg, _ = registry
        d = await reg.save(
            user_id="user_alice", driver="mock", native_id="a",
            label="A", capabilities=["lighting.set_state@1"],
        )
        # Wrong user can't delete
        assert await reg.delete(d.id, user_id="user_bob") is False
        # Correct user can
        assert await reg.delete(d.id, user_id="user_alice") is True
        assert await reg.get(d.id, user_id="user_alice") is None


class TestRegistryInvocation:
    @pytest.mark.asyncio
    async def test_invoke_unknown_device(self, registry):
        reg, _ = registry
        result = await reg.invoke(
            user_id="user_alice",
            device_id="dev_does_not_exist",
            capability="lighting.set_state@1",
            action="on",
            args={},
        )
        assert result.ok is False
        assert result.code == "device_not_found"

    @pytest.mark.asyncio
    async def test_invoke_unknown_capability(self, registry):
        reg, _ = registry
        d = await reg.save(
            user_id="user_alice", driver="mock", native_id="a",
            label="A", capabilities=["lighting.set_state@1"],
        )
        result = await reg.invoke(
            user_id="user_alice",
            device_id=d.id,
            capability="nope.fake@1",
            action="on",
            args={},
        )
        assert result.ok is False
        assert result.code == "unknown_capability"

    @pytest.mark.asyncio
    async def test_invoke_capability_not_supported_by_device(self, registry):
        reg, _ = registry
        d = await reg.save(
            user_id="user_alice", driver="mock", native_id="a",
            label="Bulb", capabilities=["lighting.set_state@1"],
        )
        result = await reg.invoke(
            user_id="user_alice",
            device_id=d.id,
            capability="media.video_play@1",
            action="play",
            args={"content_url": "http://x/y.mp4"},
        )
        assert result.ok is False
        assert result.code == "capability_not_supported"

    @pytest.mark.asyncio
    async def test_invoke_unknown_action(self, registry):
        reg, _ = registry
        d = await reg.save(
            user_id="user_alice", driver="mock", native_id="a",
            label="Bulb", capabilities=["lighting.set_state@1"],
        )
        result = await reg.invoke(
            user_id="user_alice",
            device_id=d.id,
            capability="lighting.set_state@1",
            action="dance",
            args={},
        )
        assert result.ok is False
        assert result.code == "unknown_action"

    @pytest.mark.asyncio
    async def test_invoke_user_scoped(self, registry):
        reg, _ = registry
        d = await reg.save(
            user_id="user_alice", driver="mock", native_id="a",
            label="Bulb", capabilities=["lighting.set_state@1"],
        )
        # Wrong user → device_not_found (don't leak existence)
        result = await reg.invoke(
            user_id="user_bob",
            device_id=d.id,
            capability="lighting.set_state@1",
            action="on",
            args={},
        )
        assert result.ok is False
        assert result.code == "device_not_found"

    @pytest.mark.asyncio
    async def test_invoke_success_dispatches_to_driver(self, registry):
        reg, driver = registry
        d = await reg.save(
            user_id="user_alice", driver="mock", native_id="a",
            label="Bulb", capabilities=["lighting.set_state@1"],
        )
        result = await reg.invoke(
            user_id="user_alice",
            device_id=d.id,
            capability="lighting.set_state@1",
            action="on",
            args={"foo": "bar"},
        )
        assert result.ok is True
        assert driver.invocations[-1] == (d.id, "lighting.set_state@1", "on", {"foo": "bar"})

    @pytest.mark.asyncio
    async def test_stateful_play_creates_session(self, registry):
        reg, _ = registry
        d = await reg.save(
            user_id="user_alice", driver="mock", native_id="tv_1",
            label="TV", capabilities=["media.video_play@1"],
        )
        result = await reg.invoke(
            user_id="user_alice",
            device_id=d.id,
            capability="media.video_play@1",
            action="play",
            args={"content_url": "http://x/y.mp4", "title": "Movie"},
        )
        assert result.ok is True
        sessions = await reg.list_sessions(user_id="user_alice")
        assert len(sessions) == 1
        assert sessions[0].title == "Movie"
        assert sessions[0].capability_id == "media.video_play@1"
        assert sessions[0].extra["device_label"] == "TV"

    @pytest.mark.asyncio
    async def test_stop_action_removes_session(self, registry):
        reg, _ = registry
        d = await reg.save(
            user_id="user_alice", driver="mock", native_id="tv_1",
            label="TV", capabilities=["media.video_play@1"],
        )
        await reg.invoke(
            user_id="user_alice",
            device_id=d.id,
            capability="media.video_play@1",
            action="play",
            args={"content_url": "http://x/y.mp4"},
        )
        assert len(await reg.list_sessions(user_id="user_alice")) == 1

        await reg.invoke(
            user_id="user_alice",
            device_id=d.id,
            capability="media.video_play@1",
            action="stop",
            args={},
        )
        assert len(await reg.list_sessions(user_id="user_alice")) == 0

    @pytest.mark.asyncio
    async def test_delete_device_kills_sessions(self, registry):
        reg, _ = registry
        d = await reg.save(
            user_id="user_alice", driver="mock", native_id="tv_1",
            label="TV", capabilities=["media.video_play@1"],
        )
        await reg.invoke(
            user_id="user_alice",
            device_id=d.id,
            capability="media.video_play@1",
            action="play",
            args={"content_url": "http://x/y.mp4"},
        )
        await reg.delete(d.id, user_id="user_alice")
        assert await reg.list_sessions(user_id="user_alice") == []


class TestPlayHistory:
    @pytest.mark.asyncio
    async def test_play_action_logs_history(self, registry):
        reg, _ = registry
        d = await reg.save(
            user_id="user_alice", driver="mock", native_id="tv_1",
            label="TV", capabilities=["media.video_play@1"],
        )
        await reg.invoke(
            user_id="user_alice",
            device_id=d.id,
            capability="media.video_play@1",
            action="play",
            args={
                "content_url": "http://emby/x.mkv",
                "title": "Lofi Hip Hop Mix",
            },
        )
        rows = await reg._history_store.recent_for_kind(  # noqa: SLF001
            user_id="user_alice", content_kind="video", limit=10,
        )
        assert len(rows) == 1
        assert rows[0]["content_label"] == "Lofi Hip Hop Mix"
        assert rows[0]["content_kind"] == "video"

    @pytest.mark.asyncio
    async def test_pause_does_not_log(self, registry):
        reg, _ = registry
        d = await reg.save(
            user_id="user_alice", driver="mock", native_id="tv_1",
            label="TV", capabilities=["media.video_play@1"],
        )
        await reg.invoke(
            user_id="user_alice",
            device_id=d.id,
            capability="media.video_play@1",
            action="pause",
            args={},
        )
        rows = await reg._history_store.recent_for_kind(  # noqa: SLF001
            user_id="user_alice", content_kind="video", limit=10,
        )
        assert rows == []

    @pytest.mark.asyncio
    async def test_favorite_lookup_returns_marked(self, registry):
        reg, _ = registry
        d = await reg.save(
            user_id="user_alice", driver="mock", native_id="tv_1",
            label="TV", capabilities=["media.video_play@1"],
        )
        await reg.invoke(
            user_id="user_alice",
            device_id=d.id,
            capability="media.video_play@1",
            action="play",
            args={
                "content_url": "ck:lofi",
                "content_key": "ck:lofi",
                "title": "Lofi",
            },
        )
        affected = await reg._history_store.set_favorite(  # noqa: SLF001
            user_id="user_alice", content_key="ck:lofi", is_favorite=True,
        )
        assert affected == 1
        favs = await reg._history_store.favorites_for_kind(  # noqa: SLF001
            user_id="user_alice", content_kind="video",
        )
        assert len(favs) == 1
        assert favs[0]["content_key"] == "ck:lofi"


class TestDiscovery:
    @pytest.mark.asyncio
    async def test_paired_only_driver_skipped(self, registry):
        reg, driver = registry
        # MockDriver has discovery_modes=('manual_only',); discover should skip it.
        result = await reg.discover(user_id="user_alice")
        assert result.discovered == []
        assert result.errors == {}

    @pytest.mark.asyncio
    async def test_probe_passes_through(self, registry):
        reg, driver = registry
        driver.probe_result = DiscoveredDevice(
            driver="mock",
            native_id="probed_xyz",
            label="Probed",
            capabilities=["lighting.set_state@1"],
            address={"host": "192.168.1.99"},
        )
        found = await reg.probe(
            user_id="user_alice",
            driver="mock",
            host="192.168.1.99",
        )
        assert found is not None
        assert found.native_id == "probed_xyz"
