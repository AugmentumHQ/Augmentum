"""Unit tests for DockerContainerAdapter.

The adapter is a thin shim over aiodocker; the value of these tests is
in pinning the *contract* with Docker -- env vars, labels, port
mappings, capability drops -- so a refactor doesn't silently change
what the runtime sends to docker.run.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def adapter(mock_docker):
    from augmentum.game_stream.docker_adapter import DockerContainerAdapter
    return DockerContainerAdapter(mock_docker), mock_docker


@pytest.fixture
def luanti_profile():
    from augmentum.game_stream.profiles import profile_registry
    return profile_registry.get("luanti")


def _ports():
    from augmentum.game_stream.port_pool import PortAllocation
    return PortAllocation(stream_port=30002, game_port=30003)


@pytest.mark.asyncio
async def test_start_invokes_docker_run_with_correct_image(adapter, luanti_profile):
    a, docker = adapter
    cid = await a.start(
        session_id="sess_abc",
        profile=luanti_profile,
        ports=_ports(),
        bitrate_mbps=4,
        resolution="1280x720",
        encoder="nvenc",
        world_storage_path="world_xyz",
        world_settings={"gamemode": "survival", "seed": "42"},
    )
    assert cid == "cnt_test123"
    docker.containers.run.assert_awaited_once()
    call = docker.containers.run.await_args
    config = call.kwargs.get("config") or call.args[0]
    assert config["Image"] == luanti_profile.image
    assert call.kwargs.get("name") == "agsp-sess_abc"


@pytest.mark.asyncio
async def test_start_sets_labels_for_reconcile(adapter, luanti_profile):
    a, docker = adapter
    await a.start(
        session_id="sess_abc",
        profile=luanti_profile,
        ports=_ports(),
        bitrate_mbps=4,
        resolution="1280x720",
        encoder="x264",
        world_storage_path="",
        world_settings={},
    )
    config = docker.containers.run.await_args.kwargs.get("config") or {}
    labels = config["Labels"]
    # The reconcile path filters on this label; if it changes the
    # restart-survival flow silently breaks.
    assert labels["augmentum.game_stream"] == "true"
    assert labels["augmentum.game_stream.session_id"] == "sess_abc"
    assert labels["augmentum.game_stream.profile_id"] == "luanti"


@pytest.mark.asyncio
async def test_start_publishes_ports_correctly(adapter, luanti_profile):
    a, docker = adapter
    await a.start(
        session_id="sess_p",
        profile=luanti_profile,
        ports=_ports(),
        bitrate_mbps=4,
        resolution="1280x720",
        encoder="x264",
        world_storage_path="",
        world_settings={},
    )
    config = docker.containers.run.await_args.kwargs.get("config") or {}
    bindings = config["HostConfig"]["PortBindings"]
    assert bindings["8080/tcp"][0]["HostPort"] == "30002"  # stream
    assert bindings["30000/udp"][0]["HostPort"] == "30003"  # game


@pytest.mark.asyncio
async def test_start_passes_env_to_entrypoint(adapter, luanti_profile):
    a, docker = adapter
    await a.start(
        session_id="sess_env",
        profile=luanti_profile,
        ports=_ports(),
        bitrate_mbps=6,
        resolution="1920x1080",
        encoder="vaapi",
        world_storage_path="my_world",
        world_settings={"gamemode": "creative", "seed": "abc", "pvp": True},
    )
    config = docker.containers.run.await_args.kwargs.get("config") or {}
    env = {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in config["Env"]}
    assert env["AUGMENTUM_SESSION_ID"] == "sess_env"
    assert env["AUGMENTUM_RESOLUTION"] == "1920x1080"
    assert env["AUGMENTUM_BITRATE_KBPS"] == "6000"
    assert env["AUGMENTUM_ENCODER"] == "vaapi"
    assert env["AUGMENTUM_WORLD_ID"] == "my_world"
    assert env["AUGMENTUM_WORLD_GAMEMODE"] == "creative"
    assert env["AUGMENTUM_WORLD_SEED"] == "abc"
    assert env["AUGMENTUM_WORLD_PVP"] == "true"


@pytest.mark.asyncio
async def test_start_passes_input_env_to_entrypoint(adapter, luanti_profile):
    a, docker = adapter
    await a.start(
        session_id="sess_input",
        profile=luanti_profile,
        ports=_ports(),
        bitrate_mbps=4,
        resolution="1280x720",
        encoder="x264",
        world_storage_path="",
        world_settings={},
        mouse_sensitivity=0.11,
        gamepad_enabled=False,
        controller_deadzone=0.24,
    )
    config = docker.containers.run.await_args.kwargs.get("config") or {}
    env = {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in config["Env"]}
    assert env["AUGMENTUM_MOUSE_SENSITIVITY"] == "0.11"
    assert env["AUGMENTUM_GAMEPAD_ENABLED"] == "false"
    assert env["AUGMENTUM_CONTROLLER_DEADZONE"] == "0.24"


@pytest.mark.asyncio
async def test_start_passes_agent_bridge_url_when_set(adapter, luanti_profile):
    """@example: agent_bridge_url is forwarded as AUGMENTUM_AGENT_BRIDGE_URL.

    The in-container entrypoint reads this env var and spawns
    agent-bridge.py against the URL; if the adapter doesn't pass it
    through, every AI-driven session boots into a passive container
    with no daemon dialling back.
    """

    a, docker = adapter
    await a.start(
        session_id="sess_ai",
        profile=luanti_profile,
        ports=_ports(),
        bitrate_mbps=4,
        resolution="1280x720",
        encoder="x264",
        world_storage_path="",
        world_settings={},
        agent_bridge_url=(
            "ws://augmentum:8080/api/game-agent/surfaces/emulator/"
            "bridge/s_abc?token=tok123"
        ),
    )
    config = docker.containers.run.await_args.kwargs.get("config") or {}
    env = {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in config["Env"]}
    assert env["AUGMENTUM_AGENT_BRIDGE_URL"] == (
        "ws://augmentum:8080/api/game-agent/surfaces/emulator/"
        "bridge/s_abc?token=tok123"
    )


@pytest.mark.asyncio
async def test_start_omits_agent_bridge_url_when_blank(adapter, luanti_profile):
    """@example: a non-AI session must not carry AUGMENTUM_AGENT_BRIDGE_URL.

    If the env var leaks through with an empty value the entrypoint's
    `[ -n "$AUGMENTUM_AGENT_BRIDGE_URL" ]` check still tries to launch
    agent-bridge.py with a bogus URL.
    """

    a, docker = adapter
    await a.start(
        session_id="sess_solo",
        profile=luanti_profile,
        ports=_ports(),
        bitrate_mbps=4,
        resolution="1280x720",
        encoder="x264",
        world_storage_path="",
        world_settings={},
    )
    config = docker.containers.run.await_args.kwargs.get("config") or {}
    env = {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in config["Env"]}
    assert "AUGMENTUM_AGENT_BRIDGE_URL" not in env


@pytest.mark.asyncio
async def test_start_mounts_uinput_when_gamepad_enabled(
    adapter,
    luanti_profile,
    monkeypatch,
):
    monkeypatch.setattr(
        "augmentum.game_stream.docker_adapter.os.path.exists",
        lambda path: path == "/dev/uinput",
    )
    a, docker = adapter
    await a.start(
        session_id="sess_pad",
        profile=luanti_profile,
        ports=_ports(),
        bitrate_mbps=4,
        resolution="1280x720",
        encoder="x264",
        world_storage_path="",
        world_settings={},
        gamepad_enabled=True,
    )
    hc = docker.containers.run.await_args.kwargs.get("config")["HostConfig"]
    assert hc["Devices"] == [{
        "PathOnHost": "/dev/uinput",
        "PathInContainer": "/dev/uinput",
        "CgroupPermissions": "rwm",
    }]


@pytest.mark.asyncio
async def test_start_does_not_mount_uinput_when_gamepad_disabled(
    adapter,
    luanti_profile,
    monkeypatch,
):
    monkeypatch.setattr(
        "augmentum.game_stream.docker_adapter.os.path.exists",
        lambda path: path == "/dev/uinput",
    )
    a, docker = adapter
    await a.start(
        session_id="sess_no_pad",
        profile=luanti_profile,
        ports=_ports(),
        bitrate_mbps=4,
        resolution="1280x720",
        encoder="x264",
        world_storage_path="",
        world_settings={},
        gamepad_enabled=False,
    )
    hc = docker.containers.run.await_args.kwargs.get("config")["HostConfig"]
    assert "Devices" not in hc


@pytest.mark.asyncio
async def test_start_drops_caps(adapter, luanti_profile):
    a, docker = adapter
    await a.start(
        session_id="sess_caps",
        profile=luanti_profile,
        ports=_ports(),
        bitrate_mbps=4,
        resolution="1280x720",
        encoder="x264",
        world_storage_path="",
        world_settings={},
    )
    hc = docker.containers.run.await_args.kwargs.get("config")["HostConfig"]
    assert hc["CapDrop"] == ["ALL"]
    assert "no-new-privileges:true" in hc["SecurityOpt"]
    assert hc["PidsLimit"] > 0


@pytest.mark.asyncio
async def test_start_with_gpu_adds_device_request(mock_docker, luanti_profile):
    from augmentum.game_stream.docker_adapter import DockerContainerAdapter

    a = DockerContainerAdapter(mock_docker, gpu_passthrough=True)
    await a.start(
        session_id="sess_gpu",
        profile=luanti_profile,
        ports=_ports(),
        bitrate_mbps=4,
        resolution="1280x720",
        encoder="nvenc",
        world_storage_path="",
        world_settings={},
    )
    hc = mock_docker.containers.run.await_args.kwargs.get("config")["HostConfig"]
    assert hc.get("DeviceRequests") == [
        {
            "Driver": "nvidia",
            "Count": -1,
            "Capabilities": [["gpu", "graphics", "utility", "video"]],
        },
    ]


@pytest.mark.asyncio
async def test_stop_removes_container(adapter):
    a, docker = adapter
    container = MagicMock()
    container.stop = AsyncMock()
    container.delete = AsyncMock()
    docker.containers.get = AsyncMock(return_value=container)
    await a.stop("cnt_xyz")
    container.stop.assert_awaited_once()
    container.delete.assert_awaited_once_with(force=True)


@pytest.mark.asyncio
async def test_stop_swallows_missing_container(adapter):
    """Idempotent stop: a 404 from docker.get must not raise."""
    a, docker = adapter
    docker.containers.get = AsyncMock(side_effect=Exception("404 not found"))
    await a.stop("cnt_gone")  # must not raise


@pytest.mark.asyncio
async def test_is_alive_reports_running(adapter):
    a, _ = adapter
    # mock_docker fixture's default returns Running=True
    assert await a.is_alive("cnt_test123") is True


@pytest.mark.asyncio
async def test_is_alive_false_when_get_fails(adapter):
    a, docker = adapter
    docker.containers.get = AsyncMock(side_effect=Exception("404"))
    assert await a.is_alive("cnt_gone") is False


@pytest.mark.asyncio
async def test_list_owned_filters_by_label(adapter):
    a, docker = adapter
    container = MagicMock(id="cnt_abc")
    container.show = AsyncMock(return_value={
        "Config": {"Labels": {
            "augmentum.game_stream": "true",
            "augmentum.game_stream.session_id": "sess_abc",
        }},
        "State": {"Status": "running"},
    })
    docker.containers.list = AsyncMock(return_value=[container])
    rows = await a.list_owned()
    assert len(rows) == 1
    assert rows[0]["session_id"] == "sess_abc"
    # Confirm the filter we used.
    call = docker.containers.list.await_args
    assert call.kwargs["filters"] == {
        "label": ["augmentum.game_stream=true"],
    }


# ── Runtime + adapter integration ─────────────────────────────────────


@pytest.mark.asyncio
async def test_runtime_uses_adapter_to_start_session(mock_docker):
    """End-to-end: runtime.start_session -> adapter.start -> docker.run.

    Confirms the adapter substitution path in server.py works without
    needing real Docker.
    """
    import aiosqlite

    from augmentum.game_stream import GameStreamRuntime, PortPool
    from augmentum.game_stream.docker_adapter import DockerContainerAdapter
    from augmentum.state.game_stream_store import GameStreamStore
    from tests.test_game_stream_smoke import _SCHEMA_SQL

    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA_SQL)
    await conn.execute("INSERT INTO users (id) VALUES ('u1')")
    await conn.commit()

    runtime = GameStreamRuntime(
        store=GameStreamStore(conn),
        port_pool=PortPool(base=40000, count=4),
        adapter=DockerContainerAdapter(mock_docker),
        max_concurrent_per_user=2,
    )
    info = await runtime.start_session(user_id="u1", profile_id="luanti")
    # docker.containers.run was the path of execution.
    mock_docker.containers.run.assert_awaited_once()
    # The session row picked up the container id from the adapter.
    row = await runtime._store.get_session(info.session_id, user_id="u1")
    assert row["container_id"] == "cnt_test123"
    await conn.close()
