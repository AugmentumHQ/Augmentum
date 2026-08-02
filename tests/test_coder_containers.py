"""Container manager tests (mocked Docker)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.coder.containers import (
    ContainerManager,
    _parse_ls_output,
    _parse_memory,
    _resolve_workspace_network_mode,
)
from augmentum.coder.models import ContainerInfo, FileEntry


class TestResolveWorkspaceNetworkMode:
    """Setting contract: "bridge"/"none" honored, anything else falls
    back to bridge so a typo'd settings write can't brick workspace
    creation."""

    def test_default_is_bridge(self):
        from augmentum.config import settings
        assert getattr(settings, "coder_workspace_network_mode", "bridge") == "bridge"
        assert _resolve_workspace_network_mode() == "bridge"

    @pytest.mark.parametrize("value,expected", [
        ("bridge", "bridge"),
        ("none", "none"),
        ("NONE", "none"),        # case-normalized
        (" none ", "none"),      # whitespace-tolerant
        ("host", "bridge"),      # host networking is never allowed
        ("container:x", "bridge"),
        ("", "bridge"),
        ("garbage", "bridge"),
    ])
    def test_values(self, value, expected, monkeypatch):
        from augmentum.config import settings
        monkeypatch.setattr(settings, "coder_workspace_network_mode", value)
        assert _resolve_workspace_network_mode() == expected


@pytest.fixture
def mock_docker():
    client = AsyncMock()
    client.containers = AsyncMock()
    mock_container = MagicMock()
    mock_container.id = "abc123"
    client.containers.run = AsyncMock(return_value=mock_container)
    client.containers.list = AsyncMock(return_value=[])
    # PR 2: by default no prebaked images are present, so the resolver
    # falls all the way through to ``ubuntu:24.04`` and the install-line
    # path emits (matches v1 behavior these tests were written against).
    # Tests that want to exercise the prebake-skip path mock inspect
    # explicitly — see ``tests/test_workspace_profiles.py``.
    client.images = MagicMock()
    client.images.inspect = AsyncMock(side_effect=RuntimeError("image not present"))
    return client


@pytest.fixture
def manager(mock_docker):
    return ContainerManager(docker=mock_docker, db=None)


# ------------------------------------------------------------------
# create_workspace
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_workspace(manager, mock_docker):
    info = await manager.create_workspace("test-ws", base_image="ubuntu:24.04")
    assert info.name == "test-ws"
    assert info.status == "running"
    assert info.container_id == "abc123"
    mock_docker.containers.run.assert_called_once()


@pytest.mark.asyncio
async def test_create_workspace_with_packages(manager, mock_docker):
    info = await manager.create_workspace(
        "dev-ws",
        base_image="ubuntu:24.04",
        packages=["git", "curl"],
    )
    assert info.name == "dev-ws"
    assert info.status == "running"
    call_kwargs = mock_docker.containers.run.call_args
    config = call_kwargs.kwargs.get("config") or call_kwargs.args[0]
    cmd_str = " ".join(config["Cmd"])
    assert "git" in cmd_str
    assert "curl" in cmd_str


@pytest.mark.asyncio
async def test_create_workspace_security_config(manager, mock_docker):
    """Verify cap_drop ALL and no-new-privileges are set."""
    await manager.create_workspace("secure-ws")
    call_kwargs = mock_docker.containers.run.call_args
    config = call_kwargs.kwargs.get("config") or call_kwargs.args[0]
    host_config = config["HostConfig"]
    assert host_config["CapDrop"] == ["ALL"]
    assert "no-new-privileges:true" in host_config["SecurityOpt"]


@pytest.mark.asyncio
async def test_create_workspace_resource_limits(manager, mock_docker):
    """Verify CPU, memory, and PID limits are applied."""
    await manager.create_workspace("limited-ws", cpu=1.0, memory="1g", pids=128)
    call_kwargs = mock_docker.containers.run.call_args
    config = call_kwargs.kwargs.get("config") or call_kwargs.args[0]
    host_config = config["HostConfig"]
    assert host_config["NanoCpus"] == int(1.0 * 1e9)
    assert host_config["Memory"] == 1 * 1024 * 1024 * 1024
    assert host_config["PidsLimit"] == 128


@pytest.mark.asyncio
async def test_create_workspace_labels(manager, mock_docker):
    """Verify Docker labels are set correctly."""
    info = await manager.create_workspace("labelled-ws")
    call_kwargs = mock_docker.containers.run.call_args
    config = call_kwargs.kwargs.get("config") or call_kwargs.args[0]
    labels = config["Labels"]
    assert labels["augmentum.workspace"] == "true"
    assert labels["augmentum.name"] == "labelled-ws"
    assert labels["augmentum.id"] == info.id
    assert labels["augmentum.tooling_profile"] == "browser"


@pytest.mark.asyncio
async def test_create_workspace_power_profile_installs_agent_extras(manager, mock_docker):
    info = await manager.create_workspace("power-ws", tooling_profile="power")

    assert info.tooling_profile == "power"
    call_kwargs = mock_docker.containers.run.call_args
    config = call_kwargs.kwargs.get("config") or call_kwargs.args[0]
    cmd_str = " ".join(config["Cmd"])
    labels = config["Labels"]

    assert labels["augmentum.tooling_profile"] == "power"
    assert "tooling-profile.json" in cmd_str
    assert "procps iproute2 lsof dnsutils" in cmd_str
    assert "cmake ninja-build gdb strace shellcheck shfmt" in cmd_str
    assert "uv pipx pre-commit pytest-cov ipython" in cmd_str
    assert "npm install -g pnpm yarn" in cmd_str


@pytest.mark.asyncio
async def test_create_workspace_browser_profile_installs_playwright(manager, mock_docker):
    info = await manager.create_workspace("browser-ws", tooling_profile="browser")

    assert info.tooling_profile == "browser"
    call_kwargs = mock_docker.containers.run.call_args
    config = call_kwargs.kwargs.get("config") or call_kwargs.args[0]
    cmd_str = " ".join(config["Cmd"])

    assert config["Labels"]["augmentum.tooling_profile"] == "browser"
    assert "playwright pytest-playwright" in cmd_str
    assert "python3 -m playwright install --with-deps chromium" in cmd_str


@pytest.mark.asyncio
async def test_create_workspace_git_url(manager, mock_docker):
    info = await manager.create_workspace("git-ws", git_url="https://github.com/example/repo")
    assert info.git_url == "https://github.com/example/repo"


@pytest.mark.asyncio
async def test_create_workspace_sets_git_identity(manager, mock_docker):
    await manager.create_workspace("git-config-ws")
    call_kwargs = mock_docker.containers.run.call_args
    config = call_kwargs.kwargs.get("config") or call_kwargs.args[0]
    cmd_str = " ".join(config["Cmd"])
    assert "git config --global user.name" in cmd_str
    assert "git config --global user.email" in cmd_str


@pytest.mark.asyncio
async def test_create_workspace_fallback_bootstraps_guide_critical_stack(
    manager,
    mock_docker,
):
    """Fallback ubuntu workspaces should bootstrap the guide-critical toolchain."""
    await manager.create_workspace("fallback-ws", base_image="ubuntu:24.04")
    call_kwargs = mock_docker.containers.run.call_args
    config = call_kwargs.kwargs.get("config") or call_kwargs.args[0]
    cmd_str = " ".join(config["Cmd"])

    assert "golang-go" in cmd_str
    assert "httpie" in cmd_str
    assert "python3-tk" in cmd_str
    assert "xvfb x11-utils" in cmd_str
    assert "postgresql-client redis-tools" in cmd_str
    assert "pytest ruff black mypy" in cmd_str
    assert "typescript ts-node eslint prettier" in cmd_str
    assert "npm install -g npm@latest" not in cmd_str
    assert "rm -f /usr/lib/python3*/EXTERNALLY-MANAGED" in cmd_str
    assert "ln -sf /usr/bin/fdfind /usr/local/bin/fd" in cmd_str


@pytest.mark.asyncio
async def test_enable_published_ports_recreates_container_with_bindings(mock_docker):
    manager = ContainerManager(docker=mock_docker, db=None)
    old_container = AsyncMock()
    old_container.show = AsyncMock(return_value={
        "Config": {"Image": "augmentum-workspace"},
        "State": {"Status": "running"},
        "HostConfig": {"PidsLimit": 256},
        "NetworkSettings": {"Ports": {}},
    })
    old_container.stop = AsyncMock()
    old_container.delete = AsyncMock()
    new_container = MagicMock()
    new_container.id = "new123"

    manager._get_workspace = AsyncMock(return_value=ContainerInfo(
        id="ws-1",
        name="test",
        container_id="old123",
        status="running",
        resources_cpu=2.0,
        resources_memory="2g",
        tooling_profile="power",
    ))
    mock_docker.containers.get = AsyncMock(return_value=old_container)
    mock_docker.containers.run = AsyncMock(return_value=new_container)

    info, changed = await manager.enable_published_ports("ws-1")

    assert changed is True
    assert info.container_id == "new123"
    call_kwargs = mock_docker.containers.run.call_args
    config = call_kwargs.kwargs.get("config") or call_kwargs.args[0]
    assert "3000/tcp" in config["ExposedPorts"]
    assert config["HostConfig"]["PortBindings"]["8080/tcp"][0]["HostIp"] == "127.0.0.1"
    assert config["Labels"]["augmentum.tooling_profile"] == "power"
    assert "uv pipx pre-commit pytest-cov ipython" in " ".join(config["Cmd"])
    old_container.stop.assert_awaited_once()
    old_container.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_ports_reports_unpublished_ports_as_host_zero(mock_docker):
    manager = ContainerManager(docker=mock_docker, db=None)
    manager._get_workspace = AsyncMock(return_value=ContainerInfo(
        id="ws-1",
        name="test",
        container_id="old123",
        status="running",
    ))
    container = AsyncMock()
    container.show = AsyncMock(return_value={"NetworkSettings": {"Ports": {}}})
    mock_docker.containers.get = AsyncMock(return_value=container)

    ports = await manager.list_ports("ws-1")

    assert ports
    assert all(p["host_port"] == 0 for p in ports)
    assert all(p["listening"] is False for p in ports)


# ------------------------------------------------------------------
# list_workspaces
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_workspaces_empty(manager):
    workspaces = await manager.list_workspaces()
    assert workspaces == []


@pytest.mark.asyncio
async def test_list_workspaces_no_db(mock_docker):
    """ContainerManager with db=None returns empty list."""
    mgr = ContainerManager(docker=mock_docker, db=None)
    result = await mgr.list_workspaces()
    assert result == []


# ------------------------------------------------------------------
# _parse_memory
# ------------------------------------------------------------------

def test_parse_memory_gigabytes():
    assert _parse_memory("2g") == 2 * 1024 * 1024 * 1024


def test_parse_memory_megabytes():
    assert _parse_memory("512m") == 512 * 1024 * 1024


def test_parse_memory_kilobytes():
    assert _parse_memory("1024k") == 1024 * 1024


def test_parse_memory_bytes():
    assert _parse_memory("65536") == 65536


def test_parse_memory_uppercase():
    assert _parse_memory("4G") == 4 * 1024 * 1024 * 1024


def test_parse_memory_uppercase_mb():
    assert _parse_memory("256M") == 256 * 1024 * 1024


def test_parse_memory_with_whitespace():
    assert _parse_memory("  1g  ") == 1 * 1024 * 1024 * 1024


# ------------------------------------------------------------------
# _parse_ls_output
# ------------------------------------------------------------------

def test_parse_ls_output_files():
    ls_output = """\
total 8
drwxr-xr-x 2 root root 4096 1711234567 subdir
-rw-r--r-- 1 root root  512 1711234568 file.txt
"""
    entries = _parse_ls_output(ls_output, "/workspace")
    assert len(entries) == 2

    dirs = [e for e in entries if e.is_dir]
    files = [e for e in entries if not e.is_dir]

    assert len(dirs) == 1
    assert dirs[0].name == "subdir"
    assert dirs[0].path == "/workspace/subdir"
    assert dirs[0].size == 4096

    assert len(files) == 1
    assert files[0].name == "file.txt"
    assert files[0].path == "/workspace/file.txt"
    assert files[0].size == 512
    assert files[0].modified == 1711234568.0


def test_parse_ls_output_skips_dots():
    ls_output = """\
total 0
drwxr-xr-x 2 root root 4096 1711234567 .
drwxr-xr-x 3 root root 4096 1711234560 ..
-rw-r--r-- 1 root root  100 1711234570 hello.py
"""
    entries = _parse_ls_output(ls_output, "/workspace")
    assert len(entries) == 1
    assert entries[0].name == "hello.py"


def test_parse_ls_output_empty():
    entries = _parse_ls_output("total 0\n", "/workspace")
    assert entries == []


def test_parse_ls_output_nested_path():
    ls_output = "-rw-r--r-- 1 root root 256 1711234567 main.py\n"
    entries = _parse_ls_output(ls_output, "/workspace/src")
    assert entries[0].path == "/workspace/src/main.py"


def test_parse_ls_output_trailing_slash_normalised():
    ls_output = "-rw-r--r-- 1 root root 256 1711234567 readme.md\n"
    entries = _parse_ls_output(ls_output, "/workspace/")
    assert entries[0].path == "/workspace/readme.md"


# ------------------------------------------------------------------
# ContainerInfo dataclass
# ------------------------------------------------------------------

def test_container_info_defaults():
    info = ContainerInfo(id="ws-1", name="my-workspace")
    assert info.container_id is None
    assert info.status == "stopped"
    assert info.resources_cpu == 2.0
    assert info.resources_memory == "2g"


def test_container_info_full():
    info = ContainerInfo(
        id="ws-2",
        name="dev",
        container_id="docker123",
        status="running",
        git_url="https://github.com/example/repo",
        resources_cpu=4.0,
        resources_memory="4g",
    )
    assert info.container_id == "docker123"
    assert info.status == "running"
    assert info.resources_cpu == 4.0


# ------------------------------------------------------------------
# FileEntry dataclass
# ------------------------------------------------------------------

def test_file_entry_directory():
    entry = FileEntry(name="src", path="/workspace/src", is_dir=True, size=4096)
    assert entry.is_dir is True
    assert entry.modified == 0.0


def test_file_entry_file():
    entry = FileEntry(name="main.py", path="/workspace/main.py", is_dir=False, size=1024, modified=1234567.0)
    assert entry.is_dir is False
    assert entry.modified == 1234567.0


# ------------------------------------------------------------------
# git_checkpoint
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_git_checkpoint_returns_hash_when_head_advances(manager):
    manager._run_command = AsyncMock(side_effect=[
        "",          # before HEAD
        "",          # git add -A
        "M foo.py",  # git status --porcelain
        "",          # git commit -q
        "abc1234",   # after HEAD
        "",          # git push origin HEAD:refs/heads/main
    ])

    sha = await manager.git_checkpoint("ws-1", "Agent: code_edit foo.py")

    assert sha == "abc1234"


@pytest.mark.asyncio
async def test_git_checkpoint_rejects_fatal_rev_parse_output(manager):
    manager._run_command = AsyncMock(side_effect=[
        "",                                 # before HEAD
        "",                                 # git add -A
        "M foo.py",                         # git status --porcelain
        "Author identity unknown",          # git commit -q stderr/stdout
        "fatal: Needed a single revision",  # after HEAD probe
    ])

    sha = await manager.git_checkpoint("ws-1", "Agent: code_edit foo.py")

    assert sha is None


# ------------------------------------------------------------------
# Cancel / exec tracking (2026-04-22)
# ------------------------------------------------------------------

def test_active_execs_starts_empty(manager):
    """Fresh manager has no tracked execs for any workspace."""
    assert manager._active_execs == {}


@pytest.mark.asyncio
async def test_cancel_workspace_execs_with_no_active_returns_zero(manager):
    """Cancelling when nothing is running is a no-op, no exceptions."""
    # No _get_workspace data needed because the empty-set early return
    # fires before the container lookup.
    killed = await manager.cancel_workspace_execs("ws-none")
    assert killed == 0


@pytest.mark.asyncio
async def test_cancel_workspace_execs_signals_each_tracked_exec(monkeypatch):
    """Registers two fake execs, calls cancel, verifies both get a
    kill attempt. Uses a minimal stub for container lookup because we
    don't need real docker machinery here."""
    client = AsyncMock()
    manager = ContainerManager(docker=client, db=None)

    # Populate the tracked set manually — this is what ``_run_command``
    # does when an exec is in flight.
    fake_exec_1 = MagicMock(id="exec-1")
    fake_exec_2 = MagicMock(id="exec-2")
    manager._active_execs["ws-kill"] = {
        (fake_exec_1, "exec-1"),
        (fake_exec_2, "exec-2"),
    }

    # Mock _get_workspace so the cancel path can find a container.
    from augmentum.coder.models import ContainerInfo
    manager._get_workspace = AsyncMock(return_value=ContainerInfo(
        id="ws-kill", name="kill", container_id="c1", status="running",
    ))

    # Container + exec mocks — kill_exec calls container.exec(...) to
    # issue the signal. We track the calls and assert afterward.
    kill_invocations = []

    async def fake_exec(**kwargs):
        kill_invocations.append(kwargs)
        stream = MagicMock()
        stream.read_out = AsyncMock(return_value=None)
        kill_exec_obj = MagicMock()
        kill_exec_obj.start = MagicMock(return_value=stream)
        return kill_exec_obj

    fake_container = MagicMock()
    fake_container.exec = fake_exec
    client.containers.get = AsyncMock(return_value=fake_container)

    # Neither fake exec has .inspect returning a PID, so both fall
    # through to the pkill fallback path — but both still get a
    # kill attempt, which is what we're verifying.
    fake_exec_1.inspect = AsyncMock(return_value={})
    fake_exec_2.inspect = AsyncMock(return_value={})

    killed = await manager.cancel_workspace_execs("ws-kill")
    assert killed == 2
    # Both fake execs went through kill_exec → container.exec('pkill...')
    assert len(kill_invocations) == 2


# ------------------------------------------------------------------
# file_write shell-injection hardening (P0)
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_file_write_quotes_path_against_injection(manager):
    """A malicious redirect target must be passed as a single quoted
    argument, not interpolated raw into the sh -c command."""
    import shlex

    calls = []

    async def fake_run(workspace_id, argv, **kwargs):
        calls.append(argv)
        return ""

    manager._run_command = fake_run

    evil_path = "/workspace/foo.txt; touch /tmp/pwned"
    await manager.file_write("ws", evil_path, "hello")

    assert len(calls) == 1
    sh_command = calls[0][2]  # ["sh", "-c", "<command>"]
    # The dangerous tail must be inside the quoted path, not a separate
    # shell statement.
    assert shlex.quote(evil_path) in sh_command
    assert "; touch /tmp/pwned > " not in sh_command
    # Tokenizing the redirect target yields exactly the literal path.
    tokens = shlex.split(sh_command.split(">", 1)[1].strip())
    assert tokens == [evil_path]


@pytest.mark.asyncio
async def test_file_write_preserves_content_quoting(manager):
    """Content with single quotes still round-trips through the escape."""
    calls = []

    async def fake_run(workspace_id, argv, **kwargs):
        calls.append(argv)
        return ""

    manager._run_command = fake_run

    await manager.file_write("ws", "/workspace/a.py", "x = 'hi'")
    sh_command = calls[0][2]
    assert "printf '%s'" in sh_command
    # Each single quote in the content becomes the POSIX escape '\''.
    assert r"'\''hi'\''" in sh_command


# ---------------------------------------------------------------------------
# Timeout termination — regression guard.
#
# Both timeout paths used to abandon the exec instead of signalling it:
# ``wait_for`` cancelled the Python coroutine reading stdout, but detaching
# from a docker exec never stops the process. A background Claude Code run hit
# the (then hardcoded) 900s wall-clock cap at 161 tool calls and kept editing
# /workspace for another ~12 minutes, unobserved, while the UI showed it as
# failed with the generic "claude ended without a result" — because the string
# naming the real cause was appended to a return value the streaming caller
# discards. These tests pin all three halves of that fix: kill, raise, and the
# option to have no wall-clock ceiling at all.
# ---------------------------------------------------------------------------

def _exec_harness(manager, *, stream_read):
    """Wire the minimum docker machinery ``_run_command`` needs, with a
    caller-supplied ``read_out``. Returns the exec object under test."""
    manager._get_workspace = AsyncMock(return_value=ContainerInfo(
        id="ws-t", name="t", container_id="c1", status="running",
    ))
    manager._touch_last_active = AsyncMock()

    stream = MagicMock()
    stream.read_out = stream_read
    exec_obj = MagicMock(id="exec-hang")
    exec_obj.start = MagicMock(return_value=stream)
    exec_obj.inspect = AsyncMock(return_value={"ExitCode": 0})

    container = MagicMock()
    container.exec = AsyncMock(return_value=exec_obj)
    manager._docker.containers.get = AsyncMock(return_value=container)
    return exec_obj


@pytest.mark.asyncio
async def test_wall_clock_timeout_kills_the_exec(manager):
    """A wall-clock expiry must SIGNAL the process, not just stop reading it."""
    import asyncio

    async def _never():
        await asyncio.sleep(3600)

    _exec_harness(manager, stream_read=_never)
    manager._kill_exec = AsyncMock()

    out = await manager._run_command("ws-t", ["sleep", "9999"], timeout=0.05)

    manager._kill_exec.assert_awaited()          # the orphan bug
    assert "wall-clock" in out                    # and it says so


@pytest.mark.asyncio
async def test_wall_clock_timeout_raises_when_strict(manager):
    """Streaming callers ignore the return value, so strict mode must raise."""
    import asyncio

    from augmentum.coder.containers import ExecAborted

    async def _never():
        await asyncio.sleep(3600)

    _exec_harness(manager, stream_read=_never)
    manager._kill_exec = AsyncMock()

    with pytest.raises(ExecAborted) as caught:
        await manager._run_command(
            "ws-t", ["sleep", "9999"], timeout=0.05, strict=True,
        )

    assert caught.value.kind == "wall_clock"
    manager._kill_exec.assert_awaited()


@pytest.mark.asyncio
async def test_idle_timeout_kills_the_exec(manager):
    """The idle path's message always claimed 'was killed'. Make it true."""
    import asyncio

    async def _never():
        await asyncio.sleep(3600)

    _exec_harness(manager, stream_read=_never)
    manager._kill_exec = AsyncMock()

    out = await manager._run_command(
        "ws-t", ["sleep", "9999"], timeout=None, idle_timeout=0.05,
    )

    manager._kill_exec.assert_awaited()
    assert "went silent" in out


@pytest.mark.asyncio
async def test_timeout_none_means_no_wall_clock_ceiling(manager):
    """``timeout=None`` lets an open-ended background session run to its own
    completion — the point of the fix, not merely a bigger number."""
    msgs = [MagicMock(data=b"done\n"), None]

    async def _drain():
        return msgs.pop(0)

    _exec_harness(manager, stream_read=_drain)
    manager._kill_exec = AsyncMock()

    out = await manager._run_command("ws-t", ["echo", "done"], timeout=None)

    assert out == "done\n"
    manager._kill_exec.assert_not_awaited()      # nothing abnormal happened


@pytest.mark.asyncio
async def test_strict_surfaces_nonzero_exit_code(manager):
    """A crashed process must report as a crash, not as a clean finish —
    otherwise a dead CLI reaches the user as 'ended without a result'."""
    from augmentum.coder.containers import ExecAborted

    msgs = [MagicMock(data=b"boom\n"), None]

    async def _drain():
        return msgs.pop(0)

    exec_obj = _exec_harness(manager, stream_read=_drain)
    exec_obj.inspect = AsyncMock(return_value={"ExitCode": 137})

    with pytest.raises(ExecAborted) as caught:
        await manager._run_command("ws-t", ["boom"], timeout=None, strict=True)

    assert caught.value.kind == "exit_code"
    assert "137" in caught.value.detail
    assert caught.value.partial == "boom\n"      # output is preserved, not lost
