"""Phase 1b tests for the under-test boot orchestrator.

Uses Python's stdlib ``http.server`` as the fixture app — it boots
fast and is guaranteed available in CI. Each test gets a fresh
registry + a free port so no cross-test interference.

Covers:
* Successful boot end-to-end (spawn → healthcheck → handle)
* Healthcheck timeout (app starts but never serves on the port)
* Process-exit-before-healthcheck (bad command)
* Command-error (binary not found)
* Teardown contract (clean / forced / skipped)
* Registry membership + per-run scoping
* Tool wrapper validation + output shape
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

from augmentum.bug_finder.agent_tools import (
    BootUnderTestTool,
    UnderTestStatusTool,
    build_pen_test_tools,
)
from augmentum.bug_finder.pen_test_boot import (
    BootSpec,
    _UnderTestRegistry,
    boot_under_test,
    teardown_service,
)


def _free_port() -> int:
    """Reserve a free port. Race-y but good enough for tests — the
    fixture immediately spawns onto it."""
    s = socket.socket()
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def registry() -> _UnderTestRegistry:
    return _UnderTestRegistry()


@pytest.fixture
def fake_workspace(tmp_path: Path) -> Path:
    """Workspace root that doesn't really need any specific contents
    for boot tests, but must exist so cwd resolution works."""
    (tmp_path / "stub.txt").write_text("placeholder", encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Successful boot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_boot_under_test_happy_path(
    fake_workspace: Path, registry: _UnderTestRegistry,
) -> None:
    """Spawn ``python -m http.server <port>`` and confirm healthcheck
    passes. This is the contract every other test is a variant of."""
    port = _free_port()
    spec = BootSpec(
        command=(sys.executable, "-m", "http.server", str(port)),
        port=port,
        healthcheck_path="/",
        boot_timeout_s=20.0,
        healthcheck_timeout_s=15.0,
    )
    result = await boot_under_test(fake_workspace, spec, registry=registry)
    try:
        assert result.ok, (
            f"boot failed: "
            f"{result.failure.reason if result.failure else None}: "
            f"{result.failure.detail if result.failure else None}"
        )
        svc = result.service
        assert svc.base_url == f"http://localhost:{port}"
        assert svc.pid > 0
        assert svc.healthy is True
        # Must be in the registry
        assert registry.get(svc.service_id) is svc
    finally:
        await registry.teardown_all()


@pytest.mark.asyncio
async def test_boot_persists_log_under_substrate(
    fake_workspace: Path, registry: _UnderTestRegistry,
) -> None:
    """stdout/stderr land in ``.augmentum/bug_finder/under_test_logs/``."""
    port = _free_port()
    spec = BootSpec(
        command=(sys.executable, "-m", "http.server", str(port)),
        port=port,
    )
    result = await boot_under_test(fake_workspace, spec, registry=registry)
    try:
        assert result.ok
        log_path = result.service.log_path
        assert log_path is not None
        assert log_path.is_file()
        assert log_path.parent.name == "under_test_logs"
    finally:
        await registry.teardown_all()


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_boot_reports_command_not_found(
    fake_workspace: Path, registry: _UnderTestRegistry,
) -> None:
    spec = BootSpec(
        command=("nope-this-binary-does-not-exist-xyz",),
        port=_free_port(),
        boot_timeout_s=2.0,
    )
    result = await boot_under_test(fake_workspace, spec, registry=registry)
    assert not result.ok
    assert result.failure.reason == "command_error"


@pytest.mark.asyncio
async def test_boot_reports_exit_before_healthcheck(
    fake_workspace: Path, registry: _UnderTestRegistry,
) -> None:
    """A command that exits immediately (no server) gets reported as
    ``exit`` with the actual exit code captured."""
    spec = BootSpec(
        command=(sys.executable, "-c", "import sys; sys.exit(42)"),
        port=_free_port(),
        boot_timeout_s=5.0,
        healthcheck_interval_s=0.1,
    )
    result = await boot_under_test(fake_workspace, spec, registry=registry)
    assert not result.ok
    assert result.failure.reason == "exit"
    assert result.failure.exit_code == 42


@pytest.mark.asyncio
async def test_boot_reports_healthcheck_timeout(
    fake_workspace: Path, registry: _UnderTestRegistry,
) -> None:
    """Spawn a long-running process that doesn't bind any port. The
    healthcheck should time out, and the process should be torn down."""
    port = _free_port()
    spec = BootSpec(
        # Sleep for longer than the boot timeout — process stays alive
        # but never serves anything. asyncio uses time.sleep here is fine
        # because the test process tears it down on timeout.
        command=(sys.executable, "-c", "import time; time.sleep(60)"),
        port=port,
        boot_timeout_s=3.0,
        healthcheck_timeout_s=2.0,
        healthcheck_interval_s=0.2,
    )
    result = await boot_under_test(fake_workspace, spec, registry=registry)
    assert not result.ok
    assert result.failure.reason == "timeout"


@pytest.mark.asyncio
async def test_boot_empty_command_refused(
    fake_workspace: Path, registry: _UnderTestRegistry,
) -> None:
    spec = BootSpec(command=(), port=8080)
    result = await boot_under_test(fake_workspace, spec, registry=registry)
    assert not result.ok
    assert result.failure.reason == "command_error"


@pytest.mark.asyncio
async def test_boot_missing_cwd_refused(
    tmp_path: Path, registry: _UnderTestRegistry,
) -> None:
    spec = BootSpec(
        command=(sys.executable, "-c", "print('hi')"),
        port=_free_port(),
        cwd="does/not/exist",
    )
    result = await boot_under_test(tmp_path, spec, registry=registry)
    assert not result.ok
    assert result.failure.reason == "command_error"


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_teardown_clean_when_process_exits_promptly(
    fake_workspace: Path, registry: _UnderTestRegistry,
) -> None:
    port = _free_port()
    spec = BootSpec(
        command=(sys.executable, "-m", "http.server", str(port)),
        port=port,
    )
    result = await boot_under_test(fake_workspace, spec, registry=registry)
    assert result.ok
    verdict = await teardown_service(result.service, grace_seconds=5.0)
    assert verdict in {"clean", "forced"}
    assert result.service.teardown_called


@pytest.mark.asyncio
async def test_teardown_skips_already_torn_down(
    fake_workspace: Path, registry: _UnderTestRegistry,
) -> None:
    port = _free_port()
    spec = BootSpec(
        command=(sys.executable, "-m", "http.server", str(port)),
        port=port,
    )
    result = await boot_under_test(fake_workspace, spec, registry=registry)
    assert result.ok
    await teardown_service(result.service)
    # Second call must be a no-op
    second = await teardown_service(result.service)
    assert second == "skipped"


@pytest.mark.asyncio
async def test_teardown_all_handles_multiple_services(
    fake_workspace: Path, registry: _UnderTestRegistry,
) -> None:
    p1 = _free_port()
    p2 = _free_port()
    # Ensure we don't collide with our own first port
    while p2 == p1:
        p2 = _free_port()
    r1 = await boot_under_test(
        fake_workspace,
        BootSpec(
            command=(sys.executable, "-m", "http.server", str(p1)),
            port=p1,
        ),
        registry=registry,
    )
    r2 = await boot_under_test(
        fake_workspace,
        BootSpec(
            command=(sys.executable, "-m", "http.server", str(p2)),
            port=p2,
        ),
        registry=registry,
    )
    assert r1.ok and r2.ok
    verdicts = await registry.teardown_all()
    assert set(verdicts) == {r1.service.service_id, r2.service.service_id}
    for v in verdicts.values():
        assert v in {"clean", "forced"}


# ---------------------------------------------------------------------------
# Tool wrappers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_boot_tool_validates_required_args(
    fake_workspace: Path, registry: _UnderTestRegistry,
) -> None:
    tool = BootUnderTestTool(fake_workspace, registry=registry)
    res = await tool.execute()
    assert not res.success
    assert res.validation_error


@pytest.mark.asyncio
async def test_boot_tool_validates_port_range(
    fake_workspace: Path, registry: _UnderTestRegistry,
) -> None:
    tool = BootUnderTestTool(fake_workspace, registry=registry)
    res = await tool.execute(command=["x"], port=0)
    assert not res.success
    assert res.validation_error
    assert "port" in res.error


@pytest.mark.asyncio
async def test_boot_tool_emits_parseable_json_on_success(
    fake_workspace: Path, registry: _UnderTestRegistry,
) -> None:
    port = _free_port()
    tool = BootUnderTestTool(fake_workspace, registry=registry)
    try:
        res = await tool.execute(
            command=[sys.executable, "-m", "http.server", str(port)],
            port=port,
            boot_timeout_s=20.0,
            healthcheck_timeout_s=15.0,
        )
        assert res.success
        data = json.loads(res.output)
        assert data["ok"] is True
        assert data["base_url"] == f"http://localhost:{port}"
        assert data["pid"] > 0
        assert data["service_id"].startswith("ut_")
    finally:
        await registry.teardown_all()


@pytest.mark.asyncio
async def test_boot_tool_emits_failure_with_log_tail(
    fake_workspace: Path, registry: _UnderTestRegistry,
) -> None:
    tool = BootUnderTestTool(fake_workspace, registry=registry)
    res = await tool.execute(
        command=[
            sys.executable, "-c",
            "import sys; sys.stderr.write('boom\\n'); sys.exit(7)",
        ],
        port=_free_port(),
        boot_timeout_s=3.0,
        healthcheck_timeout_s=2.0,
    )
    assert not res.success
    data = json.loads(res.output)
    assert data["ok"] is False
    assert data["reason"] == "exit"
    assert data["exit_code"] == 7
    # The log tail should contain our stderr breadcrumb.
    assert "boom" in data["log_tail"]


@pytest.mark.asyncio
async def test_under_test_status_tool(
    fake_workspace: Path, registry: _UnderTestRegistry,
) -> None:
    port = _free_port()
    boot_tool = BootUnderTestTool(fake_workspace, registry=registry)
    status_tool = UnderTestStatusTool(registry=registry)
    try:
        res = await boot_tool.execute(
            command=[sys.executable, "-m", "http.server", str(port)],
            port=port,
        )
        assert res.success
        sid = json.loads(res.output)["service_id"]
        status_res = await status_tool.execute(service_id=sid)
        assert status_res.success
        status = json.loads(status_res.output)
        assert status["known"] is True
        assert status["healthy"] is True
    finally:
        await registry.teardown_all()


@pytest.mark.asyncio
async def test_under_test_status_tool_unknown_id(
    registry: _UnderTestRegistry,
) -> None:
    tool = UnderTestStatusTool(registry=registry)
    res = await tool.execute(service_id="ut_nonexistent")
    data = json.loads(res.output)
    assert data["ok"] is False
    assert data["known"] is False


# ---------------------------------------------------------------------------
# Role isolation — still must hold after Phase 1b lands
# ---------------------------------------------------------------------------


def test_boot_under_test_not_in_deterministic_tools(
    fake_workspace: Path,
) -> None:
    from augmentum.bug_finder.agent_tools import build_deterministic_tools
    tools = build_deterministic_tools(fake_workspace)
    names = {t.name for t in tools}
    assert "boot_under_test" not in names
    assert "under_test_status" not in names


def test_pen_test_tool_names_now_contain_boot_under_test() -> None:
    """Canonical registry must list the new tools so allow-lists
    elsewhere can reference them by name."""
    from augmentum.agents.tools import PEN_TEST_TOOL_NAMES
    assert "boot_under_test" in PEN_TEST_TOOL_NAMES
    assert "under_test_status" in PEN_TEST_TOOL_NAMES
    assert "http_attack" in PEN_TEST_TOOL_NAMES


def test_pen_test_tool_names_still_isolated_from_existing_roles() -> None:
    """The role-isolation invariant from Phase 1a must hold for the
    Phase 1b tools too. No existing role's allow-list may contain
    any pen-test tool."""
    from augmentum.agents.tools import (
        COMPREHENDER_TOOL_NAMES,
        DETECTOR_TOOL_NAMES,
        FIXER_TOOL_NAMES,
        INVESTIGATOR_TOOL_NAMES,
        LEAD_TOOL_NAMES,
        PEN_TEST_TOOL_NAMES,
        PLANNER_TOOL_NAMES,
        READ_ONLY_TOOL_NAMES,
        VERIFIER_TOOL_NAMES,
    )
    for role_names in (
        READ_ONLY_TOOL_NAMES,
        PLANNER_TOOL_NAMES,
        DETECTOR_TOOL_NAMES,
        COMPREHENDER_TOOL_NAMES,
        INVESTIGATOR_TOOL_NAMES,
        LEAD_TOOL_NAMES,
        VERIFIER_TOOL_NAMES,
        FIXER_TOOL_NAMES,
    ):
        assert PEN_TEST_TOOL_NAMES.isdisjoint(role_names)


def test_build_pen_test_tools_includes_phase_1b_tools(
    fake_workspace: Path,
) -> None:
    """The exact-set contract belongs to the latest phase test.
    Phase 1b's contribution is the boot + status tools — verify they
    both land. The exact composition is owned by the canonical
    PEN_TEST_TOOL_NAMES + the test in
    test_bug_finder_pen_test_attacks.py."""
    tools = build_pen_test_tools(fake_workspace)
    names = {t.name for t in tools}
    assert "http_attack" in names
    assert "boot_under_test" in names
    assert "under_test_status" in names
