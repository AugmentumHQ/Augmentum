from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from augmentum.powers import PowerRegistry, PowerStateStore, select_controller_power
from augmentum.state.settings_store import SettingsStore

from tests.test_coder_handler import _FakeBackend, _FakeChunk, _make_request


def _write_power(root: Path, slug: str, manifest_name: str, body: str) -> None:
    pkg = root / slug
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / manifest_name).write_text(body, encoding="utf-8")


def _make_registry(tmp_path: Path) -> PowerRegistry:
    native_root = tmp_path / ".augmentum" / "powers"
    compat_root = tmp_path / ".claude" / "skills"
    _write_power(
        native_root,
        "browser-verification",
        "POWER.md",
        """---
name: Browser Verification
description: >
  Validate frontend flows and UI regressions from inside coder mode.
kind: verifier
activation_policy: controller
activation_windows:
  - post_write
  - pre_finish
modes:
  - coder
preferred_tools:
  - file_read
  - shell_exec
---
# Browser Verification

Use this Power for browser-facing debugging, reproduction, and validation.
Favor targeted checks before broad rewrites.
""",
    )
    _write_power(
        compat_root,
        "mcp-builder",
        "SKILL.md",
        """---
name: MCP Builder
description: Scaffold and validate MCP-backed integrations.
kind: integration
activation_policy: manual
user-invocable: true
---
# MCP Builder

Map external systems into a clean MCP shape before expanding orchestration.
""",
    )
    return PowerRegistry(
        search_roots=[
            (native_root, "native"),
            (compat_root, "compat"),
        ],
    )


def _repo_native_registry() -> PowerRegistry:
    repo_root = Path(__file__).resolve().parents[1]
    return PowerRegistry(search_roots=[(repo_root / ".augmentum" / "powers", "native")])


def _install_settings_store(app) -> SettingsStore:
    backend = app.state.state_manager.backend
    asyncio.get_event_loop().run_until_complete(
        backend.conn.execute(
            "CREATE TABLE IF NOT EXISTS app_settings ("
            "key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        ),
    )
    asyncio.get_event_loop().run_until_complete(
        backend.conn.execute(
            "CREATE TABLE IF NOT EXISTS user_settings ("
            "user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT, updated_at TEXT, "
            "PRIMARY KEY (user_id, key))"
        ),
    )
    asyncio.get_event_loop().run_until_complete(backend.conn.commit())
    store = SettingsStore(backend.conn)
    app.state.settings_store = store
    return store


class _DictSettingsStore:
    def __init__(self) -> None:
        self._global: dict[str, str] = {}
        self._user: dict[tuple[str, str], str] = {}

    async def get(self, key: str):
        return self._global.get(key)

    async def set(self, key: str, value: str | None):
        if value is None:
            self._global.pop(key, None)
        else:
            self._global[key] = value

    async def get_user(self, user_id: str, key: str):
        return self._user.get((user_id, key))

    async def set_user(self, user_id: str, key: str, value: str | None):
        if value is None:
            self._user.pop((user_id, key), None)
        else:
            self._user[(user_id, key)] = value


def test_registry_discovers_native_and_compat_manifests(tmp_path):
    registry = _make_registry(tmp_path)

    powers = registry.list_powers()
    assert [p.id for p in powers] == ["browser-verification", "mcp-builder"]
    browser = registry.get_power("browser-verification")
    mcp_builder = registry.get_power("mcp-builder")
    assert browser.source_kind == "native"
    assert browser.kind == "verifier"
    assert browser.activation_policy == "controller"
    assert browser.activation_windows == ["post_write", "pre_finish"]
    assert mcp_builder.source_kind == "compat"
    assert mcp_builder.kind == "integration"
    assert mcp_builder.activation_policy == "manual"


@pytest.mark.asyncio
async def test_power_state_store_tracks_workspace_activation():
    store = PowerStateStore(_DictSettingsStore())

    await store.activate_power(
        "alice",
        workspace_id="ws-alpha",
        power_id="browser-verification",
        reason="manual test",
    )
    active = await store.get_active_power("alice", workspace_id="ws-alpha")
    assert active is not None
    assert active.power_id == "browser-verification"
    assert active.reason == "manual test"

    await store.clear_active_power("alice", workspace_id="ws-alpha")
    assert await store.get_active_power("alice", workspace_id="ws-alpha") is None


@pytest.mark.asyncio
async def test_power_state_store_cache_is_per_settings_store():
    first = PowerStateStore(_DictSettingsStore())
    await first.set_enabled("", "browser-verification", False)
    assert await first.is_enabled("", "browser-verification") is False

    second = PowerStateStore(_DictSettingsStore())
    assert await second.is_enabled("", "browser-verification") is True


def test_power_routes_list_activate_and_disable(sqlite_client, tmp_path):
    app = sqlite_client.app
    _install_settings_store(app)
    app.state.power_registry = _make_registry(tmp_path)

    resp = sqlite_client.get("/api/powers")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["powers"]) == 2
    assert {p["source_kind"] for p in data["powers"]} == {"native", "compat"}
    browser = next(p for p in data["powers"] if p["id"] == "browser-verification")
    assert browser["kind"] == "verifier"
    assert browser["activation_policy"] == "controller"
    assert browser["activation_windows"] == ["post_write", "pre_finish"]

    act = sqlite_client.post(
        "/api/powers/browser-verification/activate",
        json={"workspace_id": "ws-alpha"},
    )
    assert act.status_code == 200
    assert act.json()["active"]["power_id"] == "browser-verification"

    listed = sqlite_client.get("/api/powers", params={"workspace_id": "ws-alpha"}).json()
    active_ids = [p["id"] for p in listed["powers"] if p["active"]]
    assert active_ids == ["browser-verification"]

    disable_other = sqlite_client.post(
        "/api/powers/mcp-builder/disable",
        json={"workspace_id": "ws-alpha"},
    )
    assert disable_other.status_code == 200

    listed_after = sqlite_client.get("/api/powers", params={"workspace_id": "ws-alpha"}).json()
    still_active = [p["id"] for p in listed_after["powers"] if p["active"]]
    assert still_active == ["browser-verification"]


def test_power_routes_list_auto_rescans(sqlite_client, tmp_path):
    app = sqlite_client.app
    _install_settings_store(app)
    registry = _make_registry(tmp_path)
    app.state.power_registry = registry

    first = sqlite_client.get("/api/powers")
    assert first.status_code == 200
    assert {p["id"] for p in first.json()["powers"]} == {
        "browser-verification",
        "mcp-builder",
    }

    native_root = tmp_path / ".augmentum" / "powers"
    _write_power(
        native_root,
        "release-review",
        "POWER.md",
        """---
name: Release Review
description: Final quality gate before shipping.
modes:
  - coder
---
# Release Review

Review the final changes before shipping.
""",
    )

    second = sqlite_client.get("/api/powers")
    assert second.status_code == 200
    assert {p["id"] for p in second.json()["powers"]} == {
        "browser-verification",
        "mcp-builder",
        "release-review",
    }


@pytest.mark.asyncio
async def test_coder_loads_active_power_into_system_prompt(tmp_path):
    from augmentum.modes.coder.handler import CoderHandler

    registry = _make_registry(tmp_path)
    settings_store = _DictSettingsStore()
    state = PowerStateStore(settings_store)
    await state.activate_power(
        "alice",
        workspace_id="ws-test",
        power_id="browser-verification",
        reason="manual",
    )

    handler = CoderHandler(
        _FakeBackend([_FakeChunk(done=True)]),
        session_id="sess-test",
        workspace_id="ws-test",
        container_manager=None,
        user_id="alice",
        power_registry=registry,
        settings_store=settings_store,
    )

    await handler._load_active_power_for_turn()
    event = handler._take_pending_power_activation_event()
    assert event is not None
    assert event["id"] == "browser-verification"
    assert event["source"] == "manual"
    assert event["transient"] is False
    assert event["checkpoint"] == "pre_plan"
    messages = handler._build_messages(_make_request("Validate this UI"), "extra system")
    # Power block lives in the per-turn runtime carrier (user-role
    # message before the latest user turn), not the leading system —
    # keeping the long system prefix byte-stable for prefix-cache reuse.
    combined = "\n".join(m.content for m in messages)
    assert "<active_power" in combined
    assert "Browser Verification" in combined


def test_controller_selector_defers_to_manual_power_during_pre_plan():
    registry = _repo_native_registry()
    selection = select_controller_power(
        registry.list_powers(),
        checkpoint="pre_plan",
        latest_user_text="Design and scaffold a minimal MCP server for a weather API",
        edited_paths=[],
        manual_power_id="mcp-builder",
    )
    assert selection is None


def test_controller_selector_allows_verifier_overlay_after_manual_power_write():
    registry = _repo_native_registry()
    selection = select_controller_power(
        registry.list_powers(),
        checkpoint="post_write",
        latest_user_text="Design and scaffold a minimal MCP server for a weather API",
        edited_paths=["/workspace/server.py"],
        manual_power_id="mcp-builder",
    )
    assert selection is not None
    assert selection.manifest.id == "test-author"


def test_controller_selector_prefers_test_author_after_non_test_write():
    registry = _repo_native_registry()
    selection = select_controller_power(
        registry.list_powers(),
        checkpoint="post_write",
        latest_user_text="Finish the feature and make sure it is covered",
        edited_paths=["/workspace/app.py"],
    )
    assert selection is not None
    assert selection.manifest.id == "test-author"


def test_controller_selector_detects_source_even_after_test_path():
    registry = _repo_native_registry()
    selection = select_controller_power(
        registry.list_powers(),
        checkpoint="post_write",
        latest_user_text="Finish the feature and make sure it is covered",
        edited_paths=["/workspace/tests/test_app.py", "/workspace/app.py"],
    )
    assert selection is not None
    assert selection.manifest.id == "test-author"


def test_controller_selector_prefers_failure_triage_on_verify_failed():
    registry = _repo_native_registry()
    selection = select_controller_power(
        registry.list_powers(),
        checkpoint="verify_failed",
        latest_user_text="Tests are failing after the refactor",
        edited_paths=["/workspace/ui/app.tsx"],
    )
    assert selection is not None
    assert selection.manifest.id == "failure-triage"


def test_controller_selector_prefers_dependency_doctor_on_install_failure():
    registry = _repo_native_registry()
    selection = select_controller_power(
        registry.list_powers(),
        checkpoint="verify_failed",
        latest_user_text="Module not found after install; check the package manager and lockfile.",
        edited_paths=[],
    )
    assert selection is not None
    assert selection.manifest.id == "dependency-doctor"


def test_controller_selector_prefers_performance_profiler_for_perf_failure():
    registry = _repo_native_registry()
    selection = select_controller_power(
        registry.list_powers(),
        checkpoint="verify_failed",
        latest_user_text="The benchmark is slow and p99 latency regressed.",
        edited_paths=[],
    )
    assert selection is not None
    assert selection.manifest.id == "performance-profiler"


def test_controller_selector_prefers_failure_triage_for_bug_reproducer_prompt():
    registry = _repo_native_registry()
    selection = select_controller_power(
        registry.list_powers(),
        checkpoint="pre_plan",
        latest_user_text=(
            "Create a minimal reproducer for a regression, explain the root cause, "
            "and fix the timestamp parsing bug."
        ),
        edited_paths=[],
    )
    assert selection is not None
    assert selection.manifest.id == "failure-triage"


def test_controller_selector_keeps_regression_test_prompt_with_test_author():
    registry = _repo_native_registry()
    selection = select_controller_power(
        registry.list_powers(),
        checkpoint="pre_plan",
        latest_user_text=(
            "Create a small utility and add focused regression tests for casing and "
            "whitespace behavior."
        ),
        edited_paths=[],
    )
    assert selection is None or selection.manifest.id != "failure-triage"


def test_controller_selector_prefers_multi_tenant_auditor_for_user_scoped_routes():
    registry = _repo_native_registry()
    selection = select_controller_power(
        registry.list_powers(),
        checkpoint="pre_plan",
        latest_user_text=(
            "Add a new endpoint route handler that writes user_id-scoped CRUD state."
        ),
        edited_paths=[],
    )
    assert selection is not None
    assert selection.manifest.id == "multi-tenant-auditor"


def test_controller_selector_prefers_workspace_onboarding_for_repo_orientation():
    registry = _repo_native_registry()
    selection = select_controller_power(
        registry.list_powers(),
        checkpoint="pre_plan",
        latest_user_text="This is a new workspace; onboard and explore repo conventions.",
        edited_paths=[],
    )
    assert selection is not None
    assert selection.manifest.id == "workspace-onboarding"


def test_controller_selector_ignores_tunnel_warning_artifact_text():
    registry = _repo_native_registry()
    selection = select_controller_power(
        registry.list_powers(),
        checkpoint="pre_plan",
        latest_user_text=(
            "Set abypass-tunnel-reminder request header with any value. "
            "Or, set a custom / non-standard browser user-agent request header. "
            "Are you the tunnel host?"
        ),
        edited_paths=[],
    )
    assert selection is None or selection.manifest.id != "contract-keeper"


def test_controller_selector_prefers_failure_triage_for_tunnel_bad_gateway_complaint():
    registry = _repo_native_registry()
    selection = select_controller_power(
        registry.list_powers(),
        checkpoint="pre_plan",
        latest_user_text=(
            "When I type in the IP it says bad gateway after it times out. "
            "How do I fix this?"
        ),
        edited_paths=[],
    )
    assert selection is not None
    assert selection.manifest.id == "failure-triage"


def test_controller_selector_prefers_observation_keeper_for_gotchas():
    registry = _repo_native_registry()
    selection = select_controller_power(
        registry.list_powers(),
        checkpoint="post_write",
        latest_user_text="This gotcha was not obvious; record this after the fix.",
        edited_paths=["/workspace/app.py"],
    )
    assert selection is not None
    assert selection.manifest.id == "observation-keeper"


def test_controller_selector_prefers_subagent_router_for_wide_review():
    registry = _repo_native_registry()
    selection = select_controller_power(
        registry.list_powers(),
        checkpoint="post_write",
        latest_user_text="I need a second opinion security audit of this diff.",
        edited_paths=["/workspace/app.py"],
    )
    assert selection is not None
    assert selection.manifest.id == "subagent-router"


def test_controller_selector_prefers_changelog_documenter_for_pr_summary():
    registry = _repo_native_registry()
    selection = select_controller_power(
        registry.list_powers(),
        checkpoint="pre_finish",
        latest_user_text="Write a PR description and changelog for what changed.",
        edited_paths=["/workspace/app.py"],
    )
    assert selection is not None
    assert selection.manifest.id == "changelog-documenter"


def test_controller_selector_prefers_test_baseline_keeper_for_comparison():
    registry = _repo_native_registry()
    selection = select_controller_power(
        registry.list_powers(),
        checkpoint="pre_finish",
        latest_user_text="Did this break? Compare before and after timing baseline.",
        edited_paths=["/workspace/app.py"],
    )
    assert selection is not None
    assert selection.manifest.id == "test-baseline-keeper"


def test_multi_agent_review_is_manual_workflow_power():
    registry = _repo_native_registry()
    manifest = registry.get_power("multi-agent-review")
    assert manifest is not None
    assert manifest.kind == "workflow"
    assert manifest.activation_policy == "manual"
    assert manifest.activation_windows == []


@pytest.mark.asyncio
async def test_handler_can_activate_pre_plan_controller_power():
    from augmentum.modes.coder.handler import CoderHandler

    handler = CoderHandler(
        _FakeBackend([_FakeChunk(done=True)]),
        session_id="sess-preplan",
        workspace_id="ws-preplan",
        container_manager=None,
        user_id="alice",
        power_registry=_repo_native_registry(),
        settings_store=_DictSettingsStore(),
    )

    activated = await handler._maybe_activate_controller_power(
        "pre_plan",
        latest_user_text="Write a migration to backfill slugs for existing users",
    )
    assert activated is True
    event = handler._take_pending_power_activation_event()
    assert event is not None
    assert event["id"] == "migration-safety"
    assert event["checkpoint"] == "pre_plan"
    assert handler._take_pending_power_activation_event() is None
    messages = handler._build_messages(
        _make_request("Write a migration to backfill slugs for existing users"),
        "extra system",
    )
    # Controller-power block lives in the per-turn runtime carrier
    # (user-role message before the latest user turn), not the leading
    # system — see ``CoderHandler._build_runtime_carrier_message``.
    combined = "\n".join(m.content for m in messages)
    assert "<controller_power" in combined
    assert "Migration Safety" in combined


@pytest.mark.asyncio
async def test_failure_triage_followup_nudge_is_appended_once():
    from augmentum.modes.coder.handler import CoderHandler

    handler = CoderHandler(
        _FakeBackend([_FakeChunk(done=True)]),
        session_id="sess-triage",
        workspace_id="ws-triage",
        container_manager=None,
        user_id="alice",
        power_registry=_repo_native_registry(),
        settings_store=_DictSettingsStore(),
    )
    messages: list = []
    event = {
        "id": "failure-triage",
        "checkpoint": "pre_plan",
        "source": "controller",
    }

    handler._append_power_followup_nudge(
        messages,
        event,
        goal_text="Debug the timestamp parsing regression",
    )
    handler._append_power_followup_nudge(
        messages,
        event,
        goal_text="Debug the timestamp parsing regression",
    )

    assert len(messages) == 1
    assert "Failure Triage is active" in messages[0].content


@pytest.mark.asyncio
async def test_pre_finish_review_nudges_with_release_review():
    from augmentum.modes.coder.handler import CoderHandler

    handler = CoderHandler(
        _FakeBackend([_FakeChunk(done=True)]),
        session_id="sess-prefinish",
        workspace_id="ws-prefinish",
        container_manager=None,
        user_id="alice",
        power_registry=_repo_native_registry(),
        settings_store=_DictSettingsStore(),
    )
    handler._controller_edited_paths = ["/workspace/app.py"]
    messages = []

    triggered = await handler._maybe_request_pre_finish_review(
        request=_make_request("Ship the feature"),
        messages=messages,
        total_writes=1,
        latest_input="Ship the feature",
        user_goal="Ship the feature",
    )
    assert triggered is True
    assert messages
    assert "Release Review" in messages[-1].content
    assert "release-ready" in messages[-1].content
    assert handler._controller_power_summary is not None
    assert handler._controller_power_summary["id"] == "release-review"
    event = handler._take_pending_power_activation_event()
    assert event is not None
    assert event["id"] == "release-review"
    assert event["checkpoint"] == "pre_finish"


def test_fallback_summary_includes_validation_evidence_for_test_author():
    from augmentum.modes.coder.handler import CoderHandler

    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-fallback-test",
        container_manager=None,
        workspace_id="ws-fallback-test",
    )
    handler._controller_power_summary = {
        "id": "test-author",
        "kind": "verifier",
        "display_name": "Test Author",
    }
    handler._state.tool_calls_made = 2

    out = handler._render_fallback_summary(
        iteration=2,
        total_writes=1,
        termination_reason="model_stop",
        same_file_edits={},
        messages=[],
        tool_results=[
            {
                "tool": "test_run",
                "success": True,
                "output_preview": "1 passed in 0.12s",
            },
        ],
    )

    assert "Validation evidence" in out
    assert "1 passed in 0.12s" in out


def test_fallback_summary_includes_diagnosis_for_failure_triage():
    from augmentum.modes.coder.handler import CoderHandler

    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-fallback-triage",
        container_manager=None,
        workspace_id="ws-fallback-triage",
    )
    handler._controller_power_summary = {
        "id": "failure-triage",
        "kind": "verifier",
        "display_name": "Failure Triage",
    }
    handler._state.tool_calls_made = 2

    out = handler._render_fallback_summary(
        iteration=2,
        total_writes=0,
        termination_reason="model_stop",
        same_file_edits={},
        messages=[],
        tool_results=[
            {
                "tool": "shell_exec",
                "success": False,
                "output_preview": "Traceback: ValueError: invalid isoformat string",
            },
        ],
    )

    assert "Diagnosis" in out
    assert "invalid isoformat string" in out
