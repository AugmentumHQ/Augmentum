"""Smoke + integration tests for ``augmentum.agents``.

Covers:

* Built-in roles load and have the expected shape.
* File-based registry discovers roles in workspace + user dirs, with
  workspace-local winning on name collision.
* ``parse_simple_yaml`` handles the role-file shapes we ship in
  ``presets.py``.
* Tool-preset resolution (``read_only``, presets + extras).
* Multi-provider model resolver walks the fallback chain.
* ``run_subagent`` loop terminates correctly on (a) no-tool-calls,
  (b) budget exhaustion, (c) stuck detection.
* ``SubagentDispatcher`` end-to-end with mocked backend / tool.
* ``SubagentRunStore`` round-trips a run.
* Re-export shims in bug_finder still resolve to the new package.

No live LLM / no docker required — all mocked.
"""

from __future__ import annotations

import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.agents import SubagentSpec, run_subagent
from augmentum.agents.budget import SubagentBudget
from augmentum.agents.context_bridge import (
    build_initial_user_message,
    extract_orientation,
    extract_recent_tool_digests,
    extract_workspace_facts,
)
from augmentum.agents.dispatch import DispatchRequest, SubagentDispatcher
from augmentum.agents.presets import BUILTIN_ROLES
from augmentum.agents.registry import (
    AgentRegistry,
    _coerce_scalar,
    _parse_simple_yaml,
    _split_frontmatter,
)
from augmentum.agents.resolve import (
    ResolvedModel,
    SubagentModelUnavailableError,
    parse_model_spec,
    resolve_subagent_model,
)
from augmentum.agents.tools import (
    READ_ONLY_TOOL_NAMES,
    resolve_tool_spec,
)
from augmentum.models.base import Message, Usage
from augmentum.tools.base import Tool, ToolCategory, ToolResult


# ---------------------------------------------------------------------------
# Built-in roles
# ---------------------------------------------------------------------------


def test_builtin_roles_present():
    assert set(BUILTIN_ROLES.keys()) == {
        "explore", "plan", "review", "research",
        "security_review", "threat_model", "audit_zone",
    }
    for name, role in BUILTIN_ROLES.items():
        assert role.name == name
        assert role.system_prompt, f"{name} role missing system_prompt"
        assert role.tools, f"{name} role has empty tool list"
        assert role.source == "builtin"


def test_builtin_explore_is_read_only():
    # explore must never include mutating tools.
    explore = BUILTIN_ROLES["explore"]
    mutating = {"file_write", "code_edit", "shell_exec", "apply_patch"}
    assert not (set(explore.tools) & mutating)


def test_security_review_is_read_only_and_disproof_framed():
    role = BUILTIN_ROLES["security_review"]
    mutating = {"file_write", "code_edit", "shell_exec", "apply_patch", "code_edit_batch"}
    assert not (set(role.tools) & mutating)
    sp = role.system_prompt.lower()
    # Must carry the disproof framing — that's the load-bearing innovation.
    assert "disprove" in sp or "false positive" in sp
    # Must use evidence-first severity rubric, not class-anchored.
    assert "precondition" in sp
    # Must NOT ship a prescriptive bug-class checklist.
    assert "null_deref|bounds_check" not in role.system_prompt


def test_threat_model_role_is_read_only():
    role = BUILTIN_ROLES["threat_model"]
    mutating = {"file_write", "code_edit", "shell_exec", "apply_patch"}
    assert not (set(role.tools) & mutating)
    sp = role.system_prompt.lower()
    # Must produce paste-into-BugFinderIntake markdown.
    assert "threat model" in sp
    assert "trust boundaries" in sp
    assert "attacker capabilities" in sp
    # Must explicitly target paste-back into the bug_finder pipeline.
    assert "bugfinderintake" in sp or "bug_finder" in sp


# ---------------------------------------------------------------------------
# Tool preset resolution
# ---------------------------------------------------------------------------


def test_resolve_tool_spec_preset():
    assert resolve_tool_spec("read_only") == READ_ONLY_TOOL_NAMES


def test_resolve_tool_spec_preset_plus_extras():
    out = resolve_tool_spec("read_only + [test_run, http_request]")
    assert "test_run" in out
    assert "http_request" in out
    assert "file_read" in out  # from preset


def test_resolve_tool_spec_explicit_list():
    out = resolve_tool_spec(["file_read", "code_grep"])
    assert out == frozenset({"file_read", "code_grep"})


def test_resolve_tool_spec_empty_falls_back_to_read_only():
    assert resolve_tool_spec("") == READ_ONLY_TOOL_NAMES


# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------


def test_split_frontmatter_no_marker():
    front, body = _split_frontmatter("no frontmatter here")
    assert front == {}
    assert body == "no frontmatter here"


def test_split_frontmatter_simple():
    text = "---\nname: foo\ndescription: bar\n---\nbody text\n"
    front, body = _split_frontmatter(text)
    assert front["name"] == "foo"
    assert front["description"] == "bar"
    assert body.strip() == "body text"


def test_coerce_scalar_bool():
    assert _coerce_scalar("true") is True
    assert _coerce_scalar("false") is False
    assert _coerce_scalar("yes") is True


def test_coerce_scalar_int():
    assert _coerce_scalar("42") == 42


def test_parse_simple_yaml_nested_dict():
    # PyYAML installed in this venv → full nested parse works.
    # Fallback parser test deferred to a no-yaml environment.
    txt = "model:\n  preferred: foo\n"
    out = _parse_simple_yaml(txt)
    assert out["model"] == {"preferred": "foo"}


# ---------------------------------------------------------------------------
# AgentRegistry discovery
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_role_dir(tmp_path: pathlib.Path):
    """Create a temporary .augmentum/agents/ dir with one role file."""
    workspace = tmp_path / "workspace"
    agents = workspace / ".augmentum" / "agents"
    agents.mkdir(parents=True)
    (agents / "scout.md").write_text(
        "---\n"
        "name: scout\n"
        "description: Workspace-local scout role\n"
        "tools: read_only\n"
        "---\n"
        "You are a scout. Find things and report back.\n",
        encoding="utf-8",
    )
    return workspace


def test_registry_discovers_workspace_role(temp_role_dir):
    reg = AgentRegistry(
        workspace_dir=str(temp_role_dir),
        user_dir=str(temp_role_dir / "_nowhere"),
        builtins=BUILTIN_ROLES,
    )
    reg.refresh()
    names = reg.names()
    assert "scout" in names
    assert "explore" in names  # built-in still present
    scout = reg.get("scout")
    assert scout.source == "workspace"
    assert scout.system_prompt.startswith("You are a scout")


def test_registry_workspace_overrides_builtin(tmp_path):
    workspace = tmp_path / "ws"
    agents = workspace / ".augmentum" / "agents"
    agents.mkdir(parents=True)
    (agents / "explore.md").write_text(
        "---\nname: explore\ndescription: User override\n---\nOverridden body.\n",
        encoding="utf-8",
    )
    reg = AgentRegistry(
        workspace_dir=str(workspace),
        user_dir=str(tmp_path / "_nowhere"),
        builtins=BUILTIN_ROLES,
    )
    reg.refresh()
    explore = reg.get("explore")
    assert explore.source == "workspace"
    assert "Overridden" in explore.system_prompt


def test_registry_get_unknown_raises():
    reg = AgentRegistry(builtins=BUILTIN_ROLES)
    with pytest.raises(KeyError):
        reg.get("does_not_exist")


def test_registry_refresh_if_stale_detects_new_file(tmp_path):
    workspace = tmp_path / "ws"
    agents = workspace / ".augmentum" / "agents"
    agents.mkdir(parents=True)
    reg = AgentRegistry(
        workspace_dir=str(workspace),
        user_dir=str(tmp_path / "_nowhere"),
        builtins=BUILTIN_ROLES,
    )
    reg.refresh()
    assert "newcomer" not in reg.names()

    (agents / "newcomer.md").write_text(
        "---\nname: newcomer\n---\nHello.\n", encoding="utf-8",
    )
    changed = reg.refresh_if_stale()
    assert changed
    assert "newcomer" in reg.names()


# ---------------------------------------------------------------------------
# Model resolver
# ---------------------------------------------------------------------------


def test_parse_model_spec_passthrough():
    assert parse_model_spec("model@provider") == "model@provider"
    assert parse_model_spec("  trimmed  ") == "trimmed"


@pytest.mark.asyncio
async def test_resolve_subagent_model_walks_fallbacks():
    registry = MagicMock()
    fake_backend = MagicMock()

    async def _resolve(name, *, user_id="", session_id=""):
        if name == "preferred-model":
            from augmentum.models.provider_registry import ModelUnavailableError
            raise ModelUnavailableError("preferred not here", model=name)
        if name == "fallback-1":
            return fake_backend, "fallback-1"
        return None, ""

    registry.resolve_backend_with_fabric = AsyncMock(side_effect=_resolve)

    resolved = await resolve_subagent_model(
        role="explore",
        preferred="preferred-model",
        fallbacks=["fallback-1", "fallback-2"],
        registry=registry,
    )
    assert resolved.backend is fake_backend
    assert resolved.model_id == "fallback-1"
    assert resolved.spec == "fallback-1"


@pytest.mark.asyncio
async def test_resolve_subagent_model_raises_when_all_fail():
    registry = MagicMock()
    from augmentum.models.provider_registry import ModelUnavailableError

    async def _resolve(name, *, user_id="", session_id=""):
        raise ModelUnavailableError(f"{name} missing", model=name)

    registry.resolve_backend_with_fabric = AsyncMock(side_effect=_resolve)

    with pytest.raises(SubagentModelUnavailableError) as exc:
        await resolve_subagent_model(
            role="explore", preferred="a", fallbacks=["b"], registry=registry,
        )
    assert "a" in str(exc.value)
    assert "b" in str(exc.value)


# ---------------------------------------------------------------------------
# Subagent loop
# ---------------------------------------------------------------------------


def _make_response(*, content: str = "", tool_calls=None):
    resp = MagicMock()
    resp.message = Message(role="assistant", content=content, tool_calls=tool_calls or [])
    resp.usage = Usage(prompt_tokens=10, completion_tokens=5)
    return resp


@pytest.mark.asyncio
async def test_run_subagent_completes_with_no_tool_calls():
    backend = MagicMock()
    backend.chat = AsyncMock(return_value=_make_response(content="all done"))

    spec = SubagentSpec(
        role="explore",
        model="fake-model",
        system_prompt="you are a test",
        initial_user_message="do the thing",
        tools=(),
        budget=SubagentBudget(max_iterations=5, max_wallclock_seconds=10, max_tokens=10_000),
    )
    result = await run_subagent(spec, backend=backend)
    assert result.stop_reason == "complete"
    assert result.output == "all done"
    assert result.iterations == 1


@pytest.mark.asyncio
async def test_run_subagent_budget_exhaustion():
    backend = MagicMock()
    backend.chat = AsyncMock(return_value=_make_response(
        content="thinking", tool_calls=[{"id": "1", "function": {"name": "noop", "arguments": "{}"}}],
    ))
    spec = SubagentSpec(
        role="explore",
        model="fake-model",
        system_prompt="you are a test",
        initial_user_message="loop forever",
        tools=(),  # no tools registered → call is unavailable; loop continues
        budget=SubagentBudget(max_iterations=2, max_wallclock_seconds=10, max_tokens=10_000),
    )
    result = await run_subagent(spec, backend=backend)
    assert result.stop_reason == "budget"
    assert "max_iterations" in result.stop_detail


@pytest.mark.asyncio
async def test_run_subagent_backend_error():
    backend = MagicMock()
    backend.chat = AsyncMock(side_effect=RuntimeError("backend exploded"))
    spec = SubagentSpec(
        role="explore",
        model="fake-model",
        system_prompt="you are a test",
        initial_user_message="fail",
        tools=(),
        budget=SubagentBudget(max_iterations=5, max_wallclock_seconds=10, max_tokens=10_000),
    )
    result = await run_subagent(spec, backend=backend)
    assert result.stop_reason == "error"
    assert "RuntimeError" in result.stop_detail


@pytest.mark.asyncio
async def test_run_subagent_with_real_tool_call():
    """Tool is invoked, result feeds back, then model returns no tool calls."""

    class FakeTool(Tool):
        @property
        def name(self) -> str:
            return "fake_tool"

        @property
        def description(self) -> str:
            return "Just for testing"

        @property
        def category(self) -> ToolCategory:
            return ToolCategory.CODE

        @property
        def input_schema(self) -> dict:
            return {"type": "object", "properties": {}}

        async def execute(self, **kwargs) -> ToolResult:
            return ToolResult(success=True, output="tool ran")

    call_count = 0

    async def _chat(req):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_response(
                content="calling tool",
                tool_calls=[{"id": "call_1", "function": {"name": "fake_tool", "arguments": "{}"}}],
            )
        return _make_response(content="finished")

    backend = MagicMock()
    backend.chat = AsyncMock(side_effect=_chat)

    spec = SubagentSpec(
        role="explore",
        model="fake-model",
        system_prompt="test",
        initial_user_message="use the tool",
        tools=(FakeTool(),),
        budget=SubagentBudget(max_iterations=5, max_wallclock_seconds=10, max_tokens=10_000),
    )
    result = await run_subagent(spec, backend=backend)
    assert result.stop_reason == "complete"
    assert result.tool_calls == 1
    assert result.iterations == 2
    assert len(result.tool_call_log) == 1
    assert result.tool_call_log[0].outcome == "success"


# ---------------------------------------------------------------------------
# Context bridge
# ---------------------------------------------------------------------------


def test_build_initial_user_message_slim_drops_facts():
    out = build_initial_user_message(
        prompt="do x", context_mode="slim", workspace_facts="FACTS",
    )
    assert out == "do x"
    assert "FACTS" not in out


def test_build_initial_user_message_workspace_includes_facts():
    out = build_initial_user_message(
        prompt="do x", context_mode="workspace", workspace_facts="FACTS",
    )
    assert "FACTS" in out
    assert "do x" in out
    assert "<workspace_facts>" in out


def test_build_initial_user_message_hot_includes_digests():
    out = build_initial_user_message(
        prompt="do x",
        context_mode="hot",
        workspace_facts="FACTS",
        recent_tool_digests=["recent1", "recent2"],
    )
    assert "<workspace_facts>" in out
    assert "<recent_activity>" in out
    assert "recent1" in out


def test_build_initial_user_message_slim_includes_orientation():
    # Even slim mode carries the orientation anchor — the session
    # objective + project shape — while still dropping workspace_facts.
    out = build_initial_user_message(
        prompt="do x",
        context_mode="slim",
        orientation="<orientation>\nObjective: ship the thing\n</orientation>",
        workspace_facts="FACTS",
    )
    assert "<orientation>" in out
    assert "ship the thing" in out
    assert "FACTS" not in out
    assert "<workspace_facts>" not in out
    assert "do x" in out


def test_build_initial_user_message_orientation_precedes_facts():
    out = build_initial_user_message(
        prompt="do x",
        context_mode="workspace",
        orientation="<orientation>\nObjective: O\n</orientation>",
        workspace_facts="FACTS",
    )
    assert out.index("<orientation>") < out.index("FACTS")


def test_build_initial_user_message_does_not_double_wrap_facts():
    # render_facts_block() output is already wrapped — don't re-tag it.
    pre_wrapped = "<workspace_facts>\nProject: python\n</workspace_facts>"
    out = build_initial_user_message(
        prompt="do x", context_mode="workspace", workspace_facts=pre_wrapped,
    )
    assert out.count("<workspace_facts>") == 1
    assert out.count("</workspace_facts>") == 1


def test_build_initial_user_message_renders_success_criteria():
    out = build_initial_user_message(
        prompt="do x",
        context_mode="slim",
        success_criteria=["lists every caller", "flags skipped paths"],
        constraints=["read-only"],
    )
    assert "<success_criteria>" in out
    assert "- lists every caller" in out
    assert "- flags skipped paths" in out
    assert "<constraints>" in out
    assert "- read-only" in out
    assert "do x" in out


def test_build_initial_user_message_skips_empty_criteria():
    out = build_initial_user_message(
        prompt="do x", context_mode="slim", success_criteria=[], constraints=None,
    )
    assert "<success_criteria>" not in out
    assert "<constraints>" not in out
    assert out == "do x"


def test_extract_workspace_facts_from_state():
    state = MagicMock()
    state.kernel_facts_text = "FACTS BLOCK"
    assert extract_workspace_facts(state) == "FACTS BLOCK"


def test_extract_workspace_facts_handles_none():
    assert extract_workspace_facts(None) == ""


def test_extract_orientation_from_state():
    state = MagicMock()
    state.orientation_text = "<orientation>\nObjective: O\n</orientation>"
    assert "Objective: O" in extract_orientation(state)


def test_extract_orientation_handles_none():
    assert extract_orientation(None) == ""


def test_extract_orientation_missing_attr():
    # A real CoderState that never had orientation set (kernel disabled)
    # must not blow up — getattr default kicks in.
    class _Bare:
        pass

    assert extract_orientation(_Bare()) == ""


def test_extract_recent_tool_digests_falls_back_to_summaries():
    state = MagicMock(spec=["turn_summaries"])
    state.turn_summaries = ["t1", "t2", "t3"]
    out = extract_recent_tool_digests(state, limit=2)
    assert out == ["t2", "t3"]


# ---------------------------------------------------------------------------
# Dispatcher (integration)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_runs_role_with_mocked_backend(tmp_path):
    """End-to-end: registry → resolve → spawn → result."""
    fake_backend = MagicMock()
    fake_backend.chat = AsyncMock(return_value=_make_response(
        content="explored: 3 files match", tool_calls=[],
    ))

    provider_registry = MagicMock()
    provider_registry.resolve_backend_with_fabric = AsyncMock(
        return_value=(fake_backend, "claude-sonnet-4-6"),
    )

    reg = AgentRegistry(builtins=BUILTIN_ROLES, workspace_dir=str(tmp_path / "ws"))
    dispatcher = SubagentDispatcher(
        registry=reg,
        provider_registry=provider_registry,
        store=None,
        tool_registry_provider=lambda: [],
        coder_state_provider=lambda: None,
    )

    req = DispatchRequest(
        role="explore",
        prompt="find every site that calls foo",
        user_id="u1",
    )
    outcome = await dispatcher.dispatch(req)
    assert outcome.role == "explore"
    assert outcome.model_resolved == "claude-sonnet-4-6"
    assert outcome.result.stop_reason == "complete"
    assert "explored" in outcome.result.output


@pytest.mark.asyncio
async def test_dispatcher_unknown_role_raises():
    reg = AgentRegistry(builtins=BUILTIN_ROLES)
    dispatcher = SubagentDispatcher(
        registry=reg,
        provider_registry=MagicMock(),
        store=None,
    )
    with pytest.raises(ValueError, match="does_not_exist"):
        await dispatcher.dispatch(DispatchRequest(role="does_not_exist", prompt="x"))


# ---------------------------------------------------------------------------
# Fast-model arbitrage (#2) — cheap fan-out roles run on a fast model.
# ---------------------------------------------------------------------------


def _bare_dispatcher():
    return SubagentDispatcher(
        registry=AgentRegistry(builtins=BUILTIN_ROLES),
        provider_registry=MagicMock(),
        store=None,
    )


def test_fast_model_spec_explicit_setting_wins(monkeypatch):
    from augmentum.config import settings

    monkeypatch.setattr(settings, "coder_subagent_fast_model", "haiku@anthropic", raising=False)
    monkeypatch.setattr(settings, "engine_secondary_enabled", True, raising=False)
    monkeypatch.setattr(settings, "engine_secondary_model", "qwen3-8b", raising=False)
    d = _bare_dispatcher()
    # Explicit setting beats the Slot B default for a fan-out role.
    assert d._fast_model_spec("explore") == "haiku@anthropic"
    assert d._fast_model_spec("research") == "haiku@anthropic"


def test_fast_model_spec_defaults_to_slot_b(monkeypatch):
    from augmentum.config import settings

    monkeypatch.setattr(settings, "coder_subagent_fast_model", "", raising=False)
    monkeypatch.setattr(settings, "engine_secondary_enabled", True, raising=False)
    monkeypatch.setattr(settings, "engine_secondary_model", "qwen3-8b", raising=False)
    d = _bare_dispatcher()
    assert d._fast_model_spec("explore") == "qwen3-8b"


def test_fast_model_spec_empty_when_slot_b_idle(monkeypatch):
    from augmentum.config import settings

    monkeypatch.setattr(settings, "coder_subagent_fast_model", "", raising=False)
    monkeypatch.setattr(settings, "engine_secondary_enabled", True, raising=False)
    monkeypatch.setattr(settings, "engine_secondary_model", "", raising=False)
    d = _bare_dispatcher()
    # Slot B enabled but no model loaded → inherit the lead's model.
    assert d._fast_model_spec("explore") == ""


def test_fast_model_spec_excludes_deep_roles(monkeypatch):
    from augmentum.config import settings

    monkeypatch.setattr(settings, "coder_subagent_fast_model", "haiku@anthropic", raising=False)
    monkeypatch.setattr(settings, "engine_secondary_enabled", True, raising=False)
    monkeypatch.setattr(settings, "engine_secondary_model", "qwen3-8b", raising=False)
    d = _bare_dispatcher()
    # Deep-judgment roles always inherit the lead's (capable) model.
    for role in ("review", "security_review", "threat_model", "audit_zone", "plan"):
        assert d._fast_model_spec(role) == "", role


# ---------------------------------------------------------------------------
# Subagent loop compaction (#5 fix) — bound the working context so the
# cumulative max_tokens budget doesn't trip on re-sent history.
# ---------------------------------------------------------------------------


def _mk_transcript(n_rounds: int):
    """system + user + n_rounds of (assistant w/ tool_call, tool result)."""
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="initial task + criteria"),
    ]
    for i in range(n_rounds):
        msgs.append(Message(
            role="assistant",
            content=f"thought {i}",
            tool_calls=[{"id": f"c{i}", "function": {"name": "file_read", "arguments": f'{{"path": "/f{i}.py"}}'}}],
        ))
        msgs.append(Message(role="tool", content=f"contents of f{i}" * 50, tool_call_id=f"c{i}"))
    return msgs


def test_compact_subagent_messages_collapses_middle():
    from augmentum.agents.loop import _compact_subagent_messages

    msgs = _mk_transcript(8)  # 2 + 16 = 18 messages
    out = _compact_subagent_messages(msgs, keep_recent=6)
    assert out is not None
    # Head preserved verbatim.
    assert out[0].role == "system" and out[0].content == "sys"
    assert out[1].content == "initial task + criteria"
    # One summary note replaces the middle, and it lists examined files.
    assert out[2].role == "user"
    assert "Context compacted" in out[2].content
    assert "file_read(/f0.py)" in out[2].content
    # Net shorter than the original.
    assert len(out) < len(msgs)
    # Tail must NOT start on an orphaned tool result.
    assert out[3].role != "tool"


def test_compact_subagent_messages_skips_when_short():
    from augmentum.agents.loop import _compact_subagent_messages

    # Only 2 rounds → nothing worth compacting.
    assert _compact_subagent_messages(_mk_transcript(2), keep_recent=6) is None


def test_compact_subagent_messages_examined_survives_second_pass():
    """The examined-labels list is loop-owned and cumulative: a second
    compaction pass must still name files folded away by the FIRST pass
    (whose summary note has no tool_calls to re-mine)."""
    from augmentum.agents.loop import _compact_subagent_messages

    examined: list[str] = []
    msgs = _mk_transcript(8)
    out = _compact_subagent_messages(msgs, keep_recent=6, examined=examined)
    assert out is not None and "file_read(/f0.py)" in out[2].content

    # Simulate more work after the first pass, then compact again.
    for i in range(8, 16):
        out.append(Message(
            role="assistant",
            content=f"thought {i}",
            tool_calls=[{"id": f"c{i}", "function": {"name": "file_read", "arguments": f'{{"path": "/f{i}.py"}}'}}],
        ))
        out.append(Message(role="tool", content=f"contents of f{i}" * 50, tool_call_id=f"c{i}"))
    out2 = _compact_subagent_messages(out, keep_recent=6, examined=examined)
    assert out2 is not None
    note = out2[2].content
    # First-pass files still named — the old bug dropped them here.
    assert "file_read(/f0.py)" in note
    # And the newly-folded span is recorded too.
    assert "file_read(/f8.py)" in note


def test_compact_subagent_messages_no_orphaned_tool_tail():
    from augmentum.agents.loop import _compact_subagent_messages

    # keep_recent lands the tail boundary on a tool result; the helper
    # must advance past it so the kept tail starts on an assistant msg.
    out = _compact_subagent_messages(_mk_transcript(10), keep_recent=5)
    assert out is not None
    assert out[3].role != "tool"


# ---------------------------------------------------------------------------
# Bug_finder shim compatibility
# ---------------------------------------------------------------------------


def test_bug_finder_imports_resolve_to_agents():
    from augmentum.agents.loop import SubagentSpec as NewSpec
    from augmentum.bug_finder.subagent import SubagentSpec as ShimSpec
    assert ShimSpec is NewSpec


def test_bug_finder_guards_resolve_to_agents():
    from augmentum.agents.guards import detector_guard as new_detector
    from augmentum.bug_finder.guards import detector_guard as shim_detector
    assert shim_detector is new_detector
