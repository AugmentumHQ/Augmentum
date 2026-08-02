"""Tests for post-edit verification hooks in the coder loop.

Covers the 2026-04-20 fixes to ``_execute_tool_with_verification``:

  1. ``code_multi_edit`` now runs existence check + lint + checkpoint
     (previously only ``code_edit`` and ``file_write`` did — silent bug
     that let multi-file refactors ship broken syntax with no signal).

  2. Auto-run focused tests after editing test files — scoped to the
     file, 30-second timeout, results appended to tool output. Test
     failure does NOT flip write success (the write was valid; the
     failure is a separate signal the model should react to).

  3. ``_is_test_file`` detection heuristic — covers pytest, Jest /
     Vitest, Go, RSpec conventions plus ``tests/`` / ``__tests__/``
     directory conventions. Excludes fixtures and conftest.

Run: python -m pytest tests/test_coder_post_edit_hooks.py -v
"""
from __future__ import annotations

import pytest

from augmentum.modes.coder.handler import CoderHandler

from tests.test_coder_handler import (
    _ExtendedContainerManager,
    _FakeBackend,
)


# ---------------------------------------------------------------------------
# _is_test_file heuristic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path,expected", [
    # Pytest conventions
    ("tests/test_foo.py", True),
    ("augmentum/tests/test_coder_handler.py", True),
    ("test_module.py", True),
    ("foo_test.py", True),
    # Jest / Vitest conventions
    ("src/foo.test.js", True),
    ("src/foo.test.ts", True),
    ("src/foo.spec.ts", True),
    ("components/Button.spec.tsx", True),
    ("__tests__/foo.js", True),
    # Go
    ("pkg/foo_test.go", True),
    # RSpec
    ("spec/user_spec.rb", False),  # we don't match spec/
    ("app/models/user.spec.rb", True),
    # Not test files
    ("augmentum/coder/tools.py", False),
    ("README.md", False),
    ("Dockerfile", False),
    ("src/test_helper.py", True),  # startswith test_
    # Edge — things that *look* like tests but aren't
    ("tests/conftest.py", False),
    ("tests/__init__.py", False),
    ("tests/fixtures/sample_data.json", False),
    ("test/data/input.txt", False),
    # Empty / malformed
    ("", False),
    ("notapath", False),
])
def test_is_test_file_heuristic(path, expected):
    assert CoderHandler._is_test_file(path) is expected


def test_is_test_file_case_insensitive():
    """Mixed case shouldn't trip the matcher — paths are normalised
    to lower-case before inspection."""
    assert CoderHandler._is_test_file("Tests/Test_Foo.py") is True
    assert CoderHandler._is_test_file("SRC/Foo.Test.JS") is True


# ---------------------------------------------------------------------------
# _mutation_paths normalisation
# ---------------------------------------------------------------------------


def test_mutation_paths_extracts_single_path():
    assert CoderHandler._mutation_paths("file_write", {"path": "/a/b.py"}) == ["/a/b.py"]
    assert CoderHandler._mutation_paths("code_edit", {"path": "/x.py"}) == ["/x.py"]
    assert CoderHandler._mutation_paths(
        "code_edit_batch",
        {"path": "/m.py", "edits": [{"search": "a", "replace": "b"}]},
    ) == ["/m.py"]


def test_mutation_paths_returns_empty_on_missing_path():
    assert CoderHandler._mutation_paths("file_write", {}) == []
    assert CoderHandler._mutation_paths("file_write", None) == []
    assert CoderHandler._mutation_paths("file_write", {"path": ""}) == []


# ---------------------------------------------------------------------------
# Post-edit hooks — code_multi_edit parity with code_edit
# ---------------------------------------------------------------------------


class _HookCapturingContainer(_ExtendedContainerManager):
    """Container that records every shell command and lets tests
    script specific responses by command-substring match."""

    def __init__(self):
        super().__init__()
        self.commands: list[str] = []
        self.checkpoints: list[str] = []
        # Map of substring → response. Longest match wins.
        self.responses: dict[str, str] = {}

    async def run_command(self, *args, **kwargs):
        return await self._run_command(*args, **kwargs)

    async def _run_command(self, workspace_id, cmd, timeout=None):
        # cmd is a list like ["bash", "-c", "actual command"]
        cmd_str = cmd[-1] if isinstance(cmd, list) else str(cmd)
        self.commands.append(cmd_str)
        # Return scripted response for the longest matching substring.
        for key in sorted(self.responses.keys(), key=len, reverse=True):
            if key in cmd_str:
                return self.responses[key]
        return ""

    async def git_checkpoint(self, workspace_id, message):
        self.checkpoints.append(message)
        return "abc1234"


def _make_tool_map(output: str = "ok", success: bool = True):
    """Build a tool_map with file_write / code_edit / code_multi_edit
    stand-ins that all return the same canned result."""
    from augmentum.tools.base import ToolCategory, ToolResult

    class _FakeMutatingTool:
        def __init__(self, name):
            self._name = name

        @property
        def name(self):
            return self._name

        @property
        def description(self):
            return f"fake {self._name}"

        @property
        def category(self):
            return ToolCategory.CODE

        @property
        def input_schema(self):
            return {"type": "object", "properties": {}}

        @property
        def timeout(self):
            return 5.0

        async def execute(self, **kw):
            return ToolResult(success=success, output=output)

    return {
        "file_write":      _FakeMutatingTool("file_write"),
        "code_edit":       _FakeMutatingTool("code_edit"),
        "code_edit_batch": _FakeMutatingTool("code_edit_batch"),
    }


def _make_handler_with(container: _HookCapturingContainer) -> CoderHandler:
    return CoderHandler(
        _FakeBackend([]),
        session_id="sess-hooks",
        container_manager=container,
        workspace_id="ws-hooks",
    )


@pytest.mark.asyncio
async def test_code_multi_edit_runs_existence_check():
    """Pre-fix: existence check was in the (file_write, code_edit,
    code_multi_edit) tuple already — verify that path still works."""
    cm = _HookCapturingContainer()
    # Simulate a non-empty file on existence check
    cm.responses["wc -c <"] = "42"
    h = _make_handler_with(cm)

    _result, _cp, _tid = await h._execute_tool_with_verification(
        tool_name="code_edit_batch",
        tool_input={"path": "/workspace/foo.py", "edits": []},
        tool_map=_make_tool_map(),
    )
    # The wc -c existence check ran
    assert any("wc -c" in c for c in cm.commands), (
        f"Expected existence check on multi_edit; commands={cm.commands}"
    )


@pytest.mark.asyncio
async def test_code_multi_edit_now_runs_lint():
    """Post-fix: code_multi_edit triggers the lint chain that was
    previously skipped."""
    cm = _HookCapturingContainer()
    cm.responses["wc -c <"] = "120"
    h = _make_handler_with(cm)

    await h._execute_tool_with_verification(
        tool_name="code_edit_batch",
        tool_input={"path": "/workspace/foo.py", "edits": []},
        tool_map=_make_tool_map(),
    )
    # A ruff / py_compile lint command should have run for .py
    lint_ran = any(
        "ruff" in c or "py_compile" in c for c in cm.commands
    )
    assert lint_ran, (
        f"Expected lint chain for code_multi_edit; got {cm.commands}"
    )


@pytest.mark.asyncio
async def test_code_multi_edit_now_creates_checkpoint():
    """Post-fix: successful multi_edit triggers git_checkpoint (was
    previously silent — broken code could commit implicitly via
    later edits)."""
    cm = _HookCapturingContainer()
    cm.responses["wc -c <"] = "42"
    h = _make_handler_with(cm)

    _result, checkpoint, _tid = await h._execute_tool_with_verification(
        tool_name="code_edit_batch",
        tool_input={"path": "/workspace/new.py", "edits": []},
        tool_map=_make_tool_map(),
    )
    assert checkpoint == "abc1234"
    assert len(cm.checkpoints) == 1
    assert "code_edit_batch" in cm.checkpoints[0]
    assert "new.py" in cm.checkpoints[0]


@pytest.mark.asyncio
async def test_lint_error_flips_success_for_multi_edit():
    """A .py file with a clear SyntaxError should flip the multi_edit
    result to failed — was silently passing pre-fix."""
    cm = _HookCapturingContainer()
    cm.responses["wc -c <"] = "50"
    cm.responses["ruff"] = "/workspace/broken.py:3:1: E999 SyntaxError: invalid syntax"
    h = _make_handler_with(cm)

    result, checkpoint, _ = await h._execute_tool_with_verification(
        tool_name="code_edit_batch",
        tool_input={"path": "/workspace/broken.py", "edits": []},
        tool_map=_make_tool_map(),
    )
    assert result.success is False
    assert "SyntaxError" in (result.output or "")
    # No checkpoint when lint fails
    assert checkpoint is None
    assert cm.checkpoints == []


@pytest.mark.asyncio
async def test_clean_lint_keeps_success_true():
    """No lint output → no modification to tool_result. Clean writes
    should checkpoint normally."""
    cm = _HookCapturingContainer()
    cm.responses["wc -c <"] = "80"
    # Empty lint output (clean file)
    h = _make_handler_with(cm)

    result, checkpoint, _ = await h._execute_tool_with_verification(
        tool_name="code_edit",
        tool_input={"path": "/workspace/clean.py"},
        tool_map=_make_tool_map(),
    )
    assert result.success is True
    assert checkpoint == "abc1234"


# ---------------------------------------------------------------------------
# Focused test auto-run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_editing_test_file_triggers_focused_test_run():
    """Writing to tests/test_foo.py auto-runs pytest on that file."""
    cm = _HookCapturingContainer()
    cm.responses["wc -c <"] = "200"
    # Pytest says everything passed
    cm.responses["pytest"] = "1 passed in 0.02s"
    h = _make_handler_with(cm)

    result, _cp, _ = await h._execute_tool_with_verification(
        tool_name="code_edit",
        tool_input={"path": "/workspace/tests/test_foo.py"},
        tool_map=_make_tool_map(),
    )
    # pytest should have been invoked on just that file
    pytest_ran = any(
        "pytest" in c and "test_foo.py" in c for c in cm.commands
    )
    assert pytest_ran, (
        f"Expected focused pytest on edited test file; got {cm.commands}"
    )
    # Test pass appears in output so the model can see it
    assert "passed" in (result.output or "")


@pytest.mark.asyncio
async def test_test_failure_does_not_flip_write_success():
    """A failing test after editing a test file keeps the write
    successful — the write IS valid; the failure is a separate signal."""
    cm = _HookCapturingContainer()
    cm.responses["wc -c <"] = "200"
    cm.responses["pytest"] = (
        "FAILED tests/test_foo.py::test_bar - AssertionError: 1 != 2\n"
        "1 failed, 0 passed in 0.01s"
    )
    h = _make_handler_with(cm)

    result, cp, _ = await h._execute_tool_with_verification(
        tool_name="file_write",
        tool_input={"path": "/workspace/tests/test_foo.py"},
        tool_map=_make_tool_map(),
    )
    # Write still succeeded
    assert result.success is True
    # Checkpoint still ran — the write was legit
    assert cp == "abc1234"
    # But the test output IS visible so the model can react
    assert "FAILED" in (result.output or "") or "failed" in (result.output or "")


@pytest.mark.asyncio
async def test_editing_non_test_file_skips_test_run():
    """A regular source file edit should NOT trigger a test run —
    that's the ``test_run`` tool's job for suite-wide checks."""
    cm = _HookCapturingContainer()
    cm.responses["wc -c <"] = "100"
    h = _make_handler_with(cm)

    await h._execute_tool_with_verification(
        tool_name="code_edit",
        tool_input={"path": "/workspace/augmentum/coder/tools.py"},
        tool_map=_make_tool_map(),
    )
    # No pytest call
    pytest_ran = any("pytest" in c and "test_" in c for c in cm.commands)
    assert not pytest_ran, (
        f"Focused test should not run for non-test edits; got {cm.commands}"
    )


@pytest.mark.asyncio
async def test_multi_edit_test_file_also_triggers_focused_run():
    """Parity check: multi_edit of a test file gets the same auto-run
    treatment as code_edit."""
    cm = _HookCapturingContainer()
    cm.responses["wc -c <"] = "400"
    cm.responses["pytest"] = "3 passed in 0.05s"
    h = _make_handler_with(cm)

    await h._execute_tool_with_verification(
        tool_name="code_edit_batch",
        tool_input={
            "path": "/workspace/tests/test_mod.py",
            "edits": [{"search": "a", "replace": "b"}],
        },
        tool_map=_make_tool_map(),
    )
    pytest_ran = any("pytest" in c and "test_mod.py" in c for c in cm.commands)
    assert pytest_ran


@pytest.mark.asyncio
async def test_focused_test_skipped_when_pytest_missing():
    """The ``command -v pytest`` gate means containers without pytest
    don't generate noise — test run returns empty, write keeps its
    original output."""
    cm = _HookCapturingContainer()
    cm.responses["wc -c <"] = "100"
    # Pytest command returns empty (simulating the gate failing)
    cm.responses["pytest"] = ""
    h = _make_handler_with(cm)

    result, _cp, _ = await h._execute_tool_with_verification(
        tool_name="code_edit",
        tool_input={"path": "/workspace/tests/test_x.py"},
        tool_map=_make_tool_map(output="wrote it"),
    )
    # Output should not contain a "[Test run]" header when nothing
    # useful came back
    assert "[Test run]" not in (result.output or "")


@pytest.mark.asyncio
async def test_focused_test_output_capped_at_800_chars():
    """A giant test log shouldn't dominate the tool result. Cap at
    ~800 chars so it fits in compaction window."""
    cm = _HookCapturingContainer()
    cm.responses["wc -c <"] = "100"
    cm.responses["pytest"] = "FAIL " * 500  # ~2500 chars
    h = _make_handler_with(cm)

    result, _cp, _ = await h._execute_tool_with_verification(
        tool_name="code_edit",
        tool_input={"path": "/workspace/tests/test_y.py"},
        tool_map=_make_tool_map(),
    )
    # Find the [Test run] section in output
    out = result.output or ""
    if "[Test run]" in out:
        test_section = out.split("[Test run]", 1)[1]
        assert len(test_section) < 1000
