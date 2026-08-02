"""Tests for augmentum/coder/tools.py — 8 workspace-aware coder tools."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from augmentum.coder.models import FileEntry
from augmentum.coder.state import CoderState, CoderPhase
from augmentum.coder.tools import (
    FileReadTool,
    FileWriteTool,
    FileListTool,
    DirTreeTool,
    CodeEditTool,
    CodeMultiEditTool,
    ApplyPatchTool,
    CodeGrepTool,
    CodeGlobTool,
    ShellExecTool,
    ShellReadTool,
    ContainerInfoTool,
    PublishPortsTool,
    create_coder_tools,
    READ_ONLY_TOOLS,
    ALL_CODER_TOOLS,
)
from augmentum.tools.base import ToolCategory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_state() -> CoderState:
    return CoderState(session_id="sess-1", workspace_id="ws-1")


def make_container_manager(*, run_output: str = "", file_list: list | None = None) -> MagicMock:
    cm = MagicMock()
    cm.file_read = AsyncMock(return_value=run_output)
    cm.file_write = AsyncMock(return_value=None)
    cm.file_upload = AsyncMock(return_value=None)
    cm.file_list = AsyncMock(return_value=file_list or [])
    cm.list_ports = AsyncMock(return_value=[])
    # Both shapes — the public ``run_command`` is what tools now use
    # (since the 2026-04-22 cleanup) and ``_run_command`` stays for
    # backward-compatible tests that assert the old call site.
    cm._run_command = AsyncMock(return_value=run_output)
    cm.run_command = cm._run_command
    return cm


# ---------------------------------------------------------------------------
# FileReadTool
# ---------------------------------------------------------------------------

class TestFileReadTool:
    @pytest.mark.asyncio
    async def test_basic_read(self):
        cm = make_container_manager(run_output="line1\nline2\nline3")
        state = make_state()
        tool = FileReadTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(path="/workspace/main.py")

        assert result.success
        assert "   1 | line1" in result.output
        assert "   2 | line2" in result.output
        assert "   3 | line3" in result.output

    @pytest.mark.asyncio
    async def test_tracks_files_read(self):
        cm = make_container_manager(run_output="content")
        state = make_state()
        tool = FileReadTool(container_manager=cm, workspace_id="ws-1", state=state)

        assert "/workspace/main.py" not in state.files_read
        await tool.execute(path="/workspace/main.py")
        assert "/workspace/main.py" in state.files_read

    @pytest.mark.asyncio
    async def test_error_on_container_failure(self):
        cm = make_container_manager()
        cm.file_read = AsyncMock(side_effect=RuntimeError("container not running"))
        state = make_state()
        tool = FileReadTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(path="/workspace/main.py")

        assert not result.success
        assert "container not running" in result.error

    @pytest.mark.asyncio
    async def test_missing_path_validation_error(self):
        cm = make_container_manager()
        state = make_state()
        tool = FileReadTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(path="")

        assert not result.success
        assert result.validation_error

    def test_cacheable_is_false(self):
        cm = make_container_manager()
        state = make_state()
        tool = FileReadTool(container_manager=cm, workspace_id="ws-1", state=state)
        assert tool.cacheable is False

    def test_category_is_code(self):
        cm = make_container_manager()
        state = make_state()
        tool = FileReadTool(container_manager=cm, workspace_id="ws-1", state=state)
        assert tool.category == ToolCategory.CODE

    # --- batch mode (paths=[...]) ---

    @staticmethod
    def _batch_cm(contents: dict[str, str]) -> MagicMock:
        cm = make_container_manager()

        async def _read(ws, path):
            if path in contents:
                return contents[path]
            raise RuntimeError(f"No such file: {path}")

        cm.file_read = AsyncMock(side_effect=_read)
        return cm

    @pytest.mark.asyncio
    async def test_batch_reads_multiple_files(self):
        cm = self._batch_cm({
            "/workspace/a.py": "alpha_one\nalpha_two",
            "/workspace/b.py": "beta_one",
        })
        state = make_state()
        tool = FileReadTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(paths=["/workspace/a.py", "/workspace/b.py"])

        assert result.success
        assert "=== /workspace/a.py (2 lines) ===" in result.output
        assert "   1 | alpha_one" in result.output
        assert "=== /workspace/b.py (1 lines) ===" in result.output
        assert "   1 | beta_one" in result.output
        assert result.metadata["batch"] is True
        assert result.metadata["read_ok"] == 2
        # Read-before-edit guard sees BOTH files.
        assert "/workspace/a.py" in state.files_read
        assert "/workspace/b.py" in state.files_read

    @pytest.mark.asyncio
    async def test_batch_partial_failure_stays_inline(self):
        cm = self._batch_cm({"/workspace/a.py": "content"})
        state = make_state()
        tool = FileReadTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(paths=["/workspace/a.py", "/workspace/gone.py"])

        # One good file → overall success; the miss is reported inline.
        assert result.success
        assert "No such file: /workspace/gone.py" in result.output
        assert result.metadata["read_ok"] == 1
        assert "/workspace/gone.py" not in state.files_read

    @pytest.mark.asyncio
    async def test_batch_omits_whole_files_over_budget(self):
        # First file eats the budget; second is omitted WHOLE and named,
        # never truncated mid-file.
        big = "\n".join("x" * 200 for _ in range(400))  # ~83k chars
        cm = self._batch_cm({
            "/workspace/big.py": big,
            "/workspace/small.py": "tiny",
        })
        state = make_state()
        tool = FileReadTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(paths=["/workspace/big.py", "/workspace/small.py"])

        assert result.success
        assert result.metadata["omitted"] == ["/workspace/small.py"]
        assert "tiny" not in result.output
        assert "omitted" in result.output
        # The oversized first file gets an explicit paging note.
        assert "Call file_read with path='/workspace/big.py'" in result.output

    @pytest.mark.asyncio
    async def test_batch_single_path_degrades_to_single_read(self):
        cm = make_container_manager(run_output="only line")
        state = make_state()
        tool = FileReadTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(paths=["/workspace/a.py"])

        assert result.success
        # Single-read shape (paging metadata), not batch shape.
        assert "batch" not in (result.metadata or {})
        assert result.metadata["total_lines"] == 1

    @pytest.mark.asyncio
    async def test_batch_all_misses_is_failure(self):
        cm = self._batch_cm({})
        state = make_state()
        tool = FileReadTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(paths=["/workspace/x.py", "/workspace/y.py"])

        assert not result.success
        assert "no files in the batch could be read" in result.error


# ---------------------------------------------------------------------------
# FileWriteTool
# ---------------------------------------------------------------------------

class TestFileWriteTool:
    @pytest.mark.asyncio
    async def test_basic_write(self):
        cm = make_container_manager()
        state = make_state()
        tool = FileWriteTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(path="/workspace/new.py", content="print('hi')")

        assert result.success
        cm.file_write.assert_called_once_with("ws-1", "/workspace/new.py", "print('hi')")

    @pytest.mark.asyncio
    async def test_missing_path(self):
        cm = make_container_manager()
        state = make_state()
        tool = FileWriteTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(path="", content="data")

        assert not result.success
        assert result.validation_error

    def test_missing_path_hint_explains_truncation_cause(self):
        """The error_hints copy should explain the truncation root cause
        and steer to code_edit_batch for large writes — not just re-state
        the schema. The old hint just said "send both fields", so a model
        that lost `path` to a truncated stream would retry with the same
        oversized content and lose `path` again. The 2026-05-29 transcript
        showed this looping 6× before the user intervened."""
        cm = make_container_manager()
        state = make_state()
        tool = FileWriteTool(container_manager=cm, workspace_id="ws-1", state=state)

        hint = tool.error_hints["without a 'path' argument"]
        assert "code_edit_batch" in hint
        # Names the truncation mechanism so the model picks a different
        # strategy on retry instead of re-emitting the same broken call.
        assert "output budget" in hint.lower() or "truncat" in hint.lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("outside", [
        "/etc/cron.d/job",
        "/workspace/../etc/passwd",   # normalized → /etc/passwd
        "/workspacefoo/x.py",         # prefix lookalike, not /workspace/
        "~/.bashrc",                  # home paths are read-only territory
        "/tmp/scratch.py",
    ])
    async def test_write_outside_workspace_rejected(self, outside):
        """Mutating tools stay inside /workspace: turn snapshots and the
        review/rewind substrate only track files under /workspace, so a
        write anywhere else would be unreviewable and unrevertable."""
        cm = make_container_manager()
        state = make_state()
        tool = FileWriteTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(path=outside, content="x")

        assert not result.success
        assert result.validation_error
        assert "/workspace" in result.error
        cm.file_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_inside_workspace_dot_segments_ok(self):
        # Dot segments that stay under /workspace are normalized, not rejected.
        cm = make_container_manager()
        state = make_state()
        tool = FileWriteTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(
            path="/workspace/src/../main.py", content="x",
        )

        assert result.success
        cm.file_write.assert_called_once_with("ws-1", "/workspace/main.py", "x")

    @pytest.mark.asyncio
    async def test_write_failure(self):
        cm = make_container_manager()
        cm.file_write = AsyncMock(side_effect=RuntimeError("disk full"))
        state = make_state()
        tool = FileWriteTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(path="/workspace/file.py", content="x")

        assert not result.success
        assert "disk full" in result.error

    def test_category_is_code(self):
        cm = make_container_manager()
        state = make_state()
        tool = FileWriteTool(container_manager=cm, workspace_id="ws-1", state=state)
        assert tool.category == ToolCategory.CODE


# ---------------------------------------------------------------------------
# FileListTool
# ---------------------------------------------------------------------------

class TestFileListTool:
    @pytest.mark.asyncio
    async def test_lists_directory(self):
        entries = [
            FileEntry(name="main.py", path="/workspace/main.py", is_dir=False, size=1024),
            FileEntry(name="src", path="/workspace/src", is_dir=True, size=4096),
        ]
        cm = make_container_manager(file_list=entries)
        state = make_state()
        tool = FileListTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(path="/workspace")

        assert result.success
        assert "main.py" in result.output
        assert "src" in result.output
        assert "1024" in result.output

    @pytest.mark.asyncio
    async def test_defaults_to_workspace_root(self):
        cm = make_container_manager(file_list=[])
        state = make_state()
        tool = FileListTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute()

        assert result.success
        cm.file_list.assert_called_once_with("ws-1", "/workspace")

    @pytest.mark.asyncio
    async def test_list_failure(self):
        cm = make_container_manager()
        cm.file_list = AsyncMock(side_effect=RuntimeError("no such directory"))
        state = make_state()
        tool = FileListTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(path="/nonexistent")

        assert not result.success
        assert "no such directory" in result.error

    def test_category_is_code(self):
        cm = make_container_manager()
        state = make_state()
        tool = FileListTool(container_manager=cm, workspace_id="ws-1", state=state)
        assert tool.category == ToolCategory.CODE


# ---------------------------------------------------------------------------
# DirTreeTool
# ---------------------------------------------------------------------------

class TestDirTreeTool:
    @pytest.mark.asyncio
    async def test_tree_uses_file_list_backend_for_populated_root(self):
        cm = make_container_manager()
        cm.file_list = AsyncMock(side_effect=[
            [
                FileEntry(name=".augmentum", path="/workspace/.augmentum", is_dir=True, size=4096),
                FileEntry(name="src", path="/workspace/src", is_dir=True, size=4096),
                FileEntry(name="README.md", path="/workspace/README.md", is_dir=False, size=1778),
            ],
            [
                FileEntry(name="click", path="/workspace/src/click", is_dir=True, size=4096),
                FileEntry(name="conftest.py", path="/workspace/src/conftest.py", is_dir=False, size=220),
            ],
            [
                FileEntry(name="__init__.py", path="/workspace/src/click/__init__.py", is_dir=False, size=120),
            ],
        ])
        cm.run_command = AsyncMock(side_effect=AssertionError("dir_tree should not shell out"))
        state = make_state()
        tool = DirTreeTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(path="/workspace", depth=3)

        assert result.success
        assert "/workspace/" in result.output
        assert "  src/" in result.output
        assert "  README.md  (2KB)" in result.output
        assert "    click/" in result.output
        assert "    conftest.py  (220B)" in result.output
        assert "      __init__.py  (120B)" in result.output

    @pytest.mark.asyncio
    async def test_tree_excludes_internal_noise_directories(self):
        cm = make_container_manager()
        cm.file_list = AsyncMock(return_value=[
            FileEntry(name=".git", path="/workspace/.git", is_dir=True, size=4096),
            FileEntry(name=".augmentum", path="/workspace/.augmentum", is_dir=True, size=4096),
            FileEntry(name="tests", path="/workspace/tests", is_dir=True, size=4096),
        ])
        state = make_state()
        tool = DirTreeTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(path="/workspace", depth=2)

        assert result.success
        assert ".git/" not in result.output
        assert ".augmentum/" not in result.output
        assert "  tests/" in result.output

    @pytest.mark.asyncio
    async def test_tree_empty_directory_output(self):
        cm = make_container_manager(file_list=[])
        state = make_state()
        tool = DirTreeTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(path="/workspace/empty", depth=2)

        assert result.success
        assert "/workspace/empty/" in result.output
        assert "(empty directory)" in result.output


# ---------------------------------------------------------------------------
# CodeEditTool
# ---------------------------------------------------------------------------

class TestCodeEditTool:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_cls,kwargs", [
        (CodeEditTool, {"search": "a", "replace": "b"}),
        (CodeMultiEditTool, {"edits": [{"search": "a", "replace": "b"}]}),
    ])
    async def test_edit_outside_workspace_rejected(self, tool_cls, kwargs):
        # Same confinement contract as file_write — see
        # TestFileWriteTool.test_write_outside_workspace_rejected.
        cm = make_container_manager()
        state = make_state()
        tool = tool_cls(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(path="/etc/hosts", **kwargs)

        assert not result.success
        assert result.validation_error
        assert "/workspace" in result.error
        cm.file_read.assert_not_called()

    def test_descriptions_reflect_native_guard_setting(self):
        cm = make_container_manager()
        state = make_state()
        strict_tool = CodeEditTool(
            container_manager=cm,
            workspace_id="ws-1",
            state=state,
        )
        native_tool = CodeEditTool(
            container_manager=cm,
            workspace_id="ws-1",
            state=state,
            strict_edit_guard=False,
        )
        native_batch_tool = CodeMultiEditTool(
            container_manager=cm,
            workspace_id="ws-1",
            state=state,
            strict_edit_guard=False,
        )

        assert "MUST have been read with file_read first" in strict_tool.description
        assert "not required in native mode" in native_tool.description
        assert "MUST have been read with file_read first" not in native_tool.description
        assert "not required in native mode" in native_batch_tool.description

    @pytest.mark.asyncio
    async def test_rejects_unread_file(self):
        cm = make_container_manager(run_output="old code")
        state = make_state()
        tool = CodeEditTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(
            path="/workspace/main.py",
            search="old code",
            replace="new code",
        )

        assert not result.success
        assert "read" in result.error.lower()

    @pytest.mark.asyncio
    async def test_accepts_read_file(self):
        content = "def foo():\n    return 1\n"
        cm = make_container_manager(run_output=content)
        state = make_state()
        state.record_file_read("/workspace/main.py")

        tool = CodeEditTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(
            path="/workspace/main.py",
            search="return 1",
            replace="return 2",
        )

        assert result.success
        # file_write should have been called with modified content
        cm.file_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_reports_matching_tier_exact(self):
        content = "def foo():\n    return 1\n"
        cm = make_container_manager(run_output=content)
        state = make_state()
        state.record_file_read("/workspace/main.py")

        tool = CodeEditTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(
            path="/workspace/main.py",
            search="return 1",
            replace="return 2",
        )

        assert result.success
        assert result.metadata.get("tier") == "exact"

    @pytest.mark.asyncio
    async def test_no_match_returns_error(self):
        content = "def foo():\n    return 1\n"
        cm = make_container_manager(run_output=content)
        state = make_state()
        state.record_file_read("/workspace/main.py")

        tool = CodeEditTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(
            path="/workspace/main.py",
            search="completely_absent_text",
            replace="replacement",
        )

        assert not result.success
        assert "no match" in result.error.lower() or "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_path(self):
        cm = make_container_manager()
        state = make_state()
        tool = CodeEditTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(path="", search="x", replace="y")

        assert not result.success
        assert result.validation_error

    @pytest.mark.asyncio
    async def test_missing_search(self):
        cm = make_container_manager()
        state = make_state()
        tool = CodeEditTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(path="/workspace/main.py", search="", replace="y")

        assert not result.success
        assert result.validation_error

    def test_category_is_code(self):
        cm = make_container_manager()
        state = make_state()
        tool = CodeEditTool(container_manager=cm, workspace_id="ws-1", state=state)
        assert tool.category == ToolCategory.CODE


# ---------------------------------------------------------------------------
# CodeGrepTool
# ---------------------------------------------------------------------------

class TestCodeGrepTool:
    @pytest.mark.asyncio
    async def test_basic_grep(self):
        cm = make_container_manager(run_output="/workspace/main.py:5:def hello():")
        state = make_state()
        tool = CodeGrepTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(pattern="def hello")

        assert result.success
        assert "main.py" in result.output
        cm._run_command.assert_called_once()
        cmd_arg = cm._run_command.call_args[0][1]
        # Should use grep -rn
        assert cmd_arg[0] == "grep"

    @pytest.mark.asyncio
    async def test_with_custom_path(self):
        cm = make_container_manager(run_output="")
        state = make_state()
        tool = CodeGrepTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(pattern="TODO", path="/workspace/src")

        assert result.success
        cmd_arg = cm._run_command.call_args[0][1]
        assert "/workspace/src" in cmd_arg

    @pytest.mark.asyncio
    async def test_missing_pattern(self):
        cm = make_container_manager()
        state = make_state()
        tool = CodeGrepTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(pattern="")

        assert not result.success
        assert result.validation_error

    @pytest.mark.asyncio
    async def test_grep_failure(self):
        cm = make_container_manager()
        cm._run_command = AsyncMock(side_effect=RuntimeError("grep crashed"))

        cm.run_command = cm._run_command
        state = make_state()
        tool = CodeGrepTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(pattern="foo")

        assert not result.success
        assert "grep crashed" in result.error

    def test_category_is_code(self):
        cm = make_container_manager()
        state = make_state()
        tool = CodeGrepTool(container_manager=cm, workspace_id="ws-1", state=state)
        assert tool.category == ToolCategory.CODE

    @pytest.mark.asyncio
    async def test_context_lines_adds_grep_C_and_counts_matches_only(self):
        # grep -C output: match lines use ':', context lines use '-',
        # groups separated by '--'.
        grep_out = (
            "/workspace/a.py-4-import os\n"
            "/workspace/a.py:5:def hello():\n"
            "/workspace/a.py-6-    pass\n"
            "--\n"
            "/workspace/b.py-9-# setup\n"
            "/workspace/b.py:10:def hello_again():\n"
            "/workspace/b.py-11-    pass"
        )
        cm = make_container_manager(run_output=grep_out)
        state = make_state()
        tool = CodeGrepTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(pattern="def hello", context_lines=1)

        assert result.success
        cmd_arg = cm._run_command.call_args[0][1]
        assert "-C" in cmd_arg and "1" in cmd_arg
        # Context lines are present in the output...
        assert "import os" in result.output
        # ...but only real match lines are counted.
        assert result.metadata["matches_found"] == 2
        assert result.metadata["context_lines"] == 1

    @pytest.mark.asyncio
    async def test_context_limit_caps_by_match_not_by_line(self):
        grep_out = (
            "/workspace/a.py-1-before\n"
            "/workspace/a.py:2:match one\n"
            "/workspace/a.py-3-after\n"
            "--\n"
            "/workspace/a.py-7-before\n"
            "/workspace/a.py:8:match two\n"
            "/workspace/a.py-9-after"
        )
        cm = make_container_manager(run_output=grep_out)
        state = make_state()
        tool = CodeGrepTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(pattern="match", context_lines=1, limit=1)

        assert result.success
        # First group (with its context) survives; second match is cut.
        assert "match one" in result.output
        assert "match two" not in result.output
        assert result.metadata["matches_found"] == 2
        assert result.metadata["matches_shown"] == 1


# ---------------------------------------------------------------------------
# CodeGlobTool
# ---------------------------------------------------------------------------

class TestCodeGlobTool:
    @pytest.mark.asyncio
    async def test_basic_glob(self):
        cm = make_container_manager(run_output="/workspace/main.py\n/workspace/utils.py\n")
        state = make_state()
        tool = CodeGlobTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(pattern="*.py")

        assert result.success
        assert "main.py" in result.output
        cmd_arg = cm._run_command.call_args[0][1]
        assert cmd_arg[0] == "find"

    @pytest.mark.asyncio
    async def test_missing_pattern(self):
        cm = make_container_manager()
        state = make_state()
        tool = CodeGlobTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(pattern="")

        assert not result.success
        assert result.validation_error

    @pytest.mark.asyncio
    async def test_glob_failure(self):
        cm = make_container_manager()
        cm._run_command = AsyncMock(side_effect=RuntimeError("no workspace"))

        cm.run_command = cm._run_command
        state = make_state()
        tool = CodeGlobTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(pattern="*.ts")

        assert not result.success
        assert "no workspace" in result.error

    def test_category_is_code(self):
        cm = make_container_manager()
        state = make_state()
        tool = CodeGlobTool(container_manager=cm, workspace_id="ws-1", state=state)
        assert tool.category == ToolCategory.CODE


# ---------------------------------------------------------------------------
# ShellExecTool
# ---------------------------------------------------------------------------

class TestShellExecTool:
    @pytest.mark.asyncio
    async def test_runs_command(self):
        cm = make_container_manager(run_output="npm: 9.6.7")
        state = make_state()
        tool = ShellExecTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(command="npm --version")

        assert result.success
        assert "npm: 9.6.7" in result.output
        cmd_arg = cm._run_command.call_args[0][1]
        # ``-lc`` (login shell) so /etc/profile.d/*.sh is sourced and
        # tools installed outside the system PATH (cargo, pipx, nvm,
        # etc.) are visible to the very next command.
        assert cmd_arg == ["bash", "-lc", "npm --version"]

    @pytest.mark.asyncio
    async def test_missing_command(self):
        cm = make_container_manager()
        state = make_state()
        tool = ShellExecTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(command="")

        assert not result.success
        assert result.validation_error

    @pytest.mark.asyncio
    async def test_shell_failure(self):
        cm = make_container_manager()
        cm._run_command = AsyncMock(side_effect=RuntimeError("exec failed"))

        cm.run_command = cm._run_command
        state = make_state()
        tool = ShellExecTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(command="npm install")

        assert not result.success
        assert "exec failed" in result.error

    def test_category_is_shell(self):
        cm = make_container_manager()
        state = make_state()
        tool = ShellExecTool(container_manager=cm, workspace_id="ws-1", state=state)
        assert tool.category == ToolCategory.SHELL

    def test_cacheable_is_false(self):
        cm = make_container_manager()
        state = make_state()
        tool = ShellExecTool(container_manager=cm, workspace_id="ws-1", state=state)
        assert tool.cacheable is False

    @pytest.mark.asyncio
    async def test_trailing_ampersand_records_background_process(self):
        """A command ending in `&` is treated as a backgrounded process
        and recorded in state so the sticky reminder can show it."""
        cm = make_container_manager(run_output="")
        state = make_state()
        tool = ShellExecTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(command="./server --port 4000 &")

        assert result.success
        assert len(state.background_processes) == 1
        assert state.background_processes[0]["command"].endswith("&")

    @pytest.mark.asyncio
    async def test_nohup_records_background_process(self):
        """``nohup <cmd>`` is the other common backgrounding idiom."""
        cm = make_container_manager(run_output="")
        state = make_state()
        tool = ShellExecTool(container_manager=cm, workspace_id="ws-1", state=state)

        await tool.execute(command="nohup python -m http.server 8080")

        assert len(state.background_processes) == 1

    @pytest.mark.asyncio
    async def test_double_ampersand_not_recorded_as_background(self):
        """``cmd1 && cmd2`` is logical AND, not backgrounding — must
        NOT appear in background_processes."""
        cm = make_container_manager(run_output="ok")
        state = make_state()
        tool = ShellExecTool(container_manager=cm, workspace_id="ws-1", state=state)

        await tool.execute(command="cargo build --release && ./target/release/app")

        assert state.background_processes == []

    @pytest.mark.asyncio
    async def test_repeat_background_increments_count(self):
        """Running the same bg command twice bumps the count, doesn't
        create a duplicate entry — keeps the sticky reminder compact."""
        cm = make_container_manager(run_output="")
        state = make_state()
        tool = ShellExecTool(container_manager=cm, workspace_id="ws-1", state=state)

        await tool.execute(command="./server &")
        await tool.execute(command="./server &")

        assert len(state.background_processes) == 1
        assert state.background_processes[0]["count"] == 2

    def test_tool_timeout_exceeds_inner_max(self):
        """Tool-level timeout (used by the outer asyncio.wait_for in
        _execute_tool) must exceed the maximum INNER run_command
        timeout so the outer wrap never preempts with the
        uninformative "Tool 'shell_exec' timed out after 30.0s"
        message. Inner max is 600s (long patterns); tool cap 610s."""
        cm = make_container_manager()
        state = make_state()
        tool = ShellExecTool(container_manager=cm, workspace_id="ws-1", state=state)

        assert tool.timeout > 600.0, (
            f"ShellExecTool.timeout must exceed 600s (inner long-pattern "
            f"cap); got {tool.timeout}"
        )

    @pytest.mark.asyncio
    async def test_short_command_uses_120s_60s_timeouts(self):
        """Short commands: 120s wall-clock, 60s idle."""
        cm = make_container_manager(run_output="ok")
        state = make_state()
        tool = ShellExecTool(container_manager=cm, workspace_id="ws-1", state=state)

        await tool.execute(command="ls /workspace")

        call_kwargs = cm._run_command.call_args.kwargs
        assert call_kwargs["timeout"] == 120.0
        assert call_kwargs["idle_timeout"] == 60.0

    @pytest.mark.asyncio
    async def test_install_uses_long_timeouts(self):
        """install/build/compile/etc. patterns get 600s wall, 300s idle."""
        cm = make_container_manager(run_output="Reading package lists...")
        state = make_state()
        tool = ShellExecTool(container_manager=cm, workspace_id="ws-1", state=state)

        await tool.execute(command="apt-get install -y default-jdk")

        call_kwargs = cm._run_command.call_args.kwargs
        assert call_kwargs["timeout"] == 600.0
        assert call_kwargs["idle_timeout"] == 300.0

    @pytest.mark.asyncio
    async def test_download_keyword_triggers_long_timeout(self):
        """The 2026-04-22 timeout expansion added "download" to the
        long-pattern set so ``curl -O`` / ``wget`` / ``docker pull``
        get 10-min wall + 2-min idle rather than the tight short-cap."""
        cm = make_container_manager(run_output="")
        state = make_state()
        tool = ShellExecTool(container_manager=cm, workspace_id="ws-1", state=state)

        await tool.execute(command="curl -L https://example.com/file.tar.gz -o /tmp/download.tgz")

        call_kwargs = cm._run_command.call_args.kwargs
        assert call_kwargs["timeout"] == 600.0


# ---------------------------------------------------------------------------
# ShellReadTool
# ---------------------------------------------------------------------------

class TestShellReadTool:
    @pytest.mark.asyncio
    async def test_runs_read_command(self):
        cm = make_container_manager(run_output="commit abc123\ncommit def456")
        state = make_state()
        tool = ShellReadTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(command="git log --oneline")

        assert result.success
        assert "commit abc123" in result.output
        cmd_arg = cm._run_command.call_args[0][1]
        assert cmd_arg == ["bash", "-c", "git log --oneline"]

    @pytest.mark.asyncio
    async def test_missing_command(self):
        cm = make_container_manager()
        state = make_state()
        tool = ShellReadTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(command="")

        assert not result.success
        assert result.validation_error

    def test_category_is_shell(self):
        cm = make_container_manager()
        state = make_state()
        tool = ShellReadTool(container_manager=cm, workspace_id="ws-1", state=state)
        assert tool.category == ToolCategory.SHELL


# ---------------------------------------------------------------------------
# ApplyPatchTool
# ---------------------------------------------------------------------------

class TestApplyPatchTool:
    @pytest.mark.asyncio
    async def test_applies_unified_patch_via_git_apply(self):
        cm = make_container_manager()
        # Side-effect schedule:
        #  1. _current_mtime stat for the D2 read-before-edit guard.
        #     Empty string → file treated as new → guard skipped, patch
        #     proceeds. (Test target: the patch APPLIES, not the guard.)
        #  2. git apply --check
        #  3. git apply --whitespace
        #  4. _refresh_read_mtime_after_write — stat the changed file
        #     post-apply so the next same-turn edit sees a fresh
        #     baseline. Added post-2026-04-20 alongside the mtime-
        #     aware staleness guard; previously the test expected 3
        #     calls because this refresh was inline with the apply.
        # git apply --check / --whitespace now emit an in-band success
        # marker (echo'd only on a zero exit), because WorkspaceExecutor
        # never inspects the exit code — a failed apply used to be reported
        # as success. The apply output carries the marker followed by
        # `git status --short` porcelain lines. A cleanup `rm -f` runs last
        # on the happy path.
        cm.run_command = AsyncMock(
            side_effect=[
                "",  # 1. mtime stat (read-before-edit guard) → new file
                "__AUG_PATCH_CHECK_OK__",  # 2. git apply --check succeeds
                "__AUG_PATCH_APPLY_OK__\n M src/app.py\n M README.md",  # 3. apply + status
                "",  # 4. verify-loop mtime stat → None → skip verify
                "",  # 5. rm -f cleanup of the staged diff
            ],
        )
        cm._run_command = cm.run_command
        state = make_state()
        tool = ApplyPatchTool(container_manager=cm, workspace_id="ws-1", state=state)

        patch = (
            "diff --git a/src/app.py b/src/app.py\n"
            "--- a/src/app.py\n"
            "+++ b/src/app.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        result = await tool.execute(patch=patch)

        assert result.success
        assert "src/app.py" in result.output
        cm.file_upload.assert_awaited_once()
        # 5 calls: mtime stat, git apply --check, git apply --whitespace,
        # verify-loop mtime stat, rm -f cleanup of the staged diff.
        assert cm.run_command.await_count == 5
        assert "stat -c %Y" in cm.run_command.await_args_list[0].args[1][2]
        assert "git apply --check" in cm.run_command.await_args_list[1].args[1][2]
        assert "git apply --whitespace" in cm.run_command.await_args_list[2].args[1][2]
        assert "stat -c %Y" in cm.run_command.await_args_list[3].args[1][2]
        assert "rm -f" in cm.run_command.await_args_list[4].args[1][2]

    @pytest.mark.asyncio
    async def test_requires_patch_text(self):
        cm = make_container_manager()
        state = make_state()
        tool = ApplyPatchTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(patch="")

        assert not result.success
        assert result.validation_error


# ---------------------------------------------------------------------------
# ContainerInfoTool
# ---------------------------------------------------------------------------

class TestContainerInfoTool:
    @pytest.mark.asyncio
    async def test_reports_probe_and_ports(self):
        cm = make_container_manager(
            run_output="hostname=abc123\nips=172.19.0.9 \npwd=/workspace\n",
        )
        cm.list_ports = AsyncMock(return_value=[
            {"container_port": 3000, "host_port": 49153, "listening": True},
            {"container_port": 5173, "host_port": 0, "listening": False},
        ])
        state = make_state()
        tool = ContainerInfoTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute()

        assert result.success
        assert "Workspace ID: ws-1" in result.output
        assert "172.19.0.9" in result.output
        assert "http://127.0.0.1:49153" in result.output
        assert result.metadata["ips"] == ["172.19.0.9"]


# ---------------------------------------------------------------------------
# PublishPortsTool
# ---------------------------------------------------------------------------

class TestPublishPortsTool:
    @pytest.mark.asyncio
    async def test_publish_ports_recreates_workspace(self):
        cm = make_container_manager()
        cm.enable_published_ports = AsyncMock(return_value=(
            type("Info", (), {"status": "running"})(),
            True,
        ))
        state = make_state()
        tool = PublishPortsTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(reason="Need browser access")

        assert result.success
        assert "Workspace ports were exposed" in result.output
        assert result.metadata["workspace_recreated"] is True
        cm.enable_published_ports.assert_awaited_once_with("ws-1")

    def test_category_is_code(self):
        cm = make_container_manager()
        state = make_state()
        tool = PublishPortsTool(container_manager=cm, workspace_id="ws-1", state=state)
        assert tool.category == ToolCategory.CODE


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------

class TestOutputTruncation:
    @pytest.mark.asyncio
    async def test_truncates_large_output(self):
        large_content = "x" * 60_000
        cm = make_container_manager(run_output=large_content)
        state = make_state()
        tool = ShellReadTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(command="cat bigfile")

        assert result.success
        # Output now includes a prepended TRUNCATED header + 50k body +
        # appended trailer. Total stays roughly bounded; both markers
        # must be present so compaction (which keeps first-chars preview)
        # still carries the signal.
        assert len(result.output) <= 50_500
        assert result.output.startswith("[TRUNCATED")
        assert "truncated, 60000 total chars" in result.output

    @pytest.mark.asyncio
    async def test_no_truncation_for_small_output(self):
        small_content = "hello world"
        cm = make_container_manager(run_output=small_content)
        state = make_state()
        tool = ShellReadTool(container_manager=cm, workspace_id="ws-1", state=state)

        result = await tool.execute(command="echo hello world")

        assert result.success
        assert "truncated" not in result.output


# ---------------------------------------------------------------------------
# create_coder_tools factory
# ---------------------------------------------------------------------------

class TestCreateCoderTools:
    def test_returns_all_builtin_tools(self):
        cm = make_container_manager()
        state = make_state()
        tools = create_coder_tools(cm, "ws-1", state)
        assert len(tools) == len(ALL_CODER_TOOLS)

    def test_all_tools_have_names(self):
        cm = make_container_manager()
        state = make_state()
        tools = create_coder_tools(cm, "ws-1", state)
        names = {t.name for t in tools}
        assert "file_read" in names
        assert "file_write" in names
        assert "file_list" in names
        assert "code_edit" in names
        assert "apply_patch" in names
        assert "code_grep" in names
        assert "find_files" in names
        assert "container_info" in names
        assert "publish_ports" in names
        assert "shell_exec" in names
        assert "shell_read" in names
        assert "service_start" in names
        assert "browser_verify" in names
        assert "profile_read" in names

    def test_all_coder_tools_constant(self):
        assert len(ALL_CODER_TOOLS) >= 20


# ---------------------------------------------------------------------------
# READ_ONLY_TOOLS
# ---------------------------------------------------------------------------

class TestReadOnlyTools:
    def test_has_expected_read_only_tools(self):
        # Update when adding read-only tools to READ_ONLY_TOOLS in
        # ``augmentum/coder/tools.py``. 2026-05-27: added
        # browser_evaluate, http_request, db_inspect (count 18 → 21).
        # 2026-07-02: added browser_wait, browser_extract (21 → 23) —
        # same observe-only class as snapshot/verify/evaluate.
        assert len(READ_ONLY_TOOLS) == 23

    def test_read_only_set_contents(self):
        assert "file_read" in READ_ONLY_TOOLS
        assert "file_list" in READ_ONLY_TOOLS
        assert "dir_tree" in READ_ONLY_TOOLS
        assert "code_grep" in READ_ONLY_TOOLS
        assert "find_files" in READ_ONLY_TOOLS
        assert "code_search" in READ_ONLY_TOOLS
        assert "doc_search" in READ_ONLY_TOOLS
        assert "doc_fetch" in READ_ONLY_TOOLS
        assert "env_info" in READ_ONLY_TOOLS
        assert "container_info" in READ_ONLY_TOOLS
        assert "shell_read" in READ_ONLY_TOOLS
        assert "service_list" in READ_ONLY_TOOLS
        assert "service_logs" in READ_ONLY_TOOLS
        assert "service_probe" in READ_ONLY_TOOLS
        assert "browser_snapshot" in READ_ONLY_TOOLS
        assert "browser_verify" in READ_ONLY_TOOLS
        # New verification tools (2026-05-27) — see Tier 1 audit.
        assert "browser_evaluate" in READ_ONLY_TOOLS
        # Wave-2 browser primitives (2026-07-02) — observe-only, so
        # they join their siblings here; fill_form mutates the page
        # and stays out (asserted below).
        assert "browser_wait" in READ_ONLY_TOOLS
        assert "browser_extract" in READ_ONLY_TOOLS
        assert "http_request" in READ_ONLY_TOOLS
        assert "db_inspect" in READ_ONLY_TOOLS
        assert "profile_read" in READ_ONLY_TOOLS
        assert "task_list" in READ_ONLY_TOOLS

    def test_write_tools_not_in_read_only(self):
        assert "file_write" not in READ_ONLY_TOOLS
        assert "code_edit" not in READ_ONLY_TOOLS
        assert "apply_patch" not in READ_ONLY_TOOLS
        assert "publish_ports" not in READ_ONLY_TOOLS
        assert "shell_exec" not in READ_ONLY_TOOLS
        assert "service_start" not in READ_ONLY_TOOLS
        assert "service_stop" not in READ_ONLY_TOOLS
        assert "browser_open" not in READ_ONLY_TOOLS
        assert "browser_fill_form" not in READ_ONLY_TOOLS
        assert "profile_update" not in READ_ONLY_TOOLS


@pytest.mark.asyncio
async def test_code_edit_can_bypass_read_guard_for_native_mode():
    cm = make_container_manager(run_output="function doConnect() {\n  return 1;\n}\n")
    state = make_state()
    tool = CodeEditTool(
        container_manager=cm,
        workspace_id="ws-1",
        state=state,
        strict_edit_guard=False,
    )

    result = await tool.execute(
        path="/workspace/app.js",
        search="function doConnect() {",
        replace="async function doConnect() {",
    )

    assert result.success
    cm.file_write.assert_called_once()


class TestPreWriteValidate:
    """The syntax gate must judge the NEW content, never the path.

    From 2026-07-12 to 2026-07-18 the .py probe ran ``python3 -c
    "compile(open('{path}')...)"`` in the augmentum container — where
    workspace paths don't exist — so every .py write was blocked by a
    FileNotFoundError dressed as a syntax error, with the real reason
    truncated out of the model-facing message.
    """

    def test_valid_python_passes_despite_nonexistent_path(self):
        from augmentum.coder.tools import _pre_write_validate
        assert _pre_write_validate(
            "/workspace/does/not/exist/anywhere.py",
            "def f():\n    return 1\n",
        ) is None

    def test_broken_python_blocked_with_line_detail(self):
        from augmentum.coder.tools import _pre_write_validate
        err = _pre_write_validate("/workspace/x.py", "def f(:\n    pass\n")
        assert err is not None
        assert "line 1" in err  # the model needs the location to fix it

    def test_unchecked_language_passes(self):
        from augmentum.coder.tools import _pre_write_validate
        assert _pre_write_validate("/workspace/main.rs", "fn main() {}") is None
