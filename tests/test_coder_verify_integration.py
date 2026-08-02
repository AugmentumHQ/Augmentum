"""Phase 3.2 integration: verification gate wired into the write tools.

Exercises ``_maybe_run_post_write_verify`` end-to-end through the three
write tools (file_write, code_edit, code_edit_batch). For each: clean
content passes silently, broken content surfaces the parse error in
the tool output, sets ``metadata['verification_failed']``, and records
the failure in ``CoderState.recent_tool_failures`` so Phase 2.2's
persistent ledger sees the pattern.

Lint is disabled per-test via monkeypatch so test output isn't
polluted with subprocess noise — the verify path is what we're
exercising.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.coder.state import CoderState
from augmentum.coder.tools import (
    CodeEditTool,
    CodeMultiEditTool,
    FileWriteTool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cm(file_read_content: str = "") -> MagicMock:
    cm = MagicMock()
    cm.file_read = AsyncMock(return_value=file_read_content)
    cm.file_write = AsyncMock(return_value=None)
    cm._run_command = AsyncMock(return_value="")
    cm.run_command = cm._run_command
    return cm


def _make_state_with_read(path: str) -> CoderState:
    state = CoderState(session_id="s", workspace_id="w")
    state.files_read[path] = float("inf")
    return state


@pytest.fixture
def disable_lint(monkeypatch):
    """Silence the in-container lint hook so test output is clean.
    The verify path is what we want to exercise; lint is independently
    tested elsewhere."""
    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "coder_auto_lint", False)
    yield


# ---------------------------------------------------------------------------
# file_write: clean + broken Python + broken JSON
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_write_clean_python_no_verification_flag(disable_lint):
    cm = _make_cm()
    state = CoderState(session_id="s", workspace_id="w")
    tool = FileWriteTool(container_manager=cm, workspace_id="w", state=state)

    result = await tool.execute(
        path="/workspace/clean.py",
        content="def f():\n    return 42\n",
    )

    assert result.success
    assert "verification_failed" not in result.metadata
    assert "Verification failed" not in result.output
    assert state.recent_tool_failures == []


@pytest.mark.asyncio
async def test_file_write_broken_python_flags_metadata(disable_lint):
    """A SyntaxError write must still succeed (the file IS on disk —
    that's reality) but be flagged with verification_failed metadata
    and surface the parse error in the tool output."""
    cm = _make_cm()
    state = CoderState(session_id="s", workspace_id="w")
    tool = FileWriteTool(container_manager=cm, workspace_id="w", state=state)

    result = await tool.execute(
        path="/workspace/bad.py",
        # Missing colon after function header — SyntaxError on line 1.
        content="def f(x)\n    return x\n",
    )

    assert result.success
    assert result.metadata.get("verification_failed") is True
    assert "Verification failed" in result.output
    assert "bad.py" in result.output
    assert "1" in result.output  # line number anchored
    # And the persistent ledger has the failure entry.
    assert len(state.recent_tool_failures) == 1
    entry = state.recent_tool_failures[0]
    assert entry["tool"] == "verify"
    assert entry["target"] == "/workspace/bad.py"


@pytest.mark.asyncio
async def test_file_write_broken_json_flags_metadata(disable_lint):
    cm = _make_cm()
    state = CoderState(session_id="s", workspace_id="w")
    tool = FileWriteTool(container_manager=cm, workspace_id="w", state=state)

    result = await tool.execute(
        path="/workspace/bad.json",
        content='{"a": 1, "b": 2,}',  # trailing comma
    )

    assert result.success
    assert result.metadata.get("verification_failed") is True
    assert "JSON error" in result.output
    assert "bad.json" in result.output


@pytest.mark.asyncio
async def test_file_write_unsupported_extension_silent(disable_lint):
    """An .md file has no checker — verify must NOT mark it failed."""
    cm = _make_cm()
    state = CoderState(session_id="s", workspace_id="w")
    tool = FileWriteTool(container_manager=cm, workspace_id="w", state=state)

    result = await tool.execute(
        path="/workspace/notes.md", content="# heading\nlist:",
    )

    assert result.success
    assert "verification_failed" not in result.metadata
    assert state.recent_tool_failures == []


# ---------------------------------------------------------------------------
# code_edit: clean + broken
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_code_edit_clean_python_no_flag(disable_lint):
    pre_content = "def f(x):\n    return x\n"
    cm = _make_cm(file_read_content=pre_content)
    path = "/workspace/app.py"
    state = _make_state_with_read(path)
    tool = CodeEditTool(container_manager=cm, workspace_id="w", state=state)

    result = await tool.execute(
        path=path,
        search="    return x",
        replace="    return x + 1",
    )

    assert result.success
    assert "verification_failed" not in result.metadata


@pytest.mark.asyncio
async def test_code_edit_breaks_syntax_flags_metadata(disable_lint):
    """A code_edit that lands a SyntaxError must flag and record."""
    pre_content = "def f(x):\n    return x\n"
    cm = _make_cm(file_read_content=pre_content)
    path = "/workspace/app.py"
    state = _make_state_with_read(path)
    tool = CodeEditTool(container_manager=cm, workspace_id="w", state=state)

    result = await tool.execute(
        path=path,
        # Replace turns the function header into something un-parseable.
        search="def f(x):",
        replace="def f(x",
    )

    assert result.success  # The write itself succeeded
    assert result.metadata.get("verification_failed") is True
    assert "Verification failed" in result.output
    assert len(state.recent_tool_failures) == 1
    assert state.recent_tool_failures[0]["target"] == path


# ---------------------------------------------------------------------------
# code_edit_batch: broken propagates the same way
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_code_edit_batch_breaks_syntax_flags_metadata(disable_lint):
    pre_content = "def f(x):\n    return x\n\ndef g(y):\n    return y\n"
    cm = _make_cm(file_read_content=pre_content)
    path = "/workspace/multi.py"
    state = _make_state_with_read(path)
    tool = CodeMultiEditTool(container_manager=cm, workspace_id="w", state=state)

    result = await tool.execute(
        path=path,
        edits=[
            {"search": "def f(x):", "replace": "def f(x"},
            {"search": "def g(y):", "replace": "def g(y"},
        ],
    )

    assert result.success
    assert result.metadata.get("verification_failed") is True
    assert "Verification failed" in result.output
    assert len(state.recent_tool_failures) == 1


# ---------------------------------------------------------------------------
# Disabled config bypasses the gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_via_config_bypass(monkeypatch):
    """coder_auto_verify=False must skip the gate entirely — even on
    broken code, no metadata flag and no ledger entry."""
    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "coder_auto_lint", False)
    monkeypatch.setattr(_settings, "coder_auto_verify", False)

    cm = _make_cm()
    state = CoderState(session_id="s", workspace_id="w")
    tool = FileWriteTool(container_manager=cm, workspace_id="w", state=state)

    result = await tool.execute(
        path="/workspace/bad.py",
        content="def f(x)\n    return x\n",  # SyntaxError
    )

    assert result.success
    assert "verification_failed" not in result.metadata
    assert "Verification failed" not in result.output
    assert state.recent_tool_failures == []


# ---------------------------------------------------------------------------
# Best-effort: gate exception doesn't break the write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_exception_does_not_break_write(monkeypatch, disable_lint):
    """If the verification gate itself raises, the write still
    succeeds — verify is best-effort by contract (parallels lint)."""
    cm = _make_cm()
    state = CoderState(session_id="s", workspace_id="w")
    tool = FileWriteTool(container_manager=cm, workspace_id="w", state=state)

    # Patch the import target inside _maybe_run_post_write_verify so the
    # symbol lookup hits a broken stand-in. Tools imports the gate
    # lazily inside the method, so monkeypatching the module attribute
    # is what the runtime will see.
    class _BrokenGate:
        @classmethod
        def default(cls):
            raise RuntimeError("intentional gate failure")

    monkeypatch.setattr(
        "augmentum.coder.verify.VerificationGate", _BrokenGate,
    )

    result = await tool.execute(
        path="/workspace/x.py",
        content="def f(x)\n    return x\n",  # would normally fail verify
    )

    assert result.success
    assert "verification_failed" not in result.metadata
    # No spurious ledger entry from a gate that didn't actually run.
    assert state.recent_tool_failures == []


# ---------------------------------------------------------------------------
# Recurring failures stay fresh in the ledger (Phase 2.2 integration)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recurring_verification_failures_dedupe_in_ledger(disable_lint):
    """Two separate writes that both fail verification on the same
    path collapse to one ledger entry with count=2 — Phase 2.2's
    cross-turn dedupe path. The pattern, not the count of attempts,
    is what surfaces in the next turn's sticky reminder."""
    cm = _make_cm()
    state = CoderState(session_id="s", workspace_id="w")
    tool = FileWriteTool(container_manager=cm, workspace_id="w", state=state)

    await tool.execute(
        path="/workspace/x.py", content="def f(x)\n    return x\n",
    )
    await tool.execute(
        path="/workspace/x.py", content="def g(y)\n    return y\n",
    )

    assert len(state.recent_tool_failures) == 1
    assert state.recent_tool_failures[0]["count"] == 2


# ---------------------------------------------------------------------------
# Output ordering: verify message before lint findings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_write_broken_yaml_flags_metadata(disable_lint):
    """Phase 3.3: YAML coverage. The default gate now picks up
    .yaml writes via PyYAML; verify that broken YAML surfaces the
    same way Python and JSON do."""
    from augmentum.coder.verify import _YAML_AVAILABLE
    if not _YAML_AVAILABLE:
        pytest.skip("PyYAML not installed in this env")

    cm = _make_cm()
    state = CoderState(session_id="s", workspace_id="w")
    tool = FileWriteTool(container_manager=cm, workspace_id="w", state=state)

    result = await tool.execute(
        path="/workspace/compose.yaml",
        # Unclosed flow sequence — classic YAML breaker.
        content="services:\n  web:\n    ports: [80, 443\n",
    )

    assert result.success
    assert result.metadata.get("verification_failed") is True
    assert "YAML error" in result.output
    assert "compose.yaml" in result.output
    assert len(state.recent_tool_failures) == 1
    assert state.recent_tool_failures[0]["target"] == "/workspace/compose.yaml"


@pytest.mark.asyncio
async def test_file_write_broken_toml_flags_metadata(disable_lint):
    """Phase 3.3: TOML coverage. pyproject.toml is the highest-stakes
    .toml file in this codebase — a broken write would silently break
    every subsequent pip install. Catching it at write time is the
    whole point of the gate."""
    cm = _make_cm()
    state = CoderState(session_id="s", workspace_id="w")
    tool = FileWriteTool(container_manager=cm, workspace_id="w", state=state)

    result = await tool.execute(
        path="/workspace/pyproject.toml",
        content='valid = 1\nbad = key with space\nmore = 2\n',
    )

    assert result.success
    assert result.metadata.get("verification_failed") is True
    assert "TOML error" in result.output
    assert "pyproject.toml" in result.output
    assert len(state.recent_tool_failures) == 1


@pytest.mark.asyncio
async def test_file_write_clean_yaml_no_flag(disable_lint):
    from augmentum.coder.verify import _YAML_AVAILABLE
    if not _YAML_AVAILABLE:
        pytest.skip("PyYAML not installed in this env")

    cm = _make_cm()
    state = CoderState(session_id="s", workspace_id="w")
    tool = FileWriteTool(container_manager=cm, workspace_id="w", state=state)

    result = await tool.execute(
        path="/workspace/cfg.yaml",
        content="version: 1\nservices:\n  web:\n    image: nginx\n",
    )

    assert result.success
    assert "verification_failed" not in result.metadata


@pytest.mark.asyncio
async def test_file_write_clean_toml_no_flag(disable_lint):
    cm = _make_cm()
    state = CoderState(session_id="s", workspace_id="w")
    tool = FileWriteTool(container_manager=cm, workspace_id="w", state=state)

    result = await tool.execute(
        path="/workspace/pyproject.toml",
        content='[tool.ruff]\nline-length = 100\n',
    )

    assert result.success
    assert "verification_failed" not in result.metadata


@pytest.mark.asyncio
async def test_verify_message_precedes_lint_in_output(monkeypatch):
    """Blocking errors should appear before warnings in the appended
    block so the model reads the most actionable signal first."""
    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "coder_auto_lint", True)
    monkeypatch.setattr(_settings, "coder_auto_verify", True)

    cm = _make_cm()
    # Stub lint to produce a deterministic non-empty findings string.
    async def _fake_lint(*args, **kwargs):
        return "\n\n[ruff]\nE501: line too long"
    monkeypatch.setattr(
        "augmentum.coder.lint.run_post_write_lint", _fake_lint,
    )

    state = CoderState(session_id="s", workspace_id="w")
    tool = FileWriteTool(container_manager=cm, workspace_id="w", state=state)
    result = await tool.execute(
        path="/workspace/x.py", content="def f(x)\n    return x\n",
    )

    verify_idx = result.output.find("Verification failed")
    lint_idx = result.output.find("[ruff]")
    assert verify_idx >= 0
    assert lint_idx >= 0
    assert verify_idx < lint_idx, (
        "Blocking verify message should appear before non-blocking lint "
        f"findings; got verify@{verify_idx}, lint@{lint_idx}"
    )
