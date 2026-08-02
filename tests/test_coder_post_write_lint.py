"""Post-write lint hook — Aider-style reflected lint output.

The hook lives in ``augmentum/coder/lint.py`` and is invoked from each
mutating tool's ``execute`` after a successful write. Findings get
appended to ``ToolResult.output`` so the model sees them on the same
iteration without a separate state-machine for "reflected messages".
"""
from __future__ import annotations

import pytest

from augmentum.coder.lint import _commands_for_path, run_post_write_lint


class _StubCM:
    """Stand-in for ContainerManager. Records exec calls; returns
    pre-programmed outputs keyed by the first arg of the command."""

    def __init__(self, responses: dict[str, str] | None = None,
                 raises: dict[str, Exception] | None = None) -> None:
        self.calls: list[list[str]] = []
        self._responses = responses or {}
        self._raises = raises or {}

    async def _run_command(self, workspace_id, cmd, timeout=8.0):
        self.calls.append(list(cmd))
        if cmd and cmd[0] in self._raises:
            raise self._raises[cmd[0]]
        # Match by first arg primarily; fall back to second (e.g. python3 -m).
        key = cmd[0] if cmd[0] in self._responses else (
            cmd[1] if len(cmd) > 1 and cmd[1] in self._responses else ""
        )
        # Special key for the flat command name (e.g. "ruff").
        if key == "" and cmd[0] in self._responses:
            key = cmd[0]
        return self._responses.get(key, "")


# ---------------------------------------------------------------------------
# Per-extension dispatch
# ---------------------------------------------------------------------------


def test_python_dispatch_includes_py_compile_first():
    cmds = _commands_for_path("/workspace/foo.py")
    assert cmds, "expected lint candidates for .py"
    # py_compile is the stdlib check that always works — it must lead.
    assert cmds[0].name == "py_compile"


def test_javascript_dispatch_uses_node_check():
    cmds = _commands_for_path("/workspace/foo.js")
    # node --check leads (always available); eslint follows when
    # the project has it installed.
    assert [c.name for c in cmds] == ["node --check", "eslint"]


def test_unknown_extension_no_candidates():
    cmds = _commands_for_path("/workspace/README.md")
    assert cmds == []


# ---------------------------------------------------------------------------
# run_post_write_lint behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clean_lint_returns_none():
    """Linter exited with no output → None (no findings to surface)."""
    cm = _StubCM(responses={"python3": ""})
    out = await run_post_write_lint(cm, "ws", "/workspace/foo.py")
    assert out is None


@pytest.mark.asyncio
async def test_lint_findings_returned_with_label():
    """Non-empty stdout from the linter → formatted block with label."""
    cm = _StubCM(responses={
        "python3": "  File \"/workspace/foo.py\", line 3\n    def =\n        ^\nSyntaxError: invalid syntax",
    })
    out = await run_post_write_lint(cm, "ws", "/workspace/foo.py")
    assert out is not None
    assert "[py_compile]" in out
    assert "SyntaxError" in out


@pytest.mark.asyncio
async def test_missing_binary_falls_through_to_next_candidate():
    """If py_compile output looks like 'command not found', try ruff next.
    A real ruff finding then surfaces."""
    cm = _StubCM(responses={
        "python3": "bash: python3: command not found",
        "ruff":    "foo.py:3:1: F401 unused import",
    })
    out = await run_post_write_lint(cm, "ws", "/workspace/foo.py")
    # py_compile got skipped (looked like missing), ruff was tried.
    assert out is not None
    assert "[ruff]" in out
    assert "F401" in out


@pytest.mark.asyncio
async def test_truncation_caps_long_lint_output():
    """A 50KB pylint-style dump must be truncated to max_chars to keep
    context healthy."""
    cm = _StubCM(responses={"python3": "ERROR\n" * 5000})
    out = await run_post_write_lint(
        cm, "ws", "/workspace/foo.py", max_chars=300,
    )
    assert out is not None
    assert len(out) < 500   # 300 cap + label + truncation footer
    assert "truncated" in out


@pytest.mark.asyncio
async def test_unknown_extension_skips_lint():
    cm = _StubCM()
    out = await run_post_write_lint(cm, "ws", "/workspace/data.csv")
    assert out is None
    assert cm.calls == []  # no exec happened


@pytest.mark.asyncio
async def test_exception_in_runner_swallowed():
    """A blowing-up linter never breaks the write — None gets returned
    so the caller appends nothing and the tool result stays clean."""
    cm = _StubCM(raises={"python3": RuntimeError("docker exec died")})
    # Both candidates raise — should still return None, not propagate.
    cm._raises["ruff"] = RuntimeError("nope")
    out = await run_post_write_lint(cm, "ws", "/workspace/foo.py")
    assert out is None
