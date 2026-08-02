"""Tool-call guards implementing reward-hacking defenses.

A *guard* is a callable that inspects a pending tool call and returns
either ``None`` (allow) or a short reason string (deny). The subagent
runtime invokes the guard before dispatching the tool; denied calls are
reported back to the model as a tool result so it can adjust, the same
way a tool-side validation error would be.

Lifted from bug_finder/guards.py in 2026-05-31. The exploit catalog
(git --all peek, test deletion, verifier inspection) was tuned for
SWE-bench-style audits but the guards themselves are generic — any
subagent role can opt into them.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable
from typing import Any

ToolGuard = Callable[[str, dict[str, Any]], "str | None"]


_DENIED_GIT_TOKENS: frozenset[str] = frozenset({
    "--all",
    "--branches",
    "--tags",
    "--remotes",
    "--exclude",
    "--source",
    "--reflog",
})


_DENIED_GIT_SUBCOMMANDS: frozenset[str] = frozenset({
    "tag",
    "ls-remote",
    "fetch",
    "remote",
})


_TEST_PATH_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p) for p in (
        r"(^|/)tests?/",
        r"(^|/)__tests__/",
        r"(^|/)test_[^/]+\.py$",
        r"[^/]+_test\.go$",
        r"[^/]+\.test\.[jt]sx?$",
        r"[^/]+\.spec\.[jt]sx?$",
    )
)


def _git_command_violates(tokens: list[str]) -> str | None:
    if tokens and tokens[0] == "git":
        tokens = tokens[1:]
    if not tokens:
        return None
    sub = tokens[0]
    if sub in _DENIED_GIT_SUBCOMMANDS:
        return f"git {sub} disallowed (workspace refs are deliberately stripped)"
    for tok in tokens[1:]:
        if tok in _DENIED_GIT_TOKENS:
            return f"git flag {tok} disallowed (would escape pinned ref set)"
    return None


def _shell_command_violates(command: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return "shell command failed to parse safely"
    if not tokens:
        return None
    i = 0
    while i < len(tokens):
        if tokens[i] == "git":
            slice_end = len(tokens)
            for j in range(i + 1, len(tokens)):
                if tokens[j] in {"&&", "||", ";", "|", "&"}:
                    slice_end = j
                    break
            reason = _git_command_violates(tokens[i:slice_end])
            if reason:
                return reason
            i = slice_end
        i += 1
    return None


def _path_is_test_file(path: str) -> bool:
    p = (path or "").lower()
    return any(rx.search(p) for rx in _TEST_PATH_PATTERNS)


def detector_guard(tool_name: str, args: dict[str, Any]) -> str | None:
    """Read-only guard: denies git commands that escape pinned refs."""
    if tool_name == "git":
        return _shell_command_violates(f"git {args.get('command', '')}")
    if tool_name in {"shell_exec", "shell_read"}:
        return _shell_command_violates(str(args.get("command", "")))
    return None


def fixer_guard(tool_name: str, args: dict[str, Any]) -> str | None:
    """Detector guard + denies edits to existing test files."""
    git_denial = detector_guard(tool_name, args)
    if git_denial:
        return git_denial

    if tool_name in {"file_write", "code_edit", "code_multi_edit", "code_edit_batch"}:
        target = str(args.get("path") or args.get("file_path") or "")
        if target and _path_is_test_file(target):
            return (
                f"edit to test file {target!r} disallowed for fixer role "
                "(prevents deleting/weakening tests to fake green)"
            )

    if tool_name == "apply_patch":
        patch = str(args.get("patch", ""))
        for line in patch.splitlines():
            if line.startswith("+++ "):
                target = line[4:].lstrip().removeprefix("b/").strip()
                if target and _path_is_test_file(target):
                    return (
                        f"patch targets test file {target!r}, disallowed "
                        "for fixer role"
                    )

    return None


def verifier_guard(tool_name: str, args: dict[str, Any]) -> str | None:
    """Verifier may author tests but is bound by detector_guard otherwise."""
    return detector_guard(tool_name, args)


def planner_guard(tool_name: str, args: dict[str, Any]) -> str | None:
    """Planner: read-only with detector_guard's git-ref protections."""
    return detector_guard(tool_name, args)


def role_guard(role: str) -> ToolGuard:
    """Return the appropriate guard for a role name."""
    return {
        "planner": planner_guard,
        "detector": detector_guard,
        "verifier": verifier_guard,
        "fixer": fixer_guard,
    }.get(role, detector_guard)
