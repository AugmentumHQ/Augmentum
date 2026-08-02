"""Tests for the held-out verification-command gate (Arbor principle #2).

Covers the classifier contract of ``run_verification_gate``: exit 0 →
pass, clean non-zero → fail (authoritative), every no-signal path → skip
(fail open). A ``FakeCM`` stands in for the container manager; the gate
appends its own exit-code sentinel, so the fake just echoes canned output
back the way ``run_command`` would (combined stdout/stderr, no raise on
non-zero).
"""
from __future__ import annotations

import pytest

from augmentum.coder.verify_command import (
    MAX_VERIFY_REENTRY,
    VerifyOutcome,
    _parse_sentinel,
    run_verification_gate,
)

_SENTINEL = "__AUGMENTUM_VERIFY_EXIT__"


class FakeCM:
    """Minimal ContainerManager stand-in.

    ``responses`` maps a substring-of-command → output string. The first
    matching key wins; an unmatched command returns "". ``raise_on`` is a
    substring that, when present in the command, makes run_command raise
    (to exercise the infra fail-open path). ``calls`` records every
    command for assertions.
    """

    def __init__(self, responses: dict[str, str], *, raise_on: str | None = None):
        self._responses = responses
        self._raise_on = raise_on
        self.calls: list[str] = []

    async def run_command(self, workspace_id, cmd, timeout=30.0, **_kw):
        # cmd is ["bash", "-c", "<script>"]; match against the script.
        script = cmd[-1] if isinstance(cmd, list) else str(cmd)
        self.calls.append(script)
        if self._raise_on and self._raise_on in script:
            raise RuntimeError("container is paused")
        for needle, out in self._responses.items():
            if needle in script:
                return out
        return ""


def _with_exit(body: str, code: int) -> str:
    """Append the gate's sentinel the way the wrapped shell command would."""
    return f"{body}\n{_SENTINEL}:{code}"


# ---------------------------------------------------------------------------
# _parse_sentinel
# ---------------------------------------------------------------------------

def test_parse_sentinel_extracts_code_and_strips_line():
    code, body = _parse_sentinel("3 passed\n" + f"{_SENTINEL}:0")
    assert code == 0
    assert body == "3 passed"
    assert _SENTINEL not in body


def test_parse_sentinel_missing_returns_none():
    code, body = _parse_sentinel("killed mid-run, no sentinel")
    assert code is None
    assert body == "killed mid-run, no sentinel"


def test_parse_sentinel_nonzero_code():
    code, body = _parse_sentinel(_with_exit("1 failed", 1))
    assert code == 1
    assert body == "1 failed"


# ---------------------------------------------------------------------------
# run_verification_gate — pass / fail / skip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_explicit_command_pass():
    cm = FakeCM({"pytest": _with_exit("5 passed", 0)})
    vo = await run_verification_gate(cm, "ws1", command="pytest -q")
    assert isinstance(vo, VerifyOutcome)
    assert vo.status == "pass"
    assert vo.exit_code == 0
    assert vo.command == "pytest -q"
    # Explicit command must skip detection — exactly one run_command call.
    assert len(cm.calls) == 1


@pytest.mark.asyncio
async def test_explicit_command_fail_is_authoritative():
    cm = FakeCM({"pytest": _with_exit("E   assert 1 == 2\n1 failed", 1)})
    vo = await run_verification_gate(cm, "ws1", command="pytest -q")
    assert vo.status == "fail"
    assert vo.exit_code == 1
    # The real failure output is carried back for the re-entry message,
    # with the sentinel line stripped.
    assert "assert 1 == 2" in vo.output
    assert _SENTINEL not in vo.output


@pytest.mark.asyncio
async def test_shell_failure_is_skip_not_fail():
    # Missing binary → non-zero exit BUT _shell_command_failure flags it as
    # "couldn't run", so the gate must fail open (skip), not reject the stop.
    cm = FakeCM({"pytest": _with_exit("bash: line 1: pytest: command not found", 127)})
    vo = await run_verification_gate(cm, "ws1", command="pytest -q")
    assert vo.status == "skip"
    assert vo.skip_reason.startswith("shell_failure")


@pytest.mark.asyncio
async def test_missing_sentinel_is_skip():
    # Command killed (timeout) before the echo ran → no sentinel → no signal.
    cm = FakeCM({"pytest": "running tests...\n(truncated, process killed)"})
    vo = await run_verification_gate(cm, "ws1", command="pytest -q")
    assert vo.status == "skip"
    assert vo.skip_reason == "no_sentinel"


@pytest.mark.asyncio
async def test_run_error_is_skip():
    cm = FakeCM({}, raise_on="pytest")
    vo = await run_verification_gate(cm, "ws1", command="pytest -q")
    assert vo.status == "skip"
    assert vo.skip_reason == "run_error"


# ---------------------------------------------------------------------------
# Auto-detection (empty command)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_autodetect_pytest_then_run():
    # First call is the detection probe (echoes a framework token); second
    # is the wrapped verification run.
    cm = FakeCM({
        "echo PYTEST": "PYTEST",          # detection probe branch
        "python -m pytest": _with_exit("7 passed", 0),
    })
    vo = await run_verification_gate(cm, "ws1")  # no explicit command
    assert vo.status == "pass"
    assert "pytest" in vo.command
    assert len(cm.calls) == 2  # detect + run


@pytest.mark.asyncio
async def test_autodetect_no_framework_is_skip():
    # Detection probe returns nothing recognizable → no held-out signal.
    cm = FakeCM({})  # detection returns "" ; never reaches a run
    vo = await run_verification_gate(cm, "ws1")
    assert vo.status == "skip"
    assert vo.skip_reason == "no_command"
    assert len(cm.calls) == 1  # only the detection probe


@pytest.mark.asyncio
async def test_detection_run_error_is_skip():
    cm = FakeCM({}, raise_on="echo PYTEST")
    vo = await run_verification_gate(cm, "ws1")
    assert vo.status == "skip"
    assert vo.skip_reason == "no_command"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_reentry_cap_matches_goal_judge():
    # The admission gate must be bounded so it never traps the user; it
    # shares the goal judge's cap by design.
    from augmentum.coder.goal_judge import MAX_JUDGE_REENTRY
    assert MAX_VERIFY_REENTRY == MAX_JUDGE_REENTRY == 2
