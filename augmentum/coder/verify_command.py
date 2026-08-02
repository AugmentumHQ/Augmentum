"""Held-out verification-command gate for the coder act loop.

Arbor principle #2 (RUC-NLPIR/Arbor, reviewed 2026-06-16): a
self-improving loop needs an admission gate the improving agent cannot
argue past. ``goal_judge.py`` is an *independent* judge, but it still
reasons over the agent's OWN final report + a file list — a confident
summary can talk it into ``ok=true``. Arbor's answer is to split the
signal you *optimize* (dev) from the signal that *admits* (held-out),
and make admission mechanical: a merge is kept only if it clears the
held-out gate, regardless of what the agent claims.

This gate is the coder analog. On an accepted stop that made writes, it
runs the project's verification command (the configured
``coder_verify_command`` or an auto-detected test command) **in the
workspace container** and treats a real non-zero exit as an
authoritative "not done". The agent can't fake the result: the gate
appends its own exit-code sentinel *after* the real command, so the
held-out signal is ours, not the model's.

Design contract (mirrors goal_judge / agents.verify):
* **Default OFF** (``coder_verify_command_gate_enabled``). When on, fires
  only on accepted write-stops, *before* the LLM goal judge — a red exit
  short-circuits the stop without paying for the judge call.
* **Fail-OPEN on no-signal.** No command detected, container trouble, a
  missing sentinel (timeout/kill), or a shell-level failure (missing
  binary) → ``skip``. A flaky gate must never trap the user in a
  re-entry loop, AND a silent failure must never masquerade as a pass —
  hence ``log.warning`` on the infra paths, never ``debug``.
* **Only a clean non-zero exit rejects.** That is the one case where we
  have ground truth that the work is unfinished.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Re-entries per turn before the stop is honored unconditionally. Matches
# the goal judge's cap — an admission gate must never trap the user, and
# coder iterations are expensive.
MAX_VERIFY_REENTRY = 2

# Wall-clock ceiling for the verification command. Matches the test_run
# tool's default (augmentum/coder/tools.py).
VERIFY_TIMEOUT_S = 300.0

# We append `echo "<sentinel>:$?"` after the real command so the exit
# code survives ``run_command`` (which returns combined stdout/stderr
# and does NOT raise on non-zero). The agent can't forge this — it's
# emitted by the shell after its command runs, outside the model's reach.
_SENTINEL = "__AUGMENTUM_VERIFY_EXIT__"
_SENTINEL_RE = re.compile(rf"{_SENTINEL}:(-?\d+)\s*$")

# Single-shot framework probe — one container round-trip, no reliance on
# run_command's (ambiguous) non-zero-exit behavior. Mirrors the probe set
# in tools.py::_detect_test_command; keep the two in sync if either grows.
_DETECT_CMD = (
    "cd /workspace 2>/dev/null; "
    "if [ -f pytest.ini ] || [ -f pyproject.toml ] || [ -f setup.cfg ]; then echo PYTEST; "
    "elif [ -f package.json ]; then echo NPM; "
    "elif [ -f Cargo.toml ]; then echo CARGO; "
    "elif [ -f go.mod ]; then echo GO; "
    "elif [ -f Makefile ] && grep -qE '^test:' Makefile; then echo MAKE; "
    "fi"
)
_DETECT_MAP = {
    "PYTEST": "python -m pytest -x --tb=short",
    "NPM": "npm test",
    "CARGO": "cargo test",
    "GO": "go test ./...",
    "MAKE": "make test",
}

# Excerpt of the command output fed back to the model on a failure — big
# enough to carry the failing assertion / traceback, bounded so a noisy
# suite can't blow the context budget.
_OUTPUT_BUDGET = 2500


@dataclass(frozen=True, slots=True)
class VerifyOutcome:
    """Result of one verification-gate run.

    ``status`` is the only field callers branch on:
    * ``pass`` — command ran and exited 0; honor the stop.
    * ``fail`` — command ran and exited non-zero; authoritative reject.
    * ``skip`` — no signal (no command, infra trouble, couldn't run);
      fail open and let the goal judge / TQG decide.
    """

    status: Literal["pass", "fail", "skip"]
    command: str = ""
    exit_code: int | None = None
    output: str = ""
    skip_reason: str = ""


def _parse_sentinel(output: str) -> tuple[int | None, str]:
    """Split the trailing exit-code sentinel off the command output.

    Returns ``(exit_code, body)`` where ``exit_code`` is ``None`` when no
    sentinel is present (command killed / output truncated / shell died).
    The sentinel line is stripped from ``body`` so it never leaks into the
    model-facing excerpt.
    """
    m = _SENTINEL_RE.search(output.rstrip())
    if not m:
        return None, output
    code = int(m.group(1))
    body = output[: m.start()].rstrip()
    return code, body


async def _detect_command(container_manager, workspace_id: str) -> str:
    """Auto-detect a verification command from project marker files.

    One container round-trip. Returns ``""`` when nothing recognizable is
    present — the caller treats that as ``skip`` (a project without a
    known test runner has no held-out signal to gate on).
    """
    try:
        out = await container_manager.run_command(
            workspace_id, ["bash", "-c", _DETECT_CMD], timeout=8.0,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "coder_verify_detect_failed",
            workspace_id=workspace_id, error=str(exc)[:200],
        )
        return ""
    token = (out or "").strip().splitlines()[-1].strip() if (out or "").strip() else ""
    return _DETECT_MAP.get(token, "")


async def run_verification_gate(
    container_manager,
    workspace_id: str,
    *,
    command: str = "",
    timeout: float = VERIFY_TIMEOUT_S,
) -> VerifyOutcome:
    """Run the held-out verification command and classify the result.

    ``command`` is the explicit ``coder_verify_command`` setting; when
    empty, the gate auto-detects. Never raises — every failure mode maps
    to ``status="skip"`` so the caller fails open.
    """
    cmd = (command or "").strip()
    if not cmd:
        cmd = await _detect_command(container_manager, workspace_id)
        if not cmd:
            return VerifyOutcome(status="skip", skip_reason="no_command")

    # Wrap so the real command's exit code rides out on our sentinel. The
    # outer shell always exits 0 (the echo succeeds), so run_command won't
    # treat a failing suite as an infra error.
    wrapped = f"cd /workspace && ( {cmd} ) ; echo \"{_SENTINEL}:$?\""
    try:
        raw = await container_manager.run_command(
            workspace_id, ["bash", "-c", wrapped], timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        # Container down / mid-restart / wall-clock kill. No signal — never
        # trap the user on infra trouble, but say so loudly.
        log.warning(
            "coder_verify_run_failed",
            workspace_id=workspace_id, command=cmd[:120], error=str(exc)[:200],
        )
        return VerifyOutcome(status="skip", command=cmd, skip_reason="run_error")

    exit_code, body = _parse_sentinel(raw or "")
    if exit_code is None:
        # No sentinel: the command was killed or output was truncated
        # before the echo ran. Treat as no signal.
        log.warning(
            "coder_verify_no_sentinel",
            workspace_id=workspace_id, command=cmd[:120],
        )
        return VerifyOutcome(status="skip", command=cmd, skip_reason="no_sentinel")

    if exit_code != 0:
        # A non-zero exit MIGHT be a genuine test failure (authoritative)
        # or a shell-level "couldn't run" (missing binary → no signal).
        # _shell_command_failure distinguishes them; lazy-import keeps
        # this module off tools.py's heavy import graph.
        from augmentum.coder.tools import _shell_command_failure
        shell_fail = _shell_command_failure(body)
        if shell_fail is not None:
            log.warning(
                "coder_verify_shell_failure",
                workspace_id=workspace_id, command=cmd[:120], reason=shell_fail,
            )
            return VerifyOutcome(
                status="skip", command=cmd, exit_code=exit_code,
                skip_reason=f"shell_failure:{shell_fail}",
            )
        excerpt = body[-_OUTPUT_BUDGET:] if len(body) > _OUTPUT_BUDGET else body
        return VerifyOutcome(
            status="fail", command=cmd, exit_code=exit_code, output=excerpt,
        )

    return VerifyOutcome(status="pass", command=cmd, exit_code=0)


__all__ = [
    "MAX_VERIFY_REENTRY",
    "VERIFY_TIMEOUT_S",
    "VerifyOutcome",
    "run_verification_gate",
]
