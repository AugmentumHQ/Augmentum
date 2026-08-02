"""Verifier implementations for the Promise runtime.

A verifier is an async callable ``(Promise, evidence) -> VerifyResult``.
Verifiers are deterministic checks — they do not ask the LLM. This is
the Voyager-style skill-verification pattern: the runtime, not the
model, decides whether a promise is fulfilled.

For coder mode, verifiers typically wrap a container-scoped shell
runner so that ``{"kind": "shell", "cmd": "test -f ..."}`` executes
inside the agent's sandbox.
"""
from __future__ import annotations

import shlex
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from augmentum.promises.models import Promise, Verification, VerificationKind

# Caller provides a function that runs a command in whatever sandbox is
# relevant and returns ``(exit_code, combined_output)``. This indirection
# lets the coder mode use its ContainerManager while tests can stub it.
ShellRunFn = Callable[[str, float], Awaitable[tuple[int, str]]]


@dataclass(frozen=True)
class VerifyResult:
    passed: bool
    reason: str


class VerifyFn(Protocol):
    async def __call__(
        self, promise: Promise, evidence: str | None,
    ) -> VerifyResult: ...


# --- Built-in verifiers --------------------------------------------------


async def always_pass(promise: Promise, evidence: str | None) -> VerifyResult:
    """Always returns pass. Use sparingly — only for communicative steps
    that have no observable postcondition (e.g. ``{"kind": "always"}``
    on a 'summarize findings' step)."""
    return VerifyResult(True, evidence.strip()[:200] if evidence else "no postcondition")


def shell_verifier(run_shell: ShellRunFn, default_timeout: float = 30.0) -> VerifyFn:
    """Verifier for ``{"kind": "shell", "spec": {"cmd": "..."}}``.

    The command is considered to have fulfilled the promise when it
    exits 0. The command's own output becomes the verification reason
    (truncated) so it can be shown as evidence.
    """

    async def verify(promise: Promise, evidence: str | None) -> VerifyResult:
        cmd = promise.verify.spec.get("cmd")
        if not cmd or not isinstance(cmd, str):
            return VerifyResult(False, "shell verify missing 'cmd' field")
        timeout = float(promise.verify.spec.get("timeout") or default_timeout)
        try:
            exit_code, output = await run_shell(cmd, timeout)
        except Exception as exc:  # noqa: BLE001 — runner wants any failure surfaced
            return VerifyResult(False, f"shell verify error: {exc}")
        out_trim = (output or "").strip()[:500]
        if exit_code == 0:
            return VerifyResult(True, out_trim or "exit 0")
        return VerifyResult(False, f"exit {exit_code}: {out_trim}")

    return verify


def file_verifier(run_shell: ShellRunFn) -> VerifyFn:
    """Verifier for ``{"kind": "file", "spec": {"path": "...", "must_exist": true}}``.

    ``must_exist`` defaults to True. Uses a shell ``test -e`` under the
    hood so paths are interpreted inside the same sandbox as the agent.
    """

    async def verify(promise: Promise, evidence: str | None) -> VerifyResult:
        path = promise.verify.spec.get("path")
        must_exist = bool(promise.verify.spec.get("must_exist", True))
        if not path or not isinstance(path, str):
            return VerifyResult(False, "file verify missing 'path' field")
        cmd = f"test -e {shlex.quote(path)}"
        try:
            exit_code, _ = await run_shell(cmd, 5.0)
        except Exception as exc:  # noqa: BLE001
            return VerifyResult(False, f"file verify error: {exc}")
        exists = exit_code == 0
        if exists == must_exist:
            state = "exists" if exists else "absent"
            return VerifyResult(True, f"{path} {state} as expected")
        actual = "exists" if exists else "absent"
        wanted = "exists" if must_exist else "absent"
        return VerifyResult(False, f"{path} is {actual}, expected {wanted}")

    return verify


def any_of_verifier(
    sub_verifiers: dict[VerificationKind, VerifyFn],
) -> VerifyFn:
    """Verifier for ``{"kind": "any_of", "spec": {"checks": [...]}}``.

    Passes iff at least one child check passes. Each child is a normal
    verify spec (same shape as a top-level verify dict). This is the
    semantic escape hatch: a planner that cannot predict the exact
    post-state (clone path, auto-generated binary name) can list all
    plausible targets and succeed on any match.

    Reason strings from failing children are concatenated so the retry
    prompt carries enough signal for the act layer to choose an
    approach that matches *some* check.
    """

    async def verify(promise: Promise, evidence: str | None) -> VerifyResult:
        raw_checks = promise.verify.spec.get("checks")
        if not isinstance(raw_checks, list) or not raw_checks:
            return VerifyResult(False, "any_of verify missing 'checks' list")
        failures: list[str] = []
        for raw in raw_checks:
            if not isinstance(raw, dict):
                failures.append("non-dict check ignored")
                continue
            try:
                child_verify = Verification.from_dict(raw)
            except (KeyError, ValueError) as exc:
                failures.append(f"bad child verify: {exc}")
                continue
            sub = sub_verifiers.get(child_verify.kind)
            if sub is None:
                failures.append(f"no verifier for {child_verify.kind.value}")
                continue
            child_promise = Promise(
                description=promise.description,
                verify=child_verify,
                max_attempts=promise.max_attempts,
            )
            result = await sub(child_promise, evidence)
            if result.passed:
                return VerifyResult(
                    True, f"matched {child_verify.kind.value}: {result.reason}",
                )
            failures.append(f"{child_verify.kind.value}: {result.reason}")
        return VerifyResult(
            False,
            "no any_of branch matched — " + "; ".join(failures[:4]),
        )

    return verify


# --- Convenience -------------------------------------------------------


def default_verify_fns(run_shell: ShellRunFn) -> dict[VerificationKind, VerifyFn]:
    """Bundle the phase-1 verifiers that need a shell runner.

    Callers can extend this dict with their own kinds (voice's
    user_confirm, an llm_judge wrapper, etc.).
    """
    base: dict[VerificationKind, VerifyFn] = {
        VerificationKind.SHELL: shell_verifier(run_shell),
        VerificationKind.FILE: file_verifier(run_shell),
        VerificationKind.ALWAYS: always_pass,
    }
    # any_of delegates to the other verifiers, so it must be built after
    # the base set is populated.
    base[VerificationKind.ANY_OF] = any_of_verifier(base)
    return base
