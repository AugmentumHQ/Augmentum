"""In-process verification gate for coder edits — Phase 3.1 of the
coder foundation (docs/superpowers/specs/2026-05-10-coder-foundation.md).

After (or during) a turn's writes the gate runs registered checkers
against the new file content and returns a structured report. Failures
on blocking checkers are what the wiring layer (Phase 3.2) uses to
trigger rollback or feed the failure back to the model.

Why a separate module from ``lint.py``
--------------------------------------
``augmentum/coder/lint.py`` runs inside the workspace container via
``container_manager._run_command``. It uses the project's own toolchain
(ruff, eslint, py_compile, etc.) and is authoritative — but it costs a
subprocess hop per file and depends on the container being healthy.

``verify.py`` runs in-process via stdlib (``ast.parse``, ``json.loads``).
It catches the cheapest, most common failure mode — unparseable files —
before any subprocess hop and works even when the container is
mid-restart or the project's linters aren't installed.

Both register as ``Checker`` implementations under ``VerificationGate``;
the gate composes them. ``lint.py``'s ``run_post_write_lint`` is wrapped
as :class:`LintChecker` (Phase 3.5) — opt in via
:meth:`VerificationGate.with_lint` when the caller has a live container.

Design contract
---------------
* **Pure orchestration.** No filesystem reads, no container I/O. The
  caller passes ``EditRecord`` objects with already-known new content.
  Phase 3.2 wires the act loop's per-iteration writes here.
* **Checkers are independent.** Each runs against the edits it
  ``applies_to``. Failures don't short-circuit other checkers — the
  report aggregates everything.
* **Concurrent within a batch.** ``ast.parse`` is CPU-bound; we run all
  applicable (edit × checker) pairs via ``asyncio.to_thread`` + gather
  so a 50-file batch doesn't serialize.
* **Latest-write wins per path.** If the same path appears in multiple
  ``EditRecord`` objects (rare but possible — sequential edits in one
  iteration), only the last is verified. Earlier intermediate states
  may not even be valid syntactically and aren't what landed on disk.
* **Severity is the checker's call.** Parse errors are ``blocking``;
  lint warnings will be ``warning``; metrics-only checks will be
  ``info``. The gate aggregates by severity but doesn't reinterpret it.
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import time
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from augmentum.utils.logging import get_logger

# PyYAML is in the project deps but the in-process gate must degrade
# gracefully if it ever isn't installed (running from a stripped-down
# venv, lint-only deployments, etc.). When unavailable, ``applies_to``
# returns False permanently — the checker is invisible rather than
# crash-on-load. lint.py's subprocess YAML check still covers the file
# in that case via its own graceful-skip path.
try:
    import yaml as _yaml  # noqa: F401
    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when PyYAML missing
    _yaml = None
    _YAML_AVAILABLE = False

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

Severity = Literal["blocking", "warning", "info"]


@dataclass(frozen=True, slots=True)
class EditRecord:
    """One write the gate should verify.

    ``path`` is workspace-relative or absolute — the gate doesn't read
    from it; it's identification only. ``new_content`` is the post-write
    content as the caller knows it (typically captured at the moment
    the write tool returned). ``language`` is an optional hint; if
    empty, the gate infers from the extension.
    """
    path: str
    tool: str
    new_content: str
    language: str = ""


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One checker's verdict on one edit."""
    checker: str
    target: str
    passed: bool
    severity: Severity
    message: str
    details: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Aggregated outcome of running every applicable (checker × edit) pair.

    ``passed`` is True iff no ``blocking`` checker failed. Warnings and
    info results never flip ``passed`` — they're for the trace and
    optional surfacing only.
    """
    passed: bool
    blocking_failures: tuple[CheckResult, ...] = ()
    warnings: tuple[CheckResult, ...] = ()
    successes: tuple[CheckResult, ...] = ()
    duration_ms: float = 0.0

    def model_facing_summary(self) -> str:
        """Short string the wiring layer can hand to the model on failure.

        Returns the empty string when there are no blocking failures —
        callers can treat empty as "nothing to report" without a
        separate ``passed`` check.
        """
        if not self.blocking_failures:
            return ""
        lines = [f"Verification failed ({len(self.blocking_failures)} blocking issue(s)):"]
        for r in self.blocking_failures:
            lines.append(f"  - [{r.checker}] {r.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checker protocol + built-ins
# ---------------------------------------------------------------------------


@runtime_checkable
class Checker(Protocol):
    """Interface every gate checker implements.

    ``applies_to`` is a fast classifier (extension/language match);
    ``check`` is the actual work. ``severity`` is part of the type so
    the gate can aggregate without invoking checkers.
    """
    name: str
    severity: Severity

    def applies_to(self, edit: EditRecord) -> bool: ...
    async def check(self, edit: EditRecord) -> CheckResult: ...


def _detect_language(edit: EditRecord) -> str:
    """Resolve the effective language for an edit: explicit hint wins,
    else extension lookup. Lowercased for stable matching."""
    if edit.language:
        return edit.language.strip().lower()
    _, ext = os.path.splitext(edit.path)
    return ext.lstrip(".").lower()


def _short_path(path: str, *, max_len: int = 64) -> str:
    """Trim long paths in messages so the model-facing summary stays compact."""
    if len(path) <= max_len:
        return path
    return "…" + path[-(max_len - 1):]


class PythonParseChecker:
    """Stdlib ``ast.parse`` syntax check.

    Catches the most damaging class of broken edits: an unparseable
    file. ``SyntaxError`` carries ``lineno`` and ``offset`` so the
    model-facing message can be specific. Empty content is treated as
    a pass (a deletion-like write should not parse-fail).
    """
    name: str = "python_parse"
    severity: Severity = "blocking"

    def applies_to(self, edit: EditRecord) -> bool:
        return _detect_language(edit) in {"python", "py"}

    async def check(self, edit: EditRecord) -> CheckResult:
        if not edit.new_content.strip():
            return CheckResult(
                checker=self.name, target=edit.path, passed=True,
                severity=self.severity,
                message=f"empty file {_short_path(edit.path)}",
                details={"reason": "empty"},
            )

        def _parse() -> CheckResult:
            try:
                ast.parse(edit.new_content, filename=edit.path)
            except SyntaxError as exc:
                lineno = int(exc.lineno or 0)
                offset = int(exc.offset or 0)
                # ``msg`` is short (e.g. "invalid syntax"); compose a
                # location-anchored message the model can act on.
                message = (
                    f"Syntax error in {_short_path(edit.path)}"
                    f":{lineno}: {exc.msg or 'invalid syntax'}"
                )
                return CheckResult(
                    checker=self.name, target=edit.path, passed=False,
                    severity=self.severity,
                    message=message,
                    details={
                        "lineno": lineno, "offset": offset,
                        "error_type": type(exc).__name__,
                        "raw_msg": exc.msg or "",
                    },
                )
            return CheckResult(
                checker=self.name, target=edit.path, passed=True,
                severity=self.severity,
                message=f"parses ok ({_short_path(edit.path)})",
            )

        return await asyncio.to_thread(_parse)


class JsonParseChecker:
    """``json.loads`` validity check.

    JSON is brittle (one trailing comma, one unquoted key, dead). The
    error has a deterministic line+col so the model gets actionable
    feedback. JSONC (with comments) intentionally fails — it's not
    valid JSON and tooling that accepts it should validate separately.
    """
    name: str = "json_parse"
    severity: Severity = "blocking"

    def applies_to(self, edit: EditRecord) -> bool:
        return _detect_language(edit) == "json"

    async def check(self, edit: EditRecord) -> CheckResult:
        if not edit.new_content.strip():
            return CheckResult(
                checker=self.name, target=edit.path, passed=True,
                severity=self.severity,
                message=f"empty file {_short_path(edit.path)}",
                details={"reason": "empty"},
            )

        def _parse() -> CheckResult:
            try:
                json.loads(edit.new_content)
            except json.JSONDecodeError as exc:
                message = (
                    f"JSON error in {_short_path(edit.path)}"
                    f":{exc.lineno}: {exc.msg}"
                )
                return CheckResult(
                    checker=self.name, target=edit.path, passed=False,
                    severity=self.severity,
                    message=message,
                    details={
                        "lineno": int(exc.lineno),
                        "colno": int(exc.colno),
                        "pos": int(exc.pos),
                        "raw_msg": exc.msg,
                    },
                )
            return CheckResult(
                checker=self.name, target=edit.path, passed=True,
                severity=self.severity,
                message=f"parses ok ({_short_path(edit.path)})",
            )

        return await asyncio.to_thread(_parse)


class YamlParseChecker:
    """``yaml.safe_load`` validity check.

    YAMLError carries ``problem_mark`` with 0-indexed ``line`` and
    ``column`` for the failing token, plus a short ``problem``
    description. We surface a 1-indexed line number for human / model
    consumption (matches Python + JSON conventions in this module).

    Common YAML failures this catches: mismatched flow brackets,
    bad indentation in block style, illegal characters in keys,
    duplicate keys (configurable but ``safe_load`` is permissive there).

    PyYAML's ``safe_load`` returns ``None`` for empty / whitespace-only
    documents and that's a valid pass — same semantics as the empty
    Python / JSON path.
    """
    name: str = "yaml_parse"
    severity: Severity = "blocking"

    def applies_to(self, edit: EditRecord) -> bool:
        if not _YAML_AVAILABLE:
            return False
        return _detect_language(edit) in {"yaml", "yml"}

    async def check(self, edit: EditRecord) -> CheckResult:
        if not edit.new_content.strip():
            return CheckResult(
                checker=self.name, target=edit.path, passed=True,
                severity=self.severity,
                message=f"empty file {_short_path(edit.path)}",
                details={"reason": "empty"},
            )

        def _parse() -> CheckResult:
            try:
                _yaml.safe_load(edit.new_content)
            except _yaml.YAMLError as exc:
                # Marked errors give us line + col; unmarked errors
                # (rare — usually constructor exceptions) fall through
                # with whatever ``str(exc)`` says.
                mark = getattr(exc, "problem_mark", None)
                lineno = (mark.line + 1) if mark else 0
                colno = (mark.column + 1) if mark else 0
                problem = (
                    getattr(exc, "problem", None)
                    or str(exc).splitlines()[0]
                )
                # Build a compact location-anchored message. Avoid the
                # full multi-line YAMLError __str__ (it includes the
                # context block which is useful for humans but noisy
                # for the model context budget).
                if lineno:
                    message = (
                        f"YAML error in {_short_path(edit.path)}"
                        f":{lineno}: {problem}"
                    )
                else:
                    message = (
                        f"YAML error in {_short_path(edit.path)}: {problem}"
                    )
                return CheckResult(
                    checker=self.name, target=edit.path, passed=False,
                    severity=self.severity,
                    message=message,
                    details={
                        "lineno": lineno, "colno": colno,
                        "problem": problem,
                        "error_type": type(exc).__name__,
                    },
                )
            return CheckResult(
                checker=self.name, target=edit.path, passed=True,
                severity=self.severity,
                message=f"parses ok ({_short_path(edit.path)})",
            )

        return await asyncio.to_thread(_parse)


# Match TOMLDecodeError's positional template: "Invalid value (at line 2,
# column 7)". Some errors omit the location (e.g. "Unterminated string
# (at end of document)") — those land as lineno=0 in details.
_TOML_POSITION_RE = re.compile(
    r"\(at line (\d+), column (\d+)\)", re.IGNORECASE,
)


class TomlParseChecker:
    """stdlib ``tomllib.loads`` validity check (Python 3.11+).

    TOMLDecodeError flattens to a single message — no structured
    line/col attributes. The 3.11 stdlib does embed
    ``(at line N, column M)`` in the error string for positional
    errors, which we extract via regex for the ``details`` block.
    Errors without a position (rare; unterminated EOF cases) keep
    ``lineno=0`` and rely on the message text alone.

    Common TOML failures this catches: unterminated strings, bare
    keys with spaces, malformed inline tables, duplicate table
    headers.
    """
    name: str = "toml_parse"
    severity: Severity = "blocking"

    def applies_to(self, edit: EditRecord) -> bool:
        return _detect_language(edit) == "toml"

    async def check(self, edit: EditRecord) -> CheckResult:
        if not edit.new_content.strip():
            return CheckResult(
                checker=self.name, target=edit.path, passed=True,
                severity=self.severity,
                message=f"empty file {_short_path(edit.path)}",
                details={"reason": "empty"},
            )

        def _parse() -> CheckResult:
            try:
                tomllib.loads(edit.new_content)
            except tomllib.TOMLDecodeError as exc:
                raw = str(exc)
                m = _TOML_POSITION_RE.search(raw)
                lineno = int(m.group(1)) if m else 0
                colno = int(m.group(2)) if m else 0
                if lineno:
                    message = (
                        f"TOML error in {_short_path(edit.path)}"
                        f":{lineno}: {raw}"
                    )
                else:
                    message = f"TOML error in {_short_path(edit.path)}: {raw}"
                return CheckResult(
                    checker=self.name, target=edit.path, passed=False,
                    severity=self.severity,
                    message=message,
                    details={
                        "lineno": lineno, "colno": colno,
                        "raw_msg": raw,
                        "error_type": type(exc).__name__,
                    },
                )
            return CheckResult(
                checker=self.name, target=edit.path, passed=True,
                severity=self.severity,
                message=f"parses ok ({_short_path(edit.path)})",
            )

        return await asyncio.to_thread(_parse)


# ---------------------------------------------------------------------------
# LintChecker — wraps the existing in-container lint pipeline (Phase 3.5).
# ---------------------------------------------------------------------------


class LintChecker:
    """Adapter for ``augmentum.coder.lint.run_post_write_lint``.

    Wraps the in-container subprocess lint pipeline (ruff / eslint /
    py_compile / etc.) as a ``Checker`` so it composes with the in-process
    parse checkers under one ``VerificationGate``. The model-facing
    summary, persistence ledger, and UI trace all become consistent
    across "fast in-process" checks and "slow in-container" checks.

    Severity = ``warning`` (not blocking)
    -------------------------------------
    ``PythonParseChecker`` / ``JsonParseChecker`` / ``YamlParseChecker`` /
    ``TomlParseChecker`` already catch unparseable files in-process and
    are blocking. What lint adds on top is ruff-style findings, eslint
    output, and ``node --check`` for JS — useful but not show-stopping.
    Treating them as ``warning`` keeps the trust signal proportional:
    the model isn't told "verification failed" for a missing-import nag.

    A dedicated ``JsParseChecker`` (severity ``blocking``) wrapping just
    ``node --check`` would be the natural follow-up; deferred to keep
    this commit focused.

    Constructor params
    ------------------
    Lint runs in a specific workspace container, so the checker is
    workspace-scoped (one instance per workspace). The factory
    :meth:`VerificationGate.with_lint` is the easiest way to compose.
    """
    name: str = "lint"
    severity: Severity = "warning"

    def __init__(
        self,
        container_manager,
        workspace_id: str,
        *,
        timeout: float = 8.0,
        max_chars: int = 1500,
    ):
        self._container_manager = container_manager
        self._workspace_id = workspace_id
        self._timeout = timeout
        self._max_chars = max_chars

    def applies_to(self, edit: EditRecord) -> bool:
        # Local import — lint.py is a sibling module and we want to
        # avoid pulling subprocess machinery into ``verify.py``'s
        # import-time graph (verify is supposed to be stdlib-only).
        from augmentum.coder.lint import _commands_for_path
        return bool(_commands_for_path(edit.path))

    async def check(self, edit: EditRecord) -> CheckResult:
        from augmentum.coder.lint import run_post_write_lint

        try:
            output = await run_post_write_lint(
                self._container_manager,
                self._workspace_id,
                edit.path,
                timeout=self._timeout,
                max_chars=self._max_chars,
            )
        except Exception as exc:
            # Container down / mid-restart / shell glitch — degrade to
            # "lint skipped" rather than a fake warning. The in-process
            # checkers already validated parseability; lint adds polish,
            # not safety, so a runtime miss here is acceptable.
            log.debug(
                "lint_checker_runtime_error", path=edit.path, exc_info=True,
            )
            return CheckResult(
                checker=self.name, target=edit.path, passed=True,
                severity=self.severity,
                message=f"lint skipped ({_short_path(edit.path)})",
                details={
                    "reason": "runtime_error",
                    "error":  str(exc)[:200],
                },
            )

        if output is None:
            # Clean run OR all candidate binaries missing — either way
            # there's nothing to surface.
            return CheckResult(
                checker=self.name, target=edit.path, passed=True,
                severity=self.severity,
                message=f"lint clean ({_short_path(edit.path)})",
            )

        # ``run_post_write_lint`` returns "\n\n[name]\nfindings...". We
        # keep the bracketed name marker in the message so the model
        # can tell which linter spoke; just trim the leading newlines
        # so ``model_facing_summary``'s "  - [lint] {message}" doesn't
        # render as "  - [lint] \n\n[ruff]\n...".
        cleaned = output.strip()
        return CheckResult(
            checker=self.name, target=edit.path, passed=False,
            severity=self.severity,
            message=cleaned,
            details={"raw_output": output},
        )


# ---------------------------------------------------------------------------
# Gate orchestrator
# ---------------------------------------------------------------------------


class VerificationGate:
    """Composes Checkers, runs applicable ones, aggregates results.

    Construct with explicit checkers when you want a tight gate (e.g.
    Reflex tier might run only ``PythonParseChecker``). Use
    :meth:`default` to get the full built-in set.
    """

    def __init__(self, checkers: Sequence[Checker]):
        self._checkers: tuple[Checker, ...] = tuple(checkers)

    @classmethod
    def default(cls) -> VerificationGate:
        """Built-in checker set: Python + JSON + YAML + TOML.

        All in-process, no container dependencies. PyYAML is the only
        third-party dep; ``YamlParseChecker.applies_to`` returns False
        when it's missing so the gate degrades silently to the other
        three checkers (lint.py's subprocess YAML hook still covers
        the file in that fallback).

        Note for callers using ``checker_names`` to pin a contract:
        order is stable but the set may grow as Phase 3.4 / 3.6 add
        more checkers (test runner, ScopeChecker). Lint is opt-in via
        :meth:`with_lint` since it depends on a live container.
        """
        return cls([
            PythonParseChecker(),
            JsonParseChecker(),
            YamlParseChecker(),
            TomlParseChecker(),
        ])

    @classmethod
    def with_lint(
        cls,
        container_manager,
        workspace_id: str,
        *,
        timeout: float = 8.0,
        max_chars: int = 1500,
    ) -> VerificationGate:
        """Default in-process gate + the in-container :class:`LintChecker`.

        Use this when the caller has a live container available and
        wants the unified report shape across both check sources. The
        existing ``_maybe_run_post_write_verify`` hook (Phase 3.2) keeps
        using ``default()`` for the in-process-only path; callers that
        want lint included opt in by switching to this factory.

        ``timeout`` / ``max_chars`` are forwarded to
        ``run_post_write_lint`` and bound the subprocess hop so a slow
        linter never makes the gate feel sticky.
        """
        return cls([
            PythonParseChecker(),
            JsonParseChecker(),
            YamlParseChecker(),
            TomlParseChecker(),
            LintChecker(
                container_manager, workspace_id,
                timeout=timeout, max_chars=max_chars,
            ),
        ])

    @property
    def checker_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self._checkers)

    async def verify_writes(
        self, edits: Iterable[EditRecord],
    ) -> VerificationReport:
        """Run every applicable (edit × checker) pair concurrently.

        Latest-write-wins-per-path: if multiple ``EditRecord`` objects
        target the same path, only the last one is verified. The
        caller's intermediate states are gone from disk by then anyway.
        """
        deduped = self._dedupe_latest(edits)
        if not deduped:
            return VerificationReport(passed=True, duration_ms=0.0)

        start = time.monotonic()

        # Build the work list. ``applies_to`` is sync + cheap, so we
        # filter here rather than letting checkers no-op themselves.
        # That keeps logs/traces scoped to actually-relevant work.
        tasks: list[asyncio.Task[CheckResult]] = []
        for edit in deduped:
            for checker in self._checkers:
                try:
                    if not checker.applies_to(edit):
                        continue
                except Exception:
                    # A misbehaving applies_to shouldn't tank the gate —
                    # log and skip. Checkers run untrusted-ish (third-
                    # party Powers can register some in the future).
                    log.warning(
                        "checker_applies_to_raised",
                        checker=checker.name, path=edit.path,
                        exc_info=True,
                    )
                    continue
                tasks.append(asyncio.create_task(self._run_one(checker, edit)))

        results = await asyncio.gather(*tasks) if tasks else []

        blocking_failures: list[CheckResult] = []
        warnings: list[CheckResult] = []
        successes: list[CheckResult] = []
        for r in results:
            if r.passed:
                successes.append(r)
            elif r.severity == "blocking":
                blocking_failures.append(r)
            else:
                warnings.append(r)

        duration_ms = (time.monotonic() - start) * 1000.0
        report = VerificationReport(
            passed=len(blocking_failures) == 0,
            blocking_failures=tuple(blocking_failures),
            warnings=tuple(warnings),
            successes=tuple(successes),
            duration_ms=duration_ms,
        )

        # Single structured log per gate run — gives ops visibility
        # without N log lines for N files. Only log INFO when something
        # blocked; otherwise debug (the happy path is the common case).
        log_fn = log.info if blocking_failures else log.debug
        log_fn(
            "verification_gate_completed",
            edits=len(deduped),
            checks_run=len(results),
            blocking=len(blocking_failures),
            warnings=len(warnings),
            duration_ms=round(duration_ms, 2),
        )
        return report

    async def _run_one(self, checker: Checker, edit: EditRecord) -> CheckResult:
        """Wrap an individual check so a checker exception becomes a
        ``warning`` result rather than a gate-killing ``raise``.

        Rationale: one buggy checker shouldn't cost us the verification
        of the other 49 files in the batch. The wiring layer can still
        decide to escalate based on the warning count.
        """
        try:
            return await checker.check(edit)
        except Exception as exc:
            log.warning(
                "checker_raised",
                checker=checker.name, path=edit.path, error=str(exc),
                exc_info=True,
            )
            return CheckResult(
                checker=checker.name, target=edit.path, passed=False,
                severity="warning",
                message=f"{checker.name} crashed on {_short_path(edit.path)}: {exc}",
                details={"error_type": type(exc).__name__, "error": str(exc)},
            )

    @staticmethod
    def _dedupe_latest(edits: Iterable[EditRecord]) -> list[EditRecord]:
        """Keep only the last ``EditRecord`` per path, preserving order
        of first occurrence. ``dict`` ordering is insertion-stable in
        Python 3.7+, so the natural pattern works."""
        latest: dict[str, EditRecord] = {}
        for edit in edits:
            latest[edit.path] = edit
        return list(latest.values())
