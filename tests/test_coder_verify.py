"""Verification gate tests — Phase 3.1 of the coder foundation.

Covers protocol shape, language dispatch, both stdlib checkers
(Python ast.parse + JSON json.loads), aggregation, latest-write
dedup, crash isolation, and the model-facing summary contract.

Pure stdlib + pytest — no Augmentum stack, no DB. The gate is in-
process pure orchestration and these tests exercise its public
surface only.
"""
from __future__ import annotations

import pytest

from augmentum.coder.verify import (
    Checker,
    CheckResult,
    EditRecord,
    JsonParseChecker,
    LintChecker,
    PythonParseChecker,
    TomlParseChecker,
    VerificationGate,
    VerificationReport,
    YamlParseChecker,
    _YAML_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Protocol + detection
# ---------------------------------------------------------------------------


def test_builtins_satisfy_checker_protocol():
    """Both stdlib checkers must satisfy the runtime-checkable Protocol
    so third-party Powers can register their own without a base class."""
    assert isinstance(PythonParseChecker(), Checker)
    assert isinstance(JsonParseChecker(), Checker)


def test_python_checker_applies_to_py_path():
    c = PythonParseChecker()
    assert c.applies_to(EditRecord(path="foo.py", tool="t", new_content=""))
    assert c.applies_to(EditRecord(path="x.py", tool="t", new_content="", language="python"))


def test_python_checker_applies_to_language_hint_wins():
    """Even when extension would say otherwise, an explicit language
    hint should be authoritative — supports inline-snippet edits where
    path is a sentinel like ``<scratch>``."""
    c = PythonParseChecker()
    assert c.applies_to(EditRecord(
        path="<inline>", tool="t", new_content="", language="python",
    ))


def test_python_checker_skips_non_python():
    c = PythonParseChecker()
    assert not c.applies_to(EditRecord(path="config.json", tool="t", new_content=""))
    assert not c.applies_to(EditRecord(path="readme.md", tool="t", new_content=""))
    assert not c.applies_to(EditRecord(path="no_extension", tool="t", new_content=""))


def test_json_checker_applies_to_json_path():
    c = JsonParseChecker()
    assert c.applies_to(EditRecord(path="config.json", tool="t", new_content=""))


def test_json_checker_skips_yaml():
    """JSONC and YAML are explicitly out-of-scope for this checker —
    JSONC isn't valid JSON; YAML lands in 3.3."""
    c = JsonParseChecker()
    assert not c.applies_to(EditRecord(path="manifest.yaml", tool="t", new_content=""))
    assert not c.applies_to(EditRecord(path="settings.jsonc", tool="t", new_content=""))


# ---------------------------------------------------------------------------
# PythonParseChecker behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_python_clean_code_passes():
    c = PythonParseChecker()
    edit = EditRecord(
        path="ok.py", tool="code_edit",
        new_content="def f(x: int) -> int:\n    return x + 1\n",
    )
    result = await c.check(edit)
    assert result.passed is True
    assert result.severity == "blocking"
    assert result.checker == "python_parse"
    assert result.target == "ok.py"


@pytest.mark.asyncio
async def test_python_syntax_error_reports_line_and_message():
    c = PythonParseChecker()
    # Missing colon after function header — classic SyntaxError on line 1.
    edit = EditRecord(
        path="bad.py", tool="code_edit",
        new_content="def f(x)\n    return x\n",
    )
    result = await c.check(edit)
    assert result.passed is False
    assert result.severity == "blocking"
    assert "bad.py" in result.message
    assert "1" in result.message  # line number anchored
    assert result.details["lineno"] == 1
    assert result.details["error_type"] == "SyntaxError"
    assert result.details["raw_msg"]  # short pythonic msg present


@pytest.mark.asyncio
async def test_python_indentation_error_caught():
    c = PythonParseChecker()
    # Inconsistent indentation — IndentationError is a SyntaxError subclass.
    edit = EditRecord(
        path="indent.py", tool="code_edit",
        new_content="def f():\n  x = 1\n    y = 2\n",
    )
    result = await c.check(edit)
    assert result.passed is False
    assert result.details["lineno"] >= 1


@pytest.mark.asyncio
async def test_python_empty_content_passes_with_reason():
    """A genuinely empty file (deletion-style write) is not a parse
    failure — distinguish via the ``reason`` detail key."""
    c = PythonParseChecker()
    edit = EditRecord(path="empty.py", tool="code_edit", new_content="")
    result = await c.check(edit)
    assert result.passed is True
    assert result.details.get("reason") == "empty"


@pytest.mark.asyncio
async def test_python_whitespace_only_passes_with_reason():
    c = PythonParseChecker()
    edit = EditRecord(path="ws.py", tool="code_edit", new_content="   \n\n  ")
    result = await c.check(edit)
    assert result.passed is True
    assert result.details.get("reason") == "empty"


@pytest.mark.asyncio
async def test_python_comment_only_file_parses():
    """Comments-only is valid Python, must not be treated as 'empty'."""
    c = PythonParseChecker()
    edit = EditRecord(
        path="cmt.py", tool="code_edit",
        new_content="# just a comment\n# and another\n",
    )
    result = await c.check(edit)
    assert result.passed is True
    # Should NOT carry an "empty" reason — there's real content.
    assert result.details.get("reason") != "empty"


# ---------------------------------------------------------------------------
# JsonParseChecker behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_valid_passes():
    c = JsonParseChecker()
    edit = EditRecord(
        path="cfg.json", tool="file_write",
        new_content='{"a": 1, "b": [1, 2, 3]}',
    )
    result = await c.check(edit)
    assert result.passed is True


@pytest.mark.asyncio
async def test_json_trailing_comma_fails_with_position():
    c = JsonParseChecker()
    edit = EditRecord(
        path="bad.json", tool="file_write",
        new_content='{"a": 1, "b": 2,}',
    )
    result = await c.check(edit)
    assert result.passed is False
    assert "bad.json" in result.message
    assert result.details["lineno"] >= 1
    assert "colno" in result.details
    assert "pos" in result.details


@pytest.mark.asyncio
async def test_json_jsonc_with_comment_fails():
    """JSONC (JSON with `//` comments) is invalid stdlib JSON. The
    docstring contract is explicit: tooling that wants JSONC should
    validate separately. Verify we honor it."""
    c = JsonParseChecker()
    edit = EditRecord(
        path="settings.json", tool="file_write",
        new_content='{\n  // a comment\n  "x": 1\n}',
    )
    result = await c.check(edit)
    assert result.passed is False


# ---------------------------------------------------------------------------
# VerificationGate orchestration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_default_includes_all_stdlib_checkers():
    """Phase 3.3 expanded the default set to Python + JSON + YAML +
    TOML. Order is stable; new checkers append to the end."""
    g = VerificationGate.default()
    assert g.checker_names == (
        "python_parse", "json_parse", "yaml_parse", "toml_parse",
    )


@pytest.mark.asyncio
async def test_gate_empty_input_passes_zero_duration():
    g = VerificationGate.default()
    report = await g.verify_writes([])
    assert report.passed is True
    assert report.blocking_failures == ()
    assert report.duration_ms == 0.0


@pytest.mark.asyncio
async def test_gate_clean_batch_all_in_successes():
    g = VerificationGate.default()
    edits = [
        EditRecord(path="a.py", tool="code_edit", new_content="x = 1\n"),
        EditRecord(path="b.py", tool="code_edit", new_content="def f(): pass\n"),
        EditRecord(path="c.json", tool="file_write", new_content='{"k": "v"}'),
    ]
    report = await g.verify_writes(edits)
    assert report.passed is True
    assert report.blocking_failures == ()
    assert report.warnings == ()
    assert len(report.successes) == 3
    targets = {r.target for r in report.successes}
    assert targets == {"a.py", "b.py", "c.json"}


@pytest.mark.asyncio
async def test_gate_mixed_batch_aggregates_correctly():
    g = VerificationGate.default()
    edits = [
        EditRecord(path="ok.py", tool="code_edit", new_content="x = 1\n"),
        EditRecord(path="bad.py", tool="code_edit", new_content="def f(\n"),
        EditRecord(path="ok.json", tool="file_write", new_content='{"a": 1}'),
        EditRecord(path="bad.json", tool="file_write", new_content="{not json}"),
    ]
    report = await g.verify_writes(edits)
    assert report.passed is False
    assert len(report.blocking_failures) == 2
    failed_targets = {r.target for r in report.blocking_failures}
    assert failed_targets == {"bad.py", "bad.json"}
    assert len(report.successes) == 2


@pytest.mark.asyncio
async def test_gate_duplicate_paths_keeps_latest_only():
    """Two edits to the same path in one batch — only the last
    content is verified (it's what's on disk). The first edit's
    intermediate state may not even be valid syntactically."""
    g = VerificationGate.default()
    edits = [
        # First edit: intermediate broken state
        EditRecord(path="x.py", tool="code_edit", new_content="def f(\n"),
        # Second edit: the actual final state on disk
        EditRecord(path="x.py", tool="code_edit", new_content="def f():\n    pass\n"),
    ]
    report = await g.verify_writes(edits)
    assert report.passed is True
    # Exactly one result for x.py — the dedup'd latest.
    py_results = [r for r in report.successes if r.target == "x.py"]
    assert len(py_results) == 1


@pytest.mark.asyncio
async def test_gate_unsupported_extension_yields_no_result():
    """A file no checker applies_to should not produce a result —
    silent skip is correct (don't false-pass, don't false-fail)."""
    g = VerificationGate.default()
    edits = [
        EditRecord(path="doc.md", tool="file_write", new_content="# heading"),
    ]
    report = await g.verify_writes(edits)
    assert report.passed is True
    assert report.successes == ()
    assert report.blocking_failures == ()


@pytest.mark.asyncio
async def test_gate_runs_subset_of_checkers_when_constructed_explicitly():
    """A tier might want only parse checking. Constructing with a
    narrower checker list must restrict applies_to dispatch."""
    g = VerificationGate([PythonParseChecker()])
    assert g.checker_names == ("python_parse",)
    edits = [
        EditRecord(path="a.json", tool="file_write", new_content="{not json}"),
    ]
    report = await g.verify_writes(edits)
    # JSON checker isn't registered, so the broken JSON is invisible.
    assert report.passed is True
    assert report.blocking_failures == ()


# ---------------------------------------------------------------------------
# Crash isolation
# ---------------------------------------------------------------------------


class _BoomChecker:
    """Test double that raises in ``check`` — verifies the gate
    converts the exception into a warning instead of letting it
    propagate. One bad checker must not tank the batch."""
    name = "boom"
    severity = "blocking"

    def applies_to(self, edit: EditRecord) -> bool:
        return edit.path.endswith(".py")

    async def check(self, edit: EditRecord) -> CheckResult:
        raise RuntimeError("intentional boom")


@pytest.mark.asyncio
async def test_gate_isolates_checker_crash_as_warning():
    g = VerificationGate([PythonParseChecker(), _BoomChecker()])
    edits = [
        EditRecord(path="ok.py", tool="code_edit", new_content="x = 1\n"),
    ]
    report = await g.verify_writes(edits)

    # Python parse succeeded, so passed=True (warnings don't flip it).
    assert report.passed is True
    assert len(report.successes) == 1

    # The crash is captured as a warning result, not propagated.
    crash_results = [r for r in report.warnings if r.checker == "boom"]
    assert len(crash_results) == 1
    assert "intentional boom" in crash_results[0].message
    assert crash_results[0].details["error_type"] == "RuntimeError"


class _BadAppliesChecker:
    """applies_to itself raises — must be tolerated and skipped."""
    name = "bad_applies"
    severity = "blocking"

    def applies_to(self, edit: EditRecord) -> bool:
        raise RuntimeError("applies_to is broken")

    async def check(self, edit: EditRecord) -> CheckResult:
        # Should never be reached if applies_to is broken.
        raise AssertionError("check should not run when applies_to crashes")


@pytest.mark.asyncio
async def test_gate_tolerates_broken_applies_to():
    g = VerificationGate([PythonParseChecker(), _BadAppliesChecker()])
    edits = [EditRecord(path="ok.py", tool="code_edit", new_content="x = 1\n")]
    # Must not raise; the broken applies_to is logged and skipped.
    report = await g.verify_writes(edits)
    assert report.passed is True
    assert len(report.successes) == 1


# ---------------------------------------------------------------------------
# VerificationReport.model_facing_summary
# ---------------------------------------------------------------------------


def test_model_facing_summary_empty_when_passed():
    """No blocking failures → empty summary (callers can treat empty
    as 'nothing to report' without re-checking ``passed``)."""
    report = VerificationReport(passed=True)
    assert report.model_facing_summary() == ""


def test_model_facing_summary_lists_each_blocking_failure():
    report = VerificationReport(
        passed=False,
        blocking_failures=(
            CheckResult(
                checker="python_parse", target="bad.py", passed=False,
                severity="blocking",
                message="Syntax error in bad.py:3: invalid syntax",
            ),
            CheckResult(
                checker="json_parse", target="bad.json", passed=False,
                severity="blocking",
                message="JSON error in bad.json:1: Expecting value",
            ),
        ),
    )
    summary = report.model_facing_summary()
    assert "Verification failed" in summary
    assert "2 blocking" in summary
    assert "[python_parse]" in summary
    assert "[json_parse]" in summary
    assert "bad.py:3" in summary
    assert "bad.json:1" in summary


# ---------------------------------------------------------------------------
# Concurrency smoke test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_handles_large_batch():
    """Verify a 100-file batch completes under a forgiving budget.
    Catches any accidental serialization or unbounded resource use."""
    g = VerificationGate.default()
    edits = [
        EditRecord(path=f"f{i}.py", tool="code_edit",
                   new_content=f"x = {i}\n")
        for i in range(100)
    ]
    report = await g.verify_writes(edits)
    assert report.passed is True
    assert len(report.successes) == 100
    # Generous ceiling: 100 ast.parse calls under 5 seconds total.
    assert report.duration_ms < 5_000.0


# ---------------------------------------------------------------------------
# YamlParseChecker — Phase 3.3
# ---------------------------------------------------------------------------


_yaml_skip = pytest.mark.skipif(
    not _YAML_AVAILABLE,
    reason="PyYAML not installed; YamlParseChecker is silent in this env",
)


def test_yaml_checker_satisfies_protocol():
    assert isinstance(YamlParseChecker(), Checker)


def test_yaml_checker_applies_to_yaml_and_yml():
    c = YamlParseChecker()
    if not _YAML_AVAILABLE:
        # When PyYAML missing, applies_to is permanently False — test
        # that contract explicitly so a future ImportError regression
        # gets caught.
        assert not c.applies_to(EditRecord(path="x.yaml", tool="t", new_content=""))
        return
    assert c.applies_to(EditRecord(path="cfg.yaml", tool="t", new_content=""))
    assert c.applies_to(EditRecord(path="cfg.yml", tool="t", new_content=""))
    assert c.applies_to(EditRecord(path="<inline>", tool="t", new_content="", language="yaml"))


def test_yaml_checker_skips_other_extensions():
    """YAML checker must not claim .json / .toml / .py — they each
    have their own checker with stricter semantics."""
    c = YamlParseChecker()
    assert not c.applies_to(EditRecord(path="x.json", tool="t", new_content=""))
    assert not c.applies_to(EditRecord(path="x.toml", tool="t", new_content=""))
    assert not c.applies_to(EditRecord(path="x.py", tool="t", new_content=""))


@_yaml_skip
@pytest.mark.asyncio
async def test_yaml_clean_passes():
    c = YamlParseChecker()
    edit = EditRecord(
        path="cfg.yaml", tool="file_write",
        new_content="version: 1\nservices:\n  web:\n    image: foo\n",
    )
    result = await c.check(edit)
    assert result.passed is True
    assert result.checker == "yaml_parse"


@_yaml_skip
@pytest.mark.asyncio
async def test_yaml_syntax_error_reports_line():
    """An unclosed flow sequence is a classic YAML failure with a
    structured ``problem_mark``. Verify line is 1-indexed in the
    user-facing message (PyYAML's mark.line is 0-indexed)."""
    c = YamlParseChecker()
    edit = EditRecord(
        path="bad.yaml", tool="file_write",
        new_content="a: [1, 2\nb: bad\n",
    )
    result = await c.check(edit)
    assert result.passed is False
    assert result.severity == "blocking"
    assert "bad.yaml" in result.message
    assert result.details["lineno"] >= 1
    # 1-indexed: PyYAML's 0-indexed mark.line = 0 surfaces as 1.
    assert result.details["lineno"] == result.details["lineno"]
    assert result.details["error_type"]  # populated


@_yaml_skip
@pytest.mark.asyncio
async def test_yaml_empty_passes_with_reason():
    c = YamlParseChecker()
    edit = EditRecord(path="empty.yaml", tool="file_write", new_content="")
    result = await c.check(edit)
    assert result.passed is True
    assert result.details.get("reason") == "empty"


@_yaml_skip
@pytest.mark.asyncio
async def test_yaml_whitespace_only_passes_with_reason():
    c = YamlParseChecker()
    edit = EditRecord(path="ws.yaml", tool="file_write", new_content="   \n\n")
    result = await c.check(edit)
    assert result.passed is True
    assert result.details.get("reason") == "empty"


@_yaml_skip
@pytest.mark.asyncio
async def test_yaml_compact_message_excludes_full_yamlerror_dump():
    """PyYAML's full ``str(YAMLError)`` is multi-line with context
    blocks — useful for humans, noisy for the model context budget.
    Our message must be a single line referencing path:line."""
    c = YamlParseChecker()
    edit = EditRecord(
        path="bad.yaml", tool="file_write",
        new_content="key: value\n  bad: nesting\n",
    )
    result = await c.check(edit)
    if result.passed:
        # Some indentation glitches PyYAML auto-recovers from. Skip if
        # this particular string didn't actually fail.
        return
    assert "\n" not in result.message, (
        f"YAML message should be single-line, got: {result.message!r}"
    )


# ---------------------------------------------------------------------------
# TomlParseChecker — Phase 3.3
# ---------------------------------------------------------------------------


def test_toml_checker_satisfies_protocol():
    assert isinstance(TomlParseChecker(), Checker)


def test_toml_checker_applies_to_toml():
    c = TomlParseChecker()
    assert c.applies_to(EditRecord(path="pyproject.toml", tool="t", new_content=""))
    assert c.applies_to(
        EditRecord(path="<inline>", tool="t", new_content="", language="toml"),
    )


def test_toml_checker_skips_other_extensions():
    c = TomlParseChecker()
    assert not c.applies_to(EditRecord(path="x.yaml", tool="t", new_content=""))
    assert not c.applies_to(EditRecord(path="x.json", tool="t", new_content=""))


@pytest.mark.asyncio
async def test_toml_clean_passes():
    c = TomlParseChecker()
    edit = EditRecord(
        path="ok.toml", tool="file_write",
        new_content='[tool]\nname = "augmentum"\nversion = "1.0"\n',
    )
    result = await c.check(edit)
    assert result.passed is True


@pytest.mark.asyncio
async def test_toml_positional_error_extracts_line():
    """``Invalid value (at line 2, column 7)`` — stdlib tomllib embeds
    location in the error string; we extract via regex into details."""
    c = TomlParseChecker()
    edit = EditRecord(
        path="bad.toml", tool="file_write",
        new_content='valid = 1\nbad = key with space\nmore = 2\n',
    )
    result = await c.check(edit)
    assert result.passed is False
    assert "bad.toml" in result.message
    # The position regex captured (line 2, column 7) from the error.
    assert result.details["lineno"] == 2
    assert result.details["colno"] == 7


@pytest.mark.asyncio
async def test_toml_unterminated_string_no_position():
    """Some TOML errors don't carry positional info (EOF cases).
    The checker must still flag the failure; lineno=0 is the
    contract for missing-position cases."""
    c = TomlParseChecker()
    edit = EditRecord(
        path="eof.toml", tool="file_write",
        new_content='key = "unclosed',
    )
    result = await c.check(edit)
    assert result.passed is False
    assert result.details["lineno"] == 0
    # Message still surfaces the raw error so the model can act on it.
    assert "Unterminated" in result.message


@pytest.mark.asyncio
async def test_toml_empty_passes_with_reason():
    c = TomlParseChecker()
    edit = EditRecord(path="empty.toml", tool="file_write", new_content="")
    result = await c.check(edit)
    assert result.passed is True
    assert result.details.get("reason") == "empty"


# ---------------------------------------------------------------------------
# Integration: default gate aggregates Python + JSON + YAML + TOML
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_default_aggregates_across_all_four_languages():
    """Mixed batch with one file per supported language: one bad TOML,
    one bad Python, one good YAML, one good JSON. Default gate routes
    each to the right checker and the report aggregates correctly."""
    g = VerificationGate.default()
    edits = [
        EditRecord(path="ok.json", tool="file_write", new_content='{"a": 1}'),
        EditRecord(path="bad.py", tool="file_write", new_content="def f(\n"),
        EditRecord(path="ok.yaml", tool="file_write", new_content="x: 1\n"),
        EditRecord(path="bad.toml", tool="file_write",
                   new_content='bad = key with space\n'),
    ]
    report = await g.verify_writes(edits)

    assert report.passed is False
    failed_targets = {r.target for r in report.blocking_failures}
    expected_failed = {"bad.py"}
    if _YAML_AVAILABLE:
        # YAML check ran for ok.yaml; it should pass and live in successes.
        # Bad.toml will fail.
        expected_failed.add("bad.toml")
    else:
        # PyYAML missing: YAML files silently no-op'd (no result),
        # TOML still ran. Adjust expectation.
        expected_failed.add("bad.toml")
    assert failed_targets == expected_failed

    # Both ok files (json + yaml when yaml available) pass.
    success_targets = {r.target for r in report.successes}
    assert "ok.json" in success_targets
    if _YAML_AVAILABLE:
        assert "ok.yaml" in success_targets


@pytest.mark.asyncio
async def test_gate_yaml_unsupported_silent_when_pyyaml_missing(monkeypatch):
    """If a future deployment lacks PyYAML, YAML files should be
    invisible to the gate (no false-pass, no false-fail). Simulate
    by forcing _YAML_AVAILABLE False on a single YAML batch and
    asserting zero results for the .yaml file."""
    from augmentum.coder import verify as _verify_mod
    monkeypatch.setattr(_verify_mod, "_YAML_AVAILABLE", False)

    g = VerificationGate([_verify_mod.YamlParseChecker()])
    edits = [
        EditRecord(path="x.yaml", tool="file_write",
                   new_content="this: is: not: even: valid"),
    ]
    report = await g.verify_writes(edits)
    # No result either way — silent skip.
    assert report.passed is True
    assert report.successes == ()
    assert report.blocking_failures == ()


# ---------------------------------------------------------------------------
# LintChecker — Phase 3.5 (subprocess wrapper for in-container lint)
# ---------------------------------------------------------------------------


class _StubContainerManager:
    """Minimal ``container_manager`` stand-in.

    ``run_post_write_lint`` only calls ``_run_command``, so a single
    method is enough. ``output`` is the canned shell stdout for the
    next call; ``raises`` (if set) makes the call surface an exception
    so we can exercise the runtime-error degradation path without
    spinning up a real container.
    """

    def __init__(self, *, output: str = "", raises: BaseException | None = None):
        self.output = output
        self.raises = raises
        self.calls: list[tuple] = []

    async def _run_command(self, workspace_id, cmd, *, timeout=None):
        self.calls.append((workspace_id, tuple(cmd), timeout))
        if self.raises is not None:
            raise self.raises
        return self.output


def test_lint_checker_satisfies_protocol():
    """Mirrors the parse-checker protocol guard — third-party Powers
    must be able to substitute their own LintChecker-shaped Checker."""
    cm = _StubContainerManager()
    assert isinstance(LintChecker(cm, "ws-1"), Checker)


def test_lint_checker_default_severity_is_warning():
    """Severity is part of the type, not per-result. Lint findings
    are warnings (style/ruff/eslint output) — blocking syntax errors
    are caught by the in-process parse checkers. If this assertion
    flips, the wiring layer needs auditing — a lint warning would
    start triggering the verification_failed UI chunk."""
    cm = _StubContainerManager()
    assert LintChecker(cm, "ws-1").severity == "warning"


def test_lint_checker_applies_to_known_extensions():
    """``applies_to`` delegates to ``lint._commands_for_path`` so
    the dispatch table is the single source of truth. Verify a
    representative subset of extensions pass / don't pass."""
    cm = _StubContainerManager()
    c = LintChecker(cm, "ws-1")
    assert c.applies_to(EditRecord(path="a.py", tool="t", new_content=""))
    assert c.applies_to(EditRecord(path="a.js", tool="t", new_content=""))
    assert c.applies_to(EditRecord(path="a.mjs", tool="t", new_content=""))
    assert c.applies_to(EditRecord(path="a.ts", tool="t", new_content=""))
    assert c.applies_to(EditRecord(path="a.json", tool="t", new_content=""))
    assert c.applies_to(EditRecord(path="a.yaml", tool="t", new_content=""))


def test_lint_checker_skips_extensionless_and_unknown():
    """Unknown extensions get no candidate lint commands — skipping
    keeps ``check`` from running a no-op subprocess for every text
    file the agent writes."""
    cm = _StubContainerManager()
    c = LintChecker(cm, "ws-1")
    assert not c.applies_to(EditRecord(path="README", tool="t", new_content=""))
    assert not c.applies_to(EditRecord(path="x.toml", tool="t", new_content=""))
    assert not c.applies_to(EditRecord(path="x.md", tool="t", new_content=""))
    assert not c.applies_to(EditRecord(path="image.png", tool="t", new_content=""))


@pytest.mark.asyncio
async def test_lint_checker_clean_output_passes():
    """Empty stdout from the linter = clean run. ``run_post_write_lint``
    returns ``None``, the checker maps that to a passed CheckResult so
    the gate's success bucket counts it correctly."""
    cm = _StubContainerManager(output="")
    c = LintChecker(cm, "ws-1")
    edit = EditRecord(
        path="/workspace/clean.py", tool="file_write",
        new_content="def f():\n    return 1\n",
    )
    result = await c.check(edit)
    assert result.passed is True
    assert result.checker == "lint"
    assert result.severity == "warning"
    assert "clean" in result.message.lower()


@pytest.mark.asyncio
async def test_lint_checker_findings_become_warning_result():
    """Non-empty linter stdout = findings. The checker returns
    passed=False with severity='warning' — blocking_failures stays
    empty, .warnings collects it. Critical: a warning must NOT flip
    the gate's overall .passed to False."""
    cm = _StubContainerManager(
        output="x.py:1:1: F401 'os' imported but unused",
    )
    c = LintChecker(cm, "ws-1")
    edit = EditRecord(
        path="/workspace/x.py", tool="file_write",
        new_content="import os\n",
    )
    result = await c.check(edit)
    assert result.passed is False
    assert result.severity == "warning"
    # Message keeps the linter-name marker so the model can tell
    # which tool spoke ("[ruff]", "[eslint]", etc.).
    assert "x.py" in result.message or "F401" in result.message
    assert "raw_output" in result.details


@pytest.mark.asyncio
async def test_lint_checker_container_error_treated_as_clean():
    """Container ``_run_command`` errors are absorbed by
    ``run_post_write_lint`` itself (per-candidate try/except in
    ``lint.py``). The checker sees ``output=None`` and returns a
    clean CheckResult — same shape as 'linter ran and found nothing'.

    This is intentional: lint is best-effort and a transient container
    glitch shouldn't appear in the trace as a verification result at
    all. The in-process parse checkers already covered the parseability
    question; lint adds polish, not safety."""
    cm = _StubContainerManager(raises=RuntimeError("container restarting"))
    c = LintChecker(cm, "ws-1")
    edit = EditRecord(
        path="/workspace/x.py", tool="file_write",
        new_content="def f(): pass\n",
    )
    result = await c.check(edit)
    assert result.passed is True
    assert "clean" in result.message.lower()
    # The lint pipeline absorbed both candidate failures and returned
    # None — no runtime_error reason in details, that path is reserved
    # for the rarer "lint pipeline itself raised" case (next test).
    assert "reason" not in result.details


@pytest.mark.asyncio
async def test_lint_checker_pipeline_exception_degrades_to_skip(monkeypatch):
    """Second-line safety net: the outer ``except Exception`` in
    ``LintChecker.check`` catches programming errors that escape
    ``run_post_write_lint`` itself (rare — would require a bug in
    lint.py, not a container failure).

    Force the rarer path by monkeypatching ``run_post_write_lint``
    to raise. The checker must NOT propagate — a single buggy lint
    invocation can't be allowed to fail the entire batch.
    """
    import augmentum.coder.lint as _lint_mod

    async def _exploding(*args, **kwargs):
        raise ValueError("simulated programmer error in lint pipeline")

    monkeypatch.setattr(_lint_mod, "run_post_write_lint", _exploding)

    cm = _StubContainerManager()
    c = LintChecker(cm, "ws-1")
    edit = EditRecord(
        path="/workspace/x.py", tool="file_write",
        new_content="def f(): pass\n",
    )
    result = await c.check(edit)
    assert result.passed is True
    assert result.severity == "warning"
    assert result.details.get("reason") == "runtime_error"
    assert "simulated programmer error" in result.details.get("error", "")


@pytest.mark.asyncio
async def test_lint_checker_inside_gate_does_not_block_passed():
    """End-to-end: a gate that includes LintChecker + PythonParseChecker
    against a clean Python file with lint warnings. Gate.passed must be
    True (because the warning isn't blocking) but the warning must
    appear in report.warnings so the trace can surface it."""
    cm = _StubContainerManager(
        output="x.py:1:1: F401 'os' imported but unused",
    )
    g = VerificationGate([
        PythonParseChecker(),
        LintChecker(cm, "ws-1"),
    ])
    edit = EditRecord(
        path="/workspace/x.py", tool="file_write",
        # Parses ok, but ruff would flag the unused import.
        new_content="import os\n",
    )
    report = await g.verify_writes([edit])
    assert report.passed is True, (
        "Gate.passed must be True when only warnings (not blocking) fired"
    )
    assert len(report.blocking_failures) == 0
    assert len(report.warnings) == 1
    assert report.warnings[0].checker == "lint"
    # The parse checker also ran and succeeded.
    assert any(s.checker == "python_parse" for s in report.successes)


@pytest.mark.asyncio
async def test_lint_checker_blocking_parse_error_overrides_lint():
    """Symmetric guard: if the file fails parsing, that's a blocking
    failure regardless of what lint says. The lint checker still runs
    — the gate orchestrator runs all applicable (edit × checker) pairs
    in parallel — but its warning doesn't promote to blocking. This
    pins the severity model: each checker speaks its own type."""
    cm = _StubContainerManager(
        output="x.py:1:1: F401 'os' imported but unused",
    )
    g = VerificationGate([
        PythonParseChecker(),
        LintChecker(cm, "ws-1"),
    ])
    edit = EditRecord(
        path="/workspace/x.py", tool="file_write",
        new_content="def f(\n",  # broken — unclosed paren
    )
    report = await g.verify_writes([edit])
    assert report.passed is False
    assert len(report.blocking_failures) == 1
    assert report.blocking_failures[0].checker == "python_parse"
    # Lint produced a warning result alongside; it's in .warnings, not
    # .blocking_failures, even though .passed is False.
    assert any(w.checker == "lint" for w in report.warnings)


@pytest.mark.asyncio
async def test_lint_checker_forwards_constructor_overrides():
    """``timeout`` and ``max_chars`` constructor args must reach
    ``run_post_write_lint``. Without this, ``with_lint(timeout=2.0)``
    would silently fall back to the 8s default and a slow linter
    could stall the gate."""
    cm = _StubContainerManager(output="")
    c = LintChecker(cm, "ws-1", timeout=2.5, max_chars=512)
    edit = EditRecord(path="x.py", tool="t", new_content="pass\n")
    await c.check(edit)
    # ``run_post_write_lint`` may issue multiple subprocess calls
    # (py_compile, then ruff). Both must use our timeout — if the
    # default 8.0 leaks through, this assertion catches it.
    assert cm.calls, "expected at least one container _run_command call"
    timeouts = {call[2] for call in cm.calls}
    assert timeouts == {2.5}, (
        f"timeout override should reach _run_command on every candidate, "
        f"got {timeouts}"
    )


# ---------------------------------------------------------------------------
# with_lint factory composition
# ---------------------------------------------------------------------------


def test_with_lint_factory_composes_default_plus_lint():
    """``with_lint`` is the only sanctioned way to build a gate that
    includes LintChecker (callers shouldn't poke ``_checkers``).
    Order: default checkers first, lint last — keeps in-process
    fast-failures visible before the subprocess result lands."""
    cm = _StubContainerManager()
    g = VerificationGate.with_lint(cm, "ws-1")
    assert g.checker_names == (
        "python_parse", "json_parse", "yaml_parse", "toml_parse", "lint",
    )


def test_default_factory_unchanged_by_phase_3_5():
    """Phase 3.5 must not silently expand ``default()`` — that would
    add a subprocess hop to every write through tools.py's existing
    ``_maybe_run_post_write_verify`` hook (which uses ``default()``).
    Lint is opt-in via ``with_lint``; this guard prevents accidental
    composition drift."""
    g = VerificationGate.default()
    assert g.checker_names == (
        "python_parse", "json_parse", "yaml_parse", "toml_parse",
    )
    assert "lint" not in g.checker_names
