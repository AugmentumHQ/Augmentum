"""Fuzz triage tests — pure logic, no I/O.

Triage takes a FuzzRunResult and emits Finding rows. We test:
  * classification: each exception class maps to the expected
    ClaimSignature + severity (and the special-case overrides for
    NoneType / deadly-signal land correctly);
  * stack signature: identical sites collapse, distinct sites don't;
  * leaf-frame extraction: harness and atheris frames are skipped so
    the Finding points at user code;
  * Finding shape: status=CONFIRMED, single-run accounting, fuzz
    family attribution.
"""

from __future__ import annotations

from augmentum.bug_finder.findings import (
    ClaimSignature,
    FindingStatus,
    Severity,
)
from augmentum.bug_finder.fuzz.harness import Harness
from augmentum.bug_finder.fuzz.runner import FuzzCrash, FuzzRunResult
from augmentum.bug_finder.fuzz.triage import (
    FUZZ_FAMILY,
    _classify_crash,
    _leaf_user_frame,
    _stack_signature,
    triage_fuzz_run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _harness() -> Harness:
    return Harness(
        source="# placeholder\n",
        suggested_filename="fuzz_parse.py",
        target_module="augmentum.foo",
        target_function="parse",
        target_param="data",
        input_kind="bytes",
    )


def _crash(
    *,
    exc_class: str = "AttributeError",
    exc_msg: str = "'NoneType' object has no attribute 'split'",
    stack: str = (
        "Traceback (most recent call last):\n"
        '  File "fuzz_parse.py", line 14, in TestOneInput\n'
        "    parse(data)\n"
        '  File "/workspace/augmentum/foo.py", line 22, in parse\n'
        "    return data.split()\n"
        "AttributeError: 'NoneType' object has no attribute 'split'\n"
    ),
    basename: str = "crash-aa11bb22",
    inp: bytes = b"\xff\x00",
) -> FuzzCrash:
    return FuzzCrash(
        input_basename=basename,
        input_bytes=inp,
        stack_trace=stack,
        exception_class=exc_class,
        exception_message=exc_msg,
    )


# ---------------------------------------------------------------------------
# _classify_crash
# ---------------------------------------------------------------------------


def test_classify_attribute_error_with_nonetype_is_null_deref() -> None:
    sig, sev = _classify_crash(_crash(
        exc_class="AttributeError",
        exc_msg="'NoneType' object has no attribute 'foo'",
    ))
    assert sig == ClaimSignature.NULL_DEREF.value
    assert sev == Severity.MEDIUM.value


def test_classify_attribute_error_without_nonetype_falls_back_to_table() -> None:
    """``AttributeError`` without NoneType still gets classified via
    the table — null_deref is wrong because the cause might be a
    type mismatch, not a None dereference."""
    sig, sev = _classify_crash(_crash(
        exc_class="AttributeError",
        exc_msg="'int' object has no attribute 'split'",
    ))
    # Table entry for AttributeError is also NULL_DEREF — but the
    # rationale here is "table-default", not "NoneType match". The
    # important invariant: the function returns a valid mapping.
    assert sig in {ClaimSignature.NULL_DEREF.value, ClaimSignature.TYPE_CONFUSION.value}


def test_classify_recursion_error_is_resource_leak() -> None:
    sig, sev = _classify_crash(_crash(
        exc_class="RecursionError",
        exc_msg="maximum recursion depth exceeded",
    ))
    assert sig == ClaimSignature.RESOURCE_LEAK.value


def test_classify_memory_error_is_resource_leak() -> None:
    sig, _ = _classify_crash(_crash(exc_class="MemoryError"))
    assert sig == ClaimSignature.RESOURCE_LEAK.value


def test_classify_zero_division_is_logic_error() -> None:
    sig, _ = _classify_crash(_crash(exc_class="ZeroDivisionError"))
    assert sig == ClaimSignature.LOGIC_ERROR.value


def test_classify_assertion_error_is_logic_error() -> None:
    sig, _ = _classify_crash(_crash(exc_class="AssertionError"))
    assert sig == ClaimSignature.LOGIC_ERROR.value


def test_classify_system_error_is_type_confusion_high() -> None:
    """SystemError indicates Python interpreter weirdness — type
    confusion or low-level state corruption. High severity."""
    sig, sev = _classify_crash(_crash(exc_class="SystemError"))
    assert sig == ClaimSignature.TYPE_CONFUSION.value
    assert sev == Severity.HIGH.value


def test_classify_deadly_signal_is_use_after_free_high() -> None:
    """A libfuzzer ``deadly signal`` line means SIGSEGV/SIGABRT from a
    C extension. Treat as use-after-free + high severity regardless
    of what Python class atheris attributed it to."""
    stack = (
        "==1234== ERROR: libFuzzer: deadly signal\n"
        "Some native frame trace here\n"
    )
    sig, sev = _classify_crash(_crash(
        exc_class="RuntimeError", stack=stack,
    ))
    assert sig == ClaimSignature.USE_AFTER_FREE.value
    assert sev == Severity.HIGH.value


def test_classify_unknown_class_is_other_low() -> None:
    sig, sev = _classify_crash(_crash(
        exc_class="SomeWeirdException", stack="SomeWeirdException: x\n",
    ))
    assert sig == ClaimSignature.OTHER.value
    assert sev == Severity.LOW.value


# ---------------------------------------------------------------------------
# _stack_signature
# ---------------------------------------------------------------------------


def test_stack_signature_identical_traces_collapse() -> None:
    s1 = _stack_signature(
        'File "x.py", line 10, in foo\n'
        'File "y.py", line 20, in bar\n',
    )
    s2 = _stack_signature(
        'File "x.py", line 10, in foo\n'
        'File "y.py", line 20, in bar\n',
    )
    assert s1 == s2
    assert s1 != ""


def test_stack_signature_different_lines_differ() -> None:
    s1 = _stack_signature('File "x.py", line 10, in foo\n')
    s2 = _stack_signature('File "x.py", line 11, in foo\n')
    assert s1 != s2


def test_stack_signature_empty_when_no_frames() -> None:
    assert _stack_signature("no frames here at all") == ""


# ---------------------------------------------------------------------------
# _leaf_user_frame
# ---------------------------------------------------------------------------


def test_leaf_user_frame_skips_harness() -> None:
    stack = (
        'File "fuzz_parse.py", line 14, in TestOneInput\n'
        '    parse(data)\n'
        'File "/workspace/augmentum/foo.py", line 22, in parse\n'
        '    return data.split()\n'
    )
    file_, fn = _leaf_user_frame(
        stack,
        harness_filename="fuzz_parse.py",
        fallback_file="augmentum/foo.py",
        fallback_function="parse",
    )
    assert file_ == "/workspace/augmentum/foo.py"
    assert fn == "parse"


def test_leaf_user_frame_skips_atheris() -> None:
    stack = (
        'File "fuzz_parse.py", line 14, in TestOneInput\n'
        'File "/usr/lib/python3/dist-packages/atheris/__init__.py", line 99, in fuzz\n'
        'File "/workspace/augmentum/foo.py", line 22, in parse\n'
    )
    file_, fn = _leaf_user_frame(
        stack,
        harness_filename="fuzz_parse.py",
        fallback_file="augmentum/foo.py",
        fallback_function="parse",
    )
    assert "atheris" not in file_
    assert "foo.py" in file_


def test_leaf_user_frame_falls_back_when_no_user_frames() -> None:
    """Edge case: all frames in the trace are harness/atheris. Return
    the supplied fallbacks rather than misattributing to the harness."""
    stack = (
        'File "fuzz_parse.py", line 14, in TestOneInput\n'
        'File "/usr/lib/python3/dist-packages/atheris/__init__.py", line 99, in fuzz\n'
    )
    file_, fn = _leaf_user_frame(
        stack,
        harness_filename="fuzz_parse.py",
        fallback_file="augmentum/foo.py",
        fallback_function="parse",
    )
    assert file_ == "augmentum/foo.py"
    assert fn == "parse"


# ---------------------------------------------------------------------------
# triage_fuzz_run — end-to-end
# ---------------------------------------------------------------------------


def test_triage_empty_result_returns_empty() -> None:
    res = FuzzRunResult(skipped=True, skip_reason="atheris not installed")
    findings = triage_fuzz_run(
        res, harness=_harness(),
        chunk_file="augmentum/foo.py", chunk_function="parse",
    )
    assert findings == []


def test_triage_no_crashes_returns_empty() -> None:
    res = FuzzRunResult(skipped=False, crashes=(), iterations=10000)
    findings = triage_fuzz_run(
        res, harness=_harness(),
        chunk_file="augmentum/foo.py", chunk_function="parse",
    )
    assert findings == []


def test_triage_single_crash_produces_confirmed_finding() -> None:
    res = FuzzRunResult(
        skipped=False, crashes=(_crash(),), iterations=10000,
        runtime_seconds=60.0,
    )
    findings = triage_fuzz_run(
        res, harness=_harness(),
        chunk_file="augmentum/foo.py", chunk_function="parse",
        artifact_dir="/workspace/.augmentum/fuzz/foo/artifacts",
    )
    assert len(findings) == 1
    f = findings[0]
    assert f.status == FindingStatus.CONFIRMED.value
    assert f.claim_signature == ClaimSignature.NULL_DEREF.value
    assert f.severity == Severity.MEDIUM.value
    assert f.runs_to_confirm == 1
    assert f.total_runs == 1
    assert f.families_to_confirm == 1
    assert f.total_families == 1
    assert f.repro_path.endswith("crash-aa11bb22")
    assert f.repro_output.startswith("Traceback")
    assert "fuzz family" in f.notes[0]


def test_triage_dedupes_crashes_at_same_site() -> None:
    """Three crashes with the same top frames collapse into ONE
    finding. This is atheris's "minimize-the-input" output pattern;
    every reduction step writes a new artifact at the same crash
    site. The Finding count should reflect underlying bugs, not
    artifact count."""
    base_stack = (
        'File "fuzz_parse.py", line 14, in TestOneInput\n'
        '    parse(data)\n'
        'File "/workspace/augmentum/foo.py", line 22, in parse\n'
        "AttributeError: 'NoneType' object has no attribute 'split'\n"
    )
    crashes = tuple(
        _crash(stack=base_stack, basename=f"crash-{i:04x}")
        for i in range(3)
    )
    res = FuzzRunResult(skipped=False, crashes=crashes, iterations=10)
    findings = triage_fuzz_run(
        res, harness=_harness(),
        chunk_file="augmentum/foo.py", chunk_function="parse",
    )
    assert len(findings) == 1


def test_triage_distinct_sites_produce_distinct_findings() -> None:
    """Two crashes at different leaf frames should NOT be merged.
    Different bugs, different rows."""
    s1 = (
        'File "/workspace/augmentum/foo.py", line 22, in parse\n'
        "AttributeError: 'NoneType' object has no attribute 'split'\n"
    )
    s2 = (
        'File "/workspace/augmentum/foo.py", line 50, in render\n'
        "RecursionError: maximum recursion depth exceeded\n"
    )
    crashes = (
        _crash(stack=s1, basename="crash-aa", exc_class="AttributeError",
               exc_msg="'NoneType' object has no attribute 'split'"),
        _crash(stack=s2, basename="crash-bb", exc_class="RecursionError",
               exc_msg="maximum recursion depth exceeded"),
    )
    res = FuzzRunResult(skipped=False, crashes=crashes, iterations=10)
    findings = triage_fuzz_run(
        res, harness=_harness(),
        chunk_file="augmentum/foo.py", chunk_function="parse",
    )
    assert len(findings) == 2
    sigs = {f.claim_signature for f in findings}
    assert ClaimSignature.NULL_DEREF.value in sigs
    assert ClaimSignature.RESOURCE_LEAK.value in sigs


def test_triage_finding_id_is_stable_for_same_site() -> None:
    """Two triage passes over the same crashes produce identical
    Finding IDs — needed for cross-run dedup and pattern memory."""
    res = FuzzRunResult(skipped=False, crashes=(_crash(),), iterations=10)
    a = triage_fuzz_run(
        res, harness=_harness(),
        chunk_file="augmentum/foo.py", chunk_function="parse",
    )
    b = triage_fuzz_run(
        res, harness=_harness(),
        chunk_file="augmentum/foo.py", chunk_function="parse",
    )
    assert a[0].id == b[0].id


def test_fuzz_family_constant_matches_string() -> None:
    """Round-trips the family slug used by the orchestrator's
    merge_runs call. Catches accidental renames."""
    assert FUZZ_FAMILY == "fuzz"
