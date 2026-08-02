"""Atheris fuzz runner tests.

Live execution requires Docker + Atheris in a real container — those
runs sit under ``tests/live/`` and gate on ``--run-live``. Here we
cover:

  * the parsing helpers (``_extract_exception`` / ``_find_trace_block``),
    because they read libfuzzer/atheris stdout shapes we can simulate
    with literal strings;
  * the install-skip path, by mocking ``ContainerManager`` so the
    "atheris not installed and install failed" branch hits;
  * the happy-path flow, by mocking ``ContainerManager`` to emulate a
    successful run that produces one crash artifact.

The mocked CM mirrors only the subset of methods the runner calls:
``run_command``. Real ``ContainerManager`` orchestrates Docker, which
we deliberately stay out of in unit tests.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from typing import Any

import pytest

from augmentum.bug_finder.fuzz.harness import Harness
from augmentum.bug_finder.fuzz.runner import (
    FuzzCrash,
    FuzzRunResult,
    _DEFAULT_SEEDS,
    _extract_exception,
    _find_trace_block,
    _safe_chunk_id,
    run_fuzz_harness,
)


# ---------------------------------------------------------------------------
# _extract_exception
# ---------------------------------------------------------------------------


def test_extract_exception_strict_match() -> None:
    trace = (
        "Traceback (most recent call last):\n"
        '  File "harness.py", line 12, in TestOneInput\n'
        "    parse(data)\n"
        "ValueError: malformed input\n"
    )
    cls, msg = _extract_exception(trace)
    assert cls == "ValueError"
    assert msg == "malformed input"


def test_extract_exception_qualified_class() -> None:
    trace = "json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)\n"
    cls, msg = _extract_exception(trace)
    assert cls == "json.decoder.JSONDecodeError"
    assert "Expecting value" in msg


def test_extract_exception_loose_fallback_when_no_error_suffix() -> None:
    """atheris occasionally emits its own ``UncaughtException: ...``
    summary lines that don't end in Error/Exception. Loose fallback
    still surfaces the class name and message."""
    trace = "SegFault: address 0x0\n"
    cls, msg = _extract_exception(trace)
    assert cls == "SegFault"
    assert msg == "address 0x0"


def test_extract_exception_unknown_when_no_colon() -> None:
    cls, msg = _extract_exception("some random output without a colon")
    assert cls == "UnknownException"


def test_extract_exception_empty_input() -> None:
    cls, msg = _extract_exception("")
    assert cls == "UnknownException"
    assert msg == ""


# ---------------------------------------------------------------------------
# _find_trace_block
# ---------------------------------------------------------------------------


_FAKE_ATHERIS_OUTPUT = """\
INFO: Running with entropic power schedule (0xFF, 100).
INFO: -max_total_time = 60
#1 INITED cov: 4 ft: 4 corp: 1/1b
=== Uncaught Python exception: ===
Traceback (most recent call last):
  File "harness.py", line 15, in TestOneInput
    parse(data)
  File "/workspace/x.py", line 88, in parse
    return data[0]
IndexError: index out of range

==1234== ERROR: libFuzzer: deadly signal
Test unit written to ./crash-deadbeef
"""


def test_find_trace_block_locates_traceback_before_crash_basename() -> None:
    block = _find_trace_block(_FAKE_ATHERIS_OUTPUT, "crash-deadbeef")
    assert "Traceback (most recent call last)" in block
    assert "IndexError" in block
    # The libfuzzer ==NNN== divider should not be inside the block
    assert "==1234==" not in block


def test_find_trace_block_handles_missing_basename() -> None:
    """Fallback: if we can't anchor on the basename, return the last
    traceback in the output."""
    block = _find_trace_block(_FAKE_ATHERIS_OUTPUT, "crash-not-present")
    assert "Traceback" in block
    assert "IndexError" in block


def test_find_trace_block_returns_empty_when_no_traceback() -> None:
    assert _find_trace_block("nothing here", "crash-x") == ""


# ---------------------------------------------------------------------------
# _safe_chunk_id
# ---------------------------------------------------------------------------


def test_safe_chunk_id_replaces_unsafe_chars() -> None:
    assert _safe_chunk_id("augmentum/foo.py::parse") == "augmentum_foo_py_parse"


def test_safe_chunk_id_caps_length() -> None:
    very_long = "x" * 200
    assert len(_safe_chunk_id(very_long)) == 80


def test_safe_chunk_id_falls_back_when_empty() -> None:
    assert _safe_chunk_id("") == "chunk"
    assert _safe_chunk_id("///!!!") == "chunk"


# ---------------------------------------------------------------------------
# Default seed corpus
# ---------------------------------------------------------------------------


def test_default_seeds_include_common_magic_bytes() -> None:
    seed_names = {name for name, _ in _DEFAULT_SEEDS}
    assert "empty" in seed_names
    assert "pdf-magic" in seed_names
    assert "json-empty" in seed_names


# ---------------------------------------------------------------------------
# Mocked-CM flow tests
# ---------------------------------------------------------------------------


@dataclass
class _RecordedCmd:
    cmd: list[str]
    timeout: float


@dataclass
class _MockContainerManager:
    """Stand-in for ContainerManager that satisfies ``run_command`` only.

    Responses are matched against a list of (predicate, return-value)
    tuples — first predicate that matches wins. Unmatched commands
    return the empty string (which mimics a quiet successful command
    in the real manager).
    """

    responses: list[tuple[Any, Any]] = field(default_factory=list)
    recorded: list[_RecordedCmd] = field(default_factory=list)

    async def run_command(
        self, workspace_id: str, cmd: list[str], timeout: float = 30.0,
        *, idle_timeout: float | None = None,
        progress_path: str | None = None,
        on_chunk: Any | None = None,
    ) -> str:
        self.recorded.append(_RecordedCmd(cmd=list(cmd), timeout=timeout))
        joined = " ".join(cmd)
        for predicate, value in self.responses:
            if callable(predicate) and predicate(joined):
                if callable(value):
                    return value(joined)
                if isinstance(value, Exception):
                    raise value
                return value
        return ""


def _basic_harness() -> Harness:
    return Harness(
        source="# placeholder harness\nprint('hi')\n",
        suggested_filename="fuzz_x.py",
        target_module="augmentum.x",
        target_function="parse",
        target_param="data",
        input_kind="bytes",
    )


@pytest.mark.asyncio
async def test_run_fuzz_harness_skips_when_install_fails() -> None:
    """First import check fails; install command runs (also fails);
    second import check fails. Runner returns skipped with a clear
    reason. Bug_finder pipeline continues."""
    cm = _MockContainerManager(responses=[
        (lambda c: "import atheris" in c, "ModuleNotFoundError"),
        (lambda c: "apt-get update" in c, RuntimeError("apt-get failed")),
    ])
    h = _basic_harness()
    result = await run_fuzz_harness(
        h, cm=cm, workspace_id="ws_test",
        chunk_id="foo.py::parse", max_seconds=5,
    )
    assert result.skipped
    assert "install failed" in result.skip_reason
    assert result.crashes == ()


@pytest.mark.asyncio
async def test_run_fuzz_harness_skips_when_install_completes_but_import_still_fails() -> None:
    """apt-get succeeds, pip install ostensibly succeeds, but the
    import check still fails (broken atheris install). Skip cleanly
    rather than running a harness that will crash at import."""
    call_count = {"check": 0}

    def import_check(_c: str) -> str:
        call_count["check"] += 1
        return "import fails"  # never returns "atheris-ok"

    cm = _MockContainerManager(responses=[
        (lambda c: "import atheris" in c, import_check),
        (lambda c: "apt-get update" in c, "install ok"),
    ])
    h = _basic_harness()
    result = await run_fuzz_harness(
        h, cm=cm, workspace_id="ws_test",
        chunk_id="foo.py::parse", max_seconds=5,
    )
    assert result.skipped
    assert "import still fails" in result.skip_reason
    # The runner re-checks import after install, so we should see two
    # import probes (pre-install + post-install)
    assert call_count["check"] == 2


@pytest.mark.asyncio
async def test_run_fuzz_harness_happy_path_emits_crash() -> None:
    """Simulate a successful atheris run that produced one crash
    artifact. Verify the runner reads the bytes back, parses the
    traceback into ``FuzzCrash``, and returns iteration count."""
    crash_bytes = b"\xff\xfe\x00\x01"
    crash_name = "crash-cafebabe"
    fake_output = (
        "INFO: Running with entropic power schedule.\n"
        "#42 NEW cov: 10 ft: 10 corp: 8/8b\n"
        "=== Uncaught Python exception: ===\n"
        "Traceback (most recent call last):\n"
        '  File "fuzz_x.py", line 14, in TestOneInput\n'
        "    parse(data)\n"
        "  File \"/workspace/augmentum/x.py\", line 22, in parse\n"
        "    return data[0]\n"
        "IndexError: bytes index out of range\n"
        "==1234== ERROR: libFuzzer: deadly signal\n"
        f"Test unit written to ./{crash_name}\n"
    )

    def respond(joined: str) -> str:
        if "import atheris" in joined:
            return "atheris-ok"
        if "rm -rf" in joined or "mkdir -p" in joined and "corpus" in joined:
            return ""
        if "base64 -d" in joined:
            return ""
        if "ls -1" in joined and "crash-" in joined:
            return f"/workspace/.augmentum/fuzz/foo/artifacts/{crash_name}\n"
        if "base64 -w0" in joined:
            return base64.b64encode(crash_bytes).decode("ascii")
        if "python3" in joined and "fuzz_x.py" in joined:
            return fake_output
        return ""

    cm = _MockContainerManager(responses=[(lambda _c: True, respond)])
    h = _basic_harness()
    result = await run_fuzz_harness(
        h, cm=cm, workspace_id="ws_test",
        chunk_id="foo.py::parse", max_seconds=5,
    )
    assert not result.skipped
    assert len(result.crashes) == 1
    crash = result.crashes[0]
    assert crash.input_basename == crash_name
    assert crash.input_bytes == crash_bytes
    assert crash.exception_class == "IndexError"
    assert "bytes index out of range" in crash.exception_message
    assert "Traceback" in crash.stack_trace
    assert result.iterations == 42


@pytest.mark.asyncio
async def test_run_fuzz_harness_no_crashes_returns_clean_result() -> None:
    """A clean atheris run that found nothing should land as
    ``skipped=False, crashes=()`` so the orchestrator can log "fuzz
    leg ran, no crashes" rather than thinking something failed."""
    def respond(joined: str) -> str:
        if "import atheris" in joined:
            return "atheris-ok"
        if "ls -1" in joined and "crash-" in joined:
            return ""
        if "python3" in joined and "fuzz_x.py" in joined:
            return (
                "INFO: -max_total_time = 5\n"
                "#100 INITED cov: 4 ft: 4 corp: 1/1b\n"
                "#10000 DONE   cov: 10 ft: 10 corp: 8/8b\n"
                "Done 10000 runs in 5 second(s)\n"
            )
        return ""

    cm = _MockContainerManager(responses=[(lambda _c: True, respond)])
    result = await run_fuzz_harness(
        _basic_harness(), cm=cm, workspace_id="ws_test",
        chunk_id="foo.py::parse", max_seconds=5,
    )
    assert not result.skipped
    assert result.crashes == ()
    assert result.iterations == 100  # last NEW/INITED match wins


@pytest.mark.asyncio
async def test_run_fuzz_harness_seeds_corpus_with_defaults() -> None:
    """Verify the seed corpus is written before the harness runs.
    A run that never sees seeds is brittle: the first generation is
    pure-random and shallow bugs take longer to surface."""
    cm = _MockContainerManager(responses=[
        (lambda c: "import atheris" in c, "atheris-ok"),
        (lambda c: "python3" in c and "fuzz_x.py" in c, "no crashes\n"),
        (lambda c: "ls -1" in c, ""),
    ])
    await run_fuzz_harness(
        _basic_harness(), cm=cm, workspace_id="ws_test",
        chunk_id="foo.py::parse", max_seconds=5,
    )
    # Count base64-decode commands targeting the corpus dir
    seed_writes = sum(
        1 for r in cm.recorded
        if "base64 -d" in " ".join(r.cmd) and "/corpus/" in " ".join(r.cmd)
    )
    assert seed_writes == len(_DEFAULT_SEEDS)


def test_fuzz_run_result_has_crashes_property() -> None:
    empty = FuzzRunResult()
    assert not empty.has_crashes
    with_crash = FuzzRunResult(crashes=(
        FuzzCrash(
            input_basename="crash-x", input_bytes=b"x",
            stack_trace="", exception_class="ValueError",
            exception_message="bad",
        ),
    ))
    assert with_crash.has_crashes
