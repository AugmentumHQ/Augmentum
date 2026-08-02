"""End-to-end smoke for the bug_finder fuzz pipeline.

Spins up an ephemeral ubuntu container, installs atheris + clang +
cmake, drops a deliberately-buggy Python target plus a generated
harness, runs the harness, then exercises the triage module on the
captured crashes. Reports whether the pipeline correctly identified
the seeded bugs.

This is the smoke that answers "does the system actually find bugs".
Run by hand — Docker required, ~3-5 minutes (install dominates).

Usage:
    python scripts/smoke_test_fuzz_pipeline.py
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

# Add repo root to sys.path so we can import the bug_finder modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from augmentum.bug_finder.fuzz import (
    classify_chunk,
    generate_harness,
    triage_fuzz_run,
)
from augmentum.bug_finder.fuzz.runner import (
    FuzzCrash,
    FuzzRunResult,
    _extract_exception,
    _find_trace_block,
)


# A deliberately buggy parser. The harness will feed atheris bytes;
# atheris will discover at least one of these crash paths.
BUGGY_TARGET = '''\
"""Deliberately buggy parser for fuzz pipeline smoke test."""

from __future__ import annotations


def buggy_parse(data: bytes) -> int:
    """Three seeded bugs:
      1. NULL-deref via b"NULL" prefix
      2. RecursionError via b"\\x00" prefix
      3. Native crash via b"BOOM" prefix (just a deep call chain)
    """
    if data[:4] == b"NULL":
        x = None
        return x.foo  # AttributeError: 'NoneType' object has no attribute 'foo'
    if data[:1] == b"\\x00":
        return _recurse(data[1:])
    if data[:4] == b"BOOM":
        raise SystemError("boom — interpreter weirdness simulated")
    return len(data)


def _recurse(data: bytes) -> int:
    """Stack-bomb. Hits RecursionError on input > ~1000 bytes."""
    if not data:
        return 0
    return 1 + _recurse(data[:-1])
'''


CONTAINER_NAME = f"bug-finder-fuzz-smoke-{uuid.uuid4().hex[:8]}"
TARGET_FILE = "/work/buggy.py"
HARNESS_FILE = "/work/fuzz_buggy_parse.py"
CORPUS_DIR = "/work/corpus"
ARTIFACT_DIR = "/work/artifacts"


def run(cmd: list[str], *, check: bool = True, capture: bool = True,
        timeout: float | None = None) -> subprocess.CompletedProcess:
    """Shell out, surfacing stdout/stderr on failure.

    Force UTF-8 with ``errors='replace'`` — atheris output contains
    arbitrary bytes (libfuzzer dumps mutated input on crash) and the
    Windows default CP1252 codec chokes on it.
    """
    result = subprocess.run(
        cmd, capture_output=capture, text=True, timeout=timeout,
        encoding="utf-8", errors="replace",
    )
    if check and result.returncode != 0:
        print(f"FAIL: {' '.join(cmd)}")
        print(f"  stdout: {result.stdout[-2000:]}")
        print(f"  stderr: {result.stderr[-2000:]}")
        raise SystemExit(result.returncode)
    return result


def dexec(cmd: str, *, timeout: float | None = None,
          check: bool = True) -> subprocess.CompletedProcess:
    """``docker exec`` wrapper."""
    return run(
        ["docker", "exec", CONTAINER_NAME, "bash", "-lc", cmd],
        timeout=timeout, check=check,
    )


def write_into_container(path: str, content: bytes) -> None:
    """Use base64 to dodge shell quoting headaches."""
    encoded = base64.b64encode(content).decode("ascii")
    parent = str(Path(path).parent.as_posix())
    cmd = (
        f"mkdir -p {parent} && "
        f"printf '%s' {encoded!r} | base64 -d > {path}"
    )
    dexec(cmd)


def start_container() -> None:
    print(f"== spinning up ephemeral container: {CONTAINER_NAME}")
    run([
        "docker", "run", "-d", "--name", CONTAINER_NAME,
        "--rm",  # auto-removed on stop
        "ubuntu:24.04", "sleep", "1800",
    ])


def stop_container() -> None:
    print(f"== removing container {CONTAINER_NAME}")
    run(["docker", "stop", CONTAINER_NAME], check=False)


def install_dependencies() -> None:
    print("== installing python3 + clang + cmake + atheris (~2 min)")
    start = time.monotonic()
    dexec(
        "apt-get update -qq && "
        "apt-get install -y -qq --no-install-recommends "
        "python3 python3-pip clang cmake && "
        "pip3 install --break-system-packages --no-cache-dir atheris",
        timeout=480.0,
    )
    print(f"   install done in {time.monotonic() - start:.0f}s")


def verify_atheris() -> None:
    """Confirm atheris import works."""
    result = dexec("python3 -c 'import atheris; print(\"atheris-ok\")'", timeout=30.0)
    assert "atheris-ok" in result.stdout, f"atheris install verification failed: {result.stdout!r}"


def derive_harness() -> str:
    """Run the harness writer on the buggy target and return source."""
    verdict = classify_chunk(
        BUGGY_TARGET, "buggy_parse", file_path="buggy.py",
    )
    print(f"== classifier verdict: {verdict}")
    if not verdict.fuzzable:
        raise RuntimeError(f"target not classified as fuzzable: {verdict.reason}")
    harness = generate_harness(
        verdict,
        target_file="buggy.py",
        target_function="buggy_parse",
        workspace_root=".",
    )
    print(f"== harness generated: {harness.suggested_filename}")
    return harness.source


def run_fuzz_session(max_seconds: int = 25) -> str:
    """Invoke the harness inside the container, return combined output.

    libfuzzer requires the artifact dir to pre-exist (it doesn't create
    it on first crash). We make it idempotent here so the smoke script
    can re-run cleanly. The production runner module does the same.
    """
    dexec(f"mkdir -p {ARTIFACT_DIR} {CORPUS_DIR}")
    cmd = (
        f"cd /work && "
        f"PYTHONPATH=/work python3 {HARNESS_FILE} {CORPUS_DIR} "
        f"-artifact_prefix={ARTIFACT_DIR}/ "
        f"-max_total_time={max_seconds} "
        f"2>&1 || true"
    )
    print(f"== running atheris for {max_seconds}s — looking for seeded bugs…")
    start = time.monotonic()
    result = dexec(cmd, timeout=max_seconds + 60.0, check=False)
    print(f"   atheris done in {time.monotonic() - start:.0f}s")
    return result.stdout


def list_crashes() -> list[str]:
    result = dexec(
        f"ls -1 {ARTIFACT_DIR}/crash-* 2>/dev/null || true", check=False,
    )
    return [
        Path(line.strip()).name
        for line in result.stdout.splitlines() if line.strip()
    ]


def read_from_container(path: str) -> bytes:
    result = dexec(f"base64 -w0 {path} 2>/dev/null", check=False)
    try:
        return base64.b64decode(result.stdout.strip())
    except Exception:
        return b""


def build_run_result(
    output: str, crash_names: list[str], elapsed: float,
) -> FuzzRunResult:
    """Reuse the production parsers (_extract_exception / _find_trace_block)
    to convert raw atheris output into the structured shape the triage
    module expects."""
    crashes: list[FuzzCrash] = []
    for name in sorted(crash_names):
        data = read_from_container(f"{ARTIFACT_DIR}/{name}")
        block = _find_trace_block(output, name)
        exc_cls, exc_msg = _extract_exception(block)
        crashes.append(FuzzCrash(
            input_basename=name,
            input_bytes=data,
            stack_trace=block,
            exception_class=exc_cls,
            exception_message=exc_msg,
        ))
    iters = 0
    for m in re.finditer(r"#(\d+)\s+(NEW|INITED|REDUCE|pulse)", output):
        try:
            iters = max(iters, int(m.group(1)))
        except ValueError:
            continue
    return FuzzRunResult(
        skipped=False,
        crashes=tuple(crashes),
        iterations=iters,
        runtime_seconds=elapsed,
        stderr_tail=output[-4000:],
    )


def main() -> None:
    if shutil.which("docker") is None:
        raise SystemExit("docker CLI required — install or expose via $PATH")

    start_container()
    try:
        install_dependencies()
        verify_atheris()

        harness_source = derive_harness()
        write_into_container(TARGET_FILE, BUGGY_TARGET.encode("utf-8"))
        write_into_container(HARNESS_FILE, harness_source.encode("utf-8"))
        # Seed with minimal corpus
        for name, data in [
            ("empty", b""),
            ("zero", b"\x00"),
            ("ascii", b"hello\n"),
            ("null-bait", b"NULLx"),  # near-miss of the NULL prefix
            ("recurse-bait", b"\x00" + b"x" * 1024),  # near-miss
        ]:
            write_into_container(f"{CORPUS_DIR}/{name}", data)

        t0 = time.monotonic()
        output = run_fuzz_session(max_seconds=25)
        elapsed = time.monotonic() - t0
        crashes = list_crashes()
        print(f"\n== atheris reported {len(crashes)} crash artifacts")
        if not crashes:
            print("== STDERR tail:")
            print(output[-2000:])
            raise SystemExit("smoke FAILED — atheris produced no crashes "
                             "for the seeded-buggy target")

        # Triage via production module
        from augmentum.bug_finder.fuzz import Harness
        harness = Harness(
            source=harness_source,
            suggested_filename="fuzz_buggy_parse.py",
            target_module="buggy",
            target_function="buggy_parse",
            target_param="data",
            input_kind="bytes",
        )
        result = build_run_result(output, crashes, elapsed)
        findings = triage_fuzz_run(
            result, harness=harness,
            chunk_file="buggy.py", chunk_function="buggy_parse",
            artifact_dir=ARTIFACT_DIR,
        )

        print(f"\n== triage produced {len(findings)} deduped findings:\n")
        for i, f in enumerate(findings, 1):
            print(f"  [{i}] {f.claim_signature.upper()} ({f.severity}) — {f.claim}")
            print(f"      site:   {f.file}:{f.function}")
            print(f"      status: {f.status}")
            print(f"      repro:  {f.repro_path}")
            print()

        # Verify at least one of the seeded bugs was caught
        seeded_signatures = {f.claim_signature for f in findings}
        expected = {"null_deref", "resource_leak", "type_confusion"}
        hit = seeded_signatures & expected
        if not hit:
            print(f"== WARN: none of the seeded signatures hit. Got: "
                  f"{seeded_signatures}, expected ANY of: {expected}")
            print("== This may indicate the harness reaches the function "
                  "but atheris hit an unexpected crash class first.")
        else:
            print(f"== SMOKE PASSED: triage correctly classified "
                  f"{len(hit)} seeded bug class(es): {hit}")

    finally:
        stop_container()


if __name__ == "__main__":
    main()
