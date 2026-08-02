"""End-to-end fuzz smoke against actual Augmentum source.

Picks a handful of the 53 classifier-flagged free-function fuzz targets,
copies just the files they depend on into an ephemeral ubuntu
container (with a minimal ``augmentum.utils.logging`` stub since most
targets import it), and runs the full classify → harness → fuzz →
triage pipeline against each one.

Honest expectations: most targets won't crash in 25s — they're real
code, not contrived bugs. The interesting signal is which targets DO
produce crashes, and whether the triage labelling matches what a
human would call a real bug. A "no crashes" result is also evidence
the pipeline ran correctly.

Run by hand:
    python scripts/smoke_test_fuzz_augmentum.py
"""

from __future__ import annotations

import base64
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from augmentum.bug_finder.fuzz import (  # noqa: E402
    Harness,
    classify_chunk,
    generate_harness,
    triage_fuzz_run,
)
from augmentum.bug_finder.fuzz.runner import (  # noqa: E402
    FuzzCrash,
    FuzzRunResult,
    _extract_exception,
    _find_trace_block,
)


CONTAINER_NAME = f"bug-finder-fuzz-augmentum-{uuid.uuid4().hex[:8]}"
REPO_ROOT = Path(__file__).resolve().parent.parent

# A minimal stub of ``augmentum.utils.logging`` so target modules can
# import it without dragging in structlog and the full Augmentum
# package surface. The real ``get_logger`` returns a structlog
# BoundLogger; for fuzz we just need *any* object whose ``.info``,
# ``.warning``, ``.debug`` etc. accept arbitrary kwargs without
# blowing up.
STUB_LOGGING = '''\
"""No-op stub for fuzz smoke. The real module is structlog-backed."""
class _NoopLogger:
    def __getattr__(self, name):
        def f(*a, **kw): return None
        return f
def get_logger(*a, **kw): return _NoopLogger()
'''


@dataclass(frozen=True)
class Target:
    """One real Augmentum function to fuzz."""

    file: str           # repo-relative source path
    function: str       # function name (leaf — module-level)
    extra_deps: tuple[str, ...] = ()  # additional pip packages to install


# Picked from the 53 classifier-flagged fuzz targets. Bias toward
# targets with minimal extra deps so the smoke doesn't get blocked on
# scientific stacks (numpy, pypdfium2, etc.).
TARGETS: tuple[Target, ...] = (
    # Zero non-stdlib deps — cleanest test cases.
    Target("augmentum/coder/preview_types.py", "_json_pretty"),
    Target("augmentum/coder/preview_types.py", "_text_wrap"),
    Target("augmentum/coder/preview_types.py", "_passthrough"),
    # Needs only the stub augmentum.utils.logging.
    Target("augmentum/documents/chunker.py", "extract_text"),
    # CSV parsing — bytes through csv.reader. Classic fuzz target.
    Target("augmentum/knowledge/importer.py", "_extract_csv"),
    Target("augmentum/knowledge/importer.py", "_extract_json"),
)


def run(cmd: list[str], *, check: bool = True, capture: bool = True,
        timeout: float | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd, capture_output=capture, text=True, timeout=timeout,
        encoding="utf-8", errors="replace",
    )
    if check and result.returncode != 0:
        print(f"FAIL: {' '.join(cmd)}")
        print(f"  stdout: {result.stdout[-1500:]}")
        print(f"  stderr: {result.stderr[-1500:]}")
        raise SystemExit(result.returncode)
    return result


def dexec(cmd: str, *, timeout: float | None = None,
          check: bool = True) -> subprocess.CompletedProcess:
    return run(
        ["docker", "exec", CONTAINER_NAME, "bash", "-lc", cmd],
        timeout=timeout, check=check,
    )


def write_into_container(path: str, content: bytes) -> None:
    encoded = base64.b64encode(content).decode("ascii")
    parent = str(Path(path).parent.as_posix())
    cmd = (
        f"mkdir -p {parent} && "
        f"printf '%s' {encoded!r} | base64 -d > {path}"
    )
    dexec(cmd)


def start_container() -> None:
    print(f"== spinning up: {CONTAINER_NAME}")
    run([
        "docker", "run", "-d", "--name", CONTAINER_NAME, "--rm",
        "ubuntu:24.04", "sleep", "1800",
    ])


def stop_container() -> None:
    print(f"== removing: {CONTAINER_NAME}")
    run(["docker", "stop", CONTAINER_NAME], check=False)


def install_dependencies(extra: set[str]) -> None:
    print("== installing python3 + clang + cmake + atheris (~40-90s)")
    t0 = time.monotonic()
    pip_extra = " ".join(sorted(extra)) if extra else ""
    cmd = (
        "apt-get update -qq && "
        "apt-get install -y -qq --no-install-recommends "
        "python3 python3-pip clang cmake && "
        f"pip3 install --break-system-packages --no-cache-dir atheris {pip_extra}"
    )
    dexec(cmd, timeout=600.0)
    print(f"   install done in {time.monotonic() - t0:.0f}s")


def stage_target(target: Target) -> str:
    """Write the target's source + minimal stub into the container.

    Returns the in-container module name (e.g. ``augmentum.documents.chunker``).
    """
    src_path = REPO_ROOT / target.file
    if not src_path.exists():
        raise FileNotFoundError(f"target not found: {src_path}")
    source = src_path.read_text(encoding="utf-8")

    # Drop the file at the matching dotted path inside /work.
    in_container_path = f"/work/{target.file}"
    write_into_container(in_container_path, source.encode("utf-8"))

    # Drop __init__.py files for each package along the path.
    parts = Path(target.file).parts[:-1]
    for i in range(1, len(parts) + 1):
        pkg_dir = "/work/" + "/".join(parts[:i])
        write_into_container(f"{pkg_dir}/__init__.py", b"")

    # Stub augmentum.utils.logging so target imports succeed without
    # dragging in structlog and the full Augmentum surface.
    write_into_container(
        "/work/augmentum/utils/__init__.py", b"",
    )
    write_into_container(
        "/work/augmentum/utils/logging.py", STUB_LOGGING.encode("utf-8"),
    )

    module = target.file.removesuffix(".py").replace("/", ".")
    return module


def derive_harness(target: Target, module: str) -> Harness:
    src = (REPO_ROOT / target.file).read_text(encoding="utf-8")
    verdict = classify_chunk(src, target.function, file_path=target.file)
    if not verdict.fuzzable:
        raise RuntimeError(
            f"{target.file}::{target.function} not fuzzable: {verdict.reason}",
        )
    # Override the workspace_root to "." so the resulting module path
    # is the dotted form rooted at /work.
    harness = generate_harness(
        verdict,
        target_file=target.file,
        target_function=target.function,
        workspace_root=".",
    )
    return harness


def run_fuzz_session(
    harness: Harness, target: Target, max_seconds: int = 25,
) -> tuple[FuzzRunResult, str]:
    safe = target.function.lstrip("_")
    workdir = f"/work/_fuzz_{safe}"
    corpus = f"{workdir}/corpus"
    artifact = f"{workdir}/artifacts"
    harness_path = f"{workdir}/{harness.suggested_filename}"

    dexec(
        f"rm -rf {workdir} && mkdir -p {corpus} {artifact}",
    )
    write_into_container(harness_path, harness.source.encode("utf-8"))
    # Seed corpus
    seeds = {
        "empty":      b"",
        "zero":       b"\x00",
        "ascii":      b"hello\n",
        "json-empty": b"{}",
        "pdf":        b"%PDF-1.4\n",
        "html":       b"<!DOCTYPE html>\n",
        "csv":        b"a,b,c\n1,2,3\n",
    }
    for name, data in seeds.items():
        write_into_container(f"{corpus}/{name}", data)

    print(f"   running {target.function!r} for {max_seconds}s...", flush=True)
    cmd = (
        f"cd {workdir} && "
        f"PYTHONPATH=/work python3 {harness_path} {corpus} "
        f"-artifact_prefix={artifact}/ "
        f"-max_total_time={max_seconds} 2>&1 || true"
    )
    t0 = time.monotonic()
    result = dexec(cmd, timeout=max_seconds + 60.0, check=False)
    elapsed = time.monotonic() - t0
    output = result.stdout

    # Detect harness-import failure separately from "no crashes":
    # they're very different signals.
    if "ModuleNotFoundError" in output or "ImportError" in output:
        first_err = next(
            (line for line in output.splitlines()
             if "Error" in line and "ImportError" not in line), "",
        )
        if "ModuleNotFoundError" in output:
            first_err = next(
                (line for line in output.splitlines()
                 if "ModuleNotFoundError" in line), "",
            )
        return FuzzRunResult(
            skipped=True,
            skip_reason=f"target failed to import: {first_err.strip()}",
        ), output

    # List crash artifacts
    ls = dexec(
        f"ls -1 {artifact}/crash-* 2>/dev/null || true", check=False,
    )
    crash_names = [
        Path(line.strip()).name
        for line in ls.stdout.splitlines() if line.strip()
    ]

    crashes: list[FuzzCrash] = []
    for name in sorted(crash_names):
        body = dexec(
            f"base64 -w0 {artifact}/{name} 2>/dev/null", check=False,
        )
        try:
            data = base64.b64decode(body.stdout.strip())
        except Exception:
            data = b""
        block = _find_trace_block(output, name)
        exc_cls, exc_msg = _extract_exception(block)
        crashes.append(FuzzCrash(
            input_basename=name, input_bytes=data,
            stack_trace=block,
            exception_class=exc_cls, exception_message=exc_msg,
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
        stderr_tail=output[-2000:],
    ), output


def main() -> None:
    if shutil.which("docker") is None:
        raise SystemExit("docker CLI required")

    extra_pip: set[str] = set()
    for t in TARGETS:
        extra_pip.update(t.extra_deps)

    start_container()
    try:
        install_dependencies(extra_pip)

        all_findings = []
        skipped = []
        clean = []
        for target in TARGETS:
            print(f"\n== TARGET: {target.file}::{target.function}")
            try:
                module = stage_target(target)
                harness = derive_harness(target, module)
            except Exception as exc:
                print(f"   SETUP FAILED: {exc}")
                skipped.append((target, str(exc)))
                continue
            result, output = run_fuzz_session(harness, target, max_seconds=25)
            if result.skipped:
                print(f"   SKIPPED: {result.skip_reason}")
                skipped.append((target, result.skip_reason))
                continue
            print(
                f"   ran {result.iterations} iterations in "
                f"{result.runtime_seconds:.1f}s — "
                f"{len(result.crashes)} crashes"
            )
            findings = triage_fuzz_run(
                result, harness=harness,
                chunk_file=target.file, chunk_function=target.function,
                artifact_dir=f"/work/_fuzz_{target.function.lstrip('_')}/artifacts",
            )
            if not findings:
                clean.append(target)
                continue
            for f in findings:
                print(f"     [!] {f.claim_signature.upper()} "
                      f"({f.severity}) — {f.claim}")
                print(f"         site: {f.file}:{f.function}")
            all_findings.extend((target, f) for f in findings)

        # ---- Summary ----
        print("\n" + "=" * 60)
        print("SMOKE SUMMARY")
        print("=" * 60)
        print(f"Targets fuzzed:    {len(TARGETS)}")
        print(f"Clean (no crash):  {len(clean)}")
        print(f"Skipped (setup):   {len(skipped)}")
        print(f"Findings:          {len(all_findings)}")
        if skipped:
            print("\nSkipped reasons:")
            for t, reason in skipped:
                print(f"  - {t.function}: {reason}")
        if all_findings:
            print("\nFindings detail:")
            for target, finding in all_findings:
                print(f"  - {target.file}::{target.function}")
                print(f"      {finding.claim_signature}/"
                      f"{finding.severity}: {finding.claim}")
                print(f"      repro: {finding.repro_path}")
        if not all_findings:
            print("\n(no crashes — the fuzzed targets are clean within "
                  "the 25s budget. This is the honest result for tested "
                  "Augmentum code.)")

    finally:
        stop_container()


if __name__ == "__main__":
    main()
