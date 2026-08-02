"""Run an Atheris harness inside a bug_finder workspace container.

Lazy installation pattern: ``atheris`` + ``clang`` + ``cmake`` are
installed on first use rather than baked into the augmentum image.
First fuzz run pays ~60s of install latency; subsequent runs are
instant. The trade-off: zero image bloat for users who never fuzz,
and zero Dockerfile change to land Phase 1.

If the install fails (offline, sandbox blocks apt, exotic platform),
the runner returns ``FuzzRunResult(skipped=True, skip_reason=...)`` and
the bug_finder pipeline continues with the LLM detector leg only —
the same graceful-skip shape Semgrep / symbolic_gate use.

What the runner does:

1. Ensure atheris is importable in the container.
2. Write the harness file + seed corpus to
   ``/workspace/.augmentum/fuzz/<chunk-id>/``.
3. Invoke ``python3 harness.py corpus/ -artifact_prefix=... -max_total_time=N``.
4. List the resulting ``crash-*`` artifacts, read each one's bytes,
   pair it with the traceback block from the harness output, return
   structured ``FuzzCrash`` rows.

The result is consumed by ``fuzz.triage`` which turns each crash into
a ``Finding`` row that flows through the existing dedup / ranking /
report machinery.
"""

from __future__ import annotations

import base64
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import PurePosixPath

from augmentum.bug_finder.fuzz.harness import Harness
from augmentum.coder.containers import ContainerManager
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Default seed corpus. Atheris's coverage-guided mutation benefits from
# a few starting points so the first generation isn't all pure-random
# bytes. The set below covers the most common parse-target magic-byte
# regions Augmentum's importers handle (PDF/HTML/JSON). Eight seeds is
# enough to prime the mutation engine; the spec's ``max_seed_corpus_size``
# default is 32, but eight is more honest for "we have no idea what
# the user's parser expects".
_DEFAULT_SEEDS: tuple[tuple[str, bytes], ...] = (
    ("empty",       b""),
    ("zero",        b"\x00"),
    ("zeros-16",    b"\x00" * 16),
    ("ascii",       b"hello world\n"),
    ("pdf-magic",   b"%PDF-1.4\n"),
    ("html-magic",  b"<!DOCTYPE html>\n"),
    ("json-empty",  b"{}"),
    ("json-array", b"[]"),
)

_INSTALL_CMD = (
    "apt-get update -qq && "
    "apt-get install -y -qq --no-install-recommends clang cmake && "
    "pip install --no-cache-dir atheris"
)
_FUZZ_BASE_DIR = "/workspace/.augmentum/fuzz"


@dataclass(frozen=True)
class FuzzCrash:
    """One crash captured by atheris.

    ``input_bytes`` is the exact input that triggered the crash —
    becomes the PoC artifact for the finding row. ``stack_trace`` is
    the Python traceback (or libfuzzer crash header for native
    crashes) extracted from the harness output.
    """

    input_basename: str        # e.g. "crash-a1b2c3..."
    input_bytes: bytes         # the bytes that triggered the crash
    stack_trace: str           # Python traceback text
    exception_class: str       # "ValueError" / "ZeroDivisionError" / ...
    exception_message: str     # first line of the exception text


@dataclass(frozen=True)
class FuzzRunResult:
    """Aggregate outcome of one fuzz session for one chunk."""

    skipped: bool = False
    skip_reason: str = ""
    crashes: tuple[FuzzCrash, ...] = ()
    iterations: int = 0
    runtime_seconds: float = 0.0
    stderr_tail: str = ""      # last 4 KB of combined stdout/stderr —
                               # useful for debugging "why did atheris
                               # exit so fast" cases.

    @property
    def has_crashes(self) -> bool:
        return bool(self.crashes)


# ---------------------------------------------------------------------------
# Helpers — parsing + container I/O
# ---------------------------------------------------------------------------


def _extract_exception(stack_trace: str) -> tuple[str, str]:
    """Return ``(exception_class, exception_message)`` from a traceback.

    A Python traceback's last non-empty line is conventionally
    ``ExceptionClass: message``. We anchor on a name ending in
    ``Error`` or ``Exception`` (qualified or not) so we don't grab
    arbitrary ``foo: bar`` text from inside the call chain.
    """
    last = ""
    for line in reversed(stack_trace.splitlines()):
        candidate = line.strip()
        if candidate:
            last = candidate
            break
    # Strict match — exception class names conventionally end in Error
    # or Exception.
    strict = re.match(
        r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))\s*:\s*(.*)$",
        last,
    )
    if strict:
        return strict.group(1), strict.group(2)
    # Looser fallback (e.g. atheris-specific names without the suffix).
    loose = re.match(
        r"^([A-Za-z_][A-Za-z0-9_.]+)\s*:\s*(.*)$", last,
    )
    if loose:
        return loose.group(1), loose.group(2)
    return "UnknownException", last


def _find_trace_block(output: str, crash_basename: str) -> str:
    """Locate the traceback block paired with one crash artifact name.

    Atheris prints traceback → libfuzzer crash header →
    "Test unit written to ./crash-<sha>". Anchor on the basename and
    walk backwards to the most recent ``Traceback (most recent call
    last):`` line. Falls back to "last traceback in output" when the
    anchor can't be found (atheris occasionally puts the basename only
    in stderr).
    """
    anchor = output.rfind(crash_basename) if crash_basename else -1
    if anchor < 0:
        idx = output.rfind("Traceback (most recent call last):")
        return output[idx:].strip() if idx >= 0 else ""
    block_start = output.rfind(
        "Traceback (most recent call last):", 0, anchor,
    )
    if block_start < 0:
        return ""
    # Stop at the libfuzzer divider line ("==NNN==") if present.
    block_end_match = re.search(r"\n==\d+==\s", output[block_start:anchor])
    block_end = (
        block_start + block_end_match.start() if block_end_match else anchor
    )
    return output[block_start:block_end].strip()


async def _is_atheris_installed(
    cm: ContainerManager, workspace_id: str,
) -> bool:
    try:
        out = await cm.run_command(
            workspace_id,
            ["python3", "-c", "import atheris; print('atheris-ok')"],
            timeout=15.0,
        )
    except Exception:  # noqa: BLE001 — any failure ⇒ not installed
        return False
    return "atheris-ok" in (out or "")


async def _ensure_atheris(
    cm: ContainerManager, workspace_id: str,
    *, install_timeout: float = 300.0,
) -> tuple[bool, str]:
    """Best-effort install. Returns ``(ok, reason)``."""
    if await _is_atheris_installed(cm, workspace_id):
        return True, ""
    log.info(
        "bug_finder_fuzz_install_starting", workspace_id=workspace_id,
    )
    try:
        await cm.run_command(
            workspace_id, ["bash", "-lc", _INSTALL_CMD],
            timeout=install_timeout,
            idle_timeout=90.0,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"atheris install failed: {exc}"
    if not await _is_atheris_installed(cm, workspace_id):
        return False, "atheris install reported success but import still fails"
    return True, ""


async def _write_file(
    cm: ContainerManager, workspace_id: str, path: str, content: bytes,
) -> None:
    """Write bytes into the container via base64 (quoting-safe)."""
    encoded = base64.b64encode(content).decode("ascii")
    parent = str(PurePosixPath(path).parent)
    # Empty input → base64 -d still writes a 0-byte file, which is
    # what we want for the "empty" seed.
    cmd = (
        f"mkdir -p {shlex.quote(parent)} && "
        f"printf '%s' {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}"
    )
    await cm.run_command(workspace_id, ["bash", "-lc", cmd], timeout=30.0)


async def _read_file(
    cm: ContainerManager, workspace_id: str, path: str,
) -> bytes:
    """Read a file from the container as bytes."""
    try:
        encoded = await cm.run_command(
            workspace_id,
            ["bash", "-lc", f"base64 -w0 {shlex.quote(path)} 2>/dev/null"],
            timeout=15.0,
        )
    except Exception:  # noqa: BLE001
        return b""
    try:
        return base64.b64decode((encoded or "").strip())
    except Exception:  # noqa: BLE001
        return b""


async def _list_crashes(
    cm: ContainerManager, workspace_id: str, artifact_dir: str,
) -> list[str]:
    try:
        out = await cm.run_command(
            workspace_id,
            ["bash", "-lc",
             f"ls -1 {shlex.quote(artifact_dir)}/crash-* 2>/dev/null || true"],
            timeout=10.0,
        )
    except Exception:  # noqa: BLE001
        return []
    return [
        PurePosixPath(line.strip()).name
        for line in (out or "").splitlines()
        if line.strip()
    ]


async def _seed_corpus(
    cm: ContainerManager, workspace_id: str, corpus_dir: str,
) -> None:
    for name, data in _DEFAULT_SEEDS:
        await _write_file(
            cm, workspace_id, f"{corpus_dir}/{name}", data,
        )


def _safe_chunk_id(chunk_id: str) -> str:
    """Sanitize an arbitrary chunk identifier into a path-safe slug.

    Inputs collapsing to only underscores (``///!!!`` → ``_``) fall
    through to the ``"chunk"`` placeholder. A path of
    ``/workspace/.augmentum/fuzz/_/...`` would still work but reads
    like a bug; the placeholder is more honest.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", chunk_id)[:80]
    if not cleaned or cleaned.strip("_-") == "":
        return "chunk"
    return cleaned


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_fuzz_harness(
    harness: Harness,
    *,
    cm: ContainerManager,
    workspace_id: str,
    chunk_id: str,
    max_seconds: int = 60,
    install_timeout: float = 300.0,
) -> FuzzRunResult:
    """Run ``harness`` in the workspace container, return the crash inventory.

    The runner is idempotent w.r.t. the install (skipped after first
    call) and the run dir (wiped before each session). Crashes are
    returned ordered by artifact filename so re-runs produce stable
    output for downstream dedup.

    ``max_seconds`` is the libfuzzer wall-clock budget. The outer
    container exec adds 30s grace for atheris startup + tear-down.
    """
    ok, reason = await _ensure_atheris(
        cm, workspace_id, install_timeout=install_timeout,
    )
    if not ok:
        log.warning("bug_finder_fuzz_skipped", reason=reason)
        return FuzzRunResult(skipped=True, skip_reason=reason)

    safe = _safe_chunk_id(chunk_id)
    base = f"{_FUZZ_BASE_DIR}/{safe}"
    corpus = f"{base}/corpus"
    artifact = f"{base}/artifacts"
    harness_path = f"{base}/{harness.suggested_filename}"

    # Fresh-start each session — we don't carry state across chunks for v1.
    # Persisting crash corpora across runs (the "pattern memory of seeds"
    # idea in the spec) is a Phase 3 nicety.
    await cm.run_command(
        workspace_id, ["bash", "-lc",
         f"rm -rf {shlex.quote(base)} && "
         f"mkdir -p {shlex.quote(corpus)} {shlex.quote(artifact)}"],
        timeout=10.0,
    )
    await _write_file(
        cm, workspace_id, harness_path, harness.source.encode("utf-8"),
    )
    await _seed_corpus(cm, workspace_id, corpus)

    cmd = (
        f"cd {shlex.quote(base)} && "
        f"python3 {shlex.quote(harness_path)} {shlex.quote(corpus)} "
        f"-artifact_prefix={shlex.quote(artifact)}/ "
        f"-max_total_time={int(max_seconds)} "
        f"2>&1 || true"
    )
    start = time.monotonic()
    try:
        output = await cm.run_command(
            workspace_id, ["bash", "-lc", cmd],
            timeout=max_seconds + 60.0,
            idle_timeout=max_seconds + 30.0,
        )
    except Exception as exc:  # noqa: BLE001
        return FuzzRunResult(
            skipped=True,
            skip_reason=f"harness execution failed: {exc}",
            runtime_seconds=time.monotonic() - start,
        )
    elapsed = time.monotonic() - start

    # libfuzzer reports iteration counts on lines like "#1234 INITED"
    # or "#5678 NEW". Last match wins (highest count reached).
    iters = 0
    for m in re.finditer(r"#(\d+)\s+(NEW|INITED|REDUCE|pulse)", output):
        try:
            iters = max(iters, int(m.group(1)))
        except ValueError:
            continue

    crash_names = sorted(await _list_crashes(cm, workspace_id, artifact))
    crashes: list[FuzzCrash] = []
    for name in crash_names:
        data = await _read_file(cm, workspace_id, f"{artifact}/{name}")
        block = _find_trace_block(output, name)
        exc_cls, exc_msg = _extract_exception(block)
        crashes.append(FuzzCrash(
            input_basename=name,
            input_bytes=data,
            stack_trace=block,
            exception_class=exc_cls,
            exception_message=exc_msg,
        ))

    log.info(
        "bug_finder_fuzz_complete",
        chunk_id=chunk_id, iterations=iters,
        crashes=len(crashes), runtime_seconds=round(elapsed, 1),
    )

    return FuzzRunResult(
        skipped=False,
        crashes=tuple(crashes),
        iterations=iters,
        runtime_seconds=elapsed,
        stderr_tail=(output or "")[-4000:],
    )
