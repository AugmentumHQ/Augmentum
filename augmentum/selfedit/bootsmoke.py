"""Boot-smoke verifier — "does this candidate still boot?"

The single worst failure class a self-editing agent can introduce is a change
that's fatal at *import/boot* time: the gate's compile check catches syntax
errors, but a bad top-level import, a circular import, a broken provider
registration, or a migration that won't apply on a fresh DB all sail past
``compileall`` and only blow up when the app actually starts. A dead app can't
fix itself (invariant #1), so we catch this in isolation *before* anything is
promoted.

This is the same two-step check ``audit.py --smoke`` runs, lifted to a
standalone mechanical Verifier so it can gate a single candidate cheaply
(~2 subprocesses, no full audit):

  1. ``from augmentum.proxy.server import create_app`` — exercises the entire
     route + provider import graph.
  2. Apply every migration on a fresh ``:memory:`` SQLiteBackend.

Run against the *candidate's own code* (cwd = candidate worktree), so it tests
the change on disk, not the modules already imported into the running server —
the same trust property the fitness gate relies on.

``confirms_intent=False``: booting proves the change didn't *break* the app, not
that it did what was asked. On its own a green boot-smoke yields
``human_required``, never ``verified``.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from augmentum.selfedit.verifier import (
    FAIL,
    ORACLE_MECHANICAL,
    PASS,
    SKIP,
    Verifier,
    VerifierResult,
    register_verifier,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# The default boot module — importing it exercises the whole route/provider graph.
_BOOT_IMPORT = "from augmentum.proxy.server import create_app; print('OK')"

# Apply every migration on a fresh in-memory DB (mirrors audit.py --smoke step 2).
_MIGRATE_SNIPPET = (
    "import asyncio\n"
    "async def go():\n"
    "    from augmentum.state.backends.sqlite import SQLiteBackend\n"
    "    b = SQLiteBackend(':memory:')\n"
    "    await b.connect()\n"
    "    await b.close()\n"
    "    print('OK')\n"
    "asyncio.run(go())\n"
)


@dataclass
class BootResult:
    ok: bool
    failures: list[str]      # human-readable: which sub-check failed + last error line
    launched: bool = True    # False = python itself couldn't be launched (→ SKIP, not FAIL)


# A boot runner: given a target dir, return a BootResult. Injectable so the
# verifier logic is testable without spawning the real (slow) subprocesses.
BootRunner = Callable[[str], Awaitable[BootResult]]


async def _run_snippet(code: str, *, cwd: str, timeout: float) -> tuple[int, str, bool]:
    """Run ``python -c code`` with cwd=candidate. Returns (exit, output, launched).
    Runs with a SECRET-SCRUBBED env: this imports the candidate's code, so it must
    not carry the app's API keys into code we're about to judge (W11)."""
    from augmentum.selfedit.sandbox import scrubbed_env
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code, cwd=cwd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            env=scrubbed_env(),
        )
    except Exception as exc:  # noqa: BLE001 — interpreter not launchable → can't measure
        return 127, f"could not launch interpreter: {exc!r}", False
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        with contextlib.suppress(Exception):
            proc.kill()
        return 124, f"timed out after {timeout:.0f}s", True
    return (proc.returncode or 0), (out or b"").decode("utf-8", errors="replace"), True


def _last_line(output: str) -> str:
    lines = [ln for ln in output.strip().splitlines() if ln.strip()]
    return lines[-1][:300] if lines else "no output"


_FILE_FRAME = re.compile(r'^\s*File "([^"]+)", line (\d+)')


def _error_locus(output: str) -> str:
    """Extract a LOCATION-rich failure line from a Python traceback:
    ``<file>:<line>: <ExceptionType: message>`` plus the offending source line
    when the interpreter shows one. The location is what makes a failure
    *repairable* — a model can't fix "unexpected indent" in an 8,600-line file,
    but it can fix "server.py:8635: unexpected indent · from ... import ...".

    This is the structured error capture the coder's parse-checkers do (they keep
    lineno/offset); boot-smoke used to keep only the last line and threw the
    traceback location away. Falls back to the last line when no real source frame
    is present (e.g. a bare error with no traceback)."""
    lines = output.splitlines()
    nonempty = [ln for ln in lines if ln.strip()]
    msg = nonempty[-1].strip()[:200] if nonempty else "no output"
    locus = ""
    src = ""
    # the DEEPEST real source frame is the error site (skip the -c "<string>"
    # wrapper and any <frozen ...> import-machinery frames)
    for i, ln in enumerate(lines):
        m = _FILE_FRAME.match(ln)
        if not m or m.group(1).startswith("<"):
            continue
        base = m.group(1).replace("\\", "/").split("/")[-1]
        locus = f"{base}:{m.group(2)}"
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        # the source line the interpreter prints under a frame (skip caret rows)
        if nxt and not _FILE_FRAME.match(lines[i + 1]) and not nxt.startswith("^"):
            src = nxt[:120]
    if not locus:
        return msg[:300]
    detail = f"{locus}: {msg}"
    if src:
        detail += f" · {src}"
    return detail[:300]


async def default_boot_runner(target_dir: str, *, timeout: float = 90.0) -> BootResult:
    """Real boot-smoke: import the app + apply migrations, both as subprocesses
    against ``target_dir``. A launch failure (couldn't even start python) →
    ``launched=False`` so the verifier SKIPs rather than false-failing."""
    failures: list[str] = []
    launched = True

    code, out, ok_launch = await _run_snippet(_BOOT_IMPORT, cwd=target_dir, timeout=timeout)
    launched = launched and ok_launch
    if ok_launch and (code != 0 or "OK" not in out):
        # location-rich (file:line + offending source) so a self-heal repair can
        # navigate straight to the break instead of hunting blind.
        failures.append(f"import create_app: {_error_locus(out)}")

    code, out, ok_launch = await _run_snippet(_MIGRATE_SNIPPET, cwd=target_dir, timeout=timeout)
    launched = launched and ok_launch
    if ok_launch and (code != 0 or "OK" not in out):
        failures.append(f"migrations apply: {_error_locus(out)}")

    return BootResult(ok=not failures, failures=failures, launched=launched)


def boot_smoke_verifier(*, boot_runner: BootRunner | None = None,
                        required: bool = True, cost: int = 4) -> Verifier:
    """A mechanical no-regression Verifier that proves the candidate boots.

    cheaper than the full audit (``cost=4`` runs before it) and required — a
    fatal-at-boot candidate must short-circuit the expensive checks. Skips (not
    fails) only when the interpreter itself can't be launched, so an
    infrastructure hiccup never reads as a code regression."""
    runner = boot_runner or default_boot_runner

    async def _run(ctx: dict) -> VerifierResult:
        target = ctx.get("candidate_dir") or "."
        try:
            res = await runner(target)
        except Exception as exc:  # noqa: BLE001 — runner blew up → can't measure, skip
            return VerifierResult("boot_smoke", ORACLE_MECHANICAL, SKIP, confirms_intent=False,
                                  required=required, detail=f"boot-smoke unavailable: {exc!r}")
        if not res.launched:
            return VerifierResult("boot_smoke", ORACLE_MECHANICAL, SKIP, confirms_intent=False,
                                  required=required,
                                  detail="; ".join(res.failures) or "interpreter not launchable")
        status = PASS if res.ok else FAIL
        detail = "boots clean" if res.ok else "BOOT BROKE — " + "; ".join(res.failures)
        return VerifierResult("boot_smoke", ORACLE_MECHANICAL, status, confirms_intent=False,
                              score=1.0 if res.ok else 0.0, required=required, detail=detail)

    return Verifier("boot_smoke", ORACLE_MECHANICAL, _run, ("*",), confirms_intent=False,
                    cost=cost, required=required)


def register_boot_smoke_verifier(*, boot_runner: BootRunner | None = None,
                                 required: bool = True) -> None:
    register_verifier(boot_smoke_verifier(boot_runner=boot_runner, required=required))
