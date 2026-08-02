"""Under-test app boot orchestrator for the pen-test leg.

Phase 1b — given a ``BootSpec`` (explicit command + port + healthcheck),
spawn the workspace's application as a subprocess, wait for the
healthcheck to come green, and return an ``UnderTestService`` handle
that the ``pen_tester`` subagent can target with ``http_attack``.

Why this is its own module (separate from ``pen_test.py``):
* ``pen_test.py`` is stateless — every probe is independent.
* ``pen_test_boot.py`` is stateful — running processes have to be
  tracked across tool calls and torn down at run end. Mixing the two
  would make the safety reasoning harder.

What's intentionally out of scope for Phase 1b:
* **Auto-detection** of boot commands — the planner / lead supplies an
  explicit ``BootSpec``. A future enhancement can sniff ``Procfile`` /
  ``docker-compose.yml`` / FastAPI/Flask defaults and propose specs.
* **Sibling docker containers** — Phase 1b uses subprocess in the same
  process tree as the bug_finder. Container-in-container isolation is
  a follow-up; the entire bug_finder container is already disposable.
* **The pen_tester subagent role** — that's Phase 1c. Phase 1b
  delivers the primitive so 1c just has to wire the role.

**Lifecycle contract**:
* ``boot_under_test()`` registers the service in the per-run
  ``_UnderTestRegistry``.
* The orchestrator MUST call ``teardown_all()`` at run end (success
  or failure). The LLM does not get a teardown tool — it cannot
  leak processes through tool misuse.
* If the parent process exits without teardown, the subprocess is
  orphaned. That's a bug_finder pipeline bug, not a security flaw —
  the workspace container is destroyed anyway.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from augmentum.bug_finder.pen_test import ProbeRequest, execute_probe
from augmentum.bug_finder.workspace_substrate import (
    substrate_dir,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Limits + defaults
# ---------------------------------------------------------------------------


DEFAULT_HEALTHCHECK_INTERVAL_S: float = 0.5
DEFAULT_HEALTHCHECK_TIMEOUT_S: float = 30.0
DEFAULT_BOOT_TIMEOUT_S: float = 60.0
MAX_BOOT_TIMEOUT_S: float = 300.0          # 5 minutes is plenty
MAX_LOG_TAIL_BYTES: int = 32 * 1024        # 32 KB captured for failure surfacing


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootSpec:
    """How to boot one under-test application.

    The caller (the pen_tester subagent, the orchestrator, or a script)
    must supply ``command``, ``port``, and ``healthcheck_path``. The
    booter doesn't try to guess any of these — explicit specs are
    auditable, ambiguous heuristics are not.

    ``env_overrides`` are applied on top of the inherited environment.
    ``env_allowlist`` (if set) restricts inheritance to those keys
    only — useful when the under-test app shouldn't see secrets in
    the parent env.

    ``cwd`` is interpreted relative to the workspace root when
    relative, absolute otherwise. ``None`` = workspace root.
    """

    command: tuple[str, ...]
    port: int
    healthcheck_path: str = "/"
    healthcheck_timeout_s: float = DEFAULT_HEALTHCHECK_TIMEOUT_S
    boot_timeout_s: float = DEFAULT_BOOT_TIMEOUT_S
    healthcheck_interval_s: float = DEFAULT_HEALTHCHECK_INTERVAL_S
    env_overrides: dict[str, str] = field(default_factory=dict)
    env_allowlist: tuple[str, ...] | None = None
    cwd: str | None = None
    # The healthcheck expects a status code in this range (inclusive).
    # Default 200-399 to admit redirects and "no-content but alive".
    healthcheck_status_min: int = 200
    healthcheck_status_max: int = 399


@dataclass
class UnderTestService:
    """A running under-test app.

    Held by the per-run ``_UnderTestRegistry``. The pen_tester gets
    only the read-safe fields (``service_id``, ``base_url``, ``pid``,
    ``healthy``) via the tool wrapper; the process handle stays inside
    the orchestrator.
    """

    service_id: str
    spec: BootSpec
    base_url: str
    pid: int
    started_at: int
    log_path: Path | None = None
    process: asyncio.subprocess.Process | None = None
    teardown_called: bool = False

    @property
    def healthy(self) -> bool:
        if self.process is None:
            return False
        return self.process.returncode is None and not self.teardown_called


@dataclass(frozen=True)
class BootFailure:
    """Why an under-test boot didn't reach healthy state."""

    reason: str                 # short slug: "timeout" | "exit" | "command_error"
    detail: str
    elapsed_ms: int
    exit_code: int | None = None
    log_tail: str = ""


@dataclass(frozen=True)
class BootResult:
    """Outcome of one ``boot_under_test`` call.

    Exactly one of ``service`` / ``failure`` is populated.
    """

    service: UnderTestService | None = None
    failure: BootFailure | None = None

    @property
    def ok(self) -> bool:
        return self.service is not None


# ---------------------------------------------------------------------------
# Registry — per-process state for cross-tool lifecycle
# ---------------------------------------------------------------------------


class _UnderTestRegistry:
    """In-memory registry of running under-test services.

    One per bug_finder run. The orchestrator owns the registry and is
    responsible for calling ``teardown_all`` at run end (in a try /
    finally so it runs even when the run is cancelled).

    The pen_tester subagent NEVER sees this object — it only gets the
    service handle's id + base_url through tool returns. Teardown is
    not exposed to the LLM.
    """

    def __init__(self) -> None:
        self._services: dict[str, UnderTestService] = {}

    def add(self, service: UnderTestService) -> None:
        self._services[service.service_id] = service

    def get(self, service_id: str) -> UnderTestService | None:
        return self._services.get(service_id)

    def all(self) -> tuple[UnderTestService, ...]:
        return tuple(self._services.values())

    async def teardown_all(
        self, *, grace_seconds: float = 3.0,
    ) -> dict[str, str]:
        """Tear down every registered service. Returns a per-id
        verdict map: ``"clean"`` if the process exited within the
        grace period, ``"forced"`` if we had to kill, ``"skipped"``
        if already torn down or never started."""
        verdicts: dict[str, str] = {}
        for sid, svc in list(self._services.items()):
            verdicts[sid] = await teardown_service(svc, grace_seconds=grace_seconds)
        return verdicts


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------


def _resolve_cwd(workspace_root: Path, spec: BootSpec) -> Path:
    if not spec.cwd:
        return workspace_root
    p = Path(spec.cwd)
    if p.is_absolute():
        return p
    return workspace_root / p


def _build_env(spec: BootSpec) -> dict[str, str]:
    if spec.env_allowlist is None:
        env = dict(os.environ)
    else:
        env = {k: os.environ[k] for k in spec.env_allowlist if k in os.environ}
    env.update(spec.env_overrides)
    return env


def _log_path(workspace_root: Path | None, service_id: str) -> Path | None:
    """Return a writable log path for the under-test service.

    Preferred location is the workspace substrate dir
    (``<workspace>/.augmentum/under_test_logs/``). When that dir is
    on a read-only mount (e.g. augmentum-on-augmentum where
    ``/app/.augmentum`` is RO), fall back to the system temp dir so
    boots don't crash before they even start. The boot path tolerates
    ``None`` for log_path — losing logs is preferable to losing the
    leg entirely.
    """
    if workspace_root is None:
        return None
    logs_dir = substrate_dir(workspace_root) / "under_test_logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        # Confirm writability by touching a sentinel — mkdir succeeds
        # on RO mounts when the dir already exists.
        probe = logs_dir / ".write_probe"
        probe.touch(exist_ok=True)
        probe.unlink(missing_ok=True)
        return logs_dir / f"{service_id}.log"
    except OSError:
        import tempfile
        fallback = Path(tempfile.gettempdir()) / "augmentum_under_test_logs"
        try:
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback / f"{service_id}.log"
        except OSError:
            return None


async def _read_log_tail(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    if len(raw) > MAX_LOG_TAIL_BYTES:
        raw = b"...[earlier output truncated]...\n" + raw[-MAX_LOG_TAIL_BYTES:]
    return raw.decode("utf-8", errors="replace")


async def boot_under_test(
    workspace_root: Path,
    spec: BootSpec,
    *,
    registry: _UnderTestRegistry | None = None,
) -> BootResult:
    """Boot one under-test service. Always returns a ``BootResult``;
    never raises.

    The function spawns the command, polls the healthcheck endpoint
    until 2xx (or whatever range ``spec`` permits), and registers
    the running service. On failure it captures the log tail so the
    pen_tester / orchestrator can surface why the boot didn't go.

    Side effects:
      * stdout/stderr captured to ``.augmentum/bug_finder/under_test_logs/``
      * service registered in ``registry`` for orchestrator teardown
    """
    if not spec.command:
        return BootResult(failure=BootFailure(
            reason="command_error",
            detail="BootSpec.command is empty",
            elapsed_ms=0,
        ))

    boot_timeout = max(1.0, min(spec.boot_timeout_s, MAX_BOOT_TIMEOUT_S))
    healthcheck_timeout = max(1.0, min(spec.healthcheck_timeout_s, boot_timeout))
    interval = max(0.05, spec.healthcheck_interval_s)

    workspace_root = Path(workspace_root).resolve()
    cwd = _resolve_cwd(workspace_root, spec)
    if not cwd.is_dir():
        return BootResult(failure=BootFailure(
            reason="command_error",
            detail=f"cwd does not exist: {cwd}",
            elapsed_ms=0,
        ))

    env = _build_env(spec)

    service_id = "ut_" + uuid.uuid4().hex[:12]
    log_p = _log_path(workspace_root, service_id)
    started = time.monotonic()
    started_at_wall = int(time.time())

    # --- Spawn ---
    log_handle = None
    try:
        # We pipe stdout/stderr to the log file (when available) so
        # the LLM can read it on failure. Don't use PIPE without a
        # reader — that'd deadlock on large outputs.
        if log_p is not None:
            log_handle = log_p.open("ab", buffering=0)
        proc = await asyncio.create_subprocess_exec(
            *spec.command,
            cwd=str(cwd),
            env=env,
            stdout=log_handle if log_handle is not None else asyncio.subprocess.DEVNULL,
            stderr=log_handle if log_handle is not None else asyncio.subprocess.DEVNULL,
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        if log_handle is not None:
            try:
                log_handle.close()
            except OSError:
                pass
        return BootResult(failure=BootFailure(
            reason="command_error",
            detail=f"{type(exc).__name__}: {exc}",
            elapsed_ms=int((time.monotonic() - started) * 1000),
        ))

    base_url = f"http://localhost:{int(spec.port)}"
    service = UnderTestService(
        service_id=service_id,
        spec=spec,
        base_url=base_url,
        pid=proc.pid,
        started_at=started_at_wall,
        log_path=log_p,
        process=proc,
    )

    # --- Healthcheck loop ---
    #
    # On Windows, connection-refused to a closed local port DOES NOT
    # fast-fail in httpx — the OS lets the connect attempt run to the
    # full timeout. If we used a generous per-probe timeout we'd burn
    # the whole boot budget on a handful of failed connects and never
    # notice the subprocess has died. Two countermeasures:
    #
    #   1. ``per_probe_timeout`` is capped tight (default 0.5s). The
    #      loop just iterates more.
    #   2. We start a background ``proc.wait()`` task so we can detect
    #      "subprocess exited" instantly without polling.
    healthcheck_url = base_url.rstrip("/") + "/" + spec.healthcheck_path.lstrip("/")
    deadline = started + healthcheck_timeout
    per_probe_timeout = min(1.0, max(0.2, interval * 2))
    last_error = ""

    proc_done_task = asyncio.create_task(proc.wait())

    try:
        while True:
            # Has the subprocess died since last iteration?
            if proc_done_task.done():
                log_tail = await _read_log_tail(log_p)
                return BootResult(failure=BootFailure(
                    reason="exit",
                    detail=(
                        f"process exited with code {proc.returncode} "
                        "before healthcheck went green"
                    ),
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    exit_code=proc.returncode,
                    log_tail=log_tail,
                ))

            # Try the healthcheck. Per-probe timeout is small so the
            # loop iterates quickly on Windows where connect-refused
            # is slow.
            try:
                resp, _ = await execute_probe(
                    ProbeRequest(
                        method="GET", url=healthcheck_url,
                        timeout_s=per_probe_timeout,
                        note=f"healthcheck for {service_id}",
                    ),
                    workspace_root=None,    # don't pollute receipts trail
                )
                if resp.ok and (
                    spec.healthcheck_status_min
                    <= resp.status
                    <= spec.healthcheck_status_max
                ):
                    # We're up.
                    if registry is not None:
                        registry.add(service)
                    log.info(
                        "bug_finder_under_test_booted",
                        service_id=service_id,
                        base_url=base_url,
                        pid=proc.pid,
                        elapsed_ms=int((time.monotonic() - started) * 1000),
                    )
                    return BootResult(service=service)
                last_error = (
                    f"status={resp.status} body[:80]="
                    f"{resp.body_excerpt[:80]!r}"
                    if resp.ok
                    else (resp.error or "transport error")
                )
            except Exception as exc:    # noqa: BLE001 — boot must not raise
                last_error = f"{type(exc).__name__}: {exc}"

            # One more proc-died check before deciding deadline.
            if proc_done_task.done():
                log_tail = await _read_log_tail(log_p)
                return BootResult(failure=BootFailure(
                    reason="exit",
                    detail=(
                        f"process exited with code {proc.returncode} "
                        "before healthcheck went green"
                    ),
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    exit_code=proc.returncode,
                    log_tail=log_tail,
                ))

            if time.monotonic() >= deadline:
                # Time's up. Kill the process so we don't orphan it.
                await teardown_service(service, grace_seconds=2.0)
                log_tail = await _read_log_tail(log_p)
                return BootResult(failure=BootFailure(
                    reason="timeout",
                    detail=(
                        f"healthcheck never went green within "
                        f"{healthcheck_timeout}s. last_error={last_error}"
                    ),
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    log_tail=log_tail,
                ))

            # Race a short sleep against the proc-done task — if the
            # process exits during the sleep we want to notice
            # immediately, not after the full interval.
            await asyncio.wait(
                {proc_done_task},
                timeout=interval,
                return_when=asyncio.FIRST_COMPLETED,
            )
    finally:
        # Always clean up the wait task. On the success path it's
        # still running (proc is alive on purpose); cancel it.
        if not proc_done_task.done():
            proc_done_task.cancel()
            try:
                await proc_done_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


async def teardown_service(
    service: UnderTestService,
    *,
    grace_seconds: float = 3.0,
) -> str:
    """Stop one under-test service. Returns one of:
      * ``"clean"`` — process exited gracefully within the grace
        period.
      * ``"forced"`` — had to ``kill()`` the process.
      * ``"skipped"`` — already torn down or never started.

    Best-effort: catches every exception. The pipeline cannot be
    blocked by a stuck under-test app.
    """
    if service.teardown_called:
        return "skipped"
    service.teardown_called = True
    proc = service.process
    if proc is None or proc.returncode is not None:
        return "skipped"

    # Step 1: graceful — SIGTERM on POSIX, TerminateProcess on Windows
    try:
        proc.terminate()
    except (ProcessLookupError, PermissionError, OSError) as exc:
        log.debug(
            "bug_finder_under_test_terminate_failed",
            service_id=service.service_id, error=str(exc),
        )
        # If we can't even signal it, count it as skipped — the
        # process has presumably already died.
        return "skipped"

    # Step 2: wait briefly for graceful exit
    try:
        await asyncio.wait_for(proc.wait(), timeout=max(0.5, grace_seconds))
        log.info(
            "bug_finder_under_test_torn_down",
            service_id=service.service_id,
            verdict="clean",
            exit_code=proc.returncode,
        )
        return "clean"
    except TimeoutError:
        pass

    # Step 3: force-kill if we have to
    try:
        proc.kill()
        await proc.wait()
        log.info(
            "bug_finder_under_test_torn_down",
            service_id=service.service_id,
            verdict="forced",
            exit_code=proc.returncode,
        )
        return "forced"
    except (ProcessLookupError, OSError) as exc:
        log.debug(
            "bug_finder_under_test_kill_failed",
            service_id=service.service_id, error=str(exc),
        )
        return "forced"


# ---------------------------------------------------------------------------
# Module-level state for the orchestrator
# ---------------------------------------------------------------------------


# A process-local default registry. The orchestrator passes its own
# instance into ``boot_under_test`` for per-run scoping; the
# module-level default is here for ad-hoc/script use only.
_DEFAULT_REGISTRY = _UnderTestRegistry()


def default_registry() -> _UnderTestRegistry:
    """Return the process-default registry. Mostly used by tests +
    scripts; the orchestrator should manage its own per-run registry."""
    return _DEFAULT_REGISTRY


def reset_default_registry() -> None:
    """Drop all entries from the default registry without teardown.
    Test-only helper — production code should call ``teardown_all``
    first."""
    _DEFAULT_REGISTRY._services.clear()    # noqa: SLF001
