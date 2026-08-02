"""``command_runner`` implementations — the edit driver's execution seam.

The edit driver (``edit_driver.run_engine_edit_driver``) is environment-agnostic:
it builds the agent argv + env and hands them to a ``command_runner`` that runs
them *somewhere*, streaming stdout to ``on_chunk``. This module provides the
concrete runners.

**dev-bind (Decision 3, first):** ``subprocess_command_runner`` runs the agent as
a host subprocess with ``cwd = the candidate worktree``. In the clone-and-run /
dev-bind deployment the host has the repo with a live ``.git`` and the worktree on
disk, so the agent edits the candidate directly — no container needed. This is the
genuinely-runnable first path.

**B1 container (deferred):** the dedicated RW-repo container runner has the SAME
``command_runner`` shape (``cm.run_command`` under the hood); it lands when that
substrate is targeted (needs Docker). Nothing above the seam changes.

The runner enforces a total timeout and an idle timeout (no output for N seconds →
the agent is wedged → kill), mirroring the coder run engine, so a stuck agent
can't hang the loop forever.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Awaitable, Callable

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

OnChunk = Callable[[bytes], Awaitable[None]]


class RunnerError(RuntimeError):
    """The agent process could not be launched or was killed."""


async def _stream(proc: asyncio.subprocess.Process, on_chunk: OnChunk, *,
                  idle_timeout: float) -> None:
    """Pump the process stdout to ``on_chunk`` with an idle watchdog. A read that
    blocks longer than ``idle_timeout`` means the agent produced nothing for that
    long → treat as wedged and kill it."""
    assert proc.stdout is not None
    while True:
        try:
            chunk = await asyncio.wait_for(proc.stdout.read(4096), idle_timeout)
        except TimeoutError as exc:
            with contextlib.suppress(Exception):
                proc.kill()
            raise RunnerError(f"agent idle > {idle_timeout:.0f}s — killed") from exc
        if not chunk:
            return  # EOF — process is done writing
        await on_chunk(chunk)


def make_subprocess_runner(*, timeout: float = 900.0, idle_timeout: float = 180.0,
                           inherit_env: bool = True) -> Callable[..., Awaitable[None]]:
    """Build a host-subprocess ``command_runner``.

    Runs ``argv`` with ``cwd = request.candidate.path`` so the agent edits the
    isolated worktree, never the live tree. ``environment`` (the token + config
    dir from the driver) is merged over the inherited env. Raises ``RunnerError``
    on launch failure / timeout; the driver normalizes that into a failed edit."""
    async def _run(*, request, argv: list[str], on_chunk: OnChunk,
                   environment: dict | None = None) -> None:
        env = dict(os.environ) if inherit_env else {}
        env.update(environment or {})
        cwd = request.candidate.path
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=cwd, env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
        except Exception as exc:  # noqa: BLE001 — agent binary missing / not executable
            raise RunnerError(f"could not launch agent {argv[:1]}: {exc!r}") from exc
        try:
            await asyncio.wait_for(_stream(proc, on_chunk, idle_timeout=idle_timeout), timeout)
        except TimeoutError as exc:
            with contextlib.suppress(Exception):
                proc.kill()
            raise RunnerError(f"agent exceeded total timeout {timeout:.0f}s") from exc
        finally:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), 10.0)

    return _run


# The default dev-bind runner with standard budgets.
subprocess_command_runner = make_subprocess_runner()
