"""Append-only NDJSON live-log writer and tailing reader.

The live log is the universe model of the agent. Every other component
(orchestrator, fast-path rule engine, slow-path planner, debugger,
replay tool, evaluation harness) reads or writes through this module.

Design choices
--------------
* **Append-only.** Lines are written and never rewritten in place. The
  on-disk file is the canonical truth; in-memory caches are
  convenience.
* **Line-buffered fsync-on-flush.** Writes hit the OS page cache
  immediately; ``flush()`` calls ``os.fsync`` to durably persist. Most
  call sites do not need durability per write -- the orchestrator
  flushes at end-of-tick.
* **Schema validation at the boundary.** Every entry is validated
  against :data:`augmentum.game_agent.schema.LogEntry` before being
  serialized, so the file is always parseable by future tools.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator, Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from augmentum.game_agent.schema import LogEntry

_LogEntryAdapter = TypeAdapter(LogEntry)


def now_ms() -> int:
    """Wall-clock milliseconds. Use :class:`SessionClock` for in-session timing."""

    return int(time.time() * 1000)


class SessionClock:
    """Monotonic per-session millisecond clock.

    Use when:
    - Stamping log entries; ``t=0`` corresponds to ``SessionClock.start()``.

    Expects:
    - The session is no longer than 2^31 ms (~24 days). Practical sessions
      are minutes to hours.

    Returns:
    - Milliseconds elapsed since :meth:`start` was called.
    """

    def __init__(self) -> None:
        self._origin_ns: int | None = None

    def start(self) -> None:
        self._origin_ns = time.monotonic_ns()

    def elapsed_ms(self) -> int:
        if self._origin_ns is None:
            raise RuntimeError("SessionClock not started")
        return (time.monotonic_ns() - self._origin_ns) // 1_000_000


class LiveLog:
    """Append-only NDJSON writer for one session.

    Use when:
    - The orchestrator needs to record session events, agent plans,
      adapter inputs, and fast-path rule firings.

    Expects:
    - One :class:`LiveLog` instance per session.
    - The parent directory exists or is creatable.

    Returns:
    - Validated NDJSON lines on disk, one per call to :meth:`append`.
    """

    def __init__(self, path: Path | str, *, clock: SessionClock | None = None) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or SessionClock()
        # Line-buffered so each NDJSON line hits the page cache on write,
        # avoiding partial-line garbage if the process crashes mid-session.
        self._fh = self._path.open("a", buffering=1, encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def clock(self) -> SessionClock:
        return self._clock

    def append(self, entry_dict: dict[str, Any]) -> None:
        """Validate and write one entry.

        ``entry_dict`` is validated against the discriminated
        :data:`LogEntry` union; an invalid payload raises
        :class:`pydantic.ValidationError` and is *not* written.
        """

        validated = _LogEntryAdapter.validate_python(entry_dict)
        line = json.dumps(_LogEntryAdapter.dump_python(validated), separators=(",", ":"))
        self._fh.write(line + "\n")

    def flush(self) -> None:
        """Force kernel page cache to disk."""

        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        self.flush()
        self._fh.close()

    def __enter__(self) -> LiveLog:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def read_log(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream a closed session log as raw dicts (validation deferred).

    Use when:
    - Tooling (replayers, analytics) needs to walk a finished log
      without caring about strict typing. Tools that *do* care should
      pass each dict through ``TypeAdapter(LogEntry).validate_python``.

    Expects:
    - The file is well-formed NDJSON. Blank lines are skipped.

    Returns:
    - One dict per line, in file order.
    """

    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)


async def tail_log(
    path: Path | str,
    *,
    poll_interval_s: float = 0.05,
    from_start: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    """Follow a live log as new lines appear.

    Use when:
    - The fast-path rule engine or a live dashboard wants to react to
      events as they're written, without holding a reference to the
      :class:`LiveLog` writer.

    Expects:
    - ``path`` is an open session log being written by another task.
    - The caller breaks out via cancellation or normal completion.

    Returns:
    - Yields one dict per line as it lands. Yields forever until the
      consumer stops awaiting; there is no built-in EOF.
    """

    import asyncio  # local import: this function is the only async surface

    p = Path(path)
    # Wait for file to exist if writer hasn't created it yet.
    while not p.exists():
        await asyncio.sleep(poll_interval_s)

    with p.open("r", encoding="utf-8") as fh:
        if not from_start:
            fh.seek(0, os.SEEK_END)
        while True:
            line = fh.readline()
            if not line:
                await asyncio.sleep(poll_interval_s)
                continue
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def validate_entries(entries: Iterable[dict[str, Any]]) -> list[LogEntry]:
    """Strict-validate an iterable of raw entries.

    Raises on the first invalid entry; useful for tests and CI gates
    that want to assert a log is well-formed end-to-end.
    """

    return [_LogEntryAdapter.validate_python(e) for e in entries]
