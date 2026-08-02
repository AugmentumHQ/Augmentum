"""TCP/UDP port pool for game-stream containers.

Each container needs at least two ports: one for the WebRTC signaling
endpoint (Selkies' HTTP+WebSocket front door) and one for the game
itself (e.g. Luanti server on UDP 30000). The pool hands them out
contiguously per session and recycles on release.

The pool persists nothing -- it walks live sessions in the store on
``reconcile()`` to rebuild its in-use set on startup. This keeps the
schema simpler and avoids drift between "what the DB thinks is in use"
and "what's actually bound".
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class PortPoolExhausted(RuntimeError):
    """Raised when no ports are free."""


@dataclass(frozen=True)
class PortAllocation:
    stream_port: int
    game_port: int


class PortPool:
    """Allocator over a contiguous port range.

    Each allocation reserves *two* ports: ``stream_port`` and
    ``game_port`` = ``stream_port + 1``. The pool steps in increments
    of 2 so the pair is always aligned.
    """

    def __init__(self, base: int = 30000, count: int = 50) -> None:
        if count % 2 != 0:
            raise ValueError("port pool count must be even (pair allocation)")
        self._base = base
        self._count = count
        self._lock = asyncio.Lock()
        # Set of stream_port values currently allocated.
        self._in_use: set[int] = set()

    @property
    def capacity(self) -> int:
        """Maximum number of concurrent allocations."""
        return self._count // 2

    @property
    def in_use(self) -> int:
        return len(self._in_use)

    async def allocate(self) -> PortAllocation:
        async with self._lock:
            for offset in range(0, self._count, 2):
                stream_port = self._base + offset
                if stream_port not in self._in_use:
                    self._in_use.add(stream_port)
                    return PortAllocation(
                        stream_port=stream_port,
                        game_port=stream_port + 1,
                    )
            raise PortPoolExhausted(
                f"no free ports in pool {self._base}..{self._base + self._count - 1}"
            )

    async def release(self, stream_port: int) -> None:
        async with self._lock:
            self._in_use.discard(stream_port)

    async def reserve(self, stream_port: int) -> bool:
        """Manually mark a port pair as in-use (used by reconcile())."""
        if not (self._base <= stream_port < self._base + self._count):
            return False
        async with self._lock:
            if stream_port in self._in_use:
                return False
            self._in_use.add(stream_port)
            return True

    async def reconcile(self, live_sessions: list[dict]) -> None:
        """Rebuild ``_in_use`` from store rows after a server restart."""
        async with self._lock:
            self._in_use.clear()
            for row in live_sessions:
                sp = row.get("stream_port")
                if isinstance(sp, int) and self._base <= sp < self._base + self._count:
                    self._in_use.add(sp)
