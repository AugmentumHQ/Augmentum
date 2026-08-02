"""In-memory store for rendered cast output bytes.

Renders (HTML→PNG, VRM frame→PNG, future video chunks) produce bytes
that a downstream consumer — TV receiver, browser <img>, screencap
endpoint — fetches via a tokenised URL. This module owns that
indirection: store(bytes, ...) → token, fetch(token) → bytes.

Why in-memory:

  Rendered output is ephemeral. A cast image is meaningful for as
  long as the TV is displaying it; once it's gone or replaced, the
  bytes are dead weight. SQLite-backing buys nothing — augmentum
  restarts mean the cast is over anyway — and adds cleanup chores.
  In-memory with TTL eviction matches the lifecycle.

Bounded + self-pruning to match the cast_tokens.py pattern. A few
hundred concurrent outputs is the absolute ceiling, far below any
sensible deployment's working set.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


_DEFAULT_TTL_S: float = 5 * 60.0      # 5 min — long enough for a TV fetch
_MAX_ACTIVE_OUTPUTS: int = 256        # bytes are bigger than perm tokens


@dataclass(slots=True)
class RenderOutput:
    token: str
    content_type: str            # MIME, e.g. "image/png", "video/mp4"
    body: bytes
    expires_at: float
    user_id: str = ""            # owner — bounds blast-radius if leaked
    single_use: bool = False     # delete on first fetch
    metadata: dict[str, Any] | None = None

    def is_expired(self, *, now: float | None = None) -> bool:
        return (now or time.time()) >= self.expires_at


class RenderOutputStore:
    """In-memory rendered output store. TTL-evicted, capped.

    Thread-/task-safe for typical FastAPI single-event-loop usage: every
    method is sync and dict mutations are single bytecode ops in CPython.
    For background-task writers + request-task readers we rely on the
    GIL holding through dict.__setitem__ + dict.get, same model as
    cast_tokens.py.
    """

    def __init__(
        self,
        *,
        default_ttl_s: float = _DEFAULT_TTL_S,
        max_active: int = _MAX_ACTIVE_OUTPUTS,
    ) -> None:
        self._outputs: dict[str, RenderOutput] = {}
        self._default_ttl = max(10.0, float(default_ttl_s or _DEFAULT_TTL_S))
        self._max_active = max(1, int(max_active or _MAX_ACTIVE_OUTPUTS))

    def store(
        self,
        *,
        body: bytes,
        content_type: str = "application/octet-stream",
        user_id: str = "",
        ttl_s: float | None = None,
        single_use: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> RenderOutput:
        """Stash bytes + return the RenderOutput (caller reads `.token`).

        Self-prunes when over capacity: oldest expirations evicted
        first; falls back to oldest-by-insertion when nothing has
        expired. Never blocks; eviction is O(N) over the current set
        which is bounded by max_active.
        """
        self._prune()

        token = f"ro_{secrets.token_hex(12)}"
        now = time.time()
        out = RenderOutput(
            token=token,
            content_type=content_type or "application/octet-stream",
            body=body or b"",
            expires_at=now + (float(ttl_s) if ttl_s and ttl_s > 0 else self._default_ttl),
            user_id=user_id,
            single_use=bool(single_use),
            metadata=dict(metadata) if metadata else None,
        )
        self._outputs[token] = out
        return out

    def fetch(self, token: str) -> RenderOutput | None:
        """Return the stored output for ``token`` or None.

        Drops expired entries lazily. Single-use entries delete on
        successful fetch — even if the caller throws after we return.
        """
        out = self._outputs.get(token)
        if out is None:
            return None
        if out.is_expired():
            del self._outputs[token]
            return None
        if out.single_use:
            del self._outputs[token]
        return out

    def revoke(self, token: str) -> bool:
        """Drop a token explicitly. Returns True if it was present."""
        return self._outputs.pop(token, None) is not None

    def _prune(self) -> None:
        """Drop expired entries; cap at max_active by oldest-first."""
        now = time.time()
        expired = [tok for tok, out in self._outputs.items() if out.expires_at <= now]
        for tok in expired:
            del self._outputs[tok]
        if len(self._outputs) >= self._max_active:
            # Drop the oldest-expiring entries to make room.
            ordered = sorted(self._outputs.items(), key=lambda kv: kv[1].expires_at)
            overflow = len(self._outputs) - self._max_active + 1
            for tok, _out in ordered[:overflow]:
                del self._outputs[tok]


