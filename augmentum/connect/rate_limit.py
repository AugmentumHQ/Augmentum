"""Per-user, per-verb-category sliding-window rate limiter for the
Connect signaling WebSocket.

The HTTP rate-limit middleware (``augmentum/proxy/middleware/
rate_limit.py``) explicitly skips WebSockets — it only inspects
``scope["type"] == "http"``. That leaves the Connect WS verbs
(text_send, typing, invite/offer/answer/ICE, etc.) un-throttled. A
single buggy or hostile client can flood the hub with envelopes
and starve every other user on the box.

This module fills the gap with a tiny in-memory limiter keyed by
``(user_id, category)``. State outlives any individual WS
connection (a reconnect doesn't reset the bucket) so a misbehaving
client can't escape the limit by churning sockets. State is purely
per-process — fine for single-instance Augmentum; the multi-tenant
+ federation story for limits is a Tier-2 concern.

Categories and defaults are tuned conservatively for the
single-user-with-housemates target audience. The categories aren't
exposed to the wire — they're derived from the verb. Limits live
as a dict so a future settings page (or env override) can tune
them without code change.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable

from augmentum.connect.protocol import (
    MSG_ACCEPT,
    MSG_ANSWER,
    MSG_CANDIDATES,
    MSG_DECLINE,
    MSG_HANGUP,
    MSG_INVITE,
    MSG_NEGOTIATE,
    MSG_OFFER,
    MSG_PING,
    MSG_PRESENCE_ANNOUNCE,
    MSG_SELECT_ANSWER,
    MSG_TEXT_DELETE,
    MSG_TEXT_DELIVERED,
    MSG_TEXT_EDIT,
    MSG_TEXT_REACT,
    MSG_TEXT_READ,
    MSG_TEXT_SEND,
    MSG_TYPING_START,
    MSG_TYPING_STOP,
)

# ── Category mapping ─────────────────────────────────────────────


CATEGORY_TEXT_WRITE = "text_write"
"""Outbound text traffic that creates persisted rows."""

CATEGORY_TEXT_ACK = "text_ack"
"""Receipts (read, delivered) — cheap, batchable, high cap."""

CATEGORY_EPHEMERAL = "ephemeral"
"""Typing indicators — high frequency, no persistence."""

CATEGORY_CALL_CONTROL = "call_control"
"""Invite / accept / decline / hangup / select_answer / negotiate."""

CATEGORY_CALL_HANDSHAKE = "call_handshake"
"""Offer / answer / ICE candidates — chatty during setup."""

CATEGORY_PRESENCE = "presence"
"""Presence + announce + ping — control plane."""

CATEGORY_OTHER = "other"
"""Fallback for unknown verbs."""


_VERB_CATEGORIES: dict[str, str] = {
    MSG_TEXT_SEND:        CATEGORY_TEXT_WRITE,
    MSG_TEXT_EDIT:        CATEGORY_TEXT_WRITE,
    MSG_TEXT_DELETE:      CATEGORY_TEXT_WRITE,
    MSG_TEXT_REACT:       CATEGORY_TEXT_WRITE,
    MSG_TEXT_READ:        CATEGORY_TEXT_ACK,
    MSG_TEXT_DELIVERED:   CATEGORY_TEXT_ACK,
    MSG_TYPING_START:     CATEGORY_EPHEMERAL,
    MSG_TYPING_STOP:      CATEGORY_EPHEMERAL,
    MSG_INVITE:           CATEGORY_CALL_CONTROL,
    MSG_ACCEPT:           CATEGORY_CALL_CONTROL,
    MSG_DECLINE:          CATEGORY_CALL_CONTROL,
    MSG_HANGUP:           CATEGORY_CALL_CONTROL,
    MSG_SELECT_ANSWER:    CATEGORY_CALL_CONTROL,
    MSG_NEGOTIATE:        CATEGORY_CALL_CONTROL,
    MSG_OFFER:            CATEGORY_CALL_HANDSHAKE,
    MSG_ANSWER:           CATEGORY_CALL_HANDSHAKE,
    MSG_CANDIDATES:       CATEGORY_CALL_HANDSHAKE,
    MSG_PING:             CATEGORY_PRESENCE,
    MSG_PRESENCE_ANNOUNCE: CATEGORY_PRESENCE,
}


# Reqs per 60s window. Sized to comfortably exceed normal human use
# while clipping flooding clients before they degrade the hub.
DEFAULT_LIMITS: dict[str, int] = {
    CATEGORY_TEXT_WRITE:     60,    # 1/s sustained; bursts above raise EVENT_ERROR
    CATEGORY_TEXT_ACK:       240,   # cheap; catch-up batches one ack per inbound
    CATEGORY_EPHEMERAL:      600,   # 10/s — typing debounce is the client's job
    CATEGORY_CALL_CONTROL:   60,    # 1/s — humans don't redial that fast
    CATEGORY_CALL_HANDSHAKE: 300,   # ICE batches chatter heavily during setup
    CATEGORY_PRESENCE:       120,
    CATEGORY_OTHER:          60,
}


WINDOW_SECONDS = 60


# ── Limiter ──────────────────────────────────────────────────────


class WsRateLimiter:
    """Sliding-window counter keyed by (user_id, category).

    Single-process; state outlives any one WS connection so reconnect
    churn doesn't escape the limit. Lock-free — assumes single-thread
    asyncio loop on the WS receive side, which matches how the rest
    of Augmentum's WS endpoints are structured. If we ever introduce
    a worker pool that reads from the same WS, wrap ``check`` with a
    lock.
    """

    def __init__(self, limits: dict[str, int] | None = None) -> None:
        self._limits = {**DEFAULT_LIMITS, **(limits or {})}
        self._windows: dict[tuple[str, str], deque] = defaultdict(deque)
        # Cleanup countdown — periodically prunes idle keys so a
        # one-message-per-week user doesn't keep a deque alive forever.
        self._tick = 0

    def category_for(self, verb: str) -> str:
        """Public — useful for tests and metric tags."""

        return _VERB_CATEGORIES.get(verb, CATEGORY_OTHER)

    def check(self, *, user_id: str, verb: str) -> tuple[bool, str, int]:
        """Record an attempt and return ``(allowed, category, retry_after_s)``.

        ``retry_after_s`` is 0 when allowed; otherwise the integer
        seconds until the oldest in-window entry falls out (the
        bucket has room again).

        An empty ``user_id`` means we can't bucket the request — we
        still let it through because the WS endpoint shouldn't even
        have got past auth at that point; refusing here would mask
        the real bug. The caller is expected to enforce auth itself.
        """

        category = _VERB_CATEGORIES.get(verb, CATEGORY_OTHER)
        if not user_id:
            return True, category, 0

        limit = self._limits.get(category, DEFAULT_LIMITS[CATEGORY_OTHER])
        if limit <= 0:
            # 0 = disabled bucket; let everything through.
            return True, category, 0

        now = time.monotonic()
        cutoff = now - WINDOW_SECONDS
        key = (user_id, category)
        window = self._windows[key]

        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= limit:
            # Bucket full. Suggest how long until the oldest entry
            # falls out + 1 (so a client polling at retry_after won't
            # land in the same overflow boundary).
            retry_after = max(1, int(window[0] + WINDOW_SECONDS - now) + 1)
            self._maybe_cleanup()
            return False, category, retry_after

        window.append(now)
        self._maybe_cleanup()
        return True, category, 0

    def _maybe_cleanup(self) -> None:
        self._tick += 1
        if self._tick < 200:
            return
        self._tick = 0
        cutoff = time.monotonic() - WINDOW_SECONDS
        empty: list[tuple[str, str]] = []
        for key, window in self._windows.items():
            while window and window[0] < cutoff:
                window.popleft()
            if not window:
                empty.append(key)
        for key in empty:
            del self._windows[key]

    # Test/observability helpers.

    def categories(self) -> Iterable[str]:
        return self._limits.keys()

    def limit_for(self, category: str) -> int:
        return self._limits.get(category, 0)

    def snapshot(self, user_id: str) -> dict[str, int]:
        """Current bucket fill (post-cleanup) for one user. Useful for
        debug endpoints and tests."""

        cutoff = time.monotonic() - WINDOW_SECONDS
        out: dict[str, int] = {}
        for (uid, cat), window in self._windows.items():
            if uid != user_id:
                continue
            while window and window[0] < cutoff:
                window.popleft()
            out[cat] = len(window)
        return out


class KeyedRateLimiter:
    """General per-key sliding-window limiter for PUBLIC, unauthenticated HTTP
    endpoints (invite preview/claim, guest session) — keyed by client IP.

    These endpoints can't lean on per-user auth, so a token flood from one
    address is throttled here. Token entropy still guards correctness; this just
    clips the nuisance. In-memory, single-process; clock injectable for tests.
    """

    def __init__(
        self, *, limit: int, window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit
        self._window_s = window_s
        self._clock = clock
        self._windows: dict[str, deque] = defaultdict(deque)
        self._tick = 0

    def check(self, key: str) -> tuple[bool, int]:
        """Record an attempt; return ``(allowed, retry_after_s)``.

        An empty key (no resolvable IP) is allowed through — refusing would block
        legit traffic on a misconfigured proxy more than it stops abuse.
        """
        if not key or self._limit <= 0:
            return True, 0
        now = self._clock()
        cutoff = now - self._window_s
        window = self._windows[key]
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= self._limit:
            retry_after = max(1, int(window[0] + self._window_s - now) + 1)
            self._maybe_cleanup(now)
            return False, retry_after
        window.append(now)
        self._maybe_cleanup(now)
        return True, 0

    def _maybe_cleanup(self, now: float) -> None:
        self._tick += 1
        if self._tick < 256:
            return
        self._tick = 0
        cutoff = now - self._window_s
        empty = []
        for key, window in self._windows.items():
            while window and window[0] < cutoff:
                window.popleft()
            if not window:
                empty.append(key)
        for key in empty:
            del self._windows[key]
