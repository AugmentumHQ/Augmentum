"""In-memory pair-token store for the cast receiver bootstrap flow.

The receiver flow:

  1. Receiver page (no auth) POSTs /api/cast/pair/start. We mint a
     short user-facing code + a QR encoding a phone-friendly URL,
     and return both. The pair record state is "pending".
  2. User scans the QR with their phone (already logged into
     Augmentum) → phone loads /ui/cast-pair/?code=CODE → that page
     POSTs /api/cast/pair/approve/{code} carrying the session cookie.
     We bind the record to the user_id and issue a single-use
     ws_token. State becomes "approved".
  3. Receiver page polls /api/cast/pair/poll/{code} and on seeing
     state="approved" pulls the ws_token, opens its WebSocket with
     ?token=<ws_token>. Consuming the token flips state to
     "consumed" so it can't be replayed.

Why in-memory:

  Pair codes live ~2 minutes. Cross-restart persistence buys
  nothing — every pairing in progress when augmentum restarts has
  already broken from the receiver's perspective. Bounded + self-
  pruning, same shape as render output store and cast tokens.

Security shape:

  - 8-char pair code (alphanumeric upper) — short enough for a
    fallback "type this in" entry path; long enough that brute-
    force during a 120s window is impractical.
  - 32-hex ws_token — full crypto strength for the auth bearer.
  - Single-use: consume_token() trips the state to "consumed".
  - TTL: 120s default. Long enough for a relaxed phone scan;
    short enough that codes don't sit around if pairing aborts.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


_DEFAULT_TTL_S: float = 120.0
_MAX_ACTIVE_RECORDS: int = 64
_PAIR_CODE_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no I/O/0/1 — readability


STATE_PENDING = "pending"
STATE_APPROVED = "approved"
STATE_CONSUMED = "consumed"
STATE_EXPIRED = "expired"


@dataclass(slots=True)
class PairRecord:
    pair_code: str
    state: str
    expires_at: float
    user_id: str = ""
    ws_token: str = ""
    # Where the approving user said this TV lives — chosen on the phone
    # at approve time, every pairing. "home" → long-lived session (the TV
    # stays a passive cast target, silent reconnect across restarts);
    # "away" → short-lived (re-pair sooner; don't leave a long credential
    # on a screen outside the house). establish-session maps this to the
    # session TTL + cookie max_age. Defaults to the SAFE/short option so a
    # future approve() that forgets to pass a choice can't silently mint a
    # year-long credential — the route + approve clamp already default to
    # "away"; this makes the dataclass agree.
    lifetime: str = "away"
    # True once the receiver has redeemed its HTTP session cookie via
    # /api/cast/pair/establish-session. Single-use — second call with
    # the same ws_token is rejected so a leaked token can't mint
    # multiple long-lived sessions.
    session_established: bool = False

    def is_expired(self, *, now: float | None = None) -> bool:
        return (now or time.time()) >= self.expires_at


def _generate_pair_code(length: int = 8) -> str:
    return "".join(secrets.choice(_PAIR_CODE_CHARS) for _ in range(length))


class PairStore:
    """In-memory pair record store. Process-local, self-pruning."""

    def __init__(
        self,
        *,
        default_ttl_s: float = _DEFAULT_TTL_S,
        max_active: int = _MAX_ACTIVE_RECORDS,
    ) -> None:
        self._records: dict[str, PairRecord] = {}
        self._tokens: dict[str, str] = {}  # ws_token → pair_code
        self._default_ttl = max(10.0, float(default_ttl_s or _DEFAULT_TTL_S))
        self._max_active = max(1, int(max_active or _MAX_ACTIVE_RECORDS))

    def start(self, *, ttl_s: float | None = None) -> PairRecord:
        """Create a new pair record in the pending state."""
        self._prune()

        # Regenerate on collision (vanishingly rare with 32^8 keyspace).
        for _ in range(8):
            code = _generate_pair_code()
            if code not in self._records:
                break
        else:
            # Wildly unlikely — defensive: bump the length until we find one.
            code = _generate_pair_code(12)

        ttl = float(ttl_s) if ttl_s and ttl_s > 0 else self._default_ttl
        record = PairRecord(
            pair_code=code,
            state=STATE_PENDING,
            expires_at=time.time() + ttl,
        )
        self._records[code] = record
        return record

    def approve(
        self, pair_code: str, *, user_id: str, lifetime: str = "away",
    ) -> PairRecord | None:
        """Approve a pending record for ``user_id``. Returns None if
        the code is unknown, expired, or already in a non-pending
        state. Idempotency would be tempting here but defensive
        rejection is safer — a second-call from a different account
        must not silently steal the binding.

        ``lifetime`` ("home" | "away") is the user's at-approve-time choice
        of how long this TV stays signed in; anything unrecognised falls
        back to "away" (the safer/shorter option).
        """
        record = self._records.get(pair_code)
        if record is None:
            return None
        if record.is_expired():
            record.state = STATE_EXPIRED
            return None
        if record.state != STATE_PENDING:
            return None
        if not user_id:
            return None
        record.state = STATE_APPROVED
        record.user_id = user_id
        record.lifetime = lifetime if lifetime in ("home", "away") else "away"
        record.ws_token = f"wsp_{secrets.token_hex(16)}"
        self._tokens[record.ws_token] = pair_code
        log.info(
            "cast_pair_approved",
            pair_code=pair_code, user_id=user_id,
        )
        return record

    def poll(self, pair_code: str) -> PairRecord | None:
        """Return the current record or None when expired/unknown."""
        record = self._records.get(pair_code)
        if record is None:
            return None
        if record.is_expired() and record.state == STATE_PENDING:
            record.state = STATE_EXPIRED
        return record

    def mark_session_established(self, ws_token: str) -> PairRecord | None:
        """Validate a ws_token for HTTP-session redemption.

        Called by /api/cast/pair/establish-session BEFORE the WS open,
        so the receiver's WebView gets a long-lived auth cookie for
        all subsequent HTTP requests (surface URLs, /api/* calls from
        within iframe surfaces, etc.). The ws_token itself is NOT
        consumed here — the WS handshake still claims it. This is
        single-use per pair record: a second establish-session call
        with the same token is rejected so the redemption can only
        bootstrap one HTTP session, even if the token leaks.
        """
        if not ws_token:
            return None
        pair_code = self._tokens.get(ws_token)
        if pair_code is None:
            return None
        record = self._records.get(pair_code)
        if record is None:
            return None
        if record.is_expired():
            record.state = STATE_EXPIRED
            return None
        if record.state != STATE_APPROVED:
            return None
        if record.session_established:
            return None
        record.session_established = True
        log.info(
            "cast_pair_session_established",
            pair_code=pair_code, user_id=record.user_id,
        )
        return record

    def consume_token(self, ws_token: str) -> PairRecord | None:
        """Validate a ws_token, mark consumed, return the record.

        Single-use — a token consumed once is dead. The WS endpoint
        calls this once at handshake time; subsequent calls with the
        same token must fail so a leaked token can only auth one
        connection.
        """
        if not ws_token:
            return None
        pair_code = self._tokens.get(ws_token)
        if pair_code is None:
            return None
        record = self._records.get(pair_code)
        if record is None:
            self._tokens.pop(ws_token, None)
            return None
        if record.is_expired():
            record.state = STATE_EXPIRED
            return None
        if record.state != STATE_APPROVED:
            return None
        record.state = STATE_CONSUMED
        # Token is dead — drop the index so the keyspace stays clean.
        self._tokens.pop(ws_token, None)
        log.info(
            "cast_pair_token_consumed",
            pair_code=pair_code, user_id=record.user_id,
        )
        return record

    def revoke(self, pair_code: str) -> bool:
        """Drop a record explicitly. Returns True if removed."""
        record = self._records.pop(pair_code, None)
        if record is None:
            return False
        if record.ws_token:
            self._tokens.pop(record.ws_token, None)
        return True

    def _prune(self) -> None:
        now = time.time()
        expired = [
            code for code, r in self._records.items()
            if r.expires_at <= now or r.state in (STATE_CONSUMED, STATE_EXPIRED)
            and r.expires_at <= now
        ]
        for code in expired:
            r = self._records.pop(code)
            if r.ws_token:
                self._tokens.pop(r.ws_token, None)
        if len(self._records) >= self._max_active:
            # Oldest-first eviction; tokens index follows.
            ordered = sorted(self._records.items(), key=lambda kv: kv[1].expires_at)
            overflow = len(self._records) - self._max_active + 1
            for code, r in ordered[:overflow]:
                self._records.pop(code, None)
                if r.ws_token:
                    self._tokens.pop(r.ws_token, None)
