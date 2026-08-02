"""In-memory invite-token store for cast couch co-op.

Hosts mint short-TTL invite tokens that grant temporary "join this
session" auth to guests holding the token (typically scanned from a QR
code on the TV receiver). The token IS the credential — guest phones
don't carry session cookies — so the substrate must do all the auth
work the cookie layer would otherwise handle:

  * **Bound to a session_id**: a stolen token can only join the one
    session it was minted for. Switching games invalidates the token.
  * **Bound to a host_user_id**: revoke endpoints verify the caller
    owns the invite before letting them cancel it.
  * **Slot counter**: ``slots_remaining`` decrements on each successful
    claim. Hits zero → token is dead, can't admit more guests. Lets a
    single QR serve N joiners without a per-guest mint roundtrip.
  * **Short TTL**: defaults to 300s. Long enough for "scan as friends
    arrive" pacing, short enough that a leaked QR photo doesn't admit
    a stranger an hour later.

Why in-memory (not SQLite):

  Invites are intrinsically ephemeral — no co-op session in progress
  when augmentum restarts survives the restart anyway (game-stream
  containers stop, phones reconnect to nothing). Persistence buys
  nothing and adds a cleanup-on-restart problem. Same shape as
  :mod:`augmentum.cast.pair_store`.

See spec: ``docs/superpowers/specs/2026-06-02-cast-couch-coop-design.md``
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


_DEFAULT_TTL_S: float = 300.0
_MAX_ACTIVE_RECORDS: int = 128
_DEFAULT_MAX_SLOTS: int = 3


@dataclass(slots=True)
class InviteRecord:
    token: str
    session_id: str
    host_user_id: str
    expires_at: float
    slots_remaining: int
    # guest_profile_ids that have claimed this token. Phase 1 leaves
    # this empty (anonymous joins); Phase 2's claim flow populates it.
    # We track to (a) audit who joined via which invite and (b) prevent
    # a single guest from claiming twice on the same token.
    claimed_by: list[str] = field(default_factory=list)
    revoked: bool = False

    def is_expired(self, *, now: float | None = None) -> bool:
        return (now or time.time()) >= self.expires_at

    def is_active(self, *, now: float | None = None) -> bool:
        """Token can still admit a new guest."""
        return (
            not self.revoked
            and not self.is_expired(now=now)
            and self.slots_remaining > 0
        )


def _generate_token() -> str:
    """Mint a wsi_<24-hex> token. ``wsi`` = WebSocket Invite, matches
    the ``wsp_`` pair-token shape from pair_store for visual symmetry.
    """
    return f"wsi_{secrets.token_hex(12)}"


class InviteStore:
    """In-memory invite record store. Process-local, self-pruning.

    Thread/task safety: single-event-loop access only. No locks.
    Pattern matches :class:`augmentum.cast.pair_store.PairStore`.
    """

    def __init__(
        self,
        *,
        default_ttl_s: float = _DEFAULT_TTL_S,
        max_active: int = _MAX_ACTIVE_RECORDS,
        default_max_slots: int = _DEFAULT_MAX_SLOTS,
    ) -> None:
        self._records: dict[str, InviteRecord] = {}
        self._default_ttl = max(30.0, float(default_ttl_s or _DEFAULT_TTL_S))
        self._max_active = max(1, int(max_active or _MAX_ACTIVE_RECORDS))
        self._default_max_slots = max(1, int(default_max_slots))

    # ── Lifecycle ─────────────────────────────────────────────────────

    def mint(
        self,
        *,
        session_id: str,
        host_user_id: str,
        max_slots: int | None = None,
        ttl_s: float | None = None,
    ) -> InviteRecord:
        """Create a new invite for a session. Returns the record (with
        the freshly-minted token).

        ``max_slots`` caps how many guests can claim this token.
        ``ttl_s`` overrides the store default.
        """
        if not session_id or not host_user_id:
            raise ValueError("session_id and host_user_id required")
        self._prune()

        slots = self._default_max_slots if max_slots is None else int(max_slots)
        slots = max(1, slots)
        ttl = float(ttl_s) if ttl_s and ttl_s > 0 else self._default_ttl

        # Token collisions are vanishingly rare with 24-hex keyspace but
        # the loop costs nothing.
        for _ in range(8):
            token = _generate_token()
            if token not in self._records:
                break
        else:
            token = f"wsi_{secrets.token_hex(20)}"

        record = InviteRecord(
            token=token,
            session_id=session_id,
            host_user_id=host_user_id,
            expires_at=time.time() + ttl,
            slots_remaining=slots,
        )
        self._records[token] = record
        log.info(
            "cast_invite_minted",
            token_prefix=token[:12], session_id=session_id,
            host_user_id=host_user_id, max_slots=slots, ttl_s=ttl,
        )
        return record

    def claim(
        self,
        token: str,
        *,
        guest_profile_id: str = "",
    ) -> InviteRecord | None:
        """Try to consume one slot. Returns the record on success,
        ``None`` if the token is unknown, expired, revoked, or full.

        Decrements ``slots_remaining``. The record is returned even
        when it hits 0 — the caller needs it to attach the phone. A
        subsequent claim() call on a 0-slot record returns ``None``.

        ``guest_profile_id`` is recorded for audit. Phase 1 passes ""
        (anonymous); Phase 2 passes the resolved guest profile.
        """
        record = self._records.get(token)
        if record is None:
            return None
        if not record.is_active():
            return None
        # Phase 2: refuse same-profile re-claim. Profile already has
        # a slot via an earlier claim on this token — bouncing should
        # use the existing attachment, not double-spend a slot.
        if guest_profile_id and guest_profile_id in record.claimed_by:
            return None
        record.slots_remaining -= 1
        if guest_profile_id:
            record.claimed_by.append(guest_profile_id)
        log.info(
            "cast_invite_claimed",
            token_prefix=token[:12], session_id=record.session_id,
            guest_profile_id=guest_profile_id or "anon",
            slots_remaining=record.slots_remaining,
        )
        return record

    def get(self, token: str) -> InviteRecord | None:
        """Read-only lookup. Returns None for unknown/expired tokens
        but does NOT mark expired records — that's :meth:`claim`'s job.
        """
        record = self._records.get(token)
        if record is None:
            return None
        if record.is_expired():
            return None
        return record

    def revoke(self, token: str, *, host_user_id: str = "") -> bool:
        """Cancel an invite. Returns True if removed.

        ``host_user_id`` is enforced when non-empty — defends against a
        cross-user revoke if a token leaks to another logged-in user.
        Pass "" for system-side revoke (e.g. session-stop sweep).
        """
        record = self._records.get(token)
        if record is None:
            return False
        if host_user_id and record.host_user_id != host_user_id:
            return False
        record.revoked = True
        self._records.pop(token, None)
        log.info(
            "cast_invite_revoked",
            token_prefix=token[:12], session_id=record.session_id,
            host_user_id=record.host_user_id,
        )
        return True

    def revoke_for_session(self, session_id: str) -> int:
        """Sweep all invites for a session. Returns count removed.

        Called when a game-stream session ends — no point keeping
        invites alive that point at a dead container.
        """
        targets = [
            t for t, r in self._records.items()
            if r.session_id == session_id
        ]
        for token in targets:
            record = self._records.pop(token, None)
            if record is not None:
                record.revoked = True
        if targets:
            log.info(
                "cast_invite_session_revoked",
                session_id=session_id, count=len(targets),
            )
        return len(targets)

    def list_for_session(self, session_id: str) -> list[InviteRecord]:
        return [
            r for r in self._records.values()
            if r.session_id == session_id and r.is_active()
        ]

    def list_for_host(self, host_user_id: str) -> list[InviteRecord]:
        return [
            r for r in self._records.values()
            if r.host_user_id == host_user_id and r.is_active()
        ]

    # ── Bookkeeping ───────────────────────────────────────────────────

    def _prune(self) -> None:
        now = time.time()
        expired = [
            t for t, r in self._records.items()
            if r.is_expired(now=now) or r.revoked or r.slots_remaining <= 0
        ]
        for token in expired:
            self._records.pop(token, None)
        if len(self._records) >= self._max_active:
            # Oldest-first eviction.
            ordered = sorted(
                self._records.items(), key=lambda kv: kv[1].expires_at,
            )
            overflow = len(self._records) - self._max_active + 1
            for token, _r in ordered[:overflow]:
                self._records.pop(token, None)
