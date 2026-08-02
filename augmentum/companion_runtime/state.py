"""CompanionState — 3-axis state machine (state x role x focus).

Three orthogonal axes per the design spec section 5:

- ``state`` (discrete): ``asleep | dormant | present``. Cooldown 2.0s.
- ``role`` (3-vector): ``(active, passive, reflective)`` summing to ~1.0.
  Cooldown 0.4s. Dominant axis is the argmax.
- ``focus`` (discrete with payload): ``none | <kind>:<payload>`` where
  kind ∈ ``{personal, shared, executive, social}``. Cooldown 0.8s.

State and focus are exclusive (one value at a time). Role is blended;
subagents see the full vector and branch on dominant role or weight
behavior continuously.

The cooldown discipline and listener-snapshot pattern is ported from
``ui/scripts/avatar-fsm.js``. Listener errors are isolated per-callback
so one bad subscriber cannot break the transition path.

Persistence: current state lives in ``companion_state`` (one row per
companion). Transition history is append-only in ``companion_state_log``.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Awaitable, Callable

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.state.backends.sqlite import SQLiteBackend

log = get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────

# Per-axis cooldowns in seconds (from design spec §5).
COOLDOWN_STATE_S = 2.0
COOLDOWN_ROLE_S = 0.4
COOLDOWN_FOCUS_S = 0.8


def _age_seconds(ts_text: str | None) -> float:
    """Seconds elapsed since a DB wall-clock timestamp (``datetime('now')``
    UTC text). Returns 0.0 on missing/unparseable input or a future
    timestamp (clock skew) — i.e. 'just entered', the safe default."""
    if not ts_text:
        return 0.0
    try:
        t = (
            datetime.strptime(
                str(ts_text).replace("T", " ").split(".", 1)[0],
                "%Y-%m-%d %H:%M:%S",
            )
            .replace(tzinfo=UTC)
            .timestamp()
        )
    except (ValueError, AttributeError):
        return 0.0
    return max(0.0, time.time() - t)

# Role-vector sum invariant tolerance.
ROLE_SUM_EPSILON = 0.01


# ── Enums + value objects ─────────────────────────────────────────────

class AttentionState(StrEnum):
    """Axis 1: discrete attention state."""
    ASLEEP = "asleep"        # runtime suspended; only dream housekeeping
    DORMANT = "dormant"      # awake but no presence (no devices connected)
    PRESENT = "present"      # at least one device subscribed; she's here


class FocusKind(StrEnum):
    """Axis 3: discrete focus kind."""
    NONE = "none"
    PERSONAL = "personal"        # her own activity
    SHARED = "shared"            # co-attending with user
    EXECUTIVE = "executive"      # task focus
    SOCIAL = "social"            # interpersonal with specific user


@dataclass(frozen=True, slots=True)
class RoleVector:
    """Axis 2: soft 3-vector summing to ~1.0.

    Dominant role is ``argmax``. Subagents may use the dominant for
    branching or the full weights for continuous scaling (e.g.,
    verbosity, latency targets, memory retrieval depth).
    """
    active: float
    passive: float
    reflective: float

    def __post_init__(self) -> None:
        total = self.active + self.passive + self.reflective
        if abs(total - 1.0) > ROLE_SUM_EPSILON:
            raise ValueError(
                f"RoleVector must sum to 1.0 ± {ROLE_SUM_EPSILON}; got {total:.4f}",
            )
        for name, v in (("active", self.active), ("passive", self.passive),
                        ("reflective", self.reflective)):
            if v < 0.0 or v > 1.0:
                raise ValueError(f"RoleVector.{name} must be in [0,1]; got {v}")

    def dominant(self) -> str:
        """Argmax of the three components."""
        vals = (("active", self.active), ("passive", self.passive),
                ("reflective", self.reflective))
        return max(vals, key=lambda p: p[1])[0]

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.active, self.passive, self.reflective)

    def as_json(self) -> str:
        return json.dumps({
            "active": self.active,
            "passive": self.passive,
            "reflective": self.reflective,
        })

    @classmethod
    def passive_dominant(cls) -> RoleVector:
        return cls(active=0.0, passive=1.0, reflective=0.0)


@dataclass(frozen=True, slots=True)
class FocusValue:
    """Axis 3 value: kind + opaque payload."""
    kind: FocusKind
    payload: str = ""

    def __post_init__(self) -> None:
        if self.kind == FocusKind.NONE and self.payload:
            raise ValueError("FocusKind.NONE must have empty payload")

    def as_str(self) -> str:
        """Canonical string form: ``none`` or ``<kind>:<payload>``."""
        if self.kind == FocusKind.NONE:
            return "none"
        return f"{self.kind.value}:{self.payload}"

    @classmethod
    def from_str(cls, s: str) -> FocusValue:
        if s == "none" or not s:
            return cls(kind=FocusKind.NONE)
        kind_str, _, payload = s.partition(":")
        return cls(kind=FocusKind(kind_str), payload=payload)


# ── Listener type ──────────────────────────────────────────────────────

# Subscribers receive (axis, from_value, to_value, reason). Sync or async
# callables both accepted; async ones are awaited in transition emit path.
TransitionListener = Callable[[str, str, str, str], None | Awaitable[None]]


# ── Internal state container ───────────────────────────────────────────

@dataclass
class _StateRow:
    """In-memory mirror of the companion_state row."""
    state: AttentionState
    role: RoleVector
    focus: FocusValue
    entered_state_at: float       # monotonic seconds since CompanionState boot
    entered_role_at: float
    entered_focus_at: float


# ── Public class ──────────────────────────────────────────────────────

class CompanionState:
    """3-axis state machine, persisted to ``companion_state`` /
    ``companion_state_log``.

    Use when:
    - The runtime kernel needs the current state for dispatch scoring.
    - The trigger model wants to subscribe to transitions.
    - A subagent wants the dominant role for behavior branching.

    Concurrency: a single ``asyncio.Lock`` serializes transitions per
    companion. Reads are lock-free (the in-memory snapshot is updated
    atomically after the DB write).
    """

    def __init__(
        self,
        backend: SQLiteBackend,
        companion_id: str,
        *,
        user_id: str = "",
    ) -> None:
        self._backend = backend
        self.companion_id = companion_id
        # Piece 1 — per-user invariant: each (user_id, companion_id) pair
        # has its own state row. Empty user_id is the legacy/seed
        # sentinel (pre-pivot Becca singleton after mig 179 backfill).
        self.user_id = user_id
        self._row: _StateRow | None = None
        self._listeners: dict[str, set[TransitionListener]] = {
            "state": set(),
            "role": set(),
            "focus": set(),
        }
        self._lock = asyncio.Lock()
        # Monotonic clock baseline so cooldowns are robust to wall-clock
        # changes (NTP adjustments, daylight savings, etc.). Set at first
        # load() so all entered_*_at deltas are positive.
        self._boot_at = time.monotonic()

    # ── Lifecycle ────────────────────────────────────────────────────

    async def load(self) -> None:
        """Hydrate from ``companion_state``. Per-user scoped (mig 179):
        when ``user_id`` is set, loads that specific user's row;
        otherwise loads the earliest-created row (legacy seed path)."""
        if self.user_id:
            cursor = await self._backend.conn.execute(
                "SELECT state, role_active, role_passive, role_reflective, "
                "focus, entered_state_at, entered_role_at, entered_focus_at "
                "FROM companion_state "
                "WHERE user_id = ? AND companion_id = ?",
                (self.user_id, self.companion_id),
            )
        else:
            cursor = await self._backend.conn.execute(
                "SELECT state, role_active, role_passive, role_reflective, "
                "focus, entered_state_at, entered_role_at, entered_focus_at "
                "FROM companion_state "
                "WHERE companion_id = ? "
                "ORDER BY updated_at ASC LIMIT 1",
                (self.companion_id,),
            )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            raise ValueError(
                f"no companion_state row for "
                f"({self.user_id!r}, {self.companion_id!r}) — "
                "did migration 179 backfill or lazy_provision run?",
            )
        # Back-date the monotonic "entered" baselines by the real elapsed
        # wall time from the persisted DB timestamps, so cooldowns survive
        # a restart instead of re-arming from scratch (audit 2026-06-17).
        # The two clocks are bridged: monotonic stays correct within a run;
        # only the baseline is offset by the wall-clock age. Unparseable
        # timestamps fall back to age 0 (= freshly entered), the old safe
        # behavior.
        now_mono = time.monotonic()

        def _entered(ts_text: str | None) -> float:
            return now_mono - _age_seconds(ts_text)

        self._row = _StateRow(
            state=AttentionState(row[0]),
            role=RoleVector(active=float(row[1]), passive=float(row[2]),
                            reflective=float(row[3])),
            focus=FocusValue.from_str(row[4]),
            entered_state_at=_entered(row[5]),
            entered_role_at=_entered(row[6]),
            entered_focus_at=_entered(row[7]),
        )

    # ── Read path (allocation-free where possible) ────────────────────

    def get_state(self) -> AttentionState:
        return self._row.state if self._row else AttentionState.DORMANT

    def get_role(self) -> RoleVector:
        return self._row.role if self._row else RoleVector.passive_dominant()

    def get_focus(self) -> FocusValue:
        return self._row.focus if self._row else FocusValue(kind=FocusKind.NONE)

    def snapshot(self) -> dict:
        """Cheap read-only view for telemetry and debug surfaces."""
        if self._row is None:
            return {"companion_id": self.companion_id, "loaded": False}
        return {
            "companion_id": self.companion_id,
            "loaded": True,
            "state": self._row.state.value,
            "role": {
                "active": self._row.role.active,
                "passive": self._row.role.passive,
                "reflective": self._row.role.reflective,
                "dominant": self._row.role.dominant(),
            },
            "focus": self._row.focus.as_str(),
        }

    # ── Subscription ─────────────────────────────────────────────────

    def subscribe(self, axis: str, listener: TransitionListener) -> Callable[[], None]:
        """Register a transition listener on one axis. Returns an
        unsubscribe callable.

        Axes: ``"state"``, ``"role"``, ``"focus"``. Unknown axis raises
        ``ValueError``.
        """
        if axis not in self._listeners:
            raise ValueError(f"unknown axis {axis!r}; valid: state, role, focus")
        self._listeners[axis].add(listener)

        def _unsubscribe() -> None:
            self._listeners[axis].discard(listener)

        return _unsubscribe

    # ── Transitions ──────────────────────────────────────────────────

    async def transition_state(
        self, to: AttentionState, *, reason: str = "", force: bool = False,
    ) -> bool:
        """Apply a state-axis transition.

        Returns True on success. False if rejected by cooldown (and
        ``force`` is False) or if ``to`` equals current state (no-op
        but logged as accepted).

        ``force=True`` bypasses cooldown but not validity.
        """
        async with self._lock:
            if self._row is None:
                await self.load()
            assert self._row is not None
            old = self._row.state
            if to == old:
                return True  # no-op
            now = time.monotonic()
            if not force and (now - self._row.entered_state_at) < COOLDOWN_STATE_S:
                log.debug("companion_state_cooldown",
                          companion_id=self.companion_id, axis="state",
                          remaining_s=COOLDOWN_STATE_S - (now - self._row.entered_state_at))
                return False
            await self._persist_state(to, now)
            self._row.state = to
            self._row.entered_state_at = now
            await self._emit("state", old.value, to.value, reason)
            return True

    async def transition_role(
        self, to: RoleVector, *, reason: str = "", force: bool = False,
    ) -> bool:
        """Apply a role-axis transition (3-vector swap)."""
        async with self._lock:
            if self._row is None:
                await self.load()
            assert self._row is not None
            old = self._row.role
            if to.as_tuple() == old.as_tuple():
                return True
            now = time.monotonic()
            if not force and (now - self._row.entered_role_at) < COOLDOWN_ROLE_S:
                return False
            await self._persist_role(to, now)
            self._row.role = to
            self._row.entered_role_at = now
            await self._emit("role", old.as_json(), to.as_json(), reason)
            return True

    async def transition_focus(
        self, to: FocusValue, *, reason: str = "", force: bool = False,
    ) -> bool:
        """Apply a focus-axis transition (discrete swap)."""
        async with self._lock:
            if self._row is None:
                await self.load()
            assert self._row is not None
            old = self._row.focus
            if to.kind == old.kind and to.payload == old.payload:
                return True
            now = time.monotonic()
            if not force and (now - self._row.entered_focus_at) < COOLDOWN_FOCUS_S:
                return False
            await self._persist_focus(to, now)
            self._row.focus = to
            self._row.entered_focus_at = now
            await self._emit("focus", old.as_str(), to.as_str(), reason)
            return True

    # ── Persistence helpers (called under lock) ──────────────────────

    async def _persist_state(self, to: AttentionState, _now: float) -> None:
        # Per-user scoping (mig 179): updates target the loaded row's
        # (user_id, companion_id) pair so transitions in User A's state
        # never overwrite User B's.
        conn = self._backend.conn
        await conn.execute(
            "UPDATE companion_state SET state = ?, entered_state_at = datetime('now'), "
            "updated_at = datetime('now') WHERE user_id = ? AND companion_id = ?",
            (to.value, self.user_id, self.companion_id),
        )
        await conn.commit()

    async def _persist_role(self, to: RoleVector, _now: float) -> None:
        conn = self._backend.conn
        await conn.execute(
            "UPDATE companion_state SET role_active = ?, role_passive = ?, "
            "role_reflective = ?, entered_role_at = datetime('now'), "
            "updated_at = datetime('now') WHERE user_id = ? AND companion_id = ?",
            (to.active, to.passive, to.reflective, self.user_id, self.companion_id),
        )
        await conn.commit()

    async def _persist_focus(self, to: FocusValue, _now: float) -> None:
        conn = self._backend.conn
        await conn.execute(
            "UPDATE companion_state SET focus = ?, entered_focus_at = datetime('now'), "
            "updated_at = datetime('now') WHERE user_id = ? AND companion_id = ?",
            (to.as_str(), self.user_id, self.companion_id),
        )
        await conn.commit()

    # ── Listener emission (isolated per-callback) ─────────────────────

    async def _emit(self, axis: str, from_value: str, to_value: str, reason: str) -> None:
        """Append to companion_state_log + fan out to listeners.

        Listener exceptions are caught per-callback and logged at warn
        level so one bad subscriber cannot break the transition. The
        listener set is snapshot-copied at emit time so subscribers can
        unsubscribe themselves mid-emit without corrupting the iteration.
        """
        # Persist to log (append-only). Per-user scoping (mig 179):
        # the log now carries user_id so each user's transition history
        # is isolated. Empty user_id reflects legacy seed/unprovisioned
        # state.
        await self._backend.conn.execute(
            "INSERT INTO companion_state_log (companion_id, user_id, axis, "
            "from_value, to_value, reason) VALUES (?, ?, ?, ?, ?, ?)",
            (self.companion_id, self.user_id, axis, from_value, to_value, reason),
        )
        await self._backend.conn.commit()

        # Snapshot listener set so mid-emit unsubscribes are safe
        listeners = list(self._listeners[axis])
        for cb in listeners:
            try:
                result = cb(axis, from_value, to_value, reason)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                log.warning(
                    "companion_state_listener_failed",
                    companion_id=self.companion_id,
                    axis=axis,
                    error=str(exc)[:200],
                )


__all__ = [
    "AttentionState",
    "FocusKind",
    "FocusValue",
    "RoleVector",
    "CompanionState",
    "COOLDOWN_STATE_S",
    "COOLDOWN_ROLE_S",
    "COOLDOWN_FOCUS_S",
]
