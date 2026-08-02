"""Dream scheduler — per-user threshold + idle gating."""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta

import structlog

from augmentum.memory.events import log_event

log = structlog.get_logger(__name__)


_COUNTERS_KEY = "dream._counters"
_DREAM_ENABLED_KEY = "ui.dreamEnabled"
_THRESHOLD_KEY = "ui.dreamMessageThreshold"
_IDLE_KEY = "ui.dreamIdleMinutes"
_COOLDOWN_KEY = "ui.dreamCooldownMinutes"


class DreamsDisabledError(Exception):
    """Raised when a caller asks to run a dream cycle but has opted out.

    The scheduler is a process singleton, so a cold attribute-miss can't
    signal "this user hasn't opted in" — we need an explicit error. The
    route layer translates this to a 409 response so the client can
    differentiate opted-out from "dream system not running at all"
    (which stays as 503).
    """


class DreamScheduler:
    """Schedules dream cycles per user based on activity thresholds.

    Each user has independent counters (messages_since_dream,
    approved_since_dream) and an independent last-dream timestamp. The
    background loop iterates users and runs dream cycles for whoever is
    eligible. The empty-string user_id key serves both the single-tenant
    legacy path and as a global fallback.
    """

    def __init__(
        self,
        engine,
        settings_store,
        enabled: bool = True,
        message_threshold: int = 6,
        idle_minutes: int = 30,
        cooldown_minutes: int = 60,
    ):
        self._engine = engine
        self._settings_store = settings_store
        self._enabled = enabled
        self._message_threshold = message_threshold
        self._idle_minutes = idle_minutes
        self._cooldown_minutes = cooldown_minutes
        # Per-user counters. Empty-string key is the legacy single-tenant path.
        now = datetime.now(UTC)
        self._messages_since: dict[str, int] = defaultdict(int)
        self._approved_since: dict[str, int] = defaultdict(int)
        self._last_request_at: dict[str, datetime] = defaultdict(lambda: now)
        self._last_dream_at: dict[str, datetime] = {}
        self._running_for: set[str] = set()
        self._task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Load persisted per-user counters. Call before start()."""
        if self._settings_store is None:
            return
        try:
            raw = await self._settings_store.get(_COUNTERS_KEY)
            if not raw:
                return
            data = json.loads(raw) if isinstance(raw, str) else raw
            for uid, fields in (data or {}).items():
                self._messages_since[uid] = int(fields.get("messages", 0))
                self._approved_since[uid] = int(fields.get("approved", 0))
                last_dream = fields.get("last_dream_at")
                if last_dream:
                    try:
                        self._last_dream_at[uid] = datetime.fromisoformat(last_dream)
                    except ValueError:
                        pass
            log.info("dream_scheduler_counters_loaded", users=len(data or {}))
        except Exception:
            log.warning("dream_scheduler.load_counters_failed", exc_info=True)

    def start(self) -> None:
        """Start the background check loop."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._check_loop())
        self._task.add_done_callback(self._on_task_done)
        log.info("dream_scheduler_started")

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Log unexpected scheduler exits."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            log.error("dream_scheduler_crashed", error=str(exc), exc_info=exc)
            self._task = None

    async def stop(self) -> None:
        """Stop the background loop and flush counters."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self._flush_counters()
        log.info("dream_scheduler_stopped")

    # --- Notifications --------------------------------------------------

    def notify_message(self, user_id: str = "") -> None:
        """Called on every message pair processed."""
        self._messages_since[user_id] += 1

    def notify_approval(self, memory_id: str, user_id: str = "") -> None:
        """Called when a memory is approved (auto or explicit)."""
        self._approved_since[user_id] += 1

    def notify_request(self, user_id: str = "") -> None:
        """Called on every API request to track per-user idle time."""
        self._last_request_at[user_id] = datetime.now(UTC)

    # --- Eligibility & execution ----------------------------------------

    async def _user_opted_in(self, user_id: str) -> bool:
        """Return True if ``user_id`` currently wants the dream system.

        Looks up ``ui.dreamEnabled`` from the per-user settings table,
        falling back to the install-wide default. Called on every check-
        loop pass (once per minute per active user) and on every manual
        trigger. With no settings store wired (test harness) we default
        to True to preserve the pre-multi-tenant behaviour.
        """
        if self._settings_store is None:
            return True
        try:
            val = await self._settings_store.get_user_or_global(
                user_id, _DREAM_ENABLED_KEY,
            )
        except Exception:
            log.warning("dream_scheduler.opt_in_lookup_failed", user_id=user_id, exc_info=True)
            # Fail closed — do not run cycles for a user we can't verify.
            return False
        return val == "true"

    async def _user_thresholds(self, user_id: str) -> tuple[int, int, int]:
        """Resolve ``(message_threshold, idle_minutes, cooldown_minutes)`` for a user.

        Falls back to constructor defaults when the settings store isn't
        wired or the value is absent/malformed. These settings live in
        ``user_settings`` post-Stage-D; pre-Stage-D installs with values
        in ``app_settings`` are picked up automatically via
        :meth:`SettingsStore.get_user_or_global`.
        """
        if self._settings_store is None:
            return self._message_threshold, self._idle_minutes, self._cooldown_minutes

        async def _read_int(key: str, default: int) -> int:
            try:
                raw = await self._settings_store.get_user_or_global(user_id, key)
            except Exception:
                log.warning(
                    "dream_scheduler.threshold_lookup_failed",
                    user_id=user_id, key=key, exc_info=True,
                )
                return default
            if raw in (None, ""):
                return default
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

        threshold = await _read_int(_THRESHOLD_KEY, self._message_threshold)
        idle = await _read_int(_IDLE_KEY, self._idle_minutes)
        cooldown = await _read_int(_COOLDOWN_KEY, self._cooldown_minutes)
        return threshold, idle, cooldown

    async def _is_eligible(self, user_id: str = "") -> bool:
        """Check trigger conditions for one user, using *their* thresholds.

        Does NOT check per-user opt-in — call :meth:`_user_opted_in` first.
        Thresholds are resolved per-user so tenant-level tuning of
        ``dreamMessageThreshold`` / ``dreamIdleMinutes`` /
        ``dreamCooldownMinutes`` is honoured. Falls back to constructor
        defaults if the settings store is absent or malformed.
        """
        if not self._enabled:
            return False
        if user_id in self._running_for:
            return False

        message_threshold, idle_minutes, cooldown_minutes = await self._user_thresholds(user_id)

        if self._messages_since[user_id] < message_threshold:
            return False
        if self._approved_since[user_id] < 1:
            return False

        now = datetime.now(UTC)
        idle_duration = now - self._last_request_at[user_id]
        if idle_duration < timedelta(minutes=idle_minutes):
            return False

        last_dream = self._last_dream_at.get(user_id)
        if last_dream is not None:
            cooldown_duration = now - last_dream
            if cooldown_duration < timedelta(minutes=cooldown_minutes):
                return False

        return True

    def _reset_counters(self, user_id: str) -> None:
        """Reset counters for one user after their dream cycle."""
        self._messages_since[user_id] = 0
        self._approved_since[user_id] = 0

    async def _check_loop(self) -> None:
        """Background loop — every 60s, run dream cycles for eligible users.

        A user must both (a) have ``ui.dreamEnabled = "true"`` and (b)
        pass the deterministic eligibility gate (threshold, idle,
        cooldown, not already running). The opt-in check comes first so
        users who never opted in never pay for the eligibility math.
        """
        while True:
            try:
                await asyncio.sleep(60)
                # Snapshot keys so notifications during iteration don't mutate the view
                candidates = set(self._messages_since.keys()) | set(self._approved_since.keys())
                for uid in candidates:
                    if not await self._user_opted_in(uid):
                        continue
                    if await self._is_eligible(uid):
                        await self._run_dream(uid)
            except asyncio.CancelledError:
                break
            except Exception:
                log.warning("dream_check_loop_error", exc_info=True)

    async def _run_dream(self, user_id: str) -> None:
        """Execute a dream cycle for a single user."""
        self._running_for.add(user_id)
        try:
            cycle = await self._engine.run_cycle("default", "threshold", user_id=user_id)
            self._last_dream_at[user_id] = datetime.now(UTC)
            self._reset_counters(user_id)
            log.info(
                "dream_cycle_completed",
                cycle_id=cycle.id, entries=cycle.entries_count, user_id=user_id,
            )
            try:
                conn = self._engine._journal._db
                if conn:
                    await log_event(
                        conn, "dream_cycle",
                        user_id=user_id,
                        detail={
                            "cycle_id": cycle.id,
                            "entries_count": cycle.entries_count,
                            "trigger": "threshold",
                        },
                    )
            except Exception:
                log.debug("dream_event_log_failed", exc_info=True)
        except Exception:
            log.error("dream_cycle_error", user_id=user_id, exc_info=True)
        finally:
            self._running_for.discard(user_id)

    async def trigger_manual(self, persona_id: str = "default", user_id: str = "") -> str:
        """Trigger a dream cycle immediately (bypasses threshold/idle/cooldown).

        Still honours the per-user opt-in: a caller whose
        ``ui.dreamEnabled`` is off raises :class:`DreamsDisabledError`
        rather than silently running a cycle for them. That's policy,
        not a plumbing artefact — the route layer surfaces it as 409 so
        the client can distinguish "opted out" from "system not running".
        """
        if not await self._user_opted_in(user_id):
            raise DreamsDisabledError(
                f"Dreams are not enabled for user {user_id or '(anonymous)'}"
            )
        if user_id in self._running_for:
            return "already_running"
        self._running_for.add(user_id)
        try:
            cycle = await self._engine.run_cycle(persona_id, "manual", user_id=user_id)
            self._last_dream_at[user_id] = datetime.now(UTC)
            self._reset_counters(user_id)
            return cycle.id
        finally:
            self._running_for.discard(user_id)

    async def _flush_counters(self) -> None:
        """Persist per-user counters to the settings store."""
        if self._settings_store is None:
            return
        try:
            data: dict[str, dict] = {}
            users = (
                set(self._messages_since.keys())
                | set(self._approved_since.keys())
                | set(self._last_dream_at.keys())
            )
            for uid in users:
                last_dream = self._last_dream_at.get(uid)
                data[uid] = {
                    "messages": self._messages_since[uid],
                    "approved": self._approved_since[uid],
                    "last_dream_at": last_dream.isoformat() if last_dream else None,
                }
            await self._settings_store.set(_COUNTERS_KEY, json.dumps(data))
        except Exception:
            log.warning("dream_scheduler.flush_counters_failed", exc_info=True)

    async def get_status(self, user_id: str = "") -> dict:
        """Return current scheduler status for ``user_id``.

        Async because ``next_dream_eligible`` depends on the user's own
        thresholds, which live in the settings store. The payload is
        user-scoped: two tenants calling this on the same process get
        their own counters, their own last-dream timestamp, and an
        eligibility verdict computed against their own tuning.
        """
        last_dream = self._last_dream_at.get(user_id)
        return {
            "enabled": self._enabled,
            "messages_since_dream": self._messages_since[user_id],
            "approved_memories_since_dream": self._approved_since[user_id],
            "last_dream_at": last_dream.isoformat() if last_dream else None,
            "next_dream_eligible": await self._is_eligible(user_id),
            "running": user_id in self._running_for,
        }
