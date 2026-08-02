"""In-process wakeups for surface event polling.

Surface state is durable in SQLite; this runtime is only the live
notification layer that lets TV/browser receivers long-poll without
hammering the database. If the process restarts, clients reconnect and
replay missed events from the persistent event table.
"""

from __future__ import annotations

import asyncio


class SurfaceRuntime:
    """Per-session condition variables for long-poll wakeups."""

    def __init__(self) -> None:
        self._conditions: dict[tuple[str, str], asyncio.Condition] = {}

    def _condition(self, *, user_id: str, session_id: str) -> asyncio.Condition:
        key = (str(user_id or ""), str(session_id or ""))
        cond = self._conditions.get(key)
        if cond is None:
            cond = asyncio.Condition()
            self._conditions[key] = cond
        return cond

    async def notify(self, *, user_id: str, session_id: str) -> None:
        """Wake listeners for one user-scoped surface session."""
        if not user_id or not session_id:
            return
        cond = self._condition(user_id=user_id, session_id=session_id)
        async with cond:
            cond.notify_all()

    async def wait(
        self,
        *,
        user_id: str,
        session_id: str,
        timeout_s: float,
    ) -> None:
        """Wait for a state change or timeout.

        Spurious wakeups are fine; callers always re-query SQLite.
        """
        if not user_id or not session_id or timeout_s <= 0:
            return
        cond = self._condition(user_id=user_id, session_id=session_id)
        async with cond:
            try:
                await asyncio.wait_for(cond.wait(), timeout=max(0.05, timeout_s))
            except TimeoutError:
                return
