"""Session Lifecycle Manager — coordinates state persistence across all layers.

Ensures mode state (SQLite) and KV cache (engine) stay in sync.
Triggered by session switches, model swaps, shutdown, and idle timeouts.

The proxy is the coordinator — it knows the mode, the session, the handler,
and has access to both the state manager and the engine backend.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.provider_registry import ProviderRegistry
    from augmentum.state.manager import StateManager

log = get_logger(__name__)


@dataclass
class SessionSnapshot:
    """What we know about a session's cache state."""
    session_id: str
    mode: str                          # passthrough, analytical, narrative, agentic, coder
    model: str                         # model used for this session's KV
    token_count: int = 0               # tokens in KV when saved
    kv_saved: bool = False             # is KV persisted?
    mode_state_saved: bool = False     # is mode state persisted?
    last_active: float = 0.0           # timestamp of last activity
    last_saved: float = 0.0            # timestamp of last save


class SessionLifecycle:
    """Coordinates session persistence across mode handlers and engine.

    Usage in server.py:
        lifecycle = SessionLifecycle(state_manager, provider_registry)

        # On every message:
        lifecycle.touch(session_id, mode, model)

        # On session switch:
        await lifecycle.on_session_switch(old_session_id, new_session_id)

        # On model swap:
        await lifecycle.on_model_swap(session_id)

        # On shutdown:
        await lifecycle.on_shutdown()
    """

    def __init__(
        self,
        state_manager: StateManager | None = None,
        provider_registry: ProviderRegistry | None = None,
        auto_save_interval: int = 300,  # save active session every 5 minutes
    ):
        self._state_manager = state_manager
        self._registry = provider_registry
        self._auto_save_interval = auto_save_interval

        # Track active sessions
        self._sessions: dict[str, SessionSnapshot] = {}
        self._active_session: str = ""

    def touch(self, session_id: str, mode: str, model: str):
        """Mark a session as active. Called on every message."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionSnapshot(
                session_id=session_id,
                mode=mode,
                model=model,
            )
        snap = self._sessions[session_id]
        snap.last_active = time.time()
        snap.mode = mode
        snap.model = model
        self._active_session = session_id

    async def on_session_switch(self, old_id: str, new_id: str):
        """User is switching from one session to another.

        Save the outgoing session's state. The incoming session will
        auto-restore on first message (via handler's _ensure_state_loaded
        and engine's prefix matching).
        """
        if old_id and old_id in self._sessions:
            await self._save_session(old_id)

    async def on_model_swap(self, session_id: str = ""):
        """Model is about to be swapped. Save ALL active sessions' KV.

        KV cache is model-specific — it's invalid after a model change.
        Mode state in SQLite is model-agnostic and persists fine.
        """
        for sid, snap in self._sessions.items():
            if snap.last_active > 0:
                await self._save_session(sid, kv_only=False)
                snap.kv_saved = False  # KV will be invalid after model swap

    async def on_shutdown(self):
        """Graceful shutdown. Save everything."""
        log.info("session_lifecycle_shutdown", sessions=len(self._sessions))
        for sid in list(self._sessions.keys()):
            await self._save_session(sid)
        log.info("session_lifecycle_shutdown_complete")

    async def on_message_complete(self, session_id: str, handler=None):
        """Called after a message is fully processed.

        Saves mode state immediately (cheap, SQLite).
        Saves KV cache periodically (expensive, but protects against crashes).
        """
        snap = self._sessions.get(session_id)
        if not snap:
            return

        # Always save mode state after each message
        if handler:
            await self._save_mode_state(session_id, handler)
            snap.mode_state_saved = True

        # Periodic KV save (every auto_save_interval seconds)
        now = time.time()
        if now - snap.last_saved > self._auto_save_interval:
            await self._save_kv(session_id)
            snap.last_saved = now

    async def try_restore_kv(self, session_id: str, model: str) -> bool:
        """Try to restore KV cache for a session. Returns True if successful.

        Called before the first message in a session after restart/switch.
        Only works if the same model is loaded.
        """
        snap = self._sessions.get(session_id)
        engine = self._get_engine()
        if not engine:
            return False

        try:
            restore = getattr(engine, "restore_session_state", None)
            if callable(restore):
                restored = await restore(session_id)
                if restored:
                    log.info("kv_restored", session_id=session_id)
                    if snap:
                        snap.kv_saved = True
                    return True
        except Exception as exc:
            log.debug("kv_restore_failed", session_id=session_id, error=str(exc))

        return False

    # --- Internal helpers ---

    async def _save_session(self, session_id: str, kv_only: bool = False):
        """Save both mode state and KV cache for a session."""
        if not kv_only:
            # Mode state is handled by the handler — we just trigger KV save here
            pass

        await self._save_kv(session_id)

    async def _save_kv(self, session_id: str):
        """Save KV cache to the engine's persistence layer."""
        engine = self._get_engine()
        if not engine:
            return

        try:
            save = getattr(engine, "save_session_state", None)
            if callable(save) and await save(session_id):
                snap = self._sessions.get(session_id)
                if snap:
                    snap.kv_saved = True
                    snap.last_saved = time.time()
                log.info("kv_saved", session_id=session_id)
        except Exception as exc:
            log.debug("kv_save_failed", session_id=session_id, error=str(exc))

    async def _save_mode_state(self, session_id: str, handler):
        """Save mode-specific state via the handler."""
        # Narrative handler has explicit persist
        if hasattr(handler, '_persist_state'):
            try:
                await handler._persist_state()
            except Exception as exc:
                log.warning("mode_state_save_failed",
                            session_id=session_id, error=str(exc))

    def _get_engine(self):
        """Get the engine backend if available."""
        if not self._registry:
            return None
        try:
            return self._registry.get_backend("engine")
        except (ValueError, KeyError):
            return None

    def stats(self) -> dict:
        """Current lifecycle state for debugging."""
        return {
            "active_session": self._active_session,
            "tracked_sessions": len(self._sessions),
            "sessions": {
                sid: {
                    "mode": s.mode,
                    "model": s.model,
                    "kv_saved": s.kv_saved,
                    "mode_state_saved": s.mode_state_saved,
                    "last_active": s.last_active,
                    "last_saved": s.last_saved,
                }
                for sid, s in self._sessions.items()
            },
        }
