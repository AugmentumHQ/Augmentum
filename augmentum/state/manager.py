"""State manager — coordinates state backend and session lifecycle."""

from __future__ import annotations

from augmentum.state.backends.memory import MemoryBackend
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.coder_persistence import CoderPersistence
from augmentum.state.narrative_persistence import NarrativePersistence
from augmentum.state.narrative_state import NarrativeSessionState
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class StateManager:
    """Manages state backend and provides session operations."""

    def __init__(self, backend: SQLiteBackend | MemoryBackend) -> None:
        self._backend = backend

    async def get_or_create_session(
        self, session_id: str, mode: str = "passthrough"
    ) -> dict:
        """Get an existing session or create a new one."""
        session = await self._backend.get_session(session_id)
        if session is None:
            session = await self._backend.create_session(session_id, mode)
            log.debug("session_created", session_id=session_id, mode=mode)
        return session

    async def update_session(
        self, session_id: str, **kwargs
    ) -> None:
        await self._backend.update_session(session_id, **kwargs)

    @property
    def backend(self) -> SQLiteBackend | MemoryBackend:
        return self._backend

    # --- Narrative state persistence ---

    async def save_narrative_state(
        self, session_id: str, state: NarrativeSessionState,
        *, user_id: str = "",
    ) -> None:
        """Persist narrative session state to the database.

        Only works when the backend is SQLiteBackend; silently skips
        for MemoryBackend (which has no persistent storage).

        ``user_id`` scopes the write to the session's owner so a client
        can't inject state into another user's session by forging the
        session_id in ``X-Augmentum-Session``.
        """
        if not isinstance(self._backend, SQLiteBackend):
            log.debug("save_narrative_state_skipped", reason="non-sqlite backend")
            return
        # Sessions table is user-scoped at its own layer; here we just need
        # the FK-parent row to exist. user_id is applied to the narrative
        # sub-tables by persistence.save_session_state.
        await self.get_or_create_session(session_id, mode="narrative")
        persistence = NarrativePersistence(self._backend.conn)
        await persistence.save_session_state(session_id, state, user_id=user_id)

    async def load_narrative_state(
        self, session_id: str, *, user_id: str = "",
    ) -> NarrativeSessionState | None:
        """Load narrative session state from the database.

        Returns None if the backend is not SQLite or no state exists for
        the given (session_id, user_id). Without a user_id filter, a client
        could load another user's narrative state by sending that user's
        session_id in the ``X-Augmentum-Session`` header.
        """
        if not isinstance(self._backend, SQLiteBackend):
            return None
        persistence = NarrativePersistence(self._backend.conn)
        return await persistence.load_session_state(session_id, user_id=user_id)

    # --- Coder state persistence ---

    async def save_coder_state(
        self,
        session_id: str,
        state,
        *,
        user_id: str = "",
    ) -> bool:
        """Persist coder session state to the database.

        Returns True on success, False if the write was skipped (non-
        SQLite backend) or blocked by a different user owning the row.
        """
        if not isinstance(self._backend, SQLiteBackend):
            log.debug("save_coder_state_skipped", reason="non-sqlite backend")
            return False
        persistence = CoderPersistence(self._backend.conn)
        return await persistence.save_session_state(
            session_id, state, user_id=user_id,
        )

    async def load_coder_state(
        self,
        session_id: str,
        *,
        user_id: str = "",
    ):
        """Load coder session state from the database."""
        if not isinstance(self._backend, SQLiteBackend):
            return None
        persistence = CoderPersistence(self._backend.conn)
        return await persistence.load_session_state(session_id, user_id=user_id)
