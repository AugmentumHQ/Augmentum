"""MimiAudioHistory — per-session rolling buffer of conversation turns.

Phase 2 of the presence pipeline. The forward-compatibility substrate
that makes a future Kyutai-family model swap (CSM / Moshi / Pocket-vNext)
inherit conversation context for free, instead of needing to re-tokenize
the user's entire history through a new codec.

Architecture:
  - Storage: ``companion_audio_history`` table (migration 253).
  - One row per turn. Per-(user_id, session_id) ordering via turn_index.
  - ``mimi_tokens`` is a nullable BLOB of gzipped int16 ndarray; v1
    captures Becca's transcript only. The Mimi-token capture path is
    blocked on either forking upstream ``pocket_tts`` to expose its
    internal generation tokens, or swapping to a model that exposes
    them directly. The substrate is shaped for both.

Why ship the substrate before the capture:
  Per ``[[substrate-paying-back]]`` — wire the durable representation
  now while we have one consumer (PocketTTS-as-transcript). When CSM
  or a fork-with-token-access lands, the consumers swap in without
  touching the schema, the store API, or the call sites that already
  thread sessions through here.

Multi-tenant:
  Every method requires ``user_id``. Empty string raises ValueError —
  pattern lifted from NotesStore to fail loudly on a route handler
  that forgot to thread scope.

Concurrency:
  The store wraps an aiosqlite connection. SQLite serializes writes
  inside augmentum's state layer (WAL + busy_timeout). Multiple
  PresencePipeline instances writing to different sessions in parallel
  is fine; back-to-back writes to the same session serialize on the
  unique index ``idx_companion_audio_history_session_turn``.
"""

from __future__ import annotations

import gzip
import time
import uuid
from dataclasses import dataclass

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


SPEAKER_BECCA = "becca"
SPEAKER_USER = "user"

# Retention horizon for the periodic sweep — turns older than this are
# dropped at startup. 30 days matches the auth session cap to avoid
# orphaned context referencing already-expired sessions.
DEFAULT_RETENTION_DAYS = 30

# Default window size for recent_window when the caller doesn't specify.
# 30 seconds at PocketTTS rate (~10 turns of conversation) is enough for
# prosodic continuity without dragging the LLM context past relevance.
DEFAULT_WINDOW_SECONDS = 30.0


@dataclass
class Turn:
    """A single conversation turn — what was said + (optionally) how it sounded."""

    id: str
    session_id: str
    user_id: str
    turn_index: int
    speaker: str
    transcript: str
    duration_ms: int
    created_at: str
    mimi_tokens: bytes | None = None  # gzipped int16 ndarray bytes


class MimiAudioHistory:
    """Per-app store for conversation turns + (future) Mimi token capture.

    Construct once per app lifetime; pass the same aiosqlite connection
    augmentum's state layer hands out. Per-conversation state lives in
    the database (rows keyed by user_id + session_id), NOT on the store
    itself — multiple PresencePipeline instances share one store.
    """

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    # ── Append ───────────────────────────────────────────────────

    async def append_becca_turn(
        self,
        *,
        session_id: str,
        user_id: str,
        transcript: str,
        mimi_tokens: bytes | None = None,
        duration_ms: int = 0,
    ) -> Turn:
        """Record a Becca-spoken turn.

        ``mimi_tokens`` is the gzipped serialized int16 ndarray of Mimi
        codec tokens for this turn. v1 callers pass None (transcript
        only); future callers with token access pass the serialized
        ndarray. Callers serialize via :func:`serialize_mimi_tokens`.
        """
        return await self._append(
            session_id=session_id,
            user_id=user_id,
            speaker=SPEAKER_BECCA,
            transcript=transcript,
            mimi_tokens=mimi_tokens,
            duration_ms=duration_ms,
        )

    async def append_user_turn(
        self,
        *,
        session_id: str,
        user_id: str,
        transcript: str,
        duration_ms: int = 0,
    ) -> Turn:
        """Record a user-spoken turn. Mimi token capture for user audio
        requires a server-side Mimi encoder pass — deferred until that
        landing. Stored as transcript only in v1.
        """
        return await self._append(
            session_id=session_id,
            user_id=user_id,
            speaker=SPEAKER_USER,
            transcript=transcript,
            mimi_tokens=None,
            duration_ms=duration_ms,
        )

    async def _append(
        self,
        *,
        session_id: str,
        user_id: str,
        speaker: str,
        transcript: str,
        mimi_tokens: bytes | None,
        duration_ms: int,
    ) -> Turn:
        if not user_id:
            raise ValueError("audio_history._append requires user_id")
        if not session_id:
            raise ValueError("audio_history._append requires session_id")
        if speaker not in (SPEAKER_BECCA, SPEAKER_USER):
            raise ValueError(
                f"speaker must be {SPEAKER_BECCA!r} or {SPEAKER_USER!r}",
            )
        turn_id = uuid.uuid4().hex
        # Atomic per-session turn_index allocation. The unique index on
        # (user_id, session_id, turn_index) catches any race where two
        # callers compute the same MAX+1 — the second insert raises,
        # the orchestrator retries.
        cursor = await self._conn.execute(
            "SELECT COALESCE(MAX(turn_index), -1) + 1 "
            "FROM companion_audio_history "
            "WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        )
        row = await cursor.fetchone()
        next_index = int(row[0] if row else 0)

        await self._conn.execute(
            "INSERT INTO companion_audio_history "
            "(id, user_id, session_id, turn_index, speaker, transcript, "
            " mimi_tokens, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                turn_id, user_id, session_id, next_index, speaker,
                transcript, mimi_tokens, int(duration_ms),
            ),
        )
        await self._conn.commit()
        return Turn(
            id=turn_id,
            session_id=session_id,
            user_id=user_id,
            turn_index=next_index,
            speaker=speaker,
            transcript=transcript,
            duration_ms=int(duration_ms),
            created_at="",  # filled by SQLite default; queries return it
            mimi_tokens=mimi_tokens,
        )

    # ── Read ─────────────────────────────────────────────────────

    async def recent_window(
        self,
        *,
        session_id: str,
        user_id: str,
        max_seconds: float | None = None,
        max_turns: int | None = None,
    ) -> list[Turn]:
        """Return the most recent N turns for this session, oldest-first.

        ``max_seconds`` is a soft budget summed across turn ``duration_ms``
        values — we stop walking backward once we've accumulated enough.
        ``max_turns`` is a hard cap (overrides max_seconds if smaller).
        Returns oldest-first so callers can feed it to a model in
        chronological order without reversing.

        Empty list if the session has no rows yet.
        """
        if not user_id:
            raise ValueError("audio_history.recent_window requires user_id")
        if not session_id:
            raise ValueError("audio_history.recent_window requires session_id")

        budget_ms = int((max_seconds or DEFAULT_WINDOW_SECONDS) * 1000.0)
        # Walk newest-first; SQL returns rows we then truncate by budget.
        cursor = await self._conn.execute(
            "SELECT id, session_id, user_id, turn_index, speaker, transcript, "
            "       duration_ms, created_at, mimi_tokens "
            "FROM companion_audio_history "
            "WHERE user_id = ? AND session_id = ? "
            "ORDER BY turn_index DESC",
            (user_id, session_id),
        )
        rows = await cursor.fetchall()
        if not rows:
            return []

        kept: list[Turn] = []
        accum_ms = 0
        for row in rows:
            turn = Turn(
                id=row[0],
                session_id=row[1],
                user_id=row[2],
                turn_index=int(row[3]),
                speaker=row[4],
                transcript=row[5] or "",
                duration_ms=int(row[6] or 0),
                created_at=row[7] or "",
                mimi_tokens=row[8],
            )
            kept.append(turn)
            accum_ms += turn.duration_ms
            if max_turns is not None and len(kept) >= max_turns:
                break
            if accum_ms >= budget_ms and max_turns is None:
                break

        kept.reverse()  # oldest-first for caller convenience
        return kept

    async def turn_count(
        self, *, session_id: str, user_id: str,
    ) -> int:
        """Count turns in this session — used for telemetry + tests."""
        if not user_id:
            raise ValueError("audio_history.turn_count requires user_id")
        if not session_id:
            raise ValueError("audio_history.turn_count requires session_id")
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM companion_audio_history "
            "WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    # ── Retention sweep ──────────────────────────────────────────

    async def sweep_old_turns(
        self, *, retention_days: int = DEFAULT_RETENTION_DAYS,
    ) -> int:
        """Drop rows older than the retention horizon. Returns row count.

        Intentionally NOT user-scoped — runs across all users as a
        global retention pass at startup or on a scheduled basis. The
        deletion is bounded by created_at, indexed via
        idx_companion_audio_history_created so even on a multi-million
        row table the sweep stays fast.
        """
        if retention_days <= 0:
            raise ValueError("retention_days must be positive")
        cursor = await self._conn.execute(
            "DELETE FROM companion_audio_history "
            "WHERE created_at < datetime('now', ?)",
            (f"-{int(retention_days)} days",),
        )
        await self._conn.commit()
        return cursor.rowcount if cursor.rowcount is not None else 0

    async def clear_session(
        self, *, session_id: str, user_id: str,
    ) -> int:
        """Drop all turns for one session. Used when the user explicitly
        ends a conversation + asks to forget it. Returns dropped count.
        """
        if not user_id:
            raise ValueError("audio_history.clear_session requires user_id")
        if not session_id:
            raise ValueError(
                "audio_history.clear_session requires session_id",
            )
        cursor = await self._conn.execute(
            "DELETE FROM companion_audio_history "
            "WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        )
        await self._conn.commit()
        return cursor.rowcount if cursor.rowcount is not None else 0


# ── Mimi token serialization helpers ────────────────────────────


def serialize_mimi_tokens(tokens: object) -> bytes:
    """Compress an int16 ndarray of Mimi tokens to bytes for storage.

    Takes an object that quacks like ``numpy.ndarray`` (we don't hard-
    require numpy at module import time so a test environment without
    it can still load the module). Returns gzip-compressed raw bytes
    plus a tiny header recording shape so deserialize can reconstruct.

    Tokens shape: [N_codebooks, N_frames], dtype int16 from Mimi's
    split-RVQ output. Compressed size: ~50% of raw for typical
    speech (lots of repetition in codebook 0).
    """
    import numpy as np
    arr = np.asarray(tokens, dtype=np.int16)
    if arr.ndim != 2:
        raise ValueError(
            f"Mimi tokens must be 2D [codebooks, frames], got shape {arr.shape}",
        )
    header = f"{arr.shape[0]},{arr.shape[1]}\n".encode("ascii")
    return gzip.compress(header + arr.tobytes())


def deserialize_mimi_tokens(blob: bytes) -> object:
    """Inverse of :func:`serialize_mimi_tokens`. Returns an int16 ndarray.

    Raises if numpy isn't available, ValueError if the blob is malformed.
    """
    import numpy as np
    raw = gzip.decompress(blob)
    newline_idx = raw.index(b"\n")
    header = raw[:newline_idx].decode("ascii")
    body = raw[newline_idx + 1:]
    try:
        codebooks_str, frames_str = header.split(",")
        codebooks, frames = int(codebooks_str), int(frames_str)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Malformed Mimi token blob header: {header!r}") from exc
    return np.frombuffer(body, dtype=np.int16).reshape(codebooks, frames)
