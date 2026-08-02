"""Bootstrap the L0 observation store from existing user-profile data.

Phase A seeds from a single source: ``ui_sessions`` (chat history,
which is the highest-signal phrasing corpus this user owns). Browse
history, notes, and narrative-memory sources are deferred to Phase B.

The seeder walks each session's tree, extracts assistant-message text,
and emits sliding-window (prefix, continuation) pairs into the store.
Assistant text is preferred over user text because the lookup cache
helps with *decoding* — that's what the model generates. User-typed
text never goes through the model's decode path. (The substrate spec's
autocomplete consumer would also want user text; that's a different
consumer, deferred.)

This module is idempotent at the row level via the store's upsert
semantics: re-seeding bumps observation_count rather than duplicating.
Operators can call it multiple times safely.
"""

from __future__ import annotations

import json
from typing import Any

from augmentum.observation.store import ObservationStore
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Multi-length sliding windows. Two consumers, two prefix-shape needs:
#
#   - **Lookup-cache exporter** wants long prefix → long continuation
#     pairs (8 words → 4 words) so llama.cpp's n-gram extractor has
#     real phrases to chew on.
#   - **Autocomplete consumer** wants short prefix → 1-word continuation
#     pairs (3 / 5 / 8 words → 1 word) so it can match against partial
#     user input as they type. A user 4 words into their message
#     wouldn't match an 8-word-only seed.
#
# We emit both. The shorter windows multiply the row count ~3-4×; the
# store's upsert path collapses duplicates so disk impact is bounded
# by actual vocabulary diversity, not the multiplier.
_LOOKUP_CACHE_PREFIX_WORDS = 8
_LOOKUP_CACHE_CONTINUATION_WORDS = 4

# Each entry: (prefix_word_count, continuation_word_count). Always
# include at least one short window for autocomplete; the 8/4 entry
# is what the lookup cache exporter consumes.
_AUTOCOMPLETE_WINDOWS: tuple[tuple[int, int], ...] = (
    (3, 1),
    (5, 1),
    (8, 1),
)

# Per-message cap on emitted windows (across ALL window sizes). A
# 1000-word assistant response would emit thousands of windows without
# this; the upsert collapses most via count++ but we'd still burn
# DB-write cycles on the redundant tail.
_MAX_WINDOWS_PER_MESSAGE = 240

# Per-session cap. A session with 200 long turns shouldn't dominate the
# user's profile relative to a session with 5 turns — diversity matters
# for the long-tail cache.
_MAX_WINDOWS_PER_SESSION = 4500

# Minimum word count for an assistant message to be worth processing.
# Short replies ("ok", "yes") add nothing the lookup cache can use.
_MIN_MESSAGE_WORDS = 12


async def seed_from_chat_history(
    store: ObservationStore,
    *,
    user_id: str,
    conn: Any,
    session_limit: int = 500,
) -> dict[str, int]:
    """Seed the user's L0 observations from their chat history.

    ``conn`` is the same aiosqlite connection ``store`` was constructed
    with — we don't grab it off the store internals because the store
    contract is intentionally limited to observe/query and shouldn't
    leak the connection.

    Returns counters: ``{sessions_scanned, messages_processed,
    windows_written}``. Useful for the admin endpoint's response.
    """
    if not user_id:
        return {"sessions_scanned": 0, "messages_processed": 0, "windows_written": 0}

    # Pull session blobs — newest first so a session cap that truncates
    # drops the oldest first. The user_id column was added in a later
    # migration; tolerate the legacy shape by trying the user-scoped
    # query first and falling back if the column is missing in tests.
    try:
        cursor = await conn.execute(
            """
            SELECT id, mode, data
            FROM ui_sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, session_limit),
        )
        rows = await cursor.fetchall()
    except Exception:
        log.debug("seeder_user_id_query_failed_falling_back", exc_info=True)
        cursor = await conn.execute(
            "SELECT id, mode, data FROM ui_sessions ORDER BY updated_at DESC LIMIT ?",
            (session_limit,),
        )
        rows = await cursor.fetchall()

    sessions_scanned = 0
    messages_processed = 0
    windows_written = 0

    for session_id, mode, raw_blob in rows:
        sessions_scanned += 1
        try:
            data = json.loads(raw_blob or "{}")
        except json.JSONDecodeError:
            continue

        tree = data.get("tree") or {}
        if not isinstance(tree, dict):
            continue

        session_windows = 0
        for node in tree.values():
            if session_windows >= _MAX_WINDOWS_PER_SESSION:
                break
            if not isinstance(node, dict):
                continue
            if node.get("role") != "assistant":
                continue

            content = node.get("content") or ""
            if not isinstance(content, str):
                continue
            words = content.split()
            if len(words) < _MIN_MESSAGE_WORDS:
                continue

            messages_processed += 1
            written_for_msg = await _emit_windows(
                store,
                words=words,
                user_id=user_id,
                surface="chat",
                mode=str(mode or ""),
                per_msg_cap=_MAX_WINDOWS_PER_MESSAGE,
            )
            windows_written += written_for_msg
            session_windows += written_for_msg

    log.info(
        "observation_seed_complete",
        user_id=user_id,
        sessions=sessions_scanned,
        messages=messages_processed,
        windows=windows_written,
    )
    return {
        "sessions_scanned": sessions_scanned,
        "messages_processed": messages_processed,
        "windows_written": windows_written,
    }


async def _emit_windows(
    store: ObservationStore,
    *,
    words: list[str],
    user_id: str,
    surface: str,
    mode: str,
    per_msg_cap: int,
) -> int:
    """Walk one message's words and emit (prefix, continuation) windows
    at the lookup-cache size AND the autocomplete sizes.

    Returns the total number of observations actually written across
    all window sizes. ``per_msg_cap`` is a hard ceiling — if we hit it
    mid-walk we stop, no matter how many windows would have been left.
    """
    if len(words) < _MIN_MESSAGE_WORDS:
        return 0

    written = 0

    # ── Lookup-cache window (long prefix, long continuation) ─────────
    lc_prefix_n = _LOOKUP_CACHE_PREFIX_WORDS
    lc_cont_n = _LOOKUP_CACHE_CONTINUATION_WORDS
    lc_stride_max = max(0, len(words) - lc_prefix_n - lc_cont_n + 1)
    for i in range(lc_stride_max):
        if written >= per_msg_cap:
            return written
        prefix = " ".join(words[i : i + lc_prefix_n])
        continuation = " ".join(
            words[i + lc_prefix_n : i + lc_prefix_n + lc_cont_n]
        )
        if not prefix or not continuation:
            continue
        await store.observe(
            user_id=user_id,
            prefix_text=prefix,
            continuation=continuation,
            surface=surface,
            mode=mode,
        )
        written += 1

    # ── Autocomplete windows (3/5/8 word prefix, 1-word continuation) ─
    # Short prefixes serve autocomplete's tail-match path. We emit each
    # size independently — the fingerprint differs per (prefix_text,
    # surface, mode) so a 3-word prefix never collides with a 5-word
    # one in the store.
    for prefix_n, cont_n in _AUTOCOMPLETE_WINDOWS:
        stride_max = max(0, len(words) - prefix_n - cont_n + 1)
        for i in range(stride_max):
            if written >= per_msg_cap:
                return written
            prefix = " ".join(words[i : i + prefix_n])
            continuation = " ".join(
                words[i + prefix_n : i + prefix_n + cont_n]
            )
            if not prefix or not continuation:
                continue
            await store.observe(
                user_id=user_id,
                prefix_text=prefix,
                continuation=continuation,
                surface=surface,
                mode=mode,
            )
            written += 1

    return written
