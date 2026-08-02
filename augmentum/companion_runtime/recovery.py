"""Recovery & rebuild paths (Lane 2 §9, Lane 3 §9, anti-dependency §7.2).

Three kinds of "let me change what you know about me" the user can take:

  rebuild_soft   — life event changed things. Wipe baselines + graduated
                   noticings + the about_him slice of the relationship
                   doc. Keep factual memories + about_me_with_him +
                   cooccurrence graph at half strength.
  rebuild_hard   — wipe everything in soft + factual memories. Relationship
                   doc rebuilt empty from the post-rebuild horizon. Keeps
                   the relationship entity (Becca will still know you);
                   she just doesn't carry old memories forward.
  delete_all     — hard delete cascade. ``(user_id, companion_id)`` corpus
                   wiped. Becca won't know the user next time they talk.
                   The "frictionless delete" commitment in Lane 2 §7.2.

The user-facing affordances live in the UI (Lane 4 §11 area, settings
panel); this module provides the database-level operations.

Recovery's other side: the "rerun as" affordance. When Becca picks
wrong, the user can one-click rerun the same intent against a different
tool/channel. Each rerun writes a DPO pair into companion_skill_archive
(the substrate from Sprint 4) so dispatch learns the user's preference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.state.backends.sqlite import savepoint, transactional_write
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RebuildResult:
    kind: str               # 'soft' | 'hard_reset'
    rows_affected: dict[str, int]
    rebuild_log_id: int | None
    note: str = ""


# ── Soft rebuild ─────────────────────────────────────────────────────

async def _soft_cascade(
    runtime: CompanionRuntime,
    conn,
    *,
    user_id: str,
    user_signal: str,
    note: str,
    rebuild_kind: str = "soft",
) -> tuple[dict[str, int], int]:
    """The soft-rebuild SQL, executed on the given ``conn`` with NO
    commit (the caller's ``transactional_write`` owns commit/rollback).

    ``rebuild_kind`` is written directly into the log row so a hard reset
    can run soft+hard in ONE transaction and log 'hard_reset' atomically
    — the prior INSERT-'soft'-then-UPDATE-'hard_reset' across two commits
    could crash mid-way and leave a hard reset logged as soft, then
    re-halve cooccurrence on retry (audit 2026-06-17).

    Returns ``(affected, rebuild_log_id)``.
    """
    affected: dict[str, int] = {}

    # 1. Wipe affect baselines so the next consolidation rebuilds from
    #    scratch off the post-rebuild horizon.
    cur = await conn.execute(
        "DELETE FROM companion_affect_baselines "
        "WHERE user_id = ? AND companion_id = ?",
        (user_id, runtime.companion_id),
    )
    affected["affect_baselines"] = cur.rowcount or 0

    # 2. Mark graduated noticings as suppressed (rebuild reason). We
    #    don't hard-delete — the journal entries themselves stay; the
    #    relationship doc reader filters on suppressed=0.
    cur = await conn.execute(
        "UPDATE companion_journal SET suppressed = 1, "
        "suppressed_reason = 'rebuild' "
        "WHERE companion_id = ? AND user_id = ? "
        "  AND entry_type = 'noticing' AND graduated_at IS NOT NULL "
        "  AND suppressed = 0",
        (runtime.companion_id, user_id),
    )
    affected["noticings_suppressed"] = cur.rowcount or 0

    # 3. Halve cooccurrence counts (keep the structure, fade the
    #    confidence). MAX(1, ...) preserves the signal floor so weak
    #    associations don't immediately vanish.
    cur = await conn.execute(
        "UPDATE personality_facet_cooccurrence "
        "SET count = MAX(1, CAST(count * 0.5 AS INTEGER)) "
        "WHERE user_id = ? AND companion_id = ?",
        (user_id, runtime.companion_id),
    )
    affected["cooccurrence_halved"] = cur.rowcount or 0

    # 4. Log the rebuild (kind written directly — see docstring).
    cur = await conn.execute(
        "INSERT INTO companion_rebuild_log "
        "(companion_id, user_id, rebuild_kind, user_signal, note) "
        "VALUES (?, ?, ?, ?, ?)",
        (runtime.companion_id, user_id, rebuild_kind, user_signal,
         note[:500] if note else ""),
    )
    rebuild_log_id = int(cur.lastrowid or 0)
    return affected, rebuild_log_id


async def rebuild_soft(
    runtime: CompanionRuntime,
    *,
    user_id: str,
    user_signal: str = "explicit_request",
    note: str = "",
) -> RebuildResult:
    """Soft reset: clear affect baselines + graduated noticings + the
    about_him slice. Keep memories + about_me_with_him + cooccurrence.

    Writes a row to companion_rebuild_log so the relationship-doc
    digester knows the horizon for "about_him" content. The whole
    cascade is one transaction — a mid-cascade failure rolls back fully
    (no half-state, no orphaned log row) (audit 2026-06-17).
    """
    if not user_id:
        raise ValueError("rebuild_soft: user_id required")
    backend = runtime.backend

    async with transactional_write(backend.conn) as conn:
        affected, rebuild_log_id = await _soft_cascade(
            runtime, conn, user_id=user_id,
            user_signal=user_signal, note=note, rebuild_kind="soft",
        )

    # Post-commit: in-memory cache invalidate + observers only see
    # committed state.
    try:
        runtime.memory.mark_relationship_stale(user_id)
    except Exception:
        log.warning("relationship_cache_invalidate_failed", user_id=user_id, exc_info=True)

    await runtime.bus.publish_topic(
        "rebuild.completed",
        {"user_id": user_id, "kind": "soft", "rebuild_log_id": rebuild_log_id,
         "affected": affected},
        source_companion_id=runtime.companion_id,
    )

    log.info("companion_rebuild_soft", user_id=user_id, affected=affected)
    return RebuildResult(
        kind="soft", rows_affected=affected,
        rebuild_log_id=rebuild_log_id, note=note,
    )


# ── Hard reset (relationship continues, memories wiped) ──────────────

async def rebuild_hard(
    runtime: CompanionRuntime,
    *,
    user_id: str,
    user_signal: str = "settings_panel",
    note: str = "",
) -> RebuildResult:
    """Hard reset: soft + wipe factual memories. The relationship
    entity continues (Becca will still know you); she just doesn't
    carry old memories forward."""
    if not user_id:
        raise ValueError("rebuild_hard: user_id required")
    backend = runtime.backend

    # Soft cascade + the factual-memory wipe run in ONE transaction so a
    # crash can't leave a committed soft logged as hard, and a retry
    # can't re-halve cooccurrence (audit 2026-06-17). The log row is
    # written as 'hard_reset' directly inside _soft_cascade.
    async with transactional_write(backend.conn) as conn:
        affected, rebuild_log_id = await _soft_cascade(
            runtime, conn, user_id=user_id,
            user_signal=user_signal, note=note, rebuild_kind="hard_reset",
        )

        # Wipe memories (the factual tier). Mirror what hard delete would
        # do for memories, but leave the (user_id, companion_id) row alive.
        cur = await conn.execute(
            "DELETE FROM memories WHERE user_id = ? AND companion_id = ?",
            (user_id, runtime.companion_id),
        )
        affected["memories_wiped"] = cur.rowcount or 0

        # Wipe activations & memory associations (personality substrate)
        cur = await conn.execute(
            "DELETE FROM personality_facet_activations "
            "WHERE user_id = ? AND companion_id = ?",
            (user_id, runtime.companion_id),
        )
        affected["activations_wiped"] = cur.rowcount or 0
        cur = await conn.execute(
            "DELETE FROM personality_memory_associations "
            "WHERE user_id = ? AND companion_id = ?",
            (user_id, runtime.companion_id),
        )
        affected["memory_associations_wiped"] = cur.rowcount or 0

    # Post-commit.
    try:
        runtime.memory.mark_relationship_stale(user_id)
    except Exception:
        log.warning("relationship_cache_invalidate_failed", user_id=user_id, exc_info=True)

    await runtime.bus.publish_topic(
        "rebuild.completed",
        {"user_id": user_id, "kind": "hard_reset",
         "rebuild_log_id": rebuild_log_id, "affected": affected},
        source_companion_id=runtime.companion_id,
    )

    log.info("companion_rebuild_hard", user_id=user_id, affected=affected)
    return RebuildResult(
        kind="hard_reset", rows_affected=affected,
        rebuild_log_id=rebuild_log_id, note=note,
    )


# ── Frictionless delete (Lane 2 §7.2) ────────────────────────────────

async def delete_all(
    runtime: CompanionRuntime,
    *,
    user_id: str,
) -> dict[str, int]:
    """Hard-delete cascade for ``(user_id, companion_id)``.

    No "are you sure" prompts at this layer — the UI handles
    confirmation. The promise: leaving is easy. Becca won't know the
    user next time they talk.

    Returns row-counts per table for audit.
    """
    if not user_id:
        raise ValueError("delete_all: user_id required")
    backend = runtime.backend
    affected: dict[str, int] = {}

    tables_user_companion_scoped = [
        "memories",
        "personality_facet_activations",
        "personality_facet_cooccurrence",
        "personality_memory_associations",
        "companion_affect_baselines",
    ]

    # One transaction for the whole cascade: a failure on any table rolls
    # back everything rather than leaving a half-deleted user with no
    # completion marker (audit 2026-06-17). The prior per-table
    # try/except silently skipped failing tables — removed so a real
    # failure aborts atomically. vec0 cleanup stays best-effort via a
    # savepoint (the virtual table may not be loaded).
    async with transactional_write(backend.conn) as conn:
        for table in tables_user_companion_scoped:
            cur = await conn.execute(
                f"DELETE FROM {table} WHERE user_id = ? AND companion_id = ?",
                (user_id, runtime.companion_id),
            )
            affected[table] = cur.rowcount or 0

        # Pre-collect journal ids so we can clear the vec0 mirror —
        # FTS5 cleanup is handled by the AFTER DELETE trigger from
        # migration 177, but vec0 isn't triggered (virtual table).
        cur = await conn.execute(
            "SELECT id FROM companion_journal WHERE companion_id = ? AND user_id = ?",
            (runtime.companion_id, user_id),
        )
        journal_ids = [r[0] for r in await cur.fetchall()]
        await cur.close()

        cur = await conn.execute(
            "DELETE FROM companion_journal WHERE companion_id = ? AND user_id = ?",
            (runtime.companion_id, user_id),
        )
        affected["companion_journal"] = cur.rowcount or 0

        # vec0 mirror cleanup — best-effort, isolated in a savepoint so a
        # missing extension can't abort the outer cascade.
        if journal_ids:
            try:
                async with savepoint(conn, "vec_cleanup"):
                    for i in range(0, len(journal_ids), 200):
                        batch = journal_ids[i : i + 200]
                        placeholders = ",".join("?" * len(batch))
                        await conn.execute(
                            f"DELETE FROM companion_journal_vec WHERE journal_id IN ({placeholders})",
                            batch,
                        )
            except Exception as exc:
                log.warning("journal_vec_cleanup_failed", error=str(exc)[:200])

        cur = await conn.execute(
            "DELETE FROM companion_observations "
            "WHERE companion_id = ? AND target_user_id = ?",
            (runtime.companion_id, user_id),
        )
        affected["companion_observations"] = cur.rowcount or 0

        # Rebuild log: keep the record of having existed, but mark as
        # deleted so future digestion knows. (Some users come back; we
        # don't pretend the prior relationship didn't happen even when
        # we erase its content.)
        await conn.execute(
            "INSERT INTO companion_rebuild_log "
            "(companion_id, user_id, rebuild_kind, user_signal, note) "
            "VALUES (?, ?, 'delete_all', 'settings_panel', '')",
            (runtime.companion_id, user_id),
        )

    # Invalidate the relationship cache (post-commit).
    try:
        runtime.memory.mark_relationship_stale(user_id)
    except Exception:
        log.warning("relationship_cache_invalidate_failed", user_id=user_id, exc_info=True)

    await runtime.bus.publish_topic(
        "delete_all.completed",
        {"user_id": user_id, "affected": affected},
        source_companion_id=runtime.companion_id,
    )

    log.info("companion_delete_all", user_id=user_id, affected=affected)
    return affected


# ── Rerun affordance (Lane 3 §9) ─────────────────────────────────────

async def record_rerun_pair(
    runtime: CompanionRuntime,
    *,
    user_id: str,
    intent_text: str,
    original_winner: str,
    chosen_target: str,
    delta: float = 0.7,
) -> None:
    """Write a DPO pair into companion_skill_archive when the user
    explicitly reruns a turn against a different tool/channel.

    The dispatcher's ``preference_delta`` reads these pairs at decision
    time so future similar intents bias toward what the user picked.

    feedback_strength = 0.7 — strong but not maximal; the user might
    have been curious rather than dissatisfied.
    """
    if not user_id or not intent_text or not original_winner or not chosen_target:
        return
    try:
        from augmentum.companion_runtime import skill_archive
        from augmentum.companion_runtime.runtime import Intent

        intent = Intent(text=intent_text, user_id=user_id, source="rerun")
        # Two outcomes: the user's choice (+1) and the original Becca
        # pick (-1 because they rejected it).
        await skill_archive.record_outcome(
            runtime, intent,
            chosen_subagent=chosen_target,
            outcome_signal=+1.0 * delta,
            outcome_reason="user_rerun_preferred",
            decision_ms=0.0,
            used_tiebreaker=False,
        )
        await skill_archive.record_outcome(
            runtime, intent,
            chosen_subagent=original_winner,
            outcome_signal=-1.0 * delta,
            outcome_reason="user_rerun_rejected",
            decision_ms=0.0,
            used_tiebreaker=False,
        )
    except Exception:
        log.debug("record_rerun_pair_failed", exc_info=True)


__all__ = [
    "RebuildResult",
    "rebuild_soft",
    "rebuild_hard",
    "delete_all",
    "record_rerun_pair",
]
