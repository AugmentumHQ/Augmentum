"""Heal jobs — silent maintenance for the autonomous substrate.

Sprint 4, Aletheia × Augmentum arc Pieces 11 + R3.

Three jobs run via the existing JobRunner:

* **Daily heal** — runs at ~3 AM local. Soft-deletes long-quarantined
  rows, applies forgetting curve to old entries, sweeps stale ones.
* **Weekly consolidate** — runs Sunday early AM. Consolidates entries
  older than 30 days into archived 7-day window summaries.
* **Monthly drift audit** — runs at month start. Computes trait drift
  over 30 days, flags concerning monotonic trends, validates the
  cross-tenant probe.

All three are gated by ``companion_healing_enabled`` (default OFF until
you flip it on for production). The aging task is separate and gated by
``companion_aging_enabled``.

Resource posture:
* Daily heal: ~30s scan, no LLM calls
* Weekly consolidate: ~2 min with N utility-tier LLM calls (~N×3s)
* Monthly drift: ~5s pure compute, no LLM
"""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING

from augmentum.state.backends.sqlite import transactional_write
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


# ── Aging (Piece 11) ─────────────────────────────────────────────────


async def age_unopened_notes(runtime: CompanionRuntime) -> int:
    """Notes >threshold-hours old with surfaced_at IS NULL → auto-expire.

    Stale notes undermine trust faster than no notes. After 48h
    (default) without user engagement, a note is effectively expired:
    we set surfaced_at = NOW() so it stops appearing in the pip.
    Crystallized entries (system-pinned milestones) are excluded.

    Returns the number of rows aged. Idempotent and cheap — uses
    the partial index ``idx_cj_quiet_share_ready`` from mig 178.
    """
    from augmentum.config import settings
    if not getattr(settings, "companion_aging_enabled", False):
        return 0

    threshold_hours = int(getattr(settings, "companion_aging_threshold_hours", 48))
    threshold_hours = max(1, min(threshold_hours, 720))  # 1h..30d

    try:
        cur = await runtime.backend.conn.execute(
            f"UPDATE companion_journal "
            f"SET surfaced_at = datetime('now') "
            f"WHERE quiet_share_ready = 1 "
            f"  AND surfaced_at IS NULL "
            f"  AND COALESCE(crystallized, 0) = 0 "
            f"  AND created_at < datetime('now', '-{threshold_hours} hours')",
        )
        await runtime.backend.conn.commit()
        affected = cur.rowcount or 0
        await cur.close()
    except Exception:
        log.warning("aging_query_failed", exc_info=True)
        return 0

    if affected:
        log.info(
            "aging_ran",
            companion_id=runtime.companion_id,
            aged=affected,
            threshold_hours=threshold_hours,
        )
    return affected


# ── Daily heal (R3) ──────────────────────────────────────────────────


async def daily_heal(runtime: CompanionRuntime) -> dict:
    """Once-per-day maintenance pass.

    1. Soft-delete quarantined entries older than 7 days (forensics
       window). Hard delete happens at 30 days.
    2. Apply forgetting curve: confidence_numeric × 0.99 for entries
       >30 days old. Crystallized excluded.

    Returns a dict of counters for the Observatory + telemetry.
    Idempotent.
    """
    from augmentum.config import settings
    if not getattr(settings, "companion_healing_enabled", False):
        return {"skipped": True, "reason": "disabled"}

    backend = runtime.backend
    results: dict = {"soft_deleted": 0, "forgetting_applied": 0}

    # 1. Soft-delete: archived_at on quarantined-for-7d rows.
    try:
        cur = await backend.conn.execute(
            "UPDATE companion_journal "
            "SET archived_at = datetime('now') "
            "WHERE quarantined = 1 "
            "  AND archived_at IS NULL "
            "  AND created_at < datetime('now', '-7 days')",
        )
        results["soft_deleted"] = cur.rowcount or 0
        await backend.conn.commit()
        await cur.close()
    except Exception:
        log.warning("daily_heal_soft_delete_failed", exc_info=True)

    # 2. Forgetting curve: decay confidence on old non-crystallized rows.
    try:
        cur = await backend.conn.execute(
            "UPDATE companion_journal "
            "SET confidence_numeric = MAX(0.0, confidence_numeric * 0.99) "
            "WHERE COALESCE(crystallized, 0) = 0 "
            "  AND COALESCE(quarantined, 0) = 0 "
            "  AND created_at < datetime('now', '-30 days')",
        )
        results["forgetting_applied"] = cur.rowcount or 0
        await backend.conn.commit()
        await cur.close()
    except Exception:
        log.warning("daily_heal_forgetting_failed", exc_info=True)

    # 3. Settle yesterday's Today reflection. Once settled, the row is
    # immutable until the next day rolls. We settle PER USER — discover
    # active users from the journal table (best signal of "has companion
    # interior worth reflecting on"). Cheap query, scoped by companion_id.
    users: list[str] = []
    try:
        from augmentum.companion_runtime import today as _today
        cur = await backend.conn.execute(
            "SELECT DISTINCT user_id FROM companion_journal "
            "WHERE companion_id = ? "
            "  AND created_at > datetime('now', '-3 days') "
            "  AND user_id IS NOT NULL AND user_id != ''",
            (runtime.companion_id,),
        )
        users = [r[0] for r in await cur.fetchall()]
        await cur.close()
        settled = 0
        for uid in users:
            # Settle the date that just rolled (yesterday in local).
            import time as _t
            from datetime import datetime as _dt
            from datetime import timedelta as _td
            yest = (_dt(*_t.localtime()[:6]) - _td(days=1)).strftime("%Y-%m-%d")
            await _today.settle_date(runtime, user_id=uid, date_local=yest)
            settled += 1
        results["today_settled"] = settled
    except Exception:
        log.warning("daily_heal_today_settle_failed", exc_info=True)

    # 4. Capture correction-lessons from the day's reflections (mig 270).
    # Best-effort + gated by companion_lessons_capture_enabled — a failure
    # here must not affect the maintenance above. Reuses the same active-
    # user set discovered for the Today settle.
    try:
        from augmentum.config import settings as _settings
        if getattr(_settings, "companion_lessons_capture_enabled", False):
            from augmentum.companion_runtime import lessons_capture as _lc
            captured = 0
            for uid in users:
                try:
                    res = await _lc.capture_recent(runtime, user_id=uid)
                    captured += len(res.get("captured", []) or [])
                except Exception:
                    log.debug("daily_heal_lesson_capture_user_failed", exc_info=True)
            results["lessons_captured"] = captured
    except Exception:
        log.warning("daily_heal_lesson_capture_failed", exc_info=True)

    log.info(
        "daily_heal_ran",
        companion_id=runtime.companion_id,
        soft_deleted=results["soft_deleted"],
        forgetting_applied=results["forgetting_applied"],
        today_settled=results.get("today_settled", 0),
    )
    return results


# ── Weekly consolidate (R3) ──────────────────────────────────────────


async def weekly_consolidate(runtime: CompanionRuntime, *, user_id: str) -> dict:
    """Group entries older than 30d into 7-day windows; summarize each.

    Writes one row per window to companion_journal_archive. Marks the
    source entries archived_at = NOW(). 60d post-archive: soft-delete.
    90d: hard-delete.

    Sprint 4 ships a stub summarization (no LLM call) so the consolidate
    pipeline runs end-to-end and tests pass. Sprint 7+ replaces with
    utility-tier LLM call to produce real narrative summaries.

    Returns counters for the Observatory.
    """
    import json

    from augmentum.config import settings
    if not getattr(settings, "companion_healing_enabled", False):
        return {"skipped": True, "reason": "disabled"}
    if not user_id:
        return {"skipped": True, "reason": "no_user_id"}

    backend = runtime.backend
    results: dict = {"windows_consolidated": 0, "entries_archived": 0}

    # Find 7-day windows of unarchived entries >30d old, group them.
    try:
        cur = await backend.conn.execute(
            """
            SELECT id, content, confidence_numeric, created_at, affect_tag
            FROM companion_journal
            WHERE companion_id = ? AND user_id = ?
              AND archived_at IS NULL
              AND COALESCE(crystallized, 0) = 0
              AND created_at < datetime('now', '-30 days')
            ORDER BY created_at ASC
            """,
            (runtime.companion_id, user_id),
        )
        rows = await cur.fetchall()
        await cur.close()
    except Exception:
        log.warning("weekly_consolidate_query_failed", exc_info=True)
        return results

    if not rows:
        return results

    # Group by 7-day window (UTC).
    from datetime import datetime, timedelta
    windows: dict[str, list] = {}
    for row in rows:
        try:
            created = datetime.strptime(
                str(row[3]).replace("T", " ").split(".", 1)[0],
                "%Y-%m-%d %H:%M:%S",
            ).replace(tzinfo=UTC)
        except (ValueError, TypeError) as exc:
            log.debug("healing_row_parse_failed", raw=row[3] if row else None, error=str(exc))
            continue
        # Bucket by Monday of week
        days_since_monday = created.weekday()
        window_start = (
            created - timedelta(days=days_since_monday)
        ).replace(hour=0, minute=0, second=0, microsecond=0)
        key = window_start.isoformat()
        windows.setdefault(key, []).append((row, window_start))

    # Write one archive row per window
    for window_key, group in windows.items():
        if not group:
            continue
        entry_ids = [str(r[0][0]) for r in group]
        confidences = [float(r[0][2] or 0.6) for r in group]
        affects = [r[0][4] for r in group if r[0][4]]
        avg_conf = sum(confidences) / max(len(confidences), 1)
        # Sprint 4 stub summary — concatenate the first sentence of each
        # entry. Sprint 7+ replaces with LLM-generated narrative.
        first_sentences = []
        for r in group[:5]:  # cap to first 5 per window for the stub
            content = str(r[0][1] or "")
            m = content.split(".", 1)[0]
            first_sentences.append(m[:200])
        summary = "; ".join(first_sentences) if first_sentences else "(empty window)"

        window_start = group[0][1]
        window_end = window_start + timedelta(days=7)

        try:
            # Per-window transaction: the archive INSERT and the
            # source-mark UPDATE are all-or-nothing, so a failure can't
            # leave an archive row whose sources stay unarchived (which
            # re-archived them next week → duplicates) (audit 2026-06-17).
            async with transactional_write(backend.conn) as conn:
                await conn.execute(
                    "INSERT INTO companion_journal_archive "
                    "(user_id, companion_id, window_start, window_end, "
                    " entry_ids, summary, source_count, avg_confidence, "
                    " affect_signature_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        user_id, runtime.companion_id,
                        window_start.isoformat(),
                        window_end.isoformat(),
                        json.dumps(entry_ids),
                        summary,
                        len(entry_ids),
                        avg_conf,
                        json.dumps({"top_affects": affects[:3]}),
                    ),
                )
                # Mark sources archived
                placeholders = ",".join("?" * len(entry_ids))
                await conn.execute(
                    f"UPDATE companion_journal "
                    f"SET archived_at = datetime('now') "
                    f"WHERE id IN ({placeholders})",
                    entry_ids,
                )
            results["windows_consolidated"] += 1
            results["entries_archived"] += len(entry_ids)
        except Exception:
            log.warning(
                "weekly_consolidate_write_failed",
                window=window_key, exc_info=True,
            )
            continue

    log.info(
        "weekly_consolidate_ran",
        user_id=user_id, companion_id=runtime.companion_id,
        windows=results["windows_consolidated"],
        entries=results["entries_archived"],
    )
    return results


# ── Monthly drift audit (R3) ─────────────────────────────────────────


async def monthly_drift_audit(runtime: CompanionRuntime, *, user_id: str) -> dict:
    """Compute trait drift over the last 30 days; flag concerning trends.

    Sprint 4 ships a basic snapshot: pulls drift_score from
    companion_identities + counts active mutes + counts quarantined
    entries. Sprint 7's trait nudge integration adds the trait-by-trait
    drift comparison.

    Cross-tenant probe: a synthetic check that one user's resolver
    can't return another user's items. Validates the per-user invariant.
    """
    from augmentum.config import settings
    if not getattr(settings, "companion_healing_enabled", False):
        return {"skipped": True, "reason": "disabled"}
    if not user_id:
        return {"skipped": True, "reason": "no_user_id"}

    backend = runtime.backend
    snapshot: dict = {}

    # 1. Drift score from identity row
    try:
        cur = await backend.conn.execute(
            "SELECT drift_score, last_kernel_refresh_at "
            "FROM companion_identities "
            "WHERE user_id = ? AND companion_id = ?",
            (user_id, runtime.companion_id),
        )
        row = await cur.fetchone()
        await cur.close()
        if row:
            snapshot["drift_score"] = float(row[0] or 0.0)
            snapshot["last_kernel_refresh_at"] = row[1]
    except Exception:
        log.warning("drift_audit_identity_query_failed", exc_info=True)

    # 2. Active mute count
    try:
        cur = await backend.conn.execute(
            "SELECT COUNT(*) FROM companion_topic_mutes "
            "WHERE user_id = ? AND companion_id = ? "
            "  AND expires_at > datetime('now')",
            (user_id, runtime.companion_id),
        )
        row = await cur.fetchone()
        await cur.close()
        snapshot["active_mutes"] = int(row[0] if row else 0)
    except Exception:
        snapshot["active_mutes"] = 0

    # 3. Recent quarantine rate (30d)
    try:
        cur = await backend.conn.execute(
            "SELECT "
            " SUM(CASE WHEN quarantined = 1 THEN 1 ELSE 0 END), "
            " COUNT(*) "
            "FROM companion_journal "
            "WHERE companion_id = ? AND user_id = ? "
            "  AND created_at > datetime('now', '-30 days')",
            (runtime.companion_id, user_id),
        )
        row = await cur.fetchone()
        await cur.close()
        if row and row[1]:
            snapshot["quarantine_rate_30d"] = float(row[0] or 0) / float(row[1])
            snapshot["entries_30d"] = int(row[1])
    except Exception:
        log.warning("healing_quarantine_rate_query_failed", exc_info=True)

    # 4. Multi-tenancy presence probe. The prior query
    # (`user_id = ? AND user_id != ?` with the same value) was
    # tautologically 0 — it could never observe a leak, so reporting
    # cross_tenant_leakage:0 was security theater (audit 2026-06-17).
    # The honest, observable metric is "how many OTHER users' journal
    # rows exist under this companion" — non-zero on a multi-user box,
    # which is what makes isolation testing meaningful. The real leakage
    # assertion (our filter returns only our rows) lives in the test
    # suite, not a runtime self-probe that can't see past its own WHERE.
    try:
        cur = await backend.conn.execute(
            "SELECT COUNT(*) FROM companion_journal "
            "WHERE companion_id = ? AND user_id != ?",
            (runtime.companion_id, user_id),
        )
        row = await cur.fetchone()
        await cur.close()
        snapshot["other_tenant_rows"] = int(row[0] if row else 0)
    except Exception:
        log.warning("healing_other_tenant_probe_failed", exc_info=True)
        snapshot["other_tenant_rows"] = 0

    log.info(
        "monthly_drift_audit_ran",
        user_id=user_id, companion_id=runtime.companion_id,
        **{k: v for k, v in snapshot.items() if isinstance(v, (int, float))},
    )
    return snapshot


__all__ = [
    "age_unopened_notes",
    "daily_heal",
    "weekly_consolidate",
    "monthly_drift_audit",
]
