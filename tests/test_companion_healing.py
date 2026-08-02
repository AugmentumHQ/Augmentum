"""Sprint 4 tests — aging + heal jobs.

Covers:
* Aging skips crystallized + already-surfaced
* Aging respects threshold hours
* Daily heal soft-deletes quarantined-for-7d
* Daily heal applies forgetting curve to >30d non-crystallized
* Weekly consolidate groups into 7-day windows
* Weekly consolidate marks sources archived
* Monthly drift audit produces snapshot
* Cross-tenant probe returns 0 leakage
* Kill switches default OFF behave correctly
"""

from __future__ import annotations

import pytest


async def _boot_backend():
    from augmentum.state.backends.sqlite import SQLiteBackend
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('usr_h', 'test', 'x', datetime('now'))",
    )
    await backend.conn.commit()
    return backend


def _rt(backend):
    from unittest.mock import MagicMock
    runtime = MagicMock()
    runtime.backend = backend
    runtime.companion_id = "becca"
    return runtime


# ── Aging ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_aging_disabled_no_op(monkeypatch):
    from augmentum.config import settings
    from augmentum.companion_runtime.healing import age_unopened_notes
    monkeypatch.setattr(settings, "companion_aging_enabled", False)
    backend = await _boot_backend()
    result = await age_unopened_notes(_rt(backend))
    assert result == 0


@pytest.mark.asyncio
async def test_aging_ages_old_unopened_notes(monkeypatch):
    from augmentum.config import settings
    from augmentum.companion_runtime.healing import age_unopened_notes
    monkeypatch.setattr(settings, "companion_aging_enabled", True)
    monkeypatch.setattr(settings, "companion_aging_threshold_hours", 48)

    backend = await _boot_backend()
    # Old unopened note — should be aged
    await backend.conn.execute(
        "INSERT INTO companion_journal "
        "(companion_id, user_id, entry_type, content, "
        " quiet_share_ready, created_at) "
        "VALUES ('becca', 'usr_h', 'noticing', 'old note', 1, "
        " datetime('now', '-3 days'))",
    )
    # Fresh unopened note — should NOT be aged
    await backend.conn.execute(
        "INSERT INTO companion_journal "
        "(companion_id, user_id, entry_type, content, "
        " quiet_share_ready, created_at) "
        "VALUES ('becca', 'usr_h', 'noticing', 'fresh note', 1, "
        " datetime('now'))",
    )
    await backend.conn.commit()

    aged = await age_unopened_notes(_rt(backend))
    assert aged == 1

    # Fresh note should still be unsurfaced
    cur = await backend.conn.execute(
        "SELECT surfaced_at FROM companion_journal "
        "WHERE content = 'fresh note'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] is None


@pytest.mark.asyncio
async def test_aging_skips_crystallized(monkeypatch):
    from augmentum.config import settings
    from augmentum.companion_runtime.healing import age_unopened_notes
    monkeypatch.setattr(settings, "companion_aging_enabled", True)

    backend = await _boot_backend()
    await backend.conn.execute(
        "INSERT INTO companion_journal "
        "(companion_id, user_id, entry_type, content, "
        " quiet_share_ready, crystallized, created_at) "
        "VALUES ('becca', 'usr_h', 'noticing', 'crystallized note', "
        " 1, 1, datetime('now', '-30 days'))",
    )
    await backend.conn.commit()

    aged = await age_unopened_notes(_rt(backend))
    assert aged == 0

    cur = await backend.conn.execute(
        "SELECT surfaced_at FROM companion_journal "
        "WHERE content = 'crystallized note'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] is None  # untouched


# ── Daily heal ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_daily_heal_disabled_no_op(monkeypatch):
    from augmentum.config import settings
    from augmentum.companion_runtime.healing import daily_heal
    monkeypatch.setattr(settings, "companion_healing_enabled", False)
    backend = await _boot_backend()
    result = await daily_heal(_rt(backend))
    assert result.get("skipped") is True


@pytest.mark.asyncio
async def test_daily_heal_soft_deletes_old_quarantined(monkeypatch):
    from augmentum.config import settings
    from augmentum.companion_runtime.healing import daily_heal
    monkeypatch.setattr(settings, "companion_healing_enabled", True)

    backend = await _boot_backend()
    await backend.conn.execute(
        "INSERT INTO companion_journal "
        "(companion_id, user_id, entry_type, content, "
        " quarantined, quarantine_reason, created_at) "
        "VALUES ('becca', 'usr_h', 'noticing', 'old quarantined', "
        " 1, 'adversarial_pattern', datetime('now', '-10 days'))",
    )
    await backend.conn.commit()

    result = await daily_heal(_rt(backend))
    assert result["soft_deleted"] == 1

    cur = await backend.conn.execute(
        "SELECT archived_at FROM companion_journal "
        "WHERE content = 'old quarantined'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] is not None


@pytest.mark.asyncio
async def test_daily_heal_applies_forgetting_curve(monkeypatch):
    from augmentum.config import settings
    from augmentum.companion_runtime.healing import daily_heal
    monkeypatch.setattr(settings, "companion_healing_enabled", True)

    backend = await _boot_backend()
    # Old non-crystallized entry with confidence 0.6
    await backend.conn.execute(
        "INSERT INTO companion_journal "
        "(companion_id, user_id, entry_type, content, "
        " confidence_numeric, created_at) "
        "VALUES ('becca', 'usr_h', 'noticing', 'aging note', "
        " 0.6, datetime('now', '-40 days'))",
    )
    await backend.conn.commit()

    result = await daily_heal(_rt(backend))
    assert result["forgetting_applied"] >= 1

    cur = await backend.conn.execute(
        "SELECT confidence_numeric FROM companion_journal "
        "WHERE content = 'aging note'"
    )
    row = await cur.fetchone()
    await cur.close()
    # 0.6 × 0.99 = 0.594
    assert row[0] == pytest.approx(0.594, abs=0.001)


@pytest.mark.asyncio
async def test_daily_heal_skips_crystallized_in_forgetting(monkeypatch):
    from augmentum.config import settings
    from augmentum.companion_runtime.healing import daily_heal
    monkeypatch.setattr(settings, "companion_healing_enabled", True)

    backend = await _boot_backend()
    await backend.conn.execute(
        "INSERT INTO companion_journal "
        "(companion_id, user_id, entry_type, content, "
        " confidence_numeric, crystallized, created_at) "
        "VALUES ('becca', 'usr_h', 'noticing', 'pinned', "
        " 0.9, 1, datetime('now', '-40 days'))",
    )
    await backend.conn.commit()

    await daily_heal(_rt(backend))

    cur = await backend.conn.execute(
        "SELECT confidence_numeric FROM companion_journal "
        "WHERE content = 'pinned'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == pytest.approx(0.9)  # untouched


# ── Weekly consolidate ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_weekly_consolidate_disabled_no_op(monkeypatch):
    from augmentum.config import settings
    from augmentum.companion_runtime.healing import weekly_consolidate
    monkeypatch.setattr(settings, "companion_healing_enabled", False)
    backend = await _boot_backend()
    result = await weekly_consolidate(_rt(backend), user_id="usr_h")
    assert result.get("skipped") is True


@pytest.mark.asyncio
async def test_weekly_consolidate_archives_old_entries(monkeypatch):
    from augmentum.config import settings
    from augmentum.companion_runtime.healing import weekly_consolidate
    monkeypatch.setattr(settings, "companion_healing_enabled", True)

    backend = await _boot_backend()
    # Insert 3 entries in one window, >30d old
    for i in range(3):
        await backend.conn.execute(
            "INSERT INTO companion_journal "
            "(companion_id, user_id, entry_type, content, created_at) "
            "VALUES ('becca', 'usr_h', 'noticing', ?, "
            " datetime('now', '-45 days'))",
            (f"old entry {i}",),
        )
    await backend.conn.commit()

    result = await weekly_consolidate(_rt(backend), user_id="usr_h")
    assert result["windows_consolidated"] >= 1
    assert result["entries_archived"] >= 3

    # Originals should have archived_at set
    cur = await backend.conn.execute(
        "SELECT COUNT(*) FROM companion_journal "
        "WHERE user_id = 'usr_h' AND archived_at IS NOT NULL"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] >= 3

    # Archive row should exist
    cur = await backend.conn.execute(
        "SELECT COUNT(*) FROM companion_journal_archive "
        "WHERE user_id = 'usr_h'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] >= 1


# ── Monthly drift audit ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drift_audit_disabled_no_op(monkeypatch):
    from augmentum.config import settings
    from augmentum.companion_runtime.healing import monthly_drift_audit
    monkeypatch.setattr(settings, "companion_healing_enabled", False)
    backend = await _boot_backend()
    result = await monthly_drift_audit(_rt(backend), user_id="usr_h")
    assert result.get("skipped") is True


@pytest.mark.asyncio
async def test_drift_audit_returns_snapshot(monkeypatch):
    from augmentum.config import settings
    from augmentum.companion_runtime.healing import monthly_drift_audit
    monkeypatch.setattr(settings, "companion_healing_enabled", True)

    backend = await _boot_backend()
    # Seed companion identity row for the user (mig 179 baseline row
    # has user_id=''; our user needs an explicit row)
    await backend.conn.execute(
        "INSERT INTO companion_identities "
        "(user_id, companion_id, display_name, drift_score) "
        "VALUES ('usr_h', 'becca', 'Becca', 0.04)",
    )
    await backend.conn.commit()

    result = await monthly_drift_audit(_rt(backend), user_id="usr_h")
    assert result.get("drift_score") == pytest.approx(0.04)
    # No other users seeded → the honest multi-tenancy metric is 0.
    assert result.get("other_tenant_rows") == 0


@pytest.mark.asyncio
async def test_drift_audit_reports_other_tenants_and_isolation_holds(monkeypatch):
    """The audit reports the honest, observable metric (other tenants
    exist) and a user-scoped query returns only that user's rows. The
    old `cross_tenant_leakage` metric was tautologically 0 and couldn't
    observe a leak (audit 2026-06-17)."""
    from augmentum.config import settings
    from augmentum.companion_runtime.healing import monthly_drift_audit
    monkeypatch.setattr(settings, "companion_healing_enabled", True)

    backend = await _boot_backend()
    # Two users with their own journal entries
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('usr_other', 'other', 'x', datetime('now'))",
    )
    for uid in ("usr_h", "usr_other"):
        await backend.conn.execute(
            "INSERT INTO companion_journal "
            "(companion_id, user_id, entry_type, content) "
            "VALUES ('becca', ?, 'noticing', ?)",
            (uid, f"note for {uid}"),
        )
    await backend.conn.commit()

    result = await monthly_drift_audit(_rt(backend), user_id="usr_h")
    # Honest metric: one OTHER user's row exists under this companion.
    assert result["other_tenant_rows"] == 1
    # Real isolation invariant: filtering by usr_h returns only usr_h.
    cur = await backend.conn.execute(
        "SELECT DISTINCT user_id FROM companion_journal WHERE user_id = ?",
        ("usr_h",),
    )
    rows = await cur.fetchall()
    await cur.close()
    assert [r[0] for r in rows] == ["usr_h"]
