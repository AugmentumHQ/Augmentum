"""Phase 1 substrate tests for the companion growth loop.

Covers:

  * Migration schema can be applied to ``:memory:`` cleanly
  * Economy mana regen (lazy on read, capped)
  * Economy mana debit (allowed + rejected on insufficient)
  * Economy berry earn / spend / vouch / veto / sponsor
  * GrowthStore backlog + session round-trips
  * Recall action with mock memory store (and without)
  * CompanionGrowthSession full lifecycle (plan → act → archive)
  * Reward signal channels (explicit / implicit, including unknown signals)

Pure-aiosqlite tests — no app stack, no FastAPI client — so the slice is
exercisable without spinning up the full server.
"""

from __future__ import annotations

import time
from pathlib import Path

import aiosqlite
import pytest

from augmentum.companion.growth import (
    CompanionGrowthSession,
    Economy,
    GrowthStore,
)
from augmentum.companion.growth.actions import ACTIONS, ActionRequest
from augmentum.companion.growth.rewards import (
    apply_explicit,
    apply_implicit,
    apply_restraint_credit,
)
from augmentum.companion.growth.session import SessionConfig

# ---------------------------------------------------------------------------
# Schema bootstrapping for the in-memory DB
# ---------------------------------------------------------------------------

# Tests apply the real migration file so schema drift is caught here, not
# in production. The migration is idempotent (CREATE TABLE IF NOT EXISTS).

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "augmentum" / "state" / "migrations"
    / "216_companion_growth_substrate.sql"
)

_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    applied_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
);
"""


async def _mkstore() -> tuple[GrowthStore, aiosqlite.Connection]:
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA_VERSION_TABLE)
    sql = _MIGRATION_PATH.read_text(encoding="utf-8")
    await conn.executescript(sql)
    await conn.commit()
    return GrowthStore(conn), conn


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _MockMemory:
    """Stub MemoryStore.recall return — simple list of dicts with the
    fields the Recall handler probes."""

    def __init__(self, hits: list[dict]):
        self._hits = hits

    async def recall(self, query: str, user_id: str = "", limit: int = 10):
        return list(self._hits)


# ---------------------------------------------------------------------------
# Migration smoke
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migration_applies_to_memory_db():
    store, conn = await _mkstore()
    # All four tables present.
    cursor = await conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name LIKE 'companion_%' ORDER BY name",
    )
    rows = await cursor.fetchall()
    names = [r[0] for r in rows]
    assert names == [
        "companion_economy",
        "companion_economy_tx",
        "companion_growth_backlog",
        "companion_growth_log",
    ]


# ---------------------------------------------------------------------------
# Economy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_economy_initial_snapshot_seeds_account():
    store, _ = await _mkstore()
    economy = Economy(store, user_id="u1")
    account = await economy.snapshot()
    assert account.mana == 100
    assert account.mana_cap == 100
    assert account.berries == 0
    assert account.berries_lifetime == 0


@pytest.mark.asyncio
async def test_economy_mana_regen_is_capped():
    store, _ = await _mkstore()
    economy = Economy(store, user_id="u1")
    # Burn 50 mana, then rewind the last_mana_tick to simulate elapsed time.
    debit = await economy.debit_mana(50, reason="setup")
    assert debit.ok
    assert debit.mana_after == 50

    # Rewind last_mana_tick by 24h so a fresh snapshot would regen well
    # past the cap. Verify it clamps at mana_cap.
    account = await store.get_or_create_economy(user_id="u1")
    account.last_mana_tick = int(time.time()) - 24 * 3600
    await store.save_economy(account)

    snap = await economy.snapshot()
    assert snap.mana == 100  # clamped to cap
    assert snap.mana > debit.mana_after  # regen actually happened


@pytest.mark.asyncio
async def test_economy_debit_rejected_when_insufficient():
    store, _ = await _mkstore()
    economy = Economy(store, user_id="u1")
    # Drain to 5.
    await economy.debit_mana(95, reason="drain")
    result = await economy.debit_mana(10, reason="overdraft")
    assert not result.ok
    assert result.reason == "insufficient_mana"
    # Balance unchanged.
    snap = await economy.snapshot()
    assert snap.mana <= 5 + 0.1  # may have ticked slightly during the call


@pytest.mark.asyncio
async def test_economy_earn_updates_balance_and_lifetime():
    store, _ = await _mkstore()
    economy = Economy(store, user_id="u1")
    r = await economy.earn_berries(25, signal_kind="explicit", reason="test")
    assert r.ok
    assert r.berries_after == 25
    snap = await economy.snapshot()
    assert snap.berries == 25
    assert snap.berries_lifetime == 25


@pytest.mark.asyncio
async def test_economy_veto_preserves_lifetime():
    store, _ = await _mkstore()
    economy = Economy(store, user_id="u1")
    await economy.earn_berries(50, reason="setup")
    r = await economy.veto(30, reason="user_disagrees")
    assert r.ok
    snap = await economy.snapshot()
    assert snap.berries == 20
    assert snap.berries_lifetime == 50  # not decremented


@pytest.mark.asyncio
async def test_economy_spend_rejected_on_insufficient():
    store, _ = await _mkstore()
    economy = Economy(store, user_id="u1")
    r = await economy.spend_berries(10, reason="big_swing")
    assert not r.ok
    assert r.reason == "insufficient_berries"


@pytest.mark.asyncio
async def test_economy_sponsor_is_user_action():
    store, _ = await _mkstore()
    economy = Economy(store, user_id="u1")
    await economy.sponsor(100)
    tx_rows = await store.list_tx(user_id="u1", limit=10)
    # Sponsor is signalled by tx_type=berry_earn + signal_kind=user_action,
    # not by the reason string (which can be caller-overridden).
    sponsor_tx = next(
        (t for t in tx_rows if t.tx_type == "berry_earn"
         and t.signal_kind == "user_action"),
        None,
    )
    assert sponsor_tx is not None
    assert sponsor_tx.amount == 100
    assert sponsor_tx.reason == "user_sponsor"


# ---------------------------------------------------------------------------
# GrowthStore backlog round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backlog_roundtrip():
    store, _ = await _mkstore()
    item = await store.add_backlog_item(
        user_id="u1", item_type="recall_connect",
        target_ref="quantum entanglement",
        rationale="user typed about it twice this week",
        priority=0.7,
    )
    assert item.id

    got = await store.get_backlog_item(item.id, user_id="u1")
    assert got is not None
    assert got.item_type == "recall_connect"
    assert got.target_ref == "quantum entanglement"
    assert got.priority == 0.7
    assert got.state == "pending"
    assert got.success_count == 0


@pytest.mark.asyncio
async def test_backlog_user_isolation():
    store, _ = await _mkstore()
    item = await store.add_backlog_item(
        user_id="u1", item_type="recall_connect", target_ref="topic",
    )
    # Different user can't read u1's row.
    other = await store.get_backlog_item(item.id, user_id="u2")
    assert other is None


@pytest.mark.asyncio
async def test_backlog_attempt_counters():
    store, _ = await _mkstore()
    item = await store.add_backlog_item(
        user_id="u1", item_type="recall_connect", target_ref="topic",
    )
    await store.update_backlog_attempt(item.id, user_id="u1", success=True)
    await store.update_backlog_attempt(item.id, user_id="u1", success=False)
    refreshed = await store.get_backlog_item(item.id, user_id="u1")
    assert refreshed.success_count == 1
    assert refreshed.fail_count == 1
    assert refreshed.last_attempted_at is not None


# ---------------------------------------------------------------------------
# Recall action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_action_registered():
    assert "recall_connect" in ACTIONS
    handler = ACTIONS["recall_connect"]
    assert handler.tier == 0
    assert handler.mana_cost == 2.0


@pytest.mark.asyncio
async def test_recall_action_returns_surface_event():
    from augmentum.companion.growth.actions import ActionContext

    handler = ACTIONS["recall_connect"]
    memory = _MockMemory([
        # Trivially recent — should be skipped.
        {"id": "m1", "text": "just now snippet",
         "created_at": int(time.time())},
        # Eligible — older than threshold.
        {"id": "m2", "text": "a much older insight",
         "created_at": int(time.time()) - 30 * 86400, "scope": "ARCHIVE"},
    ])
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log1",
        target_ref="some topic", memory_store=memory,
    )
    result = await handler.run(ctx)
    assert result.ok
    assert result.surface_event is not None
    assert result.surface_event["topic"] == "growth.recall.surfaced"
    assert result.surface_event["payload"]["memory_id"] == "m2"
    assert result.surface_event["payload"]["snippet"] == "a much older insight"


@pytest.mark.asyncio
async def test_recall_action_without_memory_store_fails_gracefully():
    from augmentum.companion.growth.actions import ActionContext

    handler = ACTIONS["recall_connect"]
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log1",
        target_ref="topic", memory_store=None,
    )
    result = await handler.run(ctx)
    assert not result.ok
    assert "memory_store" in result.error


@pytest.mark.asyncio
async def test_recall_action_empty_target_returns_error():
    from augmentum.companion.growth.actions import ActionContext

    handler = ACTIONS["recall_connect"]
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log1",
        target_ref="", memory_store=_MockMemory([]),
    )
    result = await handler.run(ctx)
    assert not result.ok
    assert "empty target_ref" in result.error


# ---------------------------------------------------------------------------
# CompanionGrowthSession lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_runs_recall_end_to_end():
    store, _ = await _mkstore()
    economy = Economy(store, user_id="u1")
    memory = _MockMemory([
        {"id": "m1", "text": "older insight",
         "created_at": int(time.time()) - 7 * 86400},
    ])

    session = CompanionGrowthSession(
        store=store, economy=economy, user_id="u1",
        memory_store=memory,
        config=SessionConfig(max_steps=1),
    )
    final = await session.run(
        ad_hoc_request=ActionRequest(
            action_type="recall_connect",
            target_ref="some topic",
            rationale="manual test",
        ),
    )

    assert final.outcome == "completed"
    assert final.tier == 0
    assert final.mana_spent == 2.0  # one Recall step at handler.mana_cost
    assert final.ended_at is not None
    # Act log captured the surface event.
    assert len(final.act_log) == 1
    step = final.act_log[0]
    assert step["ok"] is True
    assert step["surface_event"]["topic"] == "growth.recall.surfaced"
    # Mana balance reflects the debit.
    account = await economy.snapshot()
    assert account.mana <= 100 - 2.0 + 0.1  # tiny regen tolerance


@pytest.mark.asyncio
async def test_session_aborts_when_mana_insufficient():
    store, _ = await _mkstore()
    economy = Economy(store, user_id="u1")
    # Burn nearly all mana before the session runs.
    await economy.debit_mana(99, reason="drain")

    session = CompanionGrowthSession(
        store=store, economy=economy, user_id="u1",
        memory_store=_MockMemory([]),
        config=SessionConfig(max_steps=1),
    )
    final = await session.run(
        ad_hoc_request=ActionRequest(
            action_type="recall_connect", target_ref="topic",
        ),
    )
    assert final.outcome == "aborted"
    # Insufficient-mana debit failure appears in the act log.
    assert any(
        not step.get("ok", True) and step.get("error") == "insufficient_mana"
        for step in final.act_log
    )


@pytest.mark.asyncio
async def test_session_links_to_backlog_when_provided():
    store, _ = await _mkstore()
    economy = Economy(store, user_id="u1")
    backlog = await store.add_backlog_item(
        user_id="u1", item_type="recall_connect", target_ref="topic",
    )

    session = CompanionGrowthSession(
        store=store, economy=economy, user_id="u1",
        backlog=backlog,
        memory_store=_MockMemory([
            {"id": "m1", "text": "older",
             "created_at": int(time.time()) - 86400},
        ]),
        config=SessionConfig(max_steps=1),
    )
    final = await session.run()
    assert final.backlog_id == backlog.id

    # Backlog success counter advanced.
    refreshed = await store.get_backlog_item(backlog.id, user_id="u1")
    assert refreshed.success_count == 1
    assert refreshed.last_attempted_at is not None


@pytest.mark.asyncio
async def test_session_tx_log_links_to_growth_log_id():
    store, _ = await _mkstore()
    economy = Economy(store, user_id="u1")
    memory = _MockMemory([
        {"id": "m1", "text": "older",
         "created_at": int(time.time()) - 86400},
    ])
    session = CompanionGrowthSession(
        store=store, economy=economy, user_id="u1",
        memory_store=memory,
        config=SessionConfig(max_steps=1),
    )
    final = await session.run(
        ad_hoc_request=ActionRequest(
            action_type="recall_connect", target_ref="topic",
        ),
    )
    # Mana debit tx points back to the session.
    session_tx = await store.list_tx(
        user_id="u1", growth_log_id=final.id, limit=10,
    )
    debit_tx = [t for t in session_tx if t.tx_type == "mana_debit"]
    assert len(debit_tx) == 1
    assert debit_tx[0].amount == 2.0
    assert debit_tx[0].growth_log_id == final.id


# ---------------------------------------------------------------------------
# Reward signals
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reward_explicit_thumbs_up_banks_berries():
    store, _ = await _mkstore()
    economy = Economy(store, user_id="u1")
    outcome = await apply_explicit(
        economy, signal="thumbs_up", growth_log_id="log1",
    )
    assert outcome.ok
    assert outcome.delta == 20.0
    assert outcome.berries_after == 20.0


@pytest.mark.asyncio
async def test_reward_explicit_thumbs_down_vetoes():
    store, _ = await _mkstore()
    economy = Economy(store, user_id="u1")
    # Seed a balance to subtract from.
    await economy.earn_berries(50, reason="setup")
    outcome = await apply_explicit(
        economy, signal="thumbs_down", growth_log_id="log1",
    )
    assert outcome.ok
    assert outcome.berries_after == 30
    # Lifetime preserved.
    account = await economy.snapshot()
    assert account.berries_lifetime == 50


@pytest.mark.asyncio
async def test_reward_unknown_signal_returns_ok_false():
    store, _ = await _mkstore()
    economy = Economy(store, user_id="u1")
    outcome = await apply_explicit(
        economy, signal="not_a_real_signal", growth_log_id="log1",
    )
    assert not outcome.ok
    assert "unknown_explicit_signal" in outcome.reason


@pytest.mark.asyncio
async def test_reward_implicit_engaged_long_earns_more():
    store, _ = await _mkstore()
    economy = Economy(store, user_id="u1")
    short = await apply_implicit(
        economy, signal="engaged_short", growth_log_id="log1",
    )
    longer = await apply_implicit(
        economy, signal="engaged_long", growth_log_id="log1",
    )
    assert longer.delta > short.delta


@pytest.mark.asyncio
async def test_reward_restraint_credit_default_amount():
    store, _ = await _mkstore()
    economy = Economy(store, user_id="u1")
    outcome = await apply_restraint_credit(
        economy, growth_log_id="log1", evidence_ref="held_notification",
    )
    assert outcome.ok
    assert outcome.delta == 5.0
    # Tx records signal_kind=restraint for analytics.
    tx_rows = await store.list_tx(user_id="u1", limit=5)
    restraint_tx = [t for t in tx_rows if t.signal_kind == "restraint"]
    assert len(restraint_tx) == 1


# ---------------------------------------------------------------------------
# Unified Becca panel — new backend surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backlog_list_filters_by_state():
    """list_backlog defaults to pending; passing state=None returns all."""
    store, _ = await _mkstore()
    a = await store.add_backlog_item(
        user_id="u1", item_type="recall_connect", target_ref="topic_a",
    )
    b = await store.add_backlog_item(
        user_id="u1", item_type="recall_connect", target_ref="topic_b",
    )
    await store.set_backlog_state(b.id, user_id="u1", state="done")

    pending = await store.list_backlog(user_id="u1", state="pending")
    assert [i.id for i in pending] == [a.id]

    all_states = await store.list_backlog(user_id="u1", state=None)
    ids = {i.id for i in all_states}
    assert ids == {a.id, b.id}


@pytest.mark.asyncio
async def test_backlog_list_priority_then_age_ordering():
    """Higher priority first; same priority falls back to created_at ASC."""
    store, _ = await _mkstore()
    low = await store.add_backlog_item(
        user_id="u1", item_type="recall_connect",
        target_ref="low", priority=0.3,
    )
    high = await store.add_backlog_item(
        user_id="u1", item_type="recall_connect",
        target_ref="high", priority=1.0,
    )
    items = await store.list_backlog(user_id="u1")
    assert [i.id for i in items] == [high.id, low.id]


@pytest.mark.asyncio
async def test_sponsor_grant_appears_in_tx_ledger_recent_first():
    """Tx ledger orders by ts DESC — sponsor grant should land at the top."""
    store, _ = await _mkstore()
    economy = Economy(store, user_id="u1")
    await economy.earn_berries(10, reason="setup")
    await economy.sponsor(50, evidence_ref="some_goal_id")
    rows = await store.list_tx(user_id="u1", limit=5)
    # DESC order — sponsor (the most recent berry_earn) at index 0.
    assert rows[0].tx_type == "berry_earn"
    assert rows[0].signal_kind == "user_action"
    assert rows[0].amount == 50
    assert rows[0].evidence_ref == "some_goal_id"


@pytest.mark.asyncio
async def test_feedback_dismissed_weight_registered():
    """The `dismissed` kind must have a weight registered or the bias
    function silently treats it as 0 (effectively unweighted noise)."""
    from augmentum.companion_runtime.feedback import KIND_WEIGHTS

    assert "dismissed" in KIND_WEIGHTS
    # Softer than mute (-1.5), and net-negative (this is the whole point).
    assert -1.5 < KIND_WEIGHTS["dismissed"] < 0


@pytest.mark.asyncio
async def test_tx_list_user_isolation():
    """list_tx is per-user — u2 must not see u1's grants."""
    store, _ = await _mkstore()
    e1 = Economy(store, user_id="u1")
    e2 = Economy(store, user_id="u2")
    await e1.sponsor(100)
    await e2.sponsor(50)
    u1_rows = await store.list_tx(user_id="u1", limit=10)
    u2_rows = await store.list_tx(user_id="u2", limit=10)
    assert all(r.user_id == "u1" for r in u1_rows)
    assert all(r.user_id == "u2" for r in u2_rows)


@pytest.mark.asyncio
async def test_actions_catalog_shape():
    """The /actions endpoint reflects ACTIONS; UI uses it to enable Run.
    Each entry must carry action_type + tier + mana_cost so the panel
    can render the select + decide whether to enable the Run button."""
    for action_type, handler in ACTIONS.items():
        assert action_type  # non-empty key
        assert isinstance(getattr(handler, "tier", None), int)
        assert isinstance(getattr(handler, "mana_cost", None), (int, float))
    # Phase 1 anchor must be registered.
    assert "recall_connect" in ACTIONS


@pytest.mark.asyncio
async def test_topic_feedback_dismissed_count_field():
    """TopicFeedback dataclass gained dismissed_count — confirm shape so
    the observatory route's snapshot["feedback"] block doesn't drift."""
    from augmentum.companion_runtime.feedback import TopicFeedback
    tf = TopicFeedback(
        multiplier=1.0,
        surfaced_count=1, acknowledged_count=2,
        dismissed_count=3, muted_count=4,
    )
    assert tf.dismissed_count == 3
    # Order matters for the route's serialization path; both old fields
    # remain present and the new one slots in between acknowledged + muted.
    assert tf.surfaced_count == 1
    assert tf.muted_count == 4


# ---------------------------------------------------------------------------
# Phase 1.5 — narrate_growth (K), discovery_surface (B), care_consolidate (G)
# ---------------------------------------------------------------------------


# ── narrate_growth ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_narrate_action_registered():
    assert "narrate_growth" in ACTIONS
    handler = ACTIONS["narrate_growth"]
    assert handler.tier == 1
    assert handler.mana_cost == 1.5


@pytest.mark.asyncio
async def test_narrate_action_without_growth_store_fails_gracefully():
    from augmentum.companion.growth.actions import ActionContext

    handler = ACTIONS["narrate_growth"]
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log_self",
        growth_store=None,
    )
    result = await handler.run(ctx)
    assert not result.ok
    assert "growth_store" in result.error


@pytest.mark.asyncio
async def test_narrate_action_too_few_sessions_returns_soft_error():
    """With < 3 prior sessions the narration would be noise — skip."""
    from augmentum.companion.growth.actions import ActionContext

    store, _ = await _mkstore()
    handler = ACTIONS["narrate_growth"]
    # Seed only 2 prior sessions — below the floor.
    for i in range(2):
        await store.start_session(
            user_id="u1", agent_id="becca", backlog_id=None,
            plan={"action_type": "recall_connect"}, snapshot_ref="",
        )
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log_self",
        growth_store=store,
    )
    result = await handler.run(ctx)
    assert not result.ok
    assert "prior sessions" in result.error


@pytest.mark.asyncio
async def test_narrate_action_aggregates_recent_sessions():
    """Golden path: 5 prior sessions across 2 action types → digest
    reports both with correct counts."""
    from augmentum.companion.growth.actions import ActionContext

    store, _ = await _mkstore()
    handler = ACTIONS["narrate_growth"]
    # Seed 3 recall sessions + 2 discovery sessions, all completed.
    for action_type in ("recall_connect",) * 3 + ("discovery_surface",) * 2:
        entry = await store.start_session(
            user_id="u1", agent_id="becca", backlog_id=None,
            plan={"action_type": action_type}, snapshot_ref="",
        )
        await store.finalize_session(
            entry.id, user_id="u1", agent_id="becca",
            outcome="completed", tier=0, approval_state="n/a",
            ledger_delta={}, mana_spent=1.0, berries_spent=0,
            berries_earned=5,
        )

    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log_self",
        growth_store=store, rationale="weekly digest",
    )
    result = await handler.run(ctx)
    assert result.ok
    payload = result.surface_event["payload"]
    assert payload["window_count"] == 5
    assert payload["action_type_counts"]["recall_connect"] == 3
    assert payload["action_type_counts"]["discovery_surface"] == 2
    assert payload["outcome_counts"]["completed"] == 5
    assert payload["mana_spent_total"] == 5.0
    assert payload["berries_earned_total"] == 25.0
    assert result.ledger_delta == {"narration_surfaced": 1}


@pytest.mark.asyncio
async def test_narrate_action_excludes_own_in_progress_session():
    """The action runs INSIDE a session whose log row is open — that
    row must not show up in the digest as 'completed work she's done.'"""
    from augmentum.companion.growth.actions import ActionContext

    store, _ = await _mkstore()
    handler = ACTIONS["narrate_growth"]
    # Three completed prior sessions.
    for _ in range(3):
        entry = await store.start_session(
            user_id="u1", agent_id="becca", backlog_id=None,
            plan={"action_type": "recall_connect"}, snapshot_ref="",
        )
        await store.finalize_session(
            entry.id, user_id="u1", agent_id="becca",
            outcome="completed", tier=0, approval_state="n/a",
            ledger_delta={}, mana_spent=1.0, berries_spent=0,
            berries_earned=0,
        )
    # The in-progress one — should be excluded from the count.
    self_entry = await store.start_session(
        user_id="u1", agent_id="becca", backlog_id=None,
        plan={"action_type": "narrate_growth"}, snapshot_ref="",
    )

    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id=self_entry.id,
        growth_store=store,
    )
    result = await handler.run(ctx)
    assert result.ok
    assert result.surface_event["payload"]["window_count"] == 3
    # The in-progress narrate row didn't sneak into the action_type counts.
    assert "narrate_growth" not in result.surface_event["payload"]["action_type_counts"]


# ── discovery_surface ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_discovery_action_registered():
    assert "discovery_surface" in ACTIONS
    handler = ACTIONS["discovery_surface"]
    assert handler.tier == 1
    assert handler.mana_cost == 3.0


@pytest.mark.asyncio
async def test_discovery_action_without_memory_store_fails_gracefully():
    from augmentum.companion.growth.actions import ActionContext

    handler = ACTIONS["discovery_surface"]
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log",
        target_ref="topic", memory_store=None,
    )
    result = await handler.run(ctx)
    assert not result.ok
    assert "memory_store" in result.error


@pytest.mark.asyncio
async def test_discovery_action_empty_target_returns_error():
    from augmentum.companion.growth.actions import ActionContext

    handler = ACTIONS["discovery_surface"]
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log",
        target_ref="", memory_store=_MockMemory([]),
    )
    result = await handler.run(ctx)
    assert not result.ok
    assert "empty target_ref" in result.error


@pytest.mark.asyncio
async def test_discovery_action_picks_oldest_memory_above_floor():
    """Three memories — one recent, two old. The OLDEST of the two
    aged ones wins ("forgotten" beats "merely old")."""
    from augmentum.companion.growth.actions import ActionContext

    handler = ACTIONS["discovery_surface"]
    now = int(time.time())
    memory = _MockMemory([
        {"id": "m_recent", "text": "wrote this yesterday",
         "created_at": now - 1 * 86400},        # < floor, skipped
        {"id": "m_old", "text": "moderate aged thought",
         "created_at": now - 45 * 86400},       # > floor, candidate
        {"id": "m_ancient", "text": "very old buried gem",
         "created_at": now - 365 * 86400},      # > floor, oldest → wins
    ])
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log",
        target_ref="some topic", memory_store=memory,
    )
    result = await handler.run(ctx)
    assert result.ok
    assert result.surface_event["topic"] == "growth.discovery.surfaced"
    assert result.surface_event["payload"]["memory_id"] == "m_ancient"


@pytest.mark.asyncio
async def test_discovery_action_no_aged_candidate_returns_error():
    """All candidates younger than the age floor → no discovery."""
    from augmentum.companion.growth.actions import ActionContext

    handler = ACTIONS["discovery_surface"]
    now = int(time.time())
    memory = _MockMemory([
        {"id": "m1", "text": "fresh", "created_at": now - 3600},
        {"id": "m2", "text": "also fresh", "created_at": now - 7200},
    ])
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log",
        target_ref="topic", memory_store=memory,
    )
    result = await handler.run(ctx)
    assert not result.ok
    assert "age floor" in result.error


@pytest.mark.asyncio
async def test_discovery_action_respects_extras_min_age_override():
    """``extras['min_age_days']`` lowers the floor — a 7-day-old memory
    that would normally be skipped becomes eligible."""
    from augmentum.companion.growth.actions import ActionContext

    handler = ACTIONS["discovery_surface"]
    now = int(time.time())
    memory = _MockMemory([
        {"id": "m_week", "text": "from last week",
         "created_at": now - 7 * 86400},
    ])
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log",
        target_ref="topic", memory_store=memory,
        extras={"min_age_days": 3},  # 3-day floor; 7-day memory clears.
    )
    result = await handler.run(ctx)
    assert result.ok
    assert result.surface_event["payload"]["memory_id"] == "m_week"


# ── care_consolidate ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_care_action_registered():
    assert "care_consolidate" in ACTIONS
    handler = ACTIONS["care_consolidate"]
    assert handler.tier == 0
    assert handler.mana_cost == 4.0


@pytest.mark.asyncio
async def test_care_action_without_memory_store_fails_gracefully():
    from augmentum.companion.growth.actions import ActionContext

    handler = ACTIONS["care_consolidate"]
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log",
        memory_store=None,
    )
    result = await handler.run(ctx)
    assert not result.ok
    assert "memory_store" in result.error


@pytest.mark.asyncio
async def test_care_action_finds_cluster_of_near_duplicates():
    """Three memories share most of their prefix words — they cluster."""
    from augmentum.companion.growth.actions import ActionContext

    handler = ACTIONS["care_consolidate"]
    memory = _MockMemory([
        {"id": "m1", "text": "the cat sat quietly on the windowsill watching birds outside"},
        {"id": "m2", "text": "the cat sat watching birds outside on the windowsill quietly"},
        {"id": "m3", "text": "the cat watched birds outside sitting on the windowsill quietly"},
        # Distinct one — should NOT cluster with the cat trio.
        {"id": "m4", "text": "a complete unrelated thought about coffee and morning routines"},
    ])
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log",
        target_ref="cats", memory_store=memory,
    )
    result = await handler.run(ctx)
    assert result.ok
    payload = result.surface_event["payload"]
    assert payload["scanned_count"] == 4
    assert payload["cluster_count"] >= 1
    # The cluster has the three cat memories (in any order).
    largest = max(payload["clusters"], key=lambda c: c["size"])
    assert largest["size"] == 3
    assert set(largest["member_ids"]) == {"m1", "m2", "m3"}


@pytest.mark.asyncio
async def test_care_action_no_clusters_returns_error():
    """When every memory is distinct, no consolidation candidate exists."""
    from augmentum.companion.growth.actions import ActionContext

    handler = ACTIONS["care_consolidate"]
    memory = _MockMemory([
        {"id": "m1", "text": "completely about apples in autumn orchards"},
        {"id": "m2", "text": "totally distinct musing on quantum mechanics"},
        {"id": "m3", "text": "different topic about rainy days in spring"},
    ])
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log",
        memory_store=memory,
    )
    result = await handler.run(ctx)
    assert not result.ok
    assert "no consolidation candidates" in result.error


# ── End-to-end session integration ────────────────────────────────────


@pytest.mark.asyncio
async def test_session_runs_narrate_end_to_end():
    """CompanionGrowthSession.run dispatches the new action via the
    catalog and the act log captures the surface event."""
    store, _ = await _mkstore()
    economy = Economy(store, user_id="u1")
    # Seed 4 prior sessions so the narration clears the floor.
    for _ in range(4):
        entry = await store.start_session(
            user_id="u1", agent_id="becca", backlog_id=None,
            plan={"action_type": "recall_connect"}, snapshot_ref="",
        )
        await store.finalize_session(
            entry.id, user_id="u1", agent_id="becca",
            outcome="completed", tier=0, approval_state="n/a",
            ledger_delta={}, mana_spent=2.0,
            berries_spent=0, berries_earned=10,
        )

    session = CompanionGrowthSession(
        store=store, economy=economy, user_id="u1",
        config=SessionConfig(max_steps=1),
    )
    final = await session.run(
        ad_hoc_request=ActionRequest(
            action_type="narrate_growth",
            target_ref="",
            rationale="self-reflection",
        ),
    )
    assert final.outcome == "completed"
    assert final.mana_spent == 1.5
    assert len(final.act_log) == 1
    step = final.act_log[0]
    assert step["ok"] is True
    assert step["surface_event"]["topic"] == "growth.narrate.surfaced"
    assert step["surface_event"]["payload"]["window_count"] == 4


# ---------------------------------------------------------------------------
# Ship B — proactive_offer (D), companionship (H)
# ---------------------------------------------------------------------------


# ── proactive_offer ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_offer_action_registered():
    assert "proactive_offer" in ACTIONS
    handler = ACTIONS["proactive_offer"]
    assert handler.tier == 0
    assert handler.mana_cost == 2.0


@pytest.mark.asyncio
async def test_offer_action_empty_kind_returns_error():
    from augmentum.companion.growth.actions import ActionContext

    handler = ACTIONS["proactive_offer"]
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log",
        target_ref="",  # empty kind
        extras={"offer_label": "want music?"},
    )
    result = await handler.run(ctx)
    assert not result.ok
    assert "target_ref" in result.error


@pytest.mark.asyncio
async def test_offer_action_missing_label_returns_error():
    from augmentum.companion.growth.actions import ActionContext

    handler = ACTIONS["proactive_offer"]
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log",
        target_ref="music",  # kind ok, no label
    )
    result = await handler.run(ctx)
    assert not result.ok
    assert "offer_label" in result.error


@pytest.mark.asyncio
async def test_offer_action_emits_surface_event():
    """Golden path — kind + label produce a properly-shaped offer."""
    from augmentum.companion.growth.actions import ActionContext

    handler = ACTIONS["proactive_offer"]
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log_abc",
        target_ref="music",
        extras={
            "offer_label": "Want me to put on the focus playlist?",
            "offer_payload": {"playlist_id": "focus_v3"},
            "dismiss_after_seconds": 120,
        },
        rationale="afternoon lull",
    )
    result = await handler.run(ctx)
    assert result.ok
    se = result.surface_event
    assert se["topic"] == "growth.offer.surfaced"
    p = se["payload"]
    assert p["offer_kind"] == "music"
    assert p["offer_label"] == "Want me to put on the focus playlist?"
    assert p["offer_payload"] == {"playlist_id": "focus_v3"}
    assert p["dismiss_after_seconds"] == 120
    assert p["growth_log_id"] == "log_abc"
    assert p["offer_id"].startswith("offer_")
    # Ledger carries both the generic counter and the per-kind tally,
    # so the verifier can match accept/dismiss signals per kind.
    assert result.ledger_delta == {
        "offer_surfaced": 1, "offer_surfaced_music": 1,
    }


@pytest.mark.asyncio
async def test_offer_action_clamps_extreme_dismiss_window():
    """Caller-supplied dismiss windows get clamped to [10s, 1h]."""
    from augmentum.companion.growth.actions import ActionContext

    handler = ACTIONS["proactive_offer"]
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log",
        target_ref="break",
        extras={
            "offer_label": "5-min break?",
            "dismiss_after_seconds": 999_999,  # all-day offer would be junk
        },
    )
    result = await handler.run(ctx)
    assert result.ok
    assert result.surface_event["payload"]["dismiss_after_seconds"] == 60 * 60


@pytest.mark.asyncio
async def test_offer_action_rejects_oversized_payload():
    from augmentum.companion.growth.actions import ActionContext

    handler = ACTIONS["proactive_offer"]
    big_payload = {"blob": "x" * (5 * 1024)}  # > 4KB cap
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log",
        target_ref="document",
        extras={
            "offer_label": "Read this doc?",
            "offer_payload": big_payload,
        },
    )
    result = await handler.run(ctx)
    assert not result.ok
    assert "payload too large" in result.error


# ── companionship ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_companionship_action_registered():
    assert "companionship" in ACTIONS
    handler = ACTIONS["companionship"]
    assert handler.tier == 0
    assert handler.mana_cost == 1.5


@pytest.mark.asyncio
async def test_companionship_action_without_growth_store_fails():
    """Saturation guard requires growth_store; refuse without it."""
    from augmentum.companion.growth.actions import ActionContext

    handler = ACTIONS["companionship"]
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log",
        target_ref="good_morning",
        extras={"message": "morning"},
        growth_store=None,
    )
    result = await handler.run(ctx)
    assert not result.ok
    assert "growth_store" in result.error


@pytest.mark.asyncio
async def test_companionship_action_empty_kind_returns_error():
    from augmentum.companion.growth.actions import ActionContext

    store, _ = await _mkstore()
    handler = ACTIONS["companionship"]
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log",
        target_ref="",
        extras={"message": "hi"},
        growth_store=store,
    )
    result = await handler.run(ctx)
    assert not result.ok
    assert "moment_kind" in result.error


@pytest.mark.asyncio
async def test_companionship_action_missing_message_returns_error():
    from augmentum.companion.growth.actions import ActionContext

    store, _ = await _mkstore()
    handler = ACTIONS["companionship"]
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log",
        target_ref="good_morning",
        growth_store=store,
    )
    result = await handler.run(ctx)
    assert not result.ok
    assert "message" in result.error


@pytest.mark.asyncio
async def test_companionship_action_fires_when_clean():
    """Empty history → handler emits a clean surface event."""
    from augmentum.companion.growth.actions import ActionContext

    store, _ = await _mkstore()
    handler = ACTIONS["companionship"]
    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log_self",
        target_ref="good_morning",
        extras={"message": "morning — coffee's started"},
        growth_store=store,
        rationale="wake_signal",
    )
    result = await handler.run(ctx)
    assert result.ok
    p = result.surface_event["payload"]
    assert result.surface_event["topic"] == "growth.companionship.surfaced"
    assert p["moment_kind"] == "good_morning"
    assert p["message"] == "morning — coffee's started"
    assert result.ledger_delta == {
        "companionship_surfaced": 1, "companionship_good_morning": 1,
    }


@pytest.mark.asyncio
async def test_companionship_action_blocks_consecutive_picks():
    """If the most-recent prior session was also companionship, refuse."""
    from augmentum.companion.growth.actions import ActionContext

    store, _ = await _mkstore()
    handler = ACTIONS["companionship"]
    # Seed one prior completed companionship session — adjacent.
    entry = await store.start_session(
        user_id="u1", agent_id="becca", backlog_id=None,
        plan={"action_type": "companionship"}, snapshot_ref="",
    )
    await store.finalize_session(
        entry.id, user_id="u1", agent_id="becca",
        outcome="completed", tier=0, approval_state="n/a",
        ledger_delta={}, mana_spent=1.5,
        berries_spent=0, berries_earned=0,
    )

    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log_self",
        target_ref="checkin",
        extras={"message": "how's it going"},
        growth_store=store,
    )
    result = await handler.run(ctx)
    assert not result.ok
    assert "consecutive" in result.error


@pytest.mark.asyncio
async def test_companionship_action_blocks_at_24h_cap():
    """4th companionship attempt in 24h hits the saturation cap.

    Seeds 3 within-24h companionship sessions + 1 non-companionship
    most-recent (so the consecutive rule doesn't trip first), then
    expects the 24h-cap to refuse the 4th.
    """
    from augmentum.companion.growth.actions import ActionContext

    store, _ = await _mkstore()
    handler = ACTIONS["companionship"]

    # 3 companionship sessions in the last 24h.
    for _ in range(3):
        entry = await store.start_session(
            user_id="u1", agent_id="becca", backlog_id=None,
            plan={"action_type": "companionship"}, snapshot_ref="",
        )
        await store.finalize_session(
            entry.id, user_id="u1", agent_id="becca",
            outcome="completed", tier=0, approval_state="n/a",
            ledger_delta={}, mana_spent=1.5,
            berries_spent=0, berries_earned=0,
        )

    # Then a recall session as the most-recent — breaks the
    # consecutive-rule trip so we land on the 24h cap. Sleep just past
    # the store's second-resolution timestamp so list_sessions(ORDER BY
    # started_at DESC) puts the recall first deterministically.
    time.sleep(1.05)
    last = await store.start_session(
        user_id="u1", agent_id="becca", backlog_id=None,
        plan={"action_type": "recall_connect"}, snapshot_ref="",
    )
    await store.finalize_session(
        last.id, user_id="u1", agent_id="becca",
        outcome="completed", tier=0, approval_state="n/a",
        ledger_delta={}, mana_spent=2.0,
        berries_spent=0, berries_earned=0,
    )

    ctx = ActionContext(
        user_id="u1", agent_id="becca", growth_log_id="log_self",
        target_ref="shared_moment",
        extras={"message": "look at this"},
        growth_store=store,
    )
    result = await handler.run(ctx)
    assert not result.ok
    assert "saturation" in result.error
    assert "last 24h" in result.error
