"""CRUD layer for the companion-growth-loop substrate (migration 216).

Four tables, one store class. The store is pure persistence — business
logic (mana regen, session lifecycle, action dispatch) lives one layer
up in :class:`~augmentum.companion.growth.economy.Economy` and
:class:`~augmentum.companion.growth.session.CompanionGrowthSession`.

All writes are user-scoped + agent-scoped per the multi-tenant rules in
CLAUDE.md. ``user_id`` is required on every CRUD call; ``agent_id``
defaults to ``'becca'`` because that's the only companion today.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


DEFAULT_AGENT_ID = "becca"


# ---------------------------------------------------------------------------
# Row dataclasses — mirror the SQL schema 1:1.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BacklogItem:
    id: str = ""
    user_id: str = ""
    agent_id: str = DEFAULT_AGENT_ID
    item_type: str = ""
    target_ref: str = ""
    rationale: str = ""
    priority: float = 0.5
    source_signal: str = ""
    expected_berry_yield: float = 0.0
    expected_mana_cost: float = 0.0
    expected_berry_cost: float = 0.0
    success_count: int = 0
    fail_count: int = 0
    last_attempted_at: int | None = None
    last_consult_inconclusive_at: int | None = None
    state: str = "pending"
    created_at: int = 0


@dataclass(slots=True)
class GrowthLogEntry:
    id: str = ""
    user_id: str = ""
    agent_id: str = DEFAULT_AGENT_ID
    backlog_id: str | None = None
    started_at: int = 0
    ended_at: int | None = None
    plan: dict[str, Any] = field(default_factory=dict)
    act_log: list[dict[str, Any]] = field(default_factory=list)
    consult_records: list[dict[str, Any]] = field(default_factory=list)
    ledger_delta: dict[str, Any] = field(default_factory=dict)
    outcome: str = "in_progress"
    tier: int = 0
    approval_state: str = "n/a"
    mana_spent: float = 0.0
    berries_spent: float = 0.0
    berries_earned: float = 0.0
    snapshot_ref: str = ""


@dataclass(slots=True)
class EconomyAccount:
    user_id: str = ""
    agent_id: str = DEFAULT_AGENT_ID
    mana: float = 100.0
    mana_cap: float = 100.0
    mana_regen_per_hour: float = 10.0
    berries: float = 0.0
    berries_lifetime: float = 0.0
    last_mana_tick: int = 0


@dataclass(slots=True)
class EconomyTx:
    id: int = 0
    user_id: str = ""
    agent_id: str = DEFAULT_AGENT_ID
    growth_log_id: str | None = None
    tx_type: str = ""
    amount: float = 0.0
    reason: str = ""
    signal_kind: str = "system"
    evidence_ref: str = ""
    ts: int = 0


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _now() -> int:
    return int(time.time())


def _short_id() -> str:
    return uuid.uuid4().hex[:16]


class GrowthStore:
    """CRUD for the four growth-substrate tables.

    Pure persistence: no mana regen logic, no session orchestration. The
    upper layers (:class:`Economy`, :class:`CompanionGrowthSession`) own
    the business rules and call this for reads/writes.
    """

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    # ── Backlog ───────────────────────────────────────────────────────

    async def add_backlog_item(
        self,
        *,
        user_id: str,
        item_type: str,
        agent_id: str = DEFAULT_AGENT_ID,
        target_ref: str = "",
        rationale: str = "",
        priority: float = 0.5,
        source_signal: str = "",
        expected_berry_yield: float = 0.0,
        expected_mana_cost: float = 0.0,
        expected_berry_cost: float = 0.0,
    ) -> BacklogItem:
        if not user_id:
            raise ValueError("GrowthStore.add_backlog_item requires user_id")
        if not item_type:
            raise ValueError("GrowthStore.add_backlog_item requires item_type")
        item = BacklogItem(
            id=_short_id(),
            user_id=user_id,
            agent_id=agent_id,
            item_type=item_type,
            target_ref=target_ref,
            rationale=rationale,
            priority=priority,
            source_signal=source_signal,
            expected_berry_yield=expected_berry_yield,
            expected_mana_cost=expected_mana_cost,
            expected_berry_cost=expected_berry_cost,
            state="pending",
            created_at=_now(),
        )
        await self._conn.execute(
            """
            INSERT INTO companion_growth_backlog (
                id, user_id, agent_id, item_type, target_ref, rationale,
                priority, source_signal,
                expected_berry_yield, expected_mana_cost, expected_berry_cost,
                success_count, fail_count, state, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?)
            """,
            (
                item.id, item.user_id, item.agent_id, item.item_type,
                item.target_ref, item.rationale,
                item.priority, item.source_signal,
                item.expected_berry_yield, item.expected_mana_cost,
                item.expected_berry_cost,
                item.state, item.created_at,
            ),
        )
        await self._conn.commit()
        return item

    async def get_backlog_item(
        self, item_id: str, *, user_id: str, agent_id: str = DEFAULT_AGENT_ID,
    ) -> BacklogItem | None:
        cursor = await self._conn.execute(
            "SELECT * FROM companion_growth_backlog "
            "WHERE id = ? AND user_id = ? AND agent_id = ?",
            (item_id, user_id, agent_id),
        )
        row = await cursor.fetchone()
        return _row_to_backlog(cursor, row) if row else None

    async def list_backlog(
        self,
        *,
        user_id: str,
        agent_id: str = DEFAULT_AGENT_ID,
        state: str | None = "pending",
        limit: int = 20,
    ) -> list[BacklogItem]:
        sql = (
            "SELECT * FROM companion_growth_backlog "
            "WHERE user_id = ? AND agent_id = ?"
        )
        params: list[Any] = [user_id, agent_id]
        if state:
            sql += " AND state = ?"
            params.append(state)
        sql += " ORDER BY priority DESC, created_at ASC LIMIT ?"
        params.append(int(limit))
        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_backlog(cursor, r) for r in rows]

    async def update_backlog_attempt(
        self,
        item_id: str,
        *,
        user_id: str,
        agent_id: str = DEFAULT_AGENT_ID,
        success: bool,
    ) -> None:
        column = "success_count" if success else "fail_count"
        await self._conn.execute(
            f"UPDATE companion_growth_backlog "
            f"SET {column} = {column} + 1, last_attempted_at = ? "
            f"WHERE id = ? AND user_id = ? AND agent_id = ?",
            (_now(), item_id, user_id, agent_id),
        )
        await self._conn.commit()

    async def set_backlog_state(
        self,
        item_id: str,
        *,
        user_id: str,
        agent_id: str = DEFAULT_AGENT_ID,
        state: str,
    ) -> None:
        await self._conn.execute(
            "UPDATE companion_growth_backlog SET state = ? "
            "WHERE id = ? AND user_id = ? AND agent_id = ?",
            (state, item_id, user_id, agent_id),
        )
        await self._conn.commit()

    # ── Growth log ────────────────────────────────────────────────────

    async def start_session(
        self,
        *,
        user_id: str,
        agent_id: str = DEFAULT_AGENT_ID,
        backlog_id: str | None = None,
        plan: dict[str, Any] | None = None,
        snapshot_ref: str = "",
    ) -> GrowthLogEntry:
        if not user_id:
            raise ValueError("GrowthStore.start_session requires user_id")
        entry = GrowthLogEntry(
            id=_short_id(),
            user_id=user_id,
            agent_id=agent_id,
            backlog_id=backlog_id,
            started_at=_now(),
            plan=plan or {},
            snapshot_ref=snapshot_ref,
        )
        await self._conn.execute(
            """
            INSERT INTO companion_growth_log (
                id, user_id, agent_id, backlog_id, started_at,
                plan_json, act_log_json, consult_records_json,
                ledger_delta_json, outcome, tier, approval_state,
                mana_spent, berries_spent, berries_earned, snapshot_ref
            ) VALUES (?, ?, ?, ?, ?, ?, '[]', '[]', '{}', 'in_progress',
                      0, 'n/a', 0, 0, 0, ?)
            """,
            (
                entry.id, entry.user_id, entry.agent_id, entry.backlog_id,
                entry.started_at, json.dumps(entry.plan),
                entry.snapshot_ref,
            ),
        )
        await self._conn.commit()
        return entry

    async def append_act_step(
        self,
        session_id: str,
        *,
        user_id: str,
        agent_id: str = DEFAULT_AGENT_ID,
        step: dict[str, Any],
    ) -> None:
        # Atomic SQL-side append (json_insert '$[#]') instead of a
        # read-modify-write of the JSON list — the prior version relied on
        # a "single-active-session-per-user" contract enforced only at the
        # route layer, so two concurrent appends could read the same list
        # and the second UPDATE would clobber the first's step (audit
        # 2026-06-17). One statement = no lost update, no lock needed.
        cursor = await self._conn.execute(
            "UPDATE companion_growth_log "
            "SET act_log_json = json_insert(COALESCE(act_log_json, '[]'), '$[#]', json(?)) "
            "WHERE id = ? AND user_id = ? AND agent_id = ?",
            (json.dumps(step), session_id, user_id, agent_id),
        )
        if (cursor.rowcount or 0) == 0:
            log.warning("growth_store.append_act_step_missing", session_id=session_id)
        await self._conn.commit()

    async def finalize_session(
        self,
        session_id: str,
        *,
        user_id: str,
        agent_id: str = DEFAULT_AGENT_ID,
        outcome: str,
        tier: int = 0,
        approval_state: str = "n/a",
        ledger_delta: dict[str, Any] | None = None,
        mana_spent: float = 0.0,
        berries_spent: float = 0.0,
        berries_earned: float = 0.0,
    ) -> None:
        await self._conn.execute(
            """
            UPDATE companion_growth_log
            SET ended_at = ?, outcome = ?, tier = ?, approval_state = ?,
                ledger_delta_json = ?,
                mana_spent = ?, berries_spent = ?, berries_earned = ?
            WHERE id = ? AND user_id = ? AND agent_id = ?
            """,
            (
                _now(), outcome, int(tier), approval_state,
                json.dumps(ledger_delta or {}),
                float(mana_spent), float(berries_spent), float(berries_earned),
                session_id, user_id, agent_id,
            ),
        )
        await self._conn.commit()

    async def get_session(
        self, session_id: str, *, user_id: str, agent_id: str = DEFAULT_AGENT_ID,
    ) -> GrowthLogEntry | None:
        cursor = await self._conn.execute(
            "SELECT * FROM companion_growth_log "
            "WHERE id = ? AND user_id = ? AND agent_id = ?",
            (session_id, user_id, agent_id),
        )
        row = await cursor.fetchone()
        return _row_to_log(cursor, row) if row else None

    async def list_sessions(
        self,
        *,
        user_id: str,
        agent_id: str = DEFAULT_AGENT_ID,
        limit: int = 20,
    ) -> list[GrowthLogEntry]:
        cursor = await self._conn.execute(
            "SELECT * FROM companion_growth_log "
            "WHERE user_id = ? AND agent_id = ? "
            "ORDER BY started_at DESC LIMIT ?",
            (user_id, agent_id, int(limit)),
        )
        rows = await cursor.fetchall()
        return [_row_to_log(cursor, r) for r in rows]

    # ── Economy ───────────────────────────────────────────────────────

    async def get_or_create_economy(
        self, *, user_id: str, agent_id: str = DEFAULT_AGENT_ID,
    ) -> EconomyAccount:
        cursor = await self._conn.execute(
            "SELECT * FROM companion_economy "
            "WHERE user_id = ? AND agent_id = ?",
            (user_id, agent_id),
        )
        row = await cursor.fetchone()
        if row:
            return _row_to_economy(cursor, row)
        # Seed at defaults.
        account = EconomyAccount(
            user_id=user_id, agent_id=agent_id, last_mana_tick=_now(),
        )
        await self._conn.execute(
            """
            INSERT INTO companion_economy (
                user_id, agent_id, mana, mana_cap, mana_regen_per_hour,
                berries, berries_lifetime, last_mana_tick
            ) VALUES (?, ?, ?, ?, ?, 0, 0, ?)
            """,
            (
                account.user_id, account.agent_id,
                account.mana, account.mana_cap, account.mana_regen_per_hour,
                account.last_mana_tick,
            ),
        )
        await self._conn.commit()
        return account

    async def save_economy(self, account: EconomyAccount) -> None:
        await self._conn.execute(
            """
            UPDATE companion_economy
            SET mana = ?, mana_cap = ?, mana_regen_per_hour = ?,
                berries = ?, berries_lifetime = ?, last_mana_tick = ?
            WHERE user_id = ? AND agent_id = ?
            """,
            (
                account.mana, account.mana_cap, account.mana_regen_per_hour,
                account.berries, account.berries_lifetime, account.last_mana_tick,
                account.user_id, account.agent_id,
            ),
        )
        await self._conn.commit()

    # ── Economy transactions ─────────────────────────────────────────

    async def append_tx(
        self,
        *,
        user_id: str,
        tx_type: str,
        amount: float,
        agent_id: str = DEFAULT_AGENT_ID,
        growth_log_id: str | None = None,
        reason: str = "",
        signal_kind: str = "system",
        evidence_ref: str = "",
    ) -> int:
        cursor = await self._conn.execute(
            """
            INSERT INTO companion_economy_tx (
                user_id, agent_id, growth_log_id, tx_type,
                amount, reason, signal_kind, evidence_ref, ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, agent_id, growth_log_id, tx_type,
                float(amount), reason, signal_kind, evidence_ref, _now(),
            ),
        )
        await self._conn.commit()
        return int(cursor.lastrowid or 0)

    async def list_tx(
        self,
        *,
        user_id: str,
        agent_id: str = DEFAULT_AGENT_ID,
        growth_log_id: str | None = None,
        limit: int = 50,
    ) -> list[EconomyTx]:
        if growth_log_id:
            cursor = await self._conn.execute(
                "SELECT * FROM companion_economy_tx "
                "WHERE growth_log_id = ? AND user_id = ? AND agent_id = ? "
                "ORDER BY ts ASC, id ASC LIMIT ?",
                (growth_log_id, user_id, agent_id, int(limit)),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT * FROM companion_economy_tx "
                "WHERE user_id = ? AND agent_id = ? "
                "ORDER BY ts DESC, id DESC LIMIT ?",
                (user_id, agent_id, int(limit)),
            )
        rows = await cursor.fetchall()
        return [_row_to_tx(cursor, r) for r in rows]


# ---------------------------------------------------------------------------
# Row → dataclass helpers
# ---------------------------------------------------------------------------


def _columns(cursor: aiosqlite.Cursor) -> list[str]:
    return [d[0] for d in cursor.description]


def _row_to_backlog(cursor: aiosqlite.Cursor, row: tuple) -> BacklogItem:
    d = dict(zip(_columns(cursor), row, strict=True))
    return BacklogItem(
        id=d["id"], user_id=d["user_id"], agent_id=d["agent_id"],
        item_type=d["item_type"], target_ref=d["target_ref"],
        rationale=d["rationale"],
        priority=float(d["priority"]),
        source_signal=d["source_signal"],
        expected_berry_yield=float(d["expected_berry_yield"]),
        expected_mana_cost=float(d["expected_mana_cost"]),
        expected_berry_cost=float(d["expected_berry_cost"]),
        success_count=int(d["success_count"]),
        fail_count=int(d["fail_count"]),
        last_attempted_at=d["last_attempted_at"],
        last_consult_inconclusive_at=d["last_consult_inconclusive_at"],
        state=d["state"],
        created_at=int(d["created_at"]),
    )


def _row_to_log(cursor: aiosqlite.Cursor, row: tuple) -> GrowthLogEntry:
    d = dict(zip(_columns(cursor), row, strict=True))
    return GrowthLogEntry(
        id=d["id"], user_id=d["user_id"], agent_id=d["agent_id"],
        backlog_id=d["backlog_id"],
        started_at=int(d["started_at"]),
        ended_at=d["ended_at"],
        plan=json.loads(d["plan_json"] or "{}"),
        act_log=json.loads(d["act_log_json"] or "[]"),
        consult_records=json.loads(d["consult_records_json"] or "[]"),
        ledger_delta=json.loads(d["ledger_delta_json"] or "{}"),
        outcome=d["outcome"],
        tier=int(d["tier"]),
        approval_state=d["approval_state"],
        mana_spent=float(d["mana_spent"]),
        berries_spent=float(d["berries_spent"]),
        berries_earned=float(d["berries_earned"]),
        snapshot_ref=d["snapshot_ref"],
    )


def _row_to_economy(cursor: aiosqlite.Cursor, row: tuple) -> EconomyAccount:
    d = dict(zip(_columns(cursor), row, strict=True))
    return EconomyAccount(
        user_id=d["user_id"], agent_id=d["agent_id"],
        mana=float(d["mana"]), mana_cap=float(d["mana_cap"]),
        mana_regen_per_hour=float(d["mana_regen_per_hour"]),
        berries=float(d["berries"]),
        berries_lifetime=float(d["berries_lifetime"]),
        last_mana_tick=int(d["last_mana_tick"]),
    )


def _row_to_tx(cursor: aiosqlite.Cursor, row: tuple) -> EconomyTx:
    d = dict(zip(_columns(cursor), row, strict=True))
    return EconomyTx(
        id=int(d["id"]), user_id=d["user_id"], agent_id=d["agent_id"],
        growth_log_id=d["growth_log_id"], tx_type=d["tx_type"],
        amount=float(d["amount"]), reason=d["reason"],
        signal_kind=d["signal_kind"],
        evidence_ref=d["evidence_ref"],
        ts=int(d["ts"]),
    )
