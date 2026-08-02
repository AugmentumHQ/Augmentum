"""HTTP routes for the companion growth loop (Phase 1).

Spec: ``docs/superpowers/specs/2026-05-31-companion-growth-loop-design.md``

Endpoints exposed for the unified Becca panel + autonomous-trigger
substrate (Phase 3 will call ``/run`` internally):

  * ``POST   /api/companion/growth/backlog``       — queue an item
  * ``GET    /api/companion/growth/backlog``       — list queued items
  * ``POST   /api/companion/growth/run``           — fire a session (manual)
  * ``POST   /api/companion/growth/reward``        — post an explicit signal
  * ``POST   /api/companion/growth/sponsor``       — user-funded goal
  * ``GET    /api/companion/growth/log``           — recent sessions
  * ``GET    /api/companion/growth/actions``       — dispatchable catalog
  * ``GET    /api/companion/growth/economy``       — mana/berry balance
  * ``GET    /api/companion/growth/economy/tx``    — audit ledger

All endpoints require auth; user_id is sourced from
``request.scope["user"].id`` per the multi-tenant pattern in CLAUDE.md.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from augmentum.companion.growth import (
    CompanionGrowthSession,
    Economy,
    GrowthStore,
)
from augmentum.companion.growth.actions import ACTIONS, ActionRequest
from augmentum.companion.growth.rewards import apply_explicit, apply_implicit
from augmentum.companion.growth.session import SessionConfig
from augmentum.companion.growth.store import DEFAULT_AGENT_ID
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["companion-growth"])


# ── Helpers ──────────────────────────────────────────────────────────


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _require_user(request: Request) -> str:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "Unauthorized")
    return uid


def _growth_store(request: Request) -> GrowthStore:
    store = getattr(request.app.state, "growth_store", None)
    if store is None:
        raise HTTPException(503, "growth_store not initialized")
    return store


def _memory_store(request: Request) -> Any:
    """Return ``app.state.memory_store`` or ``None`` if not initialized.

    Actions tolerate a missing store by returning ``ok=False``; we don't
    503 here because some actions may not need it.
    """
    return getattr(request.app.state, "memory_store", None)


# ── Request models ───────────────────────────────────────────────────


class _AddBacklogBody(BaseModel):
    item_type: str = Field(..., min_length=1, max_length=64)
    target_ref: str = Field("", max_length=512)
    rationale: str = Field("", max_length=2000)
    priority: float = Field(0.5, ge=0.0, le=1.0)
    source_signal: str = Field("", max_length=128)
    expected_berry_yield: float = Field(0.0)
    expected_mana_cost: float = Field(0.0)
    expected_berry_cost: float = Field(0.0)


class _RunBody(BaseModel):
    """Fire a growth session.

    Either ``backlog_id`` or (``action_type`` + ``target_ref``) must be
    set; backlog_id wins if both supplied. Empty bodies are rejected.
    """

    backlog_id: str | None = Field(None, max_length=64)
    action_type: str | None = Field(None, max_length=64)
    target_ref: str = Field("", max_length=512)
    rationale: str = Field("", max_length=2000)
    max_steps: int = Field(5, ge=1, le=20)
    budget_mana: float = Field(50.0, ge=0.0)
    budget_berries: float = Field(0.0, ge=0.0)


class _RewardBody(BaseModel):
    growth_log_id: str = Field(..., min_length=1, max_length=64)
    signal: str = Field(..., min_length=1, max_length=64)
    channel: str = Field("explicit", max_length=16)  # explicit | implicit
    evidence_ref: str = Field("", max_length=512)


class _SponsorBody(BaseModel):
    """User-named goal to queue with high priority + optional berry grant.

    target_ref: required prose ("learn to recommend music I like").
    item_type:  optional; defaults to ``sponsored_goal`` (open vocab —
                Phase 1 backlog accepts unknown types, see add_backlog).
    berry_grant: optional non-negative; if > 0, Economy.sponsor() grants
                that many berries (user-initiated, tagged user_action).
    """

    target_ref: str = Field(..., min_length=1, max_length=512)
    item_type: str = Field("sponsored_goal", min_length=1, max_length=64)
    rationale: str = Field("", max_length=2000)
    berry_grant: float = Field(0.0, ge=0.0, le=1000.0)


# ── Routes ───────────────────────────────────────────────────────────


@router.post("/api/companion/growth/backlog")
async def add_backlog(body: _AddBacklogBody, request: Request) -> JSONResponse:
    """Add an item to the growth backlog."""
    user_id = _require_user(request)
    store = _growth_store(request)

    if body.item_type not in ACTIONS:
        # Don't 400 on unknown types — Phase 1 ships only recall_connect,
        # but the backlog vocabulary is intentionally open for Phase 3+.
        # Just log it.
        log.info(
            "growth.backlog.unknown_type_queued",
            item_type=body.item_type, user_id=user_id,
        )

    item = await store.add_backlog_item(
        user_id=user_id,
        item_type=body.item_type,
        target_ref=body.target_ref,
        rationale=body.rationale,
        priority=body.priority,
        source_signal=body.source_signal,
        expected_berry_yield=body.expected_berry_yield,
        expected_mana_cost=body.expected_mana_cost,
        expected_berry_cost=body.expected_berry_cost,
    )
    return JSONResponse({
        "ok": True,
        "id": item.id,
        "item_type": item.item_type,
        "priority": item.priority,
        "state": item.state,
    })


@router.post("/api/companion/growth/run")
async def run_session(body: _RunBody, request: Request) -> JSONResponse:
    """Fire a growth-loop session (manual / ad-hoc).

    Phase 1: caller-driven only. Autonomous triggers (Phase 3) call this
    same path internally.
    """
    user_id = _require_user(request)
    store = _growth_store(request)
    economy = Economy(store, user_id=user_id, agent_id=DEFAULT_AGENT_ID)

    backlog_item = None
    ad_hoc: ActionRequest | None = None

    if body.backlog_id:
        backlog_item = await store.get_backlog_item(
            body.backlog_id, user_id=user_id, agent_id=DEFAULT_AGENT_ID,
        )
        if backlog_item is None:
            raise HTTPException(404, f"backlog item {body.backlog_id} not found")
    elif body.action_type:
        ad_hoc = ActionRequest(
            action_type=body.action_type,
            target_ref=body.target_ref,
            rationale=body.rationale,
        )
    else:
        raise HTTPException(
            400,
            "either backlog_id or action_type must be supplied",
        )

    session = CompanionGrowthSession(
        store=store,
        economy=economy,
        user_id=user_id,
        agent_id=DEFAULT_AGENT_ID,
        backlog=backlog_item,
        config=SessionConfig(
            max_steps=body.max_steps,
            budget_mana=body.budget_mana,
            budget_berries=body.budget_berries,
        ),
        memory_store=_memory_store(request),
    )

    log_entry = await session.run(ad_hoc_request=ad_hoc)
    return JSONResponse({
        "ok": True,
        "session_id": log_entry.id,
        "outcome": log_entry.outcome,
        "tier": log_entry.tier,
        "mana_spent": log_entry.mana_spent,
        "berries_spent": log_entry.berries_spent,
        "berries_earned": log_entry.berries_earned,
        "act_log": log_entry.act_log,
        "ledger_delta": log_entry.ledger_delta,
    })


@router.post("/api/companion/growth/reward")
async def post_reward(body: _RewardBody, request: Request) -> JSONResponse:
    """Apply a reward signal to a finished growth session."""
    user_id = _require_user(request)
    store = _growth_store(request)

    session = await store.get_session(
        body.growth_log_id, user_id=user_id, agent_id=DEFAULT_AGENT_ID,
    )
    if session is None:
        raise HTTPException(404, f"session {body.growth_log_id} not found")

    economy = Economy(store, user_id=user_id, agent_id=DEFAULT_AGENT_ID)

    if body.channel == "implicit":
        outcome = await apply_implicit(
            economy, signal=body.signal,
            growth_log_id=body.growth_log_id,
            evidence_ref=body.evidence_ref,
        )
    else:
        outcome = await apply_explicit(
            economy, signal=body.signal,
            growth_log_id=body.growth_log_id,
            evidence_ref=body.evidence_ref,
        )

    return JSONResponse({
        "ok": outcome.ok,
        "delta": outcome.delta,
        "berries_after": outcome.berries_after,
        "reason": outcome.reason,
    })


@router.get("/api/companion/growth/log")
async def list_log(request: Request, limit: int = 20) -> JSONResponse:
    """Recent growth sessions for the authenticated user."""
    user_id = _require_user(request)
    store = _growth_store(request)
    sessions = await store.list_sessions(
        user_id=user_id, agent_id=DEFAULT_AGENT_ID, limit=min(limit, 100),
    )
    return JSONResponse({
        "ok": True,
        "sessions": [
            {
                "id": s.id,
                "started_at": s.started_at,
                "ended_at": s.ended_at,
                "outcome": s.outcome,
                "tier": s.tier,
                "approval_state": s.approval_state,
                "mana_spent": s.mana_spent,
                "berries_spent": s.berries_spent,
                "berries_earned": s.berries_earned,
                "plan": s.plan,
                "ledger_delta": s.ledger_delta,
            }
            for s in sessions
        ],
    })


@router.get("/api/companion/growth/economy")
async def get_economy(request: Request) -> JSONResponse:
    """Current mana / berry balance for the authenticated user."""
    user_id = _require_user(request)
    store = _growth_store(request)
    economy = Economy(store, user_id=user_id, agent_id=DEFAULT_AGENT_ID)
    account = await economy.snapshot()
    return JSONResponse({
        "ok": True,
        "mana": account.mana,
        "mana_cap": account.mana_cap,
        "mana_regen_per_hour": account.mana_regen_per_hour,
        "berries": account.berries,
        "berries_lifetime": account.berries_lifetime,
        "last_mana_tick": account.last_mana_tick,
    })


@router.get("/api/companion/growth/backlog")
async def list_backlog(
    request: Request, state: str = "pending", limit: int = 20,
) -> JSONResponse:
    """List backlog items for the authenticated user. Defaults to
    ``state='pending'``; pass ``state=''`` (empty) to see every state.
    """
    user_id = _require_user(request)
    store = _growth_store(request)
    items = await store.list_backlog(
        user_id=user_id,
        agent_id=DEFAULT_AGENT_ID,
        state=state or None,
        limit=min(int(limit), 100),
    )
    return JSONResponse({
        "ok": True,
        "items": [
            {
                "id": i.id,
                "item_type": i.item_type,
                "target_ref": i.target_ref,
                "rationale": i.rationale,
                "priority": i.priority,
                "source_signal": i.source_signal,
                "success_count": i.success_count,
                "fail_count": i.fail_count,
                "last_attempted_at": i.last_attempted_at,
                "state": i.state,
                "created_at": i.created_at,
            }
            for i in items
        ],
    })


@router.post("/api/companion/growth/sponsor")
async def sponsor_goal(body: _SponsorBody, request: Request) -> JSONResponse:
    """Queue a user-named goal at high priority + optionally grant berries.

    The sponsorship semantic from the spec is "user puts weight behind
    this thread" — modeled as (a) a backlog item with priority=1.0 and
    source_signal=user_sponsor, plus (b) an optional berry grant routed
    through Economy.sponsor() so the agent has budget for the swing.
    """
    user_id = _require_user(request)
    store = _growth_store(request)

    item = await store.add_backlog_item(
        user_id=user_id,
        item_type=body.item_type,
        target_ref=body.target_ref,
        rationale=body.rationale,
        priority=1.0,
        source_signal="user_sponsor",
    )

    grant_result = None
    if body.berry_grant > 0:
        economy = Economy(store, user_id=user_id, agent_id=DEFAULT_AGENT_ID)
        outcome = await economy.sponsor(
            body.berry_grant,
            reason=f"sponsor:{item.id}",
            evidence_ref=item.id,
        )
        grant_result = {
            "ok": outcome.ok,
            "delta": outcome.delta,
            "berries_after": outcome.berries_after,
        }

    return JSONResponse({
        "ok": True,
        "id": item.id,
        "item_type": item.item_type,
        "target_ref": item.target_ref,
        "priority": item.priority,
        "state": item.state,
        "berry_grant": grant_result,
    })


@router.get("/api/companion/growth/actions")
async def list_actions(request: Request) -> JSONResponse:
    """Catalog of registered action handlers.

    The Becca panel uses this to (a) populate the sponsor-form
    item_type select and (b) decide whether a queued backlog row can be
    dispatched via the Run button. An item_type not in this catalog
    will abort cleanly at session-time — the UI surfaces that gap so
    the user isn't promised behavioral change that won't happen.
    """
    _require_user(request)
    return JSONResponse({
        "ok": True,
        "actions": [
            {
                "action_type": h.action_type,
                "tier": int(getattr(h, "tier", 0)),
                "mana_cost": float(getattr(h, "mana_cost", 1.0)),
            }
            for h in ACTIONS.values()
        ],
    })


@router.get("/api/companion/growth/economy/tx")
async def list_economy_tx(request: Request, limit: int = 50) -> JSONResponse:
    """Recent companion_economy_tx rows — the audit ledger surfaced
    in the Becca panel so the user can see *why* berries moved.
    """
    user_id = _require_user(request)
    store = _growth_store(request)
    rows = await store.list_tx(
        user_id=user_id,
        agent_id=DEFAULT_AGENT_ID,
        limit=min(int(limit), 200),
    )
    return JSONResponse({
        "ok": True,
        "tx": [
            {
                "id": r.id,
                "ts": r.ts,
                "tx_type": r.tx_type,
                "amount": r.amount,
                "reason": r.reason,
                "signal_kind": r.signal_kind,
                "growth_log_id": r.growth_log_id,
                "evidence_ref": r.evidence_ref,
            }
            for r in rows
        ],
    })
