"""Load balancer management API routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from augmentum.auth.guards import require_admin
from augmentum.models.load_balancer import (
    AB_TEST,
    STRATEGIES,
    LoadBalancer,
    LoadBalancerRegistry,
)
from augmentum.state.balancer_store import BalancerConfig, BalancerStore
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/balancers", tags=["balancers"])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_store(request: Request) -> BalancerStore:
    store = getattr(request.app.state, "balancer_store", None)
    if not store:
        raise HTTPException(503, "Balancer store not available")
    return store


def _get_registry(request: Request) -> LoadBalancerRegistry:
    reg = getattr(request.app.state, "lb_registry", None)
    if not reg:
        raise HTTPException(503, "Balancer registry not available")
    return reg


async def _sync_balancer(store: BalancerStore, registry: LoadBalancerRegistry, balancer_id: str) -> None:
    """Reload a balancer from DB into the in-memory registry."""
    config = await store.get_balancer(balancer_id)
    if not config or not config.enabled:
        registry.unregister(balancer_id)
        return
    members = await store.list_members(balancer_id)
    lb = LoadBalancer(config, members)
    registry.register(balancer_id, lb)


# ------------------------------------------------------------------
# Request models
# ------------------------------------------------------------------

class BalancerCreateRequest(BaseModel):
    name: str
    strategy: str = "round_robin"
    fallback_enabled: bool = False
    enabled: bool = True


class BalancerUpdateRequest(BaseModel):
    name: str | None = None
    strategy: str | None = None
    fallback_enabled: bool | None = None
    enabled: bool | None = None


class MemberAddRequest(BaseModel):
    model_name: str
    backend_key: str
    weight: float = 1.0
    priority: int = 0


class MemberUpdateRequest(BaseModel):
    weight: float | None = None
    priority: int | None = None
    enabled: bool | None = None


class VoteRequest(BaseModel):
    model_name: str
    backend_key: str
    vote: str  # "up" or "down"
    session_id: str | None = None


# ------------------------------------------------------------------
# Balancer CRUD
# ------------------------------------------------------------------

@router.get("")
async def list_balancers(request: Request) -> list[dict]:
    store = _get_store(request)
    balancers = await store.list_balancers()
    result = []
    for b in balancers:
        members = await store.list_members(b.id)
        result.append({
            "id": b.id,
            "name": b.name,
            "strategy": b.strategy,
            "fallback_enabled": b.fallback_enabled,
            "enabled": b.enabled,
            "member_count": len(members),
            "members": [
                {
                    "id": m.id,
                    "model_name": m.model_name,
                    "backend_key": m.backend_key,
                    "weight": m.weight,
                    "priority": m.priority,
                    "enabled": m.enabled,
                    "last_used_at": m.last_used_at,
                }
                for m in members
            ],
            "created_at": b.created_at,
            "updated_at": b.updated_at,
        })
    return result


@router.post("")
async def create_balancer(req: BalancerCreateRequest, request: Request) -> dict:
    """Create a load balancer. Admin only — shared inference infrastructure."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    store = _get_store(request)
    registry = _get_registry(request)

    if not req.name or not req.name.strip():
        raise HTTPException(400, "Name is required")
    if req.strategy not in STRATEGIES:
        raise HTTPException(400, f"Invalid strategy. Must be one of: {', '.join(STRATEGIES)}")

    balancer_id = "lb_" + uuid.uuid4().hex[:12]
    config = BalancerConfig(
        id=balancer_id,
        name=req.name.strip(),
        strategy=req.strategy,
        fallback_enabled=req.fallback_enabled,
        enabled=req.enabled,
    )
    result = await store.create_balancer(config)
    if result.enabled:
        registry.register(result.id, LoadBalancer(result, []))

    log.info("balancer_created", id=result.id, name=result.name, strategy=result.strategy)
    return {
        "id": result.id,
        "name": result.name,
        "strategy": result.strategy,
        "fallback_enabled": result.fallback_enabled,
        "enabled": result.enabled,
    }


@router.put("/{balancer_id}")
async def update_balancer(balancer_id: str, req: BalancerUpdateRequest, request: Request) -> dict:
    """Update a load balancer. Admin only."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    store = _get_store(request)
    registry = _get_registry(request)

    fields = req.model_dump(exclude_none=True)
    if "strategy" in fields and fields["strategy"] not in STRATEGIES:
        raise HTTPException(400, f"Invalid strategy. Must be one of: {', '.join(STRATEGIES)}")

    result = await store.update_balancer(balancer_id, **fields)
    if not result:
        raise HTTPException(404, "Balancer not found")

    await _sync_balancer(store, registry, balancer_id)
    log.info("balancer_updated", id=balancer_id)
    return {
        "id": result.id,
        "name": result.name,
        "strategy": result.strategy,
        "fallback_enabled": result.fallback_enabled,
        "enabled": result.enabled,
    }


@router.delete("/{balancer_id}")
async def delete_balancer(balancer_id: str, request: Request) -> dict:
    """Delete a load balancer. Admin only."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    store = _get_store(request)
    registry = _get_registry(request)

    deleted = await store.delete_balancer(balancer_id)
    if not deleted:
        raise HTTPException(404, "Balancer not found")

    registry.unregister(balancer_id)
    log.info("balancer_deleted", id=balancer_id)
    return {"deleted": True}


# ------------------------------------------------------------------
# Members
# ------------------------------------------------------------------

@router.get("/{balancer_id}/members")
async def list_members(balancer_id: str, request: Request) -> list[dict]:
    store = _get_store(request)
    members = await store.list_members(balancer_id)
    return [
        {
            "id": m.id,
            "model_name": m.model_name,
            "backend_key": m.backend_key,
            "weight": m.weight,
            "priority": m.priority,
            "enabled": m.enabled,
            "last_used_at": m.last_used_at,
        }
        for m in members
    ]


@router.post("/{balancer_id}/members")
async def add_member(balancer_id: str, req: MemberAddRequest, request: Request) -> dict:
    """Add a member to a load balancer. Admin only."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    store = _get_store(request)
    registry = _get_registry(request)

    config = await store.get_balancer(balancer_id)
    if not config:
        raise HTTPException(404, "Balancer not found")

    member = await store.add_member(
        balancer_id, req.model_name, req.backend_key, req.weight, req.priority,
    )
    await _sync_balancer(store, registry, balancer_id)
    log.info("balancer_member_added", balancer=balancer_id, model=req.model_name, backend=req.backend_key)
    return {
        "id": member.id,
        "model_name": member.model_name,
        "backend_key": member.backend_key,
        "weight": member.weight,
        "priority": member.priority,
    }


@router.put("/{balancer_id}/members/{member_id}")
async def update_member(balancer_id: str, member_id: int, req: MemberUpdateRequest, request: Request) -> dict:
    """Update a load balancer member. Admin only."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    store = _get_store(request)
    registry = _get_registry(request)

    fields = req.model_dump(exclude_none=True)
    if not fields:
        raise HTTPException(400, "No fields to update")

    updated = await store.update_member(member_id, **fields)
    if not updated:
        raise HTTPException(404, "Member not found")

    await _sync_balancer(store, registry, balancer_id)
    return {"updated": True}


@router.delete("/{balancer_id}/members/{member_id}")
async def remove_member(balancer_id: str, member_id: int, request: Request) -> dict:
    """Remove a load balancer member. Admin only."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    store = _get_store(request)
    registry = _get_registry(request)

    removed = await store.remove_member(member_id)
    if not removed:
        raise HTTPException(404, "Member not found")

    await _sync_balancer(store, registry, balancer_id)
    log.info("balancer_member_removed", balancer=balancer_id, member_id=member_id)
    return {"removed": True}


# ------------------------------------------------------------------
# A/B Votes
# ------------------------------------------------------------------

@router.post("/{balancer_id}/vote")
async def record_vote(balancer_id: str, req: VoteRequest, request: Request) -> dict:
    store = _get_store(request)

    if req.vote not in ("up", "down"):
        raise HTTPException(400, "Vote must be 'up' or 'down'")

    config = await store.get_balancer(balancer_id)
    if not config:
        raise HTTPException(404, "Balancer not found")
    if config.strategy != AB_TEST:
        raise HTTPException(400, "Voting is only available for A/B test balancers")

    _u = request.scope.get("user")
    _uid = _u.id if _u else ""
    await store.record_vote(
        balancer_id, req.model_name, req.backend_key, req.vote, req.session_id,
        user_id=_uid,
    )
    log.info("ab_vote_recorded", balancer=balancer_id, model=req.model_name, vote=req.vote)
    return {"recorded": True}


@router.get("/{balancer_id}/stats")
async def get_stats(balancer_id: str, request: Request) -> dict:
    store = _get_store(request)
    _u = request.scope.get("user")
    _uid = _u.id if _u else ""
    stats = await store.get_vote_stats(balancer_id, user_id=_uid)
    return {
        "balancer_id": balancer_id,
        "models": [
            {
                "model_name": s.model_name,
                "backend_key": s.backend_key,
                "up": s.up,
                "down": s.down,
                "total": s.total,
                "score": round(s.score, 3),
            }
            for s in stats
        ],
    }
