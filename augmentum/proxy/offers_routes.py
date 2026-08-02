"""Offer substrate — HTTP surface.

Two small endpoints alongside the offer pieces that live on the
notification substrate:

* ``GET    /api/offers/suppressions``                       — list the
  user's "Never" rows. ("Not now" no longer suppresses anything — see
  migration 326.)
* ``DELETE /api/offers/suppressions/{kind}/{target_id}``    — Undo:
  remove a suppression row so the offer can surface again.

Active offers themselves are queried via ``GET /api/notify/feed`` with
``channel_id=system.offer``; accept/snooze/never go through the
notification action callback at ``POST /api/notify/{id}/action/{aid}``.

The ``system.offer`` action handler is registered at module import
time (same pattern Connect uses for its action handlers).

See ``docs/superpowers/specs/2026-06-02-offer-substrate-design.md``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from augmentum.config import settings as _settings
from augmentum.offers.handlers.system_offer import register_offer_action_handler
from augmentum.offers.store import (
    delete_suppression,
    list_suppressions,
)
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


router = APIRouter(prefix="/api/offers", tags=["offers"])


# Register the system.offer notification action handler at import
# time — same pattern Connect uses (``register_action_handler`` calls
# at the bottom of ``connect_routes.py``). Idempotent: re-registering
# replaces the binding cleanly.
register_offer_action_handler()


# ── Helpers ──────────────────────────────────────────────────────


def _offers_enabled() -> bool:
    return bool(getattr(_settings, "offers_enabled", True))


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    if user is None:
        raise HTTPException(status_code=401, detail="auth required")
    return user.id


def _resolve_conn(request: Request):
    sm = getattr(request.app.state, "state_manager", None)
    if sm is not None and isinstance(getattr(sm, "backend", None), SQLiteBackend):
        return sm.backend.conn
    return None


def _require_conn(request: Request):
    conn = _resolve_conn(request)
    if conn is None:
        raise HTTPException(
            status_code=503,
            detail="offers require a SQLite backend",
        )
    return conn


# ── HTTP ─────────────────────────────────────────────────────────


@router.get("/suppressions")
async def get_suppressions(request: Request) -> dict[str, Any]:
    """Return all suppression rows for the current user.

    In practice these are all ``reason='never'``: "Not now" stopped writing
    suppressions in migration 326, so there is no "Snoozed" countdown list to
    render any more, and the legacy snooze rows were cleared by that migration.

    No Settings → Offers tab exists yet — the per-chip Undo button in
    ``offer-chip.js`` is the live consumer of the DELETE route below. This
    endpoint is the read side for that tab when it gets built.
    """

    if not _offers_enabled():
        raise HTTPException(status_code=503, detail="offers disabled")
    uid = _user_id(request)
    conn = _require_conn(request)
    rows = await list_suppressions(conn, user_id=uid)
    return {
        "items": [
            {
                "kind": r.kind,
                "target_id": r.target_id,
                "suppressed_until": r.suppressed_until,
                "reason": r.reason,
                "is_permanent": r.is_permanent,
                "created_at": r.created_at,
            }
            for r in rows
        ],
    }


@router.delete("/suppressions/{kind}/{target_id}")
async def delete_suppression_row(
    kind: str, target_id: str, request: Request,
) -> dict[str, Any]:
    """Undo a snooze or never — the offer can surface again.

    Returns 200 even when no row existed (idempotent): the caller's
    intent ("don't suppress this") is satisfied either way.
    """

    if not _offers_enabled():
        raise HTTPException(status_code=503, detail="offers disabled")
    uid = _user_id(request)
    conn = _require_conn(request)
    removed = await delete_suppression(
        conn, user_id=uid, kind=kind, target_id=target_id,
    )
    return {"kind": kind, "target_id": target_id, "removed": removed}
