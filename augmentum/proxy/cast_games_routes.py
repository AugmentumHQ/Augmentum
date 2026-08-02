"""Cast game profile + classify routes.

Per-(user, title_id) profile CRUD + a synchronous classify endpoint for
the library2 cast button. Endpoints:

  GET    /api/cast/games/{title_id}/profile        — read current
  PUT    /api/cast/games/{title_id}/profile        — manual override
  DELETE /api/cast/games/{title_id}/profile        — fall back to default
  POST   /api/cast/games/{title_id}/classify       — sync classify (no persist)
  GET    /api/cast/games/profiles                  — list this user's profiles

The classify endpoint is the load-bearing one for the cast-launch
flow: library2 calls it just before POSTing /api/cast/send, gets back
the surface_url + input_chain, and uses both. Existing cast flows
that don't call it still work — the default same-origin strategy is
the historical behaviour.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from augmentum.cast.games.classifier import CastClassifier
from augmentum.cast.games.models import (
    KNOWN_ADAPTERS,
    HostCapabilities,
    KeymapProfile,
    STRATEGY_CONTAINERIZED,
    STRATEGY_PROXY,
    STRATEGY_SHIM,
    _coerce_input_chain,
)
from augmentum.cast.games.registry import CastProfileRegistry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


router = APIRouter(prefix="/api/cast/games", tags=["cast"])


# ── Pydantic shapes ──────────────────────────────────────────────


class ClassifyRequest(BaseModel):
    """Lightweight title descriptor the cast-launch flow already has
    in hand. We pass it as-is to the classifier — the same shape the
    library2 cast button has for the artifact it's casting."""

    title_id: str = ""
    kind: str = ""
    display_name: str = ""
    embed_url: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProfileOverridePayload(BaseModel):
    """Fields the user may override via PUT. Each is optional; absent
    fields preserve the existing value (or the default if no row yet)."""

    strategy: str | None = None
    embed_url: str | None = None
    container_profile_id: str | None = None
    input_chain: list[str] | None = None
    keymap: dict[str, Any] | None = None
    quirks: dict[str, Any] | None = None
    notes: str | None = None


# ── Auth helpers ─────────────────────────────────────────────────


def _require_user(request: Request) -> str:
    user = request.scope.get("user")
    if user is None:
        raise HTTPException(401, "auth required")
    user_id = getattr(user, "id", "")
    if not user_id:
        raise HTTPException(401, "auth required")
    return user_id


def _registry_or_503(request: Request) -> CastProfileRegistry:
    reg = getattr(request.app.state, "cast_profile_registry", None)
    if reg is None:
        raise HTTPException(503, "cast profile registry unavailable")
    return reg


def _classifier_or_503(request: Request) -> CastClassifier:
    cf = getattr(request.app.state, "cast_classifier", None)
    if cf is None:
        raise HTTPException(503, "cast classifier unavailable")
    return cf


def _host_capabilities(request: Request) -> HostCapabilities:
    """Inspect app.state for what this host can spend on this cast.
    Cheap; no I/O — strategies use this to bail early on options the
    host can't service (e.g. containerized without AGSP)."""
    gs_runtime = getattr(request.app.state, "game_stream_runtime", None)
    has_agsp = gs_runtime is not None
    return HostCapabilities(
        has_gpu=bool(getattr(request.app.state, "has_gpu", False)),
        has_agsp=has_agsp,
        agsp_credits_available=0,  # Phase 5 will plumb live credit counts
        has_network_egress=True,
    )


# ── Endpoints ────────────────────────────────────────────────────


@router.get("/{title_id}/profile")
async def get_profile(title_id: str, request: Request) -> dict[str, Any]:
    """Read the saved profile for this (user, title_id), or 404 if
    none — meaning "the cast will fall back to the default strategy."
    """
    user_id = _require_user(request)
    reg = _registry_or_503(request)
    profile = await reg.get(title_id, user_id=user_id)
    if profile is None:
        raise HTTPException(404, "no saved profile for this title")
    return {"profile": profile.to_dict()}


@router.get("/profiles")
async def list_profiles(request: Request) -> dict[str, Any]:
    """List every saved profile for the authed user. Used by the
    settings detail-pane to surface overridden games."""
    user_id = _require_user(request)
    reg = _registry_or_503(request)
    profiles = await reg.list_for_user(user_id=user_id)
    return {"profiles": [p.to_dict() for p in profiles]}


@router.put("/{title_id}/profile")
async def upsert_profile(
    title_id: str,
    payload: ProfileOverridePayload,
    request: Request,
) -> dict[str, Any]:
    """Manual override — merges the payload over the current profile
    (or a fresh default if none exists) and persists with
    ``classified_by='manual'``."""
    user_id = _require_user(request)
    reg = _registry_or_503(request)
    if not title_id:
        raise HTTPException(400, "title_id is required")

    fields: dict[str, Any] = {}
    if payload.strategy is not None:
        s = payload.strategy.lower().strip()
        if s not in (STRATEGY_SHIM, STRATEGY_PROXY, STRATEGY_CONTAINERIZED):
            raise HTTPException(400, f"unknown strategy: {payload.strategy!r}")
        fields["strategy"] = s
    if payload.embed_url is not None:
        fields["embed_url"] = payload.embed_url
    if payload.container_profile_id is not None:
        fields["container_profile_id"] = payload.container_profile_id
    if payload.input_chain is not None:
        coerced = _coerce_input_chain(payload.input_chain)
        # Reject if the caller asked for adapters that don't exist (we
        # already silently drop them via _coerce; surface the rejection
        # so the UI can show an error rather than fail silently).
        unknown = [a for a in payload.input_chain if a not in KNOWN_ADAPTERS]
        if unknown:
            raise HTTPException(
                400, f"unknown adapter ids: {sorted(unknown)}",
            )
        fields["input_chain"] = coerced
    if payload.keymap is not None:
        fields["keymap"] = KeymapProfile.from_dict(payload.keymap)
    if payload.quirks is not None:
        fields["quirks"] = dict(payload.quirks)
    if payload.notes is not None:
        fields["notes"] = str(payload.notes)

    profile = await reg.override(title_id, user_id=user_id, **fields)
    return {"profile": profile.to_dict()}


@router.delete("/{title_id}/profile")
async def delete_profile(title_id: str, request: Request) -> dict[str, Any]:
    """Drop the profile so the next cast classifier from scratch."""
    user_id = _require_user(request)
    reg = _registry_or_503(request)
    removed = await reg.delete(title_id, user_id=user_id)
    return {"removed": removed}


@router.post("/{title_id}/classify")
async def classify(
    title_id: str,
    payload: ClassifyRequest,
    request: Request,
) -> dict[str, Any]:
    """Synchronous classify — returns the PreparedCast the receiver
    needs to mount + the input chain config for the loader.

    No persistence side-effects (the registry is only updated by
    explicit PUT or Phase-4 probe writes). Idempotent + cheap.
    """
    user_id = _require_user(request)
    classifier = _classifier_or_503(request)

    title = {
        "id": title_id or payload.title_id,
        "title_id": title_id or payload.title_id,
        "kind": payload.kind,
        "display_name": payload.display_name,
        "embed_url": payload.embed_url,
        "metadata": payload.metadata or {},
    }
    host = _host_capabilities(request)

    result = await classifier.classify(title, host=host, user_id=user_id)
    prepared = await result.strategy.prepare(title, result.profile)

    # First cast of an unknown title (no persisted profile) → schedule a
    # background probe so the NEXT cast is pre-classified. Fire-and-forget;
    # never blocks this cast, no-op when the probe coordinator / Playwright
    # is unavailable.
    if result.source == "default":
        coordinator = getattr(request.app.state, "cast_probe_coordinator", None)
        embed = str(payload.embed_url or (payload.metadata or {}).get("embed_url") or "")
        if coordinator is not None and embed:
            try:
                coordinator.maybe_probe(
                    title_id=title_id or payload.title_id,
                    user_id=user_id,
                    embed_url=embed,
                )
            except Exception:
                log.debug("cast_probe_schedule_failed", exc_info=True)

    return {
        "source": result.source,
        "profile": result.profile.to_dict(),
        "prepared": prepared.to_dict(),
    }
