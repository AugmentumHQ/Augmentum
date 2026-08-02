"""Workspace tooling-profile offer catalog.

One entry per profile in
:mod:`augmentum.coder.profiles`. "Install" semantics: create a new
coder workspace pre-configured with that profile. We *don't* try to
rebuild an existing workspace in place — workspace creation is
already a stateful container operation, and rebuilds during active
runs would corrupt state. Additive ("here's a new workspace with
the profile you wanted") is the safe shape.

User-scoped: each user owns their own workspaces. The accept
handler reaches into ``app.state.container_manager`` and calls
``create_workspace`` with the authenticated user_id stamped on the
ownership record.

Depends on :mod:`augmentum.coder.profiles` (tooling-profile-system-v2)
which landed 2026-06-02. ``all_profiles()`` is the catalog source —
adding a profile there auto-surfaces here on next process start.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from augmentum.coder import profiles as _profiles
from augmentum.offers.catalog.base import (
    CatalogEntry,
    OfferPreview,
    register_kind,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request

    from augmentum.coder.profiles import ToolingProfile


log = get_logger(__name__)


KIND: str = "workspace_profile"


def _make_entry(profile: ToolingProfile) -> CatalogEntry:
    profile_id = profile.id
    label = profile.label
    description = profile.description
    est_size = profile.est_size_mb
    est_setup = profile.est_setup_sec

    async def _preview(_target_id: str, _user_id: str) -> OfferPreview | None:
        return OfferPreview(
            label=f"{label} workspace",
            hint=description,
            details={
                "scope": "user",
                "profile_id": profile_id,
                "inherits": profile.inherits or "",
                "est_size_mb": est_size,
                "est_setup_sec": est_setup,
                "network_policy": profile.network_policy,
            },
        )

    async def _accept(payload: dict[str, Any], request: Request) -> dict[str, Any]:
        mgr = getattr(request.app.state, "container_manager", None)
        user = request.scope.get("user")
        user_id = getattr(user, "id", "") if user is not None else ""

        if mgr is None:
            return {
                "ok": False,
                "error": "no_container_manager",
                "detail": (
                    "Coder workspaces require Docker. Install Docker and "
                    "restart, or use the existing chat surfaces."
                ),
            }
        if not user_id:
            return {
                "ok": False,
                "error": "no_user",
                "detail": "Workspace creation is per-user; no authenticated user.",
            }

        # Name uses timestamp suffix so accepting twice doesn't try to
        # create the same workspace. Short suffix keeps it readable
        # in the workspace dropdown.
        name = f"{profile_id}-{int(time.time()) % 1_000_000}"

        try:
            info = await mgr.create_workspace(
                name=name,
                tooling_profile=profile_id,
                user_id=user_id,
            )
        except Exception as exc:
            log.warning(
                "offer_workspace_profile_create_failed",
                profile_id=profile_id, error=str(exc)[:200],
            )
            return {
                "ok": False,
                "error": "create_failed",
                "detail": str(exc)[:200],
            }

        log.info(
            "offer_workspace_profile_created",
            profile_id=profile_id, workspace_id=info.id, user_id=user_id,
        )
        return {
            "ok": True,
            "workspace_id": info.id,
            "name": info.name,
            "profile_id": profile_id,
            "next_step": (
                f"Workspace '{info.name}' created with the {label} profile. "
                f"Open the Coder surface to switch to it. The new workspace "
                f"is additive — your existing workspaces are untouched."
            ),
        }

    return CatalogEntry(
        kind=KIND,
        target_id=profile_id,
        title=f"Create a {label} workspace?",
        scope="user",
        build_preview=_preview,
        accept=_accept,
        # Workspaces are a coder-mode concept; offering them from
        # passthrough or narrative is contextually wrong.
        allowed_modes=("coder",),
    )


ENTRIES: list[CatalogEntry] = [_make_entry(p) for p in _profiles.all_profiles()]


if ENTRIES:
    register_kind(KIND, ENTRIES)
