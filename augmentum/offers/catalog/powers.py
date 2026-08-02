"""Power activation catalog — discovered from `.augmentum/powers/` at import.

Powers are filesystem-backed capability packs (one POWER.md per
directory). Discovering them at catalog-import time keeps the offer
list authoritative without per-power hand-curation: drop a new
power directory in, the offer catalog picks it up on next process
start.

**Activation, not enabling.** Powers default to enabled per-user
(``PowerStateStore.is_enabled`` returns True when the key is missing).
The meaningful primitive is ``activate_power(workspace_id, power_id)``
which sets ONE active power per workspace — the active power's
prompt block is what actually gets injected into coder turns. So the
offer's accept calls ``activate_power``, not ``set_enabled``.

The substrate threads ``workspace_id`` via
``_context['workspace_id']`` (stamped by ``_execute_tool`` in
``modes/coder/handler.py``) → propose_offer tool stashes it in
``extra['_workspace_id']`` → this accept reads it back. Missing
workspace_id → ``ok=False, error="no_workspace"`` rather than
falling through to a no-op.

Only one power is active per workspace at a time. If the user accepts
an offer for power-B while power-A is active, the response reports
``replaced="power-a"`` so the chip can surface the swap clearly.

User-scoped: each user owns their own activation state, restricted
to coder mode by ``allowed_modes``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from augmentum.offers.catalog.base import (
    CatalogEntry,
    OfferPreview,
    register_kind,
)
from augmentum.powers.manifest import (
    discover_manifest_file,
    parse_power_manifest,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request

    from augmentum.powers.models import PowerManifest


log = get_logger(__name__)


KIND: str = "power"


# ── Discovery ────────────────────────────────────────────────────


def _repo_root() -> Path:
    # offers/catalog/powers.py → .../augmentum/offers/catalog → .../augmentum/offers
    # → .../augmentum → repo root (one above the augmentum/ package dir).
    return Path(__file__).resolve().parents[3]


def _discover_manifests() -> list[PowerManifest]:
    """Scan ``.augmentum/powers/<dir>/POWER.md`` and parse manifests.

    Catches per-manifest parse errors so one broken POWER.md doesn't
    take the whole offer catalog offline. The PowerRegistry does the
    same thing at runtime, but we re-scan here at import time to
    avoid coupling to app-state availability.
    """

    root = _repo_root() / ".augmentum" / "powers"
    if not root.is_dir():
        return []

    manifests: list[PowerManifest] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        manifest_path = discover_manifest_file(child)
        if manifest_path is None:
            continue
        try:
            manifests.append(parse_power_manifest(manifest_path, source_kind="native"))
        except Exception as exc:
            log.warning(
                "offer_power_manifest_parse_failed",
                path=str(manifest_path), error=str(exc)[:160],
            )
    return manifests


# ── Per-power entry factory ──────────────────────────────────────


def _make_entry(manifest: PowerManifest) -> CatalogEntry:
    power_id = manifest.id
    display = manifest.display_name or manifest.slug
    description = manifest.description or ""
    triggers = list(manifest.triggers) if manifest.triggers else []
    modes = list(manifest.mode_scope) if manifest.mode_scope else []

    async def _preview(_target_id: str, user_id: str) -> OfferPreview | None:
        # Preview can't reach app.state (no Request here), so the
        # "already active on this workspace" check happens at accept
        # time and returns ``already_active=True`` idempotently.
        # ``modes`` here is the manifest's mode_scope — what modes
        # the power's prompt block is valid for, NOT to be confused
        # with the offer's allowed_modes (coder-only at the entry
        # level below).
        return OfferPreview(
            label=f"{display} (power)",
            hint=description[:120] + ("…" if len(description) > 120 else ""),
            details={
                "scope": "user",
                "kind": manifest.kind,
                "modes": modes,
                "triggers": triggers[:6],
            },
        )

    async def _accept(payload: dict[str, Any], request: Request) -> dict[str, Any]:
        store = getattr(request.app.state, "power_state_store", None)
        user = request.scope.get("user")
        user_id = getattr(user, "id", "") if user is not None else ""

        if store is None:
            return {
                "ok": False,
                "error": "no_power_state_store",
                "detail": "Powers subsystem not initialized.",
            }
        if not user_id:
            return {
                "ok": False,
                "error": "no_user",
                "detail": "Power activation is per-user; no authenticated user.",
            }

        # Workspace_id was stashed by the propose_offer tool when the
        # coder handler dispatched the call. Missing it means either
        # the offer was proposed from a non-coder surface (the
        # allowed_modes gate should have blocked that — defense in
        # depth) or the user accepted from a session that lost the
        # workspace context. Either way, no implicit fallback.
        extra = payload.get("extra") or {}
        workspace_id = str(extra.get("_workspace_id") or "").strip()
        if not workspace_id:
            return {
                "ok": False,
                "error": "no_workspace",
                "detail": (
                    "Power activation needs a workspace. The propose_offer "
                    "call should have been dispatched from coder mode with "
                    "a workspace context."
                ),
            }

        try:
            # If THIS power is already active on THIS workspace,
            # don't re-write — idempotent ack.
            existing = await store.get_active_power(
                user_id, workspace_id=workspace_id,
            )
            if existing is not None and existing.power_id == power_id:
                return {
                    "ok": True,
                    "power_id": power_id,
                    "workspace_id": workspace_id,
                    "already_active": True,
                    "next_step": (
                        f"{display} is already the active power on this "
                        "workspace. Coder turns are already using it."
                    ),
                }

            replaced_power_id = existing.power_id if existing else ""
            await store.activate_power(
                user_id,
                workspace_id=workspace_id,
                power_id=power_id,
                source="offer",
                scope="workspace",
                reason="user accepted offer chip",
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": "store_failed",
                "detail": str(exc)[:200],
            }

        log.info(
            "offer_power_activated",
            power_id=power_id, user_id=user_id,
            workspace_id=workspace_id, replaced=replaced_power_id,
        )
        result: dict[str, Any] = {
            "ok": True,
            "power_id": power_id,
            "workspace_id": workspace_id,
            "already_active": False,
        }
        if replaced_power_id:
            result["replaced"] = replaced_power_id
            result["next_step"] = (
                f"{display} is now the active power on this workspace, "
                f"replacing {replaced_power_id}. Only one power can be "
                f"active per workspace at a time."
            )
        else:
            result["next_step"] = (
                f"{display} is now the active power on this workspace. "
                f"Its prompt block will be injected into the next coder turn."
            )
        return result

    return CatalogEntry(
        kind=KIND,
        target_id=power_id,
        title=f"Activate {display} on this workspace?",
        scope="user",
        build_preview=_preview,
        accept=_accept,
        # Powers attach to coder workspaces — surfacing from other
        # modes creates a chip the user can accept but the workspace
        # context isn't available, so accept rejects with no_workspace.
        allowed_modes=("coder",),
    )


# Build entries at import time so the catalog is ready when the
# offer dispatcher is. Re-importing the module rebuilds the list —
# matches the rest of the catalog pattern. Note: new powers added
# after process startup require a restart to surface; that's the
# same restart cadence the PowerRegistry itself uses.
ENTRIES: list[CatalogEntry] = [_make_entry(m) for m in _discover_manifests()]


# Skip registering if discovery found nothing — keeps test
# environments where .augmentum/powers/ may be absent from emitting
# a noisy "empty kind" registration.
if ENTRIES:
    register_kind(KIND, ENTRIES)
