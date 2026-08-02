"""Discover routes — unified browse + install across providers,
titles, and community content.

Spec: docs/superpowers/specs/2026-06-10-discover-surface-design.md

Endpoints:

* ``GET  /api/discover/catalog``     — filtered listing grid
* ``GET  /api/discover/categories``  — counts per category for the UI sidebar
* ``GET  /api/discover/{listing_id}``— detail
* ``POST /api/discover/{listing_id}/install`` — install via the
  ``install_via`` dispatcher

This file is the surface; install logic lives in
``augmentum/marketplace/install_dispatchers.py`` (community + provider
dispatchers) or delegates to ``TitleService`` (game-like installs that
existed before Discover).

Gated by ``settings.discover_enabled`` (default True). Disabled → 503
across all endpoints so the UI can render an empty state.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from augmentum.config import settings
from augmentum.marketplace.install_dispatchers import (
    get_dispatcher,
    get_uninstall_dispatcher,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


router = APIRouter(prefix="/api/discover", tags=["discover"])


# ── Helpers ──────────────────────────────────────────────────────────


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _gate(request: Request) -> JSONResponse | None:
    if not getattr(settings, "discover_enabled", True):
        return JSONResponse(
            {"error": "Discover is disabled"}, status_code=503,
        )
    if getattr(request.app.state, "marketplace_store", None) is None:
        return JSONResponse(
            {"error": "Catalog store not available"}, status_code=503,
        )
    return None


def _store(request: Request):
    return getattr(request.app.state, "marketplace_store", None)


def _title_service(request: Request):
    return getattr(request.app.state, "title_service", None)


# ── Routes ───────────────────────────────────────────────────────────


@router.get("/catalog")
async def list_catalog(
    request: Request,
    category: str = "",
    kind: str = "",
    publisher: str = "",
    featured: bool = False,
    q: str = "",
    limit: int = 50,
    offset: int = 0,
) -> JSONResponse:
    """List active listings with optional filters.

    Each listing carries an ``installed`` boolean populated from the
    per-user marketplace_installs audit (plus the install-wide
    managed_services check for provider listings — those are install-
    wide so any user sees them as installed when the service is on).
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    store = _store(request)
    listings = await store.list_for_discover(
        category=category or None,
        kind=kind or None,
        publisher=publisher or None,
        featured_only=bool(featured),
        search=q.strip(),
        limit=max(1, min(200, int(limit))),
        offset=max(0, int(offset)),
    )

    # Enrichment: which of these is installed for this user?
    listing_ids = [l.id for l in listings]
    per_user_installed = await store.installed_listing_ids_for_user(uid, listing_ids)
    # Install-wide provider services — enabled by any admin, every
    # user sees them as installed. Cheap lookup; one row per service.
    active_defs = await store.install_wide_active_service_definitions()

    # Add-on state comes from the Docker daemon, not from an install record
    # (see augmentum/addons/registry.py — the anchor IS the install record).
    # Fetched once for the whole page rather than per card.
    addon_states = {}
    if any(l.kind == "addon" for l in listings):
        try:
            from augmentum.addons.registry import list_states

            addon_states = await list_states(app_state=request.app.state)
        except Exception:  # noqa: BLE001 — enrichment is best-effort
            log.warning("discover_addon_states_failed", exc_info=True)

    out = []
    for l in listings:
        d = l.to_dict()
        if l.kind == "addon":
            _enrich_addon_listing(d, addon_states)
        elif l.kind == "provider_service":
            # listing id is mkt:provider:<service_id>; check the
            # underlying definition for an active managed row.
            svc_id = (l.install_payload or {}).get("service_id", "")
            d["installed"] = bool(svc_id and svc_id in active_defs)
        elif l.kind == "service":
            # Manifest services are install-wide (one shared container),
            # and installed-ness must track the managed_services row —
            # NOT the per-user install record, which goes stale the
            # moment an admin uninstalls (or the row would claim
            # "installed" forever for the installing user).
            await _enrich_service_listing(request, d, active_defs)
        elif l.id in per_user_installed:
            d["installed"] = True
        else:
            d["installed"] = False
        out.append(d)

    return JSONResponse({
        "listings": out,
        "count": len(out),
    })


def _enrich_addon_listing(d: dict, addon_states: dict) -> None:
    """Merge add-on catalog facts + live daemon state into a listing dict.

    The listing JSON carries presentation only; everything a user needs to
    decide with — build time, disk, what breaks without it, the license they
    must accept, where "Open" goes — lives in ``augmentum/addons/catalog.py``
    next to the recipe it describes, and is merged here. That keeps one
    source of truth for buildable facts while still letting the copy be
    edited without touching Python.

    ``installed`` is the presence of the image, per the daemon. There is no
    per-user install record for add-ons: a capability is either available to
    this instance or it isn't.
    """
    from augmentum.addons.catalog import addon_by_id

    addon_id = str((d.get("install_payload") or {}).get("addon_id") or "")
    spec = addon_by_id(addon_id)
    if spec is None:
        d["installed"] = False
        return

    state = addon_states.get(addon_id)
    d["installed"] = bool(state and state.installed)

    caps = dict(d.get("capabilities") or {})
    caps.update({
        "addon_id": spec.id,
        "capability": spec.capability,
        "provides": spec.provides,
        "surface": spec.surface,
        "disk_mb": spec.disk_mb,
        "build_minutes": spec.build_minutes,
        "license_notice": spec.license_notice,
        # Pinned build args are the add-on equivalent of a pinned image tag.
        # Surfaced so the card can show exactly which versions it will build
        # -- the user is the builder, so the user gets to see the recipe.
        "build_args": dict(spec.build_args),
        # Dependencies are implicit in the UI but honest in the payload: the
        # first streaming add-on installed also builds the shared base, which
        # is why its first install takes longer than the numbers on the card.
        "requires": list(spec.requires),
    })
    if state is not None:
        caps["anchored"] = state.anchored
        # Image present but held by nothing -- the precondition of the
        # 2026-07-25 sweep. Surfaced so the UI can offer to re-anchor rather
        # than waiting for a launch-time 404 months later.
        caps["at_risk"] = state.at_risk
    d["capabilities"] = caps


async def _enrich_service_listing(
    request: Request, d: dict, active_defs: set[str],
) -> None:
    """Attach truthful install state + front-door ports + system
    capabilities to a ``kind: "service"`` listing dict, in place.

    - ``installed`` mirrors the managed_services row (install-wide, one
      shared container), so it flips off on uninstall for EVERY user and
      never trusts a stale per-user install record.
    - ``capabilities.https_port`` / ``host_port`` / ``service_id`` come
      from the live runtime definition so the UI can build an "Open"
      front-door link — the same contract media-server cards use.
    - ``system.capabilities`` is built from the manifest's integration
      hooks — each hook's HookMeta provides the label, icon, and
      companion hint; live status is resolved per hook. This is what
      powers the OS-feel capability cards on the Discover home view.
    """
    svc = ((d.get("install_payload") or {}).get("service") or {})
    svc_id = str(svc.get("id") or "")
    d["installed"] = bool(svc_id and svc_id in active_defs)
    if not d["installed"]:
        return
    mgr = getattr(request.app.state, "service_manager", None)
    if mgr is None:
        return
    try:
        sd = mgr.get_definition(svc_id)
    except Exception:  # noqa: BLE001 — enrichment is best-effort
        sd = None
    if sd is None:
        return
    caps = dict(d.get("capabilities") or {})
    caps["service_id"] = svc_id
    if getattr(sd, "https_port", 0):
        caps["https_port"] = int(sd.https_port)
    if getattr(sd, "host_port", 0):
        caps["host_port"] = int(sd.host_port)

    # Gate URL: when a gate domain is configured, generic service apps are
    # published behind the access gate at <svc>.<gate_domain>:6443 — the
    # unbounded, signed-in door (no 6800-6809 port cap). Surface it so the UI
    # prefers it over the dedicated-port door. The gate listener is :6443
    # (caddy_front_door.GATE_LISTEN_PORT), shared across all subdomains.
    try:
        from augmentum.config import settings as _s
        gate_dom = (getattr(_s, "gate_domain", "") or "").strip().lower()
    except Exception:  # noqa: BLE001
        gate_dom = ""
    if gate_dom:
        caps["gate_url"] = f"https://{svc_id}.{gate_dom}:6443"

    # Update-available: the running container's image (managed_services row)
    # can lag the catalog manifest after a version bump. Surface the drift so
    # the manage sheet can offer a data-preserving Update instead of forcing a
    # uninstall/reinstall. Best-effort — never fail enrichment over this.
    try:
        installed_image = await mgr.installed_image(svc_id)
    except Exception:  # noqa: BLE001
        installed_image = ""
    target_image = str(svc.get("image") or "")
    if installed_image and target_image and installed_image != target_image:
        caps["update_available"] = True
        caps["installed_image"] = installed_image
        caps["target_image"] = target_image

    d["capabilities"] = caps

    # System capabilities — built asynchronously so DB calls don't block.
    conn = _resolve_db_conn(request)
    if conn is not None:
        try:
            d["system"] = await _build_system_capabilities(
                request, d, svc_id, conn,
            )
        except Exception:  # noqa: BLE001
            log.warning(
                "discover_system_capabilities_failed",
                listing_id=d.get("id", ""), exc_info=True,
            )
            d["system"] = {"capabilities": []}


def _resolve_db_conn(request: Request) -> Any:
    """Best-effort aiosqlite connection from app.state."""
    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm is not None else None
    return getattr(backend, "conn", None)


async def _build_system_capabilities(
    request: Request, d: dict, svc_id: str, conn: Any,
) -> dict[str, Any]:
    """Build the ``system.capabilities`` block for a service listing.

    Iterates the manifest's ``integration`` keys, resolving each to a
    capability row with label/icon/companion_hint from HookMeta, protocol
    from the manifest config, live status from per-hook resolvers, and
    connected/agent toggle state from config_json.

    Unknown hooks (no HookMeta) get a derived label + ⚙️ icon — no
    hardcoded per-hook lookup tables.
    """
    from augmentum.marketplace.hooks import KNOWN_INTEGRATION_HOOKS, HookMeta

    manifest = d.get("install_payload") or {}
    integration = manifest.get("integration") or {}
    if not isinstance(integration, dict) or not integration:
        return {"capabilities": []}

    # Read config_json for toggle state.
    config_json = {}
    try:
        mgr = getattr(request.app.state, "service_manager", None)
        if mgr is not None and hasattr(mgr, "read_config_json"):
            config_json = await mgr.read_config_json(svc_id) or {}
    except Exception:  # noqa: BLE001
        pass

    capabilities: list[dict[str, Any]] = []
    for hook_name, hook_cfg in integration.items():
        if not isinstance(hook_cfg, dict):
            hook_cfg = {}
        pair = KNOWN_INTEGRATION_HOOKS.get(hook_name)
        if pair is not None and len(pair) >= 3:
            meta: HookMeta = pair[2]
        else:
            # Unknown/community hook — derive a label from the name.
            meta = HookMeta(
                label=hook_name.replace("_", " ").title(),
                icon="⚙️",
                companion_hint="",
                status_provider="",
            )

        protocol = str(hook_cfg.get("protocol") or "")
        status = await _resolve_hook_status(
            conn, svc_id, meta.status_provider, config_json,
        )

        capabilities.append({
            "hook": hook_name,
            "label": meta.label,
            "icon": meta.icon,
            "protocol": protocol,
            "status": status,
            "companion_hint": meta.companion_hint,
            "connected": True,   # integration key present = connected
            "agent": False,      # Companion agent toggle — future
            # Install-time wiring (augmentum_backend) has no runtime toggle —
            # the UI renders it as an informational row instead of a switch.
            "toggleable": bool(getattr(meta, "toggleable", True)),
        })

    return {"capabilities": capabilities}


async def _resolve_hook_status(
    conn: Any, svc_id: str, status_provider: str, config_json: dict,
) -> str:
    """Resolve a live status string for a hook's status_provider key.

    Returns a human-readable status like "3 events today" or "connected".
    Each provider is a cheap DB query — best-effort, never raises.
    """
    try:
        if status_provider == "calendar":
            return await _calendar_status(conn, svc_id)
        if status_provider == "subsonic":
            return await _subsonic_status(conn, svc_id)
        if status_provider == "webhook":
            return _webhook_status(config_json)
        if status_provider == "media_connect":
            return await _media_connect_status(conn, svc_id)
        if status_provider == "augmentum_backend":
            # Install-time wiring — always "on" once installed.
            return "Using your models"
    except Exception:  # noqa: BLE001
        pass
    return "connected"


async def _calendar_status(conn: Any, svc_id: str) -> str:
    """Count events today for the given CalDAV service."""
    import datetime as _dt
    today = _dt.date.today().isoformat()
    tomorrow = (_dt.date.today() + _dt.timedelta(days=1)).isoformat()
    cur = await conn.execute(
        """SELECT COUNT(*) FROM calendar_events
           WHERE service_id = ?
             AND start_dt < ?
             AND (end_dt > ? OR end_dt = start_dt)""",
        (svc_id, tomorrow, today),
    )
    row = await cur.fetchone()
    await cur.close()
    n = int(row[0]) if row and row[0] is not None else 0
    if n == 0:
        return "0 events today"
    return f"{n} event{'s' if n != 1 else ''} today"


async def _subsonic_status(conn: Any, svc_id: str) -> str:
    """Check media server connection status for a Subsonic server."""
    cur = await conn.execute(
        """SELECT status FROM user_media_servers
           WHERE provider = ? AND status = 'ok'
           LIMIT 1""",
        (svc_id,),
    )
    row = await cur.fetchone()
    await cur.close()
    if row and row[0] == "ok":
        return "connected"
    return "untested"


def _webhook_status(config_json: dict) -> str:
    """Check if webhook notifications are enabled."""
    if config_json.get("webhook_enabled"):
        return "active"
    return "inactive"


async def _media_connect_status(conn: Any, svc_id: str) -> str:
    """Check media server connection status."""
    cur = await conn.execute(
        """SELECT status FROM user_media_servers
           WHERE provider = ? AND status = 'ok'
           LIMIT 1""",
        (svc_id,),
    )
    row = await cur.fetchone()
    await cur.close()
    if row and row[0] == "ok":
        return "connected"
    return "untested"


@router.get("/categories")
async def list_categories(request: Request) -> JSONResponse:
    """Return categories + counts for the Discover sidebar / filter chips."""
    if (gate := _gate(request)) is not None:
        return gate
    if not _user_id(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    conn = _store(request)._conn  # noqa: SLF001
    cursor = await conn.execute(
        "SELECT category, COUNT(*) FROM marketplace_listings "
        "WHERE delisted_at IS NULL GROUP BY category ORDER BY category",
    )
    rows = await cursor.fetchall()
    return JSONResponse({
        "categories": [{"id": r[0], "count": int(r[1])} for r in rows],
    })


# ── Community stores (service-OS phase 4) ───────────────────────────
# Declared BEFORE the /{listing_id} catch-all or "stores" would be
# swallowed as a listing id.


@router.get("/stores")
async def list_community_stores(request: Request) -> JSONResponse:
    """Registered community app stores (admin surface)."""
    from augmentum.auth.guards import is_admin
    if (gate := _gate(request)) is not None:
        return gate
    if not is_admin(request):
        return JSONResponse({"error": "Admin only"}, status_code=403)
    from augmentum.marketplace.loaders.stores import list_stores
    settings_store = getattr(request.app.state, "settings_store", None)
    if settings_store is None:
        return JSONResponse({"error": "Settings unavailable"}, status_code=503)
    return JSONResponse({"stores": await list_stores(settings_store)})


@router.post("/stores")
async def add_community_store(request: Request) -> JSONResponse:
    """Add a store by URL and sync it immediately.

    Adding a store is an install-wide trust decision — admin only, and
    the response carries the first sync's honest counts so the admin
    sees exactly what came in (and what was rejected by the gate).
    """
    from augmentum.auth.guards import is_admin
    if (gate := _gate(request)) is not None:
        return gate
    if not is_admin(request):
        return JSONResponse({"error": "Admin only"}, status_code=403)
    from augmentum.marketplace.loaders.stores import add_store, sync_store
    settings_store = getattr(request.app.state, "settings_store", None)
    http = getattr(request.app.state, "http_client", None)
    if settings_store is None or http is None:
        return JSONResponse({"error": "Not available"}, status_code=503)
    body = await request.json()
    try:
        entry = await add_store(
            settings_store,
            url=str(body.get("url") or ""),
            name=str(body.get("name") or ""),
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    try:
        stats = await sync_store(_store(request), http, entry)
    except Exception as exc:  # noqa: BLE001 — keep the registration, report the sync
        return JSONResponse({
            "store": entry,
            "sync_error": str(exc)[:300],
        }, status_code=207)
    return JSONResponse({"store": entry, "sync": stats})


@router.post("/stores/{slug}/sync")
async def sync_community_store(request: Request, slug: str) -> JSONResponse:
    from augmentum.auth.guards import is_admin
    if (gate := _gate(request)) is not None:
        return gate
    if not is_admin(request):
        return JSONResponse({"error": "Admin only"}, status_code=403)
    from augmentum.marketplace.loaders.stores import list_stores, sync_store
    settings_store = getattr(request.app.state, "settings_store", None)
    http = getattr(request.app.state, "http_client", None)
    if settings_store is None or http is None:
        return JSONResponse({"error": "Not available"}, status_code=503)
    entry = next(
        (e for e in await list_stores(settings_store) if e.get("slug") == slug),
        None,
    )
    if entry is None:
        return JSONResponse({"error": "Unknown store"}, status_code=404)
    try:
        stats = await sync_store(_store(request), http, entry)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)[:300]}, status_code=502)
    return JSONResponse({"sync": stats})


@router.delete("/stores/{slug}")
async def delete_community_store(request: Request, slug: str) -> JSONResponse:
    """Remove a store + soft-delist its listings. Installed apps stay
    installed — removal is a catalog operation, never an uninstall."""
    from augmentum.auth.guards import is_admin
    if (gate := _gate(request)) is not None:
        return gate
    if not is_admin(request):
        return JSONResponse({"error": "Admin only"}, status_code=403)
    from augmentum.marketplace.loaders.stores import remove_store
    settings_store = getattr(request.app.state, "settings_store", None)
    if settings_store is None:
        return JSONResponse({"error": "Settings unavailable"}, status_code=503)
    try:
        delisted = await remove_store(settings_store, _store(request), slug)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    return JSONResponse({"removed": slug, "delisted": delisted})


@router.get("/{listing_id}")
async def get_listing(request: Request, listing_id: str) -> JSONResponse:
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    store = _store(request)
    listing = await store.get(listing_id)
    if not listing or listing.delisted_at:
        return JSONResponse({"error": "Listing not found"}, status_code=404)

    out = listing.to_dict()
    per_user_installed = await store.installed_listing_ids_for_user(uid, [listing.id])
    if listing.kind == "provider_service":
        active_defs = await store.install_wide_active_service_definitions()
        svc_id = (listing.install_payload or {}).get("service_id", "")
        out["installed"] = bool(svc_id and svc_id in active_defs)
        # Live requirement satisfaction so the card knows whether to prompt
        # for a gated token before install (static requirements ride in
        # metadata; whether the secret is actually set is dynamic).
        reqs = ((out.get("metadata") or {}).get("requirements")) or {}
        token_req = reqs.get("token") if isinstance(reqs, dict) else None
        if token_req:
            from augmentum.config import settings as _settings
            key = str(token_req.get("setting") or "").strip()
            token_set = bool(key and str(getattr(_settings, key, "") or "").strip())
            out["preflight"] = {"token_set": token_set}
    elif listing.kind == "addon":
        try:
            from augmentum.addons.registry import list_states

            _enrich_addon_listing(
                out, await list_states(app_state=request.app.state),
            )
        except Exception:  # noqa: BLE001 — enrichment is best-effort
            log.warning("discover_addon_states_failed", exc_info=True)
            out["installed"] = False
    elif listing.kind == "service":
        active_defs = await store.install_wide_active_service_definitions()
        await _enrich_service_listing(request, out, active_defs)
    elif listing.id in per_user_installed:
        out["installed"] = True
    else:
        out["installed"] = False
    return JSONResponse({"listing": out})


class DiscoverInstallRequest(BaseModel):
    """Body for ``POST /api/discover/{listing_id}/install``."""
    confirm: bool = Field(
        True,
        description="Set false to dry-run (validate but don't install).",
    )
    options: dict[str, Any] = Field(default_factory=dict)


@router.post("/{listing_id}/install")
async def install_listing(
    request: Request, listing_id: str,
) -> JSONResponse:
    """Install a listing — routes to the appropriate dispatcher."""
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    listing = await _store(request).get(listing_id)
    if not listing or listing.delisted_at:
        return JSONResponse({"error": "Listing not found"}, status_code=404)

    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        req = DiscoverInstallRequest(**(body or {}))
    except Exception as exc:
        return JSONResponse(
            {"error": f"Invalid request body: {exc}"}, status_code=400,
        )

    if not req.confirm:
        return JSONResponse(
            {"dry_run": True, "would_install": listing.id,
             "kind": listing.kind, "install_via": listing.install_via},
        )

    dispatcher = get_dispatcher(listing.install_via)
    if dispatcher is not None:
        # Community + provider-service paths run here. The dispatcher
        # owns its own auth check (admin gates for provider/power/
        # knowledge installs).
        # Forward the request's install options to the dispatcher under a
        # reserved key (additive — dispatchers that don't need it ignore
        # it). Used by the media-server dispatcher to receive the user's
        # external media-library host path.
        artifact = {**(listing.install_payload or {}), "_install_options": req.options}
        try:
            resource_id = await dispatcher(
                request, artifact, uid,
            )
        except Exception as exc:
            # HTTPException short-circuits FastAPI; anything else bubbles
            # up as a 500 with the underlying message so the UI can show
            # something useful instead of a generic ASGI error.
            from fastapi import HTTPException as _HE
            if isinstance(exc, _HE):
                raise
            log.warning(
                "discover_install_failed",
                listing_id=listing.id, install_via=listing.install_via,
                error=str(exc),
            )
            return JSONResponse(
                {"error": str(exc)}, status_code=500,
            )
        store = _store(request)
        await store.increment_install_count(listing.id)
        await store.record_install(
            user_id=uid, listing_id=listing.id,
            install_via=listing.install_via, kind=listing.kind,
            resource_id=str(resource_id or ""),
        )
        return JSONResponse({
            "installed": True,
            "listing_id": listing.id,
            "kind": listing.kind,
            # install_via lets the UI pick the right post-install surface. The
            # staged path ("service_staged") returns a JOB id in resource_id, so
            # the card polls /api/jobs/{id} for staged progress instead of the
            # service status endpoint.
            "install_via": listing.install_via,
            "resource_id": resource_id,
            # Staged installs return a JOB id in resource_id and drive the
            # staged-progress card (Preparing → Downloading → Starting →
            # Warming up → Ready). Provider services now use the same
            # background-job path as service_staged for a uniform install UX.
            "staged": (
                listing.install_via == "service_staged"
                or listing.kind == "provider_service"
                # Add-on installs are BUILDS (minutes to tens of minutes),
                # so they take the same staged-progress path rather than
                # leaving a spinner up for half an hour.
                or listing.install_via == "addon_build"
            ),
        }, status_code=201)

    # Fallback — Source-backed installs (js13k, agsp-profile, internal,
    # marketplace). Delegate to TitleService.import_title which already
    # routes by source_id and uses the listing's install_payload to
    # forward args. Same path as /api/titles/marketplace/{id}/install.
    svc = _title_service(request)
    if svc is None:
        return JSONResponse(
            {"error": "Title service unavailable for non-dispatcher installs"},
            status_code=503,
        )
    try:
        manifest = await svc.import_title(
            user_id=uid,
            source_id="marketplace",
            manifest_data={"listing_id": listing.id},
        )
    except Exception as exc:
        log.warning(
            "discover_title_install_failed",
            listing_id=listing.id, error=str(exc),
        )
        return JSONResponse({"error": str(exc)}, status_code=400)
    store = _store(request)
    await store.increment_install_count(listing.id)
    await store.record_install(
        user_id=uid, listing_id=listing.id,
        install_via=listing.install_via, kind=listing.kind,
        resource_id=str(getattr(manifest, "id", "") or ""),
    )
    return JSONResponse({
        "installed": True,
        "listing_id": listing.id,
        "kind": listing.kind,
        "resource_id": getattr(manifest, "id", ""),
        "title": manifest.to_dict() if hasattr(manifest, "to_dict") else {},
    }, status_code=201)


@router.get("/{listing_id}/mem-limit")
async def get_listing_mem_limit(
    request: Request, listing_id: str,
) -> JSONResponse:
    """Current host-RAM ceiling for an installed service. ``""`` = unlimited.

    Read-only, so it is not admin-gated: the sheet shows the value to anyone
    who can see the app; only changing it requires admin.
    """
    if (gate := _gate(request)) is not None:
        return gate
    if not _user_id(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    listing = await _store(request).get(listing_id)
    if not listing or listing.delisted_at:
        return JSONResponse({"error": "Listing not found"}, status_code=404)
    service_id = str(((listing.install_payload or {}).get("service") or {}).get("id") or "")
    mgr = getattr(request.app.state, "service_manager", None)
    if not service_id or mgr is None:
        return JSONResponse({"mem_limit": "", "service_id": service_id})
    try:
        cfg = await mgr.read_config_json(service_id)
        current = str(cfg.get("mem_limit") or "")
    except Exception:
        log.warning("service_mem_limit_read_failed", service_id=service_id,
                    exc_info=True)
        current = ""
    return JSONResponse({"mem_limit": current, "service_id": service_id})


@router.post("/{listing_id}/mem-limit")
async def set_listing_mem_limit(
    request: Request, listing_id: str,
) -> JSONResponse:
    """Set (or clear) an installed service's host-RAM ceiling.

    Augmentum deliberately ships NO default ceiling for third-party services
    (see ``providers/manager.py::_resolve_mem_limit``): how much a service
    wants depends on what it's used for and by how many people, which only the
    operator knows. This is how they say so.

    Body: ``{"mem_limit": "2g"}``; empty string clears it. The container is
    recreated so the new ceiling takes effect — brief downtime, no data loss.
    """
    from augmentum.auth.guards import is_admin
    if (gate := _gate(request)) is not None:
        return gate
    if not _user_id(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not is_admin(request):
        return JSONResponse(
            {"error": "Changing memory limits is admin-only"}, status_code=403,
        )

    listing = await _store(request).get(listing_id)
    if not listing or listing.delisted_at:
        return JSONResponse({"error": "Listing not found"}, status_code=404)
    if listing.install_via != "service_manifest":
        return JSONResponse(
            {"error": "Only service apps have a memory limit"}, status_code=400,
        )

    service_id = str(((listing.install_payload or {}).get("service") or {}).get("id") or "")
    if not service_id:
        return JSONResponse({"error": "Listing has no service id"}, status_code=400)

    mgr = getattr(request.app.state, "service_manager", None)
    if mgr is None:
        return JSONResponse(
            {"error": "Service manager unavailable"}, status_code=503,
        )

    try:
        body = await request.json()
    except Exception:
        body = {}
    raw = str((body or {}).get("mem_limit") or "").strip()

    try:
        ms = await mgr.set_mem_limit(service_id, raw)
    except ValueError as exc:
        # A typo the user can fix — say what was wrong, don't persist it.
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        log.warning(
            "service_mem_limit_apply_failed",
            service_id=service_id, limit=raw, exc_info=True,
        )
        return JSONResponse(
            {"error": f"Couldn't apply the limit: {exc}"}, status_code=500,
        )

    return JSONResponse({
        "ok": True,
        "service_id": service_id,
        "mem_limit": raw,
        # False when the service wasn't running: the limit is saved and will
        # apply on next start. The UI says so rather than implying a restart
        # happened.
        "recreated": ms is not None,
    })


@router.post("/{listing_id}/update")
async def update_listing(
    request: Request, listing_id: str,
) -> JSONResponse:
    """Update an installed manifest service to the catalog's current pinned
    image (a version bump), preserving its data volumes.

    Admin-only and ``service_manifest`` only — the container is recreated on
    the new image, so it has install-wide side effects (image pull, brief
    downtime) but never touches named data volumes. Idempotent when already
    current: recreates on the same image (harmless) rather than erroring.
    """
    from augmentum.auth.guards import is_admin
    if (gate := _gate(request)) is not None:
        return gate
    if not _user_id(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not is_admin(request):
        return JSONResponse(
            {"error": "Service update is admin-only"}, status_code=403,
        )

    listing = await _store(request).get(listing_id)
    if not listing or listing.delisted_at:
        return JSONResponse({"error": "Listing not found"}, status_code=404)
    if listing.install_via != "service_manifest":
        return JSONResponse(
            {"error": "Only service apps can be updated"}, status_code=400,
        )

    from dataclasses import replace as _dc_replace

    from augmentum.marketplace.manifest import (
        ManifestError,
        parse_manifest,
        to_service_definition,
    )
    try:
        manifest = parse_manifest(listing.install_payload or {})
    except ManifestError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    mgr = getattr(request.app.state, "service_manager", None)
    if mgr is None:
        return JSONResponse(
            {"error": "Service manager unavailable"}, status_code=503,
        )

    active = await _store(request).install_wide_active_service_definitions()
    if manifest.service_id not in active:
        return JSONResponse(
            {"error": f"{manifest.name} is not installed"}, status_code=409,
        )

    # Preserve the front-door port allocated at install (persisted in
    # config_json) — a fresh definition would orphan the caddy snippet.
    new_def = to_service_definition(manifest)
    try:
        cfg = await mgr.read_config_json(manifest.service_id)
        https_port = int(cfg.get("https_port") or 0)
    except Exception:  # noqa: BLE001
        https_port = 0
    if https_port:
        new_def = _dc_replace(new_def, https_port=https_port)

    try:
        await mgr.update_service(manifest.service_id, new_def)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "discover_update_failed",
            listing_id=listing.id, service_id=manifest.service_id,
            error=str(exc),
        )
        return JSONResponse({"error": str(exc)}, status_code=500)

    log.info(
        "discover_service_updated",
        listing_id=listing.id, service_id=manifest.service_id,
        image=manifest.image,
    )
    return JSONResponse({
        "updated": True,
        "listing_id": listing.id,
        "service_id": manifest.service_id,
        "image": manifest.image,
    })


@router.delete("/{listing_id}/install")
async def uninstall_listing(
    request: Request, listing_id: str,
) -> JSONResponse:
    """Uninstall a listing — the inverse of ``install_listing``.

    Runs the install_via's teardown dispatcher (if any) to drop the backing
    resource — for media servers that removes the connection + cached
    library and stops the shared container — then clears the marketplace
    install record so the Discover card flips back to 'Install'. Because a
    media-server teardown is install-wide (the container is shared), the
    install record is cleared for EVERY user of that listing; per-user
    kinds clear only the caller. The teardown dispatcher owns its own auth
    (media-server uninstall is admin-only, symmetric with install).
    """
    if (gate := _gate(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    listing = await _store(request).get(listing_id)
    if not listing or listing.delisted_at:
        return JSONResponse({"error": "Listing not found"}, status_code=404)

    install_wide = False
    teardown = get_uninstall_dispatcher(listing.install_via)
    result: dict = {}
    if teardown is not None:
        artifact = {**(listing.install_payload or {})}
        try:
            result = await teardown(request, artifact, uid) or {}
        except Exception as exc:
            from fastapi import HTTPException as _HE
            if isinstance(exc, _HE):
                raise
            log.warning(
                "discover_uninstall_failed",
                listing_id=listing.id, install_via=listing.install_via,
                error=str(exc),
            )
            return JSONResponse({"error": str(exc)}, status_code=500)
        # A stopped shared container means nobody has it anymore.
        install_wide = bool(result.get("service_stopped"))
        # Add-ons are instance-wide by nature: removing the image removes the
        # capability from Augmentum for everyone, so no per-user install
        # record may survive claiming otherwise.
        if listing.kind == "addon":
            install_wide = True

    store = _store(request)
    cleared = await store.mark_uninstalled(
        listing.id, user_id="" if install_wide else uid,
    )
    return JSONResponse({
        "uninstalled": True,
        "listing_id": listing.id,
        "kind": listing.kind,
        "install_records_cleared": cleared,
        **result,
    })
