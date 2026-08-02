"""Provider management API routes."""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from augmentum.auth.guards import is_admin, require_admin
from augmentum.models.provider_profiles import (
    get_profile,
    get_profile_for_url,
)
from augmentum.models.provider_registry import create_backend_from_profile
from augmentum.proxy import system_events
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.provider_store import ProviderConfig, ProviderStore
from augmentum.utils.logging import get_logger
from augmentum.utils.safe_http import SafeHttpError, check_ssrf_user_url
from augmentum.utils.secrets import sanitize_error_detail

log = get_logger(__name__)

router = APIRouter(prefix="/api/providers", tags=["providers"])

_RESERVED_KEYS = {"ollama", "openai", "llamacpp", "claude", "gemini", "engine"}


class ProviderCreateRequest(BaseModel):
    id: str
    name: str
    base_url: str
    api_key: str | None = None
    provider_type: str = "openai"
    profile_id: str = ""


class ProviderUpdateRequest(BaseModel):
    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    is_enabled: bool | None = None
    profile_id: str | None = None


class ProbeRequest(BaseModel):
    base_url: str
    api_key: str | None = None
    provider_type: str = "openai"


class ShareRequest(BaseModel):
    shared: bool


def _user_id(request: Request) -> str:
    """Authenticated caller's id, or '' when unauthenticated."""
    user = request.scope.get("user")
    return getattr(user, "id", "") if user else ""


async def _guard_provider_url(base_url: str, *, admin: bool) -> JSONResponse | None:
    """SSRF gate for provider URLs supplied by NON-admin users.

    Admin-added providers are trusted infrastructure and frequently point at
    LAN endpoints (Ollama/vLLM on private IPs), so they keep the unrestricted
    path — matching the long-standing assumption in ``probe_provider``. A
    non-admin base_url is user-supplied, so it gets strict external-only SSRF
    filtering (no loopback / private / link-local / metadata) to stop a tenant
    from turning provider-add into an internal-network probe. Returns a 400
    JSONResponse to short-circuit the handler, or ``None`` when allowed.
    """
    if admin:
        return None
    try:
        await check_ssrf_user_url(base_url, mode="external")
    except SafeHttpError as exc:
        return JSONResponse(
            {"error": f"Provider URL not allowed: {sanitize_error_detail(str(exc))}"},
            status_code=400,
        )
    return None


def _get_store(request: Request) -> ProviderStore | None:
    """Extract ProviderStore from app state if SQLite is available."""
    state_manager = request.app.state.state_manager
    if isinstance(state_manager.backend, SQLiteBackend):
        return ProviderStore(state_manager.backend.conn)
    return None


async def _builtin_providers(request: Request) -> list[dict]:
    """List built-in providers configured via environment variables.

    Async because ``manager.status()`` shells out to nvidia-smi (5s
    timeout) and we don't want a provider-list call to block the event
    loop on a slow GPU probe.
    """
    registry = request.app.state.provider_registry
    result = []
    for key in ("ollama", "openai", "llamacpp"):
        if key in registry.backends:
            result.append({
                "id": key,
                "name": key.capitalize(),
                "type": "builtin",
                "is_enabled": True,
            })
    # Managed engine (engine v2)
    if "engine" in registry.backends:
        manager = getattr(request.app.state, "llama_manager", None)
        entry: dict = {
            "id": "engine",
            "name": "Built-in Engine",
            "type": "builtin",
            "is_enabled": True,
        }
        if manager:
            entry["base_url"] = manager.base_url
            entry["model_dirs"] = manager.model_dirs
            status = await asyncio.to_thread(manager.status)
            entry["state"] = status.get("state", "idle")
            entry["model_id"] = status.get("model_id", "")
        result.append(entry)
    # Optional vLLM safetensors engine (installed from Discover). Registered as
    # the "vllm" backend; surface it here so it appears in Settings → Providers
    # like the other local engines.
    if "vllm" in registry.backends:
        import os as _os

        from augmentum.config import settings as _s
        result.append({
            "id": "vllm",
            "name": "vLLM Engine",
            "type": "builtin",
            "is_enabled": True,
            "base_url": _s.vllm_base_url or _os.environ.get("AUGMENTUM_VLLM_BASE_URL", ""),
        })
    return result


@router.get("/profiles")
async def list_provider_profiles():
    """List built-in provider profiles (connection templates)."""
    from augmentum.models.provider_profiles import list_profiles

    return [
        {
            "id": p.id,
            "name": p.name,
            "base_url": p.base_url,
            "supports_tools": p.supports_tools,
            "supports_vision": p.supports_vision,
            "supports_thinking": p.supports_thinking,
            "notes": p.notes,
        }
        for p in list_profiles()
    ]


@router.get("/")
async def list_providers(request: Request) -> JSONResponse:
    """List all providers (built-in + user-configured)."""
    providers = await _builtin_providers(request)

    store = _get_store(request)
    if store:
        uid = _user_id(request)
        admin = is_admin(request)
        # Admin sees every provider (to manage all of them); a normal user
        # sees global + shared + their own private ones.
        db_providers = await store.list_providers(
            visible_to=None if admin else uid,
        )
        for p in db_providers:
            is_owner = bool(p.owner_user_id) and p.owner_user_id == uid
            providers.append({
                "id": p.id,
                "name": p.name,
                "base_url": p.base_url,
                "has_api_key": bool(p.api_key),
                "provider_type": p.provider_type,
                "is_enabled": p.is_enabled,
                "is_default": p.is_default,
                "type": "user",
                "created_at": p.created_at,
                "updated_at": p.updated_at,
                # Sharing metadata (migration 305) — drives the UI badges +
                # controls. ``owner_user_id`` is intentionally NOT echoed to
                # avoid leaking user ids to other admins; the booleans below
                # carry everything the UI needs.
                "shared": p.shared,
                "is_owner": is_owner,
                "can_manage": admin or is_owner,  # edit/remove
                "can_share": admin,               # only admins flip sharing
            })

    return JSONResponse({"providers": providers})


@router.post("/")
async def create_provider(body: ProviderCreateRequest, request: Request) -> JSONResponse:
    """Add a provider.

    Admins add **shared** instance-wide infrastructure (visible to every
    user). Any other authenticated user may add a **private** provider
    visible only to themselves (migration 305). Sharing a private provider
    is admin-only, via ``PUT /{id}/share``.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    admin = is_admin(request)

    if body.id in _RESERVED_KEYS:
        return JSONResponse(
            {"error": f"ID '{body.id}' is reserved for built-in backends"},
            status_code=400,
        )

    if (blocked := await _guard_provider_url(body.base_url, admin=admin)) is not None:
        return blocked

    # Non-admin providers get a namespaced backend key so two users can pick
    # the same friendly id without colliding on the process-global registry,
    # and so a 409 never reveals the existence of another user's private
    # provider. Admin providers keep the raw id (shared infra, one namespace).
    pid = body.id if admin else f"u_{uid[:8]}_{body.id}"

    registry = request.app.state.provider_registry
    if pid in registry.backends:
        return JSONResponse(
            {"error": f"Provider '{body.id}' already exists"},
            status_code=409,
        )

    store = _get_store(request)
    if not store:
        return JSONResponse(
            {"error": "SQLite backend required for provider management"},
            status_code=503,
        )

    # Policy: admin-added = shared/global; user-added = private to the owner.
    shared = admin
    owner_user_id = "" if admin else uid

    # Resolve profile: explicit selection wins, else infer from URL so a
    # user pasting NVIDIA's URL gets the NVIDIA profile attached even
    # without picking it from the dropdown.
    profile = get_profile(body.profile_id) if body.profile_id else None
    if profile is None:
        profile = get_profile_for_url(body.base_url)
    resolved_profile_id = profile.id if profile else ""

    config = ProviderConfig(
        id=pid,
        name=body.name,
        base_url=body.base_url,
        api_key=body.api_key,
        provider_type=body.provider_type,
        profile_id=resolved_profile_id,
        owner_user_id=owner_user_id,
        shared=shared,
    )
    saved = await store.create_provider(config)

    http_client = request.app.state.http_client
    backend = create_backend_from_profile(
        profile,
        api_key=saved.api_key or "",
        http_client=http_client,
        provider_type=body.provider_type,
        base_url=saved.base_url,
    )
    registry.register_backend(saved.id, backend)
    registry.set_provider_meta(saved.id, owner_user_id, shared)

    # Refresh model map
    await registry.refresh_model_map(force=True)

    log.info(
        "provider_created", id=saved.id, name=saved.name,
        owner=owner_user_id or "(global)", shared=shared,
    )
    # Broadcast only for shared/global providers. A private provider must
    # not leak its name/url onto other users' SSE streams — scope the event
    # to the owner (their own client also refreshes on the POST response).
    if shared:
        system_events.publish("providers.added", {
            "id": saved.id,
            "name": saved.name,
            "base_url": saved.base_url,
            "profile_id": saved.profile_id,
        })
    else:
        system_events.publish(
            "providers.added",
            {"id": saved.id, "name": saved.name},
            user_id=owner_user_id,
        )
    return JSONResponse(
        {"id": saved.id, "name": saved.name, "status": "created"},
        status_code=201,
    )


@router.put("/{provider_id}")
async def update_provider(provider_id: str, body: ProviderUpdateRequest, request: Request) -> JSONResponse:
    """Update a provider. Admins may edit any; a user may edit only their
    own private provider (migration 305). Sharing is toggled separately via
    ``PUT /{id}/share`` (admin-only)."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    admin = is_admin(request)
    if provider_id in _RESERVED_KEYS:
        return JSONResponse(
            {"error": "Cannot modify built-in providers"},
            status_code=400,
        )

    store = _get_store(request)
    if not store:
        return JSONResponse(
            {"error": "SQLite backend required for provider management"},
            status_code=503,
        )

    existing = await store.get_provider(provider_id)
    if existing is None:
        return JSONResponse({"error": "Provider not found"}, status_code=404)
    # Ownership: admins edit anything; a user only their own private provider.
    if not admin and existing.owner_user_id != uid:
        return JSONResponse(
            {"error": "Not permitted to modify this provider"},
            status_code=403,
        )

    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if not fields:
        return JSONResponse({"error": "No fields to update"}, status_code=400)

    # If the URL changed and the caller didn't supply a profile_id, re-run
    # URL-based matching against the new URL. Without this, editing a
    # provider's base_url could leave it pointing at NVIDIA with the old
    # (or no) profile attached.
    if "base_url" in fields and "profile_id" not in fields:
        inferred = get_profile_for_url(fields["base_url"])
        fields["profile_id"] = inferred.id if inferred else ""

    updated = await store.update_provider(provider_id, **fields)
    if not updated:
        return JSONResponse({"error": "Provider not found"}, status_code=404)

    # Re-register backend with the resolved profile so post-processing
    # rules (NVIDIA "semi", DeepSeek "semi", Perplexity "strict") apply.
    registry = request.app.state.provider_registry
    http_client = request.app.state.http_client

    if updated.is_enabled:
        profile = get_profile(updated.profile_id) if updated.profile_id else None
        if profile is None:
            profile = get_profile_for_url(updated.base_url)

        backend = create_backend_from_profile(
            profile,
            api_key=updated.api_key or "",
            http_client=http_client,
            provider_type=updated.provider_type,
            base_url=updated.base_url,
        )
        if provider_id in registry.backends:
            with contextlib.suppress(ValueError):
                registry.unregister_backend(provider_id)
        registry.register_backend(provider_id, backend)
    else:
        if provider_id in registry.backends:
            with contextlib.suppress(ValueError):
                registry.unregister_backend(provider_id)

    # Re-assert visibility metadata (unregister above dropped it; a
    # disabled provider keeps its meta so re-enabling stays correct).
    registry.set_provider_meta(provider_id, updated.owner_user_id, updated.shared)

    await registry.refresh_model_map(force=True)

    log.info("provider_updated", id=provider_id)
    # Scope the event: a private provider's change only reaches its owner.
    _evt = {"id": provider_id, "is_enabled": updated.is_enabled}
    if updated.shared or not updated.owner_user_id:
        system_events.publish("providers.updated", _evt)
    else:
        system_events.publish("providers.updated", _evt, user_id=updated.owner_user_id)
    return JSONResponse({"id": provider_id, "status": "updated"})


@router.delete("/{provider_id}")
async def delete_provider(provider_id: str, request: Request) -> JSONResponse:
    """Delete a provider. Admins may delete any; a user may delete only
    their own private provider (migration 305)."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    admin = is_admin(request)
    if provider_id in _RESERVED_KEYS:
        return JSONResponse(
            {"error": "Cannot delete built-in providers"},
            status_code=400,
        )

    store = _get_store(request)
    if not store:
        return JSONResponse(
            {"error": "SQLite backend required for provider management"},
            status_code=503,
        )

    existing = await store.get_provider(provider_id)
    if existing is None:
        return JSONResponse({"error": "Provider not found"}, status_code=404)
    if not admin and existing.owner_user_id != uid:
        return JSONResponse(
            {"error": "Not permitted to delete this provider"},
            status_code=403,
        )

    deleted = await store.delete_provider(provider_id)
    if not deleted:
        return JSONResponse({"error": "Provider not found"}, status_code=404)

    # Unregister backend + drop its visibility metadata
    registry = request.app.state.provider_registry
    if provider_id in registry.backends:
        with contextlib.suppress(ValueError):
            registry.unregister_backend(provider_id)
    registry.clear_provider_meta(provider_id)

    await registry.refresh_model_map(force=True)

    log.info("provider_deleted", id=provider_id)
    # Scope the event: a private provider's deletion only reaches its owner.
    if existing.shared or not existing.owner_user_id:
        system_events.publish("providers.deleted", {"id": provider_id})
    else:
        system_events.publish(
            "providers.deleted", {"id": provider_id},
            user_id=existing.owner_user_id,
        )
    return JSONResponse({"status": "deleted"})


@router.put("/{provider_id}/share")
async def share_provider(provider_id: str, body: ShareRequest, request: Request) -> JSONResponse:
    """Toggle whether a provider is shared with every user on the instance.

    Admin-only — only an admin may expose (or un-expose) a provider
    instance-wide. Flipping ``shared`` on a private, user-owned provider
    makes it visible to all while leaving its ``owner_user_id`` intact.
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    if provider_id in _RESERVED_KEYS:
        return JSONResponse(
            {"error": "Built-in providers are always shared"},
            status_code=400,
        )

    store = _get_store(request)
    if not store:
        return JSONResponse(
            {"error": "SQLite backend required for provider management"},
            status_code=503,
        )

    existing = await store.get_provider(provider_id)
    if existing is None:
        return JSONResponse({"error": "Provider not found"}, status_code=404)

    fields: dict = {"shared": body.shared}
    # Unsharing an ownerless provider (admin-created / pre-305 backfill)
    # stamps the acting admin as owner. Without this, "private" on an
    # owner='' row either meant "still visible to everyone" (the original
    # leak) or "visible to no one at all" (post-fix predicate) — stamping
    # makes it mean what the click says: private to the admin who clicked.
    if not body.shared and not existing.owner_user_id:
        fields["owner_user_id"] = _user_id(request)

    updated = await store.update_provider(provider_id, **fields)
    if not updated:
        return JSONResponse({"error": "Provider not found"}, status_code=404)

    registry = request.app.state.provider_registry
    registry.set_provider_meta(provider_id, updated.owner_user_id, updated.shared)

    log.info("provider_share_toggled", id=provider_id, shared=updated.shared)
    # A share-state change is inherently instance-wide news (it just became
    # visible or hidden for other users), so broadcast unconditionally.
    system_events.publish("providers.updated", {
        "id": provider_id,
        "shared": updated.shared,
    })
    return JSONResponse({"id": provider_id, "shared": updated.shared, "status": "updated"})


@router.post("/{provider_id}/test")
async def test_provider(provider_id: str, request: Request) -> JSONResponse:
    """Test an existing provider by listing its models."""
    registry = request.app.state.provider_registry
    backend = registry.get_backend(provider_id)
    if not backend:
        return JSONResponse({"error": "Provider not found"}, status_code=404)

    try:
        models = await backend.list_models()
        return JSONResponse({
            "status": "ok",
            "models": [{"name": m.name, "model": m.model} for m in models],
        })
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "error": sanitize_error_detail(str(exc))},
            status_code=502,
        )


@router.post("/probe")
async def probe_provider(body: ProbeRequest, request: Request) -> JSONResponse:
    """Test a URL before saving — powers the 'Test Connection' button.

    Admins skip SSRF filtering: their provider URLs are trusted infrastructure
    endpoints (Ollama, LM Studio, vLLM, etc.) which are almost always on
    private networks. A NON-admin probe is a user-supplied URL, so it gets the
    same strict external-only SSRF gate as create — otherwise the Test button
    becomes an internal-network probe for any tenant.
    """
    import asyncio

    if (blocked := await _guard_provider_url(
        body.base_url, admin=is_admin(request),
    )) is not None:
        return blocked

    http_client = request.app.state.http_client
    try:
        # Probe through the canonical factory so native-adapter providers
        # (gemini/google, claude) are tested against their real endpoint
        # rather than the OpenAI-compat /models path — otherwise a native
        # Gemini provider fails its pre-save Test with a false negative.
        tmp_backend = create_backend_from_profile(
            None,
            provider_type=body.provider_type,
            api_key=body.api_key or "",
            http_client=http_client,
            base_url=body.base_url,
        )
        models = await asyncio.wait_for(tmp_backend.list_models(), timeout=10.0)
        return JSONResponse({
            "status": "ok",
            "models": [{"name": m.name, "model": m.model} for m in models],
        })
    except TimeoutError:
        return JSONResponse(
            {"status": "error", "error": f"Connection timed out reaching {body.base_url}"},
            status_code=502,
        )
    except Exception as exc:
        error_msg = sanitize_error_detail(str(exc))
        # Provide helpful hints for common errors
        if "SSL" in error_msg or "certificate" in error_msg.lower():
            error_msg += " — SSL certificate verification failed. Check the URL uses https:// correctly."
        elif "ConnectError" in error_msg or "Connection refused" in error_msg:
            error_msg += " — Could not reach the server. Verify the URL and that the service is running."
        return JSONResponse(
            {"status": "error", "error": error_msg},
            status_code=502,
        )
