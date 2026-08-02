"""Cloud image generation provider routes — CRUD + generation proxy.

Manages cloud image providers (OpenAI, Together AI, Stability AI, etc.)
and proxies generation/editing requests to them. Providers are stored in
SQLite and can coexist alongside local GPU generation.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from augmentum.auth.guards import require_admin
from augmentum.config import settings
from augmentum.proxy import system_events
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.utils.http_client import SharedHTTPClient, normalize_base_url
from augmentum.utils.logging import get_logger
from augmentum.utils.secrets import decrypt_api_key, encrypt_api_key, sanitize_error_detail

log = get_logger(__name__)

router = APIRouter(tags=["cloud-image"])


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ImageProviderCreate(BaseModel):
    id: str
    name: str
    base_url: str
    api_key: str | None = None
    default_model: str = ""
    default_quality: str = "standard"


class ImageProviderUpdate(BaseModel):
    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    default_model: str | None = None
    default_quality: str | None = None
    is_enabled: bool | None = None
    is_default: bool | None = None


class CloudGenerateRequest(BaseModel):
    """Request to generate an image via a cloud provider."""
    prompt: str = Field(..., min_length=1, max_length=10000)
    negative_prompt: str = ""
    provider_id: str = ""
    model: str = ""
    width: int = 1024
    height: int = 1024
    quality: str = "standard"
    style: str = ""
    n: int = 1
    seed: int = -1
    response_format: str = "url"


class CloudEditRequest(BaseModel):
    """Request for cloud image editing (img2img / inpaint)."""
    prompt: str = Field(..., min_length=1, max_length=10000)
    provider_id: str = ""
    model: str = ""
    source_image: str  # base64
    mask_image: str = ""  # base64 (for inpainting)
    strength: float = 0.75
    width: int = 1024
    height: int = 1024
    quality: str = "standard"
    n: int = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_conn(request: Request):
    sm = getattr(request.app.state, "state_manager", None)
    if sm and isinstance(sm.backend, SQLiteBackend):
        return sm.backend.conn
    return None


async def _get_default_image_provider(conn) -> dict | None:
    cursor = await conn.execute(
        "SELECT id, name, base_url, api_key, default_model, default_quality "
        "FROM image_providers WHERE is_enabled = 1 "
        "ORDER BY is_default DESC LIMIT 1",
    )
    row = await cursor.fetchone()
    if not row:
        return None
    return {
        "id": row[0], "name": row[1], "base_url": row[2],
        "api_key": decrypt_api_key(row[3]), "default_model": row[4], "default_quality": row[5],
    }


async def _get_image_provider_by_id(conn, provider_id: str) -> dict | None:
    cursor = await conn.execute(
        "SELECT id, name, base_url, api_key, default_model, default_quality, "
        "is_enabled, is_default FROM image_providers WHERE id = ?",
        (provider_id,),
    )
    row = await cursor.fetchone()
    if not row:
        return None
    return {
        "id": row[0], "name": row[1], "base_url": row[2], "api_key": decrypt_api_key(row[3]),
        "default_model": row[4], "default_quality": row[5],
        "is_enabled": bool(row[6]), "is_default": bool(row[7]),
    }


def _is_stability(base_url: str) -> bool:
    return "stability.ai" in base_url.lower()


def _is_bfl(base_url: str) -> bool:
    return "bfl.ai" in base_url.lower() or "api.bfl" in base_url.lower()


def _is_fal(base_url: str) -> bool:
    return "fal.run" in base_url.lower() or "fal.ai" in base_url.lower()


def _build_headers(api_key: str | None, base_url: str = "") -> dict[str, str]:
    headers: dict[str, str] = {}
    if api_key:
        if _is_bfl(base_url):
            headers["x-key"] = api_key
        elif _is_fal(base_url):
            headers["Authorization"] = f"Key {api_key}"
        else:
            headers["Authorization"] = f"Bearer {api_key}"
    return headers


# Shared httpx clients — SSL verified for cloud, unverified for local
_cloud_http = SharedHTTPClient()


async def close_cloud_image_clients() -> None:
    """Close module-level httpx clients. Called during server shutdown."""
    await _cloud_http.close()


# Keep the name ``_cloud_client`` so existing call-sites work unchanged.
_cloud_client = _cloud_http.get


# ---------------------------------------------------------------------------
# Provider CRUD
# ---------------------------------------------------------------------------


@router.get("/api/image/cloud/providers")
async def list_image_providers(request: Request):
    conn = _get_conn(request)
    if not conn:
        return JSONResponse(content=[])

    cursor = await conn.execute(
        "SELECT id, name, base_url, default_model, default_quality, "
        "is_enabled, is_default FROM image_providers ORDER BY name"
    )
    rows = await cursor.fetchall()
    return JSONResponse(content=[
        {
            "id": r[0], "name": r[1], "base_url": r[2],
            "default_model": r[3], "default_quality": r[4],
            "is_enabled": bool(r[5]), "is_default": bool(r[6]),
        }
        for r in rows
    ])


@router.post("/api/image/cloud/providers")
async def create_image_provider(body: ImageProviderCreate, request: Request):
    """Add a new cloud image provider. Admin only — shared infrastructure."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    conn = _get_conn(request)
    if not conn:
        raise HTTPException(503, "Database not available")

    from augmentum.utils.safe_http import SafeHttpError, validate_provider_url
    try:
        body.base_url = validate_provider_url(body.base_url)
    except SafeHttpError as exc:
        raise HTTPException(400, str(exc))

    existing = await _get_image_provider_by_id(conn, body.id)
    if existing:
        raise HTTPException(409, f"Provider '{body.id}' already exists")

    cursor = await conn.execute(
        "SELECT COUNT(*) FROM image_providers", (),
    )
    count = (await cursor.fetchone())[0]
    is_default = 1 if count == 0 else 0

    await conn.execute(
        "INSERT INTO image_providers (id, name, base_url, api_key, "
        "default_model, default_quality, is_default) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (body.id, body.name, body.base_url, encrypt_api_key(body.api_key),
         body.default_model, body.default_quality, is_default),
    )
    await conn.commit()

    system_events.publish("image_providers.added", {"id": body.id})
    log.info("image_provider_created", id=body.id)
    return JSONResponse(content={"status": "created", "id": body.id, "is_default": bool(is_default)})


@router.put("/api/image/cloud/providers/{provider_id}")
async def update_image_provider(provider_id: str, body: ImageProviderUpdate, request: Request):
    """Update a cloud image provider. Admin only."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    conn = _get_conn(request)
    if not conn:
        raise HTTPException(503, "Database not available")

    existing = await _get_image_provider_by_id(conn, provider_id)
    if not existing:
        raise HTTPException(404, f"Provider '{provider_id}' not found")

    if body.base_url is not None:
        from augmentum.utils.safe_http import SafeHttpError, validate_provider_url
        try:
            body.base_url = validate_provider_url(body.base_url)
        except SafeHttpError as exc:
            raise HTTPException(400, str(exc))

    updates = []
    params = []
    for field_name, col in [
        ("name", "name"), ("base_url", "base_url"), ("api_key", "api_key"),
        ("default_model", "default_model"), ("default_quality", "default_quality"),
        ("is_enabled", "is_enabled"),
    ]:
        val = getattr(body, field_name, None)
        if val is not None:
            if field_name == "api_key":
                val = encrypt_api_key(val)
            updates.append(f"{col} = ?")
            params.append(val if not isinstance(val, bool) else int(val))

    if body.is_default is True:
        await conn.execute("UPDATE image_providers SET is_default = 0")
        updates.append("is_default = 1")

    if not updates:
        return JSONResponse(content={"status": "no_changes"})

    updates.append("updated_at = datetime('now')")
    params.append(provider_id)

    await conn.execute(
        f"UPDATE image_providers SET {', '.join(updates)} WHERE id = ?",
        params,
    )
    await conn.commit()
    system_events.publish("image_providers.updated", {"id": provider_id})
    return JSONResponse(content={"status": "updated"})


@router.delete("/api/image/cloud/providers/{provider_id}")
async def delete_image_provider(provider_id: str, request: Request):
    """Delete a cloud image provider. Admin only."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    conn = _get_conn(request)
    if not conn:
        raise HTTPException(503, "Database not available")

    existing = await _get_image_provider_by_id(conn, provider_id)
    if not existing:
        raise HTTPException(404, f"Provider '{provider_id}' not found")

    await conn.execute("DELETE FROM image_providers WHERE id = ?", (provider_id,))
    await conn.commit()

    if existing["is_default"]:
        try:
            await conn.execute(
                "UPDATE image_providers SET is_default = 1 "
                "WHERE is_enabled = 1 ORDER BY created_at LIMIT 1",
            )
            await conn.commit()
        except Exception:
            log.warning("image_default_promotion_failed", provider_id=provider_id)

    system_events.publish("image_providers.deleted", {"id": provider_id})
    log.info("image_provider_deleted", id=provider_id)
    return JSONResponse(content={"status": "deleted"})


@router.post("/api/image/cloud/providers/{provider_id}/test")
async def test_image_provider(provider_id: str, request: Request):
    conn = _get_conn(request)
    if not conn:
        raise HTTPException(503, "Database not available")

    provider = await _get_image_provider_by_id(conn, provider_id)
    if not provider:
        raise HTTPException(404, f"Provider '{provider_id}' not found")

    base_url = normalize_base_url(provider["base_url"])
    headers = _build_headers(provider["api_key"], base_url)

    try:
        async with _cloud_client(base_url) as client:
            if _is_stability(base_url):
                # Stability v2beta has no model list endpoint — verify auth
                # by attempting a lightweight request to a known endpoint
                resp = await client.get(
                    f"{base_url}/v2beta/stable-image/generate/core",
                    headers={**headers, "Accept": "application/json"},
                    timeout=10.0,
                )
                # 400 = bad request (expected without form data) but auth is valid
                # 401/403 = auth failed
                if resp.status_code in (401, 403):
                    raise HTTPException(401, "Invalid Stability AI API key")
                return JSONResponse(content={"status": "ok", "models": ["stable-image-core", "stable-image-ultra", "sd3.5-large", "sd3.5-large-turbo"]})

            if _is_bfl(base_url):
                # BFL has no model list endpoint — just verify auth
                return JSONResponse(content={"status": "ok", "models": [provider.get("default_model", "flux-pro-1.1")]})

            if _is_fal(base_url):
                return JSONResponse(content={"status": "ok", "models": [provider.get("default_model", "fal-ai/flux-2-pro")]})

            # OpenAI-compatible: try /v1/models
            resp = await client.get(f"{base_url}/v1/models", headers=headers, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            models_list = data.get("data", data) if isinstance(data, dict) else data
            # Filter for image models if possible
            model_ids = []
            for m in models_list[:50]:
                mid = m.get("id", m.get("name", "")) if isinstance(m, dict) else str(m)
                if mid:
                    model_ids.append(mid)
            return JSONResponse(content={"status": "ok", "models": model_ids[:20]})
    except Exception as exc:
        return JSONResponse(content={
            "status": "error", "error": sanitize_error_detail(str(exc)[:300]),
        }, status_code=200)


# ---------------------------------------------------------------------------
# Cloud model listing — merged into /api/image/models
# ---------------------------------------------------------------------------


@router.get("/api/image/cloud/models")
async def list_cloud_models(request: Request):
    """Return cloud image provider models for the unified model dropdown.

    Each model is returned with source='cloud' and a provider_id field
    so the UI can show cloud/local icons and route requests correctly.
    """
    conn = _get_conn(request)
    if not conn:
        return JSONResponse(content=[])

    cursor = await conn.execute(
        "SELECT id, name, base_url, api_key, default_model "
        "FROM image_providers WHERE is_enabled = 1 ORDER BY is_default DESC, name"
    )
    rows = await cursor.fetchall()
    if not rows:
        return JSONResponse(content=[])

    models = []
    for r in rows:
        prov_id, prov_name, prov_url = r[0], r[1], r[2]
        prov_key = decrypt_api_key(r[3])
        provider = {"id": prov_id, "name": prov_name, "base_url": prov_url, "api_key": prov_key, "default_model": r[4]}
        try:
            provider_models = await _fetch_cloud_models(provider)
        finally:
            del provider, prov_key  # clear decrypted key from scope
        for m in provider_models:
            m["provider_id"] = prov_id
            m["provider_name"] = prov_name
        models.extend(provider_models)

    return JSONResponse(content=models)


# Known model catalogs for providers without model listing endpoints
_KNOWN_CLOUD_MODELS: dict[str, list[dict]] = {
    "openai": [
        {"name": "gpt-image-1", "label": "GPT Image 1", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "yes", "img2img": "yes", "inpaint": "yes"}},
        {"name": "gpt-image-1-mini", "label": "GPT Image 1 Mini", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "yes", "img2img": "yes", "inpaint": "yes"}},
        {"name": "dall-e-3", "label": "DALL-E 3", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "yes", "img2img": "no", "inpaint": "no"}},
    ],
    "together": [
        {"name": "black-forest-labs/FLUX.1-schnell", "label": "FLUX.1 Schnell (Free)", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "yes", "img2img": "no", "inpaint": "no"}},
        {"name": "black-forest-labs/FLUX.1.1-pro", "label": "FLUX 1.1 Pro", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "yes", "img2img": "no", "inpaint": "no"}},
        {"name": "black-forest-labs/FLUX.2-max", "label": "FLUX.2 Max", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "yes", "img2img": "yes", "inpaint": "yes"}},
        {"name": "black-forest-labs/FLUX-1-kontext-pro", "label": "FLUX Kontext Pro", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "yes", "img2img": "yes", "inpaint": "no"}},
        {"name": "ideogram-ai/ideogram-v3-0", "label": "Ideogram v3", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "yes", "img2img": "yes", "inpaint": "yes"}},
    ],
    "stability": [
        {"name": "stable-image-core", "label": "Stable Image Core", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "yes", "img2img": "yes", "inpaint": "yes"}},
        {"name": "stable-image-ultra", "label": "Stable Image Ultra", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "yes", "img2img": "yes", "inpaint": "yes"}},
        {"name": "sd3.5-large", "label": "SD 3.5 Large", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "yes", "img2img": "yes", "inpaint": "yes"}},
        {"name": "sd3.5-large-turbo", "label": "SD 3.5 Large Turbo", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "yes", "img2img": "yes", "inpaint": "yes"}},
    ],
    "bfl": [
        {"name": "flux-pro-1.1", "label": "FLUX 1.1 Pro", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "yes", "img2img": "no", "inpaint": "no"}},
        {"name": "flux-pro-1.1-ultra", "label": "FLUX 1.1 Pro Ultra", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "yes", "img2img": "no", "inpaint": "no"}},
        {"name": "flux-2-pro", "label": "FLUX.2 Pro", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "yes", "img2img": "no", "inpaint": "no"}},
        {"name": "flux-dev", "label": "FLUX.1 Dev", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "yes", "img2img": "no", "inpaint": "no"}},
        {"name": "flux-kontext-pro", "label": "Kontext Pro", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "yes", "img2img": "no", "inpaint": "no"}},
    ],
    "fal": [
        {"name": "fal-ai/flux-2-pro", "label": "FLUX.2 Pro", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "yes", "img2img": "no", "inpaint": "no"}},
        {"name": "fal-ai/flux/dev", "label": "FLUX.1 Dev", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "yes", "img2img": "no", "inpaint": "no"}},
        {"name": "fal-ai/flux-pro/kontext", "label": "FLUX Kontext Pro", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "yes", "img2img": "no", "inpaint": "no"}},
        {"name": "fal-ai/flux-pro/v1/fill", "label": "FLUX Fill (Inpaint)", "pipeline_type": "cloud",
         "capabilities": {"txt2img": "no", "img2img": "no", "inpaint": "no"}},
    ],
}


def _detect_provider_type(base_url: str) -> str:
    url = base_url.lower()
    if "openai.com" in url:
        return "openai"
    if "together" in url:
        return "together"
    if _is_stability(url):
        return "stability"
    if _is_bfl(url):
        return "bfl"
    if _is_fal(url):
        return "fal"
    return "openai_compat"


async def _fetch_cloud_models(provider: dict) -> list[dict]:
    """Fetch available models from a cloud image provider."""
    base_url = normalize_base_url(provider["base_url"])
    ptype = _detect_provider_type(base_url)

    # Use known catalogs for providers we know about
    if ptype in _KNOWN_CLOUD_MODELS:
        return [m.copy() for m in _KNOWN_CLOUD_MODELS[ptype]]

    # For OpenAI-compatible providers, try to fetch models
    headers = _build_headers(provider.get("api_key"), base_url)
    try:
        async with _cloud_client(base_url) as client:
            resp = await client.get(f"{base_url}/v1/models", headers=headers, timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                models_list = data.get("data", []) if isinstance(data, dict) else data
                return [
                    {
                        "name": m.get("id", ""),
                        "label": m.get("id", ""),
                        "pipeline_type": "cloud",
                    }
                    for m in models_list[:30]
                    if isinstance(m, dict) and m.get("id")
                ]
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.TimeoutException) as exc:
        # Provider simply unreachable (backend not running, e.g. an optional
        # local cloud-image proxy that's off). Routine — log a one-liner, not
        # a 120-line ConnectError traceback. Full traceback is reserved below
        # for genuinely unexpected failures.
        log.warning("cloud_models_fetch_unreachable", base_url=base_url, error=str(exc))
    except Exception:
        log.warning("cloud_models_fetch_failed", base_url=base_url, exc_info=True)

    # Fallback: just the default model
    if provider.get("default_model"):
        return [{"name": provider["default_model"], "label": provider["default_model"], "pipeline_type": "cloud"}]
    return []


# ---------------------------------------------------------------------------
# Cloud Image Generation
# ---------------------------------------------------------------------------


async def generate_cloud_image(
    *,
    prompt: str,
    negative_prompt: str = "",
    model: str = "",
    provider_id: str = "",
    quality: str = "standard",
    width: int = 1024,
    height: int = 1024,
    n: int = 1,
    seed: int = -1,
    app_state: object | None = None,
) -> dict:
    """Core cloud image generation — reusable by both the route and the tool.

    Resolves the provider, dispatches to the correct API adapter, persists
    to the gallery, and returns a dict with ``image_id``, ``url``, etc.

    Raises on failure (callers should handle exceptions).
    """
    sm = getattr(app_state, "state_manager", None) if app_state else None
    backend = getattr(sm, "backend", None) if sm else None
    conn = backend.conn if isinstance(backend, SQLiteBackend) else None
    if not conn:
        raise RuntimeError("Database not available for cloud image generation")

    if provider_id:
        provider = await _get_image_provider_by_id(conn, provider_id)
        if not provider:
            raise RuntimeError(f"Cloud image provider '{provider_id}' not found")
    else:
        provider = await _get_default_image_provider(conn)
        if not provider:
            raise RuntimeError("No cloud image provider configured")

    base_url = normalize_base_url(provider["base_url"])
    resolved_model = model or provider["default_model"]
    if not resolved_model:
        raise RuntimeError("No model specified and provider has no default model configured")
    resolved_quality = quality or provider.get("default_quality", "standard")
    headers = _build_headers(provider["api_key"], base_url)
    ptype = _detect_provider_type(base_url)

    # Build a CloudGenerateRequest for the provider adapters
    body = CloudGenerateRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        provider_id=provider_id,
        model=resolved_model,
        width=width,
        height=height,
        quality=resolved_quality,
        n=n,
        seed=seed,
    )

    if ptype == "stability":
        result = await _generate_stability(base_url, headers, resolved_model, body, resolved_quality)
    elif ptype == "bfl":
        result = await _generate_bfl(base_url, headers, resolved_model, body)
    elif ptype == "fal":
        result = await _generate_fal(base_url, headers, resolved_model, body)
    else:
        result = await _generate_openai_compat(base_url, headers, resolved_model, body, resolved_quality)

    # Persist to gallery DB
    persistence = getattr(app_state, "image_persistence", None) if app_state else None
    if persistence:
        await _persist_cloud_generation(persistence, result, "txt2img")

    # Extract the response data for callers
    import json as _json
    data = _json.loads(result.body)
    return data


@router.post("/api/image/cloud/generate")
async def cloud_generate(body: CloudGenerateRequest, request: Request):
    """Generate an image via a cloud provider.

    Routes to the correct provider API based on the provider's base_url.
    Returns the same GenerateResponse format as local generation.
    """
    # Reject anon callers before we hit the provider — cloud generation
    # spends real credits, and post-fetch persistence at line ~1116 would
    # silently drop the orphan DB row anyway.
    _user = request.scope.get("user")
    if not _user:
        raise HTTPException(401, "Unauthorized")
    try:
        data = await generate_cloud_image(
            prompt=body.prompt,
            negative_prompt=body.negative_prompt,
            model=body.model,
            provider_id=body.provider_id,
            quality=body.quality,
            width=body.width,
            height=body.height,
            n=body.n,
            seed=body.seed,
            app_state=request.app.state,
        )
        return data  # Already a dict, FastAPI serializes automatically
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    except httpx.HTTPStatusError as exc:
        detail = sanitize_error_detail(exc.response.text[:500]) if exc.response else str(exc)
        log.warning("cloud_image_error", status=exc.response.status_code, detail=detail)
        raise HTTPException(exc.response.status_code, f"Cloud image provider error: {detail}")
    except httpx.RequestError as exc:
        log.warning("cloud_image_connection_error", error=str(exc))
        raise HTTPException(502, "Could not reach cloud image provider")


@router.post("/api/image/cloud/edit")
async def cloud_edit(body: CloudEditRequest, request: Request):
    """Edit/inpaint an image via a cloud provider."""
    # Reject anon callers before we hit the provider — same reason as
    # cloud_generate above.
    _user = request.scope.get("user")
    if not _user:
        raise HTTPException(401, "Unauthorized")
    conn = _get_conn(request)
    if not conn:
        raise HTTPException(503, "Database not available")

    if body.provider_id:
        provider = await _get_image_provider_by_id(conn, body.provider_id)
        if not provider:
            raise HTTPException(404, f"Provider '{body.provider_id}' not found")
    else:
        provider = await _get_default_image_provider(conn)
        if not provider:
            raise HTTPException(503, "No cloud image provider configured")

    base_url = normalize_base_url(provider["base_url"])
    model = body.model or provider["default_model"]
    if not model:
        raise HTTPException(422, "No model specified and provider has no default model configured")
    headers = _build_headers(provider["api_key"], base_url)
    ptype = _detect_provider_type(base_url)

    try:
        if ptype == "stability":
            result = await _edit_stability(base_url, headers, body)
        elif ptype in ("bfl", "fal"):
            # BFL and Fal don't support OpenAI-compat editing endpoints
            raise HTTPException(
                501,
                f"Image editing is not yet supported for {ptype.upper()} provider. "
                "Use a Stability AI or OpenAI provider for img2img/inpainting.",
            )
        else:
            # OpenAI, Together, generic OpenAI-compat
            result = await _edit_openai_compat(base_url, headers, model, body)

        await _persist_cloud_generation(request, result, "img2img")
        return result
    except httpx.HTTPStatusError as exc:
        detail = sanitize_error_detail(exc.response.text[:500]) if exc.response else str(exc)
        raise HTTPException(exc.response.status_code, f"Cloud image edit error: {detail}")
    except httpx.RequestError as exc:
        raise HTTPException(502, "Could not reach cloud image provider")


# ---------------------------------------------------------------------------
# Provider-specific generation adapters
# ---------------------------------------------------------------------------


async def _generate_openai_compat(
    base_url: str, headers: dict, model: str,
    body: CloudGenerateRequest, quality: str,
) -> JSONResponse:
    """OpenAI-compatible: POST /v1/images/generations."""
    headers["Content-Type"] = "application/json"
    is_openai = "openai.com" in base_url.lower()
    is_together = "together" in base_url.lower()
    # gpt-image-* always returns b64_json and REJECTS the response_format
    # param (400) — only dall-e-2/3 and generic OAI-compat accept it.
    is_gpt_image = is_openai and model.startswith("gpt-image")

    payload: dict[str, Any] = {
        "prompt": body.prompt,
        "model": model,
        "n": body.n,
    }
    if not is_gpt_image:
        payload["response_format"] = "b64_json"

    # Together uses separate width/height integers; OpenAI uses "WxH" string
    if is_together:
        payload["width"] = body.width
        payload["height"] = body.height
    else:
        payload["size"] = f"{body.width}x{body.height}"

    # quality/style only valid for DALL-E 3 (not GPT-Image, not Together)
    is_dalle3 = model == "dall-e-3"
    if is_dalle3 or not is_openai:
        payload["quality"] = quality
    if body.style and is_dalle3:
        payload["style"] = body.style

    # negative_prompt: not supported by OpenAI, supported by Together and generics
    if body.negative_prompt and not is_openai:
        payload["negative_prompt"] = body.negative_prompt

    async with _cloud_client(base_url) as client:
        resp = await client.post(
            f"{base_url}/v1/images/generations",
            json=payload, headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    # Save image(s) and return in our standard format
    images = data.get("data", [])
    if not images:
        raise HTTPException(500, "No images returned from provider")

    results = []
    for img_data in images:
        b64 = img_data.get("b64_json", "")
        url = img_data.get("url", "")
        revised_prompt = img_data.get("revised_prompt", body.prompt)
        image_id = str(uuid.uuid4())
        if b64:
            _save_cloud_image(image_id, b64)
        results.append({
            "image_id": image_id,
            "job_id": "cloud",
            "status": "completed",
            "url": f"/api/image/{image_id}" if b64 else url,
            "seed": -1,
            "prompt": revised_prompt,
            "negative_prompt": body.negative_prompt,
            "width": body.width,
            "height": body.height,
            "steps": 0,
            "model": model,
            "source": "cloud",
        })

    # Batch response when multiple images, single-image compat otherwise
    if len(results) == 1:
        return JSONResponse(content=results[0])
    return JSONResponse(content={"images": results})


async def _generate_stability(
    base_url: str, headers: dict, model: str,
    body: CloudGenerateRequest, quality: str,
) -> JSONResponse:
    """Stability AI: POST /v2beta/stable-image/generate/{model}.

    Stability requires multipart/form-data (not JSON). Uses aspect_ratio
    instead of explicit width/height for most models.
    """
    endpoint_map = {
        "stable-image-core": "core",
        "stable-image-ultra": "ultra",
        "sd3.5-large": "sd3",
        "sd3.5-large-turbo": "sd3",
    }
    endpoint = endpoint_map.get(model, "core")

    headers["Accept"] = "application/json"
    # Remove Content-Type — httpx sets multipart boundary automatically
    headers.pop("Content-Type", None)

    # Build multipart form fields as list of tuples for proper multipart encoding
    form_fields: dict[str, str] = {
        "prompt": body.prompt,
        "output_format": "png",
    }
    if body.negative_prompt:
        form_fields["negative_prompt"] = body.negative_prompt

    # Map dimensions to closest Stability aspect ratio
    # Supported: 1:1, 16:9, 9:16, 3:2, 2:3, 4:3, 3:4, 5:4, 4:5
    ratio = body.width / body.height
    if ratio > 1.65:
        form_fields["aspect_ratio"] = "16:9"
    elif ratio > 1.35:
        form_fields["aspect_ratio"] = "3:2"
    elif ratio > 1.15:
        form_fields["aspect_ratio"] = "4:3"
    elif ratio > 1.05:
        form_fields["aspect_ratio"] = "5:4"
    elif ratio >= 0.95:
        form_fields["aspect_ratio"] = "1:1"
    elif ratio >= 0.87:
        form_fields["aspect_ratio"] = "4:5"
    elif ratio >= 0.74:
        form_fields["aspect_ratio"] = "3:4"
    elif ratio >= 0.6:
        form_fields["aspect_ratio"] = "2:3"
    else:
        form_fields["aspect_ratio"] = "9:16"

    if body.seed != -1:
        form_fields["seed"] = str(body.seed)

    if endpoint == "sd3" and model == "sd3.5-large-turbo":
        form_fields["model"] = "sd3.5-large-turbo"
    elif endpoint == "sd3":
        form_fields["model"] = "sd3.5-large"

    async with _cloud_client(base_url) as client:
        # Use files= with tuple values to force multipart/form-data encoding
        multipart = {k: (None, v) for k, v in form_fields.items()}
        resp = await client.post(
            f"{base_url}/v2beta/stable-image/generate/{endpoint}",
            files=multipart, headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    image_b64 = data.get("image", "")
    image_id = str(uuid.uuid4())
    if image_b64:
        _save_cloud_image(image_id, image_b64)

    return JSONResponse(content={
        "image_id": image_id,
        "job_id": "cloud",
        "status": "completed",
        "url": f"/api/image/{image_id}" if image_b64 else "",
        "seed": data.get("seed", -1),
        "prompt": body.prompt,
        "negative_prompt": body.negative_prompt,
        "width": body.width,
        "height": body.height,
        "steps": 0,
        "model": model,
        "source": "cloud",
    })


async def _generate_bfl(
    base_url: str, headers: dict, model: str,
    body: CloudGenerateRequest,
) -> JSONResponse:
    """Black Forest Labs: async POST /v1/{model} then poll for result."""
    import asyncio

    headers["Content-Type"] = "application/json"
    payload: dict[str, Any] = {
        "prompt": body.prompt,
        "width": body.width,
        "height": body.height,
    }
    if body.seed != -1:
        payload["seed"] = body.seed

    async with _cloud_client(base_url) as client:
        # Submit generation request
        resp = await client.post(
            f"{base_url}/v1/{model}",
            json=payload, headers=headers,
        )
        resp.raise_for_status()
        task = resp.json()
        task_id = task.get("id", "")
        polling_url = task.get("polling_url", f"{base_url}/v1/get_result?id={task_id}")

        # Poll for result (max ~90 seconds)
        for _ in range(90):
            await asyncio.sleep(1)
            poll_resp = await client.get(polling_url, headers=headers, timeout=15.0)
            if poll_resp.status_code != 200:
                continue
            result = poll_resp.json()
            status = result.get("status", "")
            if status == "Ready":
                img_url = result.get("result", {}).get("sample", "")
                image_id = str(uuid.uuid4())

                # Download and save the image
                if img_url:
                    img_resp = await client.get(img_url, timeout=30.0)
                    if img_resp.status_code == 200:
                        b64 = base64.b64encode(img_resp.content).decode()
                        _save_cloud_image(image_id, b64)
                    else:
                        log.warning("bfl_image_download_failed", url=img_url, status=img_resp.status_code)
                else:
                    log.warning("bfl_no_image_url", task_id=task_id)

                return JSONResponse(content={
                    "image_id": image_id,
                    "job_id": task_id,
                    "status": "completed",
                    "url": f"/api/image/{image_id}",
                    "seed": result.get("result", {}).get("seed", -1),
                    "prompt": body.prompt,
                    "negative_prompt": body.negative_prompt,
                    "width": body.width,
                    "height": body.height,
                    "steps": 0,
                    "model": model,
                    "source": "cloud",
                })
            if status in ("Error", "Content Moderated"):
                raise HTTPException(400, f"BFL generation failed: {status}")

        raise HTTPException(504, "BFL generation timed out waiting for result")


async def _generate_fal(
    base_url: str, headers: dict, model: str,
    body: CloudGenerateRequest,
) -> JSONResponse:
    """Fal.ai: POST https://fal.run/{model}."""
    headers["Content-Type"] = "application/json"
    payload: dict[str, Any] = {
        "prompt": body.prompt,
        "image_size": {"width": body.width, "height": body.height},
        "num_images": body.n,
    }
    if body.negative_prompt:
        payload["negative_prompt"] = body.negative_prompt
    if body.seed != -1:
        payload["seed"] = body.seed

    # Fal.ai model URLs are https://fal.run/{model_id} — there is NO
    # ``/v1/`` segment (the model id itself carries the path, e.g.
    # ``fal-ai/flux/dev``). The prior ``/v1/`` 404'd every fal request.
    fal_base = base_url if "fal" in base_url.lower() else "https://fal.run"
    fal_url = f"{fal_base.rstrip('/')}/{model}"

    async with _cloud_client(base_url) as client:
        resp = await client.post(fal_url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    images = data.get("images", [])
    if not images:
        raise HTTPException(500, "No images returned from Fal.ai")

    results = []
    for img in images:
        img_url = img.get("url", "")
        image_id = str(uuid.uuid4())
        if img_url:
            async with _cloud_client("") as client:
                img_resp = await client.get(img_url, timeout=30.0)
                if img_resp.status_code == 200:
                    b64 = base64.b64encode(img_resp.content).decode()
                    _save_cloud_image(image_id, b64)
        results.append({
            "image_id": image_id,
            "job_id": "cloud",
            "status": "completed",
            "url": f"/api/image/{image_id}" if img_url else "",
            "seed": img.get("seed", data.get("seed", -1)),
            "prompt": body.prompt,
            "negative_prompt": body.negative_prompt,
            "width": img.get("width", body.width),
            "height": img.get("height", body.height),
            "steps": 0,
            "model": model,
            "source": "cloud",
        })

    if len(results) == 1:
        return JSONResponse(content=results[0])
    return JSONResponse(content={"images": results})


# ---------------------------------------------------------------------------
# Provider-specific editing adapters
# ---------------------------------------------------------------------------


async def _edit_openai_compat(
    base_url: str, headers: dict, model: str,
    body: CloudEditRequest,
) -> JSONResponse:
    """OpenAI-compatible: POST /v1/images/edits with multipart form."""
    img_bytes = base64.b64decode(body.source_image)

    files: dict[str, Any] = {
        "image": ("image.png", img_bytes, "image/png"),
    }
    if body.mask_image:
        mask_bytes = base64.b64decode(body.mask_image)
        files["mask"] = ("mask.png", mask_bytes, "image/png")

    form: dict[str, str] = {
        "prompt": body.prompt,
        "model": model,
        "n": str(body.n),
        "size": f"{body.width}x{body.height}",
        "response_format": "b64_json",
    }

    async with _cloud_client(base_url) as client:
        resp = await client.post(
            f"{base_url}/v1/images/edits",
            files=files, data=form, headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    images = data.get("data", [])
    if not images:
        raise HTTPException(500, "No images returned from provider")

    results = []
    for img_data in images:
        b64 = img_data.get("b64_json", "")
        image_id = str(uuid.uuid4())
        if b64:
            _save_cloud_image(image_id, b64)
        results.append({
            "image_id": image_id,
            "job_id": "cloud",
            "status": "completed",
            "url": f"/api/image/{image_id}" if b64 else "",
            "seed": -1,
            "prompt": body.prompt,
            "negative_prompt": "",
            "width": body.width,
            "height": body.height,
            "steps": 0,
            "model": model,
            "source": "cloud",
        })

    if len(results) == 1:
        return JSONResponse(content=results[0])
    return JSONResponse(content={"images": results})


async def _edit_stability(
    base_url: str, headers: dict,
    body: CloudEditRequest,
) -> JSONResponse:
    """Stability AI: POST /v2beta/stable-image/edit/inpaint."""
    headers["Accept"] = "application/json"
    img_bytes = base64.b64decode(body.source_image)

    files: dict[str, Any] = {
        "image": ("image.png", img_bytes, "image/png"),
    }
    form: dict[str, str] = {
        "prompt": body.prompt,
        "output_format": "png",
    }
    if hasattr(body, "strength") and body.strength and body.strength != 0.75:
        form["strength"] = str(body.strength)

    if body.mask_image:
        mask_bytes = base64.b64decode(body.mask_image)
        files["mask"] = ("mask.png", mask_bytes, "image/png")

    endpoint = "inpaint" if body.mask_image else "search-and-replace"

    # Remove Content-Type so httpx sets multipart boundary automatically
    headers.pop("Content-Type", None)

    async with _cloud_client(base_url) as client:
        resp = await client.post(
            f"{base_url}/v2beta/stable-image/edit/{endpoint}",
            files=files, data=form, headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    image_b64 = data.get("image", "")
    image_id = str(uuid.uuid4())
    if image_b64:
        _save_cloud_image(image_id, image_b64)

    return JSONResponse(content={
        "image_id": image_id,
        "job_id": "cloud",
        "status": "completed",
        "url": f"/api/image/{image_id}" if image_b64 else "",
        "seed": data.get("seed", -1),
        "prompt": body.prompt,
        "negative_prompt": "",
        "width": body.width,
        "height": body.height,
        "steps": 0,
        "model": "stability-edit",
        "source": "cloud",
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _persist_cloud_generation(
    request_or_persistence, response: JSONResponse, job_type: str = "txt2img",
) -> None:
    """Save a cloud-generated image to the persistence DB for gallery display.

    Accepts either a Request object or a persistence instance directly.
    """
    import json as _json

    if hasattr(request_or_persistence, "app"):
        persistence = getattr(request_or_persistence.app.state, "image_persistence", None)
    else:
        persistence = request_or_persistence
    if not persistence:
        return

    try:
        data = _json.loads(response.body)
        image_id = data.get("image_id", "")
        if not image_id:
            return

        output_dir = settings.image_output_dir or f"{settings.data_dir}/image_output"
        import os
        file_path = os.path.join(output_dir, f"{image_id}.png")

        # Extract user_id from request scope if available
        _cloud_uid = ""
        if hasattr(request_or_persistence, "scope"):
            _cloud_user = request_or_persistence.scope.get("user")
            _cloud_uid = _cloud_user.id if _cloud_user else ""
        if not _cloud_uid:
            log.warning("cloud_persist_skipped_no_user", image_id=image_id)
            return

        await persistence.save_generation(
            image_id=image_id,
            session_id="",
            prompt=data.get("prompt", ""),
            negative_prompt=data.get("negative_prompt", ""),
            model=data.get("model", ""),
            seed=data.get("seed", -1),
            width=data.get("width", 0),
            height=data.get("height", 0),
            steps=0,
            cfg_scale=0.0,
            preset="",
            loras=[],
            file_path=file_path,
            job_type=job_type,
            user_id=_cloud_uid,
        )
    except Exception:
        log.warning("cloud_persist_failed", exc_info=True)


def _save_cloud_image(image_id: str, b64_data: str) -> str:
    """Save a base64-encoded image to the output directory."""
    output_dir = settings.image_output_dir or f"{settings.data_dir}/image_output"
    import os
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{image_id}.png")
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_data))
    return path
