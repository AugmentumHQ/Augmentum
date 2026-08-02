"""Install dispatchers shared between community URL flow and Discover.

Each dispatcher takes ``(request, artifact, user_id)`` and returns the
installed resource id (a string — character id, flow id, power slug,
or knowledge pack job id). Raises ``HTTPException`` on validation
failures so callers can surface the status code + detail directly.

The functions used to live in ``augmentum/proxy/community_routes.py``.
They were extracted so the Discover routes (``/api/discover/{id}/
install``) and the legacy community routes (``POST /api/community/
install``) can share the same install code paths without duplication.

When adding a new installable category:
  1. Add ``_install_<category>`` here with the same signature.
  2. Register the install_via key in ``DISPATCHER_REGISTRY`` below.
  3. Update community_routes._KNOWN_CATEGORIES if the new category
     should also accept manifest-URL installs.
"""

from __future__ import annotations

import asyncio as _asyncio
import re
import time as _time
import uuid
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException, Request
from pydantic import ValidationError

from augmentum.proxy import character_routes as _char
from augmentum.utils.logging import get_logger
from augmentum.utils.safe_http import SafeHttpClient

log = get_logger(__name__)


# ── Constants ────────────────────────────────────────────────────────


_SLUG_RE = re.compile(r"^[a-z0-9-]{3,48}$")

_VALID_POWER_KINDS = frozenset(
    {"guidance", "verifier", "workflow", "integration", "bridge"}
)


def _sanitize_filename(name: str) -> str:
    """Coerce to a safe filesystem name — alnum + dash + underscore only."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")[:64] or "pack"


def _looks_like_host_path(p: str) -> bool:
    """True if ``p`` is shaped like an absolute Docker-host path.

    We can't stat the host filesystem from inside the Augmentum container,
    so this only validates shape: a POSIX absolute path (``/mnt/media``) or
    a Windows absolute path (``C:\\Media`` / ``C:/Media``). Bind sources
    must be absolute — relative paths are rejected so we never hand Docker
    an ambiguous source.
    """
    p = p.strip()
    if not p:
        return False
    if p.startswith("/"):
        return True
    return bool(re.match(r"^[A-Za-z]:[\\/]", p))


def _settings_for_max(request: Request):
    return getattr(request.app.state, "settings", None)


# ── Dispatchers ──────────────────────────────────────────────────────


async def _install_character(
    request: Request, artifact: Any, user_id: str,
) -> str:
    """Install a community character card.

    Reuses the helpers from ``character_routes`` so the import path is
    identical to the existing ``POST /api/characters/import-json``.
    """
    if not isinstance(artifact, dict):
        raise HTTPException(
            status_code=400,
            detail="Character artifact must be a JSON object",
        )

    be = _char._backend(request)
    if not be:
        raise HTTPException(status_code=503, detail="Database unavailable")

    data = _char._normalize_card(artifact)
    if not data:
        raise HTTPException(
            status_code=400, detail="Unrecognized character card format"
        )

    fields = _char._map_fields(data)
    char_id, char = _char._build_char(fields, data)

    # Avatars in community cards are referenced by URL; download is best-effort.
    avatar = ""
    avatar_url = (
        data.get("avatar")
        or data.get("photo")
        or data.get("profile_image")
        or data.get("avatar_url")
        or ""
    )
    if isinstance(avatar_url, str) and avatar_url.startswith("http"):
        try:
            avatar = await _char._download_avatar(avatar_url)
        except Exception as exc:  # avatar is non-critical
            log.warning("community_avatar_download_failed", error=str(exc))

    await _char._upsert_char(
        be, char_id, fields["name"], char, avatar, uid=user_id,
    )
    return char_id


async def _install_reasoning_flow(
    request: Request, artifact: Any, user_id: str,
) -> str:
    """Install a community reasoning flow via the existing flow store."""
    if not isinstance(artifact, dict):
        raise HTTPException(
            status_code=400,
            detail="Reasoning flow artifact must be a JSON object",
        )

    store = getattr(request.app.state, "reasoning_flow_store", None)
    if not store:
        raise HTTPException(
            status_code=503, detail="Reasoning flow store unavailable"
        )

    try:
        flow = await store.import_flow(artifact, user_id=user_id)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid flow: {exc}") from exc

    return flow.id


async def _install_power(
    request: Request, artifact: Any, user_id: str,
) -> str:
    """Install a community Power.

    Powers are install-wide (every user can see + pin them), so this
    requires admin. The POWER.md is written to
    ``{data_dir}/community-powers/<slug>/POWER.md`` — a directory
    PowerRegistry scans by default alongside the shipped packs.
    """
    from augmentum.auth.guards import is_admin
    if not is_admin(request):
        raise HTTPException(
            status_code=403,
            detail="Community Power install is admin-only (Powers are install-wide).",
        )

    if not isinstance(artifact, str):
        raise HTTPException(
            status_code=400,
            detail="Power artifact must be the raw POWER.md markdown text",
        )

    body = artifact.strip()
    if not body.startswith("---"):
        raise HTTPException(
            status_code=400,
            detail="POWER.md must start with a YAML frontmatter block (---)",
        )

    parts = body.split("---", 2)
    if len(parts) < 3:
        raise HTTPException(
            status_code=400,
            detail="POWER.md frontmatter block is malformed (missing closing ---)",
        )
    frontmatter_text = parts[1]
    try:
        meta = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Power frontmatter is not valid YAML: {exc}",
        ) from exc

    if not isinstance(meta, dict):
        raise HTTPException(
            status_code=400,
            detail="Power frontmatter must be a YAML mapping",
        )

    slug = str(meta.get("name") or "").strip()
    if not _SLUG_RE.match(slug):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Power 'name' must be kebab-case, 3-48 chars "
                f"(got {slug!r})"
            ),
        )

    kind = str(meta.get("kind") or "").strip()
    if kind not in _VALID_POWER_KINDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Power 'kind' must be one of "
                f"{', '.join(sorted(_VALID_POWER_KINDS))} (got {kind!r})"
            ),
        )

    from augmentum.config import settings as _settings
    data_dir = Path(getattr(_settings, "data_dir", "/data"))
    target_dir = data_dir / "community-powers" / slug

    if target_dir.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                f"A community Power with slug {slug!r} is already installed. "
                f"Remove it first (rm -r {target_dir}) before reinstalling."
            ),
        )

    try:
        target_dir.mkdir(parents=True, exist_ok=False)
        (target_dir / "POWER.md").write_text(body, encoding="utf-8")
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Couldn't write Power to disk: {exc}",
        ) from exc

    registry = getattr(request.app.state, "power_registry", None)
    if registry is not None:
        try:
            registry.rescan()
        except Exception as exc:
            log.warning("community_power_rescan_failed", error=str(exc))

    return slug


async def _install_knowledge_pack(
    request: Request, artifact: Any, user_id: str,
) -> str:
    """Install a community knowledge pack.

    Returns the job_id of the background download task.
    """
    from augmentum.auth.guards import is_admin
    if not is_admin(request):
        raise HTTPException(
            status_code=403,
            detail="Community knowledge pack install is admin-only.",
        )

    if not isinstance(artifact, dict):
        raise HTTPException(
            status_code=400,
            detail="Knowledge pack artifact must be a JSON object",
        )

    fmt = str(artifact.get("format") or "").strip()
    if fmt not in ("augpack", "zim"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported knowledge pack format: {fmt!r} "
                f"(expected 'augpack' or 'zim')"
            ),
        )
    if fmt == "zim":
        raise HTTPException(
            status_code=400,
            detail=(
                "ZIM-format community packs are not yet supported via the "
                "community install flow. Use POST /api/knowledge/install "
                "directly with the Kiwix catalog_id + download_url."
            ),
        )

    download_url = str(artifact.get("download_url") or "").strip()
    if not download_url:
        raise HTTPException(
            status_code=400,
            detail="Knowledge pack artifact requires a non-empty 'download_url'",
        )
    if not download_url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400, detail="download_url must be http(s)",
        )

    size_bytes = int(artifact.get("size_bytes") or 0)
    max_mb = int(
        getattr(_settings_for_max(request), "community_max_pack_size_mb", 500)
    )
    if size_bytes > 0 and size_bytes > max_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Pack size {size_bytes / 1024 / 1024:.1f} MB exceeds "
                f"community_max_pack_size_mb={max_mb}"
            ),
        )

    pack_name = str(artifact.get("name") or "Community pack").strip()
    checksum = str(artifact.get("checksum_sha256") or "").strip()

    mgr = getattr(request.app.state, "knowledge_pack_manager", None)
    if mgr is None:
        from augmentum.proxy.knowledge_routes import _get_pack_manager
        mgr = _get_pack_manager(request)
    if mgr is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge pack manager unavailable",
        )

    pack_dir = Path(mgr.pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)
    output_path = pack_dir / f"{_sanitize_filename(pack_name)}.augpack"
    if output_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"A pack named {output_path.name!r} is already installed",
        )

    job_id = uuid.uuid4().hex[:12]
    jobs: dict = getattr(request.app.state, "install_jobs", None)
    if jobs is None:
        jobs = {}
        setattr(request.app.state, "install_jobs", jobs)

    from augmentum.proxy.knowledge_routes import InstallJob
    job = InstallJob(
        job_id=job_id,
        catalog_id=f"community:{pack_name}",
        status="started",
        stage="downloading",
        started_at=_time.time(),
    )
    job._community_source = download_url  # type: ignore[attr-defined]

    async def _download_and_scan():
        try:
            client = SafeHttpClient(max_response_size=max_mb * 1024 * 1024)
            _ = client  # silence unused — kept for SafeHttpError parity
            import httpx
            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as c:
                async with c.stream("GET", download_url) as resp:
                    resp.raise_for_status()
                    with output_path.open("wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                            f.write(chunk)

            if checksum:
                import hashlib
                h = hashlib.sha256()
                with output_path.open("rb") as f:
                    for chunk in iter(lambda: f.read(64 * 1024), b""):
                        h.update(chunk)
                got = h.hexdigest()
                if got != checksum:
                    output_path.unlink(missing_ok=True)
                    job.status = "error"
                    job.error = (
                        f"Checksum mismatch (expected {checksum}, got {got})"
                    )
                    return

            await mgr.scan()
            job.status = "completed"
            job.stage = "done"
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            job.status = "error"
            job.error = str(exc)
            log.warning("community_pack_install_failed", error=str(exc))

    job.task = _asyncio.create_task(_download_and_scan())
    jobs[job_id] = job

    return job_id


# ── Provider service dispatcher (NEW for Discover) ───────────────────


def _provider_requirements_preflight(sd: Any) -> None:
    """Raise a 422 with a machine-readable code if a service's declared
    install requirements aren't met yet.

    Currently gates on ``requirements.token`` (a secret that must be present
    in settings for a gated model download — e.g. fish-tts's HuggingFace
    token). The install card reads the same ``requirements`` off the listing,
    collects the token inline, saves it via the settings endpoint, then
    retries the install. License notes are informational (surfaced by the
    card), not gated here.
    """
    reqs = getattr(sd, "requirements", None) or {}
    token_req = reqs.get("token")
    if token_req:
        setting_key = str(token_req.get("setting") or "").strip()
        from augmentum.config import settings as _settings
        val = str(getattr(_settings, setting_key, "") or "").strip() if setting_key else ""
        if not val:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "needs_token",
                    "service_id": sd.id,
                    "token": token_req,
                    "license": reqs.get("license"),
                },
            )


async def _install_provider_service(
    request: Request, artifact: Any, user_id: str,
) -> str:
    """Enable a Docker-managed provider service.

    ``artifact`` is the install_payload dict from the listing; we only
    need ``service_id``. The actual container start is delegated to
    ``ServiceManager.enable_service`` which exists today behind
    ``POST /api/marketplace/services/{id}/enable``. Admin-only —
    starting a container has side effects (port allocation, GPU
    contention, image pull).
    """
    from augmentum.auth.guards import is_admin
    if not is_admin(request):
        raise HTTPException(
            status_code=403,
            detail="Provider service install is admin-only.",
        )

    if not isinstance(artifact, dict):
        raise HTTPException(
            status_code=400,
            detail="Provider artifact must be a JSON object",
        )

    service_id = str(artifact.get("service_id") or "").strip()
    if not service_id:
        raise HTTPException(
            status_code=400,
            detail="install_payload.service_id is required",
        )

    mgr = getattr(request.app.state, "service_manager", None)
    if mgr is None:
        raise HTTPException(
            status_code=503, detail="Service manager unavailable",
        )

    # Requirements pre-flight (e.g. fish-tts's gated HuggingFace token) BEFORE
    # we enqueue anything. The install card collects + saves the token via the
    # settings endpoint first (single source of truth for encrypt-at-rest + env
    # propagation), so here we only verify it landed. A 422 with a
    # machine-readable code lets the card re-surface the inline token field
    # instead of a mystery timeout.
    sd = mgr.get_definition(service_id) if hasattr(mgr, "get_definition") else None
    if sd is not None:
        _provider_requirements_preflight(sd)

    # Model choice (e.g. fish-tts's OpenAudio S1-mini vs S1): the install card
    # sends the chosen id under _install_options.model; the job resolves it to
    # env overrides from the catalog's requirements.model.choices. Default is
    # applied by the job when absent, so this is optional.
    options = artifact.get("_install_options") if isinstance(artifact, dict) else None
    model_choice_id = ""
    if isinstance(options, dict):
        model_choice_id = str(options.get("model") or "").strip()

    # Hand the slow work (pull → start → health → register) to the shared
    # ``service_install`` background job so the Discover card shows honest
    # staged progress — identical UX to the service_staged (engine) path —
    # instead of a blocking POST behind a spinner. Returns the JOB id; the
    # install route surfaces it and the card polls /api/jobs/{id}.
    jobs_store = getattr(request.app.state, "jobs_store", None)
    job_runner = getattr(request.app.state, "job_runner", None)
    if jobs_store is None:
        raise HTTPException(status_code=503, detail="Job queue unavailable")
    job_id = await jobs_store.create(
        user_id=user_id,
        job_type="service_install",
        payload={"service_id": service_id, "model_choice": model_choice_id},
    )
    if job_runner is not None:
        job_runner.wake()
    return job_id


def _media_server_store(request: Request):
    """MediaServerStore over the live SQLite connection, or None.

    Mirrors ``media_routes._get_store`` without importing it (avoids a
    route→dispatcher import cycle)."""
    from augmentum.media.store import MediaServerStore
    from augmentum.state.backends.sqlite import SQLiteBackend

    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm else None
    if isinstance(backend, SQLiteBackend):
        return MediaServerStore(backend.conn)
    return None


async def _install_media_server(
    request: Request, artifact: Any, user_id: str,
) -> str:
    """Provision a media-server sidecar and auto-connect it to Files.

    The point of the AI-OS "one-click install" (vs. connect-an-existing):
    start a fresh Jellyfin/Suwayomi/… container via
    ``ServiceManager.enable_service``, poll until it answers, then create
    a per-user ``user_media_servers`` row pointing at the container on the
    shared Docker network. The user immediately sees the server in Files —
    no manual URL/credentials step.

    Admin-only: provisioning a container has install-wide side effects
    (image pull, port bind, disk). The *connection* row is created for the
    installing user. Idempotent on re-install via ``find_match``.
    """
    from augmentum.auth.guards import is_admin
    if not is_admin(request):
        raise HTTPException(
            status_code=403, detail="Media-server install is admin-only.",
        )

    if not isinstance(artifact, dict):
        raise HTTPException(
            status_code=400, detail="Media artifact must be a JSON object",
        )
    service_id = str(artifact.get("service_id") or "").strip()
    provider = str(artifact.get("provider") or service_id).strip()
    if not service_id:
        raise HTTPException(
            status_code=400, detail="install_payload.service_id is required",
        )

    mgr = getattr(request.app.state, "service_manager", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="Service manager unavailable")
    sd = mgr.get_definition(service_id)
    if sd is None:
        raise HTTPException(
            status_code=400, detail=f"Unknown media service: {service_id}",
        )

    # External library: if this server has a local media mount and the user
    # supplied a host directory, bind-mount it so their library lives on
    # their OWN storage (not opaque Docker storage). Blank → named-volume
    # fallback. The path is resolved on the Docker host, so we can only
    # sanity-check the shape here, not its existence.
    media_mount = str(artifact.get("media_mount") or "").strip()
    options = artifact.get("_install_options") or {}
    host_path = str((options.get("media_host_path") or "")).strip()
    volume_overrides: dict[str, str] = {}
    if media_mount and host_path:
        if not _looks_like_host_path(host_path):
            raise HTTPException(
                status_code=400,
                detail="Media folder must be an absolute host path "
                       "(e.g. /mnt/media or C:\\Media).",
            )
        volume_overrides[media_mount] = host_path

    # 1) Provision the container (with the host bind if given).
    try:
        await mgr.enable_service(service_id, volume_overrides=volume_overrides or None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.warning(
            "media_server_provision_failed",
            service_id=service_id, error=str(exc),
        )
        raise HTTPException(
            status_code=500, detail=f"Couldn't start {provider}: {exc}",
        ) from exc

    return await _connect_media_server(
        request, sd=sd, service_id=service_id, provider=provider, user_id=user_id,
    )


async def _connect_media_server(
    request: Request, *, sd: Any, service_id: str, provider: str, user_id: str,
) -> str:
    """Wire a provisioned media container into the media stack.

    Steps 2-4 of the original media dispatcher, factored out so the
    generic service-manifest dispatcher's ``media_connect`` integration
    hook runs the EXACT same code path (2026-07-18 apps-as-data design):
    reachability poll + managed-credential login, per-user
    ``user_media_servers`` row, catalog sync enqueue.
    """
    # 2) Reachable on the shared Docker network at the container alias
    #    (ServiceManager names it augmentum-<id> and aliases <id>).
    base_url = f"http://augmentum-{service_id}:{sd.internal_port}"

    # 3) Poll until the server answers, logging in with the managed
    #    credential we baked into the container (empty creds only for
    #    services that explicitly opt out of managed auth). Reuse the same
    #    provider client the manual add-server flow uses so connection
    #    semantics match; the returned token is what we persist + the app
    #    sends on every call.
    from augmentum.providers.service_auth import (
        managed_service_credentials,
        needs_first_run_setup,
        needs_managed_auth,
    )
    if needs_managed_auth(sd) or needs_first_run_setup(sd):
        auth_user, auth_pass = managed_service_credentials(service_id)
    else:
        auth_user, auth_pass = "", ""

    http = getattr(request.app.state, "http_client", None)
    token = ""
    reachable = False
    if http is not None:
        from augmentum.proxy.media_routes import _provider_client
        try:
            client = _provider_client(provider, http)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # First-run-wizard servers (Jellyfin) must bootstrap an admin
        # account before login works, and boot slower — give them a longer
        # budget. The wizard is idempotent, so retrying it is safe.
        do_wizard = needs_first_run_setup(sd) and hasattr(client, "first_run_setup")
        attempts = 40 if do_wizard else 20  # ~120s vs ~60s
        for _attempt in range(attempts):
            try:
                if do_wizard:
                    await client.first_run_setup(base_url, auth_user, auth_pass)
                token = await client.login(base_url, auth_user, auth_pass) or ""
                reachable = True
                break
            except Exception:  # noqa: BLE001 — not up yet; retry
                await _asyncio.sleep(3)

    # 4) Auto-create the per-user connection (idempotent on re-install).
    store = _media_server_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Media store unavailable")
    existing = await store.find_match(
        user_id=user_id, provider=provider, base_url=base_url,
    )
    from augmentum.media.sync import enqueue_media_sync

    if existing is not None:
        await store.update(
            existing.id, user_id=user_id,
            status="ok" if reachable else "untested",
            access_token=token if reachable else None,
        )
        if reachable:
            # Re-index on re-provision so the catalog is current.
            await enqueue_media_sync(
                request.app.state, user_id=user_id, server_id=existing.id,
            )
        return existing.id

    server = await store.create(
        user_id=user_id, provider=provider, name=sd.name,
        base_url=base_url, access_token=token,
    )
    await store.update(
        server.id, user_id=user_id,
        status="ok" if reachable else "untested",
        status_detail="" if reachable else "Container started; first scan pending.",
    )
    if reachable:
        # Auto-index the catalog into file_index now, so "play <title>" works
        # immediately after provision instead of sitting at 0 items until a
        # manual Sync.
        await enqueue_media_sync(
            request.app.state, user_id=user_id, server_id=server.id,
        )
    else:
        log.warning(
            "media_server_unreachable_after_provision",
            service_id=service_id, base_url=base_url,
        )
    return server.id


async def _uninstall_media_server(
    request: Request, artifact: Any, user_id: str,
) -> dict[str, Any]:
    """Tear down a provisioned media server — the inverse of
    :func:`_install_media_server`.

    Symmetric with install (which is admin-only): remove the caller's
    ``user_media_servers`` connection row + purge its cached library, then
    stop the shared container. The route clears the marketplace install
    record. ``disable_service`` is install-wide, so this fully uninstalls
    the server for everyone — which is why the route flags every user's
    install record, keeping the catalog honest.
    """
    from augmentum.auth.guards import is_admin
    if not is_admin(request):
        raise HTTPException(
            status_code=403, detail="Media-server uninstall is admin-only.",
        )
    if not isinstance(artifact, dict):
        raise HTTPException(
            status_code=400, detail="Media artifact must be a JSON object",
        )
    service_id = str(artifact.get("service_id") or "").strip()
    provider = str(artifact.get("provider") or service_id).strip()
    if not service_id:
        raise HTTPException(
            status_code=400, detail="install_payload.service_id is required",
        )

    # 1) Remove the caller's connection row(s) for this provider and purge
    #    the cached library FIRST — once the row is gone the orphan
    #    file_index rows can't be found by server_id, and they'd 502 on
    #    open. Guard on ownership before purge so a shared row owned by
    #    another admin is never touched.
    removed = 0
    store = _media_server_store(request)
    idx = getattr(request.app.state, "file_index", None)
    if store is not None:
        from augmentum.media.store import purge_server_data
        try:
            rows = await store.list_visible(user_id=user_id)
        except Exception:
            log.warning("media_server_uninstall_list_failed", exc_info=True)
            rows = []
        for s in rows:
            if s.provider != provider or s.user_id != user_id:
                continue
            try:
                if idx is not None:
                    await purge_server_data(idx._db, s.id, user_id=user_id)
                await store.delete(s.id, user_id=user_id)
                removed += 1
            except Exception:
                log.warning(
                    "media_server_uninstall_row_failed",
                    server_id=s.id, exc_info=True,
                )

    # 2) Stop the shared container (install-wide). Best-effort — a missing
    #    or already-stopped container shouldn't block clearing the user's
    #    connection + install record.
    mgr = getattr(request.app.state, "service_manager", None)
    stopped = False
    if mgr is not None:
        try:
            await mgr.disable_service(service_id)
            stopped = True
        except Exception:
            log.warning(
                "media_server_uninstall_disable_failed",
                service_id=service_id, exc_info=True,
            )

    return {
        "removed_connections": removed,
        "service_stopped": stopped,
        "service_id": service_id,
    }


async def _uninstall_provider_service(
    request: Request, artifact: Any, user_id: str,
) -> dict[str, Any]:
    """Tear down a Docker-managed provider service — the inverse of
    :func:`_install_provider_service`.

    Admin-only, symmetric with install (stopping a shared container is
    install-wide). Stops the container AND drops the audio/image provider
    row it registered, so the Discover card, the picker, and the running
    container all agree the service is gone. Without this dispatcher the
    uninstall route would clear only the install record and leave the
    container running — a half-uninstall that lies to the user.
    """
    from augmentum.auth.guards import is_admin
    if not is_admin(request):
        raise HTTPException(
            status_code=403,
            detail="Provider service uninstall is admin-only.",
        )
    if not isinstance(artifact, dict):
        raise HTTPException(
            status_code=400, detail="Provider artifact must be a JSON object",
        )
    service_id = str(artifact.get("service_id") or "").strip()
    if not service_id:
        raise HTTPException(
            status_code=400, detail="install_payload.service_id is required",
        )

    # 1) Stop the shared container (install-wide). Best-effort — an already
    #    stopped/missing container shouldn't block clearing the record.
    mgr = getattr(request.app.state, "service_manager", None)
    stopped = False
    if mgr is not None:
        try:
            await mgr.disable_service(service_id)
            stopped = True
        except Exception:
            log.warning(
                "provider_service_uninstall_disable_failed",
                service_id=service_id, exc_info=True,
            )

    # 2) Drop the provider row it registered so it leaves the pickers.
    try:
        from augmentum.providers.provider_bridge import (
            deregister_installed_service_provider,
        )
        await deregister_installed_service_provider(
            request.app.state, service_id,
        )
    except Exception:
        log.warning(
            "provider_bridge_deregister_failed",
            service_id=service_id, exc_info=True,
        )

    return {"service_stopped": stopped, "service_id": service_id}


# ── Registry — maps install_via to dispatcher ────────────────────────


async def _install_service_manifest(
    request: Request, artifact: Any, user_id: str,
) -> str:
    """Generic installer for ``kind: "service"`` manifest listings.

    ONE dispatcher for every service app (T2 — apps as data): validate
    manifest -> resource gate -> runtime ServiceDefinition -> provision
    via ServiceManager (same engine as media servers) -> health poll ->
    integration hooks. Spec: docs/superpowers/specs/
    2026-07-18-marketplace-service-os-design.md.
    """
    from augmentum.auth.guards import is_admin
    from augmentum.marketplace.manifest import (
        ManifestError,
        parse_manifest,
        to_service_definition,
    )

    if not is_admin(request):
        raise HTTPException(
            status_code=403, detail="Service install is admin-only.",
        )
    try:
        manifest = parse_manifest(artifact)
    except ManifestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    mgr = getattr(request.app.state, "service_manager", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="Service manager unavailable")

    # Resource gate: refuse with an honest message rather than letting a
    # heavy app OOM the box mid-pull. Declared need + 512MB slack must
    # fit in currently-available RAM. Best-effort — an unreadable
    # /proc/meminfo (non-Linux dev host) skips the gate, never blocks.
    if manifest.ram_mb:
        try:
            available_mb = 0
            with open("/proc/meminfo", encoding="ascii") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        available_mb = int(line.split()[1]) // 1024
                        break
            if available_mb and manifest.ram_mb + 512 > available_mb:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"{manifest.name} wants {manifest.ram_mb}MB RAM but only "
                        f"{available_mb}MB is available. Free memory or stop "
                        f"another service, then retry."
                    ),
                )
        except HTTPException:
            raise
        except Exception:  # noqa: BLE001 — gate is best-effort
            pass

    # Runtime definition — the seam that makes the catalog data-driven.
    sd = to_service_definition(manifest)

    # HTTPS front door: every service listing promises a browser
    # experience (manifest T4 gate), so every install gets a TLS door.
    # Manifests don't hand-pick ports (community collisions would be
    # inevitable) — allocate from the reserved range against everything
    # already claimed. Exhausted range degrades to no door, loudly.
    if not sd.https_port:
        from dataclasses import replace as _dc_replace

        from augmentum.providers.caddy_front_door import (
            allocate_front_door_port,
            claimed_snippet_ports,
        )
        used = {
            getattr(d, "https_port", 0) or 0
            for d in mgr.catalog.list_all()
        }
        used.discard(0)
        # Also include ports claimed by actual snippet files on disk —
        # pre-persistence installs have https_port=0 in memory but their
        # caddy snippets still bind real ports. Without this the allocator
        # hands a "free" port to a new install that collides with an old
        # snippet (n8n→6804 was really Navidrome).
        used |= set(claimed_snippet_ports())
        port = allocate_front_door_port(used)
        if port:
            sd = _dc_replace(sd, https_port=port)
        else:
            log.warning(
                "service_front_door_range_exhausted",
                service_id=manifest.service_id,
            )
    mgr.catalog.register_runtime(sd)
    sd = mgr.get_definition(manifest.service_id) or sd

    # Optional host-bind for the app's library dir (same pattern as media).
    options = (artifact.get("_install_options") or {}) if isinstance(artifact, dict) else {}
    volume_overrides: dict[str, str] = {}
    host_path = str(options.get("media_host_path") or "").strip()
    if manifest.media_mount and host_path:
        if not _looks_like_host_path(host_path):
            raise HTTPException(
                status_code=400,
                detail="Library folder must be an absolute host path "
                       "(e.g. /mnt/media or C:\\Media).",
            )
        volume_overrides[manifest.media_mount] = host_path

    # env_prompts answers arrive via _install_options.env — only keys the
    # manifest declared are honored (no arbitrary env injection).
    env_overrides: dict[str, str] = {}
    raw_env = options.get("env") or {}
    if isinstance(raw_env, dict):
        allowed = {pr.key for pr in manifest.env_prompts}
        env_overrides = {
            str(k): str(v) for k, v in raw_env.items() if str(k) in allowed
        }
    # Machine secrets (generate=True prompts) the user left blank get a strong
    # random value minted here — otherwise images that hard-require a session
    # key / API pepper (flatnotes, homebox) crash-loop on a fresh install. The
    # value persists via enable_service's env_overrides, so restore re-uses the
    # same secret (recreating it would invalidate issued keys/sessions).
    import secrets as _secrets
    for pr in manifest.env_prompts:
        if pr.generate and not env_overrides.get(pr.key):
            env_overrides[pr.key] = _secrets.token_urlsafe(48)

    # Augmentum-as-backend: if the manifest declares the ``augmentum_backend``
    # integration, mint a chat-scoped API key for this user and inject
    # Augmentum's own OpenAI-compatible base URL + that key into the service's
    # env — so the app uses the user's models/voices with ZERO setup. Must run
    # PRE-provision (env is baked at container create); the augmentum_backend
    # hook revokes the key on uninstall. See hooks/augmentum_backend.py.
    _backend_key_id = ""
    _backend_cfg = manifest.integration.get("augmentum_backend") or {}
    if _backend_cfg:
        from augmentum.marketplace.hooks.augmentum_backend import (
            AUGMENTUM_INTERNAL_BASE_URL,
        )
        akm = getattr(request.app.state, "api_key_manager", None)
        if akm is not None:
            try:
                # STABLE per-(service,user) key: derived deterministically so a
                # reinstall regenerates the SAME key. The app persists this key
                # in its own config volume; re-deriving + re-ensuring keeps them
                # matched across uninstall/reinstall (otherwise the app 401s on a
                # revoked old key — the PersistentConfig gotcha). Same approach
                # as media-server managed creds (providers/service_auth.py).
                from augmentum.utils.secrets import derive_secret
                _stable = "sk-aug-" + derive_secret(
                    f"backend-key:{manifest.service_id}:{user_id}", length=48,
                )
                raw_key, meta = await akm.ensure(
                    user_id, f"{manifest.name} (Discover)", _stable, "chat",
                )
                _backend_key_id = str(meta.get("id") or "")
                for env_name in _backend_cfg.get("base_url_env") or []:
                    env_overrides[str(env_name)] = AUGMENTUM_INTERNAL_BASE_URL
                for env_name in _backend_cfg.get("api_key_env") or []:
                    env_overrides[str(env_name)] = raw_key
                log.info(
                    "augmentum_backend_wired",
                    service_id=manifest.service_id, key_id=_backend_key_id,
                )
            except Exception:  # noqa: BLE001 — never fail an install over auto-wire
                log.warning(
                    "augmentum_backend_mint_failed",
                    service_id=manifest.service_id, exc_info=True,
                )

    # Optional host-RAM ceiling, chosen by the user in the install sheet's
    # advanced section. Augmentum sets no default: what a third-party service
    # needs depends on how it's used and by how many people, which we can't
    # know. Passed through rather than pre-persisted — the managed_services row
    # doesn't exist yet, so a config_json write here would match zero rows and
    # lose the choice. enable_service applies it, then persists it.
    _mem_limit = str(options.get("mem_limit") or "").strip().lower()
    if _mem_limit:
        from augmentum.providers.manager import _parse_size
        try:
            if _parse_size(_mem_limit) < 64 * 1024**2:
                raise ValueError("Memory limit must be at least 64m.")
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc) if "at least" in str(exc) else
                f"Couldn't read '{_mem_limit}' as a memory size. "
                "Use a number with a unit, e.g. 512m or 2g.",
            ) from exc

    try:
        await mgr.enable_service(
            manifest.service_id,
            volume_overrides=volume_overrides or None,
            env_overrides=env_overrides or None,
            mem_limit=_mem_limit or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.warning(
            "service_manifest_provision_failed",
            service_id=manifest.service_id, error=str(exc),
        )
        raise HTTPException(
            status_code=500, detail=f"Couldn't start {manifest.name}: {exc}",
        ) from exc

    # Persist runtime facts the DB schema has no columns for, so boot
    # rehydration (marketplace/runtime_rehydrate.py) can rebuild this
    # runtime definition with the SAME front door instead of allocating
    # a new port and orphaning the caddy snippet written above.
    try:
        await mgr.update_config_json(manifest.service_id, {
            "manifest_service": True,
            "https_port": int(getattr(sd, "https_port", 0) or 0),
            # Stash the auto-wire key id so augmentum_backend._uninstall can
            # revoke it (the hook runs post-install and has no other handle on it).
            **({"augmentum_backend_key_id": _backend_key_id} if _backend_key_id else {}),
        })
    except Exception:  # noqa: BLE001 — never fail a live install over bookkeeping
        log.warning(
            "service_manifest_config_persist_failed",
            service_id=manifest.service_id, exc_info=True,
        )

    # Integration hooks — each optional, each isolated: a hook failure
    # leaves a RUNNING app minus one integration, loudly, never a dead
    # install. Hooks dispatch through the registry in
    # augmentum/marketplace/hooks/ — no per-hook code in this dispatcher.
    result_id = manifest.service_id
    from augmentum.marketplace.hooks import KNOWN_INTEGRATION_HOOKS
    for hook_name in manifest.integration:
        pair = KNOWN_INTEGRATION_HOOKS.get(hook_name)
        if pair is None:
            # Forward compat: newer manifests on older servers.
            log.warning(
                "service_manifest_hook_unregistered",
                service_id=manifest.service_id, hook=hook_name,
            )
            continue
        _install_hook = pair[0]
        try:
            await _install_hook(request, manifest, sd, user_id)
        except HTTPException:
            raise
        except Exception:  # noqa: BLE001
            log.warning(
                "service_manifest_hook_failed",
                service_id=manifest.service_id, hook=hook_name,
                exc_info=True,
            )
    # No blocking health poll here: the install response returns as soon
    # as the container is provisioned, and the post-install card polls
    # GET /api/marketplace/services/{id}/status live — progress over a
    # multi-minute stalled "Installing…" button (image pulls for heavy
    # apps run 90-180s+). The status endpoint probes the real container,
    # so the verdict stays honest without holding the request open.

    log.info(
        "service_manifest_installed",
        service_id=manifest.service_id, image=manifest.image,
        hooks=sorted(manifest.integration),
    )
    return result_id


async def _uninstall_service_manifest(
    request: Request, artifact: Any, user_id: str,
) -> None:
    """Tear down a manifest-installed service. Volumes are PRESERVED —
    deleting user data requires an explicit choice on a dedicated
    surface, never an uninstall side effect.

    Hooks fire in REVERSE order BEFORE the container is stopped, so
    each hook can clean up its per-service rows while the service is
    still reachable (e.g. removing webhooks from Uptime Kuma)."""
    from augmentum.auth.guards import is_admin
    from augmentum.marketplace.manifest import ManifestError, parse_manifest

    if not is_admin(request):
        raise HTTPException(
            status_code=403, detail="Service uninstall is admin-only.",
        )
    try:
        manifest = parse_manifest(artifact)
    except ManifestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    mgr = getattr(request.app.state, "service_manager", None)
    if mgr is None:
        raise HTTPException(status_code=503, detail="Service manager unavailable")

    # Resolve the running definition so hooks can reach the container.
    sd = mgr.get_definition(manifest.service_id)

    # Teardown hooks in reverse install order — each best-effort.
    from augmentum.marketplace.hooks import KNOWN_INTEGRATION_HOOKS
    for hook_name in reversed(list(manifest.integration)):
        pair = KNOWN_INTEGRATION_HOOKS.get(hook_name)
        if pair is None:
            continue
        _uninstall_hook = pair[1]
        try:
            await _uninstall_hook(request, manifest, sd, user_id)
        except Exception:  # noqa: BLE001
            log.warning(
                "service_manifest_hook_uninstall_failed",
                service_id=manifest.service_id, hook=hook_name,
                exc_info=True,
            )

    await mgr.disable_service(manifest.service_id)
    log.info("service_manifest_uninstalled", service_id=manifest.service_id)


async def _install_bundle(
    request: Request, artifact: Any, user_id: str,
) -> str:
    """Install every member of a ``kind: "bundle"`` listing (spec T1 —
    a profile IS a bundle: Core/Standard/Full and any curated pack are
    just member lists over the same app units).

    Members install through their OWN dispatchers in declared order.
    Per-member isolation: one failure records and continues — a bundle
    is a shopping list, not a transaction. The summary string names
    what succeeded and what didn't so the UI can render an honest
    receipt (never a silent partial success).
    """
    if not isinstance(artifact, dict):
        raise HTTPException(status_code=400, detail="Bundle payload must be an object")
    members = artifact.get("members")
    if not isinstance(members, list) or not members:
        raise HTTPException(
            status_code=400, detail="Bundle needs a non-empty 'members' list",
        )
    store = getattr(request.app.state, "marketplace_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Marketplace unavailable")

    options = artifact.get("_install_options") or {}
    installed: list[str] = []
    failed: list[tuple[str, str]] = []
    for member_id in [str(m) for m in members]:
        listing = await store.get(member_id)
        if listing is None:
            failed.append((member_id, "not in catalog"))
            continue
        if listing.kind == "bundle":
            # No nested bundles — cycles and surprise fan-out both die here.
            failed.append((member_id, "nested bundles are not allowed"))
            continue
        dispatcher = get_dispatcher(listing.install_via)
        if dispatcher is None:
            failed.append((member_id, f"no installer for {listing.install_via}"))
            continue
        member_artifact = {
            **(listing.install_payload or {}),
            # Per-member options: bundle-level options may carry a dict
            # keyed by member id (e.g. Navidrome's library path).
            "_install_options": (
                options.get(member_id) if isinstance(options.get(member_id), dict)
                else {}
            ),
        }
        try:
            await dispatcher(request, member_artifact, user_id)
            installed.append(member_id)
        except HTTPException as exc:
            failed.append((member_id, str(exc.detail)[:160]))
        except Exception as exc:  # noqa: BLE001 — isolate members
            log.warning(
                "bundle_member_install_failed",
                bundle_member=member_id, error=str(exc), exc_info=True,
            )
            failed.append((member_id, str(exc)[:160]))

    log.info(
        "bundle_installed",
        installed=installed, failed=[f[0] for f in failed],
    )
    if failed and not installed:
        raise HTTPException(
            status_code=500,
            detail="Bundle install failed: " + "; ".join(
                f"{mid}: {why}" for mid, why in failed[:4]
            ),
        )
    summary = f"installed {len(installed)}/{len(members)}"
    if failed:
        summary += " — failed: " + ", ".join(
            f"{mid} ({why})" for mid, why in failed[:4]
        )
    return summary


async def _install_service_staged(
    request: Request, artifact: Any, user_id: str,
) -> str:
    """Staged install for slow/engine services — returns a JOB id, not a resource.

    The Augmentum Service Install Standard (spec 2026-07-22): fast-validate here
    (admin + manifest parse, so bad manifests still 400 immediately), then hand
    the slow work (pull + start + health + backend registration) to the
    ``service_install`` background job so the Discover card can show honest staged
    progress. Returns the job_id; the install route surfaces it and the card
    polls ``/api/jobs/{id}``.

    OPT-IN via ``install_via: "service_staged"`` — the generic ``service_manifest``
    dispatcher and every already-shipped listing are untouched. First consumer:
    the vLLM engine. The listing's ``install_payload`` may carry a top-level
    ``backend`` block ({key, internal_port, path}) to register an OpenAI-compatible
    engine backend once healthy.
    """
    from augmentum.auth.guards import is_admin
    from augmentum.marketplace.manifest import ManifestError, parse_manifest

    if not is_admin(request):
        raise HTTPException(status_code=403, detail="Service install is admin-only.")

    # Fast validation — surface a bad manifest as a 400 before we enqueue.
    manifest_payload = {k: v for k, v in artifact.items() if k != "_install_options"} \
        if isinstance(artifact, dict) else artifact
    try:
        parse_manifest(manifest_payload)
    except ManifestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    jobs_store = getattr(request.app.state, "jobs_store", None)
    job_runner = getattr(request.app.state, "job_runner", None)
    if jobs_store is None:
        raise HTTPException(status_code=503, detail="Job queue unavailable")

    backend = artifact.get("backend") if isinstance(artifact, dict) else None
    job_id = await jobs_store.create(
        user_id=user_id,
        job_type="service_install",
        payload={"manifest": manifest_payload, "backend": backend},
    )
    if job_runner is not None:
        job_runner.wake()
    return job_id


async def _install_addon(
    request: Request, artifact: Any, user_id: str,
) -> str:
    """Install an ADD-ON — a capability image, not a service. Returns a JOB id.

    Add-ons are the category for containers that extend Augmentum's own
    capabilities rather than standing up a product the user opens (see
    ``augmentum/addons/__init__.py`` for the full distinction). They are
    built locally from recipes in this repo, so the manifest equivalent of
    image pinning is PINNED BUILD ARGS, which live in
    ``augmentum.addons.catalog`` rather than in the listing — the listing
    carries only presentation copy plus the add-on id.

    Nothing is taken from request input beyond that id: the Dockerfile,
    image tag and build args are all catalog-resolved. That is what makes
    enabling BUILD on the docker-socket-proxy defensible.
    """
    from augmentum.addons.catalog import addon_by_id
    from augmentum.auth.guards import is_admin

    if not is_admin(request):
        raise HTTPException(status_code=403, detail="Add-on install is admin-only.")

    addon_id = ""
    if isinstance(artifact, dict):
        addon_id = str(artifact.get("addon_id") or "").strip()
    spec = addon_by_id(addon_id) if addon_id else None
    if spec is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown add-on {addon_id!r}. The listing's install_payload "
                   f"must carry an 'addon_id' from augmentum/addons/catalog.py.",
        )
    if not spec.user_facing:
        raise HTTPException(
            status_code=400,
            detail=f"{spec.title} is a dependency of other add-ons and is "
                   f"installed automatically — it can't be installed directly.",
        )

    # License acknowledgement. Only this category needs it: the user is the
    # builder, so where a recipe compiles GPL software or installs a
    # proprietary browser, the acceptance has to be theirs and explicit.
    if spec.license_notice:
        options = artifact.get("_install_options") if isinstance(artifact, dict) else None
        acknowledged = bool((options or {}).get("license_acknowledged"))
        if not acknowledged:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "license_acknowledgement_required",
                    "addon_id": spec.id,
                    "notice": spec.license_notice,
                },
            )

    jobs_store = getattr(request.app.state, "jobs_store", None)
    job_runner = getattr(request.app.state, "job_runner", None)
    if jobs_store is None:
        raise HTTPException(status_code=503, detail="Job queue unavailable")

    job_id = await jobs_store.create(
        user_id=user_id,
        job_type="addon_install",
        payload={"addon_id": spec.id},
    )
    if job_runner is not None:
        job_runner.wake()
    return job_id


async def _uninstall_addon(
    request: Request, artifact: Any, user_id: str,
) -> dict[str, Any]:
    """Remove an add-on: drop the anchor, reclaim the disk, kill the capability.

    Uninstall genuinely reclaims what install spent — the honest counterpart
    to a 25-minute, 2.3GB install. The shared streaming base is refcounted in
    ``registry.remove_addon`` so it survives until the last add-on that
    builds FROM it is gone.
    """
    from augmentum.addons.catalog import addon_by_id
    from augmentum.addons.registry import remove_addon
    from augmentum.auth.guards import is_admin

    if not is_admin(request):
        raise HTTPException(status_code=403, detail="Add-on uninstall is admin-only.")

    addon_id = ""
    if isinstance(artifact, dict):
        addon_id = str(artifact.get("addon_id") or "").strip()
    if addon_by_id(addon_id) is None:
        raise HTTPException(status_code=400, detail=f"Unknown add-on {addon_id!r}")

    try:
        return await remove_addon(addon_id, app_state=request.app.state)
    except Exception as exc:  # noqa: BLE001
        log.warning("addon_uninstall_failed", addon=addon_id, error=str(exc))
        raise HTTPException(
            status_code=500, detail=f"Add-on removal failed: {exc}",
        ) from exc


DISPATCHER_REGISTRY: dict[str, Any] = {
    "service_manifest": _install_service_manifest,
    # Add-ons: capability images built locally. Distinct from every service
    # path above — no port, no healthcheck, no long-running container.
    "addon_build": _install_addon,
    "service_staged": _install_service_staged,
    "bundle": _install_bundle,
    "community-character": _install_character,
    "community-flow": _install_reasoning_flow,
    "community-power": _install_power,
    "community-knowledge": _install_knowledge_pack,
    "provider-service": _install_provider_service,
    "media-server": _install_media_server,
}

# install_via → uninstall dispatcher. Sparse: only kinds that provision a
# real backing resource (a container + connection) need teardown. Listings
# without an entry just get their install record cleared by the route.
UNINSTALL_DISPATCHER_REGISTRY: dict[str, Any] = {
    "service_manifest": _uninstall_service_manifest,
    # Staged installs provision the same managed container via service_manager,
    # so they uninstall through the identical path — without this the Discover
    # uninstall button was a no-op for staged engines (e.g. vLLM).
    "service_staged": _uninstall_service_manifest,
    # Add-on teardown removes the image + anchor, not a container.
    "addon_build": _uninstall_addon,
    "media-server": _uninstall_media_server,
    "provider-service": _uninstall_provider_service,
}


def get_dispatcher(install_via: str):
    """Return the dispatcher callable for an install_via key, or None.

    The discover_routes install endpoint uses this to route; unknown
    install_via values mean the listing is for a Source-backed installer
    (js13k, agsp-profile, internal) handled by TitleService, not by
    one of our dispatchers.
    """
    return DISPATCHER_REGISTRY.get(install_via)


def get_uninstall_dispatcher(install_via: str):
    """Return the teardown callable for an install_via key, or None.

    None means there's no backing resource to tear down — the route still
    clears the marketplace install record so the card flips to 'Install'.
    """
    return UNINSTALL_DISPATCHER_REGISTRY.get(install_via)
