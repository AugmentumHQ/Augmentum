"""``service_install`` job handler — staged, progress-reporting service install.

This is the reusable core of the Augmentum Service Install Standard (spec:
docs/superpowers/specs/2026-07-22-unsupported-arch-serving-vllm-safetensors-
design.md). It wraps ServiceManager provisioning in the background-job queue so a
slow install (a big image pull, an engine that must warm up) shows honest staged
progress instead of an opaque spinner, and so an in-flight install survives a
server restart.

Contract (the standard every NEW service adopts):
  preparing → downloading image → starting → warming up (health) →
  registering backend (engines only) → ready.

"Ready" means USABLE — the job only completes after the healthcheck passes AND,
for an engine service, the OpenAI-compatible backend is registered.

Rollout note: this path is OPT-IN via ``install_via: "service_staged"`` — the
generic ``service_manifest`` dispatcher and every already-shipped listing are
untouched (Matt tested all 48; they don't re-run through here). vLLM is the first
consumer.

Payload:
    {
      "manifest":  <install_payload dict>,   # same shape service_manifest gets
      "backend":   {                          # optional — engine services only
          "key": "vllm",                      # provider_registry key to register
          "internal_port": 8080,              # container port the OpenAI API is on
          "path": "/v1"                       # base-url suffix
      }
    }

Idempotency: ``enable_service`` is idempotent (it reuses an existing container),
so a restart re-entry finishes provisioning rather than duplicating it.
"""

from __future__ import annotations

import asyncio
from typing import Any

from augmentum.jobs.context import JobContext, JobRetryable
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Health-poll ceiling. llama-swap (the vLLM front door) answers /health as soon
# as the container is up — it doesn't load a model until first request — so this
# is generous slack for a cold container start, not a model load.
_HEALTH_TIMEOUT_S = 180
_HEALTH_INTERVAL_S = 2.0


async def _augmentum_model_dir_mounts(mgr) -> dict[str, str]:
    """Map each configured model dir (container path) → its Docker-host source.

    The vLLM engine must READ the safetensors models the user downloads through
    the model manager, which land in the user's model dirs (multi-drive: e.g.
    /models/host, /models/spare). Those dirs are bind-mounted / volume-backed
    into the Augmentum container. We mirror THOSE mounts onto the engine
    container so both see identical paths — llama-swap can then serve
    ``vllm serve /models/host/<repo>``. Deployment-agnostic: we read whatever
    Augmentum itself has mounted (host binds on WSL, named volumes on a default
    install), never hardcoded paths. Returns {} on any inspection failure — the
    engine still installs, just without model-dir access (surfaced in logs).
    """
    import socket

    from augmentum.config import settings

    dirs: list[str] = []
    for d in (settings.engine_model_dir, settings.llamacpp_model_dir):
        if d and d not in dirs:
            dirs.append(d)
    for d in (settings.engine_extra_model_dirs or "").split(";"):
        d = d.strip()
        if d and d not in dirs:
            dirs.append(d)
    if not dirs:
        return {}

    try:
        own = await mgr._docker.containers.get(socket.gethostname())
        info = await own.show()
    except Exception as exc:  # noqa: BLE001
        log.warning("service_install_self_inspect_failed", error=str(exc))
        return {}

    mounts = info.get("Mounts") or []
    # Longest-destination-prefix wins so /data/models matches its own mount
    # before falling through to a parent /data mount.
    m_sorted = sorted(mounts, key=lambda m: len(m.get("Destination", "")), reverse=True)
    result: dict[str, str] = {}
    for d in dirs:
        for m in m_sorted:
            dest = (m.get("Destination") or "").rstrip("/")
            src = (m.get("Source") or "").rstrip("/")
            if not dest or not src:
                continue
            if d == dest:
                result[d] = src
                break
            if d.startswith(dest + "/"):
                result[d] = src + d[len(dest):]
                break
    return result


def _resolve_model_env(sd, model_choice: str) -> dict[str, str]:
    """Map a chosen model id to its env overrides from requirements.model.

    Each choice is self-describing (carries its own ``env`` block — repo,
    checkpoint paths), so this stays provider-agnostic: no fish-specific
    logic here. Falls back to the declared default; empty if the service
    offers no model choice.
    """
    reqs = getattr(sd, "requirements", None) or {}
    model_req = reqs.get("model") if isinstance(reqs, dict) else None
    if not isinstance(model_req, dict):
        return {}
    choices = model_req.get("choices") or []
    want = model_choice or str(model_req.get("default") or "")
    for c in choices:
        if isinstance(c, dict) and c.get("id") == want and isinstance(c.get("env"), dict):
            return {str(k): str(v) for k, v in c["env"].items()}
    return {}


async def _install_provider_staged(
    ctx: JobContext, app, mgr, service_id: str, *, model_choice: str = "",
) -> dict[str, Any]:
    """Staged install for a catalog provider service (STT/TTS/etc.).

    Same honest-progress contract as the manifest path — preparing →
    downloading image → warming up (health) → registering provider → ready —
    so provider services match the service_staged UX instead of the old
    blocking, spinner-only install. ``enable_service`` is idempotent, so a
    restart re-entry (or an image already pulled by a prior blocking attempt)
    finishes provisioning rather than duplicating it.
    """
    from augmentum.providers.models import ServiceStatus

    await ctx.update_progress(0.02, stage="preparing")
    await ctx.check_cancel()
    sd = mgr.get_definition(service_id) if hasattr(mgr, "get_definition") else None
    if sd is None:
        raise ValueError(f"service_install: unknown provider service {service_id!r}")

    # A chosen model (fish-tts S1-mini/S1) selects env overrides — checkpoint
    # paths + the repo the container's self-download entrypoint fetches. These
    # persist on the managed_services row so restore re-applies the same model.
    env_overrides = _resolve_model_env(sd, model_choice) or None

    await ctx.update_progress(0.10, stage="downloading image")
    await ctx.check_cancel()
    try:
        await mgr.enable_service(service_id, env_overrides=env_overrides)
    except Exception as exc:  # noqa: BLE001
        log.warning("service_install_provision_failed", service=service_id, error=str(exc))
        raise JobRetryable(f"provisioning failed: {exc}") from exc

    # Warm up: some providers (fish-tts) download a gated model on first boot,
    # so honor the catalog's start_period as the health ceiling rather than the
    # short engine default. "Ready" waits for real health where it can.
    await ctx.update_progress(0.80, stage="warming up")
    start_period = float(getattr(getattr(sd, "health_check", None), "start_period_s", 0) or 0)
    ceiling = max(_HEALTH_TIMEOUT_S, start_period)
    waited = 0.0
    while waited < ceiling:
        await ctx.check_cancel()
        status = await mgr.get_status(service_id)
        if status == ServiceStatus.RUNNING:
            break
        if status == ServiceStatus.UNHEALTHY and waited > 60:
            raise JobRetryable("service came up unhealthy")
        await asyncio.sleep(_HEALTH_INTERVAL_S)
        waited += _HEALTH_INTERVAL_S

    # Register it as a usable provider (audio_providers row / hot-load) so it
    # lands in the picker — the step the old dispatcher did inline.
    await ctx.update_progress(0.95, stage="registering provider")
    try:
        from augmentum.providers.provider_bridge import (
            register_installed_service_provider,
        )
        await register_installed_service_provider(app.state, service_id)
    except Exception:  # noqa: BLE001
        log.warning("service_install_provider_register_failed", service=service_id, exc_info=True)

    await ctx.update_progress(1.0, stage="ready")
    log.info("service_installed", service=service_id)
    return {"service_id": service_id, "ready": True}


def make_service_install_handler(app):
    """Build the staged service-install handler bound to ``app.state``."""

    async def handler(ctx: JobContext) -> dict[str, Any]:
        # Lazy imports — avoid a circular dep through marketplace → jobs.
        from augmentum.marketplace.manifest import (
            parse_manifest,
            to_service_definition,
        )
        from augmentum.providers.models import ServiceStatus

        payload = ctx.payload or {}

        mgr = getattr(app.state, "service_manager", None)
        if mgr is None:
            raise JobRetryable("service manager not ready")

        # Provider-service path: a catalog service reached by id (no manifest).
        # Same staged contract, so audio/STT/TTS providers (fish-tts, kokoro,
        # …) get the identical progress UX as manifest/engine services.
        service_id_direct = str(payload.get("service_id") or "").strip()
        if service_id_direct:
            return await _install_provider_staged(
                ctx, app, mgr, service_id_direct,
                model_choice=str(payload.get("model_choice") or "").strip(),
            )

        artifact = payload.get("manifest")
        if not isinstance(artifact, dict):
            raise ValueError("service_install: payload.manifest must be an object")

        await ctx.update_progress(0.02, stage="preparing")
        await ctx.check_cancel()

        # Fast prep (parse already validated in the dispatcher; re-parse here so
        # a restart re-entry is self-contained). Register the runtime definition
        # + allocate the HTTPS front-door port, mirroring the generic dispatcher.
        manifest = parse_manifest(artifact)
        sd = to_service_definition(manifest)
        if not sd.https_port:
            from dataclasses import replace as _dc_replace

            from augmentum.providers.caddy_front_door import (
                allocate_front_door_port,
                claimed_snippet_ports,
            )
            used = {getattr(d, "https_port", 0) or 0 for d in mgr.catalog.list_all()}
            used.discard(0)
            used |= set(claimed_snippet_ports())
            port = allocate_front_door_port(used)
            if port:
                sd = _dc_replace(sd, https_port=port)

        # Custom provision (engine services): mirror Augmentum's model dirs onto
        # this container so it can serve models the user downloads to any drive.
        # Gated on backend.mount_model_dirs so generic staged services skip it.
        volume_overrides: dict[str, str] = {}
        backend = payload.get("backend") if isinstance(payload.get("backend"), dict) else None
        if backend and backend.get("mount_model_dirs"):
            from dataclasses import replace as _dc_replace2
            mounts = await _augmentum_model_dir_mounts(mgr)
            if mounts:
                vols = dict(sd.volumes)
                for i, mount_path in enumerate(mounts):
                    vols[f"modeldir{i}"] = mount_path  # name→container path; source via override
                volume_overrides = dict(mounts)  # container path → host source (bind)

                # llama-swap model configs live in a dir ON the primary model dir
                # — which is mirror-mounted into this engine, so Augmentum writes a
                # per-model yaml there and llama-swap (--watch-config) auto-reloads.
                # No exec, no extra shared volume. The dir must exist before the
                # container starts; Augmentum has the model dir mounted, so it can
                # create it here.
                import os as _os

                from augmentum.config import settings as _settings
                primary = (_settings.engine_model_dir or "").rstrip("/")
                cmd = list(sd.command or [])
                if primary and primary in mounts:
                    cfg_dir = f"{primary}/.augmentum-vllm"
                    try:
                        _os.makedirs(cfg_dir, exist_ok=True)
                    except OSError as exc:
                        log.warning("service_install_cfgdir_failed", dir=cfg_dir, error=str(exc))
                    cmd = [
                        "--config", "/config/llama-swap.yaml",
                        "--config-dir", cfg_dir,
                        "--watch-config",
                        "--listen", "0.0.0.0:8080",
                    ]
                sd = _dc_replace2(sd, volumes=vols, command=cmd or None)
                log.info("service_install_model_dirs_mirrored",
                         service=manifest.service_id, dirs=list(mounts.keys()))
            else:
                log.warning("service_install_no_model_dirs",
                            service=manifest.service_id)
        mgr.catalog.register_runtime(sd)

        # Provision: pull image + create + start. The image pull dominates wall
        # time for a large engine image; it's opaque (no layer stream through the
        # proxy), so we bracket it with honest coarse stages rather than fake a
        # smooth bar.
        await ctx.update_progress(0.10, stage="downloading image")
        await ctx.check_cancel()
        try:
            await mgr.enable_service(
                manifest.service_id,
                volume_overrides=volume_overrides or None,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "service_install_provision_failed",
                service=manifest.service_id, error=str(exc),
            )
            # Pull/create failures are often transient (registry blip); let the
            # runner retry within max_attempts.
            raise JobRetryable(f"provisioning failed: {exc}") from exc

        # Warm up: poll until the container reports healthy (or running without a
        # healthcheck). "Ready" must not lie, so we wait for real health.
        await ctx.update_progress(0.80, stage="warming up")
        waited = 0.0
        while waited < _HEALTH_TIMEOUT_S:
            await ctx.check_cancel()
            status = await mgr.get_status(manifest.service_id)
            if status == ServiceStatus.RUNNING:
                break
            if status == ServiceStatus.UNHEALTHY and waited > 30:
                raise JobRetryable("service came up unhealthy")
            await asyncio.sleep(_HEALTH_INTERVAL_S)
            waited += _HEALTH_INTERVAL_S

        # Register the OpenAI-compatible backend (engine services only). The URL
        # is derived from the container name + internal port — the container runs
        # on the shared augmentum network as augmentum-<service_id>.
        backend = payload.get("backend")
        if isinstance(backend, dict) and backend.get("key"):
            await ctx.update_progress(0.95, stage="registering engine")
            key = str(backend["key"])
            internal_port = int(backend.get("internal_port") or manifest.internal_port)
            suffix = str(backend.get("path") or "/v1")
            url = f"http://augmentum-{manifest.service_id}:{internal_port}{suffix}"
            try:
                from augmentum.models.openai_compat import OpenAIBackend

                registry = getattr(app.state, "provider_registry", None)
                client = getattr(app.state, "http_client", None)
                if registry is not None and client is not None:
                    chat_client = getattr(registry, "_chat_http_client", None) or client
                    registry.register_backend(
                        key, OpenAIBackend(client, url, "not-needed", chat_client=chat_client),
                    )
                    log.info("service_install_backend_registered", key=key, url=url)
                # PERSIST for restart survival (F4): runtime register_backend is
                # lost on restart. Store the backend {key,url} in the service's
                # config_json; the boot service-restore step re-registers from it
                # (see server.py::_restore_managed_services). Also set it live on
                # settings for this process.
                try:
                    await mgr.update_config_json(
                        manifest.service_id, {"backend_registration": {"key": key, "url": url}},
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("service_install_backend_persist_failed",
                                key=key, error=str(exc))
                from augmentum.config import settings as _s
                setattr(_s, f"{key}_base_url", url)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "service_install_backend_register_failed",
                    key=key, url=url, error=str(exc),
                )

        await ctx.update_progress(1.0, stage="ready")
        log.info("service_installed", service=manifest.service_id)
        return {"service_id": manifest.service_id, "ready": True}

    return handler
