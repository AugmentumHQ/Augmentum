"""ServiceManager — manages Docker containers for marketplace providers.

Uses the same aiodocker client as coder mode to create, start, stop,
and remove containers for optional services (Ollama, Chatterbox, etc.).
Persists enabled services to SQLite so they survive Augmentum restarts.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import TYPE_CHECKING

from augmentum.providers.catalog import ProviderCatalog
from augmentum.providers.models import (
    ManagedService,
    ServiceCategory,
    ServiceDefinition,
    ServiceStatus,
)
from augmentum.providers.network import ensure_network
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiodocker
    import aiosqlite

log = get_logger(__name__)

_LABEL_MANAGED = "augmentum.managed"
_LABEL_SERVICE_ID = "augmentum.service.id"

# Compose-style ``${VAR}`` / ``${VAR:-default}`` references in catalog env
# values. The ServiceManager creates containers via the Docker API, NOT
# docker-compose, so these are NEVER shell-expanded on their own — passing
# them through verbatim ships a literal "${VAR}" string into the container
# (this is the fish-tts HF_TOKEN bug). Expand them explicitly at build time.
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _augmentum_env_resolution() -> dict[str, str]:
    """Trusted resolution map for ``${AUGMENTUM_*}`` catalog env refs.

    Seeded from live settings so a token set through the Settings UI (which
    applies via ``object.__setattr__(settings, ...)``, not just at boot)
    reaches the container. ``os.environ`` is consulted first in
    :func:`_expand_env_refs`, so a ``.env`` value still wins if both exist.
    """
    from augmentum.config import settings
    return {
        "AUGMENTUM_HUGGINGFACE_TOKEN": (getattr(settings, "huggingface_token", "") or ""),
    }


def _expand_env_refs(value: str, extra: dict[str, str]) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` against os.environ + ``extra``.

    Empty is treated as unset so ``${AUGMENTUM_HUGGINGFACE_TOKEN:-}`` with an
    unset token falls back to the default ("") rather than a literal string.
    """
    def _repl(m: re.Match) -> str:
        var, default = m.group(1), m.group(2)
        val = os.environ.get(var) or extra.get(var) or ""
        if not val and default is not None:
            val = default
        return val
    return _ENV_REF_RE.sub(_repl, value)


class ServiceManager:
    """Manages Docker containers for marketplace provider services."""

    def __init__(
        self,
        docker: aiodocker.Docker,
        db: aiosqlite.Connection | None,
        catalog: ProviderCatalog | None = None,
    ) -> None:
        self._docker = docker
        self._db = db
        self._catalog = catalog or ProviderCatalog()
        self._network_name: str | None = None
        self._health_task: asyncio.Task | None = None

    @property
    def catalog(self) -> ProviderCatalog:
        return self._catalog

    # ------------------------------------------------------------------
    # Container config builder
    # ------------------------------------------------------------------

    async def _prepare_service_volume(self, vol_name: str) -> None:
        """Make a per-service named volume writable by a non-root app user.

        A freshly created named Docker volume is owned ``root:root 0755``.
        An image that runs as a non-root user (n8n runs as ``node``:1000,
        TriliumNext runs non-root) therefore cannot write into its own data
        directory — the container boot-loops on ``EACCES: permission denied``
        the first time it opens its config/data file.

        We open the volume up with a one-shot alpine container that mounts
        the NAMED VOLUME directly and ``chmod -R 0777`` it. Crucially this
        does NOT touch any host ``/var/lib/docker/...`` path — that layout is
        unreachable from the Augmentum container on Docker Desktop / WSL2 and
        behind the socket proxy (the reason the previous host-path approach
        silently fell through, leaving the bare root-owned volume mounted and
        the app crashing). It only uses the Docker volume + container API,
        which the app-container create path already depends on, so if a
        service can be provisioned at all, its volume can be prepared.

        Best-effort: on failure we log and let enable_service fall through to
        the plain named-volume mount (which is exactly today's behaviour), so
        this never makes provisioning worse than it already was.
        """
        init_name = f"augmentum-volinit-{vol_name}"[:63]
        container = None
        try:
            # alpine may not be present on a fresh host — pull best-effort so
            # the init step doesn't fail on a cold cache.
            try:
                await self._docker.images.pull("alpine:latest")
            except Exception:  # noqa: BLE001 — may already be cached / offline
                pass
            container = await self._docker.containers.create_or_replace(
                init_name,
                {
                    "Image": "alpine:latest",
                    # 0777 (not a chown to a specific uid) is deliberate: we
                    # don't know the image's uid generically, and world-write
                    # lets whatever user the app runs as create its files.
                    "Cmd": ["sh", "-c", "chmod -R 0777 /mnt || true"],
                    "HostConfig": {
                        "Binds": [f"{vol_name}:/mnt"],
                        "AutoRemove": False,
                    },
                },
            )
            await container.start()
            await container.wait()
        finally:
            if container is not None:
                try:
                    await container.delete(force=True)
                except Exception:  # noqa: BLE001
                    pass

    @staticmethod
    def _build_container_config(
        sd: ServiceDefinition,
        network_name: str,
        env_overrides: dict[str, str] | None = None,
        volume_overrides: dict[str, str] | None = None,
    ) -> dict:
        """Build an aiodocker-compatible container config from a ServiceDefinition.

        ``env_overrides`` are merged on top of the catalog ``sd.env`` —
        used to inject per-install dynamic values (e.g. managed Basic-auth
        credentials for media servers) that can't live in the static
        catalog. Overrides win on key collision.

        ``volume_overrides`` maps a container mount path → host source,
        replacing that mount's default named volume with a host **bind
        mount**. This is how a provisioned media server points at the
        user's own external library directory instead of opaque
        Docker-managed storage. The source must be a real path on the
        Docker host (the daemon resolves it host-side, not inside the
        Augmentum container).
        """
        merged_env = {**sd.env, **(env_overrides or {})}
        # Expand compose-style ${VAR:-default} refs (Docker API doesn't).
        _extra = _augmentum_env_resolution()
        env_list = [
            f"{k}={_expand_env_refs(str(v), _extra)}" for k, v in merged_env.items()
        ]
        v_over = volume_overrides or {}

        # Bind the published host port to AUGMENTUM_BIND_HOST (default
        # 127.0.0.1) so a provisioned service is NOT exposed on the LAN by
        # default — same posture as the main app. The front door / gate reach
        # the container over the internal Docker network regardless, so this
        # only gates raw host-port exposure. Operators opt into LAN with
        # AUGMENTUM_BIND_HOST=0.0.0.0.
        from augmentum.config import settings
        host_ip = (settings.bind_host or "127.0.0.1").strip() or "127.0.0.1"
        port_bindings = {
            f"{sd.internal_port}/tcp": [
                {"HostIp": host_ip, "HostPort": str(sd.host_port)},
            ],
        }

        host_config: dict = {
            "PortBindings": port_bindings,
            "RestartPolicy": {"Name": "unless-stopped"},
            "LogConfig": {
                "Type": "json-file",
                "Config": {"max-size": "50m", "max-file": "3"},
            },
        }

        # GPU support
        if sd.gpu and sd.gpu.required:
            host_config["DeviceRequests"] = [{
                "Driver": sd.gpu.driver,
                "Count": -1,
                "Capabilities": [["gpu"]],
            }]

        # Volumes: named volumes by default; a mount listed in
        # volume_overrides becomes a host bind mount (host_source:mount).
        binds = []
        for vol_name, mount_path in sd.volumes.items():
            source = v_over.get(mount_path, vol_name)
            binds.append(f"{source}:{mount_path}")
        if binds:
            host_config["Binds"] = binds

        # Shared memory
        if sd.shm_size:
            host_config["ShmSize"] = _parse_size(sd.shm_size)

        # Host-RAM ceiling. Managed services are spawned through the Docker
        # API, so the mem_limits in the compose files never applied to them —
        # every marketplace install ran unbounded and could take the host down
        # on its own (spec §4.1/B3). Manifests may set an explicit limit; the
        # rest fall back to a per-category default sized generously, because a
        # cgroup OOM-kill of a model server reads to the user as a crash.
        mem_bytes = _resolve_mem_limit(sd)
        if mem_bytes > 0:
            host_config["Memory"] = mem_bytes
            # Pin swap to the same value: leaving MemorySwap unset lets Docker
            # grant swap equal to the limit again, so a runaway service gets
            # 2x the ceiling and thrashes the disk instead of being stopped.
            host_config["MemorySwap"] = mem_bytes

        # Health check
        healthcheck = None
        if sd.health_check:
            hc = sd.health_check
            healthcheck = {
                "Test": hc.test,
                "Interval": hc.interval_s * 1_000_000_000,
                "Timeout": hc.timeout_s * 1_000_000_000,
                "Retries": hc.retries,
                "StartPeriod": hc.start_period_s * 1_000_000_000,
            }

        cat_val = sd.category.value if isinstance(sd.category, ServiceCategory) else sd.category
        config: dict = {
            "Image": sd.image,
            "Env": env_list,
            "Labels": {
                _LABEL_MANAGED: "true",
                _LABEL_SERVICE_ID: sd.id,
                "augmentum.service.name": sd.name,
                "augmentum.service.category": cat_val,
            },
            "HostConfig": host_config,
            "ExposedPorts": {f"{sd.internal_port}/tcp": {}},
            "NetworkingConfig": {
                "EndpointsConfig": {
                    network_name: {"Aliases": [sd.id]},
                },
            },
        }

        if healthcheck:
            config["Healthcheck"] = healthcheck

        if sd.command:
            config["Cmd"] = sd.command

        if sd.entrypoint:
            config["Entrypoint"] = sd.entrypoint

        return config

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_network(self) -> str:
        if not self._network_name:
            self._network_name = await ensure_network(self._docker)
        return self._network_name

    def get_definition(self, service_id: str) -> ServiceDefinition | None:
        """Public accessor for a catalog ServiceDefinition by id (the
        post-install provider bridge needs it to know category / ports /
        augmentum_env). Returns None for unknown ids."""
        return self._catalog.get(service_id)

    async def enable_service(
        self,
        service_id: str,
        *,
        volume_overrides: dict[str, str] | None = None,
        env_overrides: dict[str, str] | None = None,
        mem_limit: str | None = None,
    ) -> ManagedService:
        """Enable a catalog service: pull image, create container, start it.

        ``volume_overrides`` (mount_path → host source) turn a default
        named volume into a host bind mount — e.g. pointing a media
        server at the user's external library. They're persisted in the
        managed_services row so ``restore_enabled`` re-applies them when it
        recreates the container after a restart (the path is user input,
        not derivable). Pass ``None`` to reuse whatever was persisted.
        """
        sd = self._catalog.get(service_id)
        if not sd:
            raise ValueError(f"Unknown service: {service_id}")

        # A user-set memory ceiling is applied HERE, on the one path every
        # provision funnels through, rather than at each call site. Install,
        # boot rehydration, credential recreate and version bump all end up in
        # enable_service, so they inherit the limit without four separate
        # patches — and a future provision path can't forget it.
        #
        # An explicit ``mem_limit`` is for FIRST install: the managed_services
        # row doesn't exist until _persist() below, so config_json can't be
        # written yet (update_config_json is an UPDATE and would match zero
        # rows, losing the user's choice silently). It is persisted after the
        # row exists, and every later provision reads it back from there.
        sd = await self._with_mem_limit_override(service_id, sd, explicit=mem_limit)

        # Restore / re-enable with no explicit overrides → reuse persisted.
        if volume_overrides is None:
            volume_overrides = await self._load_persisted_volume_overrides(service_id)
        if env_overrides is None:
            # Same contract as volume_overrides: env answers (manifest
            # env_prompts — service passwords etc.) are user input, not
            # derivable, so restore_enabled / re-enable must re-apply
            # them or a recreated container silently loses its secrets.
            env_overrides = (await self.read_config_json(service_id)).get(
                "env_overrides") or {}

        network = await self._ensure_network()

        # Check for existing managed container
        existing = await self._find_container(service_id)
        if existing:
            info = await existing.show()
            state = info.get("State", {})
            if not state.get("Running"):
                await existing.start()
            # Reconcile the HTTPS front door (idempotent — no-op if already
            # in place). Covers restore-on-startup and re-enable.
            await self._apply_front_door_if_media(sd)
            return await self._to_managed(service_id, sd, existing.id)

        # Pull image
        log.info("service_pulling", service=service_id, image=sd.image)
        try:
            await self._docker.images.pull(sd.image)
        except Exception as exc:
            log.error("service_pull_failed", service=service_id, error=str(exc))
            raise

        # Volume prep: each service gets its OWN named volume (the name is
        # namespaced per-service in manifest.to_service_definition, so there's
        # no cross-service collision). A fresh named volume is root-owned, so
        # before the container starts we chmod it world-writable — otherwise a
        # non-root app user (n8n's node:1000, TriliumNext) boot-loops on
        # EACCES. The volume is mounted DIRECTLY by name (no host-path bind);
        # overrides already set by the caller (media-library host paths) are
        # left untouched — prep only runs for un-overridden named volumes.
        if volume_overrides is None:
            volume_overrides = {}
        # sd.volumes maps {volume_name: mount_path} — same order as
        # _build_container_config reads it. (The previous loop unpacked these
        # reversed, so it prepared the mount path as if it were the volume
        # name; harmless only because the whole step was silently failing.)
        for vol_name, mount_path in sd.volumes.items():
            if mount_path in volume_overrides:
                continue  # caller already specified a host source
            try:
                await self._prepare_service_volume(vol_name)
            except Exception:  # noqa: BLE001 — best-effort, not fatal
                log.warning(
                    "service_volume_prepare_failed",
                    service=service_id, volume=vol_name, exc_info=True,
                )
                # Fall through — the plain named-volume mount is still used
                # (today's behaviour); a non-root app may still hit EACCES,
                # but we never made it worse.

        # Managed-auth services (e.g. media servers that must not run
        # open) get Basic-auth env injected here — derived deterministically
        # so restore_enabled re-creates with the same credential.
        from augmentum.providers.service_auth import managed_auth_env_resolved
        merged_env = {
            **(env_overrides or {}),
            **(await managed_auth_env_resolved(sd, self._db) or {}),
        }
        config = self._build_container_config(
            sd, network,
            env_overrides=merged_env or None,
            volume_overrides=volume_overrides,
        )
        container_name = f"augmentum-{sd.id}"
        try:
            container = await self._docker.containers.create_or_replace(
                container_name, config,
            )
            container_id = container.id if hasattr(container, "id") else container["Id"]
        except Exception as exc:
            log.error("service_create_failed", service=service_id, error=str(exc))
            raise

        # Start
        c = await self._docker.containers.get(container_id)
        await c.start()
        log.info("service_started", service=service_id, container=container_id[:12])

        ms = await self._to_managed(
            service_id, sd, container_id, volume_overrides=volume_overrides,
        )
        await self._persist(ms)
        if mem_limit is not None:
            # Now the row exists, so this UPDATE actually lands. Without it a
            # limit chosen at install time would apply to the first container
            # and then silently disappear on the next restart.
            await self.update_config_json(
                service_id, {"mem_limit": (mem_limit or "").strip().lower()},
            )
        if env_overrides:
            # Rides in config_json next to volume_overrides (see the
            # reuse block above). Stored like other service credentials.
            await self.update_config_json(
                service_id, {"env_overrides": dict(env_overrides)},
            )
        await self._apply_front_door_if_media(sd)
        return ms

    async def _apply_front_door_if_media(self, sd: ServiceDefinition) -> None:
        """Give a browser-facing service its HTTPS front door (Caddy TLS).

        Historically media-only (hence the name, kept for the existing
        call sites/tests); since the 2026-07-18 service-OS work any
        MEDIA or SERVICE definition with an ``https_port`` gets the
        door — "open it in your browser after install" is a listing
        gate (spec T4), so the transport has to exist for every app,
        not just media. No-op without an ``https_port``. Front-door
        failure is non-fatal — the container and the internal
        auto-connect still work — so this never raises.
        """
        cat = sd.category.value if isinstance(sd.category, ServiceCategory) else sd.category
        if cat not in (ServiceCategory.MEDIA.value, ServiceCategory.SERVICE.value):
            return
        from augmentum.providers.caddy_front_door import (
            GATE_MODE_ACCESS,
            GATE_MODE_BASIC,
            GATE_MODE_OFF,
            GATE_MODE_PROXY,
            apply_front_door,
            front_door_port_ok,
            gate_domain,
        )
        from augmentum.providers.service_auth import needs_managed_auth

        # Write a door if EITHER a dedicated port was allocated OR a gate
        # domain is configured (the unbounded <svc>.<gate> door). Bailing on
        # https_port==0 was the bug that dropped every service past the 10-port
        # pool to Caddy's catch-all ("Ollama is running") even with a gate
        # domain set. apply_front_door writes only the door(s) that apply.
        if not front_door_port_ok(getattr(sd, "https_port", 0)) and not gate_domain():
            return

        # Gate mode by service nature (no-op unless a gate domain is set):
        #   media + managed-auth → basic  (login dissolved via injected cred)
        #   service w/ own login → proxy  (straight TLS proxy; forward_auth
        #                                  would break the app's SPA/websockets)
        #   service w/o any auth → access (forward_auth is the only guard)
        #   media w/o managed auth → off  (raw-port door only)
        if cat == ServiceCategory.MEDIA.value and needs_managed_auth(sd):
            mode = GATE_MODE_BASIC
        elif cat == ServiceCategory.SERVICE.value:
            # An app that authenticates its own users ("user_set"/"generated")
            # must NOT sit behind forward_auth — it 302s the app's own XHR/WS to
            # the Augmentum login and shatters the SPA (Open WebUI, n8n, Gitea).
            # Only auth-less apps ("none") need the ACCESS forward_auth guard.
            creds = getattr(sd, "browser_credentials", "none") or "none"
            mode = GATE_MODE_ACCESS if creds == "none" else GATE_MODE_PROXY
        else:
            mode = GATE_MODE_OFF
        live = await apply_front_door(
            self._docker, sd.id, sd.https_port, sd.internal_port, gate_mode=mode,
        )
        if not live:
            log.warning("service_front_door_not_live", service=sd.id, port=sd.https_port)

    async def disable_service(self, service_id: str) -> None:
        """Stop and remove a managed service container."""
        container = await self._find_container(service_id)
        if container:
            try:
                await container.stop()
            except Exception as exc:
                log.debug("service_container_stop_failed", service=service_id, error=str(exc))
            try:
                await container.delete(force=True)
            except Exception as exc:
                log.debug("service_container_delete_failed", service=service_id, error=str(exc))
            log.info("service_disabled", service=service_id)

        # Tear down the HTTPS front door (snippet + reload) regardless of
        # whether the container delete succeeded — a stale snippet would
        # resurrect a 502 listener on the next caddy cold start.
        from augmentum.providers.caddy_front_door import remove_front_door
        await remove_front_door(self._docker, service_id)

        if self._db:
            await self._db.execute(
                "DELETE FROM managed_services WHERE id = ?", (service_id,),
            )
            await self._db.commit()

    async def _with_mem_limit_override(
        self, service_id: str, sd: ServiceDefinition,
        *, explicit: str | None = None,
    ) -> ServiceDefinition:
        """Return ``sd`` with any user-set memory ceiling applied.

        The override lives in ``config_json`` beside env/volume overrides —
        the same contract, for the same reason: it is user input and cannot be
        derived, so anything that recreates the container must re-apply it or
        the ceiling silently vanishes on the next restart.

        ``explicit`` wins when given (first install, before the row exists).
        """
        if explicit is not None:
            import dataclasses

            return dataclasses.replace(sd, mem_limit=(explicit or "").strip().lower())
        try:
            raw = str((await self.read_config_json(service_id)).get("mem_limit") or "")
        except Exception:  # noqa: BLE001 — never block a provision on bookkeeping
            log.warning("mem_limit_override_read_failed", service=service_id,
                        exc_info=True)
            return sd
        if not raw.strip():
            return sd
        import dataclasses

        return dataclasses.replace(sd, mem_limit=raw.strip())

    async def set_mem_limit(self, service_id: str, limit: str) -> ManagedService | None:
        """Set (or clear) a service's host-RAM ceiling and recreate it.

        ``limit`` is compose-style (``"2g"``, ``"512m"``); empty clears the
        ceiling and returns the service to unbounded. Docker cannot change a
        container's memory limit in place on every platform, so this recreates
        it — named data volumes, env/volume overrides and the HTTPS front door
        are all reapplied by :meth:`enable_service`.

        Raises ``ValueError`` on an unparseable value so the caller can tell
        the user, rather than persisting a typo that silently does nothing.
        """
        cleaned = (limit or "").strip().lower()
        if cleaned:
            try:
                parsed = _parse_size(cleaned)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Couldn't read '{limit}' as a memory size. "
                    "Use a number with a unit, e.g. 512m or 2g.",
                ) from exc
            # A ceiling under ~64 MiB won't survive container start for any
            # real image; refuse it here rather than let the user watch it
            # crash-loop and blame the app.
            if parsed < 64 * 1024**2:
                raise ValueError("Memory limit must be at least 64m.")

        await self.update_config_json(service_id, {"mem_limit": cleaned})
        log.info("service_mem_limit_set", service=service_id,
                 limit=cleaned or "unlimited")

        # Only recreate a service that is actually running; setting a limit on
        # a stopped one just persists for its next start.
        if not await self._find_container(service_id):
            return None
        return await self.recreate_with_new_credential(service_id)

    async def recreate_with_new_credential(self, service_id: str) -> ManagedService:
        """Recreate a managed-auth container so a changed credential takes effect.

        Suwayomi bakes its Basic-auth password into ``server.conf`` from
        ``AUTH_PASSWORD`` at boot — there's no runtime change API. After the
        override is persisted, dropping + recreating the container makes the
        entrypoint rewrite the password. Brief downtime while it restarts;
        the persisted volume overrides + front door are reapplied by
        ``enable_service``.
        """
        sd = self._catalog.get(service_id)
        if not sd:
            raise ValueError(f"Unknown service: {service_id}")
        volume_overrides = await self._load_persisted_volume_overrides(service_id)
        existing = await self._find_container(service_id)
        if existing:
            try:
                await existing.stop()
            except Exception as exc:
                log.debug("recreate_stop_failed", service=service_id, error=str(exc))
            try:
                await existing.delete(force=True)
            except Exception as exc:
                log.debug("recreate_delete_failed", service=service_id, error=str(exc))
        # No existing container now → enable_service does a fresh create,
        # injecting the override-aware managed_auth env.
        return await self.enable_service(service_id, volume_overrides=volume_overrides)

    async def update_service(
        self, service_id: str, new_def: ServiceDefinition,
    ) -> ManagedService:
        """Recreate a manifest service on a bumped catalog definition (new
        pinned image), preserving its data.

        Swaps the runtime definition, stops+deletes the running container,
        then re-provisions via :meth:`enable_service` (which pulls the new
        image). Named volumes, persisted env/volume overrides, and the HTTPS
        front door are all reapplied by ``enable_service`` — only the
        container is replaced, never its data. Brief downtime while it
        restarts. Raises ``ValueError`` if the id isn't a runtime (manifest)
        service — shipped-catalog ids are never updatable this way.
        """
        if not self._catalog.replace_runtime(new_def):
            raise ValueError(
                f"{service_id} is not an updatable manifest service",
            )
        existing = await self._find_container(service_id)
        if existing:
            try:
                await existing.stop()
            except Exception as exc:
                log.debug("update_stop_failed", service=service_id, error=str(exc))
            try:
                await existing.delete(force=True)
            except Exception as exc:
                log.debug("update_delete_failed", service=service_id, error=str(exc))
        # No container now → enable_service does a fresh pull+create on the
        # new image, reusing persisted overrides and reapplying the door.
        ms = await self.enable_service(service_id)
        log.info("service_updated", service=service_id, image=new_def.image)
        return ms

    async def installed_image(self, service_id: str) -> str:
        """Image tag of the currently-provisioned container (the
        ``managed_services`` row) — the truth of what's RUNNING, which can
        lag the catalog manifest after a version bump. Empty when the service
        isn't installed."""
        if not self._db:
            return ""
        cursor = await self._db.execute(
            "SELECT image FROM managed_services WHERE id = ?", (service_id,),
        )
        row = await cursor.fetchone()
        return str(row[0]) if row and row[0] else ""

    async def get_status(self, service_id: str) -> ServiceStatus:
        """Get the current status of a managed service."""
        container = await self._find_container(service_id)
        if not container:
            return ServiceStatus.STOPPED
        try:
            info = await container.show()
            state = info.get("State", {})
            health = state.get("Health", {})
            if health.get("Status") == "healthy":
                return ServiceStatus.RUNNING
            if health.get("Status") == "unhealthy":
                return ServiceStatus.UNHEALTHY
            if state.get("Running"):
                return ServiceStatus.STARTING
            return ServiceStatus.STOPPED
        except Exception:
            return ServiceStatus.ERROR

    async def list_managed(self) -> list[ManagedService]:
        """List all managed services with live status."""
        if not self._db:
            return []
        cursor = await self._db.execute(
            "SELECT id, definition_id, name, category, image, container_id, "
            "host_port, internal_port, config_json, enabled, status, error, "
            "created_at, updated_at FROM managed_services ORDER BY name",
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            ms = ManagedService(
                id=row[0], definition_id=row[1], name=row[2], category=row[3],
                image=row[4], container_id=row[5], host_port=row[6],
                internal_port=row[7], config_json=row[8], enabled=bool(row[9]),
                status=row[10], error=row[11], created_at=row[12], updated_at=row[13],
            )
            ms.status = (await self.get_status(ms.id)).value
            results.append(ms)
        return results

    async def restore_enabled(self) -> int:
        """On startup, re-create containers for services marked enabled."""
        if not self._db:
            return 0
        cursor = await self._db.execute(
            "SELECT id FROM managed_services WHERE enabled = 1",
        )
        rows = await cursor.fetchall()
        restored = 0
        for row in rows:
            service_id = row[0]
            try:
                await self.enable_service(service_id)
                restored += 1
            except Exception:
                log.warning("service_restore_failed", service=service_id)
        return restored

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _find_container(self, service_id: str):
        """Find a managed container by service ID label."""
        filters = {"label": [f"{_LABEL_SERVICE_ID}={service_id}"]}
        containers = await self._docker.containers.list(all=True, filters=filters)
        return containers[0] if containers else None

    async def _to_managed(
        self, service_id: str, sd: ServiceDefinition, container_id: str,
        *, volume_overrides: dict[str, str] | None = None,
    ) -> ManagedService:
        status = await self.get_status(service_id)
        cat_val = sd.category.value if isinstance(sd.category, ServiceCategory) else sd.category
        return ManagedService(
            id=service_id,
            definition_id=sd.id,
            name=sd.name,
            category=cat_val,
            image=sd.image,
            container_id=container_id,
            host_port=sd.host_port,
            internal_port=sd.internal_port,
            config_json=json.dumps({
                "augmentum_env": sd.augmentum_env,
                "volume_overrides": volume_overrides or {},
            }),
            enabled=True,
            status=status.value,
        )

    async def read_config_json(self, service_id: str) -> dict:
        """Return the managed_services config_json blob as a dict ({} if
        missing/unparseable). Complements ``update_config_json``."""
        if not self._db:
            return {}
        try:
            cursor = await self._db.execute(
                "SELECT config_json FROM managed_services WHERE id = ?",
                (service_id,),
            )
            row = await cursor.fetchone()
            data = json.loads(row[0]) if row and row[0] else {}
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 — read is best-effort
            log.warning("read_config_json_failed", service=service_id, exc_info=True)
            return {}

    async def update_config_json(self, service_id: str, patch: dict) -> None:
        """Merge ``patch`` into the managed_services config_json blob.

        Lets installers persist runtime facts that aren't columns — e.g.
        the allocated HTTPS front-door port for a marketplace manifest
        service, so boot rehydration re-registers the SAME door instead
        of allocating a fresh one (which would orphan the caddy snippet
        written at install)."""
        if not self._db:
            return
        merged = {**(await self.read_config_json(service_id)), **patch}
        await self._db.execute(
            "UPDATE managed_services SET config_json = ?, "
            "updated_at = datetime('now') WHERE id = ?",
            (json.dumps(merged), service_id),
        )
        await self._db.commit()

    async def _load_persisted_volume_overrides(
        self, service_id: str,
    ) -> dict[str, str]:
        """Read the volume_overrides stored in the managed_services row.

        Lets ``restore_enabled`` recreate a media container with the same
        host bind it was provisioned with, without the caller needing to
        re-supply the (user-chosen) path. Empty dict if unknown/none.
        """
        if not self._db:
            return {}
        try:
            cursor = await self._db.execute(
                "SELECT config_json FROM managed_services WHERE id = ?",
                (service_id,),
            )
            row = await cursor.fetchone()
            if not row or not row[0]:
                return {}
            data = json.loads(row[0])
            ov = data.get("volume_overrides") if isinstance(data, dict) else None
            return {str(k): str(v) for k, v in ov.items()} if isinstance(ov, dict) else {}
        except Exception:  # noqa: BLE001 — never let a parse error block enable
            log.warning("load_volume_overrides_failed", service=service_id, exc_info=True)
            return {}

    async def _persist(self, ms: ManagedService) -> None:
        if not self._db:
            return
        await self._db.execute(
            """INSERT INTO managed_services
               (id, definition_id, name, category, image, container_id,
                host_port, internal_port, config_json, enabled, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 container_id = excluded.container_id,
                 enabled = excluded.enabled,
                 status = excluded.status,
                 updated_at = datetime('now')""",
            (ms.id, ms.definition_id, ms.name, ms.category, ms.image,
             ms.container_id, ms.host_port, ms.internal_port, ms.config_json,
             1 if ms.enabled else 0, ms.status),
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()


def _resolve_mem_limit(sd: ServiceDefinition) -> int:
    """Host-RAM ceiling in bytes for a managed service. 0 means unlimited.

    **Only an explicit manifest value applies. There is no default ceiling,
    by design.**

    This briefly had per-category defaults derived from ``resources.ram_mb``.
    That was wrong twice over. The 2026-07-25 memory incident originated in the
    Augmentum stack itself — not one catalog service was running — so capping
    ~48 third-party apps treated uninvolved software as the cause. And
    ``ram_mb`` is a declared *minimum* (an admission check: can this host start
    the service), so deriving a ceiling from it silently asserted a maximum its
    author never stated.

    The mechanism could not have been right at any value, either: a static
    per-service ceiling encodes one concurrency level. Open WebUI sized for a
    single user is not the same container serving a team of ten alongside n8n,
    and no number in a manifest scales with users. Aggregate pressure belongs
    to a per-pool budget (spec §5.5), which is what the governance design
    already specified.

    So an operator who has measured their own deployment can set ``mem_limit``
    and it is honoured exactly. Absent that, we do not guess.
    """
    raw = (getattr(sd, "mem_limit", "") or "").strip()
    if not raw:
        return 0
    try:
        return max(0, _parse_size(raw))
    except (ValueError, TypeError):
        log.warning("service_mem_limit_unparseable", service=sd.id, value=raw)
        return 0


def _parse_size(s: str) -> int:
    """Parse Docker-style size string (e.g. '2gb') to bytes."""
    s = s.strip().lower()
    if s.endswith("gb"):
        return int(s[:-2]) * 1024 * 1024 * 1024
    if s.endswith("mb"):
        return int(s[:-2]) * 1024 * 1024
    if s.endswith("g"):
        return int(s[:-1]) * 1024 * 1024 * 1024
    if s.endswith("m"):
        return int(s[:-1]) * 1024 * 1024
    return int(s)
