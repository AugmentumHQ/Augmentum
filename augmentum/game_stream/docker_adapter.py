"""aiodocker-backed implementation of ``ContainerAdapter``.

Sibling-container pattern: the augmentum service talks to docker via
the docker-socket-proxy (``DOCKER_HOST`` env), and asks the host's
Docker daemon to start a fresh game-streaming container per session.

Mirrors the pattern in ``augmentum/coder/containers.py``:
  * Labels ``augmentum.game_stream=true`` so we can enumerate our
    containers for reconcile / cleanup.
  * Resource caps (memory, cpus, pids).
  * Capabilities dropped to ALL except a narrow allowlist.
  * Bind-mounted per-world volume so worlds survive container recycle.

Phase 2 ships the ``start`` / ``stop`` / ``is_alive`` surface required
by the runtime. Future work (Phase 5) adds GPU passthrough plumbing
and exec-into-container helpers for the bridge mod.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from augmentum.calling.turn_credentials import mint_ephemeral
from augmentum.config import settings
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.game_stream.port_pool import PortAllocation
    from augmentum.game_stream.profiles import GameProfile

log = get_logger(__name__)


# Docker labels used to identify and enumerate AGSP containers. The
# scanner / lifecycle reconciler filters on these.
LABEL_GAME_STREAM = "augmentum.game_stream"
LABEL_SESSION_ID = "augmentum.game_stream.session_id"
LABEL_USER_ID = "augmentum.game_stream.user_id"
LABEL_PROFILE_ID = "augmentum.game_stream.profile_id"


# Host paths where per-user world data lives, relative to the augmentum
# service's data root. Mirrors the named volume declared in
# compose.game-stream.yaml (mounted at /data/game_stream/worlds inside
# augmentum). The streaming container itself bind-mounts a subdir.
_HOST_WORLDS_ROOT = "/data/game_stream/worlds"


def _addon_for_image(image: str):
    """Return the add-on spec that provides ``image``, or None.

    The streaming images are installable capabilities (Discover -> Add-ons),
    so "image missing" usually means "that add-on isn't installed" rather
    than a Docker misconfiguration. Resolving the spec lets the launch path
    say which capability is absent and what installing it costs, instead of
    handing the user a daemon-level 404. Imported lazily and defensively so
    the adapter stays importable in test environments without the package.
    """
    try:
        from augmentum.addons.catalog import ADDONS

        for spec in ADDONS:
            if spec.image == image:
                return spec
    except Exception:  # noqa: BLE001 — never let this mask the real error
        return None
    return None


def _mint_session_turn_env(*, session_id: str, user_id: str) -> dict[str, str]:
    """Mint TURN creds for one streaming session.

    The streamed container's entrypoint reads
    ``AUGMENTUM_TURN_USERNAME``/``AUGMENTUM_TURN_PASSWORD`` to populate
    selkies' iceServers config. Previously these were static strings
    forwarded straight from the proxy's env; now they're HMAC ephemeral
    pairs scoped to this session so a leaked cred only relays for the
    24h TTL.

    The identity hint folds session_id + user_id so the coturn log is
    greppable by either dimension (session for incident reproduction,
    user for "who's hogging the relay?" investigations).
    """

    identity_hint = session_id[:16] or (user_id[:16] or "agsp")
    creds = mint_ephemeral(identity_hint)
    return {
        "AUGMENTUM_TURN_USERNAME": creds.username,
        "AUGMENTUM_TURN_PASSWORD": creds.password,
    }

# Volume names. We mount these BY NAME (Docker Mounts API) into the
# streamed container rather than via host-path Binds, because on
# Docker Desktop (and any setup where augmentum's /data is a named
# volume rather than a host bind) the path inside augmentum doesn't
# resolve on the daemon's filesystem — the daemon sees
# /var/lib/docker/volumes/<name>/_data/, not /data/. Mounting by
# name sidesteps the path-translation problem entirely.
#
# Names are configurable via env vars so deployments with non-default
# compose project names still work. The defaults match Augmentum's
# stock ``compose.yaml`` + ``compose.game-stream.yaml`` (project
# 'augmentum' so volumes get the 'augmentum_' prefix).
_DATA_VOLUME = os.environ.get(
    "AUGMENTUM_DATA_VOLUME_NAME", "augmentum_augmentum_data",
)
_EMULATOR_SAVES_VOLUME = os.environ.get(
    "AUGMENTUM_EMULATOR_SAVES_VOLUME_NAME",
    "augmentum_game_stream_emulator_saves",
)

# In-container mount points. The entrypoint resolves the actual ROM
# file under HOST_DATA_TARGET/blobs/<sha[:2]>/<sha[2:4]>/<sha>; saves
# get a per-(user, emulator) subdir created at first launch.
_HOST_DATA_TARGET = "/host-data"
_HOST_EMULATOR_SAVES_TARGET = "/host-emulator-saves"

# Where each bundled emulator's per-user config + saves live INSIDE the
# streamed container. The runtime layer doesn't bind these directly
# anymore — the entrypoint creates per-(user, emulator) subdirs under
# /host-emulator-saves and symlinks them to the emulator's native
# config dir (so Dolphin's ~/.dolphin-emu still works as expected).
#
# Adding a new emulator = adding a row here. Keys MUST match what the
# entrypoint (entrypoint-emulator-streamed.sh) accepts as
# AUGMENTUM_EMULATOR.
_EMULATOR_CONFIG_DIR: dict[str, str] = {
    "dolphin": "/home/player/.dolphin-emu",
    "pcsx2":   "/home/player/.config/PCSX2",
    # "rpcs3":  "/home/player/.config/rpcs3",
}


# Default resource caps. Generous enough for Luanti to run smoothly
# (Luanti server + client + Selkies + Xvfb in one container) while
# still bounding the blast radius of a runaway game.
_DEFAULT_CPU_NANOS = 4_000_000_000  # 4 cpus -- Dolphin JIT (2) + Mesa llvmpipe (1) + x264enc (1) all share this pool; 2 cpus starved them and stuttered.
_DEFAULT_MEMORY_BYTES = 3 * 1024 * 1024 * 1024  # 3 GB -- PCSX2 (~1 GB) + Mesa llvmpipe + x264enc + Selkies fits comfortably; 2 GB was tight.
_DEFAULT_PIDS_LIMIT = 512
# Docker's default /dev/shm is 64 MB. PulseAudio's shm transport and
# Dolphin's Software video backend both mmap from /dev/shm; touching an
# mmap region that can't be backed = SIGBUS (exit 135) within seconds
# of A/V init. 512 MB matches Docker Desktop's documented headroom and
# is well under the 2 GB cgroup cap above.
_DEFAULT_SHM_BYTES = 512 * 1024 * 1024  # 512 MB


# Capabilities -- drop everything, add back only what the streaming
# stack needs. Xvfb + PulseAudio + Selkies need essentially nothing
# privileged; this keeps things tight.
_CAP_DROP = ["ALL"]
_CAP_ADD = ["SETUID", "SETGID"]


def _fmt_float(value: float) -> str:
    return f"{float(value):.6g}"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        log.warning("game_stream_invalid_float_env", key=name, value=raw)
        return float(default)


class DockerContainerAdapter:
    """Real Docker-backed container adapter for ``GameStreamRuntime``.

    Constructed once at startup with an aiodocker client and reused
    for every session. Stateless apart from the client handle.
    """

    def __init__(
        self, docker_client: Any, *,
        gpu_passthrough: bool = False,
        host_network: bool = False,
    ) -> None:
        # ``docker_client`` is an ``aiodocker.Docker`` instance. Typed
        # as Any because aiodocker doesn't ship type stubs and we want
        # to keep this importable in test environments without it.
        self._docker = docker_client
        self._gpu = gpu_passthrough
        # Host networking: selkies binds host interfaces directly so
        # WebRTC ICE candidates use the host's LAN IP and are
        # reachable from a same-LAN browser. Bridge networking puts
        # selkies behind Docker NAT, where same-LAN ICE typically
        # fails (host candidate is the unrouteable Docker bridge IP,
        # STUN reflexive depends on hairpinning that Docker Desktop
        # doesn't reliably provide).
        # Trade-off: one concurrent stream per host (port conflict
        # on 8080). Acceptable for MVP; multi-stream is a follow-up
        # via per-session subdomain or assigned LAN IPs.
        self._host_network = host_network

    @property
    def host_network(self) -> bool:
        return self._host_network

    # ── ContainerAdapter protocol ──────────────────────────────────

    async def start(
        self,
        *,
        session_id: str,
        profile: GameProfile,
        ports: PortAllocation,
        bitrate_mbps: int,
        resolution: str,
        encoder: str,
        world_storage_path: str,
        world_settings: dict,
        touch_mode: bool = False,
        mouse_sensitivity: float | None = None,
        gamepad_enabled: bool = True,
        controller_deadzone: float | None = None,
        # Streamed-emulator (Dolphin etc.) extras. All None for non-
        # emulator profiles like Luanti — they fall through cleanly.
        user_id: str = "",
        emulator: str = "",
        system_id: str = "",
        rom_blob_sha: str = "",
        bios_files: list[tuple[str, str]] | tuple = (),
        # AI session bridge URL. Threaded through verbatim as
        # AUGMENTUM_AGENT_BRIDGE_URL; the entrypoint reads it and
        # spawns agent-bridge.py when set.
        agent_bridge_url: str = "",
        # Cast-input bridge URL. Threaded through verbatim as
        # AUGMENTUM_CAST_INPUT_BRIDGE_URL; the entrypoint reads it and
        # spawns cast-input-bridge.py when set. Empty for sessions
        # without phone-as-controller (e.g. AI-only or solo browser).
        cast_input_bridge_url: str = "",
    ) -> str:
        """Launch a streaming container for this session. Returns container_id."""
        bitrate_kbps = max(500, int(bitrate_mbps) * 1000)

        env = self._build_env(
            session_id=session_id,
            bitrate_kbps=bitrate_kbps,
            resolution=resolution,
            encoder=encoder,
            world_settings=world_settings,
            world_storage_path=world_storage_path,
            game_port=ports.game_port,
            touch_mode=touch_mode,
            mouse_sensitivity=mouse_sensitivity,
            gamepad_enabled=gamepad_enabled,
            controller_deadzone=controller_deadzone,
            user_id=user_id,
            emulator=emulator,
            system_id=system_id,
            rom_blob_sha=rom_blob_sha,
            bios_files=bios_files,
            agent_bridge_url=agent_bridge_url,
            cast_input_bridge_url=cast_input_bridge_url,
        )

        labels = {
            LABEL_GAME_STREAM: "true",
            LABEL_SESSION_ID: session_id,
            LABEL_PROFILE_ID: profile.id,
        }

        # Per-world bind mount. Empty/missing storage_path -> ephemeral
        # tmpfs world (each launch is a fresh world).
        binds: list[str] = []
        if world_storage_path:
            host_path = f"{_HOST_WORLDS_ROOT}/{world_storage_path}"
            binds.append(f"{host_path}:/home/player/worlds/{world_storage_path}")

        # Streamed-emulator extras: ROM blob (read-only) + per-user
        # saves volume (read-write).
        #
        # We mount the named volumes by NAME via the Mounts API rather
        # than using Binds with host paths. On Docker Desktop the
        # augmentum container's /data is itself a named volume, so the
        # path /data/blobs/<sha> exists inside augmentum but NOT on the
        # daemon's host filesystem; a Binds-style source path silently
        # gets auto-created as an empty dir, the bind appears successful,
        # and the streamed container ends up with an empty rom.bin.
        # Mounts-by-name resolves to the volume's actual storage path
        # on the daemon and bypasses the issue.
        #
        # The entrypoint reads AUGMENTUM_ROM_PATH (passed via env) to
        # find the actual file under /host-data/blobs/<sha[:2]>/<sha[2:4]>/<sha>,
        # and creates the per-user saves subdir + symlink at first launch.
        mounts: list[dict[str, Any]] = []
        if rom_blob_sha:
            mounts.append({
                "Type": "volume",
                "Source": _DATA_VOLUME,
                "Target": _HOST_DATA_TARGET,
                "ReadOnly": True,
            })
        if emulator and user_id:
            cfg_dir = _EMULATOR_CONFIG_DIR.get(emulator)
            if cfg_dir:
                mounts.append({
                    "Type": "volume",
                    "Source": _EMULATOR_SAVES_VOLUME,
                    "Target": _HOST_EMULATOR_SAVES_TARGET,
                    "ReadOnly": False,
                })
            else:
                log.warning(
                    "game_stream_emulator_saves_skipped",
                    reason="no config_dir mapping",
                    emulator=emulator,
                )

        # Port mappings: only meaningful in bridge mode. With host
        # networking the container shares the host's net namespace
        # and binds ports directly via the entrypoint's selkies
        # invocation -- PortBindings are silently ignored.
        # Per-profile security relaxation. Default: lock the container
        # down with cap-drop=ALL + no-new-privileges, which is fine for
        # most games. Profiles that use signal-handler-based soft-MMU
        # emulation (Dolphin's fastmem; future PCSX2/RPCS3) need the
        # default capabilities and standard seccomp profile to install
        # SIGBUS/SIGSEGV handlers — without them the emulator SIGBUSes
        # within ~5 seconds of trying to commit its emulated address
        # space. Set ``relax_security=True`` on those profiles.
        relax = bool(getattr(profile, "relax_security", False))
        host_config: dict[str, Any] = {
            "NanoCpus": _DEFAULT_CPU_NANOS,
            "Memory": _DEFAULT_MEMORY_BYTES,
            "MemorySwap": _DEFAULT_MEMORY_BYTES,
            "ShmSize": _DEFAULT_SHM_BYTES,
            "PidsLimit": _DEFAULT_PIDS_LIMIT,
            "Binds": binds,
            "Mounts": mounts,
        }
        if relax:
            # Default caps + default seccomp profile. Equivalent to
            # plain ``docker run`` with no security overrides.
            log.info(
                "game_stream_relaxed_security",
                profile_id=profile.id,
                reason="profile.relax_security=true (e.g. emulator fastmem)",
            )
        else:
            host_config["CapDrop"] = _CAP_DROP
            host_config["CapAdd"] = _CAP_ADD
            host_config["SecurityOpt"] = ["no-new-privileges:true"]

        # Gamepad passthrough: bind /dev/uinput so the in-container
        # bridge can synthesise /dev/input/jsX from the Selkies UDS
        # protocol. CgroupPermissions "rwm" gives the container full
        # rwx + mknod access on the device node — required because
        # python-evdev's UInput needs to call UI_DEV_CREATE which
        # the kernel routes through /dev/uinput. Skipped for profiles
        # that don't want it so we don't expand attack surface
        # unnecessarily; gated on /dev/uinput actually
        # existing on the host so we degrade cleanly on hosts where
        # the uinput kernel module isn't loaded.
        # Session input settings can disable this even for profiles
        # that advertise gamepad support.
        if (
            getattr(profile, "wants_gamepad", False)
            and gamepad_enabled
            and os.path.exists("/dev/uinput")
        ):
            host_config.setdefault("Devices", []).append({
                "PathOnHost": "/dev/uinput",
                "PathInContainer": "/dev/uinput",
                "CgroupPermissions": "rwm",
            })
            log.info(
                "game_stream_gamepad_passthrough_enabled",
                profile_id=profile.id,
            )
        elif getattr(profile, "wants_gamepad", False) and not gamepad_enabled:
            log.info(
                "game_stream_gamepad_passthrough_disabled",
                profile_id=profile.id,
                reason="session input.gamepad_enabled=false",
            )
        elif getattr(profile, "wants_gamepad", False):
            log.warning(
                "game_stream_gamepad_passthrough_skipped",
                profile_id=profile.id,
                reason="/dev/uinput not present on host (uinput module not loaded)",
            )
        exposed_ports: dict[str, dict] = {}
        if self._host_network:
            host_config["NetworkMode"] = "host"
        else:
            host_config["NetworkMode"] = "bridge"
            host_config["PortBindings"] = {
                "8080/tcp": [
                    {"HostIp": "0.0.0.0", "HostPort": str(ports.stream_port)},
                ],
                "30000/udp": [
                    {"HostIp": "0.0.0.0", "HostPort": str(ports.game_port)},
                ],
            }
            exposed_ports = {"8080/tcp": {}, "30000/udp": {}}

        # Per-profile GPU opt-out: profiles whose container is known to
        # crash on the NVIDIA GL overlay (currently emulator-streamed,
        # because Dolphin/Qt6 SIGBUSes when NVIDIA's libGL is mounted
        # over Mesa's on Xvfb) skip the DeviceRequests block. They use
        # software rendering exclusively. NVENC for the streaming
        # encoder is also unavailable for these — the entrypoint's
        # GPU probe will fall back to x264 software encode.
        profile_wants_gpu = getattr(profile, "requires_gpu", True)
        if self._gpu and profile_wants_gpu:
            # nvidia-container-toolkit translates this into the right
            # device mounts. Only set when the operator opted in via
            # the `gpu_passthrough` constructor flag (controlled in
            # server.py from the GPU compose overlay).
            # Capabilities: ``gpu`` alone gives CUDA device access but
            # NOT the OpenGL libraries -- the container falls through
            # to llvmpipe software rasterization, which is exactly the
            # "feels sluggish" symptom even though gpu=True logs from
            # the server. Adding ``graphics`` + ``utility`` gets the
            # nvidia GL drivers injected so Minetest renders on the
            # GPU. ``video`` gets the NVENC libs for hardware encode.
            host_config["DeviceRequests"] = [
                {
                    "Driver": "nvidia",
                    "Count": -1,
                    "Capabilities": [["gpu", "graphics", "utility", "video"]],
                },
            ]

        config: dict[str, Any] = {
            "Image": profile.image,
            "Env": [f"{k}={v}" for k, v in env.items()],
            "Labels": labels,
            "Tty": False,
            "OpenStdin": False,
            "ExposedPorts": exposed_ports,
            "HostConfig": host_config,
        }

        log.info(
            "game_stream_container_starting",
            session_id=session_id,
            profile_id=profile.id,
            stream_port=ports.stream_port,
            game_port=ports.game_port,
            encoder=encoder,
            resolution=resolution,
            bitrate_kbps=bitrate_kbps,
            gpu=self._gpu,
        )

        # Pre-check the image exists so the user gets an actionable
        # error rather than the generic "container start failed".
        # Differentiate "image absent" (404) from other Docker errors
        # (proxy ACL deny, daemon down, etc.) so the user message
        # points at the right fix.
        try:
            await self._docker.images.inspect(profile.image)
        except Exception as exc:
            status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
            msg = str(exc)
            log.warning(
                "game_stream_image_inspect_failed",
                image=profile.image,
                exception_type=type(exc).__name__,
                exception_message=msg,
                status=status,
            )
            if status == 404 or "no such image" in msg.lower() or "not found" in msg.lower():
                # If this image belongs to a catalogued ADD-ON, the honest
                # answer is "that capability isn't installed", plus where to
                # install it -- not a Docker troubleshooting paragraph. The
                # category exists so a missing capability is a decision made
                # in Discover rather than a 404 hit at launch.
                addon = _addon_for_image(profile.image)
                if addon is not None:
                    raise RuntimeError(
                        f"the '{addon.title}' add-on isn't installed, so "
                        f"{addon.provides} isn't available. Install it from "
                        f"Discover → Add-ons (about {addon.build_minutes} "
                        f"minutes, {addon.disk_mb / 1000:.1f}GB of disk), or "
                        f"build on the host with 'start.bat build' / "
                        f"'./start.sh build'."
                    ) from exc
                raise RuntimeError(
                    f"image {profile.image!r} not visible to the augmentum "
                    f"container (HTTP 404). On the augmentum host, run "
                    f"'docker images | grep augmentum-game-stream' to verify "
                    f"the image is actually built. If it IS built but you "
                    f"see this error, the docker-socket-proxy might be on "
                    f"a different daemon/socket than the one that built it. "
                    f"To build: 'start.bat build'."
                ) from exc
            if status in (401, 403):
                raise RuntimeError(
                    f"docker daemon refused images.inspect (HTTP {status}). "
                    f"The docker-socket-proxy may be blocking image inspect "
                    f"-- confirm IMAGES=1 in compose.yaml's docker-proxy "
                    f"environment block."
                ) from exc
            # Anything else (daemon down, network, parse error, ...)
            raise RuntimeError(
                f"could not check image {profile.image!r} "
                f"(type={type(exc).__name__}, status={status}): {msg}"
            ) from exc

        try:
            container = await self._docker.containers.run(
                config=config,
                name=f"agsp-{session_id}",
            )
        except Exception as exc:
            # Surface the underlying Docker error so the user knows
            # what to fix (port conflict, missing GPU, capability
            # rejection, etc.).
            raise RuntimeError(
                f"docker run failed for {profile.image!r}: {exc}",
            ) from exc
        return container.id

    async def stop(self, container_id: str, *, timeout: int = 10) -> None:
        """Stop + remove the container. Idempotent: missing containers
        are not an error.
        """
        if not container_id:
            return
        try:
            container = await self._docker.containers.get(container_id)
        except Exception as exc:
            # Most likely a 404 -- the container was already removed
            # (maybe the user's host did `docker rm` manually). Log
            # and move on.
            log.info(
                "game_stream_container_already_gone",
                container_id=container_id, error=str(exc),
            )
            return
        try:
            await container.stop(timeout=timeout)
        except Exception as exc:
            # Stop best-effort; if the container's already exited the
            # API will still 304 here on some Docker versions.
            log.info(
                "game_stream_container_stop_noop",
                container_id=container_id, error=str(exc),
            )
        try:
            await container.delete(force=True)
        except Exception as exc:
            log.warning(
                "game_stream_container_delete_failed",
                container_id=container_id, error=str(exc),
            )

    async def is_alive(self, container_id: str) -> bool:
        if not container_id:
            return False
        try:
            container = await self._docker.containers.get(container_id)
            data = await container.show()
        except Exception:
            return False
        state = (data.get("State") or {}) if isinstance(data, dict) else {}
        # Paused counts as alive — the container still exists, processes
        # are just cgroup-frozen. Only fully-stopped/exited means dead.
        if bool(state.get("Running")):
            return True
        return bool(state.get("Paused"))

    async def pause(self, container_id: str) -> None:
        """Suspend every process in the container via the cgroup freezer.

        Idempotent: pausing an already-paused container is a no-op
        (Docker returns 304). CPU drops to 0, RAM is held, resume via
        ``unpause`` is sub-second.
        """
        if not container_id:
            return
        try:
            container = await self._docker.containers.get(container_id)
            await container.pause()
        except Exception as exc:
            log.warning(
                "game_stream_container_pause_failed",
                container_id=container_id, error=str(exc),
            )

    async def unpause(self, container_id: str) -> None:
        """Thaw a paused container. Idempotent."""
        if not container_id:
            return
        try:
            container = await self._docker.containers.get(container_id)
            await container.unpause()
        except Exception as exc:
            log.warning(
                "game_stream_container_unpause_failed",
                container_id=container_id, error=str(exc),
            )

    # ── Optional helpers ───────────────────────────────────────────

    async def list_owned(self) -> list[dict]:
        """Enumerate all AGSP-owned containers visible to Docker.

        Used by the runtime's reconcile path to detect orphaned
        containers (rows we lost track of in the DB but Docker still
        runs). Returns a list of dicts with id/labels/state.
        """
        try:
            containers = await self._docker.containers.list(
                all=True,
                filters={"label": [f"{LABEL_GAME_STREAM}=true"]},
            )
        except Exception as exc:
            log.warning("game_stream_container_list_failed", error=str(exc))
            return []
        out = []
        for c in containers:
            try:
                data = await c.show()
            except Exception as exc:
                log.debug("game_stream_container_show_failed", container_id=getattr(c, "id", "?"), error=str(exc))
                continue
            out.append({
                "id": c.id,
                "labels": (data.get("Config") or {}).get("Labels") or {},
                "state": (data.get("State") or {}).get("Status") or "",
                "session_id": ((data.get("Config") or {}).get("Labels") or {}).get(
                    LABEL_SESSION_ID, "",
                ),
            })
        return out

    # ── Internals ──────────────────────────────────────────────────

    def _build_env(
        self,
        *,
        session_id: str,
        bitrate_kbps: int,
        resolution: str,
        encoder: str,
        world_settings: dict,
        world_storage_path: str,
        game_port: int,
        touch_mode: bool = False,
        mouse_sensitivity: float | None = None,
        gamepad_enabled: bool = True,
        controller_deadzone: float | None = None,
        # Streamed-emulator extras (empty for non-emulator profiles)
        user_id: str = "",
        emulator: str = "",
        system_id: str = "",
        rom_blob_sha: str = "",
        bios_files: list[tuple[str, str]] | tuple = (),
        agent_bridge_url: str = "",
        cast_input_bridge_url: str = "",
    ) -> dict[str, str]:
        """Compose the env-var bundle the entrypoint scripts read."""
        env: dict[str, str] = {
            "AUGMENTUM_SESSION_ID": session_id,
            "AUGMENTUM_RESOLUTION": resolution,
            "AUGMENTUM_BITRATE_KBPS": str(bitrate_kbps),
            "AUGMENTUM_ENCODER": encoder,
            # Container-side game port is fixed (30000); we still pass
            # the host-side allocation so the entrypoint's logs are
            # accurate and a future bridge mod can self-identify.
            "AUGMENTUM_GAME_PORT": "30000",
            "AUGMENTUM_HOST_GAME_PORT": str(game_port),
            # Touch-mode: tells the per-game entrypoint to render its
            # native on-screen touch UI (Minetest's enable_touch, or
            # whatever the equivalent is for future profiles).
            "AUGMENTUM_TOUCH_MODE": "true" if touch_mode else "false",
            # TURN relay config. HOST/PORT come from compose.calling.yaml;
            # USERNAME/PASSWORD are minted per-session as HMAC ephemeral
            # creds against AUGMENTUM_TURN_SECRET. The in-container
            # entrypoint reads these four env vars to build the selkies
            # iceServers config, same shape as before — the move from
            # static to ephemeral is invisible to the container side.
            "AUGMENTUM_TURN_HOST": os.environ.get(
                "AUGMENTUM_TURN_HOST", "localhost",
            ),
            "AUGMENTUM_TURN_PORT": os.environ.get(
                "AUGMENTUM_TURN_PORT", "3478",
            ),
            **_mint_session_turn_env(session_id=session_id, user_id=user_id),
            "AUGMENTUM_MOUSE_SENSITIVITY": _fmt_float(
                mouse_sensitivity
                if mouse_sensitivity is not None
                else _env_float(
                    "AUGMENTUM_GAME_STREAM_MOUSE_SENSITIVITY",
                    getattr(settings, "game_stream_mouse_sensitivity", 0.2),
                )
            ),
            "AUGMENTUM_GAMEPAD_ENABLED": "true" if gamepad_enabled else "false",
            "AUGMENTUM_CONTROLLER_DEADZONE": _fmt_float(
                controller_deadzone
                if controller_deadzone is not None
                else getattr(settings, "controller_deadzone", 0.15)
            ),
        }
        if world_storage_path:
            env["AUGMENTUM_WORLD_ID"] = world_storage_path
        # Game-specific knobs forwarded as-is. The entrypoint takes
        # responsibility for treating unknown keys as no-ops.
        if isinstance(world_settings, dict):
            mode = world_settings.get("gamemode")
            if isinstance(mode, str):
                env["AUGMENTUM_WORLD_GAMEMODE"] = mode
            seed = world_settings.get("seed")
            if isinstance(seed, str):
                env["AUGMENTUM_WORLD_SEED"] = seed
            pvp = world_settings.get("pvp")
            if isinstance(pvp, bool):
                env["AUGMENTUM_WORLD_PVP"] = "true" if pvp else "false"
            # Generic browser-stream URL. Consumed by entrypoint-
            # browser.sh as the URL to load in Chrome kiosk. This is
            # the same forwarding pattern Luanti uses for its gamemode
            # / seed knobs — just a different key the per-profile
            # entrypoint reads.
            target_url = world_settings.get("target_url")
            if isinstance(target_url, str) and target_url:
                env["AUGMENTUM_TARGET_URL"] = target_url

        # Streamed-emulator dispatch. The entrypoint dispatch
        # (case "$AUGMENTUM_EMULATOR" in dolphin) ...) reads these.
        # ROM_PATH points into the named-volume mount at
        # /host-data/blobs/<sha[:2]>/<sha[2:4]>/<sha> — same content-
        # addressed layout the augmentum-side blob_store uses.
        if emulator:
            env["AUGMENTUM_EMULATOR"] = emulator
        if system_id:
            env["AUGMENTUM_SYSTEM_ID"] = system_id
        if rom_blob_sha:
            sha = rom_blob_sha
            env["AUGMENTUM_ROM_PATH"] = (
                f"{_HOST_DATA_TARGET}/blobs/{sha[:2]}/{sha[2:4]}/{sha}"
            )
        if user_id:
            env["AUGMENTUM_USER_ID"] = user_id
        # Pass through the saves-volume mount target so the entrypoint
        # can construct the per-(user, emulator) subdir path without
        # having to know the convention.
        if emulator and user_id:
            env["AUGMENTUM_SAVES_BASE"] = (
                f"{_HOST_EMULATOR_SAVES_TARGET}/{user_id}/{emulator}"
            )
        # BIOS files for emulators that hard-require them (PCSX2, future
        # RPCS3). Format: "canonical1.bin:blob_sha256_1,canonical2.bin:
        # blob_sha256_2". The entrypoint splits on commas + colons and
        # symlinks each from /host-data/blobs/<sha[:2]>/<sha[2:4]>/<sha>
        # into the emulator's bios dir before launching. Skipped if the
        # caller passed an empty/falsy value (e.g. Dolphin, Luanti, or
        # a PS2 launch where the user has nothing installed yet — that
        # case is handled by the launcher's own missing-BIOS guard).
        if bios_files:
            env["AUGMENTUM_BIOS_FILES"] = ",".join(
                f"{name}:{sha}"
                for name, sha in bios_files
                if name and sha
            )
        # AI session: entrypoint-base.sh checks this and spawns
        # agent-bridge.py when present. The URL already carries the
        # ?token=<x> query param augmentum minted for the paired
        # game-agent session, so the daemon authenticates with no
        # extra plumbing.
        if agent_bridge_url:
            env["AUGMENTUM_AGENT_BRIDGE_URL"] = agent_bridge_url
        # Phone-as-controller: entrypoint-base.sh launches
        # cast-input-bridge.py when present. The URL carries the
        # ?token=<x> query param matching the session row's
        # cast_input_token, validated server-side at WS accept.
        if cast_input_bridge_url:
            env["AUGMENTUM_CAST_INPUT_BRIDGE_URL"] = cast_input_bridge_url
        return env
