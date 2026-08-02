"""Game streaming runtime: orchestrates store + lifecycle + port pool.

The runtime is the high-level entry point that route handlers and the
lifecycle watcher both call. Container management (docker start, exec,
wait, stop) is *intentionally pluggable* -- Phase 0 ships with a
``ContainerAdapter`` protocol whose default implementation is a no-op
stub. Phase 2 swaps in a real aiodocker-backed adapter.

This separation lets us:
* unit-test the orchestration logic without Docker
* defer the (heavier) container work to Phase 2
* later swap docker for podman / k8s without touching the orchestration
"""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from augmentum.game_stream.lifecycle import (
    GameStreamLifecycle,
    LifecycleTransitionError,
    SessionStatus,
)
from augmentum.game_stream.port_pool import PortAllocation, PortPool, PortPoolExhausted
from augmentum.game_stream.profiles import GameProfile, ProfileRegistry, profile_registry
from augmentum.state.game_stream_store import GameStreamStore
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class RuntimeError(Exception):  # noqa: N818  (project naming)
    """Generic runtime error that route handlers translate to 4xx/5xx."""


class ConcurrentStreamLimitError(RuntimeError):
    """Raised when a user exceeds their per-user concurrent stream cap."""


@dataclass(frozen=True)
class ConnectionInfo:
    """Connection details handed back to the browser client."""

    session_id: str
    profile_id: str
    status: str
    stream_port: int
    game_port: int
    bitrate_mbps: int
    resolution: str
    # Relative path the client uses to open the WebRTC signaling
    # socket; the routes layer prepends scheme/host.
    signaling_path: str
    mouse_sensitivity: float | None = None
    gamepad_enabled: bool = True
    controller_deadzone: float | None = None


# ── Container adapter protocol ────────────────────────────────────────


class ContainerAdapter(Protocol):
    """Pluggable container backend. Phase 2 plugs in aiodocker."""

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
        # Streamed-emulator extras (empty for non-emulator profiles)
        user_id: str = "",
        emulator: str = "",
        system_id: str = "",
        rom_blob_sha: str = "",
        # (canonical_filename, blob_sha256) pairs for emulators that
        # need user-installed BIOS at session start. Empty/no-op for
        # non-BIOS emulators.
        bios_files: list[tuple[str, str]] | tuple = (),
        # When non-empty, the in-container agent-bridge.py daemon
        # dials this WS URL to take over inputs + frame capture for
        # an AI-driven session. Empty = human-only session, no
        # agent-bridge process is launched in the container.
        agent_bridge_url: str = "",
        # When non-empty, the in-container cast-input-bridge.py daemon
        # dials this WS URL and writes inbound gamepad frames to virtual
        # UInput pads. Token is embedded in the URL query string and
        # validated server-side against the session row's
        # cast_input_token. Empty = no phone-as-controller surface
        # available for this session.
        cast_input_bridge_url: str = "",
    ) -> str:
        """Start the container; return container_id."""

    async def stop(self, container_id: str, *, timeout: int = 10) -> None:
        ...

    async def is_alive(self, container_id: str) -> bool:
        ...

    async def pause(self, container_id: str) -> None:
        """Suspend the container via cgroup freezer. CPU → 0, RAM held."""
        ...

    async def unpause(self, container_id: str) -> None:
        """Thaw a paused container."""
        ...


    @property
    def host_network(self) -> bool:
        """Whether the adapter spawns containers on the host network.

        Runtime uses this to override the reported stream_port: in
        host-net mode the agsp container always binds host:8080
        directly (PortBindings are ignored), so the URL the
        browser uses must point at 8080 regardless of what the
        port pool allocated.
        """
        return False


class StubContainerAdapter:
    """No-op adapter used until Phase 2 wires real Docker."""

    async def start(self, **kwargs) -> str:
        # Synthetic container id for testing. Phase 2 replaces this.
        return f"stub-{kwargs['session_id']}"

    async def stop(self, container_id: str, *, timeout: int = 10) -> None:
        return None

    async def is_alive(self, container_id: str) -> bool:
        return container_id.startswith("stub-")

    async def pause(self, container_id: str) -> None:
        return None

    async def unpause(self, container_id: str) -> None:
        return None


# ── Runtime ──────────────────────────────────────────────────────────


class GameStreamRuntime:
    """High-level orchestration."""

    def __init__(
        self,
        *,
        store: GameStreamStore,
        port_pool: PortPool | None = None,
        registry: ProfileRegistry | None = None,
        adapter: ContainerAdapter | None = None,
        max_concurrent_per_user: int = 2,
        idle_timeout_seconds: int = 600,
        prefer_hw_encoder: bool = True,
        signaling_path_prefix: str = "/api/game-stream/signal",
        render_stream_stale_seconds: int = 1800,
        # Admission-control budgets — credit-based replacement for the
        # flat max_concurrent_per_user gate. See _admit() docstring and
        # docs/superpowers/specs/2026-06-02-game-stream-admission-control.md.
        active_credit_budget: int = 8,
        resident_credit_budget: int = 16,
        user_hard_cap: int = 4,
        paused_stop_seconds: int = 1800,
        # Optional best-effort hook fired after a session reaches its
        # terminal STOPPED state. Used by the cast couch-coop substrate
        # to revoke pending invite tokens that would otherwise outlive
        # the session they pointed at. Failure is swallowed so a broken
        # hook can't block the stop path.
        on_session_stopped: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> None:
        self._store = store
        self._pool = port_pool or PortPool()
        self._registry = registry or profile_registry
        self._adapter: ContainerAdapter = adapter or StubContainerAdapter()
        self._max_concurrent_per_user = max_concurrent_per_user
        self._idle_timeout_seconds = idle_timeout_seconds
        self._prefer_hw = prefer_hw_encoder
        self._signaling_prefix = signaling_path_prefix
        # Safety net for browser-stream sessions: when last_seen_at is
        # stale beyond this threshold, sweep_idle reaps the container
        # even if the receiver-side detach hook never fired. See
        # sweep_idle docstring for the full rationale.
        self._render_stream_stale_seconds = render_stream_stale_seconds
        # Admission budgets — see _admit().
        self._active_credit_budget = max(1, int(active_credit_budget))
        self._resident_credit_budget = max(
            self._active_credit_budget, int(resident_credit_budget),
        )
        self._user_hard_cap = max(1, int(user_hard_cap))
        self._paused_stop_seconds = max(60, int(paused_stop_seconds))
        self._on_session_stopped = on_session_stopped

    @property
    def registry(self) -> ProfileRegistry:
        return self._registry

    @property
    def host_network(self) -> bool:
        return bool(getattr(self._adapter, "host_network", False))

    async def container_alive(self, container_id: str) -> bool:
        if not container_id:
            return False
        return await self._adapter.is_alive(container_id)

    # ── Lifecycle entry points ──────────────────────────────────────

    async def start_session(
        self,
        *,
        user_id: str,
        profile_id: str,
        world_id: str | None = None,
        bitrate_mbps: int | None = None,
        resolution: str | None = None,
        encoder: str | None = None,
        touch_mode: bool = False,
        mouse_sensitivity: float | None = None,
        gamepad_enabled: bool = True,
        controller_deadzone: float | None = None,
        # Streamed-emulator extras: when AgspStreamedRuntime resolves a
        # ROM-backed title to this runtime, it threads these through so
        # the entrypoint can dispatch to the right emulator binary and
        # the docker_adapter can bind-mount the ROM blob + per-user
        # saves dir. Empty for non-emulator profiles (Luanti etc.).
        emulator: str = "",
        system_id: str = "",
        rom_blob_sha: str = "",
        # (canonical_filename, blob_sha256) pairs for emulators that
        # need user-installed BIOS at session start. Resolved by the
        # caller (AgspStreamedRuntime) from bios_store; the adapter
        # threads them into the entrypoint env so the entrypoint can
        # symlink each from /host-data/blobs/ into the emulator's
        # bios dir without the proxy needing FS access to the saves
        # volume.
        bios_files: list[tuple[str, str]] | tuple = (),
        # AI session WS URL. When set, the streaming container's
        # agent-bridge.py dials it on boot and owns a parallel UInput
        # gamepad + Xvfb capture. Computed by the route layer from a
        # paired game-agent session's bridge_token URL. Empty for
        # human-only sessions.
        agent_bridge_url: str = "",
        # Cast-input bridge URL with placeholder ``{session_id}`` and
        # ``{token}``. The runtime mints the token, persists it on the
        # session row, then substitutes both into the template before
        # passing to the adapter. Empty disables phone-as-controller.
        # Route layer owns the template (host/scheme baked in).
        cast_input_bridge_url_template: str = "",
        # Ephemeral settings for profiles that don't have a persistent
        # world (browser-stream and any future "URL-driven" profiles).
        # When provided AND no world_id is given, this dict is used as
        # the per-session settings forwarded to the container via env
        # vars (e.g. ``{"target_url": "https://augmentum:6443/ui/cast-vrm/"}``).
        # When a world_id IS given, this is merged on top of the
        # world's stored settings so the caller can override on a
        # per-session basis without rewriting the world row.
        world_settings_override: dict | None = None,
    ) -> ConnectionInfo:
        if not user_id:
            raise RuntimeError("user_id required")
        profile = self._registry.get(profile_id)
        if not profile:
            raise RuntimeError(f"unknown profile {profile_id!r}")

        # Credit-budget admission. See _admit() docstring for the full
        # rule set. Replaces the old flat per-user count cap with a
        # resource-aware check that knows about profile cost, paused
        # state, and multi-user fairness. There's a small TOCTOU window
        # between this check and the create_session() insert below;
        # acceptable at our scale because the port pool has a hard
        # atomic ceiling one layer down.
        await self._admit(user_id=user_id, profile=profile)

        chosen_bitrate = int(bitrate_mbps or profile.default_bitrate_mbps)
        chosen_resolution = resolution or profile.default_resolution
        chosen_encoder = self._pick_encoder(profile, encoder)

        # Resolve world -> storage_path / settings (if any)
        storage_path = ""
        world_settings: dict = {}
        if world_id:
            world = await self._store.get_world(world_id, user_id=user_id)
            if not world:
                raise RuntimeError(f"world {world_id!r} not found for user")
            storage_path = world.get("storage_path", "")
            ws = world.get("settings_json", {})
            if isinstance(ws, dict):
                world_settings = ws
        # Apply caller-side overrides on top of any world-resolved
        # settings. Used by world-less profiles (browser-stream) to
        # pass per-session knobs like target_url that the entrypoint
        # then reads from env.
        if isinstance(world_settings_override, dict) and world_settings_override:
            world_settings = {**world_settings, **world_settings_override}

        # Allocate ports + create session row (status='starting')
        try:
            ports = await self._pool.allocate()
        except PortPoolExhausted as exc:
            raise RuntimeError("no streaming ports available") from exc

        session_id = await self._store.create_session(
            user_id=user_id,
            profile_id=profile_id,
            world_id=world_id,
            bitrate_mbps=chosen_bitrate,
            resolution=chosen_resolution,
            encoder=chosen_encoder,
            system_id=system_id,
        )

        # Mint + persist cast_input_token before the adapter call so a
        # crashing adapter still leaves a row the route can reap via
        # session_id. Empty template = no phone-as-controller, no token
        # minted, no env var passed to the container.
        cast_input_token = ""
        cast_input_bridge_url = ""
        if cast_input_bridge_url_template:
            cast_input_token = secrets.token_urlsafe(32)
            cast_input_bridge_url = cast_input_bridge_url_template.format(
                session_id=session_id, token=cast_input_token,
            )

        await self._store.update_session(
            session_id,
            user_id=user_id,
            stream_port=ports.stream_port,
            game_port=ports.game_port,
            cast_input_token=cast_input_token or None,
        )

        # Hand off to container adapter (Phase 2: real Docker; here: stub)
        try:
            container_id = await self._adapter.start(
                session_id=session_id,
                profile=profile,
                ports=ports,
                bitrate_mbps=chosen_bitrate,
                resolution=chosen_resolution,
                encoder=chosen_encoder,
                world_storage_path=storage_path,
                world_settings=world_settings,
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
        except Exception as exc:
            await self._fail_start(session_id, ports, user_id)
            log.warning(
                "game_stream_container_start_failed",
                session_id=session_id,
                profile_id=profile_id,
                error=str(exc),
            )
            # Pass through the underlying message so the route layer
            # (and therefore the user) sees the actual cause -- image
            # not built, port conflict, GPU missing, etc.
            raise RuntimeError(str(exc) or "container start failed") from exc

        await self._store.update_session(
            session_id, user_id=user_id, container_id=container_id,
        )

        if world_id:
            await self._store.update_world(
                world_id, user_id=user_id, touch_played=True,
            )

        # In host-network mode, selkies binds host:8080 directly --
        # PortBindings are ignored at the Docker layer, so the
        # allocated port pool entry never gets connected. Report
        # 8080 to the frontend so the iframe URL points at the
        # actual listening port.
        reported_stream_port = ports.stream_port
        if getattr(self._adapter, "host_network", False):
            reported_stream_port = 8080
        return ConnectionInfo(
            session_id=session_id,
            profile_id=profile_id,
            status=SessionStatus.STARTING.value,
            stream_port=reported_stream_port,
            game_port=ports.game_port,
            bitrate_mbps=chosen_bitrate,
            resolution=chosen_resolution,
            mouse_sensitivity=mouse_sensitivity,
            gamepad_enabled=gamepad_enabled,
            controller_deadzone=controller_deadzone,
            signaling_path=f"{self._signaling_prefix}/{session_id}",
        )

    async def mark_ready(self, session_id: str, *, user_id: str = "") -> bool:
        return await self._transition(session_id, SessionStatus.READY, user_id)

    async def mark_connected(self, session_id: str, *, user_id: str = "") -> bool:
        ok = await self._transition(session_id, SessionStatus.CONNECTED, user_id)
        if ok:
            await self._store.update_session(
                session_id, user_id=user_id, touch_seen=True,
            )
        return ok

    async def mark_idle(self, session_id: str, *, user_id: str = "") -> bool:
        return await self._transition(session_id, SessionStatus.IDLE, user_id)

    async def heartbeat(self, session_id: str, *, user_id: str = "") -> bool:
        """Touch an active viewer session and keep it out of idle cleanup.

        The browser stage calls this periodically while the iframe is
        mounted. It is deliberately idempotent: repeated heartbeats from
        an already-connected session should only refresh ``last_seen_at``,
        while a resumed ``ready``/``idle`` row is promoted back to
        ``connected``.
        """
        row = await self._store.get_session(session_id, user_id=user_id)
        if not row:
            return False
        status = row.get("status", "stopped")
        if GameStreamLifecycle.is_terminal(status):
            return False
        if status in (SessionStatus.READY.value, SessionStatus.IDLE.value):
            return await self._store.update_session(
                session_id,
                user_id=user_id,
                status=SessionStatus.CONNECTED.value,
                touch_seen=True,
            )
        return await self._store.update_session(
            session_id,
            user_id=user_id,
            touch_seen=True,
        )

    async def stop_session(
        self,
        session_id: str,
        *,
        user_id: str = "",
        reason: str = "clean",
    ) -> bool:
        row = await self._store.get_session(session_id, user_id=user_id)
        if not row:
            return False
        cur = row.get("status", "stopped")
        if GameStreamLifecycle.is_terminal(cur):
            return False

        # Transition to STOPPING
        try:
            GameStreamLifecycle.transition(cur, SessionStatus.STOPPING)
        except LifecycleTransitionError:
            pass
        await self._store.update_session(
            session_id, user_id=user_id, status=SessionStatus.STOPPING.value,
        )

        # Container teardown
        container_id = row.get("container_id") or ""
        if container_id:
            try:
                await self._adapter.stop(container_id)
            except Exception as exc:
                log.warning(
                    "game_stream_container_stop_failed",
                    session_id=session_id,
                    container_id=container_id,
                    error=str(exc),
                )

        # Release ports
        sp = row.get("stream_port")
        if isinstance(sp, int):
            await self._pool.release(sp)

        # Final status. Clear paused_at so a reaped paused row reads
        # cleanly in lists/inspects after the stop completes.
        await self._store.update_session(
            session_id,
            user_id=user_id,
            status=SessionStatus.STOPPED.value,
            exit_reason=reason,
            paused_at="",
        )

        # Best-effort: fire the on_session_stopped hook so external
        # subsystems (cast invite store) can clean up references to
        # a now-dead session. Pass the session's owning user_id so
        # the hook can address receiver broadcasts correctly.
        if self._on_session_stopped is not None:
            try:
                await self._on_session_stopped(
                    session_id, row.get("user_id", "") or "",
                )
            except Exception as exc:
                log.warning(
                    "game_stream_on_stop_hook_failed",
                    session_id=session_id, error=str(exc),
                )

        return True

    async def pause_session(
        self, session_id: str, *, user_id: str = "",
    ) -> bool:
        """Freeze the container via the cgroup freezer (docker pause).

        Allowed from CONNECTED / IDLE / READY. The container keeps its
        RAM and credit budget slot but stops consuming CPU. Resume is
        sub-second via ``resume_session``. The sweep loop auto-stops
        sessions whose ``paused_at`` exceeds ``paused_stop_seconds``.

        Returns False when the session isn't paused-eligible (not
        found, terminal, or already paused).
        """
        row = await self._store.get_session(session_id, user_id=user_id)
        if not row:
            return False
        cur = row.get("status", "")
        if cur == SessionStatus.PAUSED.value:
            return True  # idempotent
        if GameStreamLifecycle.is_terminal(cur):
            return False
        try:
            GameStreamLifecycle.transition(cur, SessionStatus.PAUSED)
        except LifecycleTransitionError:
            log.warning(
                "game_stream_pause_illegal_state",
                session_id=session_id, from_status=cur,
            )
            return False

        container_id = row.get("container_id") or ""
        if container_id:
            try:
                await self._adapter.pause(container_id)
            except Exception as exc:
                log.warning(
                    "game_stream_pause_adapter_failed",
                    session_id=session_id, container_id=container_id,
                    error=str(exc),
                )
                return False
        await self._store.update_session(
            session_id, user_id=user_id,
            status=SessionStatus.PAUSED.value,
            paused_at="now",
        )
        log.info(
            "game_stream_paused",
            session_id=session_id, user_id=user_id, from_status=cur,
        )
        return True

    async def resume_session(
        self, session_id: str, *, user_id: str = "",
    ) -> bool:
        """Thaw a paused container. Idempotent on already-resumed rows.

        Transitions PAUSED → CONNECTED (the client is expected to
        reconnect signaling shortly after). The receiver-side WebRTC
        peer renegotiates on the next signaling message; Selkies
        handles this on every tab-refresh today, so the resume UX is
        a sub-second blink rather than a full reload.
        """
        row = await self._store.get_session(session_id, user_id=user_id)
        if not row:
            return False
        cur = row.get("status", "")
        if cur != SessionStatus.PAUSED.value:
            return cur in (
                SessionStatus.CONNECTED.value,
                SessionStatus.READY.value,
                SessionStatus.IDLE.value,
            )

        container_id = row.get("container_id") or ""
        if container_id:
            try:
                await self._adapter.unpause(container_id)
            except Exception as exc:
                log.warning(
                    "game_stream_resume_adapter_failed",
                    session_id=session_id, container_id=container_id,
                    error=str(exc),
                )
                return False
        try:
            GameStreamLifecycle.transition(cur, SessionStatus.CONNECTED)
        except LifecycleTransitionError:
            pass
        await self._store.update_session(
            session_id, user_id=user_id,
            status=SessionStatus.CONNECTED.value,
            paused_at="",  # clear sentinel
            touch_seen=True,
        )
        log.info(
            "game_stream_resumed",
            session_id=session_id, user_id=user_id,
        )
        return True

    async def record_telemetry(
        self,
        *,
        session_id: str,
        user_id: str,
        rtt_ms: float | None = None,
        jitter_ms: float | None = None,
        packet_loss: float | None = None,
        bitrate_kbps: int | None = None,
        fps: float | None = None,
    ) -> None:
        await self._store.insert_telemetry(
            session_id=session_id,
            user_id=user_id,
            rtt_ms=rtt_ms,
            jitter_ms=jitter_ms,
            packet_loss=packet_loss,
            bitrate_kbps=bitrate_kbps,
            fps=fps,
        )

    # ── Reconcile / idle watch ──────────────────────────────────────

    async def reconcile_on_startup(self) -> None:
        """Walk live rows, rebuild port-pool, mark dead rows crashed."""
        rows = await self._store.list_running_unscoped()
        await self._pool.reconcile(rows)
        for row in rows:
            container_id = row.get("container_id") or ""
            sid = row["id"]
            uid = row["user_id"]
            alive = False
            if container_id:
                try:
                    alive = await self._adapter.is_alive(container_id)
                except Exception:
                    alive = False
            if not alive:
                cur = row.get("status", "stopped")
                # Best-effort transition; tolerate illegal jumps for
                # crash cleanup since we may be reconciling from any
                # interrupted state.
                try:
                    GameStreamLifecycle.transition(cur, SessionStatus.CRASHED)
                except LifecycleTransitionError:
                    pass
                await self._store.update_session(
                    sid,
                    user_id=uid,
                    status=SessionStatus.CRASHED.value,
                    exit_reason="reconcile_dead",
                )
                sp = row.get("stream_port")
                if isinstance(sp, int):
                    await self._pool.release(sp)

    async def sweep_idle(self) -> int:
        """Reconcile live sessions with reality and reap whatever's stale.

        Returns the number of sessions stopped. Called periodically by
        the idle watcher coroutine.

        Four reaping rules, in priority order:

        0) Liveness watchdog — for each live row, ask the adapter
           whether the container is still running. If not, the
           container died without going through stop_session (OOM,
           hard kill, daemon hiccup). Transition to CRASHED + release
           the port. This is the only path that frees credits for
           containers that died silently.
        1) IDLE status + idle_timeout elapsed → stop (clean
           game-stream idle handling driven by client-reported
           activity).
        2) PAUSED status + paused_stop_seconds elapsed → real stop
           (auto-stop a session whose container has been frozen too
           long; preserves RAM for active sessions).
        3) Profile is ``browser-stream`` AND ``last_seen_at`` is either
           missing or older than ``_render_stream_stale_seconds`` → stop.
           Belt-and-braces for the cast render path: render-stream
           sessions never transition to IDLE (no in-container client
           pings the runtime), so without this rule a missed detach
           hook leaves the agsp container burning ~4 cores forever.
           Aggressive default of 30 min — well past any reasonable real
           cast session length but short enough that a leak self-heals
           inside one watchdog cycle.
        """
        rows = await self._store.list_running_unscoped()
        now = datetime.now(timezone.utc)
        stopped = 0
        paused_stop_seconds = self._paused_stop_seconds
        for row in rows:
            status = row.get("status")
            profile_id = row.get("profile_id") or ""
            container_id = row.get("container_id") or ""
            last_seen = self._parse_ts(row.get("last_seen_at"))

            # Rule 0: liveness watchdog. Cheap (one Docker inspect per
            # row) and the only way to free credits for silently-dead
            # containers. Skip when container_id is empty — the row
            # is still in 'starting' before the adapter returned.
            if container_id:
                try:
                    alive = await self._adapter.is_alive(container_id)
                except Exception:
                    alive = False
                if not alive:
                    try:
                        GameStreamLifecycle.transition(
                            status or "", SessionStatus.CRASHED,
                        )
                    except LifecycleTransitionError:
                        pass
                    await self._store.update_session(
                        row["id"], user_id=row["user_id"],
                        status=SessionStatus.CRASHED.value,
                        exit_reason="watchdog_dead",
                    )
                    sp = row.get("stream_port")
                    if isinstance(sp, int):
                        await self._pool.release(sp)
                    log.warning(
                        "game_stream_watchdog_reaped",
                        session_id=row["id"], container_id=container_id,
                        prior_status=status,
                    )
                    stopped += 1
                    continue

            # Rule 1: standard idle timeout.
            if status == SessionStatus.IDLE.value and last_seen is not None:
                elapsed = (now - last_seen).total_seconds()
                if elapsed >= self._idle_timeout_seconds:
                    await self.stop_session(
                        row["id"], user_id=row["user_id"], reason="idle",
                    )
                    stopped += 1
                    continue

            # Rule 2: paused-stop timeout. paused_at tracks when the
            # pause began; auto-stop after the configured grace.
            if status == SessionStatus.PAUSED.value:
                paused_at = self._parse_ts(row.get("paused_at"))
                if paused_at is not None:
                    paused_s = (now - paused_at).total_seconds()
                    if paused_s >= paused_stop_seconds:
                        await self.stop_session(
                            row["id"], user_id=row["user_id"],
                            reason="paused_timeout",
                        )
                        stopped += 1
                        log.info(
                            "game_stream_paused_reaped",
                            session_id=row["id"],
                            paused_s=round(paused_s, 1),
                        )
                        continue

            # Rule 3: render-stream stale-session reaper. Catches the
            # cast disconnect path missing its teardown hook.
            if profile_id == "browser-stream":
                if last_seen is None:
                    started = self._parse_ts(row.get("created_at"))
                    if started is None:
                        continue
                    age_s = (now - started).total_seconds()
                else:
                    age_s = (now - last_seen).total_seconds()
                if age_s >= self._render_stream_stale_seconds:
                    await self.stop_session(
                        row["id"], user_id=row["user_id"],
                        reason="render_stream_stale",
                    )
                    stopped += 1
                    log.info(
                        "game_stream_render_stale_reaped",
                        session_id=row["id"], age_s=round(age_s, 1),
                    )
        return stopped

    # ── Internals ────────────────────────────────────────────────────

    def _pick_encoder(
        self, profile: GameProfile, requested: str | None,
    ) -> str:
        # Force x264enc until full GPU plumbing is verified.
        # Symptom that drove this: with NVENC selected by the
        # container probe, selkies' video pipeline crashes with
        # ``AttributeError: 'NoneType' object has no attribute
        # 'set_property'`` on cudaconvert -- the libnvidia-encode
        # libs are mounted by nvidia-container-toolkit but the
        # CUDA device (/dev/nvidia0) isn't fully exposed under
        # Docker Desktop, so cudaconvert can't initialise. The
        # crash kills the video peer; audio still works (no GPU
        # path), so the user hears it but sees an "infinite
        # loading" overlay because the viewer waits for both
        # peers to be connected.
        # x264enc has no GPU dependencies and produces a working
        # stream every time. Performance is fine on modern CPUs
        # at our 1280x720 / 30fps target.
        # TODO: re-enable hardware encode when we can probe for
        # cudaconvert specifically (not just nvh264enc) AND
        # /dev/nvidia0 is actually mountable.
        if requested and requested == "x264enc":
            return "x264enc"
        if requested and requested != "auto":
            log.info(
                "game_stream_encoder_override_to_x264",
                profile_id=profile.id,
                requested=requested,
                reason="cudaconvert nonetype crash with hw encoders",
            )
        return "x264enc"

    async def _admit(
        self, *, user_id: str, profile: GameProfile,
    ) -> None:
        """Resource-aware admission gate. Raises ConcurrentStreamLimitError
        when the request can't be accommodated.

        Two budgets, both consulted on every start:

        * **Active** — sum of cost_credits across non-paused live
          sessions. The encoder/CPU ceiling. Divided by active-user
          count for fair sharing; clamped per-user by ``user_hard_cap``.
        * **Resident** — sum of cost_credits across ALL live sessions
          (paused + non-paused). The RAM ceiling. Loose; triggers
          auto-eviction of the requesting user's oldest paused
          session before refusing.

        See docs/superpowers/specs/2026-06-02-game-stream-admission-control.md
        for the full design rationale.
        """
        rows = await self._store.list_running_unscoped()
        paused_v = SessionStatus.PAUSED.value
        stopped_v = SessionStatus.STOPPED.value
        crashed_v = SessionStatus.CRASHED.value

        def _cost(row: dict) -> int:
            p = self._registry.get(row.get("profile_id") or "")
            if p is None:
                return 1
            return max(1, int(getattr(p, "cost_credits", 1)))

        active_rows = [
            r for r in rows
            if r.get("status") not in (paused_v, stopped_v, crashed_v)
        ]
        paused_rows = [r for r in rows if r.get("status") == paused_v]

        host_active = sum(_cost(r) for r in active_rows)
        host_resident = host_active + sum(_cost(r) for r in paused_rows)
        user_active = sum(
            _cost(r) for r in active_rows if r.get("user_id") == user_id
        )
        user_resident = user_active + sum(
            _cost(r) for r in paused_rows if r.get("user_id") == user_id
        )

        active_users = len({r.get("user_id", "") for r in active_rows})
        fair_share = max(
            1, self._active_credit_budget // max(1, active_users),
        )
        # Solo user gets the whole budget (up to hard cap); shared
        # host divides budget by active-user count.
        user_active_cap = min(
            self._user_hard_cap,
            self._active_credit_budget if active_users <= 1 else fair_share,
        )

        new_cost = max(1, int(getattr(profile, "cost_credits", 1)))

        # Hard COUNT cap (number of containers, not credits). Kept as a
        # safety net alongside the credit-based caps so a misconfigured
        # cost_credits can't accidentally let a single user spawn a
        # huge number of cheap containers. max_concurrent_per_user is
        # still honored — settings can tighten it below user_hard_cap.
        user_active_count = len(
            [r for r in active_rows if r.get("user_id") == user_id],
        )
        if user_active_count + 1 > self._max_concurrent_per_user:
            raise ConcurrentStreamLimitError(
                f"user has {user_active_count}/{self._max_concurrent_per_user} "
                f"live streams",
            )

        if user_active + new_cost > user_active_cap:
            raise ConcurrentStreamLimitError(
                f"your active stream cap is {user_active_cap} credits "
                f"({active_users} user(s) sharing budget "
                f"{self._active_credit_budget}); "
                f"you'd be at {user_active + new_cost}",
            )
        if host_active + new_cost > self._active_credit_budget:
            raise ConcurrentStreamLimitError(
                f"host is at active capacity "
                f"({host_active}/{self._active_credit_budget} credits); "
                f"another active session must end first",
            )
        # User resident cap: 2× the active hard cap. Soft hoarding guard
        # that scales with the user's allowance.
        user_resident_cap = self._user_hard_cap * 2
        if user_resident + new_cost > user_resident_cap:
            raise ConcurrentStreamLimitError(
                f"you're holding too many paused sessions "
                f"({user_resident}/{user_resident_cap} resident credits); "
                f"resume or stop one first",
            )
        if host_resident + new_cost > self._resident_credit_budget:
            # Self-eviction: free the requesting user's oldest paused
            # session before blocking. Cross-user eviction is a
            # future lever; v1 fails closed when self-eviction can't
            # cover the cost.
            evicted_cost = await self._evict_oldest_paused(user_id=user_id)
            if host_resident - evicted_cost + new_cost > self._resident_credit_budget:
                raise ConcurrentStreamLimitError(
                    f"host RAM ceiling hit "
                    f"({host_resident}/{self._resident_credit_budget} "
                    f"resident credits); close another session first",
                )

    async def _evict_oldest_paused(self, *, user_id: str) -> int:
        """Stop the requesting user's oldest paused session, returning
        the credit count it was holding (0 if nothing was evicted).

        Used by ``_admit`` when the resident budget is full and the
        only way to admit is to make room. Picks the OLDEST paused row
        by paused_at to prefer evicting forgotten/stale freezes over
        ones the user just paused.
        """
        rows = await self._store.list_sessions_for_user(
            user_id=user_id, status=SessionStatus.PAUSED.value, limit=100,
        )
        if not rows:
            return 0
        # Oldest paused_at first; rows without paused_at sort to the end
        # (defensive — should never happen for a PAUSED row).
        rows.sort(key=lambda r: (r.get("paused_at") or "9999"))
        oldest = rows[0]
        sid = oldest.get("id") or ""
        if not sid:
            return 0
        profile = self._registry.get(oldest.get("profile_id") or "")
        cost = max(1, int(getattr(profile, "cost_credits", 1))) if profile else 1
        log.info(
            "game_stream_eviction_paused",
            session_id=sid, user_id=user_id, cost=cost,
        )
        await self.stop_session(sid, user_id=user_id, reason="evicted_paused")
        return cost

    async def _transition(
        self,
        session_id: str,
        target: SessionStatus,
        user_id: str,
    ) -> bool:
        row = await self._store.get_session(session_id, user_id=user_id)
        if not row:
            return False
        cur = row.get("status", "stopped")
        try:
            GameStreamLifecycle.transition(cur, target)
        except LifecycleTransitionError:
            log.warning(
                "game_stream_illegal_transition",
                session_id=session_id,
                from_status=cur,
                to_status=target.value,
            )
            return False
        await self._store.update_session(
            session_id, user_id=user_id, status=target.value,
        )
        return True

    async def _fail_start(
        self,
        session_id: str | None,
        ports: PortAllocation,
        user_id: str,
    ) -> None:
        await self._pool.release(ports.stream_port)
        if session_id:
            await self._store.update_session(
                session_id,
                user_id=user_id,
                status=SessionStatus.CRASHED.value,
                exit_reason="start_failed",
            )

    @staticmethod
    def _parse_ts(value: object) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            # SQLite datetime('now') format -- 'YYYY-MM-DD HH:MM:SS'
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc,
            )
        except ValueError:
            return None
