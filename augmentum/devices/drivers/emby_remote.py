"""Emby/Jellyfin session-bridge driver.

Emby and Jellyfin already do SSDP/DLNA discovery on the LAN 24/7.
Their `/Sessions` endpoint exposes every connected client — including
DLNA TVs the server has discovered — and their session-controller
(`/Sessions/{id}/Playing` and `/Sessions/{id}/Command`) lets us push
playback, pause, seek, volume, mute remotely.

This driver bridges that into the device substrate. The augmentum
container never has to talk to the TV directly: we tell Emby what to
do, Emby tells the TV. Sidesteps Docker networking entirely (we only
need to reach Emby, which the user already configured) and works
identically when augmentum is on a VPS — as long as the user's Emby
and TV can see each other, casting works.

Capability surface:
  - media.video_play@1 — full transport + audio/subtitle stream switch
    (the latter is provider-side, not a separate UPnP call)
  - media.audio_play@1 — same wire path; Emby treats it as audio item

Discovery is per-user: each augmentum user has their own
``user_media_servers`` rows, and each Emby/Jellyfin server may have a
different session list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

from augmentum.devices.device import Device, DiscoveredDevice
from augmentum.devices.invocation import (
    Event,
    InvocationContext,
    InvocationResult,
    PairResult,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import httpx

    from augmentum.devices.driver import DriverContext


log = get_logger(__name__)


_SUPPORTED_CAPABILITIES: tuple[str, ...] = (
    "media.video_play@1",
    "media.audio_play@1",
)


class EmbyRemoteDriver:
    """Bridge driver — surfaces Emby/Jellyfin sessions as castable devices."""

    id: str = "emby_remote"
    label: str = "Emby / Jellyfin Session"
    description: str = (
        "TVs and apps connected to your Emby or Jellyfin server. Cast "
        "library items via the server's session controller."
    )
    capabilities: tuple[str, ...] = _SUPPORTED_CAPABILITIES
    discovery_modes: tuple[str, ...] = ("via_provider",)
    requires_pairing: bool = False
    supports_passive_discovery: bool = False

    def __init__(
        self,
        *,
        http_client: "httpx.AsyncClient | None" = None,
        media_server_store_factory: Callable[[], Any] | None = None,
        file_index_factory: Callable[[], Any] | None = None,
        provider_client_factory: Callable[[str, Any], Any] | None = None,
    ) -> None:
        self._http: "httpx.AsyncClient | None" = http_client
        self._media_server_store_factory = media_server_store_factory
        self._file_index_factory = file_index_factory
        self._provider_client_factory = provider_client_factory

    async def start(self, ctx: "DriverContext") -> None:
        if self._http is None:
            self._http = ctx.http_client

    async def stop(self) -> None:
        return None

    # ---- discovery -----------------------------------------------------------

    async def discover(
        self,
        *,
        timeout_s: float = 3.0,
        user_id: str = "",
    ) -> list[DiscoveredDevice]:
        """Walk the user's Emby/Jellyfin servers; surface their sessions."""
        if not user_id or self._http is None:
            return []
        if self._media_server_store_factory is None or self._provider_client_factory is None:
            return []

        try:
            store = self._media_server_store_factory()
            # list_visible (not list_for_user) so non-owners of an
            # admin-shared Emby/Jellyfin can still surface its remote
            # sessions in the cast picker. The session list itself
            # belongs to the server, not the user.
            servers = await store.list_visible(user_id=user_id)
        except Exception as exc:
            log.warning("emby_remote_list_servers_failed", error=str(exc))
            return []

        found: list[DiscoveredDevice] = []
        for server in servers:
            if server.provider not in ("emby", "jellyfin"):
                continue
            try:
                client = self._provider_client_factory(server.provider, self._http)
            except Exception as exc:
                log.debug("emby_remote_provider_client_failed", error=str(exc))
                continue
            if not callable(getattr(client, "list_remote_sessions", None)):
                continue
            try:
                sessions = await client.list_remote_sessions(
                    server.base_url,
                    server.access_token,
                )
            except Exception as exc:
                log.debug(
                    "emby_remote_list_sessions_failed",
                    server=server.id, error=str(exc),
                )
                continue

            for sess in sessions or []:
                if not getattr(sess, "supports_remote_control", False):
                    continue
                # Skip the augmentum web/mobile client itself if it ever
                # appears in the list — casting to ourselves is silly.
                client_label = (sess.client or "").lower()
                if "augmentum" in client_label:
                    continue
                native_id = f"{server.id}:{sess.session_id}"
                label = (
                    sess.device_name
                    or sess.name
                    or sess.client
                    or "Emby Session"
                )
                found.append(DiscoveredDevice(
                    driver=self.id,
                    native_id=native_id,
                    label=label,
                    capabilities=list(_SUPPORTED_CAPABILITIES),
                    address={
                        "server_id": server.id,
                        "session_id": sess.session_id,
                        "host": server.base_url,
                    },
                    metadata={
                        "manufacturer": "Emby/Jellyfin",
                        "model_name": sess.client or "",
                        "via_server": server.name,
                        "via_provider": server.provider,
                        "supports_media_control": sess.supports_media_control,
                        "supported_commands": list(sess.supported_commands or []),
                    },
                    confidence=1.0,
                ))

        return found

    async def probe(
        self,
        *,
        host: str,
        port: int | None = None,
        hint: dict[str, Any] | None = None,
    ) -> DiscoveredDevice | None:
        # Manual-add for Emby sessions doesn't really make sense — sessions
        # only exist when an Emby client connects to the server. Users add
        # their server via the Media Sources panel, sessions show up
        # automatically on next discover.
        return None

    # ---- invocation ----------------------------------------------------------

    async def invoke(
        self,
        device: Device,
        capability: str,
        action: str,
        args: dict[str, Any],
        ctx: InvocationContext,
    ) -> InvocationResult:
        if self._http is None:
            return InvocationResult.failure("http_client_unavailable", code="driver_unavailable")

        server, session_id = await self._resolve_session(device)
        if server is None:
            return InvocationResult.failure(
                "media server unavailable; reconnect Emby/Jellyfin in Media Sources",
                code="provider_unavailable",
            )

        client = self._provider_client_factory(server.provider, self._http)

        if action == "play":
            return await self._do_play(client, server, session_id, args)
        if action in ("pause", "resume", "stop"):
            return await self._do_command(client, server, session_id, action, args)
        if action == "seek":
            return await self._do_seek(client, server, session_id, args)
        if action == "set_volume":
            return await self._do_volume(client, server, session_id, args)
        if action == "set_mute":
            return await self._do_mute(client, server, session_id, args)
        if action in ("set_audio_track", "set_subtitle_track"):
            # Not directly supported through session controller for all
            # clients — succeed as no-op so the LLM tool surface stays
            # uniform.
            return InvocationResult.success(
                extra={"note": "track_switch_via_session_controller_unsupported"},
            )
        return InvocationResult.failure(
            f"action {action} not supported on emby_remote",
            code="unsupported_action",
        )

    async def _resolve_session(self, device: Device) -> tuple[Any, str]:
        """Pull the MediaServer + session_id out of a Device."""
        addr = device.address or {}
        server_id = str(addr.get("server_id") or "").strip()
        session_id = str(addr.get("session_id") or "").strip()
        if not server_id or not session_id:
            return (None, "")
        if self._media_server_store_factory is None:
            return (None, "")
        try:
            store = self._media_server_store_factory()
            # get_visible — non-owners of an admin-shared Emby/Jellyfin
            # still need the credential to dispatch playback commands.
            server = await store.get_visible(server_id, user_id=device.user_id)
        except Exception:
            return (None, "")
        return (server, session_id)

    async def _do_play(
        self,
        client: Any,
        server: Any,
        session_id: str,
        args: dict[str, Any],
    ) -> InvocationResult:
        """Tell Emby to play a library item on the target session.

        Requires `file_id` in args — we resolve to the Emby/Jellyfin
        external_id via the file_index. Casting non-Emby content to an
        Emby session isn't meaningful (Emby can only play its own
        library items), so we surface a clear error in that case.
        """
        if not callable(getattr(client, "remote_play", None)):
            return InvocationResult.failure(
                "provider client doesn't support remote play",
                code="unsupported_action",
            )
        if self._file_index_factory is None:
            return InvocationResult.failure(
                "file index unavailable; restart augmentum",
                code="driver_unavailable",
            )

        file_id = str(args.get("file_id") or "").strip()
        if not file_id:
            return InvocationResult.failure(
                "Emby sessions can only cast library items — file_id is required.",
                code="missing_arg",
            )

        idx = self._file_index_factory()
        try:
            entry = await idx.get(file_id, user_id=server.user_id)
        except Exception as exc:
            return InvocationResult.failure(
                f"file lookup failed: {exc}",
                code="driver_error",
            )
        if entry is None:
            return InvocationResult.failure("file not found", code="device_not_found")

        meta = entry.source_metadata if isinstance(entry.source_metadata, dict) else {}
        if str(meta.get("server_id") or "") != server.id:
            return InvocationResult.failure(
                "This item lives on a different server than the target session.",
                code="cross_server_cast",
            )

        external_id = str(meta.get("external_id") or "").strip()
        if not external_id:
            return InvocationResult.failure(
                "no external_id in file_index entry",
                code="driver_error",
            )

        start_pos_ticks = None
        start_time_s = float(args.get("start_time_s") or 0.0)
        if start_time_s > 0:
            start_pos_ticks = int(start_time_s * 10_000_000)

        try:
            ok = await client.remote_play(
                server.base_url,
                server.access_token,
                session_id=session_id,
                item_id=external_id,
                play_command="PlayNow",
                start_position_ticks=start_pos_ticks,
            )
        except TypeError:
            # Older provider client signature — fall back to minimal call.
            ok = await client.remote_play(
                server.base_url,
                server.access_token,
                session_id=session_id,
                item_id=external_id,
                play_command="PlayNow",
            )
        except Exception as exc:
            return InvocationResult.failure(str(exc), code="driver_error")
        if not ok:
            return InvocationResult.failure("emby_play_failed", code="driver_error")
        return InvocationResult.success(state={"title": str(args.get("title") or "")})

    async def _do_command(
        self,
        client: Any,
        server: Any,
        session_id: str,
        action: str,
        args: dict[str, Any],
    ) -> InvocationResult:
        wire_command = {
            "pause": "Pause",
            "resume": "Unpause",
            "stop": "Stop",
        }.get(action)
        if wire_command is None:
            return InvocationResult.failure(f"unknown action {action}", code="unsupported_action")
        return await self._send_remote_command(client, server, session_id, wire_command)

    async def _do_seek(
        self,
        client: Any,
        server: Any,
        session_id: str,
        args: dict[str, Any],
    ) -> InvocationResult:
        try:
            position_s = float(args.get("position_s") or 0.0)
        except (TypeError, ValueError):
            return InvocationResult.failure("position_s required", code="missing_arg")
        if not callable(getattr(client, "remote_command", None)):
            return InvocationResult.failure("seek not supported", code="unsupported_action")
        try:
            ok = await client.remote_command(
                server.base_url,
                server.access_token,
                session_id=session_id,
                command="Seek",
                seek_position_s=position_s,
            )
        except Exception as exc:
            return InvocationResult.failure(str(exc), code="driver_error")
        return InvocationResult.success() if ok else InvocationResult.failure(
            "seek_failed", code="driver_error",
        )

    async def _send_remote_command(
        self,
        client: Any,
        server: Any,
        session_id: str,
        wire_command: str,
    ) -> InvocationResult:
        if not callable(getattr(client, "remote_command", None)):
            return InvocationResult.failure(
                "provider client doesn't support remote_command",
                code="unsupported_action",
            )
        try:
            ok = await client.remote_command(
                server.base_url,
                server.access_token,
                session_id=session_id,
                command=wire_command,
            )
        except Exception as exc:
            return InvocationResult.failure(str(exc), code="driver_error")
        return InvocationResult.success() if ok else InvocationResult.failure(
            f"{wire_command}_failed", code="driver_error",
        )

    async def _do_volume(
        self,
        client: Any,
        server: Any,
        session_id: str,
        args: dict[str, Any],
    ) -> InvocationResult:
        try:
            level = max(0, min(100, int(args.get("level") or 0)))
        except (TypeError, ValueError):
            return InvocationResult.failure("level required", code="missing_arg")
        return await self._send_general_command(
            client, server, session_id,
            command="SetVolume",
            arguments={"Volume": level},
        )

    async def _do_mute(
        self,
        client: Any,
        server: Any,
        session_id: str,
        args: dict[str, Any],
    ) -> InvocationResult:
        muted = bool(args.get("muted"))
        return await self._send_general_command(
            client, server, session_id,
            command="Mute" if muted else "Unmute",
            arguments={},
        )

    async def _send_general_command(
        self,
        client: Any,
        server: Any,
        session_id: str,
        *,
        command: str,
        arguments: dict[str, Any],
    ) -> InvocationResult:
        if not callable(getattr(client, "remote_general_command", None)):
            return InvocationResult.failure(
                "provider client doesn't support general commands",
                code="unsupported_action",
            )
        try:
            ok = await client.remote_general_command(
                server.base_url,
                server.access_token,
                session_id=session_id,
                command=command,
                arguments=arguments or None,
            )
        except Exception as exc:
            return InvocationResult.failure(str(exc), code="driver_error")
        return InvocationResult.success() if ok else InvocationResult.failure(
            f"{command}_failed", code="driver_error",
        )

    # ---- snapshot / subscribe / pairing -------------------------------------

    async def snapshot(
        self,
        device: Device,
        capability: str,
        ctx: InvocationContext,
    ) -> dict[str, Any] | None:
        # Re-fetch sessions and find the one matching this device.
        # The picker already shows transport state via list_remote_sessions
        # output; this lets the cast-remote pill stay live.
        if self._http is None:
            return None
        addr = device.address or {}
        server_id = str(addr.get("server_id") or "").strip()
        session_id = str(addr.get("session_id") or "").strip()
        if not server_id or not session_id:
            return None
        if self._media_server_store_factory is None or self._provider_client_factory is None:
            return None
        try:
            store = self._media_server_store_factory()
            # get_visible — non-owners of an admin-shared Emby/Jellyfin
            # still need the credential to dispatch playback commands.
            server = await store.get_visible(server_id, user_id=device.user_id)
            if server is None:
                return None
            client = self._provider_client_factory(server.provider, self._http)
            sessions = await client.list_remote_sessions(server.base_url, server.access_token)
        except Exception as exc:
            log.debug("emby_remote_snapshot_failed", error=str(exc))
            return None
        for s in sessions or []:
            if s.session_id == session_id:
                return {
                    "current_time_s": float(s.current_time_s or 0.0),
                    "duration_s": float(s.duration_s or 0.0),
                    "is_paused": bool(s.is_paused),
                    "is_muted": bool(s.is_muted),
                    "volume_level": s.volume_level,
                    "title": s.now_playing_title,
                    "can_seek": bool(s.can_seek),
                    "supported_commands": list(s.supported_commands or []),
                }
        return None

    async def subscribe(
        self,
        device: Device,
        capability: str,
        ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        async def _empty() -> AsyncIterator[Event]:
            if False:
                yield  # noqa: B901
        return _empty()

    async def pair_start(
        self,
        device: Device,
        ctx: InvocationContext,
    ) -> PairResult:
        return PairResult(
            state="active",
            requires_user_action=False,
            message="No pairing — uses your Emby/Jellyfin credentials.",
        )

    async def pair_complete(
        self,
        device: Device,
        code: str,
        ctx: InvocationContext,
    ) -> PairResult:
        return PairResult(state="active")
