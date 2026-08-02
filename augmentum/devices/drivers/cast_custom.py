"""Server-side Google Cast sender.

Augmentum-as-orchestrator: this driver runs the Cast SDK *inside the
augmentum container* using `pychromecast`, instead of in the user's
browser. Why server-side:

  - Off-network casts work. Phone on cell hits the augmentum API;
    augmentum reaches the TV on the LAN; phone never has to see the TV
    itself. No Tailscale required for the *user's experience*; just for
    augmentum's reachability from outside.
  - Multi-sender state stays consistent — augmentum is the single
    source of truth for what's on the TV. Both phone and laptop's
    cast-remote pills mirror the same state.
  - No browser CSP / mixed-content / cert-validation drama. The Cast
    SDK protocol is TLS + protobuf; pychromecast handles all of it.
  - Cast-Receiver custom apps (Phase 3 work) consume the same control
    plane — when we register our own App ID and ship a custom receiver,
    only this driver changes.

Capability surface (initial):
  - media.video_play@1
  - media.audio_play@1
  - display.image_show@1

The MVP targets the Default Media Receiver (`CC1AD845`) — Google's
built-in app that plays standard media URLs with title + cover art. The
custom receiver (multi-scene, comic + game emulation + VRM call, etc.)
becomes a later phase that swaps the App ID and rewrites
``_play_to_receiver`` to send custom-namespace messages.

Discovery is mDNS-based and runs server-side; on Docker Desktop where
the container can't always see LAN multicast, manual-add by IP is the
fallback and works through Docker NAT cleanly (TCP connect crosses).
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING, Any, AsyncIterator

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
    "display.image_show@1",
)


# Default Media Receiver app ID — built into every Cast device, no
# developer registration required. Renders title, artwork, transport
# controls. We swap to a custom App ID once the augmentum receiver app
# is built (Phase 3).
DEFAULT_MEDIA_RECEIVER_APP_ID = "CC1AD845"


class CastCustomDriver:
    """Google Cast sender — server-side controller."""

    id: str = "cast"
    label: str = "Google Cast"
    description: str = (
        "Chromecast Built-in, Google TV, Android TV, Cast-capable speakers — "
        "controlled from the augmentum server, so off-network and multi-sender "
        "scenarios work without browser-side Cast SDK."
    )
    capabilities: tuple[str, ...] = _SUPPORTED_CAPABILITIES
    discovery_modes: tuple[str, ...] = ("mdns",)
    requires_pairing: bool = False
    supports_passive_discovery: bool = False

    def __init__(self) -> None:
        # Per-device persistent connections. pychromecast's Chromecast
        # objects own a background socket-reader thread; reusing one
        # per device keeps latency down vs reconnecting every call.
        self._connections: dict[str, Any] = {}  # native_id -> Chromecast
        self._lock = asyncio.Lock()
        self._available: bool | None = None

    # ---- lifecycle ----------------------------------------------------------

    async def start(self, ctx: "DriverContext") -> None:
        # Confirm pychromecast is importable. If not, the driver still
        # registers (so the registry contract stays clean) but every
        # call returns "driver_unavailable" instead of crashing.
        try:
            import pychromecast  # noqa: F401
            self._available = True
            log.info("cast_driver_ready")
        except ImportError as exc:
            self._available = False
            log.warning("cast_driver_pychromecast_missing", error=str(exc))

    async def stop(self) -> None:
        async with self._lock:
            casts = list(self._connections.values())
            self._connections.clear()
        for cast in casts:
            try:
                await asyncio.to_thread(cast.disconnect, blocking=False)
            except Exception as exc:
                log.debug("cast_disconnect_failed", error=str(exc))

    # ---- discovery ----------------------------------------------------------

    async def discover(
        self,
        *,
        timeout_s: float = 3.0,
        user_id: str = "",  # noqa: ARG002 — Cast discovery is LAN-global
    ) -> list[DiscoveredDevice]:
        if not self._available:
            return []
        try:
            return await asyncio.to_thread(
                self._sync_discover,
                max(1.0, float(timeout_s)),
            )
        except Exception as exc:
            log.debug("cast_discover_failed", error=str(exc))
            return []

    @staticmethod
    def _sync_discover(timeout_s: float) -> list[DiscoveredDevice]:
        """Blocking mDNS scan; returns DiscoveredDevices.

        Critical detail: pychromecast's Chromecast objects each own a
        background socket-reader thread that keeps retrying to connect
        to its host:8009. False-positive mDNS hits (routers, IoT
        devices) would otherwise leak threads forever. We snapshot the
        metadata up front and disconnect each cast object so those
        retry loops actually terminate.
        """
        import pychromecast
        chromecasts, browser = pychromecast.get_chromecasts(timeout=timeout_s)
        try:
            return [CastCustomDriver._cast_info_to_discovered(c) for c in chromecasts]
        finally:
            for cast in chromecasts:
                try:
                    cast.disconnect(blocking=False)
                except Exception:
                    log.debug("cast_custom_disconnect_failed", exc_info=True)
            try:
                pychromecast.discovery.stop_discovery(browser)
            except Exception:
                log.debug("cast_custom_stop_discovery_failed", exc_info=True)

    async def probe(
        self,
        *,
        host: str,
        port: int | None = None,
        hint: dict[str, Any] | None = None,
    ) -> DiscoveredDevice | None:
        """Manual-add by IP. The Docker-Desktop-friendly path when mDNS
        doesn't escape the container: user enters their TV's IP, we
        TCP-connect via Cast protocol and fill in the metadata."""
        if not self._available or not host:
            return None
        try:
            cast = await asyncio.to_thread(
                self._sync_connect_by_host,
                host,
                int(port) if port else 8009,
            )
        except Exception as exc:
            log.debug("cast_probe_failed", host=host, error=str(exc))
            return None
        if cast is None:
            return None
        try:
            return self._cast_info_to_discovered(cast)
        finally:
            # Don't keep the probe-time connection cached; the registry
            # will create a real one on first invoke if the user saves
            # this device.
            try:
                await asyncio.to_thread(cast.disconnect, blocking=False)
            except Exception as exc:
                log.debug("cast_custom_probe_disconnect_failed", error=str(exc))

    @staticmethod
    def _sync_connect_by_host(host: str, port: int) -> Any | None:
        import pychromecast
        # Empty UUID/model/friendly_name — pychromecast fills them in
        # after wait(). Some models reject the connect with empty
        # values; if so, the probe fails and the user falls back to
        # browser-side Cast discovery.
        host_info = (host, port, "", "", "")
        try:
            cast = pychromecast.get_chromecast_from_host(host_info)
        except Exception:
            return None
        try:
            cast.wait(timeout=5.0)
            return cast
        except Exception:
            try:
                cast.disconnect(blocking=False)
            except Exception as exc:
                log.debug("cast_disconnect_after_wait_failed", host=host, port=port, error=str(exc))
            return None

    @staticmethod
    def _cast_info_to_discovered(cast: Any) -> DiscoveredDevice:
        info = getattr(cast, "cast_info", None)
        host = ""
        port = 8009
        uuid = ""
        friendly = ""
        model = ""
        manufacturer = ""
        if info is not None:
            host = str(getattr(info, "host", "") or "").strip()
            port = int(getattr(info, "port", 8009) or 8009)
            uuid = str(getattr(info, "uuid", "") or "").strip()
            friendly = str(getattr(info, "friendly_name", "") or "").strip()
            model = str(getattr(info, "model_name", "") or "").strip()
            manufacturer = str(getattr(info, "manufacturer", "") or "").strip()
        return DiscoveredDevice(
            driver="cast",
            native_id=uuid or f"{host}:{port}",
            label=friendly or model or host or "Cast Device",
            capabilities=list(_SUPPORTED_CAPABILITIES),
            address={"host": host, "port": port, "uuid": uuid},
            metadata={
                "manufacturer": manufacturer or "Google",
                "model_name": model,
                "via_protocol": "cast",
            },
            confidence=1.0,
        )

    # ---- invocation ---------------------------------------------------------

    async def invoke(
        self,
        device: Device,
        capability: str,
        action: str,
        args: dict[str, Any],
        ctx: InvocationContext,
    ) -> InvocationResult:
        if not self._available:
            return InvocationResult.failure(
                "pychromecast not installed in this build",
                code="driver_unavailable",
            )

        cast = await self._get_or_connect(device)
        if cast is None:
            return InvocationResult.failure(
                "could not connect to Cast device",
                code="connect_failed",
                retryable=True,
            )

        try:
            return await self._dispatch(cast, capability, action, args)
        except Exception as exc:
            log.warning(
                "cast_invoke_failed",
                action=action,
                device=device.id,
                error=str(exc),
            )
            return InvocationResult.failure(str(exc), code="driver_error")

    async def _dispatch(
        self,
        cast: Any,
        capability: str,
        action: str,
        args: dict[str, Any],
    ) -> InvocationResult:
        mc = cast.media_controller

        if action in ("play", "show"):
            content_url = str(
                args.get("content_url")
                or args.get("image_url")
                or "",
            ).strip()
            if not content_url:
                return InvocationResult.failure(
                    "content_url required",
                    code="missing_arg",
                )
            content_type = str(args.get("content_type") or "").strip()
            if not content_type:
                content_type = self._default_content_type(capability)
            title = str(args.get("title") or "Item")
            poster = str(
                args.get("poster_url")
                or args.get("cover_url")
                or "",
            ).strip()
            metadata = self._build_metadata(args, capability)
            start_time_s = float(args.get("start_time_s") or 0.0)

            await asyncio.to_thread(
                mc.play_media,
                content_url,
                content_type,
                title=title,
                thumb=poster or None,
                metadata=metadata,
                stream_type="BUFFERED",
                autoplay=True,
                current_time=start_time_s if start_time_s > 0 else None,
            )
            try:
                await asyncio.to_thread(mc.block_until_active, 8.0)
            except Exception as exc:
                log.debug("cast_block_until_active_timeout", error=str(exc))

            return InvocationResult.success(state=self._snapshot_state(cast))

        if action == "pause":
            await asyncio.to_thread(mc.pause)
            return InvocationResult.success(state=self._snapshot_state(cast))

        if action == "resume":
            await asyncio.to_thread(mc.play)
            return InvocationResult.success(state=self._snapshot_state(cast))

        if action == "stop":
            await asyncio.to_thread(mc.stop)
            return InvocationResult.success()

        if action == "seek":
            try:
                position_s = float(args.get("position_s") or 0.0)
            except (TypeError, ValueError):
                return InvocationResult.failure(
                    "position_s must be a number",
                    code="missing_arg",
                )
            await asyncio.to_thread(mc.seek, position_s)
            return InvocationResult.success(state=self._snapshot_state(cast))

        if action == "set_volume":
            try:
                level = max(0, min(100, int(args.get("level") or 0)))
            except (TypeError, ValueError):
                return InvocationResult.failure(
                    "level must be 0-100",
                    code="missing_arg",
                )
            # pychromecast wants 0.0-1.0
            await asyncio.to_thread(cast.set_volume, level / 100.0)
            return InvocationResult.success(state=self._snapshot_state(cast))

        if action == "set_mute":
            muted = bool(args.get("muted"))
            await asyncio.to_thread(cast.set_volume_muted, muted)
            return InvocationResult.success(state=self._snapshot_state(cast))

        if action in ("set_audio_track", "set_subtitle_track"):
            # Default Media Receiver doesn't expose track-switching via
            # the standard MediaController. The substrate accepts these
            # so the LLM tool surface stays uniform; we no-op cleanly.
            return InvocationResult.success(extra={
                "note": "track_switching_not_supported_on_default_receiver",
            })

        if action == "clear":
            await asyncio.to_thread(mc.stop)
            return InvocationResult.success()

        return InvocationResult.failure(
            f"action {action} not supported on cast driver",
            code="unsupported_action",
        )

    @staticmethod
    def _default_content_type(capability: str) -> str:
        if capability == "display.image_show@1":
            return "image/jpeg"
        if capability == "media.audio_play@1":
            return "audio/mpeg"
        return "video/mp4"

    @staticmethod
    def _build_metadata(args: dict[str, Any], capability: str) -> dict[str, Any]:
        """Construct a Cast metadata dict for the LOAD message.

        pychromecast accepts a plain dict; the keys map to Cast's
        MediaMetadata schema. We populate generic fields by default and
        let the receiver display whatever it understands.
        """
        meta: dict[str, Any] = {}
        if capability == "display.image_show@1":
            meta["metadataType"] = 4  # PHOTO
        elif capability == "media.audio_play@1":
            meta["metadataType"] = 3  # MUSIC_TRACK
        else:
            meta["metadataType"] = 1  # MOVIE / GENERIC
        for src_key, dest_key in [
            ("title", "title"),
            ("subtitle", "subtitle"),
            ("artist", "artist"),
            ("album", "albumName"),
            ("author", "creator"),
        ]:
            value = args.get(src_key)
            if value:
                meta[dest_key] = str(value)
        return meta

    # ---- snapshot / subscribe -----------------------------------------------

    async def snapshot(
        self,
        device: Device,
        capability: str,
        ctx: InvocationContext,
    ) -> dict[str, Any] | None:
        if not self._available:
            return None
        cast = await self._get_or_connect(device, allow_connect=False)
        if cast is None:
            return None
        return self._snapshot_state(cast)

    @staticmethod
    def _snapshot_state(cast: Any) -> dict[str, Any]:
        """Pull current MediaController + CastStatus into the substrate
        state dict the cast-remote pill expects."""
        try:
            mc_status = getattr(cast.media_controller, "status", None)
            cast_status = getattr(cast, "status", None)
            current_time_s = float(getattr(mc_status, "current_time", 0.0) or 0.0)
            duration_s = float(getattr(mc_status, "duration", 0.0) or 0.0)
            player_state = str(getattr(mc_status, "player_state", "") or "")
            is_paused = player_state in ("PAUSED",)
            volume_level = getattr(cast_status, "volume_level", None)
            volume_int = int(round(float(volume_level) * 100)) if volume_level is not None else None
            is_muted = bool(getattr(cast_status, "volume_muted", False))
            return {
                "current_time_s": current_time_s,
                "duration_s": duration_s,
                "is_paused": is_paused,
                "is_muted": is_muted,
                "volume_level": volume_int,
                "title": str(getattr(mc_status, "title", "") or ""),
                "receiver_state": player_state,
                "can_seek": True,
            }
        except Exception as exc:
            log.debug("cast_snapshot_state_failed", error=str(exc))
            return {}

    async def subscribe(
        self,
        device: Device,
        capability: str,
        ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        # Cast push events arrive on a background thread inside
        # pychromecast. Bridging that to the substrate's async
        # EventBus is Phase 3 work; for now poll-based snapshots from
        # the cast-remote pill cover the use case.
        async def _empty() -> AsyncIterator[Event]:
            if False:
                yield  # noqa: B901
        return _empty()

    # ---- pairing (no-op) ----------------------------------------------------

    async def pair_start(
        self,
        device: Device,
        ctx: InvocationContext,
    ) -> PairResult:
        return PairResult(
            state="active",
            requires_user_action=False,
            message="Cast does not require pairing",
        )

    async def pair_complete(
        self,
        device: Device,
        code: str,
        ctx: InvocationContext,
    ) -> PairResult:
        return PairResult(state="active")

    # ---- connection management ----------------------------------------------

    async def _get_or_connect(
        self,
        device: Device,
        *,
        allow_connect: bool = True,
    ) -> Any | None:
        async with self._lock:
            existing = self._connections.get(device.native_id)
            if existing is not None and self._is_alive(existing):
                return existing
            self._connections.pop(device.native_id, None)

        if not allow_connect:
            return None

        addr = device.address or {}
        host = str(addr.get("host") or "").strip()
        port = int(addr.get("port") or 8009)
        if not host:
            return None

        cast = await asyncio.to_thread(self._sync_connect_by_host, host, port)
        if cast is None:
            return None

        async with self._lock:
            # Race: another invocation may have connected first.
            existing = self._connections.get(device.native_id)
            if existing is not None and self._is_alive(existing):
                try:
                    await asyncio.to_thread(cast.disconnect, blocking=False)
                except Exception as exc:
                    log.debug("cast_race_loser_disconnect_failed", device_id=device.native_id, error=str(exc))
                return existing
            self._connections[device.native_id] = cast
            return cast

    @staticmethod
    def _is_alive(cast: Any) -> bool:
        sc = getattr(cast, "socket_client", None)
        if sc is None:
            return False
        try:
            return bool(sc.is_alive())
        except Exception:
            return False
