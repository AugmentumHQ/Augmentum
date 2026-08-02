"""DLNA / UPnP MediaRenderer driver.

Wraps the existing dependency-free implementation at
``augmentum/media/receivers/dlna.py`` (SSDP discovery, AVTransport SOAP,
RenderingControl) and adapts it to the substrate's ``DeviceDriver``
protocol.

Capability surface:
- ``media.video_play@1`` — full transport + audio/subtitle stream control
  (subtitle/audio track switching is no-op at DLNA layer; provider has
   to bake the right track into the URL when needed)
- ``media.audio_play@1`` — same wire path, audio-only items
- ``display.image_show@1`` — DLNA photo class via SetAVTransportURI

Discovery mode is SSDP (M-SEARCH multicast). Manual add probes the user-
supplied host on common UPnP description URL paths and parses the result.
DLNA requires no pairing and exposes no useful push channel, so
``subscribe()`` returns an empty stream (snapshots polled instead).
"""

from __future__ import annotations

import asyncio
import html
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from augmentum.devices.device import Device, DiscoveredDevice
from augmentum.devices.invocation import (
    Event,
    InvocationContext,
    InvocationResult,
    PairResult,
)
from augmentum.media.receivers.dlna import (
    DlnaReceiver,
    _fetch_description,
    discover_dlna_receivers,
    send_dlna_general_command,
    send_dlna_playstate_command,
    snapshot_dlna_receiver,
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


# UPnP description URL paths to try once a port answers. Ordered by
# real-world frequency: 90%+ of renderers serve the description at one
# of the first two. We only fall through the rest when the earlier
# ones come back as anything other than valid UPnP XML.
_PROBE_PATHS: tuple[str, ...] = (
    "/description.xml",
    "/rootDesc.xml",
    "/RootDevice.xml",
    "/setup.xml",
    "/upnp/desc/aios_device/aios_device.xml",  # Sonos
    "/dial/dd.xml",                             # Roku
)


# UPnP-default ports ordered by real-world frequency. We TCP-knock each
# port quickly before issuing any HTTP fetch — most LAN IPs don't have
# anything listening on UPnP ports, so the cheap connect probe lets us
# skip them in a few hundred ms total instead of waiting on HTTP timeouts.
_KNOCK_PORTS: tuple[int, ...] = (
    49152,  # generic UPnP default (most consumer TVs)
    8200,   # MiniDLNA / ReadyDLNA / many NAS units
    1400,   # Sonos
    8060,   # Roku ECP
    8080,   # generic
)


_TCP_KNOCK_TIMEOUT_S: float = 0.3   # cheap-fast LAN probe
_DESC_FETCH_TIMEOUT_S: float = 1.5  # post-knock HTTP, plenty for LAN


def _content_type_default(args: dict[str, Any], capability: str) -> str:
    declared = str(args.get("content_type") or "").strip()
    if declared:
        return declared
    if capability == "display.image_show@1":
        return "image/jpeg"
    if capability == "media.audio_play@1":
        return "audio/mpeg"
    return "video/mp4"


def _upnp_class_for_capability(capability: str) -> str:
    if capability == "display.image_show@1":
        return "object.item.imageItem"
    if capability == "media.audio_play@1":
        return "object.item.audioItem.musicTrack"
    return "object.item.videoItem"


def _build_didl(
    content_url: str,
    content_type: str,
    title: str,
    upnp_class: str,
    *,
    artist: str = "",
    album: str = "",
    creator: str = "",
    album_art_url: str = "",
) -> str:
    """DIDL-Lite metadata for SetAVTransportURI.

    Most TVs honor `dc:title`, `upnp:albumArtURI`, `dc:creator`, and
    `upnp:artist` in their now-playing card. Including them lifts the
    user-facing display from "untitled video" to a recognizable item
    with cover art. Empty fields are omitted entirely so we don't
    waste characters on blanks.
    """
    safe_title = html.escape(title or "Item")
    safe_url = html.escape(content_url or "")
    safe_ct = html.escape(content_type or "application/octet-stream")
    parts: list[str] = [f"<dc:title>{safe_title}</dc:title>"]
    if creator:
        parts.append(f"<dc:creator>{html.escape(creator)}</dc:creator>")
    if artist:
        parts.append(f"<upnp:artist>{html.escape(artist)}</upnp:artist>")
    if album:
        parts.append(f"<upnp:album>{html.escape(album)}</upnp:album>")
    if album_art_url:
        parts.append(
            '<upnp:albumArtURI '
            'dlna:profileID="JPEG_TN" '
            'xmlns:dlna="urn:schemas-dlna-org:metadata-1-0/">'
            f"{html.escape(album_art_url)}</upnp:albumArtURI>"
        )
    parts.append(f"<upnp:class>{html.escape(upnp_class)}</upnp:class>")
    parts.append(f'<res protocolInfo="http-get:*:{safe_ct}:*">{safe_url}</res>')
    return (
        '<DIDL-Lite '
        'xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/">'
        '<item id="0" parentID="-1" restricted="1">'
        + "".join(parts) +
        "</item>"
        "</DIDL-Lite>"
    )


def _device_to_receiver(device: Device) -> DlnaReceiver | None:
    """Reconstruct a DlnaReceiver from a stored Device.

    The discovery path stashes the resolved control URLs in
    ``device.address`` so we don't need to re-fetch the description
    on every invocation. If the address is missing required fields
    (e.g. legacy save), we return None so callers can re-probe.
    """
    address = device.address or {}
    av_transport_url = str(address.get("av_transport_url") or "").strip()
    if not av_transport_url:
        return None
    return DlnaReceiver(
        receiver_id=device.native_id,
        label=device.label,
        location=str(address.get("location") or ""),
        av_transport_url=av_transport_url,
        rendering_control_url=str(address.get("rendering_control_url") or ""),
        manufacturer=str((device.metadata or {}).get("manufacturer") or ""),
        model_name=str((device.metadata or {}).get("model_name") or ""),
        presentation_url=str((device.metadata or {}).get("presentation_url") or ""),
        icon_url=str((device.metadata or {}).get("icon_url") or ""),
        supported_commands=list(address.get("supported_commands") or []),
    )


def _receiver_to_discovered(receiver: DlnaReceiver, *, capabilities: list[str]) -> DiscoveredDevice:
    parts = urlsplit(receiver.location)
    host = (parts.hostname or "").strip()
    port = parts.port
    return DiscoveredDevice(
        driver="dlna",
        native_id=receiver.receiver_id,
        label=receiver.label,
        capabilities=list(capabilities),
        address={
            "host": host,
            "port": port,
            "location": receiver.location,
            "av_transport_url": receiver.av_transport_url,
            "rendering_control_url": receiver.rendering_control_url,
            "supported_commands": list(receiver.supported_commands or []),
        },
        metadata={
            "manufacturer": receiver.manufacturer,
            "model_name": receiver.model_name,
            "presentation_url": receiver.presentation_url,
            "icon_url": receiver.icon_url,
        },
        confidence=1.0,
    )


async def _tcp_alive(host: str, port: int, timeout_s: float) -> bool:
    """Return True if a TCP connect to (host, port) succeeds within timeout.

    Used as a cheap pre-filter before issuing an HTTP description fetch.
    Distinguishes 'nothing listening' (RST / refused) from 'something
    there' (handshake completes) without waiting on the full HTTP
    timeout. Cancellation-safe — the writer is closed in a finally
    block even if asyncio.wait_for raises.
    """
    writer = None
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=max(0.05, float(timeout_s)),
        )
        return True
    except (TimeoutError, OSError, ConnectionRefusedError):
        return False
    except Exception as exc:
        log.debug("tcp_knock_unexpected_error", host=host, port=port, error=str(exc))
        return False
    finally:
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception as exc:
                log.debug("tcp_knock_writer_close_failed", host=host, port=port, error=str(exc))


async def _fetch_description_fast(http_client, location: str) -> DlnaReceiver | None:
    """Same shape as media.receivers.dlna._fetch_description, but with a
    short timeout suitable for LAN probing (vs the 8s default that helper
    uses for known-good hosts)."""
    try:
        resp = await http_client.get(
            location,
            timeout=_DESC_FETCH_TIMEOUT_S,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return None
        # Reuse the shared parser for content-type-loose UPnP XML.
        from urllib.parse import urljoin, urlsplit
        from xml.etree import ElementTree as ET

        from augmentum.media.receivers.dlna import _child_text  # type: ignore

        root = ET.fromstring(resp.text)
        device = root.find(".//{*}device")
        if device is None:
            return None
        device_type = _child_text(device, "deviceType")
        if "MediaRenderer" not in device_type:
            return None
        service_map: dict[str, str] = {}
        for service in device.findall(".//{*}service"):
            stype = _child_text(service, "serviceType")
            ctrl = _child_text(service, "controlURL")
            if stype and ctrl:
                service_map[stype] = urljoin(location, ctrl)
        av = service_map.get("urn:schemas-upnp-org:service:AVTransport:1")
        if not av:
            return None
        rc = service_map.get("urn:schemas-upnp-org:service:RenderingControl:1", "")

        label = (
            _child_text(device, "friendlyName")
            or _child_text(device, "modelName")
            or urlsplit(location).hostname
            or "DLNA Renderer"
        )
        udn = _child_text(device, "UDN") or location
        icon = device.find(".//{*}icon")
        icon_url = urljoin(location, _child_text(icon, "url")) if icon is not None else ""

        supported = ["PlayPause", "Pause", "Unpause", "Stop", "Seek"]
        if rc:
            supported.extend(["VolumeUp", "VolumeDown", "SetVolume", "Mute", "Unmute", "ToggleMute"])

        # Build a deterministic receiver_id matching the shared helper's shape.
        from augmentum.media.receivers.dlna import _receiver_id  # type: ignore
        return DlnaReceiver(
            receiver_id=_receiver_id(udn, location),
            label=label,
            location=location,
            av_transport_url=av,
            rendering_control_url=rc,
            manufacturer=_child_text(device, "manufacturer"),
            model_name=_child_text(device, "modelName"),
            presentation_url=urljoin(location, _child_text(device, "presentationURL")),
            icon_url=icon_url,
            supported_commands=supported,
            extra={"device_type": device_type, "udn": udn},
        )
    except Exception as exc:
        log.debug("dlna_description_fetch_failed_fast", location=location, error=str(exc))
        return None


def _capabilities_for_receiver(receiver: DlnaReceiver) -> list[str]:
    """Most DLNA renderers handle video, audio, and image. We declare the
    full surface; consumers can capability-gate at the call site if a
    specific renderer rejects an item type. Drivers that need to deny
    a capability per-renderer should override here using model_name
    fingerprints (left as a future refinement)."""
    return list(_SUPPORTED_CAPABILITIES)


class DlnaDriver:
    """DeviceDriver implementation for DLNA / UPnP MediaRenderers."""

    id: str = "dlna"
    label: str = "DLNA / UPnP"
    description: str = (
        "Smart TVs, AV receivers, and audio systems exposing UPnP AVTransport."
    )
    capabilities: tuple[str, ...] = _SUPPORTED_CAPABILITIES
    discovery_modes: tuple[str, ...] = ("ssdp",)
    requires_pairing: bool = False
    supports_passive_discovery: bool = False

    def __init__(self, *, http_client: httpx.AsyncClient | None = None) -> None:
        self._http: httpx.AsyncClient | None = http_client

    async def start(self, ctx: DriverContext) -> None:
        if self._http is None:
            self._http = ctx.http_client

    async def stop(self) -> None:
        # No long-lived listeners to tear down.
        return None

    # ---- discovery -----------------------------------------------------------

    async def discover(
        self,
        *,
        timeout_s: float = 3.0,
        user_id: str = "",  # noqa: ARG002 — DLNA discovery is LAN-global, not per-user
    ) -> list[DiscoveredDevice]:
        if self._http is None:
            return []
        try:
            receivers = await discover_dlna_receivers(self._http, timeout_s=timeout_s)
        except Exception as exc:
            log.debug("dlna_discover_failed", error=str(exc))
            return []
        return [
            _receiver_to_discovered(r, capabilities=_capabilities_for_receiver(r))
            for r in receivers
        ]

    async def probe(
        self,
        *,
        host: str,
        port: int | None = None,
        hint: dict[str, Any] | None = None,
    ) -> DiscoveredDevice | None:
        """Two-stage probe: cheap TCP knock to find live ports, then HTTP
        description fetch on the alive ones only.

        For a /24 subnet sweep this is the difference between completing
        in seconds (TCP fast-fails on every empty IP) and timing out
        without finding anything (the HTTP-everywhere approach hangs on
        the slow timeout for every dead host).
        """
        if self._http is None or not host:
            return None
        hint = dict(hint or {})

        # User-provided full description URL takes precedence — skip
        # the knock + path probing entirely.
        explicit = str(hint.get("location_url") or "").strip()
        if explicit:
            try:
                receiver = await _fetch_description(self._http, explicit)
            except Exception as exc:
                log.debug("dlna_probe_explicit_failed", url=explicit, error=str(exc))
                return None
            if receiver is not None:
                return _receiver_to_discovered(
                    receiver, capabilities=_capabilities_for_receiver(receiver),
                )
            return None

        ports_to_try: tuple[int, ...] = (port,) if port else _KNOCK_PORTS

        for p in ports_to_try:
            # Stage 1 — TCP knock. Most IPs on a LAN have nothing on
            # UPnP ports; the connect call refuses or times out fast,
            # and we move on without burning an HTTP timeout.
            if not await _tcp_alive(host, p, _TCP_KNOCK_TIMEOUT_S):
                continue

            # Stage 2 — port answered something. Walk the description
            # paths until one yields a valid MediaRenderer doc.
            for path in _PROBE_PATHS:
                location = f"http://{host}:{p}{path}"
                try:
                    receiver = await _fetch_description_fast(self._http, location)
                except Exception as exc:
                    log.debug("dlna_probe_fetch_failed", url=location, error=str(exc))
                    continue
                if receiver is not None:
                    return _receiver_to_discovered(
                        receiver, capabilities=_capabilities_for_receiver(receiver),
                    )
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
        receiver = _device_to_receiver(device)
        if receiver is None:
            return InvocationResult.failure(
                "device address missing av_transport_url; re-probe required",
                code="device_address_invalid",
            )

        if action in ("play", "show", "load_url"):
            return await self._do_load(receiver, capability, args)

        if action in ("pause", "resume", "stop", "seek"):
            return await self._do_playstate(receiver, action, args)

        if action == "set_volume":
            level = args.get("level")
            try:
                ok = await send_dlna_general_command(
                    self._http,
                    receiver,
                    command="SetVolume",
                    arguments={"Volume": int(level) if level is not None else 0},
                )
            except Exception as exc:
                return InvocationResult.failure(str(exc), code="driver_error")
            return InvocationResult.success() if ok else InvocationResult.failure(
                "set_volume_failed", code="driver_error",
            )

        if action == "set_mute":
            muted = bool(args.get("muted"))
            try:
                ok = await send_dlna_general_command(
                    self._http,
                    receiver,
                    command="Mute" if muted else "Unmute",
                )
            except Exception as exc:
                return InvocationResult.failure(str(exc), code="driver_error")
            return InvocationResult.success() if ok else InvocationResult.failure(
                "set_mute_failed", code="driver_error",
            )

        # Subtitle/audio track switching at DLNA layer is provider-side;
        # if the caller wants a different audio/subtitle stream, the
        # provider must rebuild the URL. We accept the call so the LLM
        # tool surface stays uniform but signal it as a no-op.
        if action in ("set_audio_track", "set_subtitle_track"):
            return InvocationResult.success(
                extra={"note": "track_switching_requires_provider_url_rebuild"},
            )

        if action == "clear":
            try:
                ok = await send_dlna_playstate_command(
                    self._http, receiver, command="Stop",
                )
            except Exception as exc:
                return InvocationResult.failure(str(exc), code="driver_error")
            return InvocationResult.success() if ok else InvocationResult.failure(
                "clear_failed", code="driver_error",
            )

        return InvocationResult.failure(
            f"action {action} not supported on dlna driver",
            code="unsupported_action",
        )

    async def _do_load(
        self,
        receiver: DlnaReceiver,
        capability: str,
        args: dict[str, Any],
    ) -> InvocationResult:
        content_url = str(args.get("content_url") or args.get("url") or args.get("image_url") or "").strip()
        if not content_url:
            return InvocationResult.failure(
                "content_url is required",
                code="missing_arg",
            )
        title = str(args.get("title") or "Item")
        content_type = _content_type_default(args, capability)
        upnp_class = _upnp_class_for_capability(capability)
        start_time_s = float(args.get("start_time_s") or 0.0)

        # Pull rich metadata from args so the TV's now-playing card has
        # cover art + author/artist + album when the upstream provider
        # supplied them (Audiobookshelf has chapters and authors;
        # Emby/Jellyfin have backdrop URLs; LibriVox has reader names).
        # Empty fields fall through to terse DIDL.
        ok = await self._launch_with_class(
            receiver,
            content_url=content_url,
            content_type=content_type,
            title=title,
            upnp_class=upnp_class,
            artist=str(args.get("artist") or args.get("author") or ""),
            album=str(args.get("album") or ""),
            creator=str(args.get("creator") or args.get("author") or ""),
            album_art_url=str(args.get("poster_url") or args.get("cover_url") or ""),
            start_time_s=start_time_s,
        )
        if not ok:
            return InvocationResult.failure("launch_failed", code="driver_error")

        # Best-effort snapshot for state
        snap: dict[str, Any] = {}
        try:
            snap = await snapshot_dlna_receiver(self._http, receiver)
        except Exception as exc:
            log.debug("dlna_snapshot_failed", receiver=getattr(receiver, "udn", ""), error=str(exc))
        return InvocationResult.success(state=snap or {"title": title})

    async def _launch_with_class(
        self,
        receiver: DlnaReceiver,
        *,
        content_url: str,
        content_type: str,
        title: str,
        upnp_class: str,
        artist: str = "",
        album: str = "",
        creator: str = "",
        album_art_url: str = "",
        start_time_s: float = 0.0,
    ) -> bool:
        """SetAVTransportURI + Play with rich DIDL metadata."""
        from augmentum.media.receivers.dlna import (
            _seconds_to_time,
        )
        from augmentum.media.receivers.dlna import (
            _soap_call as soap_call,
        )

        metadata = _build_didl(
            content_url, content_type, title, upnp_class,
            artist=artist, album=album, creator=creator,
            album_art_url=album_art_url,
        )
        ok = await soap_call(
            self._http,
            control_url=receiver.av_transport_url,
            service_type=receiver.transport_service_type,
            action="SetAVTransportURI",
            arguments={
                "InstanceID": 0,
                "CurrentURI": content_url,
                "CurrentURIMetaData": metadata,
            },
        )
        if ok is None:
            return False
        play = await soap_call(
            self._http,
            control_url=receiver.av_transport_url,
            service_type=receiver.transport_service_type,
            action="Play",
            arguments={"InstanceID": 0, "Speed": 1},
        )
        if play is None:
            return False
        # Resume seek — most TVs accept a Seek immediately after Play
        # if a non-trivial start position was requested. Wait briefly
        # so the TV's transport state has time to settle.
        if start_time_s and start_time_s > 1.0:
            import asyncio
            await asyncio.sleep(0.25)
            await soap_call(
                self._http,
                control_url=receiver.av_transport_url,
                service_type=receiver.transport_service_type,
                action="Seek",
                arguments={
                    "InstanceID": 0,
                    "Unit": "REL_TIME",
                    "Target": _seconds_to_time(start_time_s),
                },
            )
        return True

    async def _do_playstate(
        self,
        receiver: DlnaReceiver,
        action: str,
        args: dict[str, Any],
    ) -> InvocationResult:
        # Map substrate-level action names to DLNA command vocab.
        wire_command = {
            "pause": "Pause",
            "resume": "Unpause",
            "stop": "Stop",
            "seek": "Seek",
        }.get(action)
        if wire_command is None:
            return InvocationResult.failure(
                f"unknown playstate action {action}",
                code="unsupported_action",
            )
        seek_position_s = None
        if action == "seek":
            try:
                seek_position_s = float(args.get("position_s") or 0.0)
            except (TypeError, ValueError):
                return InvocationResult.failure(
                    "position_s must be a number",
                    code="missing_arg",
                )
        try:
            ok = await send_dlna_playstate_command(
                self._http,
                receiver,
                command=wire_command,
                seek_position_s=seek_position_s,
            )
        except Exception as exc:
            return InvocationResult.failure(str(exc), code="driver_error")
        if not ok:
            return InvocationResult.failure(f"{action}_failed", code="driver_error")
        return InvocationResult.success()

    # ---- snapshot / subscribe ------------------------------------------------

    async def snapshot(
        self,
        device: Device,
        capability: str,
        ctx: InvocationContext,
    ) -> dict[str, Any] | None:
        if self._http is None:
            return None
        receiver = _device_to_receiver(device)
        if receiver is None:
            return None
        try:
            snap = await snapshot_dlna_receiver(self._http, receiver)
        except Exception as exc:
            log.debug("dlna_snapshot_failed", id=device.id, error=str(exc))
            return None
        return snap

    async def subscribe(
        self,
        device: Device,
        capability: str,
        ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        # DLNA has UPnP eventing (GENA) but it's flaky and most TVs don't
        # honor it. Polling via /api/devices/{id}/{capability} is the
        # supported path. This subscribe returns an empty stream.
        async def _empty() -> AsyncIterator[Event]:
            if False:
                yield  # noqa: B901 — required to make this a generator
        return _empty()

    # ---- pairing -------------------------------------------------------------

    async def pair_start(
        self,
        device: Device,
        ctx: InvocationContext,
    ) -> PairResult:
        return PairResult(
            state="active",
            requires_user_action=False,
            message="DLNA does not require pairing",
        )

    async def pair_complete(
        self,
        device: Device,
        code: str,
        ctx: InvocationContext,
    ) -> PairResult:
        return PairResult(
            state="active",
            requires_user_action=False,
            message="DLNA does not require pairing",
        )
