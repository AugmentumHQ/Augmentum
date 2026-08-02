"""Minimal DLNA/UPnP MediaRenderer support.

This adapter intentionally focuses on the control plane Augmentum needs:

- discover MediaRenderer devices over SSDP
- load a launchable video URI through AVTransport
- poll transport state, position, mute, and volume
- send basic transport/general commands

It avoids external dependencies so the project can grow the receiver layer
without taking on a full UPnP stack yet.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import socket
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlsplit
from xml.etree import ElementTree as ET

import httpx

from augmentum.media.receivers.base import ReceiverLaunchPlan
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_SSDP_ADDR = ("239.255.255.250", 1900)
_MEDIA_RENDERER_ST = "urn:schemas-upnp-org:device:MediaRenderer:1"
_SOAP_NS = "http://schemas.xmlsoap.org/soap/envelope/"


@dataclass(slots=True)
class DlnaReceiver:
    receiver_id: str
    label: str
    location: str
    av_transport_url: str
    rendering_control_url: str = ""
    transport_service_type: str = "urn:schemas-upnp-org:service:AVTransport:1"
    rendering_service_type: str = "urn:schemas-upnp-org:service:RenderingControl:1"
    receiver_profile: str = "dlna_generic_video"
    manufacturer: str = ""
    model_name: str = ""
    presentation_url: str = ""
    icon_url: str = ""
    supported_commands: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "receiver_id": self.receiver_id,
            "transport_kind": "dlna",
            "receiver_profile": self.receiver_profile,
            "label": self.label,
            "manufacturer": self.manufacturer,
            "model_name": self.model_name,
            "icon_url": self.icon_url,
            "presentation_url": self.presentation_url,
            "supported_commands": list(self.supported_commands or []),
            "extra": dict(self.extra or {}),
        }


class _SsdpCollector(asyncio.DatagramProtocol):
    def __init__(self) -> None:
        self.responses: list[dict[str, str]] = []

    def datagram_received(self, data: bytes, addr) -> None:  # noqa: ANN001
        try:
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            return
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines or "200 OK" not in lines[0].upper():
            return
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip().upper()] = value.strip()
        location = headers.get("LOCATION", "")
        if not location:
            return
        self.responses.append({
            "location": location,
            "st": headers.get("ST", ""),
            "usn": headers.get("USN", ""),
            "server": headers.get("SERVER", ""),
        })


def _receiver_id(*parts: str) -> str:
    joined = "|".join(str(part or "").strip() for part in parts if str(part or "").strip())
    digest = hashlib.sha1(joined.encode("utf-8"), usedforsecurity=False).hexdigest()  # noqa: S324
    return f"dlna_{digest[:16]}"


def _soap_envelope(service_type: str, action: str, arguments: dict[str, Any]) -> str:
    inner = "".join(
        f"<{name}>{html.escape(str(value))}</{name}>"
        for name, value in arguments.items()
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<s:Envelope '
        'xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
        's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
        "<s:Body>"
        f'<u:{action} xmlns:u="{service_type}">{inner}</u:{action}>'
        "</s:Body>"
        "</s:Envelope>"
    )


def _didl_metadata(plan: ReceiverLaunchPlan) -> str:
    title = html.escape(plan.title or "Video")
    content_url = html.escape(plan.content_url or "")
    content_type = html.escape(plan.content_type or "video/mp4")
    return (
        '<DIDL-Lite '
        'xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/">'
        '<item id="0" parentID="-1" restricted="1">'
        f"<dc:title>{title}</dc:title>"
        "<upnp:class>object.item.videoItem</upnp:class>"
        f'<res protocolInfo="http-get:*:{content_type}:*">{content_url}</res>'
        "</item>"
        "</DIDL-Lite>"
    )


def _time_to_seconds(raw: str) -> float:
    text = str(raw or "").strip()
    if not text or text in {"NOT_IMPLEMENTED", "00:00:00"}:
        return 0.0
    try:
        hours, minutes, seconds = text.split(":", 2)
        return (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)
    except (TypeError, ValueError):
        return 0.0


def _seconds_to_time(value: float) -> str:
    total = max(0, int(float(value or 0.0)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _child_text(element: ET.Element | None, name: str) -> str:
    if element is None:
        return ""
    child = element.find(f".//{{*}}{name}")
    return str(child.text or "").strip() if child is not None else ""


async def _soap_call(
    http_client: httpx.AsyncClient,
    *,
    control_url: str,
    service_type: str,
    action: str,
    arguments: dict[str, Any],
) -> ET.Element | None:
    body = _soap_envelope(service_type, action, arguments)
    headers = {
        "Content-Type": 'text/xml; charset="utf-8"',
        "SOAPAction": f'"{service_type}#{action}"',
    }
    try:
        resp = await http_client.post(
            control_url,
            headers=headers,
            content=body.encode("utf-8"),
            timeout=10.0,
            follow_redirects=True,
        )
        if resp.status_code >= 400:
            return None
        return ET.fromstring(resp.text)
    except Exception as exc:
        log.debug("dlna_soap_call_failed", action=action, url=control_url, error=str(exc))
        return None


async def _fetch_description(
    http_client: httpx.AsyncClient,
    location: str,
    *,
    usn: str = "",
    server: str = "",
) -> DlnaReceiver | None:
    try:
        resp = await http_client.get(location, timeout=8.0, follow_redirects=True)
        if resp.status_code != 200:
            return None
        root = ET.fromstring(resp.text)
    except Exception as exc:
        log.debug("dlna_description_fetch_failed", location=location, error=str(exc))
        return None

    device = root.find(".//{*}device")
    if device is None:
        return None

    device_type = _child_text(device, "deviceType")
    if "MediaRenderer" not in device_type:
        return None

    service_map: dict[str, tuple[str, str]] = {}
    for service in device.findall(".//{*}service"):
        service_type = _child_text(service, "serviceType")
        control_url = _child_text(service, "controlURL")
        if service_type and control_url:
            service_map[service_type] = (urljoin(location, control_url), service_type)

    av = service_map.get("urn:schemas-upnp-org:service:AVTransport:1")
    if not av:
        return None
    rc = service_map.get("urn:schemas-upnp-org:service:RenderingControl:1")

    label = (
        _child_text(device, "friendlyName")
        or _child_text(device, "modelName")
        or urlsplit(location).hostname
        or "DLNA Renderer"
    )
    udn = _child_text(device, "UDN") or usn or location
    icon_url = ""
    icon = device.find(".//{*}icon")
    if icon is not None:
        icon_url = urljoin(location, _child_text(icon, "url"))

    supported_commands = ["PlayPause", "Pause", "Unpause", "Stop", "Seek"]
    if rc:
        supported_commands.extend([
            "VolumeUp",
            "VolumeDown",
            "SetVolume",
            "Mute",
            "Unmute",
            "ToggleMute",
        ])

    return DlnaReceiver(
        receiver_id=_receiver_id(udn, location),
        label=label,
        location=location,
        av_transport_url=av[0],
        rendering_control_url=rc[0] if rc else "",
        manufacturer=_child_text(device, "manufacturer"),
        model_name=_child_text(device, "modelName"),
        presentation_url=urljoin(location, _child_text(device, "presentationURL")),
        icon_url=icon_url,
        supported_commands=supported_commands,
        extra={
            "device_type": device_type,
            "udn": udn,
            "server": server,
        },
    )


async def discover_dlna_receivers(
    http_client: httpx.AsyncClient,
    *,
    timeout_s: float = 2.5,
    mx: int = 2,
) -> list[DlnaReceiver]:
    loop = asyncio.get_running_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.bind(("", 0))
    transport = None
    protocol = _SsdpCollector()
    try:
        transport, _ = await loop.create_datagram_endpoint(
            lambda: protocol,
            sock=sock,
        )
        query = (
            "M-SEARCH * HTTP/1.1\r\n"
            f"HOST: {_SSDP_ADDR[0]}:{_SSDP_ADDR[1]}\r\n"
            'MAN: "ssdp:discover"\r\n'
            f"MX: {max(1, int(mx))}\r\n"
            f"ST: {_MEDIA_RENDERER_ST}\r\n"
            "\r\n"
        ).encode()
        transport.sendto(query, _SSDP_ADDR)
        await asyncio.sleep(max(0.5, float(timeout_s or 2.5)))
    finally:
        if transport is not None:
            transport.close()
        else:
            sock.close()

    seen: set[str] = set()
    tasks = []
    for response in protocol.responses:
        key = str(response.get("location") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        tasks.append(_fetch_description(
            http_client,
            key,
            usn=str(response.get("usn") or ""),
            server=str(response.get("server") or ""),
        ))
    if not tasks:
        return []
    results = await asyncio.gather(*tasks, return_exceptions=True)
    receivers: list[DlnaReceiver] = []
    for result in results:
        if isinstance(result, DlnaReceiver):
            receivers.append(result)
    receivers.sort(key=lambda receiver: receiver.label.lower())
    return receivers


async def launch_dlna_receiver(
    http_client: httpx.AsyncClient,
    receiver: DlnaReceiver,
    plan: ReceiverLaunchPlan,
) -> bool:
    metadata = _didl_metadata(plan)
    ok = await _soap_call(
        http_client,
        control_url=receiver.av_transport_url,
        service_type=receiver.transport_service_type,
        action="SetAVTransportURI",
        arguments={
            "InstanceID": 0,
            "CurrentURI": plan.content_url,
            "CurrentURIMetaData": metadata,
        },
    )
    if ok is None:
        return False
    play_ok = await _soap_call(
        http_client,
        control_url=receiver.av_transport_url,
        service_type=receiver.transport_service_type,
        action="Play",
        arguments={"InstanceID": 0, "Speed": 1},
    )
    if play_ok is None:
        return False
    if float(plan.start_time_s or 0.0) > 1.0:
        await asyncio.sleep(0.25)
        await _soap_call(
            http_client,
            control_url=receiver.av_transport_url,
            service_type=receiver.transport_service_type,
            action="Seek",
            arguments={
                "InstanceID": 0,
                "Unit": "REL_TIME",
                "Target": _seconds_to_time(plan.start_time_s),
            },
        )
    return True


async def snapshot_dlna_receiver(
    http_client: httpx.AsyncClient,
    receiver: DlnaReceiver,
) -> dict[str, Any]:
    supported = list(receiver.supported_commands or [])
    state = ""
    current_time_s = 0.0
    duration_s = 0.0
    is_muted = False
    volume_level: int | None = None

    transport_info, position_info = await asyncio.gather(
        _soap_call(
            http_client,
            control_url=receiver.av_transport_url,
            service_type=receiver.transport_service_type,
            action="GetTransportInfo",
            arguments={"InstanceID": 0},
        ),
        _soap_call(
            http_client,
            control_url=receiver.av_transport_url,
            service_type=receiver.transport_service_type,
            action="GetPositionInfo",
            arguments={"InstanceID": 0},
        ),
    )
    if transport_info is not None:
        state = _child_text(transport_info, "CurrentTransportState").upper()
    if position_info is not None:
        current_time_s = _time_to_seconds(_child_text(position_info, "RelTime"))
        duration_s = _time_to_seconds(_child_text(position_info, "TrackDuration"))

    if receiver.rendering_control_url:
        mute_resp, volume_resp = await asyncio.gather(
            _soap_call(
                http_client,
                control_url=receiver.rendering_control_url,
                service_type=receiver.rendering_service_type,
                action="GetMute",
                arguments={"InstanceID": 0, "Channel": "Master"},
            ),
            _soap_call(
                http_client,
                control_url=receiver.rendering_control_url,
                service_type=receiver.rendering_service_type,
                action="GetVolume",
                arguments={"InstanceID": 0, "Channel": "Master"},
            ),
        )
        if mute_resp is not None:
            is_muted = _child_text(mute_resp, "CurrentMute") in {"1", "true", "True"}
        if volume_resp is not None:
            try:
                volume_level = int(_child_text(volume_resp, "CurrentVolume"))
            except ValueError:
                volume_level = None

    return {
        "current_time_s": current_time_s,
        "duration_s": duration_s,
        "is_paused": state in {"PAUSED_PLAYBACK", "PAUSED_RECORDING"},
        "is_muted": is_muted,
        "can_seek": True,
        "volume_level": volume_level,
        "supported_commands": supported,
        "receiver_state": state,
        "extra": {
            "location": receiver.location,
            "presentation_url": receiver.presentation_url,
        },
    }


async def send_dlna_playstate_command(
    http_client: httpx.AsyncClient,
    receiver: DlnaReceiver,
    *,
    command: str,
    seek_position_s: float | None = None,
) -> bool:
    command = str(command or "").strip()
    if command == "Seek":
        target = _seconds_to_time(float(seek_position_s or 0.0))
        resp = await _soap_call(
            http_client,
            control_url=receiver.av_transport_url,
            service_type=receiver.transport_service_type,
            action="Seek",
            arguments={"InstanceID": 0, "Unit": "REL_TIME", "Target": target},
        )
        return resp is not None
    if command in {"Pause", "Stop"}:
        resp = await _soap_call(
            http_client,
            control_url=receiver.av_transport_url,
            service_type=receiver.transport_service_type,
            action=command,
            arguments={"InstanceID": 0},
        )
        return resp is not None
    if command in {"Unpause", "PlayPause"}:
        action = "Play"
        resp = await _soap_call(
            http_client,
            control_url=receiver.av_transport_url,
            service_type=receiver.transport_service_type,
            action=action,
            arguments={"InstanceID": 0, "Speed": 1},
        )
        return resp is not None
    return False


async def send_dlna_general_command(
    http_client: httpx.AsyncClient,
    receiver: DlnaReceiver,
    *,
    command: str,
    arguments: dict[str, Any] | None = None,
) -> bool:
    if not receiver.rendering_control_url:
        return False
    command = str(command or "").strip()
    args = arguments or {}

    async def _set_volume(value: int) -> bool:
        resp = await _soap_call(
            http_client,
            control_url=receiver.rendering_control_url,
            service_type=receiver.rendering_service_type,
            action="SetVolume",
            arguments={
                "InstanceID": 0,
                "Channel": "Master",
                "DesiredVolume": max(0, min(100, int(value))),
            },
        )
        return resp is not None

    async def _set_mute(value: bool) -> bool:
        resp = await _soap_call(
            http_client,
            control_url=receiver.rendering_control_url,
            service_type=receiver.rendering_service_type,
            action="SetMute",
            arguments={
                "InstanceID": 0,
                "Channel": "Master",
                "DesiredMute": 1 if value else 0,
            },
        )
        return resp is not None

    if command == "SetVolume":
        try:
            return await _set_volume(int(args.get("Volume", 0)))
        except (TypeError, ValueError):
            return False
    if command in {"VolumeUp", "VolumeDown"}:
        snapshot = await snapshot_dlna_receiver(http_client, receiver)
        current = int(snapshot.get("volume_level") or 0)
        delta = 5 if command == "VolumeUp" else -5
        return await _set_volume(current + delta)
    if command == "Mute":
        return await _set_mute(True)
    if command == "Unmute":
        return await _set_mute(False)
    if command == "ToggleMute":
        snapshot = await snapshot_dlna_receiver(http_client, receiver)
        return await _set_mute(not bool(snapshot.get("is_muted")))
    return False
