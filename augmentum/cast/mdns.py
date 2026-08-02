"""mDNS service advertisement so LAN clients (the Android TV receiver,
future native apps) can find this Augmentum node without a subnet sweep.

We advertise on ``_http._tcp.`` with a service name that includes
``augmentum`` so the receiver's name filter matches. The TXT record
carries the running APK ``versionCode`` so a smarter client could
short-circuit the version round-trip; today the receiver only uses the
SRV record to resolve a host:port.

Lifecycle is owned by ``augmentum/proxy/server.py``'s lifespan — we
expose ``start_mdns`` / ``stop_mdns`` and stash the registration on
``app.state`` so shutdown can deregister cleanly.

Zeroconf is a transitive dep via pychromecast; we import lazily so a
build without it (unlikely but possible in stripped environments)
degrades to "feature off" instead of import error at startup.
"""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI

log = get_logger(__name__)

# Caddy HTTPS port — receiver scripts and the Android TV client both
# probe this. Keep in sync with the client (Discovery.kt PORT).
_PORT = 6443
_SERVICE_TYPE = "_http._tcp.local."


def _primary_ipv4() -> str | None:
    """Best-effort local IPv4 the LAN can route to.

    The classic UDP-to-public-ip trick: open a UDP socket against a
    public address (no packet is sent) and read the kernel's chosen
    source IP. Handles multi-NIC hosts better than ``gethostbyname``.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("1.1.1.1", 1))
            return s.getsockname()[0]
    except OSError:
        return None


async def start_mdns(app: FastAPI) -> None:
    """Register the Augmentum service on mDNS. Safe no-op on failure."""
    if getattr(app.state, "mdns_registration", None) is not None:
        return  # already started

    try:
        from zeroconf import IPVersion, ServiceInfo
        from zeroconf.asyncio import AsyncZeroconf
    except ImportError:
        log.info("mdns_disabled_no_zeroconf")
        return

    ip = _primary_ipv4()
    if not ip:
        log.info("mdns_disabled_no_local_ip")
        return

    hostname = socket.gethostname().split(".")[0] or "augmentum"
    instance_name = f"augmentum-{hostname}.{_SERVICE_TYPE}"

    props: dict[bytes, bytes] = {
        b"path": b"/api/auth/status",
        b"scheme": b"https",
    }

    try:
        info = ServiceInfo(
            type_=_SERVICE_TYPE,
            name=instance_name,
            addresses=[socket.inet_aton(ip)],
            port=_PORT,
            properties=props,
            server=f"{hostname}.local.",
        )
        zc = AsyncZeroconf(ip_version=IPVersion.V4Only)
        await zc.async_register_service(info)
    except Exception as exc:
        log.warning("mdns_register_failed", error=str(exc))
        return

    app.state.mdns_registration = (zc, info)
    log.info("mdns_registered", name=instance_name, ip=ip, port=_PORT)


async def stop_mdns(app: FastAPI) -> None:
    """Deregister and close the zeroconf listener."""
    reg: Any = getattr(app.state, "mdns_registration", None)
    if reg is None:
        return
    zc, info = reg
    app.state.mdns_registration = None
    try:
        await zc.async_unregister_service(info)
    except Exception as exc:
        log.debug("mdns_unregister_failed", error=str(exc))
    try:
        await zc.async_close()
    except Exception as exc:
        log.debug("mdns_close_failed", error=str(exc))
