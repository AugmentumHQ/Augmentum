"""Wake-on-LAN magic packet sender for trusted receivers.

A magic packet is 102 bytes: ``0xFF * 6`` followed by the target MAC
address repeated 16 times. It's sent as a UDP broadcast on port 9
(historical "discard" port — the actual port doesn't matter, the NIC
firmware pattern-matches on the payload regardless).

Two Docker-network realities shape the design:

  - The container runs on the bridge network, so ``255.255.255.255``
    limited broadcasts get NAT-translated away and never escape to the
    LAN. Directed broadcasts (e.g. ``192.168.1.255``) survive the
    MASQUERADE rule and reach the actual subnet.
  - The container has no direct view of the LAN topology, so the
    broadcast address has to come from somewhere — either an explicit
    per-receiver override, or auto-derived from the last LAN IP we saw
    the receiver self-report.

Auto-derive policy: take the receiver's ``last_local_ip`` and replace
the host octet with ``255``. That's a textbook /24 directed broadcast,
correct for the overwhelmingly common home-network case. Power users
on /16 or odd subnets supply the override.

The helper is intentionally synchronous + fire-and-forget: the WoL
packet is one UDP send, the call site doesn't wait for an ack (there
isn't one), and any send failure is logged + swallowed rather than
raising. Wake is a hint, not a contract.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass

from augmentum.cast.trusted_receivers import (
    TrustedReceiver, normalise_ipv4, normalise_mac,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


WOL_PORT = 9  # Historical discard port; choice is convention-only.


@dataclass(slots=True)
class WakeResult:
    """Outcome of a wake attempt. Routes use this to shape the API
    response — successes report what was sent, failures the reason."""

    ok: bool
    mac: str = ""
    broadcast: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, str | bool]:
        out: dict[str, str | bool] = {"ok": self.ok}
        if self.mac:
            out["mac"] = self.mac
        if self.broadcast:
            out["broadcast"] = self.broadcast
        if self.reason:
            out["reason"] = self.reason
        return out


def derive_broadcast(last_local_ip: str) -> str:
    """Take ``192.168.1.42`` → ``192.168.1.255``.

    Returns '' when the input isn't a parseable IPv4 — callers fall
    back to asking the user to set the override explicitly.
    """
    ip = normalise_ipv4(last_local_ip)
    if not ip:
        return ""
    parts = ip.split(".")
    parts[-1] = "255"
    return ".".join(parts)


def build_magic_packet(mac: str) -> bytes:
    """Construct the 102-byte WoL payload for ``mac``.

    Raises ValueError on bad MAC (canonicalisation failed). Callers
    that catch this should treat it as a config error, not a send error.
    """
    canonical = normalise_mac(mac)
    if not canonical:
        raise ValueError(f"invalid MAC: {mac!r}")
    mac_bytes = bytes.fromhex(canonical.replace(":", ""))
    return b"\xff" * 6 + mac_bytes * 16


def send_magic_packet(mac: str, broadcast: str, *, port: int = WOL_PORT) -> WakeResult:
    """Send one magic packet and return the structured outcome.

    Sends three back-to-back to absorb the typical ~5–10% UDP drop
    rate on consumer Wi-Fi without making the user click again. NICs
    pattern-match on payload so duplicates are harmless.
    """
    canonical_mac = normalise_mac(mac)
    if not canonical_mac:
        return WakeResult(ok=False, mac=mac, reason="invalid mac")
    canonical_bcast = normalise_ipv4(broadcast)
    if not canonical_bcast:
        return WakeResult(
            ok=False, mac=canonical_mac, broadcast=broadcast,
            reason="invalid broadcast",
        )

    try:
        packet = build_magic_packet(canonical_mac)
    except ValueError as exc:
        return WakeResult(ok=False, mac=mac, reason=str(exc))

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(2.0)
            for _ in range(3):
                sock.sendto(packet, (canonical_bcast, port))
    except OSError as exc:
        log.warning(
            "wol_send_failed",
            mac=canonical_mac, broadcast=canonical_bcast,
            error=str(exc),
        )
        return WakeResult(
            ok=False, mac=canonical_mac, broadcast=canonical_bcast,
            reason=str(exc),
        )

    log.info(
        "wol_sent",
        mac=canonical_mac, broadcast=canonical_bcast,
    )
    return WakeResult(ok=True, mac=canonical_mac, broadcast=canonical_bcast)


def wake_receiver(receiver: TrustedReceiver) -> WakeResult:
    """Convenience: pick the right broadcast for the receiver and fire.

    Priority order for the broadcast address:
      1. ``wol_broadcast_override`` (user-set, per-receiver)
      2. derived /24 broadcast from ``last_local_ip``

    Returns a WakeResult; routes serialise it to JSON. The receiver
    record is the authority — if it lacks a MAC the caller should
    surface "set the MAC first" in the UI rather than calling here.
    """
    if not receiver.mac_address:
        return WakeResult(ok=False, reason="no mac on receiver")

    broadcast = receiver.wol_broadcast_override or derive_broadcast(
        receiver.last_local_ip,
    )
    if not broadcast:
        return WakeResult(
            ok=False, mac=receiver.mac_address,
            reason="no broadcast available (set override or wait for receiver to report IP once)",
        )

    return send_magic_packet(receiver.mac_address, broadcast)
