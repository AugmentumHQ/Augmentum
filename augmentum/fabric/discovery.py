"""LAN discovery for fabric peers.

Operator-triggered subnet sweep. Walks the user's LAN looking for
hosts that answer ``GET /api/fabric/hello`` with an augmentum-shaped
identity envelope. The discovery surface is non-authoritative -- it
produces *candidates*; the operator still confirms each candidate's
fingerprint and runs the existing pair flow (see
``pair_client.initiate_pair_with_remote``).

Why subnet sweep, not mDNS:

Docker Desktop hides the container in a VM whose network namespace
can't reach LAN multicast (SSDP/mDNS) -- the same constraint already
documented in ``devices/discovery/subnet_sweep.py``. TCP unicast to
LAN IPs *does* cross the Docker NAT cleanly, so we use unicast probes
against augmentum's two known ports (6443 HTTPS via Caddy, 6100 HTTP
direct) and skip multicast entirely. The sweep finishes in seconds
even on a /24 because we only probe two ports per host.

Threat model:

  - Hostile LAN host could answer with a falsified fingerprint. We
    don't care -- the fingerprint is read out-of-band before pairing
    and re-verified inside the pair handshake against the actual
    pubkey on the wire.
  - We refuse to scan non-RFC1918 ranges (catches operator typos
    that would otherwise turn augmentum into a public-internet port
    scanner) and cap subnet width at /22 (1022 hosts; anything wider
    is almost certainly mis-entered).
  - Loopback subnets are also refused -- probing localhost against
    augmentum's own port would just discover this node.

Usage::

    from augmentum.fabric.discovery import discover_fabric_peers
    result = await discover_fabric_peers(
        subnet="192.168.1.0/24",
        own_fingerprint=identity.fingerprint,
        known_node_ids={p.node_id for p in paired_peers},
    )
    for peer in result.peers:
        # peer.url + peer.fingerprint can be fed straight to the
        # existing pair flow once the operator confirms.
        ...
"""

from __future__ import annotations

import asyncio
import ipaddress
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

import httpx

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Augmentum's two known ports. Caddy's HTTPS proxy is on 6443 (when the
# https profile is active); the bare HTTP server is on 6100. See
# ``compose.yaml`` for the canonical declaration. We try HTTPS first
# because operators running fabric with TLS are the better-configured
# case and we want their nodes found preferentially.
PORTS_IN_PRIORITY: tuple[tuple[int, str], ...] = (
    (6443, "https"),
    (6100, "http"),
)

# Per-host probe timeout. Short -- a peer either answers quickly on
# its own LAN or it's not there. Anything longer just stretches sweep
# duration for closed-port hosts.
_PROBE_TIMEOUT_S = 1.5

# Bound on concurrent probes in flight. Mirrors subnet_sweep.py's
# default; tuned so that a /24 (~500 probes) finishes in roughly 5s
# without flooding the LAN switch.
_DEFAULT_CONCURRENCY = 60

# Subnet candidates tried when the caller can't supply a hint -- the
# three most common consumer-router defaults plus 10.0.0/24. Same
# list ``devices/discovery/subnet_sweep.py`` uses.
DEFAULT_FALLBACK_SUBNETS: tuple[str, ...] = (
    "192.168.0.0/24",
    "192.168.1.0/24",
    "10.0.0.0/24",
)


@dataclass(frozen=True)
class DiscoveredPeer:
    """One LAN host that answered /api/fabric/hello correctly.

    The fields exactly mirror what ``initiate_pair_with_remote``
    needs, so the UI can populate the pair form directly without
    a translation layer.
    """

    url: str          # e.g. "https://192.168.1.42:6443"
    addr: str         # e.g. "192.168.1.42:6443"
    node_id: str
    fingerprint: str
    public_key: str   # base64 ed25519
    hostname: str     # advertised, informational only
    version: str      # advertised augmentum version
    icon: str
    scheme: str       # "http" | "https"
    host: str         # bare IP
    port: int


@dataclass
class DiscoveryResult:
    """Outcome of one sweep."""

    peers: list[DiscoveredPeer] = field(default_factory=list)
    # Peers that responded but matched our own fingerprint (loopback
    # via a non-localhost IP, or the operator pointed the sweep at
    # itself). Surfaced separately so the UI can show "you found
    # yourself" rather than letting the operator try to pair with self.
    self_seen: list[DiscoveredPeer] = field(default_factory=list)
    # Peers that responded but are already paired -- node_id present
    # in the caller-supplied ``known_node_ids`` set. Lets the UI
    # render them as "already paired" rather than offering pair again.
    already_paired: list[DiscoveredPeer] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    hosts_probed: int = 0
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "peers": [p.__dict__ for p in self.peers],
            "self_seen": [p.__dict__ for p in self.self_seen],
            "already_paired": [p.__dict__ for p in self.already_paired],
            "errors": dict(self.errors),
            "hosts_probed": int(self.hosts_probed),
            "duration_s": float(self.duration_s),
        }


def derive_subnet_from_host(host_value: str) -> str | None:
    """Convert ``host:port`` (e.g. from an HTTP ``Host`` header) into a /24.

    When the operator opens the Fabric tab from ``https://192.168.1.10:6443``,
    their browser hits us with ``Host: 192.168.1.10:6443`` — that IS the
    LAN address they care about. Auto-deriving ``192.168.1.0/24`` from it
    saves the operator from having to figure out their own subnet first,
    which is the most common reason scans return zero peers (the hardcoded
    fallback list happens to not contain their actual LAN).

    Returns ``None`` for non-IP hosts (``localhost``, FQDNs), non-RFC1918
    addresses (refuse to derive a public-internet sweep target), and
    loopback/link-local. Caller falls back to :data:`DEFAULT_FALLBACK_SUBNETS`
    in those cases.
    """
    if not host_value:
        return None
    host = host_value.split(":", 1)[0].strip()
    if not host:
        return None
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    if not isinstance(ip, ipaddress.IPv4Address):
        return None
    if not ip.is_private or ip.is_loopback or ip.is_link_local:
        return None
    octets = str(ip).split(".")
    octets[3] = "0"
    return ".".join(octets) + "/24"


def enumerate_hosts(subnet: str) -> list[str]:
    """Return every host IP in the subnet (skips network + broadcast).

    Refuses to enumerate non-RFC1918 + loopback ranges -- even an
    authenticated operator shouldn't be able to weaponise augmentum's
    egress as a port scanner against the public internet. Caps at /22
    so a mistyped subnet doesn't lock up the sweep for minutes.
    """
    try:
        net = ipaddress.IPv4Network(subnet, strict=False)
    except (ValueError, ipaddress.AddressValueError):
        return []
    if net.num_addresses > 1024:
        log.warning("fabric_sweep_too_wide", subnet=subnet, hosts=net.num_addresses)
        return []
    if not (net.is_private and not net.is_loopback):
        log.warning("fabric_sweep_non_private", subnet=subnet)
        return []
    return [str(ip) for ip in net.hosts()]


def _parse_hello_response(
    body: dict,
    *,
    scheme: str,
    host: str,
    port: int,
) -> DiscoveredPeer | None:
    """Validate a /api/fabric/hello body and lift to DiscoveredPeer.

    Returns None when the shape is wrong -- treat any non-conforming
    responder as "not augmentum" rather than surfacing a malformed
    peer to the operator. A hostile responder that returns valid
    augmentum shape but a forged fingerprint still has to pass
    pair-time verification, so the only damage they can do here is
    appear in the candidate list.
    """
    if not isinstance(body, dict):
        return None
    if body.get("service") != "augmentum-fabric":
        return None
    node_id = str(body.get("node_id") or "")
    fingerprint = str(body.get("fingerprint") or "")
    public_key = str(body.get("public_key") or "")
    if not node_id or not fingerprint or not public_key:
        return None
    if not fingerprint.startswith("SHA256:"):
        return None
    return DiscoveredPeer(
        url=f"{scheme}://{host}:{port}",
        addr=f"{host}:{port}",
        node_id=node_id,
        fingerprint=fingerprint,
        public_key=public_key,
        hostname=str(body.get("hostname") or ""),
        version=str(body.get("version") or ""),
        icon=str(body.get("icon") or ""),
        scheme=scheme,
        host=host,
        port=port,
    )


async def _probe_one_host(
    client: httpx.AsyncClient,
    host: str,
) -> DiscoveredPeer | None:
    """Probe both known ports on one host. First success wins.

    Tries HTTPS:6443 first (operator-run-fabric-with-TLS is the
    better-configured case), falls back to HTTP:6100. Per-port
    timeouts are bounded so a closed port doesn't burn the budget for
    the host we actually want to find.
    """
    for port, scheme in PORTS_IN_PRIORITY:
        url = f"{scheme}://{host}:{port}/api/fabric/hello"
        try:
            resp = await client.get(url, timeout=_PROBE_TIMEOUT_S)
        except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
            continue
        except Exception as exc:
            log.debug("fabric_sweep_probe_unexpected", host=host, port=port, error=str(exc))
            continue
        if resp.status_code != 200:
            # 503 = fabric disabled on that node; that's a "real
            # augmentum, just opted out" signal but not a peer
            # candidate. Either way we don't materialise a peer
            # entry.
            continue
        try:
            body = resp.json()
        except ValueError as exc:
            log.debug("fabric_hello_parse_failed", host=host, port=port, error=str(exc))
            continue
        peer = _parse_hello_response(body, scheme=scheme, host=host, port=port)
        if peer is not None:
            return peer
    return None


async def sweep_subnet(
    *,
    subnet: str,
    own_fingerprint: str = "",
    known_node_ids: set[str] | None = None,
    timeout_s: float = 12.0,
    concurrency: int = _DEFAULT_CONCURRENCY,
) -> DiscoveryResult:
    """Probe every host in ``subnet`` for the fabric /hello endpoint.

    Returns a :class:`DiscoveryResult` partitioned into ``peers``
    (truly new candidates), ``self_seen`` (we found ourselves -- the
    sweep hit our own LAN IP), and ``already_paired`` (responder's
    node_id is in ``known_node_ids``). ``errors`` keys ``_subnet``
    or ``_timeout`` describe scope failures, not per-host failures.

    Concurrency is bounded so we don't drop the LAN switch under a
    /24 sweep. Total wall-time is bounded by ``timeout_s``; any
    probes still in flight when the deadline hits are cancelled.
    """
    hosts = enumerate_hosts(subnet)
    if not hosts:
        return DiscoveryResult(errors={"_subnet": "invalid_or_too_wide"})

    known = set(known_node_ids or ())
    own_fp = (own_fingerprint or "").strip()
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    found: list[DiscoveredPeer] = []
    start = time.monotonic()
    deadline = start + max(2.0, float(timeout_s))
    errors: dict[str, str] = {}

    async with httpx.AsyncClient(verify=False, follow_redirects=False) as client:
        async def _one(host: str) -> None:
            if time.monotonic() >= deadline:
                return
            async with sem:
                if time.monotonic() >= deadline:
                    return
                peer = await _probe_one_host(client, host)
                if peer is not None:
                    found.append(peer)

        tasks = [asyncio.create_task(_one(h)) for h in hosts]
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=max(2.0, float(timeout_s)),
            )
        except TimeoutError:
            for t in tasks:
                if not t.done():
                    t.cancel()
            errors["_timeout"] = f"{timeout_s}s"

    # Partition results AFTER all probes complete -- never compare
    # fingerprints in the hot path of the probe itself, so callers
    # can re-classify by changing the sets without re-sweeping.
    result = DiscoveryResult(
        hosts_probed=len(hosts),
        duration_s=time.monotonic() - start,
        errors=errors,
    )
    for peer in found:
        if own_fp and peer.fingerprint == own_fp:
            result.self_seen.append(peer)
        elif peer.node_id in known:
            result.already_paired.append(peer)
        else:
            result.peers.append(peer)
    return result


async def sweep_many_subnets(
    *,
    subnets: Iterable[str],
    own_fingerprint: str = "",
    known_node_ids: set[str] | None = None,
    timeout_s_per_subnet: float = 6.0,
) -> DiscoveryResult:
    """Run ``sweep_subnet`` across several CIDRs sequentially.

    Used when the caller has no subnet hint and we fall back to the
    common consumer-router defaults. Results across subnets are
    merged; the first responder for each unique node_id wins (a peer
    bridged across two subnets would otherwise appear twice). Stops
    early once we've found ~5 candidates -- the operator can re-run
    with an explicit subnet if they need more.
    """
    merged = DiscoveryResult()
    seen_node_ids: set[str] = set()
    start = time.monotonic()

    for subnet in subnets:
        partial = await sweep_subnet(
            subnet=subnet,
            own_fingerprint=own_fingerprint,
            known_node_ids=known_node_ids,
            timeout_s=timeout_s_per_subnet,
        )
        merged.hosts_probed += partial.hosts_probed
        for src, bucket in (
            (partial.peers, merged.peers),
            (partial.self_seen, merged.self_seen),
            (partial.already_paired, merged.already_paired),
        ):
            for peer in src:
                if peer.node_id in seen_node_ids:
                    continue
                seen_node_ids.add(peer.node_id)
                bucket.append(peer)
        for k, v in partial.errors.items():
            merged.errors[f"{subnet}:{k}"] = v
        if len(merged.peers) >= 5:
            break

    merged.duration_s = time.monotonic() - start
    return merged


async def discover_fabric_peers(
    *,
    subnet: str | None = None,
    own_fingerprint: str = "",
    known_node_ids: set[str] | None = None,
    timeout_s: float = 12.0,
) -> DiscoveryResult:
    """Top-level entry point. Sweep one subnet or fall back to defaults.

    Callers that know the LAN's CIDR (the operator typed it in the UI)
    pass ``subnet="192.168.1.0/24"``. Callers that don't -- e.g. an
    auto-discover triggered from a localhost session -- pass None and
    we walk the common defaults.
    """
    if subnet:
        return await sweep_subnet(
            subnet=subnet,
            own_fingerprint=own_fingerprint,
            known_node_ids=known_node_ids,
            timeout_s=timeout_s,
        )
    return await sweep_many_subnets(
        subnets=DEFAULT_FALLBACK_SUBNETS,
        own_fingerprint=own_fingerprint,
        known_node_ids=known_node_ids,
        timeout_s_per_subnet=max(2.0, timeout_s / 3.0),
    )
