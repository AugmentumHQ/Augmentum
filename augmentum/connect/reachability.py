"""Adaptive reachability for invite links — least-exposure-that-reaches.

When a host mints an invite for someone OUTSIDE the box, the link has to reach
them — but we want to expose the host as LITTLE as possible. Because we control
the link AND its lifetime, we can pick, per invite, the least-exposing path that
can actually reach the recipient, and tear any public exposure down the moment
the invite is consumed or expires.

The ladder (increasing exposure / decreasing privacy):

    LAN          — same local network only; no exposure at all
    TAILNET      — private tailnet (Tailscale 100.64/10 or *.ts.net); no public
    TS_FUNNEL    — PUBLIC via Tailscale Funnel (real cert, stable ts.net name)
    CLOUDFLARED  — PUBLIC via a throwaway cloudflared tunnel (most exposed)

At mint time we know (a) what the host can currently do (capabilities) and
(b) how far the link needs to reach (the recipient scope the operator picks).
``plan_reachability`` walks the ladder from least-to-most exposure and returns
the first tier that both reaches the recipient and is available — falling back
DOWN the privacy ladder (i.e. UP the exposure ladder) only as far as needed,
and flagging when it had to expose more than the operator asked for.

The public tiers point at a PATH-SCOPED ingress (``PUBLIC_INVITE_PATHS``): only
the invite door is reachable from the internet, never /login or the API.

This module is the *bones*: the model, capability detection, the planner, and a
pluggable engine interface with the LAN/TAILNET tiers fully wired and the
TS_FUNNEL/CLOUDFLARED engines as availability-gated adapters. The live tunnel
subprocess lifecycle plugs in behind the same interface.

See ``docs/superpowers/specs/2026-06-20-connect-comms-platform-design.md`` (P3).
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from ipaddress import ip_address, ip_network

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Tailscale CGNAT range — the same 100.64.0.0/10 the cast layer keys on.
_TAILSCALE_CGNAT = ip_network("100.64.0.0/10")
_TS_NET_SUFFIX = ".ts.net"

# Sentinel Host header every tunneled request arrives under (cloudflared is
# spawned with --http-host-header=this). The middleware guard treats requests
# on this host as "came through the public invite tunnel" and path-scopes them
# to the invite door. Not a real DNS name — purely an internal marker.
INVITE_TUNNEL_HOST = "augmentum-invite-gate.internal"

# The ONLY paths a public tier exposes. The tunnel points at an ingress that
# forwards these and 404s everything else, so even during the exposure window
# the internet can reach the invite door and nothing else (not /login, not the
# API, not the chat). Kept here so the planner, the ingress, and the middleware
# guard all agree on one list.
PUBLIC_INVITE_PATHS: tuple[str, ...] = (
    "/ui/connect-join/",      # the onboarding page + its static assets
    "/api/auth/invite/",      # public preview + claim (token-as-credential)
    # Guest comms portal — external_guest invites land HERE through the
    # tunnel (class fix: these were missing, so a tunneled portal invite
    # 404'd at its own front door). Same token-as-credential model.
    "/ui/portal/",            # the portal PWA shell + assets
    "/api/portal/register/",  # claim (token in path)
    "/api/portal/gateway",    # public gateway seal-key bundle (signed)
    "/api/portal/env",        # enveloped guest calls (device-sig authed)
    "/api/invite/",           # QR image (token in path)
    "/favicon.ico",
)


def is_invite_public_path(path: str) -> bool:
    """True when ``path`` is part of the public invite surface (ingress allow)."""
    p = (path or "").split("?", 1)[0]
    return any(p == pre.rstrip("/") or p.startswith(pre) for pre in PUBLIC_INVITE_PATHS)


# ── Active-tunnel host registry (read by the middleware guard) ──────────────
#
# A live tunnel manager publishes the set of hosts that requests arrive under
# while a public tunnel is up (the sentinel :data:`INVITE_TUNNEL_HOST`). The
# auth middleware reads it on every request to path-scope tunneled traffic. A
# provider (not a static set) so it always reflects the CURRENT tunnel state.

def _no_active_hosts() -> set:
    return set()


_active_hosts_provider: Callable[[], set] = _no_active_hosts
_active_ip_provider: Callable[[], set] = _no_active_hosts


def set_active_hosts_provider(provider: Callable[[], set]) -> None:
    """Install the callable that reports currently-active tunnel hosts."""
    global _active_hosts_provider
    _active_hosts_provider = provider


def set_active_ip_provider(provider: Callable[[], set]) -> None:
    """Install the callable that reports the active tunnel's IP allowlist.

    The provider returns a set of allowed IPs/CIDRs, or an EMPTY set meaning
    "no IP restriction" (first-contact / token-only mode). When non-empty, only
    those addresses may pass the tunnel guard — so a leaked URL is useless from
    any other IP.
    """
    global _active_ip_provider
    _active_ip_provider = provider


def active_public_hosts() -> set:
    """Hosts that requests arrive under while a public invite tunnel is live."""
    try:
        return _active_hosts_provider() or set()
    except Exception:  # pragma: no cover - a provider fault must not 500 requests
        log.warning("active_hosts_provider_failed", exc_info=True)
        return set()


def active_allowed_ips() -> set:
    """IP allowlist for the live tunnel (empty = unrestricted/first-contact)."""
    try:
        return _active_ip_provider() or set()
    except Exception:  # pragma: no cover
        log.warning("active_ip_provider_failed", exc_info=True)
        return set()


def _bare_host(host: str) -> str:
    return (host or "").split(",", 1)[0].split(":", 1)[0].strip().lower()


def _ip_allowed(client_ip: str, allowlist: set) -> bool:
    """True when ``client_ip`` falls inside any allowlisted IP/CIDR.

    Conservative: a missing/unparseable client IP is NEVER allowed when an
    allowlist is in force (fail-closed), so a tunneled request that somehow
    arrives without ``Cf-Connecting-Ip`` can't slip past the pin.
    """
    raw = (client_ip or "").strip()
    if not raw:
        return False
    try:
        ip = ip_address(raw)
    except ValueError:
        return False
    for entry in allowlist:
        entry = (entry or "").strip()
        if not entry:
            continue
        try:
            if "/" in entry:
                if ip in ip_network(entry, strict=False):
                    return True
            elif ip == ip_address(entry):
                return True
        except ValueError:
            continue
    return False


def tunnel_request_blocked(host: str, path: str, client_ip: str = "") -> bool:
    """True when a request on a live tunnel host must be refused.

    Runs FIRST in the middleware (before auth). Two regimes on a live tunnel host:

    * **Pinned (IP allowlist in force)** — the IP *is* the credential. A
      whitelisted address gets full access (the normal auth middleware still
      applies underneath); every other address is refused outright. This is what
      lets a known recipient REGAIN access (reach /login + the app) while a
      leaked URL is dead from anywhere else.
    * **Open (no allowlist — first-contact onboarding)** — token-only, so the
      path is scoped tight to the invite door; not even /login is reachable.

    Returns False (no opinion) when no tunnel is active or the host isn't a
    tunnel host, leaving normal auth to handle it.
    """
    active = active_public_hosts()
    if not active:
        return False
    if _bare_host(host) not in {_bare_host(a) for a in active}:
        return False
    allow = active_allowed_ips()
    if allow:
        # Pinned: the whitelisted IP is trusted for the full surface (auth still
        # required); anyone else is blocked, even from the invite door.
        return not _ip_allowed(client_ip, allow)
    # Open onboarding: tight path-scope — only the invite door, token-gated.
    return not is_invite_public_path(path)


class ReachTier(IntEnum):
    """Reachability tiers, ordered by EXPOSURE (lower value = more private)."""

    LAN = 0
    TAILNET = 1
    TS_FUNNEL = 2
    CLOUDFLARED = 3

    @property
    def is_public(self) -> bool:
        """Whether this tier exposes the host to the open internet."""
        return self >= ReachTier.TS_FUNNEL

    @property
    def needs_tunnel(self) -> bool:
        """Whether reaching this tier requires standing up a tunnel."""
        return self in (ReachTier.TS_FUNNEL, ReachTier.CLOUDFLARED)


class RecipientScope(IntEnum):
    """How far the link must reach — the operator's privacy intent per invite."""

    SAME_LAN = 0   # they're on my local network
    TAILNET = 1    # they're on (or I'll add them to) my tailnet
    PUBLIC = 2     # they're anywhere on the internet


_SCOPE_ALIASES = {
    "lan": RecipientScope.SAME_LAN, "same_lan": RecipientScope.SAME_LAN,
    "local": RecipientScope.SAME_LAN, "network": RecipientScope.SAME_LAN,
    "tailnet": RecipientScope.TAILNET, "tailscale": RecipientScope.TAILNET,
    "vpn": RecipientScope.TAILNET,
    "public": RecipientScope.PUBLIC, "internet": RecipientScope.PUBLIC,
    "anywhere": RecipientScope.PUBLIC, "external": RecipientScope.PUBLIC,
}


def parse_recipient_scope(value: str | None, *, default: RecipientScope = RecipientScope.SAME_LAN) -> RecipientScope:
    """Map a UI/API string ('lan' | 'tailnet' | 'public') to a RecipientScope."""
    return _SCOPE_ALIASES.get((value or "").strip().lower(), default)


# The candidate tiers for each recipient scope, in increasing-exposure order.
# The planner picks the FIRST one that's available — i.e. the least-exposing
# reachable option. A scope can step UP the exposure ladder (fallback) but never
# below the minimum that actually reaches the recipient.
_LADDER: dict[RecipientScope, tuple[ReachTier, ...]] = {
    RecipientScope.SAME_LAN: (ReachTier.LAN, ReachTier.TAILNET, ReachTier.TS_FUNNEL, ReachTier.CLOUDFLARED),
    RecipientScope.TAILNET: (ReachTier.TAILNET, ReachTier.TS_FUNNEL, ReachTier.CLOUDFLARED),
    RecipientScope.PUBLIC: (ReachTier.TS_FUNNEL, ReachTier.CLOUDFLARED),
}

# The lowest tier that still satisfies a given recipient scope — used to detect
# a privacy DOWNGRADE (we had to expose more than the operator asked for).
_MIN_TIER_FOR_SCOPE: dict[RecipientScope, ReachTier] = {
    RecipientScope.SAME_LAN: ReachTier.LAN,
    RecipientScope.TAILNET: ReachTier.TAILNET,
    RecipientScope.PUBLIC: ReachTier.TS_FUNNEL,
}


@dataclass
class ReachCapabilities:
    """What the host can do RIGHT NOW for invite reachability.

    ``lan_host`` / ``tailnet_host`` are ``host[:port]`` strings (empty if
    unknown). ``funnel_available`` / ``cloudflared_available`` gate the public
    tiers — set by detecting the operator's opt-in + the tooling.
    """

    lan_host: str = ""
    tailnet_host: str = ""
    funnel_available: bool = False
    cloudflared_available: bool = False
    # The full standing public Funnel URL (``https://<node>.<tailnet>.ts.net[:port]``)
    # when the funnel tier is usable — either derived from the ts.net name + port
    # or given explicitly. Empty = funnel tier unavailable. Distinct from
    # ``tailnet_host`` because the public funnel URL can differ (MagicDNS name +
    # a non-443 port) from the tailnet-private static host (often a raw 100.x).
    funnel_url: str = ""
    # Live-drive mode: the app will discover the ts.net name + enable funnel at
    # ensure() time, so the tier is selectable even before a URL is known.
    funnel_live: bool = False

    def host_for(self, tier: ReachTier) -> str:
        """Static base host/URL for a non-tunnel tier."""
        if tier == ReachTier.LAN:
            return self.lan_host
        if tier == ReachTier.TAILNET:
            return self.tailnet_host
        if tier == ReachTier.TS_FUNNEL:
            # Already a full https:// URL (may carry a non-default port).
            return self.funnel_url
        return ""

    def supports(self, tier: ReachTier) -> bool:
        if tier == ReachTier.LAN:
            return bool(self.lan_host)
        if tier == ReachTier.TAILNET:
            return bool(self.tailnet_host)
        if tier == ReachTier.TS_FUNNEL:
            # Usable when we have a public funnel URL to hand out (config mode)
            # OR live-drive is on (URL discovered at ensure() time).
            return self.funnel_available and (bool(self.funnel_url) or self.funnel_live)
        if tier == ReachTier.CLOUDFLARED:
            return self.cloudflared_available
        return False


@dataclass
class ReachabilityPlan:
    """The chosen path for one invite link."""

    scope: RecipientScope
    tier: ReachTier | None                 # None → cannot reach at this scope
    host: str = ""                         # static host for non-tunnel tiers
    needs_tunnel: bool = False
    privacy_downgrade: bool = False        # had to expose more than asked
    note: str = ""                         # human-readable explanation / warning

    @property
    def reachable(self) -> bool:
        return self.tier is not None


def _host_kind(host: str) -> str:
    """Classify a host string as 'tailnet' | 'lan' | 'loopback' | 'public' | 'hostname'."""
    raw = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    raw = raw.strip("[]").strip().lower()
    if not raw or raw in ("localhost", "0.0.0.0"):
        return "loopback"
    if raw.endswith(_TS_NET_SUFFIX):
        return "tailnet"
    try:
        ip = ip_address(raw)
    except ValueError:
        return "hostname"  # a real domain — treat as public-ish
    if ip in _TAILSCALE_CGNAT:
        return "tailnet"
    if ip.is_loopback:
        return "loopback"
    if ip.is_private:
        return "lan"
    return "public"


@lru_cache(maxsize=1)
def _cloudflared_present() -> bool:
    """Whether the ``cloudflared`` binary is on PATH (cached — one-time which)."""
    return bool(shutil.which("cloudflared"))


# Tailscale CGNAT 100.64.0.0/10 (the range Tailscale assigns). Same filter
# start.sh/start.bat use to populate AUGMENTUM_TLS_EXTRA_SANS.
_TAILSCALE_IP_RE = re.compile(r"^100\.(?:6[4-9]|[7-9]\d|1[0-1]\d|12[0-7])\.\d{1,3}\.\d{1,3}$")


def tailnet_ip_from_sans(env: dict | None = None) -> str:
    """The host's Tailscale (100.64/10) IP from the auto-detected TLS SANs, or "".

    ``start.sh``/``start.bat`` already discover the host's LAN + Tailscale
    addresses and write them to ``AUGMENTUM_TLS_EXTRA_SANS`` for the cert, so
    this needs ZERO operator config. Without it, tailnet-scope invites can only
    find a tailnet host when the admin happens to be *browsing* over the tailnet
    (the single Host header the resolver sees) — which is why "My tailnet"
    otherwise handed back a LAN URL. Returns the bare IP (no port) or "".
    """
    env = os.environ if env is None else env
    sans = env.get("AUGMENTUM_TLS_EXTRA_SANS", "") or ""
    for token in sans.split(","):
        tok = token.strip()
        if not tok:
            continue
        # SAN entries look like "IP:100.64.0.1" / "DNS:host.example".
        upper = tok.upper()
        if upper.startswith("IP:"):
            tok = tok[3:].strip()
        elif upper.startswith("DNS:"):
            continue
        if _TAILSCALE_IP_RE.match(tok):
            return tok
    return ""


# Tailscale Funnel is only served on these ports (hard platform limit).
ALLOWED_FUNNEL_PORTS: tuple[int, ...] = (443, 8443, 10000)


def _bare_tsnet_name(host: str) -> str:
    """The ts.net MagicDNS name from a host string ('' if not a ts.net name).

    Strips any scheme, port, trailing dot, and path. A raw 100.x IP returns ""
    because it is NOT publicly funnel-addressable — Funnel serves the DNS name.
    """
    raw = (host or "").strip()
    raw = raw.split("://", 1)[-1]              # drop scheme
    raw = raw.split("/", 1)[0]                 # drop path
    raw = raw.rsplit(":", 1)[0] if raw.count(":") == 1 else raw  # drop :port
    raw = raw.rstrip(".").strip().lower()
    return raw if raw.endswith(_TS_NET_SUFFIX) else ""


def _derive_funnel_url(ts_host: str, port: str | int | None = None) -> str:
    """Build the standing public Funnel URL for a ts.net node, or "".

    Requires a ``*.ts.net`` MagicDNS name (a bare 100.x IP is not publicly
    funnel-addressable). Appends ``:port`` only when it's a non-default allowed
    funnel port (443 is implicit in https).
    """
    name = _bare_tsnet_name(ts_host)
    if not name:
        return ""
    p = str(port or "").strip()
    if p and p not in ("443", ""):
        return f"https://{name}:{p}"
    return f"https://{name}"


def tailnet_hostname_from_env(env: dict | None = None) -> str:
    """The node's ts.net MagicDNS name from env ('' if unset/not a ts.net name)."""
    env = os.environ if env is None else env
    return _bare_tsnet_name(env.get("AUGMENTUM_TAILNET_HOSTNAME", "") or "")


def detect_capabilities(
    *, resolver_host: str = "", configured_host: str = "", env: dict | None = None,
    cloudflared_present: bool | None = None, extra_hosts: tuple[str, ...] = (),
) -> ReachCapabilities:
    """Build a capability snapshot from the resolver + configured host + env.

    - ``resolver_host`` — the best LAN/tailnet host the PublicHostResolver learned.
    - ``configured_host`` — the operator's ``AUGMENTUM_PUBLIC_HOST`` (if any).
    - ``extra_hosts`` — additional ``host[:port]`` candidates to classify (e.g.
      the machine's own Tailscale IP, which the resolver never learns unless the
      admin browsed over the tailnet). Considered AFTER the primary hosts, so a
      LAN request host still fills ``lan_host`` while the tailnet IP fills
      ``tailnet_host``.
    - ``env`` — defaults to ``os.environ``.
    - ``cloudflared_present`` — test seam; defaults to PATH auto-detection.

    **Cloudflared is zero-config**: if the binary is on PATH it's available, so a
    low-tech user just picks "Anywhere" and it works. ``AUGMENTUM_CONNECT_
    CLOUDFLARED`` (truthy/falsy) is an explicit override either way. **Funnel
    stays an explicit opt-in** (``AUGMENTUM_CONNECT_FUNNEL``) because it needs
    tailnet policy setup — not something to enable behind the operator's back.
    Either way, the per-invite "Anywhere" choice is the actual consent to expose.
    """
    env = os.environ if env is None else env
    caps = ReachCapabilities()
    # The ts.net MagicDNS name (start-script detected) is the BEST tailnet host —
    # try it first so caps.tailnet_host is the stable name, not a raw 100.x.
    ts_name = tailnet_hostname_from_env(env)
    for host in (ts_name, configured_host, resolver_host, *extra_hosts):
        host = (host or "").strip()
        if not host:
            continue
        kind = _host_kind(host)
        if kind == "tailnet" and not caps.tailnet_host:
            caps.tailnet_host = host
        elif kind in ("lan", "hostname", "public") and not caps.lan_host:
            # A configured public domain counts as a LAN-tier static host too
            # (it's reachable without a tunnel); classification only steers the
            # privacy LADDER, not whether the URL works.
            caps.lan_host = host

    # Funnel — real-signal detection (not a bare flag): usable only when we can
    # actually produce a public URL. An explicit URL wins; otherwise the opt-in
    # flag + a ts.net name derives one. A bare 100.x IP can't be funnelled, so
    # a tailnet-but-no-ts.net-name host yields no funnel URL and the planner
    # falls to cloudflared (graceful). Live-drive mode fills funnel_url itself.
    explicit_url = (env.get("AUGMENTUM_CONNECT_FUNNEL_URL") or "").strip()
    funnel_optin = _is_truthy(env.get("AUGMENTUM_CONNECT_FUNNEL"))
    if explicit_url:
        caps.funnel_url = explicit_url
    elif funnel_optin:
        caps.funnel_url = _derive_funnel_url(
            ts_name or caps.tailnet_host, env.get("AUGMENTUM_CONNECT_FUNNEL_PORT"),
        )
    caps.funnel_live = funnel_optin and _is_truthy(env.get("AUGMENTUM_CONNECT_FUNNEL_LIVE"))
    caps.funnel_available = bool(caps.funnel_url) or caps.funnel_live

    # Cloudflared — env override wins (force on/off); otherwise auto-detect.
    cf_env = env.get("AUGMENTUM_CONNECT_CLOUDFLARED")
    if cf_env not in (None, ""):
        caps.cloudflared_available = _is_truthy(cf_env)
    elif cloudflared_present is not None:
        caps.cloudflared_available = cloudflared_present
    else:
        caps.cloudflared_available = _cloudflared_present()
    return caps


def _is_truthy(v: object) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def plan_reachability(scope: RecipientScope, caps: ReachCapabilities) -> ReachabilityPlan:
    """Pick the least-exposing reachable tier for ``scope`` given ``caps``.

    Walks the scope's ladder (least→most exposure) and returns the first
    available tier. Sets ``privacy_downgrade`` when the chosen tier exposes more
    than the scope's minimum (e.g. a TAILNET recipient but no tailnet, so we had
    to go public). Returns an unreachable plan (``tier=None``) when nothing in
    the ladder is available.
    """
    for tier in _LADDER[scope]:
        if not caps.supports(tier):
            continue
        downgrade = tier > _MIN_TIER_FOR_SCOPE[scope]
        note = ""
        if downgrade:
            note = (
                f"Wanted {_MIN_TIER_FOR_SCOPE[scope].name.lower()} reach but it's "
                f"unavailable — using {tier.name.lower()} (more exposure)."
            )
        elif tier == ReachTier.TS_FUNNEL:
            note = "Public link via your Tailscale Funnel — a stable, private address that stays up."
        elif tier.is_public:
            note = f"Public link via {tier.name.lower()} — torn down when the invite is used or expires."
        return ReachabilityPlan(
            scope=scope, tier=tier, host=caps.host_for(tier),
            needs_tunnel=tier.needs_tunnel, privacy_downgrade=downgrade, note=note,
        )
    return ReachabilityPlan(
        scope=scope, tier=None,
        note="No reachable path for this recipient — set a public host, enable Tailscale Funnel, or cloudflared.",
    )


# ── Engine interface (pluggable backends) ──────────────────────────────────

class EngineUnavailable(RuntimeError):
    """Raised by an engine's ``ensure`` when it can't actually establish."""


class ReachabilityEngine:
    """Backend for one tier. Subclasses implement establish/teardown.

    Non-tunnel engines (LAN/TAILNET) just hand back a static URL. Tunnel engines
    (Funnel/Cloudflared) establish public exposure on ``ensure`` and tear it
    down on ``release``; they ref-count by invite so overlapping invites share
    one tunnel and the last release closes it.
    """

    tier: ReachTier

    def is_available(self, caps: ReachCapabilities) -> bool:
        return caps.supports(self.tier)

    async def ensure(
        self, *, invite_id: str, plan: ReachabilityPlan, ttl_seconds: int,
        allowed_ips: list | None = None,
    ) -> str:
        """Return the base URL (e.g. ``https://host``) reachable for this invite.

        ``allowed_ips`` pins the public exposure to specific IPs/CIDRs (used by
        tunnel tiers for IP-whitelisted re-access); static tiers ignore it.
        """
        raise NotImplementedError

    async def release(self, *, invite_id: str) -> None:
        """Release any exposure held for ``invite_id`` (no-op for static tiers)."""
        return None


def _scheme_for_host(host: str) -> str:
    """https for a real port-443/domain/ts.net, else the 6443 TLS front door."""
    bare = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    if host.endswith(":443") or _host_kind(host) in ("tailnet", "hostname", "public") or "." in bare:
        return "https"
    return "http"


def _static_url(host: str) -> str:
    host = (host or "").strip()
    if not host:
        return ""
    return f"{_scheme_for_host(host)}://{host}"


class _StaticEngine(ReachabilityEngine):
    """Shared base for the non-tunnel tiers — URL is the known host, no process."""

    async def ensure(
        self, *, invite_id: str, plan: ReachabilityPlan, ttl_seconds: int,
        allowed_ips: list | None = None,
    ) -> str:
        if not plan.host:
            raise EngineUnavailable(f"{self.tier.name}: no host known")
        return _static_url(plan.host)


class LanEngine(_StaticEngine):
    tier = ReachTier.LAN


class TailnetEngine(_StaticEngine):
    tier = ReachTier.TAILNET


_CF_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.IGNORECASE)


class CloudflaredEngine(ReachabilityEngine):
    """Throwaway public exposure via ``cloudflared tunnel --url``.

    The bones: ``is_available`` gates on the operator opt-in; ``ensure`` spawns
    cloudflared pointed at the path-scoped ingress and parses the assigned
    ``*.trycloudflare.com`` URL from its output; ``release`` kills the process
    (ref-counted by invite). The subprocess lifecycle is intentionally thin and
    only runs when the binary is actually present.
    """

    tier = ReachTier.CLOUDFLARED

    def __init__(self) -> None:
        self._procs: dict[str, object] = {}  # invite_id → Popen

    @staticmethod
    def parse_url(text: str) -> str:
        """Pull the assigned trycloudflare URL out of cloudflared's output."""
        m = _CF_URL_RE.search(text or "")
        return m.group(0) if m else ""

    async def ensure(
        self, *, invite_id: str, plan: ReachabilityPlan, ttl_seconds: int,
        allowed_ips: list | None = None,
    ) -> str:
        # Live spawn is environment-dependent (binary + the local ingress target)
        # and is wired by the route layer's tunnel manager; the bones expose the
        # contract + URL parsing. Raise so the planner's caller falls back when
        # the runtime isn't provisioned.
        raise EngineUnavailable(
            "cloudflared engine: live spawn provisioned by the tunnel manager (not in the bones)",
        )

    async def release(self, *, invite_id: str) -> None:
        proc = self._procs.pop(invite_id, None)
        if proc is not None:
            try:
                proc.terminate()  # type: ignore[attr-defined]
            except Exception:
                log.warning("cloudflared_release_failed", invite_id=invite_id, exc_info=True)


class FunnelEngine(ReachabilityEngine):
    """Public exposure via Tailscale Funnel — a STANDING, durable door.

    Config mode (the default, and the only mode that works when the app can't
    reach tailscaled — the common containerized case): the operator enables
    funnel host-side and the standing public URL arrives here as ``plan.host``
    (already a full ``https://<node>.ts.net[:port]`` from ``caps.funnel_url``).
    ``ensure`` just hands it back; ``release`` is a no-op — unlike the ephemeral
    cloudflared tunnel, a funnel URL is stable across restarts and is the guest's
    ONGOING transport, so it is never torn down per-invite.

    Live-drive mode (``LiveFunnelManager`` in ``funnel_manager.py``) subclasses
    this and actually toggles ``tailscale funnel`` when the app CAN reach the CLI.
    """

    tier = ReachTier.TS_FUNNEL

    async def ensure(
        self, *, invite_id: str, plan: ReachabilityPlan, ttl_seconds: int,
        allowed_ips: list | None = None,
    ) -> str:
        url = (plan.host or "").strip()
        if not url:
            # No funnel URL configured — the planner's caller falls to cloudflared.
            raise EngineUnavailable("funnel engine: no funnel url configured")
        # Already a full https:// URL (carries any non-default port). Return as-is.
        return url if "://" in url else _static_url(url)

    async def release(self, *, invite_id: str) -> None:
        # Standing door — nothing to release per-invite.
        return None


# Registry — one engine per tier. The route layer can swap in live tunnel
# engines (with real subprocess/CLI lifecycle) by re-registering.
_ENGINES: dict[ReachTier, ReachabilityEngine] = {
    ReachTier.LAN: LanEngine(),
    ReachTier.TAILNET: TailnetEngine(),
    ReachTier.TS_FUNNEL: FunnelEngine(),
    ReachTier.CLOUDFLARED: CloudflaredEngine(),
}


def get_engine(tier: ReachTier) -> ReachabilityEngine:
    return _ENGINES[tier]


def register_engine(engine: ReachabilityEngine) -> None:
    """Swap in a live engine for its tier (e.g. a real cloudflared manager)."""
    _ENGINES[engine.tier] = engine
