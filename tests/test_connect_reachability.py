"""Adaptive invite reachability — the privacy ladder, capability detection,
fallback, and the path-scoped public surface (comms platform Phase 3 bones)."""

from __future__ import annotations

from augmentum.connect.reachability import (
    PUBLIC_INVITE_PATHS,
    CloudflaredEngine,
    ReachCapabilities,
    ReachTier,
    RecipientScope,
    detect_capabilities,
    is_invite_public_path,
    parse_recipient_scope,
    plan_reachability,
    tailnet_ip_from_sans,
)

# ── tailnet host discovery from TLS SANs ──────────────────────────

def test_tailnet_ip_from_sans_extracts_cgnat():
    env = {"AUGMENTUM_TLS_EXTRA_SANS": "IP:192.168.1.42,IP:100.64.0.1,DNS:host.local"}
    assert tailnet_ip_from_sans(env) == "100.64.0.1"


def test_tailnet_ip_from_sans_empty_when_no_tailnet():
    env = {"AUGMENTUM_TLS_EXTRA_SANS": "IP:192.168.1.42,DNS:host.local"}
    assert tailnet_ip_from_sans(env) == ""
    assert tailnet_ip_from_sans({}) == ""


def test_lan_browse_still_finds_tailnet_via_extra_hosts():
    # The exact bug: admin mints from a LAN browser (resolver sees only the
    # 192.168 host). Without the tailnet IP as an extra candidate, a tailnet
    # invite has no tailnet host and falls back to LAN. With it, the tailnet
    # tier is reachable and carries the port we were reached on.
    caps = detect_capabilities(
        resolver_host="192.168.1.42:6443",
        extra_hosts=("100.64.0.1:6443",),
        cloudflared_present=False,
    )
    assert caps.lan_host == "192.168.1.42:6443"
    assert caps.tailnet_host == "100.64.0.1:6443"
    plan = plan_reachability(RecipientScope.TAILNET, caps)
    assert plan.tier == ReachTier.TAILNET
    assert plan.host == "100.64.0.1:6443"
    assert plan.privacy_downgrade is False


def test_no_tailnet_host_leaves_tailnet_scope_unreachable():
    # No tailnet IP anywhere → tailnet scope must be UNREACHABLE (so the UI
    # blocks) rather than silently satisfiable by the LAN host.
    caps = detect_capabilities(
        resolver_host="192.168.1.42:6443", cloudflared_present=False,
    )
    assert caps.tailnet_host == ""
    plan = plan_reachability(RecipientScope.TAILNET, caps)
    assert plan.tier is None
    assert not plan.reachable

# ── tier / scope basics ───────────────────────────────────────────

def test_tier_exposure_ordering():
    assert ReachTier.LAN < ReachTier.TAILNET < ReachTier.TS_FUNNEL < ReachTier.CLOUDFLARED
    assert not ReachTier.LAN.is_public and not ReachTier.TAILNET.is_public
    assert ReachTier.TS_FUNNEL.is_public and ReachTier.CLOUDFLARED.is_public
    assert ReachTier.LAN.needs_tunnel is False
    assert ReachTier.CLOUDFLARED.needs_tunnel is True


def test_parse_recipient_scope():
    assert parse_recipient_scope("lan") == RecipientScope.SAME_LAN
    assert parse_recipient_scope("tailnet") == RecipientScope.TAILNET
    assert parse_recipient_scope("public") == RecipientScope.PUBLIC
    assert parse_recipient_scope("anywhere") == RecipientScope.PUBLIC
    assert parse_recipient_scope(None) == RecipientScope.SAME_LAN  # safe default


# ── least-exposure selection ──────────────────────────────────────

def test_lan_recipient_prefers_lan_no_tunnel():
    caps = ReachCapabilities(lan_host="192.168.1.10:6443", tailnet_host="host.x.ts.net")
    plan = plan_reachability(RecipientScope.SAME_LAN, caps)
    assert plan.tier == ReachTier.LAN
    assert plan.needs_tunnel is False
    assert plan.privacy_downgrade is False
    assert plan.host == "192.168.1.10:6443"


def test_tailnet_recipient_prefers_tailnet_over_public():
    caps = ReachCapabilities(
        tailnet_host="host.x.ts.net", funnel_available=True, cloudflared_available=True,
    )
    plan = plan_reachability(RecipientScope.TAILNET, caps)
    assert plan.tier == ReachTier.TAILNET  # private wins even though public is available
    assert plan.privacy_downgrade is False


def test_public_recipient_prefers_funnel_over_cloudflared():
    # Funnel is usable only when a real public URL exists (funnel_url), not a
    # bare flag — mirrors the config-mode contract.
    caps = ReachCapabilities(
        tailnet_host="host.x.ts.net", funnel_available=True,
        funnel_url="https://host.x.ts.net:8443", cloudflared_available=True,
    )
    plan = plan_reachability(RecipientScope.PUBLIC, caps)
    assert plan.tier == ReachTier.TS_FUNNEL  # least-exposing public tier
    assert plan.host == "https://host.x.ts.net:8443"
    assert plan.privacy_downgrade is False  # funnel satisfies the public minimum


def test_public_falls_back_to_cloudflared_when_no_funnel():
    caps = ReachCapabilities(cloudflared_available=True)  # no tailnet/funnel
    plan = plan_reachability(RecipientScope.PUBLIC, caps)
    assert plan.tier == ReachTier.CLOUDFLARED
    assert plan.needs_tunnel is True


# ── fallback + privacy downgrade ──────────────────────────────────

def test_tailnet_recipient_downgrades_to_public_when_no_tailnet():
    # Recipient is on a tailnet I can't reach privately → must step UP exposure.
    caps = ReachCapabilities(lan_host="192.168.1.10:6443", cloudflared_available=True)
    plan = plan_reachability(RecipientScope.TAILNET, caps)
    assert plan.tier == ReachTier.CLOUDFLARED
    assert plan.privacy_downgrade is True
    assert "more exposure" in plan.note


def test_unreachable_when_public_asked_but_nothing_available():
    caps = ReachCapabilities(lan_host="192.168.1.10:6443")  # only LAN
    plan = plan_reachability(RecipientScope.PUBLIC, caps)
    assert plan.tier is None
    assert plan.reachable is False
    assert "No reachable path" in plan.note


def test_lan_recipient_steps_up_when_lan_host_unknown():
    # No LAN host learned yet, but a tailnet host exists → use it (still private).
    caps = ReachCapabilities(tailnet_host="host.x.ts.net")
    plan = plan_reachability(RecipientScope.SAME_LAN, caps)
    assert plan.tier == ReachTier.TAILNET
    assert plan.privacy_downgrade is True


# ── capability detection ──────────────────────────────────────────

def test_detect_classifies_tailnet_vs_lan():
    caps = detect_capabilities(
        resolver_host="100.64.0.1:6443", configured_host="", env={},
    )
    assert caps.tailnet_host == "100.64.0.1:6443"
    assert caps.lan_host == ""


def test_detect_ts_net_hostname_is_tailnet():
    caps = detect_capabilities(resolver_host="box.tail1234.ts.net", env={})
    assert caps.tailnet_host == "box.tail1234.ts.net"


def test_detect_private_ip_is_lan():
    caps = detect_capabilities(resolver_host="192.168.1.10:6443", env={})
    assert caps.lan_host == "192.168.1.10:6443"
    assert caps.tailnet_host == ""


def test_detect_funnel_needs_real_url_not_just_flag():
    # Real-signal contract: the opt-in flag ALONE isn't enough — funnel is only
    # usable when a public URL can be produced (a ts.net name to derive one, or
    # an explicit URL, or live-drive mode).
    base = detect_capabilities(resolver_host="192.168.1.10:6443", env={}, cloudflared_present=False)
    assert base.funnel_available is False

    # opt-in but only a LAN host / no ts.net name → no URL → NOT available.
    flag_only = detect_capabilities(
        resolver_host="192.168.1.10:6443", cloudflared_present=False,
        env={"AUGMENTUM_CONNECT_FUNNEL": "1"},
    )
    assert flag_only.funnel_available is False
    assert flag_only.funnel_url == ""

    # opt-in + a ts.net hostname → derives a public URL → available.
    with_name = detect_capabilities(
        cloudflared_present=False,
        env={"AUGMENTUM_CONNECT_FUNNEL": "1",
             "AUGMENTUM_TAILNET_HOSTNAME": "node.tail1234.ts.net",
             "AUGMENTUM_CONNECT_FUNNEL_PORT": "8443"},
    )
    assert with_name.funnel_available is True
    assert with_name.funnel_url == "https://node.tail1234.ts.net:8443"

    # explicit URL wins regardless of opt-in flag.
    explicit = detect_capabilities(
        cloudflared_present=False,
        env={"AUGMENTUM_CONNECT_FUNNEL_URL": "https://node.tail1234.ts.net:10000"},
    )
    assert explicit.funnel_available is True
    assert explicit.funnel_url == "https://node.tail1234.ts.net:10000"


def test_detect_cloudflared_auto_detects_binary():
    # Zero-config: binary on PATH → available, no env needed.
    present = detect_capabilities(env={}, cloudflared_present=True)
    assert present.cloudflared_available is True
    absent = detect_capabilities(env={}, cloudflared_present=False)
    assert absent.cloudflared_available is False


def test_detect_cloudflared_env_override_wins():
    # Force OFF even though the binary is present...
    off = detect_capabilities(
        env={"AUGMENTUM_CONNECT_CLOUDFLARED": "0"}, cloudflared_present=True,
    )
    assert off.cloudflared_available is False
    # ...and force ON even though the binary is absent (e.g. a sidecar).
    on = detect_capabilities(
        env={"AUGMENTUM_CONNECT_CLOUDFLARED": "1"}, cloudflared_present=False,
    )
    assert on.cloudflared_available is True


def test_detect_prefers_both_lan_and_tailnet_hosts():
    # A configured public domain (LAN-tier static) + a learned tailnet IP.
    caps = detect_capabilities(
        configured_host="connect.example.com", resolver_host="100.64.0.1:6443", env={},
    )
    assert caps.lan_host == "connect.example.com"
    assert caps.tailnet_host == "100.64.0.1:6443"


# ── path-scoped public surface ────────────────────────────────────

def test_public_invite_paths_allow_only_the_door():
    assert is_invite_public_path("/ui/connect-join/")
    assert is_invite_public_path("/ui/connect-join/connect-join.js")
    assert is_invite_public_path("/api/auth/invite/sometoken")
    assert is_invite_public_path("/api/auth/invite/sometoken/claim")
    # Everything else is closed — even login and the invite MANAGEMENT route.
    assert not is_invite_public_path("/api/auth/login")
    assert not is_invite_public_path("/api/auth/invites")  # admin mgmt (plural)
    assert not is_invite_public_path("/api/chats/sync")
    assert not is_invite_public_path("/ui/index.html")


def test_supports_matrix():
    caps = ReachCapabilities(tailnet_host="h.ts.net", funnel_available=False)
    assert caps.supports(ReachTier.TAILNET) is True
    # Funnel needs a real public URL, not just the availability flag.
    assert caps.supports(ReachTier.TS_FUNNEL) is False
    caps.funnel_available = True
    assert caps.supports(ReachTier.TS_FUNNEL) is False  # still no URL
    caps.funnel_url = "https://h.ts.net:8443"
    assert caps.supports(ReachTier.TS_FUNNEL) is True


# ── engine URL shapes (pure parts of the tunnel adapters) ─────────

def test_cloudflared_url_parsing():
    out = "2026-06-21 INF +-----+\n|  https://brave-tiger-1234.trycloudflare.com  |\n+-----+"
    assert CloudflaredEngine.parse_url(out) == "https://brave-tiger-1234.trycloudflare.com"
    assert CloudflaredEngine.parse_url("no url here") == ""


def test_funnel_url_derivation():
    from augmentum.connect.reachability import _derive_funnel_url
    # ts.net name + default port → no :443
    assert _derive_funnel_url("box.tail1234.ts.net") == "https://box.tail1234.ts.net"
    assert _derive_funnel_url("box.tail1234.ts.net", "443") == "https://box.tail1234.ts.net"
    # custom funnel port carried through
    assert _derive_funnel_url("box.tail1234.ts.net", "8443") == "https://box.tail1234.ts.net:8443"
    assert _derive_funnel_url("box.tail1234.ts.net:6443", 10000) == "https://box.tail1234.ts.net:10000"
    # a bare 100.x IP is NOT publicly funnel-addressable → empty
    assert _derive_funnel_url("100.64.0.1:6443", "8443") == ""
    assert _derive_funnel_url("") == ""


def test_public_invite_paths_is_nonempty_contract():
    assert "/api/auth/invite/" in PUBLIC_INVITE_PATHS
    assert "/api/auth/invites" not in PUBLIC_INVITE_PATHS  # mgmt stays private


# ── middleware tunnel guard ───────────────────────────────────────

def test_tunnel_guard_blocks_non_invite_paths_on_tunnel_host():
    from augmentum.connect import reachability as r

    r.set_active_hosts_provider(lambda: {r.INVITE_TUNNEL_HOST})
    try:
        # On the tunnel host, only the invite door is allowed.
        assert r.tunnel_request_blocked(r.INVITE_TUNNEL_HOST, "/ui/connect-join/") is False
        assert r.tunnel_request_blocked(r.INVITE_TUNNEL_HOST, "/api/auth/invite/tok") is False
        assert r.tunnel_request_blocked(r.INVITE_TUNNEL_HOST, "/api/auth/login") is True
        assert r.tunnel_request_blocked(r.INVITE_TUNNEL_HOST, "/api/chats/sync") is True
        # Host with a port still matches (bare-host compare).
        assert r.tunnel_request_blocked(f"{r.INVITE_TUNNEL_HOST}:443", "/api/auth/login") is True
        # A different host (normal LAN access) is never blocked by this guard.
        assert r.tunnel_request_blocked("192.168.1.10", "/api/auth/login") is False
    finally:
        r.set_active_hosts_provider(lambda: set())


def test_tunnel_guard_noop_when_no_tunnel_active():
    from augmentum.connect import reachability as r

    r.set_active_hosts_provider(lambda: set())
    # No active tunnel → the guard has no opinion on anything.
    assert r.tunnel_request_blocked(r.INVITE_TUNNEL_HOST, "/api/auth/login") is False


def test_active_hosts_provider_fault_is_swallowed():
    from augmentum.connect import reachability as r

    def _boom():
        raise RuntimeError("provider exploded")

    r.set_active_hosts_provider(_boom)
    try:
        assert r.active_public_hosts() == set()  # never propagates → never 500s
        assert r.tunnel_request_blocked(r.INVITE_TUNNEL_HOST, "/api/auth/login") is False
    finally:
        r.set_active_hosts_provider(lambda: set())


# ── IP whitelist (regain-access security) ─────────────────────────

def test_ip_allowed_matches_exact_and_cidr():
    from augmentum.connect.reachability import _ip_allowed

    assert _ip_allowed("1.2.3.4", {"1.2.3.4"}) is True
    assert _ip_allowed("1.2.3.4", {"1.2.3.0/24"}) is True
    assert _ip_allowed("9.9.9.9", {"1.2.3.0/24"}) is False
    # Fail-closed: a missing or unparseable client IP is never allowed.
    assert _ip_allowed("", {"1.2.3.4"}) is False
    assert _ip_allowed("not-an-ip", {"1.2.3.4"}) is False


def test_tunnel_guard_pinned_ip_is_the_credential():
    from augmentum.connect import reachability as r

    r.set_active_hosts_provider(lambda: {r.INVITE_TUNNEL_HOST})
    r.set_active_ip_provider(lambda: {"1.2.3.4"})
    try:
        h = r.INVITE_TUNNEL_HOST
        # Whitelisted IP gets the FULL surface for re-access — including /login
        # and the app (normal auth still applies underneath).
        assert r.tunnel_request_blocked(h, "/api/auth/invite/tok", "1.2.3.4") is False
        assert r.tunnel_request_blocked(h, "/api/auth/login", "1.2.3.4") is False
        assert r.tunnel_request_blocked(h, "/api/chats/sync", "1.2.3.4") is False
        # Any OTHER IP gets nothing — a leaked URL is dead off the pinned IP.
        assert r.tunnel_request_blocked(h, "/api/auth/invite/tok", "5.6.7.8") is True
        assert r.tunnel_request_blocked(h, "/api/auth/login", "5.6.7.8") is True
        # No client IP supplied → fail-closed.
        assert r.tunnel_request_blocked(h, "/api/auth/invite/tok", "") is True
    finally:
        r.set_active_hosts_provider(lambda: set())
        r.set_active_ip_provider(lambda: set())


def test_tunnel_guard_empty_allowlist_is_first_contact_open():
    from augmentum.connect import reachability as r

    r.set_active_hosts_provider(lambda: {r.INVITE_TUNNEL_HOST})
    r.set_active_ip_provider(lambda: set())  # unrestricted (onboarding)
    try:
        # Any IP may reach the invite door when no allowlist is in force.
        assert r.tunnel_request_blocked(r.INVITE_TUNNEL_HOST, "/api/auth/invite/tok", "5.6.7.8") is False
    finally:
        r.set_active_hosts_provider(lambda: set())
        r.set_active_ip_provider(lambda: set())


class TestScopeHostSpoofResistance:
    """The tunnel guard's host check must read the REAL Host header only.

    cloudflared pins tunneled requests' Host to the sentinel via
    --http-host-header; X-Forwarded-Host is visitor-controlled and passed
    through untouched. Honoring it let a tunnel visitor dodge the
    sentinel match and reach the full app surface (2026-07-16 open bug).
    """

    @staticmethod
    def _scope(headers: list[tuple[bytes, bytes]]) -> dict:
        return {"headers": headers}

    def test_raw_host_wins(self):
        from augmentum.auth.middleware import _scope_host
        scope = self._scope([
            (b"host", b"augmentum-invite-gate.internal"),
        ])
        assert _scope_host(scope) == "augmentum-invite-gate.internal"

    def test_x_forwarded_host_is_ignored(self):
        from augmentum.auth.middleware import _scope_host
        scope = self._scope([
            (b"x-forwarded-host", b"attacker-chosen.example"),
            (b"host", b"augmentum-invite-gate.internal"),
        ])
        # The spoofable header must never shadow the pinned Host.
        assert _scope_host(scope) == "augmentum-invite-gate.internal"

    def test_spoofed_header_cannot_unscope_tunnel(self):
        from augmentum.auth.middleware import _scope_host
        from augmentum.connect.reachability import (
            INVITE_TUNNEL_HOST,
            set_active_hosts_provider,
            set_active_ip_provider,
            tunnel_request_blocked,
        )
        set_active_hosts_provider(lambda: {INVITE_TUNNEL_HOST})
        set_active_ip_provider(set)
        try:
            scope = self._scope([
                (b"x-forwarded-host", b"not-the-tunnel.example"),
                (b"host", INVITE_TUNNEL_HOST.encode()),
            ])
            # Guard must still see tunnel traffic and block /login.
            assert tunnel_request_blocked(_scope_host(scope), "/login") is True
            # The invite door itself stays reachable.
            assert tunnel_request_blocked(
                _scope_host(scope), "/ui/connect-join/x",
            ) is False
        finally:
            set_active_hosts_provider(set)
            set_active_ip_provider(set)
