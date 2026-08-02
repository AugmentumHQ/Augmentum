"""CloudflaredTunnelManager lifecycle — driven entirely by a fake process + fake
clock, so the ref-counting / URL-capture / teardown / reap policy is verifiable
without the real cloudflared binary."""

from __future__ import annotations

import pytest

from augmentum.connect import reachability as r
from augmentum.connect.reachability import (
    INVITE_TUNNEL_HOST,
    EngineUnavailable,
    ReachabilityPlan,
    ReachTier,
    RecipientScope,
)
from augmentum.connect.tunnel_manager import CloudflaredTunnelManager


class FakeProc:
    """A scripted TunnelProcess — yields ``lines`` then EOF; records terminate."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)
        self.terminated = False

    async def readline(self) -> str:
        return self._lines.pop(0) if self._lines else ""

    def terminate(self) -> None:
        self.terminated = True

    @property
    def returncode(self):
        return None


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, secs: float) -> None:
        self.t += secs


_URL_LINE = "2026-06-21 INF |  https://brave-tiger-1234.trycloudflare.com  |"


def _plan() -> ReachabilityPlan:
    return ReachabilityPlan(scope=RecipientScope.PUBLIC, tier=ReachTier.CLOUDFLARED, needs_tunnel=True)


def _manager(procs: list[FakeProc], clock: FakeClock) -> CloudflaredTunnelManager:
    seq = iter(procs)

    async def launcher(_target: str):
        return next(seq)

    # reap_interval_s=0 disables the background reaper — tests reap explicitly.
    return CloudflaredTunnelManager(
        launcher=launcher, clock=clock, url_timeout_s=5.0, reap_interval_s=0,
    )


@pytest.fixture(autouse=True)
def _reset_provider():
    yield
    r.set_active_hosts_provider(lambda: set())
    r.set_active_ip_provider(lambda: set())


@pytest.mark.asyncio
async def test_ensure_captures_url_and_publishes_sentinel_host():
    clock = FakeClock()
    mgr = _manager([FakeProc(["booting...", _URL_LINE])], clock)

    url = await mgr.ensure(invite_id="inv1", plan=_plan(), ttl_seconds=3600)
    assert url == "https://brave-tiger-1234.trycloudflare.com"
    # While up, the guard sees the sentinel host (NOT the random URL host).
    assert mgr.active_public_hosts() == {INVITE_TUNNEL_HOST}
    assert r.active_public_hosts() == {INVITE_TUNNEL_HOST}


@pytest.mark.asyncio
async def test_shared_tunnel_is_refcounted():
    clock = FakeClock()
    # Only ONE process is ever launched even though two invites use it.
    p = FakeProc(["x", _URL_LINE])
    mgr = _manager([p, FakeProc(["should-not-be-used", _URL_LINE])], clock)

    u1 = await mgr.ensure(invite_id="a", plan=_plan(), ttl_seconds=3600)
    u2 = await mgr.ensure(invite_id="b", plan=_plan(), ttl_seconds=3600)
    assert u1 == u2  # same shared tunnel

    # First release keeps it up (b still holds a ref)...
    await mgr.release(invite_id="a")
    assert mgr.active_public_hosts() == {INVITE_TUNNEL_HOST}
    assert p.terminated is False
    # ...last release tears it down.
    await mgr.release(invite_id="b")
    assert mgr.active_public_hosts() == set()
    assert p.terminated is True


@pytest.mark.asyncio
async def test_url_capture_timeout_raises_and_cleans_up():
    clock = FakeClock()
    # Process never prints a URL → EOF with no match → unavailable.
    p = FakeProc(["starting", "no url here", ""])
    mgr = _manager([p], clock)

    with pytest.raises(EngineUnavailable):
        await mgr.ensure(invite_id="x", plan=_plan(), ttl_seconds=3600)
    assert p.terminated is True
    assert mgr.active_public_hosts() == set()
    # The failed invite left no dangling ref.
    assert mgr._refs == {}


@pytest.mark.asyncio
async def test_spawn_failure_raises_unavailable():
    async def boom(_target):
        raise FileNotFoundError("cloudflared not installed")

    mgr = CloudflaredTunnelManager(launcher=boom, clock=FakeClock())
    with pytest.raises(EngineUnavailable):
        await mgr.ensure(invite_id="x", plan=_plan(), ttl_seconds=3600)
    assert mgr.active_public_hosts() == set()


@pytest.mark.asyncio
async def test_reap_tears_down_after_ttl_expires():
    clock = FakeClock()
    p = FakeProc(["x", _URL_LINE])
    mgr = _manager([p], clock)

    await mgr.ensure(invite_id="inv", plan=_plan(), ttl_seconds=60)
    assert mgr.active_public_hosts() == {INVITE_TUNNEL_HOST}

    # Before TTL: reap is a no-op.
    clock.advance(30)
    await mgr.reap()
    assert mgr.active_public_hosts() == {INVITE_TUNNEL_HOST}
    assert p.terminated is False

    # After TTL: the unclaimed invite's ref lapses → tunnel closes.
    clock.advance(40)
    await mgr.reap()
    assert mgr.active_public_hosts() == set()
    assert p.terminated is True


@pytest.mark.asyncio
async def test_release_unknown_invite_is_safe():
    mgr = _manager([FakeProc(["x", _URL_LINE])], FakeClock())
    # Releasing before/without an ensure must not blow up (claim path calls it
    # unconditionally and it should no-op for non-tunnel invites).
    await mgr.release(invite_id="never-existed")
    assert mgr.active_public_hosts() == set()


@pytest.mark.asyncio
async def test_ttl_clamped_to_hard_ceiling():
    from augmentum.connect.tunnel_manager import MAX_TUNNEL_LIFETIME_S

    clock = FakeClock()
    mgr = _manager([FakeProc(["x", _URL_LINE])], clock)
    # Ask for an absurd TTL; the stored deadline must be clamped.
    await mgr.ensure(invite_id="inv", plan=_plan(), ttl_seconds=10**9)
    assert mgr._refs["inv"] <= clock.t + MAX_TUNNEL_LIFETIME_S


# ── IP-allowlist union semantics ──────────────────────────────────

@pytest.mark.asyncio
async def test_pinned_invite_restricts_tunnel_to_its_ips():
    mgr = _manager([FakeProc(["x", _URL_LINE])], FakeClock())
    await mgr.ensure(invite_id="a", plan=_plan(), ttl_seconds=3600, allowed_ips=["1.2.3.4"])
    # The published allowlist (and the global provider) carry the pin.
    assert mgr.active_allowed_ips() == {"1.2.3.4"}
    assert r.active_allowed_ips() == {"1.2.3.4"}


@pytest.mark.asyncio
async def test_unpinned_ref_keeps_shared_tunnel_open():
    mgr = _manager([FakeProc(["x", _URL_LINE])], FakeClock())
    # A pinned reconnect grant AND an open onboarding invite share the tunnel.
    await mgr.ensure(invite_id="pinned", plan=_plan(), ttl_seconds=3600, allowed_ips=["1.2.3.4"])
    await mgr.ensure(invite_id="open", plan=_plan(), ttl_seconds=3600, allowed_ips=[])
    # The open ref needs first-contact access → the whole tunnel stays open.
    assert mgr.active_allowed_ips() == set()
    # Once the open invite is released, the restriction snaps back to the pin.
    await mgr.release(invite_id="open")
    assert mgr.active_allowed_ips() == {"1.2.3.4"}


@pytest.mark.asyncio
async def test_pinned_ips_union_across_refs():
    mgr = _manager([FakeProc(["x", _URL_LINE])], FakeClock())
    await mgr.ensure(invite_id="a", plan=_plan(), ttl_seconds=3600, allowed_ips=["1.2.3.4"])
    await mgr.ensure(invite_id="b", plan=_plan(), ttl_seconds=3600, allowed_ips=["5.6.7.8/32"])
    assert mgr.active_allowed_ips() == {"1.2.3.4", "5.6.7.8/32"}


@pytest.mark.asyncio
async def test_allowlist_clears_on_teardown():
    mgr = _manager([FakeProc(["x", _URL_LINE])], FakeClock())
    await mgr.ensure(invite_id="a", plan=_plan(), ttl_seconds=3600, allowed_ips=["1.2.3.4"])
    await mgr.release(invite_id="a")  # last ref → teardown
    assert mgr.active_allowed_ips() == set()
    assert r.active_allowed_ips() == set()
