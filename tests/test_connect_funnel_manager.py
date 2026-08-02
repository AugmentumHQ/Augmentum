"""LiveFunnelManager policy — driven entirely by a fake tailscale runner.

The live-drive funnel engine only activates where the app can reach the
tailscale CLI (a sidecar / host-network deploy). These tests pin the policy
without any tailscale present:
  - the node's ts.net name + funnel capability are read from `status --json`,
  - a FREE allowed funnel port is chosen from `serve status --json` (never
    clobbering existing config),
  - enable is idempotent, and every unavailable condition degrades to
    `EngineUnavailable` so the planner falls to cloudflared.

Config-mode `FunnelEngine` (the standing, no-subprocess default) is covered in
test_connect_reachability.py.
"""
from __future__ import annotations

import json

import pytest

from augmentum.connect.funnel_manager import LiveFunnelManager
from augmentum.connect.reachability import (
    EngineUnavailable,
    ReachabilityPlan,
    ReachTier,
    RecipientScope,
)

_CAP = "https://tailscale.com/cap/funnel"


def _plan():
    return ReachabilityPlan(scope=RecipientScope.PUBLIC, tier=ReachTier.TS_FUNNEL, host="")


class FakeRunner:
    """Scripted tailscale runner. `status`/`serve` return canned JSON; `funnel`
    records the call and returns success unless `funnel_rc` is set."""

    def __init__(self, *, dnsname="node.tail1234.ts.net.", funnel_cap=True,
                 serve_tcp=("443",), funnel_rc=0, status_rc=0):
        self.dnsname = dnsname
        self.funnel_cap = funnel_cap
        self.serve_tcp = serve_tcp
        self.funnel_rc = funnel_rc
        self.status_rc = status_rc
        self.calls: list[list[str]] = []

    async def __call__(self, args):
        self.calls.append(list(args))
        if args[:1] == ["status"]:
            self_node = {"DNSName": self.dnsname}
            if self.funnel_cap:
                self_node["CapMap"] = {_CAP: None}
            return self.status_rc, json.dumps({"Self": self_node})
        if args[:2] == ["serve", "status"]:
            return 0, json.dumps({"TCP": {p: {"HTTPS": True} for p in self.serve_tcp}})
        if args[:1] == ["funnel"]:
            return self.funnel_rc, "" if self.funnel_rc == 0 else "funnel error"
        return 0, "{}"

    def funnel_calls(self):
        return [c for c in self.calls if c[:1] == ["funnel"]]


async def _ensure(mgr, invite_id="ref1"):
    return await mgr.ensure(invite_id=invite_id, plan=_plan(), ttl_seconds=3600)


# ── happy path + port selection ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_enables_on_free_port_and_returns_stable_url():
    runner = FakeRunner(serve_tcp=("443",))  # 443 busy → picks 8443
    mgr = LiveFunnelManager(runner=runner)
    url = await _ensure(mgr)
    assert url == "https://node.tail1234.ts.net:8443"
    fc = runner.funnel_calls()
    assert len(fc) == 1
    assert "--https=8443" in fc[0]
    assert "http://127.0.0.1:6100" in fc[0]


@pytest.mark.asyncio
async def test_prefers_443_when_free_no_port_suffix():
    runner = FakeRunner(serve_tcp=())  # nothing occupied → 443, URL has no :port
    url = await _ensure(LiveFunnelManager(runner=runner))
    assert url == "https://node.tail1234.ts.net"
    assert "--https=443" in runner.funnel_calls()[0]


@pytest.mark.asyncio
async def test_skips_to_10000_when_443_and_8443_busy():
    runner = FakeRunner(serve_tcp=("443", "8443"))
    url = await _ensure(LiveFunnelManager(runner=runner))
    assert url == "https://node.tail1234.ts.net:10000"


@pytest.mark.asyncio
async def test_no_free_port_is_unavailable():
    runner = FakeRunner(serve_tcp=("443", "8443", "10000"))
    with pytest.raises(EngineUnavailable):
        await _ensure(LiveFunnelManager(runner=runner))


@pytest.mark.asyncio
async def test_preferred_port_used_when_free():
    runner = FakeRunner(serve_tcp=("443",))
    url = await _ensure(LiveFunnelManager(runner=runner, funnel_port=10000))
    assert url == "https://node.tail1234.ts.net:10000"


@pytest.mark.asyncio
async def test_preferred_port_taken_is_unavailable_not_silent_retarget():
    # An explicit port choice that's occupied must NOT silently pick another.
    runner = FakeRunner(serve_tcp=("8443",))
    with pytest.raises(EngineUnavailable):
        await _ensure(LiveFunnelManager(runner=runner, funnel_port=8443))


# ── idempotence + standing door ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ensure_is_idempotent():
    runner = FakeRunner(serve_tcp=("443",))
    mgr = LiveFunnelManager(runner=runner)
    url1 = await _ensure(mgr, "a")
    url2 = await _ensure(mgr, "b")
    assert url1 == url2
    # Only ONE funnel enable across two ensures.
    assert len(runner.funnel_calls()) == 1


@pytest.mark.asyncio
async def test_release_is_a_noop_standing_door():
    runner = FakeRunner(serve_tcp=("443",))
    mgr = LiveFunnelManager(runner=runner)
    await _ensure(mgr, "a")
    await mgr.release(invite_id="a")
    # Never issues a `funnel off` / `reset` — the door stays up (durable).
    assert all(c[:1] != ["funnel"] or "--https" in " ".join(c) for c in runner.calls)
    assert not any("off" in c or "reset" in c for c in runner.calls)


@pytest.mark.asyncio
async def test_never_touches_other_serve_ports():
    runner = FakeRunner(serve_tcp=("443",))
    await _ensure(LiveFunnelManager(runner=runner))
    # The only mutating call is funnel on our chosen free port; no reset/off.
    muts = [c for c in runner.calls if c[:1] == ["funnel"]]
    assert len(muts) == 1 and "--https=8443" in muts[0]


# ── graceful unavailability ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_funnel_cap_absent_is_unavailable():
    runner = FakeRunner(funnel_cap=False)
    with pytest.raises(EngineUnavailable):
        await _ensure(LiveFunnelManager(runner=runner))
    assert not runner.funnel_calls()  # never attempted enable


@pytest.mark.asyncio
async def test_no_dnsname_is_unavailable():
    runner = FakeRunner(dnsname="")
    with pytest.raises(EngineUnavailable):
        await _ensure(LiveFunnelManager(runner=runner))


@pytest.mark.asyncio
async def test_enable_failure_is_unavailable():
    runner = FakeRunner(serve_tcp=("443",), funnel_rc=1)
    with pytest.raises(EngineUnavailable):
        await _ensure(LiveFunnelManager(runner=runner))


@pytest.mark.asyncio
async def test_runner_raising_is_unavailable():
    # A raising runner (tailscaled unreachable) must surface as EngineUnavailable
    # so _build_invite_link's fallback catches it and degrades to cloudflared.
    async def boom(_args):
        raise RuntimeError("tailscaled unreachable")
    with pytest.raises(EngineUnavailable):
        await _ensure(LiveFunnelManager(runner=boom))
