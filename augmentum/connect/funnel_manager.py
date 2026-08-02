"""Live Tailscale Funnel manager — the OPTIONAL live-drive engine for TS_FUNNEL.

Most deployments run the app in a container that can't reach ``tailscaled``, so
the DEFAULT funnel path is config mode (``FunnelEngine`` in ``reachability.py``):
the operator enables funnel host-side and the app is handed the stable URL via
env. This module is the opt-in (``AUGMENTUM_CONNECT_FUNNEL_LIVE``) alternative
for deployments where the app CAN reach the tailscale CLI/socket (a sidecar or
host-network Linux install): it queries tailscale, picks a free Funnel port
WITHOUT clobbering any existing ``serve``/``funnel`` config, and toggles funnel
on — returning the same stable ``https://<node>.ts.net[:port]`` URL.

Design mirrors ``tunnel_manager.py``:

* **Unit-testable without tailscale.** All policy (ts.net-name + capability
  parsing, free-port selection, idempotent enable, graceful-unavailable) runs
  against an injected ``runner`` + ``clock``. The real ``tailscale`` subprocess
  is a thin adapter used only in production.
* **Standing, not ephemeral.** Unlike the cloudflared quick-tunnel, a Funnel URL
  is stable across restarts and is the guest's ONGOING transport, so it is left
  up (``release`` is a no-op). We add exactly ONE mapping on a verified-free port
  and never ``reset`` the operator's other serve/funnel config.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable

from augmentum.connect.reachability import (
    ALLOWED_FUNNEL_PORTS,
    EngineUnavailable,
    FunnelEngine,
    ReachabilityPlan,
    register_engine,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# The tailnet capability that permits Funnel (public exposure). Present as a key
# in the node's CapMap only when the tailnet ACL grants it.
_FUNNEL_CAP_SUBSTR = "funnel"

# A runner takes tailscale CLI args and returns (returncode, combined_output).
Runner = Callable[[list[str]], Awaitable["tuple[int, str]"]]


async def _default_tailscale_runner(args: list[str]) -> tuple[int, str]:
    """Shell ``tailscale <args>`` and return (returncode, stdout+stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "tailscale", *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    return proc.returncode or 0, (out or b"").decode("utf-8", errors="replace")


class LiveFunnelManager(FunnelEngine):
    """Drives ``tailscale funnel`` when the app can reach the CLI.

    ``ensure`` is idempotent and ref-counted; once funnel is up it returns the
    cached stable URL. It never tears the funnel down per-invite (standing door);
    ``release`` only drops the ref count.
    """

    def __init__(
        self,
        *,
        target_url: str = "http://127.0.0.1:6100",
        runner: Runner | None = None,
        clock: Callable[[], float] = time.monotonic,
        funnel_port: int | str | None = None,
    ) -> None:
        self._target_url = target_url
        self._runner = runner or _default_tailscale_runner
        self._clock = clock
        self._preferred_port = int(funnel_port) if str(funnel_port or "").strip().isdigit() else None
        self._url: str = ""
        self._port: int | None = None
        self._refs: set[str] = set()
        self._lock = asyncio.Lock()

    # ReachabilityEngine surface ------------------------------------------------

    async def ensure(
        self, *, invite_id: str, plan: ReachabilityPlan, ttl_seconds: int,
        allowed_ips: list | None = None,
    ) -> str:
        async with self._lock:
            self._refs.add(invite_id)
            if self._url:
                return self._url
            self._url = await self._bring_up()
            log.info("funnel_up", url=self._url, port=self._port)
            return self._url

    async def release(self, *, invite_id: str) -> None:
        # Standing door — drop the ref but leave funnel up (durable, stable URL).
        async with self._lock:
            self._refs.discard(invite_id)

    # internals -----------------------------------------------------------------

    async def _run(self, args: list[str]) -> tuple[int, str]:
        """Run the tailscale CLI; ANY failure becomes EngineUnavailable so the
        planner degrades to cloudflared rather than 500-ing the mint."""
        try:
            return await self._runner(args)
        except EngineUnavailable:
            raise
        except Exception as exc:
            raise EngineUnavailable(f"tailscale {' '.join(args)}: {exc}") from exc

    async def _tailscale_json(self, args: list[str]) -> dict:
        rc, out = await self._run(args)
        if rc != 0:
            raise EngineUnavailable(f"tailscale {' '.join(args)}: rc={rc} {out[:160]}")
        try:
            data = json.loads(out or "{}")
        except Exception as exc:
            raise EngineUnavailable(f"tailscale {' '.join(args)}: bad json") from exc
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _funnel_capable(status: dict) -> bool:
        """True when the tailnet ACL grants this node the funnel capability."""
        self_node = status.get("Self") or {}
        capmap = self_node.get("CapMap") or {}
        caps = self_node.get("Capabilities") or []
        keys = list(capmap.keys()) + list(caps)
        return any(_FUNNEL_CAP_SUBSTR in str(k).lower() for k in keys)

    def _pick_port(self, occupied: set[int]) -> int | None:
        """A free allowed Funnel port that won't clobber existing config.

        Honours an operator-preferred port when free; otherwise the first free
        one from the allowed set (443/8443/10000). Returns None when none free.
        """
        candidates: tuple[int, ...]
        if self._preferred_port:
            if self._preferred_port in occupied:
                return None  # explicit choice is taken — don't silently retarget
            candidates = (self._preferred_port,)
        else:
            candidates = ALLOWED_FUNNEL_PORTS
        for p in candidates:
            if p not in occupied:
                return p
        return None

    async def _bring_up(self) -> str:
        status = await self._tailscale_json(["status", "--json"])
        dnsname = str((status.get("Self") or {}).get("DNSName") or "").rstrip(".").strip()
        if not dnsname:
            raise EngineUnavailable("funnel: node has no ts.net name")
        if not self._funnel_capable(status):
            raise EngineUnavailable(
                "funnel: not permitted by tailnet ACL — enable the funnel nodeAttr "
                "in the Tailscale admin console",
            )
        serve = await self._tailscale_json(["serve", "status", "--json"])
        occupied = {int(p) for p in (serve.get("TCP") or {}) if str(p).isdigit()}
        port = self._pick_port(occupied)
        if port is None:
            raise EngineUnavailable("funnel: no free funnel port (443/8443/10000 all in use)")

        # Enable funnel on exactly this one free port → our local ingress. We do
        # NOT touch any other serve/funnel mapping (no reset).
        rc, out = await self._run(
            ["funnel", "--bg", f"--https={port}", self._target_url],
        )
        if rc != 0:
            raise EngineUnavailable(f"funnel enable failed (port {port}): {out[:200]}")
        self._port = port
        return f"https://{dnsname}" if port == 443 else f"https://{dnsname}:{port}"


# ── Idempotent live-engine registration ────────────────────────────────────

_manager: LiveFunnelManager | None = None


def get_or_register_funnel_manager(
    *, target_url: str = "http://127.0.0.1:6100", funnel_port: int | str | None = None,
) -> LiveFunnelManager:
    """Construct the live funnel manager once and register it as the TS_FUNNEL
    engine (replacing the config-mode ``FunnelEngine``). Idempotent."""
    global _manager
    if _manager is None:
        _manager = LiveFunnelManager(target_url=target_url, funnel_port=funnel_port)
        register_engine(_manager)
        log.info("funnel_manager_registered", target=target_url)
    return _manager
