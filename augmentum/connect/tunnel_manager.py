"""Live cloudflared tunnel manager — ephemeral public exposure for invites.

The reachability *bones* (``reachability.py``) decide WHEN a public tunnel is
the right tier; this module is the live engine that actually stands one up and —
crucially — tears it down the moment the invite is consumed or expires.

Design goals (kept deliberately):

* **Unit-testable without the binary.** All lifecycle *policy* (ref-counting,
  URL capture, capture timeout, teardown, TTL reaping) runs against an injected
  ``launcher`` + ``clock``. The real ``cloudflared`` subprocess is a thin
  adapter used only in production. Tests drive a fake process and a fake clock.
* **Low-tech-user-friendly.** cloudflared is auto-detected (no config); the user
  only ever picks "Anywhere" on one invite. One shared tunnel is ref-counted
  across overlapping invites, so a household inviting two people spawns one
  process, and the last invite to finish closes it.
* **Path-scoped.** The tunnel is spawned with a fixed sentinel
  ``--http-host-header`` so every tunneled request arrives at the app under
  :data:`augmentum.connect.reachability.INVITE_TUNNEL_HOST`; the middleware
  guard then 404s anything but the invite door. Defense-in-depth on top of the
  normal auth middleware.

See ``docs/superpowers/specs/2026-06-20-connect-comms-platform-design.md`` (P3).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Protocol

from augmentum.connect.reachability import (
    INVITE_TUNNEL_HOST,
    CloudflaredEngine,
    EngineUnavailable,
    ReachabilityPlan,
    ReachTier,
    set_active_hosts_provider,
    set_active_ip_provider,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# How long to wait for cloudflared to print its assigned URL before giving up.
URL_CAPTURE_TIMEOUT_S = 30.0
# Hard ceiling on a single tunnel's life regardless of invite TTLs — a backstop
# so a leaked ref can't hold a public door open forever.
MAX_TUNNEL_LIFETIME_S = 7 * 24 * 3600


class TunnelProcess(Protocol):
    """Minimal process surface the manager needs (real or fake)."""

    async def readline(self) -> str:
        """Next line of output; ``""`` at EOF."""
        ...

    def terminate(self) -> None:
        ...

    @property
    def returncode(self) -> int | None:
        ...


# A launcher takes the local target URL and returns a started TunnelProcess.
Launcher = Callable[[str], Awaitable[TunnelProcess]]


class CloudflaredTunnelManager(CloudflaredEngine):
    """Ref-counted, self-tearing cloudflared engine (CLOUDFLARED tier).

    One shared quick-tunnel serves all active "Anywhere" invites; ``ensure``
    adds a ref (with a TTL), ``release`` drops one, and the tunnel is torn down
    when the last ref goes (or its TTL lapses via ``reap``). ``active_public_
    hosts`` reports the sentinel host while a tunnel is up so the middleware
    guard can path-scope it.
    """

    tier = ReachTier.CLOUDFLARED

    def __init__(
        self,
        *,
        target_url: str = "http://127.0.0.1:6100",
        launcher: Launcher | None = None,
        clock: Callable[[], float] = time.monotonic,
        url_timeout_s: float = URL_CAPTURE_TIMEOUT_S,
        reap_interval_s: float = 60.0,
    ) -> None:
        self._target_url = target_url
        self._launcher = launcher or _spawn_cloudflared
        self._clock = clock
        self._url_timeout_s = url_timeout_s
        # 0 disables the background reaper (tests reap explicitly).
        self._reap_interval_s = reap_interval_s
        self._reaper_task: asyncio.Task | None = None
        self._proc: TunnelProcess | None = None
        self._url: str = ""
        self._started_at: float = 0.0
        # invite_id → monotonic expiry deadline
        self._refs: dict[str, float] = {}
        # invite_id → allowed IPs/CIDRs ([] = this ref imposes NO IP restriction)
        self._ip_refs: dict[str, list[str]] = {}
        self._lock = asyncio.Lock()

    # ReachabilityEngine surface ------------------------------------------------

    async def ensure(
        self, *, invite_id: str, plan: ReachabilityPlan, ttl_seconds: int,
        allowed_ips: list[str] | None = None,
    ) -> str:
        async with self._lock:
            now = self._clock()
            ttl = min(max(1, int(ttl_seconds)), MAX_TUNNEL_LIFETIME_S)
            self._refs[invite_id] = now + ttl
            self._ip_refs[invite_id] = list(allowed_ips or [])
            if self._url:
                self._publish_hosts()  # refresh the IP allowlist union
                return self._url
            try:
                proc = await self._launcher(self._target_url)
            except Exception as exc:  # binary missing / spawn failure
                self._refs.pop(invite_id, None)
                raise EngineUnavailable(f"cloudflared spawn failed: {exc}") from exc
            url = await self._capture_url(proc)
            if not url:
                self._safe_terminate(proc)
                self._refs.pop(invite_id, None)
                raise EngineUnavailable("cloudflared: no URL captured before timeout")
            self._proc = proc
            self._url = url
            self._started_at = now
            self._publish_hosts()
            self._ensure_reaper()
            log.info("cloudflared_tunnel_up", url=url, refs=len(self._refs))
            return url

    async def release(self, *, invite_id: str) -> None:
        async with self._lock:
            self._refs.pop(invite_id, None)
            self._ip_refs.pop(invite_id, None)
            if not self._refs:
                await self._teardown_locked("last_ref_released")
            else:
                self._publish_hosts()  # the IP-allowlist union may have changed

    async def reap(self, *, now: float | None = None) -> None:
        """Drop expired refs and tear the tunnel down if none remain (or the
        hard lifetime ceiling is hit). Call periodically from a background loop."""
        async with self._lock:
            t = self._clock() if now is None else now
            expired = [k for k, exp in self._refs.items() if exp <= t]
            for k in expired:
                self._refs.pop(k, None)
                self._ip_refs.pop(k, None)
            lifetime_exceeded = bool(self._url) and (t - self._started_at) >= MAX_TUNNEL_LIFETIME_S
            if (not self._refs or lifetime_exceeded) and (self._proc or self._url):
                await self._teardown_locked("expired" if not self._refs else "lifetime_ceiling")
            elif expired:
                self._publish_hosts()  # allowlist union shrank

    def active_public_hosts(self) -> set[str]:
        return {INVITE_TUNNEL_HOST} if self._url else set()

    def active_allowed_ips(self) -> set[str]:
        """Union of allowed IPs across active refs — EMPTY when any active ref
        is unpinned (an open onboarding invite needs first-contact access, so
        the whole shared tunnel stays open until only pinned refs remain)."""
        if not self._url:
            return set()
        lists = [self._ip_refs.get(k, []) for k in self._refs]
        if any(not lst for lst in lists):
            return set()
        out: set[str] = set()
        for lst in lists:
            out.update(lst)
        return out

    # internals -----------------------------------------------------------------

    async def _capture_url(self, proc: TunnelProcess) -> str:
        async def _read_loop() -> str:
            while True:
                line = await proc.readline()
                if not line:  # EOF — process exited without a URL
                    return ""
                url = CloudflaredEngine.parse_url(line)
                if url:
                    return url

        try:
            return await asyncio.wait_for(_read_loop(), timeout=self._url_timeout_s)
        except TimeoutError:
            return ""

    async def _teardown_locked(self, reason: str) -> None:
        if self._proc is not None:
            self._safe_terminate(self._proc)
        self._proc = None
        self._url = ""
        self._started_at = 0.0
        self._refs.clear()
        self._ip_refs.clear()
        self._publish_hosts()
        log.info("cloudflared_tunnel_down", reason=reason)

    @staticmethod
    def _safe_terminate(proc: TunnelProcess) -> None:
        try:
            proc.terminate()
        except Exception:
            log.warning("cloudflared_terminate_failed", exc_info=True)

    def _publish_hosts(self) -> None:
        set_active_hosts_provider(self.active_public_hosts)
        set_active_ip_provider(self.active_allowed_ips)

    def _ensure_reaper(self) -> None:
        """Start the background TTL reaper if enabled and not already running."""
        if self._reap_interval_s <= 0:
            return
        if self._reaper_task is None or self._reaper_task.done():
            self._reaper_task = asyncio.create_task(self._reap_loop())

    async def _reap_loop(self) -> None:
        """Periodically reap; self-terminates once nothing is active (so an idle
        host carries no lingering task)."""
        while True:
            await asyncio.sleep(self._reap_interval_s)
            try:
                await self.reap()
            except Exception:  # pragma: no cover - reaper must never crash the loop
                log.warning("cloudflared_reap_failed", exc_info=True)
            async with self._lock:
                if not self._refs and not self._url:
                    return


# ── Real cloudflared subprocess adapter (thin; exercised live, not in CI) ───

class _AsyncProcAdapter:
    """Wraps an asyncio subprocess to the TunnelProcess protocol."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc

    async def readline(self) -> str:
        if self._proc.stdout is None:
            return ""
        raw = await self._proc.stdout.readline()
        return raw.decode("utf-8", errors="replace") if raw else ""

    def terminate(self) -> None:
        if self._proc.returncode is None:
            self._proc.terminate()

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode


async def _spawn_cloudflared(target_url: str) -> TunnelProcess:
    """Start ``cloudflared`` for a throwaway quick-tunnel at ``target_url``.

    ``--http-host-header`` pins every tunneled request to the sentinel host so
    the middleware guard can path-scope the exposure. ``--no-autoupdate`` keeps
    it quiet. stderr is merged into stdout so the URL banner is captured from
    one stream.
    """
    proc = await asyncio.create_subprocess_exec(
        "cloudflared", "tunnel",
        "--no-autoupdate",
        "--http-host-header", INVITE_TUNNEL_HOST,
        "--url", target_url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    return _AsyncProcAdapter(proc)


# ── Idempotent live-engine registration ────────────────────────────────────

_manager: CloudflaredTunnelManager | None = None


def get_or_register_cloudflared_manager(
    *, target_url: str = "http://127.0.0.1:6100",
) -> CloudflaredTunnelManager:
    """Construct the live cloudflared manager once and register it as the
    CLOUDFLARED engine (replacing the bones stub). Idempotent."""
    global _manager
    if _manager is None:
        from augmentum.connect.reachability import register_engine

        _manager = CloudflaredTunnelManager(target_url=target_url)
        register_engine(_manager)
        log.info("cloudflared_manager_registered", target=target_url)
    return _manager
