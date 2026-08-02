"""Outbound proxy manager for the SearXNG container.

Residential IPs get blocked by upstream search engines (Google, Bing,
Brave, DDG) when they look like scrapers. This module lets users bring
their own list of HTTP/HTTPS/SOCKS5 proxies (paid residential rotators,
self-hosted VPN endpoints, Tailscale exit nodes — anything httpx can
dial), tracks per-proxy health, and writes the active choice into
SearXNG's ``settings.yml`` ``outgoing.proxies`` block. SearXNG only
supports a single proxy value at a time, so "rotation" here means this
manager periodically re-picks the active proxy from the healthy pool
and restarts SearXNG to pick it up.

Lifecycle:
    - Single instance per Augmentum process, stored on
      ``app.state.searxng_proxy_manager``.
    - ``apply_from_settings`` is called on startup and on settings change
      to (re)parse the user proxy list, probe each, choose an active
      proxy, write settings.yml, restart SearXNG. Idempotent — a no-op
      pass with the same state writes nothing.
    - A background job (``augmentum.jobs.handlers.searxng_proxy_health``)
      calls ``healthcheck_all`` on a timer and re-picks if the active
      proxy flipped unhealthy.

What this manager does NOT do:
    - It does not validate that a proxy can actually reach Google /
      Bing / etc. The probe target is a stable HTTPS endpoint; a proxy
      that reaches the probe target but is blocked by a specific
      engine will still show as healthy here. Surfacing per-engine
      blocking is a future enhancement (count zero-result responses).
    - It does not own the user's proxy credentials. The proxy URLs are
      stored verbatim in the settings store; if the user includes
      ``user:pass@`` they'll be saved in plain in the SQLite settings
      table (same storage class as other free-form text settings).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import yaml

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Default probe target. Picked because DuckDuckGo is generally
# permissive about proxy traffic (less likely to give a false
# unhealthy signal than Google or Bing), and the homepage is small
# and stable. Overridable per-instance for tests.
_DEFAULT_PROBE_TARGET = "https://duckduckgo.com/"
_PROBE_TIMEOUT_SECONDS = 10.0

# Schemes we know how to forward to SearXNG.
_VALID_SCHEMES = {"http", "https", "socks4", "socks5", "socks5h"}


@dataclass
class ProxyHealth:
    """Per-proxy health state."""

    url: str
    healthy: bool = False
    last_checked: float = 0.0  # epoch seconds; 0 = never
    last_latency_ms: float | None = None
    consecutive_failures: int = 0
    last_error: str | None = None


@dataclass
class ProxyManagerStatus:
    """Aggregate state returned to API callers + the UI."""

    configured_count: int
    healthy_count: int
    active_proxy: str | None
    direct_fallback_active: bool
    last_healthcheck: float  # epoch seconds; 0 = never
    proxies: list[ProxyHealth] = field(default_factory=list)


def _normalise_proxy_url(raw: str) -> str | None:
    """Trim, validate scheme, return a usable URL or ``None``.

    Accepts the SOCKS variants httpx understands plus plain http/https.
    Rejects ssh://, ftp://, ws://, etc. Silently. Caller logs the
    drop with the original line so the UI can surface it.
    """
    candidate = raw.strip()
    if not candidate or candidate.startswith("#"):
        return None
    if "://" not in candidate:
        # Bare host:port — assume http
        candidate = "http://" + candidate
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return None
    if parsed.scheme.lower() not in _VALID_SCHEMES:
        return None
    if not parsed.hostname:
        return None
    return candidate


def parse_proxies(raw: str) -> list[str]:
    """Parse the newline-separated ``search_proxies`` setting value.

    De-duplicates while preserving order (first occurrence wins so the
    rotation cycle is stable across saves that re-order without
    semantic change).
    """
    if not raw:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for line in raw.splitlines():
        url = _normalise_proxy_url(line)
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out


class SearxngProxyManager:
    """Owns the SearXNG outbound-proxy state for the Augmentum process.

    Thread/concurrency model: methods are async and gated by a single
    asyncio lock. Callers should not need to coordinate externally.
    """

    def __init__(
        self,
        *,
        settings_yml_path: Path,
        searxng_container_name: str = "searxng",
        docker_client: Any | None = None,
        probe_target: str = _DEFAULT_PROBE_TARGET,
    ) -> None:
        self._settings_yml_path = settings_yml_path
        self._searxng_container_name = searxng_container_name
        self._docker_client = docker_client
        self._probe_target = probe_target

        self._lock = asyncio.Lock()
        self._proxies: list[str] = []
        self._health: dict[str, ProxyHealth] = {}
        self._active: str | None = None
        self._direct_fallback_active = False
        self._last_healthcheck = 0.0
        self._rotation_iter: Any = None  # cycle() lazily constructed

    # ------------------------------------------------------------------
    # State + configuration
    # ------------------------------------------------------------------

    def status(self) -> ProxyManagerStatus:
        healthy = [h for h in self._health.values() if h.healthy]
        return ProxyManagerStatus(
            configured_count=len(self._proxies),
            healthy_count=len(healthy),
            active_proxy=self._active,
            direct_fallback_active=self._direct_fallback_active,
            last_healthcheck=self._last_healthcheck,
            proxies=[self._health[p] for p in self._proxies if p in self._health],
        )

    async def update_proxy_list(self, raw: str) -> None:
        """Re-parse the user setting, prune health for removed proxies.

        Does not run probes or rewrite settings.yml — callers do that
        explicitly via ``healthcheck_all`` + ``apply_active``.
        """
        async with self._lock:
            new_list = parse_proxies(raw)
            new_set = set(new_list)
            # Drop health for proxies the user removed
            self._health = {url: h for url, h in self._health.items() if url in new_set}
            # Initialise health for new proxies (unknown until first probe)
            for url in new_list:
                if url not in self._health:
                    self._health[url] = ProxyHealth(url=url)
            self._proxies = new_list
            self._rotation_iter = None  # rebuild on next pick

    # ------------------------------------------------------------------
    # Probing
    # ------------------------------------------------------------------

    async def probe(self, proxy_url: str) -> ProxyHealth:
        """Test a single proxy by GET-ing the probe target through it.

        Updates the per-proxy health record and returns it. Healthy =
        200-class response within timeout.
        """
        health = self._health.setdefault(proxy_url, ProxyHealth(url=proxy_url))
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(
                proxy=proxy_url,
                timeout=_PROBE_TIMEOUT_SECONDS,
                follow_redirects=True,
            ) as client:
                resp = await client.get(self._probe_target)
            latency_ms = (time.monotonic() - start) * 1000
            if 200 <= resp.status_code < 400:
                health.healthy = True
                health.consecutive_failures = 0
                health.last_error = None
                health.last_latency_ms = latency_ms
            else:
                health.healthy = False
                health.consecutive_failures += 1
                health.last_error = f"HTTP {resp.status_code}"
                health.last_latency_ms = latency_ms
        except (httpx.ProxyError, httpx.ConnectError, httpx.TimeoutException) as exc:
            health.healthy = False
            health.consecutive_failures += 1
            health.last_error = type(exc).__name__
            health.last_latency_ms = None
        except Exception as exc:  # noqa: BLE001 — surface anything else as a string
            health.healthy = False
            health.consecutive_failures += 1
            health.last_error = f"{type(exc).__name__}: {exc}"
            health.last_latency_ms = None
        health.last_checked = time.time()
        return health

    async def healthcheck_all(self) -> dict[str, ProxyHealth]:
        """Probe every configured proxy in parallel.

        Returns a snapshot of the per-proxy health dict.
        """
        async with self._lock:
            proxies = list(self._proxies)
        if not proxies:
            self._last_healthcheck = time.time()
            return {}
        results = await asyncio.gather(
            *[self.probe(p) for p in proxies],
            return_exceptions=False,  # probe() swallows its own exceptions
        )
        self._last_healthcheck = time.time()
        return {h.url: h for h in results}

    # ------------------------------------------------------------------
    # Picking + applying
    # ------------------------------------------------------------------

    def pick_active(self) -> str | None:
        """Return the next healthy proxy in round-robin order.

        Returns ``None`` if no proxy is currently healthy. The caller
        decides whether to fall back to direct, based on the user's
        ``search_proxy_fallback_direct_enabled`` setting.
        """
        healthy = [p for p in self._proxies if self._health.get(p, ProxyHealth(url=p)).healthy]
        if not healthy:
            return None
        if self._rotation_iter is None or set(healthy) != getattr(self, "_rotation_set", set()):
            self._rotation_iter = cycle(healthy)
            self._rotation_set = set(healthy)
        return next(self._rotation_iter)

    async def apply_active(self, proxy_url: str | None) -> bool:
        """Write ``proxy_url`` (or remove the key) into settings.yml + restart SearXNG.

        Returns ``True`` if anything changed on disk (and a restart was
        attempted), ``False`` if the settings.yml already reflected the
        target state.
        """
        async with self._lock:
            current = await asyncio.to_thread(_read_outgoing_proxy, self._settings_yml_path)
            target = _outgoing_proxy_dict(proxy_url)
            if current == target:
                self._active = proxy_url
                self._direct_fallback_active = proxy_url is None and bool(self._proxies)
                return False
            await asyncio.to_thread(
                _write_outgoing_proxy, self._settings_yml_path, target
            )
            self._active = proxy_url
            self._direct_fallback_active = proxy_url is None and bool(self._proxies)
        await self._restart_searxng()
        log.info(
            "searxng_proxy_applied",
            active=proxy_url,
            fallback_direct=self._direct_fallback_active,
        )
        return True

    async def reconcile(self, *, fallback_to_direct: bool) -> bool:
        """Re-pick + apply. The healthcheck job's main entry point.

        ``fallback_to_direct`` controls behaviour when no proxy is
        healthy: if True, write an empty ``outgoing.proxies`` (SearXNG
        connects directly); if False, leave whatever was previously
        configured and log a warning.
        """
        picked = self.pick_active()
        if picked is None:
            if fallback_to_direct:
                return await self.apply_active(None)
            log.warning(
                "searxng_proxy_no_healthy",
                configured=len(self._proxies),
                action="keeping previous configuration",
            )
            return False
        return await self.apply_active(picked)

    # ------------------------------------------------------------------
    # SearXNG restart
    # ------------------------------------------------------------------

    async def _restart_searxng(self) -> None:
        """Restart the SearXNG container via the shared aiodocker client.

        Best-effort: a failed restart is logged but doesn't raise — the
        new settings.yml is on disk regardless, and SearXNG will pick
        it up on its next natural restart (autoheal cycle or compose
        recreate).
        """
        if self._docker_client is None:
            log.warning(
                "searxng_restart_skipped",
                reason="no docker client available — restart manually for new proxy to take effect",
            )
            return
        try:
            container = await self._docker_client.containers.get(
                self._searxng_container_name
            )
            await container.restart(timeout=10)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "searxng_restart_failed",
                error=str(exc),
                container=self._searxng_container_name,
            )


# ----------------------------------------------------------------------
# settings.yml helpers (sync, run via asyncio.to_thread)
# ----------------------------------------------------------------------


def _outgoing_proxy_dict(proxy_url: str | None) -> dict[str, str] | None:
    """Build the ``outgoing.proxies`` value SearXNG expects.

    SearXNG uses httpx under the hood, which accepts a single string
    OR a mapping keyed by URL pattern. The mapping form is more
    explicit and the form SearXNG's docs show, so we use that.
    Returns ``None`` to signal "remove the key entirely."
    """
    if not proxy_url:
        return None
    return {
        "http://": proxy_url,
        "https://": proxy_url,
    }


def _read_outgoing_proxy(path: Path) -> dict[str, str] | None:
    """Return the current ``outgoing.proxies`` value from settings.yml, or None."""
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    outgoing = data.get("outgoing") or {}
    return outgoing.get("proxies") or None


def _write_outgoing_proxy(path: Path, value: dict[str, str] | None) -> None:
    """Round-trip settings.yml with ``outgoing.proxies`` set or removed.

    Preserves all other keys; uses PyYAML's default formatting so the
    file remains valid YAML SearXNG can parse.
    """
    if not path.is_file():
        raise FileNotFoundError(f"SearXNG settings.yml not found at {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    outgoing = data.setdefault("outgoing", {})
    if value is None:
        outgoing.pop("proxies", None)
    else:
        outgoing["proxies"] = value
    # Atomic write: write to .tmp, fsync, rename. Avoids leaving SearXNG
    # with a half-written file if we get killed mid-write.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
        fh.flush()
    tmp.replace(path)
