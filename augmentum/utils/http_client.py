"""Shared httpx client factory with local-URL detection.

Provides a reusable async-context-manager that lazily creates and caches
two ``httpx.AsyncClient`` instances — one with SSL verification (for cloud
APIs) and one without (for local / self-hosted services).  The decision is
driven by ``is_local_url()``, which both ``audio_routes`` and
``cloud_image_routes`` previously duplicated inline.
"""

from __future__ import annotations

import asyncio
import contextlib

import httpx

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Default timeout shared across all callers — generous read window for
# streaming audio and large image payloads.
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=5.0)

_LOCAL_HOSTS = (
    "localhost",
    "127.0.0.1",
    "0.0.0.0",  # noqa: S104 — detection only, not binding
    "[::1]",
    "host.docker.internal",
    ".local:",
)


def normalize_base_url(url: str) -> str:
    """Strip trailing ``/v1`` (or ``/v1/``) and trailing slashes from *url*.

    Many backends hardcode ``/v1/`` in their request paths.  If a user
    configures a base URL that already includes ``/v1`` (common for
    OpenAI-compatible APIs), the result is a double ``/v1/v1/`` in the
    final URL.  Call this **before** appending any ``/v1/…`` path.
    """
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def is_local_url(url: str) -> bool:
    """Return ``True`` if *url* points to a local / self-hosted service.

    Used to decide whether SSL verification should be skipped.
    """
    lower = url.lower()
    return any(h in lower for h in _LOCAL_HOSTS)


class SharedHTTPClient:
    """Lazily-created, lock-protected pair of httpx async clients.

    Each logical subsystem (audio, cloud-image, etc.) should instantiate its
    own ``SharedHTTPClient`` so that client lifetimes stay independent while
    the creation logic is shared.
    """

    def __init__(self, timeout: httpx.Timeout | None = None) -> None:
        self._timeout = timeout or DEFAULT_TIMEOUT
        self._verified: httpx.AsyncClient | None = None
        self._unverified: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def get(self, base_url: str = ""):
        """Yield a cached ``httpx.AsyncClient``.

        SSL verification is disabled only when *base_url* points to a local
        service (as determined by :func:`is_local_url`).
        """
        async with self._lock:
            if is_local_url(base_url):
                if self._unverified is None or self._unverified.is_closed:
                    self._unverified = httpx.AsyncClient(
                        verify=False,  # noqa: S501
                        timeout=self._timeout,
                        follow_redirects=True,
                    )
                client = self._unverified
            else:
                if self._verified is None or self._verified.is_closed:
                    self._verified = httpx.AsyncClient(
                        verify=True,
                        timeout=self._timeout,
                        follow_redirects=True,
                    )
                client = self._verified

        yield client

    async def close(self) -> None:
        """Close both cached clients. Called during server shutdown."""
        for client in (self._verified, self._unverified):
            if client is not None and not client.is_closed:
                await client.aclose()
        self._verified = None
        self._unverified = None
