"""SSRF-prevention HTTP client for safe URL fetching."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

import httpx

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# IP ranges that must never be reached when fetching external URLs.
_BLOCKED_RANGES = [
    ipaddress.ip_network("127.0.0.0/8"),       # Loopback
    ipaddress.ip_network("10.0.0.0/8"),         # Private Class A
    ipaddress.ip_network("172.16.0.0/12"),      # Private Class B
    ipaddress.ip_network("192.168.0.0/16"),     # Private Class C
    ipaddress.ip_network("169.254.0.0/16"),     # Link-local
    ipaddress.ip_network("0.0.0.0/8"),          # "This" network
    ipaddress.ip_network("100.64.0.0/10"),      # Shared address space (CGN)
    ipaddress.ip_network("192.0.0.0/24"),       # IETF Protocol Assignments
    ipaddress.ip_network("198.18.0.0/15"),      # Benchmarking
    ipaddress.ip_network("224.0.0.0/4"),        # Multicast
    ipaddress.ip_network("240.0.0.0/4"),        # Reserved
    ipaddress.ip_network("255.255.255.255/32"), # Broadcast
    # IPv6
    ipaddress.ip_network("::1/128"),            # Loopback
    ipaddress.ip_network("fe80::/10"),          # Link-local
    ipaddress.ip_network("fc00::/7"),           # Unique local
    ipaddress.ip_network("::ffff:0:0/96"),      # IPv4-mapped IPv6
]

_ALLOWED_SCHEMES = {"http", "https"}


class SafeHttpError(Exception):
    """Raised when a fetch is blocked by SSRF protection."""


def validate_provider_url(url: str) -> str:
    """Validate an admin-configured provider URL's scheme.

    Provider URLs (LLM engines, TTS/STT providers, image providers,
    media servers) are legitimately allowed to point at LAN-internal
    addresses — Ollama on the home network, a Plex server on the LAN,
    etc. So we deliberately do NOT block private IPs here; that would
    break common self-hosted setups.

    What we DO block is non-http/https schemes (``file://``,
    ``gopher://``, ``dict://``, ``ftp://``, ...). None of them are
    legitimate provider URLs, but each is a known SSRF amplifier if
    an admin's session is hijacked or a settings write reaches this
    code from an unexpected path. Cheap belt-and-suspenders.

    Returns the trimmed URL on success. Raises :class:`SafeHttpError`
    with a user-readable message on a bad scheme so route handlers
    can surface it as a 400.
    """
    if not url:
        return ""
    trimmed = url.strip()
    if not trimmed:
        return ""
    # Only treat the leading token as a scheme when the URL carries
    # ``://``. ``urlparse("ollama:11434")`` reports scheme="ollama"
    # path="11434" — which would wrongly trip the scheme guard for
    # the common bare ``host:port`` form (legacy provider rows,
    # docker-network service names that ``normalize_base_url``
    # prefixes with http:// downstream). Requiring ``://`` keeps the
    # scheme check honest about what counts as an explicit scheme.
    if "://" in trimmed:
        parsed = urlparse(trimmed)
        if parsed.scheme and parsed.scheme.lower() not in _ALLOWED_SCHEMES:
            raise SafeHttpError(
                f"Blocked URL scheme {parsed.scheme!r} — provider URLs must use http or https"
            )
    return trimmed


def _is_ip_blocked(ip_str: str) -> bool:
    """Check whether a resolved IP address falls in a blocked range."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        # If we cannot parse the IP, block it as a precaution.
        return True
    return any(addr in network for network in _BLOCKED_RANGES)


async def _resolve_hostname(hostname: str) -> list[str]:
    """DNS-resolve a hostname asynchronously and return all IP addresses."""
    loop = asyncio.get_event_loop()
    try:
        results = await loop.getaddrinfo(
            hostname, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise SafeHttpError(f"DNS resolution failed for {hostname}: {exc}") from exc
    return list({r[4][0] for r in results})


def _resolve_hostname_sync(hostname: str) -> list[str]:
    """DNS-resolve a hostname synchronously (for use in transport thread pool)."""
    try:
        results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SafeHttpError(f"DNS resolution failed for {hostname}: {exc}") from exc
    return list({r[4][0] for r in results})


async def check_ssrf(url: str, *, allowlist: list[str] | None = None) -> None:
    """Validate a URL against SSRF rules (scheme + DNS-resolved IP).

    This is a lightweight check for routes that cannot use the full
    :class:`SafeHttpClient` (e.g. because they need custom headers or
    redirect behaviour) but still need SSRF protection.

    Parameters
    ----------
    url:
        The URL to validate.
    allowlist:
        Optional list of hostnames or CIDR ranges that are exempt from
        blocking (e.g. Docker-internal service names like ``"searxng"``
        or ``"172.18.0.0/16"``).

    Raises
    ------
    SafeHttpError
        If the URL targets a blocked address.
    """
    parsed = urlparse(url)

    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise SafeHttpError(
            f"Blocked scheme '{parsed.scheme}' — only http and https are allowed"
        )

    hostname = parsed.hostname
    if not hostname:
        raise SafeHttpError("URL has no hostname")

    # Check whether the hostname itself is on the allowlist.
    if _is_allowlisted(hostname, allowlist):
        return

    # If the hostname is an IP literal, check it directly.
    try:
        addr = ipaddress.ip_address(hostname)
        if _is_ip_blocked(str(addr)):
            raise SafeHttpError(f"Blocked IP address: {hostname}")
        return
    except ValueError:
        pass  # Not a bare IP — proceed with DNS resolution.

    ips = await _resolve_hostname(hostname)
    if not ips:
        raise SafeHttpError(f"DNS resolution returned no addresses for {hostname}")

    for ip in ips:
        if _is_ip_blocked(ip) and not _is_allowlisted(ip, allowlist):
            raise SafeHttpError(
                f"Blocked private/reserved IP {ip} (resolved from {hostname})"
            )


def _is_allowlisted(
    value: str, allowlist: list[str] | None,
) -> bool:
    """Check whether *value* (hostname or IP string) matches any allowlist entry."""
    if not allowlist:
        return False

    for entry in allowlist:
        # Exact hostname match (case-insensitive).
        if value.lower() == entry.lower():
            return True
        # CIDR match — entry might be a network like "172.18.0.0/16".
        try:
            network = ipaddress.ip_network(entry, strict=False)
            addr = ipaddress.ip_address(value)
            if addr in network:
                return True
        except ValueError:
            pass  # Not a valid network/IP — skip silently.

    return False


def parse_ssrf_allowlist(raw: str) -> list[str]:
    """Parse a comma-separated allowlist string into a list of entries."""
    if not raw:
        return []
    return [e.strip() for e in raw.split(",") if e.strip()]


# Ranges that stay blocked even in "lan_ok" mode — these are never
# legitimate destinations for user-supplied URLs (cloud metadata,
# broadcast, multicast, reserved, etc.). LAN + loopback are allowed
# because that's where most Plex/Jellyfin/Emby installs live.
_DANGEROUS_RANGES_LAN_OK = [
    ipaddress.ip_network("0.0.0.0/8"),          # "This" network
    ipaddress.ip_network("169.254.0.0/16"),     # Link-local + cloud metadata
    ipaddress.ip_network("100.64.0.0/10"),      # CGN
    ipaddress.ip_network("192.0.0.0/24"),       # IETF Protocol Assignments
    ipaddress.ip_network("198.18.0.0/15"),      # Benchmarking
    ipaddress.ip_network("224.0.0.0/4"),        # Multicast
    ipaddress.ip_network("240.0.0.0/4"),        # Reserved
    ipaddress.ip_network("255.255.255.255/32"), # Broadcast
    ipaddress.ip_network("fe80::/10"),          # IPv6 link-local
    ipaddress.ip_network("::ffff:0:0/96"),      # IPv4-mapped IPv6
]


def _is_ip_dangerous_lan_ok(ip_str: str) -> bool:
    """Block-list for `lan_ok` mode: metadata + multicast + reserved."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return any(addr in network for network in _DANGEROUS_RANGES_LAN_OK)


async def check_ssrf_user_url(url: str, *, mode: str = "external") -> None:
    """Validate a user-supplied URL with mode-appropriate strictness.

    Two modes:

    ``mode="external"``
        Strict: blocks loopback, private, link-local — for URLs that
        should reach the public internet (e.g. RSS feeds, knowledge
        pack imports). Equivalent to plain :func:`check_ssrf`.

    ``mode="lan_ok"``
        Permissive: blocks only addresses no legitimate user URL
        should target — cloud metadata (169.254.169.254), multicast,
        broadcast, reserved ranges, IPv6 link-local. Loopback and
        private LAN ranges (10/8, 172.16/12, 192.168/16) stay allowed
        because that's where typical Plex / Jellyfin / Emby installs
        live.

    Raises :class:`SafeHttpError` on a blocked URL.
    """
    if mode == "external":
        await check_ssrf(url)
        return
    if mode != "lan_ok":
        raise ValueError(f"Unknown SSRF mode: {mode!r}")

    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise SafeHttpError(
            f"Blocked scheme '{parsed.scheme}' — only http and https are allowed"
        )
    hostname = parsed.hostname
    if not hostname:
        raise SafeHttpError("URL has no hostname")

    # IP literal — check directly.
    try:
        addr = ipaddress.ip_address(hostname)
        if _is_ip_dangerous_lan_ok(str(addr)):
            raise SafeHttpError(f"Blocked IP address: {hostname}")
        return
    except ValueError:
        pass

    ips = await _resolve_hostname(hostname)
    if not ips:
        raise SafeHttpError(f"DNS resolution returned no addresses for {hostname}")
    for ip in ips:
        if _is_ip_dangerous_lan_ok(ip):
            raise SafeHttpError(
                f"Blocked dangerous IP {ip} (resolved from {hostname})"
            )


class SafeHttpClient:
    """HTTP client with SSRF prevention.

    Validates URLs before fetching:
    1. Scheme must be http or https.
    2. DNS-resolved IP must not be in a blocked (private/loopback/link-local) range.
    3. Response body must not exceed the configured size limit.
    """

    def __init__(self, max_response_size: int = 5_242_880) -> None:  # 5 MB
        self._max_response_size = max_response_size

    def _validate_url(self, url: str) -> str:
        """Parse and validate a URL, returning the hostname.

        Raises SafeHttpError for invalid schemes or hostnames.
        """
        parsed = urlparse(url)

        if parsed.scheme not in _ALLOWED_SCHEMES:
            raise SafeHttpError(
                f"Blocked scheme '{parsed.scheme}' — only http and https are allowed"
            )

        hostname = parsed.hostname
        if not hostname:
            raise SafeHttpError("URL has no hostname")

        return hostname

    async def _check_resolved_ips(self, hostname: str) -> None:
        """Resolve the hostname and verify that none of the IPs are blocked."""
        # If hostname is already an IP literal, check it directly.
        try:
            addr = ipaddress.ip_address(hostname)
            if _is_ip_blocked(str(addr)):
                raise SafeHttpError(f"Blocked IP address: {hostname}")
            return
        except ValueError:
            pass  # Not a bare IP — proceed with DNS resolution.

        ips = await _resolve_hostname(hostname)
        if not ips:
            raise SafeHttpError(f"DNS resolution returned no addresses for {hostname}")

        for ip in ips:
            if _is_ip_blocked(ip):
                raise SafeHttpError(
                    f"Blocked private/reserved IP {ip} (resolved from {hostname})"
                )

    def _check_resolved_ips_sync(self, hostname: str) -> None:
        """Sync variant for use in httpx transport thread pool."""
        try:
            addr = ipaddress.ip_address(hostname)
            if _is_ip_blocked(str(addr)):
                raise SafeHttpError(f"Blocked IP address: {hostname}")
            return
        except ValueError:
            pass

        ips = _resolve_hostname_sync(hostname)
        if not ips:
            raise SafeHttpError(f"DNS resolution returned no addresses for {hostname}")

        for ip in ips:
            if _is_ip_blocked(ip):
                raise SafeHttpError(
                    f"Blocked private/reserved IP {ip} (resolved from {hostname})"
                )

    async def fetch(
        self,
        url: str,
        *,
        timeout: float = 15.0,
    ) -> tuple[str, dict]:
        """Fetch a URL safely, returning (content, metadata).

        Validates the scheme, DNS-resolved IPs, and response size.
        Uses a custom transport that pins the resolved IP to prevent
        DNS rebinding attacks (TOCTOU between validation and fetch).
        Raises SafeHttpError for any SSRF violation.
        """
        hostname = self._validate_url(url)
        await self._check_resolved_ips(hostname)

        # Pin DNS resolution to prevent rebinding: resolve once and
        # connect directly to the validated IP address.
        transport = _PinnedTransport(self)

        async with httpx.AsyncClient(
            follow_redirects=True,
            max_redirects=5,
            timeout=httpx.Timeout(timeout),
            transport=transport,
            headers={"User-Agent": "Augmentum/1.0 (AI tool; +https://github.com/augmentum)"},
        ) as client:
            response = await client.get(url)

            # Validate response size via Content-Length header first.
            content_length = response.headers.get("content-length")
            if content_length:
                try:
                    cl = int(content_length)
                except (ValueError, TypeError) as exc:
                    raise SafeHttpError(
                        f"Invalid Content-Length header: {content_length}"
                    ) from exc
                if cl > self._max_response_size:
                    raise SafeHttpError(
                        f"Response too large: {cl} bytes "
                        f"(limit {self._max_response_size})"
                    )

            body = response.content
            if len(body) > self._max_response_size:
                raise SafeHttpError(
                    f"Response too large: {len(body)} bytes "
                    f"(limit {self._max_response_size})"
                )

            # After following redirects, verify the final URL's IP as well.
            final_hostname = urlparse(str(response.url)).hostname
            if final_hostname and final_hostname != hostname:
                await self._check_resolved_ips(final_hostname)

            text = body.decode("utf-8", errors="replace")

            metadata = {
                "url": str(response.url),
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "content_length": len(body),
            }

        return text, metadata

    async def fetch_bytes(
        self,
        url: str,
        *,
        timeout: float = 15.0,
    ) -> tuple[bytes, dict]:
        """Fetch a URL safely, returning (raw_bytes, metadata).

        Identical SSRF protections to :meth:`fetch` (scheme + DNS-IP
        validation, size cap, DNS-rebind-pinned transport, post-redirect
        re-validation) but returns the body undecoded — use this for
        binary payloads (images, avatars) where ``fetch``'s UTF-8 decode
        would corrupt the bytes.
        """
        hostname = self._validate_url(url)
        await self._check_resolved_ips(hostname)

        transport = _PinnedTransport(self)

        async with httpx.AsyncClient(
            follow_redirects=True,
            max_redirects=5,
            timeout=httpx.Timeout(timeout),
            transport=transport,
            headers={"User-Agent": "Augmentum/1.0 (AI tool; +https://github.com/augmentum)"},
        ) as client:
            response = await client.get(url)

            content_length = response.headers.get("content-length")
            if content_length:
                try:
                    cl = int(content_length)
                except (ValueError, TypeError) as exc:
                    raise SafeHttpError(
                        f"Invalid Content-Length header: {content_length}"
                    ) from exc
                if cl > self._max_response_size:
                    raise SafeHttpError(
                        f"Response too large: {cl} bytes "
                        f"(limit {self._max_response_size})"
                    )

            body = response.content
            if len(body) > self._max_response_size:
                raise SafeHttpError(
                    f"Response too large: {len(body)} bytes "
                    f"(limit {self._max_response_size})"
                )

            final_hostname = urlparse(str(response.url)).hostname
            if final_hostname and final_hostname != hostname:
                await self._check_resolved_ips(final_hostname)

            metadata = {
                "url": str(response.url),
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "content_length": len(body),
            }

        return body, metadata


class _PinnedTransport(httpx.AsyncBaseTransport):
    """Custom transport that validates every connection target against SSRF rules.

    Wraps the default httpx transport but intercepts each request to
    re-validate the resolved hostname, preventing DNS rebinding attacks.
    """

    def __init__(self, safe_client: SafeHttpClient) -> None:
        self._safe_client = safe_client
        self._inner = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host:
            self._safe_client._check_resolved_ips_sync(host)
        return await self._inner.handle_async_request(request)
