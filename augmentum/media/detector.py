"""Silent auto-detection of media servers on default ports.

Probes a strict whitelist of host:port combinations in parallel with a
short timeout. We do not scan LAN IP ranges — that's malware-shaped and
triggers endpoint security. The user's own box reachable via
``host.docker.internal`` (Docker Desktop) or the compose-internal
``127.0.0.1`` (rare) is the only scope worth silent probing.

Users whose server lives elsewhere (NAS, separate machine) reach it by
manually entering the URL. The detector returns an empty list for them,
and the UI shows the manual-entry form without fanfare.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from augmentum.media.providers.audiobookshelf import AudiobookshelfProvider
from augmentum.media.providers.base import DEFAULT_PORTS, ProviderInfo
from augmentum.media.providers.emby import EmbyProvider
from augmentum.media.providers.jellyfin import JellyfinProvider
from augmentum.media.providers.komga import KomgaProvider
from augmentum.media.providers.suwayomi import SuwayomiProvider
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import httpx

log = get_logger(__name__)


# Narrow whitelist. Any new host would need to be added here deliberately.
_PROBE_HOSTS: tuple[str, ...] = (
    "host.docker.internal",
    "127.0.0.1",
)


async def detect_servers(http_client: httpx.AsyncClient) -> list[ProviderInfo]:
    """Probe every (host, port) pair the registered providers care about.

    Returns a flat list of confirmed ProviderInfo entries. Failures are
    silent — that's the whole point of background detection.
    """
    abs_provider = AudiobookshelfProvider(http_client)
    emby_provider = EmbyProvider(http_client)
    jellyfin_provider = JellyfinProvider(http_client)
    komga_provider = KomgaProvider(http_client)
    suwayomi_provider = SuwayomiProvider(http_client)

    tasks: list[asyncio.Task[ProviderInfo | None]] = []
    for host in _PROBE_HOSTS:
        # Audiobookshelf
        tasks.append(asyncio.create_task(
            abs_provider.ping(f"http://{host}:{DEFAULT_PORTS['audiobookshelf']}")
        ))
        tasks.append(asyncio.create_task(
            emby_provider.ping(f"http://{host}:{DEFAULT_PORTS['emby']}")
        ))
        tasks.append(asyncio.create_task(
            jellyfin_provider.ping(f"http://{host}:{DEFAULT_PORTS['jellyfin']}")
        ))
        # Komga (comic server, REST + OPDS)
        tasks.append(asyncio.create_task(
            komga_provider.ping(f"http://{host}:{DEFAULT_PORTS['komga']}")
        ))
        # Suwayomi (Tachiyomi-extension bridge, library mirror)
        tasks.append(asyncio.create_task(
            suwayomi_provider.ping(f"http://{host}:{DEFAULT_PORTS['suwayomi']}")
        ))
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Deduplicate by (provider, base_url). Different hosts may resolve to
    # the same server (127.0.0.1 + host.docker.internal can both reach a
    # compose-internal container), and we want to present one detection.
    seen: set[tuple[str, str]] = set()
    found: list[ProviderInfo] = []
    for res in results:
        if isinstance(res, ProviderInfo):
            key = (res.provider, res.base_url)
            if key in seen:
                continue
            seen.add(key)
            found.append(res)

    if found:
        log.info("media_detect_found", servers=[(f.provider, f.base_url) for f in found])
    return found
