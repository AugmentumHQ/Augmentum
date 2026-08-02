"""Media-server loader — turn MEDIA ServiceDefinitions into Discover cards.

Reads the same ``ProviderCatalog`` as the providers loader but selects
only ``category=media`` entries (Jellyfin, Suwayomi, Audiobookshelf, …)
and upserts each as a ``kind=media_server`` listing under
``publisher=augmentum-media``.

These are the content servers an AI OS provisions for you: clicking
Install routes through ``install_dispatchers._install_media_server``,
which starts the container *and* auto-creates a per-user
``user_media_servers`` connection so the server immediately shows up in
Files — no manual URL/credential step.

The loader doesn't depend on Docker being available — ``ProviderCatalog``
is a pure JSON read. If Docker isn't running, the listings still render;
the install dispatcher fails cleanly at click time.
"""

from __future__ import annotations

from augmentum.marketplace.store import MarketplaceListing, MarketplaceStore
from augmentum.providers.catalog import ProviderCatalog
from augmentum.providers.models import ServiceCategory, ServiceDefinition
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_MEDIA_PUBLISHER = "augmentum-media"
_MEDIA_ID_PREFIX = "mkt:media:"

# Editorial curation — marquee content servers light up the Featured
# rail. Suwayomi ships first (no-auth → fully one-click); Jellyfin/ABS
# get featured once their first-run bootstrap lands.
_FEATURED_MEDIA_IDS: frozenset[str] = frozenset({"suwayomi"})

# What lands in Files once the server is connected — surfaced on the card
# so the install reads as a concrete cross-modal payoff, not a bare daemon.
_FILES_PAYOFF: dict[str, str] = {
    "suwayomi": "Manga & comics land in Files and read in the in-app comic reader.",
    "jellyfin": "Movies & shows land in Files and the Continue-watching rail.",
    "emby": "Movies & shows land in Files and the Continue-watching rail.",
    "audiobookshelf": "Audiobooks & podcasts land in Files and the Continue rail.",
    "komga": "Comics & manga land in Files and read in the in-app comic reader.",
}

# Official setup/getting-started docs per server — the post-install card
# routes users here for vetted, community-maintained setup advice (where
# to get sources, how to point at a library). Kept current here rather
# than in the shared catalog since it's a UI/editorial concern.
_SETUP_GUIDE_URL: dict[str, str] = {
    "suwayomi": "https://github.com/Suwayomi/Suwayomi-Server/wiki",
    "jellyfin": "https://jellyfin.org/docs/",
    "emby": "https://emby.media/support/",
    "audiobookshelf": "https://www.audiobookshelf.org/guides",
    "komga": "https://komga.org/docs/",
}

# Compliance/expectation note — every server installs EMPTY. We ship the
# software, never content; the user brings their own sources/files. This
# is surfaced on the install confirm AND the post-install card so the
# expectation (and the legal posture) is explicit.
_CONTENT_NOTE: dict[str, str] = {
    "suwayomi": "Installs empty — no manga or sources are included. You add your own sources and titles in Suwayomi.",
    "komga": "Installs empty — no comics are included. You point it at your own comic/manga files.",
    "jellyfin": "Installs empty — no media is included. You point it at your own movie/show library.",
    "emby": "Installs empty — no media is included. You point it at your own movie/show library.",
    "audiobookshelf": "Installs empty — no audio is included. You point it at your own audiobook/podcast files.",
}
_CONTENT_NOTE_DEFAULT = (
    "Installs empty — no content is included. You connect your own "
    "sources or files, so your library stays yours."
)

# Container mount path that should point at the user's EXTERNAL media
# library (a host bind mount) rather than opaque Docker storage. Empty/
# absent → the server has no local library to mount (Suwayomi streams from
# online sources into its own data volume, so it needs no host media path).
_MEDIA_MOUNT: dict[str, str] = {
    "jellyfin": "/media",
    "audiobookshelf": "/audiobooks",
    "komga": "/data",
}

# Media-server icons from the selfhst icon set (same source the CasaOS/
# big-bear stores use), proxied + squircle-masked client-side. Verified
# live 2026-07-19; missing ids fall back to the per-kind glyph.
_MEDIA_ICONS: dict[str, str] = {
    "jellyfin": "https://cdn.jsdelivr.net/gh/selfhst/icons/png/jellyfin.png",
    "emby": "https://cdn.jsdelivr.net/gh/selfhst/icons/png/emby.png",
    "audiobookshelf": "https://cdn.jsdelivr.net/gh/selfhst/icons/png/audiobookshelf.png",
    "komga": "https://cdn.jsdelivr.net/gh/selfhst/icons/png/komga.png",
    "suwayomi": "https://cdn.jsdelivr.net/gh/selfhst/icons/png/suwayomi.png",
}


def _service_to_listing(sd: ServiceDefinition) -> MarketplaceListing:
    """Translate a MEDIA ServiceDefinition into a Discover-shaped listing.

    ``install_via="media-server"`` routes through
    ``install_dispatchers._install_media_server``. The ``install_payload``
    carries the service id (which doubles as the media-provider name) so
    the dispatcher knows what to provision and how to connect it.
    """
    tagline = sd.description.split(". ", 1)[0] if sd.description else ""
    payoff = _FILES_PAYOFF.get(sd.id, "")
    setup_guide_url = _SETUP_GUIDE_URL.get(sd.id, "")
    content_note = _CONTENT_NOTE.get(sd.id, _CONTENT_NOTE_DEFAULT)
    media_mount = _MEDIA_MOUNT.get(sd.id, "")
    return MarketplaceListing(
        id=f"{_MEDIA_ID_PREFIX}{sd.id}",
        publisher=_MEDIA_PUBLISHER,
        title=sd.name,
        kind="media_server",
        runtime_preferred="",
        runtime_alternates=(),
        tagline=tagline[:200],
        description=sd.description,
        thumbnail_url=_MEDIA_ICONS.get(sd.id, ""),
        source_url="",
        embed_url="",
        install_via="media-server",
        install_payload={
            "service_id": sd.id, "provider": sd.id, "media_mount": media_mount,
        },
        capabilities={
            "gpu_required": sd.gpu.required,
            "vram_mb": sd.gpu.vram_mb,
            "host_port": sd.host_port,
            # Dedicated HTTPS front-door port (Caddy TLS reverse proxy to the
            # container). When set, the UI opens the server over real HTTPS
            # instead of the raw HTTP host port (which fails under HSTS).
            "https_port": getattr(sd, "https_port", 0),
            "no_auth": "no_auth" in (sd.features or []),
            "managed_auth": "managed_auth" in (sd.features or []),
            # True when Augmentum mints the server's login by either path
            # (Basic-auth env or first-run wizard) — drives the card's
            # "log in with the credentials below" affordance.
            "managed_credentials": bool(
                {"managed_auth", "first_run_wizard"} & set(sd.features or [])
            ),
            "first_run_wizard": "first_run_wizard" in (sd.features or []),
            # The server has a local library that should point at the user's
            # OWN storage — the install UI collects a host path to bind here
            # (else it falls back to a Docker-managed volume).
            "needs_media_path": bool(media_mount),
            "media_mount": media_mount,
            "files_payoff": payoff,
        },
        metadata={
            "image": sd.image,
            "features": list(sd.features),
            "files_payoff": payoff,
            "setup_guide_url": setup_guide_url,
            "content_note": content_note,
        },
        rating=None,
        install_count=0,
        signature="",
        listed_at="",                                   # INSERT default
        category="media",
        tags=tuple(sd.features or []),
        featured=sd.id in _FEATURED_MEDIA_IDS,
    )


async def load_media_servers_into_store(
    store: MarketplaceStore,
    *,
    catalog: ProviderCatalog | None = None,
) -> dict[str, int]:
    """Upsert every MEDIA catalog entry into ``marketplace_listings``.

    Returns ``{"loaded": N, "skipped": M, "delisted": K}`` matching the
    other loaders' shape. Delist sweep is scoped to
    ``publisher=augmentum-media`` so provider/title/community rows are
    untouched.
    """
    cat = catalog or ProviderCatalog()
    entries = cat.list_by_category(ServiceCategory.MEDIA)

    loaded = 0
    skipped = 0
    seen_ids: set[str] = set()
    for sd in entries:
        try:
            listing = _service_to_listing(sd)
        except Exception as exc:  # noqa: BLE001 — one bad entry mustn't break the load
            log.warning(
                "media_server_listing_build_failed",
                service_id=sd.id, error=str(exc),
            )
            skipped += 1
            continue
        await store.upsert(listing)
        seen_ids.add(listing.id)
        loaded += 1

    delisted = await store.delist_missing_for_publisher(
        seen_ids, publisher=_MEDIA_PUBLISHER,
    )

    log.info(
        "marketplace_media_servers_loaded",
        loaded=loaded, skipped=skipped, delisted=delisted,
    )
    return {"loaded": loaded, "skipped": skipped, "delisted": delisted}
