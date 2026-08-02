"""Provider service loader — turn ServiceDefinitions into MarketplaceListings.

Reads ``ProviderCatalog`` (the same one ``ServiceManager`` uses) and
upserts each entry as a ``kind=provider_service`` listing under
``publisher=augmentum-providers``. This is what makes the Settings-buried
provider marketplace discoverable from the unified Discover surface.

The loader doesn't depend on Docker being available — ``ProviderCatalog``
is a pure JSON read. If Docker isn't running, the listings still render in
Discover; the install dispatcher fails cleanly at click time with a 503.
"""

from __future__ import annotations

from augmentum.marketplace.store import MarketplaceListing, MarketplaceStore
from augmentum.providers.catalog import ProviderCatalog
from augmentum.providers.models import ServiceCategory, ServiceDefinition
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


_PROVIDER_PUBLISHER = "augmentum-providers"
_PROVIDER_ID_PREFIX = "mkt:provider:"

# Editorial curation — high-leverage entry-point providers that should
# light up Discover's Featured rail. Kept here rather than in
# providers/catalog.json because it's a UI decision, not a data model
# concern (and the source catalog is shared with non-Discover code).
# LLM runners are intentionally NOT featured: Augmentum ships its own
# llama.cpp engine (with session restore) and auto-detects external
# OpenAI-compatible endpoints, so the marketplace focuses on the pieces it
# doesn't already provide (TTS/STT).
_FEATURED_PROVIDER_IDS: frozenset[str] = frozenset({
    "kokoro-server",   # high-quality TTS, low-friction setup
    "speaches-stt",    # fast local STT
})

# Provider icons — each project's own mark (GitHub org avatar or the
# selfhst icon set the self-hosting app-store ecosystem standardizes
# on). Rendered through the browse image proxy client-side and
# squircle-masked by the Discover CSS, so provider cards land visually
# identical to the service-app icons. All URLs verified live
# 2026-07-19; ids without an entry fall back to the designed per-kind
# glyph, never a broken image.
_PROVIDER_ICONS: dict[str, str] = {
    "chatterbox-tts": "https://github.com/resemble-ai.png",
    "chatterbox-turbo": "https://github.com/resemble-ai.png",
    "speaches-stt": "https://cdn.jsdelivr.net/gh/selfhst/icons/png/speaches.png",
    "fish-tts": "https://github.com/fishaudio.png",
    "kokoro-server": "https://github.com/remsky.png",
}


def _service_to_listing(sd: ServiceDefinition) -> MarketplaceListing:
    """Translate a ServiceDefinition into a Discover-shaped listing.

    The ``install_via="provider-service"`` key routes through
    ``install_dispatchers._install_provider_service`` which calls
    ``service_manager.enable_service(service_id)``.
    """
    tagline = sd.description.split(". ", 1)[0] if sd.description else ""
    return MarketplaceListing(
        id=f"{_PROVIDER_ID_PREFIX}{sd.id}",
        publisher=_PROVIDER_PUBLISHER,
        title=sd.name,
        kind="provider_service",
        runtime_preferred="",
        runtime_alternates=(),
        tagline=tagline[:200],
        description=sd.description,
        thumbnail_url=_PROVIDER_ICONS.get(sd.id, ""),
        source_url="",
        embed_url="",
        install_via="provider-service",
        install_payload={"service_id": sd.id},
        capabilities={
            "gpu_required": sd.gpu.required,
            "vram_mb": sd.gpu.vram_mb,
            "api_type": sd.api_type,
            "host_port": sd.host_port,
        },
        metadata={
            "image": sd.image,
            "features": list(sd.features),
            "service_category": sd.category.value,
            # Install-time requirements (gated token / license) so the
            # Discover card can collect them inline before provisioning.
            "requirements": dict(sd.requirements or {}),
        },
        rating=None,
        install_count=0,
        signature="",
        listed_at="",                                   # INSERT default
        category="providers",
        tags=tuple(sd.features or []),
        featured=sd.id in _FEATURED_PROVIDER_IDS,
    )


async def load_providers_into_store(
    store: MarketplaceStore,
    *,
    catalog: ProviderCatalog | None = None,
) -> dict[str, int]:
    """Upsert every provider catalog entry into ``marketplace_listings``.

    Returns ``{"loaded": N, "skipped": M, "delisted": K}`` matching the
    titles loader's shape so server.py can log them uniformly.

    Delist sweep: provider listings that vanish from catalog.json (a
    service definition removed between server restarts) get soft-
    delisted so Discover stops surfacing them. The sweep is scoped to
    rows where ``publisher=augmentum-providers`` so titles-loader rows
    aren't touched.
    """
    cat = catalog or ProviderCatalog()
    entries = cat.list_all()

    loaded = 0
    skipped = 0
    seen_ids: set[str] = set()
    for sd in entries:
        # MEDIA-category services (Jellyfin/Suwayomi/…) are content
        # servers, not inference providers — they're surfaced by the
        # dedicated media-server loader as kind=media_server with their
        # own install dispatcher. Skip them here so they don't show up
        # as provider_service listings.
        if sd.category == ServiceCategory.MEDIA:
            continue
        try:
            listing = _service_to_listing(sd)
        except Exception as exc:  # noqa: BLE001 — never let one bad entry break the load
            log.warning(
                "provider_listing_build_failed",
                service_id=sd.id, error=str(exc),
            )
            skipped += 1
            continue
        await store.upsert(listing)
        seen_ids.add(listing.id)
        loaded += 1

    # Publisher-scoped sweep via the store helper. Only rows under our
    # publisher namespace get touched — titles + community loaders
    # operate independently.
    delisted = await store.delist_missing_for_publisher(
        seen_ids, publisher=_PROVIDER_PUBLISHER,
    )

    log.info(
        "marketplace_providers_loaded",
        loaded=loaded, skipped=skipped, delisted=delisted,
    )
    return {"loaded": loaded, "skipped": skipped, "delisted": delisted}
