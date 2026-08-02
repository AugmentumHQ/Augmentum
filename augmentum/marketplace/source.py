"""MarketplaceSource -- AXF Source bridge over MarketplaceStore.

The marketplace surfaces curated listings via ``discover()`` and
delegates installs to whichever underlying Source the listing names
in its ``install_via`` field. This keeps the marketplace as a
*curation* layer instead of a parallel install pipeline -- a listing
that wraps a js13k entry installs through Js13kSource, an AGSP-bundled
listing installs through InternalSource, etc.

After a successful install, the listing's ``install_count`` is
incremented for analytics. No user-scoped state lives on the listing
itself.
"""

from __future__ import annotations

from typing import Any

from augmentum.marketplace.store import MarketplaceListing, MarketplaceStore
from augmentum.titles.sources import (
    DiscoveryItem,
    SourceImportError,
    SourceRegistry,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class MarketplaceSource:
    """Source that surfaces curated marketplace listings."""

    id = "marketplace"
    label = "Marketplace"

    def __init__(
        self,
        *,
        store: MarketplaceStore,
        sources: SourceRegistry,
    ) -> None:
        self._store = store
        # We need to call the underlying Source's import_for_user
        # ourselves on install -- the registry resolves the right one
        # at install time, not at construction (so newly-registered
        # sources are picked up without rebuilding the marketplace).
        self._sources = sources

    # ── Discovery ────────────────────────────────────────────────────

    async def discover(
        self, query: dict[str, Any], *, user_id: str = "",
    ) -> list[DiscoveryItem]:
        kind = query.get("kind") if isinstance(query.get("kind"), str) else None
        publisher = (
            query.get("publisher")
            if isinstance(query.get("publisher"), str)
            else None
        )
        limit = max(1, min(200, int(query.get("limit", 50) or 50)))
        listings = await self._store.list_active(
            kind=kind, publisher=publisher, limit=limit,
        )
        # Optional free-text filter -- cheap substring scan over the
        # already-filtered set. Catalog is small enough that this is
        # better than maintaining FTS5 just for the marketplace.
        q = query.get("q")
        if isinstance(q, str) and q.strip():
            needle = q.strip().lower()
            listings = [
                l for l in listings
                if needle in l.title.lower()
                or needle in l.tagline.lower()
                or needle in l.description.lower()
            ]
        return [_listing_to_discovery_item(l) for l in listings]

    # ── Install ──────────────────────────────────────────────────────

    async def import_for_user(
        self, manifest_data: dict, *, user_id: str,
    ) -> str:
        if not user_id:
            raise SourceImportError("user_id required")
        listing_id = (
            manifest_data.get("listing_id")
            or manifest_data.get("source_remote_id")
            or ""
        )
        listing_id = str(listing_id).strip()
        if not listing_id:
            raise SourceImportError(
                "listing_id (or source_remote_id) is required",
            )
        listing = await self._store.get(listing_id)
        if listing is None or listing.delisted_at is not None:
            raise SourceImportError(f"listing {listing_id!r} not found")

        # Delegate to the underlying installer Source. The catalog
        # entry's ``install_via`` field names which one to use.
        installer = self._sources.get(listing.install_via)
        if installer is None:
            raise SourceImportError(
                f"listing {listing_id!r} requires source "
                f"{listing.install_via!r} which is not registered"
            )
        # Pass the listing's install_payload through. The underlying
        # Source's import_for_user takes care of validation.
        artifact_id = await installer.import_for_user(
            listing.install_payload,
            user_id=user_id,
        )
        await self._store.increment_install_count(listing_id)
        log.info(
            "marketplace_install",
            user_id=user_id,
            listing_id=listing_id,
            install_via=listing.install_via,
            artifact_id=artifact_id,
        )
        return artifact_id


def _listing_to_discovery_item(listing: MarketplaceListing) -> DiscoveryItem:
    return DiscoveryItem(
        source_id="marketplace",
        source_remote_id=listing.id,
        kind=listing.kind,
        title=listing.title,
        runtime_preferred=listing.runtime_preferred,
        runtime_alternates=listing.runtime_alternates,
        author=listing.publisher,
        tagline=listing.tagline,
        description=listing.description,
        thumbnail_url=listing.thumbnail_url,
        source_url=listing.source_url,
        embed_url=listing.embed_url,
        capabilities=dict(listing.capabilities),
        metadata={
            **dict(listing.metadata),
            "publisher": listing.publisher,
            "rating": listing.rating,
            "install_count": listing.install_count,
            "install_via": listing.install_via,
        },
    )
