"""Augmentum Marketplace -- curated catalog for the Title framework.

Listings are stored as JSON in ``data/marketplace/listings.json`` and
loaded into the ``marketplace_listings`` SQL table on startup. Every
listing names an ``install_via`` Source (js13k, agsp-profile, internal,
...) and carries an ``install_payload`` dict that gets forwarded to
that Source's ``import_for_user`` when the user clicks Install.

This means the marketplace is a *curation* layer, not a parallel
install pipeline -- adding a new bundle to the catalog is one entry in
the JSON file, no code changes. Adding a new *kind* of catalog source
(third-party signed listings, federated community shares) is a future
extension that drops in alongside the curated loader.
"""

from __future__ import annotations

from augmentum.marketplace.catalog_loader import (
    CatalogLoadError,
    load_catalog_into_store,
)
from augmentum.marketplace.loaders import (
    load_community_into_store,
    load_media_servers_into_store,
    load_providers_into_store,
    schedule_community_feed_refresh,
)
from augmentum.marketplace.source import MarketplaceSource
from augmentum.marketplace.store import MarketplaceListing, MarketplaceStore

__all__ = [
    "CatalogLoadError",
    "MarketplaceListing",
    "MarketplaceSource",
    "MarketplaceStore",
    "load_catalog_into_store",
    "load_community_into_store",
    "load_media_servers_into_store",
    "load_providers_into_store",
    "schedule_community_feed_refresh",
]
