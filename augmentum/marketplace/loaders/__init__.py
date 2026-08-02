"""Discover-surface catalog loaders.

Each loader reads a source and upserts into ``marketplace_listings``.
Loaders are idempotent (UPSERT by id) and tag rows with a ``publisher``
that scopes the ``delist_missing`` sweep to that loader's domain — so
one loader can't accidentally delist rows owned by another.

Loaders today:
  - ``providers`` — reads ``ProviderCatalog`` (catalog.json via
    ``augmentum/providers/catalog.py``) and upserts each service
    definition as a ``kind=provider_service`` listing.

Future:
  - ``community`` — fetches augmentumhq.com/community/index.json
    (Phase 2 of the Discover surface spec).
"""

from __future__ import annotations

from augmentum.marketplace.loaders.community import (
    load_community_into_store,
    schedule_community_feed_refresh,
)
from augmentum.marketplace.loaders.media_servers import (
    load_media_servers_into_store,
)
from augmentum.marketplace.loaders.providers import load_providers_into_store

__all__ = [
    "load_community_into_store",
    "load_media_servers_into_store",
    "load_providers_into_store",
    "schedule_community_feed_refresh",
]
