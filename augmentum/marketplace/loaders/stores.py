"""Community app stores — a store is just a URL serving listings JSON.

Spec takeaway T5 (2026-07-18 service-OS design): no central approval.
Anyone can host a JSON file shaped like ``data/marketplace/listings.json``
(``{"name": ..., "listings": [...]}``) and an admin adds it by URL. Its
listings sync into the shared catalog under a ``community:<slug>``
publisher, styled as third-party in Discover, and re-sync on every boot.

Trust model (phase-4 MVP, stated honestly):

* **Admin-only add** — adding a store is an install-wide trust decision.
* **The validation gate is the contract** — every entry passes the same
  ``_validate_and_build`` as the shipped catalog, so service manifests
  keep the pinned-image + browser-after-install rules, and installs keep
  their own admin gates. A hostile store can list junk; it cannot make
  junk install itself.
* **Namespace enforcement** — listing ids and publisher are FORCED into
  the store's namespace regardless of what the JSON claims, so a
  community store can never shadow official ``mkt:`` ids or impersonate
  another publisher.
* **SSRF-guarded https fetch** with a hard size cap.
* Per-store signing keys (spec §4) are NOT implemented yet — that's the
  remaining phase-4 item, tracked in the spec.
"""

from __future__ import annotations

import json
import re
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

STORES_KEY = "marketplace.community_stores"
MAX_STORE_BYTES = 2 * 1024 * 1024  # a catalog is text; 2MB is generous
MAX_STORE_LISTINGS = 500


def slug_for_url(url: str) -> str:
    """Deterministic short slug for a store URL (host + path hash)."""
    import hashlib
    host = re.sub(r"^https?://", "", url).split("/", 1)[0].lower()
    host = re.sub(r"[^a-z0-9.-]", "", host)[:40] or "store"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    return f"{host}-{digest}"


async def list_stores(settings_store) -> list[dict[str, Any]]:
    raw = await settings_store.get(STORES_KEY)
    try:
        stores = json.loads(raw) if raw else []
    except json.JSONDecodeError:
        log.warning("community_stores_config_corrupt")
        return []
    return stores if isinstance(stores, list) else []


async def _save_stores(settings_store, stores: list[dict[str, Any]]) -> None:
    await settings_store.set(STORES_KEY, json.dumps(stores))


async def add_store(settings_store, *, url: str, name: str = "") -> dict[str, Any]:
    """Register a store URL. Raises ValueError on bad/duplicate input."""
    url = (url or "").strip()
    if not url.lower().startswith("https://"):
        raise ValueError("Store URL must be https://")
    stores = await list_stores(settings_store)
    if any(s.get("url") == url for s in stores):
        raise ValueError("That store is already added")
    entry = {
        "slug": slug_for_url(url),
        "url": url,
        "name": (name or "").strip()[:80],
    }
    stores.append(entry)
    await _save_stores(settings_store, stores)
    return entry


async def remove_store(settings_store, marketplace_store, slug: str) -> int:
    """Unregister a store and delist everything it published.

    Returns the number of delisted rows. Delist is soft (same semantics
    as catalog removal) — nothing a user installed gets uninstalled.
    """
    stores = await list_stores(settings_store)
    kept = [s for s in stores if s.get("slug") != slug]
    if len(kept) == len(stores):
        raise ValueError("Unknown store")
    await _save_stores(settings_store, kept)
    delisted = await marketplace_store.delist_missing_for_publisher(
        set(), publisher=f"community:{slug}",
    )
    log.info("community_store_removed", slug=slug, delisted=delisted)
    return delisted


async def sync_store(marketplace_store, http, entry: dict[str, Any]) -> dict[str, int]:
    """Fetch one store's JSON and sync its listings into the catalog.

    Invalid entries are skipped loudly (the same contract as the shipped
    catalog loader); a fetch failure leaves the store's previous listings
    in place — a flaky host must not empty a working store.
    """
    from augmentum.marketplace.catalog_loader import _validate_and_build
    from augmentum.utils.safe_http import check_ssrf

    slug = str(entry.get("slug") or "")
    url = str(entry.get("url") or "")
    publisher = f"community:{slug}"
    await check_ssrf(url)

    resp = await http.get(url, timeout=20.0, follow_redirects=True)
    resp.raise_for_status()
    body = resp.content[: MAX_STORE_BYTES + 1]
    if len(body) > MAX_STORE_BYTES:
        raise ValueError(f"store payload exceeds {MAX_STORE_BYTES} bytes")
    doc = json.loads(body.decode("utf-8"))
    listings_raw = doc.get("listings") if isinstance(doc, dict) else None
    if not isinstance(listings_raw, list):
        raise ValueError("store JSON missing 'listings' array")
    if len(listings_raw) > MAX_STORE_LISTINGS:
        raise ValueError(f"store lists {len(listings_raw)} items (cap {MAX_STORE_LISTINGS})")

    loaded = skipped = 0
    seen_ids: set[str] = set()
    for raw in listings_raw:
        if not isinstance(raw, dict):
            skipped += 1
            continue
        namespaced = dict(raw)
        # Namespace enforcement: id + publisher are OURS to assign. A
        # community store can never claim official ids or another
        # publisher, whatever its JSON says.
        rid = str(namespaced.get("id") or "").strip()
        prefix = f"{publisher}:"
        if not rid.startswith(prefix):
            namespaced["id"] = f"{prefix}{rid}" if rid else ""
        namespaced["publisher"] = publisher
        try:
            listing = _validate_and_build(namespaced)
        except ValueError as exc:
            log.warning(
                "community_store_entry_skipped",
                store=slug, id=raw.get("id"), error=str(exc),
            )
            skipped += 1
            continue
        await marketplace_store.upsert(listing)
        seen_ids.add(listing.id)
        loaded += 1

    delisted = await marketplace_store.delist_missing_for_publisher(
        seen_ids, publisher=publisher,
    )
    log.info(
        "community_store_synced",
        store=slug, loaded=loaded, skipped=skipped, delisted=delisted,
    )
    return {"loaded": loaded, "skipped": skipped, "delisted": delisted}


async def sync_all_stores(settings_store, marketplace_store, http) -> None:
    """Best-effort boot refresh of every registered store. One store's
    failure never blocks the others or the boot."""
    for entry in await list_stores(settings_store):
        try:
            await sync_store(marketplace_store, http, entry)
        except Exception as exc:  # noqa: BLE001 — per-store isolation
            log.warning(
                "community_store_sync_failed",
                store=entry.get("slug"), error=str(exc)[:200],
            )
