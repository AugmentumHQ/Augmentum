"""Catalog loader -- read ``data/marketplace/listings.json`` into the store.

Called once at server startup (after the marketplace migration runs).
Idempotent: each listing is upserted by id, missing entries are soft-
delisted. Operators can hot-reload by editing the JSON file and
calling the loader manually via a future admin endpoint -- not wired
in v1, but the loader is structured to make that easy.

Schema of ``listings.json``:

    {
      "version": 1,
      "listings": [
        {
          "id": "mkt:foo",
          "publisher": "augmentum",
          "title": "...",
          "kind": "js13k_game" | "streamed_game" | ...,
          "runtime_preferred": "...",
          "runtime_alternates": ["...", ...],
          "tagline": "...",
          "description": "...",
          "thumbnail_url": "...",
          "source_url": "...",
          "embed_url": "...",
          "install_via": "js13k",
          "install_payload": { ... },
          "capabilities": { ... },
          "metadata": { ... }
        },
        ...
      ]
    }

Validation is intentionally lenient -- bad entries are logged and
skipped, the rest of the catalog still loads. Strict mode is
available for tests via ``raise_on_invalid=True``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from augmentum.marketplace.store import MarketplaceListing, MarketplaceStore
from augmentum.titles.manifest import TITLE_KINDS
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class CatalogLoadError(Exception):
    """Raised when the catalog file is missing/corrupt and the caller
    asked for strict loading."""


# The curated catalog ships BAKED inside the package (augmentum/marketplace/
# curated_listings.json), so every install — pull-only or checkout — populates
# Discover out of the box. (It previously read data/marketplace/listings.json,
# which is gitignored runtime data + bind-mounted empty on a fresh install, so
# Discover showed only the 8 baked providers.) An operator can still override by
# dropping their own data/marketplace/listings.json (bind-mounted); it wins when
# present.
_BAKED_CATALOG_PATH = Path(__file__).parent / "curated_listings.json"
_OPERATOR_CATALOG_PATH = Path("data/marketplace/listings.json")
_DEFAULT_CATALOG_PATH = (
    _OPERATOR_CATALOG_PATH if _OPERATOR_CATALOG_PATH.is_file() else _BAKED_CATALOG_PATH
)


async def load_catalog_into_store(
    store: MarketplaceStore,
    *,
    catalog_path: Path | str | None = None,
    raise_on_invalid: bool = False,
) -> dict[str, int]:
    """Load the JSON catalog file into the SQLite marketplace_listings
    table.

    Returns a dict with counts: ``{"loaded": N, "skipped": M, "delisted": K}``.
    """
    path = Path(catalog_path) if catalog_path else _DEFAULT_CATALOG_PATH
    if not path.exists():
        if raise_on_invalid:
            raise CatalogLoadError(f"catalog file not found: {path}")
        log.info("marketplace_catalog_missing", path=str(path))
        return {"loaded": 0, "skipped": 0, "delisted": 0}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        if raise_on_invalid:
            raise CatalogLoadError(f"catalog parse failed: {exc}") from exc
        log.warning("marketplace_catalog_parse_failed", error=str(exc))
        return {"loaded": 0, "skipped": 0, "delisted": 0}

    listings_raw = raw.get("listings") if isinstance(raw, dict) else None
    if not isinstance(listings_raw, list):
        if raise_on_invalid:
            raise CatalogLoadError("catalog missing 'listings' array")
        log.warning("marketplace_catalog_no_listings_array")
        return {"loaded": 0, "skipped": 0, "delisted": 0}

    loaded = 0
    skipped = 0
    seen_ids: set[str] = set()
    for entry in listings_raw:
        if not isinstance(entry, dict):
            skipped += 1
            continue
        try:
            listing = _validate_and_build(entry)
        except ValueError as exc:
            if raise_on_invalid:
                raise CatalogLoadError(str(exc)) from exc
            log.warning(
                "marketplace_catalog_entry_skipped",
                id=entry.get("id"),
                error=str(exc),
            )
            skipped += 1
            continue
        await store.upsert(listing)
        seen_ids.add(listing.id)
        loaded += 1

    # Publisher-scoped sweep — only delist rows we own. The titles
    # loader's domain is publisher="augmentum"; the providers loader
    # owns "augmentum-providers"; future community loaders own
    # "community:<handle>". Each loader runs independently without
    # clobbering its siblings. Listings.json without a publisher
    # field defaults to "augmentum" via _validate_and_build.
    delisted = await store.delist_missing_for_publisher(
        seen_ids, publisher="augmentum",
    )

    log.info(
        "marketplace_catalog_loaded",
        loaded=loaded, skipped=skipped, delisted=delisted,
        path=str(path),
    )
    return {"loaded": loaded, "skipped": skipped, "delisted": delisted}


# ── validation ───────────────────────────────────────────────────────


_REQUIRED_KEYS = ("id", "title", "kind", "install_via")


def _validate_and_build(entry: dict[str, Any]) -> MarketplaceListing:
    for key in _REQUIRED_KEYS:
        if not entry.get(key):
            raise ValueError(f"missing required field: {key}")
    kind = str(entry["kind"])
    if kind == "service":
        # Service app listing (2026-07-18 apps-as-data design): the
        # install_payload IS a manifest and MUST validate here — this is
        # the browser-after-install gate (spec takeaway T4). A service
        # listing that can't say what the user sees after install never
        # enters the catalog.
        from augmentum.marketplace.manifest import ManifestError, parse_manifest
        # ``service_manifest`` (generic single-image pull) and ``service_staged``
        # (staged background-install for engines that build/warm — same manifest
        # shape, different dispatcher) are both valid service install paths.
        if str(entry.get("install_via") or "") not in ("service_manifest", "service_staged"):
            raise ValueError(
                "kind 'service' requires install_via 'service_manifest' or 'service_staged'"
            )
        try:
            parse_manifest(entry.get("install_payload") or {})
        except ManifestError as exc:
            raise ValueError(f"invalid service manifest: {exc}") from exc
    elif kind == "bundle":
        # Profile/pack listing: a member list over other listing ids.
        # Existence of members is checked at INSTALL time (load order
        # between loaders isn't guaranteed); shape is checked here.
        if str(entry.get("install_via") or "") != "bundle":
            raise ValueError("kind 'bundle' requires install_via 'bundle'")
        members = (entry.get("install_payload") or {}).get("members")
        if not isinstance(members, list) or not members or not all(
            isinstance(m, str) and m.strip() for m in members
        ):
            raise ValueError(
                "bundle install_payload.members must be a non-empty list of ids"
            )
    elif kind == "addon":
        # Add-on listing: a capability image built locally from a recipe in
        # this repo. The catalog gate here is the counterpart of the
        # service path's browser-after-install gate — a listing whose
        # addon_id doesn't resolve to a real spec (missing Dockerfile,
        # renamed id, unpinned build args) never enters the catalog, so the
        # failure lands at load time with a clear message instead of at
        # install time after the user has committed to a 25-minute build.
        from augmentum.addons.catalog import addon_by_id

        if str(entry.get("install_via") or "") != "addon_build":
            raise ValueError("kind 'addon' requires install_via 'addon_build'")
        addon_id = str((entry.get("install_payload") or {}).get("addon_id") or "").strip()
        spec = addon_by_id(addon_id)
        if spec is None:
            raise ValueError(
                f"unknown addon_id {addon_id!r} — must match an entry in "
                f"augmentum/addons/catalog.py"
            )
        if not spec.user_facing:
            raise ValueError(
                f"add-on {addon_id!r} is a dependency of other add-ons and "
                f"must not be listed as its own card"
            )
    elif kind not in TITLE_KINDS:
        raise ValueError(
            f"unknown kind {kind!r} (known: service, bundle, {sorted(TITLE_KINDS)})"
        )

    runtime_alternates = entry.get("runtime_alternates") or []
    if not isinstance(runtime_alternates, list):
        runtime_alternates = []

    # Category: explicit if present, else inferred from kind. The
    # explicit override lets a future catalog entry surface a "web_app"
    # under, say, the "characters" rail when that makes sense — but
    # default behaviour matches migration 254's backfill rules.
    category = str(entry.get("category") or "").strip()
    if not category:
        category = _infer_category(kind)

    raw_tags = entry.get("tags") or []
    if not isinstance(raw_tags, list):
        raw_tags = []
    tags = tuple(str(t) for t in raw_tags if str(t).strip())

    return MarketplaceListing(
        id=str(entry["id"]),
        publisher=str(entry.get("publisher") or "augmentum"),
        title=str(entry["title"]),
        kind=kind,
        runtime_preferred=str(entry.get("runtime_preferred") or ""),
        runtime_alternates=tuple(str(a) for a in runtime_alternates),
        tagline=str(entry.get("tagline") or ""),
        description=str(entry.get("description") or ""),
        thumbnail_url=str(entry.get("thumbnail_url") or ""),
        source_url=str(entry.get("source_url") or ""),
        embed_url=str(entry.get("embed_url") or ""),
        install_via=str(entry["install_via"]),
        install_payload=entry.get("install_payload") or {},
        capabilities=entry.get("capabilities") or {},
        metadata=entry.get("metadata") or {},
        rating=_as_float(entry.get("rating")),
        install_count=int(entry.get("install_count") or 0),
        signature=str(entry.get("signature") or ""),
        listed_at="",                                   # set by INSERT default
        category=category,
        tags=tags,
        featured=bool(entry.get("featured") or False),
    )


def _infer_category(kind: str) -> str:
    if kind in ("streamed_game", "js13k_game", "web_app"):
        return "games"
    return "other"


def _as_float(v) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    return None
