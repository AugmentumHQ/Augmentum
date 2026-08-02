"""Knowledge pack install catalog.

Two kinds of entries:

* **Language packs** — installable via the existing learning route
  ``POST /api/learning/packs/{lang}/install`` which enqueues a
  background job. Per-user; the catalog is the live
  ``available_packs()`` from ``augmentum/learning/lang_pack_catalog``
  so adding a new language to that catalog automatically becomes
  offer-able.

* **External reference packs** — ZIM archives (Wikipedia, MDWiki,
  Stack Exchange, Python docs). These are too large (often 100+ GB)
  to auto-download from an offer; the accept handler returns a
  link to the Kiwix download page so the user can pick the size
  variant they want. Not admin-scoped — pointing at Kiwix is a
  user-friendly action.

Why not call ``POST /api/knowledge/download`` directly: that
endpoint needs an exact URL + filename and is admin-scoped. Showing
the user the Kiwix browse page is friendlier and avoids both
problems (URL drift, surprise multi-GB downloads).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from augmentum.learning import lang_pack_catalog
from augmentum.offers.catalog.base import (
    CatalogEntry,
    OfferPreview,
    register_kind,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request


log = get_logger(__name__)


KIND: str = "knowledge_pack"


# ── Language packs ───────────────────────────────────────────────


def _make_lang_entry(spec) -> CatalogEntry:  # type: ignore[no-untyped-def]
    lang_code = spec.lang_code
    name = spec.name
    pack_mb = spec.approx_pack_mb
    dl_mb = spec.total_download_mb
    target_id = f"lang:{lang_code}"

    async def _preview(_target_id: str, user_id: str) -> OfferPreview | None:
        # Best-effort already-installed check — short-circuit if the
        # pack manager confirms presence. The check is purely an
        # optimisation; the install endpoint also gates on it (409).
        # We can't reach app.state from here, so just skip the check
        # at preview time. The accept handler will return
        # already_installed=True idempotently.
        return OfferPreview(
            label=f"{name} ({lang_code}) language pack",
            hint=(
                f"~{pack_mb}MB built, ~{dl_mb}MB sources. "
                "Adds vocab + dictionary + example-sentence SRS for this language."
            ),
            details={
                "scope": "user",
                "lang_code": lang_code,
                "approx_pack_mb": pack_mb,
                "approx_download_mb": dl_mb,
            },
        )

    async def _accept(payload: dict[str, Any], request: Request) -> dict[str, Any]:
        # Reach into app.state to: (a) check whether already installed,
        # (b) enqueue the install job. Mirrors the /api/learning/packs/
        # {lang}/install route's logic exactly.
        mgr = getattr(request.app.state, "pack_manager", None)
        store = getattr(request.app.state, "jobs_store", None)
        user = request.scope.get("user")
        uid = getattr(user, "id", "") if user is not None else ""
        if not uid:
            return {"ok": False, "error": "no_user"}
        if store is None:
            return {"ok": False, "error": "no_job_store"}

        if mgr is not None and mgr.has_language_pack(lang_code):
            return {
                "ok": True,
                "already_installed": True,
                "lang_code": lang_code,
                "next_step": "Open Settings → Knowledge Packs to manage.",
            }

        # Coalesce: re-use any pending / running install for this lang.
        _ACTIVE = {"pending", "running"}
        _JOB_TYPE = "lang_pack_install"
        for status in _ACTIVE:
            for job in await store.list_for_user(
                user_id=uid, job_type=_JOB_TYPE, status=status,
            ):
                if (job.get("payload") or {}).get("lang_code") == lang_code:
                    return {
                        "ok": True,
                        "lang_code": lang_code,
                        "job_id": job["id"],
                        "next_step": (
                            f"Install for {lang_code} is already running. "
                            "Track in Settings → Background Jobs."
                        ),
                    }

        job_id = await store.create(
            user_id=uid, job_type=_JOB_TYPE,
            payload={"lang_code": lang_code}, priority=1,
        )
        log.info(
            "offer_lang_pack_install_enqueued",
            lang_code=lang_code, job_id=job_id, user_id=uid,
        )
        return {
            "ok": True,
            "lang_code": lang_code,
            "job_id": job_id,
            "next_step": (
                f"Background install started (~{dl_mb}MB download, ~{pack_mb}MB final). "
                "Settings → Background Jobs shows progress."
            ),
        }

    return CatalogEntry(
        kind=KIND,
        target_id=target_id,
        title=f"Install the {name} language pack?",
        scope="user",
        build_preview=_preview,
        accept=_accept,
    )


# ── External reference packs (Kiwix ZIM) ─────────────────────────


_ZIM_ENTRIES: list[dict[str, Any]] = [
    {
        "target_id": "zim:wikipedia_en",
        "title": "Open the Wikipedia (English) Kiwix download page?",
        "name": "Wikipedia (English)",
        "url": "https://library.kiwix.org/?lang=eng&category=wikipedia",
        "hint": (
            "ZIM archive. Pick a size variant (mini ~250MB / nopic ~50GB / "
            "full ~100GB). Drop the .zim into your knowledge packs dir."
        ),
    },
    {
        "target_id": "zim:mdwiki",
        "title": "Open the MDWiki Kiwix download page?",
        "name": "MDWiki (medicine)",
        "url": "https://library.kiwix.org/?lang=eng&category=wikipedia&search=mdwiki",
        "hint": "Curated medical-encyclopedia ZIM. ~9GB.",
    },
    {
        "target_id": "zim:stackoverflow",
        "title": "Open the Stack Overflow Kiwix download page?",
        "name": "Stack Overflow (programming)",
        "url": "https://library.kiwix.org/?lang=eng&category=stack_exchange",
        "hint": "Top Stack Exchange dumps in ZIM. Programming ~80GB; smaller subsets available.",
    },
    {
        "target_id": "zim:devdocs",
        "title": "Open the DevDocs Kiwix download page?",
        "name": "DevDocs (programming references)",
        "url": "https://library.kiwix.org/?lang=eng&category=other&search=devdocs",
        "hint": "Combined documentation set (Python, JS, Go, Rust, etc.). ~2-5GB.",
    },
    {
        "target_id": "zim:gutenberg",
        "title": "Open the Project Gutenberg Kiwix download page?",
        "name": "Project Gutenberg",
        "url": "https://library.kiwix.org/?lang=eng&category=gutenberg",
        "hint": "70K+ public-domain books in ZIM. ~65GB full, smaller subsets available.",
    },
]


def _make_zim_entry(record: dict[str, Any]) -> CatalogEntry:
    target_id: str = record["target_id"]
    title: str = record["title"]
    name: str = record["name"]
    url: str = record["url"]
    hint: str = record["hint"]

    async def _preview(_target_id: str, _user_id: str) -> OfferPreview | None:
        return OfferPreview(
            label=name,
            hint=hint,
            details={
                "scope": "user",
                "kind": "zim_external",
                "kiwix_url": url,
            },
        )

    async def _accept(payload: dict[str, Any], request: Request) -> dict[str, Any]:
        # ZIM downloads are huge and the canonical source is Kiwix.
        # We don't kick off an auto-download — we just return the
        # URL so the chip renders an Open link.
        return {
            "ok": True,
            "kind": "external_link",
            "url": url,
            "name": name,
            "next_step": (
                f"Open the Kiwix library at {url}, pick a size variant "
                f"that fits your disk, and drop the .zim into your "
                f"knowledge packs directory. Augmentum scans on startup."
            ),
        }

    return CatalogEntry(
        kind=KIND,
        target_id=target_id,
        title=title,
        scope="user",
        build_preview=_preview,
        accept=_accept,
    )


# ── Catalog ──────────────────────────────────────────────────────


def _build_entries() -> list[CatalogEntry]:
    entries: list[CatalogEntry] = []
    try:
        lang_packs = lang_pack_catalog.available_packs()
    except Exception as exc:
        log.warning(
            "offer_lang_pack_catalog_unavailable", error=str(exc)[:160],
        )
        lang_packs = []
    for spec in lang_packs:
        entries.append(_make_lang_entry(spec))
    for record in _ZIM_ENTRIES:
        entries.append(_make_zim_entry(record))
    return entries


ENTRIES: list[CatalogEntry] = _build_entries()


if ENTRIES:
    register_kind(KIND, ENTRIES)
