"""Merge reviewed cross-store conversions into listings.json.

Second half of the curation pipeline (curate_from_stores.py writes
tmp/curation/*.json). The converter can't know what a user SEES after
install — the REVIEW table below is the human pass: browser blocks,
category corrections, media mounts, and resource honesty, one explicit
entry per accepted app. Apps absent from REVIEW are never merged.

Run:  python scripts/curate_from_stores.py && python scripts/merge_curated_listings.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from augmentum.marketplace import catalog_loader as cl  # noqa: E402

# app -> (source store, overrides). Reviewed 2026-07-19.
REVIEW: dict[str, tuple[str, dict]] = {
    "it-tools": ("umbrel", {
        "category": "developer",
        "browser": {"after_install": "status", "path": "/", "credentials": "none"},
        "tagline": "A big box of handy tools for developers, in one page.",
    }),
    "gitea": ("casaos", {
        "category": "developer",
        "browser": {"after_install": "setup_page", "path": "/", "credentials": "user_set"},
        "source_url": "https://github.com/go-gitea/gitea",
        "website": "https://about.gitea.com",
        "tagline": "A painless self-hosted Git service (single-container, SQLite).",
    }),
    "drawio": ("runtipi", {
        "category": "developer",
        "browser": {"after_install": "status", "path": "/", "credentials": "none"},
    }),
    "cyberchef": ("runtipi", {
        "category": "developer",
        "browser": {"after_install": "status", "path": "/", "credentials": "none"},
    }),
    "ntfy": ("runtipi", {
        "category": "automation",
        "browser": {"after_install": "status", "path": "/", "credentials": "none"},
        "extra_volumes": {"lib": "/var/lib/ntfy"},
        "description_suffix": " Note: installs open (no accounts) — add "
                              "users and access control in its settings if "
                              "this server is reachable beyond your LAN.",
    }),
    "gotify": ("runtipi", {
        "category": "automation",
        "browser": {"after_install": "login", "path": "/", "credentials": "user_set"},
    }),
    "beszel": ("runtipi", {
        "category": "networking",
        "browser": {"after_install": "setup_page", "path": "/", "credentials": "user_set"},
    }),
    "actual-budget": ("runtipi", {
        "category": "files",
        "browser": {"after_install": "setup_page", "path": "/", "credentials": "user_set"},
    }),
    "privatebin": ("umbrel", {
        "category": "files",
        "browser": {"after_install": "status", "path": "/", "credentials": "none"},
    }),
    "jellyseerr": ("umbrel", {
        "category": "media",
        "browser": {"after_install": "setup_page", "path": "/", "credentials": "user_set"},
    }),
    "kavita": ("runtipi", {
        "category": "media",
        "browser": {"after_install": "setup_page", "path": "/", "credentials": "user_set"},
        "volumes": {"config": "/kavita/config"},
        "media_mount": "/media",
        "description_suffix": " Point your library at the /media folder "
                              "(optionally bound to a host folder at install).",
    }),
    "komga": ("runtipi", {
        "category": "media",
        "browser": {"after_install": "setup_page", "path": "/", "credentials": "user_set"},
        "volumes": {"config": "/config"},
        "media_mount": "/media",
        "description_suffix": " Point your library at the /media folder "
                              "(optionally bound to a host folder at install).",
    }),
    "libretranslate": ("umbrel", {
        "category": "files",
        "browser": {"after_install": "status", "path": "/", "credentials": "none"},
        "ram_mb": 2048, "disk_mb": 4000,
        "description_suffix": " Downloads translation models on first "
                              "start — allow a few GB of disk and some time.",
    }),
}
# Reviewed and REJECTED (kept here so re-runs don't resurface them):
#   baserow  — needs BASEROW_PUBLIC_URL host templating we don't support
#   wallabag — CasaOS compose doesn't persist the DB (data loss on recreate)
#   homepage/glance — config-file-driven; useless without volume editing
#   whoogle  — duplicates bundled SearXNG
#   octoprint — hardware-niche; umbrel variant needs privileged


def main() -> None:
    cur = ROOT / "tmp" / "curation"
    path = ROOT / "data" / "marketplace" / "listings.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    existing = {entry["id"] for entry in doc["listings"]}

    added = 0
    for app, (store, ov) in REVIEW.items():
        src = cur / f"{app}.{store}.json"
        if not src.exists():
            print(f"SKIP {app}: {src.name} not found (run curate first)")
            continue
        listing = json.loads(src.read_text(encoding="utf-8"))["listing"]
        if listing["id"] in existing:
            print(f"SKIP {app}: already in catalog")
            continue
        payload = listing["install_payload"]
        svc = payload["service"]
        listing["category"] = ov.get("category", listing["category"])
        payload["browser"] = ov["browser"]
        if "volumes" in ov:
            svc["volumes"] = dict(ov["volumes"])
        if "extra_volumes" in ov:
            svc["volumes"] = {**svc["volumes"], **ov["extra_volumes"]}
        if "media_mount" in ov:
            svc["media_mount"] = ov["media_mount"]
        if "tagline" in ov:
            listing["tagline"] = ov["tagline"]
        if "source_url" in ov:
            listing["source_url"] = ov["source_url"]
        if "website" in ov:
            listing["metadata"]["website"] = ov["website"]
        if "description_suffix" in ov:
            listing["description"] = (listing["description"].rstrip()
                                      + ov["description_suffix"])
        if "ram_mb" in ov:
            payload["resources"]["ram_mb"] = ov["ram_mb"]
        if "disk_mb" in ov:
            payload["resources"]["disk_mb"] = ov["disk_mb"]
        payload["lifecycle"]["backup_paths"] = sorted(svc["volumes"].values())

        cl._validate_and_build(listing)  # the real gate — throws on defects
        doc["listings"].append(listing)
        existing.add(listing["id"])
        added += 1
        print(f"ADD  {listing['id']} [{store}] {listing['category']}")

    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(f"\n{added} added; catalog now {len(doc['listings'])} listings")


if __name__ == "__main__":
    main()
