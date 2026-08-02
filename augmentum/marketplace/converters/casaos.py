"""CasaOS dialect (official IceWhaleTech store AND big-bear-casaos) →
Augmentum listing.

Everything lives in one docker-compose.yml: services with an optional
per-service ``x-casaos`` block, plus a top-level ``x-casaos`` block with
id/main/category/developer/icon/tagline/description (localized dicts —
we take en_US).
"""
from __future__ import annotations

from typing import Any

import yaml

from augmentum.marketplace.converters.base import (
    ConversionResult,
    build_listing,
    classify_env,
    container_volumes,
    gate_service,
    normalize_env,
    real_services,
)

_CATEGORY_MAP = {
    "network": "networking", "media": "media", "developer": "developer",
    "development": "developer", "utilities": "files", "productivity": "files",
    "cloud": "files", "backup": "files", "documents": "files",
    "downloader": "media", "entertainment": "media", "music": "media",
    "photo": "media", "video": "media", "home automation": "automation",
    "automation": "automation", "smart home": "automation",
    "finance": "files", "notes": "files", "security": "files",
    "monitor": "networking", "monitoring": "networking",
    "ai": "other", "games": "other", "others": "files",
}


def _en(value: Any) -> str:
    """Localized dict → en_US string; plain strings pass through."""
    if isinstance(value, dict):
        return str(value.get("en_US") or value.get("en_GB")
                   or next(iter(value.values()), ""))
    return str(value or "")


def convert_casaos(compose_text: str, *, repo_slug: str = "casaos") -> ConversionResult:
    compose: dict[str, Any] = yaml.safe_load(compose_text) or {}
    meta = compose.get("x-casaos") or {}
    services = real_services(compose)

    # App id: prefer the compose "name"; reverse-domain x-casaos ids
    # (com.bigbeartechworld.it-tools) collapse to their last segment.
    raw_id = str(compose.get("name") or meta.get("id") or "")
    app_id = raw_id.rsplit(".", 1)[-1].removeprefix("big-bear-").strip().lower()

    reasons: list[str] = []
    review: list[str] = []
    if not app_id:
        reasons.append("no app id (compose name / x-casaos id missing)")
    if len(services) != 1:
        reasons.append(
            f"multi-container ({len(services)} services: {sorted(services)})")
    if reasons:
        return ConversionResult(app_id=app_id or "?", eligible=False, reasons=reasons)

    main = str(meta.get("main") or "")
    name = main if main in services else next(iter(services))
    svc = services[name]
    reasons.extend(gate_service(name, svc))

    port = _web_port(svc, meta)
    if not (0 < port < 65536):
        reasons.append("no usable container web port")

    env, prompts, env_review = classify_env(normalize_env(svc))
    review.extend(env_review)

    if reasons:
        return ConversionResult(app_id=app_id, eligible=False,
                                reasons=reasons, review=review)

    category = _CATEGORY_MAP.get(str(meta.get("category") or "").lower(), "files")
    ram_mb = _reserved_ram_mb(svc)
    review.append("browser block defaulted (setup_page/user_set) — verify")
    tagline = _en(meta.get("tagline"))
    description = _en(meta.get("description"))
    listing = build_listing(
        app_id=app_id,
        title=_en(meta.get("title")) or app_id.replace("-", " ").title(),
        category=category,
        tagline=tagline,
        description=description[:600],
        image=str(svc.get("image") or ""),
        port=port,
        volumes=container_volumes(svc),
        env=env,
        env_prompts=prompts,
        source_url="",
        developer=str(meta.get("developer") or ""),
        website="",
        icon_url=str(meta.get("icon") or ""),
        gallery=[str(u) for u in (meta.get("screenshot_link") or [])][:3],
        tags=[t for t in [str(meta.get("category") or "").lower()] if t],
        ram_mb=ram_mb,
        source_store=repo_slug,
    )
    return ConversionResult(app_id=app_id, eligible=True,
                            review=review, listing=listing)


def _web_port(svc: dict, meta: dict) -> int:
    """Container-side web port: per-service x-casaos ports description,
    else the container side of the first published port mapping."""
    ext = svc.get("x-casaos") or {}
    for p in ext.get("ports") or []:
        if isinstance(p, dict) and p.get("container"):
            try:
                return int(p["container"])
            except (TypeError, ValueError):
                pass
    for p in svc.get("ports") or []:
        if isinstance(p, dict) and p.get("target"):
            try:
                return int(p["target"])
            except (TypeError, ValueError):
                pass
        elif isinstance(p, str):
            # "host:container[/proto]"
            part = p.split("/")[0]
            bits = part.split(":")
            if len(bits) >= 2:
                try:
                    return int(bits[-1])
                except ValueError:
                    pass
    try:
        return int(meta.get("port_map") or 0)  # last resort: host port
    except (TypeError, ValueError):
        return 0


def _reserved_ram_mb(svc: dict) -> int:
    res = (((svc.get("deploy") or {}).get("resources") or {})
           .get("reservations") or {})
    mem = str(res.get("memory") or "")
    if mem.upper().endswith("M"):
        try:
            return int(mem[:-1])
        except ValueError:
            pass
    if mem.upper().endswith("G"):
        try:
            return int(float(mem[:-1]) * 1024)
        except ValueError:
            pass
    return 256
