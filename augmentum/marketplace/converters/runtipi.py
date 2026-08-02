"""Runtipi appstore dialect → Augmentum listing.

Input: the app's ``config.json`` (id, name, port, categories, form
fields) + ``docker-compose.yml`` (schema v2 with per-service
``x-runtipi`` extensions carrying ``internal_port`` / ``is_main``).
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

_RAW = "https://raw.githubusercontent.com/runtipi/runtipi-appstore/master/apps"

# Runtipi categories → our Discover categories.
_CATEGORY_MAP = {
    "network": "networking", "media": "media", "development": "developer",
    "automation": "automation", "utilities": "files", "photography": "media",
    "security": "files", "social": "media", "featured": "files",
    "books": "media", "data": "files", "music": "media", "finance": "files",
    "gaming": "other", "ai": "other",
}


def convert_runtipi(config: dict, compose_text: str) -> ConversionResult:
    app_id = str(config.get("id") or "").strip()
    compose: dict[str, Any] = yaml.safe_load(compose_text) or {}
    services = real_services(compose)

    reasons: list[str] = []
    review: list[str] = []
    if not app_id:
        reasons.append("config.json has no id")
    if len(services) != 1:
        reasons.append(
            f"multi-container ({len(services)} services: {sorted(services)})")
    if config.get("available") is False:
        reasons.append("marked unavailable upstream")

    if reasons:
        return ConversionResult(app_id=app_id or "?", eligible=False, reasons=reasons)

    name, svc = next(iter(services.items()))
    reasons.extend(gate_service(name, svc))

    ext = svc.get("x-runtipi") or {}
    port = int(ext.get("internal_port") or 0)
    if not port:
        # Older schema: "${APP_PORT}:<container>" mappings — the
        # CONTAINER side is the app's real web port. config.json's
        # "port" is the HOST port Runtipi exposes; using it produced
        # wrong manifests (drawio 8734-vs-8080 class), so it's a last
        # resort with a review flag.
        port = _container_port_from_mappings(svc)
    if not port:
        port = int(config.get("port") or 0)
        if port:
            review.append(
                f"port {port} taken from config.json (HOST side) — verify")
    if not (0 < port < 65536):
        reasons.append("no usable internal port")

    env, prompts, env_review = classify_env(normalize_env(svc))
    review.extend(env_review)

    # Runtipi form_fields are typed install questions — exactly our
    # env_prompts. Merge them in (they name the env variable directly).
    for f in config.get("form_fields") or []:
        var = str(f.get("env_variable") or "")
        if not var or any(p["key"] == var for p in prompts):
            continue
        prompts.append({
            "key": var,
            "label": str(f.get("label") or var),
            "secret": str(f.get("type") or "") in ("password", "random"),
            **({"default": str(f["default"])} if f.get("default") not in (None, "") else {}),
        })

    if reasons:
        return ConversionResult(app_id=app_id, eligible=False,
                                reasons=reasons, review=review)

    cats = [str(c) for c in (config.get("categories") or [])]
    category = next(
        (_CATEGORY_MAP[c] for c in cats if c in _CATEGORY_MAP), "files")
    review.append("browser block defaulted (setup_page/user_set) — verify")
    listing = build_listing(
        app_id=app_id,
        title=str(config.get("name") or app_id),
        category=category,
        tagline=str(config.get("short_desc") or ""),
        description=str(config.get("description") or ""),
        image=str(svc.get("image") or ""),
        port=port,
        volumes=container_volumes(svc),
        env=env,
        env_prompts=prompts,
        source_url=str(config.get("source") or ""),
        developer=str(config.get("author") or ""),
        website=str(config.get("website") or ""),
        icon_url=f"{_RAW}/{app_id}/metadata/logo.jpg",
        gallery=[],
        tags=cats[:4],
        ram_mb=256,
        source_store="runtipi",
    )
    return ConversionResult(app_id=app_id, eligible=True,
                            review=review, listing=listing)


def _container_port_from_mappings(svc: dict) -> int:
    for p in svc.get("ports") or []:
        part = str(p).split("/")[0]
        bits = part.split(":")
        if len(bits) >= 2:
            try:
                return int(bits[-1])
            except ValueError:
                continue
    return 0
