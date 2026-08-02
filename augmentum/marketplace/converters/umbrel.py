"""Umbrel dialect → Augmentum listing.

Input: umbrel-app.yml (name/tagline/category/description/port/repo/
developer/website) + docker-compose.yml (app_proxy sidecar + real
services; APP_PORT on app_proxy names the container web port).
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

_GALLERY = "https://getumbrel.github.io/umbrel-apps-gallery"

_CATEGORY_MAP = {
    "files": "files", "media": "media", "networking": "networking",
    "automation": "automation", "developer": "developer",
    "developer-tools": "developer", "social": "media", "finance": "files",
    "ai": "other", "bitcoin": "other",
}


def convert_umbrel(app_yml: dict, compose_text: str) -> ConversionResult:
    app_id = str(app_yml.get("id") or "").strip()
    compose: dict[str, Any] = yaml.safe_load(compose_text) or {}
    services = real_services(compose)

    reasons: list[str] = []
    review: list[str] = []
    if not app_id:
        reasons.append("umbrel-app.yml has no id")
    if app_yml.get("dependencies"):
        reasons.append(f"depends on other apps: {app_yml['dependencies']}")
    if len(services) != 1:
        reasons.append(
            f"multi-container ({len(services)} services: {sorted(services)})")
    if reasons:
        return ConversionResult(app_id=app_id or "?", eligible=False, reasons=reasons)

    name, svc = next(iter(services.items()))
    reasons.extend(gate_service(name, svc))
    if svc.get("ports"):
        review.append(f"publishes extra host ports: {svc['ports']} — dropped")

    port = _app_port(compose)
    if not (0 < port < 65536):
        reasons.append("no APP_PORT on app_proxy (can't infer web port)")

    env, prompts, env_review = classify_env(normalize_env(svc))
    review.extend(env_review)

    if reasons:
        return ConversionResult(app_id=app_id, eligible=False,
                                reasons=reasons, review=review)

    category = _CATEGORY_MAP.get(str(app_yml.get("category") or "").lower(), "files")
    review.append("browser block defaulted (setup_page/user_set) — verify")
    listing = build_listing(
        app_id=app_id,
        title=str(app_yml.get("name") or app_id),
        category=category,
        tagline=str(app_yml.get("tagline") or ""),
        description=str(app_yml.get("description") or "")[:600],
        image=str(svc.get("image") or ""),
        port=port,
        volumes=container_volumes(svc),
        env=env,
        env_prompts=prompts,
        source_url=str(app_yml.get("repo") or ""),
        developer=str(app_yml.get("developer") or ""),
        website=str(app_yml.get("website") or ""),
        icon_url=f"{_GALLERY}/{app_id}/icon.svg",
        gallery=[f"{_GALLERY}/{app_id}/{n}.jpg" for n in (1, 2, 3)],
        tags=[t for t in [str(app_yml.get("category") or "").lower()] if t],
        ram_mb=256,
        source_store="umbrel",
    )
    return ConversionResult(app_id=app_id, eligible=True,
                            review=review, listing=listing)


def _app_port(compose: dict) -> int:
    proxy = (compose.get("services") or {}).get("app_proxy") or {}
    env = proxy.get("environment") or {}
    if isinstance(env, list):
        env = dict(str(i).partition("=")[::2] for i in env)
    try:
        return int(str(env.get("APP_PORT") or 0))
    except (TypeError, ValueError):
        return 0
