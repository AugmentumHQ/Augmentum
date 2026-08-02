"""Shared compose analysis + eligibility gate for store converters.

"Plug and play" for Augmentum means the app fits the service-manifest
runtime exactly: ONE real container, a web UI port, a pinned image, no
host network / privileged / added caps / devices / docker socket. The
gate reports every violated rule (not just the first) so curation can
see WHY a candidate fell out.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Compose service names that are platform plumbing, not the app.
_PLUMBING_SERVICES = {"app_proxy", "tor", "tor_server", "i2pd"}

# Host-side template variables the source platforms substitute at
# install. Env values referencing PLATFORM state can't travel as-is.
_PLATFORM_VAR_RE = re.compile(
    r"\$\{?(APP_PASSWORD|APP_SEED|DEVICE_HOSTNAME|DEVICE_DOMAIN_NAME|"
    r"APP_PROXY_PORT|APP_PROTOCOL|APP_DOMAIN|APP_HOST|LOCAL_IP|"
    r"INTERNAL_IP|PUBLIC_IP|TIPI_|RUNTIPI_|APP_DATA_DIR|APP_ID|AppID|"
    r"UMBREL_ROOT|APP_LIGHTNING_NODE|APP_BITCOIN)",
)

# Secrets the platform would auto-generate → we ask the user instead.
_SECRET_VAR_RE = re.compile(r"\$\{?(APP_PASSWORD|APP_SEED)\b")


@dataclass
class ConversionResult:
    app_id: str
    eligible: bool
    reasons: list[str] = field(default_factory=list)   # why NOT eligible
    review: list[str] = field(default_factory=list)    # needs a human eye
    listing: dict[str, Any] | None = None


def real_services(compose: dict) -> dict[str, dict]:
    """Compose services minus platform plumbing sidecars."""
    services = compose.get("services") or {}
    return {
        name: (svc or {})
        for name, svc in services.items()
        if name not in _PLUMBING_SERVICES
    }


def image_is_pinned(image: str) -> bool:
    if "@sha256:" in image:
        return True
    if ":" not in image.rsplit("/", 1)[-1]:
        return False
    tag = image.rsplit(":", 1)[-1]
    return tag not in ("latest", "stable", "main", "master", "nightly", "dev")


def strip_digest(image: str) -> str:
    """Keep the human-meaningful tag; drop the @sha256 digest suffix."""
    return image.split("@", 1)[0]


def gate_service(name: str, svc: dict) -> list[str]:
    """Return every plug-and-play rule this single service violates."""
    reasons: list[str] = []
    image = str(svc.get("image") or "")
    if not image:
        reasons.append(f"service '{name}' has no image (build-from-source)")
    elif not image_is_pinned(image):
        reasons.append(f"unpinned image tag: {image}")
    if svc.get("network_mode") == "host":
        reasons.append("requires host networking")
    if svc.get("privileged"):
        reasons.append("requires privileged mode")
    if svc.get("cap_add"):
        reasons.append(f"requires added capabilities: {svc['cap_add']}")
    if svc.get("devices"):
        reasons.append(f"requires host devices: {svc['devices']}")
    for vol in _iter_volume_strings(svc):
        if "docker.sock" in vol:
            reasons.append("mounts the docker socket")
    if svc.get("entrypoint") and isinstance(svc.get("entrypoint"), str) \
            and ("/bin/sh" in svc["entrypoint"] or ".sh" in svc["entrypoint"]):
        # Custom inline entrypoints usually depend on platform-shipped
        # scripts we don't have; flag rather than hard-refuse.
        pass
    return reasons


def _iter_volume_strings(svc: dict):
    for vol in svc.get("volumes") or []:
        if isinstance(vol, str):
            yield vol
        elif isinstance(vol, dict):
            yield f"{vol.get('source', '')}:{vol.get('target', '')}"


def container_volumes(svc: dict) -> dict[str, str]:
    """Extract container-side persistence paths as our volumes map
    (name → container path). Host-side source paths are platform
    template junk and deliberately dropped."""
    out: dict[str, str] = {}
    for vol in svc.get("volumes") or []:
        target = ""
        if isinstance(vol, str):
            parts = vol.split(":")
            # "host:container[:mode]" or a bare named path
            target = parts[1] if len(parts) >= 2 else parts[0]
        elif isinstance(vol, dict):
            target = str(vol.get("target") or "")
        target = target.strip()
        if not target or not target.startswith("/") or "docker.sock" in target:
            continue
        name = re.sub(r"[^a-z0-9]+", "-", target.strip("/").lower()).strip("-") or "data"
        # Keep names short: last path segment wins when unique.
        short = target.rstrip("/").rsplit("/", 1)[-1].lower() or "data"
        short = re.sub(r"[^a-z0-9]+", "-", short).strip("-") or "data"
        key = short if short not in out else name
        out[key] = target
    return out


def classify_env(env: dict) -> tuple[dict[str, str], list[dict], list[str]]:
    """Split source env into (safe env, env_prompts, review flags).

    - Values with platform secret vars (APP_PASSWORD/APP_SEED) become
      secret env_prompts — the user chooses, we never auto-generate
      silently.
    - Values with other platform vars are DROPPED with a review flag
      (they reference host state we can't reproduce).
    - Plain values pass through.
    """
    safe: dict[str, str] = {}
    prompts: list[dict] = []
    review: list[str] = []
    for key, raw in (env or {}).items():
        val = str(raw)
        if _SECRET_VAR_RE.search(val):
            prompts.append({
                "key": str(key),
                "label": _label_for_env(str(key)),
                "secret": True,
            })
        elif _PLATFORM_VAR_RE.search(val):
            review.append(f"dropped env {key}={val} (platform variable)")
        elif "${" in val or val.startswith("$"):
            # ANY unresolved template reference is host-side state we
            # can't reproduce — shipping it verbatim would hand the
            # container a literal "${VAR}" string. Drop + flag; where
            # the source declares a matching form field, the caller's
            # env_prompt covers the key at install time.
            review.append(f"dropped env {key}={val} (unresolved template)")
        else:
            safe[str(key)] = val
    return safe, prompts, review


def _label_for_env(key: str) -> str:
    pretty = key.lower().replace("_", " ").strip()
    if "password" in pretty:
        return "Password"
    if "secret" in pretty or "seed" in pretty or "key" in pretty:
        return "Secret key (any long random string)"
    return pretty.capitalize()


def normalize_env(svc: dict) -> dict:
    """Compose env can be a dict or a KEY=VAL list."""
    env = svc.get("environment") or {}
    if isinstance(env, dict):
        return env
    out = {}
    for item in env:
        k, _, v = str(item).partition("=")
        out[k] = v
    return out


def build_listing(
    *, app_id: str, title: str, category: str, tagline: str,
    description: str, image: str, port: int, volumes: dict[str, str],
    env: dict[str, str], env_prompts: list[dict], source_url: str,
    developer: str, website: str, icon_url: str, gallery: list[str],
    tags: list[str], ram_mb: int, source_store: str,
) -> dict[str, Any]:
    """Assemble a listing in our catalog schema. The browser block is a
    DEFAULT (setup_page / user_set) — callers must surface that in
    review flags; it's the one field no source dialect carries."""
    service: dict[str, Any] = {
        "id": app_id,
        "name": title,
        "image": strip_digest(image),
        "port": int(port),
        "volumes": volumes or {"data": "/data"},
        "healthcheck": {"path": "/", "timeout_s": 90},
    }
    if env:
        service["env"] = env
    if env_prompts:
        service["env_prompts"] = env_prompts
    meta: dict[str, Any] = {"source_store": source_store}
    if developer:
        meta["developer"] = developer
    if website:
        meta["website"] = website
    if gallery:
        meta["gallery"] = gallery
    return {
        "id": f"mkt:{app_id}",
        "title": title,
        "kind": "service",
        "publisher": "augmentum",
        "tagline": tagline,
        "description": description,
        "thumbnail_url": icon_url,
        "source_url": source_url,
        "metadata": meta,
        "install_via": "service_manifest",
        "install_payload": {
            "manifest_version": 1,
            "service": service,
            "browser": {
                "after_install": "setup_page",
                "path": "/",
                "credentials": "user_set",
            },
            "resources": {
                "ram_mb": int(ram_mb or 256),
                "disk_mb": 300,
            },
            "lifecycle": {
                "backup_paths": sorted((volumes or {"data": "/data"}).values()),
            },
        },
        "category": category,
        "tags": tags,
    }
