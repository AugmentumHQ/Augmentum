"""Service app manifests — the marketplace's apps-as-data format.

Spec: docs/superpowers/specs/2026-07-18-marketplace-service-os-design.md
(takeaway T2: apps as data, not code). A ``kind: "service"`` listing's
``install_payload`` is a versioned manifest that ONE generic dispatcher
(install_dispatchers._install_service_manifest) turns into a running,
front-doored, credentialed container — replacing the per-kind Python
dispatchers that made every new service a PR to core.

Design rules enforced here:

* **Pinned images only** — a tag or digest is required and ``:latest``
  is rejected. Updates are explicit version bumps in the catalog, never
  a silent drift on restart.
* **Browser-after-install is a gate, not a vibe** (T4) — the ``browser``
  block is REQUIRED. A listing that can't say what the user sees after
  install is rejected at catalog load.
* **Unknown integration hooks warn and no-op** (forward compatibility —
  an old server can install a newer manifest minus its newest hook).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

MANIFEST_VERSION = 1

# Integration hooks this server knows how to wire — populated by the
# hooks package at import time. Manifests may name hooks this server
# doesn't know yet: unknown names warn and no-op (forward compat).
#
# Imported LAZILY (inside parse_manifest) to avoid a circular import
# through hooks → install_dispatchers → manifest. See
# augmentum/marketplace/hooks/__init__.py for the canonical registry.

_BROWSER_AFTER_INSTALL = frozenset({"setup_page", "login", "status"})
_BROWSER_CREDENTIALS = frozenset({"generated", "none", "user_set"})


class ManifestError(ValueError):
    """A manifest failed validation. Message is operator-actionable."""


@dataclass(frozen=True)
class EnvPrompt:
    """One typed question the install modal asks before provisioning."""

    key: str
    label: str
    default: str = ""
    secret: bool = False
    # When True this is a MACHINE secret (session key, API pepper, …) the
    # user should never have to invent. If the install modal leaves it blank,
    # the dispatcher mints a strong random value. Keeps zero-config installs
    # working for images that hard-require such a secret (homebox's
    # HBOX_AUTH_API_KEY_PEPPER, flatnotes' FLATNOTES_SECRET_KEY).
    generate: bool = False


@dataclass(frozen=True)
class ServiceManifest:
    """Validated ``install_payload`` of a ``kind: "service"`` listing."""

    manifest_version: int
    # service block
    service_id: str
    image: str
    internal_port: int
    env: dict[str, str] = field(default_factory=dict)
    env_prompts: tuple[EnvPrompt, ...] = ()
    command: tuple[str, ...] = ()
    volumes: dict[str, str] = field(default_factory=dict)
    media_mount: str = ""
    healthcheck_path: str = "/health"
    healthcheck_timeout_s: int = 60
    name: str = ""
    description: str = ""
    https_port: int = 0
    # browser block (required — T4)
    browser_after_install: str = "status"
    browser_path: str = "/"
    browser_credentials: str = "none"
    # resources block
    ram_mb: int = 0
    disk_mb: int = 0
    gpu: bool = False
    # lifecycle block
    backup_paths: tuple[str, ...] = ()
    update_strategy: str = "recreate"
    # integration block — validated subset of KNOWN_INTEGRATION_HOOKS.
    # Values carry per-hook config (e.g. media_connect: {"provider": ...}).
    integration: dict[str, dict[str, Any]] = field(default_factory=dict)


def _require(payload: dict, key: str, ctx: str) -> Any:
    if key not in payload:
        raise ManifestError(f"manifest {ctx} is missing required field '{key}'")
    return payload[key]


def _validate_image_pinned(image: str) -> None:
    """Reject unpinned images — silent drift is how installs diverge."""
    if "@sha256:" in image:
        return
    # Split a possible registry host (may contain ':port') from the tag.
    tail = image.rsplit("/", 1)[-1]
    if ":" not in tail:
        raise ManifestError(
            f"image '{image}' has no tag — pin a version tag or digest"
        )
    tag = tail.rsplit(":", 1)[1]
    if tag == "latest":
        raise ManifestError(
            f"image '{image}' uses ':latest' — pin a version tag or digest "
            f"(updates are explicit catalog bumps, never restart drift)"
        )


def parse_manifest(payload: Any) -> ServiceManifest:
    """Validate and normalize a raw ``install_payload`` dict.

    Raises :class:`ManifestError` with a message good enough to fix the
    manifest from — these surface to catalog authors, not end users.
    """
    if not isinstance(payload, dict):
        raise ManifestError("install_payload must be a JSON object")

    version = payload.get("manifest_version")
    if version != MANIFEST_VERSION:
        raise ManifestError(
            f"unsupported manifest_version {version!r} "
            f"(this server supports {MANIFEST_VERSION})"
        )

    service = _require(payload, "service", "root")
    if not isinstance(service, dict):
        raise ManifestError("'service' must be an object")
    service_id = str(_require(service, "id", "service")).strip()
    if not service_id or not service_id.replace("-", "").replace("_", "").isalnum():
        raise ManifestError(f"service.id {service_id!r} must be a slug")
    image = str(_require(service, "image", "service")).strip()
    _validate_image_pinned(image)
    try:
        internal_port = int(_require(service, "port", "service"))
    except (TypeError, ValueError) as exc:
        raise ManifestError("service.port must be an integer") from exc
    if not (0 < internal_port < 65536):
        raise ManifestError(f"service.port {internal_port} out of range")

    env = service.get("env") or {}
    if not isinstance(env, dict):
        raise ManifestError("service.env must be an object of strings")
    env = {str(k): str(v) for k, v in env.items()}

    prompts: list[EnvPrompt] = []
    for i, p in enumerate(service.get("env_prompts") or []):
        if not isinstance(p, dict) or not p.get("key"):
            raise ManifestError(f"service.env_prompts[{i}] needs a 'key'")
        prompts.append(EnvPrompt(
            key=str(p["key"]),
            label=str(p.get("label") or p["key"]),
            default=str(p.get("default") or ""),
            secret=bool(p.get("secret", False)),
            generate=bool(p.get("generate", False)),
        ))

    # Optional container command override (argv). Some images need a
    # subcommand to actually serve (ntfy needs `serve`; the bare entrypoint
    # just prints help and exits). Absent → image default CMD.
    command_raw = service.get("command")
    if command_raw is not None and not isinstance(command_raw, list):
        raise ManifestError("service.command must be a list of strings")
    command = tuple(str(c) for c in (command_raw or ()))

    volumes = service.get("volumes") or {}
    if not isinstance(volumes, dict):
        raise ManifestError("service.volumes must be an object")
    volumes = {str(k): str(v) for k, v in volumes.items()}

    hc = service.get("healthcheck") or {}
    if not isinstance(hc, dict):
        raise ManifestError("service.healthcheck must be an object")

    # browser block — REQUIRED (T4: the listing gate).
    browser = _require(payload, "browser", "root")
    if not isinstance(browser, dict):
        raise ManifestError("'browser' must be an object")
    after = str(_require(browser, "after_install", "browser"))
    if after not in _BROWSER_AFTER_INSTALL:
        raise ManifestError(
            f"browser.after_install {after!r} must be one of "
            f"{sorted(_BROWSER_AFTER_INSTALL)}"
        )
    creds = str(browser.get("credentials") or "none")
    if creds not in _BROWSER_CREDENTIALS:
        raise ManifestError(
            f"browser.credentials {creds!r} must be one of "
            f"{sorted(_BROWSER_CREDENTIALS)}"
        )

    resources = payload.get("resources") or {}
    if not isinstance(resources, dict):
        raise ManifestError("'resources' must be an object")

    lifecycle = payload.get("lifecycle") or {}
    if not isinstance(lifecycle, dict):
        raise ManifestError("'lifecycle' must be an object")
    strategy = str(lifecycle.get("update_strategy") or "recreate")
    if strategy not in ("recreate", "manual"):
        raise ManifestError(
            f"lifecycle.update_strategy {strategy!r} must be recreate|manual"
        )

    integration_raw = payload.get("integration") or {}
    if not isinstance(integration_raw, dict):
        raise ManifestError("'integration' must be an object")
    # Lazy import to avoid circular: hooks → install_dispatchers → manifest.
    from augmentum.marketplace.hooks import KNOWN_INTEGRATION_HOOKS
    integration: dict[str, dict[str, Any]] = {}
    for hook, cfg in integration_raw.items():
        if hook not in KNOWN_INTEGRATION_HOOKS:
            # Forward compat: newer manifests on older servers install
            # fine, just without the hook this server doesn't know.
            log.warning(
                "manifest_unknown_integration_hook",
                service_id=service_id, hook=hook,
            )
            continue
        integration[hook] = dict(cfg) if isinstance(cfg, dict) else {}

    return ServiceManifest(
        manifest_version=MANIFEST_VERSION,
        service_id=service_id,
        image=image,
        internal_port=internal_port,
        env=env,
        env_prompts=tuple(prompts),
        command=command,
        volumes=volumes,
        media_mount=str(service.get("media_mount") or ""),
        healthcheck_path=str(hc.get("path") or "/health"),
        healthcheck_timeout_s=int(hc.get("timeout_s") or 60),
        name=str(service.get("name") or service_id),
        description=str(service.get("description") or ""),
        https_port=int(service.get("https_port") or 0),
        browser_after_install=after,
        browser_path=str(browser.get("path") or "/"),
        browser_credentials=creds,
        ram_mb=int(resources.get("ram_mb") or 0),
        disk_mb=int(resources.get("disk_mb") or 0),
        gpu=bool(resources.get("gpu", False)),
        backup_paths=tuple(str(p) for p in (lifecycle.get("backup_paths") or [])),
        update_strategy=strategy,
        integration=integration,
    )


def to_service_definition(m: ServiceManifest):
    """Build a runtime :class:`ServiceDefinition` from a manifest.

    The catalog stops being the only source of definitions — this is
    the seam that turns catalog code into catalog data (T2).
    """
    from augmentum.providers.models import (
        GpuRequirements,
        ServiceCategory,
        ServiceDefinition,
    )

    return ServiceDefinition(
        id=m.service_id,
        name=m.name,
        description=m.description,
        # Media-hook manifests behave exactly like today's media servers;
        # everything else lands in the generic bucket.
        category=(
            ServiceCategory.MEDIA
            if "media_connect" in m.integration
            else ServiceCategory.SERVICE
        ),
        image=m.image,
        internal_port=m.internal_port,
        host_port=0,  # shared-network access only; no host port grab
        https_port=m.https_port,
        env=dict(m.env),
        # Namespace each volume by service id. The manifest declares volumes
        # keyed by a bare role name ("data", "config") which was previously
        # used VERBATIM as the Docker volume name — so every app declaring
        # "data" shared one global volume literally named `data` (n8n's
        # /home/node/.n8n and trilium's /data landed on the same storage,
        # a data-corruption-grade collision). Prefixing with the service id
        # gives each app its own isolated volume. The mount path (value) is
        # unchanged. Uninstall/backup read sd.volumes, so naming stays
        # consistent end to end.
        volumes={f"augmentum_svc_{m.service_id}_{k}": v for k, v in m.volumes.items()},
        health_endpoint=m.healthcheck_path,
        command=list(m.command) or None,
        browser_credentials=m.browser_credentials,
        # GPU reservation. The manifest declares `resources.gpu: true`; the
        # service_manager already emits DeviceRequests/Capabilities:[["gpu"]] off
        # ServiceDefinition.gpu.required (manager.py:_build_container_config).
        # Previously dropped here, so no manifest service could reserve a GPU —
        # foundational for the vLLM engine and every future GPU service.
        gpu=GpuRequirements(required=True) if m.gpu else GpuRequirements(),
        # Host-RAM ceiling inputs. `resources.ram_mb` was parsed and then
        # dropped here, the same way `resources.gpu` was — it now travels so
        # the spawner can raise its category default for a hungry service.
        # It is a MINIMUM, never the cap itself (see providers/models.py).
        min_ram_mb=int(m.ram_mb or 0),
    )
