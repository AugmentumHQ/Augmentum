"""Provider catalog — loads pre-configured service definitions from JSON."""
from __future__ import annotations

import json
from pathlib import Path

from augmentum.providers.models import (
    GpuRequirements,
    HealthCheck,
    ServiceCategory,
    ServiceDefinition,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_CATALOG_PATH = Path(__file__).parent / "catalog.json"


class ProviderCatalog:
    """Read-only catalog of pre-configured provider service definitions."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _CATALOG_PATH
        self._entries: list[ServiceDefinition] = []
        self._by_id: dict[str, ServiceDefinition] = {}
        self._load()
        # Ids that came from the shipped JSON catalog. Runtime (manifest)
        # definitions must never shadow OR replace these — the shipped
        # catalog stays authoritative for its own ids.
        self._shipped_ids: set[str] = set(self._by_id)

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            log.warning("catalog_load_failed", path=str(self._path))
            return
        for item in raw:
            try:
                hc_data = item.get("health_check")
                hc = HealthCheck(**hc_data) if hc_data else None
                gpu_data = item.get("gpu", {})
                gpu = GpuRequirements(**gpu_data) if gpu_data else GpuRequirements()
                sd = ServiceDefinition(
                    id=item["id"],
                    name=item["name"],
                    description=item.get("description", ""),
                    category=ServiceCategory(item["category"]),
                    image=item["image"],
                    internal_port=item["internal_port"],
                    host_port=item["host_port"],
                    https_port=int(item.get("https_port", 0) or 0),
                    env=item.get("env", {}),
                    volumes=item.get("volumes", {}),
                    health_check=hc,
                    gpu=gpu,
                    api_type=item.get("api_type", ""),
                    health_endpoint=item.get("health_endpoint", "/health"),
                    features=item.get("features", []),
                    command=item.get("command"),
                    shm_size=item.get("shm_size"),
                    mem_limit=item.get("mem_limit", ""),
                    min_ram_mb=int(item.get("ram_mb") or 0),
                    augmentum_env=item.get("augmentum_env", {}),
                    default_model=item.get("default_model", ""),
                    model_pull=item.get("model_pull", {}),
                    entrypoint=item.get("entrypoint", []),
                    requirements=item.get("requirements", {}),
                )
                self._entries.append(sd)
                self._by_id[sd.id] = sd
            except Exception:
                log.warning("catalog_entry_invalid", entry_id=item.get("id", "?"))
        self._validate_https_ports()

    def _validate_https_ports(self) -> None:
        """Guard against front-door port collisions / out-of-range values.

        A duplicate https_port means two media servers would try to bind
        the same Caddy listener (the second silently fails to come up); an
        out-of-range value wouldn't be published on the caddy service. Log
        loudly so a bad catalog edit is caught at startup; the dedicated
        test (test_catalog_https_ports) enforces it hard.
        """
        from augmentum.providers.caddy_front_door import (
            FRONT_DOOR_PORT_MAX,
            FRONT_DOOR_PORT_MIN,
        )

        seen: dict[int, str] = {}
        for sd in self._entries:
            port = getattr(sd, "https_port", 0) or 0
            if not port:
                continue
            if not FRONT_DOOR_PORT_MIN <= port <= FRONT_DOOR_PORT_MAX:
                log.error(
                    "catalog_https_port_out_of_range",
                    service=sd.id, https_port=port,
                    allowed=f"{FRONT_DOOR_PORT_MIN}-{FRONT_DOOR_PORT_MAX}",
                )
            if port in seen:
                log.error(
                    "catalog_https_port_collision",
                    https_port=port, services=[seen[port], sd.id],
                )
            else:
                seen[port] = sd.id

    def register_runtime(self, sd: ServiceDefinition) -> None:
        """Register a definition built at runtime from a marketplace
        service manifest (2026-07-18 apps-as-data design). Runtime
        definitions never shadow catalog entries — the shipped catalog
        stays authoritative for its own ids."""
        if sd.id in self._by_id:
            log.warning("catalog_runtime_shadow_refused", service_id=sd.id)
            return
        self._entries.append(sd)
        self._by_id[sd.id] = sd
        log.info("catalog_runtime_registered", service_id=sd.id, image=sd.image)

    def replace_runtime(self, sd: ServiceDefinition) -> bool:
        """Swap the runtime definition for an already-runtime-registered id
        (a catalog version bump — new image/ports). Returns True if replaced.

        Refuses to touch a shipped-catalog id, or an id that isn't currently
        registered at all. This is the seam a marketplace ``update`` uses so
        an installed manifest app can be recreated on a bumped image without
        re-registering from scratch (which ``register_runtime`` refuses)."""
        if sd.id in self._shipped_ids:
            log.warning("catalog_shipped_replace_refused", service_id=sd.id)
            return False
        if sd.id not in self._by_id:
            return False
        self._by_id[sd.id] = sd
        for i, e in enumerate(self._entries):
            if e.id == sd.id:
                self._entries[i] = sd
                break
        log.info("catalog_runtime_replaced", service_id=sd.id, image=sd.image)
        return True

    def list_all(self) -> list[ServiceDefinition]:
        return list(self._entries)

    def list_by_category(self, category: ServiceCategory) -> list[ServiceDefinition]:
        return [e for e in self._entries if e.category == category]

    def get(self, service_id: str) -> ServiceDefinition | None:
        return self._by_id.get(service_id)
