"""Data models for the provider marketplace / managed Docker services."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field


class ServiceCategory(str, enum.Enum):
    LLM = "llm"
    TTS = "tts"
    STT = "stt"
    IMAGE = "image"
    # Content servers the AI OS can provision as sidecars (Jellyfin,
    # Suwayomi, Audiobookshelf, …). Unlike the inference categories
    # above, a MEDIA service isn't registered as a provider/backend —
    # the install dispatcher provisions the container and auto-creates a
    # per-user user_media_servers connection so it shows up in Files.
    MEDIA = "media"
    # Generic self-hosted service app installed via a marketplace
    # manifest (2026-07-18 service-OS design). Not an inference provider
    # and not media — no provider-bridge or media-connect side effects
    # unless the manifest's integration hooks ask for them.
    SERVICE = "service"


class ServiceStatus(str, enum.Enum):
    STOPPED = "stopped"
    PULLING = "pulling"
    STARTING = "starting"
    RUNNING = "running"
    UNHEALTHY = "unhealthy"
    ERROR = "error"


@dataclass
class HealthCheck:
    """Docker health check configuration."""
    test: list[str]
    interval_s: int = 10
    timeout_s: int = 5
    retries: int = 5
    start_period_s: int = 60


@dataclass
class GpuRequirements:
    """GPU requirements for the service."""
    required: bool = False
    vram_mb: int = 0
    driver: str = "nvidia"


@dataclass
class ServiceDefinition:
    """A provider service that can be managed via Docker API.

    Loaded from catalog.json for pre-configured providers, or built
    dynamically for custom providers.
    """
    id: str
    name: str
    description: str
    category: ServiceCategory
    image: str
    internal_port: int
    host_port: int
    # Dedicated HTTPS front-door port (media servers only). When set,
    # Caddy terminates TLS on this port with the trusted cert and reverse-
    # proxies to the container's internal HTTP, so the server is reachable
    # over real HTTPS in a browser and native apps. 0 = no front door.
    # Must fall inside the range published on the caddy service in
    # compose.yaml. See providers/caddy_front_door.py.
    https_port: int = 0
    env: dict[str, str] = field(default_factory=dict)
    volumes: dict[str, str] = field(default_factory=dict)
    health_check: HealthCheck | None = None
    gpu: GpuRequirements = field(default_factory=GpuRequirements)
    api_type: str = ""
    health_endpoint: str = "/health"
    features: list[str] = field(default_factory=list)
    command: list[str] | None = None
    # How the app authenticates its own users (mirrors the manifest's
    # browser.credentials). Drives gate-mode selection: an app with its OWN
    # login ("user_set"/"generated") gets a PROXY gate (straight TLS proxy —
    # forward_auth would break its SPA/websockets); an app with "none" gets an
    # ACCESS gate (forward_auth is the ONLY thing protecting it).
    browser_credentials: str = "none"
    shm_size: str | None = None
    # OPT-IN host-RAM ceiling for the spawned container, compose-style ("8g").
    # Empty (the default) means NO limit — see providers/manager.py
    # ``_resolve_mem_limit`` for why there is deliberately no default ceiling.
    # Set this only from a measurement of your own deployment; a service's
    # appetite depends on how it is used and by how many people, which neither
    # a manifest nor Augmentum can know in advance.
    mem_limit: str = ""
    # Declared MINIMUM the service needs (manifest ``resources.ram_mb``).
    # An admission-check input — "does this host have room to start it" — and
    # nothing else. It is NOT a ceiling and must never be used to derive one.
    # 0 means the manifest declared nothing.
    min_ram_mb: int = 0
    is_custom: bool = False
    augmentum_env: dict[str, str] = field(default_factory=dict)
    # Explicit default model id (e.g. the STT model speaches should serve).
    # Falls back to extraction from PRELOAD_MODELS-style env when empty.
    default_model: str = ""
    # Optional manifest for providers whose models download on demand and
    # must be pulled after the container is healthy (e.g. speaches — its
    # PRELOAD_MODELS env is ignored on the published Docker image). Keys:
    #   endpoint        POST template to pull a model, "{model}" substituted
    #   list_installed  GET path returning loaded models (OpenAI list shape)
    #   list_available  GET path returning the pullable registry
    model_pull: dict[str, str] = field(default_factory=dict)
    # Optional container ENTRYPOINT override (list form). Used when a stock
    # image needs a wrapper — e.g. fish-tts, whose image doesn't fetch its own
    # model, gets an entrypoint that downloads the chosen checkpoint (if
    # missing) then execs the image's real server. Empty = image default.
    entrypoint: list = field(default_factory=list)
    # Optional install-time requirements surfaced to the user BEFORE provision
    # (provider services have no manifest env_prompts). Recognized keys:
    #   token   {"setting","label","help_url","reason"} — a secret (e.g. a
    #           HuggingFace token) that must be present for a gated download;
    #           collected inline in the install card and saved to `setting`.
    #   license {"id","note"} — surfaced so the user makes an informed choice
    #           (e.g. Fish Speech is CC-BY-NC-SA, non-commercial).
    # gpu.required already covers the GPU dimension. Empty = no gating.
    requirements: dict = field(default_factory=dict)


@dataclass
class ManagedService:
    """A service instance persisted in the database."""
    id: str
    definition_id: str
    name: str
    category: str
    image: str
    container_id: str | None = None
    host_port: int = 0
    internal_port: int = 0
    config_json: str = "{}"
    enabled: bool = False
    status: str = "stopped"
    error: str | None = None
    created_at: str = ""
    updated_at: str = ""
