"""Post-install provider bridge — make a freshly-enabled managed service
actually usable, for every provider category, without a restart.

The gap this closes: installing a provider service through the
marketplace/Discover UI calls ``ServiceManager.enable_service`` — which
starts the container and writes a ``managed_services`` row — and then
STOPS. Nothing ever registered the running service as a *provider*
(``audio_providers`` / ``image_providers`` / an LLM backend), so it never
appeared in any picker. The only registration path,
``_auto_register_audio_providers``, runs at startup and reads
``settings.*_url`` env values (the compose path), never the
``managed_services`` table. So a UI-installed service was invisible
forever — the symptom Matt hit with speaches.

This module is the missing link. Given a ``ServiceDefinition`` (catalog)
+ the running ``ManagedService``, it:

  1. Resolves the service's reachable URL from the ``augmentum_env``
     template (``http://{container_name}:{internal_port}``).
  2. Registers it in the right place for its category:
       * ``stt`` / ``tts`` → upsert ``audio_providers`` (read live on
         every audio request — appears immediately, no restart).
       * ``image``        → upsert ``image_providers``.
       * ``llm``          → persist the URL setting AND hot-register the
         backend in the live ``ProviderRegistry`` so routing sees it now.
  3. Computes the model it will serve (from the catalog's preload env)
     and a list of concrete next steps for the UI's post-install card.

It is deliberately catalog-driven: no per-service special-casing, so the
moment a new provider service lands in ``catalog.json`` it gets full
install→register→present handling for free.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from augmentum.providers.models import ServiceCategory, ServiceDefinition
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_ENV_PREFIX = "AUGMENTUM_"

# Sensible per-api_type defaults so a freshly-registered provider is never
# left with an empty model/voice the UI can't render. Overridden by the
# catalog's own env (PRELOAD_MODELS etc.) whenever present.
_DEFAULT_MODEL_BY_API_TYPE: dict[str, str] = {
    "openai_tts": "kokoro",
    "chatterbox_tts": "chatterbox",
    "fish_tts": "fish-speech",
}
_DEFAULT_VOICE_BY_API_TYPE: dict[str, str] = {
    "openai_tts": "af_heart",
    "chatterbox_tts": "",
    "fish_tts": "",
}
# Which live ProviderRegistry backend key + class to construct for an LLM
# service, keyed by the catalog ``api_type``.
_LLM_BACKEND_KEY_BY_API_TYPE: dict[str, str] = {
    "ollama": "ollama",
    "openai_llm": "llamacpp",
}


@dataclass
class NextStep:
    """One concrete action the UI offers on the post-install card."""

    label: str
    action: str  # set_default | open_webui | test | pick_model | view_logs | retry
    detail: str = ""
    url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "action": self.action,
                "detail": self.detail, "url": self.url}


@dataclass
class ProviderRegistration:
    """Result of bridging one managed service into a usable provider."""

    service_id: str
    category: str
    provider_type: str          # stt | tts | image | llm
    provider_id: str            # row id (== service_id) or backend key
    base_url: str
    default_model: str = ""
    default_voice: str = ""
    registered: bool = False
    target: str = ""            # audio_providers | image_providers | settings
    is_default: bool = False
    detail: str = ""            # human note (esp. on failure)
    next_steps: list[NextStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_id": self.service_id,
            "category": self.category,
            "provider_type": self.provider_type,
            "provider_id": self.provider_id,
            "base_url": self.base_url,
            "default_model": self.default_model,
            "default_voice": self.default_voice,
            "registered": self.registered,
            "target": self.target,
            "is_default": self.is_default,
            "detail": self.detail,
            "next_steps": [s.to_dict() for s in self.next_steps],
        }


def resolve_service_url(sd: ServiceDefinition) -> tuple[str, str]:
    """Return ``(settings_key, base_url)`` from the single ``augmentum_env``
    entry.

    The env KEY (e.g. ``AUGMENTUM_STT_PROVIDER_URL``) names the settings
    field once the ``AUGMENTUM_`` prefix is stripped and lowercased
    (``stt_provider_url``). The VALUE is a URL template resolved against
    the container's real name + internal port. ``enable_service`` names
    the container ``augmentum-{id}`` and aliases it ``{id}`` on the shared
    network, so ``augmentum-{id}:{internal_port}`` is reachable.
    """
    if not sd.augmentum_env:
        return "", ""
    env_key, template = next(iter(sd.augmentum_env.items()))
    settings_key = env_key
    if settings_key.startswith(_ENV_PREFIX):
        settings_key = settings_key[len(_ENV_PREFIX):]
    settings_key = settings_key.lower()
    base_url = (
        str(template)
        .replace("{container_name}", f"augmentum-{sd.id}")
        .replace("{internal_port}", str(sd.internal_port))
    )
    return settings_key, base_url


def extract_default_model(sd: ServiceDefinition) -> str:
    """Best-effort default model for the provider row, derived from the
    catalog's own env — the same value that tells the container which
    model to download/preload, so the provider row and the running
    container agree on what's served.

    Handles speaches' ``PRELOAD_MODELS='["...]'`` JSON list, common
    ``MODEL`` env hints, and falls back to an api_type default.
    """
    env = sd.env or {}
    preload = env.get("PRELOAD_MODELS") or env.get("AUGMENTUM_STT_MODEL")
    if preload:
        try:
            parsed = json.loads(preload)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0]).strip()
            if isinstance(parsed, str) and parsed.strip():
                return parsed.strip()
        except (json.JSONDecodeError, TypeError):
            return str(preload).strip()
    for k in ("MODEL", "DEFAULT_MODEL", "MODEL_NAME"):
        if env.get(k):
            return str(env[k]).strip()
    return _DEFAULT_MODEL_BY_API_TYPE.get(sd.api_type, "")


def _category(sd: ServiceDefinition) -> str:
    return sd.category.value if isinstance(sd.category, ServiceCategory) else str(sd.category)


def _webui_url(sd: ServiceDefinition) -> str:
    """Best-effort host-side web UI URL. Many services expose a UI on
    their host port; the card links to it so the user can manage
    models/voices directly (speaches' model picker lives here)."""
    if not sd.host_port:
        return ""
    return f"http://localhost:{sd.host_port}"


def compute_next_steps(reg: ProviderRegistration, sd: ServiceDefinition) -> list[NextStep]:
    """The concrete, intent-completing actions for the post-install card.

    Ordered by what a user who just pressed Install most likely wants:
    make it the default, open its UI to manage models, then verify it
    works. Surfaces the served model so the user knows what's downloading.
    """
    steps: list[NextStep] = []
    if not reg.registered:
        steps.append(NextStep(
            "Retry registration", "retry",
            detail=reg.detail or "registration did not complete",
        ))
        steps.append(NextStep("View logs", "view_logs"))
        return steps

    if reg.provider_type in ("stt", "tts", "image") and not reg.is_default:
        steps.append(NextStep(
            f"Set as default {reg.provider_type.upper()}", "set_default",
            detail="route requests here by default",
        ))
    webui = _webui_url(sd)
    if webui:
        steps.append(NextStep(
            "Open web UI", "open_webui", url=webui,
            detail="manage models / voices",
        ))
    steps.append(NextStep("Test connection", "test", detail="probe the service + list models"))
    if reg.default_model:
        steps.append(NextStep(
            f"Serving: {reg.default_model}", "pick_model",
            detail="downloads on first use if not cached",
        ))
    return steps


async def _upsert_audio_provider(
    conn: Any, *, provider_id: str, provider_type: str, name: str,
    base_url: str, default_model: str, default_voice: str,
) -> bool:
    """Insert or update an ``audio_providers`` row. Returns whether it is
    the default for its type (first of its kind becomes default)."""
    cur = await conn.execute(
        "SELECT is_default FROM audio_providers WHERE id = ?", (provider_id,),
    )
    existing = await cur.fetchone()
    if existing is not None:
        await conn.execute(
            "UPDATE audio_providers SET name = ?, base_url = ?, "
            "default_model = COALESCE(NULLIF(?, ''), default_model), "
            "default_voice = COALESCE(NULLIF(?, ''), default_voice) "
            "WHERE id = ?",
            (name, base_url, default_model, default_voice, provider_id),
        )
        await conn.commit()
        return bool(existing[0])

    cur2 = await conn.execute(
        "SELECT COUNT(*) FROM audio_providers WHERE provider_type = ?",
        (provider_type,),
    )
    count = (await cur2.fetchone())[0]
    is_default = 1 if count == 0 else 0
    await conn.execute(
        "INSERT INTO audio_providers "
        "(id, provider_type, name, base_url, default_model, default_voice, is_default) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (provider_id, provider_type, name, base_url, default_model, default_voice, is_default),
    )
    await conn.commit()
    return bool(is_default)


async def _upsert_image_provider(
    conn: Any, *, provider_id: str, name: str, base_url: str, default_model: str,
) -> bool:
    """Insert or update an ``image_providers`` row. First image provider
    becomes default."""
    cur = await conn.execute(
        "SELECT is_default FROM image_providers WHERE id = ?", (provider_id,),
    )
    existing = await cur.fetchone()
    if existing is not None:
        await conn.execute(
            "UPDATE image_providers SET name = ?, base_url = ?, "
            "default_model = COALESCE(NULLIF(?, ''), default_model) WHERE id = ?",
            (name, base_url, default_model, provider_id),
        )
        await conn.commit()
        return bool(existing[0])

    cur2 = await conn.execute("SELECT COUNT(*) FROM image_providers")
    count = (await cur2.fetchone())[0]
    is_default = 1 if count == 0 else 0
    await conn.execute(
        "INSERT INTO image_providers (id, name, base_url, default_model, is_default) "
        "VALUES (?, ?, ?, ?, ?)",
        (provider_id, name, base_url, default_model, is_default),
    )
    await conn.commit()
    return bool(is_default)


def _build_llm_backend(api_type: str, base_url: str, http_client: Any) -> tuple[str, Any]:
    """Construct the live ProviderRegistry backend for an LLM service."""
    key = _LLM_BACKEND_KEY_BY_API_TYPE.get(api_type, "")
    if key == "ollama":
        from augmentum.models.ollama import OllamaBackend
        return key, OllamaBackend(http_client, base_url)
    if key == "llamacpp":
        from augmentum.models.llama_cpp import LlamaCppBackend
        return key, LlamaCppBackend(http_client, base_url, "")
    return "", None


async def _register_llm(
    sd: ServiceDefinition, *, base_url: str, settings_key: str,
    settings_store: Any, registry: Any, http_client: Any,
) -> bool:
    """Persist the URL setting (survives restart) AND hot-register the
    backend in the live registry (works now). Returns whether the live
    backend was registered."""
    # Persist for the next boot's _init_backends.
    if settings_store is not None and settings_key:
        try:
            await settings_store.set(settings_key, base_url)
        except Exception:
            log.warning("provider_bridge_settings_persist_failed",
                        service=sd.id, key=settings_key, exc_info=True)
    # Reflect onto the live settings object so anything reading it mid-run
    # is consistent immediately.
    if settings_key:
        try:
            from augmentum.config import settings as _live
            setattr(_live, settings_key, base_url)
        except Exception:
            log.debug("provider_bridge_live_settings_set_failed", key=settings_key)
    # Hot-register the routable backend.
    if registry is not None and http_client is not None:
        key, backend = _build_llm_backend(sd.api_type, base_url, http_client)
        if backend is not None and hasattr(registry, "register_backend"):
            try:
                registry.register_backend(key, backend)
                if hasattr(registry, "invalidate_model_map"):
                    registry.invalidate_model_map()
                return True
            except Exception:
                log.warning("provider_bridge_llm_register_failed",
                            service=sd.id, exc_info=True)
    return False


async def register_provider_for_service(
    sd: ServiceDefinition,
    *,
    conn: Any,
    settings_store: Any = None,
    registry: Any = None,
    http_client: Any = None,
) -> ProviderRegistration:
    """Bridge a running managed service into a usable provider.

    Category-routed, idempotent, and best-effort: a failure to register
    is captured in the returned ``ProviderRegistration`` (registered=False
    + detail + retry next-step) rather than raised, so it never fails the
    install — the container is already up; registration is recoverable.
    """
    category = _category(sd)
    settings_key, base_url = resolve_service_url(sd)
    default_model = extract_default_model(sd)
    default_voice = _DEFAULT_VOICE_BY_API_TYPE.get(sd.api_type, "")

    reg = ProviderRegistration(
        service_id=sd.id,
        category=category,
        provider_type=category,
        provider_id=sd.id,
        base_url=base_url,
        default_model=default_model,
        default_voice=default_voice,
    )

    if not base_url:
        reg.detail = "service has no augmentum_env URL template to register"
        reg.next_steps = compute_next_steps(reg, sd)
        return reg

    try:
        if category in ("stt", "tts"):
            if conn is None:
                raise RuntimeError("no database connection for audio provider upsert")
            reg.is_default = await _upsert_audio_provider(
                conn, provider_id=sd.id, provider_type=category, name=sd.name,
                base_url=base_url, default_model=default_model, default_voice=default_voice,
            )
            reg.target = "audio_providers"
            reg.registered = True
        elif category == "image":
            if conn is None:
                raise RuntimeError("no database connection for image provider upsert")
            reg.is_default = await _upsert_image_provider(
                conn, provider_id=sd.id, name=sd.name, base_url=base_url,
                default_model=default_model,
            )
            reg.target = "image_providers"
            reg.registered = True
        elif category == "llm":
            reg.provider_type = "llm"
            reg.registered = await _register_llm(
                sd, base_url=base_url, settings_key=settings_key,
                settings_store=settings_store, registry=registry, http_client=http_client,
            )
            reg.target = "settings"
            if not reg.registered:
                reg.detail = "LLM URL persisted but live backend not hot-registered"
        else:
            reg.detail = f"unknown service category {category!r}"
    except Exception as exc:  # noqa: BLE001 — never fail the install
        log.warning("provider_bridge_register_failed", service=sd.id,
                    category=category, error=str(exc), exc_info=True)
        reg.detail = f"registration error: {exc}"
        reg.registered = False

    if reg.registered:
        log.info("provider_bridge_registered", service=sd.id, category=category,
                 target=reg.target, base_url=base_url, is_default=reg.is_default)

    reg.next_steps = compute_next_steps(reg, sd)
    return reg


def _resolve_conn(app_state: Any) -> Any:
    """Best-effort aiosqlite write connection from app.state — mirrors
    audio_routes._get_conn so the upsert lands on the same DB the audio
    routes read from."""
    sm = getattr(app_state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm is not None else None
    return getattr(backend, "conn", None)


async def register_installed_service_provider(
    app_state: Any, service_id: str,
) -> ProviderRegistration | None:
    """App-state wrapper: resolve the catalog definition + live deps and
    register. Returns ``None`` only when the service manager / catalog
    can't be resolved (so the caller can no-op cleanly). Any registration
    *failure* still returns a ProviderRegistration with ``registered=False``
    + retry next-steps. Safe to call from every enable path.
    """
    mgr = getattr(app_state, "service_manager", None)
    if mgr is None or not hasattr(mgr, "get_definition"):
        return None
    sd = mgr.get_definition(service_id)
    if sd is None:
        return None
    return await register_provider_for_service(
        sd,
        conn=_resolve_conn(app_state),
        settings_store=getattr(app_state, "settings_store", None),
        registry=getattr(app_state, "provider_registry", None),
        http_client=getattr(app_state, "http_client", None),
    )


async def deregister_installed_service_provider(
    app_state: Any, service_id: str,
) -> bool:
    """Inverse of :func:`register_installed_service_provider`.

    Drops the ``audio_providers`` / ``image_providers`` row this service
    registered (its id is the ``service_id``) so an uninstalled provider
    leaves the pickers — the pickers read those tables live. Best-effort:
    returns ``True`` if a row was removed. A missing connection or row is a
    clean no-op, never an error, so it's safe on every teardown path.
    """
    conn = _resolve_conn(app_state)
    if conn is None:
        return False
    removed = False
    for table in ("audio_providers", "image_providers"):
        try:
            cur = await conn.execute(
                f"DELETE FROM {table} WHERE id = ?",  # noqa: S608 — table from hardcoded tuple
                (service_id,),
            )
            if getattr(cur, "rowcount", 0):
                removed = True
        except Exception:
            log.warning(
                "provider_deregister_failed",
                table=table, service_id=service_id, exc_info=True,
            )
    try:
        await conn.commit()
    except Exception:
        log.warning("provider_deregister_commit_failed", exc_info=True)
    return removed


__all__ = [
    "NextStep",
    "ProviderRegistration",
    "compute_next_steps",
    "deregister_installed_service_provider",
    "extract_default_model",
    "register_installed_service_provider",
    "register_provider_for_service",
    "resolve_service_url",
]
