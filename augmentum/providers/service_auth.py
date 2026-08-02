"""Managed credentials for provisioned sidecar services.

Some provisioned services must NOT run unauthenticated. A media server
like Suwayomi is a web-fetching proxy — left open on the host network it
is a real foothold. So for services flagged ``managed_auth`` we bake a
Basic-auth credential into the container at provision time, derived
deterministically from this install's secret key (``derive_secret``) so
it is:

  - stable across container restarts / recreates (``restore_enabled``
    re-derives the same value, no DB round-trip),
  - reproducible even if the service's own config volume is wiped
    (the image regenerates ``server.conf`` from the same env),
  - unique per install (different ``.secret_key`` → different creds).

Augmentum is the source of truth for the credential; it is never
user-chosen, and env-injected auth can't be changed from the service's
own web UI. This is the single source of truth shared by the
ServiceManager (env injection), the install dispatcher (connect + store
token), and the reveal endpoint (so the user can open the console).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from augmentum.providers.models import ServiceDefinition
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)

# Key under managed_services.config_json holding an optional user-set
# password override (Fernet-encrypted), which takes precedence over the
# derived credential. Lets a user set a memorable login for the server's
# own apps (TV, mobile) without losing the deterministic default.
_OVERRIDE_KEY = "credential_override"

# Catalog feature flags that opt a service into Augmentum-managed credentials.
#   managed_auth      → the server takes a Basic-auth credential via env
#                       (Suwayomi). Applied by the ServiceManager at boot.
#   first_run_wizard  → the server has a first-run setup that creates an
#                       initial admin account (Jellyfin /Startup). Applied
#                       by the install dispatcher after the container is up.
# Both mint the SAME derived (username, password); they differ only in HOW
# the container is told to require it.
MANAGED_AUTH_FEATURE = "managed_auth"
FIRST_RUN_WIZARD_FEATURE = "first_run_wizard"
_MANAGED_AUTH_USERNAME = "augmentum"
# Services whose login identifier must be an email (Komga's claim validates
# the X-Komga-Email header as an email address). The local-domain form is a
# stable, non-routable identity owned by this install.
_EMAIL_USERNAME_SERVICES = frozenset({"komga"})
_MANAGED_AUTH_EMAIL = "augmentum@augmentum.local"


def needs_managed_auth(sd: ServiceDefinition) -> bool:
    return MANAGED_AUTH_FEATURE in (sd.features or [])


def needs_first_run_setup(sd: ServiceDefinition) -> bool:
    return FIRST_RUN_WIZARD_FEATURE in (sd.features or [])


def has_managed_credentials(sd: ServiceDefinition) -> bool:
    """True if Augmentum mints + owns this service's login (either path).

    Used by the reveal endpoint / setup card: both Basic-auth and
    wizard-account servers have a derived credential the user needs to open
    the server's own console.
    """
    return needs_managed_auth(sd) or needs_first_run_setup(sd)


def managed_service_credentials(service_id: str) -> tuple[str, str]:
    """``(username, password)`` for a managed-auth service. Deterministic.

    Derived from the install secret, so every caller (provision env,
    dispatcher login, reveal endpoint) computes the identical credential.
    """
    from augmentum.utils.secrets import derive_secret

    username = (
        _MANAGED_AUTH_EMAIL if service_id in _EMAIL_USERNAME_SERVICES
        else _MANAGED_AUTH_USERNAME
    )
    return (username, derive_secret(f"media-auth:{service_id}", length=32))


def managed_auth_env(sd: ServiceDefinition) -> dict[str, str]:
    """Suwayomi-image ``AUTH_*`` env for a managed-auth service.

    Returns ``{}`` for services that don't opt in, so it's safe to call
    unconditionally in the container-config path. The image's startup
    script maps these to ``server.authMode`` / ``server.authUsername`` /
    ``server.authPassword`` and writes them before the server boots.

    Uses the *derived* password only — see ``managed_auth_env_resolved``
    for the override-aware variant used on the live provision path.
    """
    if not needs_managed_auth(sd):
        return {}
    user, password = managed_service_credentials(sd.id)
    return {
        "AUTH_MODE": "basic_auth",
        "AUTH_USERNAME": user,
        "AUTH_PASSWORD": password,
    }


# ---------------------------------------------------------------------------
# Override-aware resolution (a user-set password takes precedence)
# ---------------------------------------------------------------------------


async def _read_override(service_id: str, db: aiosqlite.Connection | None) -> str:
    """Decrypted password override for a service, or "" if none/unreadable."""
    if db is None:
        return ""
    try:
        cursor = await db.execute(
            "SELECT config_json FROM managed_services WHERE id = ?", (service_id,),
        )
        row = await cursor.fetchone()
        if not row or not row[0]:
            return ""
        data = json.loads(row[0])
        enc = data.get(_OVERRIDE_KEY) if isinstance(data, dict) else ""
        if not enc:
            return ""
        from augmentum.utils.secrets import decrypt_api_key
        return decrypt_api_key(enc) or ""
    except Exception:  # noqa: BLE001 — never let a parse error block auth
        log.warning("credential_override_read_failed", service=service_id, exc_info=True)
        return ""


async def resolve_managed_credentials(
    service_id: str, db: aiosqlite.Connection | None,
) -> tuple[str, str]:
    """``(username, password)`` honoring a stored override.

    Falls back to the deterministic derived password when no override is
    set (or the DB is unavailable), so every read path agrees.
    """
    username, derived = managed_service_credentials(service_id)
    override = await _read_override(service_id, db)
    return (username, override or derived)


async def managed_auth_env_resolved(
    sd: ServiceDefinition, db: aiosqlite.Connection | None,
) -> dict[str, str]:
    """Override-aware ``managed_auth_env`` for the live provision path."""
    if not needs_managed_auth(sd):
        return {}
    user, password = await resolve_managed_credentials(sd.id, db)
    return {"AUTH_MODE": "basic_auth", "AUTH_USERNAME": user, "AUTH_PASSWORD": password}


async def set_credential_override(
    service_id: str, new_password: str, db: aiosqlite.Connection | None,
) -> bool:
    """Persist an encrypted password override into managed_services.config_json.

    Merges into the existing config (preserving augmentum_env /
    volume_overrides). Returns False if there's no row to update or the DB
    is unavailable.
    """
    if db is None or not new_password:
        return False
    try:
        cursor = await db.execute(
            "SELECT config_json FROM managed_services WHERE id = ?", (service_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return False
        data = json.loads(row[0]) if row[0] else {}
        if not isinstance(data, dict):
            data = {}
        from augmentum.utils.secrets import encrypt_api_key
        data[_OVERRIDE_KEY] = encrypt_api_key(new_password) or ""
        await db.execute(
            "UPDATE managed_services SET config_json = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (json.dumps(data), service_id),
        )
        await db.commit()
        return True
    except Exception:  # noqa: BLE001
        log.warning("credential_override_set_failed", service=service_id, exc_info=True)
        return False
