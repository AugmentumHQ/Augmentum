"""Per-user enabled/active state for Augmentum Powers."""

from __future__ import annotations

import json
import time
from typing import Any

from augmentum.powers.models import PowerActivation

_ENABLED_KEY = "powers_enabled_v1"
_ACTIVE_KEY = "powers_active_v1"
_DEFAULT_WORKSPACE = "__default__"

# The ``/api/powers`` endpoint polls every ~30s, which fires
# ``get_enabled_map`` → ``_get_json`` → ``_get_raw`` → ``settings_store.get_user``
# → ``SELECT value FROM user_settings WHERE user_id=? AND key=?``. Measured at
# ~130ms per call on the live system (slow_db_op log) — small individually,
# but it stacks behind any concurrent write fsync on the shared aiosqlite
# worker thread. Cache the parsed dict per (user_id, key); mutations bust
# the cache via ``_invalidate_json_cache`` so we serve fresh state right
# after a toggle. TTL is generous (300s) because:
#   - state only changes on user action (toggling a power), and we
#     invalidate on every write through this store
#   - cross-process changes don't apply here (single-worker uvicorn)
#   - the poll cadence is 30s, so a 5s TTL was useless
_CACHE_TTL_S = 300.0


class PowerStateStore:
    """Persist enabled/active Power state via SettingsStore."""

    def __init__(self, settings_store: Any) -> None:
        self._settings = settings_store
        cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] | None = None
        if settings_store is not None:
            existing = getattr(settings_store, "_augmentum_power_json_cache", None)
            if isinstance(existing, dict):
                cache = existing
            else:
                cache = {}
                try:
                    settings_store._augmentum_power_json_cache = cache
                except Exception:
                    pass
        # Shared per underlying SettingsStore instance. ``power_routes`` creates
        # short-lived PowerStateStore wrappers per request, so this preserves the
        # poll fast path without leaking state across separate stores/tests.
        self._json_cache = cache if cache is not None else {}

    def _invalidate_json_cache(self, user_id: str, key: str) -> None:
        self._json_cache.pop((user_id, key), None)

    async def _get_raw(self, user_id: str, key: str) -> str | None:
        if not self._settings:
            return None
        if user_id:
            return await self._settings.get_user(user_id, key)
        return await self._settings.get(key)

    async def _set_raw(self, user_id: str, key: str, value: str | None) -> None:
        if not self._settings:
            return
        if user_id:
            await self._settings.set_user(user_id, key, value)
        else:
            await self._settings.set(key, value)
        # Mutation invalidates the read cache for this (user_id, key) so
        # the next get sees the new state. Mutations are rare (toggle
        # button / activate power); reads hit on every /api/powers poll.
        self._invalidate_json_cache(user_id, key)

    async def _get_json(self, user_id: str, key: str) -> dict[str, Any]:
        # TTL cache to keep the recurring /api/powers poll off the DB.
        cache_key = (user_id, key)
        now = time.monotonic()
        cached = self._json_cache.get(cache_key)
        if cached is not None and (now - cached[0]) < _CACHE_TTL_S:
            return dict(cached[1])

        raw = await self._get_raw(user_id, key)
        if not raw:
            data: dict[str, Any] = {}
        else:
            try:
                parsed = json.loads(raw)
                data = parsed if isinstance(parsed, dict) else {}
            except Exception:
                data = {}
        self._json_cache[cache_key] = (now, dict(data))
        return dict(data)

    async def is_enabled(self, user_id: str, power_id: str) -> bool:
        state = await self._get_json(user_id, _ENABLED_KEY)
        return state.get(power_id, True) is not False

    async def set_enabled(self, user_id: str, power_id: str, enabled: bool) -> None:
        state = dict(await self._get_json(user_id, _ENABLED_KEY))
        state[power_id] = bool(enabled)
        await self._set_raw(user_id, _ENABLED_KEY, json.dumps(state))

    async def get_enabled_map(self, user_id: str) -> dict[str, bool]:
        state = await self._get_json(user_id, _ENABLED_KEY)
        return {str(k): bool(v) for k, v in state.items()}

    async def activate_power(
        self,
        user_id: str,
        *,
        workspace_id: str,
        power_id: str,
        source: str = "manual",
        scope: str = "workspace",
        reason: str = "",
    ) -> PowerActivation:
        activations = dict(await self._get_json(user_id, _ACTIVE_KEY))
        key = workspace_id or _DEFAULT_WORKSPACE
        activations[key] = {
            "power_id": power_id,
            "workspace_id": workspace_id,
            "source": source,
            "scope": scope,
            "reason": reason,
        }
        await self._set_raw(user_id, _ACTIVE_KEY, json.dumps(activations))
        return PowerActivation(
            power_id=power_id,
            workspace_id=workspace_id,
            source=source,
            scope=scope,
            reason=reason,
        )

    async def get_active_power(
        self, user_id: str, *, workspace_id: str,
    ) -> PowerActivation | None:
        activations = dict(await self._get_json(user_id, _ACTIVE_KEY))
        payload = activations.get(workspace_id or _DEFAULT_WORKSPACE)
        if not isinstance(payload, dict):
            return None
        power_id = str(payload.get("power_id", "")).strip()
        if not power_id:
            return None
        return PowerActivation(
            power_id=power_id,
            workspace_id=str(payload.get("workspace_id", workspace_id)),
            source=str(payload.get("source", "manual") or "manual"),
            scope=str(payload.get("scope", "workspace") or "workspace"),
            reason=str(payload.get("reason", "") or ""),
        )

    async def clear_active_power(self, user_id: str, *, workspace_id: str) -> None:
        activations = await self._get_json(user_id, _ACTIVE_KEY)
        key = workspace_id or _DEFAULT_WORKSPACE
        if key in activations:
            activations.pop(key, None)
            await self._set_raw(user_id, _ACTIVE_KEY, json.dumps(activations))
