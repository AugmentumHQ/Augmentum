"""Per-user resolution of the image panel's active settings.

The image panel persists **per-user**: ``PUT /api/image/active-settings`` writes
``app_state._image_active_settings_by_user[uid]`` and
``settings_store.set_user(uid, "image_active_settings", …)`` (multi-tenant fix,
2026-06). The process-global ``app_state.image_active_settings`` attribute is now
written only by the anonymous / single-user-no-auth path and by startup load.

Therefore every READER must resolve per-user too. Reading the global mirror for
an authenticated user silently ignores that user's panel choices — most visibly
the selected **model**, which then falls through to ``settings.image_default_model``
(the install default) and produces the "uses the last-installed model, not the
one I selected" regression. This helper is the single source of truth for that
resolution so the read path can never drift from the write path again; it mirrors
``GET /api/image/active-settings`` exactly.
"""

from __future__ import annotations

import json
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


async def resolve_active_settings(app_state: Any, user_id: str = "") -> dict:
    """Return the image panel settings for *user_id* (the dict the panel saved).

    Resolution order mirrors the GET route:
      1. authenticated → in-memory per-user cache, then the persisted per-user
         row (``settings_store.get_user``); never the global mirror.
      2. anonymous / single-user-no-auth → the process-global attribute.

    Always returns a dict (``{}`` when nothing is stored), never raises.
    """
    if app_state is None:
        return {}

    if user_id:
        by_user = getattr(app_state, "_image_active_settings_by_user", None) or {}
        cached = by_user.get(user_id)
        if cached is not None:
            return cached or {}
        store = getattr(app_state, "settings_store", None)
        if store is not None:
            try:
                raw = await store.get_user(user_id, "image_active_settings")
                if raw:
                    return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {}
            except Exception:
                log.warning(
                    "image_active_settings_load_failed",
                    user_id=user_id, exc_info=True,
                )
        return {}

    # Anonymous / single-user-no-auth: the global mirror IS this user's store.
    return getattr(app_state, "image_active_settings", None) or {}
