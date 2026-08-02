"""XR-scene primitive — server-side scene state.

The XR scene's rendering lives in JS (``avatar-fsm.js``,
``avatar-spatial-director.js`` etc.). Server side, the canonical
companion scene lives in the ``companion_scene`` table (migration 157)
and is mediated by ``CompanionRuntime``. This primitive reads & writes
that row through the runtime's state/memory facades.

Action verbs:
- ``get_scene`` — return current scene blob, location, posture.
- ``set_location`` — change location (e.g. ``main_room`` → ``study``).
- ``set_posture`` — change posture (``seated`` / ``standing`` / ``walking``).
"""

from __future__ import annotations

from typing import Any

from augmentum.companion_runtime.primitives.base import (
    PrimitiveBase,
    PrimitiveContext,
    PrimitiveResult,
)
from augmentum.companion_runtime.primitives.registry import PrimitiveRegistry
from augmentum.state.backends.sqlite import transactional_write
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class XRScenePrimitive(PrimitiveBase):
    name = "xr_scene"
    description = (
        "Read/write the companion scene row (location, posture, scene "
        "blob). Rendering happens browser-side from this state."
    )

    async def call(self, ctx: PrimitiveContext, **kwargs: Any) -> PrimitiveResult:
        action = kwargs.get("action", "get_scene")
        runtime = ctx.runtime
        backend = getattr(runtime, "backend", None)
        if backend is None:
            return PrimitiveResult(ok=False, error="xr_scene: no backend on runtime")

        try:
            # backend.connect() returns None (initializer, not a context
            # manager) — use the live connection under transactional_write
            # so the set_* writes commit/rollback cleanly (audit
            # 2026-06-17). The inner explicit commits are now redundant
            # but harmless.
            async with transactional_write(backend.conn) as conn:
                if action == "get_scene":
                    cur = await conn.execute(
                        "SELECT location, posture, last_seen_with, scene_blob "
                        "FROM companion_scene WHERE companion_id = ?",
                        (runtime.companion_id,),
                    )
                    row = await cur.fetchone()
                    if not row:
                        return PrimitiveResult(
                            ok=True, payload=None,
                            metadata={"note": "no scene row yet"},
                        )
                    return PrimitiveResult(ok=True, payload={
                        "location": row[0],
                        "posture": row[1],
                        "last_seen_with": row[2],
                        "scene_blob": row[3],
                    })

                if action == "set_location":
                    loc = kwargs.get("location", "")
                    if not loc:
                        return PrimitiveResult(ok=False, error="xr_scene: empty location")
                    await conn.execute(
                        "UPDATE companion_scene SET location = ? WHERE companion_id = ?",
                        (loc, runtime.companion_id),
                    )
                    await conn.commit()
                    await ctx.bus.publish_topic(
                        "scene.location_changed",
                        {"location": loc},
                        source_companion_id=runtime.companion_id,
                    )
                    return PrimitiveResult(ok=True, payload={"location": loc})

                if action == "set_posture":
                    posture = kwargs.get("posture", "")
                    if not posture:
                        return PrimitiveResult(ok=False, error="xr_scene: empty posture")
                    await conn.execute(
                        "UPDATE companion_scene SET posture = ? WHERE companion_id = ?",
                        (posture, runtime.companion_id),
                    )
                    await conn.commit()
                    await ctx.bus.publish_topic(
                        "scene.posture_changed",
                        {"posture": posture},
                        source_companion_id=runtime.companion_id,
                    )
                    return PrimitiveResult(ok=True, payload={"posture": posture})

                return PrimitiveResult(
                    ok=False, error=f"xr_scene: unknown action {action!r}",
                )
        except Exception as exc:
            log.exception("xr_scene_failed", error=str(exc), action=action)
            return PrimitiveResult(ok=False, error=f"xr_scene_failed: {exc!s}")


PrimitiveRegistry.register(XRScenePrimitive)
