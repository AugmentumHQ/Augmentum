"""Drift-audit-writer primitive — placeholder.

The drift-audit capability is wired in the identity module
(``CompanionIdentity.refresh_persona_kernel`` computes drift), but
there's no separate audit-writer service yet. This primitive captures
the abstraction so Sprint 3 dispatch can list it; calling it writes a
companion_journal entry tagged ``drift_audit``.
"""

from __future__ import annotations

from typing import Any

from augmentum.companion_runtime.primitives.base import (
    PrimitiveBase,
    PrimitiveContext,
    PrimitiveResult,
)
from augmentum.companion_runtime.primitives.registry import PrimitiveRegistry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class DriftAuditWriterPrimitive(PrimitiveBase):
    name = "drift_audit_writer"
    description = "Record a persona-drift audit observation to the journal."

    async def call(self, ctx: PrimitiveContext, **kwargs: Any) -> PrimitiveResult:
        runtime = ctx.runtime
        memory = getattr(runtime, "memory", None)
        if memory is None:
            return PrimitiveResult(
                ok=False,
                error="drift_audit_writer: runtime has no memory facade",
            )

        identity = getattr(runtime, "identity", None)
        drift_score = kwargs.get("drift_score")
        if drift_score is None and identity is not None:
            drift_score = getattr(identity, "drift_score", None)

        note = kwargs.get("note", "")
        body = f"drift_audit: score={drift_score!r} note={note!r}"

        try:
            entry_id = await memory.journal(
                content=body,
                entry_type="drift_audit",
                user_id=ctx.user_id,
                affect_tag="reflective",
            )
        except Exception as exc:
            log.exception("drift_audit_write_failed", error=str(exc))
            return PrimitiveResult(ok=False, error=f"drift_audit_write_failed: {exc!s}")

        return PrimitiveResult(
            ok=True,
            payload={"entry_id": entry_id, "drift_score": drift_score},
        )


PrimitiveRegistry.register(DriftAuditWriterPrimitive)
