"""Drift-audit scheduler.

The drift detector seam is already in :mod:`augmentum.companion_runtime.identity`:

  - :meth:`CompanionIdentity.refresh_persona_kernel` re-digests the on-disk
    personality doc, recomputes the embedding, checks against
    :const:`DRIFT_CEILING`, and persists ``drift_score`` + ``last_kernel_refresh_at``
    into ``companion_identities``.
  - :meth:`CompanionIdentity.compute_drift` is the read-only sibling.

What was missing is anything that *invokes* a periodic rehearsal. Without
one, ``drift_score`` only updates when a human operator manually triggers
a refresh, defeating the point of a drift detector. This module fills that
gap. It runs inside the existing tick loop (no extra task), self-gates on
``companion_drift_audit_enabled`` + ``companion_drift_audit_interval_hours``,
and is no-op when the on-disk personality doc is missing.

Idempotent: ``run_if_due`` reads ``last_kernel_refresh_at`` to decide
whether the next audit window has elapsed; if not, it returns False
without touching the DB.

Design spec: docs/superpowers/specs/2026-05-14-companion-runtime-design-v2.md §10.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from augmentum.companion_runtime.identity import DriftCeilingExceeded
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


_HOUR_S = 3600.0


def _parse_db_timestamp(s: str | None) -> float | None:
    """Parse a SQLite ``datetime('now')`` ISO string to a unix timestamp.

    Returns None on missing or malformed input. SQLite writes UTC with a
    space separator and no timezone suffix ("2026-05-17 14:23:45"), so
    treat it as UTC.
    """
    if not s:
        return None
    try:
        # Accept both space and 'T' separators, with or without microseconds.
        norm = s.replace("T", " ").split(".", 1)[0]
        dt = datetime.strptime(norm, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return None


async def run_if_due(runtime: "CompanionRuntime") -> bool:
    """Run a drift-audit rehearsal if the interval has elapsed.

    Returns True when a rehearsal actually fired (and updated
    ``drift_score``). False when skipped — gate off, not yet due, or
    no on-disk personality doc.
    """
    from augmentum.config import settings

    if not getattr(settings, "companion_drift_audit_enabled", True):
        return False

    interval_h = float(getattr(settings, "companion_drift_audit_interval_hours", 24.0))
    interval_s = max(interval_h, 0.5) * _HOUR_S  # clamp to 30 min floor

    identity = runtime.identity
    if identity._row is None:
        # Identity hasn't loaded — runtime.start() hasn't completed.
        return False

    last_refresh_s = _parse_db_timestamp(identity._row.get("last_kernel_refresh_at"))
    if last_refresh_s is None:
        # Never refreshed. The runtime's auto-refresh-on-start already
        # populated it if the doc exists; if we're here with None, the
        # doc must be missing — return without thrashing.
        return False

    if (time.time() - last_refresh_s) < interval_s:
        # Not due yet.
        return False

    # Personality doc is the input — bail cleanly if absent (tests, dev
    # fixtures). The runtime startup path already logs companion_persona_doc_missing
    # in this case, so we just no-op here.
    if not identity.personality_doc_path.is_file():
        return False

    log.info(
        "drift_audit_starting",
        companion_id=runtime.companion_id,
        interval_h=interval_h,
        prior_drift=identity.drift_score,
    )
    try:
        await identity.refresh_persona_kernel()  # ceiling-checked
    except DriftCeilingExceeded as exc:
        # Doc has drifted past the cap — refuse silently, but emit a
        # bus event so an operator surface can flag it for review.
        log.warning(
            "drift_audit_ceiling_exceeded",
            companion_id=runtime.companion_id,
            error=str(exc),
        )
        await runtime.bus.publish_topic(
            "drift.audit_ceiling_exceeded",
            {
                "companion_id": runtime.companion_id,
                "error": str(exc),
                "ts": time.time(),
            },
            source_companion_id=runtime.companion_id,
        )
        return False
    except FileNotFoundError:
        # Race between is_file() and read() — treat as no-op.
        return False
    except Exception:
        log.exception("drift_audit_refresh_failed")
        return False

    # Emit on success. Consumers: telemetry, drift-audit push (if
    # companion_notify_drift_audit_push is on), debug surfaces.
    await runtime.bus.publish_topic(
        "drift.audit_run",
        {
            "companion_id": runtime.companion_id,
            "drift_score": identity.drift_score,
            "doc_version": identity._row.get("personality_doc_version") if identity._row else 0,
            "ts": time.time(),
        },
        source_companion_id=runtime.companion_id,
    )
    log.info(
        "drift_audit_completed",
        companion_id=runtime.companion_id,
        drift_score=identity.drift_score,
    )
    return True


__all__ = ["run_if_due", "_parse_db_timestamp"]
