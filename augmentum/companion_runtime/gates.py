"""Resource and attention gates for the companion runtime.

These are cheap pure checks the tick loop and scorers run before
letting Becca spend tokens, GPU time, or surface anything to the
user. The point is NOT to model the entire system's health — it's to
prevent the four specific failure modes Slice 0 closes:

1. Companion fires a creation while the user is mid-conversation with
   the primary model (KV-cache thrash, latency spike, audible
   degradation). Use :func:`is_primary_busy`.

2. Companion surfaces something while the user just spoke / typed /
   reached out. Use :func:`is_user_recently_active`.

3. Companion fires a heavy creation (image, audio) when one just ran
   or the system is under load. Use :func:`is_heavy_quiet` (and the
   companion runtime's ``heavy_quiet_until`` instance field).

4. User explicitly silenced surfacing. Use :func:`is_hushed_now`.

All gates fail-open: missing app_state, missing runtime, garbage
timestamps all read as "not gated" rather than "permanently muzzled".
A misconfigured deployment must not accidentally silence Becca
forever — that's a worse failure mode than the resource contention
the gates prevent.

Scoring sites consume these by returning a *low but non-zero* utility
when gated, so the softmax still picks something. The right response
to a gate is almost always ``no_op``, never a hard error.
"""

from __future__ import annotations

import time
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def is_primary_busy(runtime: Any) -> bool:
    """True iff the primary llama-server has an in-flight request.

    Reads :attr:`LlamaServerManager.is_busy` via the runtime's loosely
    bound app_state. Returns False on any error (missing manager,
    attribute drift) — fail-open so a misconfigured deployment can't
    accidentally mute the companion.
    """
    if runtime is None:
        return False
    app_state = getattr(runtime, "_app_state", None)
    if app_state is None:
        return False
    mgr = getattr(app_state, "llama_manager", None)
    if mgr is None:
        return False
    try:
        return bool(getattr(mgr, "is_busy", False))
    except Exception:
        return False


def is_user_recently_active(runtime: Any) -> bool:
    """True iff the user spoke / typed / engaged within the cooldown
    window. ``runtime.user_cooldown_until`` is a unix timestamp set by
    input adapters (voice STT finalize, chat send, PTT release) when
    user activity is observed. Default 0.0 means "no cooldown".
    """
    if runtime is None:
        return False
    until = float(getattr(runtime, "user_cooldown_until", 0.0) or 0.0)
    return until > time.time()


def is_heavy_quiet(runtime: Any) -> bool:
    """True iff a heavy creation (image, audio) ran recently and the
    quiet window hasn't elapsed. ``runtime.heavy_quiet_until`` is set
    by the heavy-tier performer after a successful run, providing
    backpressure even when the primary model is idle.
    """
    if runtime is None:
        return False
    until = float(getattr(runtime, "heavy_quiet_until", 0.0) or 0.0)
    return until > time.time()


def is_hushed_now() -> bool:
    """True iff the user has explicitly silenced surfacing until a
    future time. ``companion_journal_hushed_until`` in settings is an
    ISO-8601 timestamp string; empty string = not hushed.

    Accepts both "YYYY-MM-DD HH:MM:SS" and "YYYY-MM-DDTHH:MM:SSZ"
    forms because the UI may write either.
    """
    try:
        from augmentum.config import settings
        raw = (getattr(settings, "companion_journal_hushed_until", "") or "").strip()
    except Exception:
        return False
    if not raw:
        return False
    try:
        from datetime import datetime, timezone
        norm = raw.replace("T", " ").replace("Z", "").split(".", 1)[0]
        until = datetime.strptime(norm, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc,
        )
        return until.timestamp() > time.time()
    except Exception:
        log.debug("hush_parse_failed", raw=raw[:64])
        return False


__all__ = [
    "is_primary_busy",
    "is_user_recently_active",
    "is_heavy_quiet",
    "is_hushed_now",
]
