"""Live adapter — resolve the real context and run one perception pass.

This is the thin wrapper the runtime calls. It pulls the three pieces the pure
orchestrator needs from the live system and hands them to ``perceive_and_dispatch``:

  * **regret multiplier** ← ``feedback.aggregate_bias`` (the user's general
    engagement; <1 = they dismiss her output → the gate quiets);
  * **interruption budget** ← the process-wide ``BUDGET`` store (cap synced from
    ``companion_interruption_budget_per_day``);
  * **presence snapshot** ← ``presence_context.now_context`` (zero new permissions —
    browse/media/presence we already observe).

Gated OFF by default (``companion_perception_enabled``), the same rollout pattern
the initiative engine uses — inert until deliberately flipped. Never raises: a
perception pass is best-effort and must never break the tick loop.
"""

from __future__ import annotations

import time
from typing import Any

from augmentum.companion_runtime.perception.budget import BUDGET, DEFAULT_CAP
from augmentum.companion_runtime.perception.companion_sink import (
    CompanionPerceptionSink,
)
from augmentum.companion_runtime.perception.fusion import FusionContext
from augmentum.companion_runtime.perception.judgment import config_from_settings
from augmentum.companion_runtime.perception.pipeline import (
    PerceptionSink,
    perceive_and_dispatch,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def perception_enabled(settings: Any) -> bool:
    """Master kill-switch — default OFF during rollout (mirrors initiative)."""
    return bool(getattr(settings, "companion_perception_enabled", False))


async def _load_signals(runtime: Any, user_id: str, settings: Any, now: float) -> dict:
    """Fill the fusion signal bag from the L0 acquisition stores. Each stream is
    gated by its own ``companion_perception_acquire_*`` flag (default OFF) and is
    a cheap no-op until the device has actually uploaded data. The fusers read
    out of this bag and stay pure — all I/O happens here."""
    signals: dict[str, Any] = {}
    backend = getattr(runtime, "backend", None)
    if backend is None:
        return signals
    if bool(getattr(settings, "companion_perception_acquire_notifications", False)):
        try:
            from augmentum.companion_runtime.perception.acquisition import (
                recent_notifications,
            )
            signals["notifications"] = await recent_notifications(
                backend, user_id=user_id, now=now,
            )
        except Exception:  # noqa: BLE001 — a stream read must not break the pass
            log.debug("perception_load_notifications_failed", exc_info=True)
    return signals


async def _regret_multiplier(runtime: Any, user_id: str) -> float:
    try:
        from augmentum.companion_runtime import feedback as _fb
        return float(await _fb.aggregate_bias(runtime, user_id=user_id))
    except Exception:  # noqa: BLE001 — no feedback yet / read error → neutral
        log.debug("perception_regret_lookup_failed", exc_info=True)
        return 1.0


async def _snapshot(runtime: Any, user_id: str) -> dict:
    try:
        from augmentum.companion_runtime import presence_context as _pc
        conn = getattr(getattr(runtime, "backend", None), "conn", None)
        app_state = getattr(runtime, "app_state", None)
        return await _pc.now_context(conn, user_id, app_state=app_state)
    except Exception:  # noqa: BLE001 — perception degrades to an empty snapshot
        log.debug("perception_snapshot_failed", exc_info=True)
        return {}


async def evaluate_user(
    runtime: Any,
    *,
    user_id: str,
    in_conversation: bool = False,
    now: float | None = None,
    sink: PerceptionSink | None = None,
    config: Any = None,
    signals: dict | None = None,
) -> dict[str, int]:
    """Run one perception pass for ``user_id``. Returns the per-channel delivered
    counts (empty when disabled / no user / nothing fused). Best-effort."""
    from augmentum.config import settings

    if not perception_enabled(settings) or not user_id:
        return {}

    now = now if now is not None else time.time()

    # Make sure the shipped fusers are registered (idempotent by name). Without
    # this the pass runs but no fuser fires — the no-op-until-fusers state.
    try:
        from augmentum.companion_runtime.perception.fusers import (
            register_builtin_fusers,
        )
        register_builtin_fusers()
    except Exception:  # noqa: BLE001 — a registration error can't break the pass
        log.debug("perception_register_fusers_failed", exc_info=True)

    # Keep the budget cap in sync with settings (settable at runtime).
    cap = int(getattr(settings, "companion_interruption_budget_per_day", DEFAULT_CAP))
    BUDGET.set_cap(cap)

    regret = await _regret_multiplier(runtime, user_id)
    budget_remaining = BUDGET.remaining(user_id, now)
    snapshot = await _snapshot(runtime, user_id)

    # Caller-supplied signals win; otherwise pull from the L0 acquisition stores.
    if signals is None:
        signals = await _load_signals(runtime, user_id, settings, now)

    ctx = FusionContext(
        user_id=user_id, now=now, in_conversation=in_conversation,
        snapshot=snapshot, signals=signals or {},
    )
    the_sink = sink if sink is not None else CompanionPerceptionSink(runtime)
    cfg = config if config is not None else config_from_settings(settings)

    try:
        return await perceive_and_dispatch(
            ctx, regret_multiplier=regret, budget_remaining=budget_remaining,
            sink=the_sink, budget_store=BUDGET, config=cfg,
        )
    except Exception:  # noqa: BLE001 — a pass must never break the tick loop
        log.warning("perception_pass_failed", user_id=user_id, exc_info=True)
        return {}
