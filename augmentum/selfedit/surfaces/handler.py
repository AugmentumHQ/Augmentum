"""One reshape handler the verb AND the route both call — ask → engine →
present().

Centralizing this is what keeps every surface IDENTICAL: chat, voice, the
companion verb, and the HTTP route all go through ``handle_reshape_ask`` and get
the same ``ReshapePresentation`` back, so there's no per-surface drift in what the
user sees or what gets recorded. ``classify`` (the model-source-backed classifier)
and the DB ``conn`` (for the archive recorder) are injected by the caller, so this
stays testable.
"""

from __future__ import annotations

from typing import Any

from augmentum.selfedit.surfaces.engine import ReshapeRequest, run_reshape_request
from augmentum.selfedit.surfaces.presentation import ReshapePresentation, present
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


async def handle_reshape_ask(ask: str, actor: str, *, classify: Any, conn: Any = None,
                             surface_hint: str = "") -> ReshapePresentation:
    """Run a natural-language reshape ask end to end and return the one
    presentation every surface renders. Records to the never-pruned archive when
    a ``conn`` is given (terminal outcomes; applied-pending left open)."""
    on_start = on_finish = None
    if conn is not None:
        from augmentum.selfedit.surfaces.live import build_store_recorder
        on_start, on_finish = build_store_recorder(conn)

    result = await run_reshape_request(
        ReshapeRequest(ask=ask, actor=actor, surface_hint=surface_hint),
        classify=classify, on_start=on_start, on_finish=on_finish)
    return present(result)
