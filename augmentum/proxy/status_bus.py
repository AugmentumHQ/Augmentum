"""Backend stage events — `stage_start` / `stage_complete` shaped helpers.

Surfaces what's happening on the backend during the pre-token portion
of a request (model swap, slot restore, prefill). The chat stream
already yields a primitive ``{"status": "loading"}`` string at these
boundaries; this module produces structured sibling events with stable
ids, started_at timestamps, and computed durations so the frontend can
render "Loading model · deepseek-v3-instruct" with a "completed in
4.2s" follow-up instead of an opaque spinner.

Mirrors the ``tool_start`` / ``tool_progress`` / ``tool_complete``
event family (augmentum/tools/events.py) on purpose — same envelope
shape, same renderer pattern. Stages are the engine analogue of tools.

Why no contextvar writer here:
    The agentic ``progress_bus`` uses a ContextVar to pump tool
    progress events into the streaming layer from arbitrary depth.
    For engine stages we don't need that yet — the only consumer is
    ``LlamaCppBackend.chat_stream`` which IS a generator and yields
    these events directly at the call boundaries. Adding a writer
    + queue would be premature; revisit when post-stream extraction
    or memory-pipeline stages need surfacing.

The ``request_id_var`` ContextVar IS introduced now because it has
a clear single use: every log line on the request path can be
stamped with the same id for correlation, no plumbing required.
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Request ID — set at route entry, read by anything that wants log correlation.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Multi-slot KV — the codebase's current recommended default.
# ---------------------------------------------------------------------------
# Read by ``LlamaCppBackend._multislot_enabled`` and
# ``LlamaServerManager._build_slot_scheduling_args`` when
# ``settings.engine_multislot_enabled`` is ``None`` (the tri-state
# "auto" position — user hasn't expressed a preference). Flip this to
# change the recommendation without touching anyone's explicit setting:
# users who toggled "Always on" or "Always off" keep their choice;
# everyone else moves with the codebase.
#
# History:
#   2026-05-05 first introduced as ``False`` (rollout phase, opt-in).
#   2026-05-06 flipped to ``True`` after dogfood validation: regenerate
#              UX bug fixed (prepare_stable_checkpoint moves to a non-
#              chat slot), tier telemetry shows hot/cold distribution
#              as expected, no observed regressions in single-user flow.
#
# Reverting is a one-line change here — no DB migration, no spec edit.
MULTISLOT_DEFAULT_ENABLED: bool = True


request_id_var: ContextVar[str] = ContextVar("augmentum_request_id", default="")

# ---------------------------------------------------------------------------
# KV tier — set by ``_manage_slot`` when it decides the routing strategy,
# read by ``_log_performance`` to stamp ``engine_perf`` with the tier so
# distribution analysis doesn't need a log JOIN. Multi-slot architecture
# spec: docs/superpowers/specs/2026-05-05-multi-slot-kv-design.md.
#
# Values:
#   "hot"                    — session was already in occupancy at
#                              request time; engine routes via prefix.
#   "cold_with_checkpoint"   — session not in occupancy; on-disk
#                              checkpoint restored before request.
#   "cold_no_checkpoint"     — session not in occupancy, no disk state;
#                              full cold prefill on engine-picked slot.
#   ""                       — unknown / not yet decided / opaque request.
# ---------------------------------------------------------------------------

kv_tier_var: ContextVar[str] = ContextVar("augmentum_kv_tier", default="")


def bind_kv_tier(tier: str) -> Token[str]:
    """Bind the KV tier for the current async context.

    Read by ``_log_performance`` (and any future per-tier analytics)
    via ``kv_tier_var.get()``. Reset implicitly when the task ends —
    no explicit reset needed in the typical request flow.
    """
    return kv_tier_var.set(tier)


def make_request_id() -> str:
    """Generate a short opaque request id.

    8 hex chars is plenty — collision probability is negligible inside
    the operational window of a single process. Short enough to grep
    without eyestrain, distinct enough to find exactly the request you
    care about.
    """
    return uuid.uuid4().hex[:8]


def bind_request_id(req_id: str) -> Token[str]:
    """Bind a request id for the current async context.

    Returns the ContextVar token so the caller can ``reset()`` it after
    the request finishes. Idempotent: passing ``""`` clears the binding
    and matches the default behavior elsewhere on the path.
    """
    return request_id_var.set(req_id)


def reset_request_id(token: Token[str]) -> None:
    """Release a binding established by :func:`bind_request_id`."""
    request_id_var.reset(token)


# ---------------------------------------------------------------------------
# Stage events — shaped to mirror the existing tool_start/tool_complete family.
# ---------------------------------------------------------------------------

# Stable stage names. Add new entries here so the frontend can map them
# to display labels in one place. Names are short snake_case identifiers,
# stable across versions — frontend code dispatches on these.
KNOWN_STAGES = (
    "model_load",      # cold-start llama-server with the requested model
    "model_swap",      # hot-swap to a different model
    "slot_restore",    # restore KV state from saved slot
    "prefill",         # prompt processing on the loaded model
)


@dataclass
class Stage:
    """One backend stage in flight. Holds timing + identity.

    Produces ``augmentum`` payload dicts that the chat stream yields as
    sidecar metadata. Two emit points: :meth:`start_payload` right
    before the work begins, :meth:`complete_payload` right after.
    """

    name: str
    label: str = ""
    detail: str = ""
    # Stable correlation id. Frontend uses this to pair start with
    # complete — needed because two stages can be in flight back-to-
    # back and the renderer dispatches per-id.
    id: str = field(default_factory=lambda: f"stg_{uuid.uuid4().hex[:10]}")
    # ``time.monotonic`` for duration math (immune to wall-clock jumps
    # mid-request); ``time.time()`` for the wire timestamp the frontend
    # may want to display alongside other event times.
    _started_mono: float = field(default_factory=time.monotonic)
    started_at: float = field(default_factory=time.time)

    def start_payload(self) -> dict[str, Any]:
        """Augmentum dict to attach to the ``stage_start`` chunk."""
        return {
            "stage_start": {
                "id": self.id,
                "stage": self.name,
                "label": self.label or self.name,
                "detail": self.detail,
                "started_at": self.started_at,
                "request_id": request_id_var.get() or "",
            },
        }

    def complete_payload(
        self,
        *,
        success: bool = True,
        error_text: str = "",
        detail: str = "",
    ) -> dict[str, Any]:
        """Augmentum dict for the ``stage_complete`` chunk.

        ``detail`` defaults to the value passed at construction so the
        frontend can keep showing the same descriptor; pass a fresh
        string to update it (e.g., "restored from RAM cache").
        """
        duration_ms = int((time.monotonic() - self._started_mono) * 1000)
        return {
            "stage_complete": {
                "id": self.id,
                "stage": self.name,
                "success": success,
                "duration_ms": duration_ms,
                "detail": detail or self.detail,
                "error": error_text,
                "request_id": request_id_var.get() or "",
            },
        }

    def progress_payload(
        self,
        *,
        percent: int | None = None,
        message: str = "",
    ) -> dict[str, Any]:
        """Optional mid-stage update. Not used by the 1-day MVP — only
        emit progress on stages where the work is genuinely partitioned
        (e.g. multi-pass model load). Otherwise the start→complete
        pair is enough.
        """
        payload: dict[str, Any] = {
            "id": self.id,
            "stage": self.name,
        }
        if percent is not None:
            payload["percent"] = percent
        if message:
            payload["message"] = message
        return {"stage_progress": payload}
