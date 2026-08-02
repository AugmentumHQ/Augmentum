"""Shared progress-payload builders for engine load + prefill snapshots.

Single source of truth for the JSON shape the UI consumes so the three
surfaces that report it can't drift:

  - ``GET /api/engine/v2/load_progress`` / ``/prefill_progress`` (local,
    model_routes.py) — the originator watching its own engine.
  - ``GET /api/fabric/load_status`` (fabric_routes.py) — a peer reporting
    its load/prefill state back to an originating peer.
  - The fabric originator's coordinator cache fallback, which surfaces a
    peer's snapshot through the SAME local ``/v2/`` endpoints so the
    existing ``load-progress.js`` poller renders a cross-peer load with
    byte-identical UX.

The builders take the raw snapshot dicts that ``LlamaServerManager``
maintains (``_load_progress`` / ``_prefill_progress``) and return the
wire dict, or the ``{"active": False}`` form when there's nothing fresh
to show. Pure functions — no manager/request coupling — so they're
trivially testable and importable from both the routes layer and the
fabric layer without an import cycle.
"""
from __future__ import annotations

import time

# Staleness ceiling for a prefill snapshot. The upstream llama-server
# emits a print_timing line every chunk; 8s is well past that cadence,
# so a snapshot older than this means prefill finished (or the slot
# moved on) and we should report inactive rather than a frozen bar.
PREFILL_STALE_AFTER_S = 8.0

# Load-progress bar caps at 95% until the manager actually reaches READY
# and clears the snapshot — honest "still going" signalling instead of a
# bar that claims 100% while llama-server is still warming up.
_LOAD_PROGRESS_CAP = 0.95


def build_load_progress_payload(
    snapshot: dict | None, *, now_monotonic: float | None = None,
) -> dict:
    """Wire dict for a model-load snapshot (``manager._load_progress``).

    ``now_monotonic`` is injectable for tests; defaults to
    ``time.monotonic()``. Mirrors the computation that was inline in
    ``engine_v2_load_progress`` so local and cross-peer bars match.
    """
    if not snapshot:
        return {"active": False}
    now = time.monotonic() if now_monotonic is None else now_monotonic
    started_at = float(snapshot.get("started_at", 0.0))
    expected_s = float(snapshot.get("expected_s", 30.0))
    elapsed_s = max(0.0, now - started_at)
    progress = (
        min(_LOAD_PROGRESS_CAP, elapsed_s / expected_s) if expected_s > 0 else 0.0
    )
    return {
        "active": True,
        "model_id": snapshot.get("model_id", ""),
        "size_bytes": int(snapshot.get("size_bytes", 0)),
        "stage_label": snapshot.get("stage_label", "Loading model"),
        "elapsed_s": round(elapsed_s, 1),
        "expected_s": round(expected_s, 1),
        "progress": round(progress, 3),
    }


def build_prefill_progress_payload(
    snapshot: dict | None, *, now_wall: float | None = None,
) -> dict:
    """Wire dict for a prefill snapshot (``manager._prefill_progress``).

    Returns ``{"active": False}`` (optionally with ``age_s``) when no
    snapshot exists or the last one is staler than ``PREFILL_STALE_AFTER_S``.
    ``now_wall`` is injectable for tests; defaults to ``time.time()``
    because prefill snapshots are stamped with wall-clock ``updated_at``.
    """
    if not snapshot:
        return {"active": False}
    now = time.time() if now_wall is None else now_wall
    age_s = now - float(snapshot.get("updated_at", 0.0))
    if age_s > PREFILL_STALE_AFTER_S:
        return {"active": False, "age_s": round(age_s, 1)}
    return {
        "active": True,
        "tokens_done": snapshot["tokens_done"],
        "progress": snapshot["progress"],
        "elapsed_s": snapshot["elapsed_s"],
        "tps": snapshot["tps"],
        "age_s": round(age_s, 1),
    }
