"""Per-workspace ring buffer of console/error events from the USER's live preview.

The coder's browser *tools* (``augmentum/coder/browser.py``) launch a fresh
headless Chromium per call and cold-load the preview URL — they structurally
miss any error that depends on the user's session, auth, navigated app-state, or
interactions. This buffer closes that gap: a shim injected into the preview
``<head>`` (``coder_routes._PREVIEW_CONSOLE_CAPTURE_SCRIPT``) hooks
``console.error`` / ``window.onerror`` / ``unhandledrejection`` in the *real*
preview iframe, ``postMessage``s batches to the coder UI, which POSTs them to the
``/preview-console`` beacon, which lands here.

Two consumers read it:
  * ``browser_snapshot`` folds the recent entries into its result (pull) — so the
    tool reports the user's actual errors, not just its disconnected cold load.
  * the coder turn can auto-inject new-since-last-turn entries (push).

In-memory and ephemeral by design (console noise, not durable state). Keyed by
workspace_id; a monotonic per-workspace sequence id supports "since last turn".
"""

from __future__ import annotations

import time
from collections import deque
from threading import Lock

_MAX_ENTRIES = 50       # ring depth per workspace
_TEXT_CAP = 600         # per-entry text clamp
_VALID_TYPES = frozenset({"error", "warn", "exception", "unhandledrejection"})

_buffers: dict[str, deque] = {}
_seq: dict[str, int] = {}
_lock = Lock()


def record(workspace_id: str, entries: list) -> int:
    """Append captured entries; return the workspace's new high-water seq id.

    Defensive: ignores non-dict entries, clamps text, caps the batch. Never
    raises — the beacon path must not 500 on malformed client input.
    """
    if not workspace_id or not entries:
        return _seq.get(workspace_id, 0)
    with _lock:
        buf = _buffers.setdefault(workspace_id, deque(maxlen=_MAX_ENTRIES))
        n = _seq.get(workspace_id, 0)
        for e in entries[:_MAX_ENTRIES]:
            if not isinstance(e, dict):
                continue
            text = str(e.get("text") or "").strip()
            if not text:
                continue
            etype = str(e.get("type") or "error")
            if etype not in _VALID_TYPES:
                etype = "error"
            n += 1
            buf.append({
                "id": n,
                "ts": float(e.get("ts") or time.time()),
                "type": etype,
                "text": text[:_TEXT_CAP],
                "url": str(e.get("url") or "")[:200],
                "line": e.get("line") if isinstance(e.get("line"), int) else None,
            })
        _seq[workspace_id] = n
        return n


def snapshot(workspace_id: str, *, since: int = 0, limit: int = 25) -> list[dict]:
    """Buffered entries with ``id > since`` (0 = all), oldest→newest, capped."""
    with _lock:
        buf = _buffers.get(workspace_id)
        if not buf:
            return []
        out = [dict(e) for e in buf if e["id"] > since]
    return out[-limit:]


def high_water(workspace_id: str) -> int:
    """Highest seq id recorded for the workspace (0 if none)."""
    return _seq.get(workspace_id, 0)


def clear(workspace_id: str) -> None:
    """Drop a workspace's buffer (e.g. on container stop / rewind)."""
    with _lock:
        _buffers.pop(workspace_id, None)
        _seq.pop(workspace_id, None)
