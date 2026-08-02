"""Save service for the Augmentum Experience Framework.

Every runtime that produces savable state -- browser-WASM emulators,
server-streamed RetroArch, future native games with saves -- writes
through this service. The interface is uniform: PUT a slot, GET a
slot, list the slots for a title. The actual bytes live in the
shared blob store; this layer is the index + lifecycle.

Cross-runtime portability: SRAM kinds are core-agnostic and round-trip
across engines (browser ↔ streamed). State kinds are tagged with the
``core_id`` that produced them; the runtime layer enforces "same core
or refuse to load." Screenshots are PNGs and travel anywhere.
"""

from __future__ import annotations

from augmentum.saves.store import (
    SAVE_KINDS,
    SaveKind,
    SaveRecord,
    SaveServiceError,
    SaveStore,
    SaveTooLargeError,
)

__all__ = [
    "SAVE_KINDS",
    "SaveKind",
    "SaveRecord",
    "SaveServiceError",
    "SaveStore",
    "SaveTooLargeError",
]
