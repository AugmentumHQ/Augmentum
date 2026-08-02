"""Observation Substrate — cross-modal sequential pattern memory.

Phase A (this module, 2026-06-02): the L0 layer + a single consumer
(``llama-server --lookup-cache-static`` via the lazy per-model
exporter). The substrate stores observations as text (tokenizer-
agnostic); the exporter regenerates the binary cache per loaded model
via the bundled ``llama-lookup-create`` binary.

See ``docs/superpowers/specs/2026-05-30-observation-substrate-design.md``
for the multi-phase plan. The companion expression policy, the
autocomplete consumer, the inspect/edit surface, and the L1/L2 stores
are deferred to later phases — none of them are wired here.

Public surface:

- :class:`ObservationStore` — L0 CRUD + top-K query
- :func:`fingerprint_prefix` — deterministic (text, surface, mode) hash
- :func:`seed_from_chat_history` — bootstrap from ``ui_sessions``
- :func:`export_lookup_cache` — write per-model cache via llama-lookup-create
"""

from __future__ import annotations

from augmentum.observation.fingerprint import fingerprint_prefix
from augmentum.observation.store import ObservationStore

__all__ = [
    "ObservationStore",
    "fingerprint_prefix",
]
