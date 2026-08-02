"""augmentum/modes/narrative/memory_settings.py"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class SessionMemorySettings:
    """Per-session overrides for narrative LTM behaviour.

    Every field defaults to ``None`` which means "use the global setting".
    Only non-None values are persisted and take precedence.
    """

    memory_enabled: bool | None = None
    memory_mode: str | None = None
    memory_state_enabled: bool | None = None
    memory_ledger_enabled: bool | None = None
    memory_continuous_archive: bool | None = None
    smart_retrieval: bool | None = None
    smart_retrieval_count: int | None = None
    memory_ledger_ceiling: int | None = None
    memory_compaction_enabled: bool | None = None
    memory_interval: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return only non-None values (compact JSON representation)."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None:
                out[f.name] = v
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SessionMemorySettings:
        """Construct from a (possibly partial) dict.  Unknown keys ignored."""
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def init_from_globals(cls) -> SessionMemorySettings:
        """Snapshot current global settings as initial per-session values."""
        from augmentum.config import settings as cfg

        return cls(
            memory_enabled=cfg.narrative_memory_enabled,
            memory_mode=cfg.narrative_memory_mode,
            memory_state_enabled=cfg.narrative_memory_state_enabled,
            memory_ledger_enabled=cfg.narrative_memory_ledger_enabled,
            memory_continuous_archive=cfg.narrative_memory_continuous_archive,
            smart_retrieval=cfg.narrative_smart_retrieval,
            smart_retrieval_count=cfg.narrative_smart_retrieval_count,
            memory_ledger_ceiling=cfg.narrative_memory_ledger_ceiling,
            memory_compaction_enabled=cfg.narrative_memory_compaction_enabled,
            memory_interval=cfg.narrative_memory_interval,
        )


# Mapping from SessionMemorySettings field name → global config attribute name.
FIELD_TO_GLOBAL: dict[str, str] = {
    "memory_enabled": "narrative_memory_enabled",
    "memory_mode": "narrative_memory_mode",
    "memory_state_enabled": "narrative_memory_state_enabled",
    "memory_ledger_enabled": "narrative_memory_ledger_enabled",
    "memory_continuous_archive": "narrative_memory_continuous_archive",
    "smart_retrieval": "narrative_smart_retrieval",
    "smart_retrieval_count": "narrative_smart_retrieval_count",
    "memory_ledger_ceiling": "narrative_memory_ledger_ceiling",
    "memory_compaction_enabled": "narrative_memory_compaction_enabled",
    "memory_interval": "narrative_memory_interval",
}

_sentinel = object()


def resolve_memory_setting(
    session: SessionMemorySettings | None,
    key: str,
    *,
    global_value: Any = _sentinel,
) -> Any:
    """Return the effective value for *key*.

    Resolution order:
      1. Session-level override (if not None)
      2. Explicit *global_value* kwarg (if provided)
      3. Live global ``settings`` singleton
    """
    if session is not None:
        val = getattr(session, key, None)
        if val is not None:
            return val

    if global_value is not _sentinel:
        return global_value

    from augmentum.config import settings as cfg

    global_key = FIELD_TO_GLOBAL.get(key, f"narrative_{key}")
    return getattr(cfg, global_key)
