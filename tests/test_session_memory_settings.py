"""tests/test_session_memory_settings.py"""
from __future__ import annotations

import json

from augmentum.modes.narrative.memory_settings import (
    SessionMemorySettings,
    resolve_memory_setting,
)


class TestSessionMemorySettings:
    def test_defaults_are_none(self):
        s = SessionMemorySettings()
        assert s.memory_enabled is None
        assert s.memory_mode is None
        assert s.memory_state_enabled is None

    def test_from_dict_partial(self):
        s = SessionMemorySettings.from_dict({"memory_enabled": False})
        assert s.memory_enabled is False
        assert s.memory_mode is None

    def test_from_dict_ignores_unknown(self):
        s = SessionMemorySettings.from_dict({"bogus_key": 42, "memory_enabled": True})
        assert s.memory_enabled is True

    def test_from_dict_none(self):
        s = SessionMemorySettings.from_dict(None)
        assert s.memory_enabled is None

    def test_to_dict_omits_none(self):
        s = SessionMemorySettings(memory_enabled=False, memory_mode="lite")
        d = s.to_dict()
        assert d == {"memory_enabled": False, "memory_mode": "lite"}

    def test_to_dict_preserves_zero(self):
        s = SessionMemorySettings(memory_ledger_ceiling=0)
        d = s.to_dict()
        assert d == {"memory_ledger_ceiling": 0}

    def test_to_dict_empty(self):
        s = SessionMemorySettings()
        assert s.to_dict() == {}

    def test_round_trip(self):
        original = SessionMemorySettings(
            memory_enabled=True,
            memory_mode="standard",
            memory_ledger_ceiling=100,
        )
        restored = SessionMemorySettings.from_dict(original.to_dict())
        assert restored.memory_enabled == original.memory_enabled
        assert restored.memory_mode == original.memory_mode
        assert restored.memory_ledger_ceiling == original.memory_ledger_ceiling
        assert restored.smart_retrieval is None

    def test_json_round_trip(self):
        original = SessionMemorySettings(memory_enabled=True, memory_mode="standard")
        json_str = json.dumps(original.to_dict())
        restored = SessionMemorySettings.from_dict(json.loads(json_str))
        assert restored.memory_enabled is True
        assert restored.memory_mode == "standard"


class TestResolveMemorySetting:
    def test_session_override_wins(self):
        session = SessionMemorySettings(memory_enabled=False)
        result = resolve_memory_setting(session, "memory_enabled", global_value=True)
        assert result is False

    def test_global_fallback_when_none(self):
        session = SessionMemorySettings()
        result = resolve_memory_setting(session, "memory_enabled", global_value=True)
        assert result is True

    def test_global_fallback_no_session(self):
        result = resolve_memory_setting(None, "memory_enabled", global_value=True)
        assert result is True

    def test_int_setting(self):
        session = SessionMemorySettings(memory_ledger_ceiling=200)
        result = resolve_memory_setting(session, "memory_ledger_ceiling", global_value=60)
        assert result == 200

    def test_string_setting(self):
        session = SessionMemorySettings(memory_mode="lite")
        result = resolve_memory_setting(session, "memory_mode", global_value="standard")
        assert result == "lite"

    def test_zero_is_not_none(self):
        session = SessionMemorySettings(memory_ledger_ceiling=0)
        result = resolve_memory_setting(session, "memory_ledger_ceiling", global_value=60)
        assert result == 0


class TestInitFromGlobals:
    def test_captures_current_globals(self):
        s = SessionMemorySettings.init_from_globals()
        assert s.memory_enabled is not None
        assert s.memory_mode is not None
        assert s.memory_state_enabled is not None
        assert s.memory_ledger_enabled is not None
        assert s.memory_continuous_archive is not None
        assert s.smart_retrieval is not None
        assert s.smart_retrieval_count is not None
        assert s.memory_ledger_ceiling is not None
        assert s.memory_compaction_enabled is not None
        assert s.memory_interval is not None
        d = s.to_dict()
        assert len(d) == 10
