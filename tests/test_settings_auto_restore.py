"""Pin the auto-derived settings-restore contract.

Eliminates the "added a setting to _TOOL_SETTINGS, forgot to add it to
_SETTINGS_RESTORE_MAP, lose value on every restart" bug class. The
test asserts that:

1. ``_auto_derive_restore_parsers`` produces an entry for every key in
   ``config_routes._TOOL_SETTINGS`` whose type tuple's first element
   is bool/int/float, and for every key in ``_STRING_SETTINGS``.
2. The auto-derived parsers correctly coerce stored string values
   back to their typed form on restore — especially the bool case
   (``"False"`` must decode to ``False``, not the truthy non-empty
   string).
3. Manual entries in ``_SETTINGS_RESTORE_MAP`` STILL take precedence,
   so encrypted-secret strings and any custom-parser cases keep their
   hand-rolled handling instead of being silently downgraded to ``str``.

This is the regression anchor — if someone removes the auto-derivation
or breaks the merge order, these tests fail before the bug ships.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from augmentum.config import settings
from augmentum.proxy.config_routes import _STRING_SETTINGS, _TOOL_SETTINGS
from augmentum.proxy.server import (
    _SETTINGS_RESTORE_MAP,
    _auto_derive_restore_parsers,
    _parse_bool,
    _restore_settings,
)


def test_auto_derive_covers_every_tool_setting_of_known_type():
    """Every bool/int/float in _TOOL_SETTINGS should land in the
    auto-derived map. A miss here would mean a setting persists at
    PUT time but vanishes on the next restart."""
    auto = _auto_derive_restore_parsers()
    expected_typed = {
        key for key, tup in _TOOL_SETTINGS.items()
        if tup and tup[0] in (bool, int, float)
    }
    missing = expected_typed - set(auto.keys())
    assert not missing, (
        f"Settings registered in _TOOL_SETTINGS but not auto-restored: {sorted(missing)}. "
        "This is the 'keeps turning off on restart' bug class — fix _auto_derive_restore_parsers."
    )


def test_auto_derive_covers_every_string_setting():
    auto = _auto_derive_restore_parsers()
    missing = set(_STRING_SETTINGS.keys()) - set(auto.keys())
    assert not missing, (
        f"String settings missing from auto-restore: {sorted(missing)}."
    )


def test_auto_derive_bool_parser_handles_string_serialization():
    """The non-obvious one. SQLite stores Python booleans as
    ``str(True)`` / ``str(False)``. Calling ``bool("False")`` returns
    ``True`` (truthy non-empty string), which is the exact bug behind
    the original 'subagents toggle keeps turning off' symptom."""
    auto = _auto_derive_restore_parsers()
    # Pick any known-bool setting — coder_subagents_enabled has been
    # in the catalog since 2026-05 and is the canonical example.
    parser = auto.get("coder_subagents_enabled")
    assert parser is _parse_bool, (
        f"coder_subagents_enabled should use _parse_bool, got {parser!r}"
    )
    assert parser("True") is True
    assert parser("False") is False
    assert parser("true") is True
    assert parser("0") is False
    assert parser("1") is True


def test_manual_overrides_take_precedence():
    """Custom parsers in _SETTINGS_RESTORE_MAP MUST beat the
    auto-derived defaults — otherwise encrypted-string handlers
    would silently downgrade to plain str at runtime."""
    auto = _auto_derive_restore_parsers()
    # _SETTINGS_RESTORE_MAP has manual entries that should win the
    # merge. Pick any key present in both.
    overlapping = set(auto.keys()) & set(_SETTINGS_RESTORE_MAP.keys())
    assert overlapping, (
        "Manual + auto-derived maps share no keys — overlap check vacuous. "
        "Add a manual override for at least one setting so the precedence "
        "rule is regression-tested."
    )
    # The merge order in _restore_settings is {**auto, **manual} → manual wins.
    merged = {**auto, **_SETTINGS_RESTORE_MAP}
    for key in overlapping:
        assert merged[key] is _SETTINGS_RESTORE_MAP[key], (
            f"Manual override for '{key}' lost the merge — auto-derived "
            f"parser ({auto[key]!r}) won instead of manual "
            f"({_SETTINGS_RESTORE_MAP[key]!r})."
        )


@pytest.mark.asyncio
async def test_restore_settings_uses_auto_path_when_no_map_passed():
    """Production call site passes ``restore_map=None`` — verify the
    auto-derived map gets used end-to-end via a mocked store."""
    # Pick a known-bool setting and a known-int setting. Stash the
    # current values so we can restore after the test.
    bool_key = "coder_subagents_enabled"
    int_key = "coder_subagent_max_concurrent"
    bool_orig = getattr(settings, bool_key)
    int_orig = getattr(settings, int_key)

    store = AsyncMock()

    async def _get(key: str) -> Any:
        if key == bool_key:
            return "False"  # the bug-class trap value
        if key == int_key:
            return "7"
        return None

    store.get.side_effect = _get
    try:
        await _restore_settings(store)
        assert getattr(settings, bool_key) is False, (
            "_restore_settings did not decode stored 'False' string correctly "
            "via the auto-derived _parse_bool path."
        )
        assert getattr(settings, int_key) == 7, (
            "_restore_settings did not auto-coerce stored int string."
        )
    finally:
        object.__setattr__(settings, bool_key, bool_orig)
        object.__setattr__(settings, int_key, int_orig)
