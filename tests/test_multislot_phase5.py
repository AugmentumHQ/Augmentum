"""Phase 5 tests — tri-state default semantics.

Phase 5 flips the default behavior from "off, opt-in" to "auto, follow
codebase recommendation" without overwriting explicit user choices.
The field shape changes from ``bool = False`` to ``bool | None = None``,
and the runtime resolves ``None`` via ``MULTISLOT_DEFAULT_ENABLED``.

Spec: docs/superpowers/specs/2026-05-05-multi-slot-kv-design.md
"""

from __future__ import annotations

import contextlib

import httpx
import pytest

from augmentum.models.llama_cpp import LlamaCppBackend
from augmentum.models.llama_server_manager import LlamaServerManager


@contextlib.contextmanager
def _multislot_setting(value):
    """Set engine_multislot_enabled to value for the duration of a test."""
    from augmentum.config import settings
    prev = settings.engine_multislot_enabled
    settings.engine_multislot_enabled = value
    try:
        yield
    finally:
        settings.engine_multislot_enabled = prev


def _make_backend() -> LlamaCppBackend:
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={})
    )
    client = httpx.AsyncClient(transport=transport)
    return LlamaCppBackend(client, "http://llamacpp:8080")


# ---------------------------------------------------------------------------
# Resolver behavior — None / True / False all flow through correctly
# ---------------------------------------------------------------------------


class TestResolverTriState:
    """``_multislot_enabled`` on the backend and the equivalent
    resolution in ``_build_slot_scheduling_args`` on the manager must
    interpret None as "follow MULTISLOT_DEFAULT_ENABLED" while True/
    False pass through.
    """

    def test_backend_resolves_none_to_codebase_default(self):
        from augmentum.proxy.status_bus import MULTISLOT_DEFAULT_ENABLED
        backend = _make_backend()
        with _multislot_setting(None):
            assert backend._multislot_enabled() is MULTISLOT_DEFAULT_ENABLED

    def test_backend_explicit_false_overrides_default(self):
        backend = _make_backend()
        with _multislot_setting(False):
            assert backend._multislot_enabled() is False

    def test_backend_explicit_true_passes_through(self):
        backend = _make_backend()
        with _multislot_setting(True):
            assert backend._multislot_enabled() is True

    def test_manager_args_follow_codebase_default_on_none(self):
        """When the user has not toggled (None), the engine's CLI args
        come from the codebase's recommendation.
        """
        from augmentum.proxy.status_bus import MULTISLOT_DEFAULT_ENABLED
        m = LlamaServerManager.__new__(LlamaServerManager)
        # __new__ skips __init__; the args builder reads this attr
        # (added 275fd8a) — set the production default explicitly.
        m._force_single_slot = False
        with _multislot_setting(None):
            args = m._build_slot_scheduling_args()
        if MULTISLOT_DEFAULT_ENABLED:
            assert "--kv-unified" in args
            assert "--cache-ram" in args
        else:
            assert args == ["--parallel", "1"]

    def test_manager_args_explicit_false_emits_parallel_1(self):
        """Explicit "Always off" — single-slot regardless of recommendation."""
        m = LlamaServerManager.__new__(LlamaServerManager)
        # __new__ skips __init__; the args builder reads this attr
        # (added 275fd8a) — set the production default explicitly.
        m._force_single_slot = False
        with _multislot_setting(False):
            args = m._build_slot_scheduling_args()
        assert args == ["--parallel", "1"]

    def test_manager_args_explicit_true_emits_multislot(self):
        """Explicit "Always on" — multi-slot regardless of recommendation."""
        m = LlamaServerManager.__new__(LlamaServerManager)
        # __new__ skips __init__; the args builder reads this attr
        # (added 275fd8a) — set the production default explicitly.
        m._force_single_slot = False
        with _multislot_setting(True):
            args = m._build_slot_scheduling_args()
        assert "--kv-unified" in args
        assert "--cache-ram" in args
        assert "--cache-idle-slots" not in args  # evict==destroy under --kv-unified


# ---------------------------------------------------------------------------
# _parse_optional_bool — the persistence-side parser
# ---------------------------------------------------------------------------


class TestParseOptionalBool:
    """The parser used by ``_SETTINGS_RESTORE_MAP`` for tri-state keys.
    Without correct handling, persisted values lose their meaning on
    container restart (the bug Phase 4's restore_map fix solved for
    the bool case; tri-state needs its own variant).
    """

    def test_none_stays_none(self):
        from augmentum.proxy.server import _parse_optional_bool
        assert _parse_optional_bool(None) is None

    def test_empty_string_treated_as_none(self):
        """Wire shape robustness: some round-trip paths represent the
        absence-of-override as the empty string. Must not be mistaken
        for ``False``.
        """
        from augmentum.proxy.server import _parse_optional_bool
        assert _parse_optional_bool("") is None

    def test_auto_keyword_treated_as_none(self):
        """User-facing wire string: "auto" maps to None for the same
        reason "auto" is the UI label."""
        from augmentum.proxy.server import _parse_optional_bool
        assert _parse_optional_bool("auto") is None

    def test_truthy_strings(self):
        from augmentum.proxy.server import _parse_optional_bool
        assert _parse_optional_bool("True") is True
        assert _parse_optional_bool("true") is True
        assert _parse_optional_bool("1") is True
        assert _parse_optional_bool("yes") is True

    def test_falsy_strings(self):
        from augmentum.proxy.server import _parse_optional_bool
        assert _parse_optional_bool("False") is False
        assert _parse_optional_bool("0") is False
        assert _parse_optional_bool("no") is False

    def test_passthrough_bool(self):
        from augmentum.proxy.server import _parse_optional_bool
        assert _parse_optional_bool(True) is True
        assert _parse_optional_bool(False) is False


# ---------------------------------------------------------------------------
# API surface — GET returns _resolved companion, PUT accepts null
# ---------------------------------------------------------------------------


class TestApiSurface:
    """``_TRI_STATE_BOOL_SETTINGS`` registers tri-state keys with their
    resolver, used by the GET handler to expose a ``<key>_resolved``
    companion field. The frontend reads this for the "Auto · currently
    enabled" label without needing the resolution logic itself.
    """

    def test_resolver_for_multislot_registered(self):
        from augmentum.proxy.config_routes import _TRI_STATE_BOOL_SETTINGS
        assert "engine_multislot_enabled" in _TRI_STATE_BOOL_SETTINGS

    def test_resolver_returns_default_for_none(self):
        from augmentum.proxy.config_routes import _TRI_STATE_BOOL_SETTINGS
        from augmentum.proxy.status_bus import MULTISLOT_DEFAULT_ENABLED
        resolver = _TRI_STATE_BOOL_SETTINGS["engine_multislot_enabled"]
        assert resolver(None) is MULTISLOT_DEFAULT_ENABLED

    def test_resolver_passes_through_explicit(self):
        from augmentum.proxy.config_routes import _TRI_STATE_BOOL_SETTINGS
        resolver = _TRI_STATE_BOOL_SETTINGS["engine_multislot_enabled"]
        assert resolver(True) is True
        assert resolver(False) is False


# ---------------------------------------------------------------------------
# Migration safety — defaults match the documented matrix
# ---------------------------------------------------------------------------


class TestMigrationSafety:
    """Verify the documented migration matrix from the spec:

    | DB row    | Pre-Phase-5 runtime | Post-Phase-5 runtime  | Visible behavior |
    | absent    | False               | None → resolved=True  | Multi-slot now active (intentional) |
    | "True"    | True                | True (unchanged)      | No change |
    | "False"   | False               | False (unchanged)     | No change |
    """

    def test_default_unset_resolves_to_codebase_recommendation(self):
        """Brand-new install or never-toggled user: no DB row → field
        is None → resolves to MULTISLOT_DEFAULT_ENABLED.
        """
        from augmentum.proxy.status_bus import MULTISLOT_DEFAULT_ENABLED
        backend = _make_backend()
        with _multislot_setting(None):
            assert backend._multislot_enabled() is MULTISLOT_DEFAULT_ENABLED

    def test_explicit_user_choice_survives_recommendation_flip(self):
        """If a user picked "Always on" before we change the
        recommendation back to off — or vice versa — their explicit
        choice MUST stay. This is the load-bearing invariant of the
        tri-state design.
        """
        backend = _make_backend()
        # Simulate user picked "Always off" — even if codebase later
        # recommends on, this user stays off.
        with _multislot_setting(False):
            assert backend._multislot_enabled() is False
        # Simulate user picked "Always on" — stays on regardless.
        with _multislot_setting(True):
            assert backend._multislot_enabled() is True
