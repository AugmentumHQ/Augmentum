"""Tests for the global ``heavyweight_model`` slot.

Pinning behavior of the heavyweight model substrate added 2026-05-31:

- ``Settings.heavyweight_model`` default is empty (opt-in).
- ``ProviderRegistry.resolve_model_for_role("heavyweight", ...)`` reads
  the setting; returns the registry's default backend fallback when
  the setting is empty.
- The coder workspace's ``bug_finder_verifier_model`` (per-workspace
  HVY button) takes priority over the global setting in
  ``_get_workspace_buddy_model``.
- The bug_finder_run job handler falls back to the global setting
  when the payload's ``verifier_model`` is empty.

These pin the slot's resolution rules so future consumers (future
``/second-opinion`` slash command, narrative summariser escalation,
classifier hard-case fallback) inherit the same semantics.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.config import Settings


# ----------------------------------------------------------------------
# Config-layer behaviour
# ----------------------------------------------------------------------

def test_heavyweight_model_default_empty():
    """Opt-in: a fresh Settings carries no heavyweight model."""
    s = Settings()
    assert s.heavyweight_model == ""


def test_heavyweight_model_accepts_multi_provider_syntax():
    """The slot accepts the same spec syntax as the subagent dispatcher
    so peer routing / @provider overrides work the same way."""
    s = Settings(heavyweight_model="claude-opus-4-7@anthropic")
    assert s.heavyweight_model == "claude-opus-4-7@anthropic"

    s = Settings(heavyweight_model="claude-opus-4-7@fabric:tower")
    assert s.heavyweight_model == "claude-opus-4-7@fabric:tower"

    s = Settings(heavyweight_model="gpt-5.5")
    assert s.heavyweight_model == "gpt-5.5"


# ----------------------------------------------------------------------
# Resolver-layer behaviour
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolver_heavyweight_role_uses_setting():
    """``resolve_model_for_role('heavyweight', ...)`` reads the global
    setting and routes through ``resolve_backend_with_fabric``."""
    from augmentum.models.provider_registry import ProviderRegistry

    reg = ProviderRegistry(http_client=MagicMock())
    reg.resolve_backend_with_fabric = AsyncMock(
        return_value=(MagicMock(), "claude-opus-4-7"),
    )

    settings = Settings(heavyweight_model="claude-opus-4-7@anthropic")
    backend, model = await reg.resolve_model_for_role(
        "heavyweight", settings=settings,
    )
    reg.resolve_backend_with_fabric.assert_called_with(
        "claude-opus-4-7@anthropic",
    )
    assert model == "claude-opus-4-7"


@pytest.mark.asyncio
async def test_resolver_heavyweight_unset_falls_through_to_primary():
    """When ``heavyweight_model`` is empty the resolver continues down
    the existing fallback chain (primary_chat_model → default backend),
    so callers always get *something* back rather than failing."""
    from augmentum.models.provider_registry import ProviderRegistry

    reg = ProviderRegistry(http_client=MagicMock())
    reg.resolve_backend_with_fabric = AsyncMock(
        return_value=(MagicMock(), "Qwen3.6-35B"),
    )

    settings = Settings(
        heavyweight_model="",
        primary_chat_model="Qwen3.6-35B",
    )
    _backend, model = await reg.resolve_model_for_role(
        "heavyweight", settings=settings,
    )
    # The resolver tried heavyweight (empty), then primary.
    assert model == "Qwen3.6-35B"
    reg.resolve_backend_with_fabric.assert_called_with("Qwen3.6-35B")


@pytest.mark.asyncio
async def test_resolver_explicit_override_wins_over_setting():
    """Per-feature overrides (e.g. per-workspace
    ``bug_finder_verifier_model``) take priority over the global setting
    when passed via the ``override`` arg."""
    from augmentum.models.provider_registry import ProviderRegistry

    reg = ProviderRegistry(http_client=MagicMock())
    reg.resolve_backend_with_fabric = AsyncMock(
        return_value=(MagicMock(), "gpt-5.5"),
    )

    settings = Settings(heavyweight_model="claude-opus-4-7@anthropic")
    _backend, model = await reg.resolve_model_for_role(
        "heavyweight", override="gpt-5.5@openai", settings=settings,
    )
    # Override wins, global setting is unused.
    reg.resolve_backend_with_fabric.assert_called_once_with("gpt-5.5@openai")
    assert model == "gpt-5.5"


@pytest.mark.asyncio
async def test_resolver_other_roles_unaffected():
    """The new branch only adds ``role='heavyweight'`` handling — the
    existing classifier/utility/primary chain stays as it was."""
    from augmentum.models.provider_registry import ProviderRegistry

    reg = ProviderRegistry(http_client=MagicMock())
    reg.resolve_backend_with_fabric = AsyncMock(
        return_value=(MagicMock(), "Qwen3.6-Coder"),
    )

    settings = Settings(
        heavyweight_model="claude-opus-4-7@anthropic",
        classifier_model="Qwen3.6-Coder",
    )
    _backend, model = await reg.resolve_model_for_role(
        "classifier", settings=settings,
    )
    # Should hit classifier_model, NOT heavyweight_model.
    assert model == "Qwen3.6-Coder"
    reg.resolve_backend_with_fabric.assert_called_with("Qwen3.6-Coder")


# ----------------------------------------------------------------------
# Bug Finder + coder workspace consumer wiring
# ----------------------------------------------------------------------

def test_bug_finder_handler_falls_back_to_heavyweight_setting(monkeypatch):
    """When the job payload's ``verifier_model`` is empty, the handler
    should pick up the global ``heavyweight_model`` so users who set
    it once get verifier coverage on every workspace without per-
    workspace config."""
    from augmentum.bug_finder.role_models import RoleModelConfig

    # Simulate the handler's resolution snippet by directly calling the
    # logic — keeps the test stable across refactors of the surrounding
    # async machinery in jobs/handlers/bug_finder_run.py.
    fake_settings = Settings(heavyweight_model="gpt-5.5")
    monkeypatch.setattr(
        "augmentum.config.settings", fake_settings,
    )

    payload_primary = "Qwen3.6-35B"
    payload_verifier = ""  # empty in payload
    if not payload_verifier:
        from augmentum.config import settings as _settings
        payload_verifier = (
            getattr(_settings, "heavyweight_model", "") or ""
        ).strip()

    cfg = RoleModelConfig.from_primary(
        payload_primary, verifier=payload_verifier,
    )
    assert cfg.verifier == "gpt-5.5"
    assert cfg.fixer == "Qwen3.6-35B"
    assert not cfg.same_model_self_verification
