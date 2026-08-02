"""Tests for core model role resolution (utility_model / classifier_model)."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(resolve_result=None):
    """Build a ProviderRegistry-like mock with a controllable resolve_backend_for_model."""
    from augmentum.models.provider_registry import ProviderRegistry

    registry = MagicMock(spec=ProviderRegistry)
    # resolve_backend_for_model is async
    _sentinel = object() if resolve_result is None else resolve_result
    if resolve_result is None:
        default_backend = MagicMock(name="default_backend")
        _sentinel = (default_backend, "default-model")
    registry.resolve_backend_for_model = AsyncMock(return_value=_sentinel)
    # Wire resolve_model_for_role to the real implementation
    registry.resolve_model_for_role = ProviderRegistry.resolve_model_for_role.__get__(registry)
    return registry


def _make_settings(utility_model: str = "", classifier_model: str = "") -> MagicMock:
    s = MagicMock()
    s.utility_model = utility_model
    s.classifier_model = classifier_model
    return s


# ---------------------------------------------------------------------------
# resolve_model_for_role tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_role_with_override_uses_override():
    """Per-feature override takes priority over everything."""
    backend = MagicMock(name="override_backend")
    registry = _make_registry((backend, "override-model"))
    settings = _make_settings(utility_model="utility-model", classifier_model="classifier-model")

    result_backend, result_model = await registry.resolve_model_for_role(
        role="utility", override="my-override-model", settings=settings
    )

    registry.resolve_backend_for_model.assert_called_once_with("my-override-model")
    assert result_backend is backend
    assert result_model == "override-model"


@pytest.mark.asyncio
async def test_utility_role_uses_utility_model():
    """utility role resolves to utility_model when set."""
    backend = MagicMock(name="utility_backend")
    registry = _make_registry((backend, "utility-model"))
    settings = _make_settings(utility_model="utility-model")

    result_backend, result_model = await registry.resolve_model_for_role(
        role="utility", settings=settings
    )

    registry.resolve_backend_for_model.assert_called_once_with("utility-model")
    assert result_backend is backend


@pytest.mark.asyncio
async def test_classifier_role_uses_classifier_model():
    """classifier role resolves to classifier_model when set."""
    backend = MagicMock(name="classifier_backend")
    registry = _make_registry((backend, "classifier-model"))
    settings = _make_settings(classifier_model="classifier-model")

    result_backend, result_model = await registry.resolve_model_for_role(
        role="classifier", settings=settings
    )

    registry.resolve_backend_for_model.assert_called_once_with("classifier-model")
    assert result_backend is backend


@pytest.mark.asyncio
async def test_classifier_falls_back_to_utility():
    """classifier falls back to utility_model when classifier_model is empty."""
    backend = MagicMock(name="utility_backend")
    registry = _make_registry((backend, "utility-model"))
    settings = _make_settings(utility_model="utility-model", classifier_model="")

    result_backend, result_model = await registry.resolve_model_for_role(
        role="classifier", settings=settings
    )

    registry.resolve_backend_for_model.assert_called_once_with("utility-model")
    assert result_backend is backend


@pytest.mark.asyncio
async def test_utility_falls_back_to_primary():
    """utility falls back to default when utility_model is empty."""
    default_backend = MagicMock(name="primary_backend")
    registry = _make_registry((default_backend, "primary-model"))
    settings = _make_settings(utility_model="", classifier_model="")

    result_backend, result_model = await registry.resolve_model_for_role(
        role="utility", settings=settings
    )

    registry.resolve_backend_for_model.assert_called_once_with("")
    assert result_backend is default_backend


@pytest.mark.asyncio
async def test_no_settings_falls_back_to_default():
    """No settings passed at all → default backend via resolve_backend_for_model('')."""
    default_backend = MagicMock(name="primary_backend")
    registry = _make_registry((default_backend, "primary-model"))

    result_backend, result_model = await registry.resolve_model_for_role(
        role="utility", settings=None
    )

    registry.resolve_backend_for_model.assert_called_once_with("")
    assert result_backend is default_backend


@pytest.mark.asyncio
async def test_primary_role_ignores_utility_and_classifier():
    """primary role skips all role settings and goes straight to default."""
    default_backend = MagicMock(name="primary_backend")
    registry = _make_registry((default_backend, "primary-model"))
    settings = _make_settings(utility_model="utility-model", classifier_model="classifier-model")

    result_backend, result_model = await registry.resolve_model_for_role(
        role="primary", settings=settings
    )

    # Should call with "" (not utility or classifier model names)
    registry.resolve_backend_for_model.assert_called_once_with("")
    assert result_backend is default_backend


# ---------------------------------------------------------------------------
# Config field tests
# ---------------------------------------------------------------------------

def test_config_has_utility_model():
    """Settings class has utility_model field with default empty string."""
    from augmentum.config import Settings
    s = Settings()
    assert hasattr(s, "utility_model")
    assert s.utility_model == ""


def test_config_has_classifier_model():
    """Settings class has classifier_model field with default empty string."""
    from augmentum.config import Settings
    s = Settings()
    assert hasattr(s, "classifier_model")
    assert s.classifier_model == ""
