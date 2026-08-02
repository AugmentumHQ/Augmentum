"""Tests for augmentum/config.py — Settings construction and defaults."""

from __future__ import annotations

import os
from unittest.mock import patch

from augmentum.config import Settings


class TestSettingsDefaults:
    """Verify Settings() constructs with valid defaults."""

    def test_constructs_without_error(self):
        s = Settings()
        assert s is not None

    def test_host_default(self):
        s = Settings()
        assert s.host == "0.0.0.0"

    def test_port_default(self):
        s = Settings()
        assert s.port == 6100

    def test_log_level_default(self):
        s = Settings()
        assert s.log_level == "INFO"

    def test_default_backend(self):
        s = Settings()
        assert s.default_backend == "engine"

    def test_default_num_ctx(self):
        s = Settings()
        assert s.default_num_ctx == 8192

    def test_openai_base_url_default(self):
        s = Settings()
        assert s.openai_base_url == "https://api.openai.com/v1"

    def test_mcp_enabled_default(self):
        s = Settings()
        assert s.mcp_enabled is True


class TestSettingsTypes:
    """Verify settings have correct types."""

    def test_bool_settings_are_bool(self):
        s = Settings()
        bool_fields = [
            "mcp_enabled", "memory_enabled", "think_enabled",
            "image_enabled", "voice_enabled", "metrics_enabled",
            "rate_limit_enabled", "agentic_enabled",
            "reranker_enabled",
        ]
        for name in bool_fields:
            val = getattr(s, name)
            assert isinstance(val, bool), f"{name} should be bool, got {type(val)}"

    def test_string_settings_are_str(self):
        s = Settings()
        str_fields = [
            "host", "log_level", "openai_base_url",
            "searxng_base_url", "executor_base_url",
            "default_backend", "timezone",
        ]
        for name in str_fields:
            val = getattr(s, name)
            assert isinstance(val, str), f"{name} should be str, got {type(val)}"

    def test_int_settings_are_int(self):
        s = Settings()
        int_fields = [
            "port", "max_concurrent_requests", "default_num_ctx",
            "coder_idle_timeout", "coder_max_workspaces",
        ]
        for name in int_fields:
            val = getattr(s, name)
            assert isinstance(val, int), f"{name} should be int, got {type(val)}"

    def test_float_settings_are_float(self):
        s = Settings()
        float_fields = [
            "request_timeout", "http_connect_timeout",
            "uarf_phase_timeout",
        ]
        for name in float_fields:
            val = getattr(s, name)
            assert isinstance(val, float), f"{name} should be float, got {type(val)}"


class TestSettingsEnvOverride:
    """Verify environment variable overrides."""

    def test_env_overrides_port(self):
        with patch.dict(os.environ, {"AUGMENTUM_PORT": "9999"}):
            s = Settings()
            assert s.port == 9999

    def test_env_overrides_log_level(self):
        with patch.dict(os.environ, {"AUGMENTUM_LOG_LEVEL": "DEBUG"}):
            s = Settings()
            assert s.log_level == "DEBUG"

    def test_env_overrides_bool(self):
        with patch.dict(os.environ, {"AUGMENTUM_METRICS_ENABLED": "false"}):
            s = Settings()
            assert s.metrics_enabled is False


class TestToSafeDict:
    """Verify sensitive values are redacted."""

    def test_redacts_api_keys(self):
        s = Settings()
        safe = s.to_safe_dict()
        # openai_api_key is None by default, so it should be None
        assert safe["openai_api_key"] is None

    def test_redacts_set_api_key(self):
        with patch.dict(os.environ, {"AUGMENTUM_ANTHROPIC_API_KEY": "sk-secret"}):
            s = Settings()
            safe = s.to_safe_dict()
            assert safe["anthropic_api_key"] == "***"

    def test_non_sensitive_values_visible(self):
        s = Settings()
        safe = s.to_safe_dict()
        assert safe["host"] == "0.0.0.0"
        assert safe["port"] == 6100
