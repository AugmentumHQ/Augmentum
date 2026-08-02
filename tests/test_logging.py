"""Tests for augmentum/utils/logging.py — structured logging setup."""

from __future__ import annotations

import structlog

from augmentum.utils.logging import get_logger, setup_logging


class TestGetLogger:
    """Verify get_logger returns proper structlog instances."""

    def test_returns_bound_logger(self):
        logger = get_logger("test_module")
        assert logger is not None

    def test_logger_is_structlog_instance(self):
        logger = get_logger("test_module")
        # structlog returns a BoundLoggerLazyProxy or similar
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")

    def test_logger_with_none_name(self):
        logger = get_logger(None)
        assert logger is not None

    def test_logger_with_module_name(self):
        logger = get_logger(__name__)
        assert logger is not None

    def test_different_names_return_loggers(self):
        a = get_logger("module_a")
        b = get_logger("module_b")
        assert a is not None
        assert b is not None


class TestSetupLogging:
    """Verify logging setup does not crash for valid levels."""

    def test_setup_info(self):
        setup_logging("INFO")

    def test_setup_debug(self):
        setup_logging("DEBUG")

    def test_setup_warning(self):
        setup_logging("WARNING")

    def test_setup_error(self):
        setup_logging("ERROR")

    def test_setup_case_insensitive(self):
        # Should not raise even with lowercase
        setup_logging("info")
