"""Tests for ``bug_finder_run._config_from_payload``.

Pins the wiring contract between the HTTP run-kickoff payload and
``BugFinderRunConfig``. We don't run the handler — we just verify that
each documented opt-in (especially the new pen-test leg fields) lands
on the resulting config and that ``_replace_intake_with_patterns``
preserves them when the orchestrator swaps the intake."""

from __future__ import annotations

import pytest

from augmentum.jobs.handlers.bug_finder_run import (
    _config_from_payload,
    _replace_intake_with_patterns,
)


_MINIMUM = {
    "workspace_id": "ws-1",
    "primary_model": "claude-sonnet-4-6",
}


def test_pen_test_leg_off_by_default() -> None:
    config = _config_from_payload(dict(_MINIMUM))
    assert config.enable_pen_test_leg is False
    assert config.pen_test_boot_command == ""
    assert config.pen_test_boot_port == 0
    assert config.pen_test_healthcheck_path == "/"


def test_pen_test_leg_opts_in_when_payload_sets_flag() -> None:
    payload = dict(_MINIMUM)
    payload["enable_pen_test_leg"] = True
    config = _config_from_payload(payload)
    assert config.enable_pen_test_leg is True


def test_pen_test_hints_flow_through_payload() -> None:
    payload = dict(_MINIMUM)
    payload["enable_pen_test_leg"] = True
    payload["pen_test_boot_command"] = "python -m augmentum.proxy.server"
    payload["pen_test_boot_port"] = 8000
    payload["pen_test_healthcheck_path"] = "/healthz"
    config = _config_from_payload(payload)
    assert config.pen_test_boot_command == "python -m augmentum.proxy.server"
    assert config.pen_test_boot_port == 8000
    assert config.pen_test_healthcheck_path == "/healthz"


def test_pen_test_boot_port_coerces_string_to_int() -> None:
    payload = dict(_MINIMUM)
    payload["enable_pen_test_leg"] = True
    payload["pen_test_boot_port"] = "9090"
    config = _config_from_payload(payload)
    assert config.pen_test_boot_port == 9090


def test_pen_test_settings_survive_intake_replacement() -> None:
    payload = dict(_MINIMUM)
    payload["enable_pen_test_leg"] = True
    payload["pen_test_boot_command"] = "uvicorn app:app"
    payload["pen_test_boot_port"] = 8001
    payload["pen_test_healthcheck_path"] = "/readyz"
    config = _config_from_payload(payload)

    rebuilt = _replace_intake_with_patterns(config, brief="prior bugs: foo")

    # Pattern brief landed
    assert rebuilt.intake.prior_patterns == "prior bugs: foo"
    # All four pen_test fields preserved
    assert rebuilt.enable_pen_test_leg is True
    assert rebuilt.pen_test_boot_command == "uvicorn app:app"
    assert rebuilt.pen_test_boot_port == 8001
    assert rebuilt.pen_test_healthcheck_path == "/readyz"


def test_workspace_id_required() -> None:
    with pytest.raises(RuntimeError, match="workspace_id"):
        _config_from_payload({"primary_model": "claude-sonnet-4-6"})


def test_primary_model_required() -> None:
    with pytest.raises(RuntimeError, match="primary_model"):
        _config_from_payload({"workspace_id": "ws-1"})
