"""Audit hooks consumed by ``.claude/skills/augmentum-dev/scripts/audit.py``.

Each function in this module returns a list of finding dicts of the
shape ``{"key": str, "rule": str, "message": str}``. The audit script
imports this module, runs the checks, and turns findings into the
``registry_*`` subsystem score.

Phase 1A wires the checks but they produce zero findings (no settings
registered yet). Phase 1B's first migration will exercise them.
"""

from __future__ import annotations

from typing import Any

from augmentum.registry.registry import get_registry

# Constants — tunable via the suppressions file if needed; copy-pasted
# from the Phase 1A spec.
DESCRIPTION_MIN_CHARS = 20
DESCRIPTION_MAX_CHARS = 400  # generous; warnings only above this


def check_all() -> list[dict[str, Any]]:
    """Run every registry check and return the combined finding list.

    Audit.py calls this once per run. The checks are intentionally
    cheap (in-memory iteration over the registered Settings).
    """
    findings: list[dict[str, Any]] = []
    findings.extend(check_labels_non_empty())
    findings.extend(check_descriptions_meaningful())
    findings.extend(check_sections_non_empty())
    findings.extend(check_kind_default_consistent())
    return findings


def check_labels_non_empty() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in get_registry().list_all():
        if not s.label.strip():
            out.append(
                {
                    "key": s.key,
                    "rule": "registry_settings_have_label",
                    "message": "label is empty",
                }
            )
    return out


def check_descriptions_meaningful() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in get_registry().list_all():
        d = s.description.strip()
        if not d:
            out.append(
                {
                    "key": s.key,
                    "rule": "registry_settings_have_description",
                    "message": "description is empty",
                }
            )
            continue
        if len(d) < DESCRIPTION_MIN_CHARS:
            out.append(
                {
                    "key": s.key,
                    "rule": "registry_settings_have_description",
                    "message": (
                        f"description is {len(d)} chars "
                        f"(min {DESCRIPTION_MIN_CHARS})"
                    ),
                }
            )
    return out


def check_sections_non_empty() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in get_registry().list_all():
        if not s.section.strip():
            out.append(
                {
                    "key": s.key,
                    "rule": "registry_settings_have_section",
                    "message": "section is empty",
                }
            )
    return out


def check_kind_default_consistent() -> list[dict[str, Any]]:
    """The Setting dataclass already validates this on construction,
    so a registered Setting with inconsistent kind/default is
    impossible. This check is a defensive belt-and-braces in case
    a future codepath bypasses ``__post_init__`` (e.g. via
    ``dataclasses.replace`` shenanigans)."""
    out: list[dict[str, Any]] = []
    for s in get_registry().list_all():
        if s.kind == "bool" and not isinstance(s.default, bool):
            out.append(_finding(s.key, "bool kind requires bool default"))
        elif s.kind == "int" and (
            isinstance(s.default, bool) or not isinstance(s.default, int)
        ):
            out.append(_finding(s.key, "int kind requires int default"))
        elif s.kind == "float" and (
            isinstance(s.default, bool)
            or not isinstance(s.default, (int, float))
        ):
            out.append(_finding(s.key, "float kind requires numeric default"))
        elif s.kind == "str" and not isinstance(s.default, str):
            out.append(_finding(s.key, "str kind requires str default"))
        elif s.kind == "enum" and (
            not isinstance(s.default, str)
            or not s.enum_values
            or s.default not in s.enum_values
        ):
            out.append(_finding(s.key, "enum default must be in enum_values"))
        elif s.kind == "tristate" and not (
            s.default is None or isinstance(s.default, bool)
        ):
            out.append(_finding(s.key, "tristate kind requires bool or None default"))
    return out


def _finding(key: str, message: str) -> dict[str, Any]:
    return {
        "key": key,
        "rule": "registry_settings_consistent_kind",
        "message": message,
    }


def summary() -> dict[str, Any]:
    """Return a one-line summary suitable for the audit's per-subsystem
    breakdown."""
    findings = check_all()
    by_rule: dict[str, int] = {}
    for f in findings:
        by_rule[f["rule"]] = by_rule.get(f["rule"], 0) + 1
    return {
        "registered": len(get_registry().list_all()),
        "findings": len(findings),
        "by_rule": by_rule,
    }
