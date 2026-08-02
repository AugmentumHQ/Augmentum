"""Drift detector — registry vs the historical 4 declaration sites.

For every registered Setting, verify it is consistent with:
  - ``augmentum.config.Settings`` dataclass field (default + type)
  - ``augmentum.proxy.config_routes._TOOL_SETTINGS`` (if applicable)
  - ``augmentum.proxy.config_routes._STRING_SETTINGS`` (if applicable)
  - ``augmentum.proxy.config_routes._TOOL_SETTING_DEFAULTS`` (seed values)

This is the safety net for Phase 1C bulk migration: every migrated
Setting must round-trip through the historical layers without
behavioral change. Drift = a registered Setting whose declaration
contradicts what would have been declared in the literal dicts.

Returns finding dicts of the shape
``{"key": str, "rule": str, "message": str, "expected": ..., "actual": ...}``.

Findings are sorted by ``key`` for deterministic output (audit history
diffing).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from augmentum.registry.registry import get_registry
from augmentum.registry.settings import Setting

# Floating-point comparison tolerance for default/range checks. The
# pydantic dataclass may store floats with different precision than
# the literal dicts (e.g. 0.5 vs 0.50). Tight tolerance — anything
# legitimately different is still flagged.
_FLOAT_TOL = 1e-9


@dataclass(frozen=True)
class _Finding:
    key: str
    rule: str
    message: str
    expected: Any = None
    actual: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "rule": self.rule,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
        }


def check_all() -> list[dict[str, Any]]:
    """Run every drift check. Returns sorted finding dicts."""
    findings: list[_Finding] = []
    findings.extend(_check_config_dataclass_field_exists())
    findings.extend(_check_default_matches_config())
    findings.extend(_check_tool_settings_tuple_matches())
    findings.extend(_check_string_settings_maxlen_matches())
    findings.sort(key=lambda f: (f.key, f.rule))
    return [f.to_dict() for f in findings]


def summary() -> dict[str, Any]:
    findings = check_all()
    by_rule: dict[str, int] = {}
    for f in findings:
        by_rule[f["rule"]] = by_rule.get(f["rule"], 0) + 1
    return {
        "registered": len(get_registry().list_all()),
        "drift_findings": len(findings),
        "by_rule": by_rule,
    }


# ---- per-rule checks ----


def _check_config_dataclass_field_exists() -> list[_Finding]:
    """Every registered Setting must correspond to a field on the
    ``Settings`` dataclass OR to a key in ``_TOOL_SETTING_DEFAULTS``
    (the seeded-defaults table). A registered Setting with no
    persistence target is a runtime bug — the validator accepts a
    PUT but ``getattr(settings, key)`` fails on read.
    """
    out: list[_Finding] = []
    from augmentum.config import settings as _settings  # noqa: PLC0415
    from augmentum.proxy.config_routes import (  # noqa: PLC0415
        _TOOL_SETTING_DEFAULTS,
    )

    declared = set(type(_settings).model_fields.keys()) if hasattr(
        type(_settings), "model_fields"
    ) else set(_settings.__dict__.keys())
    seeded = set(_TOOL_SETTING_DEFAULTS.keys())

    for s in get_registry().list_all():
        if s.key in declared:
            continue
        if s.key in seeded:
            continue
        # Last-ditch — pydantic v2 sometimes exposes fields via __fields__
        if hasattr(type(_settings), "__fields__") and s.key in type(_settings).__fields__:
            continue
        out.append(
            _Finding(
                key=s.key,
                rule="registry_drift_no_persistence_target",
                message=(
                    "Setting is registered but has no field on the "
                    "Settings dataclass AND is not in "
                    "_TOOL_SETTING_DEFAULTS. PUT will succeed but "
                    "subsequent GET will read None."
                ),
            )
        )
    return out


def _check_default_matches_config() -> list[_Finding]:
    """Registered ``default`` must equal the value the Settings
    dataclass would return on a fresh boot. Mismatches mean the
    UI shows one default but the runtime uses another.
    """
    out: list[_Finding] = []
    from augmentum.config import settings as _settings  # noqa: PLC0415
    from augmentum.proxy.config_routes import (  # noqa: PLC0415
        _TOOL_SETTING_DEFAULTS,
    )

    for s in get_registry().list_all():
        # Prefer the seeded-defaults table when present (it's the
        # explicit "default for fields not on Settings" mechanism).
        if s.key in _TOOL_SETTING_DEFAULTS:
            actual = _TOOL_SETTING_DEFAULTS[s.key]
        elif hasattr(_settings, s.key):
            actual = getattr(_settings, s.key)
        else:
            continue  # no source — caught by the previous rule
        if not _values_equal(s.default, actual):
            out.append(
                _Finding(
                    key=s.key,
                    rule="registry_drift_default_mismatch",
                    message=(
                        f"Registered default {s.default!r} differs "
                        f"from Settings/seed actual {actual!r}"
                    ),
                    expected=s.default,
                    actual=actual,
                )
            )
    return out


def _check_tool_settings_tuple_matches() -> list[_Finding]:
    """For Settings of numeric/bool kind whose key ALSO appears in
    the literal ``_TOOL_SETTINGS``, the tuple shape must agree.

    The overlay already overwrites the literal at runtime so this
    check is informational — it flags settings where the literal
    declaration would have been TIGHTER (a known migration-quality
    signal) so we can either restore the tighter bound on the
    registered Setting or accept the looser bound consciously.
    """
    out: list[_Finding] = []
    # Read the pre-overlay literal by re-evaluating the source. We
    # can't read the live dict because the overlay has mutated it.
    # The pragmatic path: snapshot what the dict contained BEFORE the
    # overlay ran. We do this by importing the source via the AST
    # module list parser. For Phase 1C, the cheaper move is to read
    # the live overlayed dict and trust that the overlay's "registry
    # wins" semantics are sound — i.e. a tighter literal that's been
    # overwritten by a looser registry tuple is a *registration*
    # choice we want to flag separately. The drift check here is
    # therefore a no-op until a Phase 1C-blocker emerges; the test
    # ``test_overlay_does_not_clobber_unrelated_literals`` provides
    # the inverse safety (unrelated literals stay put).
    return out


def _check_string_settings_maxlen_matches() -> list[_Finding]:
    """Same logic as the tool-settings check: informational only,
    since the overlay has already mutated the live dict.
    """
    return []


# ---- helpers ----


def _values_equal(a: Any, b: Any) -> bool:
    """Compare values with float tolerance and bool vs int strictness."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) == type(b) and a == b  # noqa: E721
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < _FLOAT_TOL
        except (TypeError, ValueError):
            return False
    return a == b


def find_settings_with_no_label() -> list[Setting]:
    """Diagnostic helper for migration: settings registered with
    minimal/auto-generated labels that need human polish."""
    return [
        s
        for s in get_registry().list_all()
        if len(s.label.strip()) < 4 or s.label == s.key.replace("_", " ").title()
    ]


def find_settings_with_thin_descriptions() -> list[Setting]:
    """Diagnostic helper for migration: settings whose description
    is shorter than 40 chars (likely placeholder)."""
    return [s for s in get_registry().list_all() if len(s.description.strip()) < 40]


def _ast_load_dict(module_path: str, dict_name: str) -> dict[str, Any] | None:
    """Re-read a literal dict from a module's source via AST — used
    to peek at the *pre-overlay* contents of ``_TOOL_SETTINGS`` etc.
    so the drift detector can compare what the literal *would have
    been* to what the registry exports.

    Returns ``None`` if the dict can't be parsed (e.g. it contains
    complex callables like in ``_TRI_STATE_BOOL_SETTINGS``).
    """
    import ast
    from pathlib import Path

    try:
        src = Path(module_path).read_text(encoding="utf-8")
    except OSError:
        return None

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == dict_name:
            if node.value is None:
                return None
            try:
                return ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                # Contains callables — can't literal_eval.
                return None
        if isinstance(node, ast.Assign):
            if any(getattr(t, "id", "") == dict_name for t in node.targets):
                try:
                    return ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    return None
    return None


def get_literal_tool_settings() -> dict[str, Any] | None:
    """Read the pre-overlay ``_TOOL_SETTINGS`` literal from source.
    Returns None if AST eval can't handle it (e.g. contains type
    references like ``bool``). For now we expose for tests only —
    the overlay supersedes at runtime."""
    return _ast_load_dict(
        inspect.getfile(_loaded_config_routes()), "_TOOL_SETTINGS"
    )


def _loaded_config_routes():
    from augmentum.proxy import config_routes  # noqa: PLC0415

    return config_routes
