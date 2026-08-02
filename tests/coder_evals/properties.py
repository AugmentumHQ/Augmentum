"""Assertion helpers for coder eval cases.

Each helper returns ``(bool, reason)``. The runner aggregates failures so
a single case can report multiple violations rather than stopping at the
first. Keep helpers pure: input is the post-turn result bundle, output
is a verdict + human-readable reason.

The result bundle shape (built by ``test_runner._run_case``):

    {
        "files": {path: content},        # final workspace state
        "files_changed": [path, ...],    # diff vs. initial
        "tools_used": [name, ...],       # ordered tool-call names
        "iterations": int,               # _act_hybrid iterations
        "tokens_used": int,              # total token spend
        "tier": str | None,              # set when Phase 1 lands
        "verification": dict | None,     # set when Phase 3 lands
    }
"""
from __future__ import annotations

import re
from typing import Any


def file_contains(result: dict, *, path: str, text: str) -> tuple[bool, str]:
    content = result["files"].get(path)
    if content is None:
        return False, f"file {path} missing from workspace"
    ok = text in content
    return ok, "" if ok else f"{path} does not contain {text!r}"


def file_equals(result: dict, *, path: str, content: str) -> tuple[bool, str]:
    actual = result["files"].get(path)
    if actual is None:
        return False, f"file {path} missing from workspace"
    ok = actual == content
    return ok, "" if ok else f"{path} content mismatch"


def file_matches_regex(result: dict, *, path: str, pattern: str) -> tuple[bool, str]:
    content = result["files"].get(path)
    if content is None:
        return False, f"file {path} missing from workspace"
    ok = bool(re.search(pattern, content))
    return ok, "" if ok else f"{path} does not match /{pattern}/"


def file_unchanged(result: dict, *, path: str) -> tuple[bool, str]:
    if path in result["files_changed"]:
        return False, f"{path} was modified but should be unchanged"
    return True, ""


def tools_used_includes(result: dict, *, names: list[str]) -> tuple[bool, str]:
    missing = [n for n in names if n not in result["tools_used"]]
    return (not missing), "" if not missing else f"expected tools missing: {missing}"


def tools_used_excludes(result: dict, *, names: list[str]) -> tuple[bool, str]:
    found = [n for n in names if n in result["tools_used"]]
    return (not found), "" if not found else f"forbidden tools used: {found}"


def max_iterations(result: dict, *, n: int) -> tuple[bool, str]:
    iters = result.get("iterations", 0)
    ok = iters <= n
    return ok, "" if ok else f"used {iters} iterations, max allowed {n}"


def max_tokens(result: dict, *, n: int) -> tuple[bool, str]:
    tok = result.get("tokens_used", 0)
    ok = tok <= n
    return ok, "" if ok else f"used {tok} tokens, max allowed {n}"


# Phase-gated assertions — defined now, activated when the corresponding
# phase lands. Until then they no-op (return True) rather than failing,
# so Phase 0 cases can declare them without the runner exploding.

def tier_classified_as(result: dict, *, expected: str) -> tuple[bool, str]:
    actual = result.get("tier")
    if actual is None:
        return True, "tier classifier not yet wired (Phase 1)"
    ok = actual == expected
    return ok, "" if ok else f"tier {actual!r} != expected {expected!r}"


def verification_gate_passed(result: dict, *, _: Any = None) -> tuple[bool, str]:
    v = result.get("verification")
    if v is None:
        return True, "verification gate not yet wired (Phase 3)"
    failures = [k for k, ok in v.items() if not ok]
    return (not failures), "" if not failures else f"gate failures: {failures}"


REGISTRY = {
    "file_contains": file_contains,
    "file_equals": file_equals,
    "file_matches_regex": file_matches_regex,
    "file_unchanged": file_unchanged,
    "tools_used_includes": tools_used_includes,
    "tools_used_excludes": tools_used_excludes,
    "max_iterations": max_iterations,
    "max_tokens": max_tokens,
    "tier_classified_as": tier_classified_as,
    "verification_gate_passed": verification_gate_passed,
}


def apply_assertions(result: dict, assertions: list[dict]) -> list[str]:
    """Run all declared assertions; return list of failure reasons."""
    failures: list[str] = []
    for spec in assertions or []:
        name = spec.get("property")
        if name not in REGISTRY:
            failures.append(f"unknown property: {name}")
            continue
        kwargs = {k: v for k, v in spec.items() if k != "property"}
        ok, reason = REGISTRY[name](result, **kwargs)
        if not ok:
            failures.append(f"{name}: {reason}")
    return failures
