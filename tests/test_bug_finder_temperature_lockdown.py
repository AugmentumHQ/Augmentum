"""Regression: every SubagentSpec in bug_finder must set temperature=0.0.

Background — every call shipped at the API default (1.0) for months
because the SubagentSpec ``temperature`` field defaults to ``None``
and the agents stack omits the field from API requests when None.
That made the bug_finder pipeline non-deterministic: planner picks
different chunks per run, detector emits different findings on the
same chunks, verifier flips on identical claims. Observed: 0/1/2/3
findings across 5 runs of the same vuln_app target.

This file scans every ``SubagentSpec(`` literal under
``augmentum/bug_finder/`` and asserts ``temperature=0.0`` appears
within the constructor block. A future site that forgets it will
fail this test and the variance regression will be caught at PR
time, not when the subagent caller can't reproduce a finding.
"""

from __future__ import annotations

import ast
from pathlib import Path

BUG_FINDER_DIR = (
    Path(__file__).resolve().parent.parent / "augmentum" / "bug_finder"
)


def _is_config_detector_temperature(node: ast.AST) -> bool:
    """True when ``node`` is the AST for ``config.detector_temperature``.

    Sole exception to the literal-0.0 rule, carved out so the detector
    site can route through ``BugFinderRunConfig.detector_temperature``
    (defaults to 0.0; only research overrides raise it). The detector
    is the one role where a variance bench or thinking-mode experiment
    legitimately wants to sweep temperature.
    """
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "detector_temperature"
        and isinstance(node.value, ast.Name)
        and node.value.id == "config"
    )


def _subagent_spec_call_sites() -> list[tuple[Path, int, ast.Call]]:
    """Return every ``SubagentSpec(...)`` call site under bug_finder/.

    Walks each .py file as AST. Returns ``(path, lineno, call_node)``
    tuples for every call where the function id is ``SubagentSpec``.
    """
    sites: list[tuple[Path, int, ast.Call]] = []
    for py in BUG_FINDER_DIR.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == "SubagentSpec" or (
                isinstance(fn, ast.Attribute) and fn.attr == "SubagentSpec"
            ):
                sites.append((py, node.lineno, node))
    return sites


def test_every_subagent_spec_pins_temperature_zero() -> None:
    """Every ``SubagentSpec(...)`` literal must include
    ``temperature=0.0`` as a kwarg.

    This is the determinism floor for the bug_finder: under
    temperature=1.0 (the API default when omitted) the planner picks
    different chunks per run and the variance compounds through the
    pipeline. A subagent caller can't reason about findings if the
    same call gives different answers each time.
    """
    sites = _subagent_spec_call_sites()
    assert sites, (
        "No SubagentSpec call sites found under augmentum/bug_finder/ — "
        "the audit target may have moved; update BUG_FINDER_DIR."
    )

    offenders: list[str] = []
    for path, lineno, call in sites:
        temp_kwarg = next(
            (kw for kw in call.keywords if kw.arg == "temperature"),
            None,
        )
        if temp_kwarg is None:
            offenders.append(
                f"{path.relative_to(BUG_FINDER_DIR.parent.parent)}:{lineno} "
                f"— SubagentSpec without temperature= kwarg",
            )
            continue
        # Accept ast.Constant(0.0), ast.Constant(0), Num(0/0.0) for older
        # Python compat. Anything else (None, variable reference, non-zero)
        # is treated as suspect — the deterministic baseline only holds
        # at exactly 0.0.
        val = temp_kwarg.value
        if isinstance(val, ast.Constant) and val.value in (0, 0.0):
            continue
        # Single carve-out: the detector site may read
        # ``config.detector_temperature``. The config field's default is
        # 0.0 so the standard pipeline preserves the lockdown; only
        # explicit research overrides (variance benches, temp sweeps,
        # thinking-mode experiments) raise it. Defined at
        # ``BugFinderRunConfig.detector_temperature`` — if you rename the
        # field, update this check.
        if _is_config_detector_temperature(val):
            continue
        offenders.append(
            f"{path.relative_to(BUG_FINDER_DIR.parent.parent)}:{lineno} "
            f"— SubagentSpec temperature= is not the literal 0.0 "
            f"(found {ast.dump(val)})",
        )

    assert not offenders, (
        f"{len(offenders)} SubagentSpec site(s) violate the temperature "
        f"lockdown:\n  - " + "\n  - ".join(offenders) +
        "\nAdd `temperature=0.0,` to each. See determinism audit notes "
        "for the rationale."
    )


def test_subagent_spec_count_matches_expected_roles() -> None:
    """Soft sanity: the bug_finder ships ~10 SubagentSpec sites
    (planner, detector, fixer, comprehender, lead, investigator,
    check_writer, pen_tester, verify-is-real, verify-fix). If this
    count drifts dramatically, a role was added or removed without
    updating this test — flag it for review.

    The exact number isn't sacred — the check is "are we still
    auditing the same shape." Fail loudly if it goes <5 or >25 so
    the next maintainer notices.
    """
    sites = _subagent_spec_call_sites()
    assert 5 <= len(sites) <= 25, (
        f"Expected ~10 SubagentSpec sites (one per bug_finder role), "
        f"got {len(sites)}. If a role was added/removed, update the "
        f"bounds; if this count is unexpectedly off, audit recent "
        f"refactors."
    )
