"""Build quality gate — did the build agent actually PROVE the app works?

This is the in-package, runtime version of the floor judge that
``scripts/builder_eval.py`` applies offline. The facade calls it the moment
a build's autonomous loop finishes, grading the agent's *actual tool trail*
against the Frontend App Builder Power's definition of done
(``.augmentum/powers/frontend-app/POWER.md``):

  * wrote real code          — file_write / code_edit present
  * ran a dev server         — service_start present
  * opened it in a browser   — browser_open present
  * drove it like a user     — browser_click / browser_type / browser_fill_form present
  * ASSERTED behavior        — browser_evaluate / browser_verify / browser_extract present
  * finished clean           — status completed (finish_task)
  * published a real artifact — an artifact id resolved

The point: a model that merely writes plausible files and calls
``finish_task`` without ever serving + driving + asserting the app reports
``complete`` today and the library shows "Built ✓" — exactly how a
CSS-broken, unusable app slips through. Running this verdict inline turns
"the loop finished" into "the loop is trustworthy", and lets the surface
mark a completion ``unverified`` (with the specific unproven checks) so the
user — and the resume path — know what's left to prove.

Pure + dependency-free so it unit-tests without Docker/a backend, and so
``scripts/builder_eval.py`` can import the same judge instead of duplicating
it.
"""

from __future__ import annotations

from typing import Any

# Tool families — what each builder tool proves about the build. Names mirror
# BUILDER_CODER_TOOL_NAMES in augmentum/builds/facade.py.
WRITE_TOOLS: frozenset[str] = frozenset({"file_write", "code_edit", "code_edit_batch", "apply_patch"})
SERVE_TOOLS: frozenset[str] = frozenset({"service_start"})
OPEN_TOOLS: frozenset[str] = frozenset({"browser_open"})
DRIVE_TOOLS: frozenset[str] = frozenset({"browser_click", "browser_type", "browser_fill_form"})
ASSERT_TOOLS: frozenset[str] = frozenset({"browser_evaluate", "browser_verify", "browser_extract"})
RESOURCE_TOOLS: frozenset[str] = frozenset({"builder_design_system", "builder_reference", "builder_api_refs"})

# Human-readable label per hard-floor check, for the user-facing warnings list.
_CHECK_LABELS: dict[str, str] = {
    "wrote_code": "wrote application code",
    "ran_server": "started a dev server",
    "opened_browser": "opened the app in a browser",
    "drove_ui": "drove the UI like a user",
    "asserted_behavior": "asserted that behaviors actually work",
    "finished_clean": "finished cleanly",
    "published_artifact": "published a runnable artifact",
}

# Per-kind verification-floor depth thresholds, lifted from the Power's
# "Minimum verification floor by kind". Lenient counts — the binary floor
# (drove + asserted at all) is the hard gate; depth is a quality signal.
_KIND_FLOOR: dict[str, tuple[int, int]] = {
    "calculator": (3, 3),
    "form": (2, 2),
    "dashboard": (1, 1),
    "game": (1, 2),
    "app": (3, 3),
    "static": (1, 1),
}
_DEFAULT_FLOOR = (1, 1)


def floor_for_kind(kind: str) -> tuple[int, int]:
    """(min_drives, min_asserts) depth thresholds for a build kind."""
    return _KIND_FLOOR.get((kind or "").strip().lower(), _DEFAULT_FLOOR)


def _counts(tool_names: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in tool_names:
        t = str(name or "")
        if t:
            out[t] = out.get(t, 0) + 1
    return out


def _sum(counts: dict[str, int], family: frozenset[str]) -> int:
    return sum(n for t, n in counts.items() if t in family)


def judge_tool_names(
    tool_names: list[str],
    *,
    status: str,
    artifact_ok: bool,
    kind: str,
    min_drives: int | None = None,
    min_asserts: int | None = None,
    has_files: bool = False,
) -> dict[str, Any]:
    """Grade a build from the flat list of tool names it invoked.

    ``has_files`` lets the caller assert the build's output already contains
    real source (e.g. the published artifact has files). It satisfies the
    ``wrote_code`` floor on its own — important for a *resume* that only
    verifies an already-built app: such a session legitimately writes no new
    files, yet the app demonstrably has code, so requiring a fresh ``file_write``
    would false-negative a thorough verify-only continuation.

    Returns a verdict dict: per-check booleans (``hard``), a 0-1 ``score`` over
    the hard floor, overall ``passed`` (every hard check true), per-kind
    ``depth`` checks, ``soft`` signals, ``failed_checks`` (machine ids) and
    ``unproven`` (human labels for the surface), plus a tool-count summary.
    """
    if min_drives is None or min_asserts is None:
        d, a = floor_for_kind(kind)
        min_drives = d if min_drives is None else min_drives
        min_asserts = a if min_asserts is None else min_asserts

    counts = _counts(tool_names)
    n_drive = _sum(counts, DRIVE_TOOLS)
    n_assert = _sum(counts, ASSERT_TOOLS)

    hard = {
        "wrote_code": _sum(counts, WRITE_TOOLS) > 0 or bool(has_files),
        "ran_server": _sum(counts, SERVE_TOOLS) > 0,
        "opened_browser": _sum(counts, OPEN_TOOLS) > 0,
        "drove_ui": n_drive > 0,
        "asserted_behavior": n_assert > 0,
        "finished_clean": str(status).lower() in ("complete", "completed"),
        "published_artifact": bool(artifact_ok),
    }
    depth = {
        "enough_drives": n_drive >= int(min_drives),
        "enough_asserts": n_assert >= int(min_asserts),
    }
    soft = {"pulled_resources": _sum(counts, RESOURCE_TOOLS) > 0}

    failed = [k for k, v in hard.items() if not v]
    score = sum(1 for v in hard.values() if v) / len(hard)
    return {
        "kind": kind,
        "passed": all(hard.values()),
        "score": round(score, 3),
        "hard": hard,
        "depth": depth,
        "soft": soft,
        "failed_checks": failed,
        "unproven": [_CHECK_LABELS.get(k, k) for k in failed],
        "tool_counts": counts,
        "step_count": len(tool_names),
        "drives": n_drive,
        "asserts": n_assert,
    }


def judge_build(
    *,
    steps: list[dict],
    status: str,
    artifact_ok: bool,
    kind: str,
    min_drives: int | None = None,
    min_asserts: int | None = None,
) -> dict[str, Any]:
    """Grade a build from the ordered ``steps`` trail of ``{tool, ...}`` dicts.

    Thin wrapper over :func:`judge_tool_names` kept for the offline eval
    harness (``scripts/builder_eval.py``), which carries the step trail from
    the build snapshot rather than the raw tool-call log.
    """
    names = [str((s or {}).get("tool") or "") for s in (steps or [])]
    return judge_tool_names(
        names, status=status, artifact_ok=artifact_ok, kind=kind,
        min_drives=min_drives, min_asserts=min_asserts,
    )


def behavior_verdict(
    behaviors: list[dict], *, status: str, artifact_ok: bool,
) -> dict[str, Any]:
    """Outcome-based verdict: grade the build by how many spec-derived
    behaviors ACTUALLY PASSED in a real browser, not by which tools the agent
    called. ``passed`` (the gate) requires a clean finish, a published
    artifact, AND every checked behavior green. The ``mode: outcome`` tag lets
    the surface distinguish this from the trail-based fallback."""
    passed = [b for b in behaviors if b.get("status") == "pass"]
    failed = [b for b in behaviors if b.get("status") == "fail"]
    checked = passed + failed
    finished_clean = str(status).lower() in ("complete", "completed")
    all_passed = bool(checked) and not failed
    score = (len(passed) / len(checked)) if checked else 0.0
    return {
        "mode": "outcome",
        "passed": bool(finished_clean and artifact_ok and all_passed),
        "score": round(score, 3),
        "behaviors_total": len(behaviors),
        "behaviors_checked": len(checked),
        "behaviors_passed": len(passed),
        "behaviors_failed": len(failed),
        "failed": [
            {"id": b.get("id", ""), "description": b.get("description", ""),
             "evidence": (b.get("evidence") or "")[:200]}
            for b in failed
        ],
    }


def behavior_quality_summary(verdict: dict[str, Any]) -> dict[str, Any]:
    """Surface fields from an outcome verdict: ``unverified`` lists the
    behaviors that FAILED (real defects), not just 'never asserted'."""
    if verdict.get("passed"):
        return {"qualityStatus": "clean", "warnings": [], "blockingErrors": []}
    warnings = [
        "Behavior failed: " + f.get("description", "")
        + (f" — {f['evidence']}" if f.get("evidence") else "")
        for f in (verdict.get("failed") or [])
    ]
    if not warnings and not verdict.get("behaviors_checked"):
        warnings = ["No stated behavior could be verified in a browser."]
    return {"qualityStatus": "unverified", "warnings": warnings, "blockingErrors": []}


def quality_summary(verdict: dict[str, Any], *, final_status: str) -> dict[str, Any]:
    """Translate a verdict into the surface's quality fields.

    Returns ``{qualityStatus, warnings, blockingErrors}``:

    * a build that did not finish cleanly is left to the normal failure
      surface (qualityStatus ``clean`` — the failure itself is the signal);
    * a build that finished ``completed`` but missed the hard floor is
      ``unverified`` with a warning per unproven check, so the user sees the
      gap before opening the app and the resume path knows what to finish.
    """
    if not isinstance(verdict, dict) or not verdict:
        return {"qualityStatus": "clean", "warnings": [], "blockingErrors": []}
    finished_clean = str(final_status).lower() in ("complete", "completed")
    if not finished_clean:
        return {"qualityStatus": "clean", "warnings": [], "blockingErrors": []}
    if verdict.get("passed"):
        return {"qualityStatus": "clean", "warnings": [], "blockingErrors": []}
    unproven = verdict.get("unproven") or []
    warnings = [f"Not proven: the build never {label}." for label in unproven]
    return {
        "qualityStatus": "unverified",
        "warnings": warnings,
        "blockingErrors": [],
    }
