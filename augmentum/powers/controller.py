"""Checkpoint-based controller selection for Augmentum Powers.

This layer is intentionally narrow: it does not execute tools or mutate
workspace state. It only decides whether a controller-managed Power
should be engaged at a safe loop checkpoint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from augmentum.powers.models import PowerManifest

_FRONTEND_EXTS = {
    ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".scss", ".sass",
    ".vue", ".svelte",
}
_BACKEND_EXTS = {
    ".py", ".go", ".rb", ".java", ".kt", ".rs", ".cs", ".php",
}
_TEST_MARKERS = (
    "/tests/",
    "/__tests__/",
    ".spec.",
    ".test.",
    "_test.",
    "test_",
)
_MIGRATION_MARKERS = (
    "/migrations/",
    "/migration/",
    "/alembic/",
    "schema",
    "backfill",
    ".sql",
)
_CONTRACT_MARKERS = (
    "payload",
    "response",
    "request",
    "header",
    "contract",
    "api",
    "route",
    "schema",
    "dto",
)
_MIGRATION_TEXT_MARKERS = (
    "migration",
    "migrate",
    "schema change",
    "alter table",
    "backfill",
    "column",
    "database",
)
_FAILURE_TRIAGE_TEXT_MARKERS = (
    "bug",
    "root cause",
    "reproducer",
    "repro",
    "debug",
    "failing test",
    "failing tests",
    "failing command",
    "timestamp parsing",
    "traceback",
    "exception",
    "crash",
    "bad gateway",
    "timed out",
    "timeout",
    "503",
    "502",
    "interstitial",
    "auth token",
    "authentication failed",
)
_TEST_AUTHOR_TEXT_MARKERS = (
    "regression test",
    "regression tests",
    "add tests",
    "write tests",
    "improve coverage",
)
_DEPENDENCY_TEXT_MARKERS = (
    "dependency",
    "dependencies",
    "install",
    "package manager",
    "module not found",
    "no module named",
    "cannot find module",
    "import error",
    "lockfile",
    "environment",
    "requirements.txt",
    "pyproject",
    "package.json",
    "package-lock",
    "pnpm-lock",
    "yarn.lock",
)
_PERFORMANCE_TEXT_MARKERS = (
    "performance",
    "slow",
    "latency",
    "memory leak",
    "cpu",
    "benchmark",
    "profile",
    "profiling",
    "throughput",
    "p95",
    "p99",
)
_WORKSPACE_ONBOARDING_TEXT_MARKERS = (
    "new workspace",
    "what is this project",
    "onboard",
    "explore repo",
    "setup project",
    "understand this repo",
)
_MULTI_TENANT_TEXT_MARKERS = (
    "new table",
    "user_id",
    "route handler",
    "new endpoint",
    "crud",
    "data isolation",
    "tenant",
    "multi-tenant",
    "multi tenant",
    "persistence",
    "user-scoped",
    "user scoped",
)
_MULTI_TENANT_WEAK_TEXT_MARKERS = (
    "migration",
    "migrate",
    "schema",
)
_OBSERVATION_TEXT_MARKERS = (
    "took a while to figure out",
    "turned out that",
    "not obvious",
    "tricky",
    "gotcha",
    "had to discover",
    "remember that",
    "record this",
)
_SUBAGENT_TEXT_MARKERS = (
    "find every",
    "find all",
    "where does",
    "second opinion",
    "review my",
    "audit",
    "security",
    "design decision",
    "which approach",
    "research",
    "subagent",
    "subagents",
)
_CHANGELOG_TEXT_MARKERS = (
    "summarize the changes",
    "commit message",
    "changelog",
    "what did i change",
    "pr description",
    "pull request description",
    "handoff notes",
)
_BASELINE_TEXT_MARKERS = (
    "regression",
    "before and after",
    "benchmark",
    "timing",
    "did this break",
    "baseline",
    "compare before",
)
_ARTIFACT_QUOTE_MARKERS = (
    "you are about to visit",
    "this tunnel is hosted by",
    "please proceed with caution",
    "localtunnel is not an anonymizing service",
    "browseruseragent",
    "abypass-tunnel-reminder",
    "are you the tunnel host",
    "ip address:",
    "sign up for an account",
    "install your authtoken",
    "err_ngrok_",
)
_ARTIFACT_INLINE_MARKERS = (
    "this tunnel is hosted by",
    "you are about to visit",
    "please proceed with caution",
    "phishing",
    "browseruseragent",
    "browser user-agent",
    "abypass-tunnel-reminder",
    "are you the tunnel host",
    "localtunnel is not an anonymizing service",
    "sign up for an account",
    "install your authtoken",
    "err_ngrok_",
)
_LONG_DOUBLE_QUOTED_BLOCK_RE = re.compile(r'"([^"]{80,})"', re.DOTALL)
_SEGMENT_SPLIT_RE = re.compile(r"(?:\r?\n+|(?<=[.!?])\s+)")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class PowerSelection:
    """Controller-selected Power plus the reason it won."""

    manifest: PowerManifest
    reason: str
    score: int


@dataclass(slots=True)
class _PathFacts:
    frontend: bool = False
    backend: bool = False
    tests: bool = False
    non_test_source: bool = False
    migration: bool = False
    contractish: bool = False


def _analyze_paths(paths: list[str] | tuple[str, ...]) -> _PathFacts:
    facts = _PathFacts()
    for raw in paths:
        path = (raw or "").strip().lower()
        if not path:
            continue
        posix = PurePosixPath(path.replace("\\", "/"))
        suffix = posix.suffix.lower()
        is_frontend = suffix in _FRONTEND_EXTS
        is_backend = suffix in _BACKEND_EXTS
        is_test = any(marker in path for marker in _TEST_MARKERS)
        if is_frontend:
            facts.frontend = True
        if is_backend:
            facts.backend = True
        if is_test:
            facts.tests = True
        if (is_frontend or is_backend) and not is_test:
            facts.non_test_source = True
        if any(marker in path for marker in _MIGRATION_MARKERS):
            facts.migration = True
        if any(marker in path for marker in _CONTRACT_MARKERS):
            facts.contractish = True
    return facts


def _trigger_hits(manifest: PowerManifest, text: str) -> list[str]:
    lower = text.lower()
    return [trigger for trigger in manifest.triggers if trigger.lower() in lower]


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _looks_like_test_author_request(text: str) -> bool:
    return any(marker in text for marker in _TEST_AUTHOR_TEXT_MARKERS)


def _selector_focus_text(text: str) -> str:
    """Return a task-focused variant of user text for lexical heuristics.

    The controller should react to the user's goal, not blindly to copied
    browser/interstitial/tool output. We strip long quoted artifact blocks and
    highly specific tunnel-warning phrases before running contract/migration
    heuristics, while still allowing the full text to drive failure triage.
    """
    if not text:
        return ""

    def _replace_quoted(match: re.Match[str]) -> str:
        block = match.group(1).lower()
        if any(marker in block for marker in _ARTIFACT_QUOTE_MARKERS):
            return " "
        return match.group(0)

    focused = _LONG_DOUBLE_QUOTED_BLOCK_RE.sub(_replace_quoted, text)
    kept_segments: list[str] = []
    for segment in _SEGMENT_SPLIT_RE.split(focused):
        lower = segment.lower().strip()
        if not lower:
            continue
        if any(marker in lower for marker in _ARTIFACT_INLINE_MARKERS):
            continue
        kept_segments.append(segment)
    return _WHITESPACE_RE.sub(" ", " ".join(kept_segments)).strip()


def select_controller_power(
    manifests: list[PowerManifest],
    *,
    checkpoint: str,
    latest_user_text: str = "",
    edited_paths: list[str] | tuple[str, ...] = (),
    current_controller_power_id: str = "",
    manual_power_id: str = "",
) -> PowerSelection | None:
    """Pick the best controller-managed Power for a checkpoint.

    This is a heuristic selector, not an ML policy. It prefers:
    - explicit trigger hits for planning checkpoints
    - file-change signals for post-write checkpoints
    - deterministic verifier picks for failure/release checkpoints

    Manual workspace Powers are treated as the primary strategy for the
    turn. When one is pinned, controller overlays are suppressed during
    planning/implementation and restricted to verifier-style checkpoints
    later in the loop.
    """

    full_text = (latest_user_text or "").strip().lower()
    focus_text = _selector_focus_text(full_text)
    facts = _analyze_paths(edited_paths)
    best: PowerSelection | None = None

    if manual_power_id and checkpoint in {"pre_plan", "implementation"}:
        return None

    for manifest in manifests:
        if manifest.activation_policy != "controller":
            continue
        if checkpoint not in manifest.activation_windows:
            continue
        if manual_power_id:
            if manifest.id == manual_power_id and checkpoint != "pre_finish":
                continue
            if manifest.kind != "verifier":
                continue

        score = 0
        reasons: list[str] = []
        trigger_hits = _trigger_hits(manifest, focus_text)
        if trigger_hits:
            score += 20 * len(trigger_hits)
            reasons.append(f"matched trigger(s): {', '.join(trigger_hits[:3])}")

        if manifest.id == "migration-safety":
            if _has_any(focus_text, _MIGRATION_TEXT_MARKERS) or facts.migration:
                score += 80
                reasons.append("migration/schema signals detected")

        elif manifest.id == "contract-keeper":
            if _has_any(focus_text, _CONTRACT_MARKERS):
                score += 70
                reasons.append("API/contract language detected")
            if facts.frontend and facts.backend:
                score += 75
                reasons.append("both frontend and backend files changed")
            elif facts.contractish:
                score += 55
                reasons.append("contract-sensitive files changed")

        elif manifest.id == "browser-verification":
            if facts.frontend:
                score += 85 if checkpoint == "post_write" else 35
                reasons.append("frontend/browser-facing files changed")

        elif manifest.id == "test-author":
            if checkpoint == "post_write" and facts.non_test_source:
                score += 75
                reasons.append("source files changed and focused tests likely needed")
                if facts.tests:
                    score -= 20
                    reasons.append("test files already changed in this turn")

        elif manifest.id == "failure-triage":
            if checkpoint == "pre_plan" and _has_any(
                full_text, _FAILURE_TRIAGE_TEXT_MARKERS
            ) and not _looks_like_test_author_request(focus_text or full_text):
                score += 90
                reasons.append("bug/regression language detected")
            if checkpoint == "verify_failed":
                score += 100
                reasons.append("verification failed; triage mode is appropriate")

        elif manifest.id == "release-review":
            if checkpoint == "pre_finish" and edited_paths:
                score += 95
                reasons.append("turn has landed edits and is approaching completion")

        elif manifest.id == "dependency-doctor":
            if _has_any(focus_text, _DEPENDENCY_TEXT_MARKERS):
                score += 85 if checkpoint == "verify_failed" else 75
                reasons.append("dependency/environment signals detected")

        elif manifest.id == "performance-profiler":
            if _has_any(focus_text, _PERFORMANCE_TEXT_MARKERS):
                score += 90 if checkpoint == "verify_failed" else 80
                reasons.append("performance/profiling signals detected")

        elif manifest.id == "workspace-onboarding":
            if checkpoint == "pre_plan" and _has_any(
                focus_text, _WORKSPACE_ONBOARDING_TEXT_MARKERS
            ):
                score += 75
                reasons.append("workspace orientation requested")

        elif manifest.id == "multi-tenant-auditor":
            if _has_any(focus_text, _MULTI_TENANT_TEXT_MARKERS):
                score += 90
                reasons.append("multi-tenant data-isolation signals detected")
            elif _has_any(focus_text, _MULTI_TENANT_WEAK_TEXT_MARKERS) or facts.migration:
                score += 45
                reasons.append("persistence/schema signals may need tenant scoping")
            if checkpoint in {"post_write", "pre_finish"} and facts.contractish:
                score += 35
                reasons.append("route/API-adjacent files changed")

        elif manifest.id == "observation-keeper":
            if _has_any(focus_text, _OBSERVATION_TEXT_MARKERS):
                score += 120
                reasons.append("durable workspace discovery should be recorded")

        elif manifest.id == "subagent-router":
            if _has_any(focus_text, _SUBAGENT_TEXT_MARKERS):
                score += 85
                reasons.append("wide search/review/delegation signals detected")

        elif manifest.id == "changelog-documenter":
            if _has_any(focus_text, _CHANGELOG_TEXT_MARKERS):
                score += 125
                reasons.append("change summary or handoff requested")
            elif checkpoint == "pre_finish" and edited_paths:
                score += 40
                reasons.append("edited files may need a grounded handoff summary")

        elif manifest.id == "test-baseline-keeper":
            if _has_any(focus_text, _BASELINE_TEXT_MARKERS):
                score += 95 if checkpoint == "pre_finish" else 85
                reasons.append("baseline/regression comparison requested")

        if checkpoint == "pre_finish" and manifest.id != "release-review" and score > 0:
            score -= 25

        if current_controller_power_id and manifest.id == current_controller_power_id and score > 0:
            score += 5

        if score <= 0:
            continue

        selection = PowerSelection(
            manifest=manifest,
            reason="; ".join(reasons) or f"selected for {checkpoint}",
            score=score,
        )
        if best is None or selection.score > best.score:
            best = selection

    return best
