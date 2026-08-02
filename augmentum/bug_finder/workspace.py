"""Workspace preparation for bug-finder runs.

The workspace is created and managed by the coder mode — bug finder
just receives an existing ``workspace_id`` and prepares it for an
audit pass. Preparation = strip remote refs / branches / tags / reflog
so the ``git log --all`` peek-at-fix-commit exploits have nothing to
find, plus a best-effort baseline (language + test runner detection,
baseline test output) used by the fix verifier to spot regressions.

No clone path lives here — workspaces are coder's concern, not the
audit pipeline's. The strip is idempotent so it's safe to re-run on
every audit pass.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field

from augmentum.coder.containers import ContainerManager
from augmentum.coder.models import ContainerInfo
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class WorkspaceBaseline:
    """Best-effort baseline state captured pre-detection.

    All fields are informational. ``detected_language`` is the dominant
    one (Python, Go, JS, etc.); ``test_command`` is the command the
    orchestrator believes runs the project's tests. Both come from
    simple file-presence heuristics.
    """

    detected_language: str = ""
    test_command: str = ""
    baseline_test_stdout: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class PreparedWorkspace:
    """Output of ``prepare_workspace`` — what the orchestrator needs."""

    workspace_id: str
    container_id: str | None
    baseline: WorkspaceBaseline


# ---------------------------------------------------------------------------
# Reward-hacking strip
# ---------------------------------------------------------------------------


# Run as one bash invocation so we minimize the number of exec round-trips
# into the container. ``set -e`` so any step failing is loud rather than
# silently leaving refs behind.
_STRIP_SCRIPT = r"""
set -eu
cd /workspace

# 1. Remove origin (prevents fetch-back of stripped refs).
if git remote get-url origin >/dev/null 2>&1; then
  git remote remove origin
fi

# 2. Delete every remote-tracking ref.
for r in $(git for-each-ref refs/remotes --format='%(refname)'); do
  git update-ref -d "$r" 2>/dev/null || true
done

# 3. Delete every tag.
for t in $(git for-each-ref refs/tags --format='%(refname:short)'); do
  git tag -d "$t" >/dev/null 2>&1 || true
done

# 4. Delete every branch except HEAD's current.
current=$(git symbolic-ref --quiet --short HEAD || echo "")
for b in $(git for-each-ref refs/heads --format='%(refname:short)'); do
  if [ "$b" != "$current" ]; then
    git branch -D "$b" >/dev/null 2>&1 || true
  fi
done

# 5. Wipe the reflog so old commits aren't reachable via @{1}, @{2}, etc.
git reflog expire --expire=now --all >/dev/null 2>&1 || true

# 6. Garbage-collect aggressively so dangling commits become unreachable
# even by oid-prefix guessing.
git gc --prune=now --aggressive >/dev/null 2>&1 || true

echo "stripped"
"""


async def _strip_refs(cm: ContainerManager, workspace_id: str) -> None:
    """Run the reward-hacking-defense strip script inside the container."""
    try:
        out = await cm.run_command(
            workspace_id,
            ["bash", "-c", _STRIP_SCRIPT],
            timeout=120.0,
        )
        log.info(
            "bug_finder_workspace_stripped",
            workspace_id=workspace_id,
            output_tail=out.strip().splitlines()[-1] if out.strip() else "",
        )
    except Exception:  # noqa: BLE001
        log.warning(
            "bug_finder_workspace_strip_failed",
            workspace_id=workspace_id,
            exc_info=True,
        )
        # We don't fail the run — the strip is defense-in-depth. The
        # detector and fixer guards block `git log --all` etc. at the
        # tool-call layer regardless of whether the refs are present.


# ---------------------------------------------------------------------------
# Baseline detection
# ---------------------------------------------------------------------------


_LANGUAGE_PROBES: tuple[tuple[str, str], ...] = (
    ("pyproject.toml", "python"),
    ("setup.py", "python"),
    ("requirements.txt", "python"),
    ("go.mod", "go"),
    ("Cargo.toml", "rust"),
    ("package.json", "javascript"),
    ("composer.json", "php"),
    ("pom.xml", "java"),
    ("build.gradle", "java"),
    ("Gemfile", "ruby"),
)


_TEST_COMMANDS: dict[str, str] = {
    "python": "pytest -x --tb=short 2>&1 | tail -50",
    "go": "go test ./... 2>&1 | tail -50",
    "rust": "cargo test 2>&1 | tail -50",
    "javascript": "npm test 2>&1 | tail -50",
    "ruby": "bundle exec rake test 2>&1 | tail -50",
}


async def _detect_baseline(
    cm: ContainerManager, workspace_id: str,
) -> WorkspaceBaseline:
    """Probe for language + test runner and run baseline tests."""
    baseline = WorkspaceBaseline()
    notes: list[str] = []

    # Language detection — first match wins.
    for fname, language in _LANGUAGE_PROBES:
        try:
            out = await cm.run_command(
                workspace_id,
                ["bash", "-c", f"test -f /workspace/{shlex.quote(fname)} && echo yes || echo no"],
                timeout=5.0,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("language_probe_failed", file=fname, error=str(exc))
            continue
        if out.strip() == "yes":
            baseline.detected_language = language
            notes.append(f"language={language} (matched {fname})")
            break
    if not baseline.detected_language:
        notes.append("language detection inconclusive — verifier may need to guess")

    # Baseline test run — best effort, capped at 5 minutes wall-clock.
    test_cmd = _TEST_COMMANDS.get(baseline.detected_language, "")
    if test_cmd:
        baseline.test_command = test_cmd
        try:
            out = await cm.run_command(
                workspace_id,
                ["bash", "-c", f"cd /workspace && {test_cmd}"],
                timeout=300.0,
            )
            baseline.baseline_test_stdout = out[-4096:]  # keep the tail
            notes.append("baseline test run captured")
        except Exception:  # noqa: BLE001
            notes.append("baseline test run failed (continuing — fix-verifier handles regressions independently)")
            log.info(
                "bug_finder_baseline_tests_failed",
                workspace_id=workspace_id,
                exc_info=True,
            )

    baseline.notes = notes
    return baseline


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def prepare_workspace(
    *,
    cm: ContainerManager,
    workspace_id: str,
) -> PreparedWorkspace:
    """Prepare an existing coder workspace for an audit pass.

    Idempotent — the ref/tag strip and baseline probes can run on every
    audit without breaking anything; they're best-effort and short-
    circuit when there's nothing to do.
    """
    if not workspace_id:
        raise ValueError("prepare_workspace requires a non-empty workspace_id")
    info: ContainerInfo = await cm._get_workspace(workspace_id)  # noqa: SLF001
    await _strip_refs(cm, info.id)
    baseline = await _detect_baseline(cm, info.id)
    return PreparedWorkspace(
        workspace_id=info.id,
        container_id=info.container_id,
        baseline=baseline,
    )
