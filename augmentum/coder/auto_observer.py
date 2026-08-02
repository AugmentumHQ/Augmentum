"""Auto-observation extractor — fills the observation ledger without
the agent having to call the ``observe`` tool explicitly.

Motivation: pre-2026-05-31 the inspector's Observations / Gotchas
panels stayed empty for most sessions because models didn't reach for
``observe`` mid-turn. The ledger is one of our load-bearing memory
substrates (per [[project_coder_observation_ledger]]) — an empty
ledger means the next session re-discovers everything.

This module hooks into the tool-execution path. After every successful
tool call, it runs a small pattern matcher against the (tool_name,
input, result) triple and proposes durable facts:

* ``test_run`` success → "test runner: {runner}"
* ``service_start`` success → "dev server: {command} on port {port}"
* ``shell_exec`` matching install/build verbs → "deps installed via {pkg}"
* mutating writes to config paths → "config: {path}"
* repeated similar tool failures → gotcha entry

Design constraints
------------------

* **Pattern-based, not LLM-based.** Auto-observation should add zero
  inference latency. A second LLM round-trip per tool call would
  multiply token cost; we only spend it for the next-speaker classifier
  (which catches a known-narrow failure mode).
* **Dedupe-by-(category, fact)** — the observation ledger does this
  natively, so re-recording is safe. We aim for compact, stable facts
  ("test runner: pytest") that the deduper can match.
* **Confidence = "confirmed"** when the tool actually succeeded;
  ``"tentative"`` for inferred shapes (e.g., "this looks like a build
  command but we didn't run it"). Future render paths can prioritize
  by confidence.
* **Per-turn cap** so a fan-out of 20 file_reads doesn't fill the
  ledger with auto-noise. The handler enforces the cap.
"""
from __future__ import annotations

import re
import time
from typing import Any

from augmentum.coder.observations import Observation

# Maximum auto-observations to emit per (tool_name, input, result)
# triple. Most patterns produce 0 or 1 observations; the cap is a
# defensive bound. The CALLER also limits how many auto-observations
# accumulate per turn — this is per-call.
_PER_CALL_CAP = 2


# ---------------------------------------------------------------------------
# Test-runner detection
# ---------------------------------------------------------------------------

# Map of cmd-shape regex → human-readable runner label. Match the
# FIRST verb in the command since `cd ... && pytest -x` should still
# resolve to pytest. The patterns are anchored with `\b` to avoid
# partial matches inside longer commands.
_TEST_RUNNER_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bpytest\b"), "pytest"),
    (re.compile(r"\bpython\s+-m\s+pytest\b"), "pytest"),
    (re.compile(r"\bpython\s+-m\s+unittest\b"), "unittest"),
    (re.compile(r"\bnpx\s+jest\b|\bjest\b"), "jest"),
    (re.compile(r"\bnpx\s+vitest\b|\bvitest\b"), "vitest"),
    (re.compile(r"\bnpm\s+(?:run\s+)?test\b"), "npm test"),
    (re.compile(r"\byarn\s+(?:run\s+)?test\b"), "yarn test"),
    (re.compile(r"\bpnpm\s+(?:run\s+)?test\b"), "pnpm test"),
    (re.compile(r"\bcargo\s+test\b"), "cargo test"),
    (re.compile(r"\bgo\s+test\b"), "go test"),
    (re.compile(r"\bdotnet\s+test\b"), "dotnet test"),
    (re.compile(r"\bmix\s+test\b"), "mix test"),
    (re.compile(r"\brake\s+test\b|\bbundle\s+exec\s+rspec\b|\brspec\b"), "rspec"),
)


# ---------------------------------------------------------------------------
# Install / build / dev-server command detection
# ---------------------------------------------------------------------------

# Package-manager + verb → label. Order matters: more specific patterns
# (npm install with a specific flag) before more general ones.
_INSTALL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bnpm\s+(?:ci|install)\b"), "npm"),
    (re.compile(r"\byarn\s+(?:install)?\b(?!\s+(?:add|run|test))"), "yarn"),
    (re.compile(r"\bpnpm\s+install\b"), "pnpm"),
    (re.compile(r"\bpip\s+install\b"), "pip"),
    (re.compile(r"\bpip3\s+install\b"), "pip3"),
    (re.compile(r"\bpoetry\s+install\b"), "poetry"),
    (re.compile(r"\buv\s+(?:pip\s+)?install\b|\buv\s+sync\b"), "uv"),
    (re.compile(r"\bcargo\s+(?:build|check)\b"), "cargo"),
    (re.compile(r"\bgo\s+(?:mod\s+download|mod\s+tidy|build)\b"), "go"),
    (re.compile(r"\bbundle\s+install\b"), "bundler"),
    (re.compile(r"\bmix\s+(?:deps\.get|compile)\b"), "mix"),
)


# Common dev-server verbs that produce a long-lived process. Match on
# the verb only — port/url come from the tool's structured output.
_DEV_SERVER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bnpm\s+(?:run\s+)?(?:dev|start|serve)\b"),
    re.compile(r"\byarn\s+(?:dev|start|serve)\b"),
    re.compile(r"\bpnpm\s+(?:dev|start|serve)\b"),
    re.compile(r"\bvite\b"),
    re.compile(r"\bnext\s+dev\b"),
    re.compile(r"\buvicorn\b"),
    re.compile(r"\bhypercorn\b"),
    re.compile(r"\bgunicorn\b"),
    re.compile(r"\bdjango-admin\s+runserver\b|\bmanage\.py\s+runserver\b"),
    re.compile(r"\bflask\s+run\b"),
    re.compile(r"\brails\s+(?:s|server)\b"),
    re.compile(r"\brustup\s+run\b|\bcargo\s+run\b"),
    re.compile(r"\bgo\s+run\b"),
    re.compile(r"\bhttp\.server\b"),
)


# ---------------------------------------------------------------------------
# Config-file detection
# ---------------------------------------------------------------------------

# Paths the agent edits/writes that often carry environment info.
# Matches FILENAMES (no path component) so we catch config files no
# matter what directory they live in.
_CONFIG_FILE_PATTERN = re.compile(
    r"(?:^|/)("
    r"\.env(?:\.\w+)?"
    r"|package\.json|pnpm-lock\.yaml|yarn\.lock|package-lock\.json"
    r"|pyproject\.toml|requirements(?:-\w+)?\.txt|Pipfile|poetry\.lock|uv\.lock"
    r"|Cargo\.toml|Cargo\.lock|go\.mod|go\.sum|Gemfile|Gemfile\.lock"
    r"|tsconfig\.json|vite\.config\.\w+|next\.config\.\w+"
    r"|docker-compose\.ya?ml|Dockerfile|\.dockerignore"
    r"|Makefile|justfile|Procfile"
    r"|tailwind\.config\.\w+|eslint\.config\.\w+|\.eslintrc\.\w+|prettier\.config\.\w+|\.prettierrc(?:\.\w+)?"
    r")$",
)


def _detect_test_runner(command: str) -> str | None:
    """Return a normalized runner label for ``command`` or None."""
    if not command:
        return None
    for pat, label in _TEST_RUNNER_PATTERNS:
        if pat.search(command):
            return label
    return None


def _detect_install_manager(command: str) -> str | None:
    """Return the package manager used for an install/build command."""
    if not command:
        return None
    for pat, label in _INSTALL_PATTERNS:
        if pat.search(command):
            return label
    return None


def _is_dev_server_command(command: str) -> bool:
    if not command:
        return False
    return any(p.search(command) for p in _DEV_SERVER_PATTERNS)


def _filename_from_path(path: str) -> str:
    """Last segment of a path. Empty for an empty path."""
    if not path:
        return ""
    return path.rstrip("/").rsplit("/", 1)[-1]


def _config_match(path: str) -> str | None:
    """Return the matched filename when ``path`` is a known config file."""
    if not path:
        return None
    m = _CONFIG_FILE_PATTERN.search(path)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------


def extract_auto_observations(
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_result_success: bool,
    tool_result_output: str = "",
    tool_result_metadata: dict[str, Any] | None = None,
    source_tag: str = "auto",
) -> list[Observation]:
    """Propose Observation entries from a single tool execution.

    Inputs are the same fields the handler already has on hand at the
    post-tool hook. Returns an empty list when no patterns matched —
    most reads/grep calls produce nothing, which is fine.

    The returned observations have:

    * ``confidence='confirmed'`` — we ran the tool and it succeeded
    * ``source = source_tag`` — defaults to ``"auto"``; callers can
      include the turn id (``"auto:turn-7"``) so the inspector can
      attribute auto-entries by turn
    * Stable, deduper-friendly ``fact`` text (lower-case, no quotes)
    """
    if not tool_result_success:
        return []

    out: list[Observation] = []
    ts = time.time()
    md = tool_result_metadata or {}

    # ── test_run / shell_exec running tests ────────────────────────
    if tool_name in ("test_run", "shell_exec", "shell_read"):
        command = tool_input.get("command") or tool_input.get("cmd") or ""
        runner = _detect_test_runner(command)
        if runner:
            out.append(Observation(
                ts=ts,
                category="test",
                fact=f"test runner: {runner}",
                source=source_tag,
                confidence="confirmed",
            ))

    # ── shell_exec install / build ────────────────────────────────
    if tool_name in ("shell_exec",):
        command = tool_input.get("command") or ""
        pkg = _detect_install_manager(command)
        if pkg:
            out.append(Observation(
                ts=ts,
                category="build",
                fact=f"deps installed via {pkg}",
                source=source_tag,
                confidence="confirmed",
            ))

    # ── service_start / dev server ─────────────────────────────────
    if tool_name == "service_start":
        command = tool_input.get("command") or ""
        port = md.get("port") or tool_input.get("port") or ""
        url = md.get("url") or ""
        if command:
            parts = [f"dev server: {command.strip()[:80]}"]
            if port:
                parts.append(f"port {port}")
            elif url:
                parts.append(url)
            out.append(Observation(
                ts=ts,
                category="build",
                fact=" · ".join(parts),
                source=source_tag,
                confidence="confirmed",
            ))

    # ── Dev-server pattern inside shell_exec (best-effort) ─────────
    # When the agent backgrounds a dev server via shell_exec instead
    # of service_start (we discourage this in the prompt, but it
    # happens), still capture the command so the user knows.
    elif tool_name == "shell_exec":
        command = tool_input.get("command") or ""
        if _is_dev_server_command(command) and "&" in command:
            out.append(Observation(
                ts=ts,
                category="build",
                fact=f"dev server (backgrounded): {command.strip()[:80]}",
                source=source_tag,
                confidence="tentative",
            ))

    # ── Config-file edits ─────────────────────────────────────────
    if tool_name in ("file_write", "code_edit", "code_edit_batch", "apply_patch"):
        # apply_patch's "path" is implicit; the metadata has the list.
        path = tool_input.get("path") or ""
        if tool_name == "apply_patch":
            # Defer to metadata.changed_files if present; fall back to
            # checking the first +++/--- header in the patch. The
            # extractor's job is best-effort, not exhaustive.
            changed = md.get("changed_files") or []
            paths = changed if isinstance(changed, list) else []
        else:
            paths = [path] if path else []
        for p in paths:
            cf = _config_match(p)
            if cf:
                fname = _filename_from_path(p)
                out.append(Observation(
                    ts=ts,
                    category="env",
                    fact=f"config: {fname}",
                    source=source_tag,
                    confidence="confirmed",
                ))

    # Cap per-call observations defensively. Most patterns produce
    # 0-1; the cap stops a pathological edit-batch from emitting one
    # observation per file.
    return out[:_PER_CALL_CAP]


__all__ = [
    "extract_auto_observations",
]
