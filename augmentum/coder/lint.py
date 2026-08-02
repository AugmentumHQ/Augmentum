"""Post-write syntax / lint check for code-editing tools (Aider pattern).

After a successful ``file_write`` / ``code_edit`` / ``code_multi_edit``,
run a fast syntax-or-lint command against the touched file. Findings are
appended to the tool result's ``output`` so the model sees them on the
SAME iteration — no separate "reflected message" round-trip required.

Why not Aider's full reflection loop? Aider re-routes the lint output as
the next user message and re-prompts. We already pass the ToolResult
through the conversation and the model decodes it on the next assistant
turn — the effect is the same with one fewer state machine.

Design notes
------------
* **Best-effort.** If the linter isn't installed in the workspace
  container (exit 127), we silently skip — never block a write because
  the project's linter isn't set up.
* **Time-bounded.** ``coder_lint_timeout`` caps wall-clock per file.
  Slow linters never make the agent loop feel sticky.
* **Output-bounded.** ``coder_lint_max_chars`` truncates findings
  before they hit the conversation. A 50KB pylint dump from one
  generated file would otherwise blow the context window.
* **Per-extension dispatch.** Pythonic stdlib-only checks are first
  (``py_compile``, ``json.tool``). Project-local linters (ruff, eslint)
  are tried after, with a graceful fall-through when missing.
"""
from __future__ import annotations

from dataclasses import dataclass

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Sentinel returned by ``_run_command`` when the wrapped command died
# with exit 127 (binary not found). We don't get the raw exit code
# back — the timeout banner is the only structured signal — so we
# detect on the stdout marker instead.
_NOT_FOUND_MARKERS = ("command not found", "No such file or directory", ": not found")


@dataclass(frozen=True, slots=True)
class LintCommand:
    """One try at linting / syntax-checking a single file."""

    cmd: list[str]
    name: str  # short label that appears in the appended block


def _commands_for_path(path: str) -> list[LintCommand]:
    """Return ordered lint candidates for ``path`` based on its extension.

    Stdlib-only checks come first because they're guaranteed to work
    without any project setup. Heavier linters (ruff, eslint) come
    second so users with a configured project still get the richer
    output, but absence is silent.
    """
    lower = path.lower()
    if lower.endswith(".py"):
        return [
            # Stdlib syntax check — always available, catches the
            # most damaging class of broken edits (unparseable file).
            LintCommand(["python3", "-m", "py_compile", path], "py_compile"),
            # Style / unused-import / fast static checks.
            LintCommand(["ruff", "check", "--no-fix", "--quiet", path], "ruff"),
        ]
    if lower.endswith((".js", ".mjs", ".cjs", ".jsx")):
        return [
            LintCommand(["node", "--check", path], "node --check"),
            LintCommand(["eslint", "--no-color", "--quiet", path], "eslint"),
        ]
    if lower.endswith((".ts", ".tsx")):
        # Single-file tsc almost never works in real projects without
        # a tsconfig.json — skipping it. eslint catches the common
        # cases when configured. Full type-check belongs in a
        # dedicated LSP integration (deferred PR1.2 follow-up).
        return [LintCommand(["eslint", "--no-color", "--quiet", path], "eslint")]
    if lower.endswith(".json"):
        return [LintCommand(["python3", "-m", "json.tool", path], "json")]
    if lower.endswith((".yaml", ".yml")):
        # python3 -c "import yaml; yaml.safe_load(open(p))" — but PyYAML
        # isn't always installed. Use a tiny inline shim that's a
        # no-op when yaml isn't importable.
        return [LintCommand(
            [
                "python3", "-c",
                f"import sys\ntry:\n import yaml\nexcept ImportError:\n sys.exit(0)\nyaml.safe_load(open({path!r}))",
            ],
            "yaml",
        )]
    return []


def _looks_like_not_found(output: str) -> bool:
    """Heuristic: did the command 127 because the binary is missing?"""
    head = output[:400].lower()
    return any(marker.lower() in head for marker in _NOT_FOUND_MARKERS)


async def run_post_write_lint(
    container_manager,
    workspace_id: str,
    path: str,
    *,
    timeout: float = 8.0,
    max_chars: int = 1500,
) -> str | None:
    """Run the first applicable lint command for ``path``.

    Returns
    -------
    A formatted findings block (with linter name prefix) if the linter
    produced output AND it doesn't look like a missing-binary error.
    ``None`` when the file extension has no candidate linter, every
    candidate skipped (binary missing), or every candidate returned
    clean.
    """
    candidates = _commands_for_path(path)
    if not candidates:
        return None

    for candidate in candidates:
        try:
            output = await container_manager._run_command(
                workspace_id, candidate.cmd, timeout=timeout,
            )
        except Exception as exc:
            log.debug(
                "post_write_lint_failed", path=path,
                cmd=candidate.cmd[:2], error=str(exc),
            )
            continue

        text = (output or "").strip()
        if not text:
            # Clean: linter ran and said nothing. Skip remaining
            # candidates — one clean signal per file is enough.
            return None

        if _looks_like_not_found(text):
            # Try the next candidate; this one isn't installed.
            continue

        if len(text) > max_chars:
            text = text[: max_chars - 100] + (
                f"\n\n[... lint output truncated at {max_chars} chars ...]"
            )

        return f"\n\n[{candidate.name}]\n{text}"

    return None
