"""8 workspace-aware tools for the Coder mode agent.

Each tool wraps ContainerManager operations and follows the Tool base class
interface.  All tools share a common ``_CoderTool`` base that holds the
container manager, workspace ID, and CoderState.

Tool catalogue
--------------
FileReadTool   — read a file, add line numbers, track in state.files_read
FileWriteTool  — write (create or overwrite) a complete file
FileListTool   — list directory contents with sizes
CodeEditTool   — SEARCH/REPLACE with 4-tier matching + read-before-edit guard
CodeGrepTool   — grep -rn through workspace
CodeGlobTool   — find files by name pattern
ShellExecTool  — run any bash command (npm, pytest, git, …)
ShellReadTool  — run read-only commands (git log, cat, ls)
"""
from __future__ import annotations

import posixpath
import re
import shlex
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from augmentum.coder.editing import apply_edit
from augmentum.coder.executors import ContainerExecutor, WorkspaceExecutor
from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.coder.containers import ContainerManager
    from augmentum.coder.state import CoderState

log = get_logger(__name__)

# Maximum output size to prevent blowing up the LLM context window
_MAX_OUTPUT_CHARS = 50_000


def _truncate(text: str) -> str:
    """Truncate output to _MAX_OUTPUT_CHARS with truncation notice at BOTH ends.

    The prepended notice is the critical bit: the agent loop's compaction
    clips each tool_result to the first 160 chars of preview (400 for
    errors), so a trailing-only "truncated" hint gets clipped away — the
    model then edits / reasons thinking it saw the whole file. By
    prepending the notice, the signal survives every compaction pass.
    The trailing notice stays as a secondary confirmation for the
    uncompacted view.

    Hint is intentionally generic — earlier revisions suggested
    ``offset=50000`` for file_read, which was wrong (``offset`` is a
    zero-based LINE index for file_read, not a byte count). That bug
    sent models on a round trip: re-fetch at bogus offset → past-EOF
    error → re-fetch at offset 0 → loop. File_read's own paging hint
    (added by ``FileReadTool.execute``) knows the correct line offset
    and has always been the authoritative source.
    """
    total = len(text)
    if total <= _MAX_OUTPUT_CHARS:
        return text
    header = (
        f"[TRUNCATED — showing first {_MAX_OUTPUT_CHARS} of {total} chars. "
        "For file_read, the tool output above includes its own paging "
        "hint naming the correct next line offset — use that. For "
        "search results (code_grep / find_files), narrow your pattern "
        "or use a smaller `limit`.]\n\n"
    )
    return (
        header
        + text[:_MAX_OUTPUT_CHARS]
        + f"\n\n... (truncated, {total} total chars)"
    )


# Shell error patterns that reliably indicate the *command itself* failed
# to run — as distinct from the program running and reporting a normal
# failure. When any of these appears, the test/shell wrapper should NOT
# treat the output as parseable; it should surface a plain-English error
# up to the agent. Patterns cover bash, zsh, busybox/ash, and a couple of
# language-level "tool-not-installed" signals that tend to indicate the
# user asked for something the container isn't provisioned for.
_SHELL_FAILURE_PATTERNS = (
    # bash / zsh: "bash: line 1: pytest: command not found"
    ("command not found", "command not found"),
    # busybox / ash: "foo: not found"
    (": not found", "command not found"),
    # python module / interpreter issues
    ("ModuleNotFoundError:", "missing Python module"),
    ("ImportError: No module named", "missing Python module"),
    # go / cargo / npm "command not installed"
    ("executable file not found in $PATH", "executable not in PATH"),
    # common wrong-path case — show up when a dev runs `python3 /snake.html`
    # from /workspace and the shell resolves the path wrong
    ("No such file or directory", "file or directory not found"),
)


# Commands whose first token marks them as a download / transfer operation.
# Recognised so shell_exec can (a) grant the long-running timeout tier and
# (b) attempt to parse the output file from the rest of the arg list, so
# the idle-kill handler can verify progress by watching bytes land on disk
# rather than relying solely on stdout chatter. ``wget -q`` / ``curl -s``
# emit zero stdout by design; without disk-side liveness the 60s idle
# timer kills the download long before it completes (observed 2026-04-22
# with a 700MB RetroArch archive download that kept dying every minute).
_DOWNLOAD_FIRST_TOKENS = frozenset({
    "wget", "curl", "aria2", "aria2c", "scp", "rsync",
})


# Flags across wget / curl / aria2 that name the output file. Order matters
# for the regex — longest-first so ``--output-document=FILE`` isn't
# swallowed by the shorter ``--output=FILE`` match. Each captures either
# ``--flag=VALUE`` or ``--flag VALUE`` / ``-X VALUE`` shapes. Value can be
# a bare token OR a single/double-quoted string (paths with spaces are
# uncommon in practice but not an error to support).
_DL_VAL = r"""(?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|[^\s]+)"""
_DOWNLOAD_OUTPUT_RE = re.compile(
    rf"""
    (?:^|\s)
    (?:
        --output-document \s* =? \s* (?P<wget_long>{_DL_VAL})
      | --output          \s* =? \s* (?P<curl_long>{_DL_VAL})
      | --out             \s* =? \s* (?P<aria_long>{_DL_VAL})
      | -O                \s+         (?P<wget_short>{_DL_VAL})
      | -o                \s+         (?P<curl_short>{_DL_VAL})
    )
    """,
    re.VERBOSE,
)


def _parse_download_target(command: str) -> str | None:
    """Extract the output-file path from a download-style command.

    Returns the path when:

    * The first token of ``command`` (after a ``sh -c`` unwrap) is in
      :data:`_DOWNLOAD_FIRST_TOKENS`.
    * The command carries a recognised ``-O`` / ``-o`` / ``--output`` /
      ``--output-document`` / ``--out`` flag with a concrete value.
    * The value isn't ``-`` (stdout) or blank.

    Returns ``None`` otherwise — including for commands like ``curl URL |
    tar xz`` where the output is a pipe. In the piped case the file-size
    heuristic doesn't apply; the run_command idle handler falls back to
    its existing stdout-based behaviour (and the pipe ensures stdout
    traffic anyway, so the idle timer won't misfire).

    Intentionally does NOT attempt to expand shell variables, tilde
    paths, or quoted multi-word names. Everything past the flag is
    assumed to be the filename — a malformed value just means we can't
    track progress, which is fine.
    """
    if not command:
        return None
    stripped = command.strip()
    # Strip a leading ``sh -c '...'`` / ``bash -c "..."`` wrapper so the
    # first-token check sees the actual tool.
    m = re.match(
        r"""\A(?:sh|bash|/bin/sh|/bin/bash)\s+-l?c\s+['"](.+)['"]\s*\Z""",
        stripped,
    )
    if m:
        stripped = m.group(1).strip()

    first = stripped.split(None, 1)[0] if stripped else ""
    # Handle leading env-var assignment shape (``ENV=1 wget ...``) —
    # walk tokens until we find one that isn't of the form FOO=BAR.
    if "=" in first and first.split("=", 1)[0].replace("_", "").isalnum():
        for tok in stripped.split():
            if "=" not in tok or not tok.split("=", 1)[0].replace("_", "").isalnum():
                first = tok
                break
    first_basename = first.rsplit("/", 1)[-1]
    if first_basename not in _DOWNLOAD_FIRST_TOKENS:
        return None

    match = _DOWNLOAD_OUTPUT_RE.search(stripped)
    if match is None:
        return None

    # Extract whichever named group matched.
    path = next(
        (v for v in match.groupdict().values() if v),
        None,
    )
    if not path or path == "-":
        return None
    # Strip surrounding quotes the regex may have kept.
    if len(path) >= 2 and path[0] in ("'", '"') and path[-1] == path[0]:
        path = path[1:-1]
    return path or None


# Hard ceiling on a model-supplied shell timeout. Matches the long tier's
# wall-clock cap so an explicit timeout can extend a quiet command up to the
# same 10 minutes a build gets, but no further — anything longer belongs in
# ``run_in_background``. ``ShellExecTool.timeout`` (the OUTER dispatch wrap)
# sits 10s above this so the inner, actionable timeout always fires first.
_MAX_SHELL_TIMEOUT = 600


def _clamp_timeout(raw: object) -> int | None:
    """Coerce a model-supplied ``timeout`` to a usable wall-clock ceiling.

    Returns an int in ``[1, _MAX_SHELL_TIMEOUT]``, or ``None`` when the value
    is absent or unusable (0, negative, non-numeric) — in which case the
    caller falls back to the automatic keyword tier. Never raises: a bad
    timeout must degrade to the default, not crash the run. ``coerce_params``
    already turns ``"300"`` into ``300`` upstream, but this stays defensive
    for any path that reaches ``execute`` without coercion.
    """
    if raw is None:
        return None
    try:
        secs = int(raw)
    except (TypeError, ValueError):
        return None
    if secs <= 0:
        return None
    return min(secs, _MAX_SHELL_TIMEOUT)


def _expand_brace_glob(glob: str) -> list[str]:
    """Expand a single ``{a,b}`` brace group in a glob into concrete patterns.

    grep's ``--include`` takes a literal shell glob but does NOT expand
    braces (that's a shell feature), so ``*.{ts,tsx}`` would match nothing.
    Split one brace group so the common multi-extension case works; a glob
    without braces passes through unchanged, and a malformed one falls back
    to itself rather than raising.
    """
    if "{" not in glob or "}" not in glob:
        return [glob]
    try:
        pre, rest = glob.split("{", 1)
        body, post = rest.split("}", 1)
    except ValueError:
        return [glob]
    opts = [f"{pre}{opt}{post}" for opt in body.split(",") if opt]
    return opts or [glob]


def _resolve_workspace_path(path: str) -> str:
    """Normalize a workspace path so relative inputs still resolve.

    The tools' schemas all say "absolute path under /workspace" but weak
    models routinely pass ``README.md`` / ``./src/x.py`` / ``src/x.py``.
    Pre-2026-04-20 those reached ``cat <path>`` verbatim and failed —
    whichever container dir they ran in, the file wasn't there. This
    helper makes the tool layer tolerant:

      * Empty / whitespace                 → ``""`` (caller handles error)
      * Starts with ``/``                  → unchanged (trusted absolute)
      * Starts with ``./``                 → strip + prepend /workspace/
      * Bare ``README.md`` / ``src/x.py``  → /workspace/<path>
      * Starts with ``~/``                 → unchanged (user explicitly
        asked for a home path; we don't second-guess that)

    Paired with ``_run_command(... workdir="/workspace")`` in
    ContainerManager, this is belt-and-suspenders: the workdir fix
    handles shell commands, this fix handles tools that build absolute
    paths into command arguments (``printf '...' > <path>`` in
    file_write, where workdir doesn't help).
    """
    if not path:
        return ""
    stripped = path.strip()
    if not stripped:
        return ""
    if stripped.startswith("~/"):
        return stripped
    if stripped.startswith("/"):
        # Collapse `.` / `..` segments so a disguised path like
        # ``/workspace/../etc/passwd`` resolves to what it actually
        # names (``/etc/passwd``) before any confinement check sees
        # it. posixpath (not os.path) because these are container
        # paths regardless of host platform.
        return posixpath.normpath(stripped)
    if stripped.startswith("./"):
        stripped = stripped[2:]
    return posixpath.normpath(f"/workspace/{stripped}")


_WORKSPACE_ROOT = "/workspace"


def _workspace_confinement_error(path: str, tool_name: str) -> str | None:
    """Return an error message when a MUTATING tool targets a path
    outside ``/workspace``, else None.

    Reads are deliberately unconfined (the container is the security
    boundary, and ``shell_exec`` already sees the whole filesystem).
    Writes are different: turn snapshots, the review panel, and
    rewind only track files under ``/workspace`` — a write outside it
    would be invisible to review and impossible to revert. Callers
    must pass a path already through ``_resolve_workspace_path`` so
    ``..`` segments are collapsed before this check.
    """
    if path == _WORKSPACE_ROOT or path.startswith(_WORKSPACE_ROOT + "/"):
        return None
    return (
        f"{tool_name} can only modify files under /workspace — got "
        f"'{path}'. Files outside /workspace aren't tracked by turn "
        "snapshots, so changes there can't be reviewed or reverted. "
        "If you genuinely need to write outside the project (e.g. a "
        "/tmp scratch file), use shell_exec instead."
    )


def _shell_command_failure(output: str) -> str | None:
    """Classify shell-level command failures.

    Returns a short, human-readable tag when the output clearly indicates
    the command itself couldn't execute (missing binary, missing module,
    bad path). Returns None when the output is parseable test/program
    output that happens to contain the word "error" etc.
    """
    if not output:
        return None
    for needle, label in _SHELL_FAILURE_PATTERNS:
        if needle in output:
            return label
    return None


# ---------------------------------------------------------------------------
# Pre-write safety validation — syntax + critical-file guard
# ---------------------------------------------------------------------------

_CRITICAL_FILE_PATTERNS = (
    # Secrets / credentials
    "*.env", "*.pem", "*.key", "*.secret", "*.pfx", "*.p12",
    # Config files that gate network/auth — silent corruption breaks infra
    "credentials.*", "secrets.*", "*credentials*", "*secret*",
    # Private keys often have no extension
    "id_rsa", "id_ed25519", "id_ecdsa", "*.privatekey",
    # Token / cert files
    "*.token", "*.crt", "*.cer", "*.ca-bundle",
)

# JS syntax probes run ``node --check`` on a TEMP copy of the content —
# never on ``{path}``: workspace paths don't exist in the augmentum
# container, so a path-based probe validates the wrong thing (see
# _pre_write_validate's Python branch for the full story).
_NODE_CHECK_EXTS = (".js", ".mjs", ".cjs")


def _is_critical_file(filename: str) -> bool:
    """Check if filename matches any critical-file pattern."""
    import fnmatch
    base = filename.lower()
    for pat in _CRITICAL_FILE_PATTERNS:
        if fnmatch.fnmatch(base, pat):
            return True
    return False


def _pre_write_validate(path: str, content: str) -> str | None:
    """Run fast pre-write safety checks before committing to disk.

    Returns an error string if the write should be BLOCKED, or None
    to allow. Target: <100ms total so this is never the bottleneck.

    Checks (fail-closed — any positive hit blocks the write):
    1. Critical-file guard — secrets/credentials require explicit
       user confirmation via the task brief.
    2. Syntax check — for supported languages, verify the written
       content parses before it lands on disk.
    """
    import subprocess

    # 1. Critical-file guard
    filename = path.split("/")[-1] if "/" in path else path
    if _is_critical_file(filename):
        return (
            f"Pre-write safety block: '{path}' matches a critical-file "
            f"pattern (secrets, credentials, or private keys). These "
            f"files gate auth/infra and a wrong write is hard to recover. "
            f"If this write is intentional and authorized by your dispatch "
            f"brief, use shell_exec to bypass the guard: "
            f"`cat > {path} << 'AUGMENTUM_EOF'`. Otherwise, re-examine "
            f"whether you should be touching this file at all."
        )

    # 2. Syntax check — validate the NEW CONTENT, never the path.
    #
    # The original implementation ran ``python3 -c "compile(open(
    # '{path}').read(), ...)"`` — it opened the WORKSPACE path from
    # inside the augmentum container (where /workspace doesn't exist)
    # and ignored the piped content entirely, so from 2026-07-12 to
    # 2026-07-18 every .py code_edit/file_write was blocked by a
    # FileNotFoundError traceback dressed up as a "syntax error", with
    # the actual reason truncated out of the model-facing message
    # (found live: Ornith looping on same_validation_error_repeat).
    if filename.endswith(".py"):
        try:
            compile(content, path, "exec")
        except SyntaxError as exc:
            # Real, actionable detail — the model needs the line to fix it.
            return (
                f"Pre-write syntax check failed for '{path}': "
                f"line {exc.lineno}: {exc.msg}\n"
                f"  {(exc.text or '').strip()[:200]}\n"
                f"Fix the syntax error and retry the write."
            )
        except Exception:
            return None  # checker infra trouble — fail open, never block
        return None

    if filename.endswith(_NODE_CHECK_EXTS):
        import contextlib
        import os
        import tempfile
        tmp_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=filename[filename.rfind("."):],
                delete=False, encoding="utf-8",
            ) as tmp:
                tmp.write(content)
                tmp_name = tmp.name
            result = subprocess.run(
                ["node", "--check", tmp_name],
                capture_output=True, text=True,
                timeout=5.0,
            )
            if result.returncode != 0:
                stderr = (result.stderr or result.stdout or "").strip()
                # The temp path means nothing to the model — show the
                # intended path instead.
                stderr = stderr.replace(tmp_name, path)[:500]
                return (
                    f"Pre-write syntax check failed for '{path}'. The "
                    f"content would not parse. Fix the syntax error "
                    f"before writing.\nnode output: {stderr}"
                )
        except FileNotFoundError:
            pass  # node not available — skip
        except subprocess.TimeoutExpired:
            pass  # Check took too long — skip, not worth blocking the write
        except Exception:
            pass  # Any other infra failure — fail open
        finally:
            if tmp_name:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_name)
        return None

    return None  # No syntax check for this language

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _CoderTool(Tool):
    """Base for all workspace-aware coder tools.

    Parameters (keyword-only)
    -------------------------
    container_manager:
        The ContainerManager instance that talks to Docker.
    workspace_id:
        The workspace container to operate on.
    state:
        The CoderState for this agent session (read-before-edit guard, etc.).
    """

    def __init__(
        self,
        *,
        container_manager: ContainerManager | None = None,
        workspace_id: str,
        state: CoderState,
        executor: WorkspaceExecutor | None = None,
        profile_store=None,
        service_store=None,
        user_id: str = "",
        strict_edit_guard: bool = True,
    ) -> None:
        self._cm = container_manager
        self._workspace_id = workspace_id
        # Portable execution surface (WorkspaceExecutor). Two ways in:
        #   • inject a pre-built ``executor`` (Phase 2: RemoteEditorExecutor for
        #     the ACP "loop in the editor" path), or
        #   • pass a ``container_manager`` and we wrap it as a ContainerExecutor
        #     (today's Docker workspace — the default, behaviour-unchanged).
        # The tools only ever touch ``self._executor`` for portable file/shell
        # ops, so they don't care which backend it is. ``self._cm`` may be None
        # in remote mode — container-only ops (list_ports, enable_published_
        # ports, interactive terminal) aren't reachable there. See executors.py.
        # No raise when both are absent: a ContainerExecutor wrapping a None
        # ContainerManager is lazy — it only fails if a portable op is actually
        # invoked. Tools that never touch the workspace (e.g. AskUserTool) are
        # routinely constructed with container_manager=None, and that must keep
        # working exactly as it did before executor injection existed.
        if executor is not None:
            self._executor: WorkspaceExecutor = executor
        else:
            self._executor = ContainerExecutor(container_manager, workspace_id)
        self._state = state
        self._profile_store = profile_store
        self._service_store = service_store
        self._user_id = user_id or ""
        self._strict_edit_guard = bool(strict_edit_guard)
        # Optional live-output sink set by the agent loop before calling
        # execute() on shell tools. Lets the dispatcher stream stdout to
        # the UI as the docker exec produces it, instead of buffering the
        # whole transcript until process exit. Tools that don't run a
        # shell ignore this entirely. See modes/coder/handler.py for the
        # streaming dispatch path that sets this.
        self._on_chunk: Callable[[bytes], Awaitable[None]] | None = None

    @property
    def cacheable(self) -> bool:
        """Container operations are never cacheable."""
        return False

    async def _current_mtime(self, path: str) -> float | None:
        """Return the file's current mtime (epoch seconds), or None on
        any failure.

        Helper for the mtime-based staleness check in ``code_edit`` /
        ``code_multi_edit``. Failures (file gone, stat missing,
        permission denied) degrade to None, which the state's
        ``can_edit`` treats as "no staleness info — fresh enough".
        """
        try:
            # GNU `stat -c %Y` works on Linux containers. The `|| stat -f %m`
            # fallback is BSD/macOS form — a no-op for the current Docker
            # path, but keeps this probe correct if a native-Mac workspace
            # ever lands (or a Colima image swaps in BSD coreutils).
            out = await self._executor.run_command([
                    "bash",
                    "-c",
                    f"stat -c %Y '{path}' 2>/dev/null || stat -f %m '{path}' 2>/dev/null",
                ],
                timeout=3.0,
            )
            out = out.strip()
            if out:
                return float(out.splitlines()[0])
        except (ValueError, Exception):   # noqa: BLE001 — best-effort
            log.warning("tool.stat_failed", path=path, exc_info=True)
        return None

    async def _refresh_read_mtime_after_write(self, path: str) -> None:
        """Update ``files_read[path]`` to the post-write mtime.

        Without this, the next edit on the same file inside the same
        turn fails ``can_edit`` — the model's own write looks like an
        external mutation because the stored read-mtime is older than
        the on-disk mtime. The model then re-reads a file it just wrote
        and the error message ("modified externally") is misleading.

        Stat-failure path is intentional: leave the prior baseline
        untouched rather than erasing it to ``inf`` (which would silence
        a genuinely external edit arriving later).
        """
        new_mtime = await self._current_mtime(path)
        if new_mtime is not None:
            self._state.record_file_read(path, mtime=new_mtime)

    async def _maybe_snapshot_before_write(self, path: str) -> None:
        """Capture pre-turn disk state of ``path`` for the review flow.

        No-op when no active turn snapshot is attached (e.g. tests,
        or ``file_upload`` routes that aren't agent-initiated). Failures
        inside the snapshot layer never propagate — losing the restore
        capability for one path degrades the review UX for that file,
        it doesn't break the write.

        Idempotent per path — safe to call from a code_edit that reads
        before writing AND from a file_write targeting the same path
        earlier in the turn. Only the first call captures; the rest
        no-op.
        """
        snap = getattr(self._state, "active_turn_snapshot", None)
        if snap is None:
            return
        try:
            await snap.snapshot_before_write(path)
        except Exception:
            # Surfaces a real degradation: losing the pre-write snapshot
            # means the review flow can't restore this path if the user
            # rejects the turn. Don't block the write — but the failure
            # is worth a warning so a recurring snapshot bug is findable.
            log.warning("snapshot_pre_write_failed", path=path, exc_info=True)

    async def _maybe_run_post_write_lint(self, path: str) -> str:
        """Run the post-write lint hook for ``path`` if enabled.

        Returns the formatted lint findings (already prefixed with the
        linter name) ready for appending to ``ToolResult.output``, or
        an empty string when the hook is disabled, the file extension
        has no candidate linter, the linter isn't installed, or the
        file lints clean.

        Best-effort: any exception inside the hook is swallowed and
        treated as "no findings" — never block a successful write
        because the lint hook itself broke.
        """
        from augmentum.config import settings as _settings

        if not getattr(_settings, "coder_auto_lint", True):
            return ""
        try:
            from augmentum.coder.lint import run_post_write_lint

            findings = await run_post_write_lint(
                self._cm, self._workspace_id, path,
                timeout=float(getattr(_settings, "coder_lint_timeout", 8.0)),
                max_chars=int(getattr(_settings, "coder_lint_max_chars", 1500)),
            )
        except Exception:
            # The hook itself broke (not "the file has lint errors").
            # Without this warning a misconfigured linter / bad import
            # silently degrades to "lint never runs" with no signal.
            log.warning("post_write_lint_hook_failed", path=path, exc_info=True)
            return ""
        return findings or ""

    async def _maybe_run_post_write_verify(
        self, path: str, new_content: str,
    ) -> tuple[str, bool]:
        """In-process syntax verification gate (Phase 3.2 of the coder
        foundation). Runs ``VerificationGate.default()`` against an
        ``EditRecord`` for the just-written file content.

        Returns
        -------
        ``(message, failed)`` where:

        * ``message`` is empty on success or when disabled, else the
          gate's ``model_facing_summary()`` — short, location-anchored,
          ready to append to ``ToolResult.output`` so the model sees
          the parse error on the same iteration as the write.
        * ``failed`` is True iff the gate reported any blocking failure.
          Callers use this to flag ``ToolResult.metadata['verification_failed']``
          for downstream chunk emission and to call
          ``state.record_tool_failure(...)`` so Phase 2.2's persistent
          ledger sees the pattern across turns.

        Best-effort: any exception inside the gate is swallowed and
        treated as "no verification" — a buggy gate must never block
        an otherwise successful write (parallels lint's contract).
        """
        from augmentum.config import settings as _settings

        if not getattr(_settings, "coder_auto_verify", True):
            return "", False
        try:
            from augmentum.coder.verify import EditRecord, VerificationGate

            edit = EditRecord(
                path=path, tool=self.name, new_content=new_content or "",
            )
            report = await VerificationGate.default().verify_writes([edit])
        except Exception:
            # The verify gate itself broke (not "the file failed parse").
            # Silent degradation here would mean the model never sees
            # syntax errors on its writes — exactly the regression class
            # the gate was added to prevent.
            log.warning("post_write_verify_hook_failed", path=path, exc_info=True)
            return "", False

        if report.passed:
            return "", False

        summary = report.model_facing_summary()
        # Cross-turn persistence: feed the verification failure into the
        # same soft-failure ledger lint/tool errors land in (Phase 2.2).
        # Recurring parse errors on the same path then surface as a
        # pattern in the next turn's sticky reminder, not just a one-off
        # nudge in the current iteration's tool output.
        try:
            self._state.record_tool_failure(
                tool_name="verify",
                target=path,
                error=summary or "verification failed",
            )
        except Exception:
            # Ledger write failed — the cross-turn pattern surface won't
            # show this verification failure in next turn's sticky
            # reminder. Worth flagging so the ledger schema regression
            # gets noticed instead of silently muting Phase 2.2 output.
            log.warning("verify_record_failure_failed", path=path, exc_info=True)
        return summary, True

    async def _check_read_freshness(self, path: str):
        """Return a ToolResult error when the file-read state can't
        justify an edit, else None.

        Two guards combined:
          1. Read-before-edit (original): path must be in
             ``CoderState.files_read`` at all.
          2. Mtime-staleness (new 2026-04-20): when stat succeeds,
             require our stored read mtime ≥ current mtime, so
             external edits since the last read force a re-read.

        The failure modes collapse to the same error message naming
        both causes — the model picks the right recovery (re-read).
        When stat fails (container error, file gone), the check
        degrades to guard #1 only — no worse than pre-fix behaviour.
        """
        if not self._strict_edit_guard:
            return None
        current = await self._current_mtime(path)
        if not self._state.can_edit(path, current_mtime=current):
            return ToolResult(
                success=False,
                error=(
                    f"Cannot edit '{path}' — either you haven't read "
                    "it yet, or it has been modified externally since "
                    "your last read (mtime is newer than what we "
                    "captured). Re-read the file with file_read "
                    "before editing; the current version on disk may "
                    "differ from what you have in context."
                ),
            )
        return None


# ---------------------------------------------------------------------------
# FileReadTool
# ---------------------------------------------------------------------------

class FileReadTool(_CoderTool):
    """Read a file from the workspace and return it with line numbers."""

    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return (
            "Read one or several files from the workspace. Returns content "
            "prefixed with line numbers ('   1 | first line'). You MUST "
            "read a file before editing it (batch reads count). "
            "PREFER the batch form when you already know 2+ files you "
            "need: pass `paths` (array of strings) and one call replaces "
            "N single reads. Files that don't fit the output budget are "
            "listed under 'omitted' — call again with just those paths. "
            "For single large files, the output is paged — pass `offset` "
            "(zero-based line index) and optionally `limit` (default 2000 "
            "lines) to fetch a specific window; the response includes a "
            "`next_offset` hint when more content remains."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "No such file":
                "The path doesn't exist. Use `dir_tree` or `file_list` to "
                "find the correct path, or `find_files` to search by pattern.",
            "is a directory":
                "You passed a directory path. Use `file_list` or `dir_tree` "
                "to enumerate contents, then call `file_read` with a specific "
                "file path.",
            "Permission denied":
                "The container can't read this path. If it's outside "
                "/workspace, that's expected — stay within /workspace.",
        }

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file in the workspace (e.g. /workspace/src/main.py)",
                },
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Batch mode: read several files in ONE call (max "
                        "20). Prefer this over sequential single reads "
                        "when you already know the files you need. "
                        "Mutually exclusive with 'path'/'offset'/'limit' "
                        "paging — batch reads each file from the start."
                    ),
                },
                "offset": {
                    "type":    "integer",
                    "minimum": 0,
                    "default": 0,
                    "description": (
                        "Zero-based line index to start reading from. Use "
                        "with 'limit' to page through big files. Default 0 "
                        "(start of file). Line numbers in the output "
                        "reflect the true file position."
                    ),
                },
                "limit": {
                    "type":    "integer",
                    "minimum": 1,
                    "default": 2000,
                    "description": (
                        "Max number of lines to return in this call. Default "
                        "2000. Use a smaller limit for quick peeks; bump up "
                        "for large reads. Hard cap from output buffer is "
                        "still ~50k chars total."
                    ),
                },
                "force_raw": {
                    "type":    "boolean",
                    "default": False,
                    "description": (
                        "Force raw text content even for file types with a "
                        "registered analyzer (GGUF, GLB/VRM, safetensors, "
                        "audio, SQLite, archives). Default False — those "
                        "types return a structured summary that is far "
                        "cheaper than the raw bytes. Set True only when you "
                        "actually need the bytes (rare)."
                    ),
                },
            },
            # Neither field is JSON-Schema-required: exactly one of
            # 'path' (single) or 'paths' (batch) must be present, which
            # a static required-list can't express. Execute validates
            # and returns a self-describing error when both are missing.
            "required": [],
        }

    # Batch-read guards. _BATCH_MAX_FILES bounds container round-trips
    # per call; _BATCH_BODY_BUDGET mirrors the single-read body budget —
    # files that don't fit are OMITTED WHOLE (never truncated mid-file)
    # and named in the output so the model fetches them next call.
    _BATCH_MAX_FILES = 20
    _BATCH_BODY_BUDGET = 46_000

    async def execute(
        self,
        *,
        path: str = "",
        paths: list[str] | None = None,
        offset: int = 0,
        limit: int = 2000,
        force_raw: bool = False,
        **_kwargs,
    ) -> ToolResult:
        # Batch dispatch. A single-element paths list degrades to the
        # single-file path so paging metadata stays available for it.
        if paths:
            if isinstance(paths, str):
                paths = [paths]
            batch = [str(p).strip() for p in paths if str(p or "").strip()]
            if len(batch) > 1:
                return await self._execute_batch(batch, force_raw=force_raw)
            if len(batch) == 1 and not path:
                path = batch[0]
        # Normalize relative paths (README.md, ./src/x.py, src/x.py)
        # into absolute /workspace/... so weak models that don't follow
        # the "absolute path" schema hint still get a working read
        # instead of a cryptic ENOENT.
        path = _resolve_workspace_path(path)
        if not path:
            # Self-describing error so the model can recover without the
            # agent loop ending up in a degenerate retry cycle (observed
            # 2026-04-20 on weaker models: a terse "path is required" led
            # the model to re-invoke with the same empty params).
            return ToolResult(
                success=False,
                error=(
                    "file_read called without a 'path' argument. "
                    "Required: path (string, absolute path to a file under "
                    "/workspace) — or 'paths' (array of strings) to read "
                    "several files in one call. Example calls: "
                    '{"path": "/workspace/README.md"} or '
                    '{"paths": ["/workspace/a.py", "/workspace/b.py"]}. '
                    "If you don't know the paths yet, use dir_tree or "
                    "find_files first to list what's available."
                ),
                validation_error=True,
            )

        # Coerce + guard the pagination args defensively — weaker models
        # sometimes pass strings or negatives. Keep the call working.
        try:
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            offset = 0
        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            limit = 2000

        # Analyzer dispatch: if the file has a registered analyzer
        # (GGUF, GLB/VRM, safetensors, audio, SQLite, archives, …),
        # return its structured summary instead of raw content. This
        # is the carrot side of "don't poison the model context with
        # binary previews". The model can opt back into raw bytes by
        # passing force_raw=True for the rare case where it needs the
        # actual bytes (e.g., hex inspection of a few bytes).
        if not force_raw:
            analyzer_result = await self._maybe_run_analyzer(path)
            if analyzer_result is not None:
                return analyzer_result

        try:
            raw = await self._executor.read_file(path)
        except Exception as exc:
            log.warning("file_read_failed", path=path, error=str(exc))
            return ToolResult(success=False, error=str(exc))

        # Capture mtime NOW, right after the successful read, so the
        # staleness check in code_edit can tell "has this file changed
        # since I last saw it". `stat -c %Y` gives epoch seconds (GNU);
        # the `|| stat -f %m` fallback covers BSD/macOS coreutils so
        # this stays correct if a native-Mac workspace ever lands. A
        # container error here is non-fatal — we fall back to None
        # (files_read records inf, staleness check degrades to "always
        # fresh" for that entry, preserving pre-mtime behaviour).
        read_mtime: float | None = None
        try:
            stat_out = await self._executor.run_command([
                    "bash",
                    "-c",
                    f"stat -c %Y '{path}' 2>/dev/null || stat -f %m '{path}' 2>/dev/null",
                ],
                timeout=3.0,
            )
            stat_out = stat_out.strip()
            if stat_out:
                read_mtime = float(stat_out.splitlines()[0])
        except (ValueError, Exception):   # noqa: BLE001 — best-effort
            # Stat failure silently degrades the mtime-staleness guard
            # to "always fresh" — the model could edit truly stale reads
            # with no warning. Surface so a container/permission issue
            # doesn't quietly disable the guard fleet-wide.
            log.warning("file_read.stat_failed", path=path, exc_info=True)

        all_lines = raw.splitlines()
        total_lines = len(all_lines)

        # Compute the window. A past-EOF offset gets a clean empty-window
        # response rather than silent success with no content — the model
        # would otherwise assume it read the file successfully but saw
        # "nothing interesting here."
        if offset >= total_lines and total_lines > 0:
            return ToolResult(
                success=False,
                error=(
                    f"offset={offset} is past end of file (file has "
                    f"{total_lines} lines). Use offset=0..{total_lines - 1}."
                ),
                validation_error=True,
                metadata={"path": path, "total_lines": total_lines},
            )

        window = all_lines[offset : offset + limit]

        # Pre-clip the window BY LINES so the byte cap never fires mid-
        # line. Without this, a file with very long lines (minified JS,
        # generated SQL, etc.) could pass the ``limit`` check, then get
        # chopped mid-line by ``_truncate``'s 50k-char cap — and the
        # paging hint we append below would name a line offset past
        # what the model actually saw, sending the next read to the
        # wrong place. We target 46k chars of body to leave headroom
        # for the paging-hint footer + line-number prefixes.
        _BODY_BUDGET = 46_000
        body_chars = 0
        clipped_window: list[str] = []
        for i, line in enumerate(window):
            # +8 for "NNNN | " prefix and newline
            cost = len(line) + 8
            if clipped_window and body_chars + cost > _BODY_BUDGET:
                break
            clipped_window.append(line)
            body_chars += cost
        window = clipped_window

        end_line = offset + len(window)
        truncated = end_line < total_lines

        numbered = "\n".join(
            f"{offset + i + 1:4d} | {line}"
            for i, line in enumerate(window)
        )

        # Paging hint — the AUTHORITATIVE source for "where to read
        # next" on this tool. The byte-cap hint in ``_truncate`` is
        # generic and intentionally doesn't name a line offset; this
        # one knows the exact offset because we just clipped by lines.
        if truncated:
            numbered += (
                f"\n\n[Showing lines {offset + 1}-{end_line} of "
                f"{total_lines}. Call file_read with offset={end_line} "
                f"for the next chunk.]"
            )

        # Track in state (read-before-edit guard). Only the first read of
        # a file (offset=0) counts toward the guard — a partial re-read
        # at offset>0 implies the model already saw the file once.
        # Pass the captured mtime so the edit-time staleness check can
        # reject edits whose source file has changed externally since
        # the last read (concurrent user edit, git pull, other agent).
        if offset == 0:
            self._state.record_file_read(path, mtime=read_mtime)

        return ToolResult(
            success=True,
            output=_truncate(numbered),
            metadata={
                "path":        path,
                "total_lines": total_lines,
                "lines_shown": len(window),
                "offset":      offset,
                "next_offset": end_line if truncated else None,
                # Back-compat: pre-pagination code read metadata["lines"]
                # as the total. Keep it meaning the same thing.
                "lines":       total_lines,
            },
        )

    async def _batch_mtimes(self, resolved: list[str]) -> dict[str, float]:
        """One ``stat`` round-trip for every path in the batch.

        Missing files / stat failures simply have no entry — the
        staleness guard degrades per-file exactly like the single-read
        path does (records inf, "always fresh").
        """
        import shlex as _shlex

        if not resolved:
            return {}
        quoted = " ".join(_shlex.quote(p) for p in resolved)
        out: dict[str, float] = {}
        try:
            stat_out = await self._executor.run_command([
                    "bash",
                    "-c",
                    f"stat -c '%Y %n' {quoted} 2>/dev/null"
                    f" || stat -f '%m %N' {quoted} 2>/dev/null",
                ],
                timeout=5.0,
            )
            for line in (stat_out or "").splitlines():
                epoch, _, name = line.strip().partition(" ")
                try:
                    out[name] = float(epoch)
                except (TypeError, ValueError):
                    continue
        except Exception:  # noqa: BLE001 — best-effort, mirrors single-read
            log.warning("file_read.batch_stat_failed", exc_info=True)
        return out

    async def _execute_batch(
        self, batch: list[str], *, force_raw: bool = False,
    ) -> ToolResult:
        """Read several files in one tool call.

        Contract (never-truncate discipline): every file is either
        included COMPLETE (up to the same per-file paging window the
        single read uses, with an explicit paging note), reported as an
        inline per-file error, or OMITTED WHOLE and named — content is
        never silently clipped to fit the budget.
        """
        # De-dupe preserving order, cap the batch size loudly.
        seen: set[str] = set()
        batch = [p for p in batch if not (p in seen or seen.add(p))]
        over_cap = batch[self._BATCH_MAX_FILES:]
        batch = batch[: self._BATCH_MAX_FILES]

        resolved: list[tuple[str, str]] = []   # (requested, resolved)
        pre_errors: list[str] = []
        for raw_path in batch:
            rp = _resolve_workspace_path(raw_path)
            if rp:
                resolved.append((raw_path, rp))
            else:
                pre_errors.append(raw_path)

        mtimes = await self._batch_mtimes([rp for _, rp in resolved])

        budget = self._BATCH_BODY_BUDGET
        sections: list[str] = []
        files_meta: list[dict] = []
        omitted: list[str] = []
        any_content = False

        for _requested, rp in resolved:
            if budget <= 0:
                omitted.append(rp)
                continue

            # Analyzer dispatch — same carrot as the single read: a
            # structured summary instead of binary noise.
            if not force_raw:
                analyzer_result = await self._maybe_run_analyzer(rp)
                if analyzer_result is not None:
                    text = analyzer_result.output or analyzer_result.error
                    cost = len(text) + len(rp) + 16
                    if any_content and cost > budget:
                        omitted.append(rp)
                        continue
                    sections.append(f"=== {rp} (analyzer summary) ===\n{text}")
                    files_meta.append({
                        "path": rp, "ok": analyzer_result.success,
                        "analyzer": True,
                    })
                    budget -= cost
                    any_content = True
                    if analyzer_result.success:
                        self._state.record_file_read(rp, mtime=mtimes.get(rp))
                    continue

            try:
                raw = await self._executor.read_file(rp)
            except Exception as exc:
                # Per-file failure stays inline — one missing file must
                # not fail the other N-1 reads.
                sections.append(f"=== {rp} — ERROR: {str(exc)[:200]} ===")
                files_meta.append({"path": rp, "ok": False, "error": str(exc)[:200]})
                continue

            all_lines = raw.splitlines()
            total_lines = len(all_lines)
            window = all_lines[:2000]

            numbered_lines: list[str] = []
            body_chars = 0
            for i, line in enumerate(window):
                cost = len(line) + 8
                if numbered_lines and body_chars + cost > budget:
                    break
                numbered_lines.append(f"{i + 1:4d} | {line}")
                body_chars += cost

            shown = len(numbered_lines)
            fits_whole = shown >= total_lines
            if not fits_whole and any_content:
                # Doesn't fit complete in what's left — defer the WHOLE
                # file rather than truncate it.
                omitted.append(rp)
                continue

            header = f"=== {rp} ({total_lines} lines) ==="
            body = "\n".join(numbered_lines)
            if not fits_whole:
                # First file in the batch and larger than the whole
                # budget: window it with the same explicit paging note
                # the single read emits.
                body += (
                    f"\n\n[Showing lines 1-{shown} of {total_lines}. "
                    f"Call file_read with path={rp!r} offset={shown} "
                    f"for the next chunk.]"
                )
            sections.append(f"{header}\n{body}")
            files_meta.append({
                "path": rp, "ok": True,
                "total_lines": total_lines, "lines_shown": shown,
            })
            budget -= body_chars + len(header) + 2
            any_content = True
            self._state.record_file_read(rp, mtime=mtimes.get(rp))

        footer: list[str] = []
        for bad in pre_errors:
            sections.append(f"=== {bad!r} — ERROR: unresolvable path ===")
            files_meta.append({"path": bad, "ok": False, "error": "unresolvable path"})
        if omitted:
            footer.append(
                f"[{len(omitted)} file(s) omitted to keep this response "
                f"within the output budget (content is never truncated "
                f"mid-file): {', '.join(omitted)}. Call file_read again "
                f"with paths={omitted!r} to fetch them.]"
            )
        if over_cap:
            footer.append(
                f"[{len(over_cap)} path(s) beyond the {self._BATCH_MAX_FILES}"
                f"-file batch cap were not read: {', '.join(over_cap)}. "
                f"Send them in a follow-up batch.]"
            )

        output = "\n\n".join(sections + footer) if (sections or footer) else (
            "No readable files in batch."
        )
        ok_count = sum(1 for f in files_meta if f.get("ok"))
        return ToolResult(
            # Success iff at least one file was actually delivered —
            # an all-miss batch must look like a failure to the loop.
            success=ok_count > 0,
            output=_truncate(output),
            error="" if ok_count else "no files in the batch could be read",
            metadata={
                "batch":      True,
                "files":      files_meta,
                "read_ok":    ok_count,
                "omitted":    omitted,
                "over_cap":   over_cap,
            },
        )

    async def _maybe_run_analyzer(self, path: str) -> ToolResult | None:
        """If ``path`` matches a registered analyzer, fetch bytes and
        run it. Returns the structured-summary ToolResult, or None when
        no analyzer matches (caller falls through to the raw read).
        """
        import os
        import tempfile

        from augmentum.coder.analyzers import analyze_file, is_analyzable

        if not is_analyzable(path):
            return None
        try:
            # read_file_bytes routes to the binary-safe get-archive reader.
            # Previously this called self._cm.file_read_bytes, which does NOT
            # exist on ContainerManager — every call raised AttributeError,
            # got swallowed below, and the binary analyzer path (glTF/GGUF/
            # audio) silently died and fell through to a raw text read.
            data = await self._executor.read_file_bytes(path)
        except Exception as exc:
            log.warning(
                "file_read_analyzer_fetch_failed",
                path=path, error=str(exc)[:200],
            )
            return None

        # Most builtin analyzers want a path on the local filesystem
        # (they call into libs like pygltflib / gguf / mutagen that
        # open the file themselves). Drop the bytes into a temp file
        # preserving the extension so the lib's format dispatch works.
        suffix = os.path.splitext(path)[1] or ""
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False,
        ) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            report = await analyze_file(tmp_path, data)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if report is None:
            return None

        return ToolResult(
            success=True,
            output=report.summary,
            metadata={
                "path":            path,
                "analyzer_format": report.format,
                "analyzed":        True,
                "details":         report.details,
                "raw_size_bytes":  report.raw_size_bytes,
                "force_raw_hint":  (
                    "Pass force_raw=true to file_read to get the actual "
                    "bytes (rare — only when you need to inspect raw "
                    "content directly)."
                ),
            },
        )


# ---------------------------------------------------------------------------
# FileWriteTool
# ---------------------------------------------------------------------------

class FileWriteTool(_CoderTool):
    """Write (create or overwrite) a complete file in the workspace."""

    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        # IMPORTANT: this description is the primary lever steering the
        # model AWAY from large-payload file_write calls (the failure mode
        # where the JSON args exceed the per-response output budget and
        # `path` gets truncated off the tail). The structural fix isn't a
        # bigger budget — it's biasing the model to use code_edit_batch
        # for any non-trivial body. Every CC/Codex-class agent does this:
        # full-file Write is the exception; targeted Edit/MultiEdit/patch
        # is the default. Keep the steering language strong here.
        return (
            "Write a COMPLETE file body to the workspace (creates or "
            "overwrites). USE SPARINGLY — prefer `code_edit_batch` or "
            "`apply_patch` in almost every case. Only use file_write when "
            "ALL of the following are true: (1) creating a brand-new "
            "file, AND (2) the content is short (under ~1500 chars / "
            "~400 tokens). For larger new files, write a SHORT skeleton "
            "(imports + empty class/function stubs with `pass` bodies) "
            "via file_write FIRST, then fill each section with "
            "`code_edit_batch` blocks anchored on the stubs — each block "
            "is a small diff so the args reliably fit in one tool call. "
            "For modifying existing files, NEVER use file_write — use "
            "`code_edit_batch` (or `code_edit` for a single change). "
            "Large file_write calls regularly truncate mid-args and lose "
            "the `path` field, costing you a wasted iteration."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def error_hints(self) -> dict[str, str]:
        # NOTE: keys are substring-matched against the raw error in
        # ``base.py::enrich_error``. Be specific — bare keys like
        # ``"content"`` will spuriously match the error format text
        # itself (e.g. the "Required: path + content (string)" hint
        # in the missing-path error contains ``content``), firing the
        # wrong hint and leaving the model with a contradictory
        # message it can't recover from. Anchor each key to a phrase
        # that ONLY appears in the specific failure it targets.
        return {
            "without a 'path' argument":
                "Args arrived without `path`. Most common cause: the "
                "`content` field was so large that the model's output "
                "budget ran out before `path` was emitted — the JSON "
                "tail (including path) gets silently dropped. If your "
                "content is long, DO NOT retry with the same call: "
                "switch to `code_edit_batch` for an existing file, or "
                "for a new file write a SHORT skeleton first and fill "
                "the body with `code_edit_batch` blocks. If the file "
                "really is small, retry with both fields top-level. "
                "Example: "
                '{"path": "/workspace/hello.txt", "content": "hi"}. '
                "`path` must be absolute (e.g. /workspace/src/foo.py).",
            "No such file or directory":
                "file_write creates missing parent directories on its own, "
                "so this is NOT a 'mkdir -p first' situation — running that "
                "and retrying will fail the same way. Check the path itself: "
                "it must be absolute and under /workspace (e.g. "
                "/workspace/src/app.py). A path outside the workspace mount "
                "has no directory to create.",
            "Is a directory":
                "The path points at an existing directory, not a file. "
                "Choose a different filename.",
            "Permission denied":
                "Can't write here. /workspace and its subdirectories are "
                "writable; outside that, write will fail.",
            "missing `content`":
                "Supply the full file body in the `content` field — "
                "file_write replaces the whole file. For partial edits, "
                "use code_edit instead.",
        }

    @property
    def input_schema(self) -> dict:
        # KEY ORDER IS LOAD-BEARING. ``path`` MUST appear before
        # ``content`` in both ``properties`` and ``required``. Most
        # models trained on OpenAI/Anthropic-style tool-calling honor
        # schema declaration order when emitting JSON args — they emit
        # fields in the same order the schema declared them. That
        # matters when output is being streamed under a per-response
        # token cap: if the model runs out of budget mid-args, the TAIL
        # gets truncated. With path-first, truncation loses some of
        # `content` (the model returns a short body but a valid path,
        # which we can surface as "write succeeded with empty body,
        # retry to fill" — recoverable). With content-first, truncation
        # loses `path` entirely, which is the broken-call loop we kept
        # hitting. Python dict insertion order → JSON key order →
        # model's emission order. Don't reorder.
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to write (e.g. /workspace/src/utils.py)",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content to write",
                },
            },
            "required": ["path", "content"],
            # additionalProperties:false → cloud APIs (OpenAI Structured
            # Outputs, Anthropic tool use) and LM Studio strict mode
            # refuse to generate extra fields. Model can't accidentally
            # invent a "mode": "append" field that we'd ignore.
            "additionalProperties": False,
        }

    async def execute(self, *, path: str = "", content: str = "", **_kwargs) -> ToolResult:
        path = _resolve_workspace_path(path)
        if not path:
            return ToolResult(
                success=False,
                error=(
                    "file_write called without a 'path' argument. "
                    "Required: path (string) + content (string). Example: "
                    '{"path": "/workspace/hello.txt", "content": "hi"}.'
                ),
                validation_error=True,
            )

        confinement = _workspace_confinement_error(path, "file_write")
        if confinement:
            return ToolResult(
                success=False, error=confinement, validation_error=True,
            )

        # Opt-in oversized-content rejection. Default cap is 0
        # (uncapped), matching Claude Code's `Write` and Codex CLI's
        # `apply_patch` — both production agents skip pre-emptive
        # write caps entirely. The actual failure mode (mid-arguments
        # cutoff via finish_reason="length") is caught by the D1
        # truncation-detection layer in modes/coder/handler.py, which
        # surfaces a structured "switch to code_edit_batch" hint with
        # zero false positives. Pre-emptive rejection here is a blunt
        # backstop for weak local models with tiny output budgets
        # where the user wants a refusal before the model even tries.
        # Raise ``coder_file_write_max_tokens`` above 0 to enable.
        from augmentum.config import settings as _settings
        from augmentum.utils.tokenizer import count_tokens

        max_tokens = int(_settings.coder_file_write_max_tokens or 0)
        content_tokens = count_tokens(content) if max_tokens > 0 else 0
        if max_tokens > 0 and content_tokens > max_tokens:
            existing = await self._current_mtime(path) is not None
            if existing:
                primary = (
                    f"This file already exists. Don't rewrite the whole "
                    "thing in one call — use `code_edit_batch` with "
                    "targeted SEARCH/REPLACE blocks for the sections "
                    "that actually need to change. Each block is a small "
                    "diff, not a full-file rewrite, so the args fit in "
                    "the output budget. If you genuinely need to replace "
                    f"every line, write a SHORT (<{max_tokens // 3} "
                    "tokens) skeleton first, then add the body with "
                    "`code_edit_batch` blocks anchored on the skeleton."
                )
            else:
                primary = (
                    f"Write a SHORT (<{max_tokens // 3} tokens) skeleton "
                    "first — module docstring, imports, function/class "
                    "stubs with `pass` bodies. Then fill each stub via "
                    "`code_edit` or `code_edit_batch`, anchored on the "
                    "stub's signature line. Each edit is a small diff, "
                    "so the args reliably fit in one tool call."
                )
            return ToolResult(
                success=False,
                validation_error=True,
                error=(
                    f"file_write content is too large "
                    f"({content_tokens} tokens > {max_tokens} cap). A "
                    "single tool call can't reliably ship that much "
                    "output — the response budget runs out mid-arguments "
                    "and the tail is silently dropped, which is why "
                    "your last attempt may have arrived with missing "
                    "fields.\n\n"
                    f"Recommended: {primary}\n\n"
                    "Last resort only (rarely the right answer): split "
                    "the rewrite into multiple file_write calls each "
                    f"strictly under {max_tokens} tokens. This usually "
                    "doesn't work for a single coherent file — pieces "
                    "individually still exceed the cap."
                ),
            )

        # Read-before-edit guard for EXISTING files. The model in the
        # 2026-05-27 cascade went straight to file_write after code_edit
        # bounced — overwriting a file based on its hallucinated
        # contents because no guard checked. CodeEditTool and
        # CodeEditBatchTool already enforce this; FileWriteTool did not.
        # New-file creation is exempt — there's nothing to read first
        # when the path doesn't exist yet. ``_current_mtime`` returns
        # ``None`` on stat failure / file-not-found which is exactly
        # the existence signal we want.
        existing_mtime = await self._current_mtime(path)
        if existing_mtime is not None:
            stale = await self._check_read_freshness(path)
            if stale is not None:
                return stale

        # Review-flow snapshot: capture pre-turn state of this path
        # before the write hits disk. Idempotent per path, no-op when
        # no active turn is attached. See turn_snapshot.py for the
        # restore-on-reject semantics this enables. WITHOUT this call, an
        # OVERWRITE of an existing file is invisible in turn review AND
        # unrestorable on Reject — the exact data loss the snapshot layer
        # exists to prevent. code_edit / code_edit_batch / apply_patch all
        # snapshot here; file_write was the one write path that skipped it
        # (the helper's own docstring already lists file_write as a caller).
        await self._maybe_snapshot_before_write(path)

        # Pre-write safety validation: block on crit-files / syntax errors
        block_reason = _pre_write_validate(path, content)
        if block_reason:
            return ToolResult(
                success=False, error=block_reason, validation_error=True,
            )

        try:
            await self._executor.write_file(path, content)
        except Exception as exc:
            log.warning("file_write_failed", path=path, error=str(exc))
            return ToolResult(success=False, error=str(exc))

        await self._refresh_read_mtime_after_write(path)

        output = f"Wrote {len(content)} bytes to {path}"
        # Verify FIRST (blocking errors before warnings) so the model sees
        # the most actionable signal first when reading the tool result.
        verify_msg, verify_failed = await self._maybe_run_post_write_verify(
            path, content,
        )
        if verify_msg:
            output += "\n\n" + verify_msg
        output += await self._maybe_run_post_write_lint(path)
        metadata: dict = {"path": path, "bytes": len(content)}
        if verify_failed:
            metadata["verification_failed"] = True
        return ToolResult(success=True, output=output, metadata=metadata)


# ---------------------------------------------------------------------------
# FileListTool
# ---------------------------------------------------------------------------

class FileListTool(_CoderTool):
    """List directory contents in the workspace with sizes."""

    @property
    def name(self) -> str:
        return "file_list"

    @property
    def description(self) -> str:
        return (
            "List files and directories in a workspace directory. "
            "Returns name, type (file/dir), and size in bytes."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (default: /workspace)",
                    "default": "/workspace",
                },
            },
            "required": [],
        }

    async def execute(self, *, path: str = "/workspace", **_kwargs) -> ToolResult:
        effective_path = _resolve_workspace_path(path) if (path and path.strip()) else "/workspace"

        try:
            entries = await self._executor.list_files(effective_path)
        except Exception as exc:
            log.warning("file_list_failed", path=effective_path, error=str(exc))
            return ToolResult(success=False, error=str(exc))

        if not entries:
            return ToolResult(
                success=True,
                output=f"(empty directory: {effective_path})",
                metadata={"path": effective_path, "count": 0},
            )

        lines: list[str] = []
        for entry in entries:
            kind = "dir " if entry.is_dir else "file"
            lines.append(f"{kind}  {entry.size:>10d}  {entry.name}")

        output = f"Directory: {effective_path}\n\n" + "\n".join(lines)
        return ToolResult(
            success=True,
            output=_truncate(output),
            metadata={"path": effective_path, "count": len(entries)},
        )


# ---------------------------------------------------------------------------
# CodeEditTool
# ---------------------------------------------------------------------------

class CodeEditTool(_CoderTool):
    """Apply a SEARCH/REPLACE edit to a file using 4-tier matching.

    In strict modes the file must have been read before editing. Native
    mode can disable that guard while still requiring an exact current
    SEARCH block.
    """

    @property
    def name(self) -> str:
        return "code_edit"

    @property
    def description(self) -> str:
        # Longer description so native-tier models get the full
        # SEARCH/REPLACE contract via the tool schema — avoids the
        # separate EDIT_FORMAT_INSTRUCTIONS block we used to append to
        # every system prompt (2026-04-20: removed as redundant with
        # this description).
        if self._strict_edit_guard:
            guard_note = (
                "The file MUST have been read with file_read first; the "
                "read-before-edit guard rejects blind edits."
            )
        else:
            guard_note = (
                "A prior file_read is recommended for precise SEARCH text "
                "but is not required in native mode. The `search` field "
                "still must match text that currently exists in the file."
            )
        return (
            "Edit a file using SEARCH/REPLACE. The `search` field must match "
            "text that currently exists in the file; `replace` is the new "
            f"text. {guard_note}\n\n"
            "Matching tiers (tried in order): exact → whitespace-normalized "
            "→ indentation-preserving → unicode-fold → fuzzy. If exact fails, "
            "later tiers pick up small differences in whitespace / indent / "
            "typography automatically.\n\n"
            "Tips for reliable matches:\n"
            "• Include 1-2 context lines in `search` to guarantee uniqueness "
            "(a short snippet like `return x` probably appears many places).\n"
            "• For multi-line edits, include the full block verbatim — do "
            "NOT abbreviate with `...`.\n"
            "• If search fails: use code_grep to find the exact current text, "
            "then re-emit. If still failing, re-read the file — content may "
            "have changed since you last saw it.\n"
            "• For multiple edits on the SAME file, prefer code_edit_batch "
            "— one atomic write instead of N separate edits."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "empty 'search' block":
                "For NEW files use file_write (not code_edit). code_edit is "
                "for modifying existing file content — it needs a non-empty "
                "search block to anchor the change.",
            "no match found":
                "The search block doesn't exist in the file. Run code_grep "
                "with a short fragment of the target text to find where it "
                "actually lives, then re-emit code_edit with the exact "
                "current text as `search`.",
            "read the file first":
                "Call file_read(path=<this file>) before editing. The "
                "read-before-edit guard blocks edits on files you haven't "
                "looked at this turn.",
            "file has been modified":
                "The file changed on disk since you last read it. Re-read "
                "with file_read, update your `search` block to match the "
                "current content, then retry.",
            "No such file":
                "File doesn't exist yet. Use file_write to create it; "
                "code_edit only modifies existing files.",
        }

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path of the file to edit",
                },
                "search": {
                    "type": "string",
                    "description": "The exact block of text to find in the file",
                },
                "replace": {
                    "type": "string",
                    "description": "The text to replace the search block with",
                },
            },
            "required": ["path", "search", "replace"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        path: str = "",
        search: str = "",
        replace: str = "",
        **_kwargs,
    ) -> ToolResult:
        # Input validation
        path = _resolve_workspace_path(path)
        if not path:
            return ToolResult(
                success=False,
                error=(
                    "file_edit called without a 'path' argument. Required: "
                    "path (string), search (string - exact text to replace), "
                    "replace (string - new text). You must also have "
                    "file_read the file first in this conversation."
                ),
                validation_error=True,
            )
        if not search:
            return ToolResult(
                success=False,
                error=(
                    "file_edit called with an empty 'search' block. The "
                    "search string must match existing text in the file "
                    "exactly, character-for-character."
                ),
                validation_error=True,
            )

        confinement = _workspace_confinement_error(path, "code_edit")
        if confinement:
            return ToolResult(
                success=False, error=confinement, validation_error=True,
            )

        # Read-before-edit guard with mtime-aware staleness check.
        # Rejects both "never read" and "read but file changed since".
        stale = await self._check_read_freshness(path)
        if stale is not None:
            return stale

        # Fetch current content
        try:
            content = await self._executor.read_file(path)
        except Exception as exc:
            log.warning("code_edit_read_failed", path=path, error=str(exc))
            return ToolResult(success=False, error=f"Failed to read file for editing: {exc}")

        # Idempotence check — this edit may already be applied. If the
        # ``search`` block is absent AND the ``replace`` block is present
        # AND the two aren't identical, the file is already in the
        # desired end state. Return success with a no_op marker instead
        # of failing on "no match found" — that failure message caused
        # the retry-loop thrash observed in the 2026-04-22 Pong
        # transcript (model re-sent the same edit 6× after the first
        # write succeeded). The length guard (>=20 chars) avoids false
        # positives on trivially-short replacements like `x` → `y`
        # which would frequently match by coincidence.
        if (
            search != replace
            and len(replace) >= 20
            and search not in content
            and replace in content
        ):
            return ToolResult(
                success=True,
                output=(
                    f"No-op: this edit was already applied to '{path}'. "
                    f"The search block is absent and the replacement text "
                    f"is already present. No rewrite needed."
                ),
                metadata={
                    "path": path,
                    "tier": "already_applied",
                    "no_op": True,
                },
            )

        # Apply 4-tier edit
        new_content, tier = apply_edit(content, search, replace)

        if new_content is None:
            return ToolResult(
                success=False,
                error=(
                    f"No match found in '{path}'. The search block was not located "
                    "using any of the 4 matching tiers (exact, whitespace, indentation, fuzzy). "
                    "Re-read the file and verify the search block matches the current content."
                ),
                metadata={"tier": "none"},
            )

        # Review-flow snapshot (idempotent per path). See FileWriteTool
        # for rationale; code_edit triggers it here because the existing
        # content was read via original_content above — we snapshot
        # against current disk explicitly rather than piggybacking on
        # that read, since the snapshot layer needs its own view of
        # "what was on disk before this turn started" (which differs
        # from "what was on disk before THIS edit" if earlier edits in
        # the same turn already wrote to the file).
        await self._maybe_snapshot_before_write(path)

        # Pre-write safety validation: block on crit-files / syntax errors
        block_reason = _pre_write_validate(path, new_content)
        if block_reason:
            return ToolResult(
                success=False, error=block_reason, validation_error=True,
            )

        # Write updated content back
        try:
            await self._executor.write_file(path, new_content)
        except Exception as exc:
            log.warning("code_edit_write_failed", path=path, error=str(exc))
            return ToolResult(success=False, error=f"Edit matched (tier={tier}) but write failed: {exc}")

        await self._refresh_read_mtime_after_write(path)

        output = f"Edit applied to '{path}' (matched via {tier} tier)."
        verify_msg, verify_failed = await self._maybe_run_post_write_verify(
            path, new_content,
        )
        if verify_msg:
            output += "\n\n" + verify_msg
        output += await self._maybe_run_post_write_lint(path)
        metadata: dict = {"path": path, "tier": tier}
        if verify_failed:
            metadata["verification_failed"] = True
        return ToolResult(success=True, output=output, metadata=metadata)


# ---------------------------------------------------------------------------
# CodeMultiEditTool
# ---------------------------------------------------------------------------

class CodeMultiEditTool(_CoderTool):  # LLM name: code_edit_batch
    """Apply multiple SEARCH/REPLACE edits to one file **atomically**.

    Design choices vs. Codex's ``apply_patch`` and OpenCode's ``multiedit``:

      * **True atomicity.** Both Codex and OpenCode iterate and write per
        hunk, so a failed hunk 3 of 5 leaves the file half-edited. We run
        a PLAN phase that applies every edit to an in-memory copy; if any
        edit can't locate its search block, we return an error and the
        file on disk is untouched. Only on a fully-successful plan do we
        issue one write. (Discovered via cross-repo reading: Codex's
        comment claims atomicity but the code delivers partial writes.)

      * **Aider-style error envelope** on failure — per-edit block with a
        "did you mean?" ``SequenceMatcher`` hint + a "REPLACE already
        present" check + a summary of how many would-have-succeeded,
        mirroring ``editblock_coder.py``'s learned-the-hard-way format.

      * **Reuses our 5-tier matcher** (exact → whitespace → indentation →
        unicode → fuzzy) so every edit gets the same recovery that a
        single ``code_edit`` would. OpenCode's 9-replacer chain is richer
        but our 5 cover 95% of real failures.
    """

    @property
    def name(self) -> str:
        return "code_edit_batch"

    @property
    def description(self) -> str:
        if self._strict_edit_guard:
            guard_note = "File must be read with file_read first."
        else:
            guard_note = (
                "A prior file_read is recommended for precise SEARCH text "
                "but is not required in native mode."
            )
        return (
            "Apply multiple SEARCH/REPLACE edits to ONE file, atomically. "
            "Either all edits apply or the file is left untouched. Use for "
            "refactors that touch the same file in several places (rename a "
            "function used 5×, update imports + body, etc.) — much cheaper "
            "than 5 separate code_edit calls. Edits are applied in order; "
            "each 'search' is matched against the file as updated by earlier "
            "edits in the batch.\n\n"
            "MATCHING RULES (read these — failed matches are the #1 cause "
            "of wasted iterations):\n"
            "• Each `search` must be UNIQUE in the file. If it appears "
            "more than once, only the FIRST occurrence is replaced — and "
            "you can't tell which one. Add context lines above/below "
            "until the snippet is unambiguous.\n"
            "• Keep `search` SHORT — ideally 1-5 lines. Long blocks drift "
            "on indentation or trailing whitespace and bounce off the "
            "matcher. Anchor on the most stable lines (function "
            "signatures, decorators, distinctive comments).\n"
            "• Match the file EXACTLY — same indentation, same trailing "
            "spaces, same blank lines. If you're guessing from memory, "
            "read the file first.\n"
            "• To DELETE lines, set `replace` to empty string. To INSERT "
            "lines, set `search` to one stable existing line and "
            "`replace` to that line plus your new content.\n"
            f"{guard_note}"
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def error_hints(self) -> dict[str, str]:
        # Substring-anchored on phrases that appear ONLY in the specific
        # failure they target. file_write.py's lessons apply — bare
        # tokens like "search" or "edit" would spuriously match the
        # error format strings themselves and fire the wrong hint.
        return {
            "without a 'path' argument":
                "Args arrived without `path`. Most common cause: the "
                "`edits` array was so large that the model's output "
                "budget ran out before `path` was emitted. Split your "
                "edits across two calls — first half in one batch, "
                "second half in a follow-up. `path` must be absolute "
                "(e.g. /workspace/src/foo.py).",
            "without an 'edits' array":
                "Required shape: "
                "`{\"path\":\"/workspace/foo.py\", \"edits\":[{\"search\":\"...\","
                "\"replace\":\"...\"}]}`. For a SINGLE edit prefer "
                "`code_edit` — code_edit_batch is for ≥2 changes to the "
                "same file. Common cause of empty edits: model emitted "
                "`{path,edits:[]}` because it intended one edit, "
                "thought better of it, but forgot to populate the array.",
            "No edits were applied":
                "The whole batch was rejected because the file is "
                "ATOMIC — one failed match invalidates every edit. "
                "Don't resend only the failed edits; the applied ones "
                "haven't been written either. Re-read the file with "
                "`file_read`, fix the failing `search` blocks (tighter "
                "or wider context to make each one unique), and resend "
                "the COMPLETE edits array.",
            "REPLACE text is already present":
                "Your replacement string already exists in the file — "
                "this edit is a no-op or it would create a duplicate. "
                "Drop this edit from the batch and resend the others.",
            "stale_read":
                "The file changed on disk since you read it. Re-run "
                "`file_read` to pick up the new mtime + content before "
                "retrying the batch.",
            "Failed to read file":
                "Couldn't open the file for editing — does it exist? "
                "Use `file_list` or `dir_tree` to confirm the path. "
                "For NEW files use `file_write` (with a SHORT skeleton "
                "if the body is large), not code_edit_batch.",
        }

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path of the file to edit",
                },
                "edits": {
                    "type": "array",
                    "description": (
                        "Ordered edits. Each: {search, replace}. Applied "
                        "sequentially to in-memory content, written in ONE "
                        "atomic pass at the end if all match."
                    ),
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "search":  {"type": "string"},
                            "replace": {"type": "string"},
                        },
                        "required": ["search", "replace"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["path", "edits"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        path: str = "",
        edits: list | None = None,
        **_kwargs,
    ) -> ToolResult:
        path = _resolve_workspace_path(path)
        if not path:
            return ToolResult(
                success=False,
                validation_error=True,
                error=(
                    "code_edit_batch called without a 'path' argument. "
                    "Required: path (string), edits (array of "
                    "{search, replace})."
                ),
            )
        if not isinstance(edits, list) or not edits:
            return ToolResult(
                success=False,
                validation_error=True,
                error=(
                    "code_edit_batch called without an 'edits' array. "
                    "Required: edits: [{search, replace}, ...] with at least "
                    "one item. For a single edit, use code_edit instead."
                ),
            )

        confinement = _workspace_confinement_error(path, "code_edit_batch")
        if confinement:
            return ToolResult(
                success=False, error=confinement, validation_error=True,
            )

        # Mtime-aware read-before-edit guard (same as code_edit).
        stale = await self._check_read_freshness(path)
        if stale is not None:
            return stale

        try:
            content = await self._executor.read_file(path)
        except Exception as exc:
            log.warning("code_edit_batch_read_failed", path=path, error=str(exc))
            return ToolResult(
                success=False,
                error=f"Failed to read file for editing: {exc}",
            )

        original_content = content
        applied: list[dict] = []
        failures: list[dict] = []

        for i, raw_edit in enumerate(edits):
            if not isinstance(raw_edit, dict):
                failures.append({
                    "index":  i,
                    "search": "",
                    "reason": f"edits[{i}] is not an object",
                })
                continue
            search = raw_edit.get("search") or ""
            replace = raw_edit.get("replace", "")
            if not search:
                failures.append({
                    "index":  i,
                    "search": "",
                    "reason": f"edits[{i}].search is empty",
                })
                continue

            # Pre-check uniqueness on the EXACT tier — only meaningful
            # when the search will match verbatim, because the higher
            # tiers (whitespace/indentation/unicode/fuzzy) normalize
            # before matching and `content.count(search)` would be 0
            # even on a successful normalised match. ``apply_edit`` will
            # consume only the FIRST occurrence (``.replace(..., 1)``),
            # so a non-unique search silently changes the wrong instance.
            # We surface this as a non-fatal warning on the way back so
            # the model can audit + tighten its anchors before the next
            # edit lands on an unintended target. Worth doing because
            # this is the silent corruption failure mode of every
            # SEARCH/REPLACE tool — fail loud is better than fail
            # invisible.
            occurrences = content.count(search) if search in content else 0

            new_content, tier = apply_edit(content, search, replace)

            if new_content is None:
                # Aider-style hint — "did you mean?" via SequenceMatcher,
                # plus the idempotence check ("your replace is already in
                # the file — this edit is a no-op and you should drop it").
                already_present = replace and replace in content
                did_you_mean = _similar_blocks(content, search, k=3)
                failures.append({
                    "index":         i,
                    "search":        search,
                    "reason":        "no_match",
                    "already_present": already_present,
                    "did_you_mean":  did_you_mean,
                })
                continue

            content = new_content
            applied.append({
                "index": i,
                "tier": tier,
                "occurrences": occurrences,
            })

        # Atomicity gate: one failure → write nothing, report every
        # failed edit so the model can fix them all in one retry.
        if failures:
            return ToolResult(
                success=False,
                validation_error=True,
                error=_format_multi_edit_errors(
                    path=path,
                    failures=failures,
                    applied_count=len(applied),
                    total=len(edits),
                ),
                metadata={
                    "applied": len(applied),
                    "failed":  len(failures),
                    "total":   len(edits),
                },
            )

        # All edits resolved in the plan phase — single atomic write.
        if content == original_content:
            return ToolResult(
                success=True,
                output=(
                    f"No-op: {len(edits)} edits matched but produced no "
                    f"changes (replace == search in every edit)."
                ),
                metadata={"applied": len(applied), "no_op": True},
            )

        # Review-flow snapshot — see FileWriteTool / CodeEditTool.
        await self._maybe_snapshot_before_write(path)

        try:
            await self._executor.write_file(path, content)
        except Exception as exc:
            log.warning("code_edit_batch_write_failed", path=path, error=str(exc))
            return ToolResult(
                success=False,
                error=(
                    f"All {len(applied)} edits matched in memory but the "
                    f"atomic write failed: {exc}. The file on disk is "
                    f"unchanged."
                ),
            )

        await self._refresh_read_mtime_after_write(path)

        tier_counts: dict[str, int] = {}
        for a in applied:
            tier_counts[a["tier"]] = tier_counts.get(a["tier"], 0) + 1
        tier_summary = ", ".join(
            f"{t}: {n}" for t, n in sorted(tier_counts.items())
        )
        output = (
            f"Applied {len(applied)} edits to '{path}' atomically "
            f"({tier_summary})."
        )

        # Loud-warn on non-unique search blocks — see the uniqueness
        # comment in the apply loop above. The model needs to know
        # WHICH edits matched ambiguously so it can audit whether the
        # first occurrence was actually the right target. If most
        # edits land cleanly and one snuck through against a non-unique
        # anchor, the file silently breaks in a way no verifier or
        # linter can catch — different call site than intended, but
        # syntactically fine.
        ambiguous = [
            a for a in applied
            if a.get("tier") == "exact"
            and int(a.get("occurrences") or 0) > 1
        ]
        if ambiguous:
            lines = ["", "⚠ NON-UNIQUE SEARCH WARNING:"]
            for a in ambiguous:
                lines.append(
                    f"  edit[{a['index']}] — `search` matched "
                    f"{a['occurrences']}× in the file; only the FIRST "
                    f"occurrence was replaced."
                )
            lines.append(
                "Verify the right call site was changed (read the file "
                "and check). For future edits to this file, anchor on "
                "MORE context (surrounding decorators, signatures, or a "
                "distinctive comment) so each search is unique."
            )
            output += "\n" + "\n".join(lines)

        verify_msg, verify_failed = await self._maybe_run_post_write_verify(
            path, content,
        )
        if verify_msg:
            output += "\n\n" + verify_msg
        output += await self._maybe_run_post_write_lint(path)
        metadata: dict = {
            "path":         path,
            "applied":      len(applied),
            "tier_counts":  tier_counts,
        }
        if ambiguous:
            metadata["ambiguous_matches"] = [
                {"index": a["index"], "occurrences": a["occurrences"]}
                for a in ambiguous
            ]
        if verify_failed:
            metadata["verification_failed"] = True
        return ToolResult(success=True, output=output, metadata=metadata)


# ---------------------------------------------------------------------------
# ApplyPatchTool
# ---------------------------------------------------------------------------

class ApplyPatchTool(_CoderTool):
    """Apply a unified diff patch across one or more files atomically."""

    @property
    def name(self) -> str:
        return "apply_patch"

    @property
    def description(self) -> str:
        return (
            "Apply a unified diff patch under /workspace. Use for coordinated "
            "multi-file edits, renames, deletes, and generated diffs. The patch "
            "is checked with `git apply --check` before applying, so either the "
            "whole patch lands or nothing changes. Prefer code_edit/code_edit_batch "
            "for small surgical edits; use apply_patch when a change spans files."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": (
                        "Unified diff text, e.g. output shaped like "
                        "`diff --git a/file b/file` with ---/+++ headers "
                        "and @@ hunks."
                    ),
                },
            },
            "required": ["patch"],
            "additionalProperties": False,
        }

    @staticmethod
    def _workspace_paths_from_patch(patch: str) -> list[str]:
        """Best-effort path extraction for pre-write snapshots."""
        paths: list[str] = []
        seen: set[str] = set()
        for line in patch.splitlines():
            if not (line.startswith("+++ ") or line.startswith("--- ")):
                continue
            raw = line[4:].strip().split("\t", 1)[0].split(" ", 1)[0]
            if raw == "/dev/null":
                continue
            if raw.startswith(("a/", "b/")):
                raw = raw[2:]
            if not raw or raw.startswith("/") or "\x00" in raw:
                continue
            parts = [p for p in raw.replace("\\", "/").split("/") if p]
            if any(p == ".." for p in parts):
                continue
            path = "/workspace/" + "/".join(parts)
            if path not in seen:
                seen.add(path)
                paths.append(path)
        return paths

    async def execute(self, *, patch: str = "", **_kwargs) -> ToolResult:
        patch = patch or ""
        if not patch.strip():
            return ToolResult(
                success=False,
                error=(
                    "apply_patch called without a 'patch' string. Provide a "
                    "complete unified diff with file headers and hunks."
                ),
                validation_error=True,
            )

        patch_paths = self._workspace_paths_from_patch(patch)

        # Read-before-edit guard, multi-file edition. Same defense as
        # FileWriteTool / CodeEditTool but iterated across every path
        # the patch touches. Skip paths that don't exist yet (new-file
        # creation in the diff) — there's nothing to read first.
        for path in patch_paths:
            existing_mtime = await self._current_mtime(path)
            if existing_mtime is None:
                continue
            stale = await self._check_read_freshness(path)
            if stale is not None:
                return stale

        for path in patch_paths:
            await self._maybe_snapshot_before_write(path)

        patch_name = f"agent_patch_{uuid.uuid4().hex}.diff"
        patch_dir = "/workspace/.augmentum/patches"
        patch_path = f"{patch_dir}/{patch_name}"
        try:
            await self._executor.upload_files(patch_dir,
                [(patch_name, patch.encode("utf-8"))],
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Failed to stage patch file: {exc}",
            )

        quoted = shlex.quote(patch_path)

        # In-band success markers. ``WorkspaceExecutor.run_command`` omits
        # the ``strict`` flag (for editor/ACP-executor neutrality), so a
        # non-zero ``git apply`` exit is NEVER inspected — the command just
        # returns whatever it printed. Without a marker, a FAILED apply
        # returns its error text, that text becomes "changed files", and
        # the tool reports "Patch applied" with success=True. The echo'd
        # marker only survives when the preceding command succeeded (``&&``),
        # so its absence is an executor-agnostic proof of failure.
        _CHECK_OK = "__AUG_PATCH_CHECK_OK__"
        _APPLY_OK = "__AUG_PATCH_APPLY_OK__"

        check_cmd = (
            f"cd /workspace && git apply --check --whitespace=nowarn {quoted} "
            f"2>&1 && echo {_CHECK_OK}"
        )
        try:
            check_output = await self._executor.run_command(["bash", "-lc", check_cmd],
                timeout=30.0,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Patch check failed: {exc}",
                metadata={"patch_path": patch_path},
            )
        if _CHECK_OK not in check_output:
            # --check exited non-zero: the patch will NOT apply cleanly.
            # Gate here instead of applying anyway and reporting success
            # on the resulting error text (the A2/A3 false-success class).
            detail = check_output.replace(_CHECK_OK, "").strip() or (
                "git apply --check reported the patch does not apply"
            )
            return ToolResult(
                success=False,
                error=(
                    "Patch does not apply cleanly (git apply --check failed):\n"
                    f"{_truncate(detail)}\n\n"
                    "The target file likely changed since you built the diff — "
                    "re-read it and regenerate the patch against the current "
                    "content."
                ),
                validation_error=True,
                failure_kind="invalid_input",
                metadata={"patch_path": patch_path},
            )
        check_output = check_output.replace(_CHECK_OK, "").strip()

        apply_cmd = (
            f"cd /workspace && git apply --whitespace=nowarn {quoted} 2>&1 "
            f"&& echo {_APPLY_OK} "
            "&& git status --short -- . | sed -n '1,100p'"
        )
        try:
            apply_output = await self._executor.run_command(["bash", "-lc", apply_cmd],
                timeout=30.0,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Patch apply failed: {exc}",
                metadata={"patch_path": patch_path},
            )
        if _APPLY_OK not in apply_output:
            # Passed --check but apply still failed (race, partial index
            # state). Never report success on the error text.
            detail = apply_output.strip() or "git apply exited non-zero"
            return ToolResult(
                success=False,
                error=f"Patch failed to apply (git apply exited non-zero):\n{_truncate(detail)}",
                failure_kind="internal_error",
                metadata={"patch_path": patch_path},
            )

        # Everything AFTER the marker is the `git status --short` output.
        status_text = apply_output.split(_APPLY_OK, 1)[1]
        changed_files = [line for line in status_text.splitlines() if line.strip()]
        if changed_files:
            output = "Patch applied. Changed files:\n" + "\n".join(
                f"- {path}" for path in changed_files[:100]
            )
        else:
            output = "Patch applied. No changed files were reported by git status."
        if check_output.strip():
            output += f"\n\nPatch check output:\n{_truncate(check_output.strip())}"

        # Post-write verify + lint on every patched file that still
        # exists after apply. file_write / code_edit / code_edit_batch
        # all run these hooks per-write; apply_patch used to silently
        # skip them, which let multi-file patches land syntactically
        # broken Python/JS/JSON and the model only discovered the break
        # on the next test/run — wasting a whole turn. Mirror the same
        # discipline here so the parse-error feedback is immediate.
        #
        # We use the patch_paths we already extracted (covers ---/+++
        # headers); skip paths the patch deleted (mtime is None after
        # apply). Each file is read back via the container; if the read
        # fails we degrade silently rather than masking the apply
        # success.
        any_verify_failed = False
        for changed_path in patch_paths:
            try:
                exists_mtime = await self._current_mtime(changed_path)
            except Exception:
                exists_mtime = None
            if exists_mtime is None:
                # File was deleted by the patch (or read failed) —
                # nothing to verify.
                continue
            try:
                new_body = await self._executor.read_file(changed_path)
            except Exception:
                # Best-effort: don't fail the tool because we couldn't
                # re-read for verify. Log so persistent read failures
                # don't silently mute parse-error feedback.
                log.warning(
                    "apply_patch_post_verify_read_failed",
                    path=changed_path, exc_info=True,
                )
                continue
            verify_msg, verify_failed = await self._maybe_run_post_write_verify(
                changed_path, new_body,
            )
            if verify_msg:
                output += f"\n\n[verify: {changed_path}]\n{verify_msg}"
            if verify_failed:
                any_verify_failed = True
            lint_msg = await self._maybe_run_post_write_lint(changed_path)
            if lint_msg.strip():
                output += f"\n[lint: {changed_path}]{lint_msg}"
            await self._refresh_read_mtime_after_write(changed_path)

        # Best-effort cleanup of the staged diff. These accumulated under
        # .augmentum/patches/ indefinitely — one file per apply_patch call,
        # forever. Only reached on the success path; the failure returns
        # above leave the diff in place and surface its path so the model
        # (or user) can inspect what was attempted.
        try:
            await self._executor.run_command(["bash", "-lc", f"rm -f {quoted}"],
                timeout=5.0,
            )
        except Exception:
            log.debug("apply_patch_cleanup_failed", path=patch_path, exc_info=True)

        metadata: dict = {
            "patch_path": patch_path,
            "changed_files": changed_files[:100],
            "changed_count": len(changed_files),
        }
        if any_verify_failed:
            metadata["verification_failed"] = True
        return ToolResult(
            success=True,
            output=_truncate(output),
            metadata=metadata,
        )


def _similar_blocks(
    content: str, needle: str, *, k: int = 3,
) -> list[str]:
    """Surface the top-k file regions most similar to the missing search.

    Lifted in spirit from Aider's ``find_similar_lines`` — a small
    SequenceMatcher sweep so the error message can nudge the model
    toward the text that's actually in the file. Returns snippets, not
    line numbers, because the model edits by content not line.
    """
    import difflib

    needle_lines = needle.splitlines()
    if not needle_lines:
        return []
    content_lines = content.splitlines()
    n = len(needle_lines)
    if n > len(content_lines):
        return []
    # Sliding window ratio — same shape as the fuzzy tier.
    scored: list[tuple[float, int]] = []
    needle_str = "\n".join(needle_lines)
    for i in range(len(content_lines) - n + 1):
        window = "\n".join(content_lines[i : i + n])
        ratio = difflib.SequenceMatcher(None, needle_str, window).ratio()
        scored.append((ratio, i))
    scored.sort(reverse=True)
    out: list[str] = []
    for ratio, i in scored[:k]:
        if ratio < 0.4:   # cutoff — below this, suggestion is noise
            break
        snippet = "\n".join(content_lines[i : i + n])
        # Cap snippet length so 3 × 80-line suggestions don't blow context
        if len(snippet) > 600:
            snippet = snippet[:580] + "…"
        out.append(f"(line {i + 1}, similarity {ratio:.2f})\n{snippet}")
    return out


def _format_multi_edit_errors(
    *,
    path: str,
    failures: list[dict],
    applied_count: int,
    total: int,
) -> str:
    """Aider-style error envelope.

    Each failure becomes its own block so the model can read + fix
    without scanning a monolithic paragraph. Ends with the atomicity
    reminder — this is the key behavioural nudge: "your earlier edits
    didn't stick, the file is unchanged, fix these and resend."
    """
    lines: list[str] = [
        f"No edits were applied to '{path}' — the batch is atomic and at "
        f"least one edit failed to match. {applied_count}/{total} edits "
        f"WOULD have applied if isolated; fix the failures below and resend "
        f"the full batch.\n",
    ]
    for f in failures:
        lines.append(f"── edit[{f['index']}] ──")
        reason = f.get("reason", "no_match")
        if reason == "no_match":
            lines.append("SearchReplaceNoExactMatch: the search block was not "
                         "located by any of the 5 matching tiers.")
            if f.get("already_present"):
                lines.append(
                    "Note: the REPLACE text is already present in the file. "
                    "This edit may already be applied — drop it from the batch."
                )
            hints = f.get("did_you_mean") or []
            if hints:
                lines.append("Did you mean one of these existing blocks?")
                for h in hints:
                    lines.append(f"  {h}")
            lines.append(f"Your search was:\n{f['search']}")
        else:
            lines.append(reason)
        lines.append("")   # blank separator

    lines.append(
        "The file on disk is UNCHANGED. Fix every failure above and resend "
        "the full edits array in a single code_edit_batch call — don't send "
        "only the failed ones or the applied edits won't be re-applied."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CodeGrepTool
# ---------------------------------------------------------------------------

class CodeGrepTool(_CoderTool):
    """Search for a pattern across workspace files using grep."""

    @property
    def name(self) -> str:
        return "code_grep"

    @property
    def description(self) -> str:
        return (
            "Search for a regex pattern across workspace files (grep -rn). "
            "Returns matching lines with file paths and line numbers. "
            "Pass `context_lines` (e.g. 3) to include surrounding code "
            "with each match — often enough to act on the hit WITHOUT a "
            "follow-up file_read. "
            "Capped at `limit` matches (default 200) — when results are "
            "clipped, the output says so and metadata shows the real count. "
            "Narrow your pattern or raise `limit` for wider scans. "
            "Pass `case_insensitive: true` when the casing is uncertain "
            "(a case-sensitive miss looks like 'no matches'), and `glob` "
            "(e.g. '*.py') to restrict the search to matching files."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "pattern is required":
                "Pass the `pattern` argument — a regex string (grep -E syntax). "
                "Example: {\"pattern\": \"def authenticate\", \"path\": \"/workspace\"}",
            "Invalid regex":
                "Escape regex metacharacters (`(`, `)`, `[`, `]`, `.`, `*`, `+`, "
                "`?`, `|`, `{`, `}`) when searching for them literally, or "
                "use find_files for filename patterns.",
            "No such file or directory":
                "The `path` you scoped to doesn't exist. Default is /workspace. "
                "Use dir_tree to verify valid paths before scoping.",
        }

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for (passed to grep -E)",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (default: /workspace)",
                    "default": "/workspace",
                },
                "limit": {
                    "type":    "integer",
                    "minimum": 1,
                    "default": 200,
                    "description": (
                        "Max matches returned. Default 200 — enough for most "
                        "codebase sweeps; bump to 1000 for exhaustive scans."
                    ),
                },
                "context_lines": {
                    "type":    "integer",
                    "minimum": 0,
                    "maximum": 10,
                    "default": 0,
                    "description": (
                        "Lines of surrounding context per match (grep -C). "
                        "Use 2-3 when you need to see enough code to act "
                        "on the hit — it usually saves a follow-up "
                        "file_read. Default 0 (match lines only)."
                    ),
                },
                "case_insensitive": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Match case-insensitively (grep -i). Default false "
                        "(exact case). Set true when the casing is unknown — "
                        "a case-sensitive miss looks like 'no matches' when "
                        "the symbol really exists in another case."
                    ),
                },
                "glob": {
                    "type": "string",
                    "description": (
                        "Restrict the search to files matching this shell "
                        "glob (grep --include), e.g. '*.py' or '*.{ts,tsx}'. "
                        "Omit to search all files. Use it to skip binaries / "
                        "vendored trees and cut noise."
                    ),
                },
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        *,
        pattern: str = "",
        path: str = "/workspace",
        limit: int = 200,
        context_lines: int = 0,
        case_insensitive: bool = False,
        glob: str = "",
        **_kwargs,
    ) -> ToolResult:
        if not pattern or not pattern.strip():
            return ToolResult(
                success=False,
                error="pattern is required",
                validation_error=True,
            )

        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            limit = 200
        try:
            context_lines = max(0, min(10, int(context_lines)))
        except (TypeError, ValueError):
            context_lines = 0

        effective_path = _resolve_workspace_path(path) if (path and path.strip()) else "/workspace"
        # Ask grep for one more than `limit` so we can detect truncation
        # exactly — if we get limit+1 hits we know the real count is
        # >= limit+1 and we should tell the model to narrow.
        cmd = ["grep", "-rn", "-E", "-m", str(limit * 5)]
        if case_insensitive:
            cmd.append("-i")
        # ``--include=GLOB`` filters by filename so a scoped search skips
        # binaries / vendored trees. grep accepts it repeatedly; we only
        # take one here (the schema is a single string) but split a
        # brace list so '*.{ts,tsx}' still works via multiple flags.
        if glob and glob.strip():
            for g in _expand_brace_glob(glob.strip()):
                cmd.append(f"--include={g}")
        if context_lines:
            cmd += ["-C", str(context_lines)]
        cmd += [pattern, effective_path]

        try:
            output = await self._executor.run_command(cmd)
        except Exception as exc:
            log.warning("code_grep_failed", pattern=pattern, error=str(exc))
            return ToolResult(success=False, error=str(exc))

        if not output.strip():
            return ToolResult(
                success=True,
                output=f"No matches found for pattern '{pattern}' in {effective_path}",
                metadata={
                    "pattern": pattern, "path": effective_path,
                    "matches_found": 0, "matches_shown": 0,
                },
            )

        # Line-based cap: slice by line count (not bytes) so we never cut
        # mid-line. Per-line trim defends against a match landing on a
        # huge single-line file (minified bundles, single-line JSON dumps)
        # — without it a `grep -rn` line is "path:lineno:<entire file>"
        # and 200 of those can be megabytes. `_truncate` is the final
        # belt-and-braces byte ceiling.
        _LINE_CAP = 2000
        all_lines = output.strip().splitlines()
        if context_lines:
            # With -C, grep marks match lines "path:NN:…" and context
            # lines "path-NN-…" (groups separated by "--"). Count and
            # cap by MATCH lines so `limit` keeps meaning "matches",
            # and never cut a context group in half.
            _match_re = re.compile(r"^[^:\n]+:\d+:")
            total_hit = sum(1 for ln in all_lines if _match_re.match(ln))
            shown = []
            match_count = 0
            for ln in all_lines:
                if _match_re.match(ln):
                    match_count += 1
                    if match_count > limit:
                        break
                shown.append(ln)
        else:
            total_hit = len(all_lines)
            shown = all_lines[:limit]
        any_line_trimmed = any(len(ln) > _LINE_CAP for ln in shown)
        body = "\n".join(
            (ln[:_LINE_CAP] + " …[line trimmed]") if len(ln) > _LINE_CAP else ln
            for ln in shown
        )
        if total_hit > limit:
            body += (
                f"\n\n[Showing first {limit} of {total_hit}"
                f"{'+ (grep capped upstream)' if total_hit >= limit * 5 else ''} "
                f"matches. Narrow the pattern, scope path further, or "
                f"raise `limit` to see more.]"
            )
        return ToolResult(
            success=True,
            output=_truncate(body),
            metadata={
                "pattern":       pattern,
                "path":          effective_path,
                "matches_shown": min(total_hit, limit),
                "matches_found": total_hit,
                "likely_more":   total_hit >= limit * 5,
                "line_trimmed":  any_line_trimmed,
                "context_lines": context_lines,
            },
        )


# ---------------------------------------------------------------------------
# CodeGlobTool
# ---------------------------------------------------------------------------

class CodeGlobTool(_CoderTool):
    """Find files by name pattern using find.

    Tool name is ``find_files`` (was ``code_glob`` pre-PR-3). The class
    keeps its old Python name for backward-compat with imports; only
    the LLM-facing tool name changed. Renamed to drop the ``code_*``
    family naming pattern that biased models into hallucinating
    sibling names like ``code_read`` (which doesn't exist)."""

    @property
    def name(self) -> str:
        return "find_files"

    @property
    def description(self) -> str:
        return (
            "Find files in the workspace matching a glob pattern (e.g. '*.py', "
            "'src/**/*.ts'). Returns a list of matching file paths. "
            "Capped at `limit` files (default 200); metadata reports "
            "files_shown vs files_found so you know when to narrow."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match filenames against (e.g. '*.py')",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search from (default: /workspace)",
                    "default": "/workspace",
                },
                "limit": {
                    "type":    "integer",
                    "minimum": 1,
                    "default": 200,
                    "description": (
                        "Max file paths returned. Default 200. Bump for "
                        "wide sweeps; lower for quick peeks."
                    ),
                },
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        *,
        pattern: str = "",
        path: str = "/workspace",
        limit: int = 200,
        **_kwargs,
    ) -> ToolResult:
        if not pattern or not pattern.strip():
            return ToolResult(
                success=False,
                error="pattern is required",
                validation_error=True,
            )

        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            limit = 200

        effective_path = _resolve_workspace_path(path) if (path and path.strip()) else "/workspace"
        cmd = ["find", effective_path, "-name", pattern, "-type", "f"]

        try:
            output = await self._executor.run_command(cmd)
        except Exception as exc:
            log.warning("find_files_failed", pattern=pattern, error=str(exc))
            return ToolResult(success=False, error=str(exc))

        if not output.strip():
            return ToolResult(
                success=True,
                output=f"No files matching '{pattern}' found in {effective_path}",
                metadata={
                    "pattern": pattern, "path": effective_path,
                    "files_found": 0, "files_shown": 0,
                },
            )

        all_files = [f for f in output.strip().splitlines() if f.strip()]
        total = len(all_files)
        shown = all_files[:limit]
        body = "\n".join(shown)
        if total > limit:
            body += (
                f"\n\n[Showing first {limit} of {total} matches. "
                f"Narrow the pattern, scope path further, or raise `limit`.]"
            )
        return ToolResult(
            success=True,
            output=_truncate(body),
            metadata={
                "pattern":     pattern,
                "path":        effective_path,
                "files_shown": len(shown),
                "files_found": total,
                # Back-compat for callers that read metadata["count"].
                "count":       total,
            },
        )


# ---------------------------------------------------------------------------
# ShellExecTool
# ---------------------------------------------------------------------------

# Detects shell commands that spawn a persistent background process so
# the loop can track them across iterations. Three idioms covered:
#
#   1. trailing ``&`` — ``./server &`` or ``./server --flag &``. Excludes
#      ``&&`` (logical AND) via the negative lookbehind on ``&``.
#   2. ``nohup`` / ``disown`` / ``setsid`` — explicit backgrounding verbs.
#   3. ``; disown`` / ``&& disown`` — disown in a compound command.
#
# Observed 2026-04-22: agent ran ``./server &``, later ran it again, hit
# ``Address already in use``, then entered a 20-iter kill/restart spiral
# because nothing in its context said "you already started this."
_BACKGROUND_CMD_RE = re.compile(
    r"(?:"
        r"(?<!&)&\s*(?:$|&&|;|\|\|)"   # trailing & (not &&)
        r"|\bnohup\b"
        r"|\bdisown\b"
        r"|\bsetsid\b"
    r")"
)


class ShellExecTool(_CoderTool):
    """Run any bash command in the workspace container.

    Timeout model:
      - Default command: 120s wall-clock, 60s idle (covers 99% of
        quick checks and small builds).
      - Install/build/compile/clone/fetch/update patterns: 600s
        wall-clock, 120s idle (matches Claude Code's 10-minute cap
        for long-running shells; long builds + dependency downloads
        fit comfortably, while a hung process dies within 2 min of
        going silent).
      - Tool-level timeout (``self.timeout``): 610s — above the max
        inner cap so the outer ``asyncio.wait_for`` wrapper in
        ``_execute_tool`` never preempts the inner command timeout.
    """

    @property
    def timeout(self) -> float:
        # Max inner wall-clock cap + 10s buffer. Keeps the outer
        # tool-dispatch ``wait_for`` from firing before the inner
        # ``run_command`` timeout — the OUTER firing yields the
        # uninformative "Tool 'shell_exec' timed out after Xs" error
        # with no progress output; the INNER firing appends the
        # captured stdout + timeout note, which is actionable.
        # Tied to the model-supplied ceiling so an explicit ``timeout``
        # (clamped to ``_MAX_SHELL_TIMEOUT``) always stays under this
        # outer wrap and the inner path stays authoritative.
        return float(_MAX_SHELL_TIMEOUT + 10)

    # Commands the agent should never run (escape attempts, container
    # mutations, host-level ops). We deliberately do NOT block
    # ``curl … | sh`` / ``curl … | bash`` — the container is the blast
    # boundary, and refusing the upstream-vendor idiom breaks every
    # official install path (rustup, nvm, bun, uv, pnpm, etc.).
    # ``eval "$(cmd)"`` is also common in env-setup scripts but is
    # blocked here because it's a classic encoded-payload smuggling
    # primitive — add it back if a concrete install script needs it.
    _BLOCKED_PATTERNS = (
        "docker ", "dockerd", "containerd",
        "nsenter", "chroot",
        "mount ", "umount ",
        "reboot", "shutdown", "poweroff",
        "rm -rf /",
        "base64 -d |", "base64 --decode |",
        "eval $(", "eval \"$(", "eval '$(",
    )

    @property
    def name(self) -> str:
        return "shell_exec"

    @property
    def description(self) -> str:
        return (
            "Run a bash command in the workspace container. Use for mutations: "
            "npm install, pip install, pytest, git commit, mkdir, mv, rm, etc. "
            "stdout is returned; stderr is not captured separately. "
            "For a command that runs quietly for a long time (a computation, "
            "training run, solver, benchmark), pass `timeout` (seconds, max "
            "600) so it isn't killed for going silent. For anything longer, or "
            "that you want to keep running while you work, pass "
            "`run_in_background: true` — it detaches and you monitor it with "
            "service_logs. Do NOT run docker, mount, reboot, or other system "
            "commands."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SHELL

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "command not found":
                "The binary isn't installed in the container. Check availability "
                "with `which <cmd>` or install via apt-get (on Debian/Ubuntu) / "
                "pip / npm. env_info lists what's installed.",
            "Permission denied":
                "If it's a script, mark it executable: `chmod +x <path>`. If "
                "it's a protected path, writes outside /workspace will fail.",
            "Disallowed command":
                "Blocked by the safety list (docker, mount, reboot, chroot, "
                "eval-pipes, etc.). Rewrite the intent as a direct file or "
                "code change — don't try to work around this.",
            "timed out":
                "Command exceeded its wall-clock budget. If it's a long "
                "computation that runs quietly, pass an explicit `timeout` "
                "(seconds, max 600) — that runs it as pure wall-clock and "
                "turns OFF the go-silent check. If it needs longer than 600s, "
                "or you want to keep working while it runs, pass "
                "`run_in_background: true` and watch it with service_logs. "
                "Auto defaults when no timeout is given: 120s for quick "
                "commands, 600s for install/build/download patterns.",
            "went silent":
                "The process produced no output for the idle window and was "
                "killed as probably-hung. If it was actually busy (a quiet "
                "computation), re-run with an explicit `timeout` (max 600s), "
                "which disables the silence check, or `run_in_background: "
                "true`. If it's genuinely hung (deadlock, waiting on stdin), "
                "check state with `ps aux | grep` before retrying — don't "
                "just re-run.",
            "no such file":
                "Target doesn't exist. Use file_list / dir_tree to verify the "
                "path first.",
        }

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Bash command to execute (run as: bash -lc <command>)",
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        "Directory to run the command in. Relative paths "
                        "resolve under /workspace (e.g. 'backend' → "
                        "/workspace/backend); absolute paths are used as-is. "
                        "Default /workspace. Use this instead of prefixing "
                        "the command with `cd` yourself."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Optional wall-clock limit in seconds (max 600). When "
                        "set, the command runs up to this long with NO "
                        "go-silent kill — use it for quiet, long-running "
                        "computations (training, benchmarks, solvers) that may "
                        "print nothing for minutes. Omit to use the automatic "
                        "default (120s for quick commands, 600s for install/"
                        "build/download), which also kills a process that goes "
                        "silent. For work longer than 600s, use "
                        "run_in_background instead."
                    ),
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": (
                        "Run detached and return immediately with a service "
                        "handle; the process keeps running across turns with "
                        "no timeout. Use for ANYTHING long-running — a dev "
                        "server OR a long computation. Monitor with "
                        "service_logs, stop with service_stop, list with "
                        "service_list. Don't hand-roll nohup or &."
                    ),
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        *,
        command: str = "",
        cwd: str = "",
        timeout: int | None = None,
        run_in_background: bool = False,
        **_kwargs,
    ) -> ToolResult:
        if not command or not command.strip():
            return ToolResult(
                success=False,
                error=(
                    "shell_exec called without a 'command' argument. "
                    "Required: command (string, bash command to run). "
                    'Example: {"command": "make && ./nsnake"}. '
                    "Read INSTALL.md or the project's build file first if "
                    "you don't know the build command."
                ),
                validation_error=True,
            )

        # Block dangerous commands
        cmd_lower = command.lower().strip()
        for pattern in self._BLOCKED_PATTERNS:
            if pattern in cmd_lower:
                log.warning("shell_exec_blocked", command=command, pattern=pattern)
                return ToolResult(
                    success=False,
                    error=f"Command blocked: '{pattern}' is not allowed in workspace containers.",
                )

        # Background mode: detach through the managed-service layer so the
        # process survives across turns and is visible / tailable /
        # stoppable via service_list / service_logs / service_stop (and the
        # UI services panel). One flag on one tool — no switch to
        # service_start, no reasoning about "is this a daemon or a
        # computation." No timeout applies to a backgrounded process.
        if run_in_background:
            return await self._start_background(command, cwd=cwd)

        # ``bash -lc`` instead of ``bash -c`` so /etc/profile.d/*.sh
        # gets sourced — any tool installed into /root/.cargo, /root/
        # .local, /usr/local/go/bin, etc. is visible without the agent
        # having to remember ``source``. Without this, a fresh
        # ``cargo`` / ``pip install --user`` / ``nvm`` tool appears
        # missing in the very next shell_exec, the agent wastes an
        # iteration diagnosing "command not found", and progress stalls.
        #
        # ``cwd`` is honored by prefixing ``cd`` (portable across the
        # container and editor executors, neither of which we can assume
        # forwards a workdir). A bad cwd makes ``cd`` fail so the command
        # never runs — visible in output, not a silent wrong-dir run.
        run_str = command
        if cwd and cwd.strip():
            run_str = f"cd {shlex.quote(_resolve_workspace_path(cwd))} && {command}"
        cmd = ["bash", "-lc", run_str]

        # Two-tier timeout with activity-based idle ceiling. The tier
        # is chosen from the command string (install/build/clone/etc.
        # get the long variant). Idle timeout resets per output chunk,
        # so a streaming ``apt-get install`` / ``cargo build`` /
        # ``npm install`` keeps its full wall-clock budget as long as
        # it's producing output. Only TRUE stalls (process hung, no
        # bytes for N seconds) trigger the idle kill — which is the
        # right behaviour for active downloads.
        # Timeout-tier selection.
        #
        # ``_LONG_PATTERNS`` = keywords that mark a command as expensive-
        # by-nature (build/install chains, bulk transfers). First-token
        # download tools get the long tier automatically because their
        # command string rarely contains a pattern word otherwise
        # (``wget <url> -O <file>`` doesn't mention "download"). Observed
        # 2026-04-22: an un-patterned ``wget -q`` silent download for
        # 700MB kept dying every 60s because the short-tier idle timer
        # fired — even though the file was actively growing on disk.
        # A model-supplied ``timeout`` WINS and is pure wall-clock: no
        # idle / go-silent kill. A long computation that prints nothing for
        # minutes is exactly what it's for — the tier heuristic below can't
        # know a solver is busy, so silence must NOT be read as "hung" when
        # the model has explicitly asked for a longer run. Clamped to
        # ``_MAX_SHELL_TIMEOUT``; ``self.timeout`` keeps the outer dispatch
        # wrap 10s above it so this inner, actionable path fires first.
        explicit = _clamp_timeout(timeout)
        if explicit is not None:
            wall_timeout = float(explicit)
            idle_timeout: float | None = None
            progress_path = None
        else:
            _LONG_PATTERNS = (
                "install", "build", "compile", "make", "clone",
                "pull", "fetch", "update", "upgrade", "download",
            )
            first_token = cmd_lower.split(None, 1)[0] if cmd_lower else ""
            first_basename = first_token.rsplit("/", 1)[-1]
            is_download_tool = first_basename in _DOWNLOAD_FIRST_TOKENS
            if is_download_tool or any(p in cmd_lower for p in _LONG_PATTERNS):
                # Long-running: 10-min wall-clock (Claude Code-style cap),
                # 5-min idle. A dependency-heavy build or big archive
                # download can have multi-minute silent stretches while
                # individual chunks transfer; 2 minutes was too aggressive.
                # Anything idle >5 min AND not making disk progress (see
                # progress_path below) is truly hung.
                wall_timeout = 600.0
                idle_timeout = 300.0
            else:
                # Short commands: 2-min wall-clock, 1-min idle. Most
                # quick checks ("ls", "ps aux", "cat X") finish in
                # seconds; anything taking 60s with no output is almost
                # certainly hung. A quiet-but-busy computation should pass
                # an explicit ``timeout`` (handled above) or background.
                wall_timeout = 120.0
                idle_timeout = 60.0

            # Active-download detection: if this command names an explicit
            # output file via -O / -o / --output, pass the path to
            # run_command so its idle handler can watch the file grow. When
            # the file IS growing we emit a progress heartbeat into the
            # tool output (visible to the user) and reset the idle counter
            # instead of killing. A file that actually stalls across the
            # full idle window still gets terminated with the normal
            # diagnostic.
            progress_path = _parse_download_target(command) if is_download_tool else None

        try:
            output = await self._executor.run_command(cmd,
                timeout=wall_timeout, idle_timeout=idle_timeout,
                progress_path=progress_path,
                on_chunk=self._on_chunk,
            )
        except Exception as exc:
            log.warning("shell_exec_failed", command=command, error=str(exc))
            return ToolResult(success=False, error=str(exc))

        # Detect backgrounded commands and surface them in state so the
        # sticky reminder can show "Background processes started" next
        # iteration. Intentionally wide net: one false positive (the
        # agent sees a process listed it didn't actually background) is
        # far cheaper than one false negative (the agent forgets it
        # started a server and tries to start another).
        if _BACKGROUND_CMD_RE.search(command or ""):
            try:
                self._state.record_background_process(
                    command=command,
                    iteration=getattr(self._state, "tool_calls_made", 0),
                )
            except Exception:
                log.debug("shell_exec_bg_record_failed", exc_info=True)

        return ToolResult(
            success=True,
            output=_truncate(output) if output.strip() else "(exit 0, command succeeded with no stdout)",
            metadata={"command": command},
        )

    async def _start_background(self, command: str, *, cwd: str = "") -> ToolResult:
        """Detach ``command`` as a managed workspace service.

        Reuses :class:`WorkspaceServiceManager` so a backgrounded command
        gets the exact same tracking as ``service_start``: a durable row,
        nohup + logfile, live status, and reachability from service_list /
        service_logs / service_stop and the UI services panel. That is the
        exit-visibility surface — the model tails the log or lists status to
        see when a long computation finished, and the sticky reminder shows
        "you have a background job" next turn.

        Not available in the editor / ACP path, where ``container_manager``
        is None and the manager's ContainerExecutor has nothing to talk to.
        The honest answer there is a foreground run with an explicit
        ``timeout`` — surfaced as a validation error rather than a crash.
        """
        if self._cm is None:
            return ToolResult(
                success=False,
                validation_error=True,
                error=(
                    "run_in_background needs a container workspace and isn't "
                    "available in editor mode. Re-run in the foreground with "
                    "an explicit `timeout` (seconds, max 600) instead."
                ),
            )
        try:
            from augmentum.coder.services import WorkspaceServiceManager

            svc = await WorkspaceServiceManager(
                self._cm,
                self._workspace_id,
                store=self._service_store,
                user_id=self._user_id,
            ).start(
                command=command,
                cwd=_resolve_workspace_path(cwd) if cwd and cwd.strip() else "/workspace",
            )
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc), validation_error=True)
        except Exception as exc:
            log.warning("shell_exec_background_failed", command=command, error=str(exc))
            return ToolResult(
                success=False,
                error=f"Failed to start background command: {exc}",
            )

        # Feed the same sticky-reminder surface raw ``&`` commands use, so
        # the next turn shows the running job and steers away from a
        # double-launch.
        try:
            self._state.record_background_process(
                command=command,
                iteration=getattr(self._state, "tool_calls_made", 0),
            )
        except Exception:
            log.debug("shell_exec_bg_record_failed", exc_info=True)

        ports = ", ".join(str(p) for p in svc.ports) or "none declared"
        return ToolResult(
            success=True,
            output=(
                "Running in background — keep working; it survives across "
                "turns with no timeout.\n"
                f"Service {svc.name} (id {svc.id}, pid {svc.pid}).\n"
                f"Ports: {ports}\n"
                f"Logs: {svc.log_path}\n\n"
                f'Check progress: service_logs(service_id="{svc.id}")\n'
                f'Stop it: service_stop(service_id="{svc.id}")\n'
                "List all running jobs: service_list"
            ),
            metadata={"service": svc.to_dict()},
        )


# ---------------------------------------------------------------------------
# ShellReadTool
# ---------------------------------------------------------------------------

class ShellReadTool(_CoderTool):
    """Run a read-only bash command in the workspace container."""

    @property
    def timeout(self) -> float:
        # Inner run_command uses 30s; outer wrap must exceed that
        # or the dispatcher-level ``wait_for`` preempts with the
        # uninformative "Tool 'shell_read' timed out after 30.0s".
        return 40.0

    @property
    def name(self) -> str:
        return "shell_read"

    @property
    def description(self) -> str:
        return (
            "Run a read-only bash command in the workspace container. Use for "
            "inspection: git log, git status, cat, ls, echo, env, which, etc. "
            "Semantically identical to shell_exec but signals read-only intent."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SHELL

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Read-only bash command to execute (e.g. 'git log --oneline')",
                },
            },
            "required": ["command"],
        }

    async def execute(self, *, command: str = "", **_kwargs) -> ToolResult:
        if not command or not command.strip():
            return ToolResult(
                success=False,
                error=(
                    "shell_read called without a 'command' argument. "
                    "Required: command (string, bash command to run). "
                    'Example: {"command": "ls /workspace"}.'
                ),
                validation_error=True,
            )

        cmd = ["bash", "-c", command]

        try:
            output = await self._executor.run_command(cmd,
                on_chunk=self._on_chunk,
            )
        except Exception as exc:
            log.warning("shell_read_failed", command=command, error=str(exc))
            return ToolResult(success=False, error=str(exc))

        return ToolResult(
            success=True,
            output=_truncate(output) if output.strip() else "(exit 0, command succeeded with no stdout)",
            metadata={"command": command},
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CodeSearchTool (semantic search via codebase index)
# ---------------------------------------------------------------------------

class CodeSearchTool(_CoderTool):
    """Semantic search across the indexed codebase."""

    @property
    def name(self) -> str:
        return "code_search"

    @property
    def description(self) -> str:
        return (
            "Search the codebase by meaning, not just keywords. "
            "Finds code relevant to a natural language query using the semantic index. "
            "Use when you need to find WHERE something is implemented but don't know "
            "the exact function name or file. More powerful than code_grep for conceptual queries."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of what you're looking for (e.g. 'database connection initialization', 'user authentication logic')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def execute(self, *, query: str = "", limit: int = 5, **_kwargs) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(success=False, error="query is required", validation_error=True)

        try:
            from augmentum.coder.indexer import search_index
            results = await search_index(self._workspace_id, query, limit=limit)
        except Exception as exc:
            return ToolResult(success=False, error=f"Search failed: {exc}")

        if not results:
            return ToolResult(
                success=True,
                output=f"No results found for '{query}'. The codebase index may not be built yet.",
                metadata={"query": query, "results": 0},
            )

        lines = [f"Found {len(results)} relevant code sections:\n"]
        for r in results:
            lines.append(f"--- {r.file_path} (lines {r.start_line}-{r.end_line}, score: {r.score}) ---")
            # Show the content without the "File: ..." enrichment prefix
            content = r.content
            if content.startswith("File:"):
                content = content.split("\n", 1)[-1] if "\n" in content else content
            lines.append(content[:2000])
            lines.append("")

        return ToolResult(
            success=True,
            output=_truncate("\n".join(lines)),
            metadata={"query": query, "results": len(results)},
        )


# ---------------------------------------------------------------------------
# DirTreeTool
# ---------------------------------------------------------------------------


class DirTreeTool(_CoderTool):
    """Show directory hierarchy with depth control and line counts."""

    _EXCLUDED_NAMES = frozenset({
        ".augmentum", ".git", "node_modules", "__pycache__",
        ".venv", "venv", "dist", "build",
    })
    _MAX_ENTRIES = 500

    @property
    def name(self) -> str:
        return "dir_tree"

    @property
    def description(self) -> str:
        return (
            "Show the directory tree structure of a path in the workspace. "
            "Returns an indented tree with file sizes. PREFER the "
            "`<workspace_tree>` block already in your system prompt — "
            "it's auto-refreshed with [NEW]/[MOD]/[DEL] markers after "
            "every mutation and covers the whole repo. Only call this "
            "tool when (a) you need to inspect a subdirectory NOT "
            "already listed, (b) the workspace_tree footer says the "
            "listing was truncated and you need the rest, or (c) you "
            "suspect the snapshot is stale and want a fresh view."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (default: /workspace)",
                    "default": "/workspace",
                },
                "depth": {
                    "type": "integer",
                    "description": "Max depth to descend (default: 3, max: 6)",
                    "default": 3,
                },
            },
        }

    async def execute(self, *, path: str = "/workspace", depth: int = 3, **_kw) -> ToolResult:
        depth = max(1, min(depth, 6))
        effective_path = (
            _resolve_workspace_path(path) if (path and path.strip()) else "/workspace"
        )

        def _fmt_size(size_bytes: int) -> str:
            if size_bytes > 1_000_000:
                return f"{size_bytes / 1_000_000:.1f}MB"
            if size_bytes > 1000:
                return f"{size_bytes / 1000:.0f}KB"
            return f"{size_bytes}B"

        async def _walk(current_path: str, depth_left: int, level: int) -> bool:
            try:
                entries = await self._executor.list_files(current_path)
            except Exception as exc:
                raise RuntimeError(str(exc)) from exc

            visible = [
                entry for entry in entries
                if entry.name not in self._EXCLUDED_NAMES
            ]
            visible.sort(key=lambda entry: (not entry.is_dir, entry.name.lower(), entry.name))

            for entry in visible:
                if len(lines) >= self._MAX_ENTRIES:
                    return True
                indent = "  " * level
                if entry.is_dir:
                    lines.append(f"{indent}{entry.name}/")
                    if depth_left > 1:
                        truncated = await _walk(entry.path, depth_left - 1, level + 1)
                        if truncated:
                            return True
                else:
                    lines.append(f"{indent}{entry.name}  ({_fmt_size(entry.size)})")
            return False

        lines = [f"{effective_path.rstrip('/') or effective_path}/"]
        try:
            truncated = await _walk(effective_path, depth, 1)
        except RuntimeError as exc:
            return ToolResult(success=False, error=str(exc))

        if len(lines) == 1:
            return ToolResult(
                success=True,
                output=f"{lines[0]}\n  (empty directory)",
                metadata={"path": effective_path, "entries": 0},
            )

        if truncated:
            lines.append(f"... (truncated at {self._MAX_ENTRIES} entries)")

        return ToolResult(
            success=True,
            output=_truncate("\n".join(lines)),
            metadata={
                "path": effective_path,
                "entries": len(lines) - 1,
                "truncated": truncated,
            },
        )


# ---------------------------------------------------------------------------
# GitTool
# ---------------------------------------------------------------------------


class GitTool(_CoderTool):
    """Structured git operations with parsed output."""

    @property
    def name(self) -> str:
        return "git"

    @property
    def description(self) -> str:
        return (
            "Run git operations with structured output. Actions: status, diff, "
            "log, branch, commit. Use instead of shell_exec for git operations."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SHELL

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "diff", "log", "branch", "commit"],
                    "description": "Git operation to perform",
                },
                "message": {
                    "type": "string",
                    "description": "Commit message (required for 'commit' action)",
                },
                "branch_name": {
                    "type": "string",
                    "description": "Branch name (for 'branch' action: create and checkout)",
                },
                "staged": {
                    "type": "boolean",
                    "description": "Show only staged changes (for 'diff' action)",
                    "default": False,
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of log entries (for 'log' action, default: 10)",
                    "default": 10,
                },
            },
            "required": ["action"],
        }

    async def execute(self, *, action: str = "", **kw) -> ToolResult:
        if not action:
            return ToolResult(success=False, error="action is required", validation_error=True)

        try:
            if action == "status":
                return await self._git_status()
            elif action == "diff":
                return await self._git_diff(staged=kw.get("staged", False))
            elif action == "log":
                return await self._git_log(limit=kw.get("limit", 10))
            elif action == "branch":
                name = kw.get("branch_name", "")
                if not name:
                    return ToolResult(success=False, error="branch_name is required")
                return await self._git_branch(name)
            elif action == "commit":
                msg = kw.get("message", "")
                if not msg:
                    return ToolResult(success=False, error="message is required")
                return await self._git_commit(msg)
            else:
                return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    async def _run(self, cmd_str: str, timeout: float = 15.0) -> str:
        return await self._executor.run_command(["bash", "-c", f"cd /workspace && {cmd_str}"],
            timeout=timeout,
        )

    async def _git_status(self) -> ToolResult:
        branch = (await self._run("git branch --show-current 2>/dev/null")).strip() or "detached"
        status = await self._run("git status --porcelain 2>/dev/null")
        staged = [l[3:] for l in status.splitlines() if l and l[0] in "MADRCU"]
        unstaged = [l[3:] for l in status.splitlines() if l and len(l) > 1 and l[1] in "MADRCU"]
        untracked = [l[3:] for l in status.splitlines() if l.startswith("??")]
        clean = not status.strip()
        output = f"Branch: {branch}\nClean: {clean}\n"
        if staged:
            output += f"Staged ({len(staged)}):\n" + "\n".join(f"  {f}" for f in staged) + "\n"
        if unstaged:
            output += f"Modified ({len(unstaged)}):\n" + "\n".join(f"  {f}" for f in unstaged) + "\n"
        if untracked:
            output += f"Untracked ({len(untracked)}):\n" + "\n".join(f"  {f}" for f in untracked) + "\n"
        return ToolResult(success=True, output=output, metadata={
            "branch": branch, "clean": clean,
            "staged": len(staged), "unstaged": len(unstaged), "untracked": len(untracked),
        })

    async def _git_diff(self, staged: bool = False) -> ToolResult:
        flag = "--cached" if staged else ""
        diff = await self._run(f"git diff {flag} 2>/dev/null")
        if not diff.strip():
            return ToolResult(success=True, output="(no changes)", metadata={"lines": 0})
        return ToolResult(success=True, output=_truncate(diff), metadata={"lines": len(diff.splitlines())})

    async def _git_log(self, limit: int = 10) -> ToolResult:
        limit = max(1, min(limit, 50))
        raw = await self._run(f"git log --oneline -n {limit} 2>/dev/null")
        return ToolResult(success=True, output=raw or "(no commits)", metadata={"count": len(raw.splitlines())})

    async def _git_branch(self, name: str) -> ToolResult:
        output = await self._run(f"git checkout -b '{name}' 2>&1")
        return ToolResult(success=True, output=output.strip() or f"Switched to new branch '{name}'")

    async def _git_commit(self, message: str) -> ToolResult:
        await self._run("git add -A")
        output = await self._run(f"git commit -m '{message}' 2>&1", timeout=30.0)
        return ToolResult(success="nothing to commit" not in output.lower(), output=output.strip())


# ---------------------------------------------------------------------------
# TestRunTool
# ---------------------------------------------------------------------------


class TestRunTool(_CoderTool):
    """Run tests with structured result parsing."""

    @property
    def timeout(self) -> float:
        # Outer dispatch wrap. Must exceed the inner wall-clock ceiling
        # (up to _MAX_SHELL_TIMEOUT via the `timeout` arg) or the
        # dispatcher's wait_for kills the run early with an uninformative
        # "timed out" and no captured test output. The base default (45s)
        # sat BELOW even the old fixed 300s inner cap — a latent early-kill
        # for any suite over 45s.
        return float(_MAX_SHELL_TIMEOUT + 10)

    @property
    def name(self) -> str:
        return "test_run"

    @property
    def description(self) -> str:
        return (
            "Run the project's test suite and parse results into structured output "
            "(passed/failed/error counts with failure details). Automatically detects "
            "pytest, npm test, go test, cargo test. Or specify a custom command."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SHELL

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Test command to run. Leave empty to auto-detect from project files. "
                        "Examples: 'pytest -x', 'npm test', 'go test ./...', 'cargo test'"
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Wall-clock limit in seconds (max 600). Default 300. "
                        "Raise it for a large or integration suite that "
                        "legitimately runs longer than 5 minutes — otherwise "
                        "it's killed mid-run and reported as a failure."
                    ),
                },
            },
        }

    async def execute(
        self, *, command: str = "", timeout: int | None = None, **_kw,
    ) -> ToolResult:
        if not command:
            command = await self._detect_test_command()
            if not command:
                return ToolResult(
                    success=False,
                    error="Could not detect test framework. Specify a command explicitly.",
                )

        # Pure wall-clock, no idle kill — a test suite can legitimately run
        # quietly (a slow fixture, a long integration case). Default 300s;
        # an explicit value is clamped to the same 600s ceiling shell_exec
        # uses. For suites longer than that, run via shell_exec with
        # run_in_background and tail the log.
        wall = _clamp_timeout(timeout)
        if wall is None:
            wall = 300
        try:
            output = await self._executor.run_command(["bash", "-c", f"cd /workspace && {command} 2>&1"],
                timeout=float(wall),
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

        # Shell-level failure detection. Observed 2026-04-20: a weak model
        # ran `test_run` while pytest wasn't installed. The shell returned
        # "bash: line 1: pytest: command not found" — which contained none
        # of the old fallback's "error"/"fail"/"traceback" keywords, so
        # the parser fell through to `passed = 1` and the tool reported
        # success. The thrashing-streak breaker never fired because every
        # test "passed". Detect these shell-level failures explicitly and
        # short-circuit before frame-parsing.
        shell_fail = _shell_command_failure(output)
        if shell_fail is not None:
            full_output = (
                f"ERROR: {shell_fail}\n\n"
                f"The test command `{command}` could not execute — this "
                "is a shell-level failure, not a test failure. Fix the "
                "environment (install the missing tool, correct the path, "
                "or specify a different `command=`) before calling test_run "
                "again.\n\n"
                f"--- Raw Output ---\n{_truncate(output)}"
            )
            return ToolResult(
                success=False,
                error=shell_fail,
                output=full_output,
                metadata={
                    "passed": 0, "failed": 0, "errors": 1,
                    "failures": [],
                    "shell_failure": shell_fail,
                },
            )

        # Try to parse structured results
        parsed = self._parse_results(output, command)
        summary = (
            f"Passed: {parsed['passed']}  Failed: {parsed['failed']}  "
            f"Errors: {parsed['errors']}"
        )
        if parsed["failures"]:
            summary += "\n\nFailures:\n"
            for f in parsed["failures"][:10]:
                summary += f"  - {f['test']}: {f['error'][:200]}\n"

        full_output = f"{summary}\n\n--- Raw Output ---\n{_truncate(output)}"

        return ToolResult(
            success=parsed["failed"] == 0 and parsed["errors"] == 0,
            output=full_output,
            metadata=parsed,
        )

    async def _detect_test_command(self) -> str:
        """Auto-detect the test command from project files."""
        checks = [
            ("test -f /workspace/pytest.ini || test -f /workspace/pyproject.toml || test -f /workspace/setup.cfg", "python -m pytest -x --tb=short"),
            ("test -f /workspace/package.json", "npm test 2>&1 || true"),
            ("test -f /workspace/Cargo.toml", "cargo test 2>&1"),
            ("test -f /workspace/go.mod", "go test ./... 2>&1"),
            ("test -f /workspace/Makefile && grep -qE '^test:' /workspace/Makefile", "make test 2>&1"),
        ]
        for check_cmd, test_cmd in checks:
            try:
                await self._executor.run_command(["bash", "-c", check_cmd], timeout=5.0)
                return test_cmd
            except Exception as exc:
                log.debug("test_command_probe_failed", check=check_cmd, error=str(exc))
                continue
        return ""

    @staticmethod
    def _parse_results(output: str, command: str) -> dict:
        """Parse test output into structured results."""
        result = {"passed": 0, "failed": 0, "errors": 0, "failures": []}

        # Pytest output parsing
        if "pytest" in command or "passed" in output:
            import re
            # Match: "5 passed, 2 failed, 1 error"
            m = re.search(r"(\d+) passed", output)
            if m:
                result["passed"] = int(m.group(1))
            m = re.search(r"(\d+) failed", output)
            if m:
                result["failed"] = int(m.group(1))
            m = re.search(r"(\d+) error", output)
            if m:
                result["errors"] = int(m.group(1))
            # Extract FAILED test names
            for m in re.finditer(r"FAILED (.+?)(?:\s+-\s+(.+))?$", output, re.MULTILINE):
                result["failures"].append({"test": m.group(1), "error": m.group(2) or ""})

        # npm test / jest / mocha
        elif "npm" in command or "jest" in command or "mocha" in command:
            import re
            m = re.search(r"(\d+) passing", output)
            if m:
                result["passed"] = int(m.group(1))
            m = re.search(r"(\d+) failing", output)
            if m:
                result["failed"] = int(m.group(1))
            # Jest format: Tests: X failed, Y passed
            m = re.search(r"Tests:\s+(\d+) failed.*?(\d+) passed", output)
            if m:
                result["failed"] = int(m.group(1))
                result["passed"] = int(m.group(2))

        # Go test
        elif "go test" in command:
            import re
            result["passed"] = len(re.findall(r"--- PASS:", output))
            result["failed"] = len(re.findall(r"--- FAIL:", output))
            for m in re.finditer(r"--- FAIL: (\S+)", output):
                result["failures"].append({"test": m.group(1), "error": ""})

        # Cargo test
        elif "cargo test" in command:
            import re
            m = re.search(r"test result:.*?(\d+) passed.*?(\d+) failed", output)
            if m:
                result["passed"] = int(m.group(1))
                result["failed"] = int(m.group(2))

        # Fallback when none of the framework parsers matched. Pre-
        # 2026-04-20 this defaulted to `passed = 1` on any output lacking
        # "error"/"fail"/"traceback" keywords — which mis-classified
        # silent failures (e.g. "command not found", empty output, fatal
        # warnings without the magic words) as passes. Default now is
        # ERROR, not PASS: if we couldn't parse the output at all, that
        # IS the failure signal. Shell-level failures are caught earlier
        # in execute() so they don't reach here.
        if result["passed"] == 0 and result["failed"] == 0:
            lower = output.lower()
            if "error" in lower or "fail" in lower or "traceback" in lower or output.strip() == "":
                result["errors"] = 1
            else:
                # Output exists and looks benign but we can't confirm a
                # pass — mark as error so the agent doesn't assume green.
                # Worst case: a legitimate silent-success run gets retried
                # with an explicit command=, which is harmless.
                result["errors"] = 1

        return result


# ---------------------------------------------------------------------------
# BlenderRunTool
# ---------------------------------------------------------------------------


class BlenderRunTool(_CoderTool):
    """Run a Blender bpy build script headless and capture its outputs.

    The bpy Python API IS the creation interface — the model writes a
    ``.py`` build script (via ``file_write``) that adds/edits geometry,
    materials, cameras, and optionally exports a GLB / renders a PNG, and
    this tool executes it under a virtual framebuffer:

        ``xvfb-run -a blender --background --python <script> -- <args>``

    Requires the workspace to be on the ``creative`` tooling profile
    (Blender + xvfb installed). On any other profile the tool reports a
    clear "not installed" error rather than pretending to run.

    It does NOT invent a DSL: everything the model wants Blender to do it
    expresses in the script. The tool's job is deterministic execution +
    structured reporting (exit cause, bpy stdout, discovered artifacts).
    """

    @property
    def timeout(self) -> float:
        # Outer dispatch wrap must exceed the inner wall-clock ceiling for
        # the same reason TestRunTool sets this — a legitimate multi-minute
        # bake must not be killed early by the dispatcher's wait_for.
        return float(_MAX_SHELL_TIMEOUT + 10)

    @property
    def name(self) -> str:
        return "blender_run"

    @property
    def description(self) -> str:
        return (
            "Run a Blender bpy Python build script headless (xvfb) and capture "
            "its outputs (exported GLB/.blend, rendered PNG). The bpy API is the "
            "interface — write a .py script that builds/exports/renders, then run "
            "it here. Requires the 'creative' workspace profile. Use Eevee "
            "(default, CPU-fast) unless you need Cycles ray-tracing.\n"
            "REFERENCE (read before writing bpy — a version-exact Blender API "
            "reference is baked into this image at /opt/blender-reference/): "
            "read the cheatsheet with `shell_exec: cat /opt/blender-reference/"
            "bpy_cheatsheet.md` (golden build-script pattern + the common "
            "mistakes — read it FIRST), and look up any operator's exact "
            "signature with `shell_exec: grep <name> /opt/blender-reference/"
            "bpy_api_reference.md` (e.g. `grep primitive_cube_add /opt/"
            "blender-reference/bpy_api_reference.md`). It is also copied to "
            "`.reference/blender/` in the workspace when available.\n"
            "GOTCHA: `bpy.ops.*` operators return a status set `{'FINISHED'}`, "
            "NOT the created object — get the object via "
            "`bpy.context.active_object`. Always set `bpy.context.scene.camera` "
            "before rendering or you get a black frame."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SHELL

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": (
                        "Path to the bpy Python script to execute (relative to "
                        "/workspace or absolute). Write it first with file_write."
                    ),
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Extra arguments forwarded to the script after `--` "
                        "(read via sys.argv in the script)."
                    ),
                },
                "render": {
                    "type": "string",
                    "description": (
                        "Optional output PNG path the script renders to. When "
                        "set, the tool confirms the file exists and reports its "
                        "size so a downstream visual-verify step can pick it up."
                    ),
                },
                "engine": {
                    "type": "string",
                    "enum": ["BLENDER_EEVEE", "CYCLES"],
                    "description": (
                        "Render engine hint forwarded to the script as "
                        "`--engine <value>`. Eevee (default) renders on CPU "
                        "under xvfb; Cycles needs a GPU-capable workspace."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Wall-clock limit in seconds (max 600). Default 300. "
                        "Raise for a heavy Cycles bake."
                    ),
                },
            },
            "required": ["script"],
        }

    async def execute(
        self,
        *,
        script: str = "",
        args: list | None = None,
        render: str = "",
        engine: str = "BLENDER_EEVEE",
        timeout: int | None = None,
        **_kw,
    ) -> ToolResult:
        if not script:
            return ToolResult(
                success=False,
                error="blender_run requires a `script` path (write it with file_write first).",
            )
        wall = _clamp_timeout(timeout)
        if wall is None:
            wall = 300

        # Forward the engine hint as a trailing script arg so the build
        # script can honor it (bpy sets scene.render.engine itself — the
        # CLI has no reliable cross-version engine flag).
        fwd: list[str] = ["--engine", engine]
        for a in (args or []):
            fwd.append(str(a))
        quoted_script = shlex.quote(script)
        quoted_args = " ".join(shlex.quote(a) for a in fwd)
        # `xvfb-run -a` picks a free display; `--background` = headless;
        # `--python-exit-code 1` makes a script exception a non-zero exit so
        # the shell-failure detector below catches it instead of a silent 0.
        cmd = (
            f"cd /workspace && xvfb-run -a blender --background "
            f"--python-exit-code 1 --python {quoted_script} -- {quoted_args} 2>&1"
        )
        try:
            output = await self._executor.run_command(
                ["bash", "-c", cmd], timeout=float(wall),
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

        shell_fail = _shell_command_failure(output)
        if shell_fail is not None:
            hint = ""
            if "blender" in shell_fail and "not found" in shell_fail.lower():
                hint = (
                    "\n\nBlender is not installed — this workspace is not on the "
                    "'creative' tooling profile. Recreate the workspace with the "
                    "creative profile (Blender + xvfb) before using blender_run."
                )
            return ToolResult(
                success=False,
                error=shell_fail,
                output=f"ERROR: {shell_fail}{hint}\n\n--- Raw Output ---\n{_truncate(output)}",
                metadata={"exit": "error", "artifacts": []},
            )

        # Discover produced artifacts. Blender prints "Saved: '/path'" on
        # render/save; also confirm the explicit render path and scan for
        # freshly written GLBs so the loop knows what to hand downstream.
        artifacts = await self._discover_artifacts(output, render)
        summary_lines = [f"Blender run completed (engine={engine})."]
        if render:
            r = next((a for a in artifacts if a["path"] == render), None)
            if r:
                summary_lines.append(f"Render: {render} ({_fmt_size(r['bytes'])})")
            else:
                summary_lines.append(
                    f"WARNING: expected render {render} was not produced."
                )
        glbs = [a for a in artifacts if a["path"].endswith(".glb")]
        if glbs:
            summary_lines.append(
                "GLB: " + ", ".join(f"{a['path']} ({_fmt_size(a['bytes'])})" for a in glbs)
            )
        summary = "\n".join(summary_lines)
        full_output = f"{summary}\n\n--- Raw Output ---\n{_truncate(output)}"
        # Success = command ran cleanly AND, if a render was requested, it
        # exists ("designed ≠ applied" — don't claim success for a missing file).
        ok = (not render) or any(a["path"] == render for a in artifacts)
        return ToolResult(
            success=ok,
            output=full_output,
            metadata={"exit": "ok", "artifacts": artifacts, "engine": engine},
        )

    async def _discover_artifacts(
        self, output: str, render: str,
    ) -> list[dict]:
        """Return [{path, bytes}] for outputs this run produced.

        Sources: paths Blender logged via ``Saved: '<path>'``, the explicit
        ``render`` path, and any ``.glb`` under /workspace touched in the
        last two minutes. Sizes come from a single stat sweep; unreadable
        paths are dropped rather than raising.
        """
        import re

        candidates: set[str] = set()
        for m in re.finditer(r"Saved:\s*'([^']+)'", output):
            candidates.add(m.group(1))
        if render:
            candidates.add(render if render.startswith("/") else f"/workspace/{render}")
        # Recently-written GLBs (export target the model may not have logged).
        try:
            recent = await self._executor.run_command(
                ["bash", "-c",
                 "cd /workspace && find . -name '*.glb' -mmin -2 2>/dev/null | head -20"],
                timeout=10.0,
            )
            for line in recent.splitlines():
                line = line.strip()
                if line:
                    candidates.add(line if line.startswith("/") else f"/workspace/{line.lstrip('./')}")
        except Exception:
            pass

        out: list[dict] = []
        for path in sorted(candidates):
            try:
                sz = await self._executor.run_command(
                    ["bash", "-c", f"stat -c %s {shlex.quote(path)} 2>/dev/null"],
                    timeout=5.0,
                )
                sz = sz.strip()
                if sz.isdigit():
                    # Normalize the explicit render path back to the form the
                    # caller passed so the caller can match it verbatim.
                    disp = render if (render and path.endswith(render.lstrip('./'))) else path
                    out.append({"path": disp, "bytes": int(sz)})
            except Exception:
                continue
        return out


def _fmt_size(n: int) -> str:
    """Compact human byte count for tool summaries."""
    f = float(max(n, 0))
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024.0:
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024.0
    return f"{f:.1f} TB"


# ---------------------------------------------------------------------------
# EnvInfoTool
# ---------------------------------------------------------------------------


class EnvInfoTool(_CoderTool):
    """Return a structured snapshot of the workspace environment."""

    @property
    def name(self) -> str:
        return "env_info"

    @property
    def description(self) -> str:
        return (
            "Get a snapshot of the workspace environment: installed language runtimes, "
            "package managers, project type, disk/memory usage. Call this at the start "
            "of complex tasks to understand what tools and runtimes are available."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **_kw) -> ToolResult:
        profile = "browser"
        try:
            get_workspace = getattr(self._cm, "_get_workspace", None)
            if get_workspace is not None:
                info = await get_workspace(self._workspace_id)
                profile = getattr(info, "tooling_profile", None) or "browser"
        except Exception:
            log.debug("env_info_profile_lookup_failed", exc_info=True)

        script = r"""
echo "=== Runtimes ==="
python3 --version 2>/dev/null || echo "python3: not installed"
node --version 2>/dev/null || echo "node: not installed"
go version 2>/dev/null || echo "go: not installed"
rustc --version 2>/dev/null || echo "rustc: not installed"
java -version 2>&1 | head -1 || echo "java: not installed"
echo ""
echo "=== Package Managers ==="
pip --version 2>/dev/null || echo "pip: not installed"
npm --version 2>/dev/null && echo "(npm)" || true
cargo --version 2>/dev/null || true
uv --version 2>/dev/null || true
pipx --version 2>/dev/null | sed 's/^/pipx /' || true
pnpm --version 2>/dev/null | sed 's/^/pnpm /' || true
yarn --version 2>/dev/null | sed 's/^/yarn /' || true
echo ""
echo "=== Tooling ==="
cmake --version 2>/dev/null | head -1 || true
ninja --version 2>/dev/null | sed 's/^/ninja /' || true
ss --version 2>/dev/null | head -1 || true
python3 -m playwright --version 2>/dev/null || true
echo ""
echo "=== Project Files ==="
ls /workspace/package.json /workspace/requirements.txt /workspace/Pipfile \
   /workspace/pyproject.toml /workspace/Cargo.toml /workspace/go.mod \
   /workspace/Makefile /workspace/docker-compose.yml /workspace/Dockerfile \
   2>/dev/null || echo "(none detected)"
echo ""
echo "=== Disk ==="
df -h /workspace 2>/dev/null | tail -1 | awk '{print "Used: "$3" / "$2" (Free: "$4")"}'
echo ""
echo "=== Memory ==="
free -h 2>/dev/null | grep Mem | awk '{print "Used: "$3" / "$2}'
"""
        try:
            output = await self._executor.run_command(["bash", "-c", script],
                timeout=15.0,
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

        prefix = f"=== Workspace ===\nTooling profile: {profile}\n\n"
        return ToolResult(success=True, output=(prefix + output.strip()).strip())


class ContainerInfoTool(_CoderTool):
    """Return container identity, network, and published-port facts."""

    @property
    def name(self) -> str:
        return "container_info"

    @property
    def description(self) -> str:
        return (
            "Inspect the current workspace container: workspace/container IDs, "
            "hostname, container IP addresses, Docker state/image when available, "
            "and published dev-server ports. Use this for questions like "
            "'what is this container's IP?', 'which ports are exposed?', or "
            "'what workspace am I in?' instead of guessing shell commands."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, **_kw) -> ToolResult:
        lines = [f"Workspace ID: {self._workspace_id}"]
        metadata: dict = {"workspace_id": self._workspace_id}

        info = None
        try:
            get_ws = getattr(self._cm, "_get_workspace", None)
            if get_ws is not None:
                info = await get_ws(self._workspace_id)
        except Exception:
            info = None
        if info is not None:
            container_id = getattr(info, "container_id", None) or ""
            if container_id:
                lines.append(f"Container ID: {container_id[:12]}")
                metadata["container_id"] = container_id
            name = getattr(info, "name", None) or ""
            if name:
                lines.append(f"Workspace name: {name}")
                metadata["workspace_name"] = name
            status = getattr(info, "status", None) or ""
            if status:
                lines.append(f"Workspace status: {status}")
                metadata["status"] = status
            profile = getattr(info, "tooling_profile", None) or "browser"
            lines.append(f"Tooling profile: {profile}")
            metadata["tooling_profile"] = profile

        try:
            details = None
            container_id = metadata.get("container_id")
            docker = getattr(self._cm, "_docker", None)
            if docker is not None and container_id:
                container = await docker.containers.get(container_id)
                details = await container.show()
            if details:
                docker_name = (details.get("Name") or "").lstrip("/")
                image = (details.get("Config") or {}).get("Image") or ""
                state = (details.get("State") or {}).get("Status") or ""
                if docker_name:
                    lines.append(f"Docker name: {docker_name}")
                    metadata["docker_name"] = docker_name
                if image:
                    lines.append(f"Image: {image}")
                    metadata["image"] = image
                if state:
                    lines.append(f"Docker state: {state}")
                    metadata["docker_state"] = state
                networks = (details.get("NetworkSettings") or {}).get("Networks") or {}
                network_ips = {
                    name: (net or {}).get("IPAddress")
                    for name, net in networks.items()
                    if (net or {}).get("IPAddress")
                }
                if network_ips:
                    lines.append("Docker network IPs:")
                    for name, ip in sorted(network_ips.items()):
                        lines.append(f"- {name}: {ip}")
                    metadata["network_ips"] = network_ips
        except Exception:
            log.debug("container_info_docker_inspect_failed", exc_info=True)

        try:
            probe = await self._executor.run_command([
                    "sh", "-lc",
                    "printf 'hostname='; hostname 2>/dev/null || true; "
                    "printf 'ips='; hostname -I 2>/dev/null || true; "
                    "printf 'pwd='; pwd 2>/dev/null || true",
                ],
                timeout=5.0,
            )
        except Exception as exc:
            probe = f"probe_error={exc}"
        probe_data: dict[str, str] = {}
        for line in probe.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                probe_data[key.strip()] = value.strip()
        if probe_data.get("hostname"):
            lines.append(f"Hostname: {probe_data['hostname']}")
            metadata["hostname"] = probe_data["hostname"]
        if probe_data.get("ips"):
            ips = probe_data["ips"].split()
            lines.append("Container IPs from hostname -I: " + ", ".join(ips))
            metadata["ips"] = ips
        if probe_data.get("pwd"):
            lines.append(f"Current working directory: {probe_data['pwd']}")
        if probe_data.get("probe_error"):
            lines.append(f"Shell probe error: {probe_data['probe_error']}")

        try:
            ports = await self._cm.list_ports(self._workspace_id)
        except Exception:
            ports = []
        if ports:
            metadata["ports"] = ports
            lines.append("Published dev ports:")
            for port in ports:
                cport = int(port.get("container_port") or 0)
                hport = int(port.get("host_port") or 0)
                listening = bool(port.get("listening"))
                if hport:
                    url = f"http://127.0.0.1:{hport}"
                    lines.append(
                        f"- {cport}/tcp -> {url} "
                        f"({'listening' if listening else 'not listening'})"
                    )
                else:
                    lines.append(
                        f"- {cport}/tcp -> not published "
                        f"({'listening' if listening else 'not listening'})"
                    )

        return ToolResult(
            success=True,
            output=_truncate("\n".join(lines)),
            metadata=metadata,
        )


class PublishPortsTool(_CoderTool):
    """Expose common dev-server ports for the current workspace."""

    @property
    def name(self) -> str:
        return "publish_ports"

    @property
    def description(self) -> str:
        return (
            "Expose common dev-server ports (3000, 5173, 8000, 8080, etc.) "
            "for the CURRENT workspace by recreating the container against the "
            "same files. Use this instead of shell tunnels when the user wants "
            "browser access to a local app in this workspace. Requires approval."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": (
                        "Short reason for exposing ports, e.g. "
                        "'Need browser access to the local dev server'."
                    ),
                },
            },
            "additionalProperties": False,
        }

    async def execute(self, *, reason: str = "", **_kwargs) -> ToolResult:
        if self._cm is None:
            return ToolResult(
                success=False,
                error="Container manager unavailable; cannot publish workspace ports.",
            )
        try:
            info, changed = await self._cm.enable_published_ports(self._workspace_id)
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Failed to publish workspace ports: {exc}",
            )

        output = (
            "Workspace ports were exposed. Start or keep running the dev server, "
            "then open the port badge in the coder status bar once it appears."
            if changed else
            "Workspace ports were already exposed. Start or keep running the dev "
            "server, then open the port badge in the coder status bar once it appears."
        )
        reason = (reason or "").strip()
        if reason:
            output = f"{output}\nReason: {reason}"
        return ToolResult(
            success=True,
            output=output,
            metadata={
                "workspace_id": self._workspace_id,
                "workspace_recreated": bool(changed),
                "ports_published": True,
                "workspace_status": info.status,
            },
        )


#: Tool names that are safe to use during the Plan phase (read-only operations).
# ``task_list`` is included here so the planning phase can seed the initial
# task list — it mutates only agent-internal state, never the workspace.
READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "file_read",
    "file_list",
    "dir_tree",
    "code_grep",
    "find_files",
    "code_search",
    "find_symbol",
    "file_outline",
    "doc_search",
    "doc_fetch",
    "pack_search",
    "shell_read",
    "env_info",
    "container_info",
    "service_list",
    "service_logs",
    "service_probe",
    "browser_snapshot",
    "browser_verify",
    # browser_evaluate is read-only in the workspace sense (no file or
    # service mutation), but the JS itself can mutate the page DOM. We
    # treat it as read-only for plan-phase eligibility — verifying a
    # page's state shouldn't be gated by the strict-edit guard.
    "browser_evaluate",
    # Same classification as snapshot/verify/evaluate: they observe the
    # page, never the workspace. Membership here is what lets the plan
    # phase and explanatory turns use them (browser_fill_form stays
    # out — it submits forms). They stay serialized in the act loop via
    # NATIVE_SERIAL_TOOL_NAMES (shared Chromium session).
    "browser_wait",
    "browser_extract",
    # Sidecar-native observers: browser_get/browser_console read page or
    # console state; browser_interact/navigate/tabs/find stay out (they
    # act on the page, same classification as click/type).
    "browser_get",
    "browser_console",
    "http_request",
    "db_inspect",
    "profile_read",
    "task_list",
})

# ---------------------------------------------------------------------------
# TaskListTool — structured todo tracking (Claude Code / Codex style)
# ---------------------------------------------------------------------------

_TASK_STATUSES = frozenset({"pending", "in_progress", "completed"})


class TaskListTool(_CoderTool):
    """Create or replace the agent's task list for this session.

    This is the Claude Code / Codex ``TodoWrite`` / ``update_plan``
    equivalent. The agent uses it to track multi-step work; each call
    replaces the list wholesale (partial updates aren't supported — the
    model just re-sends the full state). The resulting list is rendered
    into the sticky ``<system-reminder>`` block every iteration so it
    survives compaction, and the UI can surface it as a live checklist.

    Invariant (enforced here): at most one item in ``in_progress``. This
    is the focus signal — violating it means the agent lost track of
    what it was doing. We return a ``validation_error`` rather than
    silently coercing so the model corrects rather than drifts.
    """

    @property
    def name(self) -> str:
        return "task_list"

    @property
    def description(self) -> str:
        return (
            "Create or update your task list for this session. Use for "
            "multi-step work (3+ steps) to track progress and keep "
            "yourself oriented. Each call replaces the full list — "
            "include EVERY task (pending, in_progress, completed). "
            "Exactly one task may be 'in_progress' at a time. Mark a "
            "task 'completed' only when it's fully done; keep blocked "
            "work in 'in_progress' and add a new task for the blocker. "
            "Skip this tool for trivial or single-step requests."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": (
                        "Full task list. Replaces prior state. Each item: "
                        "content (imperative — 'Run tests'), activeForm "
                        "(present continuous — 'Running tests'), status "
                        "(pending | in_progress | completed)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "content":    {"type": "string"},
                            "activeForm": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["content", "status"],
                    },
                },
            },
            "required": ["items"],
        }

    async def execute(
        self, *, items: list | None = None, **_kwargs,
    ) -> ToolResult:
        if items is None or not isinstance(items, list):
            return ToolResult(
                success=False,
                error=(
                    "task_list called without an 'items' array. Required: "
                    'items: [{content, activeForm, status}]. Example: '
                    '{"items": [{"content": "Read INSTALL.md", '
                    '"activeForm": "Reading INSTALL.md", '
                    '"status": "in_progress"}]}.'
                ),
                validation_error=True,
            )

        # Normalise + validate each item. Reject the whole batch if any
        # item is malformed — partial lists are a common drift source.
        normalised: list[dict] = []
        in_progress_count = 0
        for idx, raw in enumerate(items):
            if not isinstance(raw, dict):
                return ToolResult(
                    success=False,
                    error=f"items[{idx}] must be an object, got {type(raw).__name__}",
                    validation_error=True,
                )
            content = (raw.get("content") or "").strip()
            status = (raw.get("status") or "pending").strip().lower()
            active_form = (raw.get("activeForm") or raw.get("active_form") or "").strip()
            if not content:
                return ToolResult(
                    success=False,
                    error=f"items[{idx}].content is empty — every task needs an imperative description",
                    validation_error=True,
                )
            if status not in _TASK_STATUSES:
                return ToolResult(
                    success=False,
                    error=(
                        f"items[{idx}].status is {status!r}; must be one of "
                        f"pending, in_progress, completed"
                    ),
                    validation_error=True,
                )
            if status == "in_progress":
                in_progress_count += 1
            # activeForm is optional; synthesise a fallback from content
            # so the UI can still render a spinner label.
            if not active_form:
                active_form = content
            normalised.append({
                "content":    content,
                "activeForm": active_form,
                "status":     status,
            })

        if in_progress_count > 1:
            return ToolResult(
                success=False,
                error=(
                    f"{in_progress_count} items marked 'in_progress'. Exactly "
                    "one task should be in_progress at a time — mark the "
                    "others 'pending' until you switch to them."
                ),
                validation_error=True,
            )

        self._state.set_tasks(normalised)

        # Render a human-readable summary so the model's tool_result view
        # doubles as confirmation of what got stored.
        lines = [f"Task list updated — {len(normalised)} item(s):"]
        for t in normalised:
            marker = {
                "completed":   "[x]",
                "in_progress": "[~]",
                "pending":     "[ ]",
            }.get(t["status"], "[ ]")
            lines.append(f"  {marker} {t['content']}")
        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={
                "count":       len(normalised),
                "in_progress": in_progress_count,
            },
        )


# ---------------------------------------------------------------------------
# FinishTaskTool — safe loop exit when tool_choice="required" forbids a
# text-only stop. Also a generally useful "I'm done" signal for weak
# models that otherwise keep retrying because the loop can't tell
# "substantive answer" from "another plan restatement".
# ---------------------------------------------------------------------------

class FinishTaskTool(_CoderTool):
    """Signal that the current user request is fully addressed.

    Calling this tool terminates the act loop on the next iteration with
    ``termination_reason="finish_task_called"``. The ``summary`` argument
    becomes the user-visible answer — the synthesis phase uses it as
    the authoritative task outcome rather than trying to reconstruct
    one from tool results.

    Why this exists: when the plan-prose detector escalates to
    ``tool_choice="required"`` (see ``phase_act.py:_is_plan_prose_dominated``),
    the model can no longer emit a terminal text response. Without an
    explicit "I'm done" tool, a genuinely-complete task would loop
    forever on forced tool calls. ``finish_task`` is that escape hatch.

    Also useful outside the force-tools path: weak models routinely
    repeat themselves rather than stop. A direct ``finish_task`` signal
    is unambiguous in a way "no more tool calls + terse prose" is not.
    """

    @property
    def name(self) -> str:
        return "finish_task"

    @property
    def description(self) -> str:
        return (
            "Signal that the user's request is fully addressed — the "
            "task is DONE. Call this instead of emitting a final "
            "summary as prose when you're finished. The 'summary' "
            "argument is what the user sees. Only call when genuinely "
            "complete: all requested work delivered, or blocked with "
            "no further action possible. DO NOT use to escape a "
            "difficult task — if stuck, use ask_user or explain the "
            "blocker and call finish_task with that as the summary."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "User-facing summary of what was done (or why "
                        "the task couldn't be completed). 1-4 "
                        "sentences. This IS the final answer — write "
                        "it for the user, not as a log entry."
                    ),
                    "minLength": 1,
                    "maxLength": 2000,
                },
            },
            "required": ["summary"],
            "additionalProperties": False,
        }

    async def execute(
        self, *, summary: str = "", **_kwargs,
    ) -> ToolResult:
        summary = (summary or "").strip()
        if not summary:
            return ToolResult(
                success=False,
                validation_error=True,
                error=(
                    "finish_task called without a 'summary'. The "
                    "summary IS the user-facing answer — it can't be "
                    "empty. Pass a 1-4 sentence description of what "
                    "you delivered, or what blocked you."
                ),
            )
        self._state.finish_requested = True
        self._state.finish_summary = summary
        return ToolResult(
            success=True,
            output=(
                "Task marked complete. The act loop will terminate on "
                "the next iteration and deliver your summary to the "
                "user. Do not call any further tools."
            ),
            metadata={"summary_chars": len(summary)},
        )


# ---------------------------------------------------------------------------
# CompactTool — model-initiated history compaction with a self-written
# handoff note. The agent decides WHEN to fold (semantic seams: phase
# closure, resolved dead-ends, landed verdicts) instead of waiting for
# the harness threshold to force it mid-thought.
# ---------------------------------------------------------------------------

_COMPACT_MAX_USES_PER_TURN = 2
_COMPACT_FIELD_MAX = 1_200


class CompactTool(_CoderTool):
    """Fold older working history into a self-written handoff note.

    Signal-flag tool (same pattern as ``finish_task``): execution sets
    ``state.compact_requested`` + ``state.compact_note``; the act loop's
    compaction step consumes the flag at the top of the next iteration
    and performs the actual fold with the model's note as the synthesis
    segment (``_compact_messages_with_synthesis`` in the coder handler).
    The note uses the exact four-line shape the second-model synthesis
    emits, so downstream consumers (the ``<compacted>`` block, training
    windows) see one format regardless of who wrote it.

    The model writing its own note is the point: the note is what it
    will remember, so authorship forces an honest state inventory. The
    mechanical segment (file tallies, tool counts, previews) still rides
    alongside as the grounded backbone the note can't fabricate over.
    """

    @property
    def name(self) -> str:
        return "compact"

    @property
    def description(self) -> str:
        return (
            "Fold your older working history into a handoff note you "
            "write yourself. Call at a natural seam: a phase just "
            "closed (exploration done, moving to implementation), a "
            "dead-end was resolved (keep the lesson, drop the flailing), "
            "or a verdict landed and its evidence trail is no longer "
            "needed. Recent messages stay verbatim; your note becomes "
            "your memory of the folded region, so include every fact "
            "you'll still need. Do NOT call mid-hypothesis, while "
            "recently-read file contents are still needed, or when "
            "little has accumulated."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        def _field(desc: str) -> dict:
            return {
                "type": "string",
                "description": desc,
                "minLength": 1,
                "maxLength": _COMPACT_FIELD_MAX,
            }
        return {
            "type": "object",
            "properties": {
                "state": _field(
                    "What is completed and verified, right now. "
                    "Specific: file paths, function names, error strings."
                ),
                "decisions": _field(
                    "Choices made and WHY, including approaches ruled "
                    "out. 'none' if none."
                ),
                "learnings": _field(
                    "Gotchas, constraints, or facts that would be "
                    "expensive to rediscover. 'none' if none."
                ),
                "next": _field(
                    "What remains or was in flight when you wrote this."
                ),
            },
            "required": ["state", "decisions", "learnings", "next"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        state: str = "",
        decisions: str = "",
        learnings: str = "",
        next: str = "",  # noqa: A002 — schema field name, shadows builtin
        **_kwargs,
    ) -> ToolResult:
        fields = {
            "State": (state or "").strip(),
            "Decisions": (decisions or "").strip(),
            "Learnings": (learnings or "").strip(),
            "Next": (next or "").strip(),
        }
        missing = [k for k, v in fields.items() if not v]
        if missing:
            return ToolResult(
                success=False,
                validation_error=True,
                error=(
                    f"compact called with empty field(s): "
                    f"{', '.join(missing)}. The note IS your memory of "
                    "the folded history — every field is required "
                    "(write 'none' only when genuinely empty)."
                ),
            )
        if self._state.compact_requested:
            return ToolResult(
                success=False,
                error=(
                    "A compaction is already pending from this "
                    "iteration. Continue working; it applies before "
                    "your next step."
                ),
            )
        if self._state.compact_tool_uses >= _COMPACT_MAX_USES_PER_TURN:
            return ToolResult(
                success=False,
                error=(
                    "compact has already been used "
                    f"{_COMPACT_MAX_USES_PER_TURN} times this turn. "
                    "Automatic compaction still protects the context "
                    "ceiling — keep working."
                ),
            )
        note = "\n".join(f"{k}: {v}" for k, v in fields.items())
        self._state.compact_requested = True
        self._state.compact_note = note
        self._state.compact_tool_uses += 1
        return ToolResult(
            success=True,
            output=(
                "Handoff note recorded. Older history folds into the "
                "<compacted> block before your next step; recent "
                "messages stay verbatim. Re-anchor on the goal and "
                "continue from your note."
            ),
            metadata={"note_chars": len(note)},
        )


# ---------------------------------------------------------------------------
# AskUserTool — mid-task clarification via the handler's question callback
# ---------------------------------------------------------------------------

class AskUserTool(_CoderTool):
    """Pause the agent loop to ask the user a clarifying question.

    Architectural reuse: the handler already has a ``permission_callback``
    that suspends-then-resumes for yes/no tool approvals. This tool
    extends the same primitive to multi-option questions — studied the
    qwen-code ``askUserQuestion`` and OpenCode ``question`` tools and
    both resolve to "reuse your confirmation gate, don't build a parallel
    event channel." We do the same.

    The callback is injected via ``create_coder_tools(question_callback=...)``
    from the handler. When None (tests, or a deployment without an
    interactive frontend), the tool returns a degraded error so the
    model can fall back to guessing rather than hanging on an awaitable
    that never resolves.
    """

    def __init__(
        self,
        *,
        container_manager: ContainerManager,
        workspace_id: str,
        state: CoderState,
        question_callback=None,
        executor: WorkspaceExecutor | None = None,
        profile_store=None,
        service_store=None,
        user_id: str = "",
        strict_edit_guard: bool = True,
    ) -> None:
        super().__init__(
            container_manager=container_manager,
            workspace_id=workspace_id,
            state=state,
            executor=executor,
            profile_store=profile_store,
            service_store=service_store,
            user_id=user_id,
            strict_edit_guard=strict_edit_guard,
        )
        self._question_callback = question_callback

    @property
    def name(self) -> str:
        return "ask_user"

    @property
    def description(self) -> str:
        return (
            "Ask the user a clarifying question when you hit ambiguity you "
            "can't resolve from the codebase alone. Use sparingly — only "
            "when the right choice genuinely can't be inferred (API design "
            "preference, which of N approaches to take, domain facts not "
            "in the code). DO NOT use for: yes/no approval of destructive "
            "actions (the permission system handles that), things you can "
            "verify with file_read or shell_read, or to stall on straightforward "
            "tasks. Prefer specific, closed-form questions with 2-5 options."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "questions": {
                    "type":  "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "description": (
                        "Usually one question. Up to 5 if they're tightly "
                        "related (e.g. 'should I use X or Y?' plus 'and "
                        "name it A or B?') and the user would naturally "
                        "answer them together."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The question, phrased for the user.",
                            },
                            "options": {
                                "type":  "array",
                                "items": {"type": "string"},
                                "description": (
                                    "2-5 concrete options. User always has an "
                                    "'Other' escape hatch for free-text. Mark "
                                    "a recommended option with '(Recommended)' "
                                    "suffix."
                                ),
                            },
                            "multi_select": {
                                "type":    "boolean",
                                "default": False,
                                "description": (
                                    "Set true only when multiple options can "
                                    "coexist (e.g. 'which test frameworks?'). "
                                    "Default false (single-choice)."
                                ),
                            },
                        },
                        "required": ["prompt", "options"],
                    },
                },
            },
            "required": ["questions"],
        }

    async def execute(
        self, *, questions: list | None = None, **_kwargs,
    ) -> ToolResult:
        if not isinstance(questions, list) or not questions:
            return ToolResult(
                success=False,
                validation_error=True,
                error=(
                    "ask_user called without a 'questions' array. Required: "
                    'questions: [{prompt, options}]. Example: '
                    '{"questions": [{"prompt": "Which test framework?", '
                    '"options": ["pytest", "unittest"]}]}.'
                ),
            )

        # Shape + validate every question. Reject the whole batch on any
        # malformed item — partial question sets confuse the UI.
        cleaned: list[dict] = []
        for i, raw in enumerate(questions):
            if not isinstance(raw, dict):
                return ToolResult(
                    success=False,
                    validation_error=True,
                    error=f"questions[{i}] must be an object",
                )
            prompt = (raw.get("prompt") or "").strip()
            options = raw.get("options") or []
            if not prompt:
                return ToolResult(
                    success=False,
                    validation_error=True,
                    error=f"questions[{i}].prompt is empty",
                )
            if not isinstance(options, list) or len(options) < 2:
                return ToolResult(
                    success=False,
                    validation_error=True,
                    error=(
                        f"questions[{i}].options must be an array of at "
                        "least 2 strings. Add real choices — a single "
                        "option is not a question."
                    ),
                )
            # Defensive cap — a 20-option list is almost certainly a mistake
            # and kills the UI's ability to render cleanly.
            if len(options) > 8:
                options = options[:8]
            cleaned.append({
                "prompt":       prompt,
                "options":      [str(o) for o in options],
                "multi_select": bool(raw.get("multi_select", False)),
            })

        if self._question_callback is None:
            # Option-3 short-circuit (pending a full question_registry +
            # modal bridge — tracked separately). Rather than a hard
            # refusal that forces the model to guess, format the
            # questions as a user-facing chat message, reuse the
            # finish_task signal to terminate the turn cleanly, and let
            # the user answer in their next message. The user's reply
            # becomes normal chat history — the model picks it up on
            # the next turn with no extra harness required.
            #
            # Semantic note: we reuse finish_requested/finish_summary
            # because both pathways — "task done" and "waiting on
            # clarification" — share the same loop-termination + visible-
            # summary shape. A future question_registry build-out would
            # replace this branch with a real callback invocation; the
            # tool contract doesn't change.
            lines = [
                "I need a bit more info before continuing:",
                "",
            ]
            for q in cleaned:
                lines.append(f"**{q['prompt']}**")
                for i, opt in enumerate(q["options"], start=1):
                    lines.append(f"  {i}. {opt}")
                if q.get("multi_select"):
                    lines.append(
                        "  _(pick one or more — reply with the numbers "
                        "or names, comma-separated)_",
                    )
                lines.append("")
            lines.append(
                "Reply with your choice(s) and I'll pick up from there.",
            )
            message = "\n".join(lines).rstrip()

            self._state.finish_requested = True
            self._state.finish_summary = message
            return ToolResult(
                success=True,
                output=(
                    "Question posted to the user; the turn will end "
                    "after this so they can reply. Do not call any "
                    "further tools — wait for their answer on the "
                    "next turn."
                ),
                metadata={
                    "pending_question": True,
                    "questions_count":  len(cleaned),
                },
            )

        try:
            answers = await self._question_callback(cleaned)
        except Exception as exc:
            log.warning("coder.ask_user_callback_failed", error=str(exc))
            return ToolResult(
                success=False,
                error=f"User interaction failed: {exc}. Proceed with best judgement.",
            )

        if answers is None:
            # Explicit user cancellation — not a bug. Signal it cleanly
            # so the model can either ask more specifically or give up.
            return ToolResult(
                success=False,
                error=(
                    "User declined to answer. Proceed with your best "
                    "inference or stop and summarise what's blocking."
                ),
            )

        # Render the answers as plain text so the tool_result the LLM
        # reads is self-contained — no schema archaeology required.
        lines = ["User answered:"]
        for q, a in zip(cleaned, answers, strict=False):
            if isinstance(a, dict):
                chosen = a.get("answer") or a.get("value") or str(a)
            elif isinstance(a, list):
                chosen = ", ".join(str(x) for x in a)
            else:
                chosen = str(a)
            lines.append(f"  Q: {q['prompt']}")
            lines.append(f"  A: {chosen}")
        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"answers": answers},
        )


#: Ordered list of all coder tool classes for registration.
# Import web tools (doc_search, doc_fetch) from the web_tools module
# Sidecar-native browser verbs (persistent agent-browser service)
from augmentum.coder.browser_tools_native import (
    BrowserConsoleTool,
    BrowserFindTool,
    BrowserGetTool,
    BrowserInteractTool,
    BrowserNavigateTool,
    BrowserTabsTool,
)

# Offline knowledge-pack search (dynamic description lists installed packs)
from augmentum.coder.knowledge_tools import PackSearchTool
from augmentum.coder.runtime_tools import (
    BrowserClickTool,
    BrowserEvaluateTool,
    BrowserExtractTool,
    BrowserFillFormTool,
    BrowserOpenTool,
    BrowserScreenshotTool,
    BrowserSnapshotTool,
    BrowserTypeTool,
    BrowserVerifyTool,
    BrowserWaitTool,
    DbInspectTool,
    HttpRequestTool,
    ObserveTool,
    ProfileReadTool,
    ProfileUpdateTool,
    ServiceListTool,
    ServiceLogsTool,
    ServiceProbeTool,
    ServiceStartTool,
    ServiceStopTool,
)
from augmentum.coder.terminal_tools import (
    TermCloseTool,
    TermListTool,
    TermOpenTool,
    TermSendTool,
    TermSnapshotTool,
)
from augmentum.coder.web_tools import DocFetchTool, DocSearchTool


class TaskDispatchTool(_CoderTool):
    """Spawn a focused subagent with its own model + tool subset + budget.

    Mirrors Claude Code's ``Task`` tool. The lead agent calls this when
    a subtask (exploration, design pass, second-opinion review, grounded
    research) benefits from its own context budget rather than crowding
    the lead's. Built-in roles ship in ``augmentum/agents/presets.py``;
    user roles drop into ``.augmentum/agents/*.md`` (workspace-local) or
    ``~/.augmentum/agents/*.md`` (global).

    Multi-provider: ``model`` accepts ``model@provider`` or
    ``model@fabric:peer_id`` syntax so a subagent can run on a different
    backend (Anthropic API, local llama-server, peer machine) than the
    lead. Falls back through the role's preferred → fallbacks chain
    when the override resolves to no available backend.
    """

    def __init__(
        self,
        *,
        container_manager: ContainerManager,
        workspace_id: str,
        state: CoderState,
        dispatcher,
        profile_store=None,
        service_store=None,
        user_id: str = "",
        strict_edit_guard: bool = True,
    ) -> None:
        super().__init__(
            container_manager=container_manager,
            workspace_id=workspace_id,
            state=state,
            profile_store=profile_store,
            service_store=service_store,
            user_id=user_id,
            strict_edit_guard=strict_edit_guard,
        )
        self._dispatcher = dispatcher
        # Live-progress sink installed by the streaming wrapper in
        # ``_execute_tool_with_subagent_stream``. Mirrors the
        # ``_on_chunk`` pattern on ShellExecTool: the wrapper sets this
        # to a producer-side push function before invoking execute(),
        # and clears it afterwards. None = the tool runs without
        # surfacing live activity (e.g. unit-test direct call).
        self._on_progress = None

    @property
    def name(self) -> str:
        return "task_dispatch"

    @property
    def description(self) -> str:
        return (
            "Spawn a focused subagent (Claude Code Task-style) when "
            "delegating beats grinding. Use ANY of:\n"
            "  - About to read 5+ files searching for one thing → "
            "role=explore\n"
            "  - Stuck between 2-3 approaches for a non-trivial change → "
            "role=plan\n"
            "  - Just made a complex multi-file change worth a second "
            "opinion → role=review\n"
            "  - Need an API/library answer beyond your training → "
            "role=research\n"
            "  - Auditing a file/diff for vulnerabilities → "
            "role=security_review\n"
            "  - Need a threat model document for downstream security "
            "tooling → role=threat_model\n\n"
            "The subagent runs in its own context budget — its file_reads "
            "don't crowd yours. Result is a structured tool output; "
            "treat it like any other tool's answer and continue.\n\n"
            "Multi-provider: pass `model='name@provider'` (e.g. "
            "`claude-haiku-4-5@anthropic`) or `name@fabric:peer_id` to "
            "route to a different backend than the lead. Don't dispatch "
            "for single-file edits or work the user explicitly asked you "
            "to do — those belong to YOU."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "description": (
                        "Registered role name. Built-ins: explore, plan, "
                        "review, research, security_review, threat_model. "
                        "User-defined roles in .augmentum/agents/*.md "
                        "override built-ins of the same name."
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "The focused task to hand to the subagent. State "
                        "the goal; do NOT include your full conversation "
                        "history. Put the definition-of-done in "
                        "`success_criteria`, not buried in prose."
                    ),
                },
                "success_criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Definition-of-done: each item is one concrete, "
                        "checkable condition the subagent must satisfy (or "
                        "explicitly report it couldn't). The subagent self-"
                        "checks against these before returning. Strongly "
                        "recommended — it's the difference between an aligned "
                        "result and one that technically answered the prompt "
                        "but missed the point. e.g. ['lists every caller of "
                        "resolve_backend with file:line', 'flags any caller "
                        "that skips the fabric path']."
                    ),
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Hard limits the subagent must respect beyond its "
                        "role's own rules. e.g. ['read-only — propose, don't "
                        "apply', 'ignore the vendored/ tree']."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": (
                        "Optional model override. Format: `model_id` | "
                        "`model_id@provider` | `model_id@fabric:peer_id`. "
                        "Empty = use the role's preferred model (which "
                        "falls back to the parent's model)."
                    ),
                },
                "context": {
                    "type": "string",
                    "enum": ["slim", "workspace", "hot"],
                    "description": (
                        "How much parent context to inherit. "
                        "slim = nothing; workspace = kernel facts "
                        "(default); hot = workspace + recent tool digests."
                    ),
                },
            },
            "required": ["role", "prompt"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        role: str = "",
        prompt: str = "",
        success_criteria: list | tuple | str | None = None,
        constraints: list | tuple | str | None = None,
        model: str = "",
        context: str = "",
        **_kwargs,
    ) -> ToolResult:
        from augmentum.agents.dispatch import DispatchRequest
        from augmentum.agents.resolve import SubagentModelUnavailableError

        role = (role or "").strip()
        prompt = (prompt or "").strip()
        if not role:
            return ToolResult(
                success=False,
                validation_error=True,
                error=(
                    "task_dispatch requires `role`. Built-in choices: "
                    "explore, plan, review, research. User-defined roles "
                    "live in .augmentum/agents/*.md."
                ),
            )
        if not prompt:
            return ToolResult(
                success=False,
                validation_error=True,
                error=(
                    "task_dispatch requires `prompt` — the focused task "
                    "to hand to the subagent."
                ),
            )
        if self._dispatcher is None:
            return ToolResult(
                success=False,
                error=(
                    "task_dispatch is unavailable: subagent dispatcher "
                    "is not wired in this session. Check that "
                    "coder_subagents_enabled is True in settings."
                ),
            )

        def _as_tuple(val) -> tuple[str, ...]:
            # Models sometimes hand back a single string, a JSON-ish list,
            # or a newline-joined blob. Normalize all of them to a tuple of
            # non-empty trimmed strings so the contract is predictable.
            if val is None:
                return ()
            if isinstance(val, str):
                parts = [p.strip() for p in val.replace("\r", "").split("\n")]
                return tuple(p for p in parts if p)
            if isinstance(val, (list, tuple)):
                return tuple(str(p).strip() for p in val if str(p).strip())
            return ()

        req = DispatchRequest(
            role=role,
            prompt=prompt,
            success_criteria=_as_tuple(success_criteria),
            constraints=_as_tuple(constraints),
            model_override=(model or "").strip(),
            context_mode_override=(context or "").strip(),
            workspace_id=self._workspace_id,
            session_id=getattr(self._state, "session_id", "") or "",
            user_id=self._user_id,
            parent_run_id=getattr(self._state, "current_run_id", "") or "",
            parent_turn_id=getattr(self._state, "current_turn_id", "") or "",
            progress_callback=self._on_progress,
        )

        try:
            outcome = await self._dispatcher.dispatch(req)
        except ValueError as exc:
            return ToolResult(
                success=False,
                validation_error=True,
                error=str(exc),
            )
        except SubagentModelUnavailableError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        result = outcome.result
        # ``cancelled`` is NOT a success state — the lead should see
        # the recovery hint and decide whether to retry or work inline.
        # ``budget`` IS treated as success-with-warning because some
        # roles intentionally hit budget on long jobs and produce
        # useful partial output (the explore role digests N files
        # and stops at iteration cap — the digest is still valuable).
        #
        # A clean ``complete`` whose VERIFICATION FAILED is demoted to
        # non-success: the subagent stopped tidily but an independent
        # judge found its success_criteria unmet, so the lead must NOT
        # bank it as done. The recovery_hint (set by the loop) names the
        # unmet criteria; surfacing it through the error path — not just
        # the prose footer — engages the lead's retry/repair handling.
        verification = getattr(result, "verification", "unchecked")
        success = (
            result.stop_reason in {"complete", "budget"}
            and verification != "failed"
        )

        # Compose the model-facing tool output — the subagent's final
        # text plus a compact metadata footer so the lead can decide
        # whether to trust it or retry. When stop_reason != complete
        # we tack on the structured recovery_hint so the lead model
        # gets explicit guidance ("Subagent looped on REPEATED_TOOL_
        # CALLS — re-dispatch with a sharper prompt that…") instead
        # of inferring from raw stop_reason. This is the difference
        # between the lead burning 5 iterations to recover and the
        # lead doing the right thing on its first follow-up.
        output_parts: list[str] = []
        if result.output.strip():
            output_parts.append(result.output.strip())
        output_parts.append(
            f"\n--- subagent {outcome.role!r} via {outcome.model_resolved!r}"
            f" — stop:{result.stop_reason}"
            + (f" ({result.stop_detail})" if result.stop_detail else "")
            + (f" — verify:{verification}" if verification != "unchecked" else "")
            + f" — iters:{result.iterations}"
            f" tokens:{result.tokens_in + result.tokens_out}"
            f" wall:{result.wallclock_ms}ms ---"
        )
        if result.recovery_hint:
            output_parts.append(f"\n[recovery] {result.recovery_hint}")
        if verification == "error":
            # The independent judge could not run (backend/parse failure)
            # — fail-open by design, but the lead must know the report is
            # UNVERIFIED rather than reading `verify:error` as noise. A
            # silent fail-open masquerading as a checked result is the
            # trust bug this line exists to prevent.
            output_parts.append(
                "\n[unverified] The independent success-criteria check "
                "could not run — the report above is the subagent's OWN "
                "claim. Spot-check any load-bearing result before "
                "building on it."
            )
        if verification == "failed":
            err = (
                "subagent verification failed: success criteria unmet"
                + (f" — {result.verification_reason}" if result.verification_reason else "")
            )
        elif not success:
            err = f"subagent stopped: {result.stop_reason} ({result.stop_detail or 'no detail'})"
        else:
            err = ""
        return ToolResult(
            success=success,
            output="\n".join(output_parts),
            error=err,
            metadata={
                "subagent_id": outcome.subagent_id,
                "role": outcome.role,
                "model_spec": outcome.model_spec,
                "model_resolved": outcome.model_resolved,
                "stop_reason": result.stop_reason,
                "stop_detail": result.stop_detail,
                "stuck_pattern": result.stuck_pattern,
                "iterations": result.iterations,
                "tool_calls": result.tool_calls,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "wallclock_ms": result.wallclock_ms,
                "recovery_hint": result.recovery_hint,
                "verification": verification,
                "verification_reason": getattr(result, "verification_reason", ""),
                "verification_unmet": list(getattr(result, "verification_unmet", []) or []),
            },
        )


class ExploreCodebaseTool(TaskDispatchTool):
    """Concrete-surface alias for an ``explore`` subagent dispatch.

    Local/open models reliably DON'T call the abstract ``task_dispatch``
    meta-tool — delegation is an RL-trained behaviour they lack, so they
    grind through files themselves instead (validated 2026-06-19:
    Qwen3-Coder-30B chose to self-read on every explore-shaped task with
    ``task_dispatch`` offered, but DELEGATED the moment the same subagent
    was dressed as a concrete ``explore_codebase(query)`` tool). This
    class is that dressing: an in-distribution "search the codebase" verb
    with one obvious arg, whose body is a full ``explore`` subagent.

    It inherits the ENTIRE dispatch path from :class:`TaskDispatchTool`
    (persistence to ``coder_subagent_runs``, the ``_on_progress`` live
    stream, success-criteria verification, the metadata footer) — the only
    thing this subclass does is supply the meta-cognition the model
    skipped: it fixes ``role=explore``, frames the ``prompt``, and derives
    ``success_criteria`` programmatically from the single ``query`` arg.
    The model never sees "subagent", "role", or "success_criteria".
    """

    @property
    def name(self) -> str:
        return "explore_codebase"

    @property
    def description(self) -> str:
        return (
            "Search the codebase to find where and how something is "
            "implemented or used. Reads across as many files as needed and "
            "returns a structured summary of every relevant location "
            "(file:line + what it does). Use this instead of opening files "
            "one by one when the answer is spread across the repo — e.g. "
            "'find every caller of X', 'where is Y handled', 'how does Z "
            "work across the codebase'. Runs in its own context budget, so "
            "it won't crowd your working context with file dumps."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What to find, in plain words — e.g. 'every call "
                        "site of resolve_backend_for_model and what it "
                        "passes', 'where session tokens are validated', "
                        "'how the migration runner discovers files'."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, *, query: str = "", **_kwargs) -> ToolResult:
        # Tolerate a model that fills `prompt`/`q` instead of `query`.
        query = (query or _kwargs.get("prompt") or _kwargs.get("q") or "").strip()
        if not query:
            return ToolResult(
                success=False,
                validation_error=True,
                error=(
                    "explore_codebase requires `query` — what to find in "
                    "the codebase, in plain words."
                ),
            )
        # Supply the meta-cognition the model skipped: frame the task for an
        # explore subagent and derive a checkable definition-of-done. These
        # are exactly what a well-formed `task_dispatch(role=explore, …)`
        # would carry — we just author them from the one arg the model gave.
        prompt = (
            f"Explore the codebase to answer this: {query}\n"
            "Read as many files as needed. Report EVERY relevant location "
            "as file:line with a one-line note on what it does, then a short "
            "summary. If nothing relevant exists, say so explicitly."
        )
        success_criteria = (
            f"Addresses the query across the codebase, not just the first file: {query}",
            "Every relevant location is reported as file:line with a brief note",
            "States explicitly if nothing relevant was found",
        )
        # Delegate to the inherited dispatch path. context='workspace' is the
        # default explore context (kernel facts, no hot tool digests).
        return await super().execute(
            role="explore",
            prompt=prompt,
            success_criteria=success_criteria,
            context="workspace",
        )


class BugFinderRunTool(_CoderTool):
    """Kick off an autonomous bug-finder audit against the current workspace.

    The bug-finder pipeline is an eight-stage agentic loop (plan →
    detect → verify → fix) that the lead model invokes when it wants a
    second-opinion security/correctness audit on the workspace. Each
    role runs as its own bounded subagent. The full design lives in
    ``docs/superpowers/specs/2026-05-10-bug-finder-mode-design.md``.

    This tool ENQUEUES the run — it returns immediately with a
    ``run_id`` and ``job_id`` while the audit executes in the
    background. Use :class:`BugFinderStatusTool` (``bug_finder_status``)
    to poll for results. Runs typically take 5-30 minutes depending on
    repo size + chunk count.
    """

    def __init__(
        self,
        *,
        container_manager: ContainerManager,
        workspace_id: str,
        state: CoderState,
        jobs_store,
        bug_finder_store,
        verifier_model_resolver,
        profile_store=None,
        service_store=None,
        user_id: str = "",
        strict_edit_guard: bool = True,
    ) -> None:
        super().__init__(
            container_manager=container_manager,
            workspace_id=workspace_id,
            state=state,
            profile_store=profile_store,
            service_store=service_store,
            user_id=user_id,
            strict_edit_guard=strict_edit_guard,
        )
        self._jobs_store = jobs_store
        self._bug_finder_store = bug_finder_store
        self._verifier_model_resolver = verifier_model_resolver

    @property
    def name(self) -> str:
        return "bug_finder_run"

    @property
    def description(self) -> str:
        return (
            "Kick off an autonomous bug-finder audit on the current "
            "workspace. Use when the user asks for a security/correctness "
            "review or before shipping a substantial change. Returns a "
            "run_id immediately; poll with `bug_finder_status` to read "
            "the report.\n\n"
            "The pipeline is expensive (5-30 minutes, thousands of tokens) "
            "and runs in its own context — your conversation isn't "
            "blocked while it executes. Pass `focus_paths` to scope the "
            "scan to specific directories and `threat_model` (markdown) "
            "to bias detector + verifier toward the codebase's real risk "
            "surface.\n\n"
            "Do NOT use for: single-file review (use `task_dispatch` "
            "with role=review or role=security_review instead); generic "
            "lint/format/typecheck (run those directly); your own "
            "code-comprehension reads (use file_read)."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "primary_model": {
                    "type": "string",
                    "description": (
                        "OPTIONAL. Model id for planner / detector / fixer. "
                        "Leave EMPTY to use the user's configured "
                        "heavyweight model (falling back to their coder "
                        "model) — that is the default and is almost always "
                        "correct. Only set this if the user explicitly "
                        "named a different model that is actually installed "
                        "locally or served by a connected peer. Do NOT "
                        "invent or guess a model id from memory (e.g. a "
                        "cloud model name) — only locally-served models "
                        "work, and a wrong id silently dead-ends the whole "
                        "run."
                    ),
                },
                "verifier_model": {
                    "type": "string",
                    "description": (
                        "Optional override for the verifier role. Empty "
                        "string = single-model self-verification "
                        "(default; see RoleModelConfig docstring for the "
                        "correlated-error rationale)."
                    ),
                },
                "focus_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Restrict the planner to these workspace-relative "
                        "paths. Empty = survey the whole repo. Strongly "
                        "recommended for large repos."
                    ),
                },
                "threat_model": {
                    "type": "string",
                    "description": (
                        "Optional markdown threat model. Prepended to "
                        "detector + verifier prompts. Sections that help: "
                        "Assets, Trust boundaries, Attacker capabilities, "
                        "In scope, Out of scope."
                    ),
                },
                "force_below_minimum": {
                    "type": "boolean",
                    "description": (
                        "Bypass the capability gate. Use only when you "
                        "know the primary model can produce the strict "
                        "JSON output the parsers require."
                    ),
                },
            },
            "required": [],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        primary_model: str = "",
        verifier_model: str = "",
        focus_paths: list | None = None,
        threat_model: str = "",
        force_below_minimum: bool = False,
        **_kwargs,
    ) -> ToolResult:
        import uuid as _uuid

        from augmentum.bug_finder.capability import (
            capability_floor_label,
            is_capable,
        )

        primary_model = (primary_model or "").strip()
        if not primary_model:
            # Default server-side to a real, locally-served model instead
            # of forcing the calling LLM to name one. The #1 cause of dead
            # bug-finder runs was the coder agent inventing a cloud id from
            # memory (e.g. ``claude-sonnet-4-...``) that no local backend or
            # peer serves — it passes the family-name capability gate, then
            # fails at backend resolution deep inside the job. Prefer the
            # heavyweight slot (strongest model = best audit); fall back to
            # the user's configured coder model when it's unset.
            try:
                from augmentum.config import settings as _settings
                primary_model = (
                    (getattr(_settings, "heavyweight_model", "") or "").strip()
                    or (getattr(_settings, "primary_chat_model", "") or "").strip()
                )
            except Exception:
                primary_model = ""
        if not primary_model:
            return ToolResult(
                success=False,
                validation_error=True,
                error=(
                    "bug_finder_run: no model given and no default "
                    "configured. Set a heavyweight or primary chat model in "
                    "Settings → Models, or pass an installed `primary_model`."
                ),
            )
        if not force_below_minimum and not is_capable(primary_model):
            return ToolResult(
                success=False,
                validation_error=True,
                error=(
                    f"primary_model '{primary_model}' is below the bug-"
                    "finder capability floor. The detector / verifier / "
                    "fixer prompts target capable instruction-followers; "
                    "below-floor models tend to produce malformed JSON "
                    "the parsers silently reject. Pick one of: "
                    f"{capability_floor_label()}. Or set "
                    "`force_below_minimum: true` if you're certain the "
                    "model is up to the task."
                ),
            )
        if self._jobs_store is None:
            return ToolResult(
                success=False,
                error=(
                    "bug_finder_run is unavailable: background job queue "
                    "not wired in this session."
                ),
            )

        # Verifier defaults: explicit override > per-workspace HVY slot
        # > global heavyweight_model setting > "" (single-model self-
        # verification). The resolver thunk owns the lookup so the tool
        # doesn't need direct DB access for the per-workspace slot.
        verifier = (verifier_model or "").strip()
        if not verifier and self._verifier_model_resolver is not None:
            try:
                verifier = await self._verifier_model_resolver(
                    self._workspace_id, self._user_id,
                )
            except Exception:
                verifier = ""

        run_id = f"bfr_{_uuid.uuid4().hex[:12]}"
        focus = [str(p).strip() for p in (focus_paths or []) if str(p).strip()]
        payload: dict[str, Any] = {
            "run_id": run_id,
            "workspace_id": self._workspace_id,
            "primary_model": primary_model,
            "verifier_model": verifier,
            "focus_paths": focus,
            "threat_model": (threat_model or "").strip(),
        }
        if force_below_minimum:
            payload["force_below_minimum"] = True

        try:
            job_id = await self._jobs_store.create(
                user_id=self._user_id,
                job_type="bug_finder_run",
                payload=payload,
                priority=5,
                max_attempts=1,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False,
                error=f"bug_finder_run enqueue failed: {exc}",
            )

        return ToolResult(
            success=True,
            output=(
                f"Bug-finder run enqueued.\n"
                f"  run_id: {run_id}\n"
                f"  job_id: {job_id}\n"
                f"  workspace: {self._workspace_id}\n"
                f"  primary_model: {primary_model}\n"
                + (f"  verifier_model: {verifier}\n" if verifier else "")
                + (f"  focus_paths: {focus}\n" if focus else "")
                + "\nThe audit is running in the background. Continue "
                "with other work and call `bug_finder_status` with this "
                "run_id when you want the report."
            ),
            metadata={
                "run_id": run_id,
                "job_id": job_id,
                "workspace_id": self._workspace_id,
                "primary_model": primary_model,
                "verifier_model": verifier,
                "focus_paths": focus,
            },
        )


class BugFinderStatusTool(_CoderTool):
    """Poll the status / final report of a previously-enqueued audit.

    Returns the run's current ``stop_reason`` (``running``, ``complete``,
    ``cancelled``, ``error``, ``wallclock``) and — when terminal — the
    full :class:`BugFinderRunReport` rendered as a compact markdown
    summary. Lead models should call this every few iterations after
    dispatching with :class:`BugFinderRunTool` rather than waiting in a
    tight loop.
    """

    def __init__(
        self,
        *,
        container_manager: ContainerManager,
        workspace_id: str,
        state: CoderState,
        bug_finder_store,
        profile_store=None,
        service_store=None,
        user_id: str = "",
        strict_edit_guard: bool = True,
    ) -> None:
        super().__init__(
            container_manager=container_manager,
            workspace_id=workspace_id,
            state=state,
            profile_store=profile_store,
            service_store=service_store,
            user_id=user_id,
            strict_edit_guard=strict_edit_guard,
        )
        self._bug_finder_store = bug_finder_store

    @property
    def name(self) -> str:
        return "bug_finder_status"

    @property
    def description(self) -> str:
        return (
            "Poll the status of a bug-finder run previously dispatched "
            "with `bug_finder_run`. Returns running/complete/error + the "
            "report summary when terminal. Cheap (single DB read); safe "
            "to call repeatedly."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "The run_id returned by bug_finder_run.",
                },
            },
            "required": ["run_id"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        run_id: str = "",
        **_kwargs,
    ) -> ToolResult:
        run_id = (run_id or "").strip()
        if not run_id:
            return ToolResult(
                success=False,
                validation_error=True,
                error="bug_finder_status requires `run_id`.",
            )
        if self._bug_finder_store is None:
            return ToolResult(
                success=False,
                error=(
                    "bug_finder_status is unavailable: store not wired "
                    "in this session."
                ),
            )
        row = await self._bug_finder_store.get_run(
            run_id, user_id=self._user_id,
        )
        if row is None:
            return ToolResult(
                success=False,
                error=f"run_id '{run_id}' not found.",
            )
        status = (row.get("stop_reason") or "running").strip() or "running"

        if status == "running":
            return ToolResult(
                success=True,
                output=(
                    f"Bug-finder run {run_id} is still running.\n"
                    f"  workspace: {row.get('workspace_id', '')}\n"
                    f"  started_at: {row.get('started_at', '')}\n"
                    "Check again later — runs typically take 5-30 minutes."
                ),
                metadata={"run_id": run_id, "status": "running"},
            )

        # Terminal: format the report for the lead model.
        total = int(row.get("findings_total") or 0)
        fixed = int(row.get("findings_fixed") or 0)
        confirmed = int(row.get("findings_confirmed") or 0)
        fix_failed = int(row.get("findings_fix_failed") or 0)
        report = row.get("report") or {}
        notes = list(report.get("notes") or [])
        same_model = bool(report.get("same_model_self_verification"))
        findings = list(report.get("findings") or [])

        # Top critical / high findings, capped — keep the tool output
        # compact even when the audit surfaced many issues.
        critical_high = [
            f for f in findings
            if f.get("severity") in ("critical", "high")
            and f.get("status") in ("confirmed", "fixed", "fix_failed")
        ]
        top_lines: list[str] = []
        for f in critical_high[:5]:
            sev = (f.get("severity") or "").upper()
            st = (f.get("status") or "").upper()
            top_lines.append(
                f"  - [{sev}/{st}] {f.get('file', '')}:{f.get('function', '')} "
                f"— {f.get('claim', '(no claim)')}"
            )
        top_block = (
            "\nTop critical/high findings:\n" + "\n".join(top_lines)
            if top_lines else ""
        )
        notes_block = (
            "\nNotes:\n" + "\n".join(f"  - {n}" for n in notes)
            if notes else ""
        )
        same_model_warn = (
            "\n[warning] Single-model self-verification — interpret "
            "findings with care (correlated detector/verifier errors)."
            if same_model else ""
        )

        return ToolResult(
            success=status == "complete",
            output=(
                f"Bug-finder run {run_id} finished: {status}.\n"
                f"  workspace: {row.get('workspace_id', '')}\n"
                f"  findings: {total} total — {fixed} fixed, "
                f"{confirmed} confirmed, {fix_failed} fix-failed.\n"
                f"  tokens: in={row.get('total_tokens_in', 0)} "
                f"out={row.get('total_tokens_out', 0)}, "
                f"wall={int(row.get('total_wallclock_ms') or 0) // 1000}s"
                + top_block + notes_block + same_model_warn
            ),
            error=(
                ""
                if status == "complete"
                else f"run stopped: {status} ({row.get('stop_detail') or 'no detail'})"
            ),
            metadata={
                "run_id": run_id,
                "status": status,
                "findings_total": total,
                "findings_fixed": fixed,
                "findings_confirmed": confirmed,
                "findings_fix_failed": fix_failed,
                "same_model_self_verification": same_model,
            },
        )


class ThinkTool(_CoderTool):
    """Elective reasoning scratchpad (Anthropic think-tool pattern).

    A no-op tool the model calls when it wants to plan or reason before
    acting — the middle ground between forcing per-turn native thinking ON
    (reasons before every tool call: too much) and OFF (no planning at
    all). It touches nothing: no file, no shell, no workspace state. The
    model's reasoning lives in the ``thought`` argument; execute() just
    returns a light acknowledgement so the loop continues. Exposure is
    gated in ``phase_act._act_native`` — only when native per-turn thinking
    is OFF for the turn, and behind the ``coder_think_tool_enabled`` flag.
    """

    @property
    def name(self) -> str:
        return "think"

    @property
    def description(self) -> str:
        return (
            "Think through a problem before acting — internal working notes, "
            "NOT code execution or file edits (nothing runs, nothing changes). "
            "Use it at decision points: to plan a multi-step change, reason "
            "about a surprising or failing tool result, choose between "
            "approaches, or reconsider when you're stuck. Write ONE focused "
            "thought, then take the next concrete action — do not narrate "
            "routine steps and do not call this repeatedly in a row."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": (
                        "Your reasoning: what you're trying to do, the options "
                        "you're weighing, and what you'll do next."
                    ),
                },
            },
            "required": ["thought"],
        }

    async def execute(self, *, thought: str = "", **_kwargs) -> ToolResult:
        # Pure no-op: the value is the model's reasoning in `thought`. We
        # only acknowledge so the agentic loop continues. The light
        # "take the next action" nudge discourages think-spirals (repeated
        # think calls that produce no work).
        return ToolResult(
            success=True,
            output="Noted. Now take the next concrete action.",
        )


class FindSymbolTool(_CoderTool):
    """Symbol-definition lookup against the per-workspace code-intel index."""

    @property
    def name(self) -> str:
        return "find_symbol"

    @property
    def description(self) -> str:
        return (
            "Find where a symbol (function/class/method/const) is DEFINED, "
            "in one hop — no grep chains. Accepts bare names ('search_index') "
            "or qualified ('PackManager.search'). Falls back to fuzzy "
            "substring match when nothing matches exactly. For CALL SITES "
            "use code_grep. Builds the index on first use if missing."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Symbol name, optionally 'Class.method' qualified",
                },
                "kind": {
                    "type": "string",
                    "enum": ["class", "function", "method", "const", "type"],
                    "description": "Optional filter by symbol kind",
                },
            },
            "required": ["name"],
        }

    async def execute(self, *, name: str = "", kind: str = "", **_kw) -> ToolResult:
        from augmentum.config import settings as _settings
        if not getattr(_settings, "coder_code_intel_enabled", True):
            return ToolResult(success=False, error="code intelligence is disabled")
        if not name or not name.strip():
            return ToolResult(success=False, error="name is required", validation_error=True)
        from augmentum.coder import code_intel
        try:
            if not await code_intel.has_index(self._workspace_id):
                await code_intel.build_code_intel(self._cm, self._workspace_id)
            results = await code_intel.find_symbol(
                self._workspace_id, name.strip(), kind=(kind or None),
            )
        except Exception as exc:
            log.warning("find_symbol_failed", name=name, error=str(exc))
            return ToolResult(success=False, error=str(exc))
        if not results:
            return ToolResult(
                success=True,
                output=(
                    f"No definition found for '{name}' in the index. It may be "
                    "defined dynamically or in an unindexed language — try "
                    "code_grep as a fallback."
                ),
                metadata={"matches": 0},
            )
        lines = []
        if not results[0].get("exact", True):
            lines.append(f"No exact match for '{name}' — closest names:")
        for r in results:
            qual = f"{r['scope']}." if r["scope"] else ""
            lines.append(
                f"{r['path']}:{r['line']} [{r['kind']}] {qual}{r['name']}{r['signature']}"
            )
        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"matches": len(results), "exact": results[0].get("exact", True)},
        )


class FileOutlineTool(_CoderTool):
    """Structural outline (symbols + imports) for files, batched."""

    @property
    def name(self) -> str:
        return "file_outline"

    @property
    def description(self) -> str:
        return (
            "Get the structural outline of files WITHOUT reading them: "
            "classes/functions/methods with line ranges, plus imports. "
            "Batch multiple paths in one call. Use before file_read to "
            "jump straight to the right line range."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File paths to outline (batch them)",
                },
            },
            "required": ["paths"],
        }

    async def execute(self, *, paths: list | None = None, **_kw) -> ToolResult:
        from augmentum.config import settings as _settings
        if not getattr(_settings, "coder_code_intel_enabled", True):
            return ToolResult(success=False, error="code intelligence is disabled")
        if not paths:
            return ToolResult(success=False, error="paths is required", validation_error=True)
        from augmentum.coder import code_intel
        blocks: list[str] = []
        for path in [str(p) for p in paths][:20]:
            try:
                outline = await code_intel.file_outline(self._workspace_id, path)
                if outline is None:
                    # New/unindexed file — targeted reindex, retry once.
                    await code_intel.reindex_paths(self._cm, self._workspace_id, [path])
                    outline = await code_intel.file_outline(self._workspace_id, path)
            except Exception as exc:
                log.warning("file_outline_failed", path=path, error=str(exc))
                blocks.append(f"{path}: outline failed ({exc})")
                continue
            if outline is None:
                blocks.append(f"{path}: not in index (does the file exist?)")
                continue
            lines = [f"{outline['path']} ({outline['lang'] or 'no symbols'}, {outline['size']}b)"]
            if outline["imports"]:
                mods = sorted({i["module"] for i in outline["imports"]})
                lines.append("  imports: " + ", ".join(mods[:20]))
            for s in outline["symbols"]:
                qual = f"{s['scope']}." if s["scope"] else ""
                span = f"{s['line']}-{s['end_line']}" if s["end_line"] else str(s["line"])
                lines.append(f"  {span}: [{s['kind']}] {qual}{s['name']}{s['signature']}")
            if not outline["symbols"]:
                lines.append("  (no symbols extracted)")
            blocks.append("\n".join(lines))
        return ToolResult(success=True, output="\n\n".join(blocks), metadata={"files": len(blocks)})


ALL_CODER_TOOLS: list[type[_CoderTool]] = [
    ThinkTool,
    FileReadTool,
    FileWriteTool,
    FileListTool,
    DirTreeTool,
    CodeEditTool,
    CodeMultiEditTool,
    ApplyPatchTool,
    CodeGrepTool,
    CodeGlobTool,
    CodeSearchTool,
    FindSymbolTool,
    FileOutlineTool,
    DocSearchTool,
    DocFetchTool,
    PackSearchTool,
    ShellExecTool,
    ShellReadTool,
    TermOpenTool,
    TermSendTool,
    TermSnapshotTool,
    TermListTool,
    TermCloseTool,
    GitTool,
    TestRunTool,
    BlenderRunTool,
    EnvInfoTool,
    ContainerInfoTool,
    HttpRequestTool,
    DbInspectTool,
    PublishPortsTool,
    ServiceStartTool,
    ServiceListTool,
    ServiceLogsTool,
    ServiceStopTool,
    ServiceProbeTool,
    BrowserOpenTool,
    BrowserSnapshotTool,
    BrowserClickTool,
    BrowserTypeTool,
    BrowserScreenshotTool,
    BrowserEvaluateTool,
    BrowserVerifyTool,
    BrowserWaitTool,
    BrowserExtractTool,
    BrowserFillFormTool,
    # Sidecar-native verbs (persistent browser; clear error when the
    # compose.browser.yaml overlay isn't running)
    BrowserInteractTool,
    BrowserNavigateTool,
    BrowserGetTool,
    BrowserConsoleTool,
    BrowserTabsTool,
    BrowserFindTool,
    ProfileReadTool,
    ProfileUpdateTool,
    ObserveTool,
    TaskListTool,
    FinishTaskTool,
    AskUserTool,
]
# TaskDispatchTool is constructed separately by create_coder_tools when
# a dispatcher is supplied; it isn't in ALL_CODER_TOOLS because it
# requires a non-standard constructor argument (the SubagentDispatcher).


def _parse_allowlist(raw: str) -> list[tuple[str, str]] | None:
    """Parse the AUGMENTUM_CODER_MCP_ALLOWLIST env value into [(server, tool), …].

    Each comma-separated entry is of the form ``server/tool``, where
    either component may be ``*`` to match any value. Returns ``None``
    for empty input (meaning: allow everything — no filter).
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    patterns: list[tuple[str, str]] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "/" in entry:
            server, tool = entry.split("/", 1)
        else:
            server, tool = entry, "*"
        patterns.append((server or "*", tool or "*"))
    return patterns or None


def _allowlist_matches(patterns: list[tuple[str, str]], full_name: str) -> bool:
    """True if ``full_name`` ("server/tool") matches any pattern."""
    if "/" not in full_name:
        return False
    server, tool = full_name.split("/", 1)
    for p_server, p_tool in patterns:
        if (p_server in ("*", server)) and (p_tool in ("*", tool)):
            return True
    return False


def create_coder_tools(
    container_manager: ContainerManager | None,
    workspace_id: str,
    state: CoderState,
    *,
    executor: WorkspaceExecutor | None = None,
    tool_registry=None,
    question_callback=None,
    profile_store=None,
    service_store=None,
    user_id: str = "",
    strict_edit_guard: bool = True,
    planning_mode: str = "default",
    subagent_dispatcher=None,
    db_conn=None,
    jobs_store=None,
) -> list[Tool]:
    """Instantiate all built-in coder tools plus any MCP tools registered.

    Parameters
    ----------
    container_manager:
        The ContainerManager that wraps the Docker API.
    workspace_id:
        The workspace container to operate on.
    state:
        The CoderState for this agent session.
    tool_registry:
        Optional ``ToolRegistry`` whose MCP-backed tools (instances of
        ``MCPToolWrapper``) should be folded into the returned list. Pass
        ``None`` to skip MCP discovery (useful for tests and environments
        without MCP servers configured).

    Returns
    -------
    list[Tool]
        Built-in coder tools in canonical order, followed by MCP tools
        from the registry (filtered by ``AUGMENTUM_CODER_MCP_ALLOWLIST``
        if that env var is set). MCP tool names are namespaced
        ``{server_name}/{tool_name}`` so they cannot collide with
        built-ins.

    Allowlist
    ---------
    Set ``AUGMENTUM_CODER_MCP_ALLOWLIST`` to a comma-separated list of
    ``server/tool`` patterns to restrict which MCP tools the coder agent
    can see. ``*`` is a wildcard in either component. Example:
    ``github/*,linear/create_issue`` allows every github tool plus one
    specific linear tool, blocking everything else. Leave unset to
    expose every registered MCP tool (current default).
    """
    import os

    def _construct(cls):
        # AskUserTool needs the handler's question callback; every other
        # tool takes the standard _CoderTool kwargs. ``executor`` is passed
        # ONLY when injected (remote/editor mode) — when it's None the kwarg
        # is omitted entirely so the container path is byte-identical to
        # before (tools default to wrapping container_manager themselves).
        kw = dict(
            container_manager=container_manager,
            workspace_id=workspace_id,
            state=state,
            profile_store=profile_store,
            service_store=service_store,
            user_id=user_id,
            strict_edit_guard=strict_edit_guard,
        )
        if executor is not None:
            kw["executor"] = executor
        if cls is AskUserTool:
            kw["question_callback"] = question_callback
        return cls(**kw)

    # In the ACP "loop in the editor" path there is no Docker container:
    # container_manager is None and workspace I/O rides an injected
    # WorkspaceExecutor. Runtime tools that reach ``self._cm`` directly
    # (service/browser/terminal/http/db) can't work there and would crash on
    # a None access — so don't advertise them. File/shell/git/etc. route
    # through the executor and stay available. See _RuntimeCoderTool.
    _no_container = container_manager is None
    _registry = [
        cls for cls in ALL_CODER_TOOLS
        if not (_no_container and getattr(cls, "requires_container", False))
    ]
    tools: list[Tool] = [_construct(cls) for cls in _registry]

    # Subagent dispatch — only present when the handler wired a
    # dispatcher (which only happens when coder_subagents_enabled).
    # Construction is separate because TaskDispatchTool needs a non-
    # standard kwarg (the dispatcher) that the rest of the registry
    # doesn't know about.
    if subagent_dispatcher is not None:
        _dispatch_kwargs = {
            "container_manager": container_manager,
            "workspace_id": workspace_id,
            "state": state,
            "dispatcher": subagent_dispatcher,
            "profile_store": profile_store,
            "service_store": service_store,
            "user_id": user_id,
            "strict_edit_guard": strict_edit_guard,
        }
        tools.append(TaskDispatchTool(**_dispatch_kwargs))
        # Concrete-surface alias for explore — the abstract task_dispatch
        # meta-tool is reliably ignored by local models; explore_codebase
        # is the same subagent dressed as an in-distribution verb they DO
        # call (see ExploreCodebaseTool). Additive: both are offered.
        tools.append(ExploreCodebaseTool(**_dispatch_kwargs))

    # Bug Finder dispatch tools — only when a job queue AND a sqlite
    # connection are both wired. The pair always lands together (status
    # reads are useless without dispatch); skipping in any other
    # configuration keeps the surface contract honest. Verifier-model
    # resolution looks at per-workspace override first, falls back to
    # the global heavyweight_model setting.
    if jobs_store is not None and db_conn is not None:
        from augmentum.bug_finder.store import BugFinderRunStore
        bf_store = BugFinderRunStore(db_conn)

        async def _resolve_verifier(ws_id: str, uid: str) -> str:
            try:
                async with db_conn.execute(
                    "SELECT bug_finder_verifier_model FROM project_checkouts "
                    "WHERE id = ? AND user_id = ?",
                    (ws_id, uid),
                ) as cursor:
                    row = await cursor.fetchone()
                if row and row[0]:
                    return str(row[0]).strip()
            except Exception:
                pass
            try:
                from augmentum.config import settings as _settings
                return (getattr(_settings, "heavyweight_model", "") or "").strip()
            except Exception:
                return ""

        tools.append(BugFinderRunTool(
            container_manager=container_manager,
            workspace_id=workspace_id,
            state=state,
            jobs_store=jobs_store,
            bug_finder_store=bf_store,
            verifier_model_resolver=_resolve_verifier,
            profile_store=profile_store,
            service_store=service_store,
            user_id=user_id,
            strict_edit_guard=strict_edit_guard,
        ))
        tools.append(BugFinderStatusTool(
            container_manager=container_manager,
            workspace_id=workspace_id,
            state=state,
            bug_finder_store=bf_store,
            profile_store=profile_store,
            service_store=service_store,
            user_id=user_id,
            strict_edit_guard=strict_edit_guard,
        ))

    # Phase 2 LTM recall tools — only registered when the SQLite conn
    # was wired through. Callsites that don't pass db_conn (tests,
    # legacy spawn paths, bug_finder) skip the recall surface and
    # the agent operates as before without it. See
    # [[project_coder_turn_archive]] for the full design.
    if db_conn is not None:
        from augmentum.coder.turn_archive_tools import (
            RecallExpandTool,
            RecallTool,
        )
        recall_kwargs = dict(
            container_manager=container_manager,
            workspace_id=workspace_id,
            state=state,
            profile_store=profile_store,
            service_store=service_store,
            user_id=user_id,
            strict_edit_guard=strict_edit_guard,
        )
        tools.append(RecallTool(db_conn=db_conn, **recall_kwargs))
        tools.append(RecallExpandTool(db_conn=db_conn, **recall_kwargs))

    if tool_registry is not None:
        allowlist = _parse_allowlist(os.environ.get("AUGMENTUM_CODER_MCP_ALLOWLIST", ""))
        try:
            from augmentum.mcp.bridge import MCPToolWrapper
            registry_tools = getattr(tool_registry, "_tools", None)
            if registry_tools:
                for t in registry_tools.values():
                    if not isinstance(t, MCPToolWrapper):
                        continue
                    if allowlist is not None and not _allowlist_matches(allowlist, t.name):
                        continue
                    tools.append(t)
        except ImportError:
            # MCP module missing — silently skip so tests without MCP still work.
            log.debug("mcp_bridge_unavailable_skipping_discovery")
        except Exception as exc:
            log.warning("mcp_tool_discovery_failed", error=str(exc))

    # Plan-mode filter: drop write/shell/exec tools so the model can
    # only explore, never mutate. The system prompt nudges "propose,
    # don't edit" but the hard floor here makes that an enforced
    # invariant — Plan mode IS read-only by tool-list construction,
    # not by the model's discretion. Cycle is in
    # migrations/207_coder_workspaces_planning_mode.sql.
    # Plan-mode used to hard-filter write/shell/MCP tools at
    # construction. Retired post-migration 208 in favor of soft
    # guidance: the model has every tool available and is nudged by
    # ``_plan_mode_addendum`` in the system prompt to propose-first.
    # The shift matches how mature agents (Claude Code, Cursor
    # Composer) treat planning — natural collaboration over enforced
    # restriction — and aligns with the broader "trust the model"
    # default. Operators who genuinely need read-only enforcement
    # should write a deny rule in .augmentum/permissions.toml; that
    # path stays bulletproof at the permission layer.
    return tools
