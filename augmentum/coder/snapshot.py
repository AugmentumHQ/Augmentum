"""WorkspaceSnapshot — programmatic, self-refreshing workspace tree.

Single source of truth for "what files exist in /workspace right now."
Injected into the coder loop's system prompt + observation refresh cycle
so the model never has to run ``dir_tree`` / ``ls`` just to find out where
files live — especially relevant after it creates / deletes / moves files
mid-turn.

Behavioural contract
--------------------
* Lazy: the container ``find`` runs only when ``refresh_if_stale`` is
  invoked AND the snapshot is marked stale. Callers should ``mark_stale``
  after any tool call that mutates the workspace (``file_write``,
  ``code_edit``, ``code_multi_edit``, successful ``shell_exec``).
* Query-agnostic: unlike ``build_repo_map``, the snapshot does not
  re-rank files against the user's current prompt. The same file set
  shows up every turn so the model builds a stable mental model of the
  repo instead of watching files flicker in and out of view across turns.
* Cheap render: flat sorted list with ``(Nlines)`` suffix. Large repos
  (>``max_files``) emit a "showing first N of M" footer directing the
  model to ``dir_tree`` / ``find_files`` if it needs the rest.
* Delta markers: the previous snapshot is retained so ``render()`` can
  emit ``[NEW]`` on files that appeared since the last refresh, ``[DEL]``
  on disappeared ones. Line-count jumps of ±20% emit ``[MOD]``.

This module deliberately has no awareness of ``CoderState`` — it's keyed
by ``workspace_id`` and owns its own lifecycle. That keeps it safe to
use from both the handler (hot path) and stateless helpers (tests).
"""
from __future__ import annotations

import re
import time

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Directories to prune from the scan. Mirrors ``repomap._SKIP_DIRS`` so
# the two views of the workspace stay consistent — a file hidden in one
# must be hidden in the other, or the model will see "find says it
# doesn't exist, repomap says it does."
_SKIP_DIRS = (
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".nuxt", "target", "vendor",
    ".tox", ".mypy_cache", ".pytest_cache", "coverage",
    ".cargo", "pkg", "bin", "obj",
)

# File extensions worth listing. Matches the coder's likely edit targets;
# binaries / assets / cache files go uncounted even if the ``find`` pass
# walks over them. Conservative by default — a project that needs an
# exotic extension can ``dir_tree`` on demand.
_INCLUDE_EXTENSIONS = (
    "py", "js", "ts", "jsx", "tsx", "rs", "go", "rb",
    "java", "c", "cpp", "h", "hpp", "cs", "php", "swift", "kt",
    "sh", "bash", "zsh", "yaml", "yml", "toml", "json", "lock",
    "html", "css", "scss", "sass", "less", "sql", "md", "mdx",
    "txt", "ini", "cfg", "conf", "env", "proto", "graphql",
    "vue", "svelte", "astro",
)

# Fast-detectable dotfiles worth surfacing even though they have no
# extension. These drive tooling behaviour and the model should see them.
_INCLUDE_DOTFILES = (
    ".gitignore", ".dockerignore", ".eslintrc", ".prettierrc",
    ".editorconfig", ".env.example",
    "Dockerfile", "Makefile", "LICENSE", "README", "CHANGELOG",
)


class WorkspaceSnapshot:
    """Lazy, delta-tracked snapshot of a container's workspace tree.

    Lifecycle::

        snap = WorkspaceSnapshot(container_manager, workspace_id)
        await snap.refresh_if_stale(force=True)   # initial build
        # ... agent iterates, mutates files ...
        snap.mark_stale()                           # after file_write
        await snap.refresh_if_stale()               # next render
        prompt_block = snap.render()                # inject into context
    """

    __slots__ = (
        "_cm", "_workspace_id", "_max_files",
        "_current", "_previous",
        "_stale", "_last_refresh_at", "_refresh_count",
        "_total_found",
    )

    def __init__(
        self,
        container_manager,
        workspace_id: str,
        max_files: int = 250,
    ) -> None:
        self._cm = container_manager
        self._workspace_id = workspace_id
        self._max_files = max_files
        self._current: dict[str, int] | None = None     # path → line_count
        self._previous: dict[str, int] | None = None
        self._stale: bool = True
        self._last_refresh_at: float = 0.0
        self._refresh_count: int = 0
        # Total files the scan saw (before the ``max_files`` clip). Lets
        # ``render()`` tell the model "showing 250 of 1834 — use dir_tree
        # to look elsewhere."
        self._total_found: int = 0

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    def is_stale(self) -> bool:
        return self._stale

    def mark_stale(self) -> None:
        """Flip the staleness bit — next ``refresh_if_stale`` will re-scan.

        Cheap (just sets a flag). Callers should invoke this after any
        tool that *might* have mutated the workspace — it's fine to over-
        call; the scan is still lazy.
        """
        self._stale = True

    @property
    def refresh_count(self) -> int:
        return self._refresh_count

    @property
    def current_paths(self) -> set[str]:
        """Current workspace-relative file set from the last refresh."""
        return set(self._current or {})

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    async def refresh_if_stale(self, *, force: bool = False) -> bool:
        """Re-scan the container if ``_stale`` (or ``force=True``).

        Returns True when a refresh actually ran, False otherwise. A
        failure to scan (no container, shell error) leaves the existing
        snapshot in place and logs at debug; the model still sees the
        last-known-good tree, which is better than an empty block.
        """
        if not self._stale and not force:
            return False
        if self._cm is None:
            return False

        try:
            rows = await self._scan()
        except Exception:
            # Stale tree gets injected into the next prompt with no
            # signal to the model that it's stale. Worth a warning so
            # a recurring scan failure (container down, find unavailable)
            # is findable rather than masked by the keep-last-known-good
            # fallback.
            log.warning("workspace_snapshot.scan_failed", exc_info=True)
            return False

        # Rotate previous → current → new.
        self._previous = self._current
        self._total_found = len(rows)
        # Apply the clip here, not in the scan, so the previous-snapshot
        # diff still sees the full file set for the portion that stayed
        # under the cap.
        self._current = dict(rows[: self._max_files])
        self._stale = False
        self._last_refresh_at = time.time()
        self._refresh_count += 1
        return True

    async def _scan(self) -> list[tuple[str, int]]:
        """Run ``find`` + ``wc -l`` in the container and parse the output.

        Returns a list of ``(workspace-relative path, line_count)``
        sorted by path so downstream renders are stable. Paths are
        always workspace-relative — the ``/workspace/`` prefix is
        stripped so the tree reads naturally.
        """
        # Build the prune + include filters. The ``\\( ... \\)`` grouping
        # escapes through bash -c cleanly.
        prune = " ".join(f"-path '*/{d}' -prune -o" for d in _SKIP_DIRS)
        ext_filter = " -o ".join(
            f"-name '*.{e}'" for e in _INCLUDE_EXTENSIONS
        )
        dotfile_filter = " -o ".join(
            f"-name '{d}'" for d in _INCLUDE_DOTFILES
        )
        cmd = (
            f"find /workspace {prune} "
            f"\\( {ext_filter} -o {dotfile_filter} \\) -type f -print "
            # xargs -r: don't run wc if input is empty (handles empty repo)
            f"2>/dev/null | sort | xargs -r wc -l 2>/dev/null"
        )

        output = await self._cm._run_command(
            self._workspace_id, ["bash", "-c", cmd], timeout=15.0,
        )

        rows: list[tuple[str, int]] = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            # ``wc -l`` emits a trailing "total N" line when multiple
            # files are given. Skip it; it's not a file.
            if line.endswith(" total"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            count_raw, path = parts
            try:
                count = int(count_raw)
            except ValueError:
                continue
            # Strip the workspace prefix; downstream renders don't need it.
            if path.startswith("/workspace/"):
                rel = path[len("/workspace/"):]
            elif path == "/workspace":
                continue
            else:
                rel = path
            if not rel:
                continue
            rows.append((rel, count))

        # Sort by path — deterministic rendering, and the directory
        # prefix ordering makes the flat list read like a tree.
        rows.sort(key=lambda r: r[0])
        return rows

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, *, with_header: bool = True) -> str:
        """Render the current snapshot as a ``<workspace_tree>`` block.

        When the workspace is empty or unreachable, returns an explicit
        "(empty)" block rather than ``""`` — without this, the model
        gets no signal that /workspace has nothing in it and tends to
        confabulate a project from its training data (observed: DTLN,
        random HF repos). With ``with_header=False`` the tags are
        omitted — useful when the caller wants to embed the body in a
        larger context block.
        """
        if self._current is None or not self._current:
            body = (
                "(empty — /workspace contains no files yet. "
                "Do not assume any project exists here; ask the user "
                "what they want to build, or wait for them to add files.)"
            )
            if not with_header:
                return body
            return (
                "<workspace_tree files=\"0\" total=\"0\" "
                f"refresh=\"{self._refresh_count}\">\n{body}\n"
                "</workspace_tree>"
            )

        delta_new, delta_del, delta_mod = self._compute_delta()
        body_lines: list[str] = []

        # Flat sorted format: ``path (N L) [MARKER]``. Padding to a
        # fixed column keeps the markers visually aligned so the model
        # can scan for changes quickly. The cap of 120 chars accounts
        # for deep nested paths (e.g. ui/scripts/surfaces/narrative-
        # surface.js is 38 chars, augmentum/state/migrations/... can hit
        # 70+).
        for path in sorted(self._current.keys()):
            count = self._current[path]
            if path in delta_new:
                marker = " [NEW]"
            elif path in delta_mod:
                marker = " [MOD]"
            else:
                marker = ""
            body_lines.append(f"{path} ({count}L){marker}")

        # Deletions appear at the tail so the model sees what's gone.
        # These won't be in ``_current`` (by definition), so we append
        # them as a separate section rather than trying to interleave.
        if delta_del:
            body_lines.append("")
            body_lines.append("# Deleted since last refresh:")
            for path in sorted(delta_del):
                body_lines.append(f"{path} [DEL]")

        body = "\n".join(body_lines)

        truncated = self._total_found > len(self._current)
        footer = ""
        if truncated:
            footer = (
                f"\n\n[Showing {len(self._current)} of {self._total_found} "
                f"files. Use `dir_tree` or `find_files` with a narrower "
                f"path if you need the rest.]"
            )

        if not with_header:
            return body + footer

        # The explanatory preamble tells a weaker model explicitly what
        # this block is for — observed to drop the "should I run
        # dir_tree to check?" pattern on mid-tier models.
        return (
            f"<workspace_tree files=\"{len(self._current)}\" "
            f"total=\"{self._total_found}\" refresh=\"{self._refresh_count}\">\n"
            "Auto-refreshed view of /workspace. Files list is authoritative "
            "for \"what exists right now\" — you don't need to run dir_tree "
            "or ls to discover files. [NEW]/[MOD]/[DEL] markers show "
            "changes since the last refresh.\n\n"
            f"{body}{footer}\n"
            "</workspace_tree>"
        )

    def _compute_delta(self) -> tuple[set[str], set[str], set[str]]:
        """Return (new, deleted, modified) path sets since ``_previous``.

        ``modified`` is defined as "line count differs by more than 20 %
        or more than 20 lines" — small edits don't trigger the marker
        because the model shouldn't flag a 350 → 352 line file as
        interesting. Both thresholds matter: 20 % handles small files
        (50 → 60 is an interesting delta), the absolute 20-line minimum
        handles big files (1000 → 1005 isn't).
        """
        if self._previous is None:
            return set(), set(), set()

        curr = self._current or {}
        prev = self._previous

        curr_paths = set(curr.keys())
        prev_paths = set(prev.keys())

        new = curr_paths - prev_paths
        deleted = prev_paths - curr_paths
        modified: set[str] = set()
        for path in curr_paths & prev_paths:
            c = curr[path]
            p = prev[path]
            diff = abs(c - p)
            if diff >= 20 or (p > 0 and diff / p >= 0.20):
                modified.add(path)
        return new, deleted, modified


# Regex used by tests to extract ``count="N"`` from rendered output.
# Kept at module level so tests don't re-compile per call.
_FILES_ATTR_RE = re.compile(r'files="(\d+)"')
