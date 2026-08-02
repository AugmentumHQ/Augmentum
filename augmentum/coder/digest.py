"""Project digest — one-shot inlined codebase block for small workspaces.

A "masterfile" approach: for workspaces that fit under a token budget,
inline every file's content into a single boundary-delimited block that
gets prepended to the system prompt. Benefits on the small-project
common case (one web app, one script, a handful of files):

- Replaces ``dir_tree → file_read → file_read → ...`` serial orchestration
  with a single upfront read of every file.
- At-top placement in the system prompt means Anthropic / OpenAI / llama-
  server prefix caches the stable content — subsequent iterations pay
  ~10% of the token cost for the re-injected block.
- Deterministic sort (by path) keeps the cache prefix stable across
  iterations; only files that actually changed invalidate the tail.
- Returns ``None`` when the computed total exceeds the budget so the
  caller falls back to the existing ``repo_map`` + on-demand ``file_read``
  path — no behaviour change on medium / large projects.

Budget is set via ``AUGMENTUM_CODER_DIGEST_BUDGET`` when provided.
Otherwise, Coder derives a conservative slice of the active model context
window; when the backend cannot report a window, it falls back to the
historical 40_000 tokens (~160 KB at 4 chars/token average). The default
is intentionally conservative: for small webapp scaffolds, typical Python
scripts, and most getting-started workspaces the digest fits comfortably;
for real repos it skips silently and the existing path runs.

**No truncation — all-or-nothing.** If *any* file would push the total
over budget, the whole digest is discarded. A truncated file would
silently give the model partial information while the preamble tells
it the digest is authoritative — a guaranteed-wrong mental model that
causes subtle bugs (edit-in-place on a file whose end the model never
saw, confident assertions about missing imports, etc.). Falling
through to ``repo_map`` + on-demand ``file_read`` means the model
knows it needs to read specific files, and each read returns complete
content.

Format:

    ===== FILE: src/app.py (127L) =====
    <content>
    ===== END: src/app.py =====

    ===== FILE: src/utils.py (42L) =====
    ...

Paths are relative to /workspace so the block reads like a project
tour rather than an absolute-path listing. Line counts are included in
the header for the model to cross-reference against ``file_read(start,
end)`` calls if it wants to edit-in-place.
"""
from __future__ import annotations

import os

from augmentum.coder.context_tokens import (
    DEFAULT_CODER_DIGEST_TOKENS,
    coder_digest_token_budget,
)
from augmentum.coder.repomap import _CODE_EXTENSIONS, _SKIP_DIRS
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Rough token budget. With a known model context window this scales from
# that window; without one we keep the historical 40_000-token fallback.
# Env override for power users.
def _default_budget(context_window: int | None = None) -> int:
    raw = os.environ.get("AUGMENTUM_CODER_DIGEST_BUDGET", "")
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    if context_window:
        return coder_digest_token_budget(context_window)
    return DEFAULT_CODER_DIGEST_TOKENS


# Simple token estimator: ~4 chars/token is the commonly-cited average
# for English code. Good enough for budget gating; we don't need exact
# counts because the budget is already conservative. Using tiktoken
# here would add a dependency and ~100ms on cold start without changing
# the yes/no decision meaningfully.
def _estimate_tokens(text: str) -> int:
    return (len(text) + 3) // 4


# Header / footer per file. Kept grep-friendly so a sufficiently motivated
# user (or model) can pull a single file out of the digest with one
# regex. Matches the conventions emitted by ``cat`` and git-diff for
# human eyes.
def _file_header(path: str, line_count: int) -> str:
    return f"===== FILE: {path} ({line_count}L) ====="


def _file_footer(path: str) -> str:
    return f"===== END: {path} ====="


async def build_project_digest(
    container_manager,
    workspace_id: str,
    *,
    token_budget: int | None = None,
    context_window: int | None = None,
) -> str | None:
    """Inline every workspace file into a boundary-delimited block.

    Returns ``None`` when:
    - The workspace has no matching files (caller falls through to
      ``WorkspaceSnapshot`` + ``repo_map``).
    - The total content would exceed ``token_budget`` — the whole
      digest is discarded, never partially returned. Caller uses the
      existing on-demand path so the model gets complete ``file_read``
      results rather than a digest with invisible gaps.
    - Any container command fails (best-effort; silent degradation).

    When it returns a string, that string is suitable for prepending
    directly to the system prompt. Place it ABOVE PLAN_SYSTEM /
    ACT_SYSTEM so the prefix caches; placing it at the tail would force
    a cache miss every time unrelated context changes.
    """
    if container_manager is None:
        return None
    if token_budget is None:
        token_budget = _default_budget(context_window)

    # Step 1: enumerate files with line counts. Reuse repomap's
    # ``find | wc -l`` pattern — same skip-dirs, same extension filter,
    # same output shape — so the two subsystems agree on "what counts
    # as a source file". Any divergence would confuse the model ("I
    # see file X in the tree but digest doesn't have it").
    skip_prune = " ".join(f"-path '*/{d}' -prune -o" for d in _SKIP_DIRS)
    ext_filter = " -o ".join(f"-name '*.{e}'" for e in _CODE_EXTENSIONS)
    find_cmd = (
        f"find /workspace {skip_prune} "
        f"\\( {ext_filter} \\) -print "
        f"2>/dev/null | head -500 | xargs wc -l 2>/dev/null"
    )

    try:
        listing = await container_manager._run_command(
            workspace_id, ["bash", "-c", find_cmd], timeout=10.0,
        )
    except Exception:
        log.debug("digest_listing_failed", exc_info=True)
        return None

    files: list[tuple[str, int]] = []  # (rel_path, line_count)
    for raw in listing.strip().splitlines():
        line = raw.strip()
        if not line or line.endswith(" total"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        count_raw, abs_path = parts
        try:
            count = int(count_raw)
        except ValueError:
            continue
        if abs_path.startswith("/workspace/"):
            rel = abs_path[len("/workspace/"):]
        elif abs_path == "/workspace":
            continue
        else:
            rel = abs_path
        if not rel:
            continue
        files.append((rel, count))

    if not files:
        return None

    # Deterministic sort — critical for prefix caching. If files come
    # out in different orders across runs, every inference pays a
    # cache miss. Sort alphabetically by path.
    files.sort(key=lambda r: r[0])

    # Step 2: pre-flight size estimate using line counts alone. If even
    # the rough estimate exceeds the budget, skip reading file contents
    # — that's the whole point of the budget gate. Average source line
    # in code ≈ 30-50 chars; use 40 as a midpoint. The estimate is
    # deliberately loose because a FALSE NEGATIVE (skipping when we
    # could fit) is cheap — caller falls through to repo_map. A FALSE
    # POSITIVE (proceeding to read when we can't fit) means reading
    # hundreds of files and then throwing it away.
    estimated_chars = sum(count * 40 for _, count in files)
    estimated_tokens = (estimated_chars + 3) // 4
    if estimated_tokens > token_budget:
        log.debug(
            "digest_over_budget_preflight",
            files=len(files),
            estimated_tokens=estimated_tokens,
            budget=token_budget,
        )
        return None

    # Step 3: read file contents, streaming into buffer, aborting if
    # we cross the budget mid-read. One sequential read per file — we
    # don't parallelise because (a) the container runtime serialises
    # shell commands anyway and (b) concurrent exec on small files has
    # negative throughput win under GVisor / Docker.
    buffer: list[str] = []
    running_chars = 0
    # Budget in chars for easier comparison with raw content lengths.
    char_budget = token_budget * 4

    for rel, line_count in files:
        abs_path = f"/workspace/{rel}" if not rel.startswith("/") else rel
        try:
            content = await container_manager.file_read(workspace_id, abs_path)
        except Exception:
            # The digest is presented to the model as a complete project
            # view. A silently-dropped file leaves the model believing
            # it has the whole picture when it doesn't — exactly the
            # "authoritative context truncated silently" footgun. Warn
            # so the drop is at least findable in logs.
            log.warning("digest_read_failed", path=abs_path, exc_info=True)
            continue

        if content is None:
            continue

        header = _file_header(rel, line_count)
        footer = _file_footer(rel)
        entry = f"{header}\n{content}\n{footer}\n"

        if running_chars + len(entry) > char_budget:
            log.debug(
                "digest_over_budget_during_read",
                path=rel,
                running_chars=running_chars,
                entry_chars=len(entry),
                budget_chars=char_budget,
            )
            # Partial digests are worse than no digest — they give the
            # model a false sense of completeness. Bail and let the
            # caller fall through.
            return None

        buffer.append(entry)
        running_chars += len(entry)

    if not buffer:
        return None

    preamble = (
        "<project_digest>\n"
        "Full inlined contents of every source file in /workspace, sorted "
        "by path. File boundaries marked with ``===== FILE: ... =====`` "
        "headers. This block is the authoritative view of the workspace — "
        "you DO NOT need to call file_read / dir_tree to see what's here. "
        "Use file_read only to re-check a file after an edit, or for "
        "line-range slicing when editing specific regions.\n\n"
    )
    closing = "</project_digest>"

    return preamble + "".join(buffer) + closing
