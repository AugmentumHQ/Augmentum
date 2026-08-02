"""ScratchStore — externalise large tool results to the container filesystem.

The coder's mid-turn compaction clips tool results to ~1500 chars (400 for
errors). For a 40k-char ``file_read`` or multi-megabyte ``shell_exec`` log,
that clip is *irreversible* — the content is gone once compacted, and the
model has to re-run the tool to see it again (and the re-run re-bloats
context, which re-triggers compaction, which re-clips...).

This module implements Manus's "filesystem as ultimate context" pattern:
when a tool result exceeds ``_SCRATCH_THRESHOLD`` bytes, write the full
content to ``/workspace/.augmentum/scratch/<tool>-<hash>.txt`` inside the
container. Replace the inline message body with a short summary + path
reference + preview. The model can re-read via the normal ``file_read``
tool when it actually needs the detail.

Reversible compression — the full content is recoverable at any point,
but doesn't occupy conversation-window tokens while idle.

Behavioural contract
--------------------
* Content under the threshold passes through unchanged (no indirection
  cost when it isn't needed).
* Content that externalises returns a structured ``ScratchRef`` so the
  handler can render a consistent "Large result externalised: ..." line.
* Writes are best-effort — if the container write fails (read-only FS,
  OOM, etc.) we fall back to returning the original content, so tool
  results never silently vanish. The fallback path preserves the old
  behaviour (full content inline, subject to compaction).
* Scratch files are grouped under ``/workspace/.augmentum/scratch/`` so
  users can inspect them and the auto-refresh workspace_tree will surface
  them explicitly.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from augmentum.coder.executors import ContainerExecutor
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Below this size, tool results pass through unchanged. Picked at 8k
# chars (~2000 tokens) because: (a) compaction keeps successful tool
# results at 1500 chars verbatim, so anything under that isn't at risk
# of clip anyway, and (b) above ~2000 tokens the result starts to
# dominate the conversation window. Threshold is conservative —
# externalising too eagerly costs the model a file_read round trip.
_SCRATCH_THRESHOLD = 8_000

# Where scratch files live inside the container. Under /workspace so
# the auto-refresh WorkspaceSnapshot surfaces them, and inside
# .augmentum/ so they don't pollute the project root.
_SCRATCH_DIR = "/workspace/.augmentum/scratch"

# Size of the inline preview shown in the compacted message body. Big
# enough for the model to tell "is this the content I want" without
# needing a file_read; small enough to stay under the 1500-char
# compaction cap.
_PREVIEW_CHARS = 500

# Size of the trailing tail rendered alongside the head preview.
# pytest summaries, traceback frames, and shell error lines almost
# always live at the END of long output. Showing only the head loses
# them. Head + tail with an "elided" middle marker matches what
# `head` + `tail` does in shell pipelines and matches opencode's
# truncate.ts pattern.
_TAIL_CHARS = 300

# Sanitise tool names so ``tool/name`` (MCP-namespaced) doesn't create
# stray subdirectories in the scratch dir.
_TOOL_NAME_SANITISE = re.compile(r"[^a-zA-Z0-9_\-]+")


@dataclass(frozen=True, slots=True)
class ScratchRef:
    """Handle for an externalised tool result.

    Fields
    ------
    path:
        Absolute path inside the container to the full-content file.
    original_size:
        Total byte length of the original content (so the model knows
        how much is still available if it re-reads).
    preview:
        First ``_PREVIEW_CHARS`` of the content. Included inline in
        the replacement message so the model can make decisions
        without a round-trip.
    tail:
        Last ``_TAIL_CHARS`` of the content (empty when the content
        is short enough that head already covers everything, or when
        head + tail would overlap). Surfaces error summaries / stack
        traces / final shell lines that would otherwise be lost when
        only the head is kept.
    source_tool:
        The tool name that produced the content. Surfaced in the
        replacement message body so the model can trace provenance.
    """

    path: str
    original_size: int
    preview: str
    source_tool: str
    tail: str = ""


class ScratchStore:
    """Container-backed scratch storage for oversized tool outputs.

    Instances are scoped to one ``workspace_id`` and lazy — no IO until
    ``maybe_externalise`` actually needs to write something. Init is
    cheap so the handler can unconditionally construct one in
    ``__init__`` and call ``maybe_externalise`` per tool result.
    """

    __slots__ = ("_cm", "_executor", "_workspace_id", "_threshold", "_dir_ensured")

    def __init__(
        self,
        container_manager,
        workspace_id: str,
        *,
        threshold: int = _SCRATCH_THRESHOLD,
    ) -> None:
        self._cm = container_manager
        self._executor = ContainerExecutor(container_manager, workspace_id)
        self._workspace_id = workspace_id
        self._threshold = threshold
        self._dir_ensured = False

    async def maybe_externalise(
        self,
        *,
        content: str,
        source_tool: str,
    ) -> ScratchRef | None:
        """Externalise ``content`` if it exceeds the threshold.

        Returns a ``ScratchRef`` when a write happened, ``None``
        otherwise (so the caller can keep the original inline). The
        ``None`` branch is hit both when content is small AND when a
        write fails — handler code treats both the same (inline the
        content), keeping the fallback path simple and safe.
        """
        if self._cm is None:
            return None
        if content is None:
            return None
        size = len(content)
        if size <= self._threshold:
            return None

        # Hash the FIRST 4096 bytes + total size — cheap enough, and
        # avoids hashing a megabyte blob on every write. Collisions on
        # 8-char truncated SHA-1 of (prefix, size) are astronomically
        # unlikely for normal tool output; if one happens the worst
        # case is two results share a file, which is harmless — we
        # overwrite and the second read still gets valid content.
        prefix = content[:4096].encode("utf-8", errors="replace")
        digest = hashlib.sha1(
            prefix + str(size).encode(), usedforsecurity=False,
        ).hexdigest()[:12]
        safe_tool = _TOOL_NAME_SANITISE.sub("-", source_tool or "unknown")[:32]
        path = f"{_SCRATCH_DIR}/{safe_tool}-{digest}.txt"

        if not await self._ensure_dir():
            return None

        try:
            # `file_write` on ContainerManager handles parent dirs and
            # encoding. We call through the same pathway the
            # FileWriteTool uses so behaviour stays consistent.
            await self._executor.write_file(path, content)
        except Exception:
            # Externalised tool output is lost — the model only sees the
            # preview, not the full result it was promised it could
            # re-read from disk. Surface so a recurring scratch-dir
            # failure (permissions, full disk) doesn't masquerade as
            # "preview-only mode is normal".
            log.warning("scratch.write_failed", path=path, exc_info=True)
            return None

        preview = content[:_PREVIEW_CHARS]
        # Tail starts after the head ends; only emit a non-empty tail
        # when there's a real gap between head and tail (otherwise the
        # head already covers everything and a duplicate "tail" is just
        # noise). The +1 ensures at least one character of separation.
        tail_start = max(_PREVIEW_CHARS, size - _TAIL_CHARS)
        tail = content[tail_start:] if tail_start < size - 1 else ""
        return ScratchRef(
            path=path,
            original_size=size,
            preview=preview,
            tail=tail,
            source_tool=source_tool or "unknown",
        )

    async def _ensure_dir(self) -> bool:
        """mkdir -p the scratch directory once per instance."""
        if self._dir_ensured:
            return True
        try:
            await self._cm._run_command(
                self._workspace_id,
                ["bash", "-c", f"mkdir -p {_SCRATCH_DIR}"],
                timeout=5.0,
            )
            self._dir_ensured = True
            return True
        except Exception:
            log.warning("scratch.mkdir_failed", dir=_SCRATCH_DIR, exc_info=True)
            return False


def render_scratch_message(ref: ScratchRef) -> str:
    """Render the replacement tool_result body when externalising.

    Preview leads; the externalisation machinery is footer metadata.
    Rationale (2026-04-22 transcript audit): when the bureaucratic
    header ran first, models spent attention on the path/size line
    and often missed that the answer they wanted was right there in
    the preview — leading to a redundant ``file_read`` round trip and
    sometimes a hallucinated "the command produced no useful output"
    summary. Preview-first means the tool result reads like a normal
    truncated shell output, with the externalisation note as an
    ordinary footer.

    When ``ref.tail`` is non-empty, the head is followed by an explicit
    elision marker and the tail. pytest summaries, traceback frames
    and final shell-error lines live at the END of long output —
    showing only the head loses them.
    """
    if ref.tail:
        elided = ref.original_size - len(ref.preview) - len(ref.tail)
        body = (
            f"{ref.preview}\n"
            f"\n[... {elided} bytes elided ...]\n\n"
            f"{ref.tail}\n"
            "\n---\n"
            f"[Head + tail above ({len(ref.preview)} + {len(ref.tail)} "
            f"of {ref.original_size} bytes, source: `{ref.source_tool}`). "
            f"Full content saved to `{ref.path}` — use `file_read` with "
            "`offset`/`limit` to load the elided middle when the "
            "head/tail isn't enough.]"
        )
        return body
    return (
        f"{ref.preview}\n"
        "\n---\n"
        f"[Preview above ({len(ref.preview)} of {ref.original_size} bytes, "
        f"source: `{ref.source_tool}`). "
        f"Full content saved to `{ref.path}` — use `file_read` with "
        "`offset`/`limit` to load the full content when the preview "
        "isn't enough.]"
    )
