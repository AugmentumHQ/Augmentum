"""Resolve the running code's git SHA — for fabric drift visibility.

The augmentum container runs a prebuilt image with the host repo overlaid
onto /app at boot, and the repo (incl. .git) is mounted read-only at
/host-augmentum-src. So the SHA that identifies "which code is this node
running" lives in the host repo, not the image. We parse it from the
.git files directly — no git binary in the container required.

Used by the fabric layer so the main can see which nodes are behind
(see scripts/deploy-nodes.sh --status for the SSH-side equivalent).
"""

from __future__ import annotations

import functools
import os

# Where the host repo (with .git) might be visible, most-specific first.
_CANDIDATE_REPOS = (
    os.environ.get("AUGMENTUM_SRC", ""),
    "/host-augmentum-src",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)


def _read_head_sha(repo: str) -> str | None:
    """Resolve HEAD → 40-char SHA by reading .git files (no git binary)."""
    git_dir = os.path.join(repo, ".git")
    head_file = os.path.join(git_dir, "HEAD")
    try:
        with open(head_file, encoding="utf-8") as f:
            head = f.read().strip()
    except OSError:
        return None

    if not head.startswith("ref:"):
        return head or None  # detached HEAD — the SHA itself

    ref_path = head[4:].strip()  # e.g. refs/heads/main
    loose = os.path.join(git_dir, ref_path)
    try:
        with open(loose, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        pass
    # Packed refs fallback.
    packed = os.path.join(git_dir, "packed-refs")
    try:
        with open(packed, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(("#", "^")) and line.endswith(ref_path):
                    return line.split(" ", 1)[0]
    except OSError:
        pass
    return None


@functools.lru_cache(maxsize=1)
def get_code_version() -> str:
    """Short git SHA (12 chars) of the running code, or 'unknown'.

    Cached for the process lifetime — the code can't change without a
    restart (the overlay re-syncs at boot), which resets the cache.
    """
    for repo in _CANDIDATE_REPOS:
        if not repo:
            continue
        sha = _read_head_sha(repo)
        if sha:
            return sha[:12]
    return "unknown"
