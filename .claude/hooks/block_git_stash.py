#!/usr/bin/env python3
"""PreToolUse hook: hard-block `git stash` in this repo.

`git stash` has caused lost-work incidents here (stashes that masked
multi-session WIP and got buried). Memory/instructions didn't stop agents from
reaching for it, so this enforces the rule at the harness level: any Bash call
whose command runs `git stash` (any subcommand, any flags, anywhere in a
compound command) is DENIED before it executes.

Matches `git stash` only as the git SUBCOMMAND at a command boundary (start of
command or after a shell operator), tolerant of whitespace and global flags
between `git` and `stash`. Does NOT match `stash` appearing inside a quoted
argument such as a commit message (e.g. `git commit -m "fix stash bug"`).

Reads the PreToolUse JSON on stdin; emits a deny decision as JSON on stdout.
Fails OPEN (exits 0 silently) if the input can't be parsed — a hook must never
wedge the agent over a parse error.
"""

from __future__ import annotations

import json
import re
import sys

# git, then optional global flags (--no-pager, -c, ...), then the `stash`
# subcommand — anchored to a command boundary so quoted "stash" text is ignored.
_STASH = re.compile(r"(?:^|[;&|\n(])\s*git\s+(?:-{1,2}\S+\s+)*stash(?:\s|$|[;&|)])")


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0  # unparseable → don't block
    cmd = ((data.get("tool_input") or {}).get("command")) or ""
    if not isinstance(cmd, str) or not _STASH.search(cmd):
        return 0  # not a git stash → allow
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                "git stash is FORBIDDEN in this repo — it has caused lost-work "
                "incidents (buried multi-session WIP). The working tree is kept "
                "committed; instead make a WIP commit on the current branch or a "
                "new branch. To shelve work without committing, use `git worktree` "
                "or a throwaway branch — never stash."
            ),
        }
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
