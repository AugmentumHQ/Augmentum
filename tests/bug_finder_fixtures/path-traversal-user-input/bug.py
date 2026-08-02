"""Toy avatar download endpoint."""

from __future__ import annotations

from pathlib import Path

AVATAR_DIR = Path("/var/data/avatars")


def read_avatar(filename: str) -> bytes:
    # BUG: filename is concatenated without sanitization. `../../etc/passwd`
    # escapes AVATAR_DIR and reads anything readable by the process.
    target = AVATAR_DIR / filename
    with open(target, "rb") as f:
        return f.read()
