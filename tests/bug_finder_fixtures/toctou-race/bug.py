"""Toy file-uploader with permission check."""

from __future__ import annotations

import os


def upload_to(path: str, body: bytes) -> bool:
    # BUG: classic TOCTOU. We check the path is a regular file (e.g., not a
    # symlink to /etc/passwd) and writable, then we open it — but between
    # those calls an attacker can swap the file for a symlink.
    if not os.path.isfile(path):
        return False
    if not os.access(path, os.W_OK):
        return False
    with open(path, "wb") as f:
        f.write(body)
    return True
