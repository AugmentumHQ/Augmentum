"""Toy ping endpoint."""

from __future__ import annotations

import subprocess


def ping(host: str) -> str:
    # BUG: shell=True with user-controlled host. `8.8.8.8; rm -rf /` runs
    # both commands. Even `8.8.8.8 && cat /etc/passwd` works.
    out = subprocess.run(
        f"ping -c 1 {host}", shell=True, capture_output=True, text=True,
    )
    return out.stdout
