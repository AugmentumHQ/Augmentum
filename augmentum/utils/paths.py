"""Cross-platform path resolution for bundled model directories.

The Docker images pre-bake model weights into ``/home/augmentum/.<name>``
because that's where the `augmentum` system user owns its home in the
container. Native installs (Apple Silicon, Linux without Docker,
Windows-native — none of which are the primary deployment target but all
of which someone will eventually try) can't write to ``/home/augmentum``.

This helper resolves a default model directory in this order:

1. The Docker-bundled path ``/home/augmentum/.<name>`` if it exists —
   covers every Docker-on-Linux-host scenario without any change in
   behaviour.
2. Platform-appropriate user cache:
   * macOS:   ``~/Library/Caches/augmentum/<name>``
   * Windows: ``%LOCALAPPDATA%\\augmentum\\<name>``
   * Linux:   ``~/.cache/augmentum/<name>`` (XDG-friendly).

The directory is *not* created — callers decide whether to mkdir or
fail with a clear error. This is a path-resolver, not a downloader.
"""

from __future__ import annotations

import os
import sys


def resolve_model_dir(name: str) -> str:
    """Return the default on-disk location for a bundled model.

    ``name`` is the short dirname (e.g. ``"kokoro"``, ``"dtln"``). The
    Docker path uses ``.<name>``; the user-cache paths use ``<name>``.
    """
    docker_path = f"/home/augmentum/.{name}"
    if os.path.isdir(docker_path):
        return docker_path

    if sys.platform == "darwin":
        return os.path.expanduser(f"~/Library/Caches/augmentum/{name}")
    if sys.platform == "win32":
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            return os.path.join(local_app, "augmentum", name)
        return os.path.expanduser(f"~/AppData/Local/augmentum/{name}")
    return os.path.expanduser(f"~/.cache/augmentum/{name}")
