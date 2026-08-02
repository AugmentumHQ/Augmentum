"""Augmentum's vendored pocket-tts override surface.

Public exports + the version pin constants. See VENDOR.md for the
rationale on the pinned-override strategy vs wholesale vendor.
"""

from __future__ import annotations

from augmentum.voice._vendored.pocket_tts.tap import (
    MimiCodesCallback,
    MimiTappedTTSModel,
    try_install_tap,
    upstream_available,
)
from augmentum.voice._vendored.pocket_tts.upstream_pin import (
    MIN_TESTED_VERSION,
    REQUIRED_UPSTREAM_METHODS,
    UPSTREAM_COMMIT,
    UPSTREAM_DATE,
    UPSTREAM_REPO,
)

__all__ = [
    "MIN_TESTED_VERSION",
    "MimiCodesCallback",
    "MimiTappedTTSModel",
    "REQUIRED_UPSTREAM_METHODS",
    "UPSTREAM_COMMIT",
    "UPSTREAM_DATE",
    "UPSTREAM_REPO",
    "try_install_tap",
    "upstream_available",
]
