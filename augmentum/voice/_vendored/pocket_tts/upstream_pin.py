"""Frozen upstream pin for the pocket-tts override.

Kept in its own dep-free module so a quick

    python -c "from augmentum.voice._vendored.pocket_tts.upstream_pin import UPSTREAM_COMMIT"

works without pulling torch / safetensors / scipy through `tap.py`.

When upgrading the pin, update both UPSTREAM_COMMIT and UPSTREAM_DATE,
then re-run the smoke tests in tests/test_smoke_pocket_tts_vendor.py.
See VENDOR.md for the full upgrade procedure.
"""

from __future__ import annotations

# Commit hash of the upstream tree this override was validated against.
# When upstream moves, the tap.py loader emits a warning (not an error)
# but the override stays best-effort active. See VENDOR.md for why we
# warn instead of raise.
UPSTREAM_COMMIT: str = "15a6c1817b360f9b37691aef9734435a85610c68"
UPSTREAM_DATE: str = "2026-06-03"
UPSTREAM_REPO: str = "https://github.com/kyutai-labs/pocket-tts"

# Minimum pocket-tts PyPI version that has the override-required method
# `_decode_audio_worker` reachable for subclass override. If upstream
# refactors below this version, the tap won't install and we fall back
# to the upstream model unmodified (audio history captures transcript
# only for that runtime).
MIN_TESTED_VERSION: str = "0.0.0"  # update when pocket-tts ships __version__ reliably

# Public method names on TTSModel that the tap depends on staying in
# place. The smoke tests in tests/test_smoke_pocket_tts_vendor.py check
# these survive any patch we apply.
REQUIRED_UPSTREAM_METHODS: tuple[str, ...] = (
    "generate_audio_stream",
    "generate_audio",
    "load_model",
    "get_state_for_audio_prompt",
    "_decode_audio_worker",
)
