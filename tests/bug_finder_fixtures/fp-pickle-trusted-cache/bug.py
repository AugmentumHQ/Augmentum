"""FP-bait: looks like a pickle RCE sink, but the input is trusted."""

from __future__ import annotations

import pickle
from pathlib import Path

# Cache file is written by this same process, in a directory we control.
# No external write path exists. Loading it cannot RCE us — we're loading
# our own data.
CACHE_PATH = Path("/var/lib/myapp/_internal_cache.pkl")


def save_cache(payload: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_bytes(pickle.dumps(payload))


def load_cache() -> dict:
    # pickle.load() on disk content. A naive detector flags this as
    # arbitrary code execution. But the only writer is save_cache above —
    # there's no untrusted producer in this trust boundary.
    if not CACHE_PATH.exists():
        return {}
    return pickle.loads(CACHE_PATH.read_bytes())
