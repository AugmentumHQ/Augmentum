"""Refresh the vendored LiteLLM cost table.

Fetches the latest model_prices_and_context_window.json from
BerriAI/litellm (MIT) and overwrites augmentum/fabric/model_costs.json.

Run after major LLM provider pricing changes or when a new model
you want to route to isn't recognised by the cost-aware scorer.

Usage:
    python scripts/refresh_litellm_costs.py
    python scripts/refresh_litellm_costs.py --dry-run

The script preserves the __metadata__ block (source + license
attribution); LiteLLM's file lacks one, so we re-stamp ours after
each refresh.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

_UPSTREAM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
_VENDORED_PATH = (
    Path(__file__).parent.parent
    / "augmentum" / "fabric" / "model_costs.json"
)


def fetch_upstream() -> dict:
    """Download the latest cost table from BerriAI/litellm.

    Raises on any HTTP error so the operator sees the problem
    instead of silently keeping stale data.
    """
    print(f"Fetching: {_UPSTREAM_URL}")
    resp = httpx.get(_UPSTREAM_URL, timeout=60.0, follow_redirects=True)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(
            f"upstream returned {type(data).__name__}, expected dict"
        )
    return data


def merge_with_metadata(upstream: dict) -> dict:
    """Stamp our metadata block on the upstream content.

    LiteLLM doesn't include __metadata__; we keep our license
    attribution + vendored_at timestamp every refresh so the
    file's provenance stays self-documenting.
    """
    # Count entries (skip any nested metadata keys).
    entry_count = sum(
        1 for k in upstream if not k.startswith("__")
        and isinstance(upstream[k], dict)
    )
    out = {
        "__metadata__": {
            "source": (
                "https://github.com/BerriAI/litellm/blob/main/"
                "model_prices_and_context_window.json"
            ),
            "license": "MIT",
            "license_url": (
                "https://github.com/BerriAI/litellm/blob/main/LICENSE"
            ),
            "purpose": (
                "Reference table for cost-aware fabric routing. "
                "Local + peer models default to 0.0 (sunk hardware cost); "
                "cloud-backed peer capabilities populate from these "
                "entries by model_id lookup."
            ),
            "format_note": (
                "Costs are per-TOKEN (matching LiteLLM's wire format). "
                "$0.00003/token = $30 per 1M tokens. Multiply by 1e6 "
                "when displaying $/1M to operators."
            ),
            "format_version": 1,
            "vendored_at": datetime.now(timezone.utc).date().isoformat(),
            "vendored_count": entry_count,
        }
    }
    # Drop any upstream __* keys + copy real entries.
    for k, v in upstream.items():
        if k.startswith("__"):
            continue
        out[k] = v
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch + count, but don't overwrite the vendored file.",
    )
    args = parser.parse_args()

    try:
        upstream = fetch_upstream()
    except (httpx.HTTPError, RuntimeError) as exc:
        print(f"ERROR: refresh failed — {exc}", file=sys.stderr)
        return 1

    merged = merge_with_metadata(upstream)
    new_count = merged["__metadata__"]["vendored_count"]

    if args.dry_run:
        print(f"[dry-run] would write {new_count} entries to {_VENDORED_PATH}")
        return 0

    # Pretty-print so diff-review is readable; UTF-8 explicit.
    text = json.dumps(merged, indent=2, ensure_ascii=False, sort_keys=False)
    _VENDORED_PATH.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote {new_count} entries to {_VENDORED_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
