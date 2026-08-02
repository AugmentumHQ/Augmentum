"""Cost-aware routing: lookup table for cloud model pricing.

Loads ``augmentum/fabric/model_costs.json`` (vendored from
BerriAI/litellm under MIT) into a process-local dict for fast
per-request lookup. The director's scoring function (Phase 10.4)
uses ``input_cost_per_token`` + ``output_cost_per_token`` to bias
candidates away from expensive cloud peers when local or cheaper
peers can serve the same model.

Local-hosted + peer-hosted LLMs default to 0.0 (sunk hardware cost
— operator already paid for electricity). Only cloud-routed
capabilities populate real costs from this table.

Operators refresh the table by running
``scripts/refresh_litellm_costs.py``; this module never fetches
remotely, so a misbehaving network can't break startup.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Path is module-relative so the file is found whether the package
# is installed or imported from a source checkout.
_COST_JSON_PATH = Path(__file__).parent / "model_costs.json"


def _normalise_model_id(model_id: str) -> str:
    """The cost table keys use various conventions
    (``gpt-4o``, ``groq/llama-3.1-70b-versatile``,
    ``openrouter/anthropic/claude-3.5-sonnet``). Caller may pass
    bare ``llama-3.1-70b-versatile`` or the fully-qualified form.
    Lookup is case-insensitive; provider-prefix variants tried
    when the bare lookup misses.

    Returns the normalised key the caller should use against the
    cached dict; the lookup helpers handle the prefix-fallback
    internally.
    """
    return model_id.strip().lower()


@lru_cache(maxsize=1)
def _load_cost_table() -> dict[str, dict[str, Any]]:
    """Load the JSON file once + cache the dict.

    Returns an empty dict if the file is missing or malformed
    (operator hasn't refreshed yet, file got corrupted, etc.).
    Cost-aware routing degrades gracefully to "treat all candidates
    as cost-equal" rather than crashing.
    """
    try:
        with open(_COST_JSON_PATH, encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        log.warning(
            "fabric_cost_table_missing",
            path=str(_COST_JSON_PATH),
            msg="run scripts/refresh_litellm_costs.py to populate",
        )
        return {}
    except json.JSONDecodeError as exc:
        log.warning(
            "fabric_cost_table_corrupt",
            path=str(_COST_JSON_PATH), error=str(exc)[:160],
        )
        return {}

    # Drop the __metadata__ key from the lookup dict; case-insensitive
    # keys for resilient matching.
    out: dict[str, dict[str, Any]] = {}
    for k, v in raw.items():
        if k.startswith("__"):
            continue
        if isinstance(v, dict):
            out[_normalise_model_id(k)] = v
    return out


def lookup_cost(model_id: str) -> tuple[float, float]:
    """Return (input_cost_per_token, output_cost_per_token) for a
    model. Returns ``(0.0, 0.0)`` when the model isn't in the table
    — that's the right default for local + peer-hosted LLMs that
    aren't burning cloud API credits.

    Tries the bare key first, then a few common provider-prefix
    fallbacks (``openrouter/``, ``together_ai/``, ``groq/``).
    """
    table = _load_cost_table()
    norm = _normalise_model_id(model_id)
    entry = table.get(norm)
    if entry is None:
        # Fallback prefixes — many entries are filed under provider
        # namespaces (groq/llama-3.1-70b, etc.). Try a small set.
        for prefix in ("openrouter/", "together_ai/", "groq/",
                       "fireworks_ai/", "mistral/", "anthropic/",
                       "deepseek/", "azure/"):
            entry = table.get(prefix + norm)
            if entry is not None:
                break
    if entry is None:
        return 0.0, 0.0
    return (
        float(entry.get("input_cost_per_token", 0.0) or 0.0),
        float(entry.get("output_cost_per_token", 0.0) or 0.0),
    )


def reload_cost_table() -> int:
    """Drop the lru_cache so the next lookup re-reads the JSON.
    Called by the refresh script after it overwrites the file.
    Returns the new entry count.
    """
    _load_cost_table.cache_clear()
    return len(_load_cost_table())
