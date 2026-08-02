"""Registry grounding — show the synthesizer the verbs that already exist.

The OpenRoom discipline again: their agent MUST call ``list_apps`` / read the
manifest before acting, so it reuses real capabilities instead of inventing
duplicates. Our synthesizer had no such view — it would happily mint
``navigate.open_browse`` when ``browse.open`` already does the job (redundant
clutter) or reuse an existing id and *shadow a hand-authored verb*
(``registry.add`` overwrites on duplicate id — silent and dangerous for a
machine-authored verb).

This module hands the live registry's (id, summary) digest to the synthesis
prompt so the model can decline with ``{"duplicate": "<id>"}`` when the request
is already covered, and provides an exact-id gate so a synthesized spec can never
overwrite an existing verb even if the model ignores the hint.

Injectable everywhere (``known_verbs`` lists are passed in) so synthesis stays
unit-testable without importing the whole intent package; production callers let
it lazily read ``REGISTRY``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# (text) -> a match object with ``.action_id`` (the tier-1 matcher), or None.
MatchFn = Callable[[str], Any]

# Keep the prompt digest bounded — summaries are hints, not contracts.
_SUMMARY_CAP = 90
_LIST_CAP = 120  # we have ~59 verbs today; cap guards against unbounded growth


def known_verbs_from_registry() -> list[tuple[str, str]]:
    """Live (id, summary) pairs from the action registry, sorted by id.

    Imports lazily and never raises — a registry that can't be read yields an
    empty list, and the caller's gate/grounding both degrade safely to off."""
    try:
        from augmentum.intent import REGISTRY  # noqa: PLC0415 — lazy: avoids import cycle
    except Exception as exc:  # noqa: BLE001
        log.info("registry_grounding_unavailable", error=repr(exc))
        return []
    pairs = [(a.id, (a.summary or "").strip()) for a in REGISTRY.all()]
    return sorted(pairs, key=lambda p: p[0])


def find_exact_duplicate(spec_id: str, known: list[tuple[str, str]]) -> str:
    """Return the existing summary if ``spec_id`` already exists, else ""."""
    for vid, summary in known:
        if vid == spec_id:
            return summary or "(no summary)"
    return ""


def default_match_fn() -> MatchFn:
    """The live tier-1 matcher, imported lazily (avoids an import cycle). Returns a
    no-op matcher if the intent package can't be imported."""
    try:
        from augmentum.intent.matcher import match_intent  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        log.info("collision_matcher_unavailable", error=repr(exc))
        return lambda _text: None
    return match_intent


def find_example_collisions(
    examples: list[str], match_fn: MatchFn,
) -> list[tuple[str, str]]:
    """For each example phrasing, if it ALREADY matches an existing tier-1 verb,
    return ``(example, existing_id)``. A new verb registers last, so the registry
    (walked in registration order) hands that phrasing to the existing verb first
    -- the new verb would lose tier-1 dispatch on it. Never raises."""
    out: list[tuple[str, str]] = []
    for ex in examples:
        if not (ex or "").strip():
            continue
        try:
            m = match_fn(ex)
        except Exception:  # noqa: BLE001 — a matcher hiccup ≠ a collision
            continue
        if m is not None:
            out.append((ex, str(getattr(m, "action_id", "?"))))
    return out


def describe_known_verbs(known: list[tuple[str, str]]) -> str:
    """A grounding block listing existing verbs so the model reuses or declines.
    Empty string when there are none (omit the section, don't print an empty list)."""
    if not known:
        return ""
    shown = known[:_LIST_CAP]
    lines = [f"  {vid} - {summary[:_SUMMARY_CAP]}" for vid, summary in shown]
    more = "" if len(known) <= _LIST_CAP else f"\n  ...and {len(known) - _LIST_CAP} more"
    return (
        "EXISTING VERBS — the assistant ALREADY has these. If one of them already "
        "covers the request, do NOT author a duplicate; respond "
        '{"duplicate": "<existing.id>"} instead. Never reuse an existing id for a '
        "new verb.\n" + "\n".join(lines) + more
    )
