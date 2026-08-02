"""Tier 1 matcher — regex match across the registry.

Walks the registered actions in registration order and returns the
first action whose compiled patterns match the transcript. Tier 2
(embedding) and Tier 3 (LLM tools) are out of scope here — the
dispatch layer calls into them if this returns None.
"""

from __future__ import annotations

from typing import Any

from augmentum.intent.action import IntentMatch
from augmentum.intent.registry import REGISTRY
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _normalize(text: str) -> str:
    """Lowercase + strip trailing punctuation so the regex doesn't have
    to tolerate every variant. Patterns themselves are case-insensitive
    via re.IGNORECASE, but trailing '?' / '!' / '.' would otherwise
    require explicit handling in every pattern."""
    s = text.strip()
    # Drop a single trailing punctuation char — "what?" → "what",
    # "stop." → "stop". Multi-char trailing strings ("..." / "!!") are
    # collapsed too.
    while s and s[-1] in ".!?,":
        s = s[:-1]
    return s


def match_intent(
    text: str, mode: str | None = None, *, fast_path_only: bool = False,
) -> IntentMatch | None:
    """Return the first Tier 1 hit, or None.

    Named capture groups in a matched pattern become ``args`` on the
    returned IntentMatch — handler signatures can declare them
    directly. Empty groups are dropped so a handler can default-via-
    kwarg-presence rather than threading None checks.

    ``fast_path_only`` restricts the walk to conversation-control verbs
    (``fanout.fast_path``) — stop / repeat / slower / louder / goodbye /
    nevermind / strike. The voice route runs a fast-path-only pass
    BEFORE the address router so these can't be swallowed by the
    always-listening converse-skip or wait on LLM latency.
    """
    if not text:
        return None
    norm = _normalize(text)
    if not norm:
        return None

    # Two-pass strategy:
    #   Pass 1 — strict regex ``patterns`` (Tier 1a). These are
    #            hand-authored and match exactly what the author
    #            wrote. Sub-ms; preferred when they hit.
    #   Pass 2 — compiled ``templates`` (Tier 1b). Hassil-style
    #            syntax compiled to regex at registration. Catches
    #            natural phrasing variants the strict regex misses.
    #
    # Both passes use ``search`` against the normalized transcript
    # and return the first hit. Patterns are tried first because
    # strict regex authors had specific phrasings in mind and the
    # matcher should honor that.
    for action in REGISTRY.all():
        if not action.fanout.tier1:
            continue
        if fast_path_only and not action.fanout.fast_path:
            continue
        if not action.available_in(mode):
            continue
        for pattern in action.patterns:
            m = pattern.search(norm)
            if m is None:
                continue
            args: dict[str, Any] = {
                k: v.strip() if isinstance(v, str) else v
                for k, v in m.groupdict().items() if v is not None
            }
            log.debug(
                "intent_tier1_hit",
                action=action.id, args=args, pattern=pattern.pattern,
            )
            return IntentMatch(action_id=action.id, args=args, tier=1)

    # Pass 2 — templates. Same skip rules as patterns; same shape
    # for the returned IntentMatch (slot names become args).
    for action in REGISTRY.all():
        if not action.fanout.tier1:
            continue
        if fast_path_only and not action.fanout.fast_path:
            continue
        if not action.available_in(mode):
            continue
        for compiled in action.compiled_templates:
            m = compiled.pattern.search(norm)
            if m is None:
                continue
            args = {
                k: v.strip() if isinstance(v, str) else v
                for k, v in m.groupdict().items() if v is not None
            }
            log.debug(
                "intent_tier1b_template_hit",
                action=action.id, args=args, template=compiled.source,
            )
            return IntentMatch(action_id=action.id, args=args, tier=1)

    return None
