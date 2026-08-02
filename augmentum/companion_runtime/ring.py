"""Results ring — turn-decayed memory of what she recently looked at.

The lifecycle every perceptual result follows (2026-06-12 design):

    full result (the turn it ran)
      -> compressed digest line (N turns, touch-to-refresh)
        -> pull-only (re-fetch via context.peek / re-inflation)

Tool results, presence depth (page excerpt / note tail), initiative
deliveries, and image generations all write entries here through
:func:`record`; ``compose_becca_prompt`` renders the survivors as the
"recently looked at" block and RE-INFLATES entries the current turn
clearly references (deterministic overlap — the model never has to
decide to re-fetch for the common case).

Design rules, each one a named failure mode:

* **Digests are indexical, not informational.** A digest that
  half-enumerates specifics ("3 rules: ...") invites confabulating the
  rest. Name that things exist; don't describe what she didn't say.
* **Same-slot supersede.** Three peeks of the same page while
  discussing it must not eat three ring slots.
* **Turn clock, not wall clock.** Conversational relevance decays by
  exchange; source freshness (AttentionStore) decays by minutes. Two
  different clocks, deliberately not conflated.
* **One write helper.** voice dispatch + native loop + presence all
  call :func:`record` — per the mode-handler-parallel lesson, the
  decay logic exists exactly once.

The ring lives on ``ReferentCache.results_ring`` and persists through
``working_state`` (per-user, survives restarts + voice session churn).
"""

from __future__ import annotations

import re
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

RING_CAP = 4
DEFAULT_KEEP_TURNS = 3

# Words that carry no referential signal — overlap on these must not
# refresh (or re-inflate) an entry. Mirrors the roster scorer's spirit.
_STOPWORDS = frozenset(
    ["the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "about", "from", "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "these", "those", "i", "you", "she", "he", "we", "they", "me", "my", "your", "her", "his", "our", "their", "what", "which", "who", "when", "how", "do", "does", "did", "can", "could", "would", "should", "will", "just", "so", "not", "no", "yes", "okay", "please", "tell", "show", "open", "get", "make", "let"]
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]+")


def _content_tokens(text: str) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall((text or "").lower())
        if len(t) > 2 and t not in _STOPWORDS
    }


def bump_turn(refs: Any) -> int:
    """Advance the per-conversation decay clock. One bump per USER turn."""
    if refs is None:
        return 0
    refs.turn_seq = int(getattr(refs, "turn_seq", 0) or 0) + 1
    return refs.turn_seq


def record(
    refs: Any,
    *,
    kind: str,
    label: str,
    digest: str = "",
    slot: str = "",
    detail: str = "",
    refetch: dict[str, Any] | None = None,
) -> None:
    """Write (or supersede) a ring entry. Never raises.

    ``slot`` is the supersede key — entries sharing a slot replace each
    other (same page re-peeked, same note re-touched). Empty slot means
    the entry is its own one-off. ``detail`` is the FULL result text,
    stored but never rendered after the turn it was born — it's what
    re-inflation and ``context.peek(recent)`` hand back, so a follow-up
    never has to re-run a non-idempotent tool to recover what it said.
    ``refetch`` carries context.peek args for sources that are better
    re-fetched live (page/note/file); shape ``{"slot": ..., **extra}``.
    """
    if refs is None or not (label or digest):
        return
    try:
        ring = getattr(refs, "results_ring", None)
        if ring is None:
            return
        turn = int(getattr(refs, "turn_seq", 0) or 0)
        entry = {
            "kind": str(kind or "tool")[:32],
            "slot": str(slot or "")[:64],
            "label": str(label or "")[:120],
            "digest": str(digest or "")[:200],
            "detail": str(detail or "")[:1200],
            "refetch": refetch if isinstance(refetch, dict) else None,
            "born_turn": turn,
            "touch_turn": turn,
        }
        if entry["slot"]:
            ring[:] = [e for e in ring if e.get("slot") != entry["slot"]]
        ring.append(entry)
        # Cap eviction prefers tool/action entries — presence entries
        # (bounded to one per slot by the supersede above) double as
        # the cold-tier depth cache for re-inflation and peek, so a
        # burst of tool calls must not flush the open page's excerpt.
        while len(ring) > RING_CAP:
            for i, e in enumerate(ring):
                if e.get("kind") != "presence":
                    del ring[i]
                    break
            else:
                del ring[0]
    except Exception:  # noqa: BLE001 — the ring is best-effort memory
        log.debug("ring_record_failed", exc_info=True)


def alive(refs: Any, *, keep_turns: int = DEFAULT_KEEP_TURNS) -> list[dict]:
    """Surviving entries, oldest first. Evicts the dead in place.

    Presence entries (kind="presence") are EXEMPT from age eviction:
    they persist as the cold-tier depth cache while their source is
    still on screen — evicting them made the feeder see an empty slot
    and re-record a newborn, so a decayed page excerpt oscillated back
    to full fidelity every few turns. Their RENDER tier (full vs
    pointer) is the renderer's freshness check, not liveness here.
    """
    if refs is None:
        return []
    ring = getattr(refs, "results_ring", None)
    if not ring:
        return []
    turn = int(getattr(refs, "turn_seq", 0) or 0)
    kept = [
        e for e in ring
        if e.get("kind") == "presence"
        or turn - int(e.get("touch_turn") or e.get("born_turn") or 0) <= keep_turns
    ]
    if len(kept) != len(ring):
        ring[:] = kept
    return list(kept)


def touch_and_match(
    refs: Any, scoring_text: str, *, keep_turns: int = DEFAULT_KEEP_TURNS,
) -> list[dict]:
    """Refresh entries the current turn references; return the matches.

    The relevance "evaluation" is deterministic content-token overlap
    against label+digest — the same trust level as the roster ranker,
    zero latency, zero tokens. Matched entries get their decay clock
    reset AND are returned so the renderer can re-inflate them (pull
    the full detail back into the prompt server-side, before the model
    ever has to decide anything).
    """
    entries = alive(refs, keep_turns=keep_turns)
    if not entries:
        return []
    turn_tokens = _content_tokens(scoring_text)
    if not turn_tokens:
        return []
    turn = int(getattr(refs, "turn_seq", 0) or 0)
    matched: list[dict] = []
    for e in entries:
        entry_tokens = _content_tokens(
            f"{e.get('label') or ''} {e.get('digest') or ''}",
        )
        if len(turn_tokens & entry_tokens) >= 2 or (
            # Single-token overlap still counts when the token is rare-
            # shaped (long word, likely a name/title fragment).
            any(len(t) >= 6 for t in (turn_tokens & entry_tokens))
        ):
            e["touch_turn"] = turn
            matched.append(e)
    return matched


def age_turns(refs: Any, entry: dict) -> int:
    turn = int(getattr(refs, "turn_seq", 0) or 0)
    return max(0, turn - int(entry.get("born_turn") or 0))


def record_action_result(
    refs: Any, *, action_id: str, args: dict | None, result: Any,
) -> None:
    """Ring feeder for verb dispatches (all three dispatch layers).

    Clarifies don't record — an open question is pending_intent's job,
    not a result. The digest defaults to what she SAID (``speak``),
    which is indexical by construction: it can't half-describe more
    than the user already heard. Never raises.
    """
    try:
        if result is None or getattr(result, "clarify", None):
            return
        arg_hint = ""
        for key in ("query", "title", "location", "prompt", "duration", "intent"):
            val = (args or {}).get(key)
            if val:
                arg_hint = str(val)[:60]
                break
        label = f"{action_id}: {arg_hint}" if arg_hint else action_id
        digest = (
            getattr(result, "digest", "")
            or getattr(result, "speak", "")
            or getattr(result, "toast", "")
            or "done"
        )
        detail_bits = [
            getattr(result, "speak", "") or "",
            getattr(result, "toast", "") or "",
            getattr(result, "prompt_addendum", "") or "",
        ]
        record(
            refs, kind="action", slot=f"action:{action_id}",
            label=label, digest=digest,
            detail="\n".join(b for b in detail_bits if b),
        )
    except Exception:  # noqa: BLE001
        log.debug("ring_record_action_failed", exc_info=True)
