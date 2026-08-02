"""Registry grounding — the synthesizer reuses/declines instead of duplicating.

Load-bearing:
  - describe_known_verbs renders a real "already exists" block (or nothing);
  - find_exact_duplicate spots an id collision;
  - the {"duplicate": ...} escape is a clean decline (not a retryable error,
    no repair call burned);
  - the exact-id gate rejects a spec that would SHADOW an existing verb;
  - known_verbs is injectable so synthesis is testable without the registry, and
    the live loader degrades to [] rather than raising.
"""

from __future__ import annotations

import json

from augmentum.selfedit.capabilities import (
    describe_known_verbs,
    find_exact_duplicate,
    known_verbs_from_registry,
)
from augmentum.selfedit.capabilities.registry_grounding import (
    default_match_fn,
    find_example_collisions,
)
from augmentum.selfedit.capabilities.synthesize import synthesize_capability_spec


def _NO_MATCH(_t):   # disable the collision gate deterministically
    return None


class _Hit:
    def __init__(self, action_id):
        self.action_id = action_id

_KNOWN = [
    ("browse.open", "Open the browse panel for the user."),
    ("control.stop", "Stop the current TTS playback immediately."),
]

_NEW_JSON = json.dumps({
    "id": "fun.coin_flip",
    "summary": "Say a coin-flip quip.",
    "examples": ["flip a coin"],
    "behavior": "speak",
    "speak": "Heads you win.",
    "stakes": "trivial_reversible",
})


# --- helpers ---------------------------------------------------------------

def test_describe_lists_existing_verbs():
    block = describe_known_verbs(_KNOWN)
    assert "browse.open" in block and "control.stop" in block
    assert "duplicate" in block          # tells the model how to decline
    assert "Never reuse an existing id" in block


def test_describe_empty_is_blank():
    assert describe_known_verbs([]) == ""


def test_find_exact_duplicate():
    assert find_exact_duplicate("browse.open", _KNOWN).startswith("Open the browse")
    assert find_exact_duplicate("nope.missing", _KNOWN) == ""


# --- synthesis behavior ----------------------------------------------------

async def test_grounding_block_reaches_the_prompt():
    seen: list[str] = []

    async def mi(prompt: str) -> str:
        seen.append(prompt)
        return _NEW_JSON

    spec, errs = await synthesize_capability_spec(
        "flip a coin", model_invoke=mi, known_verbs=_KNOWN, match_fn=_NO_MATCH,
    )
    assert spec is not None and errs == []
    assert "browse.open" in seen[0]      # existing verbs grounded the prompt


async def test_model_declines_duplicate_no_repair():
    calls: list[str] = []

    async def mi(prompt: str) -> str:
        calls.append(prompt)
        return '{"duplicate": "browse.open"}'

    spec, errs = await synthesize_capability_spec(
        "open browsing", model_invoke=mi, known_verbs=_KNOWN,
    )
    assert spec is None
    assert any("duplicate" in e and "browse.open" in e for e in errs)
    assert len(calls) == 1               # a clean decline does NOT burn a repair call


async def test_exact_id_collision_rejected_then_repaired():
    calls: list[str] = []

    async def mi(prompt: str) -> str:
        calls.append(prompt)
        # first: reuse an existing id (would shadow browse.open) → rejected;
        # repair: a fresh id → accepted
        if len(calls) == 1:
            return json.dumps({
                "id": "browse.open", "summary": "my own open",
                "examples": ["open"], "behavior": "speak", "speak": "hi",
                "stakes": "trivial_reversible",
            })
        return _NEW_JSON

    spec, errs = await synthesize_capability_spec(
        "say a quip", model_invoke=mi, known_verbs=_KNOWN, match_fn=_NO_MATCH,
    )
    assert spec is not None and spec.id == "fun.coin_flip"
    assert len(calls) == 2               # id collision triggered exactly one repair
    assert "already exists" in calls[1]  # repair prompt carried the real reason


async def test_no_known_verbs_still_works():
    async def mi(_p: str) -> str:
        return _NEW_JSON
    spec, errs = await synthesize_capability_spec(
        "flip a coin", model_invoke=mi, known_verbs=[], match_fn=_NO_MATCH,
    )
    assert spec is not None and errs == []


# --- #4: trigger-collision gate --------------------------------------------

def test_find_example_collisions_reports_existing_verb():
    def matcher(text):
        return _Hit("navigate.open_surface") if "browse" in text else None
    hits = find_example_collisions(["open the browse panel", "flip a coin"], matcher)
    assert hits == [("open the browse panel", "navigate.open_surface")]


def test_find_example_collisions_tolerates_matcher_errors():
    def boom(_t):
        raise RuntimeError("matcher down")
    assert find_example_collisions(["x"], boom) == []   # never raises


async def test_synthesize_rejects_colliding_example_then_repairs():
    calls: list[str] = []

    async def mi(prompt: str) -> str:
        calls.append(prompt)
        # both attempts return the same speak verb; only the EXAMPLES differ
        examples = ["open the browse panel"] if len(calls) == 1 else ["zap me a quip"]
        return json.dumps({
            "id": "fun.quip", "summary": "say a quip", "examples": examples,
            "behavior": "speak", "speak": "hi", "stakes": "trivial_reversible",
        })

    # matcher: the first example collides with an existing verb, the repaired one is free
    def matcher(text):
        return _Hit("navigate.open_surface") if "browse" in text else None

    spec, errs = await synthesize_capability_spec(
        "make a quip", model_invoke=mi, known_verbs=[], match_fn=matcher,
    )
    assert spec is not None and spec.examples == ["zap me a quip"]
    assert len(calls) == 2
    assert "lose tier-1 dispatch" in calls[1]   # repair carried the collision reason


def test_default_match_fn_is_callable():
    fn = default_match_fn()
    assert callable(fn)
    # against the live registry, a clearly-conversational string shouldn't match a verb
    assert fn("xyzzy not a command") is None


# --- live loader is safe ---------------------------------------------------

def test_live_loader_returns_real_pairs_or_empty():
    pairs = known_verbs_from_registry()
    # In the test process the intent package imports fine → real pairs; the
    # contract is just "list of (id, summary), never raises".
    assert isinstance(pairs, list)
    for p in pairs[:3]:
        assert isinstance(p, tuple) and len(p) == 2
    if pairs:
        ids = [vid for vid, _ in pairs]
        assert ids == sorted(ids)        # sorted by id (stable prompt ordering)
