"""Guards the invariant: load-bearing verbs are never budget-clipped from the
companion tool roster.

tonight's failure (2026-07-27): "launch a coder run to make a game" produced a
chat reply and no run, because ``coder.delegate`` — the SOLE path from the
companion to a coder run — scored ~0 relevance on the turn and got clipped into
the deferred tail of ``enumerate_tools``. A clipped tool is invisible AND
uncallable (it's neither in the prompt roster nor the assembled native tools),
so the model literally could not act. ``fanout.always_offer`` makes such verbs
budget- and family-cap-exempt; these tests pin that.
"""
from __future__ import annotations

# Importing the intent package registers the builtin actions (coder.delegate
# among them) into REGISTRY — enumerate_tools reads that registry.
import augmentum.intent  # noqa: F401
from augmentum.companion_runtime.tools import enumerate_tools


def _roster_names(turn_text: str, budget: int) -> list[str]:
    return [t["name"] for t in enumerate_tools(turn_text, context_budget_chars=budget)]


def test_coder_delegate_marked_load_bearing():
    from augmentum.intent.registry import REGISTRY
    action = REGISTRY.get("coder.delegate")
    assert action is not None, "coder.delegate not registered"
    assert action.fanout.always_offer is True


def test_coder_delegate_survives_clip_on_unrelated_turn():
    # Realistic floor budget (1200 = _TOOL_ROSTER_CHAR_BUDGET, matches the live
    # logged value) + a turn with zero coding relevance. Without the exemption
    # coder.delegate sorts to the deferred tail and vanishes; with it, it stays.
    names = _roster_names("what's the weather in tokyo tomorrow", 1200)
    assert "coder.delegate" in names, (
        "coder.delegate was clipped from the roster — the exact tonight failure"
    )


def test_coder_delegate_present_even_at_tiny_registry_budget():
    # Even when the budget only affords the catalogue + a verb or two, the
    # load-bearing verb must remain (it's the guarantee, not a nice-to-have).
    names = _roster_names("tell me a joke about cats", 1300)
    assert "coder.delegate" in names


def test_unrelated_turn_still_clips_the_long_tail():
    # Sanity: the budget IS doing its job — an unrelated turn does not carry the
    # entire registry. (If nothing ever clipped, the test above would be vacuous.)
    small = _roster_names("what's the weather in tokyo tomorrow", 1200)
    large = _roster_names("what's the weather in tokyo tomorrow", 100_000)
    assert len(small) < len(large), "expected the char budget to clip some verbs"
    # ...but the load-bearing verb is in BOTH.
    assert "coder.delegate" in small and "coder.delegate" in large
