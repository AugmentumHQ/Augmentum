"""Earned Understanding P1: the Calibrated Voice (tier → register).

Recalled memories carry their earned-tier confidence cue so the companion
speaks a CORE fact plainly and an unproven impression tentatively, instead of
asserting everything at the same flat confidence. See
docs/superpowers/specs/2026-06-20-earned-understanding-design.md.
"""

from __future__ import annotations

from augmentum.memory.models import Memory, MemoryTier, MemoryType
from augmentum.memory.register import (
    HONEST_EMPTY_NOTE,
    calibrated_bullets,
    register_label,
)


def _mem(content, tier):
    return Memory(
        id="m", user_id="u1", content=content,
        memory_type=MemoryType.FACT, tier=tier,
        created_at="2026-06-01T00:00:00+00:00",
    )


def test_register_label_per_tier():
    assert register_label(MemoryTier.CORE) == "certain"
    assert register_label(MemoryTier.ACTIVE) == "fairly sure"
    assert "hedge" in register_label(MemoryTier.PROVISIONAL)
    assert "faded" in register_label(MemoryTier.ARCHIVE)
    # string tiers work too
    assert register_label("core") == "certain"


def test_register_label_rounds_down_on_unknown():
    """Ambiguous/unknown tier defaults to 'fairly sure', never 'certain'."""
    assert register_label("") == "fairly sure"
    assert register_label("bogus") == "fairly sure"
    assert register_label(None) == "fairly sure"


def test_calibrated_bullets_prefixes_each_with_register():
    rows = [
        _mem("User's name is Matt", MemoryTier.CORE),
        _mem("User mentioned liking jazz", MemoryTier.ACTIVE),
    ]
    out = calibrated_bullets(rows)
    assert "- [certain] User's name is Matt" in out
    assert "- [fairly sure] User mentioned liking jazz" in out


def test_calibrated_bullets_handles_dict_rows_and_caps():
    rows = [{"content": "fact A", "tier": "core"}, {"content": "", "tier": "active"}]
    out = calibrated_bullets(rows, limit=6)
    assert "- [certain] fact A" in out
    # empty content skipped, not rendered as a blank bullet
    assert out.count("\n- ") == 0 and out.count("- ") == 1


def test_calibrated_bullets_respects_limit():
    rows = [_mem(f"fact {i}", MemoryTier.ACTIVE) for i in range(10)]
    out = calibrated_bullets(rows, limit=3)
    assert len(out.splitlines()) == 3


def test_honest_empty_note_is_directive():
    # Must tell the model to SAY it doesn't have it, not invent.
    assert "don't have" in HONEST_EMPTY_NOTE
    assert "guessing" in HONEST_EMPTY_NOTE or "invent" in HONEST_EMPTY_NOTE


def test_calibrated_bullets_truncates_long_content():
    rows = [_mem("x" * 500, MemoryTier.ACTIVE)]
    out = calibrated_bullets(rows, max_chars=240)
    # prefix + 240 chars, not 500
    assert "x" * 240 in out and "x" * 241 not in out
