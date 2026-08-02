"""Pure-logic tests for the numeric-watch substrate + judge parsing.

Everything here runs against plain values and recorded HTML fixtures —
no DB, no network, no model. The state machine is exercised as
sequences (spec §13.1): feed readings in, assert the verdict stream.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "watch_pages"


def _page(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# ── Extraction ladder ────────────────────────────────────────────────────


def test_extract_jsonld_price():
    from augmentum.companion_runtime.metrics import extract_value
    ext = extract_value(_page("product_jsonld.html"))
    assert ext.value == 549.99
    assert ext.method == "json-ld"
    assert "549.99" in ext.evidence


def test_extract_ambiguous_two_prices_is_error_not_guess():
    """changedetection.io's MoreThanOnePriceFound rule: distinct
    candidates → ambiguous, never a coin flip."""
    from augmentum.companion_runtime.metrics import extract_value
    ext = extract_value(_page("product_two_prices.html"))
    assert ext.ambiguous
    assert ext.value is None
    assert sorted(ext.candidates) == [449.0, 649.0]


def test_extract_pattern_fallback_when_no_structured_data():
    from augmentum.companion_runtime.metrics import extract_value
    ext = extract_value(_page("product_no_structured.html"), hint="Price")
    assert ext.value == 129.5
    assert ext.method == "pattern"


def test_extract_pattern_handles_currency_and_thousands():
    """The '$29.99 → 2999' parse-bug class, pinned forever."""
    from augmentum.companion_runtime.metrics import (
        extract_near_hint,
        to_scaled_int,
    )
    ext = extract_near_hint("Total cost: $1,299.99 incl. tax", "cost")
    assert ext.value == 1299.99
    assert to_scaled_int(29.99) == 2999
    assert to_scaled_int(19.999) == 2000  # round, don't truncate


def test_extract_nothing_returns_none():
    from augmentum.companion_runtime.metrics import extract_value
    ext = extract_value("<html><body>words only, no numbers</body></html>")
    assert ext.value is None and not ext.ambiguous


def test_pinned_method_tried_first():
    """price-data-follower promotion: a pinned rung wins when it still
    works, even when another rung would also bind."""
    from augmentum.companion_runtime.metrics import extract_value
    page = _page("product_jsonld.html")  # both rungs can bind here
    ext = extract_value(page, hint="Price", pinned_method="pattern")
    assert ext.method == "pattern"


# ── Quarantine / confirm / hysteresis state machine ──────────────────────


def _run_sequence(readings, condition=None, confirm=2, quarantine=60.0):
    from augmentum.companion_runtime.metrics import classify_reading
    state: dict = {}
    out = []
    for r in readings:
        v = classify_reading(
            r, state=state, condition=condition,
            quarantine_pct=quarantine, confirm_readings=confirm,
        )
        state = v.state
        out.append(v)
    return out


def test_sequence_glitch_is_quarantined_then_recovers():
    """[549, 549, 5.49, 549, 467, 467] — the spec's canonical sequence.
    The 5.49 glitch dies alone in quarantine; the real drop to 467
    confirms over two readings and fires the <500 condition."""
    verdicts = _run_sequence(
        [549.0, 549.0, 5.49, 549.0, 467.0, 467.0],
        condition={"op": "<", "value": 500},
    )
    statuses = [v.status for v in verdicts]
    fires = [v.fire for v in verdicts]
    assert statuses == ["ok", "ok", "quarantined", "ok", "ok", "ok"]
    # 467 is within quarantine bounds (15% off 549) so it's accepted
    # immediately; the condition still needs 2 consecutive readings.
    assert fires == [False, False, False, False, False, True]


def test_sequence_real_crash_confirms_into_new_level():
    """A genuine 90% drop: first reading quarantined, second agreeing
    reading confirms the new level and the condition fires once the
    confirm count is met."""
    verdicts = _run_sequence(
        [549.0, 49.0, 49.0, 49.0],
        condition={"op": "<", "value": 500},
    )
    assert [v.status for v in verdicts] == [
        "ok", "quarantined", "ok", "ok",
    ]
    assert [v.fire for v in verdicts] == [False, False, False, True]


def test_condition_fires_once_not_every_reading():
    verdicts = _run_sequence(
        [467.0, 467.0, 466.0, 465.0],
        condition={"op": "<", "value": 500},
    )
    assert [v.fire for v in verdicts] == [False, True, False, False]


def test_condition_rearms_after_clearing():
    """Boundary flap: fire → clear → re-confirm → fire again. Exactly
    two fires, not four."""
    verdicts = _run_sequence(
        [467.0, 467.0, 510.0, 467.0, 467.0],
        condition={"op": "<", "value": 500},
    )
    assert [v.fire for v in verdicts] == [False, True, False, False, True]


def test_missing_is_distinct_and_resets_nothing():
    from augmentum.companion_runtime.metrics import classify_reading
    state: dict = {}
    v1 = classify_reading(549.0, state=state)
    v2 = classify_reading(None, state=v1.state)
    v3 = classify_reading(549.0, state=v2.state)
    assert v2.status == "missing"
    assert v3.status == "ok"
    assert v3.state["last_accepted"] == 549.0


def test_daily_interval_confirm_of_one_fires_immediately():
    verdicts = _run_sequence(
        [467.0], condition={"op": "<", "value": 500}, confirm=1,
    )
    assert verdicts[0].fire


def test_condition_met_is_code_not_llm():
    from augmentum.companion_runtime.metrics import condition_met
    assert condition_met(467.0, {"op": "<", "value": 500}) is True
    assert condition_met(510.0, {"op": "<", "value": 500}) is False
    assert condition_met(467.0, None) is None
    assert condition_met(467.0, {"op": "between", "value": 500}) is None


# ── Judge verdict parsing + evidence rule ────────────────────────────────


CONTENT = "Price dropped from $549.99 to $467.00 — limited time offer."


def test_judge_verified_suppression_suppresses():
    from augmentum.companion_runtime.watch_judge import parse_verdict
    v = parse_verdict(
        '{"important": false, "reason": "ad copy changed, not price",'
        ' "evidence": "limited time offer"}',
        CONTENT,
    )
    assert v is not None and v.consulted
    assert v.evidence_verified
    assert v.important is False


def test_judge_unverified_suppression_delivers():
    """P7: an LLM claim that doesn't point at bytes in the source cannot
    silence the user's watch."""
    from augmentum.companion_runtime.watch_judge import parse_verdict
    v = parse_verdict(
        '{"important": false, "reason": "just a banner",'
        ' "evidence": "free shipping on orders over $50"}',
        CONTENT,
    )
    assert v is not None
    assert not v.evidence_verified
    assert v.important is True          # delivery decision overridden
    assert v.raw_important is False     # the model's own answer, recorded


def test_judge_unverified_importance_still_delivers():
    from augmentum.companion_runtime.watch_judge import parse_verdict
    v = parse_verdict(
        '{"important": true, "reason": "price moved",'
        ' "evidence": "not actually in the content"}',
        CONTENT,
    )
    assert v is not None and v.important is True


def test_judge_evidence_check_is_whitespace_tolerant():
    from augmentum.companion_runtime.watch_judge import parse_verdict
    v = parse_verdict(
        '{"important": false, "reason": "x",'
        ' "evidence": "dropped from  $549.99   to $467.00"}',
        CONTENT,
    )
    assert v is not None and v.evidence_verified


def test_judge_garbage_reply_returns_none_for_fail_open():
    from augmentum.companion_runtime.watch_judge import parse_verdict
    assert parse_verdict("I think it matters a lot!", CONTENT) is None
    assert parse_verdict('{"no_important_key": 1}', CONTENT) is None
    assert parse_verdict("", CONTENT) is None


@pytest.mark.asyncio
async def test_judge_change_fails_open_without_model():
    """No usable backend → important=True, consulted=False. The watch
    delivers; the run row shows the judge never weighed in."""
    from unittest.mock import MagicMock

    from augmentum.companion_runtime.watch_judge import judge_change

    runtime = MagicMock()  # tiers.primary will raise on this
    v = await judge_change(
        runtime, intent="only price changes", diff_content=CONTENT,
    )
    assert v.important is True
    assert v.consulted is False
