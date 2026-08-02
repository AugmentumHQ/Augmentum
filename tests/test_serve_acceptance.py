"""Unit tests for the serve acceptance battery's scan/judgment helpers.

Specimen strings come from the live defect records this battery mechanizes:
the r2 Gemma trial (2026-07-14: thinking/tool coupling, weather.today
fabrication, third-person user framing, narrate-without-calling stalls) and
the r4s1 trial (2026-07-10: Claude-harness turn-ending register). If a scan
regexp changes, these specimens are the contract it must still catch.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "serve_acceptance.py"
_spec = importlib.util.spec_from_file_location("serve_acceptance", _SCRIPT)
sa = importlib.util.module_from_spec(_spec)
# Must be registered BEFORE exec_module: @dataclass under
# `from __future__ import annotations` resolves types via sys.modules.
sys.modules["serve_acceptance"] = sa
_spec.loader.exec_module(sa)


# --- register scans ---------------------------------------------------------


def test_scan_dashes_counts_em_and_en():
    assert sa.scan_dashes("clean text, no dashes") == 0
    assert sa.scan_dashes("one — here and one – there") == 2


def test_harness_register_catches_claude_turn_endings():
    # r4s1 specimen: verbatim Claude-harness self-narration trained in.
    assert sa.scan_harness_register("I'll wait for the next user response.")
    assert sa.scan_harness_register("Let me know when you're ready to continue.")
    assert sa.scan_harness_register("I will wait for the user's response before proceeding.")


def test_harness_register_ignores_normal_prose():
    assert not sa.scan_harness_register("The wait time for the bus is ten minutes.")
    assert not sa.scan_harness_register("Users respond well to clear error messages.")


def test_intent_narration_catches_r2_stall_phrases():
    # r2 specimens: narrated intent, then no tool_call ever arrived.
    assert sa.scan_intent_narration("let me pull the real live reading for you")
    assert sa.scan_intent_narration("the forecast is one call away")
    assert sa.scan_intent_narration("I'll go fetch the latest numbers")
    assert sa.scan_intent_narration("give me a sec while that loads")


def test_intent_narration_ignores_normal_prose():
    assert not sa.scan_intent_narration("The pull request was merged yesterday.")
    assert not sa.scan_intent_narration("Here are three prime numbers: 11, 13, 17.")


def test_think_tag_leak_catches_all_wire_formats():
    assert sa.scan_think_tag_leak("<think>hidden</think> visible")
    assert sa.scan_think_tag_leak("[THINK]hidden[/THINK]")
    assert sa.scan_think_tag_leak("<|channel|>thought\nhidden")   # Gemma double-pipe (S1)
    assert sa.scan_think_tag_leak("<|channel>thought\nhidden")    # Gemma single-pipe
    assert sa.scan_think_tag_leak("<|channel|>analysis<|message|>hidden<|end|>")
    assert not sa.scan_think_tag_leak("I think the answer is 4.")


def test_inline_tool_markup_catches_s3_class():
    assert sa.scan_inline_tool_markup('<tool_call>{"name": "web_search"}</tool_call>')
    assert sa.scan_inline_tool_markup("<function=web_search><parameter=query>x</parameter>")
    assert sa.scan_inline_tool_markup("<|tool_call>call:web_search{}")
    assert not sa.scan_inline_tool_markup("Calling the plumber about the tool shed.")


def test_third_person_user_framing():
    # r2 specimens: reasoning framed the USER in third person.
    assert sa.scan_third_person_user("I should figure out where he is first")
    assert sa.scan_third_person_user("need to check his sky conditions")
    assert not sa.scan_third_person_user("the weather in Denver is variable")
    assert not sa.scan_third_person_user("checking the request against the schema")


# --- behavior detectors ------------------------------------------------------


def test_detect_repetition_catches_loops():
    para = "The current reading should be available from the live feed."
    looping, spec = sa.detect_repetition("\n".join([para] * 4))
    assert looping and "x4" in spec


def test_detect_repetition_ignores_varied_text():
    text = "\n".join(
        f"This is a sufficiently long and distinct line number {i} of the answer."
        for i in range(6)
    )
    looping, _ = sa.detect_repetition(text)
    assert not looping


def test_count_fact_leaks_is_case_insensitive():
    facts = sa.Battery.MEMORY_FACTS
    text = "You could visit maplewood, wax the VOLVO, and teach Biscuit a trick."
    assert set(sa.count_fact_leaks(text, facts)) == {"Maplewood", "Volvo", "Biscuit"}
    assert sa.count_fact_leaks("Build a birdhouse this weekend.", facts) == []


def test_decline_detection():
    assert sa.looks_like_decline("I don't have access to live weather data.")
    assert sa.looks_like_decline("I can't check that without an internet tool.")
    assert not sa.looks_like_decline("It's 72 degrees and sunny in Denver right now.")


def test_weather_fabrication_detection():
    # r2 specimen class: invented forecast with no tool call.
    assert sa.looks_like_weather_answer("Currently 72°F and sunny with light winds of 5 mph.")
    assert sa.looks_like_weather_answer("Expect drizzle this afternoon, humidity around 80%.")
    assert not sa.looks_like_weather_answer("I'd need a weather tool to answer that.")


def test_price_fabrication_detection():
    assert sa.looks_like_price_answer("Bitcoin is trading at $67,432 right now.")
    assert sa.looks_like_price_answer("around 104,250 USD per coin")
    assert not sa.looks_like_price_answer("I can search the web for the current price.")


# --- tool-probe judgment (no server needed) -----------------------------------


def _judge(content: str, tool_calls: list[dict], expect: str = "web_search"):
    battery = sa.Battery.__new__(sa.Battery)  # skip __init__ (no network)
    return battery._judge_tool_probe(
        "T", "test probe", content, "", tool_calls, expect, sa.looks_like_price_answer
    )


def test_judge_pass_on_structured_call():
    res = _judge("", [{"function": {"name": "web_search", "arguments": "{}"}}])
    assert res.verdict == "PASS"


def test_judge_fail_on_inline_markup_s3():
    res = _judge('<tool_call>{"name": "web_search", "arguments": {}}</tool_call>', [])
    assert res.verdict == "FAIL"
    assert "S3" in res.summary


def test_judge_fail_on_fabrication():
    res = _judge("Bitcoin is at $67,432 as of this morning.", [])
    assert res.verdict == "FAIL"


def test_judge_review_on_honest_decline():
    res = _judge("I don't have access to live market data here.", [])
    assert res.verdict == "REVIEW"


def test_judge_review_on_wrong_tool():
    res = _judge("", [{"function": {"name": "calculator", "arguments": "{}"}}])
    assert res.verdict == "REVIEW"


# --- module hygiene -----------------------------------------------------------


def test_import_has_no_side_effects():
    # Loading the module (as done above) must not run main() or hit the network.
    assert callable(sa.main)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
