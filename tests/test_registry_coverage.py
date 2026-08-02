"""Unit tests for registry_coverage row parsing and phantom suggestion.

The parser must handle every row shape in the tree: ShareGPT waves (inline
tool_call JSON / Qwen XML / Gemma call markup + the primer's ``tools:`` line),
messages rows with structured tool_calls (BOTH the wrapped and the flat shape
— the flat one silently escaped trace_transform v1), and v3 capture traces.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "registry_coverage.py"
_spec = importlib.util.spec_from_file_location("registry_coverage", _SCRIPT)
rc = importlib.util.module_from_spec(_spec)
sys.modules["registry_coverage"] = rc
_spec.loader.exec_module(rc)


# --- extraction primitives ----------------------------------------------------


def test_structured_names_wrapped_and_flat():
    calls = [
        {"function": {"name": "web_search", "arguments": "{}"}},
        {"name": "calculator", "arguments": {}},          # flat — the v1 hole
        {"id": "x"},                                       # nameless: ignored
    ]
    assert rc._names_from_structured(calls) == ["web_search", "calculator"]


def test_text_names_all_markups():
    text = (
        '<tool_call>{"name": "wikipedia", "arguments": {"query": "x"}}</tool_call>'
        "\n<function=python_exec><parameter=code>1</parameter></function>"
        "\n<|tool_call>call:weather.today{}"
    )
    assert rc._names_from_text(text) == ["wikipedia", "python_exec", "weather.today"]


def test_reasoning_detection_empty_think_is_off():
    # Qwen3.5 no-think = EMPTY <think> block, and that emptiness IS the
    # trained thinking-off signal — must classify as OFF.
    assert not rc._has_reasoning_text("<think>\n\n</think>\n\nPlain answer.")
    assert rc._has_reasoning_text("<think>let me check the rate</think>ok")
    assert rc._has_reasoning_text("<|channel>thought\nreasoning here")
    assert not rc._has_reasoning_text("no reasoning markup at all")


# --- whole-row parsing ---------------------------------------------------------


def test_parse_sharegpt_row_with_primer_exposure():
    row = {
        "conversations": [
            {"from": "system", "value": ":C\n[now: 2026-07-15 12:00]\ntools: web_search, wikipedia"},
            {"from": "human", "value": "look it up"},
            {"from": "gpt", "value": '<think>need the wiki</think>\n<tool_call>{"name": "wikipedia", "arguments": {}}</tool_call>'},
            {"from": "tool", "value": "result"},
            {"from": "gpt", "value": "answer"},
        ],
        "metadata": {"tag": ":C"},
    }
    u = rc.parse_row(row)
    assert u.tools_used == {"wikipedia"}
    assert u.tools_available == {"web_search", "wikipedia"}
    assert u.thinking is True
    assert u.mode == ":C"


def test_parse_messages_row_structured():
    row = {
        "messages": [
            {"role": "system", "content": ":-"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": "file_read", "arguments": "{}"}}]},
        ],
    }
    u = rc.parse_row(row)
    assert u.tools_used == {"file_read"}
    assert u.thinking is False


def test_parse_v3_trace():
    row = {
        "mode": "coder",
        "tools_used": ["shell_exec"],
        "tools_available": ["shell_exec", "file_read", "code_edit"],
        "final_thinking": "planning the fix",
        "chain": [
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": "file_write"}]},
        ],
    }
    u = rc.parse_row(row)
    assert u.tools_used == {"shell_exec", "file_write"}
    assert u.tools_available == {"shell_exec", "file_read", "code_edit"}
    assert u.thinking is True
    assert u.mode == "coder"


# --- phantom suggestion ---------------------------------------------------------


def test_suggest_near_catches_sanitization_class():
    known = {"web_search", "image_generation", "schedule_request"}
    assert rc.suggest_near("web.search", known) == "web_search"
    assert rc.suggest_near("image_generate", known) == "image_generation"
    assert rc.suggest_near("totally_invented_verb", known) == ""


# --- scan end-to-end on a temp corpus -------------------------------------------


def test_scan_counts_and_phantoms(tmp_path):
    rows = [
        {"conversations": [
            {"from": "system", "value": ":C\ntools: web_search"},
            {"from": "gpt", "value": '<tool_call>{"name": "web_search", "arguments": {}}</tool_call>'},
        ]},
        {"conversations": [
            {"from": "gpt", "value": '<think>hm</think><tool_call>{"name": "web_search", "arguments": {}}</tool_call>'},
        ]},
        {"conversations": [
            {"from": "gpt", "value": '<tool_call>{"name": "ghost_tool", "arguments": {}}</tool_call>'},
        ]},
    ]
    f = tmp_path / "corpus" / "waves.jsonl"
    f.parent.mkdir()
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    stats = {"web_search": rc.ToolStats(sources={"live"})}
    total, phantoms = rc.scan([tmp_path / "corpus"], stats, {"web_search"})

    assert total == 3
    ws = stats["web_search"]
    assert ws.rows == 2
    assert ws.think_on == 1 and ws.think_off == 1
    assert ws.exposed_count == 1                      # one primer tools: line
    assert "ghost_tool" in phantoms
    assert phantoms["ghost_tool"]["rows"] == 1


def test_scan_usage_before_exposure_is_not_phantom(tmp_path):
    # Regression: a tool whose USAGE row precedes the exposure row that first
    # names it must still classify as covered (classification runs after the
    # full walk, not order-dependently mid-scan).
    rows = [
        {"conversations": [{"from": "gpt",
            "value": '<tool_call>{"name": "late_tool", "arguments": {}}</tool_call>'}]},
        {"tools_available": ["late_tool"], "tools_used": [], "mode": "coder"},
    ]
    f = tmp_path / "c" / "rows.jsonl"
    f.parent.mkdir()
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    stats: dict = {}
    total, phantoms = rc.scan([tmp_path / "c"], stats, set())
    assert total == 2
    assert "late_tool" not in phantoms
    assert stats["late_tool"].rows == 1


def test_scan_skips_bak_files(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "wave.jsonl.bak").write_text(
        json.dumps({"conversations": [{"from": "gpt",
            "value": '<tool_call>{"name": "web_search", "arguments": {}}</tool_call>'}]}),
        encoding="utf-8")
    stats = {"web_search": rc.ToolStats(sources={"live"})}
    total, _ = rc.scan([d], stats, {"web_search"})
    assert total == 0 and stats["web_search"].rows == 0


def test_import_has_no_side_effects():
    assert callable(rc.main)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
