"""Regression tests for the extended repetition-tracking ledger.

Original tracking (2026-04-20) only covered file/shell/grep tools.
Browser/http/db probes were untracked, so the model could loop on
``browser_click('.connect-btn')`` indefinitely — the cap=5 hard-block
never fired. the dogfood 2026-05-30 trace showed exactly this: ~10
consecutive identical browser_click calls with no intermediate
progress, and the agent had no signal to break out.

These tests verify:
1. New tool families get an intent key derived from their distinguishing arg
2. ``hit_repeat_cap`` fires at the configured cap for the new tools
3. Tools with empty/missing args fall back to "" (skip tracking) cleanly
4. The existing tools still work (no regression in path/command/query)
"""

from __future__ import annotations

from augmentum.loops.ledger import (
    TRACKED_TOOLS_BY_COMMAND,
    TRACKED_TOOLS_BY_EXPRESSION,
    TRACKED_TOOLS_BY_PATH,
    TRACKED_TOOLS_BY_QUERY,
    TRACKED_TOOLS_BY_QUERY_DB,
    TRACKED_TOOLS_BY_REQUEST,
    TRACKED_TOOLS_BY_SELECTOR,
    TRACKED_TOOLS_BY_URL,
    ObservationLedger,
    _intent_key,
)

# ---------------------------------------------------------------------------
# Intent-key derivation per tool family
# ---------------------------------------------------------------------------

class TestIntentKeyDerivation:
    def test_browser_click_uses_selector(self):
        # The exact pattern from the live trace.
        assert _intent_key("browser_click", {"selector": ".connect-btn"}) == ".connect-btn"

    def test_browser_type_combines_selector_and_text(self):
        # Typing different text into the same field counts as distinct
        # intents so the model can fill a form one step at a time.
        k1 = _intent_key("browser_type", {"selector": "#chat", "text": "hello"})
        k2 = _intent_key("browser_type", {"selector": "#chat", "text": "world"})
        assert k1 != k2
        assert k1 == "#chat|hello"
        assert k2 == "#chat|world"

    def test_browser_type_without_text_collapses_to_selector(self):
        assert _intent_key("browser_type", {"selector": "#chat"}) == "#chat"

    def test_browser_evaluate_uses_expression(self):
        assert _intent_key("browser_evaluate", {"expression": "window.foo"}) == "window.foo"

    def test_browser_open_uses_url(self):
        assert _intent_key("browser_open", {"url": "http://x"}) == "http://x"

    def test_http_request_combines_method_and_url(self):
        # GET and POST to the same URL are legitimately different.
        get = _intent_key("http_request", {"method": "GET", "url": "http://x"})
        post = _intent_key("http_request", {"method": "POST", "url": "http://x"})
        assert get != post
        assert get == "GET http://x"
        assert post == "POST http://x"

    def test_http_request_defaults_method_to_GET(self):
        assert _intent_key("http_request", {"url": "http://x"}) == "GET http://x"

    def test_db_inspect_uses_query(self):
        assert _intent_key(
            "db_inspect", {"query": "SELECT 1"},
        ) == "SELECT 1"

    def test_unknown_tool_returns_empty(self):
        assert _intent_key("totally_made_up_tool", {"selector": "x"}) == ""

    def test_missing_args_return_empty(self):
        # Cleanly skips tracking rather than producing a "" key that
        # would collide with every other untrackable call.
        assert _intent_key("browser_click", {}) == ""
        assert _intent_key("http_request", {}) == ""


# ---------------------------------------------------------------------------
# Pre-existing tool families still work (regression check)
# ---------------------------------------------------------------------------

class TestExistingTrackingStillWorks:
    def test_file_read_still_tracks_path(self):
        assert _intent_key("file_read", {"path": "/a.py"}) == "/a.py"

    def test_shell_exec_still_tracks_command(self):
        assert _intent_key("shell_exec", {"command": "ls"}) == "ls"

    def test_code_grep_still_tracks_pattern(self):
        assert _intent_key("code_grep", {"pattern": "foo"}) == "foo"


# ---------------------------------------------------------------------------
# The full integration: hit_repeat_cap fires on the new tracked tools
# ---------------------------------------------------------------------------

class TestRepeatCapFiresOnNewTools:
    """The whole point of extending the tracked sets is so the existing
    hit_repeat_cap=5 hard-block fires for browser/http loops. Reproduces
    the dogfood .connect-btn trace pattern as a regression check.
    """

    def test_browser_click_repeat_cap_fires_at_5(self):
        ledger = ObservationLedger()
        for i in range(5):
            ledger.record_tool_call(
                tool_name="browser_click",
                tool_input={"selector": ".connect-btn"},
                iteration=i,
            )
        # Cap fires at count >= 5 — the model would now see a
        # validation_error injection telling it to stop looping.
        assert ledger.hit_repeat_cap(
            tool_name="browser_click",
            tool_input={"selector": ".connect-btn"},
            cap=5,
        )

    def test_browser_click_different_selectors_dont_collide(self):
        ledger = ObservationLedger()
        # Different selectors → different intents → independent counters
        for _ in range(5):
            ledger.record_tool_call(
                tool_name="browser_click",
                tool_input={"selector": ".connect-btn"},
                iteration=0,
            )
        ledger.record_tool_call(
            tool_name="browser_click",
            tool_input={"selector": ".logout-btn"},
            iteration=1,
        )
        # connect-btn caps, logout-btn doesn't
        assert ledger.hit_repeat_cap(
            tool_name="browser_click",
            tool_input={"selector": ".connect-btn"},
            cap=5,
        )
        assert not ledger.hit_repeat_cap(
            tool_name="browser_click",
            tool_input={"selector": ".logout-btn"},
            cap=5,
        )

    def test_http_request_repeat_cap_distinguishes_methods(self):
        ledger = ObservationLedger()
        for _ in range(5):
            ledger.record_tool_call(
                tool_name="http_request",
                tool_input={"method": "GET", "url": "http://x/health"},
                iteration=0,
            )
        assert ledger.hit_repeat_cap(
            tool_name="http_request",
            tool_input={"method": "GET", "url": "http://x/health"},
            cap=5,
        )
        # POST to same URL is a different intent — not capped
        assert not ledger.hit_repeat_cap(
            tool_name="http_request",
            tool_input={"method": "POST", "url": "http://x/health"},
            cap=5,
        )

    def test_browser_evaluate_repeat_cap_on_same_expression(self):
        # The trace's other loop — re-evaluating the same expression
        # after navigations destroyed the context.
        ledger = ObservationLedger()
        for _ in range(5):
            ledger.record_tool_call(
                tool_name="browser_evaluate",
                tool_input={"expression": "document.querySelector('.x').textContent"},
                iteration=0,
            )
        assert ledger.hit_repeat_cap(
            tool_name="browser_evaluate",
            tool_input={"expression": "document.querySelector('.x').textContent"},
            cap=5,
        )

    def test_empty_args_dont_falsely_cap(self):
        # A tool called with missing/empty distinguishing args has
        # _intent_key="" → not tracked → never caps. Confirms our
        # opt-out on degenerate input doesn't break the model.
        ledger = ObservationLedger()
        for _ in range(10):
            ledger.record_tool_call(
                tool_name="browser_click",
                tool_input={},  # no selector
                iteration=0,
            )
        assert not ledger.hit_repeat_cap(
            tool_name="browser_click",
            tool_input={},
            cap=5,
        )


# ---------------------------------------------------------------------------
# Tracked-set membership sanity
# ---------------------------------------------------------------------------

class TestTrackedSetMembership:
    """Make sure every browser/http/db tool we expect to track is
    actually in the set. Drift check — if a tool gets renamed or a
    new mutation tool lands, the test breaks loudly.
    """

    def test_all_browser_interaction_tools_tracked(self):
        for name in (
            "browser_click", "browser_type", "browser_verify",
            # Wave-2 primitives (2026-07-02) — same repeat-loop risk.
            "browser_wait", "browser_extract", "browser_fill_form",
        ):
            assert name in TRACKED_TOOLS_BY_SELECTOR, f"{name} not in selector set"

    def test_all_browser_navigation_tools_tracked(self):
        for name in ("browser_open", "browser_snapshot", "browser_screenshot"):
            assert name in TRACKED_TOOLS_BY_URL, f"{name} not in url set"

    def test_browser_evaluate_tracked(self):
        assert "browser_evaluate" in TRACKED_TOOLS_BY_EXPRESSION

    def test_http_request_tracked(self):
        assert "http_request" in TRACKED_TOOLS_BY_REQUEST

    def test_db_inspect_tracked(self):
        assert "db_inspect" in TRACKED_TOOLS_BY_QUERY_DB

    def test_no_overlap_between_sets(self):
        # Each tool should be classified into exactly one family.
        all_sets = [
            TRACKED_TOOLS_BY_PATH,
            TRACKED_TOOLS_BY_COMMAND,
            TRACKED_TOOLS_BY_QUERY,
            TRACKED_TOOLS_BY_URL,
            TRACKED_TOOLS_BY_SELECTOR,
            TRACKED_TOOLS_BY_EXPRESSION,
            TRACKED_TOOLS_BY_REQUEST,
            TRACKED_TOOLS_BY_QUERY_DB,
        ]
        seen: set[str] = set()
        for s in all_sets:
            overlap = s & seen
            assert not overlap, f"Tool(s) in multiple tracked sets: {overlap}"
            seen |= s
