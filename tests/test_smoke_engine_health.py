"""Pin the SearXNG engine-health tracker + resilient search helper.

Task #29: SearXNG suspends engines internally but never exposes the
remaining suspension time over JSON, so the tool layer keeps its own
estimate. These tests pin:

- reason-aware TTL recording and monotonic-clock expiry
- success (engine appears in results) clears a suspension
- healthy_fallback_engines excludes suspended candidates, preserves
  preference order, and goes empty when everything is down
- searxng_search_resilient: records health, reissues against healthy
  engines only on infra-empty responses, swallows fallback errors
- WebSearchTool integration: dynamic fallback set + structured
  rate-limited result instead of a bare "No results found."
- standing_tasks handlers route through the shared resilient helper
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.tools.engine_health import (
    EngineHealthTracker,
    searxng_search_resilient,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def _response(data: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json = MagicMock(return_value=data)
    resp.raise_for_status = MagicMock()
    return resp


# ─── Tracker semantics ──────────────────────────────────────────────────


class TestTracker:
    def test_suspension_recorded_with_reason_ttl(self):
        clock = FakeClock()
        t = EngineHealthTracker(clock=clock)
        t.record_response({"unresponsive_engines": [
            ["startpage", "CAPTCHA"],
            ["brave", "too many requests"],
            ["yep", "some unknown error"],
        ]})
        assert t.is_suspended("startpage")
        assert t.is_suspended("brave")
        assert t.is_suspended("yep")
        # Unknown-reason default (180s) expires first…
        clock.now += 181
        assert not t.is_suspended("yep")
        # …then "too many requests" (300s)…
        clock.now += 120  # t=1301
        assert not t.is_suspended("brave")
        # …CAPTCHA (600s) outlives both.
        assert t.is_suspended("startpage")
        clock.now += 300  # t=1601
        assert not t.is_suspended("startpage")

    def test_result_engine_clears_suspension(self):
        clock = FakeClock()
        t = EngineHealthTracker(clock=clock)
        t.record_response({"unresponsive_engines": [["brave", "too many requests"]]})
        assert t.is_suspended("brave")
        # Brave shows up in results → demonstrably answering again.
        t.record_response({"results": [{"engine": "brave", "url": "https://x.com"}]})
        assert not t.is_suspended("brave")

    def test_result_engines_list_also_clears(self):
        t = EngineHealthTracker(clock=FakeClock())
        t.record_response({"unresponsive_engines": [["bing", "access denied"]]})
        t.record_response({"results": [{"engines": ["bing", "duckduckgo"]}]})
        assert not t.is_suspended("bing")

    def test_longer_estimate_never_shortened(self):
        clock = FakeClock()
        t = EngineHealthTracker(clock=clock)
        t.record_response({"unresponsive_engines": [["startpage", "CAPTCHA"]]})
        # A later, shorter-TTL report must not pull the estimate in.
        clock.now += 10
        t.record_response({"unresponsive_engines": [["startpage", "some blip"]]})
        clock.now += 200  # past the 190s blip estimate, inside the CAPTCHA one
        assert t.is_suspended("startpage")

    def test_healthy_fallback_engines_order_and_exclusion(self):
        t = EngineHealthTracker(clock=FakeClock())
        assert t.healthy_fallback_engines() == "bing,duckduckgo,wikipedia,brave,mojeek"
        t.record_response({"unresponsive_engines": [
            ["brave", "too many requests"],
            ["mojeek", "access denied"],
        ]})
        assert t.healthy_fallback_engines() == "bing,duckduckgo,wikipedia"

    def test_all_suspended_returns_empty(self):
        t = EngineHealthTracker(clock=FakeClock())
        t.record_response({"unresponsive_engines": [
            [e, "access denied"]
            for e in ("bing", "duckduckgo", "wikipedia", "brave", "mojeek")
        ]})
        assert t.healthy_fallback_engines() == ""

    def test_earliest_retry_seconds(self):
        clock = FakeClock()
        t = EngineHealthTracker(clock=clock)
        assert t.earliest_retry_seconds() is None
        t.record_response({"unresponsive_engines": [
            ["brave", "too many requests"],   # 300s
            ["startpage", "CAPTCHA"],          # 600s — not a fallback candidate
        ]})
        # Only fallback candidates count; brave recovers first.
        assert t.earliest_retry_seconds() == 300
        # Non-candidate suspensions alone → None (nothing the fallback
        # path is waiting on).
        clock.now += 301
        assert t.earliest_retry_seconds() is None

    def test_suspended_summary_shape(self):
        clock = FakeClock()
        t = EngineHealthTracker(clock=clock)
        t.record_response({"unresponsive_engines": [["qwant", "Suspended: access denied"]]})
        summary = t.suspended_summary()
        assert summary == [{
            "engine": "qwant",
            "reason": "Suspended: access denied",
            "retry_in_seconds": 600,
        }]

    def test_malformed_entries_ignored(self):
        t = EngineHealthTracker(clock=FakeClock())
        t.record_response({"unresponsive_engines": [[], None, "bare-string", 42]})
        assert t.suspended_summary() == []
        t.record_response(None)  # type: ignore[arg-type]
        t.record_response({"results": ["not-a-dict"]})


# ─── Resilient search helper ────────────────────────────────────────────


class TestResilientSearch:
    async def test_results_returned_without_fallback(self):
        client = AsyncMock()
        client.get = AsyncMock(return_value=_response(
            {"results": [{"engine": "bing", "url": "https://a.com"}]}
        ))
        data = await searxng_search_resilient(
            client, "http://searxng:8080", "q",
            tracker=EngineHealthTracker(clock=FakeClock()),
        )
        assert len(data["results"]) == 1
        assert client.get.call_count == 1

    async def test_infra_empty_reissues_with_healthy_engines(self):
        tracker = EngineHealthTracker(clock=FakeClock())
        primary = _response({
            "results": [],
            "unresponsive_engines": [["brave", "too many requests"],
                                     ["mojeek", "access denied"]],
        })
        fallback = _response({"results": [{"engine": "bing", "url": "https://a.com"}]})
        client = AsyncMock()
        client.get = AsyncMock(side_effect=[primary, fallback])

        data = await searxng_search_resilient(
            client, "http://searxng:8080", "q", tracker=tracker,
        )
        assert data.get("augmentum_fallback_used") is True
        assert len(data["results"]) == 1
        # The reissue must constrain to engines NOT just observed down.
        fb_params = client.get.call_args_list[1].kwargs["params"]
        assert fb_params["engines"] == "bing,duckduckgo,wikipedia"

    async def test_genuine_no_match_does_not_reissue(self):
        """Empty results with NO unresponsive engines = honest no-match."""
        client = AsyncMock()
        client.get = AsyncMock(return_value=_response(
            {"results": [], "unresponsive_engines": []}
        ))
        data = await searxng_search_resilient(
            client, "http://searxng:8080", "qzx",
            tracker=EngineHealthTracker(clock=FakeClock()),
        )
        assert data["results"] == []
        assert client.get.call_count == 1

    async def test_all_candidates_suspended_skips_reissue(self):
        tracker = EngineHealthTracker(clock=FakeClock())
        tracker.record_response({"unresponsive_engines": [
            [e, "access denied"]
            for e in ("bing", "duckduckgo", "wikipedia", "brave", "mojeek")
        ]})
        client = AsyncMock()
        client.get = AsyncMock(return_value=_response(
            {"results": [], "unresponsive_engines": [["bing", "access denied"]]}
        ))
        data = await searxng_search_resilient(
            client, "http://searxng:8080", "q", tracker=tracker,
        )
        assert data["results"] == []
        assert client.get.call_count == 1  # no doomed second query

    async def test_fallback_network_error_returns_primary(self):
        tracker = EngineHealthTracker(clock=FakeClock())
        primary = _response({
            "results": [],
            "unresponsive_engines": [["brave", "too many requests"]],
        })
        client = AsyncMock()
        client.get = AsyncMock(side_effect=[primary, RuntimeError("boom")])
        data = await searxng_search_resilient(
            client, "http://searxng:8080", "q", tracker=tracker,
        )
        assert data["results"] == []
        assert "augmentum_fallback_used" not in data


# ─── WebSearchTool integration ──────────────────────────────────────────


class TestWebSearchToolIntegration:
    def _make_tool(self, client, tracker):
        from augmentum.tools.web_search import WebSearchTool
        return WebSearchTool(client, base_url="http://searxng:8080",
                             tracker=tracker)

    async def test_fallback_uses_dynamic_healthy_set(self):
        tracker = EngineHealthTracker(clock=FakeClock())
        primary = _response({
            "results": [],
            "unresponsive_engines": [["brave", "too many requests"],
                                     ["bing", "too many requests"]],
        })
        fallback = _response({"results": []})
        client = AsyncMock()
        client.get = AsyncMock(side_effect=[primary, fallback])
        tool = self._make_tool(client, tracker)

        await tool.execute(query="anything")
        fb_params = client.get.call_args_list[1].kwargs["params"]
        # brave + bing just got suspended → excluded from the reissue.
        assert fb_params["engines"] == "duckduckgo,wikipedia,mojeek"

    async def test_rate_limited_result_is_structured(self):
        """Infra blackout → explicit rate-limited message + retry estimate,
        not a bare 'No results found.' the model would relay as no-match."""
        tracker = EngineHealthTracker(clock=FakeClock())
        primary = _response({
            "results": [],
            "unresponsive_engines": [["bing", "too many requests"]],
        })
        fallback = _response({"results": []})
        client = AsyncMock()
        client.get = AsyncMock(side_effect=[primary, fallback])
        tool = self._make_tool(client, tracker)

        result = await tool.execute(query="latest news")
        assert result.success is True
        assert "rate-limited" in result.output.lower()
        assert result.metadata["rate_limited"] is True
        assert result.metadata["retry_in_seconds"] == 300
        assert result.metadata["engines_unavailable"]

    async def test_genuine_no_match_keeps_plain_message(self):
        tracker = EngineHealthTracker(clock=FakeClock())
        client = AsyncMock()
        client.get = AsyncMock(return_value=_response(
            {"results": [], "unresponsive_engines": []}
        ))
        tool = self._make_tool(client, tracker)
        result = await tool.execute(query="qzxv unfindable")
        assert "no results" in result.output.lower()
        assert "rate_limited" not in (result.metadata or {})


# ─── Standing-task wiring ───────────────────────────────────────────────


class TestStandingTasksWiring:
    """recurring_search + briefing must route through the shared
    resilient helper so they record health and reissue against the
    engines most likely to answer (the user-requested integration)."""

    def test_handlers_use_resilient_helper(self):
        src = Path("augmentum/companion_runtime/standing_tasks.py").read_text(
            encoding="utf-8",
        )
        # Both call sites converted (recurring_search + briefing)…
        assert src.count("searxng_search_resilient(") >= 2
        # …and no raw SearXNG GET remains anywhere in the module.
        assert "searxng_base.rstrip('/')}/search" not in src

    def test_recurring_search_surfaces_rate_limit(self):
        src = Path("augmentum/companion_runtime/standing_tasks.py").read_text(
            encoding="utf-8",
        )
        assert "rate-limited" in src
