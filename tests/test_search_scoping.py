"""Tests for search context scoping and tool settings API."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from augmentum.config import settings
from augmentum.modes.analytical.prompts import (
    SEARCH_CONTEXT_SECTION,
    scope_search_context,
)

# --- scope_search_context tests ---


class TestScopeSearchContext:
    """Tests for per-phase search context scoping."""

    SAMPLE_CONTEXT = SEARCH_CONTEXT_SECTION.format(
        search_results=(
            'Search: "test query"\n\n'
            "[1] Example Title\n"
            "    URL: https://example.com/page1\n"
            "    This is the snippet text for result one.\n\n"
            "[2] Another Title\n"
            "    URL: https://example.com/page2\n"
            "    This is the snippet text for result two."
        ),
    )

    def test_empty_context_returns_empty(self):
        assert scope_search_context("", "apply") == ""
        assert scope_search_context("", "verify") == ""

    def test_apply_gets_full_context(self):
        result = scope_search_context(self.SAMPLE_CONTEXT, "apply")
        assert result == self.SAMPLE_CONTEXT
        assert "snippet text" in result

    def test_respond_gets_full_context(self):
        result = scope_search_context(self.SAMPLE_CONTEXT, "respond")
        assert result == self.SAMPLE_CONTEXT

    def test_identify_gets_summary(self):
        result = scope_search_context(self.SAMPLE_CONTEXT, "identify")
        # Should get brief summary, not full context
        assert result != self.SAMPLE_CONTEXT
        assert "Search Context" in result
        # Should not contain full snippet text
        assert "snippet text" not in result

    def test_relevant_gets_summary(self):
        result = scope_search_context(self.SAMPLE_CONTEXT, "relevant")
        assert result != self.SAMPLE_CONTEXT
        assert "Search Context" in result
        assert "snippet text" not in result

    def test_verify_gets_urls_only(self):
        result = scope_search_context(self.SAMPLE_CONTEXT, "verify")
        assert "URL: https://example.com/page1" in result
        assert "URL: https://example.com/page2" in result
        assert "[1] Example Title" in result
        assert "[2] Another Title" in result
        # Should NOT contain the full snippet text
        assert "snippet text for result one" not in result
        assert "snippet text for result two" not in result

    def test_verify_has_reference_header(self):
        result = scope_search_context(self.SAMPLE_CONTEXT, "verify")
        assert "Reference URLs" in result

    def test_assess_gets_nothing(self):
        result = scope_search_context(self.SAMPLE_CONTEXT, "assess")
        assert result == ""

    def test_conclude_gets_nothing(self):
        result = scope_search_context(self.SAMPLE_CONTEXT, "conclude")
        assert result == ""

    def test_unknown_phase_gets_nothing(self):
        result = scope_search_context(self.SAMPLE_CONTEXT, "unknown")
        assert result == ""

    def test_verify_no_urls_returns_empty(self):
        context = "Some search context with no URL lines at all."
        result = scope_search_context(context, "verify")
        assert result == ""


# --- Context marking tests ---


class TestContextMarking:
    """Tests for clear provenance marking in search context."""

    def test_web_results_delimited(self):
        formatted = SEARCH_CONTEXT_SECTION.format(search_results="some results")
        assert "--- BEGIN WEB RESULTS ---" in formatted
        assert "--- END WEB RESULTS ---" in formatted
        assert "some results" in formatted

    def test_source_label_present(self):
        formatted = SEARCH_CONTEXT_SECTION.format(search_results="data")
        assert "[SOURCE: WEB SEARCH]" in formatted

    def test_search_results_marked_as_real_data(self):
        formatted = SEARCH_CONTEXT_SECTION.format(search_results="data")
        assert "REAL, CURRENT data" in formatted


# --- Tool settings API tests ---


class TestToolSettingsAPI:
    @pytest.fixture
    def app(self):
        from unittest.mock import AsyncMock, MagicMock

        from fastapi import FastAPI

        from augmentum.proxy.config_routes import router

        app = FastAPI()
        app.include_router(router)
        app.state.settings_store = MagicMock()
        app.state.settings_store.set = AsyncMock()
        return app

    @pytest.fixture
    def client(self, app):
        return TestClient(app)

    def test_get_tool_settings(self, client):
        resp = client.get("/api/config/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert "uarf_auto_search" in data
        assert "uarf_auto_search_queries" in data
        assert "uarf_auto_search_results_per_query" in data
        assert "uarf_auto_search_max_context_chars" in data
        assert "uarf_auto_verify" in data
        assert "uarf_max_tool_calls_per_phase" in data

    def test_update_tool_settings(self, client):
        original_queries = settings.uarf_auto_search_queries
        try:
            resp = client.put(
                "/api/config/tools",
                json={"uarf_auto_search_queries": 7},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["updated"]["uarf_auto_search_queries"] == 7
            assert data["current"]["uarf_auto_search_queries"] == 7
            assert settings.uarf_auto_search_queries == 7
        finally:
            object.__setattr__(settings, "uarf_auto_search_queries", original_queries)

    def test_update_rejects_unknown_setting(self, client):
        resp = client.put(
            "/api/config/tools",
            json={"unknown_setting": 42},
        )
        data = resp.json()
        assert "errors" in data
        assert any("Unknown" in e for e in data["errors"])

    def test_update_rejects_out_of_range(self, client):
        resp = client.put(
            "/api/config/tools",
            json={"uarf_auto_search_queries": 99},
        )
        data = resp.json()
        assert "errors" in data
        assert any("out of range" in e for e in data["errors"])

    def test_update_boolean_setting(self, client):
        original = settings.uarf_auto_search
        try:
            resp = client.put(
                "/api/config/tools",
                json={"uarf_auto_search": False},
            )
            assert resp.status_code == 200
            assert resp.json()["updated"]["uarf_auto_search"] is False
            assert settings.uarf_auto_search is False
        finally:
            object.__setattr__(settings, "uarf_auto_search", original)

    def test_update_persists_to_store(self, client, app):
        original = settings.uarf_auto_search_queries
        try:
            client.put(
                "/api/config/tools",
                json={"uarf_auto_search_queries": 8},
            )
            app.state.settings_store.set.assert_called()
            # Find the call with our key
            calls = app.state.settings_store.set.call_args_list
            assert any(c.args == ("uarf_auto_search_queries", "8") for c in calls)
        finally:
            object.__setattr__(settings, "uarf_auto_search_queries", original)

    def test_multiple_updates_at_once(self, client):
        originals = {
            "uarf_auto_search_queries": settings.uarf_auto_search_queries,
            "uarf_auto_search_results_per_query": settings.uarf_auto_search_results_per_query,
        }
        try:
            resp = client.put(
                "/api/config/tools",
                json={
                    "uarf_auto_search_queries": 3,
                    "uarf_auto_search_results_per_query": 6,
                },
            )
            data = resp.json()
            assert data["updated"]["uarf_auto_search_queries"] == 3
            assert data["updated"]["uarf_auto_search_results_per_query"] == 6
        finally:
            for k, v in originals.items():
                object.__setattr__(settings, k, v)


# --- Prompt integration tests ---


class TestPromptSearchScoping:
    """Verify that get_phase_prompt uses scoped search context."""

    SAMPLE_CONTEXT = SEARCH_CONTEXT_SECTION.format(
        search_results=(
            "[1] Title\n"
            "    URL: https://example.com\n"
            "    Snippet text here."
        ),
    )

    def test_apply_prompt_includes_full_search(self):
        from augmentum.modes.analytical.prompts import get_phase_prompt

        _, user = get_phase_prompt("apply", query="test", search_context=self.SAMPLE_CONTEXT)
        assert "Snippet text here" in user
        assert "BEGIN WEB RESULTS" in user

    def test_verify_prompt_includes_urls_only(self):
        from augmentum.modes.analytical.prompts import get_phase_prompt

        _, user = get_phase_prompt(
            "verify",
            query="test",
            apply_output="some analysis",
            search_context=self.SAMPLE_CONTEXT,
        )
        assert "URL: https://example.com" in user
        assert "Snippet text here" not in user

    def test_assess_prompt_excludes_search(self):
        from augmentum.modes.analytical.prompts import get_phase_prompt

        _, user = get_phase_prompt("assess", query="test", search_context=self.SAMPLE_CONTEXT)
        assert "WEB" not in user
        assert "URL" not in user

    def test_conclude_prompt_excludes_search(self):
        from augmentum.modes.analytical.prompts import get_phase_prompt

        _, user = get_phase_prompt(
            "conclude",
            query="test",
            apply_output="analysis",
            search_context=self.SAMPLE_CONTEXT,
        )
        assert "WEB" not in user
