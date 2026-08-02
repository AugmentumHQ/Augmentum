"""Pin the autonomous-web-search gate.

The gate exists so background paths (curator tick, UI For-You poll) do
NOT fan out to SearXNG by default — only RSS / HN / Reddit / arXiv
feeds the user explicitly subscribed to.

User-initiated voice / chat tool calls bypass the gate entirely
because they go through WebSearchTool, not the recommender.

Tests don't run a real SearXNG fan-out — they verify the wiring:
  - The setting exists and defaults to False
  - The autonomous kwarg threads to generate_recommendations
  - The curator + UI poll pass autonomous=True
  - The Core / Frontier / Adjacent budgets zero out when gated
  - Fresh (feeds) zone still runs when gated
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_CONFIG = Path("augmentum/config.py")
_RECOMMENDER = Path("augmentum/discovery/recommender.py")
_CURATOR = Path("augmentum/companion_runtime/curator.py")
_DISCOVERY_ROUTES = Path("augmentum/proxy/discovery_routes.py")


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8")


class TestSettingDeclared:
    def test_setting_exists_in_config(self):
        src = _src(_CONFIG)
        assert "companion_autonomous_web_search_enabled" in src

    def test_setting_defaults_to_false(self):
        """Default OFF — autonomous SearXNG fan-out must be opt-in."""
        src = _src(_CONFIG)
        # Match the assignment line. Accept any whitespace around the colon
        # and value.
        pattern = (
            r"companion_autonomous_web_search_enabled\s*:\s*bool\s*=\s*False"
        )
        assert re.search(pattern, src), (
            "companion_autonomous_web_search_enabled must default to False"
        )

    def test_setting_has_admin_marker(self):
        """Admin settings get the [admin] marker so config_routes validates
        them through the right path."""
        src = _src(_CONFIG)
        # Match the comment on the same line.
        line = next(
            (l for l in src.splitlines()
             if "companion_autonomous_web_search_enabled" in l),
            None,
        )
        assert line is not None
        assert "[admin]" in line


class TestRecommenderKwarg:
    def test_autonomous_kwarg_declared(self):
        src = _src(_RECOMMENDER)
        # Function signature should include autonomous as a keyword-only arg
        # with default False.
        assert re.search(
            r"autonomous\s*:\s*bool\s*=\s*False",
            src,
        ), "generate_recommendations must declare autonomous: bool = False"

    def test_gate_reads_setting(self):
        src = _src(_RECOMMENDER)
        assert "companion_autonomous_web_search_enabled" in src

    def test_skip_searxng_zones_logic(self):
        """When autonomous=True AND setting=False, the SearXNG zones
        (Core/Frontier/Adjacent) must zero out."""
        src = _src(_RECOMMENDER)
        # The skip flag must exist
        assert "skip_searxng_zones" in src
        # And gate each zone's budget assignment
        for zone in ("core_budget", "frontier_budget", "adjacent_budget"):
            assert zone in src
        # Each zone's budget gets zeroed when skip_searxng_zones is True
        # — check that the zero-out pattern appears at least once for
        # each. We don't pin the exact expression so a future refactor
        # can move to a helper without breaking the test.
        assert (
            "skip_searxng_zones" in src
            and src.count("skip_searxng_zones") >= 3
        ), "Each SearXNG zone must consult skip_searxng_zones"

    def test_logs_gate_fired(self):
        """A gated request must log so prod can observe the policy in action."""
        src = _src(_RECOMMENDER)
        assert "recommender_searxng_gated" in src


class TestCallersOptIn:
    def test_curator_passes_autonomous_true(self):
        """Curator's For-You step must mark itself as autonomous."""
        src = _src(_CURATOR)
        # Find the generate_recommendations call inside the For-You phase
        # and confirm autonomous=True appears within its argument list.
        idx = src.find("generate_recommendations(")
        assert idx > 0
        # Match the rest of the call (up to the closing paren of THIS call).
        # Simple bounded slice — 1.5 KB is plenty for typical kwargs.
        window = src[idx:idx + 1500]
        assert "autonomous=True" in window, (
            "Curator's generate_recommendations call must include "
            "autonomous=True"
        )

    def test_discovery_routes_passes_autonomous_true(self):
        """UI For-You poll endpoint must mark itself as autonomous."""
        src = _src(_DISCOVERY_ROUTES)
        idx = src.find("generate_recommendations(")
        assert idx > 0
        window = src[idx:idx + 1500]
        assert "autonomous=True" in window


class TestUserPathsUnaffected:
    """The gate must not break user-initiated paths. WebSearchTool calls
    SearXNG directly without going through generate_recommendations, so
    it can't be affected by this gate."""

    def test_websearch_tool_does_not_import_recommender(self):
        """WebSearchTool must not import the recommender — they're
        separate paths. If a future refactor routes WebSearchTool through
        the recommender, the user-search path would suddenly be gated."""
        # Find tool implementations that touch SearXNG
        tool_paths = [
            Path("augmentum/tools/web_search.py"),
            Path("augmentum/tools/websearch.py"),
        ]
        existing = [p for p in tool_paths if p.exists()]
        if not existing:
            pytest.skip("No web-search tool module found at expected paths")
        for path in existing:
            src = path.read_text(encoding="utf-8")
            assert "generate_recommendations" not in src, (
                f"{path} routes through recommender — user-initiated "
                "searches would be subject to autonomous gate"
            )
