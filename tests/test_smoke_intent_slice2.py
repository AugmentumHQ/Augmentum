"""Smoke tests — Slice 2 verb modules (media / search / chat).

Verifies the verbs register correctly, fan out only to Tier 3 (LLM
tool exposure — never regex pattern match), and their handlers
dispatch the expected surface channels.

Architectural contract (Phase 0 of the build plan): natural-language
verbs orchestrate through the LLM, not through pattern matching. Tier
1 / 2 are reserved for latency-critical control phrases (stop / pause
/ bye / never mind) where sub-100ms response matters. Discovery and
navigation verbs (play / search / new chat) pay the ~1-2s LLM
round-trip in exchange for understanding STT noise and conversational
phrasing the switchboard pattern can't handle.

If a future refactor accidentally drops a module from
``augmentum/intent/__init__.py``'s eager-import list, this file will
fail loudly at the registry-shape assertion stage.
"""

from __future__ import annotations

import pytest

# Match the discipline in test_smoke_intent — pull architect in so its
# primitives don't get spurious-missing flags if it's the import path
# that resolves the shared REGISTRY.
import augmentum.architect  # noqa: F401

SLICE2_VERB_IDS = (
    "media.play",
    "search.knowledge", "search.local",
    "chat.new", "chat.history",
)


class TestSlice2Registry:
    """Every new module's verbs land in the registry on import."""

    def test_media_verbs_registered(self):
        from augmentum.intent import REGISTRY
        ids = {a.id for a in REGISTRY.all()}
        # Discovery verbs are owned here; transport verbs (pause / next /
        # previous) are owned by augmentum/architect/primitives/media_control.py
        # and we explicitly do NOT redefine them.
        assert "media.play" in ids
        # media.search retired 2026-06-11 — duplicated search.local
        # byte-for-byte (same files.search_open emit); must stay gone.
        assert "media.search" not in ids

    def test_search_verbs_registered(self):
        from augmentum.intent import REGISTRY
        ids = {a.id for a in REGISTRY.all()}
        for expected in (
            "search.knowledge", "search.local",
        ):
            assert expected in ids, f"missing {expected}"
        # Open-web screen search consolidated onto the architect
        # primitive (2026-06-11) — the slice-2 twin must stay gone.
        assert "search.web" not in ids
        assert "web.search" in ids

    def test_chat_verbs_registered(self):
        from augmentum.intent import REGISTRY
        ids = {a.id for a in REGISTRY.all()}
        for expected in ("chat.new", "chat.history"):
            assert expected in ids, f"missing {expected}"


class TestSlice2Fanout:
    """All Slice 2 verbs are Tier-3-only — the LLM orchestrates them.

    Pattern-matching natural-language verbs creates two failure modes:
    (a) false-positives on conversational uses of common words ("I
    played piano yesterday" → media.play); (b) brittleness to STT
    noise ("play do, not yet?" loses to "play Dune audiobook"). The
    Tier-3-only contract avoids both — Becca picks the verb from
    context, the model handles ambiguity.
    """

    @pytest.mark.parametrize("verb_id", SLICE2_VERB_IDS)
    def test_verb_is_tier3_only(self, verb_id):
        from augmentum.intent import REGISTRY
        action = REGISTRY.get(verb_id)
        assert action is not None, f"{verb_id} not in registry"
        assert action.fanout.tier1 is False, (
            f"{verb_id} should not fan out to Tier 1 — "
            "natural-language verbs route through the LLM"
        )
        assert action.fanout.tier2 is False, (
            f"{verb_id} should not fan out to Tier 2"
        )
        assert action.fanout.tier3 is True, (
            f"{verb_id} should expose as LLM tool (Tier 3)"
        )
        assert action.fanout.fast_path is False, (
            f"{verb_id} should not be on the frontend fast-path"
        )

    @pytest.mark.parametrize("verb_id", SLICE2_VERB_IDS)
    def test_verb_has_no_compiled_templates(self, verb_id):
        """Hassil templates would fire even with tier1=True so they
        must be empty. Regex patterns may be auto-derived from
        examples (registry feature for LLM tool description), but the
        matcher gates on ``fanout.tier1`` before evaluating them — so
        Tier-3-only verbs are functionally invisible regardless."""
        from augmentum.intent import REGISTRY
        action = REGISTRY.get(verb_id)
        assert action is not None, f"{verb_id} not in registry"
        assert action.compiled_templates == [], (
            f"{verb_id} should not have compiled templates — "
            "Tier-3-only verbs don't template-match"
        )

    def test_slice2_verbs_never_match_transcripts(self):
        """End-to-end — match_intent never returns a Slice 2 verb.

        Tier-3-only verbs are invisible to ``match_intent``. They only
        surface to the LLM via ``register_action_tools``.
        """
        from augmentum.intent import match_intent
        slice2_ids = set(SLICE2_VERB_IDS)
        # Phrases that would have matched the old regex patterns.
        for text in (
            "play the dune audiobook",
            "find me some sci-fi",
            "search for sourdough recipes",
            "look up the battle of agincourt",
            "find on my computer the foundation",
            "new chat",
            "show me my chat history",
        ):
            m = match_intent(text)
            if m is not None:
                assert m.action_id not in slice2_ids, (
                    f"slice 2 verb {m.action_id!r} should not have "
                    f"pattern-matched {text!r} — Tier 3 only"
                )


class TestSlice2HandlerShape:
    """Handlers return well-formed ActionResult with the right surface channel.

    Even though Tier 1/2 are off, the handlers must work — they're
    invoked when the LLM picks the verb via Tier 3 tool exposure.
    """

    # media.play left this table 2026-06-11: it resolves against the
    # library (play/offer/miss, see media/resolver.py) instead of
    # emitting a search channel — covered by the resolver tests.
    # media.search retired the same day (duplicated search.local).
    @pytest.mark.asyncio
    @pytest.mark.parametrize("verb_id,args,expected_channel", [
        ("web.search",       {"query": "sourdough"}, "browse.search"),
        ("search.knowledge", {"query": "rome"},      "browse.search"),
        ("search.local",     {"query": "budget"},    "files.search_open"),
        ("chat.new",     {}, "chat.new"),
        ("chat.history", {}, "navigate.open_surface"),
    ])
    async def test_handler_dispatches_expected_channel(
        self, verb_id, args, expected_channel,
    ):
        from augmentum.intent import REGISTRY
        action = REGISTRY.get(verb_id)
        assert action is not None, f"{verb_id} not in registry"
        # Handlers expect (text, session, args). SessionContext is a
        # plain dataclass so we can construct one with empty defaults.
        from augmentum.intent.action import SessionContext
        session = SessionContext(
            session_id="test", user_id="test_user",
        )
        result = await action.handler("", session, args)
        assert result is not None
        assert result.surface_emit is not None
        assert result.surface_emit["channel"] == expected_channel

    # media.play's empty-query path now parks a clarify question
    # (ReferentCache.pending_intent) instead of yanking a panel —
    # pinned in test_parked_intent.py, not here.
    @pytest.mark.asyncio
    @pytest.mark.parametrize("verb_id,empty_args,fallback_channel", [
        ("web.search",       {}, "navigate.open_surface"),
        ("search.knowledge", {}, "navigate.open_surface"),
        ("search.local",     {}, "navigate.open_surface"),
    ])
    async def test_handler_graceful_on_empty_query(
        self, verb_id, empty_args, fallback_channel,
    ):
        """No query → open the destination surface unfiltered."""
        from augmentum.intent import REGISTRY
        from augmentum.intent.action import SessionContext
        action = REGISTRY.get(verb_id)
        session = SessionContext(session_id="test", user_id="test_user")
        result = await action.handler("", session, empty_args)
        assert result is not None
        assert result.surface_emit["channel"] == fallback_channel
