"""Tests for LLM-based narrative semantic extraction."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.modes.narrative.llm_extractor import (
    CharacterExtraction,
    FactExtraction,
    NarrativeExtraction,
    PlotExtraction,
    WorldExtraction,
    _parse_extraction,
    extract_narrative_state,
)

# --- _parse_extraction tests ---


class TestParseExtraction:
    def test_empty_json(self):
        result = _parse_extraction('{"characters":[],"world":{},"plots":{},"facts":[]}')
        assert result is not None
        assert result.characters == []
        assert result.world is None
        assert result.plots is None
        assert result.facts == []

    def test_full_extraction(self):
        data = {
            "characters": [
                {
                    "name": "Lyra",
                    "emotion": "happy",
                    "emotion_confidence": 0.9,
                    "physical_state": "sitting",
                    "location": "the tavern",
                    "inventory_add": ["map"],
                    "inventory_remove": [],
                    "relationship_changes": {"Kael": "growing trust"},
                }
            ],
            "world": {
                "location": "the tavern",
                "time_of_day": "evening",
                "weather": "raining",
                "atmosphere": "tense",
            },
            "plots": {
                "new_threads": ["A mysterious letter arrives"],
                "progressions": ["Getting closer to the artifact"],
                "resolutions": [],
            },
            "facts": [
                {"content": "Lyra is left-handed", "confidence": 0.95}
            ],
        }
        result = _parse_extraction(json.dumps(data))
        assert result is not None
        assert len(result.characters) == 1
        assert result.characters[0].name == "Lyra"
        assert result.characters[0].emotion == "happy"
        assert result.characters[0].emotion_confidence == 0.9
        assert result.characters[0].physical_state == "sitting"
        assert result.characters[0].location == "the tavern"
        assert result.characters[0].inventory_add == ["map"]
        assert result.characters[0].relationship_changes == {"Kael": "growing trust"}

        assert result.world is not None
        assert result.world.location == "the tavern"
        assert result.world.time_of_day == "evening"
        assert result.world.weather == "raining"
        assert result.world.atmosphere == "tense"

        assert result.plots is not None
        assert result.plots.new_threads == ["A mysterious letter arrives"]
        assert result.plots.progressions == ["Getting closer to the artifact"]
        assert result.plots.resolutions == []

        assert len(result.facts) == 1
        assert result.facts[0].content == "Lyra is left-handed"
        assert result.facts[0].confidence == 0.95

    def test_null_values_treated_as_none(self):
        data = {
            "characters": [
                {
                    "name": "Kael",
                    "emotion": "null",
                    "physical_state": "null",
                    "location": "null",
                }
            ],
            "world": {},
            "plots": {},
            "facts": [],
        }
        result = _parse_extraction(json.dumps(data))
        assert result is not None
        assert result.characters[0].emotion is None
        assert result.characters[0].physical_state is None
        assert result.characters[0].location is None

    def test_markdown_code_fences_stripped(self):
        raw = '```json\n{"characters":[],"world":{},"plots":{},"facts":[]}\n```'
        result = _parse_extraction(raw)
        assert result is not None

    def test_invalid_json_returns_none(self):
        assert _parse_extraction("not json at all") is None

    def test_non_dict_returns_none(self):
        assert _parse_extraction("[1,2,3]") is None

    def test_missing_character_name_skipped(self):
        data = {
            "characters": [{"emotion": "happy"}, {"name": "Valid", "emotion": "sad"}],
            "world": {},
            "plots": {},
            "facts": [],
        }
        result = _parse_extraction(json.dumps(data))
        assert result is not None
        assert len(result.characters) == 1
        assert result.characters[0].name == "Valid"

    def test_empty_fact_content_skipped(self):
        data = {
            "characters": [],
            "world": {},
            "plots": {},
            "facts": [{"content": "", "confidence": 0.5}, {"content": "Real fact"}],
        }
        result = _parse_extraction(json.dumps(data))
        assert result is not None
        assert len(result.facts) == 1
        assert result.facts[0].content == "Real fact"

    def test_world_all_null_gives_none(self):
        data = {
            "characters": [],
            "world": {"location": "null", "time_of_day": "null"},
            "plots": {},
            "facts": [],
        }
        result = _parse_extraction(json.dumps(data))
        assert result is not None
        assert result.world is None

    def test_plots_all_empty_gives_none(self):
        data = {
            "characters": [],
            "world": {},
            "plots": {"new_threads": [], "progressions": [], "resolutions": []},
            "facts": [],
        }
        result = _parse_extraction(json.dumps(data))
        assert result is not None
        assert result.plots is None

    def test_multiple_characters(self):
        data = {
            "characters": [
                {"name": "A", "emotion": "happy"},
                {"name": "B", "emotion": "sad"},
            ],
            "world": {},
            "plots": {},
            "facts": [],
        }
        result = _parse_extraction(json.dumps(data))
        assert result is not None
        assert len(result.characters) == 2

    def test_non_dict_character_skipped(self):
        data = {
            "characters": ["not a dict", {"name": "Valid"}],
            "world": {},
            "plots": {},
            "facts": [],
        }
        result = _parse_extraction(json.dumps(data))
        assert result is not None
        assert len(result.characters) == 1

    def test_non_dict_fact_skipped(self):
        data = {
            "characters": [],
            "world": {},
            "plots": {},
            "facts": ["not a dict", {"content": "Valid"}],
        }
        result = _parse_extraction(json.dumps(data))
        assert result is not None
        assert len(result.facts) == 1


# --- extract_narrative_state tests ---


class TestExtractNarrativeState:
    @pytest.fixture()
    def mock_backend(self):
        backend = AsyncMock()
        return backend

    @pytest.mark.asyncio()
    async def test_successful_extraction(self, mock_backend):
        llm_response = json.dumps({
            "characters": [{"name": "Lyra", "emotion": "excited"}],
            "world": {"location": "forest"},
            "plots": {},
            "facts": [],
        })
        response = MagicMock()
        response.message.content = llm_response
        mock_backend.chat.return_value = response

        result = await extract_narrative_state(
            backend=mock_backend,
            text="Lyra ran through the forest excitedly.",
            known_characters=["Lyra"],
        )

        assert result is not None
        assert len(result.characters) == 1
        assert result.characters[0].name == "Lyra"
        assert result.characters[0].emotion == "excited"
        assert result.world is not None
        assert result.world.location == "forest"
        mock_backend.chat.assert_called_once()

    @pytest.mark.asyncio()
    async def test_llm_error_returns_none(self, mock_backend):
        mock_backend.chat.side_effect = RuntimeError("LLM down")
        result = await extract_narrative_state(
            backend=mock_backend,
            text="Some text",
            known_characters=[],
        )
        assert result is None

    @pytest.mark.asyncio()
    async def test_empty_response_returns_none(self, mock_backend):
        response = MagicMock()
        response.message.content = ""
        mock_backend.chat.return_value = response

        result = await extract_narrative_state(
            backend=mock_backend,
            text="Some text",
            known_characters=[],
        )
        assert result is None

    @pytest.mark.asyncio()
    async def test_no_message_returns_none(self, mock_backend):
        response = MagicMock()
        response.message = None
        mock_backend.chat.return_value = response

        result = await extract_narrative_state(
            backend=mock_backend,
            text="Some text",
            known_characters=[],
        )
        assert result is None

    @pytest.mark.asyncio()
    async def test_invalid_json_response_returns_none(self, mock_backend):
        response = MagicMock()
        response.message.content = "I couldn't parse that"
        mock_backend.chat.return_value = response

        result = await extract_narrative_state(
            backend=mock_backend,
            text="Some text",
            known_characters=[],
        )
        assert result is None

    @pytest.mark.asyncio()
    async def test_truncates_long_input(self, mock_backend):
        response = MagicMock()
        response.message.content = '{"characters":[],"world":{},"plots":{},"facts":[]}'
        mock_backend.chat.return_value = response

        long_text = "x" * 10000
        await extract_narrative_state(
            backend=mock_backend,
            text=long_text,
            known_characters=[],
        )

        call_args = mock_backend.chat.call_args[0][0]
        user_msg = [m for m in call_args.messages if m.role == "user"][0]
        # The text portion should be truncated to _MAX_INPUT_CHARS
        assert len(user_msg.content) < 10000

    @pytest.mark.asyncio()
    async def test_no_known_characters(self, mock_backend):
        response = MagicMock()
        response.message.content = '{"characters":[],"world":{},"plots":{},"facts":[]}'
        mock_backend.chat.return_value = response

        result = await extract_narrative_state(
            backend=mock_backend,
            text="Some text",
            known_characters=[],
        )
        assert result is not None
        call_args = mock_backend.chat.call_args[0][0]
        user_msg = [m for m in call_args.messages if m.role == "user"][0]
        assert "(none known yet)" in user_msg.content


# --- Engine merge tests ---


class TestEngineMerge:
    def _make_engine(self):
        from augmentum.modes.narrative.engine import NarrativeEngine
        engine = NarrativeEngine(session_id="test")
        engine._initialized = True
        return engine

    def test_merge_character_existing(self):
        from augmentum.state.narrative_state import Entity, EntityState, EntityType, _new_id
        engine = self._make_engine()

        entity = Entity(
            id=_new_id(),
            session_id="test",
            entity_type=EntityType.CHARACTER,
            name="Lyra",
            state=EntityState(emotional_state="neutral"),
        )
        engine._state.entities[entity.id] = entity

        extraction = NarrativeExtraction(
            characters=[CharacterExtraction(
                name="Lyra",
                emotion="excited",
                emotion_confidence=0.9,
                location="the forest",
            )],
        )
        engine.merge_llm_extraction(extraction, message_index=5)

        updated = engine._state.get_entity_by_name("Lyra")
        assert updated is not None
        assert updated.state.emotional_state == "excited"
        assert updated.state.location == "the forest"

    def test_merge_character_new(self):
        engine = self._make_engine()
        extraction = NarrativeExtraction(
            characters=[CharacterExtraction(
                name="NewChar",
                emotion="calm",
                emotion_confidence=0.7,
            )],
        )
        engine.merge_llm_extraction(extraction, message_index=3)

        entity = engine._state.get_entity_by_name("NewChar")
        assert entity is not None
        assert entity.state.emotional_state == "calm"

    def test_merge_world_state(self):
        engine = self._make_engine()
        extraction = NarrativeExtraction(
            world=WorldExtraction(
                location="the castle",
                time_of_day="night",
                weather="stormy",
                atmosphere="tense",
            ),
        )
        engine.merge_llm_extraction(extraction, message_index=2)

        assert engine.world_state.location == "the castle"
        assert engine.world_state.time_of_day == "night"
        assert engine.world_state.weather == "stormy"

    def test_merge_new_plot_thread(self):
        engine = self._make_engine()
        extraction = NarrativeExtraction(
            plots=PlotExtraction(
                new_threads=["A dragon attacks the village"],
            ),
        )
        engine.merge_llm_extraction(extraction, message_index=4)

        active = engine._plot_tracker.active_threads
        assert len(active) == 1
        assert "dragon" in active[0].title.lower()

    def test_merge_plot_resolution(self):
        engine = self._make_engine()
        engine._plot_tracker.add_thread(
            session_id="test",
            title="Find the artifact",
            message_index=0,
        )
        extraction = NarrativeExtraction(
            plots=PlotExtraction(
                resolutions=["The artifact is found"],
            ),
        )
        engine.merge_llm_extraction(extraction, message_index=5)
        assert len(engine._plot_tracker.active_threads) == 0

    def test_merge_facts(self):
        engine = self._make_engine()
        extraction = NarrativeExtraction(
            facts=[
                FactExtraction(content="Lyra is left-handed", confidence=0.95),
                FactExtraction(content="The tavern is called The Broken Crown", confidence=0.8),
            ],
        )
        engine.merge_llm_extraction(extraction, message_index=6)

        facts = [f for f in engine._state.facts if f.source == "llm_extraction"]
        assert len(facts) == 2
        assert any("left-handed" in f.content for f in facts)

    def test_merge_empty_extraction_noop(self):
        engine = self._make_engine()
        extraction = NarrativeExtraction()
        engine.merge_llm_extraction(extraction, message_index=1)
        assert len(engine._state.entities) == 0
        assert len(engine._state.facts) == 0

    def test_merge_partial_world(self):
        engine = self._make_engine()
        # Set initial state
        engine._world_tracker.apply_delta(
            {"location": "forest", "weather": "clear"}, 0, "main"
        )
        # LLM only updates weather
        extraction = NarrativeExtraction(
            world=WorldExtraction(weather="raining"),
        )
        engine.merge_llm_extraction(extraction, message_index=2)
        # Location should be preserved, weather updated
        assert engine.world_state.location == "forest"
        assert engine.world_state.weather == "raining"


# --- Handler integration tests ---


class TestHandlerLLMExtraction:
    @pytest.mark.asyncio()
    async def test_llm_extraction_fires_on_response(self):
        from augmentum.models.base import (
            InternalChatRequest,
            InternalChatResponse,
            Message,
        )
        from augmentum.modes.narrative.engine import NarrativeEngine
        from augmentum.modes.narrative.handler import NarrativeHandler

        engine = NarrativeEngine(session_id="test")
        backend = AsyncMock()

        response = MagicMock(spec=InternalChatResponse)
        response.message = MagicMock()
        response.message.content = "*Lyra smiled warmly*"
        backend.chat.return_value = response

        from augmentum.modes.narrative.memory_settings import SessionMemorySettings
        engine.state.memory_settings = SessionMemorySettings(memory_enabled=False)
        handler = NarrativeHandler(
            backend=backend,
            engine=engine,
            session_id="test",
        )

        request = InternalChatRequest(
            model="test",
            messages=[
                Message(role="system", content="You are Lyra."),
                Message(role="user", content="Hello"),
            ],
            stream=False,
        )

        with patch("augmentum.modes.narrative.handler.NarrativeHandler._maybe_llm_extract") as mock_extract:
            await handler.handle(request)
            mock_extract.assert_called_once_with("*Lyra smiled warmly*")

    @pytest.mark.asyncio()
    async def test_llm_extraction_disabled(self):
        from augmentum.modes.narrative.engine import NarrativeEngine
        from augmentum.modes.narrative.handler import NarrativeHandler

        engine = NarrativeEngine(session_id="test")
        backend = AsyncMock()
        handler = NarrativeHandler(
            backend=backend,
            engine=engine,
            session_id="test",
        )

        with patch("augmentum.config.settings") as mock_settings:
            mock_settings.narrative_llm_extraction = False
            result = handler._maybe_llm_extract("some text")
            assert result is None

    @pytest.mark.asyncio()
    async def test_llm_extraction_enabled_returns_task(self):
        from augmentum.modes.narrative.engine import NarrativeEngine
        from augmentum.modes.narrative.handler import NarrativeHandler

        engine = NarrativeEngine(session_id="test")
        backend = AsyncMock()
        handler = NarrativeHandler(
            backend=backend,
            engine=engine,
            session_id="test",
        )

        with patch("augmentum.config.settings") as mock_settings:
            mock_settings.narrative_llm_extraction = True
            mock_settings.narrative_extraction_interval = 1
            with patch.object(handler, "_run_llm_extraction", new_callable=AsyncMock) as mock_run:
                task = handler._maybe_llm_extract("some text")
                assert task is not None
                # Let the task finish
                await task
