"""Smoke tests -- verify every dream module imports and primary classes construct."""

from __future__ import annotations

import importlib


class TestDreamModuleImports:
    """Every module under augmentum/dream/ must import without error."""

    def test_import_engine(self):
        mod = importlib.import_module("augmentum.dream.engine")
        assert hasattr(mod, "DreamEngine")

    def test_import_scheduler(self):
        mod = importlib.import_module("augmentum.dream.scheduler")
        assert hasattr(mod, "DreamScheduler")

    def test_import_context(self):
        mod = importlib.import_module("augmentum.dream.context")
        assert hasattr(mod, "DreamContextBuilder")

    def test_import_journal(self):
        mod = importlib.import_module("augmentum.dream.journal")
        assert hasattr(mod, "DreamJournal")

    def test_import_portrait(self):
        mod = importlib.import_module("augmentum.dream.portrait")
        assert hasattr(mod, "PortraitManager")

    def test_import_prompts(self):
        mod = importlib.import_module("augmentum.dream.prompts")
        assert hasattr(mod, "build_dream_prompt")
        assert hasattr(mod, "DREAM_ANTI_PATTERNS")

    def test_import_models(self):
        mod = importlib.import_module("augmentum.dream.models")
        assert hasattr(mod, "DreamEntry")
        assert hasattr(mod, "DreamCycle")
        assert hasattr(mod, "DreamPortrait")
        assert hasattr(mod, "DreamEntryType")
        assert hasattr(mod, "ContextSegment")
