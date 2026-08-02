"""Smoke tests — verify every module under augmentum/memory/ can be imported."""

from __future__ import annotations


class TestMemoryImports:
    """Import every module in the memory package."""

    def test_import_store(self):
        from augmentum.memory import store  # noqa: F401

    def test_import_llm_extractor(self):
        from augmentum.memory import llm_extractor  # noqa: F401

    def test_import_graph(self):
        from augmentum.memory import graph  # noqa: F401

    def test_import_graph_extractor(self):
        from augmentum.memory import graph_extractor  # noqa: F401

    def test_import_embeddings(self):
        from augmentum.memory import embeddings  # noqa: F401

    def test_import_reranker(self):
        from augmentum.memory import reranker  # noqa: F401

    def test_import_extractor(self):
        from augmentum.memory import extractor  # noqa: F401

    def test_import_consolidator(self):
        from augmentum.memory import consolidator  # noqa: F401

    def test_import_compactor(self):
        from augmentum.memory import compactor  # noqa: F401

    def test_import_core_profile(self):
        from augmentum.memory import core_profile  # noqa: F401

    def test_import_models(self):
        from augmentum.memory import models  # noqa: F401

    def test_import_events(self):
        from augmentum.memory import events  # noqa: F401

    def test_import_notifications(self):
        from augmentum.memory import notifications  # noqa: F401

    def test_import_reflection(self):
        from augmentum.memory import reflection  # noqa: F401

    def test_import_integration(self):
        from augmentum.memory import integration  # noqa: F401

    def test_memory_model_types(self):
        from augmentum.memory.models import MemoryType

        assert MemoryType.FACT.value == "fact"
        assert MemoryType.PREFERENCE.value == "preference"

    def test_memory_tier_values(self):
        from augmentum.memory.models import MemoryTier

        assert MemoryTier.CORE.value == "core"
        assert MemoryTier.PROVISIONAL.value == "provisional"

    def test_extracted_fact_defaults(self):
        from augmentum.memory.models import ExtractedFact, MemoryType

        f = ExtractedFact(content="User likes Python")
        assert f.type == MemoryType.FACT
        assert f.importance == 0.5
        assert f.confidence == 0.8
