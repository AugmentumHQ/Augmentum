"""Smoke tests — verify every module under augmentum/state/ can be imported."""

from __future__ import annotations


class TestStateImports:
    """Import every module in the state package to catch syntax/import errors."""

    def test_import_manager(self):
        from augmentum.state import manager  # noqa: F401

    def test_import_memory_backend(self):
        from augmentum.state.backends import memory  # noqa: F401

    def test_import_sqlite_backend(self):
        from augmentum.state.backends import sqlite  # noqa: F401

    def test_import_settings_store(self):
        from augmentum.state import settings_store  # noqa: F401

    def test_import_provider_store(self):
        from augmentum.state import provider_store  # noqa: F401

    def test_import_narrative_state(self):
        from augmentum.state import narrative_state  # noqa: F401

    def test_import_narrative_persistence(self):
        from augmentum.state import narrative_persistence  # noqa: F401

    def test_import_notes_store(self):
        from augmentum.state import notes_store  # noqa: F401

    def test_import_balancer_store(self):
        from augmentum.state import balancer_store  # noqa: F401

    def test_import_discovery_store(self):
        from augmentum.state import discovery_store  # noqa: F401

    def test_import_backup(self):
        from augmentum.state import backup  # noqa: F401

    def test_import_tree_utils(self):
        from augmentum.state import tree_utils  # noqa: F401

    def test_memory_backend_dataclass(self):
        from augmentum.state.backends.memory import MemorySession

        s = MemorySession(id="test-1", mode="analytical")
        assert s.id == "test-1"
        assert s.mode == "analytical"

    def test_tree_utils_linearize(self):
        from augmentum.state.tree_utils import linearize_to_node

        tree = {
            "root": {"content": "hi", "children": ["child1"]},
            "child1": {"content": "hello", "children": []},
        }
        path = linearize_to_node(tree, "root", "child1")
        assert path is not None
        assert len(path) == 2

    def test_tree_utils_missing_target(self):
        from augmentum.state.tree_utils import linearize_to_node

        tree = {"root": {"content": "hi", "children": []}}
        assert linearize_to_node(tree, "root", "missing") is None
