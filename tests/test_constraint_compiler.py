"""Tests for constraint-to-test compiler — spec → QuickJS test scripts."""
from __future__ import annotations

from augmentum.tools.constraint_compiler import (
    compile_tests,
    generate_css_foundation,
    generate_skeleton,
)
from augmentum.tools.constraint_schema import AppSpec, Constraint, Element


def _make_spec(**overrides) -> AppSpec:
    """Helper to build a minimal valid spec."""
    defaults = {
        "name": "Test App",
        "state_schema": {"items": "array"},
        "elements": [
            Element(id="app", tag="div", role="container"),
            Element(id="add-btn", tag="button", role="action", label="Add"),
            Element(id="item-list", tag="div", role="display"),
        ],
        "constraints": [
            Constraint(id="c1", behavior="render-app", description="App container renders",
                       type="structural", depends_on=[]),
        ],
    }
    defaults.update(overrides)
    return AppSpec(**defaults)


class TestGenerateSkeleton:
    def test_produces_valid_html(self):
        spec = _make_spec()
        html = generate_skeleton(spec)
        assert "<!DOCTYPE html>" in html
        assert '<div id="app"' in html
        assert '<button id="add-btn"' in html
        assert "Add" in html

    def test_nests_children_in_parent(self):
        spec = _make_spec(elements=[
            Element(id="modal", tag="dialog", role="modal"),
            Element(id="modal-input", tag="input", role="field", parent="modal"),
            Element(id="modal-save", tag="button", role="action", label="Save", parent="modal"),
        ])
        html = generate_skeleton(spec)
        dialog_start = html.index("<dialog")
        dialog_end = html.index("</dialog>")
        input_pos = html.index('id="modal-input"')
        assert dialog_start < input_pos < dialog_end

    def test_column_role_generates_card_list(self):
        spec = _make_spec(elements=[
            Element(id="col-todo", tag="section", role="column", label="To Do"),
        ])
        html = generate_skeleton(spec)
        assert "card-list" in html
        assert "To Do" in html

    def test_includes_viewport_meta(self):
        spec = _make_spec()
        html = generate_skeleton(spec)
        assert "viewport" in html

    def test_title_uses_spec_name(self):
        spec = _make_spec(name="My Kanban")
        html = generate_skeleton(spec)
        assert "<title>My Kanban</title>" in html


class TestCompileTests:
    def test_structural_constraint_produces_test(self):
        spec = _make_spec(constraints=[
            Constraint(id="c1", behavior="render-app", description="App container renders",
                       type="structural", depends_on=[]),
        ])
        result = compile_tests(spec)
        assert "c1" in result.tests
        assert "_verifyErrors" in result.tests["c1"]
        assert "c1" not in result.unmapped

    def test_click_interaction_produces_test(self):
        spec = _make_spec(constraints=[
            Constraint(id="c2", behavior="create-item",
                       description="Clicking #add-btn creates an item",
                       type="interaction",
                       trigger={"event": "click", "target": "#add-btn"},
                       expected={"new_element": ".item", "parent": "#item-list"},
                       depends_on=["c1"]),
        ])
        result = compile_tests(spec)
        assert "c2" in result.tests
        assert "click" in result.tests["c2"]
        assert "add-btn" in result.tests["c2"]

    def test_persistence_save_produces_test(self):
        spec = _make_spec(constraints=[
            Constraint(id="c3", behavior="persist-state",
                       description="State persists to localStorage",
                       type="persistence",
                       trigger={"event": "state-change"},
                       expected={"localStorage_key": "app-state"},
                       depends_on=[]),
        ])
        result = compile_tests(spec)
        assert "c3" in result.tests
        assert "localStorage" in result.tests["c3"]

    def test_unmappable_constraint_flagged(self):
        spec = _make_spec(constraints=[
            Constraint(id="c99", behavior="custom-thing",
                       description="Does something very specific",
                       type="custom", depends_on=[]),
        ])
        result = compile_tests(spec)
        assert "c99" in result.unmapped or "c99" in result.tests

    def test_drag_interaction_produces_test(self):
        spec = _make_spec(
            elements=[
                Element(id="board", tag="main", role="container"),
                Element(id="col-a", tag="section", role="column", label="A"),
                Element(id="col-b", tag="section", role="column", label="B"),
            ],
            constraints=[
                Constraint(id="c4", behavior="drag-card",
                           description="Cards can be dragged between columns",
                           type="interaction",
                           trigger={"event": "drag", "source": ".card", "target": ".card-list"},
                           expected={"element_moved": True},
                           depends_on=[]),
            ],
        )
        result = compile_tests(spec)
        assert "c4" in result.tests
        assert "dragstart" in result.tests["c4"]
        assert "drop" in result.tests["c4"]

    def test_all_tests_combined_is_valid_js(self):
        spec = _make_spec(constraints=[
            Constraint(id="c1", behavior="render", description="Renders", type="structural", depends_on=[]),
            Constraint(id="c2", behavior="click", description="Click works",
                       type="interaction",
                       trigger={"event": "click", "target": "#add-btn"},
                       expected={"new_element": ".item"},
                       depends_on=["c1"]),
        ])
        result = compile_tests(spec)
        combined = result.combined_script()
        assert combined.count("{") == combined.count("}")
        assert "_verifyErrors" in combined


class TestGenerateCssFoundation:
    def test_produces_css_string(self):
        spec = _make_spec()
        css = generate_css_foundation(spec)
        assert ":root" in css
        assert "--" in css

    def test_includes_responsive(self):
        spec = _make_spec()
        css = generate_css_foundation(spec)
        assert "@media" in css
