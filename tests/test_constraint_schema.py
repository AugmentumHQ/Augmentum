"""Tests for constraint schema — spec dataclasses, parsing, validation."""
from __future__ import annotations

import pytest

from augmentum.tools.constraint_schema import (
    AppSpec,
    Constraint,
    Element,
    parse_spec,
    sort_constraints,
    validate_spec,
)


class TestConstraintDataclasses:
    def test_element_creation(self):
        el = Element(id="board", tag="main", role="container")
        assert el.id == "board"
        assert el.tag == "main"
        assert el.role == "container"
        assert el.label == ""
        assert el.parent == ""

    def test_element_with_label_and_parent(self):
        el = Element(id="card-input", tag="input", role="field", label="Card Title", parent="edit-modal")
        assert el.label == "Card Title"
        assert el.parent == "edit-modal"

    def test_constraint_creation(self):
        c = Constraint(
            id="c1", behavior="render-columns",
            description="Three columns visible",
            type="structural", depends_on=[],
        )
        assert c.id == "c1"
        assert c.trigger == {}
        assert c.expected == {}

    def test_constraint_with_trigger_and_expected(self):
        c = Constraint(
            id="c2", behavior="create-card",
            description="Clicking #add-btn creates a card",
            type="interaction",
            trigger={"event": "click", "target": "#add-btn"},
            expected={"new_element": ".card", "parent": "#todo-list"},
            depends_on=["c1"],
        )
        assert c.trigger["event"] == "click"
        assert c.expected["new_element"] == ".card"
        assert c.depends_on == ["c1"]

    def test_app_spec_creation(self):
        spec = AppSpec(
            name="Test App",
            state_schema={"items": "array"},
            elements=[Element(id="app", tag="div", role="container")],
            constraints=[Constraint(id="c1", behavior="init", description="App renders", type="structural", depends_on=[])],
        )
        assert spec.name == "Test App"
        assert len(spec.elements) == 1
        assert len(spec.constraints) == 1


class TestParseSpec:
    def test_parse_valid_json(self):
        raw = '''{
            "name": "Todo",
            "state_schema": {"items": "array"},
            "elements": [{"id": "app", "tag": "div", "role": "container"}],
            "constraints": [{"id": "c1", "behavior": "init", "description": "App renders", "type": "structural", "depends_on": []}]
        }'''
        spec = parse_spec(raw)
        assert spec.name == "Todo"
        assert isinstance(spec.elements[0], Element)
        assert isinstance(spec.constraints[0], Constraint)

    def test_parse_missing_name_uses_default(self):
        raw = '{"state_schema": {}, "elements": [], "constraints": []}'
        spec = parse_spec(raw)
        assert spec.name == "Untitled App"

    def test_parse_invalid_json_raises(self):
        with pytest.raises(ValueError, match="parse"):
            parse_spec("not json at all {{{")

    def test_parse_extracts_json_from_markdown(self):
        raw = 'Here is the spec:\n```json\n{"name": "X", "state_schema": {}, "elements": [], "constraints": []}\n```'
        spec = parse_spec(raw)
        assert spec.name == "X"


class TestValidateSpec:
    def test_valid_spec_passes(self):
        spec = AppSpec(
            name="Test",
            state_schema={"count": "number"},
            elements=[Element(id="btn", tag="button", role="action")],
            constraints=[Constraint(id="c1", behavior="click", description="Click works", type="interaction", depends_on=[])],
        )
        errors = validate_spec(spec)
        assert errors == []

    def test_no_constraints_error(self):
        spec = AppSpec(name="Empty", state_schema={}, elements=[], constraints=[])
        errors = validate_spec(spec)
        assert any("constraint" in e.lower() for e in errors)

    def test_duplicate_constraint_ids_error(self):
        spec = AppSpec(
            name="Dup", state_schema={}, elements=[],
            constraints=[
                Constraint(id="c1", behavior="a", description="A", type="structural", depends_on=[]),
                Constraint(id="c1", behavior="b", description="B", type="structural", depends_on=[]),
            ],
        )
        errors = validate_spec(spec)
        assert any("duplicate" in e.lower() for e in errors)

    def test_dangling_dependency_error(self):
        spec = AppSpec(
            name="Dangle", state_schema={}, elements=[],
            constraints=[
                Constraint(id="c2", behavior="b", description="B", type="structural", depends_on=["c1"]),
            ],
        )
        errors = validate_spec(spec)
        assert any("c1" in e for e in errors)


class TestSortConstraints:
    def test_topological_sort_linear(self):
        constraints = [
            Constraint(id="c3", behavior="c", description="C", type="structural", depends_on=["c2"]),
            Constraint(id="c1", behavior="a", description="A", type="structural", depends_on=[]),
            Constraint(id="c2", behavior="b", description="B", type="structural", depends_on=["c1"]),
        ]
        sorted_c = sort_constraints(constraints)
        ids = [c.id for c in sorted_c]
        assert ids == ["c1", "c2", "c3"]

    def test_topological_sort_diamond(self):
        constraints = [
            Constraint(id="c1", behavior="a", description="A", type="structural", depends_on=[]),
            Constraint(id="c2", behavior="b", description="B", type="structural", depends_on=["c1"]),
            Constraint(id="c3", behavior="c", description="C", type="structural", depends_on=["c1"]),
            Constraint(id="c4", behavior="d", description="D", type="structural", depends_on=["c2", "c3"]),
        ]
        sorted_c = sort_constraints(constraints)
        ids = [c.id for c in sorted_c]
        assert ids.index("c1") < ids.index("c2")
        assert ids.index("c1") < ids.index("c3")
        assert ids.index("c2") < ids.index("c4")
        assert ids.index("c3") < ids.index("c4")

    def test_cycle_raises(self):
        constraints = [
            Constraint(id="c1", behavior="a", description="A", type="structural", depends_on=["c2"]),
            Constraint(id="c2", behavior="b", description="B", type="structural", depends_on=["c1"]),
        ]
        with pytest.raises(ValueError, match="cycle"):
            sort_constraints(constraints)
