"""Structured spec schema for constraint-driven code synthesis.

Defines the data model for extracted application specs: elements, state,
and behavioral constraints. Includes parser (JSON from LLM output),
validator (catches structural issues before compilation), and topological
sort (dependency-ordered constraint queue).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Element:
    """A DOM element in the app skeleton."""
    id: str
    tag: str
    role: str  # container, column, action, modal, field, display
    label: str = ""
    parent: str = ""  # parent element ID (for nesting)


@dataclass
class Constraint:
    """One testable behavior the app must satisfy."""
    id: str
    behavior: str  # short name: "create-card", "drag-between-columns"
    description: str  # human-readable: "Clicking #add-btn creates a card"
    type: str  # structural, interaction, persistence, timer, canvas
    depends_on: list[str] = field(default_factory=list)
    trigger: dict = field(default_factory=dict)  # {event, target, key, ctrl, ...}
    expected: dict = field(default_factory=dict)  # {new_element, parent, visible, ...}
    status: str = "pending"  # pending, passed, failed, skipped


@dataclass
class AppSpec:
    """Complete structured specification for an application."""
    name: str
    state_schema: dict = field(default_factory=dict)
    elements: list[Element] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)


def parse_spec(raw: str) -> AppSpec:
    """Parse LLM output into an AppSpec.

    Handles: raw JSON, JSON wrapped in markdown fences, JSON with
    trailing text. Raises ValueError if no valid JSON found.
    """
    # Try extracting JSON from markdown code fences first
    fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", raw)
    text = fence_match.group(1) if fence_match else raw

    # Try to find the JSON object boundaries
    start = text.find("{")
    if start == -1:
        raise ValueError("Could not parse spec: no JSON object found in response")

    # Find matching closing brace
    depth = 0
    end = start
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    raw_json = text[start:end]
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        # LLMs produce broken JSON constantly — fix common issues:
        # 1. Trailing commas: ,} or ,]
        fixed = re.sub(r",\s*([}\]])", r"\1", raw_json)
        # 2. Single quotes → double quotes (only outside existing double-quoted strings)
        #    Simple heuristic: replace ' with " when it's a key/value delimiter
        fixed = re.sub(r"(?<=[{,\[])\s*'([^']+)'\s*:", r' "\1":', fixed)
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
        # 3. Unquoted keys
        fixed = re.sub(r"(?<=[{,])\s*([a-zA-Z_]\w*)\s*:", r' "\1":', fixed)
        try:
            data = json.loads(fixed)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse spec JSON: {exc}") from exc

    elements = [
        Element(
            id=e.get("id", ""),
            tag=e.get("tag", "div"),
            role=e.get("role", "container"),
            label=e.get("label", ""),
            parent=e.get("parent", ""),
        )
        for e in data.get("elements", [])
    ]

    constraints = [
        Constraint(
            id=c.get("id", f"c{i+1}"),
            behavior=c.get("behavior", ""),
            description=c.get("description", ""),
            type=c.get("type", "structural"),
            depends_on=c.get("depends_on") or [],
            trigger=c.get("trigger") or {},
            expected=c.get("expected") or {},
        )
        for i, c in enumerate(data.get("constraints", []))
    ]

    return AppSpec(
        name=data.get("name", "Untitled App"),
        state_schema=data.get("state_schema", {}),
        elements=elements,
        constraints=constraints,
    )


def validate_spec(spec: AppSpec) -> list[str]:
    """Validate a parsed spec for structural issues.

    Returns a list of error strings. Empty list = valid.
    """
    errors = []

    if not spec.constraints:
        errors.append("Spec has no constraints — nothing to implement")

    # Check for duplicate constraint IDs
    seen_ids: set[str] = set()
    for c in spec.constraints:
        if c.id in seen_ids:
            errors.append(f"Duplicate constraint ID: {c.id}")
        seen_ids.add(c.id)

    # Check for dangling dependencies
    for c in spec.constraints:
        for dep in c.depends_on:
            if dep not in seen_ids:
                errors.append(f"Constraint {c.id} depends on {dep} which doesn't exist")

    # Check element IDs referenced in constraints exist
    element_ids = {e.id for e in spec.elements}
    for c in spec.constraints:
        target = (c.trigger or {}).get("target", "")
        if target.startswith("#"):
            ref_id = target[1:].split(".")[0].split("[")[0]
            if ref_id and ref_id not in element_ids:
                errors.append(f"Constraint {c.id} references #{ref_id} but no element with that ID in spec")

    return errors


def sort_constraints(constraints: list[Constraint]) -> list[Constraint]:
    """Topological sort of constraints by depends_on.

    Raises ValueError if there's a dependency cycle.
    """
    by_id = {c.id: c for c in constraints}
    visited: set[str] = set()
    in_stack: set[str] = set()
    result: list[Constraint] = []

    def visit(cid: str) -> None:
        if cid in in_stack:
            raise ValueError(f"Dependency cycle detected involving {cid}")
        if cid in visited:
            return
        in_stack.add(cid)
        c = by_id.get(cid)
        if c:
            for dep in c.depends_on:
                visit(dep)
        in_stack.discard(cid)
        visited.add(cid)
        if c:
            result.append(c)

    for c in constraints:
        visit(c.id)

    return result
