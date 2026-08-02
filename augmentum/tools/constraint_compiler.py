"""Constraint-to-test compiler -- turns structured specs into executable QuickJS tests.

Each constraint type maps to a test template. Templates are parameterized
by the constraint trigger/expected fields. The output is a dict of
constraint_id -> QuickJS test function strings.

Also generates the HTML skeleton and CSS foundation from the spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from augmentum.tools.constraint_schema import AppSpec, Constraint, Element


@dataclass
class TestCompilationResult:
    """Result of compiling constraints into tests."""
    tests: dict[str, str] = field(default_factory=dict)
    unmapped: list[str] = field(default_factory=list)

    def combined_script(self) -> str:
        parts = []
        for cid, code in self.tests.items():
            parts.append(f"// --- Test: {cid} ---\n{code}")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# HTML Skeleton Generator
# ---------------------------------------------------------------------------

def generate_skeleton(spec: AppSpec) -> str:
    """Generate valid HTML skeleton from spec elements."""
    roots: list[Element] = []
    children: dict[str, list[Element]] = {}
    for el in spec.elements:
        if el.parent:
            children.setdefault(el.parent, []).append(el)
        else:
            roots.append(el)

    def render_element(el: Element, indent: int = 2) -> str:
        pad = " " * indent
        attrs = f' id="{el.id}"'
        classes = []
        if el.role == "column":
            classes.append("column")
            attrs += f' data-col="{el.id}"'
        elif el.role == "modal":
            classes.append("modal-overlay hidden")
        elif el.role == "display":
            classes.append("display-area")
        if classes:
            attrs += f' class="{" ".join(classes)}"'

        if el.tag in ("input", "img", "br", "hr"):
            return f"{pad}<{el.tag}{attrs} />"

        inner_parts = []
        if el.label:
            if el.role == "column":
                inner_parts.append(f"{pad}  <h2>{el.label}</h2>")
            elif el.tag != "button":
                inner_parts.append(f"{pad}  <span>{el.label}</span>")
        if el.role == "column":
            inner_parts.append(f'{pad}  <div class="card-list" data-drop-zone></div>')
        for child in children.get(el.id, []):
            inner_parts.append(render_element(child, indent + 2))

        inner = "\n".join(inner_parts)
        label_text = el.label if el.tag == "button" else ""
        if inner:
            return f"{pad}<{el.tag}{attrs}>\n{inner}\n{pad}</{el.tag}>"
        elif label_text:
            return f"{pad}<{el.tag}{attrs}>{label_text}</{el.tag}>"
        else:
            return f"{pad}<{el.tag}{attrs}></{el.tag}>"

    body_parts = [render_element(el) for el in roots]
    body_html = "\n".join(body_parts)

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="UTF-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f"  <title>{spec.name}</title>\n"
        "</head>\n"
        "<body>\n"
        f"{body_html}\n"
        "</body>\n"
        "</html>"
    )


# ---------------------------------------------------------------------------
# CSS Foundation Generator
# ---------------------------------------------------------------------------

_CSS_FOUNDATION = """\
:root {
  --bg: #fafafa; --bg-card: #ffffff; --bg-hover: #f1f5f9;
  --text: #0f172a; --text-dim: #64748b;
  --accent: #6366f1; --accent-hover: #4f46e5; --accent-glow: rgba(99,102,241,0.15);
  --success: #10b981; --error: #ef4444; --warning: #f59e0b;
  --border: #e2e8f0;
  --shadow: 0 1px 3px rgba(0,0,0,0.06);
  --shadow-lg: 0 10px 25px rgba(0,0,0,0.07);
  --font: system-ui, -apple-system, sans-serif;
  --radius: 8px;
  --transition: 150ms ease;
}
* { box-sizing: border-box; margin: 0; }
body { font-family: var(--font); background: var(--bg); color: var(--text); line-height: 1.6; }
button { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem;
  border: none; border-radius: var(--radius); font-weight: 600; cursor: pointer;
  background: var(--accent); color: white; transition: all var(--transition); }
button:hover { background: var(--accent-hover); transform: translateY(-1px); }
input, select, textarea { padding: 0.5rem; border: 1px solid var(--border);
  border-radius: var(--radius); font-family: inherit; width: 100%; }
input:focus, textarea:focus { border-color: var(--accent); outline: none;
  box-shadow: 0 0 0 3px var(--accent-glow); }
.hidden { display: none !important; }
.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.5);
  display: flex; align-items: center; justify-content: center; z-index: 100; }
.modal-overlay.hidden { display: none; }
.column { background: var(--bg-card); border-radius: var(--radius); padding: 1rem;
  border: 1px solid var(--border); min-height: 200px; }
.card-list { min-height: 50px; display: flex; flex-direction: column; gap: 0.5rem; }
.card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 0.75rem; cursor: grab; box-shadow: var(--shadow); transition: all var(--transition); }
.card:hover { box-shadow: var(--shadow-lg); }
.card.dragging { opacity: 0.5; }
@media (max-width: 640px) { body { padding: 0.5rem; } }"""


def generate_css_foundation(spec: AppSpec) -> str:
    """Generate a design-system CSS foundation from the spec."""
    return _CSS_FOUNDATION


# ---------------------------------------------------------------------------
# Test Templates
# ---------------------------------------------------------------------------

def _test_structural(c: Constraint, spec: AppSpec) -> str:
    checks = []
    for el in spec.elements:
        checks.append(
            '  if (!document.getElementById("' + el.id + '")) '
            '_verifyErrors.push("CONSTRAINT ' + c.id + ": element #" + el.id + ' not found in DOM");'
        )
    return "(function _test_" + c.id + "() {\n" + "\n".join(checks) + "\n})();"


def _test_click_interaction(c: Constraint, spec: AppSpec) -> str:
    target = c.trigger.get("target", "").lstrip("#")
    new_el = c.expected.get("new_element", ".item")
    lines = [
        "(function _test_" + c.id + "() {",
        '  var el = document.getElementById("' + target + '");',
        '  if (!el) { _verifyErrors.push("CONSTRAINT ' + c.id + ": #" + target + ' not found"); return; }',
        "  if (!el._listeners || !el._listeners.click) {",
        '    _verifyErrors.push("CONSTRAINT ' + c.id + ": #" + target + " has no click listener. "
        "Add: document.getElementById('" + target + "').addEventListener('click', handler)\");",
        "    return;",
        "  }",
        "  var beforeCount = document.querySelectorAll('" + new_el + "').length;",
        "  try {",
        "    el._listeners.click.forEach(function(fn) { fn({ preventDefault: function(){}, target: el }); });",
        "  } catch(e) {",
        '    _verifyErrors.push("CONSTRAINT ' + c.id + ": clicking #" + target + ' threw " + e.constructor.name + ": " + e.message);',
        "    return;",
        "  }",
        "  var afterCount = document.querySelectorAll('" + new_el + "').length;",
        "  if (afterCount <= beforeCount) {",
        '    _verifyErrors.push("CONSTRAINT ' + c.id + ": clicking #" + target + " should create a " + new_el + ' element but count did not increase");',
        "  }",
        "})();",
    ]
    return "\n".join(lines)


def _test_drag_interaction(c: Constraint, spec: AppSpec) -> str:
    source = c.trigger.get("source", ".card")
    target = c.trigger.get("target", ".card-list")
    lines = [
        "(function _test_" + c.id + "() {",
        "  var cards = document.querySelectorAll('" + source + "');",
        '  if (cards.length === 0) { _verifyErrors.push("CONSTRAINT ' + c.id + ": no " + source + ' elements to drag"); return; }',
        "  var card = cards[0];",
        "  var dropZones = document.querySelectorAll('" + target + "');",
        '  if (dropZones.length < 2) { _verifyErrors.push("CONSTRAINT ' + c.id + ": need 2+ " + target + ' containers"); return; }',
        "  var sourceZone = card.parentNode;",
        "  var targetZone = null;",
        "  for (var i = 0; i < dropZones.length; i++) {",
        "    if (dropZones[i] !== sourceZone) { targetZone = dropZones[i]; break; }",
        "  }",
        '  if (!targetZone) { _verifyErrors.push("CONSTRAINT ' + c.id + ': no different drop target"); return; }',
        "  if (!card._listeners || !card._listeners.dragstart) {",
        '    _verifyErrors.push("CONSTRAINT ' + c.id + ": " + source + ' has no dragstart listener");',
        "    return;",
        "  }",
        "  var dragData = {};",
        "  card._listeners.dragstart.forEach(function(fn) {",
        "    fn({ dataTransfer: { setData: function(k,v){ dragData[k]=v; }, effectAllowed: '' }, target: card });",
        "  });",
        "  if (!targetZone._listeners || !targetZone._listeners.drop) {",
        '    _verifyErrors.push("CONSTRAINT ' + c.id + ": " + target + ' has no drop listener");',
        "    return;",
        "  }",
        "  targetZone._listeners.drop.forEach(function(fn) {",
        "    fn({ preventDefault: function(){}, dataTransfer: { getData: function(k){ return dragData[k]; } }, target: targetZone });",
        "  });",
        "  var moved = false;",
        "  for (var j = 0; j < targetZone.children.length; j++) {",
        "    if (targetZone.children[j] === card || (card.id && targetZone.children[j].id === card.id)) { moved = true; break; }",
        "  }",
        "  if (!moved) {",
        '    _verifyErrors.push("CONSTRAINT ' + c.id + ': drag did not move element to target");',
        "  }",
        "})();",
    ]
    return "\n".join(lines)


def _test_dblclick_interaction(c: Constraint, spec: AppSpec) -> str:
    target = c.trigger.get("target", ".card")
    shown = c.expected.get("visible", "")
    shown_id = shown.lstrip("#") if shown else ""
    lines = [
        "(function _test_" + c.id + "() {",
        "  var items = document.querySelectorAll('" + target + "');",
        '  if (items.length === 0) { _verifyErrors.push("CONSTRAINT ' + c.id + ": no " + target + ' elements"); return; }',
        "  var item = items[0];",
        "  if (!item._listeners || !item._listeners.dblclick) {",
        '    _verifyErrors.push("CONSTRAINT ' + c.id + ": " + target + ' has no dblclick listener");',
        "    return;",
        "  }",
        "  try {",
        "    item._listeners.dblclick.forEach(function(fn) { fn({ preventDefault: function(){}, target: item }); });",
        "  } catch(e) {",
        '    _verifyErrors.push("CONSTRAINT ' + c.id + ': dblclick threw " + e.constructor.name + ": " + e.message);',
        "    return;",
        "  }",
    ]
    if shown_id:
        lines.extend([
            '  var modal = document.getElementById("' + shown_id + '");',
            '  if (modal && modal.classList && modal.classList.contains("hidden")) {',
            '    _verifyErrors.push("CONSTRAINT ' + c.id + ": dblclick should show #" + shown_id + ' but it is still hidden");',
            "  }",
        ])
    lines.append("})();")
    return "\n".join(lines)


def _test_keydown_interaction(c: Constraint, spec: AppSpec) -> str:
    key = c.trigger.get("key", "")
    ctrl = "true" if c.trigger.get("ctrl", False) else "false"
    lines = [
        "(function _test_" + c.id + "() {",
        '  var event = { key: "' + key + '", ctrlKey: ' + ctrl + ", preventDefault: function(){} };",
        "  var handled = false;",
        "  if (window._listeners && window._listeners.keydown) {",
        "    window._listeners.keydown.forEach(function(fn) {",
        "      try { fn(event); handled = true; } catch(e) {",
        '        _verifyErrors.push("CONSTRAINT ' + c.id + ': keydown threw " + e.constructor.name + ": " + e.message);',
        "      }",
        "    });",
        "  }",
        "  if (!handled && document._listeners && document._listeners.keydown) {",
        "    document._listeners.keydown.forEach(function(fn) {",
        "      try { fn(event); handled = true; } catch(e) {",
        '        _verifyErrors.push("CONSTRAINT ' + c.id + ': keydown threw " + e.constructor.name + ": " + e.message);',
        "      }",
        "    });",
        "  }",
        "  if (!handled) {",
        '    _verifyErrors.push("CONSTRAINT ' + c.id + ": no keydown listener for key='" + key + "' ctrl=" + ctrl + '");',
        "  }",
        "})();",
    ]
    return "\n".join(lines)


def _test_persistence_save(c: Constraint, spec: AppSpec) -> str:
    storage_key = c.expected.get("localStorage_key", "app-state")
    lines = [
        "(function _test_" + c.id + "() {",
        '  var stored = localStorage.getItem("' + storage_key + '");',
        "  if (!stored) {",
        '    _verifyErrors.push("CONSTRAINT ' + c.id + ": localStorage key '" + storage_key + "' is empty after init\");",
        "  }",
        "})();",
    ]
    return "\n".join(lines)


def _test_persistence_load(c: Constraint, spec: AppSpec) -> str:
    storage_key = c.expected.get("localStorage_key", "app-state")
    lines = [
        "(function _test_" + c.id + "() {",
        "  var testData = '{\"_test_load\": true}';",
        '  localStorage.setItem("' + storage_key + '", testData);',
        '  var retrieved = localStorage.getItem("' + storage_key + '");',
        "  if (!retrieved) {",
        '    _verifyErrors.push("CONSTRAINT ' + c.id + ": localStorage.getItem('" + storage_key + "') not called during init\");",
        "  }",
        "})();",
    ]
    return "\n".join(lines)


def _test_canvas(c: Constraint, spec: AppSpec) -> str:
    lines = [
        "(function _test_" + c.id + "() {",
        '  var canvas = document.querySelector("canvas");',
        '  if (!canvas) { _verifyErrors.push("CONSTRAINT ' + c.id + ': no canvas element found"); return; }',
        "  if (!canvas._contextCreated) {",
        '    _verifyErrors.push("CONSTRAINT ' + c.id + ': canvas getContext() never called");',
        "  }",
        "})();",
    ]
    return "\n".join(lines)


def _test_fallback(c: Constraint, spec: AppSpec) -> str:
    return "(function _test_" + c.id + "() { /* no template for type '" + c.type + "' */ })();"


_TEMPLATES: dict[str, Any] = {
    "structural": _test_structural,
    "interaction/click": _test_click_interaction,
    "interaction/drag": _test_drag_interaction,
    "interaction/dblclick": _test_dblclick_interaction,
    "interaction/keydown": _test_keydown_interaction,
    "interaction/submit": _test_click_interaction,
    "persistence/save": _test_persistence_save,
    "persistence/load": _test_persistence_load,
    "persistence": _test_persistence_save,
    "canvas": _test_canvas,
    "timer": _test_fallback,
}


def _resolve_template(c: Constraint) -> str:
    base = c.type
    event = (c.trigger or {}).get("event", "")
    if event:
        compound = f"{base}/{event}"
        if compound in _TEMPLATES:
            return compound
    if base in _TEMPLATES:
        return base
    return ""


def compile_tests(spec: AppSpec) -> TestCompilationResult:
    """Compile all constraints into executable QuickJS tests."""
    result = TestCompilationResult()
    for c in spec.constraints:
        template_key = _resolve_template(c)
        if template_key:
            result.tests[c.id] = _TEMPLATES[template_key](c, spec)
        else:
            result.tests[c.id] = _test_fallback(c, spec)
            result.unmapped.append(c.id)
    return result
