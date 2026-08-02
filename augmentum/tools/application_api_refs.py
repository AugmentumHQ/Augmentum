"""Category-specific API quick references (toolkit spec §2).

When the detected project category implies a specific API surface
(Canvas 2D for games, Chart.js for dashboards, etc.) the generator's
prompt gets a compact reference block with verified signatures. This
prevents small-model hallucinations like ``ctx.fillCircle()`` before
they happen; the ``API_CORRECTIONS`` enrichment in
``application_scaffolds.py`` catches mistakes after the fact — refs
here catch them up front.

Refs are intentionally TERSE — each is a single idiom or signature, not
a tutorial. The goal is to anchor the model's output toward real APIs,
not to replace documentation.
"""

from __future__ import annotations

# Per-category verified API signatures + common idioms. Category keys
# match those produced by ``_detect_categories`` in application_scaffolds.
CATEGORY_API_REFS: dict[str, list[str]] = {
    "canvas_game": [
        "# Canvas 2D — verified signatures",
        "const canvas = document.getElementById('game'); const ctx = canvas.getContext('2d');",
        "# Size the canvas via its ATTRIBUTES, not CSS, to avoid blurry rendering:",
        "canvas.width = 800; canvas.height = 600;",
        "# Clear + redraw each frame:",
        "ctx.clearRect(0, 0, canvas.width, canvas.height);",
        "# Shapes — Canvas 2D has NO fillCircle/drawCircle; use arc():",
        "ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.fill();",
        "ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h);",
        "# Text:",
        "ctx.font = '16px sans-serif'; ctx.fillText(str, x, y);",
        "# Transforms — always wrap in save/restore so siblings aren't affected:",
        "ctx.save(); ctx.translate(cx, cy); ctx.rotate(angle); ctx.fillRect(-w/2, -h/2, w, h); ctx.restore();",
        "# Game loop — use requestAnimationFrame, NOT setInterval:",
        "function loop(t) { update(dt); render(); requestAnimationFrame(loop); }",
        "requestAnimationFrame(loop);",
        "# Input — listen on the document for keyboard, on the canvas for mouse:",
        "document.addEventListener('keydown', (e) => { keys[e.key] = true; });",
        "canvas.addEventListener('mousedown', (e) => { const r = canvas.getBoundingClientRect(); const x = e.clientX - r.left; });",
    ],
    "charts_dashboard": [
        "# Chart.js v4 — verified signatures",
        "# Chart.js must be included via CDN (script tag) before your code runs.",
        "const ctx = document.getElementById('myChart').getContext('2d');",
        "const chart = new Chart(ctx, { type: 'line', data: { labels: [...], datasets: [...] }, options: {...} });",
        "# Type: 'line' | 'bar' | 'pie' | 'doughnut' | 'scatter' | 'radar'",
        "# Dataset shape:",
        "{ label: 'Sales', data: [10, 20, 30], borderColor: 'rgb(75, 192, 192)', backgroundColor: 'rgba(75, 192, 192, 0.2)' }",
        "# Make charts responsive AND resizable inside flex/grid layouts:",
        "options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' } } }",
        "# Update data without recreating:",
        "chart.data.datasets[0].data = newValues; chart.update();",
        "# When re-using a canvas for a new chart, destroy the old one first:",
        "oldChart.destroy();",
    ],
    "interactive_form": [
        "# Form handling — verified signatures",
        "form.addEventListener('submit', (e) => { e.preventDefault(); /* do work */ });",
        "# Collect all form values in one call:",
        "const data = Object.fromEntries(new FormData(form));",
        "# Native validation (HTML5 constraints like required/pattern):",
        "if (!form.checkValidity()) { form.reportValidity(); return; }",
        "# Persist + restore:",
        "localStorage.setItem('draft', JSON.stringify(data));",
        "const saved = JSON.parse(localStorage.getItem('draft') || 'null');",
        "# Input types worth knowing: 'email', 'tel', 'url', 'number', 'date', 'color', 'range'",
        "# Show validation errors inline (custom):",
        "input.setCustomValidity('must be at least 8 characters'); input.reportValidity();",
    ],
    "data_visualization": [
        "# Data rendering — verified idioms",
        "# Sort a copy so the original isn't mutated:",
        "const sorted = [...rows].sort((a, b) => a.value - b.value);",
        "# Aggregate:",
        "const total = rows.reduce((sum, r) => sum + r.value, 0);",
        "const byCategory = rows.reduce((acc, r) => { (acc[r.cat] ??= []).push(r); return acc; }, {});",
        "# Format numbers / currency / dates with Intl (browser built-in):",
        "new Intl.NumberFormat('en-US').format(1234567);  // '1,234,567'",
        "new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(n);",
        "new Date(ts).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });",
        "# Empty-state handling: render a message when arrays are empty — DO NOT render tables with zero rows and no explanation.",
    ],
}


def api_refs_for_categories(categories: list[str]) -> str:
    """Return a single prompt-ready block of refs for the given categories.

    Categories that have no ref entry are ignored. Returns an empty
    string when ``categories`` is empty or none of them match — callers
    can then skip adding the section to the prompt entirely.
    """
    if not categories:
        return ""
    sections: list[str] = []
    for cat in categories:
        lines = CATEGORY_API_REFS.get(cat)
        if not lines:
            continue
        sections.append("\n".join(lines))
    if not sections:
        return ""
    return "## API quick reference (use these — don't guess signatures)\n\n" + "\n\n".join(sections)
