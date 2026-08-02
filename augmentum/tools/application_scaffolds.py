"""Application builder scaffolds — project templates with defaults and prompts."""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# GBNF Grammars for llama.cpp constrained decoding
# ---------------------------------------------------------------------------
# These force models to produce valid structured output, eliminating parsing
# failures entirely on supported backends.

# Grammar for SEARCH/REPLACE fix output.
# Allows multiple === FILE: ... === sections, each with one or more
# SEARCH/REPLACE blocks, ending with __PASS_COMPLETE__.
GRAMMAR_SEARCH_REPLACE = r"""
root        ::= (file-section)+ pass-end
file-section ::= file-header (sr-block)+
file-header ::= "=== FILE: " filename " ===" nl
filename    ::= [^\n=]+
sr-block    ::= search-marker content divider content replace-marker nl?
search-marker ::= "<<<<<<< SEARCH" nl
divider     ::= "=======" nl
replace-marker ::= ">>>>>>> REPLACE" nl
content     ::= line*
line        ::= [^\n]* nl
nl          ::= "\n"
pass-end    ::= "__PASS_COMPLETE__" nl?
""".strip()

# Grammar for plan pass output (new build).
# Forces every file to declare its contract: ROLE/LANG/DESCRIPTION plus the
# three contract columns PROVIDES/DEPENDS/WIRES. The validator relies on
# these to catch missing wiring before generation runs, so they're not
# optional. Use the literal token "none" for empty contracts.
GRAMMAR_PLAN = r"""
root        ::= (file-line nl)+ pass-end
file-line   ::= "FILE: " path " | ROLE: " role " | LANG: " lang " | DESCRIPTION: " description " | PROVIDES: " contract-value " | DEPENDS: " contract-value " | WIRES: " contract-value
path        ::= [^ |]+
role        ::= "entry" | "style" | "script" | "module" | "data"
lang        ::= [^ |]+
description ::= [^|\n]+
contract-value ::= [^|\n]+
nl          ::= "\n"
pass-end    ::= "__PASS_COMPLETE__" nl?
""".strip()

# Iteration variant: existing files are described by ACTION (modify/create)
# and don't need contract emission — modifications are local edits, not
# new architecture. Kept separate so the new-build grammar above can stay
# strict without breaking the iterate path.
GRAMMAR_PLAN_ITERATE = r"""
root        ::= (file-line nl)+ pass-end
file-line   ::= "FILE: " path " | " (modify-line | create-line)
modify-line ::= "ACTION: modify | DESCRIPTION: " description
create-line ::= "ROLE: " role " | LANG: " lang " | ACTION: create | DESCRIPTION: " description
path        ::= [^ |]+
role        ::= "entry" | "style" | "script" | "module" | "data"
lang        ::= [^ |]+
description ::= [^\n]+
nl          ::= "\n"
pass-end    ::= "__PASS_COMPLETE__" nl?
""".strip()

SCAFFOLDS = {
    "static": {
        "name": "Static Web App",
        "description": "Single-page app with HTML + CSS + JS. No build step.",
        "default_files": [
            {"path": "index.html", "role": "entry", "lang": "html"},
            {"path": "styles.css", "role": "style", "lang": "css"},
            {"path": "app.js", "role": "script", "lang": "javascript"},
        ],
        "cdn_includes": [],
    },
    "dashboard": {
        "name": "Data Dashboard",
        "description": "Dashboard with charts and data tables. Includes Chart.js CDN.",
        "default_files": [
            {"path": "index.html", "role": "entry", "lang": "html"},
            {"path": "styles.css", "role": "style", "lang": "css"},
            {"path": "app.js", "role": "script", "lang": "javascript"},
        ],
        "cdn_includes": ["https://cdn.jsdelivr.net/npm/chart.js@4"],
        "entry_template": '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1.0">\n<title>{{title}}</title>\n<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>\n</head>\n<body>\n{{content}}\n</body>\n</html>',
    },
    "game": {
        "name": "Browser Game",
        "description": "Canvas-based game with game loop boilerplate.",
        "default_files": [
            {"path": "index.html", "role": "entry", "lang": "html"},
            {"path": "game.js", "role": "script", "lang": "javascript"},
        ],
        "cdn_includes": [],
        "entry_template": '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<title>{{title}}</title>\n<style>*{margin:0}canvas{display:block}</style>\n</head>\n<body>\n<canvas id="game"></canvas>\n</body>\n</html>',
    },
    "form": {
        "name": "Form / Tool",
        "description": "Interactive form or utility tool with validation and output.",
        "default_files": [
            {"path": "index.html", "role": "entry", "lang": "html"},
            {"path": "styles.css", "role": "style", "lang": "css"},
            {"path": "app.js", "role": "script", "lang": "javascript"},
        ],
        "cdn_includes": [],
    },
}


# ---------------------------------------------------------------------------
# Design Framework — model-agnostic guardrails for code generation quality
# ---------------------------------------------------------------------------
# These are injected into plan and generate prompts to compensate for model
# inconsistency. They're structural (testable), not stylistic (opinion).

DESIGN_FRAMEWORK = {
    # --- Universal rules (all scaffolds) ---
    "universal": [
        "Every class, function, and constant is defined EXACTLY once across all files — never duplicate definitions",
        "If splitting JS across files: each file uses an IIFE `(function(){ ... })()` and exports via `window.ModuleName = { ... }`",
        "CSS uses custom properties (:root { --color-primary: ... }) for all colors — enables theming",
        "Every DOM ID referenced in JS must exist in the HTML — plan IDs before writing JS",
        "Responsive: must work on mobile (320px+). Use CSS Grid or Flexbox, never fixed pixel widths for layout",
        "No stubs or placeholders — every function must have a real implementation",
    ],

    # --- Category-specific patterns (keyed by scaffold or auto-detected) ---
    "canvas_game": [
        "Game loop: requestAnimationFrame with delta-time `(timestamp - lastTime)` passed to update()",
        "Game states: at minimum MENU → PLAYING → GAME_OVER with clear transition functions",
        "Input: keydown/keyup state tracking (object of booleans), not keypress events",
        "Entities: single array, each has update(dt) + render(ctx), remove via markedForDeletion flag (iterate backwards)",
        "Canvas resizes to window on 'resize' event",
    ],
    "charts_dashboard": [
        "Each chart: unique canvas ID + dedicated init function",
        "Destroy existing chart before recreating (prevents canvas overlay bug)",
        "Data centralized in one source — chart inits reference it, don't inline data",
        "Layout: CSS Grid dashboard with sidebar + main content area",
    ],
    "interactive_form": [
        "Use <form> with addEventListener('submit', ...) + preventDefault() — not inline onclick",
        "Validation: inline error messages next to fields, not alert()",
        "Output/results shown in a dedicated section, not alert()",
        "If persisting: localStorage with JSON.stringify/parse and a named key",
    ],
    "data_visualization": [
        "Separate data layer from presentation — data in one file/object, rendering in another",
        "Include number formatting (toLocaleString, currency, percentages as appropriate)",
        "Color palette defined as constants — not hardcoded hex values scattered in code",
    ],

    # --- Anti-patterns (common LLM mistakes to explicitly forbid) ---
    "anti_patterns": [
        "NEVER use document.write() — it replaces the entire page",
        "NEVER put <script> or <link> tags referencing project files in HTML — files are assembled automatically",
        "NEVER use var — always const or let",
        "NEVER use alert() for user feedback — use DOM elements",
        "NEVER generate the same code twice in one file — if a class is complete, do not repeat it",
        "NEVER reference a function before defining it in the same file (except inside event listeners or DOMContentLoaded)",
    ],
}


def _detect_categories(description: str, scaffold_id: str) -> list[str]:
    """Auto-detect which design framework categories apply based on description + scaffold.

    Returns a list of category keys from DESIGN_FRAMEWORK to include in prompts.
    This is intentionally fuzzy — it's better to include a category unnecessarily
    than to miss one that was needed.
    """
    desc_lower = description.lower()
    categories = []

    # Scaffold-based detection
    if scaffold_id == "game":
        categories.append("canvas_game")
    elif scaffold_id == "dashboard":
        categories.append("charts_dashboard")
    elif scaffold_id == "form":
        categories.append("interactive_form")

    # Keyword-based detection (catches cross-scaffold usage)
    if (
        any(kw in desc_lower for kw in ("chart", "graph", "plot", "visualization", "analytics"))
        and "charts_dashboard" not in categories
    ):
        categories.append("charts_dashboard")
    if (
        any(kw in desc_lower for kw in ("canvas", "game", "sprite", "player", "enemy", "score"))
        and "canvas_game" not in categories
    ):
        categories.append("canvas_game")
    if (
        any(kw in desc_lower for kw in ("form", "input", "submit", "validation", "calculator", "converter"))
        and "interactive_form" not in categories
    ):
        categories.append("interactive_form")
    if (
        any(kw in desc_lower for kw in ("data", "table", "stats", "metrics", "kpi", "dashboard"))
        and "data_visualization" not in categories
    ):
        categories.append("data_visualization")

    return categories


def build_design_rules(description: str, scaffold_id: str) -> str:
    """Build the design framework rules string for prompts.

    Combines the computed design system (palette + typography —
    toolkit spec §1), category-specific API quick references
    (toolkit spec §2), universal rules, detected category rules, and
    anti-patterns into a concise prompt section.
    """
    from augmentum.tools.application_api_refs import api_refs_for_categories
    from augmentum.tools.application_design_system import compute_design_system

    categories = _detect_categories(description, scaffold_id)

    sections = []

    # Computed design system (palette + typography). Goes first so the
    # LLM sees the concrete palette before the abstract rules.
    ds = compute_design_system(description, scaffold_id)
    sections.append(ds.guidance_for_prompt())

    # Category-specific API refs — verified signatures for Canvas 2D,
    # Chart.js, forms, etc. Prevents small-model hallucinations up front.
    api_block = api_refs_for_categories(categories)
    if api_block:
        sections.append(api_block)

    # Universal rules always included
    sections.append("## Design Rules (MUST follow)\n" + "\n".join(f"- {r}" for r in DESIGN_FRAMEWORK["universal"]))

    # Category-specific design rules
    for cat in categories:
        rules = DESIGN_FRAMEWORK.get(cat, [])
        if rules:
            label = cat.replace("_", " ").title()
            sections.append(f"## {label} Patterns\n" + "\n".join(f"- {r}" for r in rules))

    # Anti-patterns always included
    sections.append("## Anti-Patterns (NEVER do these)\n" + "\n".join(f"- {r}" for r in DESIGN_FRAMEWORK["anti_patterns"]))

    return "\n\n".join(sections)

SCALE_GUIDE = """Project scale guide — use your judgment:

MICRO (2-3 files): Single-purpose tools, calculators, simple forms
  → index.html + styles.css + app.js

SMALL (4-5 files): Apps with state management, multiple UI sections
  → entry + styles + app logic + data/state module + (config or utils)

MEDIUM (6-7 files): Apps with distinct subsystems, components, themes
  → entry + styles + app + components + state/api + utils + (data or config)

FULL (8-10 files): Multi-view apps, dashboards, complex interactions
  → entry + layout styles + component styles + router + views + components + state + api + utils + config

Split a file when it would exceed ~150 lines or handles two unrelated concerns.
Don't split for the sake of splitting — a 3-file app that works is better than a 7-file app with unnecessary abstraction."""


def build_plan_prompt(description: str, scaffold_id: str, existing_files: list | None = None) -> list[dict]:
    """Build the messages list for the plan phase."""
    scaffold = SCAFFOLDS.get(scaffold_id, SCAFFOLDS["static"])
    is_iteration = existing_files is not None and len(existing_files) > 0

    if is_iteration:
        # Build semantic file summaries (purpose + exports, not raw code)
        file_summaries = []
        for f in existing_files:

            content = f["content"]
            lines = content.split("\n")
            count = len(lines)

            # Extract what the file provides (semantic summary)
            parts = [f'{f["path"]} ({f.get("role", "unknown")}, {count} lines)']

            if f.get("role") == "entry":
                # For HTML: list IDs, forms, interactive elements
                ids = re.findall(r'id=["\'](\w[\w-]*)["\']', content)
                forms = re.findall(r'<form[^>]*(?:id=["\'](\w+)["\'])?', content)
                buttons = len(re.findall(r'<button\b', content))
                parts.append(f"  IDs: {', '.join(ids[:10]) if ids else 'none'}")
                if forms:
                    parts.append(f"  Forms: {len(forms)}")
                if buttons:
                    parts.append(f"  Buttons: {buttons}")
            elif f.get("role") == "style":
                # For CSS: list custom properties, key selectors
                props = re.findall(r'--[\w-]+(?=\s*:)', content)
                classes = re.findall(r'\.(\w[\w-]*)\s*[{,]', content)
                has_responsive = '@media' in content
                parts.append(f"  Custom props: {', '.join(props[:8]) if props else 'none'}")
                parts.append(f"  Key classes: {', '.join(classes[:10]) if classes else 'none'}")
                if has_responsive:
                    parts.append("  Has responsive breakpoints")
            elif f.get("role") in ("script", "module"):
                # For JS: list functions, classes, window exports, event listeners
                funcs = re.findall(r'(?:function\s+(\w+)|(?:const|let)\s+(\w+)\s*=\s*(?:function|\(.*?\)\s*=>))', content)
                func_names = [m[0] or m[1] for m in funcs]
                classes = re.findall(r'class\s+(\w+)', content)
                exports = re.findall(r'window\.(\w+)\s*=', content)
                listeners = re.findall(r"addEventListener\s*\(\s*['\"](\w+)['\"]", content)
                parts.append(f"  Functions: {', '.join(func_names[:10]) if func_names else 'none'}")
                if classes:
                    parts.append(f"  Classes: {', '.join(classes)}")
                if exports:
                    parts.append(f"  Exports (window.X): {', '.join(exports[:8])}")
                if listeners:
                    parts.append(f"  Event listeners: {', '.join(set(listeners))}")

            file_summaries.append("\n".join(parts))

        files_context = "\n\n".join(file_summaries)

        system = (
            "You are planning modifications to an existing web application.\n\n"
            "RULES:\n"
            "1. ONLY list files that NEED changes. Unchanged files are kept as-is automatically.\n"
            "2. Be precise — if only CSS needs fixing, list only the CSS file.\n"
            "3. Think about which files are ACTUALLY affected by the request:\n"
            "   - Visual/styling changes → usually just the CSS file\n"
            "   - New interactive behavior → JS file (and maybe HTML for new elements)\n"
            "   - Layout/structure changes → HTML file (and maybe CSS)\n"
            "   - New feature with its own UI → may need a new file + HTML changes for navigation\n"
            "4. For each file, describe the SPECIFIC change — not a vague summary.\n\n"
            "OUTPUT FORMAT:\n"
            "For files to MODIFY:\n"
            "FILE: <path> | ACTION: modify | DESCRIPTION: <specific change>\n\n"
            "For NEW files:\n"
            "FILE: <path> | ROLE: <entry|style|script|module|data> | LANG: <language> | ACTION: create | DESCRIPTION: <purpose>\n\n"
            "EXAMPLE — user says 'add dark mode toggle':\n"
            "FILE: index.html | ACTION: modify | DESCRIPTION: Add a toggle button with id=\"dark-toggle\" in the header\n"
            "FILE: styles.css | ACTION: modify | DESCRIPTION: Add body.dark class with inverted color scheme using CSS custom properties\n"
            "FILE: app.js | ACTION: modify | DESCRIPTION: Add click handler for #dark-toggle that toggles body.dark class and saves to localStorage\n\n"
            "EXAMPLE — user says 'fix the button hover color':\n"
            "FILE: styles.css | ACTION: modify | DESCRIPTION: Change .btn:hover background-color\n\n"
            "End with __PASS_COMPLETE__"
        )
        user = f"Project files:\n\n{files_context}\n\nUser request: {description}"
    else:
        from augmentum.tools.application_intent import format_intent_for_plan_prompt

        defaults_desc = ", ".join(f["path"] for f in scaffold["default_files"])
        design_rules = build_design_rules(description, scaffold_id)
        intent_block = format_intent_for_plan_prompt(description)
        # Empty when no intent signals fire — keep the prompt tight in that case.
        intent_section = f"{intent_block}\n\n" if intent_block else ""
        system = (
            f"You are a senior frontend architect planning a production-quality web application.\n"
            f"Scaffold: '{scaffold['name']}' — Default files: {defaults_desc}\n"
            f"{'CDN libraries: ' + ', '.join(scaffold['cdn_includes']) if scaffold['cdn_includes'] else 'No CDN libraries.'}\n\n"
            f"{SCALE_GUIDE}\n\n"
            f"{design_rules}\n\n"
            f"{intent_section}"
            "Plan a well-architected file structure. Each file should have a clear, single responsibility.\n"
            "Think about: separation of concerns, reusable components, clean data flow.\n\n"
            "Output a file list. EVERY file MUST declare its contract — the "
            "validator uses these columns to catch missing wiring before "
            "generation runs. Use this exact single-line format with ALL "
            "seven columns present:\n"
            "FILE: <path> | ROLE: <entry|style|script|module|data> | LANG: <language>"
            " | DESCRIPTION: <specific purpose>"
            " | PROVIDES: <comma-separated exports (window.X, className, functionName)>"
            " | DEPENDS: <comma-separated imports this file consumes from others>"
            " | WIRES: <DOM selectors + events this file attaches, e.g. '#btn-go click, #form submit'>\n\n"
            "PROVIDES/DEPENDS/WIRES are REQUIRED. Use the literal token 'none' "
            "when a column is genuinely empty (e.g. an HTML entry typically has "
            "PROVIDES: none, DEPENDS: none, WIRES: none). Put ONLY bare symbol "
            "names — no parenthetical descriptions, no types, no comments. "
            "Examples:\n"
            "  PROVIDES: window.calculate, window.Calculator\n"
            "  DEPENDS: window.formatNumber\n"
            "  WIRES: #btn-calc click, #form submit\n\n"
            "DO write:  PROVIDES: window.TimerController, window.saveState\n"
            "DON'T write: PROVIDES: window.TimerController (main class), window.saveState (persistence helper)\n\n"
            "Your descriptions should be specific about WHAT the file does, not generic ('handles logic').\n"
            "Good: 'Manages task CRUD operations, localStorage persistence, and drag-drop state'\n"
            "Bad: 'Main application logic'\n\n"
            "Choose the right number of files for the complexity. Don't over-split simple apps.\n"
            "End with __PASS_COMPLETE__"
        )
        user = f"Build: {description}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_generate_prompt(files_to_generate: list, existing_files: dict, project_description: str) -> list[dict]:
    """Build the messages list for the generate phase (one iteration)."""
    existing_context = ""
    if existing_files:
        parts = []
        for path, content in existing_files.items():
            parts.append(f"=== {path} ===\n{content}")
        existing_context = "\n\n".join(parts) + "\n\n---\n\n"

    file_list = "\n".join(f"- {f['path']} ({f.get('role', 'script')}): {f.get('description', '')}" for f in files_to_generate)

    system = (
        "You are generating files for a web application.\n"
        "Output each file as a fenced code block with the filename:\n\n"
        "```filename.ext\n"
        "file content here\n"
        "```\n\n"
        "Rules:\n"
        "- Generate complete, working code — not stubs or placeholders\n"
        "- Use modern, clean HTML/CSS/JS\n"
        "- Make it visually polished — good spacing, colors, typography, transitions\n"
        "- Include responsive design (mobile-friendly)\n"
        "- Add hover states, focus styles, and subtle animations\n"
        "- Use CSS custom properties for theming\n"
        "- If all requested files are generated, end with __PASS_COMPLETE__\n"
        "- If you realize additional files are needed, end with __NEEDS_ANOTHER_PASS__: <reason>"
    )
    user = f"{existing_context}Project: {project_description}\n\nGenerate these files:\n{file_list}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Error enrichment — toolkit spec §4
#
# Maps common LLM-hallucinated API mistakes to concrete corrections. When
# the validator surfaces one of these strings, the fix prompt gets an
# explicit suggestion appended, which small models (≤13B) can't reliably
# derive from the raw error alone. Format: literal substring → suggestion.
# ---------------------------------------------------------------------------

API_CORRECTIONS: dict[str, str] = {
    # Canvas 2D — methods LLMs invent that don't exist
    "fillCircle":
        "Canvas 2D has no fillCircle(). Use ctx.beginPath(); "
        "ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();",
    "drawCircle":
        "Canvas 2D has no drawCircle(). Use ctx.arc(x, y, r, 0, Math.PI * 2) "
        "followed by ctx.stroke() or ctx.fill().",
    "fillText is not a function":
        "ctx.fillText(text, x, y) requires an initialized 2D context — "
        "check that getContext('2d') succeeded before calling fillText.",
    # jQuery-style methods that don't exist on DOM elements
    "addClass is not a function":
        "Vanilla DOM uses element.classList.add('foo'), not addClass().",
    "removeClass is not a function":
        "Vanilla DOM uses element.classList.remove('foo'), not removeClass().",
    "hasClass is not a function":
        "Vanilla DOM uses element.classList.contains('foo'), not hasClass().",
    "toggleClass is not a function":
        "Vanilla DOM uses element.classList.toggle('foo'), not toggleClass().",
    ".val is not a function":
        "Use element.value for inputs, element.textContent for spans/divs. "
        ".val() is a jQuery idiom.",
    ".html is not a function":
        "Use element.innerHTML = '...' or .textContent, not .html().",
    # document.getElementByID (common typo — correct is getElementById)
    "getElementByID is not a function":
        "JavaScript is case-sensitive: document.getElementById (lowercase 'd'), "
        "not getElementByID.",
    # localStorage.save / .get — common LLM confusion with real API
    "localStorage.save is not a function":
        "localStorage uses setItem(key, value) / getItem(key), not save/get.",
    "localStorage.get is not a function":
        "Use localStorage.getItem(key), not .get().",
    # Chart.js common mistakes
    "Chart is not defined":
        "Include Chart.js via <script src='https://cdn.jsdelivr.net/npm/chart.js'></script> "
        "in the HTML head before your script that uses it.",
    "canvas.getContext is not a function":
        "You probably passed the Chart constructor a selector string instead of a "
        "canvas element. Use document.getElementById('myCanvas') or new Chart(ctx, …).",
    # Event-listener signature mistakes
    "addEventListener is not a function":
        "addEventListener exists on Element, not on a NodeList. "
        "If you used querySelectorAll, iterate with .forEach before attaching listeners.",
    # Module/import errors in a browser context
    "Cannot use import statement outside a module":
        "The pipeline assembles all files into one HTML document with a single "
        "global scope — do NOT use ES module import/export. Expose cross-file "
        "symbols via window.X and reference them as window.X from other files.",
    "export is not a function":
        "See above — the bundled app has no ES modules. Replace `export X` with "
        "`window.X = X`.",
    # Reference errors — suggest checking spelling / window scope
    "ReferenceError:":
        "If a symbol is defined in another file, reference it as window.SYMBOL. "
        "If it's meant to be local, check for typos or that you're calling it "
        "after the definition runs.",
}


def enrich_errors(errors: list[str]) -> list[str]:
    """Return ``errors`` with concrete correction hints appended.

    Matches each error against ``API_CORRECTIONS`` (case-sensitive
    substring) and appends the suggestion inline. Errors without a match
    pass through unchanged. Deduplicates so the same suggestion isn't
    appended twice when multiple errors share a root cause.

    This closes the small-model gap where the fix prompt received raw
    browser errors and expected the LLM to infer the correction — which
    works for frontier models but routinely fails at 4-13B sizes.
    """
    out: list[str] = []
    seen_hints: set[str] = set()
    for err in errors:
        err_s = str(err)
        suggestion = None
        for needle, hint in API_CORRECTIONS.items():
            if needle in err_s:
                suggestion = hint
                break
        if suggestion and suggestion not in seen_hints:
            out.append(f"{err_s}\n    Hint: {suggestion}")
            seen_hints.add(suggestion)
        else:
            out.append(err_s)
    return out


# ---------------------------------------------------------------------------
# Model-adaptive prompt complexity — toolkit spec §5
#
# Small models (≤8B) drop instructions silently when the system prompt
# exceeds ~500 tokens; frontier models tolerate 4x that budget. We pick
# a tier from the model name and attach a compact steering preamble for
# the small tiers rather than surgically rewriting every prompt.
# ---------------------------------------------------------------------------

_MODEL_PARAM_RE = re.compile(
    r"(?:(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])|-(\d+(?:\.\d+)?)b-)",
)
_FRONTIER_SUBSTRINGS = (
    "gpt-4", "gpt-5", "claude", "gemini", "grok",
    "o1-", "o3-", "o4-", "mistral-large", "deepseek-r1",
)


def detect_model_tier(model_name: str) -> str:
    """Classify ``model_name`` into ``small`` / ``medium`` / ``large`` /
    ``frontier`` to drive prompt complexity. Uses parameter-count hints
    in the name (``"qwen3-7b-instruct"`` → small) and a known-family
    allowlist for hosted frontier models.

    Returns ``"medium"`` for unknown/empty names — a safe default that
    keeps full prompts flowing without aggressive trimming.
    """
    if not model_name:
        return "medium"
    lower = model_name.lower()
    for marker in _FRONTIER_SUBSTRINGS:
        if marker in lower:
            return "frontier"
    m = _MODEL_PARAM_RE.search(lower)
    if m:
        params = float(m.group(1) or m.group(2))
        if params <= 8:
            return "small"
        if params <= 32:
            return "medium"
        return "large"
    return "medium"


def adapt_prompt_for_tier(system_prompt: str, tier: str) -> str:
    """Prepend a tier-appropriate steering preamble to a system prompt.

    Small tier gets an explicit "stay on the happy path" preamble so it
    doesn't reach for advanced APIs it will probably hallucinate. Larger
    tiers get the prompt unmodified. Non-destructive — the original
    instructions still follow verbatim.
    """
    if tier != "small":
        return system_prompt
    preamble = (
        "You are a small model — stick to the happy path. Prefer vanilla "
        "DOM APIs (document.getElementById, element.classList, "
        "element.addEventListener). Avoid ES modules (no import/export). "
        "Use plain <script> tags and window.X for cross-file symbols. "
        "When in doubt, choose the simpler approach.\n\n"
    )
    return preamble + system_prompt


def build_fix_prompt(
    files: list,
    errors: list,
    previous_attempts: list[str] | None = None,
    *,
    model_name: str = "",
) -> list[dict]:
    """Build the messages list for the fix phase.

    Uses targeted context compression (Manus pattern): files mentioned in
    error messages get full content; other files get a compact signature
    listing their global exports.  This reduces tokens by ~60-80% for
    multi-file projects while giving the LLM everything it needs.

    ``previous_attempts`` (Manus error-preservation pattern): if prior fix
    attempts failed, their SEARCH/REPLACE patches are included so the model
    knows what was already tried and avoids repeating the same broken fix.

    ``model_name`` (toolkit spec §5) is used to tier-adapt the system
    prompt — small models get an extra steering preamble that keeps
    them on the vanilla-JS happy path instead of reaching for ES
    modules they'll fail to wire up correctly.
    """


    enriched = enrich_errors(errors)
    error_text = "\n".join(f"- {e}" for e in enriched)

    # Determine which files are mentioned in error messages
    affected: set[str] = set()
    for e in errors:
        err_str = str(e)
        for f in files:
            if f["path"] in err_str:
                affected.add(f["path"])
    # If we can't determine affected files, send all in full (safe fallback)
    if not affected:
        affected = {f["path"] for f in files}

    context_parts = []
    for f in files:
        if f["path"] in affected:
            context_parts.append(f"=== {f['path']} (FULL) ===\n{f['content']}")
        else:
            # Compact signature: global exports only
            exports = []
            for m in re.finditer(
                r'(?:window\.(\w+)\s*=|^(?:function|class|const|var|let)\s+(\w+))',
                f["content"], re.MULTILINE,
            ):
                exports.append(m.group(1) or m.group(2))
            sig = ", ".join(exports[:20]) if exports else "none"
            context_parts.append(f"=== {f['path']} (signature) ===\nExports: {sig}")

    file_context = "\n\n".join(context_parts)

    system = (
        "You are fixing code errors. Follow these steps exactly:\n"
        "1. Read the error messages below\n"
        "2. Find the root cause in the file content\n"
        "3. Output SEARCH/REPLACE blocks to fix it\n"
        "4. The SEARCH section must contain EXACT lines from the current file\n\n"
        "Output format (no other text):\n"
        "=== FILE: <filename> ===\n"
        "<<<<<<< SEARCH\n"
        "exact existing lines\n"
        "=======\n"
        "fixed lines\n"
        ">>>>>>> REPLACE\n\n"
        "Important:\n"
        "- Copy the SEARCH lines EXACTLY from the file (whitespace matters)\n"
        "- Fix the root cause, not symptoms\n"
        "- Do not change unrelated code\n"
        "- Files marked (FULL) contain the error. Files marked (signature) show exports only\n"
        "- All files share ONE global scope when assembled. Use window.X for cross-file access\n"
        "- End with __PASS_COMPLETE__"
    )
    # Build user message with optional previous attempts
    parts = [f"Files:\n\n{file_context}\n\n---\nErrors:\n{error_text}"]
    if previous_attempts:
        attempts_text = "\n\n".join(
            f"Attempt {i+1} (FAILED — do NOT repeat this):\n{a}"
            for i, a in enumerate(previous_attempts)
        )
        parts.append(f"\n\n---\nPrevious fix attempts that FAILED (try a different approach):\n{attempts_text}")
    parts.append("\n\nFix now.")
    user = "".join(parts)

    system = adapt_prompt_for_tier(system, detect_model_tier(model_name))

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_judge_prompt(files: list, description: str) -> list[dict]:
    """Build the messages list for the judge phase."""
    file_list = ", ".join(f["path"] for f in files)

    system = (
        "You are reviewing a web application for quality.\n"
        "Score it 1-10 on: functionality, visual polish, code quality, UX.\n"
        "Output format:\n\n"
        "SCORE: <N>/10\n"
        "STRENGTHS: <bullet list>\n"
        "IMPROVEMENTS: <bullet list of specific, actionable improvements>\n\n"
        "If score >= 8.5, end with __PASS_COMPLETE__\n"
        "If score < 8.5, end with __NEEDS_ANOTHER_PASS__: improvements listed above"
    )

    # Send abbreviated files to save tokens
    file_summaries = []
    for f in files:
        lines = f["content"].split("\n")
        if len(lines) <= 30:
            file_summaries.append(f"=== {f['path']} ({len(lines)} lines) ===\n{f['content']}")
        else:
            head = "\n".join(lines[:15])
            tail = "\n".join(lines[-10:])
            file_summaries.append(f"=== {f['path']} ({len(lines)} lines) ===\n{head}\n... ({len(lines) - 25} lines omitted) ...\n{tail}")

    user = f"Project: {description}\nFiles: {file_list}\n\n" + "\n\n".join(file_summaries)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Comprehension Prompt (Constraint-Driven Synthesis v2)
# ---------------------------------------------------------------------------

def build_comprehension_prompt(description: str, scaffold_id: str) -> list[dict]:
    """Build prompt for Phase 1: extract structured spec from user description.

    The model outputs a JSON spec with elements, state schema, and behavioral
    constraints. This is a classification/extraction task, not generation --
    even small models can fill this structure.
    """
    # SCAFFOLDS lookup kept as a tripwire: a bogus scaffold_id would
    # have tripped a KeyError here instead of later in the pipeline.
    _ = SCAFFOLDS.get(scaffold_id, SCAFFOLDS["static"])
    scaffold_hint = ""
    if scaffold_id == "game":
        scaffold_hint = "\nThis is a GAME project. Include a canvas element and game-related constraints (game loop, input, entities, score)."
    elif scaffold_id == "dashboard":
        scaffold_hint = "\nThis is a DASHBOARD project. Include chart containers and data-related constraints (chart rendering, data entry, export)."
    elif scaffold_id == "form":
        scaffold_hint = "\nThis is a FORM/TOOL project. Include form elements and validation constraints."

    system = (
        "You are a requirements analyst. Read the user's app description and output a JSON specification.\n\n"
        "CRITICAL: You are writing a SPECIFICATION, not code. Do NOT write JavaScript, HTML, or CSS.\n"
        "The JSON describes WHAT the app does, not HOW. No implementation details.\n\n"
        "EXAMPLE input: \"todo list with add and delete\"\n"
        "EXAMPLE output:\n"
        '{"name":"Todo List","state_schema":{"todos":"array","nextId":"number"},'
        '"elements":['
        '{"id":"app","tag":"main","role":"container","label":"","parent":""},'
        '{"id":"todo-input","tag":"input","role":"field","label":"","parent":""},'
        '{"id":"add-btn","tag":"button","role":"action","label":"Add","parent":""},'
        '{"id":"todo-list","tag":"ul","role":"display","label":"","parent":""}'
        '],"constraints":['
        '{"id":"c1","behavior":"render","description":"App renders with input, button, and list","type":"structural","trigger":{},"expected":{},"depends_on":[]},'
        '{"id":"c2","behavior":"add-todo","description":"Clicking #add-btn creates a todo in #todo-list","type":"interaction","trigger":{"event":"click","target":"#add-btn"},"expected":{"new_element":".todo-item"},"depends_on":["c1"]},'
        '{"id":"c3","behavior":"delete-todo","description":"Clicking .delete-btn removes the todo","type":"interaction","trigger":{"event":"click","target":".delete-btn"},"expected":{},"depends_on":["c2"]}'
        "]}\n\n"
        "RULES:\n"
        "1. EVERY element MUST have a unique \"id\" (e.g. \"board\", \"add-card-btn\", \"edit-modal\")\n"
        "2. Each constraint describes ONE behavior with: id, behavior, description, type, trigger, expected, depends_on\n"
        "3. trigger and expected are objects — use {} if not applicable, NEVER null\n"
        "4. Constraint types: structural, interaction, persistence, canvas, timer\n"
        "5. Extract EVERY behavior mentioned in the description as a separate constraint\n"
        "6. First constraint is always structural (elements render)\n"
        "7. Persistence needs TWO constraints: save AND restore\n"
        "8. Undo and redo are separate constraints\n"
        f"{scaffold_hint}\n\n"
        "Output ONLY the JSON object in COMPACT form (no pretty-printing, no newlines).\n"
        "state_schema values should be simple type names (string, number, array), not descriptions.\n"
        "No code. No explanations. No markdown."
    )

    user = f"Build: {description}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

