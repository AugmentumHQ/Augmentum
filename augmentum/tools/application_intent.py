"""Derive feature suggestions from the app description (toolkit spec §6).

The App Builder's current ``verify_intent`` step catches "description
said X but no X in code" mismatches by keyword-grepping generated code.
That's a bolt-on check at the END of the pipeline — by the time it
fires, the plan is locked in and a fix means running the full improve
loop.

This module pushes intent into the PLAN step instead. Given the user's
description, it suggests feature names whose contract entries
(PROVIDES / WIRES) should appear in the generated plan. The plan prompt
then nudges the LLM to include those in the contract columns, which
means the runtime-contract validator (toolkit spec §3) catches missing
features natively — no separate verify_intent loop needed.

We don't prescribe exact symbol names: describing a feature as
"addition operation" and letting the LLM pick ``window.add`` or
``window.calc.add`` keeps both flexibility and verification.
"""

from __future__ import annotations

import re

# --- Feature definitions ---------------------------------------------------
# Each entry carries both PLAN-time hints (for derive_intent_features /
# format_intent_for_plan_prompt, nudging the LLM to surface contracts)
# AND VERIFY-time patterns (for verify_intent at end of pipeline, catching
# descriptions-mention-but-code-doesn't gaps). Keeping these unified
# prevents the tables drifting as new features are added.
#
# Fields:
#   label            — human-readable feature label (used for plan bullets)
#   keywords         — description-match triggers (whole-word)
#   provides         — PROVIDES hint for plan prompt (None → skip)
#   wires            — WIRES hint for plan prompt (None → skip)
#   verify_label     — short feature name used in verify_intent error msgs
#                      (None → skip verify — useful for CRUD rules where
#                      "add item" and "addition" share keywords; only the
#                      arithmetic rule should verify).
#   verify_patterns  — list of regex strings (OR logic). If ANY matches
#                      in all_code (case-insensitive), feature is
#                      considered implemented. Empty list → skip verify.
#
# Pattern design rule: patterns must be specific enough that a random
# JS file without the feature won't match. Raw single operators like
# ``+ `` are too loose — they hit CSS combinators, string concat, and
# import wildcards. Prefer named functions, compound operators, case
# dispatch strings, and feature-specific APIs.

# List preserves order so the suggestion block is stable across runs.
_INTENT_RULES: list[dict] = [
    # ===== Arithmetic — calculators, converters, tools =====
    # ``context_keywords`` disambiguates "add" (CRUD add-item) from "add"
    # (arithmetic). The rule fires only when one of the context words
    # *also* appears in the description, so a todo app's "add item,
    # remove item" doesn't spuriously demand addition operators.
    {
        "label": "addition",
        "keywords": ("add", "addition", "plus", "sum"),
        "context_keywords": ("calculator", "calc", "compute", "arithmetic", "math", "operation"),
        "provides": "an add/sum function (e.g. window.add, window.calc.add)",
        "wires": None,
        "verify_label": "addition operation",
        "verify_patterns": [
            r"\bfunction\s+(?:add|plus|sum)\b",
            r"\b(?:const|let|var)\s+(?:add|plus|sum)\s*=\s*(?:function|\(|async)",
            r"window\.(?:add|plus|sum)\b",
            r"case\s*['\"]\+['\"]",
            r"['\"]\+['\"]\s*:",
            r"operator\s*===?\s*['\"]\+['\"]",
            r"\+\+(?![+=])",
            r"\+=\s*\d",
            r"\.reduce\s*\(",
        ],
    },
    {
        "label": "subtraction",
        "keywords": ("subtract", "minus", "difference"),
        "context_keywords": ("calculator", "calc", "compute", "arithmetic", "math", "operation"),
        "provides": "a subtract function",
        "wires": None,
        "verify_label": "subtraction operation",
        "verify_patterns": [
            r"\bfunction\s+(?:subtract|minus)\b",
            r"\b(?:const|let|var)\s+(?:subtract|minus)\s*=\s*(?:function|\(|async)",
            r"window\.(?:subtract|minus)\b",
            r"case\s*['\"]-['\"]",
            r"['\"]-['\"]\s*:",
            r"operator\s*===?\s*['\"]-['\"]",
            r"--(?![-=])",
            r"-=\s*\d",
        ],
    },
    {
        "label": "multiplication",
        "keywords": ("multiply", "multiplication", "times", "product"),
        "context_keywords": ("calculator", "calc", "compute", "arithmetic", "math", "operation"),
        "provides": "a multiply function",
        "wires": None,
        "verify_label": "multiplication operation",
        "verify_patterns": [
            r"\bfunction\s+(?:multiply|times)\b",
            r"\b(?:const|let|var)\s+(?:multiply|times)\s*=\s*(?:function|\(|async)",
            r"window\.(?:multiply|times)\b",
            r"case\s*['\"]\*['\"]",
            r"['\"]\*['\"]\s*:",
            r"operator\s*===?\s*['\"]\*['\"]",
            r"\*=\s*\d",
        ],
    },
    {
        "label": "division",
        "keywords": ("divide", "division"),
        "context_keywords": ("calculator", "calc", "compute", "arithmetic", "math", "operation"),
        "provides": "a divide function (with zero-division guard)",
        "wires": None,
        "verify_label": "division operation",
        "verify_patterns": [
            r"\bfunction\s+divide\b",
            r"\b(?:const|let|var)\s+divide\s*=\s*(?:function|\(|async)",
            r"window\.divide\b",
            r"case\s*['\"]/['\"]",
            r"['\"]/['\"]\s*:",
            r"operator\s*===?\s*['\"]/['\"]",
            r"/=\s*\d",
        ],
    },
    {
        "label": "percent / percentage",
        "keywords": ("percent", "percentage"),
        "context_keywords": ("calculator", "calc", "compute", "arithmetic", "math", "tip"),
        "provides": "a percent function (e.g. window.percent or case '%')",
        "wires": None,
        "verify_label": "percent operation",
        "verify_patterns": [
            r"\bfunction\s+percent\b",
            r"window\.percent\b",
            r"case\s*['\"]%['\"]",
            r"['\"]%['\"]\s*:",
            r"/\s*100\b",
            r"\*\s*0\.01\b",
        ],
    },
    # ===== CRUD =====
    {
        "label": "add-item / create",
        # Bare "add" is ambiguous between CRUD and arithmetic; the context
        # gate is what keeps a todo/kanban app's "add" from missing the
        # CRUD rule while also not tripping arithmetic.
        "keywords": ("add", "add item", "create", "new task", "new entry"),
        "context_keywords": ("todo", "list", "kanban", "board", "task", "tasks",
                             "item", "items", "note", "notes", "card", "cards",
                             "entry", "entries"),
        "provides": "an add/create function that appends to the collection",
        "wires": "the form or button that triggers creation (e.g. #form submit or #add click)",
        "verify_label": "add/create functionality",
        "verify_patterns": [
            r"\bfunction\s+(?:add|create)\w*\s*\(",
            r"\b(?:const|let|var)\s+(?:add|create)\w*\s*=\s*(?:function|\(|async)",
            r"\.push\s*\(",
            r"\.unshift\s*\(",
            r"submit['\"]\s*,",
            r"#form\s",
        ],
    },
    {
        "label": "delete / remove",
        "keywords": ("delete", "remove"),
        "provides": "a remove function that drops an item from the collection",
        "wires": "per-item delete buttons (e.g. .item-delete click)",
        "verify_label": "delete/remove functionality",
        "verify_patterns": [
            r"\bfunction\s+(?:delete|remove)\w*\s*\(",
            r"\.splice\s*\(",
            r"\.filter\s*\(",
            r"\.delete\s*\(",
            r"removeChild\s*\(",
            r"\.remove\s*\(\s*\)",
        ],
    },
    {
        "label": "edit / update",
        "keywords": ("edit", "update", "rename", "modify"),
        "provides": "an update/edit function",
        "wires": "per-item edit affordance (e.g. .item-edit click or dblclick)",
        "verify_label": "edit/update functionality",
        "verify_patterns": [
            r"\bfunction\s+(?:edit|update|rename|modify)\w*\s*\(",
            r"\bdblclick\b",
            r"contenteditable",
            r"\.edit[\w-]*",
        ],
    },
    # ===== Persistence =====
    {
        "label": "persistence",
        "keywords": ("persist", "save state", "remember", "localstorage"),
        "provides": "save/load helpers using localStorage.setItem / getItem",
        "wires": None,
        "verify_label": "data persistence",
        "verify_patterns": [
            r"localStorage\.(?:setItem|getItem)",
            r"\bindexedDB\b",
            r"sessionStorage\.",
        ],
    },
    # ===== Search / sort / filter =====
    {
        "label": "search / filter",
        "keywords": ("search", "filter", "find"),
        "provides": "a filter/search function over the data",
        "wires": "the search input (e.g. #search input)",
        "verify_label": "search/filter functionality",
        "verify_patterns": [
            r"\.filter\s*\(",
            r"\.includes\s*\(",
            r"\.indexOf\s*\(",
            r"\.find\s*\(",
            r"\.search\b",
            r"#search\b",
        ],
    },
    {
        "label": "sort",
        "keywords": ("sort", "reorder", "arrange"),
        "provides": "a sort function",
        "wires": "the sort control (e.g. #sort-by change)",
        "verify_label": "sorting functionality",
        "verify_patterns": [
            r"\.sort\s*\(",
            r"\bfunction\s+sort\w*\s*\(",
            r"sortBy\s*[=(]",
        ],
    },
    # ===== Drag and drop =====
    {
        "label": "drag-and-drop",
        "keywords": ("drag", "draggable", "drag and drop"),
        "provides": "handlers for dragstart / dragover / drop",
        "wires": "the draggable container (dragstart/dragover/drop events)",
        "verify_label": "drag-and-drop",
        "verify_patterns": [
            r"\bdragstart\b",
            r"\bdragover\b",
            r"\bdraggable\b",
            r"\bdrop\b.*=.*function",
            r"addEventListener\s*\(\s*['\"]drop['\"]",
        ],
    },
    {
        "label": "undo / redo",
        "keywords": ("undo", "redo"),
        "provides": "an undo/redo history stack with snapshot on every mutation",
        "wires": "keyboard shortcuts (Ctrl+Z / Ctrl+Shift+Z) or undo buttons",
        "verify_label": "undo/redo history",
        "verify_patterns": [
            r"\bfunction\s+(?:undo|redo)\b",
            r"\bhistory\.(?:past|future|stack|undo|redo)\b",
            r"undoStack\b",
            r"\.key\s*[=.]*\s*['\"]z['\"]",
        ],
    },
    # ===== Game mechanics =====
    {
        "label": "score tracking",
        "keywords": ("score", "points"),
        "provides": "a score counter (e.g. window.gameState.score) and increment function",
        "wires": None,
        "verify_label": "score tracking",
        "verify_patterns": [
            r"\bscore\b",
            r"\bpoints\b",
        ],
    },
    {
        "label": "game over state",
        "keywords": ("game over", "gameover", "end game"),
        "provides": "a game-over function that freezes input and shows the state",
        "wires": None,
        "verify_label": "game over state",
        "verify_patterns": [
            r"\b(?:gameOver|game_over|gameover|endGame|end_game)\b",
        ],
    },
    {
        "label": "restart",
        "keywords": ("restart", "retry", "play again", "new game"),
        "provides": "a reset/restart function that zeroes state",
        "wires": "the restart button (e.g. #btn-restart click)",
        "verify_label": "restart/retry",
        "verify_patterns": [
            r"\b(?:restart|retry|newGame|startGame)\b",
            r"#btn-(?:restart|retry|new-game)\b",
        ],
    },
    {
        "label": "collision detection",
        "keywords": ("collision", "collide", "hit detect"),
        "provides": "a collision/intersection check between entities",
        "wires": None,
        "verify_label": "collision detection",
        "verify_patterns": [
            r"\b(?:collide|collision|intersect|overlap|hits?)\b.*\(",
            r"getBoundingClientRect",
        ],
    },
    # ===== Time =====
    {
        "label": "timer / countdown",
        "keywords": ("timer", "countdown", "stopwatch", "clock"),
        "provides": "start/stop/tick helpers using setInterval",
        "wires": "timer control buttons (e.g. #btn-start click, #btn-stop click)",
        "verify_label": "timer/countdown",
        "verify_patterns": [
            r"setInterval\s*\(",
            r"setTimeout\s*\(",
            r"requestAnimationFrame\s*\(",
            r"\btick\b",
        ],
    },
    {
        "label": "pause / resume",
        "keywords": ("pause", "resume"),
        "provides": "pause/resume state management for the timer",
        "wires": "a pause button (e.g. #btn-pause click)",
        "verify_label": "pause/resume",
        "verify_patterns": [
            r"\bfunction\s+(?:pause|resume)\b",
            r"\bpause\s*\(\s*\)",
            r"\brunning\s*=\s*false",
            r"\bpaused\s*=\s*(?:true|!)",
            r"clearInterval\s*\(",
            r"#btn-pause\b",
        ],
    },
    {
        "label": "lap times",
        "keywords": ("lap", "laps", "lap time", "lap times"),
        "provides": "a lap recorder that appends elapsed times to a list",
        "wires": "a lap button (e.g. #btn-lap click)",
        "verify_label": "lap-times recorder",
        "verify_patterns": [
            r"\blaps?\s*(?:\[|\.push|List|Array)",
            r"\brecordLap\b",
            r"#btn-lap\b",
            r"\bfunction\s+lap\w*\s*\(",
        ],
    },
    {
        "label": "progress ring / circular progress",
        "keywords": ("progress ring", "progress circle", "circular progress", "circular progress ring"),
        "provides": "an SVG ring whose stroke-dashoffset reflects elapsed/total",
        "wires": None,
        "verify_label": "circular progress ring",
        "verify_patterns": [
            r"stroke-dash(?:array|offset)",
            r"strokeDash(?:Array|Offset)",
            r"2\s*\*\s*Math\.PI",
        ],
    },
    {
        "label": "audio / sound alert",
        "keywords": ("sound alert", "audio alert", "beep", "chime", "alert sound"),
        "provides": "audio playback triggered at phase-end",
        "wires": None,
        "verify_label": "audio alert",
        "verify_patterns": [
            r"<audio\b",
            r"new\s+Audio\s*\(",
            r"\.play\s*\(\s*\)",
            r"AudioContext",
        ],
    },
    {
        "label": "memory store / recall (calculator)",
        "keywords": ("memory store", "memory recall", "memory clear", "mem store"),
        "provides": "memory store/recall functions (M+, MR, MC) backed by a variable",
        "wires": "memory buttons (e.g. #btn-ms click, #btn-mr click, #btn-mc click)",
        "verify_label": "memory store/recall",
        "verify_patterns": [
            r"\bmemory\s*[:=]",
            r"memoryStore\b",
            r"memoryRecall\b",
            r"memoryClear\b",
            r"#btn-m[srcp]\b",
            r"\bMS\b|\bMR\b|\bMC\b",
        ],
    },
    {
        "label": "history log",
        "keywords": ("history log", "expression history", "history of expressions", "last N expressions", "last 10 expressions"),
        "provides": "a bounded history array + render function",
        "wires": "a history list container (e.g. #history-list)",
        "verify_label": "history log",
        "verify_patterns": [
            r"history\s*(?:\[|\.push|\.unshift|List|Array|\.slice)",
            r"expressions\s*\[",
            r"#history\b",
        ],
    },
    # ===== Theming =====
    {
        "label": "dark mode toggle",
        "keywords": ("dark mode", "dark theme", "light mode", "theme toggle"),
        "provides": "a theme-toggle function that flips a class on <body>",
        "wires": "the toggle control (e.g. #theme-toggle click)",
        "verify_label": "dark/light theme",
        "verify_patterns": [
            r"\bdark\b",
            r"\btheme\b",
            r"\btoggle\b",
            r"prefers-color-scheme",
        ],
    },
    # ===== Validation =====
    {
        "label": "form validation",
        "keywords": ("validation", "validate", "required field"),
        "provides": "a validate function or use of form.checkValidity()",
        "wires": "the form submit (e.g. #form submit)",
        "verify_label": "form validation",
        "verify_patterns": [
            r"\bfunction\s+validate\w*\s*\(",
            r"\bcheckValidity\s*\(",
            r"\brequired\b",
            r"setCustomValidity",
        ],
    },
    # ===== Export / download =====
    {
        "label": "export / download",
        "keywords": ("export", "download", "pdf"),
        "provides": "an export function (Blob + URL.createObjectURL)",
        "wires": "the export button (e.g. #btn-export click)",
        "verify_label": "export/download",
        "verify_patterns": [
            r"new\s+Blob\s*\(",
            r"createObjectURL",
            r"download\s*=",
            r"\.toDataURL\s*\(",
        ],
    },
    {
        "label": "CSV export",
        "keywords": ("csv", "csv export", "export csv", "download csv"),
        "provides": "CSV serializer + Blob download",
        "wires": "the export button (e.g. #btn-csv click)",
        "verify_label": "CSV export",
        "verify_patterns": [
            r"text/csv",
            r"\.csv['\"]",
            r"join\s*\(\s*['\"],['\"]\s*\)",
            r"new\s+Blob\s*\([^)]+csv",
        ],
    },
    # ===== Copy to clipboard =====
    {
        "label": "copy to clipboard",
        "keywords": ("copy to clipboard", "clipboard", "copy hex", "copy button"),
        "provides": "a copy-to-clipboard function using navigator.clipboard",
        "wires": "copy buttons on each item (e.g. .copy click)",
        "verify_label": "copy-to-clipboard",
        "verify_patterns": [
            r"navigator\.clipboard",
            r"clipboard\.writeText",
            r"execCommand\s*\(\s*['\"]copy",
        ],
    },
    # ===== Color / palette math =====
    {
        "label": "color scheme math",
        "keywords": ("analogous", "complementary", "triadic", "tetradic", "color scheme", "color palette"),
        "provides": "hue-rotation helpers + scheme generators (analogous/complementary/triadic/tetradic)",
        "wires": "swatch grids, one per scheme",
        "verify_label": "color-scheme math (HSL / hue rotation)",
        "verify_patterns": [
            r"\b(?:hslToRgb|rgbToHsl|hslToHex|hexToHsl|rotateHue)\b",
            r"scheme(?:Analogous|Complementary|Triadic|Tetradic)",
            r"\b(?:analogous|complementary|triadic|tetradic)\b",
            r"\bhsl\s*\(",
        ],
    },
    # ===== Live preview / split pane =====
    {
        "label": "live preview (editor ↔ preview)",
        "keywords": ("live preview", "live render", "as you type"),
        "provides": "an input listener on the editor that renders into the preview",
        "wires": "editor textarea input → preview pane innerHTML",
        "verify_label": "live preview",
        "verify_patterns": [
            r"addEventListener\s*\(\s*['\"]input['\"]",
            r"\boninput\b",
            r"\bpreview\b.*innerHTML",
        ],
    },
    {
        "label": "split pane layout",
        "keywords": ("split pane", "split view", "side by side", "split screen"),
        "provides": "a two-pane layout (flex or grid) that splits content + preview",
        "wires": None,
        "verify_label": "split-pane layout",
        "verify_patterns": [
            r"display:\s*(?:flex|grid)",
            r"grid-template-columns",
            r"\bsplit-(?:left|right|pane|div)\b",
            r"\bsplit-view\b",
        ],
    },
    # ===== Chart types (dashboard-specific, fire only on the exact keyword) =====
    {
        "label": "pie chart",
        "keywords": ("pie chart", "pie"),
        "provides": "a Chart.js pie/doughnut chart instance",
        "wires": "a canvas for the pie chart (e.g. #chart-pie)",
        "verify_label": "pie chart",
        "verify_patterns": [
            r"type:\s*['\"]pie['\"]",
            r"type:\s*['\"]doughnut['\"]",
            r"\bpie\b.*\bChart\b",
        ],
    },
    {
        "label": "bar chart",
        "keywords": ("bar chart", "bar graph"),
        "provides": "a Chart.js bar chart instance",
        "wires": "a canvas for the bar chart (e.g. #chart-bar)",
        "verify_label": "bar chart",
        "verify_patterns": [
            r"type:\s*['\"]bar['\"]",
            r"\bbar\b.*\bChart\b",
        ],
    },
    {
        "label": "line chart",
        "keywords": ("line chart", "line graph"),
        "provides": "a Chart.js line chart instance",
        "wires": "a canvas for the line chart (e.g. #chart-line)",
        "verify_label": "line chart",
        "verify_patterns": [
            r"type:\s*['\"]line['\"]",
            r"\bline\b.*\bChart\b",
        ],
    },
    # ===== Notifications / toasts =====
    {
        "label": "toast / notification",
        "keywords": ("toast", "notification", "snackbar"),
        "provides": "a showToast/showNotification function",
        "wires": None,
        "verify_label": "toast/notification",
        "verify_patterns": [
            r"\b(?:showToast|showNotification|toast|snackbar)\b",
        ],
    },
]


def derive_intent_features(description: str) -> list[dict]:
    """Return feature suggestions that match the description.

    Each returned dict has ``label``, ``provides`` (str describing the
    expected PROVIDES contract entry or ``None`` if the feature is
    non-functional), and ``wires`` (str describing WIRES entry or
    ``None``). The list preserves :data:`_INTENT_RULES` order so the
    plan prompt block is stable — helpful for snapshotting / eval.

    Whole-word keyword match; "addition" won't accidentally hit in
    "additional". Returns ``[]`` for empty or signal-free descriptions.
    """
    if not description:
        return []
    lower = description.lower()
    suggestions: list[dict] = []
    seen_labels: set[str] = set()
    for rule in _INTENT_RULES:
        matched, _kw = _rule_matches_description(rule, lower)
        if not matched:
            continue
        if rule["label"] in seen_labels:
            continue
        seen_labels.add(rule["label"])
        suggestions.append({
            "label": rule["label"],
            "provides": rule["provides"],
            "wires": rule["wires"],
        })
    return suggestions


def _rule_matches_description(rule: dict, desc_lower: str) -> tuple[bool, str | None]:
    """Return (matched, matched_keyword). A rule matches when one of its
    keywords appears as a whole word AND, if ``context_keywords`` is set,
    at least one context word is also present. The context gate is what
    lets "add" in a calculator description trigger arithmetic rules
    while "add" in a todo description does not."""
    matched_kw = None
    for kw in rule["keywords"]:
        if re.search(rf"\b{re.escape(kw)}\b", desc_lower):
            matched_kw = kw
            break
    if not matched_kw:
        return False, None

    context = rule.get("context_keywords")
    if context:
        has_context = any(
            re.search(rf"\b{re.escape(ck)}\b", desc_lower) for ck in context
        )
        if not has_context:
            return False, None

    return True, matched_kw


def format_intent_for_plan_prompt(description: str) -> str:
    """Render the intent suggestions as a prompt block for the plan step.

    Empty when no features trigger — avoids padding the prompt with a
    "Required features: (none)" section for apps whose descriptions
    don't mention specific mechanics.
    """
    suggestions = derive_intent_features(description)
    if not suggestions:
        return ""
    lines = [
        "## Required features (map each to a contract entry in your plan)",
        "Your description implies these features. Each should surface in "
        "a PROVIDES or WIRES entry on the file that implements it — that "
        "way the validator catches missing wiring before runtime.",
        "",
    ]
    for s in suggestions:
        parts = [f"- **{s['label']}**"]
        if s["provides"]:
            parts.append(f"PROVIDES: {s['provides']}")
        if s["wires"]:
            parts.append(f"WIRES: {s['wires']}")
        lines.append("  \u2192 ".join(parts))
    return "\n".join(lines)


def verify_intent_gaps(description: str, all_code: str) -> list[str]:
    """Return INTENT: issue strings for features mentioned in the
    description but missing from the generated code.

    Single source of truth for the verify-time check. Iterates the unified
    :data:`_INTENT_RULES` table and consults ``verify_patterns`` on each
    rule whose ``keywords`` match the description. Patterns are regex —
    see the module-level comment for the design rule.

    Rules without a ``verify_label`` are plan-only (e.g. hints that are
    too ambiguous at verify time) and are skipped.
    """
    if not description or not all_code:
        return []

    desc_lower = description.lower()
    issues: list[str] = []

    for rule in _INTENT_RULES:
        verify_label = rule.get("verify_label")
        patterns = rule.get("verify_patterns") or []
        if not verify_label or not patterns:
            continue

        matched, matched_kw = _rule_matches_description(rule, desc_lower)
        if not matched:
            continue

        implemented = any(
            re.search(pat, all_code, re.IGNORECASE) for pat in patterns
        )
        if implemented:
            continue

        sample_pat = patterns[0].replace("\\b", "").replace("\\", "")[:60]
        issues.append(
            f"INTENT: Description mentions '{matched_kw}' but no {verify_label} "
            f"implementation found in the code. Expected something like: "
            f"{sample_pat}. Add the missing {verify_label} logic."
        )

    return issues
