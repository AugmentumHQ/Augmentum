"""Intent-keyed tool shortlist for coder priming.

The flat ``TOOL_REFERENCE`` prose catalog in ``prompts.py`` describes
every coder tool (~1.3K tokens). Most intents don't need every tool —
an INSPECT turn shouldn't be tempted by ``code_edit``; a RESEARCH turn
shouldn't see service controls. This module filters the catalog by
intent and returns a focused subset for prompt injection.

Single source of truth: per-tool descriptions live in
``TOOL_DESCRIPTIONS`` here. The flat ``TOOL_REFERENCE`` constant is kept
as a fallback (used by code paths that haven't been migrated yet).
"""
from __future__ import annotations

from augmentum.modes.coder.intent import TurnIntentKind

# ---------------------------------------------------------------------------
# Per-tool prose. Format mirrors TOOL_REFERENCE so a model that has read
# either source sees the same shape. Keep each entry under ~80 tokens.
# ---------------------------------------------------------------------------

TOOL_DESCRIPTIONS: dict[str, str] = {
    # --- Navigation & Reading ---
    "dir_tree": (
        "**dir_tree** — Directory hierarchy with depth control.\n"
        "- Input: `{\"path\": \"/workspace/src\", \"depth\": 3}`\n"
        "- Use first to understand project layout. Skips .git, node_modules, __pycache__."
    ),
    "file_read": (
        "**file_read** — Read one or several files with line numbers.\n"
        "- Single: `{\"path\": \"/workspace/src/main.py\"}`\n"
        "- Batch: `{\"paths\": [\"/workspace/a.py\", \"/workspace/b.py\"]}` — "
        "PREFER this when you already know 2+ files you need; one call "
        "replaces N.\n"
        "- **REQUIRED** before `code_edit` — the system enforces this "
        "(batch reads count).\n"
        "- Output format: `   1 | first line`."
    ),
    "file_list": (
        "**file_list** — Flat directory listing with sizes.\n"
        "- Input: `{\"path\": \"/workspace\"}`\n"
        "- Use `dir_tree` instead for hierarchical view."
    ),
    "env_info": (
        "**env_info** — Snapshot of workspace environment.\n"
        "- Input: `{}`\n"
        "- Returns installed runtimes, packages, project files, disk/memory.\n"
        "- Call at the start of complex tasks to avoid wasted turns."
    ),
    "container_info": (
        "**container_info** — Container identity, network, published ports.\n"
        "- Input: `{}`\n"
        "- Use for IP/port/runtime questions, esp. before reporting URLs."
    ),

    # --- Search ---
    "code_grep": (
        "**code_grep** — Regex search across files.\n"
        "- Input: `{\"pattern\": \"def login\", \"path\": \"/workspace/src\"}`\n"
        "- Add `\"context_lines\": 3` to see surrounding code — often "
        "saves the follow-up file_read.\n"
        "- When `code_edit` search fails, use this to find the current text."
    ),
    "find_files": (
        "**find_files** — Find files by name pattern.\n"
        "- Input: `{\"pattern\": \"*.test.js\", \"path\": \"/workspace\"}`\n"
        "- Finds files, not content. Use `code_grep` for content search."
    ),
    "code_search": (
        "**code_search** — Semantic search (find code by meaning).\n"
        "- Input: `{\"query\": \"authentication middleware\", \"limit\": 5}`\n"
        "- Best for: \"where is X handled?\" questions."
    ),

    # --- Editing ---
    "code_edit": (
        "**code_edit** — SEARCH/REPLACE with 4-tier matching.\n"
        "- Input: `{\"path\": \"...\", \"search\": \"old\", \"replace\": \"new\"}`\n"
        "- The `search` field must match text currently in the file.\n"
        "- Include 1-2 context lines for unique matching. If search fails, re-read.\n"
        "- Preferred over `file_write` for modifications."
    ),
    "code_edit_batch": (
        "**code_edit_batch** — Multiple SEARCH/REPLACE edits to one file atomically.\n"
        "- Input: `{\"path\": \"...\", \"edits\": [{\"search\": ..., \"replace\": ...}]}`\n"
        "- Use when several edits touch the same file; all apply or none do."
    ),
    "apply_patch": (
        "**apply_patch** — Apply a unified diff across files atomically.\n"
        "- Input: `{\"patch\": \"diff --git ...\"}`\n"
        "- Use for coordinated multi-file edits, renames, deletes."
    ),
    "file_write": (
        "**file_write** — Create or overwrite a complete file.\n"
        "- Input: `{\"path\": \"...\", \"content\": \"...\"}`\n"
        "- Use for NEW files. Use `code_edit` for modifying existing files."
    ),

    # --- Shell & Execution ---
    "shell_exec": (
        "**shell_exec** — Run any bash command (mutations allowed).\n"
        "- Input: `{\"command\": \"python3 script.py\"}`\n"
        "- `timeout` (secs, max 600): for a quiet long computation — runs "
        "pure wall-clock, no go-silent kill.\n"
        "- `run_in_background: true`: detach for a dev server or a run "
        "longer than 600s; watch with service_logs.\n"
        "- Use python3 (not python). Docker/mount/reboot commands are blocked."
    ),
    "shell_read": (
        "**shell_read** — Run read-only commands.\n"
        "- Input: `{\"command\": \"cat /etc/os-release\"}`\n"
        "- Like `shell_exec` but signals read-only intent."
    ),

    # --- Git ---
    "git": (
        "**git** — Structured git operations with parsed output.\n"
        "- Actions: status, diff, log, branch, commit.\n"
        "- Diff: `{\"action\": \"diff\", \"staged\": true}`.\n"
        "- Preferred over raw `shell_exec` for git operations."
    ),

    # --- Testing ---
    "test_run": (
        "**test_run** — Run tests with structured result parsing.\n"
        "- Input: `{\"command\": \"pytest -x\"}` or `{}` (auto-detects).\n"
        "- Returns: passed/failed/error counts + failure details.\n"
        "- Auto-detects: pytest, npm test, go test, cargo test, make test."
    ),

    # --- Documentation ---
    "doc_search": (
        "**doc_search** — Search programming documentation.\n"
        "- Input: `{\"query\": \"python asyncio gather\", \"language\": \"python\"}`\n"
        "- Searches official docs. Boosts trusted sources."
    ),
    "doc_fetch": (
        "**doc_fetch** — Read a documentation page.\n"
        "- Input: `{\"url\": \"https://docs.python.org/...\"}`\n"
        "- Extracts clean text with code examples."
    ),
    "pack_search": (
        "**pack_search** — Search installed OFFLINE knowledge packs.\n"
        "- Input: `{\"query\": \"asyncio TaskGroup cancellation\"}`\n"
        "- First resort for API/stdlib reference when a pack covers the "
        "language (the tool description lists installed packs); "
        "doc_search for the live web / anything time-sensitive."
    ),

    # --- Services ---
    "service_start": (
        "**service_start** — Start a long-running process.\n"
        "- Input: `{\"command\": \"uvicorn app:create_app --factory --port 8000\"}`\n"
        "- Returns a service id; runtime tracks lifecycle.\n"
        "- Preferred over `shell_exec &` for dev servers."
    ),
    "service_stop": (
        "**service_stop** — Stop a managed service.\n"
        "- Input: `{\"id\": \"svc-7a3b\"}`."
    ),
    "service_list": (
        "**service_list** — List managed services in this workspace.\n"
        "- Input: `{}`."
    ),
    "service_logs": (
        "**service_logs** — Tail logs for a managed service.\n"
        "- Input: `{\"id\": \"svc-7a3b\", \"tail\": 200}`."
    ),
    "service_probe": (
        "**service_probe** — HTTP-probe a managed service.\n"
        "- Input: `{\"id\": \"svc-7a3b\", \"path\": \"/health\"}`\n"
        "- Returns status code, latency, response preview."
    ),
    "publish_ports": (
        "**publish_ports** — Publish container ports for browser access.\n"
        "- Use when the user needs to hit the service from the host."
    ),

    # --- Browser ---
    "browser_open": (
        "**browser_open** — Open a URL in the headless browser.\n"
        "- Input: `{\"url\": \"http://localhost:8000\"}`\n"
        "- Returns the page snapshot (status, title, visible elements) "
        "directly — no follow-up browser_snapshot needed."
    ),
    "browser_snapshot": (
        "**browser_snapshot** — Capture page state (DOM + accessibility tree).\n"
        "- Input: `{}`."
    ),
    "browser_verify": (
        "**browser_verify** — Assert page conditions after an action.\n"
        "- Input: `{\"check\": \"text 'Welcome' is visible\"}`."
    ),
    "browser_screenshot": (
        "**browser_screenshot** — Capture a PNG of the page.\n"
        "- Input: `{}` for a fast best-effort full-page shot with viewport fallback.\n"
        "- Optional: `wait_until` (`domcontentloaded` default, `networkidle` strict), "
        "`timeout_ms`, `full_page`, `wait_for_selector`."
    ),
    "browser_evaluate": (
        "**browser_evaluate** — Run a JS expression in the page and get its value back.\n"
        "- Bare value: `{\"expression\": \"document.title\"}`\n"
        "- Function:  `{\"expression\": \"() => document.querySelectorAll('li').length\"}`\n"
        "- With args: `{\"expression\": \"(arg) => document.querySelector(arg.sel).textContent\", \"args\": {\"sel\": \"h1\"}}`\n"
        "- Scoped:    `{\"selector\": \"li.first\", \"expression\": \"el.textContent.trim()\"}`\n"
        "  (`el` is bound to the matched element when `selector` is set)\n"
        "- JS errors return structured {message, name, line, column, stack}.\n"
        "- Output JSON-trimmed by structure (arrays head + total, strings cap,\n"
        "  depth cap) so the value stays parseable.\n"
        "- Use to verify runtime state a screenshot can't show: hydration,\n"
        "  store contents, computed properties, fetch results, localStorage."
    ),

    "browser_wait": (
        "**browser_wait** — Wait for a page condition (NO setTimeout in "
        "browser_evaluate).\n"
        "- Selector: `{\"selector\": \".results\", \"state\": \"visible\"}`\n"
        "- Text: `{\"text\": \"Saved\"}` · Neither = network idle.\n"
        "- `timeout_ms` default 10000. On timeout returns the CURRENT page text."
    ),
    "browser_extract": (
        "**browser_extract** — Structured page data as JSON (NO "
        "querySelectorAll loops in browser_evaluate).\n"
        "- Kinds: text · links · table · list · meta · attr.\n"
        "- Input: `{\"kind\": \"table\", \"selector\": \"#results\"}` or "
        "`{\"kind\": \"attr\", \"selector\": \"img\", \"attribute\": \"src\"}`."
    ),
    "browser_fill_form": (
        "**browser_fill_form** — Fill several fields + submit in ONE call "
        "(instead of browser_type per field).\n"
        "- Input: `{\"fields\": {\"#email\": \"a@b.c\", \"#agree\": true}, "
        "\"submit\": \"button[type=submit]\", \"wait_after_text\": \"Welcome\"}`\n"
        "- Submit fires only when every field succeeded."
    ),

    # --- Sidecar-native browser verbs (persistent agent-browser session;
    #     page state carries across calls — click then screenshot works) ---
    "browser_interact": (
        "**browser_interact** — hover/dblclick/focus/check/uncheck/select/"
        "press/scroll/scrollintoview/drag/highlight in the persistent browser.\n"
        "- Input: `{\"action\": \"hover\", \"selector\": \"@e3\"}` · "
        "`{\"action\": \"press\", \"value\": \"Enter\"}` · "
        "`{\"action\": \"select\", \"selector\": \"#lang\", \"value\": \"fr\"}`\n"
        "- Selector accepts CSS or a snapshot @ref."
    ),
    "browser_navigate": (
        "**browser_navigate** — back/forward/reload/pushstate (SPA nav).\n"
        "- Input: `{\"action\": \"back\"}` or `{\"action\": \"pushstate\", \"url\": \"/settings\"}`."
    ),
    "browser_get": (
        "**browser_get** — Read one thing from the page: text/html/value/"
        "attr/title/url/count/box/styles.\n"
        "- Input: `{\"what\": \"text\", \"selector\": \"@e2\"}` or "
        "`{\"what\": \"count\", \"selector\": \".item\"}`."
    ),
    "browser_console": (
        "**browser_console** — Console log or uncaught errors accumulated "
        "across interactions (not just page load).\n"
        "- Input: `{}` or `{\"source\": \"errors\"}`; `\"clear\": true` resets."
    ),
    "browser_tabs": (
        "**browser_tabs** — list/new/switch/close tabs.\n"
        "- Input: `{}` (list) · `{\"action\": \"new\", \"url\": \"...\"}` · "
        "`{\"action\": \"switch\", \"tab\": \"t2\"}`."
    ),
    "browser_find": (
        "**browser_find** — Semantic (accessibility) element lookup — more "
        "robust than CSS when ids/classes churn.\n"
        "- Input: `{\"locator\": \"role\", \"value\": \"button\", \"action\": \"click\"}` "
        "or `{\"locator\": \"label\", \"value\": \"Email\", \"action\": \"fill\", \"text\": \"a@b.c\"}`."
    ),

    # --- HTTP / DB introspection ---
    "http_request": (
        "**http_request** — Structured HTTP probe (GET/POST/PUT/PATCH/DELETE/HEAD/OPTIONS).\n"
        "- Input: `{\"url\": \"...\", \"method\": \"POST\", \"headers\": {...}, \"body\": \"...\"}`\n"
        "- Returns status, headers, body. Use instead of hand-rolling curl\n"
        "  in shell_exec when probing an API or verifying a started service."
    ),
    "db_inspect": (
        "**db_inspect** — Read-only SQLite introspection.\n"
        "- Input: `{\"db_path\": \"/workspace/app.db\", \"action\": \"schema\"}`\n"
        "- Actions: schema, tables (with row counts), sample (top-N from a table),\n"
        "  query (SELECT/WITH/EXPLAIN/PRAGMA), integrity (PRAGMA integrity_check).\n"
        "- Writes refused — use shell_exec for those."
    ),

    # --- Coordination ---
    "task_list": (
        "**task_list** — Maintain a visible todo list for multi-step work.\n"
        "- Input: `{\"items\": [{\"content\": \"Run tests\", \"status\": \"in_progress\"}]}`\n"
        "- Use for 3+ step tasks; exactly one item should be in_progress."
    ),
    "ask_user": (
        "**ask_user** — Ask a clarifying question.\n"
        "- Use sparingly; prefer inspecting files/tools first."
    ),
    "finish_task": (
        "**finish_task** — Signal the turn is complete.\n"
        "- Call after the verifier passes (tests, validator, probe) or after\n"
        "  the answer prose is written for INSPECT/REVIEW/RESEARCH turns."
    ),

    # --- Profile (workspace memory) ---
    "profile_read": (
        "**profile_read** — Read this workspace's profile (conventions, facts).\n"
        "- Input: `{}` or `{\"category\": \"runtime\"}`."
    ),
    "profile_update": (
        "**profile_update** — Save a fact about this workspace.\n"
        "- Use for concise, validated facts; not every observation."
    ),
    "observe": (
        "**observe** — Record a durable cross-session fact.\n"
        "- Input: `{\"category\": \"build\", \"fact\": \"pytest is the test runner\"}`\n"
        "- Categories: build, test, deploy, api, data, env, constraint,\n"
        "  gotcha, style, other. Persists to "
        "/workspace/.augmentum/observations.jsonl.\n"
        "- Use for facts the NEXT session shouldn't have to re-discover.\n"
        "- Distinct from profile_update (structured per-user profile)\n"
        "  and from prose summary (workspace-scoped, jsonl-queryable)."
    ),
}


# ---------------------------------------------------------------------------
# Intent → tool subset. Order in each list matches presentation order.
# Goal: each intent sees only the tools relevant to its task shape.
# Frontier models can request anything they need; smaller models won't
# be tempted by tools that don't fit. finish_task always last.
# ---------------------------------------------------------------------------

_INSPECT_TOOLS = (
    "file_read", "file_list", "dir_tree", "code_grep", "find_files",
    "code_search", "env_info", "container_info", "shell_read",
    "db_inspect", "observe",
    "finish_task",
)

_REVIEW_TOOLS = (
    "file_read", "code_grep", "find_files", "code_search", "git",
    "observe", "finish_task",
)

_RESEARCH_TOOLS = (
    "pack_search", "doc_search", "doc_fetch", "browser_open", "browser_snapshot",
    "browser_extract", "browser_wait", "browser_evaluate", "browser_get",
    "browser_find", "http_request",
    "file_read", "code_grep", "observe", "finish_task",
)

_IMPLEMENT_TOOLS = (
    "file_read", "file_list", "dir_tree", "code_grep", "find_files",
    "code_search", "env_info",
    "code_edit", "code_edit_batch", "apply_patch", "file_write",
    "shell_exec", "test_run", "git",
    "task_list", "observe", "ask_user", "finish_task",
)

_DEBUG_TOOLS = (
    "file_read", "file_list", "dir_tree", "code_grep", "find_files",
    "code_search", "env_info",
    "test_run", "shell_exec", "code_edit", "code_edit_batch", "git",
    "http_request", "db_inspect", "browser_evaluate", "browser_wait",
    "browser_extract", "browser_console", "browser_get",
    "observe", "ask_user", "finish_task",
)

_OPERATE_TOOLS = (
    "file_read", "env_info", "container_info",
    "service_start", "service_list", "service_logs", "service_stop",
    "service_probe", "publish_ports",
    "browser_open", "browser_verify", "browser_snapshot",
    "browser_wait", "browser_extract", "browser_fill_form",
    "browser_evaluate", "browser_interact", "browser_find",
    "browser_navigate", "browser_get", "browser_console", "browser_tabs",
    "http_request",
    "shell_exec", "task_list", "observe", "finish_task",
)


INTENT_TOOLS: dict[TurnIntentKind, tuple[str, ...]] = {
    TurnIntentKind.INSPECT:   _INSPECT_TOOLS,
    TurnIntentKind.REVIEW:    _REVIEW_TOOLS,
    TurnIntentKind.RESEARCH:  _RESEARCH_TOOLS,
    TurnIntentKind.IMPLEMENT: _IMPLEMENT_TOOLS,
    TurnIntentKind.DEBUG:     _DEBUG_TOOLS,
    TurnIntentKind.OPERATE:   _OPERATE_TOOLS,
    # UNKNOWN falls back to IMPLEMENT (safe write-capable superset).
    TurnIntentKind.UNKNOWN:   _IMPLEMENT_TOOLS,
}


def tools_for_intent(intent_kind: TurnIntentKind | None) -> tuple[str, ...]:
    """Return the tool name list for an intent. Falls back to IMPLEMENT."""
    if intent_kind is None:
        return _IMPLEMENT_TOOLS
    return INTENT_TOOLS.get(intent_kind, _IMPLEMENT_TOOLS)


def render_tool_shortlist(intent_kind: TurnIntentKind | None) -> str:
    """Render the intent-filtered tool catalog as markdown prose.

    Used by text/structured-tier callers that can't pass native tool
    schemas. Native tier gets schemas via the request's ``tools=`` arg —
    don't inject this prose there; use ``render_native_intent_hint``
    for native tier instead.
    """
    names = tools_for_intent(intent_kind)
    blocks = ["## Available Tools"]
    for name in names:
        desc = TOOL_DESCRIPTIONS.get(name)
        if desc:
            blocks.append(desc)
    return "\n\n".join(blocks)


def render_native_intent_hint(intent_kind: TurnIntentKind | None) -> str:
    """Render a one-line preferred-tools hint for native-tier callers.

    Native models get full tool schemas via the request's ``tools=``
    arg — they have access to everything. This hint just nudges them
    toward the intent-relevant subset without removing options.

    Returns "" for None / UNKNOWN to avoid teaching a wrong shape on
    unclassified turns (full toolbox is the safe default).
    """
    if intent_kind is None or intent_kind == TurnIntentKind.UNKNOWN:
        return ""
    names = tools_for_intent(intent_kind)
    if not names:
        return ""
    intent_label = (
        intent_kind.value if hasattr(intent_kind, "value") else str(intent_kind)
    ).upper()
    tools_list = ", ".join(names)
    return (
        "## For this turn\n\n"
        f"Intent classified as: **{intent_label}**. "
        "Lean on these tools first: "
        f"{tools_list}. "
        "Others remain available — this is a soft preference, not a constraint."
    )
