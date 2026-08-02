"""Tool-name allow-lists and filter helpers for subagent roles.

Three layers:

* **Tool presets** (``READ_ONLY_TOOL_NAMES``, ``VERIFIER_TOOL_NAMES``,
  ``FIXER_TOOL_NAMES``) — frozensets of tool names grouped by risk
  level. Used by role files via the ``tools: read_only`` shorthand.
* **Role allow-lists** (``PLANNER_TOOL_NAMES``, ``DETECTOR_TOOL_NAMES``)
  — same presets aliased per bug-finder role, kept separate so future
  role-specific tools (``rank_files_by_churn`` for planner) have a home
  without touching the shared presets.
* **``filter_tools()``** — applies an allow-list to a tool registry.

Generic across all subagent consumers: bug_finder and coder both
import from here.
"""

from __future__ import annotations

from augmentum.tools.base import Tool

READ_ONLY_TOOL_NAMES: frozenset[str] = frozenset({
    "file_read",
    "file_list",
    "dir_tree",
    "code_grep",
    "code_glob",
    "code_search",
    "find_files",
    "doc_search",
    "doc_fetch",
    "pack_search",
    "shell_read",
    "env_info",
    "container_info",
    "service_list",
    "service_logs",
    "service_probe",
    "http_request",
    "db_inspect",
    "profile_read",
})


# Deterministic substrate tools — agent-callable wrappers around the
# augmentum-dev scanner suite + structured references. Available to
# every read-only role; the lead PREFERS these over LLM grepping
# because they return ground truth in milliseconds instead of model
# best-guesses.
DETERMINISTIC_TOOL_NAMES: frozenset[str] = frozenset({
    "list_routes",
    "find_callers_of_endpoint",
    "red_team_scan",
    "code_quality",
    "security_check",
    "runtime_checks",
    # wiring inspection — middleware chain / decorator chain / static
    # constants / one-hop origin trace. Cover the FP patterns LLM-only
    # audits surface: "trusts unvalidated scope['user']" without
    # accounting for middleware order, "missing auth check" on a
    # handler that actually carries @require_auth, etc.
    "middleware_chain",
    "decorators_on",
    "get_constant",
    "trace_origin",
    # call-graph queries — reverse / forward / reachability over the
    # workspace's function-level CFG.
    "who_calls",
    "callees_of",
    "is_reachable_from",
})


# Pen-test probing tools — DELIBERATELY NOT folded into any existing
# role allow-list. These send real HTTP traffic; only the dedicated
# pen_tester role (landing in Phase 1c) may invoke them. Defining the
# constant now keeps the registry contract visible — anyone adding a
# probing tool must declare it here, not just expose it via
# ``build_pen_test_tools``.
PEN_TEST_TOOL_NAMES: frozenset[str] = frozenset({
    "http_attack",
    "boot_under_test",
    "under_test_status",
    "authz_matrix_probe",
    "concurrent_probe",
})


# Pen-tester role allow-list — read-only (source inspection) + the
# probing toolset + deterministic substrate (for picking targets based
# on routes/wiring). The pen_tester is the ONLY role that gets probing
# capability; all other roles' allow-lists must stay disjoint from
# PEN_TEST_TOOL_NAMES. Use ``test_pen_tester_role_isolation`` to pin.
PEN_TESTER_TOOL_NAMES: frozenset[str] = (
    READ_ONLY_TOOL_NAMES | DETERMINISTIC_TOOL_NAMES | PEN_TEST_TOOL_NAMES
)


PLANNER_TOOL_NAMES: frozenset[str] = READ_ONLY_TOOL_NAMES | DETERMINISTIC_TOOL_NAMES
DETECTOR_TOOL_NAMES: frozenset[str] = READ_ONLY_TOOL_NAMES | DETERMINISTIC_TOOL_NAMES
COMPREHENDER_TOOL_NAMES: frozenset[str] = READ_ONLY_TOOL_NAMES | DETERMINISTIC_TOOL_NAMES
# Investigators follow threads across the codebase — same surface as
# the detector since their job is also "read and reason about code".
INVESTIGATOR_TOOL_NAMES: frozenset[str] = READ_ONLY_TOOL_NAMES | DETERMINISTIC_TOOL_NAMES
# The lead agent has no direct file access — its tools are queue
# manipulation + subagent dispatch (filled in by lead.py at run time).
# Granting it the deterministic substrate lets it call list_routes /
# red_team_scan / etc. directly as part of its decision-making.
LEAD_TOOL_NAMES: frozenset[str] = READ_ONLY_TOOL_NAMES | DETERMINISTIC_TOOL_NAMES


VERIFIER_TOOL_NAMES: frozenset[str] = READ_ONLY_TOOL_NAMES | frozenset({
    "file_write",
    "shell_exec",
    "test_run",
    "browser_open",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_evaluate",
    "browser_verify",
    "browser_wait",
    "browser_extract",
    "browser_fill_form",
    # Sidecar-native verbs (persistent browser session)
    "browser_interact",
    "browser_navigate",
    "browser_get",
    "browser_console",
    "browser_tabs",
    "browser_find",
})


FIXER_TOOL_NAMES: frozenset[str] = READ_ONLY_TOOL_NAMES | frozenset({
    "file_write",
    "code_edit",
    "code_edit_batch",
    "code_multi_edit",
    "apply_patch",
    "shell_exec",
    "test_run",
    "git",
    "observe",
    "profile_update",
})


FULL_TOOL_NAMES: frozenset[str] = FIXER_TOOL_NAMES | VERIFIER_TOOL_NAMES | frozenset({
    "service_start",
    "service_stop",
    "publish_ports",
    "task_list",
})


# Preset name → frozenset. Role files reference these by name
# (``tools: read_only``); custom roles can also enumerate tool names
# explicitly.
TOOL_PRESETS: dict[str, frozenset[str]] = {
    "read_only": READ_ONLY_TOOL_NAMES,
    "verify": VERIFIER_TOOL_NAMES,
    "edit": FIXER_TOOL_NAMES,
    "full": FULL_TOOL_NAMES,
}


def filter_tools(tools: list[Tool], allowed_names: frozenset[str]) -> tuple[Tool, ...]:
    """Return only the tools in ``tools`` whose ``.name`` is in ``allowed_names``."""
    return tuple(t for t in tools if t.name in allowed_names)


def tool_names_for_role(role: str) -> frozenset[str]:
    """Return the allowed tool names for a bug-finder role (legacy)."""
    return {
        "planner": PLANNER_TOOL_NAMES,
        "detector": DETECTOR_TOOL_NAMES,
        "verifier": VERIFIER_TOOL_NAMES,
        "fixer": FIXER_TOOL_NAMES,
        "pen_tester": PEN_TESTER_TOOL_NAMES,
    }.get(role, READ_ONLY_TOOL_NAMES)


def resolve_tool_spec(spec: str | list[str] | frozenset[str]) -> frozenset[str]:
    """Resolve a role file's ``tools:`` declaration into a frozenset.

    Accepts:
      * preset name: ``"read_only"`` → ``READ_ONLY_TOOL_NAMES``
      * preset + extras: ``"read_only + [test_run, http_request]"``
      * explicit list: ``["file_read", "code_grep"]``
      * frozenset (passthrough)
    """
    if isinstance(spec, frozenset):
        return spec
    if isinstance(spec, list):
        return frozenset(str(s) for s in spec if s)
    if not isinstance(spec, str):
        return READ_ONLY_TOOL_NAMES

    text = spec.strip()
    if not text:
        return READ_ONLY_TOOL_NAMES

    if "+" in text:
        preset_name, _, extras = text.partition("+")
        preset_name = preset_name.strip()
        base = TOOL_PRESETS.get(preset_name, frozenset())
        extras_clean = extras.strip().lstrip("[").rstrip("]")
        extra_names = {
            tok.strip().strip("'\"")
            for tok in extras_clean.split(",")
            if tok.strip()
        }
        return base | frozenset(extra_names)

    return TOOL_PRESETS.get(text, frozenset({text}))
