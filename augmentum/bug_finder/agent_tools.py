"""Agent-callable tool wrappers around the deterministic substrate.

The lead, investigator, and detector can call these tools instead of
asking an LLM to grep the workspace. Each tool wraps the matching
``refs`` / ``dev_tools`` / ``intelligence`` / adapter function and
returns plain text the agent loop can consume as-is.

Why a separate module: the substrate functions (``refs.list_routes``,
``dev_tools.red_team_scan``, ``intelligence.who_calls``) work on
Python objects (lists of dataclasses, ``CallGraph`` instances). The
agent layer wants ``Tool`` instances with ``execute(**kwargs) ->
ToolResult``. This module is the thin adapter.

Tools land in the ``READ_ONLY_TOOL_NAMES`` set so the lead, detector,
investigator, and comprehender all have access. The fixer doesn't —
it shouldn't be doing structural reads; it should be patching the
code the verifier already confirmed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from augmentum.bug_finder import (
    dev_tools, intelligence, pen_test, pen_test_attacks, pen_test_boot,
    refs, wiring,
)
from augmentum.bug_finder.adapters import adapter_for_framework
from augmentum.bug_finder.call_graph import CallGraph, build_from_directory
from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# All workspace tools expect the workspace root on disk. The
# orchestrator + agent runtime pass it in at tool-construction time.


def _bound_root(root: Path) -> Path:
    """Coerce + sanity-check the workspace root.

    Returns the absolute path of ``root``. Doesn't validate existence
    — the individual tools handle missing-directory gracefully by
    returning empty results.
    """
    return Path(root).resolve()


# ---------------------------------------------------------------------------
# list_routes — query the cached or live-scanned route inventory
# ---------------------------------------------------------------------------


class ListRoutesTool(Tool):
    """Return all HTTP routes from the workspace.

    Reads ``routes.json`` when augmentum-dev caches are present
    (instant); falls back to the framework adapter's live AST scan
    otherwise. Supports filtering by method / path-substring /
    file-substring so the lead can ask "what auth routes do we
    have?" or "what POST routes touch billing?".
    """

    def __init__(self, root: Path, framework: str = "") -> None:
        self._root = _bound_root(root)
        self._framework = (framework or "").strip().lower() or "fastapi"

    @property
    def name(self) -> str:
        return "list_routes"

    @property
    def description(self) -> str:
        return (
            "Return HTTP routes (method + path + handler file:line) from "
            "the workspace. Use this BEFORE asking the investigator to "
            "grep for routes — this returns ground truth from the "
            "cached or AST-scanned inventory. Filterable by method, "
            "path substring, file substring."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": (
                        "Optional HTTP method filter ('GET' / 'POST' / etc). "
                        "Empty = any."
                    ),
                    "default": "",
                },
                "path_substr": {
                    "type": "string",
                    "description": (
                        "Optional case-insensitive substring of the route "
                        "path. Example: '/auth/' surfaces all auth-shaped "
                        "routes."
                    ),
                    "default": "",
                },
                "file_substr": {
                    "type": "string",
                    "description": (
                        "Optional case-insensitive substring of the source "
                        "file path. Example: 'proxy/' narrows to proxy routes."
                    ),
                    "default": "",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 50,
                    "description": (
                        "Max routes returned. Default 50."
                    ),
                },
            },
        }

    async def execute(
        self,
        *,
        method: str = "",
        path_substr: str = "",
        file_substr: str = "",
        limit: int = 50,
        **_kwargs,
    ) -> ToolResult:
        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            limit = 50

        # Prefer cached refs when the workspace ships augmentum-dev
        # (instant). Otherwise live-scan via the framework adapter
        # (seconds on a large repo).
        if refs.has_augmentum_dev_refs(self._root):
            rows = refs.list_routes(
                self._root,
                method=method, path_substr=path_substr,
                file_substr=file_substr, limit=limit,
            )
            data = [
                {
                    "method": r.method, "path": r.path,
                    "handler": r.handler,
                    "file": r.file, "line": r.line,
                }
                for r in rows
            ]
        else:
            adapter = adapter_for_framework(self._framework)
            raw = adapter.list_routes(self._root)
            method_q = method.upper().strip()
            path_q = path_substr.lower().strip()
            file_q = file_substr.lower().strip()
            data = []
            for r in raw:
                if method_q and r.method != method_q:
                    continue
                if path_q and path_q not in r.path.lower():
                    continue
                if file_q and file_q not in r.file.lower():
                    continue
                data.append({
                    "method": r.method, "path": r.path,
                    "handler": r.handler,
                    "file": r.file, "line": r.line,
                })
                if len(data) >= limit:
                    break

        body = json.dumps(data, indent=2)
        return ToolResult(
            success=True,
            output=body,
            metadata={
                "route_count": len(data),
                "filters": {
                    "method": method, "path_substr": path_substr,
                    "file_substr": file_substr,
                },
            },
        )


# ---------------------------------------------------------------------------
# find_callers_of — frontend → endpoint cross-reference
# ---------------------------------------------------------------------------


class FindCallersOfEndpointTool(Tool):
    """Return frontend (UI) callers of a given HTTP endpoint.

    Useful for the lead to ask: "this route exists, but does anyone
    actually call it from the frontend?" A route with no caller is a
    candidate for dead-code investigation or server-to-server-only
    classification.
    """

    def __init__(self, root: Path) -> None:
        self._root = _bound_root(root)

    @property
    def name(self) -> str:
        return "find_callers_of_endpoint"

    @property
    def description(self) -> str:
        return (
            "Return frontend (JS) fetch / WebSocket calls that hit a "
            "given endpoint path (substring match). Empty result = "
            "the endpoint has no frontend caller (orphan or "
            "server-to-server)."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path_substr": {
                    "type": "string",
                    "description": (
                        "Substring of the endpoint path to search for in "
                        "frontend calls. Example: '/api/auth/login'."
                    ),
                },
                "method": {
                    "type": "string",
                    "description": "Optional HTTP method filter.",
                    "default": "",
                },
            },
            "required": ["path_substr"],
        }

    async def execute(
        self,
        *,
        path_substr: str = "",
        method: str = "",
        **_kwargs,
    ) -> ToolResult:
        if not path_substr.strip():
            return ToolResult(
                success=False,
                error="path_substr is required",
                validation_error=True,
            )
        rows = refs.find_callers_of(
            self._root, path_substr=path_substr, method=method,
        )
        data = [
            {"url": r.url, "method": r.method, "file": r.file}
            for r in rows
        ]
        return ToolResult(
            success=True,
            output=json.dumps(data, indent=2),
            metadata={"caller_count": len(data)},
        )


# ---------------------------------------------------------------------------
# red_team_scan — adversarial deterministic scan
# ---------------------------------------------------------------------------


class RedTeamScanTool(Tool):
    """Run the augmentum-dev adversarial scanner over the workspace.

    Surfaces data-isolation gaps, auth bypass patterns, token
    exposure, IDOR, AI context leaks. Returns structured findings
    the lead can dispatch verifiers against. Suppression-filtered by
    default. Cached for 5 minutes per workspace.
    """

    def __init__(self, root: Path) -> None:
        self._root = _bound_root(root)

    @property
    def name(self) -> str:
        return "red_team_scan"

    @property
    def description(self) -> str:
        return (
            "Run the deterministic adversarial scanner. Returns "
            "structured findings: data-isolation gaps, auth bypass, "
            "token exposure, IDOR, AI context leaks. "
            "Use this BEFORE asking the investigator to grep for "
            "security patterns — the scanner has been tuned over "
            "months and applies known-intentional suppressions. "
            "Empty result = workspace has no findings in these "
            "classes; that's a real signal, not a no-op."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "apply_suppressions": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "Filter known-intentional findings (default True). "
                        "Set False to see EVERY raw scanner hit including "
                        "ones we've decided not to flag."
                    ),
                },
                "severity_floor": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "Optional severity floor: info / low / medium / "
                        "high / critical. Findings below this level are "
                        "filtered out."
                    ),
                },
            },
        }

    async def execute(
        self,
        *,
        apply_suppressions: bool = True,
        severity_floor: str = "",
        **_kwargs,
    ) -> ToolResult:
        rows = dev_tools.run_scanner(
            "red_team_scan", root=self._root,
            apply_suppressions=bool(apply_suppressions),
        )
        sev_rank = {
            "critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4,
        }
        floor_norm = (severity_floor or "").strip().lower()
        floor_idx = sev_rank.get(floor_norm)
        if floor_idx is not None:
            rows = [
                r for r in rows
                if sev_rank.get(r.severity, 5) <= floor_idx
            ]
        data = [
            {
                "rule_id": r.rule_id,
                "severity": r.severity,
                "category": r.category,
                "file": r.file, "line": r.line,
                "message": r.message,
                "fix": r.fix,
            }
            for r in rows
        ]
        return ToolResult(
            success=True,
            output=json.dumps(data, indent=2),
            metadata={"finding_count": len(data)},
        )


# ---------------------------------------------------------------------------
# Generic scanner tool — one class, parameterized by scanner slug
# ---------------------------------------------------------------------------


class _ScannerTool(Tool):
    """Generic wrapper around any ``dev_tools.run_scanner`` slug.

    All scanner tools share the same shape (severity floor, suppression
    toggle); only the slug + human-facing description differ. One
    class with three instances is cleaner than three near-identical
    classes.
    """

    _SCANNER_DESCRIPTIONS = {
        "code_quality": (
            "Run the deterministic code-quality scanner. Surfaces silent "
            "catch blocks, console.log in production, mixed error "
            "response formats, websocket contract gaps, tech-debt "
            "markers. Returns structured findings — call BEFORE asking "
            "the investigator to grep for these patterns."
        ),
        "security_check": (
            "Run the deterministic security scanner. Surfaces SSRF, SQL "
            "injection, template XSS, key exposure, and stale exception "
            "rules. Returns structured findings with file + line + "
            "description. The lead should dispatch detector verification "
            "against each finding rather than re-grepping."
        ),
        "runtime_checks": (
            "Run the deterministic runtime-pattern scanner. Surfaces "
            "empty model strings, silent exception swallowing, "
            "unhandled fetch failures, state outside handlers. "
            "Catches runtime bugs the static type system misses."
        ),
    }

    def __init__(self, root: Path, slug: str) -> None:
        self._root = _bound_root(root)
        self._slug = slug

    @property
    def name(self) -> str:
        return self._slug

    @property
    def description(self) -> str:
        return self._SCANNER_DESCRIPTIONS.get(
            self._slug,
            f"Run the {self._slug} deterministic scanner.",
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "apply_suppressions": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "Filter known-intentional findings (default True)."
                    ),
                },
                "severity_floor": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "Optional severity floor: info / low / medium / "
                        "high / critical."
                    ),
                },
            },
        }

    async def execute(
        self,
        *,
        apply_suppressions: bool = True,
        severity_floor: str = "",
        **_kwargs,
    ) -> ToolResult:
        rows = dev_tools.run_scanner(
            self._slug, root=self._root,
            apply_suppressions=bool(apply_suppressions),
        )
        sev_rank = {
            "critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4,
        }
        floor_norm = (severity_floor or "").strip().lower()
        floor_idx = sev_rank.get(floor_norm)
        if floor_idx is not None:
            rows = [
                r for r in rows
                if sev_rank.get(r.severity, 5) <= floor_idx
            ]
        data = [
            {
                "rule_id": r.rule_id,
                "severity": r.severity,
                "category": r.category,
                "file": r.file, "line": r.line,
                "message": r.message,
                "fix": r.fix,
            }
            for r in rows
        ]
        return ToolResult(
            success=True,
            output=json.dumps(data, indent=2),
            metadata={
                "scanner": self._slug,
                "finding_count": len(data),
            },
        )


# ---------------------------------------------------------------------------
# Wiring tools — middleware chain, decorators, constants, origin trace
# ---------------------------------------------------------------------------


class MiddlewareChainTool(Tool):
    """Return the workspace's ASGI middleware registration chain.

    Use BEFORE assuming a value (``scope['user']``, ``request.state.x``)
    is unvalidated input — the middleware chain tells you what runs
    before any route handler. ASGI semantics: the LAST registered
    middleware wraps the OUTERMOST so it runs FIRST on the request.
    """

    def __init__(self, root: Path) -> None:
        self._root = _bound_root(root)

    @property
    def name(self) -> str:
        return "middleware_chain"

    @property
    def description(self) -> str:
        return (
            "List every ASGI middleware (``app.add_middleware(...)``) "
            "registered in the workspace, in registration order, with "
            "request-flow ordering also returned. Use this BEFORE "
            "flagging a scope['key'] / request.state.x value as "
            "unvalidated — the chain tells you whether a gatekeeping "
            "middleware ran first."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **_kwargs) -> ToolResult:
        entries = wiring.extract_middleware_chain(self._root)
        flow = wiring.runs_first_to_last(entries)
        data = {
            "registration_order": [
                {
                    "name": e.name, "order": e.order,
                    "file": e.file, "line": e.line,
                    "kwargs": e.kwargs_repr,
                }
                for e in entries
            ],
            "request_flow_order": [e.name for e in flow],
            "note": (
                "request_flow_order shows the order an incoming request "
                "actually traverses the chain — the LAST registered "
                "middleware runs FIRST."
            ),
        }
        return ToolResult(
            success=True,
            output=json.dumps(data, indent=2),
            metadata={"middleware_count": len(entries)},
        )


class DecoratorsOnTool(Tool):
    """Return the decorator chain on a function defined near ``file:line``."""

    def __init__(self, root: Path) -> None:
        self._root = _bound_root(root)

    @property
    def name(self) -> str:
        return "decorators_on"

    @property
    def description(self) -> str:
        return (
            "Return the decorator chain on the function whose source "
            "range contains the given file:line. Use this BEFORE "
            "flagging a route handler as missing auth — the @require_auth "
            "decorator is often invisible from a single chunk read."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Workspace-relative path to the source file.",
                },
                "line": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Any line within the function (including a "
                        "decorator line). The tool resolves to the "
                        "function whose source range contains this line."
                    ),
                },
            },
            "required": ["file", "line"],
        }

    async def execute(
        self, *, file: str = "", line: int = 0, **_kwargs,
    ) -> ToolResult:
        if not file or not line:
            return ToolResult(
                success=False,
                error="file and line are required",
                validation_error=True,
            )
        decos = wiring.decorators_on(self._root, file=file, line=int(line))
        data = [
            {
                "name": d.name, "line": d.line,
                "args": d.args_repr, "kwargs": d.kwargs_repr,
            }
            for d in decos
        ]
        return ToolResult(
            success=True,
            output=json.dumps(data, indent=2),
            metadata={"decorator_count": len(data)},
        )


class GetConstantTool(Tool):
    """Resolve a module-level constant by name."""

    def __init__(self, root: Path) -> None:
        self._root = _bound_root(root)

    @property
    def name(self) -> str:
        return "get_constant"

    @property
    def description(self) -> str:
        return (
            "Resolve a module-level ``name = <literal>`` binding to its "
            "static value. Returns confident=True for pure literals; "
            "confident=False when the value depends on runtime/env. "
            "Use this BEFORE flagging a feature-flag-gated path as "
            "unreachable — the flag's default may be the opposite of "
            "what the local code suggests."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The bare constant name to resolve.",
                },
            },
            "required": ["name"],
        }

    async def execute(self, *, name: str = "", **_kwargs) -> ToolResult:
        if not name.strip():
            return ToolResult(
                success=False, error="name is required",
                validation_error=True,
            )
        binding = wiring.get_constant(self._root, name.strip())
        if binding is None:
            return ToolResult(
                success=True,
                output=json.dumps({"found": False, "name": name}),
                metadata={"found": False},
            )
        data = {
            "found": True,
            "name": binding.name,
            "value_repr": binding.value_repr,
            "file": binding.file,
            "line": binding.line,
            "confident": binding.confident,
        }
        return ToolResult(
            success=True,
            output=json.dumps(data, indent=2),
            metadata={"found": True, "confident": binding.confident},
        )


class TraceOriginTool(Tool):
    """One-hop origin trace for a variable at a specific use site."""

    def __init__(self, root: Path) -> None:
        self._root = _bound_root(root)

    @property
    def name(self) -> str:
        return "trace_origin"

    @property
    def description(self) -> str:
        return (
            "Return where ``var`` was last bound before ``line`` in "
            "``file`` — parameter, latest assignment, for/with target, "
            "or import. ONE-HOP only: we report the immediate binding, "
            "not the transitive source. Use this BEFORE asking 'where "
            "does this value come from?'"
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Workspace-relative source file.",
                },
                "line": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "The use-site line number.",
                },
                "var": {
                    "type": "string",
                    "description": (
                        "The variable name to trace. Bare identifier — "
                        "no subscript / attribute chains."
                    ),
                },
            },
            "required": ["file", "line", "var"],
        }

    async def execute(
        self, *, file: str = "", line: int = 0, var: str = "", **_kwargs,
    ) -> ToolResult:
        if not file or not line or not var:
            return ToolResult(
                success=False, error="file, line, and var are required",
                validation_error=True,
            )
        trace = wiring.trace_origin(
            self._root, file=file, line=int(line), var=var.strip(),
        )
        if trace is None:
            return ToolResult(
                success=True,
                output=json.dumps({"found": False, "var": var, "note": (
                    "No static binding found — value may be closure-"
                    "captured, dynamically injected, or framework-supplied."
                )}),
                metadata={"found": False},
            )
        data = {
            "found": True,
            "var": trace.var,
            "use": {"file": trace.file, "line": trace.line},
            "origin": {
                "kind": trace.origin_kind,
                "file": trace.origin_file,
                "line": trace.origin_line,
                "expr": trace.origin_expr,
                "confident": trace.confident,
                "note": trace.note,
            },
        }
        return ToolResult(
            success=True,
            output=json.dumps(data, indent=2),
            metadata={"found": True, "kind": trace.origin_kind},
        )


# ---------------------------------------------------------------------------
# Call-graph tools — who_calls / callees / reachability
# ---------------------------------------------------------------------------


class _CallGraphHolder:
    """Lazy + shared CallGraph for one workspace.

    Building the graph touches every .py file in the repo. Sharing one
    instance across all call-graph tools means one parse per run instead
    of one per tool invocation.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._graph: CallGraph | None = None

    def get(self) -> CallGraph:
        if self._graph is None:
            self._graph = build_from_directory(self._root)
        return self._graph


class WhoCallsTool(Tool):
    """Bare-name reverse call lookup over the workspace."""

    def __init__(self, root: Path, holder: _CallGraphHolder) -> None:
        self._root = _bound_root(root)
        self._holder = holder

    @property
    def name(self) -> str:
        return "who_calls"

    @property
    def description(self) -> str:
        return (
            "Return qualified callers of a bare function name across the "
            "workspace. Bare-name matching — ``eval`` returns every "
            "``eval(...)`` site regardless of which ``eval`` it is. The "
            "caller weighs which match is real. Use BEFORE concluding a "
            "risky function is unreachable."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Bare function/method name to look up.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1, "default": 20,
                    "description": "Max callers returned.",
                },
            },
            "required": ["target"],
        }

    async def execute(
        self, *, target: str = "", limit: int = 20, **_kwargs,
    ) -> ToolResult:
        if not target.strip():
            return ToolResult(
                success=False, error="target is required",
                validation_error=True,
            )
        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            limit = 20
        graph = self._holder.get()
        callers = intelligence.who_calls(graph, target.strip(), limit=limit)
        data = [
            {"caller": c.caller, "file": c.file, "line": c.line}
            for c in callers
        ]
        return ToolResult(
            success=True,
            output=json.dumps(data, indent=2),
            metadata={"caller_count": len(data)},
        )


class CalleesOfTool(Tool):
    """Forward call lookup — what does this caller invoke?"""

    def __init__(self, root: Path, holder: _CallGraphHolder) -> None:
        self._root = _bound_root(root)
        self._holder = holder

    @property
    def name(self) -> str:
        return "callees_of"

    @property
    def description(self) -> str:
        return (
            "Return bare names called from a fully-qualified caller "
            "(``module.Class.method``). Use to learn what a handler "
            "actually reaches — particularly useful when verifying a "
            "claim that 'this function never touches the database'."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "qualified_caller": {
                    "type": "string",
                    "description": (
                        "Qualified caller name: ``module.Class.method`` "
                        "or ``module.function``."
                    ),
                },
            },
            "required": ["qualified_caller"],
        }

    async def execute(
        self, *, qualified_caller: str = "", **_kwargs,
    ) -> ToolResult:
        if not qualified_caller.strip():
            return ToolResult(
                success=False, error="qualified_caller is required",
                validation_error=True,
            )
        graph = self._holder.get()
        callees = intelligence.callees_of(graph, qualified_caller.strip())
        return ToolResult(
            success=True,
            output=json.dumps(callees, indent=2),
            metadata={"callee_count": len(callees)},
        )


class IsReachableTool(Tool):
    """BFS reachability between two qualified names in the call graph."""

    def __init__(self, root: Path, holder: _CallGraphHolder) -> None:
        self._root = _bound_root(root)
        self._holder = holder

    @property
    def name(self) -> str:
        return "is_reachable_from"

    @property
    def description(self) -> str:
        return (
            "Is ``sink`` reachable from ``source`` within ``max_depth`` "
            "call hops? Bare-name targets; qualified source. Use to ask "
            "'can this risky sink be hit from a public route?' — "
            "promotes severity when True, demotes when False."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Qualified source name (e.g. ``module.handler``).",
                },
                "sink": {
                    "type": "string",
                    "description": "Bare name of the sink (e.g. ``eval``, ``subprocess.Popen``).",
                },
                "max_depth": {
                    "type": "integer", "minimum": 1, "default": 4,
                    "description": "Hop limit; default 4.",
                },
            },
            "required": ["source", "sink"],
        }

    async def execute(
        self, *, source: str = "", sink: str = "",
        max_depth: int = 4, **_kwargs,
    ) -> ToolResult:
        if not source.strip() or not sink.strip():
            return ToolResult(
                success=False, error="source and sink are required",
                validation_error=True,
            )
        try:
            max_depth = max(1, int(max_depth))
        except (TypeError, ValueError):
            max_depth = 4
        graph = self._holder.get()
        reachable = intelligence.is_reachable_from(
            graph, source=source.strip(), sink=sink.strip(), max_depth=max_depth,
        )
        return ToolResult(
            success=True,
            output=json.dumps({
                "source": source, "sink": sink,
                "max_depth": max_depth, "reachable": reachable,
            }),
            metadata={"reachable": reachable},
        )


# ---------------------------------------------------------------------------
# Tool name constants — exported for the agent tool-set declarations
# ---------------------------------------------------------------------------


DETERMINISTIC_TOOL_NAMES: frozenset[str] = frozenset({
    "list_routes",
    "find_callers_of_endpoint",
    "red_team_scan",
    "code_quality",
    "security_check",
    "runtime_checks",
    "middleware_chain",
    "decorators_on",
    "get_constant",
    "trace_origin",
    "who_calls",
    "callees_of",
    "is_reachable_from",
})


def build_deterministic_tools(
    root: Path, framework: str = "fastapi",
) -> tuple[Tool, ...]:
    """Return the deterministic tool set bound to one workspace.

    The orchestrator calls this once per run, mixes the result into
    the existing READ_ONLY tools, and passes the combined tuple to
    each subagent spec. Tools are stateless w.r.t. requests so a
    single instance can be reused across all subagents in one run.
    """
    call_graph_holder = _CallGraphHolder(_bound_root(root))
    return (
        ListRoutesTool(root=root, framework=framework),
        FindCallersOfEndpointTool(root=root),
        RedTeamScanTool(root=root),
        _ScannerTool(root=root, slug="code_quality"),
        _ScannerTool(root=root, slug="security_check"),
        _ScannerTool(root=root, slug="runtime_checks"),
        MiddlewareChainTool(root=root),
        DecoratorsOnTool(root=root),
        GetConstantTool(root=root),
        TraceOriginTool(root=root),
        WhoCallsTool(root=root, holder=call_graph_holder),
        CalleesOfTool(root=root, holder=call_graph_holder),
        IsReachableTool(root=root, holder=call_graph_holder),
    )


# ---------------------------------------------------------------------------
# Pen-test probing tools — DELIBERATELY SEPARATE from deterministic
# ---------------------------------------------------------------------------
#
# These send real HTTP traffic; they MUST NOT be added to the read-only
# role allow-lists by accident. ``build_pen_test_tools`` is called
# only by the pen_tester subagent role (landing in Phase 1c). Until
# then the tool exists in the codebase but is dormant — no role can
# instantiate it through the standard tool-build path.
# ---------------------------------------------------------------------------


class HTTPAttackTool(Tool):
    """Send one HTTP request to a probe target.

    Phase 1a primitive. The model fully specifies the request —
    method, URL, headers, body — and gets back the response status,
    headers, body excerpt, and latency. Use this to:

    * Confirm a SQL-injection hypothesis by sending the payload and
      observing the error/response.
    * Test authz claims by sending the same request under a different
      session token.
    * Verify input validation by sending malformed payloads and
      reading the response.

    Host policy: by default only loopback, the Docker bridge
    (172.16/12), ``host.docker.internal``, and single-label DNS
    names (compose-internal) are permitted. External hosts require
    ``allow_external=true`` — pass it explicitly when probing a
    known staging environment you own.
    """

    def __init__(
        self, root: Path | None = None,
        *,
        workspace_root_for_receipts: Path | None = None,
    ) -> None:
        # ``root`` kept for parity with other workspace-scoped tools;
        # we just use it as the receipts dir unless a separate path
        # is supplied.
        self._receipts_root = (
            _bound_root(workspace_root_for_receipts)
            if workspace_root_for_receipts
            else (_bound_root(root) if root else None)
        )

    @property
    def name(self) -> str:
        return "http_attack"

    @property
    def description(self) -> str:
        return (
            "Send one HTTP request to a probe target and return the "
            "response (status, headers, body excerpt, latency). Use to "
            "actively verify findings: confirm SQLi with a payload, "
            "test authz by swapping tokens, exercise input validation. "
            "Default host policy permits ONLY loopback / Docker bridge "
            "/ single-label DNS. External hosts require allow_external=true."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": (
                        "HTTP method. One of GET, POST, PUT, PATCH, "
                        "DELETE, HEAD, OPTIONS."
                    ),
                },
                "url": {
                    "type": "string",
                    "description": (
                        "Target URL. Must use http:// or https:// scheme."
                    ),
                },
                "headers": {
                    "type": "object",
                    "description": (
                        "Request headers. Sensitive header values "
                        "(Authorization, Cookie, X-API-Key) are "
                        "redacted in the receipts trail but sent "
                        "verbatim to the target."
                    ),
                    "additionalProperties": {"type": "string"},
                    "default": {},
                },
                "body": {
                    "type": "string",
                    "description": (
                        "Request body. For JSON, include the "
                        "Content-Type header explicitly."
                    ),
                    "default": "",
                },
                "timeout_s": {
                    "type": "number",
                    "minimum": 0.5, "maximum": 120,
                    "default": 30,
                    "description": "Per-request timeout in seconds.",
                },
                "follow_redirects": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Follow HTTP redirects. Off by default so a "
                        "302 to a different host doesn't silently "
                        "bypass the host allow-list."
                    ),
                },
                "allow_external": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Opt-in to probe a host outside the default "
                        "loopback / docker-bridge / single-label "
                        "allow-list. Set ONLY when the caller has "
                        "authorization to probe the external target."
                    ),
                },
                "finding_id": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "Optional — link this probe to a finding in "
                        "the receipts trail."
                    ),
                },
                "run_id": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "Optional — link this probe to a bug_finder run."
                    ),
                },
                "note": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "Optional free-form annotation captured in "
                        "the receipt."
                    ),
                },
            },
            "required": ["method", "url"],
        }

    async def execute(
        self,
        *,
        method: str = "",
        url: str = "",
        headers: dict | None = None,
        body: str = "",
        timeout_s: float = pen_test.DEFAULT_TIMEOUT_S,
        follow_redirects: bool = False,
        allow_external: bool = False,
        finding_id: str = "",
        run_id: str = "",
        note: str = "",
        **_kwargs,
    ) -> ToolResult:
        if not method or not url:
            return ToolResult(
                success=False,
                error="method and url are required",
                validation_error=True,
            )
        # Defensive: the model sometimes passes nulls
        if headers is None:
            headers = {}
        elif not isinstance(headers, dict):
            return ToolResult(
                success=False,
                error="headers must be an object (dict)",
                validation_error=True,
            )
        try:
            timeout_s = float(timeout_s)
        except (TypeError, ValueError):
            timeout_s = pen_test.DEFAULT_TIMEOUT_S
        req = pen_test.ProbeRequest(
            method=method,
            url=url,
            headers={str(k): str(v) for k, v in headers.items()},
            body=body or "",
            timeout_s=timeout_s,
            follow_redirects=bool(follow_redirects),
            allow_external=bool(allow_external),
            finding_id=finding_id or "",
            run_id=run_id or "",
            note=note or "",
        )
        resp, receipt = await pen_test.execute_probe(
            req, workspace_root=self._receipts_root,
        )
        # Return shape mirrors the other deterministic tools — a JSON
        # blob the LLM can parse. We surface the receipt fields the
        # LLM needs to make a verdict: status, headers, body excerpt,
        # latency, plus the policy bucket so it knows whether the
        # probe was admitted.
        data = {
            "ok": resp.ok,
            "status": resp.status,
            "headers": resp.headers,
            "body_excerpt": resp.body_excerpt,
            "body_size": resp.body_size,
            "body_truncated": resp.body_truncated,
            "latency_ms": resp.latency_ms,
            "final_url": resp.final_url,
            "host_policy": receipt.host_policy,
        }
        if resp.error:
            data["error"] = resp.error
        return ToolResult(
            # ``success`` here reports whether the *tool call* worked,
            # not whether the probe succeeded. A 500 response is still
            # a successful tool call.
            success=resp.ok or receipt.host_policy == "refused",
            output=json.dumps(data, indent=2),
            error=resp.error if not resp.ok else "",
            metadata={
                "status": resp.status,
                "host_policy": receipt.host_policy,
                "latency_ms": resp.latency_ms,
            },
        )


class BootUnderTestTool(Tool):
    """Boot the workspace's application as a subprocess so the
    pen_tester can target it with HTTP probes.

    Phase 1b primitive. The caller supplies an explicit ``BootSpec``
    (command, port, healthcheck path) — the tool spawns the process,
    polls the healthcheck until it goes green, and returns the base
    URL the pen_tester should use with ``http_attack``.

    Teardown is NOT exposed as a tool — the orchestrator's
    ``_UnderTestRegistry.teardown_all`` handles lifecycle. This
    prevents an LLM from leaking processes or denying-of-service the
    pipeline through tool misuse.

    The booter writes stdout/stderr to a log file under
    ``.augmentum/bug_finder/under_test_logs/``. If the healthcheck
    times out, the failure response includes the tail of those logs
    so the LLM can reason about why the app didn't come up.
    """

    def __init__(
        self,
        workspace_root: Path,
        *,
        registry: pen_test_boot._UnderTestRegistry | None = None,
    ) -> None:
        self._workspace_root = _bound_root(workspace_root)
        # When no registry is supplied, fall back to the default. The
        # orchestrator passes a per-run registry so teardown is scoped.
        self._registry = registry or pen_test_boot.default_registry()

    @property
    def name(self) -> str:
        return "boot_under_test"

    @property
    def description(self) -> str:
        return (
            "Boot the workspace's app as a subprocess inside the "
            "container, wait for its healthcheck to come green, and "
            "return the base URL the pen-tester should target with "
            "http_attack. The caller MUST supply an explicit command "
            "+ port + healthcheck path; this tool does not auto-detect. "
            "Teardown is handled by the orchestrator — no manual stop "
            "tool is exposed."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "argv-style command to invoke the app. "
                        "Example: ['uvicorn', 'app:app', '--host', "
                        "'0.0.0.0', '--port', '8080']."
                    ),
                },
                "port": {
                    "type": "integer",
                    "minimum": 1, "maximum": 65535,
                    "description": "Port the app will listen on.",
                },
                "healthcheck_path": {
                    "type": "string",
                    "default": "/",
                    "description": (
                        "URL path to GET for the readiness probe. "
                        "The default ``/`` works for most apps; "
                        "/healthz, /readyz, /api/healthz are common."
                    ),
                },
                "healthcheck_timeout_s": {
                    "type": "number",
                    "minimum": 1, "maximum": 300,
                    "default": 30,
                    "description": "Seconds to wait for healthcheck.",
                },
                "boot_timeout_s": {
                    "type": "number",
                    "minimum": 1, "maximum": 300,
                    "default": 60,
                    "description": "Overall boot timeout.",
                },
                "cwd": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "Working directory relative to workspace root "
                        "(or absolute). Empty = workspace root."
                    ),
                },
                "env_overrides": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "default": {},
                    "description": (
                        "Environment variables to set on top of "
                        "inherited env."
                    ),
                },
            },
            "required": ["command", "port"],
        }

    async def execute(
        self,
        *,
        command: list[str] | None = None,
        port: int = 0,
        healthcheck_path: str = "/",
        healthcheck_timeout_s: float = pen_test_boot.DEFAULT_HEALTHCHECK_TIMEOUT_S,
        boot_timeout_s: float = pen_test_boot.DEFAULT_BOOT_TIMEOUT_S,
        cwd: str = "",
        env_overrides: dict | None = None,
        **_kwargs,
    ) -> ToolResult:
        if not command or not isinstance(command, list):
            return ToolResult(
                success=False,
                error="command (non-empty list of strings) is required",
                validation_error=True,
            )
        try:
            port_i = int(port)
        except (TypeError, ValueError):
            port_i = 0
        if port_i <= 0 or port_i > 65535:
            return ToolResult(
                success=False,
                error="port must be in 1..65535",
                validation_error=True,
            )
        spec = pen_test_boot.BootSpec(
            command=tuple(str(c) for c in command),
            port=port_i,
            healthcheck_path=healthcheck_path or "/",
            healthcheck_timeout_s=float(healthcheck_timeout_s),
            boot_timeout_s=float(boot_timeout_s),
            cwd=cwd or None,
            env_overrides={
                str(k): str(v)
                for k, v in (env_overrides or {}).items()
            },
        )
        result = await pen_test_boot.boot_under_test(
            self._workspace_root, spec, registry=self._registry,
        )
        if result.ok:
            svc = result.service
            data = {
                "ok": True,
                "service_id": svc.service_id,
                "base_url": svc.base_url,
                "pid": svc.pid,
                "started_at": svc.started_at,
                "healthcheck_url": (
                    svc.base_url.rstrip("/")
                    + "/" + spec.healthcheck_path.lstrip("/")
                ),
            }
            return ToolResult(
                success=True,
                output=json.dumps(data, indent=2),
                metadata={
                    "service_id": svc.service_id,
                    "base_url": svc.base_url,
                },
            )
        f = result.failure
        data = {
            "ok": False,
            "reason": f.reason,
            "detail": f.detail,
            "elapsed_ms": f.elapsed_ms,
            "exit_code": f.exit_code,
            "log_tail": f.log_tail[-4000:],   # cap for prompt budget
        }
        return ToolResult(
            success=False,
            output=json.dumps(data, indent=2),
            error=f"boot failed: {f.reason}: {f.detail}"[:300],
            metadata={"reason": f.reason},
        )


class UnderTestStatusTool(Tool):
    """Check whether an under-test service booted earlier is still
    healthy. Useful when the pen_tester is partway through a probe
    sequence and wants to verify the app hasn't crashed."""

    def __init__(
        self,
        registry: pen_test_boot._UnderTestRegistry | None = None,
    ) -> None:
        self._registry = registry or pen_test_boot.default_registry()

    @property
    def name(self) -> str:
        return "under_test_status"

    @property
    def description(self) -> str:
        return (
            "Report whether a previously-booted under-test service is "
            "still running. Returns ok, pid, healthy flag, and the "
            "tail of its log when stopped."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "service_id": {
                    "type": "string",
                    "description": (
                        "service_id from a prior boot_under_test call."
                    ),
                },
            },
            "required": ["service_id"],
        }

    async def execute(
        self, *, service_id: str = "", **_kwargs,
    ) -> ToolResult:
        if not service_id.strip():
            return ToolResult(
                success=False,
                error="service_id is required",
                validation_error=True,
            )
        svc = self._registry.get(service_id.strip())
        if svc is None:
            return ToolResult(
                success=True,
                output=json.dumps({
                    "ok": False,
                    "known": False,
                    "note": "no service with this id in the current run",
                }),
                metadata={"known": False},
            )
        proc = svc.process
        exit_code = (
            proc.returncode if proc is not None else None
        )
        data = {
            "ok": True,
            "known": True,
            "service_id": svc.service_id,
            "base_url": svc.base_url,
            "pid": svc.pid,
            "healthy": svc.healthy,
            "teardown_called": svc.teardown_called,
            "exit_code": exit_code,
        }
        return ToolResult(
            success=True,
            output=json.dumps(data, indent=2),
            metadata={"healthy": svc.healthy},
        )


class AuthzMatrixProbeTool(Tool):
    """Systematically probe cross-tenant access against one endpoint.

    Phase 2 attack primitive — the canonical "hard to detect
    statically, easy to verify dynamically" check. Given two or
    more user tokens and a victim's resource id, sends GET requests
    using each attacker's token against each victim's resource and
    reports which probes returned 2xx (a leak).

    Use after ``boot_under_test`` returns a booted service, when
    you have a finding hypothesis like "this endpoint reads a row
    by id without tenant filtering". The leak_indicators dict (when
    you can predict what victim-owned response data looks like) lets
    you upgrade ambiguous 2xx responses to confirmed leaks.
    """

    def __init__(
        self,
        *,
        workspace_root_for_receipts: Path | None = None,
    ) -> None:
        self._receipts_root = (
            _bound_root(workspace_root_for_receipts)
            if workspace_root_for_receipts else None
        )

    @property
    def name(self) -> str:
        return "authz_matrix_probe"

    @property
    def description(self) -> str:
        return (
            "Systematically test cross-tenant access against an "
            "endpoint pattern containing {victim_id}. Sweeps every "
            "(attacker_token, victim_user, victim_resource) tuple "
            "and reports which probes returned 2xx (the leak). "
            "Optional leak_indicators dict upgrades 2xx-with-payload "
            "to confirmed leak. Required for multi-tenant audits — "
            "this class of bug is invisible to chunk-level static "
            "review."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "base_url": {
                    "type": "string",
                    "description": (
                        "Booted target base URL (from boot_under_test)."
                    ),
                },
                "endpoint_pattern": {
                    "type": "string",
                    "description": (
                        "Path template containing '{victim_id}', e.g. "
                        "'/api/notes/{victim_id}'. Required."
                    ),
                },
                "attacker_tokens": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Auth token strings. Each will be tried against "
                        "each victim. Use tokens that match victim "
                        "user_ids so same-tenant probes get skipped."
                    ),
                },
                "victims": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2, "maxItems": 2,
                    },
                    "description": (
                        "List of [user_id, resource_id] pairs. The "
                        "user_id is used to skip same-tenant probes; "
                        "the resource_id substitutes into "
                        "endpoint_pattern."
                    ),
                },
                "auth_header_name": {
                    "type": "string", "default": "Authorization",
                    "description": (
                        "Header to carry the auth value. Default "
                        "'Authorization'."
                    ),
                },
                "auth_header_format": {
                    "type": "string", "default": "Bearer {token}",
                    "description": (
                        "Format string for the header value. Use "
                        "{token} for substitution."
                    ),
                },
                "leak_indicators": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": (
                        "Optional {user_id -> indicator-string} map. "
                        "When a 2xx response body contains the victim's "
                        "indicator, the leak is upgraded to confirmed. "
                        "Useful when you know what victim-data looks "
                        "like (their known title, their user_id literal)."
                    ),
                    "default": {},
                },
                "finding_id": {
                    "type": "string", "default": "",
                    "description": "Audit-trail link to a finding.",
                },
                "run_id": {
                    "type": "string", "default": "",
                    "description": "Audit-trail link to a bug_finder run.",
                },
            },
            "required": ["base_url", "endpoint_pattern",
                         "attacker_tokens", "victims"],
        }

    async def execute(
        self,
        *,
        base_url: str = "",
        endpoint_pattern: str = "",
        attacker_tokens: list[str] | None = None,
        victims: list[list[str]] | None = None,
        auth_header_name: str = "Authorization",
        auth_header_format: str = "Bearer {token}",
        leak_indicators: dict | None = None,
        finding_id: str = "",
        run_id: str = "",
        **_kwargs,
    ) -> ToolResult:
        if not base_url or not endpoint_pattern:
            return ToolResult(
                success=False,
                error="base_url and endpoint_pattern are required",
                validation_error=True,
            )
        if not attacker_tokens or not victims:
            return ToolResult(
                success=False,
                error="attacker_tokens and victims must be non-empty",
                validation_error=True,
            )
        # Coerce victim rows from list-of-list to tuple-of-tuple
        try:
            victims_norm = tuple(
                (str(v[0]), str(v[1])) for v in victims
                if isinstance(v, (list, tuple)) and len(v) >= 2
            )
        except (TypeError, ValueError):
            return ToolResult(
                success=False,
                error="victims must be a list of [user_id, resource_id] pairs",
                validation_error=True,
            )
        tokens_norm = tuple(str(t) for t in attacker_tokens)
        verdict = await pen_test_attacks.authz_matrix_probe(
            base_url=base_url,
            endpoint_pattern=endpoint_pattern,
            auth_header_name=auth_header_name,
            auth_header_format=auth_header_format,
            attacker_tokens=tokens_norm,
            victims=victims_norm,
            leak_indicators={
                str(k): str(v) for k, v in (leak_indicators or {}).items()
            },
            workspace_root=self._receipts_root,
            finding_id=finding_id, run_id=run_id,
        )
        data = {
            "ok": True,
            "endpoint_pattern": verdict.endpoint_pattern,
            "vulnerable": verdict.vulnerable,
            "rationale": verdict.rationale,
            "rows": [
                {
                    "attacker_token": r.attacker_token,
                    "victim_id": r.victim_id,
                    "url": r.url,
                    "status": r.status,
                    "leak_indicator_matched": r.leak_indicator_matched,
                    "latency_ms": r.latency_ms,
                    "body_excerpt": r.body_excerpt[:200],
                    "error": r.error,
                }
                for r in verdict.rows
            ],
            "leaked_count": len(verdict.leaked_rows),
        }
        return ToolResult(
            success=True,
            output=json.dumps(data, indent=2),
            metadata={
                "vulnerable": verdict.vulnerable,
                "leaked_count": len(verdict.leaked_rows),
            },
        )


class ConcurrentProbeTool(Tool):
    """Fire N parallel identical requests at one endpoint and analyze
    the response distribution.

    Phase 2 primitive aimed at the bug class LLMs systematically miss:
    TOCTOU races, uniqueness-invariant violations, scheduler-timing-
    dependent bugs (per arXiv 2508.16419). Static analysis can't model
    concurrency outcomes; this primitive does.

    When ``expected_success_count`` is set, the verdict flags
    "uniqueness violation" if more 2xx responses arrive than the
    caller's atomicity claim. Also flags 5xx (handler can't cope with
    concurrency) and non-deterministic status divergence.
    """

    def __init__(
        self,
        *,
        workspace_root_for_receipts: Path | None = None,
    ) -> None:
        self._receipts_root = (
            _bound_root(workspace_root_for_receipts)
            if workspace_root_for_receipts else None
        )

    @property
    def name(self) -> str:
        return "concurrent_probe"

    @property
    def description(self) -> str:
        return (
            "Fire N parallel identical requests at one endpoint and "
            "analyze the response distribution. Targets TOCTOU "
            "races, uniqueness-invariant violations, and concurrency-"
            "safety failures. Use when the static finding hypothesis "
            "involves check-then-act, transactional invariants, "
            "rate-limiting, or unique-resource claims."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "base_url": {"type": "string"},
                "path": {
                    "type": "string",
                    "description": "Path to attack (e.g. '/inventory/claim').",
                },
                "method": {
                    "type": "string", "default": "POST",
                    "description": (
                        "HTTP method. POST/PATCH/DELETE are common for "
                        "race-prone endpoints; GET only when the read "
                        "itself has side effects."
                    ),
                },
                "headers": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "default": {},
                },
                "body": {"type": "string", "default": ""},
                "replicas": {
                    "type": "integer",
                    "minimum": 2, "maximum": 50, "default": 10,
                    "description": (
                        "Parallel request count. 10 is enough for most "
                        "races; widen for tight TOCTOU windows."
                    ),
                },
                "expected_success_count": {
                    "type": "integer", "minimum": 0, "default": 1,
                    "description": (
                        "How many 2xx responses SHOULD occur if the "
                        "endpoint is atomic. For 'claim one unique "
                        "resource' this is 1; for 'rate-limit allows "
                        "5/sec' this is 5; for an idempotent endpoint "
                        "set to replicas."
                    ),
                },
                "finding_id": {"type": "string", "default": ""},
                "run_id": {"type": "string", "default": ""},
            },
            "required": ["base_url", "path"],
        }

    async def execute(
        self,
        *,
        base_url: str = "",
        path: str = "",
        method: str = "POST",
        headers: dict | None = None,
        body: str = "",
        replicas: int = 10,
        expected_success_count: int = 1,
        finding_id: str = "",
        run_id: str = "",
        **_kwargs,
    ) -> ToolResult:
        if not base_url or not path:
            return ToolResult(
                success=False,
                error="base_url and path are required",
                validation_error=True,
            )
        try:
            replicas_i = max(2, min(int(replicas), 50))
            expected_i = max(0, int(expected_success_count))
        except (TypeError, ValueError):
            return ToolResult(
                success=False,
                error="replicas and expected_success_count must be integers",
                validation_error=True,
            )
        verdict = await pen_test_attacks.concurrent_probe(
            base_url=base_url, path=path,
            method=(method or "POST").upper(),
            headers=(
                {str(k): str(v) for k, v in (headers or {}).items()}
            ),
            body=body or "",
            replicas=replicas_i,
            expected_success_count=expected_i,
            workspace_root=self._receipts_root,
            finding_id=finding_id, run_id=run_id,
        )
        data = {
            "ok": True,
            "endpoint": verdict.endpoint,
            "replicas": verdict.replicas,
            "vulnerable": verdict.vulnerable,
            "uniqueness_violation": verdict.uniqueness_violation,
            "error_class_divergence": verdict.error_class_divergence,
            "inconsistency": verdict.inconsistency,
            "status_distribution": verdict.status_distribution,
            "success_count": verdict.success_count,
            "error_count": verdict.error_count,
            "expected_success_count": verdict.expected_success_count,
            "rationale": verdict.rationale,
        }
        return ToolResult(
            success=True,
            output=json.dumps(data, indent=2),
            metadata={
                "vulnerable": verdict.vulnerable,
                "uniqueness_violation": verdict.uniqueness_violation,
            },
        )


PEN_TEST_TOOL_NAMES: frozenset[str] = frozenset({
    "http_attack",
    "boot_under_test",
    "under_test_status",
    "authz_matrix_probe",
    "concurrent_probe",
})


def build_pen_test_tools(
    root: Path | None = None,
    *,
    workspace_root_for_receipts: Path | None = None,
    under_test_registry: pen_test_boot._UnderTestRegistry | None = None,
) -> tuple[Tool, ...]:
    """Return the pen-test probing toolset.

    Kept separate from ``build_deterministic_tools`` so the read-only
    role allow-lists never silently grant probing capability. Only
    the pen_tester role (Phase 1c) calls this.

    ``under_test_registry`` is the per-run registry the orchestrator
    will tear down at run end. When ``None``, falls back to the
    process-default registry (ad-hoc / script usage).
    """
    workspace = root if root is not None else Path(".")
    return (
        HTTPAttackTool(
            root=root,
            workspace_root_for_receipts=workspace_root_for_receipts,
        ),
        BootUnderTestTool(
            workspace_root=workspace,
            registry=under_test_registry,
        ),
        UnderTestStatusTool(registry=under_test_registry),
        AuthzMatrixProbeTool(
            workspace_root_for_receipts=workspace_root_for_receipts,
        ),
        ConcurrentProbeTool(
            workspace_root_for_receipts=workspace_root_for_receipts,
        ),
    )
