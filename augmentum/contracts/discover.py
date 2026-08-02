"""Route discovery — introspect a FastAPI app into ``RouteSpec`` records.

The first half of the contract harness: given a constructed app, enumerate
every callable HTTP entrypoint with enough detail to synthesize a probe
(method, path, path params, required query params, whether it takes a body).

FastAPI carries a ``dependant`` on each ``APIRoute`` that already resolved
the handler's path/query/body params from its type signature — we read that
rather than re-parsing the signature ourselves. Falls back to a regex over
the path template when the dependant isn't present (plain Starlette routes).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PATH_PARAM = re.compile(r"\{([^}:]+)(?::[^}]+)?\}")


@dataclass(frozen=True, slots=True)
class RouteSpec:
    """One callable HTTP entrypoint, discovered from ``app.routes``."""

    method: str                       # "GET" | "POST" | ...
    path: str                         # "/api/chats/{id}"
    name: str                         # route name (handler __name__ usually)
    handler: str                      # "module:qualname"
    path_params: tuple[str, ...]      # ("id",)
    required_query: tuple[str, ...]   # query params with no default
    has_body: bool                    # takes a request body model

    @property
    def key(self) -> str:
        return f"{self.method} {self.path}"


def _handler_ref(endpoint: object) -> str:
    mod = getattr(endpoint, "__module__", "?")
    qual = getattr(endpoint, "__qualname__", getattr(endpoint, "__name__", "?"))
    return f"{mod}:{qual}"


def _dependant_params(route: object) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    """Pull (path_params, required_query, has_body) from a FastAPI dependant.

    Returns empty/False on anything that isn't a FastAPI APIRoute — the caller
    falls back to a path-template regex for path params.
    """
    dep = getattr(route, "dependant", None)
    if dep is None:
        return (), (), False
    path_names: list[str] = []
    req_query: list[str] = []
    try:
        for f in getattr(dep, "path_params", []) or []:
            name = getattr(f, "name", "")
            if name:
                path_names.append(name)
        for f in getattr(dep, "query_params", []) or []:
            name = getattr(f, "name", "")
            if name and getattr(f, "required", False):
                req_query.append(name)
        has_body = bool(getattr(dep, "body_params", []) or [])
    except Exception:  # noqa: BLE001 — introspection best-effort, never fatal
        return (), (), False
    return tuple(path_names), tuple(req_query), has_body


def discover_routes(app: object) -> list[RouteSpec]:
    """Enumerate every HTTP route on a constructed FastAPI/Starlette app.

    WebSocket routes, mounts, and routes with no concrete methods are skipped.
    A route advertising N methods yields N specs (one per method).
    """
    specs: list[RouteSpec] = []
    for route in getattr(app, "routes", []):
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        endpoint = getattr(route, "endpoint", None)
        if not methods or not path or endpoint is None:
            continue  # Mount / WebSocketRoute / static — not a probeable HTTP route

        dep_path, req_query, has_body = _dependant_params(route)
        # Prefer the dependant's path params; fall back to the template regex.
        path_params = dep_path or tuple(_PATH_PARAM.findall(path))
        handler = _handler_ref(endpoint)
        name = getattr(route, "name", "") or getattr(endpoint, "__name__", "?")

        for method in sorted(methods):
            if method in ("HEAD", "OPTIONS"):
                continue
            specs.append(
                RouteSpec(
                    method=method,
                    path=path,
                    name=name,
                    handler=handler,
                    path_params=path_params,
                    required_query=req_query,
                    has_body=has_body,
                )
            )
    specs.sort(key=lambda s: (s.path, s.method))
    return specs
