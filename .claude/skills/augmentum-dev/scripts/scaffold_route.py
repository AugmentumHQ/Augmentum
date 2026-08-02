#!/usr/bin/env python3
"""Generate boilerplate for a new Augmentum API route file.

Usage:
    python scaffold_route.py feature_name

Example:
    python scaffold_route.py bookmarks
    -> Creates augmentum/proxy/bookmarks_routes.py
    -> Prints the server.py registration lines to add manually
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------

def _find_root() -> Path:
    """Walk up from this script to find the project root (has augmentum/ and ui/)."""
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "augmentum" / "proxy").is_dir() and (parent / "ui").is_dir():
            return parent
    print("ERROR: Cannot find Augmentum project root.", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Terminal colors (skip on Windows without ANSI support)
# ---------------------------------------------------------------------------

_COLOR = os.environ.get("TERM") or os.name != "nt"

def _red(s: str) -> str:    return f"\033[91m{s}\033[0m" if _COLOR else s
def _yellow(s: str) -> str: return f"\033[93m{s}\033[0m" if _COLOR else s
def _green(s: str) -> str:  return f"\033[92m{s}\033[0m" if _COLOR else s
def _cyan(s: str) -> str:   return f"\033[96m{s}\033[0m" if _COLOR else s
def _bold(s: str) -> str:   return f"\033[1m{s}\033[0m" if _COLOR else s


# ---------------------------------------------------------------------------
# Route file template
# ---------------------------------------------------------------------------

_ROUTE_TEMPLATE = '''\
"""{title} API routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/{name}", tags=["{name}"])


# ---------------------------------------------------------------------------
# GET — list / read
# ---------------------------------------------------------------------------

@router.get("")
async def list_{name}(request: Request) -> JSONResponse:
    """Return all {name} entries."""
    try:
        # TODO: replace with real data source
        items: list = []
        return JSONResponse(items)
    except Exception:
        log.exception("Failed to list {name}")
        return JSONResponse({{"error": "Failed to list {name}"}}, status_code=500)


# ---------------------------------------------------------------------------
# POST — create
# ---------------------------------------------------------------------------

@router.post("")
async def create_{name_singular}(request: Request) -> JSONResponse:
    """Create a new {name_singular} entry."""
    try:
        body = await request.json()
        # TODO: validate body, persist to DB
        log.info("{name_singular}_created", data=body)
        return JSONResponse({{"status": "created", "data": body}}, status_code=201)
    except Exception:
        log.exception("Failed to create {name_singular}")
        return JSONResponse({{"error": "Failed to create {name_singular}"}}, status_code=500)


# ---------------------------------------------------------------------------
# Add more endpoints below:
#
#   @router.get("/{{item_id}}")    — get single item (200)
#   @router.put("/{{item_id}}")    — update item    (200)
#   @router.delete("/{{item_id}}") — delete item    (204, no body)
# ---------------------------------------------------------------------------
'''


def _to_singular(name: str) -> str:
    """Naive singularisation: strip trailing 's' if present."""
    if name.endswith("ies"):
        return name[:-3] + "y"
    if name.endswith("ses") or name.endswith("xes") or name.endswith("zes"):
        return name[:-2]
    if name.endswith("s") and not name.endswith("ss"):
        return name[:-1]
    return name


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: scaffold_route.py <feature_name>")
        print('Example: scaffold_route.py bookmarks')
        sys.exit(1)

    raw = sys.argv[1]
    # Normalise to snake_case
    name = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    if not name:
        print(_red("ERROR: Invalid feature name."))
        sys.exit(1)

    root = _find_root()
    route_file = root / "augmentum" / "proxy" / f"{name}_routes.py"

    # ---- guard against overwrite ----
    if route_file.exists():
        print(_red(f"ERROR: {route_file.relative_to(root)} already exists. Will not overwrite."))
        sys.exit(1)

    # ---- generate route file ----
    name_singular = _to_singular(name)
    title = name.replace("_", " ").title()

    content = _ROUTE_TEMPLATE.format(
        name=name,
        name_singular=name_singular,
        title=title,
    )

    route_file.write_text(content, encoding="utf-8")
    print(_green(f"Created: {route_file.relative_to(root)}"))

    # ---- print server.py registration lines ----
    import_line = f"from augmentum.proxy.{name}_routes import router as {name}_router"
    register_line = f"app.include_router({name}_router)"

    print()
    print(_bold("Add these lines to server.py:"))
    print()
    print(f"  {_cyan('Import:')}  {import_line}")
    print(f"  {_cyan('Register:')}  {register_line}")
    print()
    print(_yellow("Remember: register the router alongside the other include_router() calls."))


if __name__ == "__main__":
    main()
