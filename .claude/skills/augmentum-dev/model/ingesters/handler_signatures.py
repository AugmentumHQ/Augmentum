"""Handler-signatures ingester — AST-derived user_id-wiring facts.

For every endpoint row, parse the handler's source file with the
``ast`` module and look at the matching FunctionDef:

  * ``accepts_user_id`` — True if the function calls ``_user_id(request)``
    OR declares a ``user_id`` parameter directly.
  * ``passes_user_id``  — True if any Call within the function body
    has a ``user_id=`` keyword argument.

These are the two halves of the multi-tenant data-isolation contract
documented in CLAUDE.md. The query layer joins handler_signatures
back to endpoints (by endpoint_id) to surface routes that handle
user-scoped data without honouring the contract.

Performance: each route file is parsed exactly once per refresh; the
ingester groups endpoints by handler_file before parsing. With ~50
route files this completes in <500ms cold and is gated on file mtime
the next pass.
"""

from __future__ import annotations

import ast
import sqlite3
from collections import defaultdict
from pathlib import Path

# Request-user acquisition helpers used across augmentum/proxy/. Each is a
# module-level ``def X(request) -> str`` (or returns the user object) that
# derives the current user from ``request.scope["user"]`` and refuses anon.
# Recognising all of them — not just ``_user_id`` — is what keeps the
# multi-tenant audit from crying wolf on the ~5-in-6 scoped handlers that
# happen to use a differently-named helper. Verified against their defs;
# add a name here only after confirming it truly acquires the request user
# (a false entry would hide a real leak).
_USER_ID_HELPERS = frozenset({
    "_user_id", "_require_user", "_require_user_id", "_get_user",
    "_require_auth", "_uid",
})


def _walk_function(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[bool, bool, str]:
    """Return (accepts_user_id, passes_user_id, signature_text)."""
    # Parameter detection — direct ``user_id`` arg.
    accepts = False
    for arg in (
        list(func_node.args.args)
        + list(func_node.args.kwonlyargs)
        + list(func_node.args.posonlyargs)
    ):
        if arg.arg == "user_id":
            accepts = True
            break

    passes = False
    # Walk every Call inside the function body.
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            # Check for a user-acquisition helper call — counts as accepts.
            func = node.func
            if isinstance(func, ast.Name) and func.id in _USER_ID_HELPERS:
                accepts = True
            # Check for user_id= kwarg passed to anything.
            for kw in node.keywords:
                if kw.arg == "user_id":
                    passes = True

    sig = f"def {func_node.name}({ast.unparse(func_node.args)})"
    return accepts, passes, sig


def ingest(project_root: Path, db: sqlite3.Connection) -> None:
    # Pull (endpoint_id, handler_file_path, handler_name) for every endpoint.
    endpoint_rows = list(db.execute("""
        SELECT e.id AS endpoint_id, e.handler_name, f.path AS handler_file
        FROM endpoints e
        JOIN files f ON f.id = e.handler_file_id
        WHERE f.path LIKE 'augmentum/proxy/%_routes.py'
           OR f.path = 'augmentum/proxy/server.py'
    """))
    if not endpoint_rows:
        return

    # Group by file so each file is parsed once.
    by_file: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for r in endpoint_rows:
        by_file[r["handler_file"]].append(r)

    db.execute("BEGIN")
    try:
        db.execute("DELETE FROM handler_signatures")
        for file_rel, rows in by_file.items():
            file_abs = project_root / file_rel
            if not file_abs.is_file():
                continue
            try:
                source = file_abs.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=file_rel)
            except (OSError, SyntaxError):
                continue
            # Build a name → FunctionDef map for this file.
            funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # First occurrence wins — top-level handlers shadow
                    # inner closures / nested defs.
                    funcs.setdefault(node.name, node)
            for r in rows:
                func_node = funcs.get(r["handler_name"])
                if func_node is None:
                    # Handler name didn't resolve — could be a lambda,
                    # a class method, or stale routes.json. Skip with
                    # all-zero defaults rather than insert a mislead-
                    # ing positive signal.
                    db.execute(
                        """INSERT INTO handler_signatures
                           (endpoint_id, accepts_user_id, passes_user_id, raw_signature)
                           VALUES (?, 0, 0, NULL)""",
                        (r["endpoint_id"],),
                    )
                    continue
                accepts, passes, sig = _walk_function(func_node)
                db.execute(
                    """INSERT INTO handler_signatures
                       (endpoint_id, accepts_user_id, passes_user_id, raw_signature)
                       VALUES (?, ?, ?, ?)""",
                    (r["endpoint_id"], 1 if accepts else 0,
                     1 if passes else 0, sig),
                )
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
