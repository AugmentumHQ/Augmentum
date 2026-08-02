"""Deliberately vulnerable FastAPI app for pen-test integration tests.

Every endpoint is broken in a specific, documented way. The pen-test
integration tests boot this app and verify that the corresponding
primitive (http_attack / authz_matrix_probe / injection_sweep / etc.)
catches each class of vulnerability.

Run directly (for ad-hoc verification only):
    python -m tests.fixtures.vuln_app.app --port 8765

DO NOT use this app as a starting point for real code. Every block
labeled ``# VULN-N:`` is intentionally exploitable.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import traceback
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# In-memory state — re-built on each boot so tests get a clean slate
# ---------------------------------------------------------------------------


def _build_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE notes ("
        "id INTEGER PRIMARY KEY, "
        "user_id TEXT NOT NULL, "
        "title TEXT NOT NULL, "
        "body TEXT NOT NULL DEFAULT '')",
    )
    cur.execute(
        "INSERT INTO notes (id, user_id, title, body) VALUES "
        "(1, 'alice', 'alice secret', 'private to alice'),"
        "(2, 'bob',   'bob secret',   'private to bob'),"
        "(3, 'alice', 'alice public', 'shared')",
    )
    conn.commit()
    return conn


_DB = _build_db()


# ---------------------------------------------------------------------------
# Token auth — bearer ``alice`` or ``bob`` maps to that user. The
# fixture deliberately uses an absurdly simple scheme; tests don't
# care about the auth implementation, only about whether the
# endpoints CHECK auth.
# ---------------------------------------------------------------------------


_VALID_USERS: frozenset[str] = frozenset({"alice", "bob"})


def _current_user(authorization: str | None) -> str | None:
    """Return the user id implied by the Authorization header.

    None means "no auth provided" (or invalid). Endpoints that should
    require auth must check this; the broken endpoints DON'T.
    """
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token if token in _VALID_USERS else None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


app = FastAPI(title="bug_finder vuln fixture")


@app.get("/")
def root() -> dict:
    """Healthcheck — always 200, no auth required."""
    return {"status": "ok"}


@app.get("/notes")
def list_notes(authorization: str = Header(None)) -> dict:
    """SAFE: lists the current user's notes; filters by user_id.

    Counter-example to the broken endpoints — pen_tester should
    correctly identify this is NOT vulnerable.
    """
    user = _current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="auth required")
    cur = _DB.cursor()
    cur.execute(
        "SELECT id, title FROM notes WHERE user_id = ?", (user,),
    )
    return {"notes": [{"id": r[0], "title": r[1]} for r in cur.fetchall()]}


# ----- VULN-1: Broken access control --------------------------------------
# Endpoint listed in /docs as "admin only" but the auth check is missing.
# Should require auth; doesn't.
@app.get("/admin/stats")
def admin_stats() -> dict:
    """VULN-1: missing auth check. The docstring claims admin-only;
    no token is verified. ANY caller can read.

    Detection: http_attack with NO Authorization header should still
    return 200 with sensitive payload. A safe analog would return 401.
    """
    cur = _DB.cursor()
    cur.execute("SELECT COUNT(*) FROM notes")
    count = cur.fetchone()[0]
    return {
        "total_notes": count,
        "admin_secret": "this should never be exposed",
    }


# ----- VULN-2: Cross-tenant data leak (the killer for multi-tenant apps)
# Reads "note by id" without checking that the note belongs to the
# requesting user.
@app.get("/notes/{note_id}")
def get_note(note_id: int, authorization: str = Header(None)) -> dict:
    """VULN-2: cross-tenant leak. Returns ANY note by id, regardless
    of which user the auth token belongs to. Alice can read Bob's
    note 2 just by guessing the id.

    Detection: authz_matrix_probe asks "can user A read user B's
    resource?". For this endpoint, the answer is YES — and it should
    be NO.
    """
    user = _current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="auth required")
    cur = _DB.cursor()
    cur.execute(
        "SELECT id, user_id, title, body FROM notes WHERE id = ?",
        (note_id,),
    )
    row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    # VULN: should check row[1] == user before returning
    return {"id": row[0], "owner": row[1], "title": row[2], "body": row[3]}


# ----- VULN-3: SQL injection ----------------------------------------------
# Search endpoint that string-formats the user input into the query.
@app.get("/search")
def search(q: str = "", authorization: str = Header(None)) -> dict:
    """VULN-3: SQL injection via f-string. The ``q`` query param is
    inserted into the SQL with no quoting. A payload like
    ``'%' UNION SELECT user_id, title FROM notes--`` returns all
    notes regardless of user filter.

    Detection: injection_sweep with classic SQLi payloads should
    trigger either a SQL error (with verbose-errors on it leaks
    schema) or an over-broad result set.
    """
    user = _current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="auth required")
    cur = _DB.cursor()
    # VULN: f-string directly into SQL
    query = (
        f"SELECT id, title FROM notes WHERE user_id = '{user}' "
        f"AND title LIKE '%{q}%'"
    )
    cur.execute(query)
    rows = cur.fetchall()
    return {"results": [{"id": r[0], "title": r[1]} for r in rows]}


# ----- VULN-4: Path traversal ---------------------------------------------
_FIXTURE_DIR = Path(__file__).parent


@app.get("/file")
def read_file(name: str = "") -> JSONResponse:
    """VULN-4: path traversal. ``name`` is joined to a base directory
    with no normalization, so ``../../etc/passwd`` style payloads
    escape the intended folder.

    Detection: http_attack with ``name=../app.py`` should return the
    fixture source code (or any other reachable file). A safe analog
    would reject the path or normalize it to fail.
    """
    # VULN: no normalization, no allow-list
    target = _FIXTURE_DIR / name
    try:
        body = target.read_text(encoding="utf-8")
    except OSError:
        return JSONResponse({"error": "could not read"}, status_code=404)
    return JSONResponse({"name": name, "body": body[:500]})


# ----- VULN-5: Mass assignment --------------------------------------------
@app.post("/notes")
async def create_note(request: Request, authorization: str = Header(None)) -> dict:
    """VULN-5: mass assignment. The body is unpacked into the INSERT
    with no field allow-list, so a caller can set fields they
    shouldn't be able to (e.g. ``user_id`` to impersonate someone
    else, or future-added admin flags).

    Detection: http_attack with a body that includes ``user_id`` set
    to a DIFFERENT user should still succeed AND the resulting note
    should belong to the impersonated user.
    """
    user = _current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="auth required")
    body = await request.json()
    # VULN: trusts body['user_id'] over the auth-derived user
    target_user = body.get("user_id") or user
    title = body.get("title") or "untitled"
    note_body = body.get("body") or ""
    cur = _DB.cursor()
    cur.execute(
        "INSERT INTO notes (user_id, title, body) VALUES (?, ?, ?)",
        (target_user, title, note_body),
    )
    _DB.commit()
    return {"id": cur.lastrowid, "owner": target_user, "title": title}


# ----- VULN-7: Race condition (TOCTOU) on a counter --------------------
# Classic non-atomic read-modify-write. Concurrent decrement requests
# can drive the counter below zero because the read-check and the write
# happen in separate steps without locking.
_INVENTORY: dict[str, int] = {"items": 1}


@app.post("/inventory/claim")
async def claim_item(authorization: str = Header(None)) -> dict:
    """VULN-7: TOCTOU race condition. ``items`` count is read, checked,
    then decremented as separate operations. Under concurrent load the
    same item can be claimed by multiple requests because the
    check-then-act window is non-atomic.

    Detection: concurrent_probe firing N parallel POSTs should
    return 200 from more requests than the inventory starting count.
    """
    user = _current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="auth required")
    import asyncio
    current = _INVENTORY["items"]
    # VULN: deliberate sleep widens the TOCTOU window so the race is
    # reliably triggerable in tests. Real-world bugs of this shape have
    # microsecond windows but are still exploitable under load.
    await asyncio.sleep(0.05)
    if current <= 0:
        raise HTTPException(status_code=409, detail="sold out")
    _INVENTORY["items"] = current - 1
    return {"claimed_by": user, "remaining": _INVENTORY["items"]}


@app.post("/inventory/reset")
def reset_inventory(count: int = 1) -> dict:
    """Test helper — resets the counter. Not a vulnerability, just
    makes the test deterministic."""
    _INVENTORY["items"] = count
    return {"items": _INVENTORY["items"]}


# ----- VULN-6: Verbose error info disclosure ------------------------------
@app.get("/debug/explode")
def debug_explode() -> dict:
    """VULN-6: verbose error disclosure. Any 500 from this endpoint
    leaks the full Python traceback in the response body, including
    file paths and code excerpts.

    Detection: http_attack triggering the failure should return a
    500 whose body contains substrings like 'Traceback' or
    'File "<path>"'.
    """
    raise RuntimeError("oops")


@app.exception_handler(RuntimeError)
async def runtime_error_handler(_req: Request, exc: RuntimeError) -> JSONResponse:
    """VULN: returns the full traceback in the response. A safe
    handler would log the trace server-side and return a generic
    message client-side."""
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return JSONResponse(
        {"error": str(exc), "traceback": tb},
        status_code=500,
    )


# ---------------------------------------------------------------------------
# CLI for ad-hoc verification
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="error")
    return 0


if __name__ == "__main__":
    sys.exit(main())
