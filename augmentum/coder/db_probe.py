"""SQLite introspection helper — workspace databases through the ``db_inspect`` tool.

Coder workspaces frequently include local SQLite databases (test fixtures,
seed data, persistent app state). The base image ships both the ``sqlite3``
CLI and the stdlib ``sqlite3`` Python module — this helper picks Python for
structured JSON output the model can parse cleanly.

Supported actions:

- ``schema`` — every ``CREATE`` statement from ``sqlite_master``.
- ``tables`` — name + row count per table.
- ``sample`` — first N rows from a table (``limit`` defaults to 5).
- ``query`` — caller-supplied SELECT, capped at ``limit`` rows. Non-SELECT
  statements are refused — this is a read-only inspection tool, not a
  migration runner. The agent has ``shell_exec`` if it needs to write.
- ``integrity`` — ``PRAGMA integrity_check`` plus a quick row-count summary.
"""

from __future__ import annotations

import json
import shlex
from typing import Any

_ACTIONS = frozenset({"schema", "tables", "sample", "query", "integrity"})

# Cap on returned rows + total result size. Each row is JSON-encoded so
# the bytes cap dominates for wide tables; the row count keeps us from
# wandering through a million-row table even if each row is tiny.
_DEFAULT_LIMIT = 5
_MAX_LIMIT = 200
_MAX_RESULT_BYTES = 50_000


async def run_db_inspect(
    cm,
    workspace_id: str,
    *,
    db_path: str,
    action: str = "schema",
    table: str = "",
    limit: int = _DEFAULT_LIMIT,
    query: str = "",
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Inspect a SQLite database file inside the workspace.

    Returns a dict with: ``ok``, ``action``, action-specific payload
    (``schema`` / ``tables`` / ``rows`` / ``integrity``), and ``error``
    when failing. Always read-only — ``query`` actions are vetted to
    refuse anything that isn't a bare ``SELECT`` / ``WITH`` / ``EXPLAIN``.
    """
    action = (action or "schema").strip().lower()
    if action not in _ACTIONS:
        return {
            "ok": False,
            "error": f"action must be one of {sorted(_ACTIONS)}; got {action!r}",
            "validation_error": True,
        }
    if not db_path:
        return {
            "ok": False,
            "error": "db_path is required",
            "validation_error": True,
        }
    limit_int = max(1, min(_MAX_LIMIT, int(limit or _DEFAULT_LIMIT)))

    if action == "sample" and not table:
        return {
            "ok": False,
            "error": "sample requires table=<name>",
            "validation_error": True,
        }
    if action == "query" and not query.strip():
        return {
            "ok": False,
            "error": "query requires query=<SELECT statement>",
            "validation_error": True,
        }
    if action == "query":
        head = query.lstrip().split(None, 1)[0].upper() if query.strip() else ""
        if head not in {"SELECT", "WITH", "EXPLAIN", "PRAGMA"}:
            return {
                "ok": False,
                "error": (
                    "db_inspect is read-only — query must begin with "
                    "SELECT / WITH / EXPLAIN / PRAGMA. Use shell_exec "
                    "for writes."
                ),
                "validation_error": True,
            }

    script = (
        "import json,sqlite3,os,time\n"
        f"db={db_path!r}; action={action!r}; table={table!r}; "
        f"limit={limit_int!r}; query={query!r}; max_bytes={_MAX_RESULT_BYTES!r}\n"
        "start=time.time()\n"
        "if not os.path.exists(db):\n"
        "    print(json.dumps({'ok':False,'error':f'db not found: {db}'})); raise SystemExit(0)\n"
        "try:\n"
        "    # Read-only URI mount — defense in depth on top of the\n"
        "    # SELECT/WITH/EXPLAIN/PRAGMA filter on the caller side.\n"
        "    conn=sqlite3.connect(f'file:{db}?mode=ro', uri=True, timeout=5.0)\n"
        "    conn.row_factory=sqlite3.Row\n"
        "    cur=conn.cursor()\n"
        "    out={'ok':True,'action':action}\n"
        "    def _rows_to_list(cursor, cap):\n"
        "        cols=[d[0] for d in (cursor.description or [])]\n"
        "        rows=[]\n"
        "        size=0\n"
        "        for r in cursor.fetchmany(cap):\n"
        "            row={k:r[k] for k in cols}\n"
        "            rows.append(row)\n"
        "            size += len(json.dumps(row, default=str))\n"
        "            if size > max_bytes: break\n"
        "        return cols, rows, (size > max_bytes)\n"
        "    if action == 'schema':\n"
        "        cur.execute(\"SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name\")\n"
        "        items=[{'type':r['type'],'name':r['name'],'sql':r['sql'] or ''} for r in cur.fetchall()]\n"
        "        out['schema']=items[:200]\n"
        "    elif action == 'tables':\n"
        "        cur.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name\")\n"
        "        tables=[]\n"
        "        for r in cur.fetchall():\n"
        "            try:\n"
        "                c=conn.execute(f'SELECT COUNT(*) FROM \"{r[\"name\"]}\"')\n"
        "                cnt=int(c.fetchone()[0])\n"
        "            except Exception as exc:\n"
        "                cnt=-1\n"
        "            tables.append({'name':r['name'],'rows':cnt})\n"
        "        out['tables']=tables[:200]\n"
        "    elif action == 'sample':\n"
        "        # Validate table name against sqlite_master to refuse injection.\n"
        "        cur.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name=?\", (table,))\n"
        "        if not cur.fetchone():\n"
        "            print(json.dumps({'ok':False,'error':f'table not found: {table}'})); raise SystemExit(0)\n"
        "        cur.execute(f'SELECT * FROM \"{table}\" LIMIT {limit}')\n"
        "        cols, rows, truncated=_rows_to_list(cur, limit)\n"
        "        out['columns']=cols; out['rows']=rows; out['truncated']=truncated\n"
        "    elif action == 'query':\n"
        "        cur.execute(query)\n"
        "        cols, rows, truncated=_rows_to_list(cur, limit)\n"
        "        out['columns']=cols; out['rows']=rows; out['truncated']=truncated\n"
        "    elif action == 'integrity':\n"
        "        cur.execute('PRAGMA integrity_check')\n"
        "        rows=[r[0] for r in cur.fetchall()]\n"
        "        out['integrity']=rows[:50]\n"
        "        cur.execute(\"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'\")\n"
        "        out['table_count']=int(cur.fetchone()[0])\n"
        "    out['latency_ms']=int((time.time()-start)*1000)\n"
        "    conn.close()\n"
        "    print(json.dumps(out, default=str))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'ok':False,'error':str(exc),"
        "'latency_ms':int((time.time()-start)*1000)}))\n"
    )
    out = await cm.run_command(
        workspace_id,
        ["bash", "-lc", f"python3 -c {shlex.quote(script)}"],
        timeout=float(timeout) + 5.0,
    )
    try:
        return json.loads((out or "").strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "error": out or "no output from db_inspect"}
