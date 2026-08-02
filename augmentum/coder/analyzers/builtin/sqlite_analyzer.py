"""SQLite database analyzer — table list, row counts, schema preview."""

from __future__ import annotations

import sqlite3

from augmentum.coder.analyzers.registry import (
    AnalysisReport,
    register_analyzer,
)

# Canonical identifier-quoting helper lives in augmentum.utils.sql now
# (audit 2026-06-17). Kept under the local name so the call sites below
# read unchanged. Table names come from sqlite_master here; quoting is
# defense-in-depth + keeps red_team_scan's f-string-in-execute flag a
# clean false-positive rather than a "trust me" assertion.
from augmentum.utils.sql import quote_ident as _quote_ident


class SQLiteAnalyzer:
    name = "sqlite"
    extensions = ("db", "sqlite", "sqlite3")
    magic_bytes = (b"SQLite format 3\x00",)

    async def analyze(self, path: str, raw: bytes) -> AnalysisReport:
        # Open read-only via URI so we never accidentally mutate the file
        # — analyzers are observers, not editors.
        uri = f"file:{path}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        except sqlite3.OperationalError as exc:
            return AnalysisReport(
                format="SQLite (locked or unreadable)",
                summary=f"SQLite file detected but couldn't open read-only: {exc}",
            )

        try:
            tables = [
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            indexes = [
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            views = [
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='view' "
                    "ORDER BY name"
                )
            ]

            table_lines: list[str] = []
            table_details: dict[str, dict] = {}
            for tbl in tables[:30]:
                ident = _quote_ident(tbl)
                try:
                    cols = list(conn.execute(f"PRAGMA table_info({ident})"))
                    row = conn.execute(
                        f"SELECT COUNT(*) FROM {ident}"
                    ).fetchone()
                    row_count = int(row[0]) if row else 0
                except sqlite3.Error as exc:
                    cols = []
                    row_count = -1
                    table_details[tbl] = {"error": str(exc)[:200]}
                else:
                    table_details[tbl] = {
                        "row_count": row_count,
                        "columns": [
                            {"name": c[1], "type": c[2], "pk": bool(c[5])}
                            for c in cols
                        ],
                    }
                col_label = (
                    f"{len(cols)} cols" if cols else "(unreadable)"
                )
                count_label = (
                    f"{row_count:,} rows" if row_count >= 0 else "(unreadable)"
                )
                table_lines.append(f"  - {tbl}: {count_label}, {col_label}")
            if len(tables) > 30:
                table_lines.append(f"  …and {len(tables) - 30} more tables")
        finally:
            conn.close()

        size_mb = len(raw) / (1024 * 1024) if raw else 0
        bullets = [
            f"- File size: {size_mb:,.2f} MB",
            f"- Tables: {len(tables)}",
            f"- Indexes: {len(indexes)}",
            f"- Views: {len(views)}",
        ]

        sections = ["SQLite database\n", "\n".join(bullets)]
        if table_lines:
            sections.append("\n**Tables:**\n" + "\n".join(table_lines))
        sections.append(
            "\nFor schema or sample rows on a specific table, run "
            "`shell` with `sqlite3 <path> '.schema <table>'` or "
            "`SELECT * FROM <table> LIMIT 5;`."
        )

        return AnalysisReport(
            format="SQLite database",
            summary="\n".join(sections),
            details={
                "table_count": len(tables),
                "index_count": len(indexes),
                "view_count": len(views),
                "tables": table_details,
                "size_mb": round(size_mb, 2),
            },
            raw_size_bytes=len(raw),
        )


register_analyzer(SQLiteAnalyzer())
