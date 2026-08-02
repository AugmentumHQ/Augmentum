"""Facts registry — named queries against the codebase model.

Each FACT entry maps a stable name (e.g. ``tables.user_scoped.count``)
to a SQL query that returns ONE row with ONE column. The rendered
value is whatever ``str(row[0])`` produces.

Doc files reference facts via fenced HTML comments:

    <!--fact:tables.user_scoped.count-->77<!--/-->

``refresh_docs.py`` walks docs, looks up each named fact, runs its
query, and replaces the inner content. Adding a new fact = one new
entry here + one fenced reference in the doc.
"""

from __future__ import annotations

from .registry import FACTS, check_fact, render_fact

__all__ = ["FACTS", "check_fact", "render_fact"]
