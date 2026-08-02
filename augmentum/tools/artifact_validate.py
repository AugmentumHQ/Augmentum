"""Data-quality gates for chart / spreadsheet artifacts.

Local models routinely emit charts and spreadsheets with too few data
points, empty value arrays, or all-zero series. Those render as blank or
broken artifacts while the success card still claims "ready" — the bug
this module exists to close.

Two consumers:

- ``artifact_pipeline.py`` (build mode) calls ``chart_quality`` /
  ``sheet_quality`` after the drafting LLM returns, and re-drafts once
  with a stricter prompt (``chart_repair_user`` / ``sheet_repair_user``)
  when the result is degenerate.
- ``ChartTool`` / ``SpreadsheetTool`` call the same checks at render time
  and, when the data is still thin, append an honest warning to
  ``ToolResult.warnings`` instead of shipping a silently-blank artifact.

Keeping both paths on one set of detectors means "what counts as empty"
is defined once.
"""

from __future__ import annotations

from dataclasses import dataclass

# Density floors we ASK drafters to hit (used in prompts) and BELOW WHICH we
# warn (used in detection). Deliberately low — the goal is to catch "1 bar /
# 2 rows / all zeros", not to police genuinely small-but-valid datasets.
MIN_CHART_POINTS = 4
MIN_SHEET_ROWS = 4


@dataclass
class DataQuality:
    """Verdict for a drafted dataset.

    Three distinct states keep the user-facing warning honest while still
    letting the build pipeline push for richer data:

    ``usable``     — there is enough to render something meaningful.
    ``degenerate`` — genuinely broken: empty, all-zero, ragged, no rows.
                     The render tools surface a warning AND the pipeline
                     re-drafts once.
    ``thin``       — renderable and valid, but below the density floor
                     (very few rows/points). The pipeline re-drafts to
                     enrich it, but the render tools DON'T warn — a 2-row
                     comparison is legitimate, not a bug.
    ``reason``     — short human-readable note, "" when the data is fine.
    """

    usable: bool
    degenerate: bool
    reason: str = ""
    thin: bool = False

    @property
    def needs_repair(self) -> bool:
        """The pipeline re-drafts on either degeneracy or thinness."""
        return self.degenerate or self.thin

    @property
    def ok(self) -> bool:
        return self.usable and not self.degenerate and not self.thin


def _numeric_values(datasets: list) -> tuple[list[float], bool]:
    """Flatten every numeric value across series; also report whether ANY
    series carried a non-empty values array at all."""
    flat: list[float] = []
    any_values = False
    for ds in datasets or []:
        if not isinstance(ds, dict):
            continue
        vals = ds.get("values") or []
        if vals:
            any_values = True
        flat.extend(v for v in vals if isinstance(v, int | float) and not isinstance(v, bool))
    return flat, any_values


def chart_quality(labels: list, datasets: list) -> DataQuality:
    """Classify a chart dataset for renderability."""
    labels = labels or []
    datasets = datasets or []
    flat, any_values = _numeric_values(datasets)

    if not labels or not datasets or not any_values or not flat:
        return DataQuality(usable=False, degenerate=True, reason="no data points")

    if all(v == 0 for v in flat):
        return DataQuality(usable=True, degenerate=True, reason="all values are zero")

    # A series that doesn't reach the label count plots as a truncated/blank
    # tail — a common local-model failure (labels filled, values short).
    ragged = any(
        len(ds.get("values") or []) < len(labels)
        for ds in datasets
        if isinstance(ds, dict)
    )
    if ragged:
        return DataQuality(usable=True, degenerate=True, reason="series shorter than labels")

    # Below here the data is valid — only thin. Pipeline enriches; no warning.
    if len(labels) < 2:
        return DataQuality(usable=True, degenerate=False, thin=True, reason="only one data point")
    if len(labels) < MIN_CHART_POINTS:
        return DataQuality(usable=True, degenerate=False, thin=True, reason="few data points")

    return DataQuality(usable=True, degenerate=False)


def _filled_row_count(rows: list) -> int:
    """Rows with at least one non-empty cell."""
    n = 0
    for row in rows or []:
        if any(str(c).strip() for c in (row or []) if c is not None):
            n += 1
    return n


def sheet_quality(sheets: list) -> DataQuality:
    """Classify a spreadsheet for renderability.

    Judged on the densest data sheet so an auto-appended "Sources" sheet
    can't mask an empty primary sheet (and vice-versa).
    """
    sheets = sheets or []
    if not sheets:
        return DataQuality(usable=False, degenerate=True, reason="no sheets")

    best_filled = -1
    best_headers = 0
    for sh in sheets:
        if not isinstance(sh, dict):
            continue
        headers = sh.get("headers") or []
        filled = _filled_row_count(sh.get("rows") or [])
        if filled > best_filled:
            best_filled = filled
            best_headers = len(headers)

    if best_filled < 0:
        return DataQuality(usable=False, degenerate=True, reason="no usable sheet")
    if best_headers == 0:
        return DataQuality(usable=False, degenerate=True, reason="no columns")
    if best_filled == 0:
        return DataQuality(usable=True, degenerate=True, reason="no data rows")
    if best_filled < MIN_SHEET_ROWS:
        # Valid but sparse — pipeline enriches; render tools don't warn (a
        # short table is often exactly what the user asked for).
        return DataQuality(usable=True, degenerate=False, thin=True, reason="few rows")

    return DataQuality(usable=True, degenerate=False)


# ---------------------------------------------------------------------------
# Repair prompts — fed back to the drafting LLM for one stricter retry
# ---------------------------------------------------------------------------

def chart_repair_user(topic: str, research_context: str, reason: str) -> tuple[str, str]:
    """(system, user) for a stricter chart re-draft after a degenerate result."""
    system = (
        "You extract chart data from research and output ONLY valid JSON — no "
        "markdown, no prose. Your previous attempt was unusable; fix it now."
    )
    user = (
        f"The chart you produced was rejected: {reason}.\n\n"
        f"Topic: {topic}\n\n"
        f"Data:\n{research_context}\n\n"
        f"Produce a COMPLETE dataset:\n"
        f"- At least {MIN_CHART_POINTS} category labels.\n"
        f"- Every series must give one real number for EVERY label "
        f"(equal-length arrays).\n"
        f"- Use actual figures from the data; if one is missing, infer a "
        f"reasonable value — never 0, blank, or a placeholder.\n\n"
        f'Output JSON exactly in this shape:\n'
        f'{{"chart_type": "bar", "x_label": "Quarter", "y_label": "Revenue ($M)", '
        f'"labels": ["Q1", "Q2", "Q3", "Q4"], '
        f'"datasets": [{{"name": "2025", "values": [12.4, 15.1, 18.9, 22.3]}}]}}'
    )
    return system, user


def sheet_repair_user(topic: str, research_context: str, reason: str) -> tuple[str, str]:
    """(system, user) for a stricter spreadsheet re-draft after a degenerate result."""
    system = (
        "You organize data into spreadsheet structure and output ONLY valid "
        "JSON — no markdown, no prose. Your previous attempt was unusable; fix it."
    )
    user = (
        f"The spreadsheet you produced was rejected: {reason}.\n\n"
        f"Topic: {topic}\n\n"
        f"Data:\n{research_context}\n\n"
        f"Produce a COMPLETE sheet:\n"
        f"- At least {MIN_SHEET_ROWS} data rows.\n"
        f"- Every row must have one value per header (no ragged rows).\n"
        f"- Use real values from the data; estimate and mark '(est)' if a "
        f"figure is unknown — never leave cells as TBD, N/A, or blank.\n\n"
        f'Output JSON exactly in this shape:\n'
        f'{{"sheets": [{{"name": "Summary", '
        f'"headers": ["Item", "2024", "2025", "Change"], '
        f'"rows": [["Revenue", 120, 145, 25], ["Costs", 80, 92, 12], '
        f'["Profit", 40, 53, 13], ["Margin %", 33, 37, 4]], '
        f'"column_formats": {{}}, "summary_row": "none"}}]}}'
    )
    return system, user
