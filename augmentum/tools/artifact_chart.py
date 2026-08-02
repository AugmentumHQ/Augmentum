"""Chart artifact tool — generates professional chart images from data.

The renderer is theme-aware (charts adopt the same palette as the PDF/PPTX/
XLSX they're embedded in), formats numbers like a human would ($142K, 45%,
1.2M) on both axes and data labels, and honours the design rules the artifact
templates ask for (pie "Other" grouping, sorted comparison bars, trend lines).
"""

from __future__ import annotations

import io
import json
import re
from typing import TYPE_CHECKING

from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.tools.artifact_storage import ArtifactStore

log = get_logger(__name__)

# Module-level guard: matplotlib.use() must be called before any pyplot
# import and is process-global. Calling it on every render is wasteful
# and can emit UserWarnings if another component sets a different
# backend in the meantime. We pin Agg once on first chart render and
# never again. matplotlib stays lazy-imported because it is a
# Docker-only dependency (see Dockerfile.gpu); plain test environments
# may not have it installed.
_MPL_BACKEND_PINNED = False

_VALID_VALUE_FORMATS = ("auto", "number", "currency", "percent", "abbreviated")
_VALID_SORTS = ("none", "asc", "desc")


def _pin_matplotlib_backend() -> None:
    global _MPL_BACKEND_PINNED
    if _MPL_BACKEND_PINNED:
        return
    import matplotlib
    matplotlib.use("Agg")
    _MPL_BACKEND_PINNED = True


class ChartTool(Tool):
    """Generate professional chart images (PNG) from structured data."""

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._store = artifact_store

    @property
    def name(self) -> str:
        return "create_chart"

    @property
    def description(self) -> str:
        return (
            "Create a professional chart image (PNG) from structured data, "
            "shown inline in your reply. Call it whenever your answer contains "
            "numbers that read better as a picture: comparing quantities across "
            "categories, a trend over time, or a breakdown of a whole. You do "
            "not need to be asked — if you are about to write a table of "
            "figures or describe a pattern in data, draw it instead. "
            "Supports bar, line, pie, scatter, area, stacked_bar, stacked_area, "
            "and horizontal_bar charts. Numbers are formatted automatically "
            "($, %, K/M/B) — set value_format to override. Use show_values=true "
            "to label points, sort to rank bars, and subtitle/caption for context. "
            "Returns a download link and an image URL for embedding. "
            "Use descriptive labels and real numbers — no placeholder text."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.ARTIFACT

    @property
    def error_hints(self) -> dict[str, str]:
        """Recovery guidance keyed on substrings of the errors this tool emits.

        Matched by ``Tool.enrich_error`` and appended to what the model sees, so
        a mis-formatted call can be fixed on the next attempt instead of turning
        into "I wasn't able to create the chart." Each key must stay a substring
        of a real error string in ``execute`` / ``_render_chart``.
        """
        return {
            "Labels and datasets are required": (
                "Pass BOTH: `labels` is a flat array of category names "
                '(["Q1","Q2"]) and `datasets` is an array of series objects '
                '([{"name":"Revenue","values":[12,15]}]). A series whose values '
                "were empty or non-numeric is dropped, which can leave "
                "`datasets` empty — check you sent numbers, not strings like "
                '"N/A" or "unknown". If you do not have the figures yet, look '
                "them up first, then call this tool."
            ),
            "Unsupported chart type": (
                "Use one of: bar, line, pie, scatter, area, stacked_bar, "
                "stacked_area, horizontal_bar. For a ranking use bar (or "
                "horizontal_bar for long labels); for change over time use "
                "line; for shares of a whole use pie."
            ),
            # matplotlib raises this when a series length != len(labels).
            "shape mismatch": (
                "Every series in `datasets` must have exactly as many values as "
                "there are `labels`. Count them and pad or trim — use 0 for a "
                "genuinely missing data point rather than omitting it."
            ),
            "could not convert": (
                "Values must be plain numbers — no currency symbols, commas, "
                "units, or ranges. Send 1200000, not \"$1.2M\". Put the unit in "
                "`y_label` and let value_format handle display."
            ),
            "No module named 'matplotlib'": (
                "The chart renderer is not installed on this server, so no "
                "retry will succeed. Do NOT call this tool again in this "
                "conversation — present the numbers as a markdown table "
                "instead and tell the user charting is unavailable."
            ),
        }

    @property
    def model_hint(self) -> str:
        return (
            "Pass real labels and values in `labels`/`datasets` — never "
            "placeholders, and never call it with data you don't have (look it "
            "up first, then chart it). One chart per distinct comparison: if "
            "the series share an axis and unit, put them in ONE call as "
            "multiple datasets rather than emitting several charts. Skip it for "
            "a single number or two, which prose says better than a picture."
        )

    def health_check(self) -> bool:
        """False when matplotlib isn't installed.

        matplotlib is a Docker-only dependency (see the backend-pinning note
        above), so a bare-metal or plain-test install can have this tool
        registered but unable to render. Charts are exposed to the model in
        passthrough Auto mode, and a tool on the roster that always fails at
        execute time is worse than one that simply isn't offered — the model
        promises a chart and then apologises. ``find_spec`` only resolves the
        module path, so this stays cheap enough for the tools dropdown and
        keeps matplotlib lazy.
        """
        from importlib.util import find_spec
        try:
            return find_spec("matplotlib") is not None
        except (ImportError, ValueError):
            return False

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Chart title",
                },
                "subtitle": {
                    "type": "string",
                    "description": "Optional one-line subtitle under the title (context/timeframe)",
                    "default": "",
                },
                "chart_type": {
                    "type": "string",
                    "enum": [
                        "bar", "line", "pie", "scatter", "area",
                        "stacked_bar", "stacked_area", "horizontal_bar",
                    ],
                    "description": "Type of chart (default: bar)",
                    "default": "bar",
                },
                "x_label": {
                    "type": "string",
                    "description": "X-axis label (optional)",
                    "default": "",
                },
                "y_label": {
                    "type": "string",
                    "description": "Y-axis label — include units e.g. 'Revenue ($M)' so values format correctly",
                    "default": "",
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Category labels (x-axis for bar/line, slice labels for pie)",
                },
                "datasets": {
                    "type": "array",
                    "description": "One or more data series",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Series name (for legend)",
                            },
                            "values": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Data values",
                            },
                        },
                        "required": ["values"],
                    },
                },
                "value_format": {
                    "type": "string",
                    "enum": list(_VALID_VALUE_FORMATS),
                    "description": (
                        "How to format numbers: 'currency' ($1.2K), 'percent' (45%), "
                        "'abbreviated' (1.2M), 'number' (1,234), or 'auto' to infer "
                        "from the axis label and magnitude (default: auto)"
                    ),
                    "default": "auto",
                },
                "sort": {
                    "type": "string",
                    "enum": list(_VALID_SORTS),
                    "description": (
                        "Sort categories by value for single-series bar/pie charts: "
                        "'desc' (largest first), 'asc', or 'none' to keep order "
                        "(default: none — keep none for chronological data)"
                    ),
                    "default": "none",
                },
                "caption": {
                    "type": "string",
                    "description": "Optional source/attribution caption shown at the bottom",
                    "default": "",
                },
                "show_values": {
                    "type": "boolean",
                    "description": "Display data values on the chart (default: false; auto-on for simple bar/pie)",
                    "default": False,
                },
            },
            "required": ["title", "labels", "datasets"],
        }

    async def execute(
        self,
        *,
        title: str = "Chart",
        subtitle: str = "",
        chart_type: str = "bar",
        x_label: str = "",
        y_label: str = "",
        labels: list | None = None,
        datasets: list | None = None,
        value_format: str = "auto",
        sort: str = "none",
        caption: str = "",
        show_values: bool = False,
        theme: str = "",
        design: dict | None = None,
        highlight_max: bool = False,
        trend_line: bool | None = None,
        donut: bool = False,
        task_id: str = "",
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        from augmentum.tools.artifact_normalize import (
            normalize_bool,
            normalize_chart_datasets,
            normalize_chart_labels,
            normalize_str,
        )

        title = normalize_str(title, "Chart")
        subtitle = normalize_str(subtitle)
        caption = normalize_str(caption)
        x_label = normalize_str(x_label)
        y_label = normalize_str(y_label)
        labels = normalize_chart_labels(labels)
        datasets = normalize_chart_datasets(datasets)

        from augmentum.tools.artifact_sanitize import sanitize_chart_labels
        labels = sanitize_chart_labels(labels)
        show_values = normalize_bool(show_values)

        value_format = (value_format or "auto").lower()
        if value_format not in _VALID_VALUE_FORMATS:
            value_format = "auto"
        sort = (sort or "none").lower()
        if sort not in _VALID_SORTS:
            sort = "none"

        if not labels or not datasets:
            return ToolResult(success=False, error="Labels and datasets are required")

        ct = chart_type.lower()
        valid_types = (
            "bar", "line", "pie", "scatter", "area",
            "stacked_bar", "stacked_area", "horizontal_bar",
        )
        if ct not in valid_types:
            return ToolResult(success=False, error=f"Unsupported chart type: {ct}")

        # Honest low-data signal — covers both the direct tool-call path and the
        # build pipeline. We still render (the chart gets a "No data" placeholder
        # rather than a blank canvas) but flag thin/empty/all-zero data instead
        # of claiming a sparse chart is "ready".
        from augmentum.tools.artifact_validate import chart_quality
        quality = chart_quality(labels, datasets)
        warnings: list[str] = []
        if quality.degenerate:
            warnings.append(
                f"Chart has limited data ({quality.reason}) — it may look sparse "
                "or empty. Try a more specific request or supply the numbers to plot."
            )

        try:
            data = _render_chart(
                title=title,
                chart_type=ct,
                x_label=x_label,
                y_label=y_label,
                labels=labels,
                datasets=datasets,
                show_values=show_values,
                theme_name=theme,
                design=design,
                value_format=value_format,
                sort=sort,
                subtitle=subtitle,
                caption=caption,
                highlight_max=highlight_max,
                trend_line=trend_line,
                donut=donut,
            )

            safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')[:60]
            filename = f"{safe_title}.png"

            info = await self._store.save(
                data=data,
                filename=filename,
                fmt="png",
                task_id=task_id,
                session_id=session_id,
                display_name=f"{title}.png",
                user_id=Tool.extract_user_id(kwargs),
                metadata={
                    "page_type": "chart",
                    "chart_type": ct,
                    "series_count": len(datasets),
                },
                source_json=json.dumps({
                    "type": "chart",
                    "title": title,
                    "subtitle": subtitle,
                    "chart_type": ct,
                    "x_label": x_label,
                    "y_label": y_label,
                    "labels": labels,
                    "datasets": datasets,
                    "show_values": show_values,
                    "value_format": value_format,
                    "sort": sort,
                    "caption": caption,
                    "theme": theme,
                }),
            )

            from augmentum.tools.base import (
                format_output_with_warnings,
                make_artifact_card,
            )

            if quality.degenerate:
                summary = (
                    f"{ct.replace('_', ' ').title()} chart '{title}' was created, "
                    f"but it has limited data ({quality.reason}) so it may render "
                    "sparse or empty."
                )
            else:
                summary = (
                    f"{ct.replace('_', ' ').title()} chart '{title}' is ready — "
                    f"{len(datasets)} series across {len(labels)} points. "
                    "Available in the artifact library."
                )
            card = make_artifact_card(
                info,
                kind="image",  # PNG renders inline, treat like an image
                title=title,
                subtitle=ct.replace("_", " "),
                summary=summary,
                preview={
                    "artifact_kind": "chart",
                    "format": "png",
                    "size_bytes": info.get("size_bytes", 0),
                    "image_url": info.get("download_url", ""),
                    "chart_type": ct,
                    "series_count": len(datasets),
                    "label_count": len(labels),
                    "x_label": x_label,
                    "y_label": y_label,
                    "value_format": value_format,
                    "sort": sort,
                    "low_data": quality.degenerate,
                    "quality_note": quality.reason,
                },
            )
            return ToolResult(
                success=True,
                output=format_output_with_warnings(summary, warnings),
                metadata=info,
                card=card,
                warnings=warnings,
            )
        except Exception as e:
            log.error("chart_creation_failed", error=str(e), exc_info=True)
            return ToolResult(success=False, error=f"Chart creation failed: {e}")


# ---------------------------------------------------------------------------
# Color + style resolution (theme-aware)
# ---------------------------------------------------------------------------

# Fallback palette used only when a theme accent can't be parsed.
_FALLBACK_PALETTE = [
    "#2563EB", "#F97316", "#059669", "#DC2626",
    "#7C3AED", "#0891B2", "#CA8A04", "#DB2777",
]

# matplotlib generic family names per design font_family pick.
_MPL_FONT_FAMILY = {
    "system": "sans-serif", "sans": "sans-serif",
    "serif": "serif", "mono": "monospace", "dyslexic": "sans-serif",
}


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int] | None:
    h = (hex_color or "").strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6 or any(c not in "0123456789abcdefABCDEF" for c in h):
        return None
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hsl(r: int, g: int, b: int) -> tuple[float, float, float]:
    rf, gf, bf = r / 255, g / 255, b / 255
    mx, mn = max(rf, gf, bf), min(rf, gf, bf)
    lum = (mx + mn) / 2
    d = mx - mn
    if d == 0:
        return 0.0, 0.0, lum
    s = d / (2 - mx - mn) if lum > 0.5 else d / (mx + mn)
    if mx == rf:
        h = ((gf - bf) / d + (6 if gf < bf else 0)) * 60
    elif mx == gf:
        h = ((bf - rf) / d + 2) * 60
    else:
        h = ((rf - gf) / d + 4) * 60
    return h, s, lum


def _hsl_to_hex(h: float, s: float, lum: float) -> str:
    h = h % 360
    c = (1 - abs(2 * lum - 1)) * s
    xx = c * (1 - abs((h / 60) % 2 - 1))
    m = lum - c / 2
    if h < 60:
        rp, gp, bp = c, xx, 0
    elif h < 120:
        rp, gp, bp = xx, c, 0
    elif h < 180:
        rp, gp, bp = 0, c, xx
    elif h < 240:
        rp, gp, bp = 0, xx, c
    elif h < 300:
        rp, gp, bp = xx, 0, c
    else:
        rp, gp, bp = c, 0, xx
    r = round((rp + m) * 255)
    g = round((gp + m) * 255)
    b = round((bp + m) * 255)
    return f"#{r:02X}{g:02X}{b:02X}"


# Hue offsets for the accent-anchored palette — tuned (like the Studio
# chart preview) to land on visually distinct hues without going garish.
_PALETTE_OFFSETS = (0, 32, 64, 192, 262, 326, 140, 96, 220, 300)


def _palette(accent: str, n: int) -> list[str]:
    """Build an ``n``-color palette anchored on the theme accent.

    The accent stays first (single-series charts read as "the brand color");
    the rest are HSL hue rotations at a consistent saturation/lightness so
    multi-series charts are distinguishable but cohesive.
    """
    rgb = _hex_to_rgb(accent)
    if rgb is None:
        return [_FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)] for i in range(max(n, 1))]
    h, s, lum = _rgb_to_hsl(*rgb)
    s = min(max(s, 0.5), 0.85)
    lum = min(max(lum, 0.42), 0.6)
    accent_norm = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
    out = [accent_norm]
    for i in range(1, max(n, 1)):
        out.append(_hsl_to_hex(h + _PALETTE_OFFSETS[i % len(_PALETTE_OFFSETS)], s, lum))
    return out[:max(n, 1)]


class _ChartStyle:
    """Resolved visual style for one chart render (theme + design)."""

    __slots__ = (
        "bg", "text", "text_secondary", "grid", "accent", "accent_dark",
        "font_family", "title_size", "subtitle_size", "label_size",
        "tick_size", "value_size", "caption_size",
    )

    def __init__(self, theme_name: str, design: dict | None):
        from augmentum.tools.artifact_theme import (
            apply_design,
            get_theme,
            normalize_design,
        )

        theme = apply_design(get_theme(theme_name), design)
        d = normalize_design(design, fallback_theme=theme.name)
        scale = float(d.get("font_size_scale", 1.0))

        self.bg = theme.background
        self.text = theme.text
        self.text_secondary = theme.text_secondary
        self.grid = theme.border
        self.accent = theme.accent
        self.accent_dark = theme.accent_dark
        self.font_family = _MPL_FONT_FAMILY.get(d.get("font_family", "system"), "sans-serif")
        # Chart-specific type scale (independent of the document body scale).
        self.title_size = 16.0 * scale
        self.subtitle_size = 11.0 * scale
        self.label_size = 11.0 * scale
        self.tick_size = 9.5 * scale
        self.value_size = 8.5 * scale
        self.caption_size = 8.0 * scale


# ---------------------------------------------------------------------------
# Value formatting (axes + labels)
# ---------------------------------------------------------------------------

_CURRENCY_HINTS = (
    "$", "£", "€", "¥", "usd", "eur", "gbp", "revenue", "cost", "price",
    "sales", "budget", "profit", "income", "expense", "spend", "salary",
    "dollar", "valuation", "$m", "$k", "$b",
)
_PERCENT_HINTS = ("%", "percent", "rate", "share", "ratio", "growth", "margin", "ctr")


def _detect_value_format(value_format: str, x_label: str, y_label: str, flat: list[float]) -> str:
    if value_format and value_format != "auto":
        return value_format
    text = f"{y_label} {x_label}".lower()
    if any(h in text for h in _PERCENT_HINTS):
        return "percent"
    if any(h in text for h in _CURRENCY_HINTS):
        return "currency"
    maxabs = max((abs(v) for v in flat), default=0)
    if maxabs >= 10000:
        return "abbreviated"
    return "number"


def _abbrev(v: float) -> str:
    a = abs(v)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            num = v / div
            s = f"{num:.1f}".rstrip("0").rstrip(".")
            return f"{s}{suf}"
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:,.1f}"


def _make_value_formatter(fmt: str, flat: list[float]):
    """Return a ``value -> str`` function for the resolved format."""
    maxabs = max((abs(v) for v in flat), default=0)
    # Percent values supplied as fractions (0..1.5) get scaled to 0..100.
    pct_scale = 100.0 if (flat and maxabs <= 1.5) else 1.0

    def fmt_number(v: float) -> str:
        if v == int(v):
            return f"{int(v):,}"
        return f"{v:,.2f}".rstrip("0").rstrip(".")

    def f(v: float) -> str:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return str(v)
        if fmt == "percent":
            p = v * pct_scale
            return f"{p:.0f}%" if p == int(p) else f"{p:.1f}%"
        if fmt == "abbreviated":
            return _abbrev(v)
        if fmt == "currency":
            if abs(v) >= 10000:
                return f"${_abbrev(v)}" if v >= 0 else f"-${_abbrev(abs(v))}"
            if v == int(v):
                return f"${int(v):,}" if v >= 0 else f"-${abs(int(v)):,}"
            return f"${v:,.2f}" if v >= 0 else f"-${abs(v):,.2f}"
        return fmt_number(v)

    return f


# ---------------------------------------------------------------------------
# Chart rendering via matplotlib
# ---------------------------------------------------------------------------


def _coerce_values(values, n: int) -> list[float]:
    """Pad/truncate a series to length ``n`` with numeric floats."""
    out: list[float] = []
    for v in (values or [])[:n]:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(0.0)
    out.extend([0.0] * (n - len(out)))
    return out


def _render_chart(
    title: str,
    chart_type: str,
    x_label: str = "",
    y_label: str = "",
    labels: list | None = None,
    datasets: list | None = None,
    show_values: bool = False,
    theme_name: str = "",
    design: dict | None = None,
    value_format: str = "auto",
    sort: str = "none",
    subtitle: str = "",
    caption: str = "",
    highlight_max: bool = False,
    trend_line: bool | None = None,
    donut: bool = False,
) -> bytes:
    """Render chart data to a professional PNG using matplotlib.

    Theme-aware (light background + brand-accent palette), with human number
    formatting on axes and labels. Uses explicit Figure/Axes creation (not
    pyplot global state) to stay thread-safe under concurrent renders; the
    Agg backend is pinned once via ``_pin_matplotlib_backend()``.
    """
    _pin_matplotlib_backend()
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter

    st = _ChartStyle(theme_name, design)
    labels = [str(la) for la in (labels or [])]
    datasets = [d for d in (datasets or []) if isinstance(d, dict)]
    n = len(labels)

    # Flatten numeric values up front for formatter detection + baseline.
    flat = [
        float(v) for d in datasets for v in (d.get("values") or [])
        if isinstance(v, int | float) and not isinstance(v, bool)
    ]
    resolved_fmt = _detect_value_format(value_format, x_label, y_label, flat)
    fmt_value = _make_value_formatter(resolved_fmt, flat)
    axis_formatter = FuncFormatter(lambda v, _pos: fmt_value(v))

    is_horizontal = chart_type == "horizontal_bar"
    n_series = max(len(datasets), 1)

    # --- Adaptive figure size ---------------------------------------------
    if is_horizontal:
        height = min(max(2.4 + 0.5 * n, 5), 16)
        figsize = (10, height)
    elif chart_type == "pie":
        figsize = (8.5, 6.5)
    else:
        width = min(max(5.5 + 0.45 * n, 8), 18)
        figsize = (width, 6)

    fig = Figure(figsize=figsize, dpi=150, facecolor=st.bg)
    ax = fig.subplots()
    ax.set_facecolor(st.bg)
    ax.tick_params(colors=st.text_secondary, labelsize=st.tick_size)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(st.grid)

    x = np.arange(n)

    # --- Empty-state ------------------------------------------------------
    if not flat:
        ax.text(
            0.5, 0.5, "No data available", transform=ax.transAxes,
            ha="center", va="center", fontsize=15, color=st.text_secondary,
            alpha=0.7, fontfamily=st.font_family,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        _apply_titles(fig, ax, st, title, subtitle, caption)
        return _save_png(fig)

    palette = _palette(st.accent, max(n_series, n if chart_type == "pie" else n_series))

    def label_color(_i: int) -> str:
        return st.text_secondary

    # --- Optional sort (single-series category charts) --------------------
    single = n_series == 1
    if single and sort in ("asc", "desc") and chart_type in ("bar", "horizontal_bar", "pie"):
        vals0 = _coerce_values(datasets[0].get("values"), n)
        order = sorted(range(n), key=lambda i: vals0[i], reverse=(sort == "desc"))
        labels = [labels[i] for i in order]
        datasets = [{
            "name": datasets[0].get("name", ""),
            "values": [vals0[i] for i in order],
        }]
        x = np.arange(n)
        # Highlighting the top bar reads naturally once sorted.
        if sort == "desc" and chart_type in ("bar", "horizontal_bar"):
            highlight_max = True

    # Auto-enable value labels for simple single-series category charts.
    if not show_values and single and chart_type in ("bar", "horizontal_bar") and n <= 12:
        show_values = True

    # ----------------------------------------------------------------------
    if chart_type == "pie":
        _render_pie(ax, st, labels, datasets, palette, donut, fmt_value)

    elif chart_type == "scatter":
        for i, ds in enumerate(datasets):
            vals = _coerce_values(ds.get("values"), n)
            ax.scatter(x, vals, label=ds.get("name") or f"Series {i+1}",
                       color=palette[i % len(palette)], s=64, alpha=0.85,
                       edgecolors=st.bg, linewidth=0.6, zorder=3)
            if show_values:
                _label_points(ax, st, x, vals, fmt_value)
        _category_axis(ax, st, x, labels, axis="x")
        ax.yaxis.set_major_formatter(axis_formatter)
        ax.grid(True, axis="both", linestyle="--", alpha=0.25, color=st.grid)

    elif chart_type == "stacked_area":
        all_values, names, colors = [], [], []
        for i, ds in enumerate(datasets):
            all_values.append(_coerce_values(ds.get("values"), n))
            names.append(ds.get("name") or f"Series {i+1}")
            colors.append(palette[i % len(palette)])
        ax.stackplot(x, *all_values, labels=names, colors=colors, alpha=0.8)
        _category_axis(ax, st, x, labels, axis="x")
        ax.yaxis.set_major_formatter(axis_formatter)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25, color=st.grid)

    elif chart_type == "area":
        for i, ds in enumerate(datasets):
            vals = _coerce_values(ds.get("values"), n)
            color = palette[i % len(palette)]
            ax.fill_between(x, vals, alpha=0.18, color=color)
            ax.plot(x, vals, label=ds.get("name") or f"Series {i+1}",
                    color=color, linewidth=2.4, zorder=3)
            if show_values:
                _label_points(ax, st, x, vals, fmt_value)
        _category_axis(ax, st, x, labels, axis="x")
        ax.yaxis.set_major_formatter(axis_formatter)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25, color=st.grid)
        _apply_baseline(ax, flat, "auto")

    elif chart_type == "line":
        for i, ds in enumerate(datasets):
            vals = _coerce_values(ds.get("values"), n)
            color = palette[i % len(palette)]
            ax.plot(x, vals, label=ds.get("name") or f"Series {i+1}",
                    color=color, linewidth=2.4, marker="o", markersize=5,
                    markeredgecolor=st.bg, markeredgewidth=0.8, zorder=3)
            if show_values:
                _label_points(ax, st, x, vals, fmt_value)
        # Trend line: auto for a single-series line with enough points.
        want_trend = trend_line if trend_line is not None else (single and n >= 4)
        if want_trend and single:
            _draw_trend_line(ax, st, x, _coerce_values(datasets[0].get("values"), n))
        _category_axis(ax, st, x, labels, axis="x")
        ax.yaxis.set_major_formatter(axis_formatter)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25, color=st.grid)
        _apply_baseline(ax, flat, "auto")

    elif chart_type == "stacked_bar":
        bottom = np.zeros(n)
        for i, ds in enumerate(datasets):
            vals = np.array(_coerce_values(ds.get("values"), n), dtype=float)
            bars = ax.bar(x, vals, 0.62, bottom=bottom,
                          label=ds.get("name") or f"Series {i+1}",
                          color=palette[i % len(palette)], alpha=0.92,
                          edgecolor=st.bg, linewidth=0.6)
            if show_values:
                ax.bar_label(bars, labels=[fmt_value(v) if v else "" for v in vals],
                             label_type="center", fontsize=st.value_size,
                             color=st.bg, fontfamily=st.font_family)
            bottom += vals
        _category_axis(ax, st, x, labels, axis="x")
        ax.yaxis.set_major_formatter(axis_formatter)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25, color=st.grid)

    elif chart_type == "horizontal_bar":
        _render_grouped_bars(
            ax, st, x, labels, datasets, palette, fmt_value, show_values,
            highlight_max and single, horizontal=True,
        )
        ax.xaxis.set_major_formatter(axis_formatter)
        ax.set_xlim(left=0)
        ax.grid(True, axis="x", linestyle="--", alpha=0.25, color=st.grid)
        ax.invert_yaxis()  # first label on top

    else:  # bar
        _render_grouped_bars(
            ax, st, x, labels, datasets, palette, fmt_value, show_values,
            highlight_max and single, horizontal=False,
        )
        ax.yaxis.set_major_formatter(axis_formatter)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25, color=st.grid)

    ax.set_axisbelow(True)

    if x_label:
        ax.set_xlabel(x_label, fontsize=st.label_size, color=st.text_secondary,
                      fontfamily=st.font_family, labelpad=8)
    if y_label and chart_type != "pie":
        ax.set_ylabel(y_label, fontsize=st.label_size, color=st.text_secondary,
                      fontfamily=st.font_family, labelpad=8)

    # --- Legend (outside the plot, never covering data) -------------------
    if chart_type != "pie" and (n_series > 1 or (single and datasets and datasets[0].get("name"))):
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.14),
            ncol=min(n_series, 4), frameon=False, fontsize=st.tick_size,
            labelcolor=st.text_secondary,
        )

    _apply_titles(fig, ax, st, title, subtitle, caption)
    return _save_png(fig)


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _category_axis(ax, st: _ChartStyle, x, labels: list, axis: str) -> None:
    rotation = 45 if len(labels) > 6 else 0
    if axis == "x":
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=rotation,
                           ha="right" if rotation else "center")
        for lbl in ax.get_xticklabels():
            lbl.set_color(st.text_secondary)
            lbl.set_fontfamily(st.font_family)
    else:
        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        for lbl in ax.get_yticklabels():
            lbl.set_color(st.text_secondary)
            lbl.set_fontfamily(st.font_family)


def _render_grouped_bars(ax, st, x, labels, datasets, palette, fmt_value,
                         show_values, highlight, *, horizontal: bool) -> None:
    n = len(labels)
    n_series = len(datasets)
    span = 0.8
    bar_size = span / max(n_series, 1)
    single = n_series == 1
    for i, ds in enumerate(datasets):
        vals = _coerce_values(ds.get("values"), n)
        offset = (i - n_series / 2 + 0.5) * bar_size
        # Single-series + highlight: top value in accent_dark, rest accent.
        if single and highlight:
            mx = max(vals) if vals else 0
            colors = [st.accent_dark if v == mx else st.accent for v in vals]
        elif single:
            colors = st.accent
        else:
            colors = palette[i % len(palette)]
        name = ds.get("name") or f"Series {i+1}"
        if horizontal:
            bars = ax.barh(x + offset, vals, bar_size, label=name,
                           color=colors, alpha=0.92, edgecolor=st.bg, linewidth=0.5)
            if show_values:
                ax.bar_label(bars, labels=[fmt_value(v) for v in vals],
                             padding=3, fontsize=st.value_size,
                             color=st.text_secondary, fontfamily=st.font_family)
        else:
            bars = ax.bar(x + offset, vals, bar_size, label=name,
                          color=colors, alpha=0.92, edgecolor=st.bg, linewidth=0.5)
            if show_values:
                ax.bar_label(bars, labels=[fmt_value(v) for v in vals],
                             padding=3, fontsize=st.value_size,
                             color=st.text_secondary, fontfamily=st.font_family)
    if horizontal:
        _category_axis(ax, st, x, labels, axis="y")
    else:
        _category_axis(ax, st, x, labels, axis="x")


def _render_pie(ax, st, labels, datasets, palette, donut: bool, fmt_value) -> None:
    """Pie/donut with desc sort + 'Other' grouping (max 6 slices)."""
    n = len(labels)
    values = _coerce_values(datasets[0].get("values") if datasets else [], n)
    pairs = [(labels[i], values[i]) for i in range(n) if values[i] > 0]
    pairs.sort(key=lambda p: p[1], reverse=True)

    max_slices = 6
    if len(pairs) > max_slices:
        head = pairs[:max_slices - 1]
        other = sum(v for _, v in pairs[max_slices - 1:])
        pairs = head + [("Other", other)]

    if not pairs:
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes,
                ha="center", va="center", fontsize=15, color=st.text_secondary,
                alpha=0.7)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    slice_labels = [p[0] for p in pairs]
    slice_values = [p[1] for p in pairs]
    colors = _palette(st.accent, len(pairs))

    wedge_props = {"width": 0.42, "edgecolor": st.bg, "linewidth": 1.5} if donut \
        else {"edgecolor": st.bg, "linewidth": 1.5}
    wedges, texts, autotexts = ax.pie(
        slice_values, labels=slice_labels, colors=colors,
        autopct=lambda pct: f"{pct:.0f}%" if pct >= 4 else "",
        startangle=90, counterclock=False, pctdistance=0.78 if donut else 0.7,
        textprops={"color": st.text, "fontsize": st.tick_size,
                   "fontfamily": st.font_family},
        wedgeprops=wedge_props,
    )
    for at in autotexts:
        at.set_color(st.bg if not donut else st.text)
        at.set_fontsize(st.value_size)
        at.set_fontweight("bold")
    ax.set_aspect("equal")


def _label_points(ax, st, x_vals, y_vals, fmt_value) -> None:
    for xv, yv in zip(x_vals, y_vals, strict=False):
        ax.annotate(
            fmt_value(yv), xy=(xv, yv), xytext=(0, 6),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=st.value_size, color=st.text_secondary,
            fontfamily=st.font_family,
        )


def _draw_trend_line(ax, st, x, y) -> None:
    import numpy as np
    if len(x) < 2:
        return
    try:
        coeffs = np.polyfit(np.asarray(x, dtype=float), np.asarray(y, dtype=float), 1)
    except Exception:  # noqa: BLE001 — degenerate fit, skip the overlay
        return
    fit = np.poly1d(coeffs)
    ax.plot(x, fit(x), linestyle="--", linewidth=1.6, color=st.text_secondary,
            alpha=0.6, label="_trend", zorder=2)


def _apply_baseline(ax, flat: list[float], mode: str) -> None:
    """Non-zero y-baseline for line/area when data is clustered high.

    Bars always start at zero (set by their branches) — only value-position
    charts get a tightened baseline, and only when it won't mislead.
    """
    if mode != "auto" or not flat:
        return
    lo, hi = min(flat), max(flat)
    if lo > 0 and hi > 0 and (lo / hi) > 0.4:
        pad = (hi - lo) * 0.15 or hi * 0.05
        ax.set_ylim(bottom=max(0, lo - pad))


def _apply_titles(fig, ax, st: _ChartStyle, title: str, subtitle: str, caption: str) -> None:
    if subtitle:
        ax.set_title(title, fontsize=st.title_size, fontweight="bold",
                     color=st.text, fontfamily=st.font_family, pad=24, loc="left")
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes,
                fontsize=st.subtitle_size, color=st.text_secondary,
                fontfamily=st.font_family, ha="left", va="bottom")
    else:
        ax.set_title(title, fontsize=st.title_size, fontweight="bold",
                     color=st.text, fontfamily=st.font_family, pad=14, loc="left")
    if caption:
        fig.text(0.01, 0.005, caption, fontsize=st.caption_size,
                 color=st.text_secondary, fontfamily=st.font_family,
                 ha="left", va="bottom", style="italic")


def _save_png(fig) -> bytes:
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    return buf.getvalue()
