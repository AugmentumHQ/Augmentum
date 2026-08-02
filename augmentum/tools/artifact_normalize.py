"""Input normalization for artifact tools.

LLMs produce inconsistent JSON structures — body as list vs string,
numbers as strings, nested arrays as JSON strings, None where strings
are expected.  This module normalizes inputs before artifact renderers
see them, preventing crashes from model format variations.
"""

from __future__ import annotations

import json

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def normalize_str(value: object, default: str = "") -> str:
    """Coerce a value to string.  Handles list, None, int, float."""
    if value is None:
        return default
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return str(value)


def normalize_int(value: object, default: int = 0) -> int:
    """Coerce a value to int.  Handles string, float, None."""
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except (ValueError, TypeError):
            try:
                return int(float(value))
            except (ValueError, TypeError):
                return default
    return default


def normalize_bool(value: object, default: bool = False) -> bool:
    """Coerce a value to bool.  Handles string, int, None."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    return default


def normalize_list(value: object, default: list | None = None) -> list:
    """Coerce a value to list.  Handles JSON string, None, single item."""
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        # Try parsing as JSON array
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        # Single value → wrap in list
        if stripped:
            return [stripped]
        return default
    return default


def normalize_number(value: object, default: float = 0.0) -> float:
    """Coerce a value to float.  Handles string with units, commas, None."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Strip common suffixes and formatting
        cleaned = value.strip().rstrip("%").replace(",", "")
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return default
    return default


# ---------------------------------------------------------------------------
# Document section normalization
# ---------------------------------------------------------------------------


def normalize_sections(sections: object) -> list[dict]:
    """Normalize document sections.  Handles JSON string, missing fields, type coercion."""
    raw = normalize_list(sections)
    result = []
    for item in raw:
        if isinstance(item, str):
            # Model returned a string instead of an object
            result.append({"heading": "", "body": item})
            continue
        if not isinstance(item, dict):
            continue
        section = {
            "heading": normalize_str(item.get("heading", "")),
            "body": normalize_str(item.get("body", "")),
            "level": normalize_int(item.get("level", 1), default=1),
            "image_url": normalize_str(item.get("image_url", "")),
            "image_caption": normalize_str(item.get("image_caption", "")),
        }
        # Clamp level to 1-4
        section["level"] = max(1, min(4, section["level"]))
        result.append(section)
    return result


# ---------------------------------------------------------------------------
# Slide normalization
# ---------------------------------------------------------------------------


_VALID_LAYOUTS = frozenset({"title", "content", "two_column", "blank"})


def normalize_slides(slides: object) -> list[dict]:
    """Normalize presentation slides.  Handles list/string body, missing fields."""
    raw = normalize_list(slides)
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        layout = normalize_str(item.get("layout", "content"))
        if layout not in _VALID_LAYOUTS:
            layout = "content"
        additional = item.get("additional_images") or []
        if isinstance(additional, list):
            additional = [normalize_str(u) for u in additional if u]
        else:
            additional = []
        slide = {
            "layout": layout,
            "title": normalize_str(item.get("title", "")),
            "body": normalize_str(item.get("body", "")),
            "notes": normalize_str(item.get("notes", "")),
            "image_url": normalize_str(item.get("image_url", "")),
            "additional_images": additional[:3],  # picker hard cap
        }
        result.append(slide)
    return result


# ---------------------------------------------------------------------------
# Sheet normalization
# ---------------------------------------------------------------------------


def normalize_sheets(sheets: object) -> list[dict]:
    """Normalize spreadsheet sheets.  Handles missing fields, type coercion."""
    raw = normalize_list(sheets)
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        headers = normalize_list(item.get("headers", []))
        # Ensure all headers are strings
        headers = [normalize_str(h) for h in headers]

        rows = normalize_list(item.get("rows", []))
        # Ensure each row is a list (not a dict or string)
        clean_rows = []
        for row in rows:
            if isinstance(row, list):
                clean_rows.append(row)
            elif isinstance(row, dict):
                # Convert dict row to list in header order
                clean_rows.append([row.get(h, "") for h in headers])
            elif isinstance(row, str):
                # Try to parse as JSON array
                parsed = normalize_list(row)
                clean_rows.append(parsed if parsed else [row])
            else:
                clean_rows.append([row])

        # Reconcile row width to the header count so the grid editor and the
        # XLSX renderer never misalign. Short rows (a common local-model
        # failure) are padded with empty cells; longer rows are left intact
        # so no model-supplied data is silently dropped.
        if headers:
            n = len(headers)
            clean_rows = [
                (row + [""] * (n - len(row))) if len(row) < n else row
                for row in clean_rows
            ]

        sheet = {
            "name": normalize_str(item.get("name", "Sheet"))[:31],
            "headers": headers,
            "rows": clean_rows,
            "freeze_header": normalize_bool(item.get("freeze_header", True), default=True),
            "column_formats": item.get("column_formats") or {},
            "summary_row": normalize_str(item.get("summary_row", "none")) or "none",
        }
        result.append(sheet)
    return result


# ---------------------------------------------------------------------------
# Chart data normalization
# ---------------------------------------------------------------------------


def normalize_chart_labels(labels: object) -> list[str]:
    """Normalize chart labels to list of strings."""
    raw = normalize_list(labels)
    return [normalize_str(label) for label in raw]


# Series-value keys we accept besides the documented ``values``. Models are
# trained overwhelmingly on Chart.js, whose series are ``{label, data}``, so
# ``data`` arrives constantly. Before this, ``item.get("values")`` missed it and
# the series became an EMPTY list — no error, just a blank chart, which is the
# worst failure mode available: the model believes it succeeded and the user
# gets an empty picture. Aliases are cheaper than a retry and cheaper than a
# schema the model will ignore anyway.
_DATASET_VALUE_KEYS: tuple[str, ...] = ("values", "data", "y", "series", "points")
_DATASET_NAME_KEYS: tuple[str, ...] = ("name", "label", "title", "series_name")


def normalize_chart_datasets(datasets: object) -> list[dict]:
    """Normalize chart datasets into ``[{name, values}]``.

    Deliberately permissive about the SHAPE the caller used, because the caller
    is usually an LLM and every rejected shape costs a round trip (or, worse,
    silently renders an empty chart). Accepted:

    - the documented ``[{"name": ..., "values": [...]}]``
    - Chart.js style ``[{"label": ..., "data": [...]}]`` (see
      ``_DATASET_VALUE_KEYS`` / ``_DATASET_NAME_KEYS``)
    - a bare list of numbers — ``[1, 2, 3]`` — as one unnamed series
    - a list of lists — ``[[1, 2], [3, 4]]`` — as one series each
    - a mapping of name to values — ``{"Q1": [1, 2]}``

    Values are always coerced to floats. A series carrying no usable numbers is
    DROPPED rather than kept as an empty list, so the caller's
    "labels and datasets are required" check can actually fire and the model
    gets a real error instead of a blank canvas.
    """
    # A dict of series → treat keys as names. Do this before normalize_list,
    # which would otherwise reduce a mapping to a single opaque item.
    if isinstance(datasets, dict) and datasets:
        items: list[object] = [
            {"name": key, "values": value} for key, value in datasets.items()
        ]
    else:
        items = normalize_list(datasets)

    # A bare list of scalars is ONE series, not N series of one point each.
    if items and all(isinstance(v, (int, float, str)) for v in items):
        nums = [normalize_number(v) for v in items]
        return [{"name": "Series 1", "values": nums}] if nums else []

    result: list[dict] = []
    for i, item in enumerate(items):
        if isinstance(item, dict):
            raw_values: object = []
            for key in _DATASET_VALUE_KEYS:
                if item.get(key) not in (None, ""):
                    raw_values = item[key]
                    break
            name = ""
            for key in _DATASET_NAME_KEYS:
                name = normalize_str(item.get(key, ""))
                if name:
                    break
        elif isinstance(item, (list, tuple)):
            raw_values, name = list(item), ""
        else:
            continue

        clean_values = [normalize_number(v) for v in normalize_list(raw_values)]
        if not clean_values:
            # Nothing plottable — dropping it lets the caller report a real
            # error instead of rendering an empty series.
            continue
        result.append({
            "name": name or f"Series {i + 1}",
            "values": clean_values,
        })
    return result
