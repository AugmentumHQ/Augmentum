"""Numeric-watch logic: extraction, conditions, quarantine, hysteresis.

Pure functions only — no DB, no network, no runtime. The standing-task
kind runners (url_watch with a condition, metric_watch) orchestrate;
every judgment about a number lives here so it can be tested as plain
sequences (see tests/test_watch_metrics.py).

Design provenance (scheduled-requests spec §7.4, research 2026-06-11):
  * Keepa — append-only change-events, scaled integers, missing ≠ zero
  * changedetection.io — JSON-LD-first extraction ladder, ambiguity is
    an error (MoreThanOnePriceFound), metadata can lie
  * Prometheus ``for:`` / UptimeRobot N-of-M — a threshold crossing
    needs consecutive confirmation before it fires
  * Walmart MoatPlus / ScrapeHero — implausible jumps are quarantined,
    not alerted ("price dropped 99%!" is almost always a parse glitch)

The iron rule (P7): code does the comparison. The LLM may *locate* a
number (Phase 3+ ladder rung, quote-verified); it never decides whether
the condition is met.
"""

from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ─── Extraction ─────────────────────────────────────────────────────────

# Currency/number shaped token, optionally symbol-prefixed: $1,299.99 /
# 85.2 / €45. Group 1 = the numeric text.
_NUMBER_RE = re.compile(
    r"[$€£¥]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
)

# schema.org price-ish keys, checked case-insensitively at any JSON-LD
# nesting depth. Ordered: an explicit offer price beats a generic value.
_JSONLD_PRICE_KEYS = ("price", "lowprice", "highprice", "value")


@dataclass(slots=True)
class Extraction:
    """One extraction attempt's outcome."""
    value: float | None          # None = nothing extractable
    method: str = ""             # 'json-ld' | 'pattern' | ''
    evidence: str = ""           # verbatim snippet the value came from
    ambiguous: bool = False      # >1 distinct candidate — error, not a guess
    candidates: list[float] = field(default_factory=list)


def _walk_jsonld(node: Any, found: list[tuple[float, str]]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            lk = str(k).lower()
            if lk in _JSONLD_PRICE_KEYS and isinstance(v, int | float | str):
                with contextlib.suppress(TypeError, ValueError):
                    found.append((float(str(v).replace(",", "")), f"{k}: {v}"))
            else:
                _walk_jsonld(v, found)
    elif isinstance(node, list):
        for item in node:
            _walk_jsonld(item, found)


def extract_from_jsonld(html: str) -> Extraction:
    """Rung 1: schema.org JSON-LD blocks. Regex pre-extraction of the
    script bodies (no DOM parse), then a key walk. Ambiguity across
    *distinct* values is reported, never resolved by guessing."""
    found: list[tuple[float, str]] = []
    for m in re.finditer(
        r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>',
        html or "", re.DOTALL | re.IGNORECASE,
    ):
        try:
            _walk_jsonld(json.loads(m.group(1).strip()), found)
        except (json.JSONDecodeError, ValueError):
            continue
    if not found:
        return Extraction(value=None)
    distinct = sorted({v for v, _ in found})
    if len(distinct) > 1:
        return Extraction(
            value=None, method="json-ld", ambiguous=True,
            candidates=distinct,
        )
    return Extraction(
        value=distinct[0], method="json-ld", evidence=found[0][1][:200],
    )


def extract_near_hint(text: str, hint: str) -> Extraction:
    """Rung 2: first number within the window around ``hint`` — the
    creation probe records where it saw the baseline (a label like
    'Price' or a snippet) so later fires look in the same place."""
    if not text:
        return Extraction(value=None)
    window = text
    if hint:
        idx = text.lower().find(hint.lower())
        if idx >= 0:
            window = text[max(0, idx - 40): idx + len(hint) + 120]
    m = _NUMBER_RE.search(window)
    if not m:
        return Extraction(value=None)
    try:
        value = float(m.group(1).replace(",", ""))
    except (TypeError, ValueError):
        return Extraction(value=None)
    snippet = window[max(0, m.start() - 30): m.end() + 30].strip()
    return Extraction(value=value, method="pattern", evidence=snippet[:200])


def extract_value(html_or_text: str, *, hint: str = "",
                  pinned_method: str = "") -> Extraction:
    """The ladder: json-ld → pattern-near-hint. A pinned method (set
    after the first success — changedetection.io's price-data-follower
    promotion) is tried first; falling off the pin is reported via the
    returned method so callers can surface extraction drift."""
    order = ["json-ld", "pattern"]
    if pinned_method in order:
        order.remove(pinned_method)
        order.insert(0, pinned_method)
    for method in order:
        ext = (
            extract_from_jsonld(html_or_text) if method == "json-ld"
            else extract_near_hint(html_or_text, hint)
        )
        if ext.ambiguous or ext.value is not None:
            return ext
    return Extraction(value=None)


# ─── Condition ──────────────────────────────────────────────────────────

_OPS = {
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
}


def condition_met(value: float, condition: dict[str, Any] | None) -> bool | None:
    """Evaluate a compiled condition in code. None = no/invalid condition
    (caller treats the reading as informational, not a trigger)."""
    if not isinstance(condition, dict):
        return None
    op = _OPS.get(str(condition.get("op") or ""))
    if op is None:
        return None
    try:
        return bool(op(float(value), float(condition.get("value"))))
    except (TypeError, ValueError):
        return None


# ─── Quarantine + confirmation + hysteresis state machine ───────────────
#
# State travels in the task's params JSON (the engine already persists
# params mutations per fire):
#   {
#     "last_accepted": float | None,   # last sane reading
#     "pending_value": float | None,   # candidate new level under confirm
#     "pending_count": int,            # consecutive agreeing readings
#     "cond_fired": bool,              # condition currently in fired state
#   }

DEFAULT_QUARANTINE_PCT = 60.0    # |Δ%| beyond this vs last accepted → quarantine
DEFAULT_CONFIRM_READINGS = 2     # consecutive readings to accept / to fire
# Two quarantined readings agreeing within this tolerance count as the
# same new level (a price isn't byte-stable across fetches: cents, FX).
_AGREE_TOLERANCE_PCT = 2.0


@dataclass(slots=True)
class Verdict:
    """One reading's classification + what (if anything) to tell the user."""
    status: str                  # 'ok' | 'quarantined' | 'missing'
    fire: bool = False           # condition newly satisfied (confirmed)
    note: str = ""               # honest one-liner for the run row
    state: dict[str, Any] = field(default_factory=dict)  # → params


def classify_reading(
    value: float | None,
    *,
    state: dict[str, Any] | None,
    condition: dict[str, Any] | None = None,
    quarantine_pct: float = DEFAULT_QUARANTINE_PCT,
    confirm_readings: int = DEFAULT_CONFIRM_READINGS,
) -> Verdict:
    """Classify one reading against the running state.

    Sequence semantics (tested as plain lists):
      * first sane reading → accepted baseline, never fires
      * |Δ%| > quarantine_pct vs last accepted → quarantined; the new
        level is accepted only after ``confirm_readings`` consecutive
        agreeing readings (glitch readings die alone)
      * condition crossings fire only after ``confirm_readings``
        consecutive satisfying readings, and re-arm only after a
        reading clears the condition (no ping-pong at the boundary)
      * value None → 'missing'; resets nothing (absence is not data)
    """
    s = dict(state or {})
    last = s.get("last_accepted")
    confirm_readings = max(1, int(confirm_readings))

    if value is None:
        return Verdict(status="missing",
                       note="couldn't read a value this time", state=s)

    # ── Quarantine gate ──
    if last is not None and float(last) != 0 and quarantine_pct > 0:
        delta_pct = abs((value - float(last)) / float(last)) * 100.0
        if delta_pct > quarantine_pct:
            pend = s.get("pending_value")
            agrees = (
                pend is not None and float(pend) != 0
                and abs((value - float(pend)) / float(pend)) * 100.0
                <= _AGREE_TOLERANCE_PCT
            )
            count = int(s.get("pending_count") or 0) + 1 if agrees else 1
            if count >= confirm_readings:
                # Confirmed new level — fall through to acceptance below.
                s.pop("pending_value", None)
                s.pop("pending_count", None)
            else:
                s["pending_value"] = value
                s["pending_count"] = count
                return Verdict(
                    status="quarantined",
                    note=(
                        f"read {value:g} — {delta_pct:.0f}% off the last "
                        f"accepted {float(last):g}; holding for "
                        f"confirmation ({count}/{confirm_readings})"
                    ),
                    state=s,
                )
        else:
            s.pop("pending_value", None)
            s.pop("pending_count", None)

    # ── Accept ──
    s["last_accepted"] = value

    # ── Condition with confirmation + re-arm hysteresis ──
    met = condition_met(value, condition)
    if met is None:
        s.pop("cond_count", None)
        return Verdict(status="ok", note=f"read {value:g}", state=s)
    if met:
        count = int(s.get("cond_count") or 0) + 1
        s["cond_count"] = count
        already = bool(s.get("cond_fired"))
        if not already and count >= confirm_readings:
            s["cond_fired"] = True
            return Verdict(
                status="ok", fire=True,
                note=f"condition met: read {value:g}", state=s,
            )
        return Verdict(
            status="ok",
            note=(
                f"read {value:g} (condition "
                f"{'still met' if already else f'met {count}/{confirm_readings}, confirming'})"
            ),
            state=s,
        )
    # Condition not met → re-arm.
    s["cond_count"] = 0
    s["cond_fired"] = False
    return Verdict(status="ok", note=f"read {value:g}", state=s)


def to_scaled_int(value: float, scale: int = 100) -> int:
    """Storage form: cents-style scaled integer (Keepa). Round, don't
    truncate — 19.999 is 2000 cents, not 1999."""
    return round(float(value) * int(scale))
