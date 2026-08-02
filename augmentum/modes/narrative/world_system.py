"""World-system manifest — card-declared game mechanics for narrative mode.

Design: docs/superpowers/specs/2026-07-15-world-system-manifest-design.md

The card declares (``extensions.world_system``), the engine provides,
absence means invisible. This module is strictly generic: it knows
trackers, tables, dice, and sheets — never a specific world. The bundled
Cyraeth card is a definition file, not a code path.

Authority model (spec D1): the MODEL moves trackers only via
``world.track.shift``; this module validates (band trackers move one band
per call unless force-flagged, counters take deltas) and logs every write
with provenance. USER corrections (drawer) are sticky: the model cannot
overwrite a user-set value for ``USER_LOCK_TURNS`` turns.

The sheet and the [World State] prompt block render ONLY from the store —
never from prose — so narration can drift but the numbers cannot
(peer-research finding: fake tracking breaks trust worse than absence).
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

SPEC_VERSION = "augmentum_world_v1"
MODULES = ("trackers", "tables", "dice", "locations", "sheet")
USER_LOCK_TURNS = 2

# Bounded per-tracker history so world_state stays small in the DB blob.
_HISTORY_CAP = 40


# ---------------------------------------------------------------------------
# Manifest


@dataclass
class TrackerDef:
    id: str
    label: str
    kind: str = "band"           # band | counter | flag | scalar
    scope: str = "character"     # character | party | world
    bands: list[str] = field(default_factory=list)
    start: Any = None
    visible: bool = True
    reveal_on: str = ""          # "" | "exposed" (hidden until != start)
    min: float | None = None     # scalar/counter bounds (optional)
    max: float | None = None


@dataclass
class TableDef:
    id: str
    label: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)


@dataclass
class DiceDef:
    system: str = "d20"
    player_roller: bool = True


@dataclass
class SheetSection:
    id: str
    label: str = ""
    trackers: list[str] = field(default_factory=list)
    when: str = ""               # "" | "revealed"


@dataclass
class WorldManifest:
    name: str = ""
    modules: list[str] = field(default_factory=list)
    trackers: list[TrackerDef] = field(default_factory=list)
    tables: list[TableDef] = field(default_factory=list)
    dice: DiceDef | None = None
    sheet_sections: list[SheetSection] = field(default_factory=list)
    sheet_command: str = "/status"

    def has(self, module: str) -> bool:
        return module in self.modules

    def tracker(self, tracker_id: str) -> TrackerDef | None:
        for t in self.trackers:
            if t.id == tracker_id:
                return t
        return None

    def table(self, table_id: str) -> TableDef | None:
        for t in self.tables:
            if t.id == table_id:
                return t
        return None


def parse_manifest(card_raw_data: dict | None) -> WorldManifest | None:
    """Parse ``extensions.world_system`` from a card's raw data.

    Returns None when absent or unusable — absence is the designed
    invisible state, so parse failures log a warning and return None
    rather than raising (a malformed manifest must not break chat).
    """
    if not card_raw_data:
        return None
    ext = card_raw_data.get("extensions") or {}
    ws = ext.get("world_system")
    if not isinstance(ws, dict):
        return None
    if ws.get("spec") != SPEC_VERSION:
        log.warning(
            "world_manifest_unknown_spec", spec=ws.get("spec"),
            expected=SPEC_VERSION,
        )
        return None
    try:
        modules = [m for m in (ws.get("modules") or []) if m in MODULES]
        manifest = WorldManifest(
            name=str(ws.get("name") or ""), modules=modules,
        )
        for t in ws.get("trackers") or []:
            kind = t.get("kind", "band")
            bands = [str(b) for b in (t.get("bands") or [])]
            if kind == "band" and len(bands) < 2:
                log.warning("world_tracker_bad_bands", tracker=t.get("id"))
                continue
            start = t.get("start")
            if kind == "band" and start not in bands:
                start = bands[0]
            if kind == "counter" and not isinstance(start, int | float):
                start = 0
            if kind == "flag":
                start = bool(start)
            manifest.trackers.append(TrackerDef(
                id=str(t.get("id") or "").strip(),
                label=str(t.get("label") or t.get("id") or ""),
                kind=kind, scope=t.get("scope", "character"),
                bands=bands, start=start,
                visible=bool(t.get("visible", True)),
                reveal_on=str(t.get("reveal_on") or ""),
                min=t.get("min"), max=t.get("max"),
            ))
        manifest.trackers = [t for t in manifest.trackers if t.id]
        for tb in ws.get("tables") or []:
            cols = [str(c) for c in (tb.get("columns") or [])]
            rows = [list(r) for r in (tb.get("rows") or []) if isinstance(r, list)]
            if tb.get("id") and cols:
                manifest.tables.append(TableDef(
                    id=str(tb["id"]), label=str(tb.get("label") or tb["id"]),
                    columns=cols, rows=rows,
                ))
        d = ws.get("dice")
        if isinstance(d, dict):
            manifest.dice = DiceDef(
                system=str(d.get("system") or "d20"),
                player_roller=bool(d.get("player_roller", True)),
            )
        sh = ws.get("sheet")
        if isinstance(sh, dict):
            manifest.sheet_command = str(sh.get("command") or "/status")
            for s in sh.get("sections") or []:
                if s.get("id"):
                    manifest.sheet_sections.append(SheetSection(
                        id=str(s["id"]), label=str(s.get("label") or s["id"]),
                        trackers=[str(x) for x in (s.get("trackers") or [])],
                        when=str(s.get("when") or ""),
                    ))
        # Prune declared-but-empty modules so UI/tool gating stays honest.
        if not manifest.trackers:
            manifest.modules = [m for m in manifest.modules if m != "trackers"]
        if not manifest.tables:
            manifest.modules = [m for m in manifest.modules if m != "tables"]
        if manifest.dice is None:
            manifest.modules = [m for m in manifest.modules if m != "dice"]
        if not manifest.modules:
            return None
        return manifest
    except Exception:
        log.warning("world_manifest_parse_failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Tracker store — operates on the state dict persisted in
# NarrativeSessionState.world_state (migration 312). Shape:
# {
#   "values":  {"<owner>|<tracker_id>": <value>},
#   "history": {"<owner>|<tracker_id>": [{"turn","from","to","by","reason"}]},
#   "locks":   {"<owner>|<tracker_id>": <turn locked until (exclusive)>}
# }
# owner = character/party name for character|party scope, "world" for world.


def _key(tracker_id: str, owner: str) -> str:
    # "" = the player character ("pc"); world-scope trackers also key at
    # the default owner — tracker ids are unique across scopes so the
    # namespaces can't collide.
    return f"{owner or 'pc'}|{tracker_id}"


class WorldStore:
    """Validated read/write layer over the persisted world-state dict."""

    def __init__(self, manifest: WorldManifest, state_dict: dict) -> None:
        self.manifest = manifest
        self.data = state_dict
        self.data.setdefault("values", {})
        self.data.setdefault("history", {})
        self.data.setdefault("locks", {})

    # -- reads ------------------------------------------------------------

    def get(self, tracker_id: str, owner: str = "") -> Any:
        t = self.manifest.tracker(tracker_id)
        if t is None:
            return None
        return self.data["values"].get(_key(tracker_id, owner), t.start)

    def owners(self) -> list[str]:
        seen: list[str] = []
        for k in self.data["values"]:
            owner = k.split("|", 1)[0]
            if owner not in seen:
                seen.append(owner)
        return seen

    def revealed(self, t: TrackerDef, owner: str = "") -> bool:
        if t.reveal_on != "exposed":
            return True
        return self.get(t.id, owner) != t.start

    # -- writes -----------------------------------------------------------

    def shift(
        self, tracker_id: str, *, owner: str = "", turn: int = 0,
        to: Any = None, delta: float | None = None,
        by: str = "model", reason: str = "",
    ) -> tuple[bool, str, Any]:
        """Validated write. Returns (ok, message, new_value)."""
        t = self.manifest.tracker(tracker_id)
        if t is None:
            return False, f"Unknown tracker '{tracker_id}'.", None
        key = _key(tracker_id, owner)
        cur = self.get(tracker_id, owner)

        lock_until = self.data["locks"].get(key, 0)
        if by == "model" and turn < lock_until:
            return False, (
                f"'{t.label}' was set by the user recently and is locked "
                f"(current value: {cur}). Narrate from the current value."
            ), cur

        force = "force:" in (reason or "")
        if t.kind == "band":
            if to not in t.bands:
                return False, (
                    f"'{t.label}' bands are {t.bands}; '{to}' is not one."
                ), cur
            if by == "model" and not force:
                step = abs(t.bands.index(to) - t.bands.index(cur))
                if step > 1:
                    return False, (
                        f"'{t.label}' can move one band per shift "
                        f"(currently '{cur}'). Use reason='force: <major "
                        f"story event>' for larger jumps."
                    ), cur
            new = to
        elif t.kind == "counter":
            if delta is None:
                if not isinstance(to, int | float):
                    return False, f"'{t.label}' takes a numeric delta.", cur
                delta = float(to) - float(cur)
            new = float(cur) + float(delta)
            if t.min is not None:
                new = max(float(t.min), new)
            if t.max is not None:
                new = min(float(t.max), new)
            if float(new).is_integer():
                new = int(new)
        elif t.kind == "flag":
            new = bool(to)
        else:  # scalar
            try:
                new = float(to)
            except (TypeError, ValueError):
                return False, f"'{t.label}' takes a number.", cur
            if t.min is not None:
                new = max(float(t.min), new)
            if t.max is not None:
                new = min(float(t.max), new)

        self.data["values"][key] = new
        hist = self.data["history"].setdefault(key, [])
        hist.append({
            "turn": turn, "from": cur, "to": new, "by": by,
            "reason": (reason or "")[:200],
        })
        del hist[:-_HISTORY_CAP]
        if by == "user":
            self.data["locks"][key] = turn + USER_LOCK_TURNS
        log.info(
            "world_tracker_shift", tracker=tracker_id, owner=owner or "world",
            from_value=cur, to_value=new, by=by, turn=turn,
        )
        return True, f"{t.label}: {cur} -> {new}", new

    # -- prompt/state rendering --------------------------------------------

    def state_block(self, *, primary_owner: str = "") -> str:
        """Compact one-line-per-tracker block for the per-turn injection.

        Hidden (reveal_on) trackers ARE included once revealed; invisible
        trackers (visible=False) are engine-only and never enter the
        prompt. ~30-60 tokens for a typical manifest.
        """
        if not self.manifest.has("trackers"):
            return ""
        lines: list[str] = []
        owners = self.owners() or ["pc"]
        if "pc" not in owners:
            owners.insert(0, "pc")
        for t in self.manifest.trackers:
            if not t.visible:
                continue
            if t.scope == "world":
                if self.revealed(t):
                    lines.append(f"{t.label}: {self.get(t.id)}")
                continue
            parts = []
            for owner in owners:
                o = "" if owner == "pc" else owner
                if self.revealed(t, o):
                    label = "PC" if owner == "pc" else owner
                    parts.append(f"{label}: {self.get(t.id, o)}")
            if parts:
                lines.append(f"{t.label} — " + "; ".join(parts))
        if not lines:
            return ""
        return (
            "[World State — engine-tracked, authoritative. Narrate from "
            "these values; change them only via world.track.shift]\n"
            + "\n".join(lines)
        )

    def sheet(self, *, owner: str = "") -> dict:
        """Structured sheet render (UI draws it; also textified for chat)."""
        sections: list[dict] = []
        for s in self.manifest.sheet_sections:
            rows: list[dict] = []
            for tid in s.trackers:
                t = self.manifest.tracker(tid)
                if t is None or not t.visible:
                    continue
                t_owner = "" if t.scope == "world" else owner
                if s.when == "revealed" and not self.revealed(t, t_owner):
                    continue
                rows.append({
                    "id": t.id, "label": t.label, "kind": t.kind,
                    "value": self.get(t.id, t_owner),
                    "bands": t.bands or None,
                })
            if rows:
                sections.append({"id": s.id, "label": s.label, "rows": rows})
        return {
            "world": self.manifest.name, "owner": owner or "PC",
            "sections": sections,
        }


def sheet_text(sheet: dict) -> str:
    """Plain-text sheet for the transcript event card fallback."""
    lines = [f"— {sheet['world'] or 'World'} · {sheet['owner']} —"]
    for s in sheet["sections"]:
        lines.append(f"[{s['label']}]")
        for r in s["rows"]:
            lines.append(f"  {r['label']}: {r['value']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dice

_ROLL_RE = re.compile(
    r"^\s*(?P<n>\d{0,2})d(?P<sides>\d{1,4})\s*(?:(?P<sign>[+-])\s*(?P<mod>\d{1,3}))?\s*$",
    re.IGNORECASE,
)


def roll_dice(expression: str, *, rng: random.Random | None = None) -> dict | None:
    """Evaluate an NdM(+/-K) expression. Returns dict or None if invalid."""
    m = _ROLL_RE.match(expression or "")
    if not m:
        return None
    n = int(m.group("n") or 1)
    sides = int(m.group("sides"))
    if not (1 <= n <= 20 and 2 <= sides <= 1000):
        return None
    mod = int(m.group("mod") or 0)
    if m.group("sign") == "-":
        mod = -mod
    r = rng or random
    rolls = [r.randint(1, sides) for _ in range(n)]
    return {
        "expression": expression.strip(), "rolls": rolls, "modifier": mod,
        "total": sum(rolls) + mod,
    }


# ---------------------------------------------------------------------------
# Table lookup


def lookup_table(manifest: WorldManifest, table_id: str, query: str = "") -> str:
    """Return matching rows (or the whole table when small) as text.

    Whole rows only — selection, never truncation of a row's content.
    """
    t = manifest.table(table_id)
    if t is None:
        names = ", ".join(tb.id for tb in manifest.tables) or "(none)"
        return f"Unknown table '{table_id}'. Available: {names}"
    q = (query or "").strip().lower()
    rows = t.rows
    if q:
        rows = [r for r in t.rows if any(q in str(c).lower() for c in r)]
        if not rows:
            return f"No rows in '{t.label}' match '{query}'."
    if len(rows) > 20:
        rows = rows[:20]
    header = " | ".join(t.columns)
    body = "\n".join(" | ".join(str(c) for c in r) for r in rows)
    return f"{t.label}\n{header}\n{body}"


# ---------------------------------------------------------------------------
# Sheet command detection (tier-1 intercept, never reaches the model)

_SHEET_COMMANDS = {
    "/s": "condition", "/status": "", "/inv": "inventory",
    "/inventory": "inventory", "/loc": "location", "/location": "location",
}


def match_sheet_command(text: str) -> str | None:
    """Return the section filter ('' = full sheet) or None if not a command."""
    cmd = (text or "").strip().lower()
    return _SHEET_COMMANDS.get(cmd)


def serialize_world_state(data: dict) -> str:
    try:
        return json.dumps(data or {})
    except (TypeError, ValueError):
        log.warning("world_state_serialize_failed", exc_info=True)
        return "{}"


def deserialize_world_state(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except (ValueError, TypeError):
        log.warning("world_state_deserialize_failed", exc_info=True)
        return {}


# ---------------------------------------------------------------------------
# Drift reconciliation (Wave 2, spec "Adopted from peers" #1)
#
# Peer finding (Multihog): a second-pass model catches HP/loot changes the
# narrator described but never applied. We keep D1 (no silent writes): the
# extraction only produces SUGGESTIONS the user taps to accept.

_RECONCILE_PROMPT = """You audit a story turn against a game-state tracker.

TRACKERS (id | kind | current value | allowed values):
{trackers}

STORY TEXT (the narrator's latest turn):
---
{text}
---

List ONLY tracker changes that the story text CLEARLY establishes and the
current values do not yet reflect. Physical wounds, spending/earning money,
eating/drinking, exhaustion, rank changes. Do NOT infer moods or guesses.
Band trackers: pick the nearest allowed value. Counters: give the delta.

Reply with a JSON array (empty array if nothing changed):
[{{"tracker": "<id>", "to": "<band value>", "delta": <number or null>,
   "reason": "<short quote or paraphrase from the text>"}}]
JSON only, no commentary."""


async def extract_drift_suggestions(
    backend, model: str, store: WorldStore, response_text: str,
) -> list[dict]:
    """Ask the utility model for narration/store drift. Returns suggestion
    dicts (validated against the manifest); [] on any failure — this is a
    best-effort background pass, never a turn blocker."""
    manifest = store.manifest
    if not manifest.has("trackers") or not (response_text or "").strip():
        return []
    lines = []
    for t in manifest.trackers:
        if not t.visible:
            continue
        allowed = "/".join(t.bands) if t.bands else t.kind
        lines.append(f"{t.id} | {t.kind} | {store.get(t.id)} | {allowed}")
    if not lines:
        return []
    prompt = _RECONCILE_PROMPT.format(
        trackers="\n".join(lines), text=response_text[:6000],
    )
    try:
        from augmentum.models.base import InternalChatRequest, Message
        request = InternalChatRequest(
            model=model,
            messages=[Message(role="user", content=prompt)],
            max_tokens=400,
            temperature=0.1,
        )
        response = await backend.chat(request)
        raw = (response.message.content or "") if response.message else ""
    except Exception:
        log.warning("world_reconcile_call_failed", exc_info=True)
        return []
    # Tolerant JSON pull — models love to wrap arrays in prose/fences.
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except (ValueError, TypeError):
        return []
    out: list[dict] = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        t = manifest.tracker(str(it.get("tracker") or ""))
        if t is None:
            continue
        to = it.get("to")
        delta = it.get("delta")
        if t.kind == "band":
            if to not in t.bands or to == store.get(t.id):
                continue
            delta = None
        elif t.kind == "counter":
            if not isinstance(delta, int | float) or not delta:
                continue
            to = None
        else:
            continue  # flags/scalars: too error-prone for auto-suggestion
        out.append({
            "tracker": t.id, "label": t.label, "to": to, "delta": delta,
            "current": store.get(t.id),
            "reason": str(it.get("reason") or "")[:160],
        })
        if len(out) >= 4:
            break
    return out
