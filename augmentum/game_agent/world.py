"""World blackboard: provenance-ranked facts + a measurable goal stack.

One shared state every lane reads and writes, instead of prose passed
between prompts. Two disciplines give it power:

* **Provenance rank** — every fact knows where it came from. RAM probes
  outrank the scene narrator, which outranks model inference; a lower-
  rank writer can never overwrite a fresher higher-rank fact, so pixel
  guesses structurally cannot override memory truth.
* **Goals as data** — FINAL / MEDIUM / SHORT goals carry an optional
  machine-checkable progress ``metric`` over the fact store (map id
  changed, badge bit set, party_count >= 1). Progress becomes a
  measurement, not a vibe — the stall watchdog and the correct-move
  metric both fall out of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Source = Literal["ram", "scene", "model"]
_RANK: dict[str, int] = {"ram": 3, "scene": 2, "model": 1}

Horizon = Literal["final", "medium", "short"]


@dataclass
class Fact:
    value: Any
    source: Source
    t_ms: int


@dataclass
class Goal:
    """One horizon of the goal stack.

    ``metric`` (optional): {"probe": <fact name>, "op": "eq|ne|ge|le",
    "value": <target>} — checked against the fact store. A goal without
    a metric is judged only by the planner.
    """

    text: str
    metric: dict[str, Any] | None = None
    set_t_ms: int = 0
    done: bool = False

    def check(self, world: WorldState) -> bool:
        if self.metric is None:
            return False
        fact = world.facts.get(str(self.metric.get("probe") or ""))
        if fact is None:
            return False
        op = str(self.metric.get("op") or "eq")
        target = self.metric.get("value")
        v = fact.value
        try:
            if op == "eq":
                return v == target
            if op == "ne":
                return v != target
            if op == "ge":
                return float(v) >= float(target)
            if op == "le":
                return float(v) <= float(target)
        except (TypeError, ValueError):
            return False
        return False


class WorldState:
    """The blackboard. Cheap, synchronous, single-event-loop."""

    def __init__(self) -> None:
        self.facts: dict[str, Fact] = {}
        self.goals: dict[str, Goal] = {}
        # Last time any fact VALUE changed — the stall watchdog's pulse.
        self.last_change_ms: int = 0
        # Novelty tracker: progress = novelty across tracked dimensions
        # (tiles, screens, dialogue lines, goal completions) — the
        # collapsed premise behind RL exploration bonuses. Per-dimension
        # visit counts + the last key seen (so sitting on one tile does
        # not inflate its count), plus one shared "when did anything
        # NOVEL last happen" pulse. Unlike ``last_change_ms``, menu
        # churn cannot reset this: a menu is only novel once.
        self._novelty_counts: dict[str, dict[Any, int]] = {}
        self._novelty_last: dict[str, Any] = {}
        self.last_novel_ms: int = 0

    # ── facts ─────────────────────────────────────────────────────

    def update(self, name: str, value: Any, *, source: Source, t_ms: int) -> bool:
        """Write a fact. Returns True when the value actually changed.

        A lower-rank source cannot overwrite a different value written
        by a higher-rank source within the freshness window (5s) — the
        narrator can't clobber RAM truth, but CAN fill fields RAM never
        reports or refresh stale ones.
        """

        prev = self.facts.get(name)
        if prev is not None and _RANK[source] < _RANK[prev.source] and (
            t_ms - prev.t_ms < 5000
        ):
            return False
        changed = prev is None or prev.value != value
        self.facts[name] = Fact(value=value, source=source, t_ms=t_ms)
        if changed:
            self.last_change_ms = t_ms
        return changed

    def update_probes(self, probes: dict[str, Any], *, t_ms: int) -> None:
        for k, v in probes.items():
            self.update(k, v, source="ram", t_ms=t_ms)

    def update_many(
        self, probes: dict[str, Any], *, source: Source, t_ms: int,
    ) -> dict[str, Any]:
        """Rank-aware bulk write; returns the subset that was ACCEPTED.

        Accepted = the fact now holds our value from our source at our
        timestamp (i.e. the provenance gate didn't reject the write).
        Callers that mirror probe values into rank-blind stores (the
        prompt overlay, lore, novelty) must only mirror this subset —
        otherwise a lower-rank writer sneaks past the blackboard's
        provenance discipline via the side doors.
        """

        accepted: dict[str, Any] = {}
        for k, v in probes.items():
            self.update(k, v, source=source, t_ms=t_ms)
            fact = self.facts.get(k)
            if (
                fact is not None
                and fact.t_ms == t_ms
                and fact.source == source
                and fact.value == v
            ):
                accepted[k] = v
        return accepted

    # ── goals ─────────────────────────────────────────────────────

    def set_goal(
        self, horizon: Horizon, text: str,
        metric: dict[str, Any] | None = None, *, t_ms: int,
    ) -> None:
        text = str(text)[:200]
        cur = self.goals.get(horizon)
        if cur is not None and cur.text == text and cur.metric == metric:
            return  # unchanged — keep its age so stall detection is honest
        self.goals[horizon] = Goal(text=text, metric=metric, set_t_ms=t_ms)

    def apply_goal_update(self, update: dict[str, Any], *, t_ms: int) -> None:
        """Consume a plan's ``goal_update``: per-horizon {"text", "metric"?}."""

        if not isinstance(update, dict):
            return
        for horizon in ("final", "medium", "short"):
            raw = update.get(horizon)
            if isinstance(raw, str) and raw.strip():
                self.set_goal(horizon, raw, None, t_ms=t_ms)
            elif isinstance(raw, dict) and str(raw.get("text") or "").strip():
                metric = raw.get("metric")
                self.set_goal(
                    horizon, str(raw["text"]),
                    metric if isinstance(metric, dict) else None, t_ms=t_ms,
                )

    def check_goals(self) -> list[str]:
        """Mark newly-completed metric goals; return their horizons."""

        completed = []
        for horizon, goal in self.goals.items():
            if not goal.done and goal.check(self):
                goal.done = True
                completed.append(horizon)
        return completed

    # ── views ─────────────────────────────────────────────────────

    def goals_line(self) -> str:
        """Dense one-line goal view for prompts. Empty when no goals."""

        bits = []
        for horizon in ("final", "medium", "short"):
            g = self.goals.get(horizon)
            if g is not None:
                mark = "DONE " if g.done else ""
                bits.append(f"{horizon.upper()}: {mark}{g.text}")
        return " | ".join(bits)

    # ── novelty ───────────────────────────────────────────────────

    def note(self, dimension: str, key: Any, *, t_ms: int) -> int:
        """Record a visit along one novelty dimension; return the count.

        Consecutive repeats of the same key don't inflate the count
        (standing on a tile is one visit until you leave and return).
        A count of 1 means NOVEL and bumps :attr:`last_novel_ms`.
        """

        if self._novelty_last.get(dimension) == key:
            return self._novelty_counts.get(dimension, {}).get(key, 1)
        self._novelty_last[dimension] = key
        counts = self._novelty_counts.setdefault(dimension, {})
        counts[key] = counts.get(key, 0) + 1
        if counts[key] == 1:
            self.last_novel_ms = t_ms
        return counts[key]

    def distinct_count(self, dimension: str) -> int:
        """How many DISTINCT keys were ever seen along one dimension."""

        return len(self._novelty_counts.get(dimension, {}))

    def novelty_snapshot(self) -> dict[str, int]:
        """Distinct-key count per novelty dimension.

        The read side of :meth:`note`, for the end-of-session scorecard.
        Counts only — the keys themselves (dialogue lines, visual
        buckets) can be large and are not worth carrying into a trailer.
        """

        return {dim: len(keys) for dim, keys in self._novelty_counts.items()}

    def novelty_keys(self, dimension: str) -> list[Any]:
        """Distinct keys seen along one dimension.

        Only safe for small-vocabulary dimensions (``screen``). Do NOT
        call for ``visual``/``dialog`` — those are unbounded by design.
        """

        return list(self._novelty_counts.get(dimension, {}))

    def goals_completed(self) -> int:
        """Metric goals marked done. Planner-authored — see progress.py."""

        return sum(1 for g in self.goals.values() if g.done)

    def mark_progress(self, t_ms: int) -> None:
        """External progress signal (goal completed, …) counts as novelty."""

        self.last_novel_ms = max(self.last_novel_ms, t_ms)

    # ── stall watchdog ────────────────────────────────────────────

    def stalled_for_ms(self, now_ms: int) -> int:
        """Milliseconds since the world last changed at all."""

        return max(0, now_ms - self.last_change_ms)

    def novelty_stalled_for_ms(self, now_ms: int) -> int:
        """Milliseconds since anything NOVEL happened.

        The honest stall measure: fact churn (menu open/close, HP
        animation) resets :meth:`stalled_for_ms` forever without any
        progress being made; this only resets on first-time tiles/
        screens/dialogue or an explicit :meth:`mark_progress`.
        """

        return max(0, now_ms - self.last_novel_ms)


__all__ = ["Fact", "Goal", "WorldState"]
