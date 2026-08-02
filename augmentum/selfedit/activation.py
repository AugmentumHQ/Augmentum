"""The verified skill graph — capability that GROWS, derived from the archive.

A modest local model cannot be made smarter by editing itself, but the *capability
of the system* can grow without bound if growth lives in a **grown, verified,
weighted graph** rather than in frozen weights (the agent-memory literature's
dominant lever: the gap between has-memory and no-memory often exceeds the gap
between backbones). This module is **P1** of that graph — and it is built to the
integration doctrine's sharpest rule: *integrate as DATA on a stable spine, not as
code.*

So there is **no new table.** The skill graph is a **pure, deterministic fold over
the never-pruned ``self_edit_attempts`` archive** (the spine), replayed in
chronological order. The edge weights *are* the archive, viewed through a
verification-gated plasticity lens. Delete this module and nothing is lost — the
archive is untouched. That is the reversibility the doctrine demands.

Three properties the research says a naive co-occurrence graph cannot skip (it
*fails* without them) are first-class here:

* **Verification-gated, never raw frequency** (anti-Goodhart). Edges strengthen on
  a *verified/shipped* outcome, weaken on a *rolled-back/rejected* one, and do
  nothing on a pending attempt. The verdict is the dopamine — a three-factor /
  reward-modulated rule, never "fire together, wire together."
* **Stability — BCM sliding threshold + Oja normalization** (gap #1). Pure Hebbian,
  even reward-modulated, is provably unstable (a strong edge fires→strengthens→
  runaway, the graph collapses onto monopoly nodes). A node's contribution to new
  weight is gated by ``(1 - frequency)`` — a *rising bar* for ubiquitous nodes
  (BCM) that doubles as anti-monopoly diversity — and each node's outgoing vector
  is L2-bounded at the end (Oja's essence). Old verdicts decay (homeostatic, recent
  matters more) — but **decay ≠ delete**: the archive is never pruned, only the
  *active/callable* weighting fades.

What's deliberately **out of scope** here (later phases, pulled by real need, not
pre-built — doctrine #9): per-skill applicability preconditions (gap #2, P2), the
causal "why" harvested from the verifier-sensitive minimal change (gap #3, P3), and
the exploration curriculum (gap #4, P4).

The output is a **read-only selection signal** — ``score(context) -> {score,
confidence, ...}`` — exposed to advisors/routers as advice. It changes no autonomy;
it is shadow by construction (doctrine #4), accumulating against real outcomes
before it is ever trusted to act.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# --- plasticity tunables (deliberately conservative) ----------------------
LEARNING_RATE = 1.0       # base step; modulation r ∈ [-1, 1] scales it
DECAY = 0.01              # per-attempt homeostatic decay of edge weight (recency)
W_MAX = 4.0               # hard clamp before normalization (runaway backstop)
CONF_K = 5.0              # support at which confidence ≈ 0.5 (evidence half-life)
HOP_DISCOUNT = 0.5        # one-hop neighborhood weight in activation spread
ANTI_MONOPOLY = 0.6       # how hard the BCM threshold damps a ubiquitous node
                          # (0 = off; 1 = fully zeroed at frequency 1). A node that
                          # fires in *every* attempt is dampened, never silenced —
                          # so a consistently-successful region stays signal.

# Verification signal (the third factor) by terminal archive status. Pending /
# in-flight statuses contribute NOTHING (0.0) — only a real verdict moves a weight.
_MODULATION: dict[str, float] = {
    "promoted": 1.0,      # shipped + endorsed → strongest positive
    "live": 1.0,          # applied to the running app
    "rolled_back": -1.0,  # was applied, then reverted → a real mistake
    "rejected": -0.5,     # never made it past the gate → soft negative
    "failed": -0.5,       # broke during the attempt → soft negative
}

# Provenance damping (ingest-all-work). The engine's own attempts carry a real
# oracle verdict → full weight. An ingested git commit's verdict is implicit
# (kept-in-history / reverted) → real but weaker evidence. A coder turn's
# "done" is the weakest keep signal we accept → heavily damped, never zero
# (zero would make ingestion pointless). Rows without a source (pre-306) are
# the engine's own → full weight; UNKNOWN future sources get half weight until
# someone consciously rates them (conservative, not silent full-trust).
_SOURCE_WEIGHT: dict[str, float] = {
    "autonomous": 1.0,
    "": 1.0,
    "git": 0.6,
    "coder": 0.25,
}


def modulation_for_attempt(attempt: dict) -> float:
    """The verification signal r ∈ [-1, 1] for one archived attempt. Driven by the
    terminal ``status`` (the honest verdict), not by anything the editing agent can
    author — so it can't be gamed from inside the loop. Damped by the row's
    provenance (``source``) so ingested implicit verdicts never outshout the
    engine's own oracle-confirmed ones."""
    base = _MODULATION.get(str(attempt.get("status", "")).strip().lower(), 0.0)
    if base == 0.0:
        return 0.0
    source = str(attempt.get("source", "")).strip().lower()
    return base * _SOURCE_WEIGHT.get(source, 0.5)


def _subsystem(path: str) -> str:
    """The 2-level path prefix a change touched (e.g. ``augmentum/selfedit``,
    ``ui/scripts``) — the unit at which 'this region behaves well/badly' is legible."""
    p = (path or "").strip().lstrip("./").replace("\\", "/")
    parts = [seg for seg in p.split("/") if seg]
    if not parts:
        return ""
    return "/".join(parts[:2])


def atoms_for_attempt(attempt: dict) -> set[str]:
    """The set of co-activating nodes for one attempt — extracted only from columns
    the archive actually stores (surface, tier, files_changed). Namespaced so a
    surface can never collide with a subsystem."""
    atoms: set[str] = set()
    surface = str(attempt.get("surface", "")).strip().lower()
    if surface:
        atoms.add(f"shape:{surface}")
    tier = str(attempt.get("tier", "")).strip().lower()
    if tier:
        atoms.add(f"tier:{tier}")
    target = str(attempt.get("target", "")).strip().lower()
    if target:
        # the structured debt class (``scanner.metric``) — lets the graph carry
        # "have we historically LANDED this class of fix" as first-class signal,
        # which transfers across files/surfaces the way a raw region cannot.
        atoms.add(f"target:{target}")
    for path in attempt.get("files_changed") or []:
        sub = _subsystem(str(path))
        if sub:
            atoms.add(f"sub:{sub}")
    return atoms


def query_atoms(*, surface: str = "", files: list[str] | None = None,
                tier: str = "", target: str = "") -> set[str]:
    """The same extractor, for a *prospective* context (a candidate target) so the
    query and the learned graph speak the same vocabulary."""
    return atoms_for_attempt({"surface": surface, "tier": tier, "target": target,
                              "files_changed": files or []})


@dataclass
class ActivationScore:
    """A read-only selection signal for one query context.

    ``score`` ∈ [-1, 1] (tanh-squashed): positive = a region of verified success
    (safe/fruitful to attempt), negative = a region of repeated failure (deprioritize
    or escalate harder). ``confidence`` ∈ [0, 1] grows with how much real evidence
    backs the query atoms — low confidence ⇒ unknown ⇒ defer (never act on a guess).
    """

    score: float = 0.0
    confidence: float = 0.0
    support: float = 0.0                 # total verdict-evidence on the query atoms
    contributors: list[tuple[str, float]] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "support": round(self.support, 2),
            "contributors": [[a, round(w, 4)] for a, w in self.contributors],
            "rationale": self.rationale,
        }


@dataclass
class SkillGraph:
    """A weighted graph folded from the archive. Edges are directed-symmetric
    associations; ``activity`` is per-node verdict evidence; ``attempts`` is the
    number of archived attempts that fed the fold."""

    edges: dict[str, dict[str, float]] = field(default_factory=dict)
    activity: dict[str, float] = field(default_factory=dict)
    attempts: int = 0

    # --- scoring (activation spread) --------------------------------------

    def score(self, atoms: set[str]) -> ActivationScore:
        """Activation spread from a query context: internal coherence (have these
        atoms co-succeeded?) + a one-hop discounted neighborhood (what verified
        structure radiates from this region?), squashed to [-1, 1]."""
        present = [a for a in atoms if a in self.edges or a in self.activity]
        if not present:
            return ActivationScore(rationale="no learned evidence for this context")

        contrib: dict[str, float] = {}
        # internal coherence: weights among the query atoms themselves.
        for a in present:
            row = self.edges.get(a, {})
            for b in present:
                if b != a and b in row:
                    contrib[f"{a}↔{b}"] = contrib.get(f"{a}↔{b}", 0.0) + row[b]
        coherence = sum(contrib.values()) / 2.0  # each undirected pair counted twice

        # one-hop neighborhood: verified structure radiating out of the region.
        neigh: dict[str, float] = {}
        for a in present:
            for b, w in self.edges.get(a, {}).items():
                if b not in atoms:
                    neigh[b] = neigh.get(b, 0.0) + w
        neighborhood = sum(neigh.values())

        raw = coherence + HOP_DISCOUNT * neighborhood
        squashed = math.tanh(raw)

        support = sum(self.activity.get(a, 0.0) for a in present)
        confidence = support / (support + CONF_K)

        # top contributors for the rationale: strongest signed neighbors + pairs.
        merged = dict(contrib)
        for b, w in neigh.items():
            merged[f"→{b}"] = HOP_DISCOUNT * w
        top = sorted(merged.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        rationale = self._rationale(squashed, confidence, top)
        return ActivationScore(score=squashed, confidence=confidence, support=support,
                               contributors=top, rationale=rationale)

    @staticmethod
    def _rationale(score: float, confidence: float, top: list[tuple[str, float]]) -> str:
        if confidence < 0.2:
            return "thin evidence — treat as unknown"
        lean = ("verified-success region" if score > 0.15
                else "repeated-failure region" if score < -0.15
                else "mixed/neutral region")
        bits = ", ".join(f"{a} ({w:+.2f})" for a, w in top[:3])
        return f"{lean}; strongest signals: {bits}" if bits else lean

    def score_target(self, scanner: str, metric: str,
                     files: list[str] | None = None) -> ActivationScore:
        """Trust signal for a structured debt class (``scanner.metric``), optionally
        narrowed by the files it would touch. The honest read for a STRUCTURAL item
        the human is deciding on: have we landed this class before, or does the
        archive show it keeps getting reverted? Cold (confidence ~0) until the loop
        has recorded attempts of this class — never a fabricated guess."""
        return self.score(query_atoms(target=f"{scanner}.{metric}", files=files or []))

    def top_regions(self, n: int = 10) -> list[tuple[str, float]]:
        """Nodes ranked by net outgoing weight — the regions the system has learned
        to trust (positive) or distrust (negative). For the Workshop view."""
        net = {a: sum(row.values()) for a, row in self.edges.items()}
        return sorted(net.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def to_dict(self) -> dict:
        return {
            "attempts": self.attempts,
            "nodes": len(self.activity),
            "edges": sum(len(r) for r in self.edges.values()) // 2,
            "top_regions": [[a, round(w, 4)] for a, w in self.top_regions(12)],
        }


# --- the fold (the heart) -------------------------------------------------

def build_graph(attempts: list[dict], *, learning_rate: float = LEARNING_RATE,
                decay: float = DECAY, w_max: float = W_MAX) -> SkillGraph:
    """Replay the archive in chronological order, applying the verification-gated,
    BCM-stabilized plasticity rule. PURE — same archive in, same graph out.

    The rule, per attempt at step ``t`` with modulation ``r`` over its co-activating
    atom set ``A``:
      * ``theta[n] = ANTI_MONOPOLY · appearances[n] / t`` — the BCM sliding
        threshold: a node active in most attempts has a *rising* bar, so it
        contributes proportionally less new weight (anti-monopoly diversity). The
        ``ANTI_MONOPOLY`` factor (<1) damps a ubiquitous node without silencing it —
        a region that consistently appears AND succeeds stays signal.
      * for each pair (a, b) in A:  ``Δ = lr · r · (1 - theta[a]) · (1 - theta[b])``
        applied symmetrically and clamped to ``±w_max``.
    Edge decay is applied lazily (recency) and Oja L2-normalization bounds each
    node's outgoing vector once at the end (stability). ``activity`` accumulates
    only *verdict* evidence (``|r| > 0``) so confidence reflects real signal.
    """
    # sort by created_at then id for a stable, deterministic replay order.
    ordered = sorted(attempts, key=lambda a: (str(a.get("created_at", "")), str(a.get("id", ""))))

    edges: dict[str, dict[str, float]] = {}
    last_step: dict[str, dict[str, int]] = {}
    seen_count: dict[str, int] = {}    # times a node appeared (for theta)
    activity: dict[str, float] = {}    # accumulated |verdict evidence| (for support)
    keep = 1.0 - decay

    def _decayed(a: str, b: str, t: int) -> float:
        prior = edges.get(a, {}).get(b, 0.0)
        if prior == 0.0:
            return 0.0
        gap = t - last_step.get(a, {}).get(b, t)
        return prior * (keep ** gap) if gap > 0 else prior

    def _set(a: str, b: str, val: float, t: int) -> None:
        edges.setdefault(a, {})[b] = max(-w_max, min(w_max, val))
        last_step.setdefault(a, {})[b] = t

    folded = 0
    for attempt in ordered:
        atoms = atoms_for_attempt(attempt)
        if not atoms:
            continue
        # every appearance counts toward the BCM frequency threshold...
        for n in atoms:
            seen_count[n] = seen_count.get(n, 0) + 1
        folded += 1
        t = folded

        r = modulation_for_attempt(attempt)
        if r == 0.0:
            continue  # pending/unknown: structure noted (theta), but no weight moved
        # ...but only a real verdict accrues support evidence.
        for n in atoms:
            activity[n] = activity.get(n, 0.0) + abs(r)

        items = sorted(atoms)
        for i, a in enumerate(items):
            theta_a = ANTI_MONOPOLY * seen_count[a] / t
            for b in items[i + 1:]:
                theta_b = ANTI_MONOPOLY * seen_count[b] / t
                gain = (1.0 - theta_a) * (1.0 - theta_b)
                if gain <= 0.0:
                    continue
                delta = learning_rate * r * gain
                _set(a, b, _decayed(a, b, t) + delta, t)
                _set(b, a, _decayed(b, a, t) + delta, t)

    # finalize: bring every edge to the common final step (lazy decay flush)...
    T = folded
    for a, row in edges.items():
        for b in list(row):
            gap = T - last_step.get(a, {}).get(b, T)
            if gap > 0:
                row[b] = row[b] * (keep ** gap)
    # ...then Oja-style L2 bound: no node's outgoing influence exceeds unit norm
    # (competition / stability — caps runaway without erasing sign or ranking).
    for row in edges.values():
        norm = math.sqrt(sum(w * w for w in row.values()))
        if norm > 1.0:
            for b in row:
                row[b] /= norm

    log.info("selfedit_skill_graph_built", attempts=folded,
             nodes=len(seen_count), with_verdict=len(activity))
    return SkillGraph(edges=edges, activity=activity, attempts=folded)


# --- calibration (the signal earns the right to act) ----------------------
MIN_HISTORY = 3           # prior attempts needed before a prediction counts
CALIB_CONF_FLOOR = 0.5    # a prediction must be this confident to be scored
CALIB_SIGNAL_THRESH = 0.15  # ...and this far from neutral to be a directional call
CALIB_MIN_GRADUATE = 8    # directional calls needed before the signal can graduate
CALIB_ACC_FLOOR = 0.7     # realized directional accuracy required to graduate
CALIB_CAP = 200           # backtest at most this many recent attempts (bounded cost)


@dataclass
class Calibration:
    """How trustworthy the skill graph's signal actually is, measured by a
    *prequential backtest*: for each archived attempt, predict its region's
    outcome from ONLY the attempts before it, then compare to what really
    happened. Honest by construction (no future leakage), pure (a fold over the
    archive), and the basis for *earned* activation — the router applies the
    hint only once ``graduated`` is true."""

    n_attempts: int = 0
    n_predictions: int = 0     # confident, directional calls scored against an outcome
    n_correct: int = 0
    brier: float = 1.0         # mean (p − outcome)² over confident predictions; lower better
    graduated: bool = False
    truncated: bool = False    # backtest hit CALIB_CAP (older attempts not scored)
    rationale: str = ""

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_predictions if self.n_predictions else 0.0

    def to_dict(self) -> dict:
        return {
            "n_attempts": self.n_attempts, "n_predictions": self.n_predictions,
            "n_correct": self.n_correct, "accuracy": round(self.accuracy, 4),
            "brier": round(self.brier, 4), "graduated": self.graduated,
            "truncated": self.truncated, "rationale": self.rationale,
        }


def backtest_calibration(attempts: list[dict], *, min_history: int = MIN_HISTORY,
                         conf_floor: float = CALIB_CONF_FLOOR,
                         signal_thresh: float = CALIB_SIGNAL_THRESH,
                         min_graduate: int = CALIB_MIN_GRADUATE,
                         acc_floor: float = CALIB_ACC_FLOOR,
                         cap: int = CALIB_CAP) -> Calibration:
    """Prequential (predict-then-observe) backtest over the archive. For each
    attempt with a real verdict, score its region against the graph built from the
    attempts BEFORE it, and check whether the sign of that prediction matched the
    actual outcome. The signal *graduates* (earns the right to be applied, not just
    logged) once it has made enough confident directional calls at/above the
    accuracy floor. PURE."""
    ordered = sorted(attempts, key=lambda a: (str(a.get("created_at", "")), str(a.get("id", ""))))
    truncated = len(ordered) > cap
    if truncated:
        ordered = ordered[-cap:]
        log.info("selfedit_calibration_truncated", cap=cap, total=len(attempts))

    n_correct = 0
    records = 0
    brier_terms: list[float] = []
    for t, a in enumerate(ordered):
        r = modulation_for_attempt(a)
        if r == 0.0 or t < min_history:
            continue  # no ground truth, or not enough prior history to predict from
        s = build_graph(ordered[:t]).score(atoms_for_attempt(a))
        if s.confidence < conf_floor:
            continue
        outcome = 1 if r > 0 else 0
        p = (s.score + 1.0) / 2.0  # map score∈[-1,1] → keep-probability∈[0,1]
        brier_terms.append((p - outcome) ** 2)
        if abs(s.score) >= signal_thresh:
            records += 1
            if (s.score > 0) == (outcome == 1):
                n_correct += 1

    brier = sum(brier_terms) / len(brier_terms) if brier_terms else 1.0
    acc = n_correct / records if records else 0.0
    graduated = records >= min_graduate and acc >= acc_floor
    if records < min_graduate:
        rationale = (f"shadow — only {records} confident call(s); needs "
                     f"{min_graduate} to graduate")
    elif graduated:
        rationale = f"graduated — {n_correct}/{records} directional calls correct ({acc:.0%})"
    else:
        rationale = (f"shadow — {acc:.0%} accuracy over {records} calls, below the "
                     f"{acc_floor:.0%} floor")
    return Calibration(n_attempts=len(ordered), n_predictions=records,
                       n_correct=n_correct, brier=brier, graduated=graduated,
                       truncated=truncated, rationale=rationale)


# --- routing hint (read-only; the "router beats cascade" use of the signal) ---

def recommend_start_rung(score: ActivationScore, n_rungs: int, *,
                         fail_threshold: float = -0.3, conf_threshold: float = 0.5,
                         max_skip: int | None = None) -> int:
    """How many cheap rungs of the escalation ladder to SKIP for this target's
    region — a read-only routing hint, NOT an autonomy change (it only reorders
    which model tries first; it never auto-promotes anything).

    The research's point: an always-cheap-first cascade wastes the cheap tier on
    hard targets. When the archive shows a region is *confidently* failure-prone,
    start higher instead of burning the doomed cheap pass on it. Conservative by
    construction:
      * fires ONLY on real evidence (``confidence ≥ conf_threshold``) AND a
        clearly-negative region (``score < fail_threshold``) — unknown/neutral/
        positive regions always start at rung 0;
      * skips one rung normally, two only for a *strongly* failure-prone region;
      * never skips the whole ladder — ``max_skip`` (e.g. the count of non-frontier
        rungs − 1) keeps the expensive/frontier rung opt-in and at least one rung
        always runs.
    """
    if n_rungs <= 1:
        return 0
    cap = (n_rungs - 1) if max_skip is None else max(0, min(max_skip, n_rungs - 1))
    if cap <= 0:
        return 0
    if score.confidence < conf_threshold or score.score >= fail_threshold:
        return 0
    skip = 2 if score.score < fail_threshold * 2 else 1
    return min(skip, cap)


# --- async reader (thin IO over the growth connection) --------------------

async def load_attempts(conn: Any, *, user_id: str, limit: int = 2000) -> list[dict]:
    """Pull terminal-and-pending attempts for the fold, oldest→newest. Read-only;
    selects only the columns the graph needs."""
    import json as _json
    # NOTE: this SELECT must cover every column ``atoms_for_attempt`` and
    # ``modulation_for_attempt`` consume. ``target`` was missing until
    # 2026-07-01 — the `target:` atoms folded fine from dicts in tests but were
    # silently absent from every graph built off the live archive, killing the
    # per-debt-class transfer signal the debt loop queries (loop.py routing).
    cur = await conn.execute(
        "SELECT id, surface, tier, status, files_changed, created_at, source, "
        "target FROM self_edit_attempts WHERE user_id=? "
        "ORDER BY created_at ASC LIMIT ?",
        (user_id, max(1, min(int(limit or 2000), 10000))),
    )
    rows = await cur.fetchall()
    out: list[dict] = []
    for r in rows:
        try:
            files = _json.loads(r[4] or "[]")
        except (ValueError, TypeError):
            files = []
        out.append({"id": r[0], "surface": r[1], "tier": r[2], "status": r[3],
                    "files_changed": files, "created_at": r[5],
                    "source": (r[6] if len(r) > 6 else "") or "autonomous",
                    "target": (r[7] if len(r) > 7 else "") or ""})
    return out


async def load_graph(conn: Any, *, user_id: str) -> SkillGraph:
    """Build the skill graph from a user's archive on the growth connection."""
    return build_graph(await load_attempts(conn, user_id=user_id))


async def load_graph_and_calibration(
    conn: Any, *, user_id: str,
) -> tuple[SkillGraph, Calibration]:
    """Load the archive ONCE and derive both the graph and its calibration — the
    pair a caller needs to use the signal responsibly (the graph says *what*, the
    calibration says *whether to trust it yet*)."""
    attempts = await load_attempts(conn, user_id=user_id)
    return build_graph(attempts), backtest_calibration(attempts)


async def load_calibration(conn: Any, *, user_id: str) -> Calibration:
    """Backtest the signal's reliability from a user's archive."""
    return backtest_calibration(await load_attempts(conn, user_id=user_id))
