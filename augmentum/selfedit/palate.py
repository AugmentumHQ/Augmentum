"""The Palate — a per-user, few-shot, legible model of YOUR judgment.

The self-edit engine has two oracles and nothing between them: **mechanical**
(crisp correctness, free, narrow) and **human** (taste + correctness, total, but
priced in your attention). So every taste-laden change — layout, density, feel,
"I'd keep that" — must interrupt you, and the system can never act on *feel* on
its own. Taste is not a test; there is no benchmark coming for "make this better
for **this** person."

The Palate is the **third oracle**: given a proposed change it returns
``p_keep`` ∈ [0,1], a ``confidence`` (how much *relevant* evidence backs it), and
a plain-language ``rationale`` — distilled from the never-pruned archive's own
keep/revert labels. It is deliberately **not** a trained net but a transparent
weighted-evidence scorer, so it is:

* **legible** — it explains itself in one line you can read and correct;
* **useful from the first label** — a shape prior backbone at N≈0, sharpening
  per verdict (the reward-model literature needs a *distribution*; we have one
  person and dozens of labels, so we model evidence, not weights);
* **cold-start honest** — with no relevant evidence it says so (low confidence)
  and defers to you; it never fabricates a taste verdict.

This module is **P1: read-only**. The Palate only *informs* — it annotates what
the Workshop surfaces and can advise routing. It does NOT change autonomy: the
``palatable`` verdict tier and any auto-apply are later phases, earned per shape
only once the Palate is a *calibrated* predictor of your verdict there.
Verification leads; capability follows. It can never lower the safety floor
(mechanical oracles stay the gate), and — like the acceptance-test oracle — it is
computed by us over the settled archive, never authored by the editing agent, so
it can't be gamed.

Spec: ``docs/superpowers/specs/2026-06-24-palate-taste-oracle-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from augmentum.selfedit.preferences import change_shape
from augmentum.selfedit.retrodiction import _KEPT_STATUSES, _REVERTED_STATUSES
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "ChangeFeatures",
    "PalateVerdict",
    "Palate",
    "features_from_attempt",
    "features_from_target",
    "build_palate",
    "palate_profile",
]

# Evidence needed before the Palate "speaks" with meaningful confidence. Below
# this it stays honest — low confidence, defer to the human. Confidence ramps as
# relevant evidence accrues (n / (n + _CONFIDENCE_K)).
_CONFIDENCE_K = 4
_MIN_CONFIDENT = 0.5           # a verdict at/above this is worth acting on as a signal
_NEUTRAL = 0.5                 # p_keep with no evidence — pure "I don't know yet"


@dataclass(frozen=True)
class ChangeFeatures:
    """The interpretable features of a change the Palate scores over — all
    already derivable from an archive row or a proposed target. Discrete +
    legible on purpose (no embeddings): the overlap of these fields is the
    similarity metric, which keeps the whole model explainable."""

    shape: str                 # surface:intent_class (preferences.change_shape)
    surface: str
    intent_class: str
    subsystem: str             # top path prefix, e.g. "augmentum/selfedit"
    origin: str = "audit"      # audit | demand | capability — provenance of the ask

    def overlap(self, other: ChangeFeatures) -> float:
        """Transparent similarity ∈ [0,1]: weighted agreement on the fields that
        matter most for taste. Shape carries the most signal, then subsystem,
        then the raw surface. No hidden features — you can read why two changes
        are "similar.\""""
        score = 0.0
        if self.shape and self.shape == other.shape:
            score += 0.5
        elif self.surface and self.surface == other.surface:
            score += 0.2   # partial credit: same surface, different intent
        if self.subsystem and self.subsystem == other.subsystem:
            score += 0.3
        if self.intent_class and self.intent_class == other.intent_class:
            score += 0.2
        return min(1.0, score)


@dataclass
class PalateVerdict:
    """What the Palate thinks — and how much to trust the thought."""

    p_keep: float
    confidence: float
    rationale: str
    evidence_count: int = 0
    shape: str = ""

    @property
    def speaks(self) -> bool:
        """True when there is enough relevant evidence to treat this as a real
        signal (vs an honest "I don't know yet" that must defer to the human)."""
        return self.confidence >= _MIN_CONFIDENT

    @property
    def lean(self) -> str:
        if not self.speaks:
            return "unknown"
        return "keep" if self.p_keep >= 0.6 else "revert" if self.p_keep <= 0.4 else "unsure"

    def to_dict(self) -> dict:
        return {
            "p_keep": round(self.p_keep, 4), "confidence": round(self.confidence, 4),
            "rationale": self.rationale, "evidence_count": self.evidence_count,
            "shape": self.shape, "speaks": self.speaks, "lean": self.lean,
        }


def _subsystem(files: list[str]) -> str:
    """The dominant top-2 path segment across the changed files — the 'where' of a
    change (``augmentum/selfedit``, ``ui/scripts``, …). Legible and cheap."""
    counts: dict[str, int] = {}
    for f in files or []:
        parts = str(f).replace("\\", "/").split("/")
        if len(parts) >= 2:
            key = "/".join(parts[:2])
        elif parts:
            key = parts[0]
        else:
            continue
        counts[key] = counts.get(key, 0) + 1
    return max(counts, key=lambda k: counts[k]) if counts else ""


def features_from_attempt(attempt: dict) -> ChangeFeatures:
    """Extract features from an archived attempt row (has files_changed + a
    verdict carrying intent_class)."""
    verdict = attempt.get("gate_verdict") or {}
    intent_class = ""
    if isinstance(verdict, dict):
        intent_class = str(verdict.get("intent_class", "") or "")
    surface = str(attempt.get("surface", "") or "")
    files = attempt.get("files_changed") or []
    if not isinstance(files, list):
        files = []
    return ChangeFeatures(
        shape=change_shape(surface, intent_class), surface=surface,
        intent_class=intent_class, subsystem=_subsystem(files),
        origin=str(attempt.get("source", "") or "audit"),
    )


def features_from_target(*, surface: str, intent_class: str,
                         files: list[str] | None = None,
                         origin: str = "audit") -> ChangeFeatures:
    """Features for a PROPOSED change (a debt/demand target, before it's an
    attempt) — the live-annotation path."""
    return ChangeFeatures(
        shape=change_shape(surface, intent_class), surface=surface,
        intent_class=intent_class, subsystem=_subsystem(files or []),
        origin=origin,
    )


@dataclass
class _Example:
    features: ChangeFeatures
    kept: bool
    objective: str = ""


@dataclass
class Palate:
    """The per-user taste model: labeled examples (settled keep/revert history)
    for prototype similarity, plus a per-shape keep/revert tally for the prior
    backbone. Built by ``build_palate``; ``assess`` is pure.

    ``shape_tally`` is the CANONICAL human-verdict signal (the preference store):
    it records every keep/revert the human gave, independent of whether the git
    apply succeeded — because a keep is a *taste* judgment, not a git mechanic.
    Sourcing the shape prior from it (not re-derived from attempt status) keeps
    the Palate in lock-step with the Learned lane and means an apply-pending keep
    still teaches taste. Falls back to attempt-derived counts when a shape has no
    tally (e.g. git-ingested history)."""

    examples: list[_Example] = field(default_factory=list)
    shape_tally: dict[str, tuple[int, int]] = field(default_factory=dict)  # shape → (kept, reverted)

    @property
    def n_labels(self) -> int:
        # the human's total verdicts (tally) if present, else settled examples
        tallied = sum(k + r for k, r in self.shape_tally.values())
        return max(tallied, len(self.examples))

    @property
    def base_keep_rate(self) -> float:
        """Global prior — the fallback when a query has no shape/neighbor signal."""
        tk = sum(k for k, _ in self.shape_tally.values())
        tr = sum(r for _, r in self.shape_tally.values())
        if tk + tr:
            return tk / (tk + tr)
        if not self.examples:
            return _NEUTRAL
        return sum(1 for e in self.examples if e.kept) / len(self.examples)

    def _shape_prior(self, shape: str) -> tuple[float, int]:
        """(keep_rate, n) for this exact shape — the cold-start backbone. Prefers
        the canonical human-verdict tally; falls back to settled attempt counts."""
        if shape in self.shape_tally:
            kept, reverted = self.shape_tally[shape]
            n = kept + reverted
            if n:
                return kept / n, n
        same = [e for e in self.examples if e.features.shape == shape]
        if not same:
            return _NEUTRAL, 0
        return sum(1 for e in same if e.kept) / len(same), len(same)

    def assess(self, features: ChangeFeatures) -> PalateVerdict:
        """Score a proposed change: blend the shape prior with prototype
        similarity (nearest-kept vs nearest-reverted), confidence-weighted by
        how much *relevant* evidence exists. Every term is legible."""
        if not self.examples and not self.shape_tally:
            return PalateVerdict(
                p_keep=_NEUTRAL, confidence=0.0, evidence_count=0, shape=features.shape,
                rationale="no taste history yet — surfacing for your verdict")

        shape_rate, shape_n = self._shape_prior(features.shape)

        # Prototype similarity: weight each example by its overlap with the query,
        # then the weighted keep-rate is "how much this resembles things you kept."
        weighted_keep = 0.0
        weight_sum = 0.0
        best_kept_sim = 0.0
        best_kept_obj = ""
        for e in self.examples:
            w = features.overlap(e.features)
            if w <= 0:
                continue
            weight_sum += w
            if e.kept:
                weighted_keep += w
                if w > best_kept_sim:
                    best_kept_sim, best_kept_obj = w, e.objective
        proto_rate = (weighted_keep / weight_sum) if weight_sum > 0 else _NEUTRAL

        # Relevant evidence = shape labels + fractional neighbor mass. Confidence
        # ramps with it; the shape prior dominates when present, else prototypes.
        relevant = shape_n + weight_sum
        confidence = relevant / (relevant + _CONFIDENCE_K)
        if shape_n > 0:
            # shape prior is the backbone; prototypes nudge it toward the neighbors
            p_keep = 0.7 * shape_rate + 0.3 * proto_rate
        elif weight_sum > 0:
            p_keep = proto_rate
        else:
            # no shape, no neighbors → only the global base rate, held at low conf
            p_keep = self.base_keep_rate
            confidence = min(confidence, 0.25)

        rationale = self._rationale(features.shape, shape_rate, shape_n,
                                    best_kept_sim, best_kept_obj, p_keep, confidence)
        return PalateVerdict(
            p_keep=p_keep, confidence=confidence, evidence_count=int(round(relevant)),
            shape=features.shape, rationale=rationale)

    @staticmethod
    def _rationale(shape: str, shape_rate: float, shape_n: int,
                   best_sim: float, best_obj: str, p_keep: float, conf: float) -> str:
        if conf < _MIN_CONFIDENT:
            if shape_n or best_sim:
                return (f"only weak signal for '{shape}' so far — deferring to your "
                        "verdict (it teaches the most here)")
            return "no comparable changes in your history yet — surfacing for you"
        parts: list[str] = []
        if shape_n:
            kr = round(shape_rate * 100)
            parts.append(f"you've kept {kr}% of '{shape}' changes ({shape_n})")
        if best_sim >= 0.5 and best_obj:
            verb = "kept" if p_keep >= 0.5 else "seen"
            parts.append(f"resembles a change you {verb}: “{best_obj[:60]}”")
        lead = "likely a keep" if p_keep >= 0.6 else "likely a revert" if p_keep <= 0.4 else "could go either way"
        return f"{lead} — " + "; ".join(parts) if parts else lead


def build_palate(attempts: list[dict],
                 preferences: list[dict] | None = None) -> Palate:
    """Distill the Palate from SETTLED archive history (prototype examples) + the
    per-shape human-verdict tally (``preferences``, the canonical keep/revert
    signal from the preference store). ``preferences`` items are dicts with
    ``shape``/``kept``/``reverted`` (``PreferenceStat.to_dict()``). Passing them
    makes the Palate learn from every human verdict — even a keep whose git-apply
    is still pending — and keeps it consistent with the Learned lane."""
    examples: list[_Example] = []
    for a in attempts:
        status = str(a.get("status", "")).strip().lower()
        if status in _KEPT_STATUSES:
            kept = True
        elif status in _REVERTED_STATUSES:
            kept = False
        else:
            continue
        examples.append(_Example(
            features=features_from_attempt(a), kept=kept,
            objective=str(a.get("objective", "") or "")))
    tally: dict[str, tuple[int, int]] = {}
    for p in preferences or []:
        shape = str(p.get("shape", "") or "")
        if not shape:
            continue
        tally[shape] = (int(p.get("kept", 0) or 0), int(p.get("reverted", 0) or 0))
    return Palate(examples=examples, shape_tally=tally)


def palate_profile(attempts: list[dict],
                   preferences: list[dict] | None = None) -> dict:
    """"What the Palate has learned about you" — a legible, correctable profile:
    per-shape keep-rates rendered as plain statements, the evidence behind each,
    and the model's overall coverage. Read-only; feeds ``GET /api/selfedit/palate``
    and the Workshop Learned lane. ``preferences`` (the canonical human-verdict
    tally) is the primary source; settled attempts fill shapes it doesn't cover."""
    palate = build_palate(attempts, preferences)
    by_shape: dict[str, tuple[int, int]] = dict(palate.shape_tally)  # canonical first
    for e in palate.examples:                                         # fill uncovered shapes
        if e.features.shape in palate.shape_tally:
            continue
        k, r = by_shape.get(e.features.shape, (0, 0))
        by_shape[e.features.shape] = (k + (1 if e.kept else 0), r + (0 if e.kept else 1))

    statements: list[dict] = []
    for shape, (kept, reverted) in sorted(by_shape.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
        n = kept + reverted
        if n == 0:
            continue
        rate = kept / n
        if rate >= 0.75:
            verb = f"tend to KEEP '{shape}' changes"
        elif rate <= 0.25:
            verb = f"tend to REVERT '{shape}' changes"
        else:
            verb = f"are split on '{shape}' changes"
        statements.append({
            "shape": shape, "statement": f"You {verb}",
            "kept": kept, "reverted": n - kept, "samples": n,
            "keep_rate": round(rate, 3),
            # legibility: a shape needs a few labels before its statement is firm
            "firm": n >= 3,
        })
    return {
        "n_labels": palate.n_labels,
        "base_keep_rate": round(palate.base_keep_rate, 3),
        "statements": statements,
        # honest cold-start banner for the UI
        "warming_up": palate.n_labels < _CONFIDENCE_K,
        "min_confident_evidence": _CONFIDENCE_K,
    }
