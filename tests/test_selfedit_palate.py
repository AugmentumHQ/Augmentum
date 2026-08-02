"""The Palate — the per-user, few-shot, cold-start-honest taste oracle.

Locks the properties the whole design rests on:
* cold-start HONESTY — with no/thin labels it says so (low confidence, defers),
  never fabricates a taste verdict;
* the shape-prior backbone — a consistently kept/reverted shape drives p_keep;
* prototype similarity — a change that resembles kept work leans keep;
* confidence ramps with RELEVANT evidence, and only 'speaks' past the floor;
* legibility — the rationale is plain and the profile renders correctable
  statements;
* it distills from SETTLED (human-decided) history only — the honest labels.
"""

from __future__ import annotations

from augmentum.selfedit import palate as P


def _attempt(*, surface, intent_class, status, files=None, objective="o", aid="a"):
    return {
        "id": aid, "surface": surface, "status": status, "objective": objective,
        "files_changed": files or [f"augmentum/{surface}/x.py"],
        "gate_verdict": {"tier": "human_required", "intent_class": intent_class},
    }


def _kept(surface, intent_class, **kw):
    return _attempt(surface=surface, intent_class=intent_class, status="promoted", **kw)


def _reverted(surface, intent_class, **kw):
    return _attempt(surface=surface, intent_class=intent_class, status="rolled_back", **kw)


def _feat(surface, intent_class, files=None):
    return P.features_from_target(surface=surface, intent_class=intent_class, files=files or [])


# ---------------------------------------------------------------------------
# cold-start honesty
# ---------------------------------------------------------------------------

def test_empty_palate_defers_to_human():
    pal = P.build_palate([])
    v = pal.assess(_feat("frontend", "style"))
    assert v.confidence == 0.0 and v.speaks is False
    assert v.p_keep == 0.5 and v.lean == "unknown"
    assert "no taste history" in v.rationale


def test_thin_evidence_does_not_speak():
    # one label for a shape → still below the confidence floor; honest deferral
    pal = P.build_palate([_kept("frontend", "style")])
    v = pal.assess(_feat("frontend", "style"))
    assert v.evidence_count >= 1
    assert v.speaks is False        # one datapoint is not a taste model
    assert v.lean == "unknown"


# ---------------------------------------------------------------------------
# the shape-prior backbone
# ---------------------------------------------------------------------------

def test_consistently_kept_shape_leans_keep_and_speaks():
    attempts = [_kept("frontend", "style", aid=f"k{i}") for i in range(6)]
    pal = P.build_palate(attempts)
    v = pal.assess(_feat("frontend", "style"))
    assert v.speaks is True
    assert v.p_keep >= 0.7 and v.lean == "keep"
    assert "kept" in v.rationale.lower()


def test_consistently_reverted_shape_leans_revert():
    attempts = [_reverted("frontend", "style", aid=f"r{i}") for i in range(6)]
    pal = P.build_palate(attempts)
    v = pal.assess(_feat("frontend", "style"))
    assert v.speaks is True
    assert v.p_keep <= 0.4 and v.lean == "revert"


def test_confidence_ramps_with_evidence():
    small = P.build_palate([_kept("backend", "bugfix", aid=f"k{i}") for i in range(2)])
    big = P.build_palate([_kept("backend", "bugfix", aid=f"k{i}") for i in range(20)])
    cs = small.assess(_feat("backend", "bugfix")).confidence
    cb = big.assess(_feat("backend", "bugfix")).confidence
    assert cb > cs                   # more relevant labels → more confidence


# ---------------------------------------------------------------------------
# prototype similarity (looks-like-things-you-kept)
# ---------------------------------------------------------------------------

def test_overlap_is_transparent():
    a = P.ChangeFeatures("frontend:style", "frontend", "style", "ui/scripts")
    same = P.ChangeFeatures("frontend:style", "frontend", "style", "ui/scripts")
    diff = P.ChangeFeatures("backend:bugfix", "backend", "bugfix", "augmentum/models")
    assert a.overlap(same) == 1.0
    assert a.overlap(diff) == 0.0
    # partial: same surface, different intent → partial credit
    partial = P.ChangeFeatures("frontend:feature", "frontend", "feature", "ui/scripts")
    assert 0.0 < a.overlap(partial) < 1.0


def test_unseen_shape_borrows_from_similar_neighbors():
    # kept a lot of ui/scripts frontend work; a NEW frontend intent in the same
    # subsystem should lean keep via prototype similarity even with no exact shape.
    attempts = [_kept("frontend", "style", files=["ui/scripts/a.js"], aid=f"k{i}")
                for i in range(6)]
    pal = P.build_palate(attempts)
    v = pal.assess(_feat("frontend", "feature", files=["ui/scripts/b.js"]))
    assert v.p_keep > 0.5            # resembles kept frontend/ui work
    assert v.shape == "frontend:feature"


# ---------------------------------------------------------------------------
# distillation: settled labels only
# ---------------------------------------------------------------------------

def test_only_settled_rows_become_labels():
    attempts = [
        _kept("backend", "bugfix", aid="k1"),
        _reverted("backend", "bugfix", aid="r1"),
        _attempt(surface="backend", intent_class="bugfix", status="gated", aid="g1"),
        _attempt(surface="backend", intent_class="bugfix", status="failed", aid="f1"),
        _attempt(surface="backend", intent_class="bugfix", status="rejected", aid="x1"),
    ]
    pal = P.build_palate(attempts)
    assert pal.n_labels == 2         # only promoted + rolled_back count


# ---------------------------------------------------------------------------
# the legible profile
# ---------------------------------------------------------------------------

def test_profile_renders_plain_statements():
    attempts = ([_kept("frontend", "style", aid=f"k{i}") for i in range(4)]
                + [_reverted("backend", "refactor", aid=f"r{i}") for i in range(4)])
    prof = P.palate_profile(attempts)
    assert prof["n_labels"] == 8
    shapes = {s["shape"]: s for s in prof["statements"]}
    assert "KEEP" in shapes["frontend:style"]["statement"]
    assert "REVERT" in shapes["backend:refactor"]["statement"]
    assert shapes["frontend:style"]["firm"] is True   # >=3 labels
    assert prof["warming_up"] is False                # 8 >= floor


def test_profile_empty_is_warming_up():
    prof = P.palate_profile([])
    assert prof["n_labels"] == 0 and prof["warming_up"] is True
    assert prof["statements"] == []


# ---------------------------------------------------------------------------
# the preference tally = canonical human-verdict signal (live-found gap fix):
# a keep teaches taste even when its git-apply is pending, so the Palate must
# learn from the preference tally, not just terminal attempt status.
# ---------------------------------------------------------------------------

def _pref(shape, kept, reverted):
    return {"shape": shape, "kept": kept, "reverted": reverted}


def test_palate_learns_from_preference_tally_without_promoted_attempts():
    # NO settled attempts (all gated / apply-pending), but the human kept 5
    # backend:feature changes → the Palate must still speak from the tally.
    prefs = [_pref("backend:feature", 5, 0)]
    pal = P.build_palate([], prefs)
    v = pal.assess(_feat("backend", "feature"))
    assert v.speaks is True and v.lean == "keep"
    assert pal.n_labels == 5


def test_tally_is_the_shape_prior_backbone():
    prefs = [_pref("frontend:style", 0, 6)]   # you revert these
    pal = P.build_palate([], prefs)
    v = pal.assess(_feat("frontend", "style"))
    assert v.speaks and v.lean == "revert" and v.p_keep <= 0.4


def test_profile_uses_the_tally():
    prof = P.palate_profile([], [_pref("backend:feature", 4, 1)])
    shapes = {s["shape"]: s for s in prof["statements"]}
    assert "KEEP" in shapes["backend:feature"]["statement"]
    assert shapes["backend:feature"]["samples"] == 5
    assert prof["warming_up"] is False   # 5 >= floor


def test_settled_attempts_fill_shapes_the_tally_misses():
    # tally covers backend:feature; a git-ingested kept frontend:style attempt
    # (no tally) still contributes a statement.
    prof = P.palate_profile(
        [_kept("frontend", "style", aid="k1")],
        [_pref("backend:feature", 3, 0)])
    shapes = {s["shape"] for s in prof["statements"]}
    assert "backend:feature" in shapes and "frontend:style" in shapes
