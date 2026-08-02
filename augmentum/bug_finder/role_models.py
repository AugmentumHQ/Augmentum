"""Per-role model selection + model-family classification.

The Plausible Completion Trap (SWE-bench Verified: 18.8-22%
logically-wrong-but-passing patches) describes the failure mode where
the same model that produced incorrect code cannot reliably verify it
because their errors are correlated. Running a *different* verifier
model mitigates this — but on local hardware, swapping a second model
on and off the GPU causes worse practical pain than the correlated-
error risk does. So the default is single-model, and the user opts
in to a verifier override per workspace when they have the resources
(cloud models, multi-GPU rigs).

We do NOT enforce ``verifier != fixer`` at construction. Instead we
expose ``same_model_self_verification`` so the orchestrator can stamp
the run report with which mode the audit ran in. Combined with the
``runs_to_confirm`` histogram, that gives users the data to assess
trust in the findings without forcing them into a configuration that
breaks on their hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    COMPREHENDER = "comprehender"
    PLANNER = "planner"
    DETECTOR = "detector"
    VERIFIER = "verifier"
    FIXER = "fixer"
    LEAD = "lead"               # CC-style top-level orchestrator
    INVESTIGATOR = "investigator"   # branches off finding threads
    PEN_TESTER = "pen_tester"   # dynamic-probe verification leg


@dataclass(frozen=True)
class RoleModelConfig:
    """Per-role model selection.

    Built from a primary model (used for planner/detector/fixer) plus an
    optional verifier override. When the verifier override is empty or
    equal to the primary model, ``same_model_self_verification`` is
    True — a fact the orchestrator records on the run report.
    """

    planner: str
    detector: str
    verifier: str
    fixer: str
    comprehender: str = ""  # falls back to planner when empty
    lead: str = ""          # falls back to planner — the lead's
                            # meta-reasoning needs a capable model
    investigator: str = ""  # falls back to detector — same scrutiny
                            # as detection, just across more code
    pen_tester: str = ""    # falls back to verifier — same role
                            # (active confirmation), different mechanism

    def __post_init__(self) -> None:
        for field_name in ("planner", "detector", "verifier", "fixer"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"role_models.{field_name} must be a non-empty model id; "
                    f"got {value!r}",
                )

    @property
    def same_model_self_verification(self) -> bool:
        """True when verifier == fixer (or planner/detector). Recorded on
        the report so users can interpret findings in context."""
        return self.verifier == self.fixer

    @classmethod
    def from_primary(
        cls, primary: str, *, verifier: str = "",
    ) -> RoleModelConfig:
        """Build a config from one default model plus an optional verifier
        override. The expected call-site shape: caller has the user's
        currently-selected model (``primary``) plus, optionally, a
        per-workspace verifier override (empty string when unset)."""
        primary = (primary or "").strip()
        if not primary:
            raise ValueError("from_primary requires a non-empty primary model")
        verifier_clean = (verifier or "").strip() or primary
        return cls(
            planner=primary,
            detector=primary,
            verifier=verifier_clean,
            fixer=primary,
        )

    @classmethod
    def from_primary_with_cross_family_verifier(
        cls,
        primary: str,
        *,
        available_models: tuple[str, ...] | list[str] = (),
        explicit_verifier: str = "",
    ) -> RoleModelConfig:
        """Build a config that AUTOMATICALLY selects a verifier from a
        different model family than ``primary``.

        Closes the same-family correlated-failure gap (arXiv 2604.07650)
        without requiring the caller to know which models are available
        — they pass the list of reachable model ids and the helper
        does the family math.

        ``explicit_verifier`` takes priority when supplied (caller's
        choice always wins). Otherwise ``pick_cross_family_verifier``
        chooses from ``available_models``; if none is cross-family,
        verifier falls back to ``primary`` and
        ``same_model_self_verification`` will be True on the resulting
        config (visible on the run report so the caller can interpret
        findings accordingly).
        """
        primary = (primary or "").strip()
        if not primary:
            raise ValueError(
                "from_primary_with_cross_family_verifier requires "
                "a non-empty primary model",
            )
        explicit = (explicit_verifier or "").strip()
        if explicit:
            chosen_verifier = explicit
        else:
            chosen_verifier = pick_cross_family_verifier(
                primary, available_models,
            )
        return cls(
            planner=primary,
            detector=primary,
            verifier=chosen_verifier,
            fixer=primary,
        )

    def for_role(self, role: Role) -> str:
        return {
            Role.COMPREHENDER: self.comprehender or self.planner,
            Role.PLANNER: self.planner,
            Role.DETECTOR: self.detector,
            Role.VERIFIER: self.verifier,
            Role.FIXER: self.fixer,
            Role.LEAD: self.lead or self.planner,
            Role.INVESTIGATOR: self.investigator or self.detector,
            Role.PEN_TESTER: self.pen_tester or self.verifier,
        }[role]


# ---------------------------------------------------------------------------
# Model-family classification (for cross-family confirmation tracking)
# ---------------------------------------------------------------------------
#
# The Plausible Completion Trap (above) extends to detection: two
# detector runs of the *same* model produce correlated errors, so
# stacking N runs of one model only weakly drives down FPs. Anthropic's
# bug-finder research measured the strongest precision gains when the
# ensemble crosses vendor families — Claude + GPT + Qwen flags the
# same bug = high confidence; Claude flagging it three times = mediocre
# confidence. We expose `families_to_confirm` on each Finding for
# exactly this reason.
#
# The mapping is deliberately coarse — we only need to know "would
# these models share training data and RLHF". A finer split (e.g.
# Mixtral vs. Mistral 7B) would track marginal differences but each
# class would have too few representatives to be useful.

_FAMILY_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("anthropic",  ("claude",)),
    ("openai",     ("gpt", "o1", "o3", "o4", "o5", "codex")),
    ("qwen",       ("qwen",)),
    ("deepseek",   ("deepseek", "deep-seek")),
    ("google",     ("gemini", "gemma", "palm")),
    ("mistral",    ("mistral", "codestral", "magistral", "ministral",
                    "mixtral", "minimax")),
    ("meta",       ("llama", "code-llama", "codellama")),
    ("xai",        ("grok",)),
    ("hunyuan",    ("hunyuan",)),
    ("nemotron",   ("nemotron", "nvidia/")),
    ("glm",        ("glm", "chatglm")),
    ("exaone",     ("exaone", "lg-ai")),
    ("kimi",       ("kimi", "moonshot")),
)


def family_for_model(model_id: str) -> str:
    """Coarse vendor-family label for `model_id`.

    Returns ``"unknown"`` when nothing matches — better than guessing.
    The matcher is lowercased prefix-based after stripping any
    ``@provider:`` peer suffix (fabric routing) and any leading
    ``a/`` / ``n/`` / ``p/`` mode prefixes.
    """
    raw = (model_id or "").strip().lower()
    if not raw:
        return "unknown"
    # Strip fabric peer suffix (e.g. "claude-opus-4-7@fabric:peerid")
    if "@" in raw:
        raw = raw.split("@", 1)[0]
    # Strip mode prefix (e.g. "a/claude-opus-4-7")
    for pref in ("a/", "n/", "p/", "d/"):
        if raw.startswith(pref):
            raw = raw[len(pref):]
            break
    for family, prefixes in _FAMILY_PREFIXES:
        for p in prefixes:
            if raw.startswith(p):
                return family
    return "unknown"


def families_for_models(model_ids: list[str] | tuple[str, ...]) -> list[str]:
    """Per-model family labels, preserving order + duplicates. Useful as
    the `families` argument to `merge_runs`."""
    return [family_for_model(m) for m in model_ids]


def pick_cross_family_verifier(
    primary: str,
    available_models: tuple[str, ...] | list[str],
) -> str:
    """Return a model from ``available_models`` whose family differs
    from ``primary``'s family. Returns ``primary`` when no cross-family
    candidate exists.

    Rationale: when the detector and verifier come from the same model
    family, agreement reflects shared blind spots, not validation.
    arXiv 2604.07650 measured Spearman 0.64-0.71 correlated-failure
    between same-family detector/verifier pairs. Routing verification
    across families closes that correlation.

    Selection rules:
    1. Skip any model in ``available_models`` that's the same family
       as ``primary``.
    2. Among the remaining cross-family models, prefer in this
       order: anthropic, openai, google, qwen — these are the families
       with the most distinct training distributions in published
       benchmarks. Other families are picked next in list order.
    3. If no cross-family model is found, return ``primary`` (the
       caller can flag ``same_model_self_verification`` on the run
       report so users see they're running in the higher-correlated-
       error mode).

    Args:
        primary: The model id used for detector/fixer.
        available_models: Iterable of candidate model ids the system
            can reach (from the bridge / provider registry).
    """
    primary_family = family_for_model(primary)
    candidates: list[tuple[str, str]] = []
    for m in available_models or ():
        if not m or m == primary:
            continue
        fam = family_for_model(m)
        if fam == primary_family or fam == "unknown":
            continue
        candidates.append((m, fam))

    if not candidates:
        return primary

    # Preferred-family ordering — first match wins per the rationale
    preferred_order = ("anthropic", "openai", "google", "qwen")
    for fam in preferred_order:
        for m, mfam in candidates:
            if mfam == fam:
                return m
    # Fall back to first candidate
    return candidates[0][0]
