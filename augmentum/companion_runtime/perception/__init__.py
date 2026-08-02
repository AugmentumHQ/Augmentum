"""Perception layer modules.

- ``topical`` — distill observer.recent into named threads (Sprint 2).
- ``pad`` — project facet activations onto Pleasure/Arousal/Dominance (Sprint 6).
- ``affect`` — baseline distributions + texture detection (Lane 2 §1).
- ``user_affect`` — per-user affect tracker (in-memory).
- ``insight`` / ``budget`` / ``judgment`` — the Sovereign Perception Pipeline's
  L3 judgment gate: turn fused insights into delivery decisions (silent / pull /
  speak / act) under a regret-gated interruption budget. The thing that keeps
  proactive data aware, not an echo machine
  (``docs/superpowers/specs/2026-06-25-sovereign-perception-pipeline-design.md``).
"""

from __future__ import annotations

from augmentum.companion_runtime.perception.affect import (
    CandidateNotice,
    BaselineDistribution,
    build_baseline,
    load_baseline,
    perceive,
    rebuild_all_baselines,
)
from augmentum.companion_runtime.perception.budget import (
    BUDGET,
    DEFAULT_CAP,
    InterruptionBudgetStore,
    can_spend,
    remaining,
)
from augmentum.companion_runtime.perception.companion_sink import (
    CompanionPerceptionSink,
)
from augmentum.companion_runtime.perception.fusion import (
    FusionContext,
    Fuser,
    clear_fusers,
    fuse,
    register_fuser,
    registered_fusers,
)
from augmentum.companion_runtime.perception.insight import (
    ACT_WITH_CONSENT,
    FILE_FOR_PULL,
    SILENT,
    SPEAK,
    DeliveryDecision,
    Insight,
)
from augmentum.companion_runtime.perception.judgment import (
    JudgmentConfig,
    config_from_settings,
    decide_delivery,
)
from augmentum.companion_runtime.perception.live import (
    evaluate_user,
    perception_enabled,
)
from augmentum.companion_runtime.perception.pipeline import (
    PerceptionSink,
    dispatch,
    perceive_and_dispatch,
    run_perception,
)
from augmentum.companion_runtime.perception.pad import (
    PAD,
    project_pad,
)
from augmentum.companion_runtime.perception.topical import (
    Thread,
    aggregate_threads,
)

__all__ = [
    "Thread", "aggregate_threads",
    "PAD", "project_pad",
    "BaselineDistribution", "CandidateNotice",
    "build_baseline", "load_baseline",
    "rebuild_all_baselines", "perceive",
    # Sovereign Perception Pipeline — L3 judgment gate
    "Insight", "DeliveryDecision",
    "SILENT", "FILE_FOR_PULL", "SPEAK", "ACT_WITH_CONSENT",
    "decide_delivery", "JudgmentConfig", "config_from_settings",
    "InterruptionBudgetStore", "BUDGET", "can_spend", "remaining", "DEFAULT_CAP",
    # L1+L2 fusion + the orchestrator
    "FusionContext", "Fuser", "fuse", "register_fuser", "clear_fusers",
    "registered_fusers",
    "PerceptionSink", "run_perception", "dispatch", "perceive_and_dispatch",
    # live wiring — sink onto the initiative queue/bus + the runtime adapter
    "CompanionPerceptionSink", "evaluate_user", "perception_enabled",
]
