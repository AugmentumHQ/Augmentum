"""Request classifier — routes requests to the appropriate processing mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from augmentum.classifier.complexity_analyzer import ComplexityAnalyzer, ComplexityLevel
from augmentum.classifier.narrative_detector import NarrativeDetector
from augmentum.models.base import InternalChatRequest
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class Mode(str, Enum):
    PASSTHROUGH = "passthrough"
    ANALYTICAL = "analytical"
    NARRATIVE = "narrative"
    AGENTIC = "agentic"
    CODER = "coder"
    # Becca-direct: chat routes through the companion's own prompt
    # composer + tier stream. Only reachable when the chat router
    # picks the ``becca_direct`` subagent. The classifier itself
    # never returns this — explicit override or dispatch wins it.
    BECCA_DIRECT = "becca_direct"
    # Direct: external-API "raw pipe" tier. EVERY Augmentum injector
    # at the route level is skipped — no memory recall, no knowledge
    # packs, no dream context, no media context, no vision caption
    # fallback, no file-token expansion, no SSOS, no datetime
    # injection. Backend resolution + auth + telemetry only. Intended
    # for external API-key clients (Claude Code / Aider / Cline /
    # external coding tools / headless integrations) that want a
    # zero-overhead path to the underlying model. Functionally the
    # opposite of BECCA_DIRECT (which routes ALL the way through the
    # companion) — same enum, different semantics, the comments here
    # exist to keep the two distinct in reviewers' minds.
    # Explicit-only: classifier never returns this from heuristics.
    DIRECT = "direct"


@dataclass
class ClassificationResult:
    mode: Mode
    confidence: float
    reason: str
    metadata: dict = field(default_factory=dict)


# Model name prefixes for explicit mode override
MODE_PREFIXES = {
    "p/": Mode.PASSTHROUGH,
    "a/": Mode.ANALYTICAL,
    "n/": Mode.NARRATIVE,
    "g/": Mode.AGENTIC,
    "c/": Mode.CODER,
    "d/": Mode.DIRECT,
}

# Header for explicit mode override
MODE_HEADER = "X-Augmentum-Mode"

# Mode string mapping for header values
MODE_MAP = {
    "passthrough": Mode.PASSTHROUGH,
    "analytical": Mode.ANALYTICAL,
    "narrative": Mode.NARRATIVE,
    "agentic": Mode.AGENTIC,
    "coder": Mode.CODER,
    "becca_direct": Mode.BECCA_DIRECT,
    "direct": Mode.DIRECT,
}


class RequestClassifier:
    """Classifies incoming requests to determine processing mode.

    Priority chain:
    1. Explicit override (model name prefix or X-Augmentum-Mode header)
    2. System prompt structural analysis (character card patterns)
    3. Content heuristics (complexity, research triggers)
    4. Session history (what mode was this session using?)
    5. Default: passthrough
    """

    def __init__(self) -> None:
        self._narrative_detector = NarrativeDetector()
        self._complexity_analyzer = ComplexityAnalyzer()

    def classify(
        self,
        request: InternalChatRequest,
        *,
        mode_override: str | None = None,
        session_mode: str | None = None,
        default_mode: str | None = None,
    ) -> ClassificationResult:
        """Classify a request and return the recommended mode.

        Args:
            request: The internal chat request to classify.
            mode_override: Explicit mode from header (X-Augmentum-Mode).
            session_mode: Current mode of the session (for continuity).
            default_mode: User-pinned default mode from user_settings
                (``default_mode`` key). Wins over content heuristics +
                session continuity so a user who pinned DIRECT isn't
                auto-promoted to NARRATIVE by a system prompt pattern.
                Loses to explicit per-request override (prefix/header).
        """
        # 0. /flow command — always passthrough (needs flow store)
        flow_result = self._check_flow_command(request)
        if flow_result:
            return flow_result

        # 1. Explicit model name prefix
        prefix_result = self._check_model_prefix(request)
        if prefix_result:
            return prefix_result

        # 2. Explicit header override
        if mode_override:
            mode = MODE_MAP.get(mode_override.lower())
            if mode:
                return ClassificationResult(
                    mode=mode,
                    confidence=1.0,
                    reason=f"explicit header override: {mode_override}",
                )

        # 3. User-pinned default — beats heuristics so the pin
        #    actually sticks. Caller decides what value to thread in
        #    (typically user_settings['default_mode']); empty/unknown
        #    falls through. Confidence is 0.95: lower than explicit
        #    override but higher than heuristic detection.
        if default_mode:
            pinned = MODE_MAP.get(default_mode.lower())
            if pinned:
                return ClassificationResult(
                    mode=pinned,
                    confidence=0.95,
                    reason=f"user pinned default_mode: {default_mode}",
                )

        # 4. Narrative detection (system prompt analysis)
        narrative_result = self._narrative_detector.detect(request)
        if narrative_result.confidence >= 0.7:
            return ClassificationResult(
                mode=Mode.NARRATIVE,
                confidence=narrative_result.confidence,
                reason=narrative_result.reason,
                metadata=narrative_result.metadata,
            )

        # 5. Complexity analysis (content heuristics)
        complexity = self._complexity_analyzer.analyze(request)
        if complexity.level == ComplexityLevel.COMPLEX:
            return ClassificationResult(
                mode=Mode.ANALYTICAL,
                confidence=complexity.confidence,
                reason=complexity.reason,
                metadata={"complexity": complexity.level.value, "signals": complexity.signals},
            )

        # 6. Session continuity — stay in current mode if not passthrough
        if session_mode and session_mode != Mode.PASSTHROUGH.value:
            mode = MODE_MAP.get(session_mode, Mode.PASSTHROUGH)
            return ClassificationResult(
                mode=mode,
                confidence=0.6,
                reason=f"session continuity: {session_mode}",
            )

        # 7. Moderate complexity with some analytical signals
        if complexity.level == ComplexityLevel.MODERATE and complexity.confidence >= 0.6:
            return ClassificationResult(
                mode=Mode.ANALYTICAL,
                confidence=complexity.confidence * 0.8,
                reason=complexity.reason,
                metadata={"complexity": complexity.level.value, "signals": complexity.signals},
            )

        # 8. Default: passthrough
        return ClassificationResult(
            mode=Mode.PASSTHROUGH,
            confidence=1.0,
            reason="default passthrough",
        )

    @staticmethod
    def _check_flow_command(request: InternalChatRequest) -> ClassificationResult | None:
        """Route ``/flow <name>`` messages to passthrough mode."""
        if not request.messages:
            return None
        for msg in reversed(request.messages):
            if msg.role == "user":
                text = (msg.content or "").strip().lower()
                if text == "/flow" or text.startswith("/flow "):
                    return ClassificationResult(
                        mode=Mode.PASSTHROUGH,
                        confidence=1.0,
                        reason="explicit /flow command",
                    )
                break
        return None

    def _check_model_prefix(self, request: InternalChatRequest) -> ClassificationResult | None:
        """Check if the model name has a mode prefix (e.g., 'a/llama3.1:8b')."""
        model = request.model
        for prefix, mode in MODE_PREFIXES.items():
            if model.startswith(prefix):
                # Strip the prefix from the model name
                request.model = model[len(prefix):]
                return ClassificationResult(
                    mode=mode,
                    confidence=1.0,
                    reason=f"model prefix override: {prefix}",
                    metadata={"original_model": model},
                )
        return None

    @staticmethod
    def strip_model_prefix(model_name: str) -> tuple[str, Mode | None]:
        """Strip mode prefix from model name and return (clean_name, mode).

        Useful for /api/tags augmentation.
        """
        for prefix, mode in MODE_PREFIXES.items():
            if model_name.startswith(prefix):
                return model_name[len(prefix):], mode
        return model_name, None
