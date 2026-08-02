"""Smoke tests — import and construct every module in augmentum/classifier/."""

from __future__ import annotations

from augmentum.models.base import InternalChatRequest, Message


class TestRequestClassifierConstruct:
    """Verify RequestClassifier can be constructed."""

    def test_construct(self):
        from augmentum.classifier.router import RequestClassifier

        rc = RequestClassifier()
        assert rc._narrative_detector is not None
        assert rc._complexity_analyzer is not None

    def test_mode_enum_values(self):
        from augmentum.classifier.router import Mode

        assert Mode.PASSTHROUGH.value == "passthrough"
        assert Mode.ANALYTICAL.value == "analytical"
        assert Mode.NARRATIVE.value == "narrative"
        assert Mode.AGENTIC.value == "agentic"
        assert Mode.CODER.value == "coder"

    def test_classification_result_defaults(self):
        from augmentum.classifier.router import ClassificationResult, Mode

        result = ClassificationResult(
            mode=Mode.PASSTHROUGH,
            confidence=1.0,
            reason="test",
        )
        assert result.metadata == {}

    def test_mode_prefixes(self):
        from augmentum.classifier.router import MODE_PREFIXES, Mode

        assert MODE_PREFIXES["a/"] == Mode.ANALYTICAL
        assert MODE_PREFIXES["n/"] == Mode.NARRATIVE
        assert MODE_PREFIXES["p/"] == Mode.PASSTHROUGH

    def test_strip_model_prefix(self):
        from augmentum.classifier.router import Mode, RequestClassifier

        name, mode = RequestClassifier.strip_model_prefix("a/llama3:8b")
        assert name == "llama3:8b"
        assert mode == Mode.ANALYTICAL

    def test_strip_model_prefix_no_prefix(self):
        from augmentum.classifier.router import RequestClassifier

        name, mode = RequestClassifier.strip_model_prefix("llama3:8b")
        assert name == "llama3:8b"
        assert mode is None


class TestNarrativeDetectorConstruct:
    """Verify NarrativeDetector constructs and has expected interface."""

    def test_construct(self):
        from augmentum.classifier.narrative_detector import NarrativeDetector

        nd = NarrativeDetector()
        assert nd is not None

    def test_detect_no_system_prompt(self):
        from augmentum.classifier.narrative_detector import NarrativeDetector

        nd = NarrativeDetector()
        req = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="hello")],
        )
        result = nd.detect(req)
        assert result.confidence == 0.0

    def test_detect_sillytavern_pattern(self):
        from augmentum.classifier.narrative_detector import NarrativeDetector

        nd = NarrativeDetector()
        req = InternalChatRequest(
            model="test",
            messages=[
                Message(role="system", content="You are {{char}}. {{user}} talks to you. Stay in character."),
                Message(role="user", content="Hello there!"),
            ],
        )
        result = nd.detect(req)
        assert result.confidence > 0.5

    def test_detection_result_fields(self):
        from augmentum.classifier.narrative_detector import NarrativeDetection

        nd = NarrativeDetection(confidence=0.8, reason="test")
        assert nd.metadata == {}


class TestComplexityAnalyzerConstruct:
    """Verify ComplexityAnalyzer constructs and has expected interface."""

    def test_construct(self):
        from augmentum.classifier.complexity_analyzer import ComplexityAnalyzer

        ca = ComplexityAnalyzer()
        assert ca is not None

    def test_complexity_level_enum(self):
        from augmentum.classifier.complexity_analyzer import ComplexityLevel

        assert ComplexityLevel.SIMPLE.value == "simple"
        assert ComplexityLevel.MODERATE.value == "moderate"
        assert ComplexityLevel.COMPLEX.value == "complex"

    def test_analyze_simple(self):
        from augmentum.classifier.complexity_analyzer import ComplexityAnalyzer, ComplexityLevel

        ca = ComplexityAnalyzer()
        req = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="hi")],
        )
        result = ca.analyze(req)
        assert result.level == ComplexityLevel.SIMPLE

    def test_analyze_complex(self):
        from augmentum.classifier.complexity_analyzer import ComplexityAnalyzer, ComplexityLevel

        ca = ComplexityAnalyzer()
        req = InternalChatRequest(
            model="test",
            messages=[
                Message(
                    role="user",
                    content=(
                        "Analyze and compare the pros and cons of React vs Vue "
                        "step-by-step. Research the performance implications and "
                        "evaluate the ecosystem comprehensively."
                    ),
                ),
            ],
        )
        result = ca.analyze(req)
        assert result.level == ComplexityLevel.COMPLEX

    def test_analyze_no_user_content(self):
        from augmentum.classifier.complexity_analyzer import ComplexityAnalyzer, ComplexityLevel

        ca = ComplexityAnalyzer()
        req = InternalChatRequest(
            model="test",
            messages=[Message(role="system", content="You are helpful.")],
        )
        result = ca.analyze(req)
        assert result.level == ComplexityLevel.SIMPLE
