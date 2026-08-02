"""Integration tests for RequestClassifier routing decisions."""

from __future__ import annotations

from augmentum.classifier.router import ClassificationResult, Mode, RequestClassifier
from augmentum.models.base import InternalChatRequest, Message


def _make_request(user_content: str, system_content: str = "", model: str = "llama3:8b") -> InternalChatRequest:
    messages = []
    if system_content:
        messages.append(Message(role="system", content=system_content))
    messages.append(Message(role="user", content=user_content))
    return InternalChatRequest(model=model, messages=messages)


class TestExplicitOverrides:
    """Test explicit mode overrides via model prefix and header."""

    def test_analytical_prefix(self):
        rc = RequestClassifier()
        req = _make_request("hello", model="a/llama3:8b")
        result = rc.classify(req)
        assert result.mode == Mode.ANALYTICAL
        assert result.confidence == 1.0
        assert req.model == "llama3:8b"  # prefix stripped

    def test_narrative_prefix(self):
        rc = RequestClassifier()
        req = _make_request("hello", model="n/llama3:8b")
        result = rc.classify(req)
        assert result.mode == Mode.NARRATIVE

    def test_passthrough_prefix(self):
        rc = RequestClassifier()
        req = _make_request("hello", model="p/llama3:8b")
        result = rc.classify(req)
        assert result.mode == Mode.PASSTHROUGH

    def test_agentic_prefix(self):
        rc = RequestClassifier()
        req = _make_request("hello", model="g/llama3:8b")
        result = rc.classify(req)
        assert result.mode == Mode.AGENTIC

    def test_coder_prefix(self):
        rc = RequestClassifier()
        req = _make_request("hello", model="c/llama3:8b")
        result = rc.classify(req)
        assert result.mode == Mode.CODER

    def test_header_override_analytical(self):
        rc = RequestClassifier()
        req = _make_request("hello")
        result = rc.classify(req, mode_override="analytical")
        assert result.mode == Mode.ANALYTICAL
        assert result.confidence == 1.0

    def test_header_override_narrative(self):
        rc = RequestClassifier()
        req = _make_request("hello")
        result = rc.classify(req, mode_override="narrative")
        assert result.mode == Mode.NARRATIVE

    def test_header_override_case_insensitive(self):
        rc = RequestClassifier()
        req = _make_request("hello")
        result = rc.classify(req, mode_override="ANALYTICAL")
        assert result.mode == Mode.ANALYTICAL


class TestNarrativeDetection:
    """Test narrative/RP content triggers narrative mode."""

    def test_sillytavern_character_card(self):
        rc = RequestClassifier()
        system = (
            "You are {{char}}, a brave warrior. {{user}} approaches you in the tavern. "
            "Stay in character at all times. Personality: bold, confident."
        )
        req = _make_request("Hello there!", system_content=system)
        result = rc.classify(req)
        assert result.mode == Mode.NARRATIVE

    def test_wpp_format(self):
        rc = RequestClassifier()
        system = '[character= "Luna"] [personality= "kind, gentle, wise"]'
        req = _make_request("Hi Luna!", system_content=system)
        result = rc.classify(req)
        assert result.mode == Mode.NARRATIVE

    def test_v2_json_card(self):
        rc = RequestClassifier()
        system = '{"spec": "chara_card_v2", "first_mes": "Hello traveler", "character_book": {}}'
        req = _make_request("Greetings", system_content=system)
        result = rc.classify(req)
        assert result.mode == Mode.NARRATIVE

    def test_rp_instruction(self):
        rc = RequestClassifier()
        system = (
            "You roleplay as a medieval knight. Stay in character. "
            "Write as if you are speaking in first person. "
            "Setting: A dark forest at midnight."
        )
        req = _make_request("Who goes there?", system_content=system)
        result = rc.classify(req)
        assert result.mode == Mode.NARRATIVE


class TestAnalyticalRouting:
    """Test complex analytical queries route to analytical mode."""

    def test_research_query(self):
        rc = RequestClassifier()
        req = _make_request(
            "Research and analyze the differences between React and Vue. "
            "Compare their performance characteristics step-by-step and "
            "evaluate the pros and cons of each framework comprehensively."
        )
        result = rc.classify(req)
        assert result.mode == Mode.ANALYTICAL

    def test_math_problem(self):
        rc = RequestClassifier()
        req = _make_request(
            "Solve this equation step-by-step: \\frac{d}{dx}(x^2 + 3x) "
            "and verify the derivative using the limit definition."
        )
        result = rc.classify(req)
        assert result.mode == Mode.ANALYTICAL

    def test_code_analysis_request(self):
        rc = RequestClassifier()
        req = _make_request(
            "Analyze and refactor this code. Debug the performance issue "
            "and investigate why the algorithm runs slowly. Compare "
            "different optimization strategies:\n```python\ndef slow(): pass\n```"
        )
        result = rc.classify(req)
        assert result.mode == Mode.ANALYTICAL


class TestPassthroughDefault:
    """Test simple queries fall through to passthrough."""

    def test_greeting(self):
        rc = RequestClassifier()
        req = _make_request("hi")
        result = rc.classify(req)
        assert result.mode == Mode.PASSTHROUGH

    def test_simple_question(self):
        rc = RequestClassifier()
        req = _make_request("What is Python?")
        result = rc.classify(req)
        assert result.mode == Mode.PASSTHROUGH

    def test_empty_messages(self):
        rc = RequestClassifier()
        req = InternalChatRequest(model="test", messages=[])
        result = rc.classify(req)
        assert result.mode == Mode.PASSTHROUGH

    def test_no_system_no_complexity(self):
        rc = RequestClassifier()
        req = _make_request("thanks")
        result = rc.classify(req)
        assert result.mode == Mode.PASSTHROUGH


class TestSessionContinuity:
    """Test session mode continuity."""

    def test_stays_in_analytical(self):
        rc = RequestClassifier()
        req = _make_request("ok, continue")
        result = rc.classify(req, session_mode="analytical")
        assert result.mode == Mode.ANALYTICAL
        assert "session continuity" in result.reason

    def test_stays_in_narrative(self):
        rc = RequestClassifier()
        req = _make_request("I draw my sword")
        result = rc.classify(req, session_mode="narrative")
        assert result.mode == Mode.NARRATIVE

    def test_passthrough_session_doesnt_stick(self):
        rc = RequestClassifier()
        req = _make_request("hello")
        result = rc.classify(req, session_mode="passthrough")
        assert result.mode == Mode.PASSTHROUGH


class TestFlowCommand:
    """Test /flow command always routes to passthrough."""

    def test_flow_command(self):
        rc = RequestClassifier()
        req = _make_request("/flow my-workflow")
        result = rc.classify(req)
        assert result.mode == Mode.PASSTHROUGH
        assert "flow" in result.reason

    def test_bare_flow_command(self):
        rc = RequestClassifier()
        req = _make_request("/flow")
        result = rc.classify(req)
        assert result.mode == Mode.PASSTHROUGH

    def test_flow_overrides_prefix(self):
        rc = RequestClassifier()
        req = InternalChatRequest(
            model="a/llama3:8b",
            messages=[Message(role="user", content="/flow test")],
        )
        result = rc.classify(req)
        # /flow is checked before prefix
        assert result.mode == Mode.PASSTHROUGH


class TestClassificationResultShape:
    """Verify the shape of ClassificationResult across all paths."""

    def test_result_has_required_fields(self):
        rc = RequestClassifier()
        req = _make_request("hello")
        result = rc.classify(req)
        assert isinstance(result, ClassificationResult)
        assert isinstance(result.mode, Mode)
        assert isinstance(result.confidence, float)
        assert isinstance(result.reason, str)
        assert 0.0 <= result.confidence <= 1.0

    def test_metadata_on_narrative(self):
        rc = RequestClassifier()
        system = "You are {{char}}. {{user}} is here. Stay in character."
        req = _make_request("hello", system_content=system)
        result = rc.classify(req)
        if result.mode == Mode.NARRATIVE:
            assert "matched_patterns" in result.metadata

    def test_metadata_on_analytical(self):
        rc = RequestClassifier()
        req = _make_request(
            "Analyze and compare the pros and cons step-by-step. "
            "Research the implications comprehensively."
        )
        result = rc.classify(req)
        if result.mode == Mode.ANALYTICAL:
            assert "complexity" in result.metadata
