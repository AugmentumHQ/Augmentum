"""Tests for the request classifier — narrative detection, complexity analysis, overrides."""

from __future__ import annotations

from augmentum.classifier.complexity_analyzer import ComplexityAnalyzer, ComplexityLevel
from augmentum.classifier.narrative_detector import NarrativeDetector
from augmentum.classifier.router import Mode, RequestClassifier
from augmentum.models.base import InternalChatRequest, Message

# --- Helpers ---


def make_request(
    user_content: str,
    system_content: str = "",
    model: str = "llama3.1:8b",
) -> InternalChatRequest:
    messages = []
    if system_content:
        messages.append(Message(role="system", content=system_content))
    messages.append(Message(role="user", content=user_content))
    return InternalChatRequest(model=model, messages=messages)


# === Narrative Detector Tests ===


class TestNarrativeDetector:
    def setup_method(self):
        self.detector = NarrativeDetector()

    def test_sillytavern_character_card(self):
        """SillyTavern character cards should be detected with high confidence."""
        req = make_request(
            "Hello!",
            system_content=(
                "{{char}} is a brave warrior from the northern lands. "
                "{{char}}'s personality is bold and fearless. "
                "{{user}} is a traveling merchant. "
                "{{scenario}}: You meet in a tavern during a storm."
            ),
        )
        result = self.detector.detect(req)
        assert result.confidence >= 0.7
        assert "sillytavern_char" in result.metadata.get("matched_patterns", [])

    def test_wpp_format(self):
        """W++ character definitions should be detected."""
        req = make_request(
            "Hi there",
            system_content='[character= "Elena"] [personality= "kind" + "brave" + "curious"]',
        )
        result = self.detector.detect(req)
        assert result.confidence >= 0.5
        assert "wpp_character" in result.metadata.get("matched_patterns", [])

    def test_plist_format(self):
        """PList-style character descriptions should be detected."""
        req = make_request(
            "Hello",
            system_content=(
                "Personality: Cheerful, optimistic, slightly mischievous\n"
                "Appearance: Tall elf with silver hair and green eyes\n"
                "Species: High Elf\n"
                "Background: Former court mage who left to explore the world"
            ),
        )
        result = self.detector.detect(req)
        assert result.confidence >= 0.6

    def test_cai_format(self):
        """Character.AI format should be detected."""
        req = make_request(
            "Hey",
            system_content=(
                "Name: Captain Aria Blackwood\n"
                "Greeting: *Aria adjusts her captain's hat* Welcome aboard!\n"
                "Example Dialogue:\n"
                "User: Where are we heading?\n"
                "Aria: To the edge of the known world!"
            ),
        )
        result = self.detector.detect(req)
        assert result.confidence >= 0.5

    def test_v2_json_markers(self):
        """Character Card V2 JSON format should be detected."""
        req = make_request(
            "Hello",
            system_content='{"spec": "chara_card_v2", "data": {"first_mes": "Hi!", "mes_example": ""}}',
        )
        result = self.detector.detect(req)
        assert result.confidence >= 0.7

    def test_rp_instructions(self):
        """Roleplay instructions should be detected."""
        req = make_request(
            "Let's begin",
            system_content=(
                "You are playing as a medieval knight. Stay in character at all times. "
                "Write as if you are in the world. Use *asterisks* for actions."
            ),
        )
        result = self.detector.detect(req)
        assert result.confidence >= 0.5

    def test_no_system_prompt(self):
        """No system prompt should return 0 confidence."""
        req = make_request("Hello, how are you?")
        result = self.detector.detect(req)
        assert result.confidence == 0.0

    def test_plain_system_prompt(self):
        """A plain assistant prompt should not trigger narrative detection."""
        req = make_request(
            "What's 2+2?",
            system_content="You are a helpful assistant.",
        )
        result = self.detector.detect(req)
        assert result.confidence < 0.5

    def test_long_character_card(self):
        """Long character cards with multiple signals should get high confidence."""
        req = make_request(
            "*waves*",
            system_content=(
                "{{char}} is Lyra, a 200-year-old elven sorceress.\n"
                "Personality: Wise, patient, slightly sardonic\n"
                "Appearance: Tall with silver hair that flows like moonlight\n"
                "{{char}} lives in a tower at the edge of the Whispering Forest.\n"
                "{{user}} is a young adventurer seeking {{char}}'s wisdom.\n"
                "{{scenario}}: {{user}} arrives at the tower during a thunderstorm.\n"
                "Stay in character at all times. Use *asterisks* for actions."
            ),
        )
        result = self.detector.detect(req)
        assert result.confidence >= 0.85


# === Complexity Analyzer Tests ===


class TestComplexityAnalyzer:
    def setup_method(self):
        self.analyzer = ComplexityAnalyzer()

    def test_simple_greeting(self):
        """Simple greetings should be classified as simple."""
        req = make_request("Hello!")
        result = self.analyzer.analyze(req)
        assert result.level == ComplexityLevel.SIMPLE

    def test_simple_question(self):
        """Basic factual questions should be simple."""
        req = make_request("What is the capital of France?")
        result = self.analyzer.analyze(req)
        assert result.level == ComplexityLevel.SIMPLE

    def test_analytical_request(self):
        """Analytical requests should be moderate or complex."""
        req = make_request(
            "Analyze the economic implications of rising interest rates "
            "on the housing market. Compare the effects in urban vs rural areas."
        )
        result = self.analyzer.analyze(req)
        assert result.level in (ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX)
        assert "analyze" in result.signals

    def test_research_request(self):
        """Research requests should be complex."""
        req = make_request(
            "Research the latest advances in quantum computing. "
            "Provide a comprehensive analysis of the pros and cons of different approaches. "
            "Compare superconducting qubits vs trapped ion vs photonic systems."
        )
        result = self.analyzer.analyze(req)
        assert result.level == ComplexityLevel.COMPLEX

    def test_math_notation(self):
        """Math notation should trigger complexity."""
        req = make_request("Solve the equation: \\frac{d}{dx} \\int_0^x f(t) dt = f(x)")
        result = self.analyzer.analyze(req)
        assert result.level in (ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX)
        assert "math_notation" in result.signals

    def test_code_request(self):
        """Code-related requests should increase complexity."""
        req = make_request(
            "Refactor this function to use async/await:\n"
            "```python\ndef fetch_data():\n    return requests.get(url)\n```"
        )
        result = self.analyzer.analyze(req)
        assert result.level in (ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX)

    def test_multi_question(self):
        """Multiple questions should boost complexity."""
        req = make_request(
            "What is machine learning? How does it differ from deep learning? "
            "What are the best frameworks? Which one should I use for NLP?"
        )
        result = self.analyzer.analyze(req)
        assert "multi_question" in result.signals

    def test_yes_no_simple(self):
        """Short affirmative responses should be simple."""
        req = make_request("yes")
        result = self.analyzer.analyze(req)
        assert result.level == ComplexityLevel.SIMPLE


# === Request Classifier (Router) Tests ===


class TestRequestClassifier:
    def setup_method(self):
        self.classifier = RequestClassifier()

    def test_default_passthrough(self):
        """Simple requests default to passthrough."""
        req = make_request("What time is it?")
        result = self.classifier.classify(req)
        assert result.mode == Mode.PASSTHROUGH

    def test_model_prefix_analytical(self):
        """a/ prefix forces analytical mode."""
        req = make_request("Hello", model="a/llama3.1:8b")
        result = self.classifier.classify(req)
        assert result.mode == Mode.ANALYTICAL
        assert result.confidence == 1.0
        assert req.model == "llama3.1:8b"  # Prefix stripped

    def test_model_prefix_narrative(self):
        """n/ prefix forces narrative mode."""
        req = make_request("Hello", model="n/llama3.1:8b")
        result = self.classifier.classify(req)
        assert result.mode == Mode.NARRATIVE
        assert req.model == "llama3.1:8b"

    def test_model_prefix_passthrough(self):
        """p/ prefix forces passthrough mode."""
        req = make_request(
            "Analyze this deeply",
            model="p/llama3.1:8b",
        )
        result = self.classifier.classify(req)
        assert result.mode == Mode.PASSTHROUGH
        assert result.confidence == 1.0

    def test_header_override(self):
        """X-Augmentum-Mode header overrides classification."""
        req = make_request("Hello")
        result = self.classifier.classify(req, mode_override="analytical")
        assert result.mode == Mode.ANALYTICAL
        assert result.confidence == 1.0

    def test_header_override_case_insensitive(self):
        """Header override is case-insensitive."""
        req = make_request("Hello")
        result = self.classifier.classify(req, mode_override="NARRATIVE")
        assert result.mode == Mode.NARRATIVE

    def test_character_card_routes_to_narrative(self):
        """Character cards should route to narrative mode."""
        req = make_request(
            "*waves hello*",
            system_content=(
                "{{char}} is a brave knight named Sir Aldric.\n"
                "{{user}} is a traveling merchant.\n"
                "{{scenario}}: Meeting at a crossroads.\n"
                "Stay in character. Use *asterisks* for actions."
            ),
        )
        result = self.classifier.classify(req)
        assert result.mode == Mode.NARRATIVE
        assert result.confidence >= 0.7

    def test_complex_query_routes_to_analytical(self):
        """Complex analytical queries should route to analytical mode."""
        req = make_request(
            "Research and analyze the comparative advantages of Rust vs Go "
            "for building distributed systems. Provide a comprehensive evaluation "
            "of performance, developer experience, and ecosystem maturity."
        )
        result = self.classifier.classify(req)
        assert result.mode == Mode.ANALYTICAL

    def test_session_continuity(self):
        """Session mode is respected for continuity."""
        req = make_request("Continue the story")
        result = self.classifier.classify(req, session_mode="narrative")
        assert result.mode == Mode.NARRATIVE

    def test_prefix_takes_priority_over_everything(self):
        """Model prefix overrides even character cards."""
        req = make_request(
            "*waves*",
            system_content="{{char}} is a knight. Stay in character.",
            model="p/llama3.1:8b",
        )
        result = self.classifier.classify(req)
        assert result.mode == Mode.PASSTHROUGH

    def test_header_takes_priority_over_content(self):
        """Header override takes priority over content analysis."""
        req = make_request(
            "*waves*",
            system_content="{{char}} is a knight. Stay in character.",
        )
        result = self.classifier.classify(req, mode_override="passthrough")
        assert result.mode == Mode.PASSTHROUGH

    def test_strip_model_prefix_utility(self):
        """strip_model_prefix correctly parses prefixed model names."""
        name, mode = RequestClassifier.strip_model_prefix("a/llama3.1:8b")
        assert name == "llama3.1:8b"
        assert mode == Mode.ANALYTICAL

        name, mode = RequestClassifier.strip_model_prefix("llama3.1:8b")
        assert name == "llama3.1:8b"
        assert mode is None


# === Integration test: tags include prefixed variants ===


def test_ollama_tags_include_prefixed_models(client):
    """GET /api/tags should include a/, n/, p/ prefixed model variants."""
    resp = client.get("/api/tags")
    assert resp.status_code == 200
    data = resp.json()
    names = [m["name"] for m in data["models"]]
    assert "llama3.1:8b" in names
    assert "a/llama3.1:8b" in names
    assert "n/llama3.1:8b" in names
    assert "p/llama3.1:8b" in names


def test_ollama_chat_with_prefix_strips_model(client):
    """Sending a/model should classify as analytical and strip the prefix."""
    resp = client.post(
        "/api/chat",
        json={
            "model": "a/llama3.1:8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )
    assert resp.status_code == 200
    # The response model should have the prefix stripped
    data = resp.json()
    assert data["model"] == "llama3.1:8b"
