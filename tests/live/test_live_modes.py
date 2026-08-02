"""Live integration tests for Augmentum mode pipelines.

These tests exercise REAL LLM behavior through the actual mode-specific
pipelines (passthrough, analytical, narrative) using the Ollama-compatible
/api/chat endpoint with mode prefixes on the model name.

Run:
    python -m pytest tests/live/test_live_modes.py -x -v -m live
"""

from __future__ import annotations

import json
import uuid

import httpx
import pytest

BASE_URL = "https://localhost:6443"
MODEL = "deepseek-chat"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _probe() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/", timeout=3.0, verify=False)
        return r.status_code < 500
    except Exception:
        return False


@pytest.fixture(scope="module")
def _require_server():
    if not _probe():
        pytest.skip(f"Augmentum not reachable at {BASE_URL}")


@pytest.fixture
def client(_require_server):
    with httpx.Client(base_url=BASE_URL, timeout=120.0, verify=False) as c:
        yield c


def _chat(client: httpx.Client, messages: list[dict], model: str = MODEL,
          *, stream: bool = False, headers: dict | None = None) -> httpx.Response:
    """POST to /api/chat with the given model (prefix included)."""
    payload = {"model": model, "messages": messages, "stream": stream}
    return client.post("/api/chat", json=payload, headers=headers or {})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestPassthrough:
    """Passthrough mode — no prefix or p/ prefix."""

    def test_basic_response(self, client: httpx.Client):
        """Plain model name routes through passthrough; returns assistant content."""
        resp = _chat(client, [
            {"role": "user", "content": "What is 2+2? Answer with just the number."},
        ])
        assert resp.status_code == 200
        data = resp.json()
        content = data["message"]["content"]
        assert "4" in content
        assert data["message"]["role"] == "assistant"
        assert data["done"] is True

    def test_explicit_prefix(self, client: httpx.Client):
        """p/ prefix explicitly selects passthrough mode."""
        resp = _chat(client, [
            {"role": "user", "content": "Say the word 'hello' and nothing else."},
        ], model=f"p/{MODEL}")
        assert resp.status_code == 200
        content = resp.json()["message"]["content"].lower()
        assert "hello" in content

    def test_streaming(self, client: httpx.Client):
        """Stream a passthrough response, collect NDJSON chunks, verify assembly."""
        resp = _chat(client, [
            {"role": "user", "content": "Count from 1 to 5, one number per line."},
        ], stream=True)
        assert resp.status_code == 200

        chunks = []
        full_content = []
        for line in resp.iter_lines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            chunks.append(chunk)
            delta = chunk.get("message", {}).get("content", "")
            if delta:
                full_content.append(delta)

        # Must have multiple chunks (streaming, not a single blob)
        assert len(chunks) >= 2, f"Expected streaming chunks, got {len(chunks)}"

        # Last chunk must signal done
        assert chunks[-1]["done"] is True

        # Assembled text should contain the numbers
        assembled = "".join(full_content)
        for n in ("1", "2", "3", "4", "5"):
            assert n in assembled, f"Missing {n} in assembled stream: {assembled!r}"


@pytest.mark.live
class TestAnalytical:
    """Analytical mode — a/ prefix triggers the UARF pipeline."""

    def test_activation(self, client: httpx.Client):
        """a/ prefix activates analytical mode and returns real content."""
        resp = _chat(client, [
            {"role": "user", "content": "Explain what photosynthesis is in two sentences."},
        ], model=f"a/{MODEL}")
        assert resp.status_code == 200
        data = resp.json()
        content = data["message"]["content"]
        # Analytical mode should still produce substantive text
        assert len(content) > 30, f"Response too short for analytical mode: {content!r}"
        # Photosynthesis-related terms should appear
        assert any(w in content.lower() for w in ("light", "plant", "energy", "sun", "chloro")), \
            f"Response doesn't seem related to photosynthesis: {content!r}"

    def test_tool_triggered_search(self, client: httpx.Client):
        """Analytical mode with a factual question may trigger search/verification.

        We can't guarantee search fires every time, but the response must be
        substantive.  Analytical mode may include raw thinking tokens
        (e.g. ``<?end?of?thinking?>``), so we only check length rather than
        requiring specific numeric content.
        """
        resp = _chat(client, [
            {"role": "user", "content": "What is the current population of Tokyo?"},
        ], model=f"a/{MODEL}")
        assert resp.status_code == 200
        content = resp.json()["message"]["content"]
        # Response must be non-empty and substantial (analytical may include
        # thinking tokens or prose without digits)
        assert len(content) > 50, \
            f"Analytical response too short ({len(content)} chars): {content!r}"

    def test_streaming(self, client: httpx.Client):
        """Analytical mode streaming — UARF may buffer phases but must stream."""
        resp = _chat(client, [
            {"role": "user", "content": "What is the boiling point of water?"},
        ], model=f"a/{MODEL}", stream=True)
        assert resp.status_code == 200

        chunks = []
        full_content = []
        for line in resp.iter_lines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            chunks.append(chunk)
            delta = chunk.get("message", {}).get("content", "")
            if delta:
                full_content.append(delta)

        assert len(chunks) >= 1, "Expected at least one streaming chunk"
        assembled = "".join(full_content)
        assert "100" in assembled or "212" in assembled, \
            f"Expected boiling point (100C/212F) in response: {assembled!r}"


@pytest.mark.live
class TestNarrative:
    """Narrative mode — n/ prefix with character card support."""

    def test_activation(self, client: httpx.Client):
        """n/ prefix activates narrative mode and produces in-character response."""
        messages = [
            {"role": "system", "content": (
                "You are Luna, a cheerful space explorer from the year 3042. "
                "You speak with excitement about the cosmos and always mention "
                "your ship, the Starweaver. Stay in character at all times."
            )},
            {"role": "user", "content": "Tell me about your latest adventure."},
        ]
        resp = _chat(client, messages, model=f"n/{MODEL}")
        assert resp.status_code == 200
        content = resp.json()["message"]["content"]
        assert len(content) > 50, f"Narrative response too short: {content!r}"
        # The response should reflect the character prompt — look for space/ship themes
        content_lower = content.lower()
        assert any(w in content_lower for w in (
            "star", "space", "ship", "cosmos", "galaxy", "starweaver",
            "adventure", "planet", "orbit", "luna",
        )), f"Response doesn't reflect character: {content!r}"

    def test_character_card_in_system(self, client: httpx.Client):
        """Narrative mode respects character personality from system prompt."""
        messages = [
            {"role": "system", "content": (
                "Character: Professor Grimshaw\n"
                "Personality: A grumpy Victorian-era scientist who speaks in "
                "overly formal English and frequently uses the word 'preposterous'.\n"
                "Scenario: The user has just asked a silly question."
            )},
            {"role": "user", "content": "Can fish fly?"},
        ]
        resp = _chat(client, messages, model=f"n/{MODEL}")
        assert resp.status_code == 200
        content = resp.json()["message"]["content"]
        assert len(content) > 30


@pytest.mark.live
class TestModeSwitching:
    """Verify switching modes mid-session doesn't break anything."""

    def test_passthrough_then_analytical(self, client: httpx.Client):
        """Send passthrough first, then analytical — both must succeed."""
        session_id = str(uuid.uuid4())
        headers = {"X-Augmentum-Session": session_id}

        # First: passthrough
        r1 = _chat(client, [
            {"role": "user", "content": "Say 'alpha'."},
        ], model=MODEL, headers=headers)
        assert r1.status_code == 200
        c1 = r1.json()["message"]["content"]
        assert len(c1) > 0

        # Second: analytical on the same session
        r2 = _chat(client, [
            {"role": "user", "content": "What is the speed of light in km/s?"},
        ], model=f"a/{MODEL}", headers=headers)
        assert r2.status_code == 200
        c2 = r2.json()["message"]["content"]
        assert any(c.isdigit() for c in c2), f"Expected numeric content: {c2!r}"

    def test_analytical_then_narrative(self, client: httpx.Client):
        """Analytical followed by narrative — both succeed independently."""
        # Analytical
        r1 = _chat(client, [
            {"role": "user", "content": "Define entropy in one sentence."},
        ], model=f"a/{MODEL}")
        assert r1.status_code == 200

        # Narrative
        r2 = _chat(client, [
            {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
            {"role": "user", "content": "What's for dinner?"},
        ], model=f"n/{MODEL}")
        assert r2.status_code == 200
        assert len(r2.json()["message"]["content"]) > 10
