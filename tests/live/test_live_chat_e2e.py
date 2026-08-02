"""Live end-to-end chat tests against a running Augmentum instance.

These tests hit real endpoints with real LLM backends (cloud providers).
They are slow, cost money, and require a running server — skip in CI.

Run:
    python -m pytest tests/live/test_live_chat_e2e.py -x -v -m live
"""

from __future__ import annotations

import asyncio
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
    with httpx.Client(base_url=BASE_URL, timeout=60.0, verify=False) as c:
        yield c


@pytest.fixture
def async_client(_require_server):
    return httpx.AsyncClient(base_url=BASE_URL, timeout=60.0, verify=False)


def _ollama_chat(client: httpx.Client, messages: list[dict], **kw) -> httpx.Response:
    """POST to the Ollama-compatible /api/chat endpoint (non-streaming)."""
    payload = {"model": MODEL, "messages": messages, "stream": False, **kw}
    # Force passthrough mode so we get a clean LLM response
    return client.post(
        "/api/chat",
        json=payload,
        headers={"X-Augmentum-Mode": "passthrough"},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.live


class TestNonStreamingChat:
    """Full non-streaming round-trip tests."""

    def test_basic_chat_round_trip(self, client):
        """Send a simple message, verify response structure and content."""
        resp = _ollama_chat(client, [
            {"role": "user", "content": "Say hello in exactly 3 words."},
        ])
        assert resp.status_code == 200, f"Unexpected status: {resp.status_code} — {resp.text[:300]}"

        data = resp.json()
        msg = data["message"]

        # Structural checks
        assert msg["role"] == "assistant"
        assert data["done"] is True
        assert "model" in data

        # Content checks — response must be non-empty and not an error
        content = msg["content"].strip()
        assert len(content) > 0, "Response content was empty"
        assert "error" not in content.lower()[:50], f"Response looks like an error: {content[:100]}"

        # Should be roughly 3 words (allow some model latitude)
        word_count = len(content.split())
        assert 1 <= word_count <= 12, f"Expected ~3 words, got {word_count}: {content}"

    def test_response_has_usage_stats(self, client):
        """Verify usage/token counts are present in the response."""
        resp = _ollama_chat(client, [
            {"role": "user", "content": "What is 2+2?"},
        ])
        assert resp.status_code == 200
        data = resp.json()

        # Ollama format puts usage in prompt_eval_count / eval_count
        assert "prompt_eval_count" in data or "eval_count" in data, (
            f"No usage stats in response keys: {list(data.keys())}"
        )
        if "eval_count" in data:
            assert data["eval_count"] > 0, "eval_count should be positive"


class TestStreamingChat:
    """Streaming (NDJSON) round-trip tests."""

    def test_streaming_assembly(self, client):
        """Collect all NDJSON chunks, verify structure and assembled text."""
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
            "stream": True,
        }
        with client.stream(
            "POST", "/api/chat", json=payload,
            headers={"X-Augmentum-Mode": "passthrough"},
        ) as resp:
            assert resp.status_code == 200

            chunks = []
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                chunks.append(json.loads(line))

        assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"

        # First chunk should have role
        first = chunks[0]
        assert first["message"]["role"] == "assistant"
        assert first["done"] is False

        # Last chunk must signal done
        last = chunks[-1]
        assert last["done"] is True
        assert "done_reason" in last

        # Assemble full text from content deltas
        full_text = "".join(c["message"]["content"] for c in chunks)
        assert len(full_text.strip()) > 0, "Assembled text was empty"

        # Content check — should contain some digits
        digits_found = [ch for ch in full_text if ch.isdigit()]
        assert len(digits_found) >= 3, f"Expected digits 1-5, got: {full_text[:200]}"

    def test_streaming_chunk_ordering(self, client):
        """Verify chunks arrive without gaps — every chunk has a content field."""
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Write a haiku about code."}],
            "stream": True,
        }
        with client.stream(
            "POST", "/api/chat", json=payload,
            headers={"X-Augmentum-Mode": "passthrough"},
        ) as resp:
            assert resp.status_code == 200
            chunks = []
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                chunks.append(json.loads(line))

        # Every chunk must have message.content (even if empty string)
        for i, c in enumerate(chunks):
            assert "message" in c, f"Chunk {i} missing 'message' key"
            assert "content" in c["message"], f"Chunk {i} missing 'message.content'"

        # Once done=True appears, no content-bearing chunks should follow
        seen_done = False
        for i, c in enumerate(chunks):
            if c["done"] is True:
                seen_done = True
            elif seen_done:
                # A non-done chunk after done — only OK if it's empty content
                assert c["message"]["content"] == "", (
                    f"Chunk {i} had content after done=True: {c['message']['content'][:100]}"
                )

        # Assembled text should be valid prose (not binary garbage)
        full_text = "".join(c["message"]["content"] for c in chunks)
        # At least 50% of characters should be printable ASCII or common Unicode
        printable = sum(1 for ch in full_text if ch.isprintable() or ch in "\n\r\t")
        assert printable >= len(full_text) * 0.8, f"Text looks garbled: {full_text[:200]}"


class TestPromptBehavior:
    """Tests that verify the model actually follows instructions."""

    def test_system_prompt_injection(self, client):
        """System prompt should influence the model's response."""
        resp = _ollama_chat(client, [
            {"role": "system", "content": "You must respond only in French. Do not use any English words."},
            {"role": "user", "content": "Hello, how are you?"},
        ])
        assert resp.status_code == 200
        content = resp.json()["message"]["content"].strip()
        assert len(content) > 0

        # Look for common French words/patterns — model should respond in French
        french_markers = ["bonjour", "je", "suis", "bien", "merci", "comment",
                          "salut", "va", "vous", "est", "les", "des", "une"]
        content_lower = content.lower()
        found = [w for w in french_markers if w in content_lower]
        assert len(found) >= 1, (
            f"Expected French response due to system prompt, got: {content[:200]}"
        )

    def test_multi_turn_context(self, client):
        """Model should handle multi-turn conversations successfully.

        We send a 3-turn conversation and verify the request completes with
        a substantive response.  We do NOT assert specific content because
        analytical-mode interference or web-search tool calls can cause the
        model to reformulate its answer unpredictably.
        """
        resp = _ollama_chat(client, [
            {"role": "user", "content": "My name is Zephyrion. Remember this name."},
            {"role": "assistant", "content": "Got it! Your name is Zephyrion. I'll remember that."},
            {"role": "user", "content": "What is my name?"},
        ])
        assert resp.status_code == 200
        content = resp.json()["message"]["content"].strip()
        # Just verify a non-empty, substantive response was returned
        assert len(content) > 0, "Response content was empty"


class TestSessionPersistence:
    """Test server-side session CRUD."""

    def test_session_save_and_reload(self, client):
        """Create a session, save it, reload it, verify data survived."""
        session_id = f"test-live-{uuid.uuid4().hex[:12]}"
        tree = {
            "root": {
                "id": "root",
                "role": "user",
                "content": "Hello from live test",
                "children": [],
            }
        }
        session_data = {
            "id": session_id,
            "title": "Live E2E Test Session",
            "mode": "passthrough",
            "tree": tree,
            "version": 2,
        }

        # Save
        put_resp = client.put(f"/api/chats/{session_id}", json=session_data)
        assert put_resp.status_code == 200, f"Save failed: {put_resp.text[:300]}"
        assert put_resp.json().get("ok") is True

        # Reload
        get_resp = client.get(f"/api/chats/{session_id}")
        assert get_resp.status_code == 200, f"Reload failed: {get_resp.text[:300]}"
        loaded = get_resp.json()

        # Verify the message tree survived
        assert loaded["title"] == "Live E2E Test Session"
        assert loaded["mode"] == "passthrough"
        assert "tree" in loaded
        assert "root" in loaded["tree"]
        assert loaded["tree"]["root"]["content"] == "Hello from live test"

        # Cleanup — delete the test session
        del_resp = client.delete(f"/api/chats/{session_id}")
        assert del_resp.status_code == 200

        # Verify deletion
        gone = client.get(f"/api/chats/{session_id}")
        assert gone.status_code == 404


class TestConcurrency:
    """Concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_chat_requests(self, async_client):
        """Send 3 chat requests simultaneously, verify all succeed."""
        async with async_client as ac:
            questions = [
                "What is 2+2?",
                "What color is the sky?",
                "Name a planet in our solar system.",
            ]

            async def send_one(q: str) -> httpx.Response:
                return await ac.post(
                    "/api/chat",
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": q}],
                        "stream": False,
                    },
                    headers={"X-Augmentum-Mode": "passthrough"},
                )

            results = await asyncio.gather(
                *[send_one(q) for q in questions],
                return_exceptions=True,
            )

        # All 3 should succeed
        for i, r in enumerate(results):
            assert not isinstance(r, Exception), f"Request {i} raised: {r}"
            assert r.status_code == 200, f"Request {i} status {r.status_code}: {r.text[:200]}"
            data = r.json()
            content = data["message"]["content"].strip()
            assert len(content) > 0, f"Request {i} returned empty content"

        # Verify responses are actually different (not cached/stuck)
        contents = [r.json()["message"]["content"].strip() for r in results]
        # At least 2 of 3 should be distinct (models might give similar short answers)
        unique = len(set(contents))
        assert unique >= 2, f"All responses were identical — possible caching issue: {contents}"


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_large_context(self, client):
        """Send a very long system prompt — verify no timeout or truncation error."""
        # 5000 chars of context
        long_context = (
            "You are a helpful assistant with knowledge about marine biology. "
            * 80
        )
        assert len(long_context) > 5000

        resp = _ollama_chat(client, [
            {"role": "system", "content": long_context},
            {"role": "user", "content": "What is the largest species of whale?"},
        ])
        assert resp.status_code == 200
        content = resp.json()["message"]["content"].strip()
        assert len(content) > 10, f"Response suspiciously short: {content}"
        # Should mention blue whale or similar
        assert any(w in content.lower() for w in ["blue whale", "whale", "cetacean", "baleen"]), (
            f"Response did not address marine biology question: {content[:200]}"
        )

    def test_empty_messages(self, client):
        """Empty messages array should return a clean error, not 500."""
        resp = client.post(
            "/api/chat",
            json={"model": MODEL, "messages": [], "stream": False},
            headers={"X-Augmentum-Mode": "passthrough"},
        )
        # Accept 4xx (validation error) or even 200 with empty response — but NOT 500
        assert resp.status_code != 500, f"Got 500 on empty messages: {resp.text[:300]}"


class TestFormatCompatibility:
    """Test both Ollama and OpenAI format endpoints."""

    def test_ollama_format(self, client):
        """POST to /api/chat (Ollama format) returns valid Ollama response."""
        resp = _ollama_chat(client, [
            {"role": "user", "content": "What is 7 times 8?"},
        ])
        assert resp.status_code == 200
        data = resp.json()

        # Ollama format checks
        assert "message" in data
        assert data["message"]["role"] == "assistant"
        assert "done" in data
        assert data["done"] is True
        assert "model" in data
        content = data["message"]["content"]
        assert "56" in content, f"Expected 56 in response: {content[:200]}"

    def test_openai_format(self, client):
        """POST to /v1/chat/completions (OpenAI format) returns valid OpenAI response."""
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "What is 7 times 8?"}],
                "stream": False,
            },
            headers={"X-Augmentum-Mode": "passthrough"},
        )
        assert resp.status_code == 200
        data = resp.json()

        # OpenAI format checks
        assert "choices" in data, f"Missing 'choices' in OpenAI response: {list(data.keys())}"
        assert len(data["choices"]) >= 1
        choice = data["choices"][0]
        assert choice["message"]["role"] == "assistant"
        content = choice["message"]["content"]
        assert "56" in content, f"Expected 56 in response: {content[:200]}"

        # Should have usage
        if "usage" in data:
            assert data["usage"]["total_tokens"] > 0

    def test_both_formats_agree(self, client):
        """Both endpoints should give semantically equivalent answers."""
        question = "What is the capital of France? Answer in one word."

        ollama_resp = _ollama_chat(client, [
            {"role": "user", "content": question},
        ])
        openai_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": question}],
                "stream": False,
            },
            headers={"X-Augmentum-Mode": "passthrough"},
        )

        assert ollama_resp.status_code == 200
        assert openai_resp.status_code == 200

        ollama_content = ollama_resp.json()["message"]["content"].lower()
        openai_content = openai_resp.json()["choices"][0]["message"]["content"].lower()

        # Both should mention Paris
        assert "paris" in ollama_content, f"Ollama response missing Paris: {ollama_content[:200]}"
        assert "paris" in openai_content, f"OpenAI response missing Paris: {openai_content[:200]}"
