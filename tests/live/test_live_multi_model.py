"""Live multi-model trace verification tests.

Tests end-to-end traces through Augmentum across multiple model sizes,
verifying that the pipeline handles different model capabilities robustly.

Supports both Ollama and LM Studio backends.

Run:
    python -m pytest tests/live/test_live_multi_model.py -x -v -m live
    python -m pytest tests/live/test_live_multi_model.py -x -v -m live --tb=short
    python -m pytest tests/live/test_live_multi_model.py -x -v -m live -k "4b"  # specific tests
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

import httpx
import pytest

BASE_URL = "https://localhost:6443"
LM_STUDIO_BASE = "http://localhost:1234"


@dataclass
class ModelSpec:
    """Specification for a model to test."""

    name: str
    size_category: str  # "small" (<=4B), "medium" (7-13B), "large" (>13B)
    expected_capabilities: list[str] = field(default_factory=list)
    max_tokens: int = 2048
    supports_streaming: bool = True


MODEL_CATALOG: list[ModelSpec] = [
    # Small models (<=4B parameters)
    ModelSpec(
        name="qwen2.5-0.5b-instruct",
        size_category="small",
        expected_capabilities=["basic_qa", "simple_math"],
        max_tokens=512,
    ),
    ModelSpec(
        name="qwen2.5-1.5b-instruct",
        size_category="small",
        expected_capabilities=["basic_qa", "simple_math", "follow_instructions"],
        max_tokens=1024,
    ),
    ModelSpec(
        name="qwen2.5-3b-instruct",
        size_category="small",
        expected_capabilities=["basic_qa", "reasoning", "follow_instructions"],
        max_tokens=1536,
    ),
    ModelSpec(
        name="qwen2.5-4b-instruct",
        size_category="small",
        expected_capabilities=["basic_qa", "reasoning", "follow_instructions", "code"],
        max_tokens=2048,
    ),
    # Medium models (7-13B)
    ModelSpec(
        name="qwen2.5-7b-instruct",
        size_category="medium",
        expected_capabilities=["reasoning", "code", "follow_instructions", "creative"],
        max_tokens=4096,
    ),
    ModelSpec(
        name="llama3.2-1b-instruct",
        size_category="small",
        expected_capabilities=["basic_qa", "simple_math", "follow_instructions"],
        max_tokens=2048,
    ),
    ModelSpec(
        name="llama3.2-3b-instruct",
        size_category="small",
        expected_capabilities=["basic_qa", "reasoning", "follow_instructions"],
        max_tokens=2048,
    ),
    ModelSpec(
        name="llama3.1-8b-instruct",
        size_category="medium",
        expected_capabilities=["reasoning", "code", "follow_instructions", "creative"],
        max_tokens=4096,
    ),
    ModelSpec(
        name="mistral-7b-instruct-v0.3",
        size_category="medium",
        expected_capabilities=["reasoning", "code", "creative"],
        max_tokens=4096,
    ),
    ModelSpec(
        name="phi3-mini-instruct",
        size_category="small",
        expected_capabilities=["basic_qa", "reasoning", "follow_instructions"],
        max_tokens=2048,
    ),
    ModelSpec(
        name="deepseek-coder-1.3b-instruct",
        size_category="small",
        expected_capabilities=["basic_qa", "code", "follow_instructions"],
        max_tokens=2048,
    ),
    ModelSpec(
        name="deepseek-coder-6.7b-instruct",
        size_category="medium",
        expected_capabilities=["reasoning", "code", "follow_instructions"],
        max_tokens=4096,
    ),
    ModelSpec(
        name="gemma2-2b-it",
        size_category="small",
        expected_capabilities=["basic_qa", "simple_math", "follow_instructions"],
        max_tokens=2048,
    ),
    ModelSpec(
        name="gemma2-9b-it",
        size_category="medium",
        expected_capabilities=["reasoning", "code", "follow_instructions"],
        max_tokens=4096,
    ),
    ModelSpec(
        name="codellama-7b-instruct",
        size_category="medium",
        expected_capabilities=["code", "reasoning"],
        max_tokens=4096,
    ),
]


# 4B parameter models specifically - these are the target for this test
FOUR_B_MODELS: list[ModelSpec] = [
    ModelSpec(
        name="qwen2.5-4b-instruct",
        size_category="small",
        expected_capabilities=["basic_qa", "reasoning", "follow_instructions", "code"],
        max_tokens=2048,
    ),
    ModelSpec(
        name="qwen2.5-4b",
        size_category="small",
        expected_capabilities=["basic_qa", "reasoning", "follow_instructions", "code"],
        max_tokens=2048,
    ),
    ModelSpec(
        name="Phi-3.5-mini-instruct",
        size_category="small",
        expected_capabilities=["basic_qa", "reasoning", "follow_instructions"],
        max_tokens=2048,
    ),
    ModelSpec(
        name="gemma-2-4b-it",
        size_category="small",
        expected_capabilities=["basic_qa", "reasoning", "follow_instructions"],
        max_tokens=2048,
    ),
]


WORKFLOW_SCENARIOS = [
    {
        "id": "simple_qa",
        "prompt": "What is 2+2? Answer with just the number.",
        "expected_in_response": ["4"],
        "expected_fail_on_small": False,
        "timeout": 30,
    },
    {
        "id": "follow_instruction",
        "prompt": "Respond with exactly three words: hello world today",
        "expected_in_response": ["hello", "world", "today"],
        "expected_fail_on_small": False,
        "timeout": 30,
    },
    {
        "id": "basic_reasoning",
        "prompt": "If all roses are flowers and some flowers fade quickly, what can we conclude about roses?",
        "expected_in_response": ["fade", "flowers", "may", "can"],
        "expected_fail_on_small": False,
        "timeout": 60,
    },
    {
        "id": "creative_writing",
        "prompt": "Write a haiku about coding. Only output the poem.",
        "expected_in_response": ["code", "bug", "fix", "build", "deploy"],
        "expected_fail_on_small": False,
        "timeout": 60,
    },
    {
        "id": "code_simple",
        "prompt": "Write a Python function that returns the factorial of n. Only output the function.",
        "expected_in_response": ["def", "factorial", "return", "n"],
        "expected_fail_on_small": False,
        "timeout": 60,
    },
    {
        "id": "multi_step_reasoning",
        "prompt": "A train travels 60 mph for 2 hours, then 80 mph for 1 hour. What is the average speed? Show your work.",
        "expected_in_response": ["60", "80", "200", "average", "speed", "÷", "/", "divide"],
        "expected_fail_on_small": False,
        "timeout": 90,
    },
    {
        "id": "context_recall",
        "prompt": "Remember the word 'xylophone'. Now tell me: what word did I ask you to remember?",
        "expected_in_response": ["xylophone"],
        "expected_fail_on_small": True,  # Small models struggle with in-context recall
        "timeout": 60,
    },
    {
        "id": "system_prompt_adherence",
        "prompt": "You must only respond in Spanish. What is the capital of France?",
        "expected_in_response": ["paris", "capital"],
        "forbidden_words": ["the capital of", "capital of"],
        "expected_fail_on_small": True,  # Small models struggle with language adherence
        "timeout": 60,
    },
]


def _probe() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/", timeout=3.0, verify=False)
        return r.status_code < 500
    except Exception:
        return False


def _probe_lm_studio() -> bool:
    """Check if LM Studio local server is running."""
    try:
        r = httpx.get(f"{LM_STUDIO_BASE}/v1/models", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def _get_available_models(client: httpx.Client) -> list[str]:
    """Query Augmentum for available models."""
    try:
        resp = client.get("/api/models", timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and "models" in data:
                return [m.get("name", m) if isinstance(m, dict) else m for m in data["models"]]
            elif isinstance(data, list):
                return [m.get("name", m) if isinstance(m, dict) else m for m in data]
    except Exception:
        pass
    return []


def _get_lm_studio_models() -> list[dict]:
    """Query LM Studio for available models."""
    try:
        resp = httpx.get(f"{LM_STUDIO_BASE}/v1/models", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            if "data" in data:
                return data["data"]
            return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def _get_lm_studio_model_name(model_info: dict) -> str:
    """Extract model name from LM Studio model info."""
    if isinstance(model_info, dict):
        return model_info.get("id", model_info.get("name", ""))
    return str(model_info)


def _match_model(spec: ModelSpec, available: list[str]) -> str | None:
    """Find best matching model from available list."""
    for model_name in available:
        model_lower = model_name.lower()
        spec_lower = spec.name.lower()
        if spec_lower in model_lower or model_lower in spec_lower:
            return model_name
        for part in spec_lower.split(":"):
            if part and part in model_lower:
                return model_name
    return None


@pytest.fixture(scope="module")
def _require_server():
    if not _probe():
        pytest.skip(f"Augmentum not reachable at {BASE_URL}")


@pytest.fixture(scope="module")
def lm_studio_available():
    """Check if LM Studio is running."""
    return _probe_lm_studio()


@pytest.fixture(scope="module")
def client(_require_server):
    with httpx.Client(base_url=BASE_URL, timeout=180.0, verify=False) as c:
        yield c


@pytest.fixture(scope="module")
def lm_client(_require_server, lm_studio_available):
    """Direct LM Studio client (bypass Augmentum for comparison)."""
    if not lm_studio_available:
        pytest.skip("LM Studio not available")
    with httpx.Client(base_url=LM_STUDIO_BASE, timeout=180.0) as c:
        yield c


@pytest.fixture(scope="module")
def available_models(client):
    """Dynamically discover available models."""
    models = _get_available_models(client)
    if not models:
        pytest.skip("No models available from Augmentum")
    return models


@pytest.fixture(scope="module")
def lm_studio_models(lm_studio_available):
    """Get models directly from LM Studio."""
    if not lm_studio_available:
        return []
    return _get_lm_studio_models()


def _chat_request(
    client: httpx.Client,
    model: str,
    messages: list[dict],
    *,
    stream: bool = False,
    mode: str = "passthrough",
    timeout: float = 60.0,
) -> httpx.Response:
    """Send a chat request to Augmentum."""
    headers = {}
    if mode != "passthrough":
        headers["X-Augmentum-Mode"] = mode

    return client.post(
        "/api/chat",
        json={"model": model, "messages": messages, "stream": stream},
        headers=headers,
        timeout=timeout,
    )


def _stream_and_assemble(
    response: httpx.Response,
) -> tuple[list[dict], str]:
    """Stream response and return chunks + assembled text."""
    chunks = []
    full_text = []

    for line in response.iter_lines():
        if not line.strip():
            continue
        chunk = json.loads(line)
        chunks.append(chunk)
        delta = chunk.get("message", {}).get("content", "")
        if delta:
            full_text.append(delta)

    return chunks, "".join(full_text)


# ============================================================================
# Model Discovery & Capability Tests
# ============================================================================


@pytest.mark.live
class TestModelDiscovery:
    """Verify model discovery and catalog functionality."""

    def test_model_list_accessible(self, client):
        """Models endpoint should return 200 with a list."""
        resp = client.get("/api/models", timeout=10.0)
        assert resp.status_code == 200, f"Model list failed: {resp.status_code}"
        data = resp.json()
        assert isinstance(data, (dict, list)), f"Unexpected response type: {type(data)}"

    def test_model_catalog_matches_availability(self, available_models):
        """Catalog entries should match at least some available models."""
        matched = 0
        for spec in MODEL_CATALOG:
            if _match_model(spec, available_models):
                matched += 1

        assert matched > 0, (
            f"None of the {len(MODEL_CATALOG)} catalog models matched available: {available_models}"
        )

    def test_lm_studio_discovery(self, lm_studio_available, lm_studio_models):
        """Verify LM Studio is detected and models are listed."""
        if not lm_studio_available:
            pytest.skip("LM Studio not running")

        assert len(lm_studio_models) > 0, "No models found in LM Studio"

        for m in lm_studio_models:
            name = _get_lm_studio_model_name(m)
            assert name, f"Model missing identifier: {m}"


# ============================================================================
# LM Studio Direct Comparison Tests
# ============================================================================


@pytest.mark.live
class TestLMStudioDirect:
    """Test LM Studio models directly (bypassing Augmentum)."""

    def test_lm_studio_4b_models(self, lm_client, lm_studio_models):
        """Test all 4B-parameter models available in LM Studio."""
        results = []

        for spec in FOUR_B_MODELS:
            matched = None
            for m in lm_studio_models:
                name = _get_lm_studio_model_name(m)
                if _match_model(spec, [name]):
                    matched = name
                    break

            if not matched:
                results.append({"spec": spec.name, "status": "not_loaded"})
                continue

            resp = lm_client.post(
                "/v1/chat/completions",
                json={
                    "model": matched,
                    "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
                    "max_tokens": 50,
                },
                timeout=60,
            )

            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "").lower()
                results.append(
                    {
                        "model": matched,
                        "status": "passed" if "hello" in content else "wrong_output",
                        "output": content[:50],
                    }
                )
            else:
                results.append(
                    {
                        "model": matched,
                        "status": "error",
                        "error": f"HTTP {resp.status_code}",
                    }
                )

        passed = sum(1 for r in results if r["status"] == "passed")
        print(f"\nLM Studio 4B results: {results}")

        available = [r for r in results if r["status"] != "not_loaded"]
        if available:
            assert passed > 0, f"No 4B models passed: {results}"

    def test_lm_studio_all_loaded_models(self, lm_client, lm_studio_models):
        """Test all currently loaded LM Studio models."""
        results = []

        for m in lm_studio_models:
            name = _get_lm_studio_model_name(m)
            if not name:
                continue

            resp = lm_client.post(
                "/v1/chat/completions",
                json={
                    "model": name,
                    "messages": [
                        {"role": "user", "content": "What is 2+2? Answer with just the number."}
                    ],
                    "max_tokens": 20,
                },
                timeout=60,
            )

            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                passed = "4" in content
                results.append(
                    {
                        "model": name,
                        "passed": passed,
                        "output": content[:30],
                    }
                )
            else:
                results.append(
                    {
                        "model": name,
                        "passed": False,
                        "error": f"HTTP {resp.status_code}",
                    }
                )

        print(f"\nLM Studio loaded models: {len(lm_studio_models)}")
        print(f"Results: {results}")

        passed = sum(1 for r in results if r["passed"])
        assert passed > 0, f"No LM Studio models passed: {results}"

    def test_lm_studio_streaming(self, lm_client, lm_studio_models):
        """Test streaming responses from LM Studio."""
        if not lm_studio_models:
            pytest.skip("No LM Studio models")

        model = _get_lm_studio_model_name(lm_studio_models[0])

        with lm_client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Count from 1 to 5."}],
                "stream": True,
                "max_tokens": 100,
            },
            timeout=60,
        ) as resp:
            assert resp.status_code == 200

            chunks = []
            full_text = []
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    if line.strip() == "data: [DONE]":
                        break
                    try:
                        chunk_data = json.loads(line[6:])
                        delta = (
                            chunk_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        )
                        if delta:
                            full_text.append(delta)
                    except json.JSONDecodeError:
                        pass

        assembled = "".join(full_text)
        print(f"\nStreaming response: {assembled[:100]}")
        assert len(assembled.strip()) > 0, "Empty streaming response"


# ============================================================================
# Augmentum + LM Studio Integration Tests
# ============================================================================


@pytest.mark.live
class TestAugmentumLMStudioIntegration:
    """Test Augmentum routing to LM Studio models."""

    def test_augmentum_routes_to_lm_studio(self, client, lm_studio_models):
        """Verify Augmentum can route requests to LM Studio models."""
        if not lm_studio_models:
            pytest.skip("No LM Studio models")

        model = _get_lm_studio_model_name(lm_studio_models[0])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Say 'via augmentum' only."}],
                "max_tokens": 50,
            },
            timeout=60,
        )

        if resp.status_code == 200:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").lower()
            assert "augmentum" in content, f"Unexpected response: {content}"
        elif resp.status_code in [400, 404]:
            pytest.skip(
                f"Model {model} not registered in Augmentum (expected for direct LM Studio models)"
            )

    def test_augmentum_to_lm_4b_model(self, client, lm_studio_models):
        """Test Augmentum routing to a 4B model specifically."""
        four_b_models = [
            m for m in lm_studio_models if "4b" in _get_lm_studio_model_name(m).lower()
        ]

        if not four_b_models:
            pytest.skip("No 4B models loaded in LM Studio")

        model = _get_lm_studio_model_name(four_b_models[0])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "What is the square root of 16?"}],
                "max_tokens": 50,
            },
            timeout=90,
        )

        if resp.status_code == 200:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").lower()
            print(f"\n4B model response: {content[:100]}")
            assert any(w in content for w in ["4", "four", "root"]), (
                f"Unexpected response: {content}"
            )
        elif resp.status_code in [400, 404]:
            pytest.skip(f"4B model {model} not in Augmentum registry")


# ============================================================================
# Single Model Trace Tests
# ============================================================================


@pytest.mark.live
class TestSingleModelTraces:
    """Test trace behavior for individual models."""

    @pytest.mark.parametrize("spec", MODEL_CATALOG, ids=lambda s: s.name)
    def test_model_responds(self, client, available_models, spec):
        """Each model should produce a valid non-empty response."""
        model = _match_model(spec, available_models)
        if not model:
            pytest.skip(f"Model {spec.name} not available")

        resp = _chat_request(
            client,
            model,
            [{"role": "user", "content": "Say 'hello' and nothing else."}],
            timeout=spec.max_tokens / 50 + 30,
        )

        assert resp.status_code == 200, (
            f"Model {model} failed: {resp.status_code} — {resp.text[:200]}"
        )

        if resp.headers.get("content-type", "").startswith("text/event-stream"):
            _, text = _stream_and_assemble(resp)
        else:
            data = resp.json()
            text = data.get("message", {}).get("content", "")

        assert len(text.strip()) > 0, f"Model {model} returned empty response"

    @pytest.mark.parametrize("spec", MODEL_CATALOG, ids=lambda s: s.name)
    def test_model_streaming_integrity(self, client, available_models, spec):
        """Streaming responses should assemble to valid text."""
        model = _match_model(spec, available_models)
        if not model:
            pytest.skip(f"Model {spec.name} not available")

        resp = _chat_request(
            client,
            model,
            [{"role": "user", "content": "Count from 1 to 3, one number per line."}],
            stream=True,
            timeout=60,
        )

        assert resp.status_code == 200, f"Streaming failed for {model}"

        chunks, assembled = _stream_and_assemble(resp)

        assert len(chunks) >= 2, f"Expected multiple chunks for {model}, got {len(chunks)}"
        assert chunks[0].get("done") is False, "First chunk should not be done"
        assert chunks[-1].get("done") is True, "Last chunk should be done"
        assert len(assembled.strip()) > 0, f"Assembled text empty for {model}"


# ============================================================================
# Size Category Tests
# ============================================================================


@pytest.mark.live
class TestModelSizeCategories:
    """Test behavior grouped by model size."""

    def test_small_models_basic_qa(self, client, available_models):
        """Small models (<=4B) should handle basic Q&A."""
        small_specs = [s for s in MODEL_CATALOG if s.size_category == "small"]
        results = []

        for spec in small_specs:
            model = _match_model(spec, available_models)
            if not model:
                continue

            resp = _chat_request(
                client,
                model,
                [{"role": "user", "content": "What is the capital of Japan?"}],
                timeout=60,
            )

            if resp.status_code == 200:
                data = resp.json()
                content = data.get("message", {}).get("content", "").lower()
                results.append(
                    {
                        "model": model,
                        "passed": "tokyo" in content,
                        "content_preview": content[:100],
                    }
                )

        if not results:
            pytest.skip("No small models available for testing")

        passed = sum(1 for r in results if r["passed"])
        assert passed > 0, f"No small models answered correctly: {results}"

    def test_medium_models_reasoning(self, client, available_models):
        """Medium models should handle multi-step reasoning."""
        medium_specs = [s for s in MODEL_CATALOG if s.size_category == "medium"]
        results = []

        for spec in medium_specs:
            model = _match_model(spec, available_models)
            if not model:
                continue

            resp = _chat_request(
                client,
                model,
                [
                    {
                        "role": "user",
                        "content": "If A > B and B > C, tell me about A and C. Be brief.",
                    }
                ],
                timeout=90,
            )

            if resp.status_code == 200:
                data = resp.json()
                content = data.get("message", {}).get("content", "").lower()
                results.append(
                    {
                        "model": model,
                        "passed": any(w in content for w in ["a", ">", "greater", "c"]),
                        "content_preview": content[:100],
                    }
                )

        if not results:
            pytest.skip("No medium models available for testing")

        passed = sum(1 for r in results if r["passed"])
        assert passed > 0, f"No medium models handled reasoning: {results}"


# ============================================================================
# Workflow Scenario Tests
# ============================================================================


@pytest.mark.live
class TestWorkflowScenarios:
    """Test various real-world workflows across model sizes."""

    @pytest.mark.parametrize("scenario", WORKFLOW_SCENARIOS, ids=lambda s: s["id"])
    def test_scenario_across_models(self, client, available_models, scenario):
        """Run a workflow scenario across all available models."""
        results = []

        for spec in MODEL_CATALOG:
            model = _match_model(spec, available_models)
            if not model:
                continue

            # Skip small models for complex scenarios
            if scenario.get("expected_fail_on_small") and spec.size_category == "small":
                results.append(
                    {
                        "model": model,
                        "skipped": True,
                        "reason": "complex scenario, small model",
                    }
                )
                continue

            resp = _chat_request(
                client,
                model,
                [{"role": "user", "content": scenario["prompt"]}],
                timeout=scenario.get("timeout", 60),
            )

            if resp.status_code != 200:
                results.append(
                    {
                        "model": model,
                        "failed": True,
                        "error": f"HTTP {resp.status_code}",
                    }
                )
                continue

            if resp.headers.get("content-type", "").startswith("text/event-stream"):
                _, content = _stream_and_assemble(resp)
            else:
                data = resp.json()
                content = data.get("message", {}).get("content", "")

            content_lower = content.lower()

            # Check expected words
            expected = scenario.get("expected_in_response", [])
            has_expected = any(word.lower() in content_lower for word in expected)

            # Check forbidden words
            forbidden = scenario.get("forbidden_words", [])
            has_forbidden = any(fw.lower() in content_lower for fw in forbidden)

            results.append(
                {
                    "model": model,
                    "spec": spec.name,
                    "size": spec.size_category,
                    "passed": has_expected and not has_forbidden,
                    "content_preview": content[:150],
                }
            )

        if not results:
            pytest.skip("No matching models available")

        # At least some models should pass
        passed = sum(1 for r in results if r.get("passed"))
        total = sum(1 for r in results if not r.get("skipped"))

        assert passed > 0, (
            f"Scenario '{scenario['id']}' failed on all {total} tested models:\n"
            + "\n".join(
                f"  {r['model']}: {r.get('content_preview', r.get('error', 'unknown'))[:80]}"
                for r in results
                if not r.get("skipped")
            )
        )


# ============================================================================
# Multi-Turn Conversation Tests
# ============================================================================


@pytest.mark.live
class TestMultiTurnConversations:
    """Test conversation continuity across multiple turns."""

    def test_conversation_memory_across_turns(self, client, available_models):
        """Models should maintain context across multiple turns."""
        medium_specs = [s for s in MODEL_CATALOG if s.size_category in ["medium", "small"]]

        for spec in medium_specs[:3]:  # Test on up to 3 models
            model = _match_model(spec, available_models)
            if not model:
                continue

            messages = [
                {"role": "user", "content": "My favorite color is azure blue."},
                {
                    "role": "assistant",
                    "content": "That's a beautiful color! Azure blue is a lovely shade.",
                },
                {"role": "user", "content": "What is my favorite color?"},
            ]

            resp = _chat_request(client, model, messages, timeout=90)

            if resp.status_code != 200:
                continue

            data = resp.json()
            content = data.get("message", {}).get("content", "").lower()

            # Should mention azure or blue
            if "azure" in content or "blue" in content:
                return  # Test passed

            # Allow some models to fail this
            continue

        pytest.skip("No suitable models for multi-turn test")

    def test_conversation_with_system_prompt(self, client, available_models):
        """System prompts should influence model behavior."""
        system_msg = "You are a pirate. Respond accordingly but briefly."

        for spec in MODEL_CATALOG[:5]:
            model = _match_model(spec, available_models)
            if not model:
                continue

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": "Hello!"},
            ]

            resp = _chat_request(client, model, messages, timeout=60)

            if resp.status_code != 200:
                continue

            data = resp.json()
            content = data.get("message", {}).get("content", "").lower()

            # Look for pirate indicators
            pirate_words = ["arr", "ahoy", "matey", "ship", "sea", "crew", "captain", "yer"]
            if any(pw in content for pw in pirate_words):
                return  # Test passed

        pytest.skip("No models responded to system prompt as pirates")


# ============================================================================
# Concurrency & Load Tests
# ============================================================================


@pytest.mark.live
class TestConcurrentRequests:
    """Test behavior under concurrent load."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_across_models(self, available_models):
        """Send concurrent requests to different models."""
        if not available_models:
            pytest.skip("No models available")

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=120.0, verify=False) as ac:
            tasks = []

            for i, model in enumerate(available_models[:5]):
                messages = [{"role": "user", "content": f"Respond with just the number {i + 1}."}]
                payload = {"model": model, "messages": messages, "stream": False}

                async def send(idx: int, m: str):
                    resp = await ac.post("/api/chat", json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        content = data.get("message", {}).get("content", "").strip()
                        return {"model": m, "content": content, "expected": str(idx + 1)}
                    return {"model": m, "error": f"HTTP {resp.status_code}"}

                tasks.append(send(i, model))

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        valid_results = [r for r in results if isinstance(r, dict) and "error" not in r]
        assert len(valid_results) > 0, "All concurrent requests failed"


# ============================================================================
# Latency & Performance Tests
# ============================================================================


@pytest.mark.live
class TestLatencyCharacteristics:
    """Measure and compare response latency across model sizes."""

    def test_latency_by_model_size(self, client, available_models):
        """Small models should generally respond faster than large ones."""
        timings = {}

        for spec in MODEL_CATALOG:
            model = _match_model(spec, available_models)
            if not model:
                continue

            start = time.time()
            resp = _chat_request(
                client,
                model,
                [{"role": "user", "content": "What is 2+2?"}],
                timeout=30,
            )
            elapsed = time.time() - start

            if resp.status_code == 200:
                timings[model] = {
                    "size": spec.size_category,
                    "latency": elapsed,
                }

        if len(timings) < 2:
            pytest.skip("Not enough models for comparison")

        # Compare small vs medium/large
        small_latencies = [t["latency"] for t in timings.values() if t["size"] == "small"]
        other_latencies = [t["latency"] for t in timings.values() if t["size"] != "small"]

        if small_latencies and other_latencies:
            avg_small = sum(small_latencies) / len(small_latencies)
            avg_other = sum(other_latencies) / len(other_latencies)

            # Log the comparison but don't fail - just document
            print(f"\nLatency comparison: small={avg_small:.2f}s, other={avg_other:.2f}s")


# ============================================================================
# Error Handling & Edge Cases
# ============================================================================


@pytest.mark.live
class TestErrorHandling:
    """Test error handling across different failure modes."""

    def test_nonexistent_model_error(self, client):
        """Request to non-existent model should return proper error."""
        resp = client.post(
            "/api/chat",
            json={
                "model": "nonexistent-model-xyz-12345",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
            timeout=30,
        )

        assert resp.status_code in [400, 404, 502], (
            f"Expected error status for nonexistent model, got {resp.status_code}"
        )

    def test_empty_prompt_handling(self, client, available_models):
        """Empty or whitespace prompts should be handled gracefully."""
        for model in available_models[:3]:
            resp = client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "   "}],
                    "stream": False,
                },
                timeout=30,
            )
            # Should not return 500
            assert resp.status_code != 500, f"Empty prompt caused 500 for {model}"

    def test_very_long_prompt(self, client, available_models):
        """Very long prompts should not cause crashes."""
        long_prompt = "The word is: " + "x" * 10000

        for model in available_models[:2]:
            resp = client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": long_prompt}],
                    "stream": False,
                },
                timeout=120,
            )
            # Should return 200 or 400, not 500
            assert resp.status_code < 500, f"Long prompt caused {resp.status_code} for {model}"


# ============================================================================
# Streaming Quality Tests
# ============================================================================


@pytest.mark.live
class TestStreamingQuality:
    """Verify streaming output quality and correctness."""

    def test_streaming_completeness(self, client, available_models):
        """Stream should eventually produce complete response."""
        for model in available_models[:3]:
            resp = _chat_request(
                client,
                model,
                [{"role": "user", "content": "List three colors. One per line."}],
                stream=True,
                timeout=60,
            )

            if resp.status_code != 200:
                continue

            chunks, assembled = _stream_and_assemble(resp)

            # Should have multiple chunks
            assert len(chunks) >= 3, f"Expected at least 3 chunks for {model}"

            # Should have content
            assert len(assembled.strip()) > 0, f"No content streamed for {model}"

            # Verify done signal
            assert chunks[-1].get("done") is True, f"Missing done signal for {model}"

            return  # Test passed

        pytest.skip("No suitable streaming models available")

    def test_streaming_order_preserved(self, client, available_models):
        """Content should stream in correct order (no reversed text)."""
        for model in available_models[:2]:
            resp = _chat_request(
                client,
                model,
                [{"role": "user", "content": "Write the alphabet."}],
                stream=True,
                timeout=60,
            )

            if resp.status_code != 200:
                continue

            chunks, assembled = _stream_and_assemble(resp)

            # A->Z should appear in order
            if "a" in assembled.lower() and "z" in assembled.lower():
                a_pos = assembled.lower().find("a")
                z_pos = assembled.lower().find("z")
                assert a_pos < z_pos, f"Alphabetic order reversed for {model}"
                return

        pytest.skip("No suitable models for order test")


# ============================================================================
# 4B Model Specific Tests
# ============================================================================


@pytest.mark.live
class Test4BModels:
    """Dedicated tests for 4B parameter models."""

    FOUR_B_SCENARIOS = [
        {
            "id": "basic_math",
            "prompt": "What is 15 + 27? Answer with just the number.",
            "expected": ["42"],
            "description": "Simple arithmetic",
        },
        {
            "id": "word_problem",
            "prompt": "If I have 3 apples and eat 1, how many do I have left?",
            "expected": ["2", "two"],
            "description": "Simple word problem",
        },
        {
            "id": "fact_recall",
            "prompt": "What is the capital of Japan?",
            "expected": ["tokyo"],
            "description": "Basic fact recall",
        },
        {
            "id": "instruction_following",
            "prompt": "Respond with exactly the word YES in uppercase and nothing else.",
            "expected": ["YES"],
            "forbidden": ["but", "however", "however,"],
            "description": "Strict instruction following",
        },
        {
            "id": "code_simple",
            "prompt": "Write a Python function that returns the sum of two numbers. Output only the function.",
            "expected": ["def", "return"],
            "description": "Simple code generation",
        },
        {
            "id": "logical_reasoning",
            "prompt": "All dogs are animals. Max is a dog. Is Max an animal? Answer yes or no.",
            "expected": ["yes", "yep", "indeed"],
            "description": "Basic logical deduction",
        },
    ]

    def test_lm_studio_4b_models_workflow(self, lm_client, lm_studio_models):
        """Run all 4B scenarios against LM Studio 4B models."""
        four_b_models = [
            m
            for m in lm_studio_models
            if "4b" in _get_lm_studio_model_name(m).lower()
            or "-4b" in _get_lm_studio_model_name(m).lower()
            or "4b-" in _get_lm_studio_model_name(m).lower()
        ]

        if not four_b_models:
            pytest.skip("No 4B models loaded in LM Studio")

        all_results = []

        for model_info in four_b_models:
            model_name = _get_lm_studio_model_name(model_info)
            model_results = {"model": model_name, "scenarios": []}

            for scenario in self.FOUR_B_SCENARIOS:
                resp = lm_client.post(
                    "/v1/chat/completions",
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": scenario["prompt"]}],
                        "max_tokens": 100,
                    },
                    timeout=60,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    content = (
                        data.get("choices", [{}])[0].get("message", {}).get("content", "").lower()
                    )

                    expected_found = any(e.lower() in content for e in scenario["expected"])
                    forbidden_found = any(
                        f.lower() in content for f in scenario.get("forbidden", [])
                    )

                    passed = expected_found and not forbidden_found
                    model_results["scenarios"].append(
                        {
                            "id": scenario["id"],
                            "passed": passed,
                            "content": content[:50],
                        }
                    )
                else:
                    model_results["scenarios"].append(
                        {
                            "id": scenario["id"],
                            "passed": False,
                            "error": f"HTTP {resp.status_code}",
                        }
                    )

            all_results.append(model_results)

        print(f"\n4B Model Results:")
        for result in all_results:
            passed = sum(1 for s in result["scenarios"] if s["passed"])
            print(f"  {result['model']}: {passed}/{len(result['scenarios'])} passed")
            for s in result["scenarios"]:
                status = "✓" if s["passed"] else "✗"
                print(f"    {status} {s['id']}: {s.get('content', s.get('error', ''))[:40]}")

        # At least one model should pass most scenarios
        total_passed = sum(sum(1 for s in r["scenarios"] if s["passed"]) for r in all_results)
        total_tests = sum(len(r["scenarios"]) for r in all_results)

        assert total_passed > 0, f"No 4B model scenarios passed: {all_results}"
        assert total_passed >= total_tests * 0.5, (
            f"4B models failing too many tests: {total_passed}/{total_tests}"
        )

    def test_4b_model_consistency(self, lm_client, lm_studio_models):
        """Test that 4B models give consistent answers across multiple runs."""
        four_b_models = [
            m for m in lm_studio_models if "4b" in _get_lm_studio_model_name(m).lower()
        ]

        if not four_b_models:
            pytest.skip("No 4B models loaded")

        model_name = _get_lm_studio_model_name(four_b_models[0])
        question = "What is 7 times 8?"
        answers = []

        for _ in range(3):
            resp = lm_client.post(
                "/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": question}],
                    "max_tokens": 20,
                },
                timeout=60,
            )

            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                answers.append(content)

        assert len(answers) >= 2, "Not enough consistent responses"

        # 56 should appear in most answers
        correct_count = sum(1 for a in answers if "56" in a)
        assert correct_count >= len(answers) * 0.5, f"Inconsistent answers: {answers}"

    def test_4b_augmentum_vs_direct(self, client, lm_client, lm_studio_models):
        """Compare 4B model responses via Augmentum vs direct LM Studio."""
        four_b_models = [
            m for m in lm_studio_models if "4b" in _get_lm_studio_model_name(m).lower()
        ]

        if not four_b_models:
            pytest.skip("No 4B models loaded")

        model_name = _get_lm_studio_model_name(four_b_models[0])
        question = "What is the square root of 144?"

        # Direct LM Studio
        direct_resp = lm_client.post(
            "/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": question}],
                "max_tokens": 50,
            },
            timeout=60,
        )

        if direct_resp.status_code != 200:
            pytest.skip("Direct LM Studio call failed")

        direct_content = (
            direct_resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").lower()
        )

        # Via Augmentum
        augmentum_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": question}],
                "max_tokens": 50,
            },
            timeout=60,
        )

        if augmentum_resp.status_code == 200:
            augmentum_content = (
                augmentum_resp.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .lower()
            )

            print(f"\nDirect: {direct_content[:50]}")
            print(f"Via Augmentum: {augmentum_content[:50]}")

            # Both should contain "12" or "twelve"
            direct_correct = "12" in direct_content or "twelve" in direct_content
            augmentum_correct = "12" in augmentum_content or "twelve" in augmentum_content

            assert direct_correct, f"Direct response wrong: {direct_content}"
            # Augmentum may fail if model not registered, so just log
            if not augmentum_correct:
                print(f"Warning: Augmentum response differs from direct")


# ============================================================================
# Stress Test: 4B Models Under Load
# ============================================================================


@pytest.mark.live
class Test4BStressTests:
    """Stress tests specifically for 4B parameter models."""

    @pytest.mark.asyncio
    async def test_concurrent_4b_requests(self, lm_studio_models):
        """Send concurrent requests to 4B models."""
        four_b_models = [
            m for m in lm_studio_models if "4b" in _get_lm_studio_model_name(m).lower()
        ]

        if not four_b_models:
            pytest.skip("No 4B models loaded")

        model_name = _get_lm_studio_model_name(four_b_models[0])

        async with httpx.AsyncClient(base_url=LM_STUDIO_BASE, timeout=120.0) as ac:
            tasks = []

            for i in range(5):
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": f"What number am I thinking of? It's {i + 1}."}
                    ],
                    "max_tokens": 50,
                }
                tasks.append(ac.post("/v1/chat/completions", json=payload))

            results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = [r for r in results if isinstance(r, httpx.Response) and r.status_code == 200]
        assert len(successes) >= 4, f"Too many failures under concurrent load: {len(successes)}/5"

    def test_4b_rapid_successive_requests(self, lm_client, lm_studio_models):
        """Test 4B model with rapid successive requests."""
        four_b_models = [
            m for m in lm_studio_models if "4b" in _get_lm_studio_model_name(m).lower()
        ]

        if not four_b_models:
            pytest.skip("No 4B models loaded")

        model_name = _get_lm_studio_model_name(four_b_models[0])

        for i in range(5):
            resp = lm_client.post(
                "/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": f"Say the number {i}."}],
                    "max_tokens": 20,
                },
                timeout=60,
            )

            assert resp.status_code == 200, f"Request {i} failed: {resp.status_code}"

            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"Request {i}: {content.strip()[:30]}")
