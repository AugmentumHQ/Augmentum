# Augmentum Test Rulebook

This is not a generic testing guide. This is how tests work in **this** project.

## Rules

### 1. Every route file gets a test file

If `augmentum/proxy/foo_routes.py` exists, `tests/test_foo_routes.py` must exist.
The `dead_code.py` scanner enforces this. Minimum coverage: each endpoint returns
the correct status code for success and the most common error case.

### 2. Sync for HTTP, async for DB

```python
# HTTP endpoint → sync test via TestClient (no @pytest.mark.asyncio needed)
def test_get_characters(client):
    resp = client.get("/api/characters/")
    assert resp.status_code == 200

# Database/state operation → async test
@pytest.mark.asyncio
async def test_session_roundtrip():
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    ...
```

`asyncio_mode = "auto"` is set in pyproject.toml, so async tests just work.

### 3. Use conftest fixtures for stateless routes, custom fixtures for stateful

**Stateless** (proxy, health, config, models) — use `client` from conftest:
```python
def test_health(client):
    resp = client.get("/")
    assert resp.status_code == 200
```

**Stateful** (chat, character, persona, narrative, browse) — create a `sqlite_client`:
```python
import asyncio
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.manager import StateManager

@pytest.fixture
def sqlite_client(app):
    backend = SQLiteBackend(":memory:")
    asyncio.get_event_loop().run_until_complete(backend.connect())
    app.state.state_manager = StateManager(backend)
    yield TestClient(app)
    asyncio.get_event_loop().run_until_complete(backend.close())
```

This gives you a fresh in-memory database for every test. No cleanup needed.

### 4. Name tests as `test_<action>_<scenario>`

```python
# Good — reads like a sentence:
def test_create_character_with_avatar()
def test_delete_nonexistent_returns_404()
def test_sync_empty_body_returns_400()

# Bad — vague:
def test_character()
def test_error()
def test_it_works()
```

### 5. Assert the contract, not the implementation

Test what the API promises (status codes, response shape, field presence), not
how it does it internally:

```python
# Good — tests the API contract:
def test_list_characters_empty(sqlite_client):
    resp = sqlite_client.get("/api/characters/")
    assert resp.status_code == 200
    data = resp.json()
    assert "characters" in data
    assert isinstance(data["characters"], list)
    assert len(data["characters"]) == 0

# Bad — tests internal implementation:
def test_list_characters_calls_sqlite(sqlite_client):
    with patch("augmentum.proxy.character_routes._backend") as mock:
        ...  # Testing that it calls a specific function
```

### 6. Test the round-trip, not just the write

For any CRUD endpoint, the test should verify data survives:

```python
def test_character_roundtrip(sqlite_client):
    # Create
    resp = sqlite_client.put("/api/characters/ch_test", json={
        "name": "Luna", "description": "A helpful assistant"
    })
    assert resp.status_code == 200

    # Read back
    resp = sqlite_client.get("/api/characters/ch_test")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Luna"

    # Update
    resp = sqlite_client.put("/api/characters/ch_test", json={
        "name": "Luna v2", "description": "Updated"
    })
    assert resp.status_code == 200

    # Verify update
    resp = sqlite_client.get("/api/characters/ch_test")
    assert resp.json()["name"] == "Luna v2"

    # Delete
    resp = sqlite_client.delete("/api/characters/ch_test")
    assert resp.status_code == 200

    # Verify deletion
    resp = sqlite_client.get("/api/characters/ch_test")
    assert resp.status_code == 404
```

### 7. Test error cases, not just happy path

Every endpoint should have at least one error test:

```python
def test_get_missing_character(sqlite_client):
    resp = sqlite_client.get("/api/characters/nonexistent")
    assert resp.status_code == 404

def test_save_character_empty_body(sqlite_client):
    resp = sqlite_client.put("/api/characters/ch_test", json={})
    assert resp.status_code == 200  # upsert with defaults, not error

def test_import_bad_format(sqlite_client):
    resp = sqlite_client.post("/api/characters/import", json={"invalid": True})
    assert resp.status_code == 400
```

### 8. Mock external services, never real ones

External HTTP calls (LLM backends, TTS providers, SearXNG) are always mocked.
Never make real network calls in tests.

```python
# conftest.py already mocks the LLM backend via MockOllamaBackend.
# For additional services:
from unittest.mock import AsyncMock, MagicMock, patch

def test_tts_with_mock_provider(client):
    with patch("augmentum.proxy.audio_routes._get_default_provider") as mock:
        mock.return_value = {"base_url": "http://fake", "api_key": "", ...}
        ...
```

### 9. Group related tests in classes

Use classes to group tests for the same endpoint or feature:

```python
class TestCharacterCRUD:
    def test_list_empty(self, sqlite_client): ...
    def test_create(self, sqlite_client): ...
    def test_read(self, sqlite_client): ...
    def test_update(self, sqlite_client): ...
    def test_delete(self, sqlite_client): ...

class TestCharacterImport:
    def test_import_v2_card(self, sqlite_client): ...
    def test_import_janitorai_card(self, sqlite_client): ...
    def test_import_bulk(self, sqlite_client): ...
```

### 10. No parametrize, no snapshot; use standard markers

This project uses simple, explicit test functions. Each scenario gets its own
function with a descriptive name. This is intentional — it's easier to read,
debug, and maintain than parameterized matrices.

Standard markers:
- `@pytest.mark.asyncio` — for async tests (auto-detected by asyncio_mode=auto)
- `@pytest.mark.live` — requires running external services, skipped unless `--run-live`
- `@pytest.mark.contract` — external service contract validation
- `@pytest.mark.slow` — tests >5s (benchmarks, stress tests)

### 11. Three-tier test organization

Tests are organized in three tiers:

**Tier 1 — Smoke** (`tests/test_smoke_*.py`): Every module imports, primary class
constructs, basic operation works. Fast (<100ms each).

**Tier 2 — Contract & Integration**:
- `tests/test_contract_*.py` — Mock external service boundaries, verify request/response
  shapes (headers, body format, URL patterns, streaming event sequences)
- `tests/test_integration_*.py` — Cross-subsystem chains with real SQLite :memory:

**Tier 3 — Live** (`tests/live/test_live_*.py`): Real services, real streaming, real
audio. Auto-skipped when services unavailable. Run with `--run-live`.

### 12. Contract test pattern for external services

When testing code that calls external APIs (LLM providers, TTS, Docker, SearXNG),
write contract tests that verify the exact request shape we send:

```python
@pytest.mark.contract
def test_claude_sends_correct_headers():
    """Verify Claude adapter sends required auth headers."""
    with patch("augmentum.models.adapters.claude.httpx.AsyncClient") as mock:
        instance = mock.return_value.__aenter__.return_value
        instance.post = AsyncMock(return_value=MagicMock(
            status_code=200,
            json=lambda: load_fixture("claude_messages.json"),
        ))
        # ... call the adapter ...
        call_kwargs = instance.post.call_args
        assert "x-api-key" in call_kwargs.kwargs["headers"]
        assert "anthropic-version" in call_kwargs.kwargs["headers"]
```

### 13. Canned response fixtures

Realistic API responses live in `tests/fixtures/responses/`:
- `ollama_chat.json`, `ollama_tags.json`
- `openai_chat_completion.json`, `openai_models.json`
- `claude_messages.json`
- `gemini_generate.json`
- `searxng_results.json`
- `deepgram_transcript.json`

Load with the `load_fixture` conftest helper:
```python
def test_parse_ollama_response(load_fixture):
    data = load_fixture("ollama_chat.json")
    assert data["done"] is True
```

### 14. Shared fixtures

| Fixture | Purpose | Scope |
|---------|---------|-------|
| `client` | Stateless route tests | TestClient with mock backend |
| `sqlite_client` | Stateful route tests | TestClient with real SQLite :memory: |
| `mock_backend` | MockOllamaBackend | Canned LLM responses |
| `mock_docker` | AsyncMock aiodocker | Container/network testing |
| `load_fixture` | Load canned JSON responses | From tests/fixtures/responses/ |
| `audio_silence` | 1s PCM silence (32KB) | Voice pipeline tests |
| `audio_tone` | 1s 440Hz sine PCM | Voice/HBE tests |

---

## What NOT to Test

### Don't test the framework
FastAPI's routing, Pydantic validation, and SQLite's SQL engine are already tested
by their own projects. Don't re-test them:

```python
# Don't do this — you're testing Pydantic, not your code:
def test_request_model_validates_types():
    with pytest.raises(ValidationError):
        MyRequestModel(name=123)  # name should be str
```

### Don't test LLM output quality
The LLM's response content is non-deterministic. Test that the pipeline processes
responses correctly, not what the model says:

```python
# Good — tests pipeline behavior:
def test_stream_response_includes_done_signal(client):
    resp = client.post("/api/chat", json={..., "stream": True})
    lines = resp.text.strip().split("\n")
    last = json.loads(lines[-1])
    assert last["done"] is True

# Bad — tests model intelligence:
def test_model_gives_correct_answer(client):
    resp = client.post("/api/chat", json={"messages": [{"role": "user", "content": "What is 2+2?"}]})
    assert "4" in resp.json()["message"]["content"]  # Depends on model!
```

### Don't test internal helper functions unless they're complex

If a function is 5 lines and only called from one place, test it through the
endpoint. Save direct unit tests for complex logic:

```python
# Worth testing directly — complex parsing with edge cases:
def test_parse_native_tool_call_with_nested_json(): ...
def test_extract_emotion_instruct_rp_markers(): ...
def test_is_refusal_text_compound_phrases(): ...

# Not worth testing directly — simple wiring:
def test_get_settings_store_returns_store(): ...  # Just reads app.state
```

### Don't mock what you own

If you can use the real thing (in-memory), do:

```python
# Good — real SQLite, real StateManager:
backend = SQLiteBackend(":memory:")
await backend.connect()
manager = StateManager(backend)

# Bad — unnecessary mock:
manager = MagicMock(spec=StateManager)
manager.create_session = AsyncMock(return_value={"id": "test"})
```

---

## Test File Template

For a new route file `augmentum/proxy/foo_routes.py`:

```python
"""Tests for foo_routes.py endpoints."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient


# --- Fixtures ---

@pytest.fixture
def sqlite_client(app):
    """Client with real SQLite backend for stateful tests."""
    from augmentum.state.backends.sqlite import SQLiteBackend
    from augmentum.state.manager import StateManager

    backend = SQLiteBackend(":memory:")
    asyncio.get_event_loop().run_until_complete(backend.connect())
    app.state.state_manager = StateManager(backend)
    yield TestClient(app)
    asyncio.get_event_loop().run_until_complete(backend.close())


# --- Tests ---

class TestFooCRUD:
    def test_list_empty(self, sqlite_client):
        resp = sqlite_client.get("/api/foo/")
        assert resp.status_code == 200
        assert resp.json() == {"items": []}

    def test_create(self, sqlite_client):
        resp = sqlite_client.post("/api/foo/", json={"name": "test"})
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data

    def test_get_missing(self, sqlite_client):
        resp = sqlite_client.get("/api/foo/nonexistent")
        assert resp.status_code == 404

    def test_roundtrip(self, sqlite_client):
        # Create
        create = sqlite_client.post("/api/foo/", json={"name": "test"})
        foo_id = create.json()["id"]

        # Read back
        read = sqlite_client.get(f"/api/foo/{foo_id}")
        assert read.status_code == 200
        assert read.json()["name"] == "test"

        # Delete
        delete = sqlite_client.delete(f"/api/foo/{foo_id}")
        assert delete.status_code == 200

        # Verify deleted
        verify = sqlite_client.get(f"/api/foo/{foo_id}")
        assert verify.status_code == 404
```

---

## Running Tests

```bash
# All tests:
python -m pytest tests/ -x

# Specific file:
python -m pytest tests/test_foo_routes.py -x

# Specific test:
python -m pytest tests/test_foo_routes.py::TestFooCRUD::test_create -x

# Verbose with output:
python -m pytest tests/test_foo_routes.py -x -v -s

# Stop at first failure:
python -m pytest tests/ -x --tb=short
```

Tests run inside the Docker container (pytest is installed there, not on the host).
From the host: `docker compose exec augmentum python -m pytest tests/ -x`

---

## Live Integration Tests

Unit tests mock everything. **Live tests** hit the actual running Augmentum
instance and verify real production paths. They're standalone scripts
(not pytest) that run manually when Docker is up.

### When to write a live test
- Verifying a new service integration (TTS provider, LLM backend, search)
- Testing end-to-end flows that cross multiple services (voice pipeline, RAG)
- Validating performance characteristics (latency, streaming behavior)
- Reproducing bugs that only occur with real backends

### Naming
Live test files use the `live_` prefix: `tests/live_voice_pipeline_test.py`

### Pattern
```python
"""Live integration test for [feature].

Requires running Docker services. Run manually:
    python tests/live_feature_test.py [--url URL] [--verbose]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Windows UTF-8 fix
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

async def test_feature(base_url: str, verbose: bool = False):
    """Test the feature against a live instance."""
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        # Health check
        resp = await client.get("/")
        assert resp.status_code == 200, f"Health check failed: {resp.status_code}"

        # Test the actual feature
        resp = await client.post("/api/chat", json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        if verbose:
            print(f"  Response: {data['message']['content'][:100]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:6100")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print(f"Testing against {args.url}")
    asyncio.run(test_feature(args.url, args.verbose))
    print("PASS")
```

### What live tests verify that unit tests can't
| Scenario | Unit Test | Live Test |
|----------|-----------|-----------|
| LLM responds to prompts | Mock returns canned text | Real model generates real text |
| TTS produces audio | Mock returns empty bytes | Real audio plays correctly |
| Search returns results | Mock returns fixture data | SearXNG returns live web results |
| Voice pipeline latency | Not measurable | Actual TTFB and inter-sentence gap |
| Character card import from URL | Mock HTTP response | Real Chub/RisuRealm fetch |
| Image generation | Mock returns placeholder | Real SD/GGUF model generates image |
| WebSocket voice flow | Can't test with TestClient | Real bidirectional audio streaming |

### Don't
- Don't run live tests in CI (they need GPU services)
- Don't make live tests fragile (check status codes and shapes, not exact content)
- Don't skip error handling (always assert status codes before parsing JSON)

## Choosing the Oracle (verification spine)

The rules above are pytest-shaped because most Augmentum claims are. But not
every claim is honestly falsifiable by a pytest test, and force-fitting one (or
skipping verification because pytest doesn't fit) are both failure modes. The
doctrine, from `docs/superpowers/specs/2026-07-06-coder-verification-spine-design.md`:
**every change is a claim; match each claim to the cheapest oracle that would
FAIL if the claim were false.**

| Claim class | Cheapest falsifiable oracle |
|---|---|
| Pure logic | Unit/example tests. Broad or ambiguous input space → property/metamorphic checks: invariants, round-trips, idempotence, seeded determinism. |
| Bug fix | Red reproduction test first; the **same verifier** goes green and closes the bug — never a different check for the green leg. |
| Route/API | Rule 1 pairing (status + most-common-error per endpoint) for the dev loop; `python -m augmentum.contracts.probe` differentially for admission-grade sweeps. |
| Persistence/state | Real `:memory:`/local round-trip; restart/reload survival; wrong-user and wrong-session isolation cases are **mandatory** for user-scoped tables (`scope = ? OR scope IS NULL` is not isolation). |
| External provider | Mocked contract tests: fixture responses + request-shape assertions. Never live third-party calls (see Live Integration Tests for the sanctioned exception). |
| Website/app UI | Browser probe: one meaningful user-visible interaction, web-first assertion, console clean, no failed requests. Screenshot is evidence, not the assertion. |
| Game/simulation | Seeded deterministic replay; tick/physics/rule invariants; save/load round-trip. Pixel tolerance only when "renders X" is the actual claim. |
| Visual/media | Render probe: output exists, nonblank, correct dimensions; golden-tolerance compare where a golden exists. |
| Performance | Budget smoke against a stated tolerance, differential vs a baseline where one exists — deltas over absolutes. |
| Security/auth | Abuse cases: wrong-user, missing-auth, malformed input, privilege boundary, concurrency. |
| Docs/config/build | Render/lint/link/build check. No test unless behavior changed — inventing one here is itself a failure. |
| No honest oracle | Say so. Strongest available proxy + explicit statement of what it doesn't cover + human-review note. Never dress a proxy up as proof. |

Cross-cutting sanity check before trusting any check: **could its output change
if the code were wrong?** A script that prints status lines and exits 0 is a
demo, not verification.
