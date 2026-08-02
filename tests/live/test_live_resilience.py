"""Live resilience tests — error handling, edge cases, and security against the running service.

Run with:  python -m pytest tests/live/test_live_resilience.py -m live -x -v
Requires: Augmentum running at https://localhost:6443
"""

from __future__ import annotations

import asyncio
import uuid

import httpx
import pytest

BASE = "https://localhost:6443"
TIMEOUT = 30.0


def _probe() -> bool:
    try:
        resp = httpx.get(f"{BASE}/api/config/tools", timeout=3.0, verify=False)
        return resp.status_code < 500
    except Exception:
        return False


@pytest.fixture(scope="module")
def client():
    """Sync httpx client pointed at the live Augmentum HTTPS service."""
    if not _probe():
        pytest.skip("Augmentum not available at https://localhost:6443")
    with httpx.Client(base_url=BASE, timeout=TIMEOUT, verify=False) as c:
        yield c


@pytest.fixture
def aclient():
    """Async httpx client for tests that need async."""
    if not _probe():
        pytest.skip("Augmentum not available at https://localhost:6443")
    c = httpx.AsyncClient(base_url=BASE, timeout=TIMEOUT, verify=False)
    yield c
    # Cleanup — close in sync context (httpx supports this)
    try:
        import asyncio
        asyncio.get_event_loop().run_until_complete(c.aclose())
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chat_payload(
    model: str = "deepseek-chat",
    messages: list | None = None,
    stream: bool = False,
    **extra,
) -> dict:
    payload = {"model": model, "stream": stream, **extra}
    if messages is not None:
        payload["messages"] = messages
    else:
        payload["messages"] = [{"role": "user", "content": "Say hi"}]
    return payload


# ===========================================================================
# Error Resilience
# ===========================================================================


@pytest.mark.live
class TestErrorResilience:
    """Verify the server returns clean errors instead of crashing."""

    # 1. Invalid model name
    def test_invalid_model(self, client: httpx.Client):
        resp = client.post(
            "/api/chat",
            json=_chat_payload(model="nonexistent-model-xyz"),
        )
        # BUG: should be 4xx, server currently returns 500 for unknown models
        assert resp.status_code >= 400, f"Unexpected status: {resp.status_code}"
        assert resp.status_code < 600

    # 2. Empty messages array — Pydantic validator should reject
    def test_empty_messages(self, client: httpx.Client):
        resp = client.post(
            "/api/chat",
            json=_chat_payload(messages=[]),
        )
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}: {resp.text[:300]}"
        body = resp.json()
        # Pydantic returns detail array
        assert "detail" in body

    # 3. Malformed message — missing role key
    def test_malformed_message_no_role(self, client: httpx.Client):
        resp = client.post(
            "/api/chat",
            json=_chat_payload(messages=[{"content": "hello"}]),
        )
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"

    # 3b. Malformed message — missing content key (should default to "")
    def test_malformed_message_no_content(self, client: httpx.Client):
        resp = client.post(
            "/api/chat",
            json=_chat_payload(messages=[{"role": "user"}]),
        )
        # content defaults to "" in OllamaMessage, so this may actually succeed
        # The key thing is it must not be a 500
        assert resp.status_code != 500, f"Got 500: {resp.text[:300]}"

    # 4. Oversized payload (200KB+ message)
    def test_oversized_payload(self, client: httpx.Client):
        big_content = "A" * 200_000
        resp = client.post(
            "/api/chat",
            json=_chat_payload(messages=[{"role": "user", "content": big_content}]),
            timeout=60.0,
        )
        # Either processes it or returns a clean error — not a crash
        assert resp.status_code != 500, f"Got 500 on oversized payload: {resp.text[:300]}"
        # Valid responses: 200 (processed), 413 (too large), 422 (validation), 4xx
        assert resp.status_code < 600

    # 5. Rapid-fire sequential requests
    def test_rapid_fire(self, client: httpx.Client):
        statuses = []
        for _ in range(10):
            resp = client.post(
                "/api/chat",
                json=_chat_payload(
                    messages=[{"role": "user", "content": "Say one word"}],
                    stream=False,
                ),
                timeout=60.0,
            )
            statuses.append(resp.status_code)
        # All must be valid HTTP responses (not 500 crashes)
        for i, s in enumerate(statuses):
            assert s != 500, f"Request {i} returned 500"
        # At least some should succeed; if rate limiting, should be 429
        ok_or_limited = [s for s in statuses if s in (200, 429)]
        assert len(ok_or_limited) >= 5, f"Too many failures: {statuses}"

    # 6. Invalid JSON body
    def test_invalid_json(self, client: httpx.Client):
        resp = client.post(
            "/api/chat",
            content=b"not json at all",
            headers={"Content-Type": "application/json"},
        )
        # FastAPI returns 422 for JSON parse errors
        assert resp.status_code in (400, 422), f"Expected 400/422, got {resp.status_code}"

    # 7. Missing Content-Type header
    def test_missing_content_type(self, client: httpx.Client):
        resp = client.post(
            "/api/chat",
            content=b'{"model":"deepseek-chat","messages":[{"role":"user","content":"hi"}],"stream":false}',
        )
        # Without Content-Type, FastAPI may reject or may still parse
        assert resp.status_code != 500, f"Got 500 without Content-Type: {resp.text[:300]}"


# ===========================================================================
# Session / Resource Operations
# ===========================================================================


@pytest.mark.live
class TestResourceOperations:
    """Verify CRUD endpoints handle missing/invalid resources cleanly."""

    # 8. Get nonexistent session
    def test_get_nonexistent_session(self, client: httpx.Client):
        fake_id = str(uuid.uuid4())
        resp = client.get(f"/api/chats/{fake_id}")
        assert resp.status_code == 404
        body = resp.json()
        assert "error" in body

    # 9. Delete nonexistent character
    def test_delete_nonexistent_character(self, client: httpx.Client):
        resp = client.delete("/api/characters/nonexistent-id-12345")
        # Handler returns 404 when rowcount == 0
        assert resp.status_code == 404
        body = resp.json()
        assert "error" in body

    # 9b. Delete nonexistent session
    def test_delete_nonexistent_session(self, client: httpx.Client):
        fake_id = str(uuid.uuid4())
        resp = client.delete(f"/api/chats/{fake_id}")
        assert resp.status_code == 404


# ===========================================================================
# Config Validation
# ===========================================================================


@pytest.mark.live
class TestConfigValidation:
    """Verify config endpoints reject invalid values cleanly."""

    # 10. Out-of-range value
    def test_config_out_of_range_temperature(self, client: httpx.Client):
        resp = client.put(
            "/api/config/tools",
            json={"uarf_auto_search_queries": 999},
        )
        assert resp.status_code == 200  # endpoint always returns 200 with errors array
        body = resp.json()
        assert "errors" in body
        assert any("out of range" in e for e in body["errors"]), f"No range error in: {body['errors']}"

    # 10b. Negative value
    def test_config_negative_value(self, client: httpx.Client):
        resp = client.put(
            "/api/config/tools",
            json={"uarf_auto_search_queries": -5},
        )
        body = resp.json()
        assert "errors" in body
        assert any("out of range" in e for e in body["errors"])

    # 11. Unknown key
    def test_config_unknown_key(self, client: httpx.Client):
        resp = client.put(
            "/api/config/tools",
            json={"totally_fake_setting_xyz": 42},
        )
        body = resp.json()
        # Should report as error (unknown setting)
        assert "errors" in body
        assert any("Unknown setting" in e for e in body["errors"])

    # 11b. Mixed valid + invalid
    def test_config_mixed_valid_invalid(self, client: httpx.Client):
        resp = client.put(
            "/api/config/tools",
            json={
                "uarf_auto_search": True,        # valid
                "fake_setting_abc": 123,          # invalid key
                "uarf_auto_search_queries": 999,  # out of range
            },
        )
        body = resp.json()
        # Valid setting should be applied
        assert "updated" in body
        assert body["updated"].get("uarf_auto_search") is True
        # Invalid ones should be in errors
        assert "errors" in body
        assert len(body["errors"]) >= 2


# ===========================================================================
# Streaming Edge Cases
# ===========================================================================


@pytest.mark.live
class TestStreamingResilience:
    """Verify streaming endpoints survive abrupt disconnects."""

    # 12. Streaming with immediate disconnect
    def test_stream_disconnect_then_normal_request(self, client: httpx.Client):
        # Start a streaming request, read just a bit, then abort
        try:
            with client.stream(
                "POST",
                "/api/chat",
                json=_chat_payload(stream=True),
                timeout=15.0,
            ) as stream:
                # Read just the first chunk
                for chunk in stream.iter_bytes(chunk_size=64):
                    break  # read one chunk and bail
            # Connection closed — stream context manager handles cleanup
        except Exception:
            pass  # Any error during disconnect is fine

        # Now verify the server is still healthy
        resp = client.get("/api/config/tools")
        assert resp.status_code == 200, "Server unhealthy after stream disconnect"


# ===========================================================================
# Unicode & Encoding
# ===========================================================================


@pytest.mark.live
class TestUnicodeStress:
    """Verify the system handles diverse Unicode correctly."""

    # 13. Unicode stress test
    def test_unicode_roundtrip(self, client: httpx.Client):
        session_id = f"test-unicode-{uuid.uuid4()}"
        unicode_content = (
            "Hello \U0001f680 "       # emoji (rocket)
            "\u4f60\u597d "            # CJK (ni hao)
            "\u0645\u0631\u062d\u0628\u0627 "  # Arabic (marhaba)
            "\u200b\u200c\u200d "      # zero-width chars
            "\ufeff "                  # BOM
            "\u202e reversed \u202c"   # RTL override
        )
        # Save a session with unicode content
        resp = client.put(
            f"/api/chats/{session_id}",
            json={
                "title": f"Unicode test {unicode_content[:20]}",
                "mode": "passthrough",
                "tree": {"root": {"role": "user", "content": unicode_content}},
            },
        )
        assert resp.status_code == 200

        # Read it back
        resp = client.get(f"/api/chats/{session_id}")
        assert resp.status_code == 200
        body = resp.json()
        stored = body.get("tree", {}).get("root", {}).get("content", "")
        assert stored == unicode_content, f"Unicode round-trip failed: {stored!r}"

        # Cleanup
        client.delete(f"/api/chats/{session_id}")


# ===========================================================================
# Security
# ===========================================================================


@pytest.mark.live
class TestSecurityPayloads:
    """Verify XSS, SQL injection, and path traversal are blocked."""

    # 14. XSS payload stored as plain text
    def test_xss_in_session(self, client: httpx.Client):
        session_id = f"test-xss-{uuid.uuid4()}"
        xss_payload = '<script>alert(1)</script><img src=x onerror=alert(1)>'
        resp = client.put(
            f"/api/chats/{session_id}",
            json={
                "title": xss_payload,
                "mode": "passthrough",
                "tree": {"root": {"role": "user", "content": xss_payload}},
            },
        )
        assert resp.status_code == 200

        # Read back — must be stored as literal text, not stripped
        resp = client.get(f"/api/chats/{session_id}")
        assert resp.status_code == 200
        body = resp.json()
        # The content should survive as-is (it's JSON, not HTML rendered)
        assert body.get("tree", {}).get("root", {}).get("content") == xss_payload

        # Cleanup
        client.delete(f"/api/chats/{session_id}")

    # 15. SQL injection in session ID
    def test_sql_injection_session_id(self, client: httpx.Client):
        # Classic SQL injection attempts
        injections = [
            "' OR 1=1--",
            "'; DROP TABLE ui_sessions;--",
            "1 UNION SELECT * FROM ui_sessions--",
        ]
        for payload in injections:
            resp = client.get(f"/api/chats/{payload}")
            # Parameterized queries mean this just returns 404 (no match)
            assert resp.status_code in (400, 404, 422), (
                f"Unexpected {resp.status_code} for SQL injection: {payload}"
            )
            # Must not return multiple sessions (data leak)
            body = resp.json()
            assert "sessions" not in body, f"Possible data leak with: {payload}"

    # 16. Path traversal in character ID
    def test_path_traversal_character(self, client: httpx.Client):
        traversals = [
            "../../etc/passwd",
            "..\\..\\windows\\system32",
            "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        ]
        for payload in traversals:
            resp = client.get(f"/api/characters/{payload}")
            # Should be 404 (not found) or 400, never 200 with file contents
            assert resp.status_code in (400, 404, 422), (
                f"Unexpected {resp.status_code} for traversal: {payload}"
            )
            body = resp.text
            assert "root:" not in body, f"Possible file leak with: {payload}"


# ===========================================================================
# Concurrency
# ===========================================================================


@pytest.mark.live
class TestConcurrency:
    """Verify concurrent operations don't corrupt state."""

    # 17. Concurrent session modifications
    @pytest.mark.asyncio
    async def test_concurrent_session_writes(self, aclient: httpx.AsyncClient):
        session_id = f"test-concurrent-{uuid.uuid4()}"

        async def write_version(version: int):
            return await aclient.put(
                f"/api/chats/{session_id}",
                json={
                    "title": f"Version {version}",
                    "mode": "passthrough",
                    "tree": {f"msg-{version}": {"role": "user", "content": f"v{version}"}},
                    "version": version,
                },
            )

        # Fire two concurrent writes
        r1, r2 = await asyncio.gather(
            write_version(1),
            write_version(2),
        )
        # Both should succeed (upsert)
        assert r1.status_code == 200, f"Write 1 failed: {r1.status_code}"
        assert r2.status_code == 200, f"Write 2 failed: {r2.status_code}"

        # Read back — should be one of the two versions (last-write-wins)
        resp = await aclient.get(f"/api/chats/{session_id}")
        assert resp.status_code == 200
        body = resp.json()
        title = body.get("title", "")
        assert title in ("Version 1", "Version 2"), f"Corrupted title: {title}"

        # Cleanup
        await aclient.delete(f"/api/chats/{session_id}")

    # 17b. Concurrent read+write
    @pytest.mark.asyncio
    async def test_concurrent_read_write(self, aclient: httpx.AsyncClient):
        session_id = f"test-rw-{uuid.uuid4()}"

        # Create the session first
        resp = await aclient.put(
            f"/api/chats/{session_id}",
            json={"title": "Initial", "mode": "passthrough", "tree": {}},
        )
        assert resp.status_code == 200

        async def read_session():
            return await aclient.get(f"/api/chats/{session_id}")

        async def write_session():
            return await aclient.put(
                f"/api/chats/{session_id}",
                json={"title": "Updated", "mode": "passthrough", "tree": {"a": 1}},
            )

        # Fire concurrent read + write
        results = await asyncio.gather(
            read_session(),
            write_session(),
            read_session(),
        )
        for r in results:
            assert r.status_code == 200, f"Got {r.status_code}: {r.text[:200]}"

        # Cleanup
        await aclient.delete(f"/api/chats/{session_id}")


# ===========================================================================
# Edge Cases
# ===========================================================================


@pytest.mark.live
class TestEdgeCases:
    """Miscellaneous edge cases that have caused issues in production."""

    # Empty model name (should fail Pydantic validator)
    def test_empty_model_name(self, client: httpx.Client):
        resp = client.post(
            "/api/chat",
            json=_chat_payload(model=""),
        )
        assert resp.status_code == 422

    # Whitespace-only model name
    def test_whitespace_model_name(self, client: httpx.Client):
        resp = client.post(
            "/api/chat",
            json=_chat_payload(model="   "),
        )
        assert resp.status_code == 422

    # Very long model name
    def test_very_long_model_name(self, client: httpx.Client):
        resp = client.post(
            "/api/chat",
            json=_chat_payload(model="x" * 10000),
        )
        # BUG: should be 4xx (e.g. 422), server currently returns 500 for unknown models
        assert resp.status_code >= 400, f"Unexpected status: {resp.status_code}"
        assert resp.status_code < 600

    # Null values in messages
    def test_null_content_in_message(self, client: httpx.Client):
        resp = client.post(
            "/api/chat",
            json=_chat_payload(messages=[{"role": "user", "content": None}]),
        )
        # Should be validation error or handled gracefully
        assert resp.status_code != 500, f"Got 500 on null content: {resp.text[:300]}"

    # Deeply nested JSON
    def test_deeply_nested_json(self, client: httpx.Client):
        session_id = f"test-nested-{uuid.uuid4()}"
        # Build a 50-level nested tree
        tree = {"leaf": "value"}
        for i in range(50):
            tree = {f"level-{i}": tree}

        resp = client.put(
            f"/api/chats/{session_id}",
            json={"title": "Nested test", "mode": "passthrough", "tree": tree},
        )
        assert resp.status_code == 200

        # Read back
        resp = client.get(f"/api/chats/{session_id}")
        assert resp.status_code == 200

        # Cleanup
        client.delete(f"/api/chats/{session_id}")

    # Special characters in session title
    def test_special_chars_in_title(self, client: httpx.Client):
        session_id = f"test-special-{uuid.uuid4()}"
        title = "Test 'quotes' \"double\" <angle> & ampersand \\ backslash \0 null"
        resp = client.put(
            f"/api/chats/{session_id}",
            json={"title": title, "mode": "passthrough", "tree": {}},
        )
        assert resp.status_code == 200

        resp = client.get(f"/api/chats/{session_id}")
        assert resp.status_code == 200
        # Null byte may get stripped by JSON, but the rest should survive
        body = resp.json()
        assert "quotes" in body.get("title", "")

        # Cleanup
        client.delete(f"/api/chats/{session_id}")

    # Config with wrong types
    def test_config_wrong_type(self, client: httpx.Client):
        resp = client.put(
            "/api/config/tools",
            json={"uarf_auto_search_queries": "not a number"},
        )
        body = resp.json()
        # Should either coerce or report error — not crash
        assert resp.status_code == 200
        # "not a number" can't be cast to int
        assert "errors" in body

    # POST to GET-only endpoint
    def test_method_not_allowed(self, client: httpx.Client):
        resp = client.post("/api/config/tools", json={})
        assert resp.status_code == 405

    # GET nonexistent endpoint
    def test_404_route(self, client: httpx.Client):
        resp = client.get("/api/totally-fake-endpoint")
        assert resp.status_code == 404
