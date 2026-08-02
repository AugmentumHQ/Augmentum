"""Tests for the multi-model fan-out plan endpoint + settings wiring.

POST /api/chats/fanout-plan groups compare models by backend so the UI
knows which can stream in parallel and which must serialize on a
single-slot local engine.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from augmentum.config import settings


# ---------------------------------------------------------------------------
# POST /api/chats/fanout-plan
# ---------------------------------------------------------------------------


class TestFanoutPlan:
    def test_empty_models_returns_empty_plan(self, client):
        resp = client.post("/api/chats/fanout-plan", json={"models": []})
        assert resp.status_code == 200
        assert resp.json()["plan"] == []

    def test_malformed_body_returns_empty_plan(self, client):
        resp = client.post(
            "/api/chats/fanout-plan",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.json()["plan"] == []

    def test_models_resolve_in_order(self, client):
        resp = client.post(
            "/api/chats/fanout-plan",
            json={"models": ["llama3.1:8b", "gpt-test"]},
        )
        assert resp.status_code == 200
        plan = resp.json()["plan"]
        assert [p["model"] for p in plan] == ["llama3.1:8b", "gpt-test"]
        for p in plan:
            # Mock registry exposes no _backends map — key stays unknown,
            # and the mock backend class is not a single-slot local engine.
            assert p["exclusive"] is False

    def test_duplicate_and_blank_models_deduped(self, client):
        resp = client.post(
            "/api/chats/fanout-plan",
            json={"models": ["m1", "m1", "  ", "", "m2", 42]},
        )
        plan = resp.json()["plan"]
        assert [p["model"] for p in plan] == ["m1", "m2"]

    def test_resolve_failure_surfaces_error(self, client, app):
        app.state.provider_registry.resolve_backend_for_model = AsyncMock(
            side_effect=ValueError("model not found"),
        )
        resp = client.post("/api/chats/fanout-plan", json={"models": ["ghost"]})
        assert resp.status_code == 200
        plan = resp.json()["plan"]
        assert plan[0]["model"] == "ghost"
        assert plan[0]["exclusive"] is False
        assert "model not found" in plan[0].get("error", "")

    def test_local_engine_marked_exclusive(self, client, app):
        class AugmentumEngineBackend:  # name match is the contract
            pass

        fake_engine = AugmentumEngineBackend()
        app.state.provider_registry.resolve_backend_for_model = AsyncMock(
            return_value=(fake_engine, "local-model"),
        )
        app.state.provider_registry._backends = {"engine": fake_engine}

        resp = client.post("/api/chats/fanout-plan", json={"models": ["local-model"]})
        plan = resp.json()["plan"]
        assert plan[0]["backend"] == "engine"
        assert plan[0]["exclusive"] is True

    def test_secondary_engine_key_marked_exclusive(self, client, app):
        """Slot B (engine_secondary) is exclusive but a distinct backend key,
        so two local models on separate slots get separate plan groups."""

        class AugmentumEngineBackend:
            pass

        class SecondarySlotBackend:  # unknown class — key prefix carries it
            pass

        primary = AugmentumEngineBackend()
        secondary = SecondarySlotBackend()

        async def _resolve(name):
            return (secondary if name == "pinned-b" else primary, name)

        app.state.provider_registry.resolve_backend_for_model = AsyncMock(
            side_effect=_resolve,
        )
        app.state.provider_registry._backends = {
            "engine": primary,
            "engine_secondary": secondary,
        }

        resp = client.post(
            "/api/chats/fanout-plan",
            json={"models": ["model-a", "pinned-b"]},
        )
        plan = resp.json()["plan"]
        assert plan[0]["backend"] == "engine"
        assert plan[0]["exclusive"] is True
        assert plan[1]["backend"] == "engine_secondary"
        assert plan[1]["exclusive"] is True
        # Distinct backend keys → the UI runs these two in parallel.
        assert plan[0]["backend"] != plan[1]["backend"]


# ---------------------------------------------------------------------------
# Settings wiring round-trip
# ---------------------------------------------------------------------------


class TestMultiModelSettings:
    def test_settings_roundtrip(self, client):
        orig_enabled = settings.multi_model_enabled
        orig_models = settings.multi_model_models
        try:
            resp = client.put(
                "/api/config/tools",
                json={
                    "multi_model_enabled": True,
                    "multi_model_models": "gpt-test,claude-test",
                },
            )
            assert resp.status_code == 200

            resp = client.get("/api/config/tools")
            data = resp.json()
            assert data["multi_model_enabled"] is True
            assert data["multi_model_models"] == "gpt-test,claude-test"
        finally:
            object.__setattr__(settings, "multi_model_enabled", orig_enabled)
            object.__setattr__(settings, "multi_model_models", orig_models)

    def test_models_string_length_capped(self, client):
        orig = settings.multi_model_models
        try:
            resp = client.put(
                "/api/config/tools",
                json={"multi_model_models": "x" * 5000},
            )
            assert resp.status_code == 200
            assert len(settings.multi_model_models) <= 2048
        finally:
            object.__setattr__(settings, "multi_model_models", orig)


# ---------------------------------------------------------------------------
# Concurrent fan-out streams (server side)
# ---------------------------------------------------------------------------


class TestConcurrentFanoutStreams:
    """The UI fan-out sends N independent /api/chat requests sharing one
    X-Augmentum-Session. Passthrough handlers are built per-request (not
    cached), so concurrent per-model streams on one session must not
    contend — this pins that property."""

    def _send(self, client, model, session_id):
        import json as _json

        with client.stream(
            "POST",
            "/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
            headers={
                "X-Augmentum-Mode": "passthrough",
                "X-Augmentum-Session": session_id,
            },
        ) as resp:
            assert resp.status_code == 200
            content = ""
            done = False
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                data = _json.loads(line)
                msg = data.get("message") or {}
                content += msg.get("content", "")
                if data.get("done"):
                    done = True
                    break
            assert done, f"stream for {model} never emitted done"
            return content

    def test_two_concurrent_streams_same_session(self, client, app, mock_backend):
        """Two streams, one session, same model — the best-of-2 fan-out
        case (and the only model the mock catalog knows; unknown names
        are rejected by request validation before reaching the handler,
        which the UI never does since picks come from the catalog).

        ``resolve_backend_with_fabric`` is configured here because the
        shared conftest mock predates fabric routing and never gained
        it — without this, /api/chat 400s on a mock-unpack ValueError
        (the same pre-existing gap that currently fails
        test_ollama_routes::test_chat_non_streaming)."""
        from concurrent.futures import ThreadPoolExecutor

        app.state.provider_registry.resolve_backend_with_fabric = AsyncMock(
            return_value=(mock_backend, "llama3.1:8b"),
        )

        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = [
                ex.submit(self._send, client, "llama3.1:8b", "fanout-sess-1")
                for _ in range(2)
            ]
            results = [f.result(timeout=60) for f in futs]
        assert all(isinstance(r, str) and r for r in results)
