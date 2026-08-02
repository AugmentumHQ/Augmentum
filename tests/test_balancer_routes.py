"""Tests for balancer_routes.py — load balancer management API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def _setup_balancer_mocks(app):
    """Wire up mock balancer_store and lb_registry on app.state."""
    store = MagicMock()
    store.list_balancers = AsyncMock(return_value=[])
    store.get_balancer = AsyncMock(return_value=None)
    store.create_balancer = AsyncMock()
    store.update_balancer = AsyncMock(return_value=None)
    store.delete_balancer = AsyncMock(return_value=False)
    store.list_members = AsyncMock(return_value=[])
    store.add_member = AsyncMock()
    store.get_vote_stats = AsyncMock(return_value=[])
    app.state.balancer_store = store

    registry = MagicMock()
    registry.register = MagicMock()
    registry.unregister = MagicMock()
    app.state.lb_registry = registry
    return store, registry


# ---------------------------------------------------------------------------
# GET /api/balancers
# ---------------------------------------------------------------------------


class TestListBalancers:
    def test_no_store_returns_503(self, client):
        resp = client.get("/api/balancers")
        assert resp.status_code == 503

    def test_list_empty(self, client):
        _setup_balancer_mocks(client.app)
        resp = client.get("/api/balancers")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 0


# ---------------------------------------------------------------------------
# POST /api/balancers — create
# ---------------------------------------------------------------------------


class TestCreateBalancer:
    def test_create_success(self, client):
        store, registry = _setup_balancer_mocks(client.app)
        mock_result = MagicMock()
        mock_result.id = "lb_abc123"
        mock_result.name = "Test Balancer"
        mock_result.strategy = "round_robin"
        mock_result.fallback_enabled = False
        mock_result.enabled = True
        store.create_balancer = AsyncMock(return_value=mock_result)

        resp = client.post("/api/balancers", json={
            "name": "Test Balancer",
            "strategy": "round_robin",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Test Balancer"
        assert data["strategy"] == "round_robin"

    def test_create_empty_name_returns_400(self, client):
        _setup_balancer_mocks(client.app)
        resp = client.post("/api/balancers", json={
            "name": "  ",
            "strategy": "round_robin",
        })
        assert resp.status_code == 400

    def test_create_invalid_strategy_returns_400(self, client):
        _setup_balancer_mocks(client.app)
        resp = client.post("/api/balancers", json={
            "name": "Test",
            "strategy": "invalid_strategy",
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# PUT /api/balancers/{id}
# ---------------------------------------------------------------------------


class TestUpdateBalancer:
    def test_update_not_found_returns_404(self, client):
        store, _ = _setup_balancer_mocks(client.app)
        store.update_balancer = AsyncMock(return_value=None)

        resp = client.put("/api/balancers/lb_nonexistent", json={"name": "Updated"})
        assert resp.status_code == 404

    def test_update_invalid_strategy_returns_400(self, client):
        _setup_balancer_mocks(client.app)
        resp = client.put("/api/balancers/lb_test", json={"strategy": "bad"})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# DELETE /api/balancers/{id}
# ---------------------------------------------------------------------------


class TestDeleteBalancer:
    def test_delete_not_found_returns_404(self, client):
        store, _ = _setup_balancer_mocks(client.app)
        store.delete_balancer = AsyncMock(return_value=False)

        resp = client.delete("/api/balancers/lb_nonexistent")
        assert resp.status_code == 404

    def test_delete_success(self, client):
        store, _ = _setup_balancer_mocks(client.app)
        store.delete_balancer = AsyncMock(return_value=True)

        resp = client.delete("/api/balancers/lb_test")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True


# ---------------------------------------------------------------------------
# GET /api/balancers/{id}/members
# ---------------------------------------------------------------------------


class TestBalancerMembers:
    def test_list_members_empty(self, client):
        _setup_balancer_mocks(client.app)
        resp = client.get("/api/balancers/lb_test/members")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_add_member_not_found_returns_404(self, client):
        store, _ = _setup_balancer_mocks(client.app)
        store.get_balancer = AsyncMock(return_value=None)

        resp = client.post("/api/balancers/lb_nonexistent/members", json={
            "model_name": "llama3.1:8b",
            "backend_key": "ollama",
        })
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/balancers/{id}/stats
# ---------------------------------------------------------------------------


class TestBalancerStats:
    def test_stats_returns_shape(self, client):
        _setup_balancer_mocks(client.app)
        resp = client.get("/api/balancers/lb_test/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "balancer_id" in data
        assert "models" in data
        assert isinstance(data["models"], list)


# ---------------------------------------------------------------------------
# POST /api/balancers/{id}/vote
# ---------------------------------------------------------------------------


class TestBalancerVote:
    def test_vote_invalid_value_returns_400(self, client):
        _setup_balancer_mocks(client.app)
        resp = client.post("/api/balancers/lb_test/vote", json={
            "model_name": "model-a",
            "backend_key": "ollama",
            "vote": "sideways",
        })
        assert resp.status_code == 400

    def test_vote_nonexistent_balancer_returns_404(self, client):
        store, _ = _setup_balancer_mocks(client.app)
        store.get_balancer = AsyncMock(return_value=None)

        resp = client.post("/api/balancers/lb_fake/vote", json={
            "model_name": "model-a",
            "backend_key": "ollama",
            "vote": "up",
        })
        assert resp.status_code == 404
