"""Live integration tests for the auth system.

Requires a running Augmentum server. Skipped when unavailable.
Run: pytest tests/live/test_live_auth.py -v --run-live

WARNING: These tests create and delete users on the running server.
They assume the server has an active session (already set up).
"""

from __future__ import annotations

import httpx
import pytest

BASE = "http://localhost:6100"


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


async def _probe() -> bool:
    """Check if server is reachable."""
    try:
        async with httpx.AsyncClient(base_url=BASE, timeout=5) as c:
            r = await c.get("/api/version")
            return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
async def auth_client():
    """Get an authenticated httpx client. Assumes server is set up."""
    if not await _probe():
        pytest.skip("Augmentum server not reachable")

    async with httpx.AsyncClient(base_url=BASE, timeout=30) as client:
        # Check if we can authenticate
        status = await client.get("/api/auth/status")
        data = status.json()

        if data.get("setup_required"):
            pytest.skip("Server requires setup — cannot run auth tests in this state")

        if not data.get("authenticated"):
            # Try to login with test credentials
            login = await client.post("/api/auth/login", json={
                "username": "admin",
                "password": "testpassword123",
            })
            if login.status_code != 200:
                pytest.skip("Cannot authenticate — unknown credentials")
            # Extract cookie
            client.cookies = login.cookies

        yield client


@pytest.mark.live
class TestAuthStatus:
    @pytest.mark.anyio
    async def test_status_endpoint_reachable(self, auth_client):
        r = await auth_client.get("/api/auth/status")
        assert r.status_code == 200
        data = r.json()
        assert "setup_required" in data
        assert "authenticated" in data

    @pytest.mark.anyio
    async def test_status_shows_authenticated(self, auth_client):
        r = await auth_client.get("/api/auth/status")
        data = r.json()
        assert data["authenticated"] is True
        assert data["user"] is not None
        assert "id" in data["user"]
        assert "username" in data["user"]


@pytest.mark.live
class TestAuthMe:
    @pytest.mark.anyio
    async def test_me_returns_profile(self, auth_client):
        r = await auth_client.get("/api/auth/me")
        assert r.status_code == 200
        data = r.json()
        assert "id" in data
        assert "username" in data
        assert "role" in data
        # Should NOT contain password hash
        assert "password" not in str(data).lower()

    @pytest.mark.anyio
    async def test_me_unauthenticated(self):
        """Unauthenticated request to /me should fail."""
        async with httpx.AsyncClient(base_url=BASE, timeout=10) as client:
            r = await client.get("/api/auth/me")
            assert r.status_code == 401


@pytest.mark.live
class TestWsTicket:
    @pytest.mark.anyio
    async def test_get_ticket(self, auth_client):
        r = await auth_client.post("/api/auth/ws-ticket")
        assert r.status_code == 200
        data = r.json()
        assert "ticket" in data
        assert len(data["ticket"]) > 16  # Should be hex string

    @pytest.mark.anyio
    async def test_ticket_unauthenticated(self):
        async with httpx.AsyncClient(base_url=BASE, timeout=10) as client:
            r = await client.post("/api/auth/ws-ticket")
            assert r.status_code == 401


@pytest.mark.live
class TestAdminUserManagement:
    """Tests for admin user CRUD. Creates and cleans up test users."""

    @pytest.mark.anyio
    async def test_list_users(self, auth_client):
        r = await auth_client.get("/api/auth/users")
        assert r.status_code == 200
        data = r.json()
        assert "users" in data
        assert len(data["users"]) >= 1  # At least the admin

    @pytest.mark.anyio
    async def test_create_and_delete_user(self, auth_client):
        """Full lifecycle: create → verify → delete."""
        # Create
        r = await auth_client.post("/api/auth/users", json={
            "username": "test_user_crud",
            "password": "testpass123",
            "role": "user",
        })
        assert r.status_code == 201
        user = r.json()["user"]
        user_id = user["id"]
        assert user["username"] == "test_user_crud"
        assert user["role"] == "user"

        # Verify in list
        r = await auth_client.get("/api/auth/users")
        usernames = [u["username"] for u in r.json()["users"]]
        assert "test_user_crud" in usernames

        # Delete
        r = await auth_client.delete(
            f"/api/auth/users/{user_id}",
            headers={"X-Confirm-Delete": "true"},
        )
        assert r.status_code == 200

        # Verify gone
        r = await auth_client.get("/api/auth/users")
        usernames = [u["username"] for u in r.json()["users"]]
        assert "test_user_crud" not in usernames

    @pytest.mark.anyio
    async def test_cannot_delete_self(self, auth_client):
        """Admin cannot delete themselves."""
        me = await auth_client.get("/api/auth/me")
        my_id = me.json()["id"]
        r = await auth_client.delete(
            f"/api/auth/users/{my_id}",
            headers={"X-Confirm-Delete": "true"},
        )
        assert r.status_code == 400

    @pytest.mark.anyio
    async def test_delete_requires_confirm_header(self, auth_client):
        """Delete without X-Confirm-Delete should fail."""
        # Create a user to try to delete
        r = await auth_client.post("/api/auth/users", json={
            "username": "test_no_confirm",
            "password": "testpass123",
            "role": "user",
        })
        user_id = r.json()["user"]["id"]

        # Try delete without header
        r = await auth_client.delete(f"/api/auth/users/{user_id}")
        assert r.status_code == 400

        # Cleanup
        await auth_client.delete(
            f"/api/auth/users/{user_id}",
            headers={"X-Confirm-Delete": "true"},
        )

    @pytest.mark.anyio
    async def test_duplicate_username_rejected(self, auth_client):
        """Cannot create two users with same username."""
        r = await auth_client.post("/api/auth/users", json={
            "username": "test_duplicate",
            "password": "testpass123",
            "role": "user",
        })
        user_id = r.json()["user"]["id"]

        r2 = await auth_client.post("/api/auth/users", json={
            "username": "test_duplicate",
            "password": "testpass456",
            "role": "user",
        })
        assert r2.status_code == 409

        # Cleanup
        await auth_client.delete(
            f"/api/auth/users/{user_id}",
            headers={"X-Confirm-Delete": "true"},
        )


@pytest.mark.live
class TestLoginSecurity:
    @pytest.mark.anyio
    async def test_wrong_password_returns_401(self):
        """Wrong password should return 401 with generic message."""
        async with httpx.AsyncClient(base_url=BASE, timeout=10) as client:
            r = await client.post("/api/auth/login", json={
                "username": "admin",
                "password": "definitely_wrong_password",
            })
            assert r.status_code == 401
            data = r.json()
            assert "Invalid username or password" in data.get("error", "")

    @pytest.mark.anyio
    async def test_nonexistent_user_returns_401(self):
        """Nonexistent username should return same 401 (no user enumeration)."""
        async with httpx.AsyncClient(base_url=BASE, timeout=10) as client:
            r = await client.post("/api/auth/login", json={
                "username": "user_does_not_exist_xyz",
                "password": "somepassword123",
            })
            assert r.status_code == 401
            data = r.json()
            assert "Invalid username or password" in data.get("error", "")

    @pytest.mark.anyio
    async def test_invalid_username_format(self):
        """Invalid username format should still return 401 (not 400)."""
        async with httpx.AsyncClient(base_url=BASE, timeout=10) as client:
            r = await client.post("/api/auth/login", json={
                "username": "",
                "password": "somepassword",
            })
            assert r.status_code == 401


@pytest.mark.live
class TestProtectedEndpoints:
    """Verify that protected endpoints reject unauthenticated requests."""

    @pytest.mark.anyio
    async def test_chat_requires_auth(self):
        async with httpx.AsyncClient(base_url=BASE, timeout=10) as client:
            r = await client.get("/api/chats/")
            assert r.status_code == 401

    @pytest.mark.anyio
    async def test_settings_requires_auth(self):
        async with httpx.AsyncClient(base_url=BASE, timeout=10) as client:
            r = await client.get("/api/config/tools")
            assert r.status_code == 401

    @pytest.mark.anyio
    async def test_admin_requires_admin_role(self, auth_client):
        """Create a regular user, login as them, verify admin endpoints fail."""
        # Create regular user
        r = await auth_client.post("/api/auth/users", json={
            "username": "test_regular_user",
            "password": "testpass123",
            "role": "user",
        })
        if r.status_code != 201:
            pytest.skip("Could not create test user")
        user_id = r.json()["user"]["id"]

        try:
            # Login as regular user
            async with httpx.AsyncClient(base_url=BASE, timeout=10) as regular:
                login = await regular.post("/api/auth/login", json={
                    "username": "test_regular_user",
                    "password": "testpass123",
                })
                if login.status_code != 200:
                    pytest.skip("Could not login as test user")
                regular.cookies = login.cookies

                # Try admin endpoint
                r = await regular.get("/api/auth/users")
                assert r.status_code == 403
        finally:
            # Cleanup
            await auth_client.delete(
                f"/api/auth/users/{user_id}",
                headers={"X-Confirm-Delete": "true"},
            )
