"""Smoke tests — module imports, router is registered, settings are wired."""

from __future__ import annotations


class TestCommunityImports:
    """Import every module that ships in the community subsystem."""

    def test_import_routes(self):
        from augmentum.proxy import community_routes  # noqa: F401

    def test_router_defined(self):
        from augmentum.proxy.community_routes import router

        assert router is not None
        paths = {route.path for route in router.routes}
        assert "/community-install" in paths
        assert "/api/community/install" in paths

    def test_known_categories(self):
        from augmentum.proxy.community_routes import _KNOWN_CATEGORIES

        assert "characters" in _KNOWN_CATEGORIES
        assert "reasoning-flows" in _KNOWN_CATEGORIES
        assert "powers" in _KNOWN_CATEGORIES
        assert "knowledge" in _KNOWN_CATEGORIES

    def test_builtin_trusted_origins(self):
        from augmentum.proxy.community_routes import _BUILTIN_TRUSTED_ORIGINS

        # The two canonical AugmentumHQ raw URLs (case variants) must be
        # default-allowlisted so the launch community items work without
        # any admin configuration.
        assert any("AugmentumHQ" in p for p in _BUILTIN_TRUSTED_ORIGINS)
        assert any("augmentumhq" in p for p in _BUILTIN_TRUSTED_ORIGINS)

    def test_settings_field_exists(self):
        """community_install_enabled must be declared on Settings."""
        from augmentum.config import Settings

        s = Settings()
        assert s.community_install_enabled is True
        assert s.community_trusted_origins == []
        assert s.community_max_pack_size_mb == 500

    def test_auth_middleware_exempts_route(self):
        """The /community-install route MUST be in the auth public-path
        set so cross-origin navigations from augmentumhq.com reach the
        handler (which then redirects to /login if no session) instead
        of getting fail-closed 401'd by the middleware."""
        from augmentum.auth.middleware import _PUBLIC_PATHS

        assert "/community-install" in _PUBLIC_PATHS

    def test_migration_present(self):
        """Migration 236 ships the community_installs table."""
        from pathlib import Path

        root = Path(__file__).parent.parent / "augmentum" / "state" / "migrations"
        files = list(root.glob("236_community_installs*.sql"))
        assert len(files) == 1
        body = files[0].read_text()
        assert "CREATE TABLE IF NOT EXISTS community_installs" in body
        assert "user_id" in body
        assert "REFERENCES users(id)" in body
        assert "ON DELETE CASCADE" in body
