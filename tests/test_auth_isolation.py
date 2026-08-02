"""Tests that verify cross-tenant data isolation patterns."""

from __future__ import annotations


class TestScopingHelpers:
    def test_user_where(self):
        from augmentum.auth.scoping import user_where
        clause, params = user_where("usr_abc123")
        assert "user_id = ?" in clause
        assert params == ("usr_abc123",)

    def test_user_insert_fields(self):
        from augmentum.auth.scoping import user_insert_fields
        assert user_insert_fields() == ", user_id"

    def test_user_insert_placeholder(self):
        from augmentum.auth.scoping import user_insert_placeholder
        assert user_insert_placeholder() == ", ?"


class TestUserModel:
    def test_to_public_dict_no_password(self):
        from augmentum.auth.models import User
        u = User(id="usr_1", username="testuser", role="admin")
        d = u.to_public_dict()
        assert "password" not in str(d).lower()
        assert d["id"] == "usr_1"
        assert d["role"] == "admin"

    def test_is_admin(self):
        from augmentum.auth.models import User
        admin = User(id="usr_1", username="admin", role="admin")
        user = User(id="usr_2", username="user", role="user")
        assert admin.is_admin is True
        assert user.is_admin is False
