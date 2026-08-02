"""Toy user-config lookup."""

from __future__ import annotations


class Config:
    def __init__(self) -> None:
        self.user_settings: dict[str, dict] | None = None

    def get_theme(self, user_id: str) -> str:
        # BUG: user_settings can be None (initial state). Calling .get on it
        # raises AttributeError; the caller doesn't expect that.
        return self.user_settings.get(user_id, {}).get("theme", "default")


def demo(cfg: Config, user: str) -> str:
    return cfg.get_theme(user)
