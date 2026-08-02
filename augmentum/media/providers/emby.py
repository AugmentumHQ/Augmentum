"""Emby media-server provider."""

from __future__ import annotations

from augmentum.media.providers.emby_compat import EmbyCompatBase


class EmbyProvider(EmbyCompatBase):
    name = "emby"
    api_prefix = "/emby"
    auth_scheme = "Emby"
    item_list_path_uses_user = True
    user_views_path_uses_user = True
    user_data_path_uses_user = True
