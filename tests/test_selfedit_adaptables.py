"""Adaptable-settings catalog tests — discoverability for the Adapt lane."""

from __future__ import annotations

from augmentum.selfedit import adaptables


class _Store:
    def __init__(self, data=None):
        self.data = data or {}

    async def get_user_or_global(self, user_id, key):
        return self.data.get(key)


def test_catalog_keys_are_namespaced_and_real():
    # every catalogued key writes under the per-user ui.* prefix the app reads
    for a in adaptables.CATALOG:
        assert a.settings_key == f"ui.{a.key}"
        assert a.type in ("bool", "number", "text", "choice")
    assert adaptables.get_adaptable("voiceAutoRead").type == "bool"
    assert adaptables.get_adaptable("ui.voiceAutoRead") is adaptables.get_adaptable("voiceAutoRead")


async def test_catalog_with_values_fills_current_and_flags_set():
    store = _Store({"ui.thinkEnabled": "true", "ui.aiName": "Ada"})
    out = await adaptables.catalog_with_values(settings_store=store, user_id="u1")
    by_key = {d["key"]: d for d in out}
    assert by_key["thinkEnabled"]["value"] == "true" and by_key["thinkEnabled"]["is_set"] is True
    assert by_key["aiName"]["value"] == "Ada" and by_key["aiName"]["is_set"] is True
    # unset → falls to default display, not flagged set
    assert by_key["voiceSpeed"]["value"] == "1.0" and by_key["voiceSpeed"]["is_set"] is False


def test_auto_derives_from_app_settings_registry():
    # given the app's real settings dict, the catalog auto-extends: a NEW key the
    # app added shows up (type inferred), denylisted blobs don't, curated metadata
    # still sharpens known keys.
    ui_settings = {
        "voiceAutoRead": 8,        # curated → bool, nice label
        "newFancyToggle": 8,       # NEW bool (maxlen 8) — auto-appears
        "newReplyLimit": 8,        # NEW number (name hint) even at maxlen 8
        "aiName": 256,             # curated text
        "systemPrompt": 8000,      # denylisted blob — must NOT appear
    }
    by = {a.key: a for a in adaptables.derive(ui_settings)}
    assert by["voiceAutoRead"].type == "bool" and by["voiceAutoRead"].curated is True
    assert by["newFancyToggle"].type == "bool" and by["newFancyToggle"].curated is False
    assert by["newReplyLimit"].type == "number"          # "Limit" hint beats maxlen
    assert by["newFancyToggle"].label == "New Fancy Toggle"   # humanized from camelCase
    assert "systemPrompt" not in by                      # denylisted


async def test_value_lookup_never_raises():
    class _Boom:
        async def get_user_or_global(self, *_):
            raise RuntimeError("down")
    out = await adaptables.catalog_with_values(settings_store=_Boom(), user_id="u1")
    assert len(out) == len(adaptables.CATALOG)   # degraded, not crashed
