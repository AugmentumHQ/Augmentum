"""Catalog smoke + per-kind invariants.

These tests pin the *shape* of each registered catalog kind without
exercising the underlying install paths (that's the job of each
subsystem's tests). The goal is to catch:

* Kind registration regressions (a catalog stops loading silently).
* CatalogEntry schema drift (missing build_preview / accept / etc.).
* Preview return-type drift (None vs OfferPreview).
* Reference-entry presence — the canonical examples each phase
  promises must keep existing.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from augmentum.offers.catalog import base
# Importing the offers package wires up every catalog kind via
# augmentum/offers/catalog/__init__.py's side-effect imports.
from augmentum.offers import catalog as _catalog_root  # noqa: F401


MIG_221 = Path("augmentum/state/migrations/221_notification_substrate.sql").read_text()
MIG_224 = Path("augmentum/state/migrations/224_offer_suppressions.sql").read_text()


U1 = "user-alpha"


@pytest.fixture
async def conn():
    async with aiosqlite.connect(":memory:") as c:
        await c.executescript(MIG_221)
        await c.executescript(MIG_224)
        await c.commit()
        yield c


def _all_entries():
    out = []
    for kind in base.list_kinds():
        for target in base.list_targets(kind):
            out.append(base.get_entry(kind, target))
    return out


# ── Shape invariants (apply to every entry, every kind) ──────────


class TestEntryShape:
    def test_every_entry_has_required_fields(self) -> None:
        entries = _all_entries()
        assert entries, "expected at least one catalog entry registered"
        for e in entries:
            assert e.kind, f"entry missing kind: {e}"
            assert e.target_id, f"entry missing target_id: {e}"
            assert e.title, f"entry missing title: {e.target_id}"
            assert e.scope in ("user", "admin"), (
                f"unknown scope {e.scope!r} on {e.kind}/{e.target_id}"
            )
            assert callable(e.build_preview), (
                f"build_preview missing on {e.kind}/{e.target_id}"
            )
            assert callable(e.accept), (
                f"accept missing on {e.kind}/{e.target_id}"
            )

    def test_unique_target_ids_per_kind(self) -> None:
        for kind in base.list_kinds():
            targets = base.list_targets(kind)
            assert len(targets) == len(set(targets)), (
                f"duplicate target_id in {kind}: {targets}"
            )


# ── Per-kind canonical-entry presence ────────────────────────────


class TestCanonicalEntries:
    """Each kind has reference entries the spec / docs name.

    If a future refactor accidentally drops one (e.g. renames
    ``mcp_server:linear``), the test fires with a clear message.
    """

    def test_mcp_server_canonical_entries(self) -> None:
        # The 10 vendor remote-HTTP + 6 Anthropic stdio + Gmail
        # reference = 17. Verifying a representative subset rather
        # than the exact count so adding a new entry isn't a test
        # churn — but a missing flagship entry IS.
        targets = set(base.list_targets("mcp_server"))
        for canonical in ("linear", "notion", "stripe", "atlassian", "cloudflare",
                          "sentry", "github_copilot", "fetch", "filesystem",
                          "memory", "gmail"):
            assert canonical in targets, (
                f"mcp_server:{canonical} dropped from catalog"
            )

    def test_mcp_client_config_canonical_entries(self) -> None:
        targets = set(base.list_targets("mcp_client_config"))
        for canonical in ("claude-desktop", "cursor", "cline", "continue",
                          "windsurf", "vscode", "zed", "generic"):
            assert canonical in targets, (
                f"mcp_client_config:{canonical} dropped from catalog"
            )

    def test_subagent_role_canonical_entries(self) -> None:
        targets = set(base.list_targets("subagent_role"))
        for canonical in ("explore", "plan", "review", "research",
                          "security_review", "threat_model"):
            assert canonical in targets, (
                f"subagent_role:{canonical} dropped — check BUILTIN_ROLES"
            )

    def test_power_kind_has_entries(self) -> None:
        # Power catalog is filesystem-driven; can be empty in test
        # environments without .augmentum/powers/. Only assert the
        # kind registers when discovery actually found manifests.
        if "power" in base.list_kinds():
            assert base.list_targets("power"), (
                "power kind registered but empty"
            )

    def test_knowledge_pack_canonical_entries(self) -> None:
        # ZIM external entries should always be present (they're
        # hardcoded, not discovery-driven).
        targets = set(base.list_targets("knowledge_pack"))
        for canonical in ("zim:wikipedia_en", "zim:mdwiki", "zim:gutenberg"):
            assert canonical in targets, (
                f"knowledge_pack:{canonical} dropped from catalog"
            )

    def test_mode_switch_canonical_entries(self) -> None:
        # Every Mode enum value must round-trip through the catalog.
        # Missing one means a recent enum edit didn't propagate.
        targets = set(base.list_targets("mode_switch"))
        for canonical in ("passthrough", "analytical", "narrative",
                          "agentic", "coder", "direct", "becca_direct"):
            assert canonical in targets, (
                f"mode_switch:{canonical} dropped from catalog"
            )

    def test_model_swap_canonical_entries(self) -> None:
        targets = set(base.list_targets("model_swap"))
        for canonical in ("heavyweight", "utility"):
            assert canonical in targets, (
                f"model_swap:{canonical} dropped from catalog"
            )

    def test_setting_tweak_canonical_entries(self) -> None:
        targets = set(base.list_targets("setting_tweak"))
        for canonical in ("ghost_text", "knowledge_in_chat",
                          "emotion_aware_tts", "voice_moonshine",
                          "companion_dispatch"):
            assert canonical in targets, (
                f"setting_tweak:{canonical} dropped from whitelist"
            )

    def test_workspace_profile_canonical_entries(self) -> None:
        # 3 profiles from tooling-profile-system-v2.
        targets = set(base.list_targets("workspace_profile"))
        for canonical in ("standard", "power", "browser"):
            assert canonical in targets, (
                f"workspace_profile:{canonical} dropped from catalog"
            )

    def test_memory_save_canonical_entries(self) -> None:
        # 3 user-facing memory kinds; internal types (entity, analysis,
        # skill, relationship) are extraction-only — NOT offer-able.
        targets = set(base.list_targets("memory_save"))
        for canonical in ("preference", "fact", "instruction"):
            assert canonical in targets, (
                f"memory_save:{canonical} dropped from catalog"
            )

    def test_coder_only_kinds_are_mode_gated(self) -> None:
        # Entries that are meaningless outside coder mode must declare
        # ``allowed_modes=("coder",)`` so the dispatcher rejects them
        # when surfaced from passthrough/narrative — otherwise the
        # user gets an inert chip they can accept but observes no
        # effect (powers wait for activation, workspace creation is
        # contextually wrong, subagent role files only feed
        # task_dispatch).
        coder_only_kinds = ("power", "workspace_profile", "subagent_role")
        for kind in coder_only_kinds:
            for target in base.list_targets(kind):
                entry = base.get_entry(kind, target)
                assert entry.allowed_modes == ("coder",), (
                    f"{kind}:{target} must restrict allowed_modes to coder; "
                    f"got {entry.allowed_modes!r}"
                )

    def test_universal_kinds_are_not_mode_gated(self) -> None:
        # Sanity: cross-mode kinds (mode_switch, model_swap, memory_save,
        # etc.) must NOT restrict — they're useful from anywhere.
        universal = ("mode_switch", "model_swap", "setting_tweak",
                     "memory_save", "knowledge_pack", "mcp_server",
                     "mcp_client_config")
        for kind in universal:
            for target in base.list_targets(kind):
                entry = base.get_entry(kind, target)
                assert entry.allowed_modes == (), (
                    f"{kind}:{target} unexpectedly mode-gated to "
                    f"{entry.allowed_modes!r} — these kinds should be "
                    "surfaceable from any mode"
                )


# ── Preview builders return correctly-typed values ──────────────


@pytest.mark.asyncio
class TestPreviewContracts:
    async def test_mcp_server_previews_render_when_not_installed(self) -> None:
        from augmentum.config import settings
        prev_servers = getattr(settings, "mcp_servers", "")
        object.__setattr__(settings, "mcp_servers", "")  # ensure none installed
        try:
            entry = base.get_entry("mcp_server", "linear")
            preview = await entry.build_preview("linear", U1)
            assert preview is not None
            assert preview.label
            assert preview.details.get("transport") == "http"
            assert preview.details.get("url") == "https://mcp.linear.app/mcp"
        finally:
            object.__setattr__(settings, "mcp_servers", prev_servers)

    async def test_mcp_server_previews_skipped_when_installed(self) -> None:
        from augmentum.config import settings
        import json
        prev_servers = getattr(settings, "mcp_servers", "")
        object.__setattr__(
            settings, "mcp_servers",
            json.dumps([{"name": "linear", "url": "https://mcp.linear.app/mcp"}]),
        )
        try:
            entry = base.get_entry("mcp_server", "linear")
            preview = await entry.build_preview("linear", U1)
            assert preview is None  # already installed → skip
        finally:
            object.__setattr__(settings, "mcp_servers", prev_servers)

    async def test_mcp_client_config_preview_renders(self) -> None:
        entry = base.get_entry("mcp_client_config", "cursor")
        preview = await entry.build_preview("cursor", U1)
        assert preview is not None
        assert preview.details.get("scope") == "user"
        assert preview.details.get("transport") == "client-config"

    async def test_subagent_role_preview_renders(self) -> None:
        entry = base.get_entry("subagent_role", "explore")
        # Preview returns None when the role file already exists
        # locally (every dev machine has its own state); in CI / a
        # fresh container the file is absent and the preview renders.
        # Tolerate either branch — what matters is the call doesn't
        # raise.
        preview = await entry.build_preview("explore", U1)
        if preview is not None:
            assert "explore" in preview.label.lower() or "subagent" in preview.label.lower()

    async def test_mode_switch_preview_renders(self) -> None:
        entry = base.get_entry("mode_switch", "direct")
        preview = await entry.build_preview("direct", U1)
        assert preview is not None
        assert preview.details.get("mode") == "direct"
        assert preview.details.get("scope") == "user"

    async def test_model_swap_preview_skipped_when_unconfigured(self) -> None:
        # When the install has no heavyweight model configured, the
        # entry shouldn't surface — accept would fail anyway.
        from augmentum.config import settings
        prev = getattr(settings, "heavyweight_model", "")
        object.__setattr__(settings, "heavyweight_model", "")
        try:
            entry = base.get_entry("model_swap", "heavyweight")
            preview = await entry.build_preview("heavyweight", U1)
            assert preview is None
        finally:
            object.__setattr__(settings, "heavyweight_model", prev)

    async def test_model_swap_preview_renders_when_configured(self) -> None:
        from augmentum.config import settings
        prev = getattr(settings, "heavyweight_model", "")
        object.__setattr__(settings, "heavyweight_model", "qwen3-30b-thinking")
        try:
            entry = base.get_entry("model_swap", "heavyweight")
            preview = await entry.build_preview("heavyweight", U1)
            assert preview is not None
            assert preview.details.get("resolved_model") == "qwen3-30b-thinking"
        finally:
            object.__setattr__(settings, "heavyweight_model", prev)

    async def test_setting_tweak_preview_skipped_when_already_set(self) -> None:
        # ghost_text_enabled defaults to False — flipping to True
        # should make the preview render; setting it to True
        # directly should make it return None.
        from augmentum.config import settings
        prev = getattr(settings, "ghost_text_enabled", False)
        object.__setattr__(settings, "ghost_text_enabled", True)
        try:
            entry = base.get_entry("setting_tweak", "ghost_text")
            preview = await entry.build_preview("ghost_text", U1)
            assert preview is None
        finally:
            object.__setattr__(settings, "ghost_text_enabled", prev)

    async def test_workspace_profile_preview_renders(self) -> None:
        entry = base.get_entry("workspace_profile", "browser")
        preview = await entry.build_preview("browser", U1)
        assert preview is not None
        assert preview.details.get("profile_id") == "browser"

    async def test_memory_save_preview_renders(self) -> None:
        entry = base.get_entry("memory_save", "preference")
        preview = await entry.build_preview("preference", U1)
        assert preview is not None
        assert preview.details.get("memory_type") == "preference"
        assert preview.details.get("source_type") == "explicit"


# ── Accept handlers don't crash on minimal inputs ────────────────


def _make_request_with_app_state(**state_kwargs):
    """Build a FastAPI Request-shaped MagicMock.

    The accept handlers reach into ``request.app.state`` and
    ``request.scope.get('user')`` — those are the only surfaces we
    actually need. Test doesn't go through the full ASGI stack.
    """
    req = MagicMock()
    req.app = MagicMock()
    req.app.state = SimpleNamespace(**state_kwargs)
    req.scope = {"user": SimpleNamespace(id=U1, is_admin=True)}
    # base_url is a starlette URL — stringify reproduces the host.
    # Used by mcp_clients.py to derive the Augmentum /mcp URL.
    req.base_url = "https://augmentum.local"
    return req


@pytest.mark.asyncio
class TestAcceptHandlerSafety:
    async def test_mcp_client_config_accept_returns_snippet(self) -> None:
        entry = base.get_entry("mcp_client_config", "cursor")
        req = _make_request_with_app_state()
        result = await entry.accept(
            {"kind": "mcp_client_config", "target_id": "cursor", "extra": {}},
            req,
        )
        assert result["ok"] is True
        assert result["kind"] == "snippet"
        assert "augmentum" in result["snippet"]
        assert "https://augmentum.local/mcp" in result["snippet"]

    async def test_mcp_client_config_continue_emits_yaml(self) -> None:
        entry = base.get_entry("mcp_client_config", "continue")
        req = _make_request_with_app_state()
        result = await entry.accept(
            {"kind": "mcp_client_config", "target_id": "continue", "extra": {}},
            req,
        )
        assert result["ok"] is True
        assert result["language"] == "yaml"
        # YAML-form Continue config uses top-level `mcpServers:` list.
        assert "mcpServers:" in result["snippet"]
        assert "type: streamable-http" in result["snippet"]

    async def test_knowledge_pack_zim_returns_link(self) -> None:
        entry = base.get_entry("knowledge_pack", "zim:wikipedia_en")
        req = _make_request_with_app_state()
        result = await entry.accept(
            {"kind": "knowledge_pack", "target_id": "zim:wikipedia_en"},
            req,
        )
        assert result["ok"] is True
        assert result["kind"] == "external_link"
        assert "kiwix" in result["url"].lower()

    async def test_power_accept_requires_store(self) -> None:
        # Without ``power_state_store`` on app.state, returns a
        # friendly error rather than crashing.
        if not base.list_targets("power"):
            pytest.skip("no power entries in this environment")
        target = base.list_targets("power")[0]
        entry = base.get_entry("power", target)
        req = _make_request_with_app_state()  # no power_state_store
        result = await entry.accept(
            {"kind": "power", "target_id": target,
             "extra": {"_workspace_id": "ws-1"}}, req,
        )
        assert result["ok"] is False
        assert result["error"] == "no_power_state_store"

    async def test_power_accept_requires_workspace_id(self) -> None:
        # Workspace_id is stashed in ``extra['_workspace_id']`` by
        # the propose_offer tool; missing it means the offer was
        # proposed without a workspace context. Reject rather than
        # falling through to a no-op or implicit lookup.
        if not base.list_targets("power"):
            pytest.skip("no power entries in this environment")
        target = base.list_targets("power")[0]
        entry = base.get_entry("power", target)
        store = MagicMock()
        req = _make_request_with_app_state(power_state_store=store)
        result = await entry.accept(
            {"kind": "power", "target_id": target}, req,  # no extra
        )
        assert result["ok"] is False
        assert result["error"] == "no_workspace"

    async def test_power_accept_activates_on_workspace(self) -> None:
        if not base.list_targets("power"):
            pytest.skip("no power entries in this environment")
        target = base.list_targets("power")[0]
        entry = base.get_entry("power", target)

        store = MagicMock()
        store.get_active_power = AsyncMock(return_value=None)
        store.activate_power = AsyncMock()
        req = _make_request_with_app_state(power_state_store=store)

        result = await entry.accept(
            {
                "kind": "power", "target_id": target,
                "extra": {"_workspace_id": "ws-42"},
            },
            req,
        )
        assert result["ok"] is True
        assert result["already_active"] is False
        assert result.get("replaced", "") == ""
        store.activate_power.assert_awaited_once()
        call_kwargs = store.activate_power.await_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-42"
        assert call_kwargs["power_id"] == target
        assert call_kwargs["source"] == "offer"

    async def test_power_accept_already_active_is_idempotent(self) -> None:
        if not base.list_targets("power"):
            pytest.skip("no power entries in this environment")
        target = base.list_targets("power")[0]
        entry = base.get_entry("power", target)

        existing = SimpleNamespace(power_id=target, workspace_id="ws-42")
        store = MagicMock()
        store.get_active_power = AsyncMock(return_value=existing)
        store.activate_power = AsyncMock()
        req = _make_request_with_app_state(power_state_store=store)

        result = await entry.accept(
            {
                "kind": "power", "target_id": target,
                "extra": {"_workspace_id": "ws-42"},
            },
            req,
        )
        assert result["ok"] is True
        assert result["already_active"] is True
        store.activate_power.assert_not_awaited()

    async def test_power_accept_replaces_prior_active(self) -> None:
        if not base.list_targets("power"):
            pytest.skip("no power entries in this environment")
        # Need 2 powers to test the "replaces another" case.
        targets = base.list_targets("power")
        if len(targets) < 2:
            pytest.skip("need >= 2 powers to test replace path")
        new_target = targets[0]
        old_target = targets[1]
        entry = base.get_entry("power", new_target)

        existing = SimpleNamespace(power_id=old_target, workspace_id="ws-42")
        store = MagicMock()
        store.get_active_power = AsyncMock(return_value=existing)
        store.activate_power = AsyncMock()
        req = _make_request_with_app_state(power_state_store=store)

        result = await entry.accept(
            {
                "kind": "power", "target_id": new_target,
                "extra": {"_workspace_id": "ws-42"},
            },
            req,
        )
        assert result["ok"] is True
        assert result["already_active"] is False
        assert result["replaced"] == old_target
        assert "replacing" in result["next_step"].lower()
        store.activate_power.assert_awaited_once()

    async def test_mode_switch_accept_writes_user_setting(self) -> None:
        entry = base.get_entry("mode_switch", "direct")
        store = MagicMock()
        store.get_user = AsyncMock(return_value=None)
        store.set_user = AsyncMock()
        req = _make_request_with_app_state(settings_store=store)

        result = await entry.accept(
            {"kind": "mode_switch", "target_id": "direct"}, req,
        )
        assert result["ok"] is True
        assert result["mode"] == "direct"
        assert result["already_pinned"] is False
        store.set_user.assert_awaited_once_with(U1, "default_mode", "direct")

    async def test_mode_switch_accept_idempotent(self) -> None:
        entry = base.get_entry("mode_switch", "direct")
        store = MagicMock()
        store.get_user = AsyncMock(return_value="direct")  # already pinned
        store.set_user = AsyncMock()
        req = _make_request_with_app_state(settings_store=store)

        result = await entry.accept(
            {"kind": "mode_switch", "target_id": "direct"}, req,
        )
        assert result["ok"] is True
        assert result["already_pinned"] is True
        store.set_user.assert_not_awaited()

    async def test_mode_switch_accept_no_store(self) -> None:
        entry = base.get_entry("mode_switch", "direct")
        req = _make_request_with_app_state()  # no settings_store
        result = await entry.accept(
            {"kind": "mode_switch", "target_id": "direct"}, req,
        )
        assert result["ok"] is False
        assert result["error"] == "no_settings_store"

    async def test_model_swap_accept_writes_primary_chat_model(self) -> None:
        from augmentum.config import settings
        prev = getattr(settings, "heavyweight_model", "")
        object.__setattr__(settings, "heavyweight_model", "qwen3-30b-thinking")
        try:
            entry = base.get_entry("model_swap", "heavyweight")
            store = MagicMock()
            store.get_user = AsyncMock(return_value=None)
            store.set_user = AsyncMock()
            req = _make_request_with_app_state(settings_store=store)

            result = await entry.accept(
                {"kind": "model_swap", "target_id": "heavyweight"}, req,
            )
            assert result["ok"] is True
            assert result["model"] == "qwen3-30b-thinking"
            store.set_user.assert_awaited_once_with(
                U1, "primary_chat_model", "qwen3-30b-thinking",
            )
        finally:
            object.__setattr__(settings, "heavyweight_model", prev)

    async def test_model_swap_accept_unconfigured_role(self) -> None:
        from augmentum.config import settings
        prev = getattr(settings, "heavyweight_model", "")
        object.__setattr__(settings, "heavyweight_model", "")
        try:
            entry = base.get_entry("model_swap", "heavyweight")
            store = MagicMock()
            req = _make_request_with_app_state(settings_store=store)
            result = await entry.accept(
                {"kind": "model_swap", "target_id": "heavyweight"}, req,
            )
            assert result["ok"] is False
            assert result["error"] == "unconfigured_role"
        finally:
            object.__setattr__(settings, "heavyweight_model", prev)

    async def test_setting_tweak_admin_required_for_admin_scope(self) -> None:
        entry = base.get_entry("setting_tweak", "ghost_text")
        store = MagicMock()
        store.get = AsyncMock(return_value=None)
        store.set = AsyncMock()
        req = _make_request_with_app_state(settings_store=store)
        # Override the user to non-admin
        req.scope = {"user": SimpleNamespace(id=U1, is_admin=False)}

        result = await entry.accept(
            {"kind": "setting_tweak", "target_id": "ghost_text"}, req,
        )
        assert result["ok"] is False
        assert result["error"] == "admin_required"
        store.set.assert_not_awaited()

    async def test_setting_tweak_admin_writes_app_setting(self) -> None:
        from augmentum.config import settings
        prev = getattr(settings, "ghost_text_enabled", False)
        object.__setattr__(settings, "ghost_text_enabled", False)
        try:
            entry = base.get_entry("setting_tweak", "ghost_text")
            store = MagicMock()
            store.get = AsyncMock(return_value=None)
            store.set = AsyncMock()
            req = _make_request_with_app_state(settings_store=store)

            result = await entry.accept(
                {"kind": "setting_tweak", "target_id": "ghost_text"}, req,
            )
            assert result["ok"] is True
            store.set.assert_awaited_once_with("ghost_text_enabled", "true")
            # Live mirror updated for current process.
            assert settings.ghost_text_enabled is True
        finally:
            object.__setattr__(settings, "ghost_text_enabled", prev)

    async def test_workspace_profile_accept_requires_container_manager(self) -> None:
        entry = base.get_entry("workspace_profile", "browser")
        req = _make_request_with_app_state()  # no container_manager
        result = await entry.accept(
            {"kind": "workspace_profile", "target_id": "browser"}, req,
        )
        assert result["ok"] is False
        assert result["error"] == "no_container_manager"

    async def test_workspace_profile_accept_creates_workspace(self) -> None:
        entry = base.get_entry("workspace_profile", "browser")

        info = SimpleNamespace(id="ws-abc", name="browser-12345")
        mgr = MagicMock()
        mgr.create_workspace = AsyncMock(return_value=info)
        req = _make_request_with_app_state(container_manager=mgr)

        result = await entry.accept(
            {"kind": "workspace_profile", "target_id": "browser"}, req,
        )
        assert result["ok"] is True
        assert result["workspace_id"] == "ws-abc"
        assert result["profile_id"] == "browser"
        mgr.create_workspace.assert_awaited_once()
        call_kwargs = mgr.create_workspace.await_args.kwargs
        assert call_kwargs["tooling_profile"] == "browser"
        assert call_kwargs["user_id"] == U1

    async def test_memory_save_accept_writes_to_store(self) -> None:
        entry = base.get_entry("memory_save", "preference")
        store = MagicMock()
        store.store = AsyncMock(return_value="mem-xyz")
        req = _make_request_with_app_state(memory_store=store)

        payload = {
            "kind": "memory_save",
            "target_id": "preference",
            "extra": {"content": "User prefers dark mode and concise prose"},
        }
        result = await entry.accept(payload, req)
        assert result["ok"] is True
        assert result["memory_id"] == "mem-xyz"
        assert result["memory_type"] == "preference"
        store.store.assert_awaited_once()
        call_kwargs = store.store.await_args.kwargs
        assert call_kwargs["user_id"] == U1
        assert call_kwargs["content"] == "User prefers dark mode and concise prose"
        # EXPLICIT source ensures PII scrub is skipped + dedup gates relaxed.
        from augmentum.memory.models import SourceType
        assert call_kwargs["source_type"] == SourceType.EXPLICIT

    async def test_memory_save_refuses_missing_content(self) -> None:
        entry = base.get_entry("memory_save", "fact")
        store = MagicMock()
        store.store = AsyncMock()
        req = _make_request_with_app_state(memory_store=store)

        result = await entry.accept(
            {"kind": "memory_save", "target_id": "fact", "extra": {}},
            req,
        )
        assert result["ok"] is False
        assert result["error"] == "missing_content"
        store.store.assert_not_awaited()

    async def test_memory_save_refuses_oversized_content(self) -> None:
        entry = base.get_entry("memory_save", "fact")
        store = MagicMock()
        store.store = AsyncMock()
        req = _make_request_with_app_state(memory_store=store)

        result = await entry.accept(
            {
                "kind": "memory_save", "target_id": "fact",
                "extra": {"content": "x" * 5000},
            },
            req,
        )
        assert result["ok"] is False
        assert result["error"] == "content_too_long"
        store.store.assert_not_awaited()

    async def test_memory_save_no_store(self) -> None:
        entry = base.get_entry("memory_save", "fact")
        req = _make_request_with_app_state()  # no memory_store
        result = await entry.accept(
            {
                "kind": "memory_save", "target_id": "fact",
                "extra": {"content": "x"},
            },
            req,
        )
        assert result["ok"] is False
        assert result["error"] == "no_memory_store"
