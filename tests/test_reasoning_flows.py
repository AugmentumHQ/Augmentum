"""Tests for reasoning flow store, templates, and API routes."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.reasoning.models import FlowCreateRequest, FlowStep, FlowUpdateRequest, ReasoningFlow
from augmentum.reasoning.models import VALID_ROLES
from augmentum.reasoning.templates import (
    BUILTIN_TEMPLATES,
    get_template,
    list_templates,
    research_flow,
)


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------


class TestTemplates:
    def test_builtin_count(self):
        assert len(BUILTIN_TEMPLATES) == 17

    def test_all_templates_produce_valid_flows(self):
        for name, factory in BUILTIN_TEMPLATES.items():
            flow = factory()
            assert flow.name, f"{name} has no name"
            assert len(flow.steps) >= 1, f"{name} has no steps"
            assert flow.is_builtin is True

    def test_research_has_six_steps(self):
        flow = research_flow()
        assert len(flow.steps) == 6
        names = [s.name for s in flow.steps]
        assert names == ["Classify", "Search", "Cross-Reference", "Synthesize", "Fact-Check", "Respond"]

    def test_auto_routing_is_default(self):
        from augmentum.reasoning.templates import auto_routing_flow
        flow = auto_routing_flow()
        assert flow.is_default is True
        assert flow.name == "Auto Routing"

    def test_research_is_not_default(self):
        flow = research_flow()
        assert flow.is_default is False

    def test_research_has_classify_role(self):
        flow = research_flow()
        classify_steps = [s for s in flow.steps if s.role == "classify"]
        assert len(classify_steps) >= 1
        assert classify_steps[0].name == "Classify"

    def test_research_has_respond_role(self):
        flow = research_flow()
        respond_steps = [s for s in flow.steps if s.role == "respond"]
        assert len(respond_steps) == 1
        assert respond_steps[0].stream_to_user is True

    def test_research_complexity_gating(self):
        flow = research_flow()
        # Synthesize step should always run (empty gate)
        synthesize = next(s for s in flow.steps if s.name == "Synthesize")
        assert synthesize.complexity_gate == []  # always runs

    def test_quick_answer_single_step(self):
        flow = get_template("quick_answer")
        assert flow is not None
        assert len(flow.steps) == 1
        assert flow.steps[0].role == "respond"

    def test_research_has_search_step(self):
        flow = get_template("research")
        assert flow is not None
        search_steps = [s for s in flow.steps if s.role == "search"]
        assert len(search_steps) >= 1

    def test_code_review_has_python_exec(self):
        flow = get_template("code_review")
        assert flow is not None
        fix_step = next(s for s in flow.steps if s.name == "Suggest Fixes")
        assert "python_exec" in fix_step.tool_names

    def test_math_has_calculator_tools(self):
        flow = get_template("math")
        assert flow is not None
        solve_step = next(s for s in flow.steps if s.name == "Solve")
        assert "calculator" in solve_step.tool_names
        assert "math_verify" in solve_step.tool_names

    def test_math_solve_step_forces_tool_use(self):
        # Regression guard: the Solve step's prompt says "Use tools for ALL
        # calculations" — pinning the tool_choice override keeps small models
        # honest, so a future template edit that drops it should fail loudly.
        flow = get_template("math")
        solve_step = next(s for s in flow.steps if s.name == "Solve")
        assert solve_step.tool_choice == "required"

    def test_debate_has_both_sides(self):
        flow = get_template("debate")
        assert flow is not None
        names = [s.name for s in flow.steps]
        assert "Argue For" in names
        assert "Argue Against" in names

    def test_creative_disables_search(self):
        flow = get_template("creative")
        assert flow is not None
        assert flow.auto_search is False

    def test_quick_answer_single_step_via_template(self):
        flow = get_template("quick_answer")
        assert flow is not None
        assert len(flow.steps) == 1
        assert flow.steps[0].stream_to_user is True

    def test_unknown_template_returns_none(self):
        assert get_template("nonexistent") is None

    def test_list_templates(self):
        templates = list_templates()
        assert len(templates) == 17
        names = {t["name"] for t in templates}
        assert "research" in names
        assert "quick_answer" in names
        for t in templates:
            assert "display_name" in t
            assert "description" in t
            assert "step_count" in t
            assert t["step_count"] >= 1

    def test_all_steps_have_ids(self):
        for name, factory in BUILTIN_TEMPLATES.items():
            flow = factory()
            for step in flow.steps:
                assert step.id, f"{name}/{step.name} has no id"

    def test_steps_have_sequential_sort_order(self):
        for name, factory in BUILTIN_TEMPLATES.items():
            flow = factory()
            orders = [s.sort_order for s in flow.steps]
            assert orders == list(range(len(flow.steps))), f"{name} sort order is wrong"

    def test_step_roles_are_valid(self):
        for name, factory in BUILTIN_TEMPLATES.items():
            flow = factory()
            for step in flow.steps:
                assert step.role in VALID_ROLES, f"{name}/{step.name} has invalid role '{step.role}'"

    def test_last_step_streams_to_user(self):
        for name, factory in BUILTIN_TEMPLATES.items():
            flow = factory()
            # At least one step should stream to user
            streaming = [s for s in flow.steps if s.stream_to_user]
            assert streaming, f"{name} has no streaming step"


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestModels:
    def test_flow_step_defaults(self):
        step = FlowStep()
        assert step.role == "analyze"
        assert step.tool_categories == []
        assert step.complexity_gate == []
        assert step.stream_to_user is False
        assert step.output_cap == 800
        assert step.enabled is True

    def test_reasoning_flow_defaults(self):
        flow = ReasoningFlow()
        assert flow.auto_search is True
        assert flow.max_tool_calls_per_step == 3
        assert flow.is_builtin is False
        assert flow.steps == []

    def test_flow_create_request_requires_name(self):
        with pytest.raises(Exception):
            FlowCreateRequest(name="")

    def test_flow_update_request_all_optional(self):
        req = FlowUpdateRequest()
        d = req.model_dump(exclude_none=True)
        assert d == {}

    def test_flow_serialization_roundtrip(self):
        flow = research_flow()
        data = flow.model_dump()
        restored = ReasoningFlow(**data)
        assert restored.name == flow.name
        assert len(restored.steps) == len(flow.steps)
        assert restored.steps[0].name == flow.steps[0].name

    def test_flow_step_tool_choice_default_empty(self):
        # Default = empty string = no override (executor passes None).
        # Backwards-compatible with every existing FlowStep.
        assert FlowStep().tool_choice == ""

    def test_flow_step_tool_choice_accepts_sentinels_and_names(self):
        for v in ("", "auto", "required", "none", "calculator", "web_search"):
            assert FlowStep(tool_choice=v).tool_choice == v

    def test_flow_step_tool_choice_length_capped(self):
        with pytest.raises(Exception):
            FlowStep(tool_choice="x" * 201)


# ---------------------------------------------------------------------------
# Tool-choice translation helper
# ---------------------------------------------------------------------------


class _MockTool:
    """Minimal stand-in for tools.Tool — only ``.name`` is read."""
    def __init__(self, name: str):
        self.name = name


class TestToolChoiceTranslation:
    """Cover every branch of ``_translate_step_tool_choice``."""

    @staticmethod
    def _translate(value: str, tool_names: list[str]):
        from augmentum.reasoning.executor import _translate_step_tool_choice
        step = FlowStep(tool_choice=value)
        tools = [_MockTool(n) for n in tool_names]
        return _translate_step_tool_choice(step, tools)

    def test_empty_yields_none(self):
        assert self._translate("", ["calculator"]) is None

    def test_passthrough_sentinels(self):
        assert self._translate("auto", ["calculator"]) == "auto"
        assert self._translate("required", ["calculator"]) == "required"
        assert self._translate("none", ["calculator"]) == "none"

    def test_no_tools_yields_none_even_with_sentinel(self):
        # tool_choice with no tools would 400 on most providers — drop quietly.
        assert self._translate("required", []) is None
        assert self._translate("auto", []) is None

    def test_specific_tool_name_becomes_pin_dict(self):
        result = self._translate("calculator", ["calculator", "web_search"])
        assert result == {
            "type": "function",
            "function": {"name": "calculator"},
        }

    def test_unknown_tool_name_yields_none(self):
        # Typo / dropped tool — drop the override rather than 400 the call.
        result = self._translate("calcualtor", ["calculator"])
        assert result is None


# ---------------------------------------------------------------------------
# Store tests (using in-memory SQLite)
# ---------------------------------------------------------------------------


UID = "user_test"


@pytest.fixture
async def store():
    import aiosqlite
    from pathlib import Path

    db = await aiosqlite.connect(":memory:")
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=OFF")

    # Create schema_version table (normally created by migration 001)
    await db.execute(
        "CREATE TABLE IF NOT EXISTS schema_version "
        "(version INTEGER PRIMARY KEY, description TEXT, applied_at TEXT DEFAULT (datetime('now')))"
    )

    # Apply migration
    migration_path = Path(__file__).parent.parent / "augmentum" / "state" / "migrations" / "011_reasoning_flows.sql"
    sql = migration_path.read_text()
    await db.executescript(sql)
    # Later schema additions — add columns directly so we don't drag in
    # later migration dependencies.
    for table in ("reasoning_flows", "reasoning_flow_steps"):
        try:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT")
        except Exception:
            pass
    try:
        await db.execute(
            "ALTER TABLE reasoning_flow_steps ADD COLUMN model_override TEXT DEFAULT ''"
        )
    except Exception:
        pass
    try:
        await db.execute(
            "ALTER TABLE reasoning_flow_steps ADD COLUMN tool_choice TEXT DEFAULT ''"
        )
    except Exception:
        pass
    await db.commit()

    from augmentum.reasoning.store import FlowStore
    s = FlowStore(db)
    yield s
    await db.close()


@pytest.mark.asyncio
class TestFlowStore:
    async def test_seed_builtins(self, store):
        count = await store.seed_builtins()
        assert count == 13

    async def test_seed_builtins_idempotent(self, store):
        await store.seed_builtins()
        count2 = await store.seed_builtins()
        # seed_builtins updates existing builtins too, so count is always 13
        assert count2 == 13

    async def test_list_flows(self, store):
        await store.seed_builtins()
        flows = await store.list_flows()
        assert len(flows) == 13
        # Each entry is (flow, step_count)
        for flow, step_count in flows:
            assert flow.name
            assert step_count >= 1

    async def test_get_flow_with_steps(self, store):
        await store.seed_builtins()
        flows = await store.list_flows()
        flow_id = flows[0][0].id
        flow = await store.get_flow(flow_id)
        assert flow is not None
        assert len(flow.steps) >= 1

    async def test_get_default_flow(self, store):
        await store.seed_builtins()
        default = await store.get_default_flow()
        assert default is not None
        assert default.name == "Auto Routing"
        assert default.is_default is True

    async def test_get_nonexistent(self, store):
        flow = await store.get_flow("nonexistent")
        assert flow is None

    async def test_create_flow(self, store):
        flow = ReasoningFlow(
            name="Test Flow",
            description="A test",
            steps=[
                FlowStep(name="Step 1", system_prompt="Do something", role="respond", stream_to_user=True),
            ],
        )
        created = await store.create_flow(flow, user_id=UID)
        assert created.id
        assert created.name == "Test Flow"

        fetched = await store.get_flow(created.id)
        assert fetched is not None
        assert len(fetched.steps) == 1
        assert fetched.steps[0].name == "Step 1"

    async def test_update_flow(self, store):
        flow = ReasoningFlow(name="Original", steps=[
            FlowStep(name="S1", role="respond", stream_to_user=True),
        ])
        created = await store.create_flow(flow, user_id=UID)

        updated = await store.update_flow(created.id, {"name": "Updated"})
        assert updated is not None
        assert updated.name == "Updated"
        assert updated.version == 2

    async def test_update_builtin_returns_none(self, store):
        await store.seed_builtins()
        default = await store.get_default_flow()
        result = await store.update_flow(default.id, {"name": "Hacked"})
        assert result is None

    async def test_update_steps(self, store):
        flow = ReasoningFlow(name="Test", steps=[
            FlowStep(name="Old Step", role="respond", stream_to_user=True),
        ])
        created = await store.create_flow(flow, user_id=UID)

        new_steps = [
            FlowStep(name="New Step 1", role="analyze").model_dump(),
            FlowStep(name="New Step 2", role="respond", stream_to_user=True).model_dump(),
        ]
        updated = await store.update_flow(created.id, {"steps": new_steps})
        assert updated is not None
        assert len(updated.steps) == 2
        assert updated.steps[0].name == "New Step 1"
        assert updated.steps[1].name == "New Step 2"

    async def test_delete_flow(self, store):
        flow = ReasoningFlow(name="Deletable", steps=[
            FlowStep(name="S", role="respond", stream_to_user=True),
        ])
        created = await store.create_flow(flow, user_id=UID)
        assert await store.delete_flow(created.id) is True
        assert await store.get_flow(created.id) is None

    async def test_tool_choice_round_trips(self, store):
        flow = ReasoningFlow(name="Forced", steps=[
            FlowStep(
                name="Compute", role="analyze",
                tool_names=["calculator"], tool_choice="required",
            ),
            FlowStep(
                name="Answer", role="respond", stream_to_user=True,
            ),
        ])
        created = await store.create_flow(flow, user_id=UID)
        fetched = await store.get_flow(created.id)
        assert fetched is not None
        compute = next(s for s in fetched.steps if s.name == "Compute")
        answer = next(s for s in fetched.steps if s.name == "Answer")
        assert compute.tool_choice == "required"
        # Default ("") still round-trips as empty, not NULL.
        assert answer.tool_choice == ""

    async def test_delete_builtin_fails(self, store):
        await store.seed_builtins()
        default = await store.get_default_flow()
        assert await store.delete_flow(default.id) is False

    async def test_set_default(self, store):
        await store.seed_builtins()
        # Create a custom flow and make it default
        flow = ReasoningFlow(name="Custom Default", steps=[
            FlowStep(name="S", role="respond", stream_to_user=True),
        ])
        created = await store.create_flow(flow, user_id=UID)
        assert await store.set_default(created.id) is True

        new_default = await store.get_default_flow()
        assert new_default.id == created.id

        # Old default should no longer be default
        flows = await store.list_flows()
        auto_routing = next((f for f, _ in flows if f.name == "Auto Routing"), None)
        assert auto_routing is not None
        assert auto_routing.is_default is False

    async def test_clone_flow(self, store):
        await store.seed_builtins()
        default = await store.get_default_flow()

        clone = await store.clone_flow(default.id, "My Standard", user_id=UID)
        assert clone is not None
        assert clone.name == "My Standard"
        assert clone.is_builtin is False
        assert clone.id != default.id
        assert len(clone.steps) == len(default.steps)

        # Steps should have different IDs
        original_ids = {s.id for s in default.steps}
        clone_ids = {s.id for s in clone.steps}
        assert original_ids.isdisjoint(clone_ids)

    async def test_clone_nonexistent(self, store):
        assert await store.clone_flow("nonexistent") is None

    async def test_export_flow(self, store):
        await store.seed_builtins()
        default = await store.get_default_flow()
        data = await store.export_flow(default.id)
        assert data is not None
        assert data["name"] == "Auto Routing"
        assert len(data["steps"]) >= 1

    async def test_import_flow(self, store):
        data = {
            "name": "Imported Flow",
            "description": "From JSON",
            "steps": [
                {"name": "S1", "system_prompt": "test", "role": "respond", "stream_to_user": True},
            ],
        }
        flow = await store.import_flow(data, user_id=UID)
        assert flow.id
        assert flow.name == "Imported Flow"
        assert flow.is_builtin is False

        fetched = await store.get_flow(flow.id)
        assert fetched is not None
        assert len(fetched.steps) == 1

    async def test_steps_preserve_tool_config(self, store):
        flow = ReasoningFlow(name="Tools Test", steps=[
            FlowStep(
                name="S1",
                tool_categories=["search", "execute"],
                tool_names=["python_exec", "calculator"],
                complexity_gate=["moderate", "complex"],
                role="analyze",
            ),
            FlowStep(name="S2", role="respond", stream_to_user=True),
        ])
        created = await store.create_flow(flow, user_id=UID)
        fetched = await store.get_flow(created.id)
        s1 = fetched.steps[0]
        assert s1.tool_categories == ["search", "execute"]
        assert s1.tool_names == ["python_exec", "calculator"]
        assert s1.complexity_gate == ["moderate", "complex"]

    async def test_steps_preserve_prompts(self, store):
        prompt = "You are a helpful assistant.\n\nDo the thing."
        template = "## Query\n{query}\n\n{previous_output}"
        flow = ReasoningFlow(name="Prompt Test", steps=[
            FlowStep(
                name="S1",
                system_prompt=prompt,
                user_template=template,
                role="respond",
                stream_to_user=True,
            ),
        ])
        created = await store.create_flow(flow, user_id=UID)
        fetched = await store.get_flow(created.id)
        assert fetched.steps[0].system_prompt == prompt
        assert fetched.steps[0].user_template == template


# ---------------------------------------------------------------------------
# Route tests (using TestClient)
# ---------------------------------------------------------------------------


@pytest.fixture
async def app_with_flows():
    """Create a minimal FastAPI app with flow store for route testing."""
    import aiosqlite
    from pathlib import Path
    from fastapi import FastAPI
    from augmentum.proxy.reasoning_routes import router
    from augmentum.reasoning.store import FlowStore

    db = await aiosqlite.connect(":memory:")
    # Foreign keys off so we don't need to materialize the users table just
    # to satisfy the ALTER's REFERENCES users(id) clause.
    await db.execute("PRAGMA foreign_keys=OFF")
    await db.execute(
        "CREATE TABLE IF NOT EXISTS schema_version "
        "(version INTEGER PRIMARY KEY, description TEXT, applied_at TEXT DEFAULT (datetime('now')))"
    )
    migration_path = Path(__file__).parent.parent / "augmentum" / "state" / "migrations" / "011_reasoning_flows.sql"
    await db.executescript(migration_path.read_text())
    # Later schema additions — add columns directly so we don't drag in
    # later migration dependencies.
    for table in ("reasoning_flows", "reasoning_flow_steps"):
        try:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT")
        except Exception:
            pass
    try:
        await db.execute(
            "ALTER TABLE reasoning_flow_steps ADD COLUMN model_override TEXT DEFAULT ''"
        )
    except Exception:
        pass
    try:
        await db.execute(
            "ALTER TABLE reasoning_flow_steps ADD COLUMN tool_choice TEXT DEFAULT ''"
        )
    except Exception:
        pass
    await db.commit()

    app = FastAPI()

    # Inject a fake auth scope so routes can extract user_id from request.scope.
    @app.middleware("http")
    async def _fake_auth(request, call_next):
        request.scope["user"] = type("U", (), {"id": UID})()
        return await call_next(request)

    app.include_router(router)

    store = FlowStore(db)
    await store.seed_builtins()
    app.state.flow_store = store

    yield app
    await db.close()


@pytest.mark.asyncio
class TestRoutes:
    async def test_list_flows(self, app_with_flows):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(transport=ASGITransport(app=app_with_flows), base_url="http://test") as client:
            resp = await client.get("/api/reasoning/flows")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 13
            assert all("step_count" in f for f in data)

    async def test_get_flow(self, app_with_flows):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(transport=ASGITransport(app=app_with_flows), base_url="http://test") as client:
            # List to get an ID
            flows = (await client.get("/api/reasoning/flows")).json()
            flow_id = flows[0]["id"]

            resp = await client.get(f"/api/reasoning/flows/{flow_id}")
            assert resp.status_code == 200
            flow = resp.json()
            assert "steps" in flow
            assert len(flow["steps"]) >= 1

    async def test_get_nonexistent(self, app_with_flows):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(transport=ASGITransport(app=app_with_flows), base_url="http://test") as client:
            resp = await client.get("/api/reasoning/flows/nonexistent")
            assert resp.status_code == 404

    async def test_create_flow(self, app_with_flows):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(transport=ASGITransport(app=app_with_flows), base_url="http://test") as client:
            resp = await client.post("/api/reasoning/flows", json={
                "name": "My Custom",
                "steps": [
                    {"name": "Think", "role": "analyze", "system_prompt": "Think carefully."},
                    {"name": "Answer", "role": "respond", "stream_to_user": True, "system_prompt": "Answer."},
                ],
            })
            assert resp.status_code == 201
            flow = resp.json()
            assert flow["name"] == "My Custom"
            assert len(flow["steps"]) == 2

    async def test_create_from_template(self, app_with_flows):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(transport=ASGITransport(app=app_with_flows), base_url="http://test") as client:
            resp = await client.post("/api/reasoning/flows", json={
                "name": "My Research",
                "template": "research",
            })
            assert resp.status_code == 201
            flow = resp.json()
            assert flow["name"] == "My Research"
            assert flow["is_builtin"] is False
            assert len(flow["steps"]) >= 5

    async def test_create_from_unknown_template(self, app_with_flows):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(transport=ASGITransport(app=app_with_flows), base_url="http://test") as client:
            resp = await client.post("/api/reasoning/flows", json={
                "name": "Bad",
                "template": "nonexistent",
            })
            assert resp.status_code == 400

    async def test_clone_builtin(self, app_with_flows):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(transport=ASGITransport(app=app_with_flows), base_url="http://test") as client:
            flows = (await client.get("/api/reasoning/flows")).json()
            research = next(f for f in flows if f["name"] == "Research")

            resp = await client.post(
                f"/api/reasoning/flows/{research['id']}/clone",
                params={"name": "My Research"},
            )
            assert resp.status_code == 201
            clone = resp.json()
            assert clone["name"] == "My Research"
            assert clone["is_builtin"] is False

    async def test_delete_custom_flow(self, app_with_flows):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(transport=ASGITransport(app=app_with_flows), base_url="http://test") as client:
            # Create then delete
            resp = await client.post("/api/reasoning/flows", json={
                "name": "Temp",
                "steps": [{"name": "S", "role": "respond", "stream_to_user": True}],
            })
            flow_id = resp.json()["id"]

            resp = await client.delete(f"/api/reasoning/flows/{flow_id}")
            assert resp.status_code == 200

    async def test_delete_builtin_fails(self, app_with_flows):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(transport=ASGITransport(app=app_with_flows), base_url="http://test") as client:
            flows = (await client.get("/api/reasoning/flows")).json()
            builtin_id = flows[0]["id"]

            resp = await client.delete(f"/api/reasoning/flows/{builtin_id}")
            assert resp.status_code == 404

    async def test_set_default(self, app_with_flows):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(transport=ASGITransport(app=app_with_flows), base_url="http://test") as client:
            # Create custom flow
            resp = await client.post("/api/reasoning/flows", json={
                "name": "New Default",
                "steps": [{"name": "S", "role": "respond", "stream_to_user": True}],
            })
            flow_id = resp.json()["id"]

            resp = await client.put(f"/api/reasoning/flows/{flow_id}/default")
            assert resp.status_code == 200

    async def test_export_import(self, app_with_flows):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(transport=ASGITransport(app=app_with_flows), base_url="http://test") as client:
            flows = (await client.get("/api/reasoning/flows")).json()
            research = next(f for f in flows if f["name"] == "Research")

            # Export
            resp = await client.get(f"/api/reasoning/flows/{research['id']}/export")
            assert resp.status_code == 200
            exported = resp.json()

            # Import
            exported["name"] = "Imported Research"
            resp = await client.post("/api/reasoning/flows/import", json=exported)
            assert resp.status_code == 201
            imported = resp.json()
            assert imported["name"] == "Imported Research"
            assert imported["id"] != research["id"]

    async def test_get_templates(self, app_with_flows):
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(transport=ASGITransport(app=app_with_flows), base_url="http://test") as client:
            resp = await client.get("/api/reasoning/templates")
            assert resp.status_code == 200
            templates = resp.json()
            assert len(templates) == 17
