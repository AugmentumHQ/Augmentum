"""Tests for FORMAT-aware, capability-matched dual-source illustration.

The Tutorial builder used to free-fire ``image_generation`` with whatever the
user's default image model was — so an anime checkpoint (Lumina) drew an
"alien" how-to-change-a-tire guide. The Illustrate step now runs a
deterministic loop that, per section, gathers BOTH real-photo search results
and a *capability-matched* generated image, leads with the FORMAT-appropriate
source, and NEVER generates a real-world depiction with a stylised model.

Covers:
- ``ImageGenerationTool.select_model_for`` — the capability gate
- ``_parse_document_sections`` — shared parser keeps section indices aligned
- ``_detect_plan_format`` — reads the Plan's FORMAT label
- ``_execute_illustrate_step`` — dual-source loop, both topic shapes
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock

import pytest

# ---------------------------------------------------------------------------
# select_model_for — capability gate
# ---------------------------------------------------------------------------


class _ModelInfo:
    def __init__(self, name):
        self.name = name


class _FakePersistence:
    def __init__(self, names):
        self._models = [_ModelInfo(n) for n in names]

    async def list_models(self):
        return self._models


class _FakeAppState:
    def __init__(self, installed, active_model):
        self.image_persistence = _FakePersistence(installed)
        self.image_active_settings = {"model": active_model}


def _make_gen_tool(installed, active_model, monkeypatch):
    from augmentum.config import settings
    from augmentum.tools.image_generation import ImageGenerationTool

    # Force the "default" to be the active UI model, not any env override.
    monkeypatch.setattr(settings, "agentic_image_model", "", raising=False)
    monkeypatch.setattr(settings, "image_default_model", "", raising=False)
    return ImageGenerationTool(
        queue=None, app_state=_FakeAppState(installed, active_model),
    )


@pytest.mark.asyncio
async def test_select_model_keeps_default_when_it_fits(monkeypatch):
    tool = _make_gen_tool(["flux.1-schnell"], "flux.1-schnell", monkeypatch)
    assert await tool.select_model_for(need_photoreal=True) == "flux.1-schnell"


@pytest.mark.asyncio
async def test_select_model_swaps_anime_default_for_photoreal(monkeypatch):
    # Lumina default, but a photoreal model is installed → pick that instead.
    tool = _make_gen_tool(["lumina2", "juggernaut-xl"], "lumina2", monkeypatch)
    assert await tool.select_model_for(need_photoreal=True) == "juggernaut-xl"


@pytest.mark.asyncio
async def test_select_model_returns_empty_when_nothing_capable(monkeypatch):
    # Anime-only install + a real-world photo need → no model, skip generation.
    tool = _make_gen_tool(["lumina2"], "lumina2", monkeypatch)
    assert await tool.select_model_for(need_photoreal=True) == ""


@pytest.mark.asyncio
async def test_select_model_no_need_returns_default(monkeypatch):
    tool = _make_gen_tool(["lumina2"], "lumina2", monkeypatch)
    assert await tool.select_model_for() == "lumina2"


@pytest.mark.asyncio
async def test_select_model_diagram_need(monkeypatch):
    tool = _make_gen_tool(["lumina2", "sdxl-base-1.0"], "lumina2", monkeypatch)
    assert await tool.select_model_for(need_diagram=True) == "sdxl-base-1.0"


# ---------------------------------------------------------------------------
# _parse_document_sections + _detect_plan_format
# ---------------------------------------------------------------------------


def test_parse_document_sections_section_markers():
    from augmentum.modes.agentic.handler import _parse_document_sections

    draft = (
        "Here's the guide:\n\n"
        "## SECTION: Introduction\nThis is a reasonably long intro paragraph "
        "describing what the reader will accomplish in this guide overall.\n\n"
        "## SECTION: Step 1\nLoosen the lug nuts before jacking the car.\n\n"
        "## SECTION: Step 2\nRaise the vehicle with the jack."
    )
    out = _parse_document_sections(draft, fallback_title="T")
    assert [s["heading"] for s in out] == ["Introduction", "Step 1", "Step 2"]


def test_parse_document_sections_fallback_single():
    from augmentum.modes.agentic.handler import _parse_document_sections

    out = _parse_document_sections("no markers here at all", fallback_title="FB")
    assert len(out) == 1
    assert out[0]["heading"] == "FB"


class _FmtTask:
    def __init__(self, step_outputs):
        self.step_outputs = step_outputs


class _FmtWmem:
    def __init__(self, plan_output):
        self._plan = plan_output

    @property
    def all_step_names(self):
        return ["Plan", "Draft Tutorial"]

    def get_step_output(self, name):
        return self._plan if name == "Plan" else ""


def test_detect_plan_format_from_wmem():
    from augmentum.modes.agentic.handler import _detect_plan_format

    wmem = _FmtWmem("Target audience: beginner\nFORMAT: procedure\nPrereqs: jack")
    assert _detect_plan_format(_FmtTask({}), wmem) == "procedure"


def test_detect_plan_format_from_step_outputs():
    from augmentum.modes.agentic.handler import _detect_plan_format

    task = _FmtTask({0: "FORMAT: 'code'\nTarget: advanced"})
    assert _detect_plan_format(task, _FmtWmem("")) == "code"


def test_detect_plan_format_absent():
    from augmentum.modes.agentic.handler import _detect_plan_format

    assert _detect_plan_format(_FmtTask({0: "no label"}), _FmtWmem("")) == ""


# ---------------------------------------------------------------------------
# _execute_illustrate_step — dual-source loop
# ---------------------------------------------------------------------------


class _Result:
    def __init__(self, success, metadata=None):
        self.success = success
        self.metadata = metadata or {}
        self.error = ""


class _FakeSearchTool:
    """Returns two photo candidates per query."""

    async def execute(self, **kwargs):
        return _Result(True, {"images": [
            {"embed_url": "http://photo/1", "thumb_url": "http://photo/1t",
             "source": "wikimedia", "title": "real photo 1"},
            {"embed_url": "http://photo/2", "thumb_url": "http://photo/2t",
             "source": "flickr", "title": "real photo 2"},
        ]})


class _FakeGenTool:
    def __init__(self, model_to_return):
        self._model = model_to_return
        self.calls = 0

    async def select_model_for(self, *, need_photoreal=False, need_diagram=False):
        return self._model

    async def execute(self, **kwargs):
        self.calls += 1
        return _Result(True, {"url": "/api/image/gen1"})


class _FakeRegistry:
    def __init__(self, tools):
        self._tools = tools

    def resolve(self, name):
        return self._tools.get(name)


class _LoopTask:
    def __init__(self):
        self.id = "task1"
        self.title = "Changing a Tire"
        self.original_query = "how to change a tire"
        self.step_outputs = {}
        self.image_candidates = {}
        self.slide_image_picks = {}


class _LoopWmem:
    def __init__(self, draft, plan):
        self._draft = draft
        self._plan = plan
        self._chain_results = {}

    @property
    def all_step_names(self):
        return ["Plan", "Draft Tutorial"]

    def get_step_output(self, name):
        if name == "Plan":
            return self._plan
        return self._draft if "Draft" in name else ""


_DRAFT = (
    "## SECTION: Introduction\n"
    "This guide walks you through safely swapping a flat tire on the roadside "
    "using the tools in your trunk so you can get moving again quickly.\n\n"
    "## SECTION: Loosen the lug nuts\n"
    "Before lifting the car, break each lug nut loose by a quarter turn so the "
    "wheel does not spin while it is in the air.\n\n"
    "## SECTION: Raise and swap\n"
    "Position the jack under the frame rail, raise the car, remove the nuts, "
    "swap the wheel, and lower it back down."
)


def _make_handler(registry):
    from augmentum.modes.agentic.handler import AgenticHandler

    handler = AgenticHandler(
        backend=AsyncMock(), session_id="ses", task_store=None,
        tool_registry=registry,
    )
    handler._user_id = "u1"
    handler._inject_tool_context = lambda tool, args: None
    return handler


def _patch_crafter(monkeypatch):
    import augmentum.tools.artifact_pipeline as ap

    async def fake_craft(units, caller):
        return [
            {"index": i + 1, "query": f"q{i + 1}", "description": f"d{i + 1}",
             "prefer_charts": False}
            for i in range(len(units))
        ]

    monkeypatch.setattr(ap, "craft_initial_slide_queries", fake_craft)
    monkeypatch.setattr(ap, "build_agentic_pipeline_caller",
                        lambda *a, **k: AsyncMock())


@pytest.mark.asyncio
async def test_illustrate_procedure_anime_only_skips_generation(monkeypatch):
    """The reported bug: anime-only install + a tire how-to.

    Generation must be skipped (no photoreal model) and every primary pick
    must be a real photo — never a stylised render.
    """
    _patch_crafter(monkeypatch)
    gen = _FakeGenTool(model_to_return="")  # no capable model
    registry = _FakeRegistry({"image_search": _FakeSearchTool(),
                              "image_generation": gen})
    handler = _make_handler(registry)
    task = _LoopTask()
    wmem = _LoopWmem(_DRAFT, "FORMAT: procedure")
    step = types.SimpleNamespace(
        role="illustrate", tool_names=["image_search", "image_generation"],
    )

    out = await handler._execute_illustrate_step(step, "m", None, task, wmem)

    assert gen.calls == 0  # never generated with an unsuitable model
    assert task.image_candidates  # photos were found
    for pool in task.image_candidates.values():
        assert all(c["kind"] == "photo" for c in pool)
    # Every primary is a real photo.
    for idx, pick in task.slide_image_picks.items():
        primary = next(c for c in task.image_candidates[idx]
                       if c["candidate_id"] == pick["primary"])
        assert primary["kind"] == "photo"
    assert "photo-primary" in out


@pytest.mark.asyncio
async def test_illustrate_code_topic_leads_with_generated_diagram(monkeypatch):
    """A code/technical topic with a diagram-capable model leads with the
    generated diagram, with photos available as alternates."""
    _patch_crafter(monkeypatch)
    gen = _FakeGenTool(model_to_return="sdxl-base-1.0")
    registry = _FakeRegistry({"image_search": _FakeSearchTool(),
                              "image_generation": gen})
    handler = _make_handler(registry)
    task = _LoopTask()
    wmem = _LoopWmem(_DRAFT, "FORMAT: code")
    step = types.SimpleNamespace(
        role="illustrate", tool_names=["image_search", "image_generation"],
    )

    out = await handler._execute_illustrate_step(step, "m", None, task, wmem)

    assert gen.calls > 0  # diagrams were generated
    # Primary for each illustrated section is the generated diagram.
    for idx, pick in task.slide_image_picks.items():
        primary = next(c for c in task.image_candidates[idx]
                       if c["candidate_id"] == pick["primary"])
        assert primary["kind"] == "generated"
        assert primary["embed_url"] == "/api/image/gen1"
    # Pools mix both sources (real photos kept as alternates).
    kinds = {c["kind"] for pool in task.image_candidates.values() for c in pool}
    assert kinds == {"generated", "photo"}
    assert "diagram-primary" in out


@pytest.mark.asyncio
async def test_illustrate_presentation_path_is_photo_only(monkeypatch):
    """A deck lists only image_search → generation never triggers (no regression)."""
    _patch_crafter(monkeypatch)
    gen = _FakeGenTool(model_to_return="flux.1-schnell")
    registry = _FakeRegistry({"image_search": _FakeSearchTool(),
                              "image_generation": gen})
    handler = _make_handler(registry)
    task = _LoopTask()
    wmem = _LoopWmem(_DRAFT, "")
    step = types.SimpleNamespace(role="illustrate", tool_names=["image_search"])

    await handler._execute_illustrate_step(step, "m", None, task, wmem)

    assert gen.calls == 0
    for pool in task.image_candidates.values():
        assert all(c["kind"] == "photo" for c in pool)
