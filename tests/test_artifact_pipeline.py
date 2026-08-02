from __future__ import annotations

import json
import pytest

import os

from augmentum.tools.artifact_pipeline import (
    ArtifactRequest,
    ArtifactResult,
    ContentInventory,
    DataTable,
    ImageRef,
    PipelineContext,
    ResearchItem,
    build_agentic_pipeline_caller,
    build_backend_pipeline_caller,
    download_web_image,
    resolve_pipeline_tools,
)


def test_artifact_request_defaults():
    req = ArtifactRequest(format="pdf", topic="renewable energy")
    assert req.format == "pdf"
    assert req.topic == "renewable energy"
    assert req.title is None
    assert req.theme is None
    assert req.tool_params == {}


def test_pipeline_context_defaults():
    ctx = PipelineContext()
    assert ctx.message_history == []
    assert ctx.working_memory is None
    assert ctx.tool_results == []
    assert ctx.generated_images == []


def test_content_inventory_defaults():
    inv = ContentInventory(topic="test")
    assert inv.existing_research == []
    assert inv.existing_data == []
    assert inv.existing_images == []
    assert inv.existing_draft is None
    assert inv.coverage_gaps == []
    assert inv.needs_research is True
    assert inv.needs_images is False


def test_research_item():
    item = ResearchItem(content="fact", source_url="https://example.com", topic="energy")
    assert item.content == "fact"
    assert item.source_url == "https://example.com"


def test_data_table():
    dt = DataTable(headers=["Year", "Value"], rows=[[2020, 100]], source="https://example.com")
    assert len(dt.rows) == 1


def test_image_ref():
    img = ImageRef(url="/api/image/abc", description="solar panel", source="generated")
    assert img.url == "/api/image/abc"


def test_artifact_result():
    res = ArtifactResult(
        artifact_id="abc123",
        download_url="/api/artifacts/abc123/download",
        display_name="Report.pdf",
        metadata={"size_bytes": 1000},
        source_json={},
    )
    assert res.artifact_id == "abc123"


# ---------------------------------------------------------------------------
# Task 2: Context Scan Phase
# ---------------------------------------------------------------------------
from augmentum.tools.artifact_pipeline import scan_context


@pytest.mark.asyncio
async def test_scan_context_empty():
    """Cold start — no context at all."""
    req = ArtifactRequest(format="pdf", topic="renewable energy")
    ctx = PipelineContext()

    async def mock_llm(system: str, user: str) -> str:
        return json.dumps(["solar power", "wind energy", "policy"])

    inv = await scan_context(req, ctx, mock_llm)
    assert inv.topic == "renewable energy"
    assert inv.needs_research is True
    assert inv.existing_research == []
    assert len(inv.coverage_gaps) == 3


@pytest.mark.asyncio
async def test_scan_context_with_message_history():
    """Should extract research from tool result messages."""
    req = ArtifactRequest(format="pdf", topic="solar panels")
    ctx = PipelineContext(message_history=[
        {"role": "user", "content": "tell me about solar panels"},
        {"role": "tool", "content": "Solar panels convert sunlight to electricity. Source: https://energy.gov/solar"},
        {"role": "assistant", "content": "Solar panels are great."},
    ])

    async def mock_llm(system: str, user: str) -> str:
        return json.dumps(["manufacturing", "costs"])

    inv = await scan_context(req, ctx, mock_llm)
    assert len(inv.existing_research) >= 1
    assert inv.existing_research[0].source_url == "https://energy.gov/solar"
    assert len(inv.coverage_gaps) == 2


@pytest.mark.asyncio
async def test_scan_context_pptx_needs_images():
    """PPTX format should default to needs_images=True."""
    req = ArtifactRequest(format="pptx", topic="test")
    ctx = PipelineContext()

    async def mock_llm(system: str, user: str) -> str:
        return json.dumps([])

    inv = await scan_context(req, ctx, mock_llm)
    assert inv.needs_images is True


@pytest.mark.asyncio
async def test_scan_context_image_keywords():
    """Keywords like 'illustrated' should set needs_images."""
    req = ArtifactRequest(format="pdf", topic="an illustrated guide to birds")
    ctx = PipelineContext()

    async def mock_llm(system: str, user: str) -> str:
        return json.dumps([])

    inv = await scan_context(req, ctx, mock_llm)
    assert inv.needs_images is True


@pytest.mark.asyncio
async def test_scan_context_with_generated_images():
    """Should pick up generated images from context."""
    req = ArtifactRequest(format="pdf", topic="test")
    ctx = PipelineContext(generated_images=["/api/image/abc123"])

    async def mock_llm(system: str, user: str) -> str:
        return json.dumps([])

    inv = await scan_context(req, ctx, mock_llm)
    assert len(inv.existing_images) == 1
    assert inv.existing_images[0].url == "/api/image/abc123"


@pytest.mark.asyncio
async def test_scan_context_rich_context_no_research_needed():
    """With lots of existing research, needs_research should be False."""
    req = ArtifactRequest(format="pdf", topic="test")
    ctx = PipelineContext(message_history=[
        {"role": "tool", "content": f"Research finding {i}. Source: https://example.com/{i}"}
        for i in range(10)
    ])

    async def mock_llm(system: str, user: str) -> str:
        return json.dumps([])  # no gaps

    inv = await scan_context(req, ctx, mock_llm)
    assert inv.needs_research is False


@pytest.mark.asyncio
async def test_scan_context_working_memory():
    """Should extract from agentic working memory."""

    class FakeWMem:
        @property
        def all_step_names(self):
            return ["Research", "Draft"]

        def get_step_output(self, name):
            if name == "Research":
                return "Found data about X. Source: https://example.com"
            if name == "Draft":
                return "## SECTION: Introduction\nThis is the draft content about X."
            return ""

        @property
        def _artifacts(self):
            return []

        @property
        def _steps(self):
            return [("Research", "chain", ""), ("Draft", "generative", "")]

    req = ArtifactRequest(format="pdf", topic="test")
    ctx = PipelineContext(working_memory=FakeWMem())

    async def mock_llm(system: str, user: str) -> str:
        return json.dumps([])

    inv = await scan_context(req, ctx, mock_llm)
    assert len(inv.existing_research) >= 1
    assert inv.existing_draft is not None
    assert "draft content" in inv.existing_draft.lower()


# ---------------------------------------------------------------------------
# Task 3: Research Phase
# ---------------------------------------------------------------------------
from unittest.mock import AsyncMock, MagicMock
from augmentum.tools.artifact_pipeline import run_research, run_research_data


@pytest.mark.asyncio
async def test_run_research_fills_gaps():
    """Research should search for each coverage gap."""
    inventory = ContentInventory(topic="solar energy", coverage_gaps=["costs", "efficiency"])

    search_tool = AsyncMock()
    search_tool.execute = AsyncMock(return_value=MagicMock(
        success=True, output="Solar panel costs have dropped 90%. Source: https://irena.org/costs"
    ))
    fetch_tool = AsyncMock()
    fetch_tool.execute = AsyncMock(return_value=MagicMock(
        success=True, output="Detailed article about solar costs from IRENA..."
    ))

    await run_research(inventory, search_tool, fetch_tool)

    assert search_tool.execute.call_count == 2  # one per gap
    assert len(inventory.existing_research) >= 2


@pytest.mark.asyncio
async def test_run_research_skips_when_not_needed():
    """Should do nothing if needs_research is False."""
    inventory = ContentInventory(topic="test", needs_research=False)
    search_tool = AsyncMock()
    fetch_tool = AsyncMock()

    await run_research(inventory, search_tool, fetch_tool)

    search_tool.execute.assert_not_called()


@pytest.mark.asyncio
async def test_run_research_handles_search_failure():
    """Should gracefully handle search failures."""
    inventory = ContentInventory(topic="test", coverage_gaps=["subtopic"])

    search_tool = AsyncMock()
    search_tool.execute = AsyncMock(return_value=MagicMock(success=False, output="", error="timeout"))
    fetch_tool = AsyncMock()

    await run_research(inventory, search_tool, fetch_tool)
    # Should not crash, just proceed with what we have


@pytest.mark.asyncio
async def test_run_research_data_for_xlsx():
    """Research data should add data-oriented search terms."""
    inventory = ContentInventory(topic="cloud provider pricing")

    search_tool = AsyncMock()
    search_tool.execute = AsyncMock(return_value=MagicMock(
        success=True,
        output="AWS: $0.023/GB, Azure: $0.018/GB, GCP: $0.020/GB. Source: https://cloudcosts.com"
    ))
    fetch_tool = AsyncMock()
    fetch_tool.execute = AsyncMock(return_value=MagicMock(
        success=True,
        output="Provider,Storage,Compute\nAWS,0.023,0.10\nAzure,0.018,0.08"
    ))

    await run_research_data(inventory, search_tool, fetch_tool)

    assert len(inventory.existing_research) >= 1


# ---------------------------------------------------------------------------
# Task 4: Draft & Illustrate Phases
# ---------------------------------------------------------------------------
from augmentum.tools.artifact_pipeline import (
    draft_sections,
    draft_slides,
    draft_sheet_structure,
    draft_chart_dataset,
    run_illustrate,
)


@pytest.mark.asyncio
async def test_draft_sections():
    """Should produce markdown with SECTION markers."""
    inventory = ContentInventory(
        topic="solar energy",
        existing_research=[
            ResearchItem(content="Solar costs dropped 90% since 2010.", source_url="https://irena.org"),
        ],
    )

    async def mock_llm(system: str, user: str) -> str:
        return (
            "## SECTION: Introduction\nSolar energy is growing.\n\n"
            "## SECTION: Cost Trends\nCosts dropped 90% since 2010. [Source: https://irena.org]\n\n"
            "## SECTION: Conclusion\nSolar is the future."
        )

    draft = await draft_sections(inventory, mock_llm)
    assert "## SECTION:" in draft
    assert "Cost Trends" in draft


@pytest.mark.asyncio
async def test_draft_slides():
    """Should produce markdown with Slide N markers."""
    inventory = ContentInventory(topic="quarterly results")

    async def mock_llm(system: str, user: str) -> str:
        return (
            "### Slide 1: Title Slide\nQ4 2025 Results\n**Notes:** Welcome everyone\n\n"
            "### Slide 2: Revenue\n- Revenue up 15%\n- New markets entered\n**Notes:** Key growth areas"
        )

    draft = await draft_slides(inventory, mock_llm)
    assert "### Slide" in draft


@pytest.mark.asyncio
async def test_draft_sheet_structure():
    """Should return valid JSON sheet structure."""
    inventory = ContentInventory(
        topic="cloud pricing",
        existing_research=[
            ResearchItem(content="AWS $0.023/GB, Azure $0.018/GB", source_url=""),
        ],
    )

    async def mock_llm(system: str, user: str) -> str:
        return json.dumps({
            "sheets": [{
                "name": "Pricing",
                "headers": ["Provider", "Storage ($/GB)"],
                "rows": [["AWS", 0.023], ["Azure", 0.018]],
                "column_formats": {"Storage ($/GB)": "$#,##0.000"},
                "summary_row": "none",
            }]
        })

    sheets = await draft_sheet_structure(inventory, mock_llm)
    assert len(sheets) == 1
    assert sheets[0]["name"] == "Pricing"


@pytest.mark.asyncio
async def test_draft_chart_dataset():
    """Should return valid chart data structure."""
    inventory = ContentInventory(
        topic="monthly sales",
        existing_research=[
            ResearchItem(content="Jan: 100, Feb: 150, Mar: 200", source_url=""),
        ],
    )

    async def mock_llm(system: str, user: str) -> str:
        return json.dumps({
            "chart_type": "bar",
            "labels": ["Jan", "Feb", "Mar"],
            "datasets": [{"name": "Sales", "values": [100, 150, 200]}],
            "x_label": "Month",
            "y_label": "Units",
        })

    chart = await draft_chart_dataset(inventory, mock_llm)
    assert chart["chart_type"] == "bar"
    assert len(chart["labels"]) == 3


@pytest.mark.asyncio
async def test_draft_sections_uses_existing_draft():
    """If a draft already exists, should return it without LLM call."""
    inventory = ContentInventory(
        topic="test",
        existing_draft="## SECTION: Intro\nExisting draft content.",
    )

    async def mock_llm(system: str, user: str) -> str:
        pytest.fail("LLM should not be called when draft exists")

    draft = await draft_sections(inventory, mock_llm)
    assert "Existing draft content" in draft


@pytest.mark.asyncio
async def test_run_illustrate():
    """Should search for images and build mapping."""
    sections = [
        {"heading": "Solar Panels", "body": "How solar panels work"},
        {"heading": "Wind Energy", "body": "Wind turbines generate power"},
    ]
    existing_images = []

    image_search = AsyncMock()
    image_search.execute = AsyncMock(return_value=MagicMock(
        success=True,
        output="Found image: solar panel installation",
        metadata={"url": "https://example.com/solar.jpg"},
    ))

    async def mock_llm(system: str, user: str) -> str:
        return json.dumps({"Solar Panels": "https://example.com/solar.jpg", "Wind Energy": None})

    mapping = await run_illustrate(sections, existing_images, image_search, mock_llm)
    assert isinstance(mapping, dict)


# ---------------------------------------------------------------------------
# Task 5: Assemble Phases
# ---------------------------------------------------------------------------
from augmentum.tools.artifact_pipeline import (
    assemble_sections,
    assemble_slides,
    assemble_sheets,
    assemble_chart,
    parse_draft_sections,
    parse_draft_slides,
)


def test_parse_draft_sections():
    """Should parse ## SECTION: markers into section dicts."""
    draft = (
        "## SECTION: Introduction\nThis is the intro.\n\n"
        "## SECTION: Body\nThis is the body with data.\n"
    )
    sections = parse_draft_sections(draft)
    assert len(sections) == 2
    assert sections[0]["heading"] == "Introduction"
    assert "intro" in sections[0]["body"].lower()


def test_parse_draft_sections_fallback():
    """Should fall back to ## heading parsing."""
    draft = "## Introduction\nIntro text.\n\n## Analysis\nAnalysis text."
    sections = parse_draft_sections(draft)
    assert len(sections) == 2


def test_parse_draft_sections_single():
    """No markers — single section."""
    draft = "Just a paragraph of text."
    sections = parse_draft_sections(draft, fallback_title="Report")
    assert len(sections) == 1
    assert sections[0]["heading"] == "Report"


def test_parse_draft_slides():
    """Should parse ### Slide N: markers into slide dicts."""
    draft = (
        "### Slide 1: Welcome\nTitle slide content\n**Notes:** Hello\n\n"
        "### Slide 2: Data\n- Point 1\n- Point 2\n**Notes:** Discuss data"
    )
    slides = parse_draft_slides(draft)
    assert len(slides) == 2
    assert slides[0]["title"] == "Welcome"
    assert "Hello" in slides[0]["notes"]


def test_assemble_sections_wires_images():
    """Should wire image URLs into sections."""
    sections = [
        {"heading": "Solar", "body": "Solar text", "level": 1},
        {"heading": "Wind", "body": "Wind text", "level": 1},
    ]
    image_map = {"Solar": "https://example.com/solar.jpg"}
    research = [ResearchItem(content="fact", source_url="https://example.com")]

    result = assemble_sections(sections, image_map, research, title="Report", tool_params={})
    assert result["sections"][0].get("image_url") == "https://example.com/solar.jpg"
    assert result["sections"][1].get("image_url") is None
    # Should have references section
    assert any("reference" in s["heading"].lower() for s in result["sections"])


def test_assemble_slides_wires_images():
    """Should wire image URLs into slides."""
    slides = [
        {"layout": "content", "title": "Solar", "body": "Text", "notes": ""},
    ]
    image_map = {"Solar": "https://example.com/solar.jpg"}
    research = [ResearchItem(content="fact", source_url="https://example.com")]

    result = assemble_slides(slides, image_map, research, title="Deck", tool_params={})
    assert result["slides"][0].get("image_url") == "https://example.com/solar.jpg"


def test_assemble_sheets():
    """Should pass through valid sheet structure."""
    sheets = [{"name": "Data", "headers": ["A", "B"], "rows": [[1, 2]]}]
    research = [ResearchItem(content="data", source_url="https://example.com")]

    result = assemble_sheets(sheets, research, title="Sheet", tool_params={})
    assert len(result["sheets"]) >= 1


def test_assemble_chart():
    """Should pass through valid chart data."""
    chart_data = {
        "chart_type": "bar",
        "labels": ["A", "B"],
        "datasets": [{"name": "S1", "values": [1, 2]}],
        "x_label": "X",
        "y_label": "Y",
    }

    result = assemble_chart(chart_data, title="Chart", tool_params={})
    assert result["chart_type"] == "bar"
    assert result["title"] == "Chart"


# ---------------------------------------------------------------------------
# Task 6: Pipeline Orchestrator
# ---------------------------------------------------------------------------
from augmentum.tools.artifact_pipeline import execute_artifact_pipeline


@pytest.mark.asyncio
async def test_pipeline_pdf_cold_start():
    """Full pipeline for PDF with no prior context — exercises all phases."""
    req = ArtifactRequest(format="pdf", topic="renewable energy trends")
    ctx = PipelineContext()

    llm_call_count = 0

    async def mock_llm(system: str, user: str) -> str:
        nonlocal llm_call_count
        llm_call_count += 1
        if "gaps" in system.lower() or "missing" in user.lower():
            return json.dumps(["solar costs", "wind capacity"])
        if "section" in system.lower():
            return (
                "## SECTION: Introduction\nRenewable energy is growing.\n\n"
                "## SECTION: Solar Costs\nSolar costs dropped 90%.\n\n"
                "## SECTION: Conclusion\nThe future is bright."
            )
        return "{}"

    # Mock tools
    mock_search = AsyncMock()
    mock_search.execute = AsyncMock(return_value=MagicMock(
        success=True, output="Solar costs dropped. Source: https://irena.org/solar"
    ))
    mock_fetch = AsyncMock()
    mock_fetch.execute = AsyncMock(return_value=MagicMock(
        success=True, output="Detailed solar cost data..."
    ))

    mock_doc_tool = AsyncMock()
    mock_doc_tool.execute = AsyncMock(return_value=MagicMock(
        success=True,
        output="Document created: Report.pdf\nDownload: /api/artifacts/test123/download",
        metadata={"id": "test123", "download_url": "/api/artifacts/test123/download",
                   "display_name": "Report.pdf", "size_bytes": 5000},
    ))

    result = await execute_artifact_pipeline(
        req, ctx, mock_llm,
        _search_tool=mock_search,
        _fetch_tool=mock_fetch,
        _render_tools={"create_document": mock_doc_tool},
    )

    assert result.artifact_id == "test123"
    assert "/api/artifacts/" in result.download_url
    assert mock_search.execute.called
    assert mock_doc_tool.execute.called


@pytest.mark.asyncio
async def test_pipeline_xlsx_with_research():
    """XLSX pipeline should research data and structure sheets."""
    req = ArtifactRequest(format="xlsx", topic="cloud pricing comparison")
    ctx = PipelineContext()

    async def mock_llm(system: str, user: str) -> str:
        if "gaps" in system.lower() or "missing" in user.lower():
            return json.dumps([])
        if "spreadsheet" in system.lower() or "sheets" in user.lower():
            return json.dumps({"sheets": [{"name": "Pricing", "headers": ["Provider", "Cost"], "rows": [["AWS", 0.023]], "column_formats": {}, "summary_row": "none"}]})
        return "[]"

    mock_search = AsyncMock()
    mock_search.execute = AsyncMock(return_value=MagicMock(
        success=True, output="AWS pricing: $0.023/GB. Source: https://aws.amazon.com"
    ))
    mock_fetch = AsyncMock()
    mock_fetch.execute = AsyncMock(return_value=MagicMock(success=True, output="AWS data"))

    mock_xlsx_tool = AsyncMock()
    mock_xlsx_tool.execute = AsyncMock(return_value=MagicMock(
        success=True,
        output="Spreadsheet created\nDownload: /api/artifacts/xlsx123/download",
        metadata={"id": "xlsx123", "download_url": "/api/artifacts/xlsx123/download",
                   "display_name": "Pricing.xlsx", "size_bytes": 3000},
    ))

    result = await execute_artifact_pipeline(
        req, ctx, mock_llm,
        _search_tool=mock_search,
        _fetch_tool=mock_fetch,
        _render_tools={"create_spreadsheet": mock_xlsx_tool},
    )

    assert result.artifact_id == "xlsx123"
    assert mock_xlsx_tool.execute.called


@pytest.mark.asyncio
async def test_pipeline_skips_research_with_rich_context():
    """With existing research from analytical, should skip research phase."""
    req = ArtifactRequest(format="pdf", topic="test")

    class FakeToolCallRecord:
        def __init__(self, tool_name, output):
            self.tool_name = tool_name
            self.output = output
            self.success = True

    ctx = PipelineContext(
        message_history=[
            {"role": "tool", "content": f"Research finding {i}. Source: https://example.com/{i}"}
            for i in range(5)
        ],
        tool_results=[
            FakeToolCallRecord("web_search", "More research data"),
            FakeToolCallRecord("web_fetch", "Fetched content about the topic"),
        ],
    )

    async def mock_llm(system: str, user: str) -> str:
        if "gaps" in system.lower() or "missing" in user.lower():
            return json.dumps([])  # no gaps
        if "section" in system.lower():
            return "## SECTION: Summary\nAll the research findings compiled."
        return "{}"

    mock_search = AsyncMock()
    mock_fetch = AsyncMock()

    mock_doc_tool = AsyncMock()
    mock_doc_tool.execute = AsyncMock(return_value=MagicMock(
        success=True,
        output="Download: /api/artifacts/rich123/download",
        metadata={"id": "rich123", "download_url": "/api/artifacts/rich123/download",
                   "display_name": "Report.pdf", "size_bytes": 5000},
    ))

    result = await execute_artifact_pipeline(
        req, ctx, mock_llm,
        _search_tool=mock_search,
        _fetch_tool=mock_fetch,
        _render_tools={"create_document": mock_doc_tool},
    )

    # Research should NOT have been called — context was sufficient
    mock_search.execute.assert_not_called()
    assert result.artifact_id == "rich123"


# ---------------------------------------------------------------------------
# Backend / Agentic pipeline callers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_backend_pipeline_caller():
    """build_backend_pipeline_caller wraps backend.chat into LLMCaller."""
    from unittest.mock import AsyncMock, MagicMock

    mock_backend = MagicMock()
    resp = MagicMock()
    resp.content = "hello from backend"
    mock_backend.chat = AsyncMock(return_value=resp)

    caller = build_backend_pipeline_caller(mock_backend, model="test-model")
    result = await caller("system prompt", "user prompt")

    assert result == "hello from backend"
    mock_backend.chat.assert_awaited_once()
    call_args = mock_backend.chat.call_args[0][0]
    assert call_args.model == "test-model"
    assert len(call_args.messages) == 2
    assert call_args.messages[0].role == "system"
    assert call_args.messages[1].role == "user"


@pytest.mark.asyncio
async def test_build_backend_pipeline_caller_none_content():
    """build_backend_pipeline_caller returns empty string when content is None."""
    from unittest.mock import AsyncMock, MagicMock

    mock_backend = MagicMock()
    resp = MagicMock()
    resp.content = None
    mock_backend.chat = AsyncMock(return_value=resp)

    caller = build_backend_pipeline_caller(mock_backend)
    result = await caller("sys", "usr")
    assert result == ""


@pytest.mark.asyncio
async def test_build_agentic_pipeline_caller():
    """build_agentic_pipeline_caller wraps _call_llm into LLMCaller."""
    from unittest.mock import AsyncMock, MagicMock

    mock_call_llm = AsyncMock(return_value="agentic response")
    mock_request = MagicMock()

    caller = build_agentic_pipeline_caller(mock_call_llm, model="agent-model", request=mock_request)
    result = await caller("system prompt", "user prompt")

    assert result == "agentic response"
    mock_call_llm.assert_awaited_once_with(
        model="agent-model",
        system_prompt="system prompt",
        user_message="user prompt",
        request=mock_request,
    )


# ---------------------------------------------------------------------------
# Task 10: Production Tool Resolution
# ---------------------------------------------------------------------------


def test_resolve_pipeline_tools_no_registry():
    """Should return all None when no tool_registry on app."""
    app = MagicMock()
    del app.state.tool_registry  # ensure getattr returns None
    app.state = MagicMock(spec=[])  # no tool_registry attribute
    result = resolve_pipeline_tools(app)
    assert result == {"search_tool": None, "fetch_tool": None, "image_search_tool": None}


def test_resolve_pipeline_tools_with_registry():
    """Should call registry.resolve for each tool name."""
    app = MagicMock()
    mock_registry = MagicMock()
    mock_registry.resolve.side_effect = lambda name: f"resolved_{name}"
    app.state.tool_registry = mock_registry

    result = resolve_pipeline_tools(app)
    assert result["search_tool"] == "resolved_web_search"
    assert result["fetch_tool"] == "resolved_web_fetch"
    assert result["image_search_tool"] == "resolved_image_search"


# ---------------------------------------------------------------------------
# Task 11: Web Image Download
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_web_image_converts_to_local():
    """Should write test bytes to a temp file and return its path."""
    result = await download_web_image("https://example.com/photo.jpg", _test_bytes=b"fake-image")
    assert result is not None
    assert os.path.exists(result)
    os.unlink(result)


@pytest.mark.asyncio
async def test_download_web_image_passes_internal_urls():
    """Internal /api/ URLs should be returned unchanged."""
    result = await download_web_image("/api/image/abc123")
    assert result == "/api/image/abc123"
