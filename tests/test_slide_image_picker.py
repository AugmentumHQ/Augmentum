"""Phase 1 substrate tests for the slide image-search picker.

Covers the pieces the picker depends on without exercising the full
agentic flow:
- Prompt-output parsers (initial crafter, expansion crafter)
- ``craft_initial_slide_queries`` retry-on-parse-failure path
- ``craft_expansion_queries`` deficit-based K request
- ``convert_description_to_gen_prompt`` chart-pivot fallback
- ``_apply_pipeline_image_picks`` end-to-end projection (handler-side
  helper is verified by smoke-testing the pipeline-side mirror — same
  logic, same indexing contract)
- ``_pick_artifact_draft_from_outputs`` heuristic
- PPTX additional_images normalisation contract
"""

from __future__ import annotations

import pytest

from augmentum.tools.artifact_pipeline import (
    _apply_pipeline_image_picks,
    _parse_expansion_payload,
    _parse_slide_query_payload,
    _strip_json_fence,
    convert_description_to_gen_prompt,
    craft_expansion_queries,
    craft_initial_slide_queries,
)


# ---------------------------------------------------------------------------
# Strip-fence helper
# ---------------------------------------------------------------------------


def test_strip_json_fence_removes_markdown():
    assert _strip_json_fence("```json\n{\"a\": 1}\n```") == "{\"a\": 1}"
    assert _strip_json_fence("```\n[1, 2]\n```") == "[1, 2]"
    assert _strip_json_fence('{"x": 1}') == '{"x": 1}'
    assert _strip_json_fence("") == ""


# ---------------------------------------------------------------------------
# _parse_slide_query_payload
# ---------------------------------------------------------------------------


def test_parse_slide_query_normal_envelope():
    raw = (
        '{"slides": ['
        '{"index": 1, "description": "d1", "query": "q1", "prefer_charts": false},'
        '{"index": 2, "description": "d2", "query": "", "prefer_charts": true}'
        ']}'
    )
    out = _parse_slide_query_payload(raw)
    assert len(out) == 2
    assert out[0] == {"index": 1, "description": "d1", "query": "q1", "prefer_charts": False}
    assert out[1]["query"] == ""  # text-only slide preserved


def test_parse_slide_query_bare_list_accepted():
    """Some models drop the outer {"slides": [...]} envelope."""
    raw = '[{"index": 1, "query": "q"}]'
    out = _parse_slide_query_payload(raw)
    assert len(out) == 1
    assert out[0]["index"] == 1


def test_parse_slide_query_fenced():
    raw = '```json\n{"slides": [{"index": 1, "query": "x"}]}\n```'
    out = _parse_slide_query_payload(raw)
    assert len(out) == 1


def test_parse_slide_query_string_index_coerced():
    raw = '{"slides": [{"index": "3", "query": "x"}]}'
    out = _parse_slide_query_payload(raw)
    assert out == [{"index": 3, "description": "", "query": "x", "prefer_charts": False}]


def test_parse_slide_query_empty_and_garbage():
    assert _parse_slide_query_payload("") == []
    assert _parse_slide_query_payload("not json at all") == []
    assert _parse_slide_query_payload("null") == []


# ---------------------------------------------------------------------------
# _parse_expansion_payload — returns None on hard failure so retry triggers
# ---------------------------------------------------------------------------


def test_parse_expansion_normal():
    raw = (
        '{"queries": ['
        '{"description": "d1", "query": "q1", "prefer_charts": true},'
        '{"description": "d2", "query": "q2", "prefer_charts": false}'
        ']}'
    )
    out = _parse_expansion_payload(raw, expected_k=2)
    assert out is not None
    assert len(out) == 2
    assert out[0]["query"] == "q1"
    assert out[1]["prefer_charts"] is False


def test_parse_expansion_caps_at_k():
    raw = '{"queries": [{"query": "q1"}, {"query": "q2"}, {"query": "q3"}]}'
    out = _parse_expansion_payload(raw, expected_k=2)
    assert out is not None
    assert len(out) == 2  # never more than k


def test_parse_expansion_drops_empty_queries():
    raw = '{"queries": [{"query": ""}, {"query": "real"}]}'
    out = _parse_expansion_payload(raw, expected_k=2)
    assert out == [{"description": "", "query": "real", "prefer_charts": False}]


def test_parse_expansion_garbage_returns_none():
    """None signals retry — empty list would falsely indicate "model said no queries"."""
    assert _parse_expansion_payload("garbage", expected_k=2) is None
    assert _parse_expansion_payload("", expected_k=2) is None


# ---------------------------------------------------------------------------
# craft_initial_slide_queries — retry on first-pass parse failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_craft_initial_retries_on_first_parse_failure():
    """First call returns garbage, second returns valid JSON. Result == retry output."""
    call_count = {"n": 0}

    async def caller(system, user):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "this is not json"
        return '{"slides": [{"index": 1, "query": "fixed"}]}'

    slides = [{"title": "Slide One", "body": "body"}]
    out = await craft_initial_slide_queries(slides, caller)
    assert call_count["n"] == 2
    assert len(out) == 1
    assert out[0]["query"] == "fixed"


@pytest.mark.asyncio
async def test_craft_initial_returns_empty_on_double_failure():
    """Both passes garbage → empty list. Caller decides whether to skip illustrate."""
    async def caller(system, user):
        return "still not json"

    slides = [{"title": "Slide", "body": "body"}]
    out = await craft_initial_slide_queries(slides, caller)
    assert out == []


@pytest.mark.asyncio
async def test_craft_initial_passes_all_slides_to_llm():
    """The full slide list (1-based indexed) reaches the prompt body."""
    captured = {}

    async def caller(system, user):
        captured["system"] = system
        captured["user"] = user
        return '{"slides": []}'

    slides = [
        {"title": "S1", "body": "B1"},
        {"title": "S2", "body": "B2"},
        {"title": "S3", "body": "B3"},
    ]
    await craft_initial_slide_queries(slides, caller)
    assert '"index": 1' in captured["user"]
    assert '"index": 2' in captured["user"]
    assert '"index": 3' in captured["user"]
    assert "S1" in captured["user"]
    assert "S3" in captured["user"]


# ---------------------------------------------------------------------------
# craft_expansion_queries — existing-queries-in-prompt + K capping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_craft_expansion_includes_existing_queries_in_prompt():
    """Diversity constraint: existing queries reach the LLM user message."""
    captured = {}

    async def caller(system, user):
        captured["user"] = user
        return '{"queries": [{"query": "new1"}, {"query": "new2"}]}'

    out = await craft_expansion_queries(
        slide_title="Solar panels",
        slide_body="cost decline",
        existing_queries=[{"query": "solar cost IRENA", "description": "chart"}],
        k=2,
        llm_caller=caller,
    )
    assert len(out) == 2
    assert "solar cost IRENA" in captured["user"]
    assert "Form 2 new queries" in captured["user"]


@pytest.mark.asyncio
async def test_craft_expansion_retries_then_gives_up():
    """Double parse failure → empty list, no crash."""
    async def caller(system, user):
        return "trash"

    out = await craft_expansion_queries(
        slide_title="t", slide_body="b",
        existing_queries=[], k=2, llm_caller=caller,
    )
    assert out == []


# ---------------------------------------------------------------------------
# convert_description_to_gen_prompt — chart-pivot fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_convert_description_returns_llm_output_when_available():
    async def caller(system, user):
        return "A photorealistic solar farm at sunset"

    out = await convert_description_to_gen_prompt(
        description="Chart of solar cost decline",
        slide_title="Solar panels",
        llm_caller=caller,
    )
    assert out == "A photorealistic solar farm at sunset"


@pytest.mark.asyncio
async def test_convert_description_strips_quotes_and_newlines():
    async def caller(system, user):
        return '"A solar farm.\nGolden hour."'

    out = await convert_description_to_gen_prompt(
        description="d", slide_title="t", llm_caller=caller,
    )
    assert out.startswith("A solar farm.")
    assert "\n" not in out


@pytest.mark.asyncio
async def test_convert_description_fallback_pivots_chart_to_scene():
    """When the LLM returns nothing, fallback replaces chart/graph/etc. with 'scene'.

    This is the failure-mode handling promised by the spec: numeric-chart
    descriptions don't get passed verbatim to a gen model that can't
    draw charts.
    """
    async def caller(system, user):
        return ""

    out = await convert_description_to_gen_prompt(
        description="Bar chart showing revenue growth quarter over quarter",
        slide_title="Q3 results",
        llm_caller=caller,
    )
    assert "chart" not in out.lower()
    assert "scene" in out.lower()


@pytest.mark.asyncio
async def test_convert_description_handles_llm_exception():
    async def caller(system, user):
        raise RuntimeError("llm down")

    out = await convert_description_to_gen_prompt(
        description="diagram of the OAuth flow",
        slide_title="Auth", llm_caller=caller,
    )
    assert out  # non-empty fallback string
    assert "diagram" not in out.lower()  # pivoted to "scene"


# ---------------------------------------------------------------------------
# _apply_pipeline_image_picks — projection across primary + additional
# ---------------------------------------------------------------------------


def test_apply_picks_projects_primary_and_additional():
    slides = [
        {"layout": "title", "title": "Welcome", "body": ""},
        {"layout": "content", "title": "Slide 2", "body": "B"},
        {"layout": "content", "title": "Slide 3", "body": "B"},
    ]
    candidates = {
        2: [
            {"candidate_id": "a1", "embed_url": "http://a/1"},
            {"candidate_id": "a2", "embed_url": "http://a/2"},
        ],
        3: [
            {"candidate_id": "b1", "embed_url": "http://b/1"},
        ],
    }
    picks = {
        2: {"primary": "a1", "additional": ["a2"]},
        3: {"primary": "b1", "additional": []},
    }
    out = _apply_pipeline_image_picks(slides, picks, candidates)
    assert out[0].get("image_url") is None  # slide without picks unchanged
    assert out[1]["image_url"] == "http://a/1"
    assert out[1]["additional_images"] == ["http://a/2"]
    assert out[2]["image_url"] == "http://b/1"
    assert "additional_images" not in out[2]


def test_apply_picks_accepts_string_keys_from_json_roundtrip():
    """SQLite-stored picks come back keyed by string. Helper must tolerate both."""
    slides = [{"title": "A"}, {"title": "B"}]
    candidates = {"2": [{"candidate_id": "x", "embed_url": "http://b"}]}
    picks = {"2": {"primary": "x", "additional": []}}
    out = _apply_pipeline_image_picks(slides, picks, candidates)
    assert out[1]["image_url"] == "http://b"


def test_apply_picks_caps_additional_at_three():
    slides = [{"title": "X"}]
    pool = [
        {"candidate_id": f"c{i}", "embed_url": f"http://x/{i}"} for i in range(5)
    ]
    candidates = {1: pool}
    picks = {1: {"primary": "c0", "additional": ["c1", "c2", "c3", "c4"]}}
    out = _apply_pipeline_image_picks(slides, picks, candidates)
    assert len(out[0]["additional_images"]) == 3  # hard cap


def test_apply_picks_skips_unknown_candidate_id():
    slides = [{"title": "X"}]
    candidates = {1: [{"candidate_id": "real", "embed_url": "http://real"}]}
    picks = {1: {"primary": "ghost", "additional": []}}
    out = _apply_pipeline_image_picks(slides, picks, candidates)
    assert "image_url" not in out[0]  # ghost id ignored


def test_apply_picks_empty_inputs_are_passthrough():
    slides = [{"title": "X"}]
    assert _apply_pipeline_image_picks(slides, {}, {}) == slides
    assert _apply_pipeline_image_picks(slides, {1: {"primary": "x"}}, {}) == slides


# ---------------------------------------------------------------------------
# _pick_artifact_draft_from_outputs — picker re-render reads here
# ---------------------------------------------------------------------------


class _FakeTask:
    def __init__(self, step_outputs):
        self.step_outputs = step_outputs


def test_pick_draft_from_outputs_prefers_longest_non_status():
    from augmentum.modes.agentic.handler import _pick_artifact_draft_from_outputs

    task = _FakeTask({
        0: "task plan...",  # too short
        1: "Solar costs dropped 90%. " * 30,  # research, accepted
        2: ("### Slide 1: Welcome\n" + "real draft content. " * 50),  # draft
        3: "Image generated successfully. Do NOT call image_generation again." * 5,  # status
    })
    out = _pick_artifact_draft_from_outputs(task)
    assert "Slide 1" in out
    assert "Image generated" not in out


def test_pick_draft_from_outputs_empty_when_no_qualifying_step():
    from augmentum.modes.agentic.handler import _pick_artifact_draft_from_outputs

    task = _FakeTask({0: "short", 1: "Image generated successfully. " * 50})
    assert _pick_artifact_draft_from_outputs(task) == ""


# ---------------------------------------------------------------------------
# normalize_slides preserves additional_images
# ---------------------------------------------------------------------------


def test_normalize_slides_preserves_additional_images():
    from augmentum.tools.artifact_normalize import normalize_slides

    raw = [{
        "layout": "content",
        "title": "S",
        "body": "B",
        "image_url": "/api/image/x",
        "additional_images": ["http://a", "http://b"],
    }]
    out = normalize_slides(raw)
    assert out[0]["additional_images"] == ["http://a", "http://b"]


def test_normalize_slides_caps_additional_at_three():
    from augmentum.tools.artifact_normalize import normalize_slides

    raw = [{"title": "X", "additional_images": [f"http://{i}" for i in range(6)]}]
    out = normalize_slides(raw)
    assert len(out[0]["additional_images"]) == 3


def test_normalize_slides_handles_missing_additional_images():
    from augmentum.tools.artifact_normalize import normalize_slides

    raw = [{"title": "X"}]
    out = normalize_slides(raw)
    assert out[0]["additional_images"] == []
