"""Parser-robustness tests: the strict-JSON plan must survive the prose and
trailing junk that thinking-capable models (and tool-injecting chat routes)
wrap around it.
"""

from __future__ import annotations

import pytest

from augmentum.game_agent.prompt import (
    PlanParseError,
    _extract_first_json_object,
    parse_plan_output,
)
from augmentum.game_agent.schema import SurfaceCapsPayload

_CAPS = SurfaceCapsPayload(
    semantic_inputs=["confirm", "cancel", "nav_up"],
    log_schema="pokemon_rs.v1",
    observation_modalities=["log", "frame"],
)

_PLAN = (
    '{"observations":["title screen"],'
    '"state_update":"on title",'
    '"actions":[{"semantic":"confirm","duration_ms":100}],'
    '"confidence":0.8,"next_check_in_ms":500}'
)


def test_extract_first_object_ignores_braces_in_strings():
    text = 'prefix {"a":"has } brace","b":1} trailing'
    assert _extract_first_json_object(text) == '{"a":"has } brace","b":1}'


def test_parses_clean_json():
    plan = parse_plan_output(_PLAN, _CAPS)
    assert plan.actions[0].semantic == "confirm"


def test_parses_with_trailing_markdown_junk():
    # The exact failure we saw: valid JSON, then a trailing image link.
    raw = f"```json\n{_PLAN}\n```\n\n![Generated Image](/api/image/abc123)"
    plan = parse_plan_output(raw, _CAPS)
    assert plan.actions[0].semantic == "confirm"


def test_parses_with_leading_prose():
    raw = f"Here is my plan after thinking it through:\n{_PLAN}"
    plan = parse_plan_output(raw, _CAPS)
    assert plan.next_check_in_ms == 500


def test_still_rejects_genuine_garbage():
    with pytest.raises(PlanParseError):
        parse_plan_output("no json here at all", _CAPS)
