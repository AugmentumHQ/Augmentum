"""Smoke tests for the AgentWorld world-model driver (Slot B)."""

from __future__ import annotations

import pytest

from augmentum.models.base import InternalChatResponse, Message, Usage
from augmentum.world_model import (
    RESPONSE_MARKER,
    WORLD_DOMAINS,
    AgentWorldDriver,
    WorldModelUnavailable,
    extract_observation,
    serialize_episode,
)
from augmentum.world_model.driver import load_domain_prompt
from augmentum.world_model.judging import (
    SCORE_DIMENSIONS,
    build_judge_user_prompt,
    load_judge_prompt,
    parse_judge_output,
)

_ON_CONTRACT = (
    "<predicted_observation>**Environment Observation:**\n"
    "total 0\nuser@host:~$ </predicted_observation>"
)


class FakeBackend:
    def __init__(self, content=_ON_CONTRACT, thinking="cd then ls"):
        self.content = content
        self.thinking = thinking
        self.requests = []

    async def chat(self, request):
        self.requests.append(request)
        return InternalChatResponse(
            message=Message(role="assistant", content=self.content, thinking=self.thinking),
            model="Qwen-AgentWorld-35B-A3B-Q4_K_M",
            finish_reason="stop",
            usage=Usage(prompt_tokens=1200, completion_tokens=64, total_tokens=1264),
        )


class FakeSlot:
    def __init__(self, model_id="Qwen-AgentWorld-35B-A3B-Q4_K_M", loaded=True, backend=None):
        self._model_id = model_id
        self._loaded = loaded
        self.backend = backend or FakeBackend()

    def status(self):
        return {"loaded": self._loaded, "model_id": self._model_id}


def test_all_domain_prompts_present_and_nonempty():
    for domain in WORLD_DOMAINS:
        text = load_domain_prompt(domain)
        assert len(text) > 500, f"{domain} prompt suspiciously short"


def test_all_judge_prompts_present_and_nonempty():
    for domain in WORLD_DOMAINS:
        text = load_judge_prompt(domain)
        assert len(text) > 500, f"{domain} judge prompt suspiciously short"


def test_unknown_domain_rejected():
    with pytest.raises(ValueError, match="unknown world domain"):
        load_domain_prompt("minecraft")


def test_status_unavailable_when_slot_disabled():
    driver = AgentWorldDriver(None)
    st = driver.status()
    assert st["available"] is False
    assert "disabled" in st["reason"]


def test_status_rejects_non_world_model():
    driver = AgentWorldDriver(FakeSlot(model_id="Qwen3.5-4B-Q4_K_M"))
    st = driver.status()
    assert st["available"] is False
    assert st["is_world_model"] is False
    assert "does not look like a world model" in st["reason"]


def test_status_honors_pin_when_not_resident(monkeypatch):
    # After a restart Slot B keeps its pin and lazy-loads on first use —
    # a pinned world model must report available even before residency.
    from augmentum.config import settings

    monkeypatch.setattr(
        settings, "engine_secondary_model", "Qwen.Qwen-AgentWorld-35B-A3B.Q4_K_M",
        raising=False,
    )
    driver = AgentWorldDriver(FakeSlot(loaded=False, model_id=""))
    st = driver.status()
    assert st["available"] is True
    assert st["resident"] is False
    assert st["model_id"] == "Qwen.Qwen-AgentWorld-35B-A3B.Q4_K_M"


def test_status_pin_of_non_world_model_still_refused(monkeypatch):
    from augmentum.config import settings

    monkeypatch.setattr(
        settings, "engine_secondary_model", "Qwen3.5-4B-Q4_K_M", raising=False,
    )
    driver = AgentWorldDriver(FakeSlot(loaded=False, model_id=""))
    st = driver.status()
    assert st["available"] is False
    assert "does not look like a world model" in st["reason"]


# -- extraction (AgentWorldBench parse_model_output contract) -------------


def test_extract_on_contract_output():
    obs, tag_found = extract_observation(_ON_CONTRACT)
    assert tag_found is True
    assert obs == "total 0\nuser@host:~$"


def test_extract_last_tag_block_wins():
    raw = (
        "<predicted_observation>fake early block</predicted_observation>\n"
        "<predicted_observation>real block</predicted_observation>"
    )
    obs, tag_found = extract_observation(raw)
    assert tag_found is True
    assert obs == "real block"


def test_extract_tolerates_missing_closer():
    obs, tag_found = extract_observation("<predicted_observation>no closer here")
    assert tag_found is True
    assert obs == "no closer here"


def test_extract_fallback_without_tags_strips_marker():
    obs, tag_found = extract_observation(f"{RESPONSE_MARKER}\nplain output")
    assert tag_found is False
    assert obs == "plain output"


def test_extract_unclosed_think_does_not_eat_the_answer():
    raw = (
        "<think>reasoning that never closes... "
        "<predicted_observation>survived</predicted_observation>"
    )
    obs, tag_found = extract_observation(raw)
    assert tag_found is True
    assert obs == "survived"


def test_extract_fake_tag_inside_think_is_ignored():
    raw = (
        "<think>maybe <predicted_observation>hallucinated</predicted_observation> "
        "hmm</think><predicted_observation>actual</predicted_observation>"
    )
    obs, _ = extract_observation(raw)
    assert obs == "actual"


# -- serialization (one-turn-per-trajectory trained format) ----------------


def test_serialize_episode_marks_observations():
    text = serialize_episode(
        [
            {"role": "user", "content": "Action: execute_bash\nCommand: ls"},
            {"role": "assistant", "content": "total 0"},
            {"role": "user", "content": "Action: execute_bash\nCommand: pwd"},
        ],
        initial_state="Empty home dir on Ubuntu 24.04, cwd=/home/user",
    )
    blocks = text.split("\n\n")
    assert blocks[0].startswith(RESPONSE_MARKER)  # initial state leads
    assert "Ubuntu 24.04" in blocks[0]
    assert blocks[1] == "Action: execute_bash\nCommand: ls"
    assert blocks[2] == f"{RESPONSE_MARKER}\ntotal 0"
    assert blocks[3] == "Action: execute_bash\nCommand: pwd"


def test_serialize_episode_does_not_double_marker():
    text = serialize_episode(
        [
            {"role": "user", "content": "ls"},
            {"role": "assistant", "content": f"{RESPONSE_MARKER}\nalready marked"},
            {"role": "user", "content": "pwd"},
        ]
    )
    assert text.count(RESPONSE_MARKER) == 1


# -- simulate --------------------------------------------------------------


@pytest.mark.asyncio
async def test_simulate_refuses_when_empty_slot(monkeypatch):
    from augmentum.config import settings

    monkeypatch.setattr(settings, "engine_secondary_model", "", raising=False)
    driver = AgentWorldDriver(FakeSlot(loaded=False, model_id=""))
    with pytest.raises(WorldModelUnavailable, match="no model loaded in Slot B"):
        await driver.simulate("terminal", [{"role": "user", "content": "ls\n"}])


@pytest.mark.asyncio
async def test_simulate_happy_path_serializes_and_extracts():
    slot = FakeSlot()
    driver = AgentWorldDriver(slot)
    history = [
        {"role": "user", "content": "Action: execute_bash\nCommand: mkdir demo"},
        {"role": "assistant", "content": "user@host:~$ "},
        {"role": "user", "content": "Action: execute_bash\nCommand: ls -la demo"},
    ]
    step = await driver.simulate(
        "terminal", history, initial_state="Fresh Ubuntu container, cwd=/root"
    )

    assert step.observation == "total 0\nuser@host:~$"
    assert step.tag_found is True
    assert step.raw_content == _ON_CONTRACT
    assert step.thinking == "cd then ls"
    assert step.domain == "terminal"
    assert step.prompt_tokens == 1200

    sent = slot.backend.requests[0]
    # system prompt + ONE serialized user message — the trained format
    assert [m.role for m in sent.messages] == ["system", "user"]
    assert "Terminal World Model" in sent.messages[0].content
    body = sent.messages[1].content
    assert body.startswith(f"{RESPONSE_MARKER}\nFresh Ubuntu container")
    assert "Command: mkdir demo" in body
    assert f"{RESPONSE_MARKER}\nuser@host:~$" in body
    assert body.rstrip().endswith("Command: ls -la demo")
    # model-card sampling + qwen3.5 thinking-family guidance
    assert sent.temperature == 0.6
    assert sent.top_p == 0.95
    assert sent.top_k == 20
    assert sent.think is True


@pytest.mark.asyncio
async def test_simulate_chat_framing_escape_hatch():
    slot = FakeSlot()
    driver = AgentWorldDriver(slot)
    await driver.simulate(
        "terminal",
        [
            {"role": "user", "content": "ls"},
            {"role": "assistant", "content": "total 0"},
            {"role": "user", "content": "pwd"},
        ],
        serialized=False,
    )
    sent = slot.backend.requests[0]
    assert [m.role for m in sent.messages] == ["system", "user", "assistant", "user"]


@pytest.mark.asyncio
async def test_simulate_system_override_wins():
    slot = FakeSlot()
    driver = AgentWorldDriver(slot)
    await driver.simulate(
        "web",
        [{"role": "user", "content": "click #submit"}],
        system_override="CUSTOM SIM PROMPT",
    )
    assert slot.backend.requests[0].messages[0].content == "CUSTOM SIM PROMPT"


@pytest.mark.asyncio
async def test_simulate_rejects_history_not_ending_on_action():
    driver = AgentWorldDriver(FakeSlot())
    with pytest.raises(ValueError, match="must end with a user-role"):
        await driver.simulate(
            "terminal",
            [{"role": "user", "content": "ls"}, {"role": "assistant", "content": "ok"}],
        )


@pytest.mark.asyncio
async def test_simulate_rejects_system_in_history():
    driver = AgentWorldDriver(FakeSlot())
    with pytest.raises(ValueError, match="roles must be user/assistant"):
        await driver.simulate(
            "terminal",
            [
                {"role": "system", "content": "you are a terminal"},
                {"role": "user", "content": "ls"},
            ],
        )


@pytest.mark.asyncio
async def test_simulate_empty_observation_is_an_error():
    slot = FakeSlot(backend=FakeBackend(content="", thinking=None))
    driver = AgentWorldDriver(slot)
    with pytest.raises(WorldModelUnavailable, match="empty observation"):
        await driver.simulate("terminal", [{"role": "user", "content": "ls"}])


@pytest.mark.asyncio
async def test_simulate_off_format_output_still_usable():
    # No tags at all — the driver falls back to the cleaned text and
    # flags it via tag_found=False rather than erroring.
    slot = FakeSlot(backend=FakeBackend(content="bare output", thinking=None))
    driver = AgentWorldDriver(slot)
    step = await driver.simulate("terminal", [{"role": "user", "content": "ls"}])
    assert step.observation == "bare output"
    assert step.tag_found is False


# -- judging ----------------------------------------------------------------


def test_parse_judge_output_happy_path():
    raw = (
        "Reasoning about the sim quality...\n"
        "<final_evaluation>\n```json\n"
        '{"strengths": ["plausible ls"], "weaknesses": ["wrong prompt"],'
        ' "scores": {"format": 5, "factuality": 4, "consistency": 4,'
        ' "realism": 5, "quality": 4}}\n'
        "```\n</final_evaluation>"
    )
    parsed = parse_judge_output(raw)
    assert parsed["success"] is True
    assert set(parsed["scores"]) == set(SCORE_DIMENSIONS)
    assert parsed["total_score"] == pytest.approx(4.4)
    assert parsed["strengths"] == ["plausible ls"]


def test_parse_judge_output_rejects_incomplete_scores():
    raw = '<final_evaluation>```json\n{"scores": {"format": 5}}\n```</final_evaluation>'
    assert parse_judge_output(raw)["success"] is False


def test_parse_judge_output_rejects_garbage():
    assert parse_judge_output("no evaluation here")["success"] is False
    assert parse_judge_output("")["success"] is False


def test_build_judge_user_prompt_shape():
    text = build_judge_user_prompt(
        current_turn="Command: ls",
        simulated="total 0",
        ground_truth="total 4",
        context="earlier turns",
    )
    assert "# Context (Historical Interactions):" in text
    assert "**World Model Output (Simulated):**" in text
    assert "**Ground Truth (Real Output):**" in text
    assert "<final_evaluation>" in text
