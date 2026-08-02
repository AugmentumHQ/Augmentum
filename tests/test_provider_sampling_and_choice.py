"""Per-provider sampling clamps + tool_choice/effort demotion. 2026-07-01.

Covers CORRECTIONS backlog items:
  * #10 xAI — ``minimal`` reasoning_effort demoted to ``low`` (Grok 400s on minimal)
  * #13 Moonshot — temperature clamped to [0, 1]
  * #26 Z.AI — temperature clamped to [0, 1], top_p to [0.01, 1]
  * #12 Moonshot — tool_choice="required" demoted to "auto"
  * #7/#9 AI21 — max_output ceiling + json_schema→json_object demotion

The clamp is a strict no-op for providers that declare no range, so a
local engine using temperature > 2 is never touched (regression guard).
"""

from __future__ import annotations

import httpx

from augmentum.models.base import InternalChatRequest, Message
from augmentum.models.openai_compat import OpenAIBackend
from augmentum.models.provider_profiles import PROFILES


def _backend(profile_id: str | None) -> OpenAIBackend:
    profile = PROFILES[profile_id] if profile_id else None
    base = profile.base_url if profile else "http://localhost:8090/v1"
    return OpenAIBackend(httpx.AsyncClient(), base or "http://localhost:8090/v1", None, profile=profile)


def _req(model: str = "test-model", **kw) -> InternalChatRequest:
    return InternalChatRequest(
        model=model,
        messages=[Message(role="user", content="what is 2+2?")],
        **kw,
    )


# --- #10 xAI: minimal reasoning_effort demoted to low -----------------

def test_xai_does_not_accept_minimal():
    assert not PROFILES["xai"].supports_reasoning_effort_minimal


def test_xai_minimal_demoted_to_low():
    payload = _backend("xai")._build_openai_payload(
        _req("grok-4.3", think=True, reasoning_effort="minimal")
    )
    assert payload["reasoning_effort"] == "low"


def test_xai_high_passes_through():
    payload = _backend("xai")._build_openai_payload(
        _req("grok-4.3", think=True, reasoning_effort="high")
    )
    assert payload["reasoning_effort"] == "high"


# --- #13 Moonshot: temperature clamped to [0, 1] ----------------------

def test_moonshot_temperature_range():
    assert PROFILES["moonshot"].temperature_range == (0.0, 1.0)


def test_moonshot_temp_clamped():
    payload = _backend("moonshot")._build_openai_payload(_req(temperature=1.7))
    assert payload["temperature"] == 1.0


def test_moonshot_temp_in_range_untouched():
    payload = _backend("moonshot")._build_openai_payload(_req(temperature=0.6))
    assert payload["temperature"] == 0.6


# --- #26 Z.AI: temperature [0, 1], top_p [0.01, 1] --------------------

def test_zai_temp_clamped():
    payload = _backend("zai")._build_openai_payload(_req("glm-4.7", temperature=2.0))
    assert payload["temperature"] == 1.0


def test_zai_top_p_clamped_low_and_high():
    hi = _backend("zai")._build_openai_payload(_req("glm-4.7", top_p=1.5))
    assert hi["top_p"] == 1.0
    lo = _backend("zai")._build_openai_payload(_req("glm-4.7", top_p=0.0))
    assert lo["top_p"] == 0.01


# --- Regression: providers with no declared range pass through --------

def test_openai_temp_not_clamped():
    # OpenAI accepts [0, 2]; no temperature_range declared → untouched.
    assert PROFILES["openai"].temperature_range is None
    payload = _backend("openai")._build_openai_payload(_req("gpt-5.5", temperature=1.6))
    assert payload["temperature"] == 1.6


def test_local_engine_high_temp_untouched():
    # No profile (anonymous local endpoint) → temperature > 2 preserved.
    payload = _backend(None)._build_openai_payload(_req(temperature=2.5))
    assert payload["temperature"] == 2.5


# --- #12 Moonshot: tool_choice="required" demoted to "auto" -----------

def test_moonshot_rejects_required_tool_choice():
    assert not PROFILES["moonshot"].supports_tool_choice_required


def test_moonshot_required_demoted_to_auto():
    payload = _backend("moonshot")._build_openai_payload(
        _req(tool_choice="required")
    )
    assert payload["tool_choice"] == "auto"


def test_moonshot_auto_tool_choice_untouched():
    payload = _backend("moonshot")._build_openai_payload(_req(tool_choice="auto"))
    assert payload["tool_choice"] == "auto"


def test_other_provider_required_passes_through():
    payload = _backend("openai")._build_openai_payload(
        _req("gpt-5.5", tool_choice="required")
    )
    assert payload["tool_choice"] == "required"


# --- #7/#9 AI21: max_output ceiling + json_schema demotion ------------

def test_ai21_max_output_capped():
    assert PROFILES["ai21"].max_output == 4_096


def test_ai21_no_json_schema():
    assert not PROFILES["ai21"].supports_response_format_json_schema
    payload = _backend("ai21")._build_openai_payload(
        _req(raw_options={"json_schema": {"type": "object", "properties": {}}})
    )
    assert payload["response_format"] == {"type": "json_object"}


# --- #2 Sampler extras: min_p / top_k / repetition_penalty parity ------
#
# Model-card recommended sampling (min_p/top_k/repeat_penalty from
# sampling_profiles) reaches LOCAL engines via the full llama.cpp param set.
# For CLOUD it must be emitted ONLY to providers whose official API documents
# each knob (profile.sampler_extras), clamped to the documented range, with the
# llama.cpp source key ``repeat_penalty`` mapped to the wire ``repetition_
# penalty``. Ranges verified from each provider's official API reference.


def test_fireworks_forwards_all_sampler_extras():
    payload = _backend("fireworks")._build_openai_payload(
        _req("accounts/fireworks/models/qwq-32b", top_k=40,
             raw_options={"min_p": 0.02, "repeat_penalty": 1.1})
    )
    assert payload["top_k"] == 40
    assert payload["min_p"] == 0.02
    # llama.cpp source key repeat_penalty → OpenAI-compat repetition_penalty.
    assert payload["repetition_penalty"] == 1.1
    assert "repeat_penalty" not in payload


def test_together_forwards_and_clamps_min_p():
    payload = _backend("together")._build_openai_payload(
        _req("Qwen/Qwen3-235B", raw_options={"min_p": 1.5})
    )
    assert payload["min_p"] == 1.0  # clamped into Together's documented 0–1


def test_fireworks_top_k_clamped_to_100():
    payload = _backend("fireworks")._build_openai_payload(_req("m", top_k=250))
    assert payload["top_k"] == 100
    assert isinstance(payload["top_k"], int)  # wire type preserved


def test_siliconflow_forwards_min_p_top_k_but_not_repetition():
    # SiliconFlow documents top_k + min_p, but NOT repetition_penalty.
    payload = _backend("siliconflow")._build_openai_payload(
        _req("Qwen/Qwen3-8B", top_k=20,
             raw_options={"min_p": 0.05, "repeat_penalty": 1.1})
    )
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0.05
    assert "repetition_penalty" not in payload
    assert "repeat_penalty" not in payload


def test_openrouter_forwards_sampler_extras():
    payload = _backend("openrouter")._build_openai_payload(
        _req("qwen/qwq-32b", top_k=40, raw_options={"min_p": 0.0, "repeat_penalty": 1.05})
    )
    assert payload["top_k"] == 40
    assert payload["min_p"] == 0.0
    assert payload["repetition_penalty"] == 1.05


# --- Strict providers: extras (incl. top_k) gated OFF → no latent 400 --


def test_deepseek_drops_all_sampler_extras():
    # DeepSeek documents temperature/top_p ONLY; top_k/min_p/repetition would
    # 400 as unknown keys. This is the latent-bug fix: top_k used to be sent
    # to every cloud provider.
    payload = _backend("deepseek")._build_openai_payload(
        _req("deepseek-chat", top_k=40, raw_options={"min_p": 0.02, "repeat_penalty": 1.1})
    )
    assert "top_k" not in payload
    assert "min_p" not in payload
    assert "repetition_penalty" not in payload


def test_openai_drops_top_k_from_recommended_profile():
    # A model with a recommended top_k routed through OpenAI must NOT leak it.
    payload = _backend("openai")._build_openai_payload(_req("gpt-5.5", top_k=20))
    assert "top_k" not in payload


def test_mistral_drops_sampler_extras():
    payload = _backend("mistral")._build_openai_payload(
        _req("mistral-large", top_k=40, raw_options={"min_p": 0.02})
    )
    assert "top_k" not in payload
    assert "min_p" not in payload


# --- Local engine (no profile): top_k preserved, extras left to engine -


def test_local_engine_keeps_top_k():
    # The local classifier path sets top_k on a llama-server that accepts it —
    # must still be forwarded even though there's no profile.
    payload = _backend(None)._build_openai_payload(_req(top_k=40))
    assert payload["top_k"] == 40


def test_together_temperature_clamped_to_one():
    payload = _backend("together")._build_openai_payload(_req("m", temperature=1.8))
    assert payload["temperature"] == 1.0


# --- DeepSeek deprecated penalties are not sent -----------------------


def test_deepseek_skips_deprecated_penalties():
    payload = _backend("deepseek")._build_openai_payload(
        _req("deepseek-chat", frequency_penalty=0.5, presence_penalty=0.5)
    )
    assert "frequency_penalty" not in payload
    assert "presence_penalty" not in payload


def test_other_provider_still_sends_penalties():
    payload = _backend("openai")._build_openai_payload(
        _req("gpt-5.5", frequency_penalty=0.5, presence_penalty=0.3)
    )
    assert payload["frequency_penalty"] == 0.5
    assert payload["presence_penalty"] == 0.3


# --- supported_sampler_params: the UI-affordance capability -----------
#
# The Tuning editor reads this so it never offers a knob the backend drops.
# It MUST mirror _build_openai_payload's emission exactly (wire == UI).


def test_supported_openai_only_standard_knobs():
    s = _backend("openai").supported_sampler_params("gpt-5.5")
    assert s == {"temperature", "top_p", "presence_penalty"}
    # The three non-OpenAI knobs the editor must hide for OpenAI:
    assert not ({"top_k", "min_p", "repeat_penalty"} & s)


def test_supported_deepseek_drops_presence_too():
    # DeepSeek deprecated presence_penalty → the editor should hide it.
    s = _backend("deepseek").supported_sampler_params("deepseek-chat")
    assert s == {"temperature", "top_p"}


def test_supported_fireworks_full_set():
    s = _backend("fireworks").supported_sampler_params("qwq-32b")
    assert s == {"temperature", "top_p", "presence_penalty",
                 "top_k", "min_p", "repeat_penalty"}


def test_supported_siliconflow_has_top_k_min_p_not_repeat():
    s = _backend("siliconflow").supported_sampler_params("Qwen/Qwen3-8B")
    assert "top_k" in s and "min_p" in s
    assert "repeat_penalty" not in s


def test_supported_together_full_set():
    s = _backend("together").supported_sampler_params("Qwen/Qwen3-235B")
    assert {"top_k", "min_p", "repeat_penalty"} <= s


def test_supported_local_engine_has_top_k_not_min_p():
    # Anonymous local endpoint (is_local_engine=True): the openai-compat local
    # path forwards top_k but leaves min_p/repeat to the engine, so the editor
    # reflects exactly that.
    s = _backend(None).supported_sampler_params("some-local-model")
    assert "top_k" in s
    assert {"temperature", "top_p", "presence_penalty"} <= s
    assert "min_p" not in s and "repeat_penalty" not in s


def test_supported_mirrors_payload_emission():
    # Contract check: every key supported_sampler_params claims for a provider
    # is actually emitted by _build_openai_payload (and vice-versa) given a
    # request that sets all knobs.
    for pid in ("openai", "deepseek", "fireworks", "siliconflow", "together"):
        b = _backend(pid)
        model = "test-model"
        claimed = b.supported_sampler_params(model)
        payload = b._build_openai_payload(_req(
            model, temperature=0.7, top_p=0.9, top_k=20, presence_penalty=0.1,
            raw_options={"min_p": 0.02, "repeat_penalty": 1.1},
        ))
        emitted = set()
        for key, wire in (("temperature", "temperature"), ("top_p", "top_p"),
                          ("top_k", "top_k"), ("min_p", "min_p"),
                          ("repeat_penalty", "repetition_penalty"),
                          ("presence_penalty", "presence_penalty")):
            if wire in payload:
                emitted.add(key)
        assert claimed == emitted, f"{pid}: claimed {claimed} != emitted {emitted}"
