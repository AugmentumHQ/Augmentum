"""Tests for per-model sampling profiles + layered resolution.

Load-bearing:
  - recommended_for maps families to known-good sampling (Qwen3=0.6/0.95/20 NOT
    greedy; Gemma=1.0/0.95/64; qwen2 distinct from qwen3; unknown→default);
  - resolve_sampling honors precedence (call ▸ chat ▸ model ▸ global ▸ family) and
    each layer fills ONLY what the layer above left unset;
  - the per-model store round-trips via an injected settings store.
"""

from __future__ import annotations

from augmentum.models.sampling_profiles import (
    SamplingParams,
    load_overrides,
    recommended_for,
    resolve_sampling,
    save_overrides,
)

# --- the auto-import brain -------------------------------------------------

def test_recommended_per_family():
    q3 = recommended_for("Qwen3-4B-Q4_K_M")                          # base Qwen3 → family
    assert (q3.temperature, q3.top_p, q3.top_k) == (0.6, 0.95, 20)   # thinking, not greedy
    g = recommended_for("gemma-4-E4B-it")
    assert (g.temperature, g.top_p, g.top_k) == (1.0, 0.95, 64)
    q2 = recommended_for("Qwen2.5-7B-Instruct")
    assert q2.temperature == 0.7 and q2.top_k == 20                  # distinct from qwen3
    assert recommended_for("some-unknown-model").temperature == 0.7  # default


def test_arch_hint_is_used():
    assert recommended_for("mystery", arch="qwen3").top_k == 20


# --- exact per-model table (official model-card values) --------------------

def test_exact_model_overrides_family():
    # Qwen3.6 general thinking = temp 1.0 (NOT the base-Qwen3 0.6 family value)
    q36_27 = recommended_for("Qwen3.6-27B-Q4_K_S")
    assert (q36_27.temperature, q36_27.top_p, q36_27.top_k) == (1.0, 0.95, 20)
    assert q36_27.presence_penalty == 0.0                            # dense → pp 0.0
    q36_35 = recommended_for("Qwen3.6-35B-A3B-IQ4_XS")
    assert q36_35.temperature == 1.0 and q36_35.presence_penalty == 1.5  # MoE → pp 1.5
    # Coding finetune drops back to the precise-coding temp
    assert recommended_for("Qwen3.6-40B-Deck-Opus-NEO-CODE").temperature == 0.6
    # GLM-4.x → official temp 1.0, and NO top_k (over-constraining breaks Flash)
    glm = recommended_for("Dolphin-Mistral-GLM-4.7-Flash-24B-Venice")
    assert glm.temperature == 1.0 and glm.top_k is None
    # Meta Llama 3.3 generation_config default
    assert recommended_for("Llama-3.3-70B-Instruct-IQ2_XXS").temperature == 0.6
    # Nemotron Nano reasoning
    assert recommended_for("NVIDIA-Nemotron-3-Nano-4B-Q8_0").temperature == 0.6
    # Rocinante (RP) — creative-leaning + min_p
    roci = recommended_for("Rocinante-XL-16B-v1a-Q6_K")
    assert roci.temperature == 1.0 and roci.min_p == 0.02


def test_popular_local_models_self_hosters_may_pull():
    # QwQ-32B reasoning — min_p 0, top_k 40
    qwq = recommended_for("QwQ-32B-Q4_K_M")
    assert (qwq.temperature, qwq.top_k, qwq.min_p) == (0.6, 40, 0.0)
    # Mistral Small 3.x — notably low temp 0.15
    assert recommended_for("Mistral-Small-3.2-24B-Instruct-2506").temperature == 0.15
    # Mistral Nemo — lower temp 0.3 (bigger vocab)
    assert recommended_for("Mistral-Nemo-Instruct-2407").temperature == 0.3
    # Magistral (Mistral reasoning) ≠ caught by the mistral family
    assert recommended_for("Magistral-Small-2506").temperature == 0.7
    # Cohere Command — A creative vs R grounded
    assert recommended_for("command-a-03-2025").temperature == 0.9
    assert recommended_for("c4ai-command-r-v01").temperature == 0.3
    # Phi-4 reasoning variants want 0.8/k50; base phi-4 ~0.5 (order matters)
    assert recommended_for("Phi-4-reasoning-plus").top_k == 50
    assert recommended_for("Phi-4-mini-reasoning").temperature == 0.8
    assert recommended_for("phi-4-Q4_K_M").temperature == 0.5
    # Llama 3.x → Meta generation_config default via the family (0.6/0.9)
    eff = recommended_for("Meta-Llama-3.1-8B-Instruct")
    assert (eff.temperature, eff.top_p) == (0.6, 0.9)


# --- merge semantics -------------------------------------------------------

def test_merged_over_fills_only_none():
    top = SamplingParams(temperature=0.2)               # only temp set
    base = SamplingParams(temperature=0.9, top_p=0.8, top_k=40)
    m = top.merged_over(base)
    assert m.temperature == 0.2                          # self wins
    assert m.top_p == 0.8 and m.top_k == 40              # filled from base


def test_to_request_kwargs_drops_none():
    assert SamplingParams(temperature=0.5).to_request_kwargs() == {"temperature": 0.5}


# --- layered precedence ----------------------------------------------------

def test_resolve_precedence_call_wins_then_chat_then_model_then_global():
    eff = resolve_sampling(
        "Qwen3.6-35B",
        request=SamplingParams(temperature=0.1),        # per-call
        per_chat=SamplingParams(temperature=0.3, top_p=0.5),
        per_model=SamplingParams(temperature=0.4, top_k=10),
        global_default=SamplingParams(temperature=0.9, min_p=0.05))
    assert eff.temperature == 0.1                        # per-call wins
    assert eff.top_p == 0.5                              # from per-chat
    assert eff.top_k == 10                               # from per-model
    assert eff.min_p == 0.05                             # from global


def test_resolve_falls_through_to_family_default():
    # nothing set anywhere → the family recommendation fills it (base Qwen3)
    eff = resolve_sampling("Qwen3-4B")
    assert (eff.temperature, eff.top_p, eff.top_k) == (0.6, 0.95, 20)


def test_resolve_partial_layers_compose():
    # only a per-model temp override; the rest comes from the family
    eff = resolve_sampling("gemma-4-E4B", per_model=SamplingParams(temperature=0.5))
    assert eff.temperature == 0.5                        # per-model override
    assert eff.top_p == 0.95 and eff.top_k == 64         # gemma family fills the rest


# --- per-model store -------------------------------------------------------

class _Store:
    """Mirrors SettingsStore's real surface: get/set (global) +
    get_user/set_user/get_user_or_global (per-tenant)."""

    def __init__(self):
        self.g: dict = {}
        self.u: dict = {}

    async def get(self, key):
        return self.g.get(key)

    async def set(self, key, value):
        self.g[key] = value

    async def get_user(self, user_id, key):
        return self.u.get((user_id, key))

    async def set_user(self, user_id, key, value):
        self.u[(user_id, key)] = value

    async def get_user_or_global(self, user_id, key):
        v = self.u.get((user_id, key))
        return v if v is not None else self.g.get(key)


async def test_store_round_trips():
    store = _Store()
    assert (await load_overrides("Qwen3.6-35B", store)).temperature is None  # none yet
    await save_overrides("Qwen3.6-35B", SamplingParams(temperature=0.55, top_k=30), store)
    got = await load_overrides("Qwen3.6-35B", store)
    assert got.temperature == 0.55 and got.top_k == 30
    assert (await load_overrides("other", store)).temperature is None        # keyed by model


async def test_store_absent_is_empty_not_crash():
    assert (await load_overrides("m", None)).to_request_kwargs() == {}


async def test_per_user_overrides_global_seed_with_fallback():
    store = _Store()
    # download auto-seed writes the install-wide default (no user_id)
    await save_overrides("Qwen3.6-35B", SamplingParams(temperature=0.6, top_k=20), store)
    # a user with no edit inherits the seed via user→global fallback
    inherited = await load_overrides("Qwen3.6-35B", store, user_id="u1")
    assert inherited.temperature == 0.6 and inherited.top_k == 20
    # the user edits it in the library → their value wins, seed untouched
    await save_overrides("Qwen3.6-35B", SamplingParams(temperature=0.9), store, user_id="u1")
    assert (await load_overrides("Qwen3.6-35B", store, user_id="u1")).temperature == 0.9
    assert (await load_overrides("Qwen3.6-35B", store)).temperature == 0.6   # global seed intact
    # a different user still sees the seed
    assert (await load_overrides("Qwen3.6-35B", store, user_id="u2")).temperature == 0.6


# --- the apply-point (mutates a request through the full stack) -------------

class _Req:
    def __init__(self, model, temperature=None, top_p=None, top_k=None):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.raw_options = {}


async def test_apply_fills_unset_from_family_and_keeps_request():
    from augmentum.models.sampling_profiles import apply_to_request
    req = _Req("Qwen3.6-35B", temperature=0.2)            # per-call temp set
    await apply_to_request(req, None)                      # no store/global → family fills rest
    assert req.temperature == 0.2                          # per-call preserved
    assert req.top_p == 0.95 and req.top_k == 20           # Qwen3 family filled the gaps
    # min_p has no dataclass field here → lands in raw_options only if set (it isn't for qwen3)
    assert "min_p" not in req.raw_options


async def test_apply_per_model_override_beats_family():
    from augmentum.models.sampling_profiles import apply_to_request
    store = _Store()
    await save_overrides("gemma-4-E4B", SamplingParams(temperature=0.5), store)
    req = _Req("gemma-4-E4B")
    await apply_to_request(req, store)
    assert req.temperature == 0.5                          # per-model override
    assert req.top_p == 0.95 and req.top_k == 64           # gemma family fills the rest
