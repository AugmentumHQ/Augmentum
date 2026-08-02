"""Per-model sampling profiles + layered resolution.

The gap: Augmentum had ONE global temperature for every model — swap models and the
sampling doesn't follow, and nothing knew Qwen3 wants 0.6 while Gemma-4 wants 1.0.
This adds the OpenWebUI-style layering:

    per-call (request) ▸ per-chat (session) ▸ per-model (auto-imported on download,
    editable in the model library) ▸ global (Settings) ▸ family default

Each layer fills only what the layer above left unset (``resolve_sampling``).

``recommended_for(model, arch)`` is the AUTO-IMPORT brain — name/arch → the
family's known-good sampling (seed this on download). ``load_overrides`` /
``save_overrides`` are the per-model store (mirrors ``image/model_profiles.py``),
the library's manual edits. Pure + injected store, so it's testable.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, fields
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SamplingParams:
    """All fields optional → ``None`` means "defer to the next layer"."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    presence_penalty: float | None = None

    def merged_over(self, other: SamplingParams | None) -> SamplingParams:
        """Return self with each ``None`` field filled from ``other`` (self wins)."""
        if other is None:
            return SamplingParams(**{f.name: getattr(self, f.name) for f in fields(self)})
        return SamplingParams(**{
            f.name: (getattr(self, f.name) if getattr(self, f.name) is not None
                     else getattr(other, f.name))
            for f in fields(self)})

    def to_request_kwargs(self) -> dict:
        """Only the set fields — to splat onto a chat request without clobbering
        with Nones."""
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) is not None}

    def to_dict(self) -> dict:
        return self.to_request_kwargs()

    @classmethod
    def from_dict(cls, d: dict | None) -> SamplingParams:
        d = d or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known and v is not None})


# Family → known-good sampling (the COARSE fallback when no exact-model match).
# Sources: Qwen3 card (thinking), Gemma 3/4 card, DeepSeek-R1 guidance, GLM/Z.AI
# card, general chat defaults. The fine-grained per-model overrides live in
# _MODEL_SAMPLING below; see docs/model-cards/sampling-reference.md for sources.
_FAMILY_SAMPLING: dict[str, SamplingParams] = {
    "qwen3":   SamplingParams(0.6, 0.95, 20),          # base Qwen3 thinking — NOT greedy
    "qwen2":   SamplingParams(0.7, 0.8, 20),           # Qwen2/2.5 chat
    "gemma":   SamplingParams(1.0, 0.95, 64),          # Gemma 3/4
    "deepseek": SamplingParams(0.6, 0.95, None),       # R1/V3 guidance ~0.5-0.7
    "glm":     SamplingParams(1.0, 0.95, None),        # Z.AI card: temp 1.0; do NOT add top_k
    "exaone":  SamplingParams(0.6, 0.95, None),
    "minimax": SamplingParams(0.6, 0.95, None),
    "mistral": SamplingParams(0.7, 0.95, None),
    "llama":   SamplingParams(0.6, 0.9, None),          # Meta generation_config default (Llama 3.x)
}
_DEFAULT_SAMPLING = SamplingParams(0.7, 0.95, None)


# Exact/substring per-model recommendations pulled from the official model
# cards (verbatim values; sources in docs/model-cards/sampling-reference.md).
# Checked BEFORE the family fallback — the FIRST substring that matches the
# lowercased model name wins, so order most-specific → least-specific. These
# encode the documented GENERAL-task profile; the coding profile (lower temp)
# is noted in the reference doc and reachable via per-chat/per-model override.
_MODEL_SAMPLING: tuple[tuple[str, SamplingParams], ...] = (
    # === Models you run, plus the popular local-oriented ones a self-hoster on
    # bigger hardware is likely to pull. General-task profile; coding variants
    # (lower temp) noted in the reference doc. Order = most-specific first. ===

    # --- Qwen 3.6 (general thinking = temp 1.0; coding finetune = 0.6) -------
    ("qwen3.6-40b-deck", SamplingParams(0.6, 0.95, 20)),                        # coding finetune
    ("qwen3.6-35b",      SamplingParams(1.0, 0.95, 20, presence_penalty=1.5)),  # MoE: pp 1.5
    ("qwen3.6-27b",      SamplingParams(1.0, 0.95, 20, presence_penalty=0.0)),  # dense: pp 0.0
    ("qwen3.6",          SamplingParams(1.0, 0.95, 20)),
    # --- Qwen 3.5 (general thinking = temp 1.0) -----------------------------
    ("qwen3.5",          SamplingParams(1.0, 0.95, 20, presence_penalty=1.5)),
    # --- QwQ-32B (Qwen reasoning; min_p 0, top_k 20-40) ---------------------
    ("qwq",              SamplingParams(0.6, 0.95, 40, min_p=0.0)),
    # --- GLM-5.x (Z.AI card: temp 1.0, top_p 0.95, min_p 0.01) --------------
    ("glm-5",            SamplingParams(temperature=1.0, top_p=0.95, min_p=0.01)),
    # --- GLM-4.x (Z.AI: temp 1.0; over-constraining breaks Flash tool-calls) -
    ("glm-4.7",          SamplingParams(temperature=1.0, top_p=0.95)),
    ("glm-4",            SamplingParams(temperature=1.0, top_p=0.95)),
    # --- MiniMax M3 (Jun 2026 card: temp 1.0, top_p 0.95, top_k 40) — does NOT
    # inherit the M2.x family default of temp 0.6; new arch wants higher temp.
    ("minimax-m3",       SamplingParams(1.0, 0.95, 40)),
    # --- NVIDIA Nemotron Nano (reasoning-on) --------------------------------
    ("nemotron-3-nano",  SamplingParams(temperature=0.6, top_p=0.95)),
    ("nemotron",         SamplingParams(temperature=0.6, top_p=0.95)),
    # --- Mistral popular locals (each deviates from the 0.7 family default) --
    ("mistral-small-3",  SamplingParams(temperature=0.15)),                     # Small 3.x: temp 0.15
    ("magistral",        SamplingParams(temperature=0.7, top_p=0.95)),          # Mistral reasoning
    ("mistral-nemo",     SamplingParams(temperature=0.3)),                      # Nemo: lower temp
    # --- Cohere Command (Command-A creative vs Command-R grounded) ----------
    ("command-a",        SamplingParams(temperature=0.9, top_p=0.95, repeat_penalty=1.04)),
    ("command-r",        SamplingParams(temperature=0.3)),
    # --- Microsoft Phi-4 (reasoning variants want temp 0.8; base ~0.5) ------
    ("phi-4-mini-reasoning", SamplingParams(0.8, 0.95, 50)),
    ("phi-4-reasoning",  SamplingParams(0.8, 0.95, 50)),
    ("phi-4",            SamplingParams(temperature=0.5)),
    # --- TheDrummer Rocinante (Mistral-Nemo RP; 0.7 chill → 1.2 creative) ----
    ("rocinante",        SamplingParams(temperature=1.0, top_p=0.95, min_p=0.02)),
)


def _family_of(model_name: str, arch: str = "") -> str:
    """Lightweight family detection for SAMPLING (name first, arch as a hint).
    Distinguishes qwen3 vs qwen2.x — they want different sampling."""
    s = f"{model_name} {arch}".lower()
    if "qwen3" in s or "qwen-3" in s or "qwen3.5" in s or "qwen3.6" in s:
        return "qwen3"
    if "qwen2" in s or "qwen-2" in s or "qwen2.5" in s:
        return "qwen2"
    if "gemma" in s:
        return "gemma"
    for fam in ("deepseek", "glm", "exaone", "minimax", "mistral", "llama"):
        if fam in s:
            return fam
    return ""


def _copy(params: SamplingParams) -> SamplingParams:
    return SamplingParams(**{f.name: getattr(params, f.name) for f in fields(params)})


def recommended_for(model_name: str, arch: str = "") -> SamplingParams:
    """The AUTO-IMPORT brain: a model's known-good sampling. Tries the exact
    per-model table (official model-card values) first, then the coarse family
    fallback. Seed this into the per-model profile on download; the user can
    edit it after. See docs/model-cards/sampling-reference.md for sources."""
    name = (model_name or "").lower()
    for needle, params in _MODEL_SAMPLING:
        if needle in name:
            log.info("sampling_recommended", model=model_name[:80], match=needle)
            return _copy(params)
    fam = _family_of(model_name, arch)
    params = _FAMILY_SAMPLING.get(fam, _DEFAULT_SAMPLING)
    log.info("sampling_recommended", model=model_name[:80], family=fam or "default")
    return _copy(params)


# --- per-model store (mirrors image/model_profiles.py) --------------------

_KEY_PREFIX = "sampling_profile:"


def _profile_key(model_name: str) -> str:
    short = (model_name or "").rsplit("/", 1)[-1].lower()
    return f"{_KEY_PREFIX}{short}"


async def load_overrides(model_name: str, settings_store: Any, *,
                         user_id: str = "") -> SamplingParams:
    """The saved per-model sampling (empty SamplingParams if none/no store).

    With a ``user_id`` the read falls back user → global (the auto-seeded
    install default), so a user inherits the download-seeded profile until
    they edit it in the model library. Without a ``user_id`` it reads the
    global (seed) layer directly."""
    if not settings_store:
        return SamplingParams()
    key = _profile_key(model_name)
    try:
        if user_id:
            raw = await settings_store.get_user_or_global(user_id, key)
        else:
            raw = await settings_store.get(key)
    except Exception as exc:  # noqa: BLE001 — a read miss is just "no override"
        log.warning("sampling_load_failed", model=model_name[:80], error=repr(exc))
        return SamplingParams()
    if not raw:
        return SamplingParams()
    try:
        return SamplingParams.from_dict(json.loads(raw))
    except (ValueError, TypeError):
        return SamplingParams()


async def save_overrides(model_name: str, params: SamplingParams, settings_store: Any, *,
                         user_id: str = "") -> None:
    """Persist a per-model profile. ``user_id`` → the user's library edit
    (``set_user``); no ``user_id`` → the install-wide seed (``set``, what the
    download auto-import writes)."""
    if not settings_store:
        return
    key = _profile_key(model_name)
    value = json.dumps(params.to_dict())
    if user_id:
        await settings_store.set_user(user_id, key, value)
    else:
        await settings_store.set(key, value)
    log.info("sampling_saved", model=model_name[:80], user=user_id or "global",
             params=params.to_dict())


# --- the resolver: merge all layers by precedence -------------------------

def resolve_sampling(model_name: str, *, request: SamplingParams | None = None,
                     per_chat: SamplingParams | None = None,
                     per_model: SamplingParams | None = None,
                     global_default: SamplingParams | None = None,
                     arch: str = "") -> SamplingParams:
    """Effective sampling = per-call ▸ per-chat ▸ per-model ▸ global ▸ family
    default. Each layer fills only what the layer above left unset."""
    return (
        (request or SamplingParams())
        .merged_over(per_chat)
        .merged_over(per_model)
        .merged_over(global_default)
        .merged_over(recommended_for(model_name, arch)))


def _from_request(req: Any) -> SamplingParams:
    """The per-call layer: sampling already on the incoming request."""
    raw = getattr(req, "raw_options", None) or {}
    def pick(name):
        v = getattr(req, name, None)
        return v if v is not None else raw.get(name)
    return SamplingParams(temperature=pick("temperature"), top_p=pick("top_p"),
                          top_k=pick("top_k"), min_p=pick("min_p"),
                          repeat_penalty=pick("repeat_penalty"),
                          presence_penalty=pick("presence_penalty"))


async def apply_to_request(req: Any, settings_store: Any, *,
                           user_id: str = "",
                           per_chat: SamplingParams | None = None,
                           global_default: SamplingParams | None = None,
                           arch: str = "") -> Any:
    """THE apply-point: resolve the full layer stack for ``req.model`` and write
    the effective sampling back onto ``req`` (a dataclass field if it has one,
    else into ``raw_options``). Per-model overrides load from the store; the
    per-call layer is whatever the request already carried. Mutates + returns req.
    Never raises — a failure leaves the request untouched (falls back to whatever
    it had)."""
    try:
        model = getattr(req, "model", "") or ""
        per_model = await load_overrides(model, settings_store, user_id=user_id)
        eff = resolve_sampling(model, request=_from_request(req), per_chat=per_chat,
                               per_model=per_model, global_default=global_default, arch=arch)
        # Ensure a raw_options dict to mirror into: local backends (llama.cpp)
        # read temperature/top_p from the dataclass FIELDS but top_k/min_p/
        # repeat_penalty from raw_options. Writing both channels guarantees the
        # effective value reaches the backend regardless of which it consumes.
        raw = getattr(req, "raw_options", None)
        if not isinstance(raw, dict):
            raw = {}
            with contextlib.suppress(Exception):
                req.raw_options = raw
        for key, value in eff.to_request_kwargs().items():
            if hasattr(req, key):
                setattr(req, key, value)
            raw[key] = value
    except Exception as exc:  # noqa: BLE001 — sampling must never break a chat
        log.warning("sampling_apply_failed", error=repr(exc))
    return req
