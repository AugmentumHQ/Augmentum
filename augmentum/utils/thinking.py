"""Thinking block extraction and streaming normalization.

Reasoning models use several wire formats for hidden reasoning:

* ``<think>...</think>`` blocks — DeepSeek-R1, Qwen3, most newer llama.cpp
  derivatives
* Channelized Gemma / GPT-OSS output —
  ``<|channel|>analysis<|message|>…<|end|>`` (symmetric delimiters, channel
  names: ``analysis``, ``commentary``, ``final``)
* Channelized **Gemma 4** output — ``<|channel>thought\\n…<channel|>`` with
  *asymmetric* delimiters. Not a slash-variant of the opener; it is a new
  format introduced with Gemma 4 (April 2026) and is NOT interchangeable
  with the earlier Gemma/GPT-OSS markers above.
* Native side channels — Ollama ``thinking``, OpenAI ``reasoning_content``

This module normalizes all of them through a single streaming buffer and a
post-hoc helper. The stream buffer runs the relevant parsers in parallel per
chunk and routes the result into ``(clean_content, thinking_text)``.

Parsers are selected by a family hint (``family`` or ``model`` arg). With no
hint, the buffer runs every format parser so behavior is unchanged for legacy
callers. New callers should pass a hint so we can skip false-positive checks
and enforce format-specific quirks (e.g. Gemma 4 requires ``skip_special_
tokens=False`` at the tokenizer boundary — if that isn't honored upstream the
channel markers never reach us and parsing silently no-ops).
"""

from __future__ import annotations

import re

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns + partial-token tables per format
# ---------------------------------------------------------------------------

# <think>...</think> — DeepSeek-R1 / Qwen3 / most reasoning models
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# Mistral Magistral — `[THINK]...[/THINK]` SPECIAL TOKENS (single token IDs).
# Symmetric: both tokens appear in the response stream (controlled by the
# Magistral system prompt, not a chat template kwarg). Same parser shape as
# `<think>` — just different delimiter strings, so the streaming buffer
# parameterizes its markers rather than running a separate state machine.
#
# Gotcha: Magistral requires `mistral-common >= 1.8.5` for proper special-token
# encoding. With llama.cpp + GGUF the tokenizer config in the GGUF determines
# whether `[THINK]` / `[/THINK]` survive serialization. If they're stripped
# (skip_special_tokens=True at the boundary), parsing silently no-ops — this
# is a GGUF-build issue, not something we can fix here.
_MAGISTRAL_THINK_PATTERN = re.compile(r"\[THINK\](.*?)\[/THINK\]", re.DOTALL)
# Families whose models emit ``[THINK]`` / ``[/THINK]`` SPECIAL TOKENS
# instead of literal ``<think>`` / ``</think>``. The streaming buffer
# parameterises its delimiters based on family membership.
#
# Mistral Magistral 1/2 originated the format; Ministral 3 Reasoning
# (Mistral, Dec 2025 - Mar 2026) reuses the same special-token
# convention per the unsloth.ai/docs/models/tutorials/ministral-3
# guide. All Ministral-3 *Reasoning-2512* variants (3B / 8B / 14B)
# fall here. Plain Ministral-3 (non-reasoning) emits no thinking
# block — gated by the family-detect path returning the matching
# family key only for reasoning checkpoints.
_MAGISTRAL_FAMILIES: frozenset[str] = frozenset({
    "magistral", "magistral2",
    "ministral3",
})

# Gemma / GPT-OSS channelized output — SYMMETRIC <|…|> delimiters.
# Channel types: analysis (reasoning), commentary (hidden), final (visible).
# Segments terminated by <|end|>.
_GEMMA_CHANNEL_RE = re.compile(
    r"(?:<\|start\|>assistant\s*)?"
    r"<\|channel\|>(analysis|commentary|final)"
    r"(?:\s*<\|constrain\|>[A-Za-z0-9_-]+)?"
    r"<\|message\|>",
    re.DOTALL,
)
_GEMMA_PARTIAL_TOKENS: tuple[str, ...] = (
    "<|start|>assistant",
    "<|channel|>",
    "<|channel|>analysis<|message|>",
    "<|channel|>commentary<|message|>",
    "<|channel|>final<|message|>",
    "<|message|>",
    "<|constrain|>",
    "<|end|>",
)

# Gemma 4 channel — ASYMMETRIC delimiters. Opener: "<|channel>" (left pipe
# only) followed by the literal word "thought" + newline. Closer:
# "<channel|>" (right pipe only). Everything between is reasoning; everything
# after the closer is the final answer. Closer is NOT guaranteed on truncated
# streams — treat missing closer as "still thinking".
_GEMMA4_CHANNEL_OPEN_RE = re.compile(r"<\|channel>thought\n?", re.DOTALL)
_GEMMA4_CHANNEL_CLOSE = "<channel|>"
_GEMMA4_PARTIAL_TOKENS: tuple[str, ...] = (
    # Any prefix of the opener or closer can appear split across chunks.
    "<|channel>thought\n",
    "<|channel>thought",
    "<|channel>",
    "<|channel",
    "<channel|>",
    "<channel|",
    "<channel",
)


# ---------------------------------------------------------------------------
# Family detection / parser selection
# ---------------------------------------------------------------------------

# Model-family → which format parsers to run. Unknown families run all
# parsers (backward-compatible default). Values are the logical format keys
# used internally below.
_FAMILY_PARSERS: dict[str, tuple[str, ...]] = {
    # <think> lineage
    "qwen3":     ("think",),
    "qwen35":    ("think",),
    "qwen35moe": ("think",),
    "qwen3moe":  ("think",),
    "qwen3next": ("think",),
    # DeepSeek lineage. R1/V2/V3 emit symmetric <think>...</think>. V3.2 and V4
    # (Pro/Flash) switched to the GLM-style asymmetric format — opener in prompt
    # prefix, only </think> in the response stream. See _STARTS_THINKING_FAMILIES.
    "deepseek2":  ("think",),
    "deepseek3":  ("think",),
    "deepseek32": ("think",),
    "deepseek4":  ("think",),
    # GLM thinking models (Z.AI, GLM-4.5/4.6/4.7). See _STARTS_THINKING_FAMILIES
    # below for the critical bit: GLM's chat template puts <think> in the prompt
    # prefix (via add_generation_prompt), so the response stream starts directly
    # with reasoning content with no opening <think> in the visible stream.
    # The parser must initialize in "inside thinking" state — mirroring the
    # approach in Ollama's GLM47Parser (model/parsers/glm47.go).
    "glm":     ("think",),
    "glm4":    ("think",),
    "glm45":   ("think",),
    "glm46":   ("think",),
    "glm47":   ("think",),
    # GLM-5.x (Z.AI, GLM-5 / 5.1 / 5.2, Jun 2026). Same asymmetric-closer
    # lineage as 4.5+. Resolved via the generic "glm" name needle before;
    # listed explicitly so the arch-first path is deterministic for GGUF arch
    # strings like ``glm5`` / ``glm52`` and future GLM-5.x point releases.
    "glm5":    ("think",),
    "glm52":   ("think",),
    "chatglm": ("think",),
    # MiniMax M-series (M2 / M2.5 / M2.7). Template ends prompt with
    # `]~b]ai\n<think>\n`, so response stream starts inside a think block —
    # same asymmetric pattern as GLM.
    "minimax":   ("think",),
    "minimaxm2": ("think",),
    # MiniMax M3 (Jun 2026) — new MSA architecture + native multimodal (GGUF
    # arch ``minimax_m3_vl``). Reasoning handling is INHERITED from M2
    # (asymmetric <think>). VERIFY against the real M3 chat template before
    # trusting at scale: if M3 turns out symmetric or non-reasoning, remove it
    # from ``_STARTS_THINKING_FAMILIES`` (a wrong asymmetric assumption would
    # route the first real content into reasoning).
    "minimaxm3":     ("think",),
    "minimax_m3_vl": ("think",),
    # LG AI EXAONE 4.0 / EXAONE-Deep. README documents
    # `enable_thinking=True` opening a reasoning block via the chat template,
    # so opener is in the prompt prefix — same asymmetric pattern as GLM.
    "exaone":  ("think",),
    "exaone4": ("think",),
    # NVIDIA Nemotron 3 Nano (Reasoning variants) — symmetric <think>
    # with `enable_thinking` kwarg. Listed here so the family hint scopes
    # parsing cleanly and skips false-positive Gemma checks. `nemotron_h_moe`
    # is the MoE variant (Nemotron 3 Nano Omni 30B-A3B); reasoning behavior
    # is identical to `nemotron_h`, only the inference-engine MoE expert
    # offload path differs (handled in llama_server_manager).
    "nemotron":       ("think",),
    "nemotron_h":     ("think",),
    "nemotron_h_moe": ("think",),
    "nemotronh":      ("think",),
    # Tencent Hunyuan Hy3 — symmetric <think>. Listed for clean family
    # scoping (parser already worked via the default fallback path).
    "hunyuan": ("think",),
    "hy3":     ("think",),
    # Mistral Magistral — symmetric `[THINK]...[/THINK]` special tokens.
    # The same `think` parser key drives extraction; the streaming buffer
    # picks the correct delimiter strings based on family membership in
    # _MAGISTRAL_FAMILIES (above).
    "magistral":  ("think",),
    "magistral2": ("think",),
    # Mistral Ministral-3 Reasoning-2512 (3B / 8B / 14B). Reuses the
    # Magistral [THINK] bracket-token convention per unsloth's
    # tutorials. ``mistral-common >= 1.8.5`` still required so the
    # GGUF tokenizer encodes the special tokens correctly.
    "ministral3": ("think",),
    # Gemma 3 / GPT-OSS symmetric channels
    "gemma":    ("gemma_channel", "think"),
    "gemma2":   ("gemma_channel", "think"),
    "gemma3":   ("gemma_channel", "think"),
    "gemma3n":  ("gemma_channel", "think"),
    "gpt_oss":  ("gemma_channel",),
    "gpt-oss":  ("gemma_channel",),
    # Gemma 4 asymmetric channels
    "gemma4":   ("gemma4_channel", "think"),
    # Moonshot Kimi K2.6 / K2.6-Thinking (Apr 2026). Emits symmetric
    # ``<think>...</think>``; vLLM ships a dedicated kimi_k2 reasoning
    # parser but the on-the-wire format is standard. Toggle goes through
    # ``thinking`` (bool), not ``enable_thinking`` — handled in
    # llama_cpp.py::_chat_template_kwargs.
    "kimi":    ("think",),
    "kimi2":   ("think",),
    "kimi_k2": ("think",),
    # Xiaomi MiMo-V2.5 / MiMo-V2.5-Pro (Apr-May 2026). Symmetric
    # ``<think>...</think>``, ``enable_thinking`` kwarg (Qwen-compatible
    # name). New GGUF arch ``mimo2`` — confirm pinned LLAMA_SERVER_VERSION
    # supports it before bulk-installing.
    "mimo":   ("think",),
    "mimo2":  ("think",),
    # NVIDIA Nemotron-Cascade 2 (Mar 2026). Symmetric ``<think>`` with a
    # ChatML base. The toggle pattern is NEW — empty ``<think></think>``
    # prefix to disable thinking, not a kwarg — but extraction itself
    # uses the same symmetric parser. Family key distinct from
    # ``nemotron`` (Nemotron-3 Nano) so the toggle dispatcher can branch.
    "nemotron_cascade":   ("think",),
    "nemotron_cascade2":  ("think",),
    # Liquid AI LFM2 / LFM2.5 (Nov 2025 – Jan 2026). Hybrid conv + GQA
    # architecture, edge-tier. Symmetric ``<think>...</think>`` —
    # LFM2.5-1.2B-Thinking, LFM2-8B-A1B, LFM2-2.6B all emit standard
    # think markers. Verified 2026-06-10 in voice path logs: model
    # routed via fabric peer was emitting visible ``<think>`` blocks
    # because parser dispatch had no LFM family entry → reasoning
    # leaked into deliver output and into TTS.
    "lfm":   ("think",),
    "lfm2":  ("think",),
    "lfm25": ("think",),
    # Non-reasoning models — no parsers needed, but we still run none and
    # return the content unchanged.
    "llama":   (),
    "mistral": (),
}

# Families whose chat template puts the opening <think> tag in the PROMPT
# PREFIX rather than the response. The model's response begins inside a
# thinking block and the only delimiter that arrives in the stream is the
# closing </think>. The parser must initialize in "inside thinking" state
# to route everything before </think> into reasoning_content.
#
# This is exactly what Ollama's GLM47Parser does (see
# https://github.com/ollama/ollama/blob/main/model/parsers/glm47.go):
#   if thinkValue == nil || thinkValue.Bool() {
#       p.state = glm46ParserState_CollectingThinking
#   }
#
# When `enable_thinking` is explicitly false, the chat template puts </think>
# in the prompt prefix instead and the response starts in the "after thinking"
# state — no special handling needed since there's no thinking content at all.
_STARTS_THINKING_FAMILIES: frozenset[str] = frozenset({
    # GLM (Z.AI) — original asymmetric family. glm5/glm52 = GLM-5.x (Jun 2026),
    # same asymmetric-closer behavior.
    "glm", "glm4", "glm45", "glm46", "glm47", "glm5", "glm52", "chatglm",
    # DeepSeek V3.2 / V4 (Pro/Flash) — switched to GLM-style asymmetric in
    # late 2026. Earlier R1/V2/V3 variants stay symmetric and are NOT in
    # this set.
    "deepseek32", "deepseek4",
    # MiniMax M-series — chat template ends with `<think>\n` in
    # `add_generation_prompt`, identical pattern to GLM. minimaxm3 = M3 (Jun
    # 2026) — inherited assumption, see the verify-note in _FAMILY_PARSERS.
    "minimax", "minimaxm2", "minimaxm3", "minimax_m3_vl",
    # LG AI EXAONE 4.0 / EXAONE-Deep — `enable_thinking=True` opens a
    # reasoning block that the response continues directly.
    "exaone", "exaone4",
    # Qwen 3.x family — verified 2026-05-10 via /apply-template against
    # Qwen3.6-35B-A3B-UD-Q4_K_XL: prompt rendering ends with
    # `<|im_start|>assistant\n<think>\n`, so the response stream starts
    # INSIDE a think block and only `</think>` arrives in the visible
    # stream. Symptom before adding here: chain-of-thought ("Let me start
    # by exploring...") leaked as visible content under native strategy
    # because the symmetric-think parser saw no opening tag. Gated on
    # ``thinking_enabled is not False`` in the streaming buffer, so
    # disabling thinking via UI still works (the template puts `</think>`
    # in the prefix in that case and there's no leading think block).
    "qwen3", "qwen35", "qwen35moe", "qwen3moe", "qwen3next",
})


# Name-substring → family registry. Used as fallback when GGUF
# ``general.architecture`` is missing or doesn't directly match a
# family key in ``_FAMILY_PARSERS`` (the arch field encodes only the
# major lineage — e.g. ``glm4`` covers GLM-4.5/4.6/4.7 which need
# different parsers due to the asymmetric-closer change in 4.5+).
#
# IMPORTANT: ordering is computed automatically — sorted by needle
# length descending at module load time — so longer / more-specific
# needles take precedence over shorter ones. Adding a new entry is
# a single-line change and the sort handles "qwen3.5 must match
# before qwen3" by construction. Pre-T2-3 the list was hand-ordered
# and a misplaced insert silently routed a model family to the
# wrong parser.
_NAME_NEEDLES_RAW: dict[str, str] = {
    # Gemma family
    "gemma-4": "gemma4",
    "gemma4":  "gemma4",
    "gemma-3": "gemma3",
    "gemma3":  "gemma3",
    "gpt-oss": "gpt_oss",
    # Qwen 3 / 3.5 / 3.6 (3.6 reuses qwen3.5 arch)
    "qwen3.5": "qwen35",
    "qwen3.6": "qwen35",
    "qwen3":   "qwen3",
    # DeepSeek — V3.2 and V4 use asymmetric (opener in prompt).
    # R1/V2/V3 stay symmetric.
    "deepseek-v4":   "deepseek4",
    "deepseek-v3.2": "deepseek32",
    "deepseek-v3":   "deepseek3",
    "deepseek-v2":   "deepseek2",
    "deepseek-r1":   "deepseek3",
    "deepseek":      "deepseek2",
    # GLM (Z.AI). 5.x + 4.5+ are asymmetric; 4 baseline is symmetric.
    "glm-5.2": "glm52",
    "glm-5.1": "glm5",
    "glm-5":   "glm5",
    "glm5":    "glm5",
    "glm-4.7": "glm47",
    "glm4.7":  "glm47",
    "glm-4.6": "glm46",
    "glm4.6":  "glm46",
    "glm-4.5": "glm45",
    "glm4.5":  "glm45",
    "glm-4":   "glm4",
    "glm4":    "glm4",
    "chatglm": "chatglm",
    "glm":     "glm",
    # MiniMax M-series — M2.x all share the asymmetric pattern. M3 (Jun 2026)
    # is mapped from the inherited M2 assumption (verify per _FAMILY_PARSERS).
    "minimax-m3": "minimaxm3",
    "minimaxm3":  "minimaxm3",
    "minimax-m2": "minimaxm2",
    "minimaxm2":  "minimaxm2",
    "minimax":    "minimax",
    # LG EXAONE 4.0 / EXAONE-Deep (asymmetric).
    "exaone-4":    "exaone4",
    "exaone4":     "exaone4",
    "exaone-deep": "exaone4",
    "exaone":      "exaone",
    # NVIDIA Nemotron 3 Nano (Reasoning). Symmetric.
    # Omni variant (text+vision+audio+video MoE) reuses the same parser;
    # ordering handled by longest-first sort at module load.
    "nemotron-3-nano-omni": "nemotron_h_moe",
    "nemotron-3-omni":      "nemotron_h_moe",
    "nemotron-h-moe":       "nemotron_h_moe",
    "nemotron_h_moe":       "nemotron_h_moe",
    "nemotron-h":           "nemotron_h",
    "nemotronh":            "nemotron_h",
    "nemotron":             "nemotron",
    # Tencent Hunyuan Hy3 — symmetric.
    "hy3":     "hy3",
    "hunyuan": "hunyuan",
    # Mistral Magistral. "magistral" is the only Mistral lineage with
    # reasoning markers — vanilla Mistral 7B / Small / Large do NOT
    # emit [THINK]/[/THINK]. Keep needles narrow to avoid false
    # positives on plain Mistral models.
    "magistral-2": "magistral2",
    "magistral2":  "magistral2",
    "magistral":   "magistral",
    # Liquid AI LFM2 / LFM2.5. GGUF arch field is typically ``lfm2``
    # so the arch path catches it first; these needles cover the
    # name-based fallback for HuggingFace repo / filename matches
    # like "LFM2.5-8B-A1B-Q4_0" and "LiquidAI/LFM2-2.6B-GGUF". The
    # 2.5 needles come before plain "lfm2"/"lfm" so the longest-first
    # sort picks the most specific variant.
    #
    # Note: no ``liquidai`` org-prefix needle. Length-8 ``liquidai``
    # would beat length-6 ``lfm2.5`` in the longest-first sort, so a
    # name like ``LiquidAI/LFM2.5-1.2B-Thinking`` would resolve to the
    # plain ``lfm2`` family key (via the liquidai needle) instead of
    # the more-specific ``lfm25``. The LFM-prefixed needles cover
    # every real model name; the org prefix would only fire on a name
    # like ``LiquidAI/Custom`` which doesn't exist in practice.
    "lfm-2.5":    "lfm25",
    "lfm2.5":     "lfm25",
    "lfm-2":      "lfm2",
    "lfm2":       "lfm2",
    "lfm":        "lfm",
}

# Sort longest-first so a query for ``qwen-3.5-coder-instruct`` matches
# the ``qwen3.5`` (7-char) needle before falling through to the
# ``qwen3`` (5-char) one. Ties broken by the original dict insertion
# order which is stable in Python 3.7+.
_NAME_NEEDLES: tuple[tuple[str, str], ...] = tuple(
    sorted(_NAME_NEEDLES_RAW.items(), key=lambda kv: -len(kv[0]))
)


def detect_reasoning_family(
    model: str | None = None, arch: str | None = None
) -> str | None:
    """Family lookup for parser selection.

    Resolution priority:

    1. ``arch`` directly matches a key in ``_FAMILY_PARSERS`` (the
       authoritative path — GGUF ``general.architecture`` is the
       model author's declared identity).
    2. Substring match against ``_NAME_NEEDLES`` (sorted longest-first
       so ``qwen3.5`` beats ``qwen3``).
    3. ``None`` — caller defaults to running every parser.

    A non-empty ``arch`` that doesn't resolve via either path triggers
    a debug log so future-unknown architectures surface in
    diagnostics rather than silently routing to the all-parsers
    fallback.
    """
    if arch:
        key = arch.strip().lower()
        if key in _FAMILY_PARSERS:
            return key

    if model:
        m = model.strip().lower()
        for needle, family in _NAME_NEEDLES:
            if needle in m:
                return family

    if arch:
        # We had an architecture hint but neither it nor the name
        # resolved — surface for diagnostics. Quiet at debug; a louder
        # warning would spam logs for every plain-llama load.
        log.debug(
            "reasoning_family_unresolved",
            arch=arch,
            model=(model or "")[:80],
        )
    return None


def _resolve_active_parsers(
    family: str | None,
) -> tuple[str, ...]:
    """Map a family key to the tuple of parser format keys to run.

    ``None`` means "no hint — run everything defensively", which matches the
    pre-refactor behavior for callers that don't pass a family.
    """
    if family is None:
        return ("think", "gemma_channel", "gemma4_channel")
    return _FAMILY_PARSERS.get(family, ("think", "gemma_channel", "gemma4_channel"))


# ---------------------------------------------------------------------------
# Shared helpers (partial-suffix trimming)
# ---------------------------------------------------------------------------


def _trim_partial_suffix(text: str, partial_tokens: tuple[str, ...]) -> str:
    """Trim a trailing partial control token from *text*.

    Used during streaming so we don't leak half-markers into visible content.
    Keeps up to ``len(longest_token) - 1`` chars buffered for the next chunk.
    """
    if not text:
        return text
    hold = 0
    for token in partial_tokens:
        max_check = min(len(text), len(token) - 1)
        for size in range(max_check, 0, -1):
            if token.startswith(text[-size:]):
                hold = max(hold, size)
                break
    if hold:
        return text[:-hold]
    return text


def _trim_partial_gemma_suffix(text: str) -> str:
    return _trim_partial_suffix(text, _GEMMA_PARTIAL_TOKENS)


def _trim_partial_gemma4_suffix(text: str) -> str:
    return _trim_partial_suffix(text, _GEMMA4_PARTIAL_TOKENS)


# ---------------------------------------------------------------------------
# Gemma 3 / GPT-OSS extractor (symmetric channels) — preserved from earlier
# ---------------------------------------------------------------------------


def _extract_gemma_channels(
    content: str, *, partial: bool = False
) -> tuple[str, str, bool]:
    """Extract Gemma/GPT-OSS analysis channels from content.

    Returns ``(clean_content, thinking_text, matched)`` where ``matched``
    indicates whether any complete Gemma channel header was found.
    """
    matches = list(_GEMMA_CHANNEL_RE.finditer(content))
    if not matches:
        if partial and "<|" in content:
            return _trim_partial_gemma_suffix(content), "", False
        return content, "", False

    clean_parts: list[str] = []
    thinking_parts: list[str] = []
    cursor = 0

    for index, match in enumerate(matches):
        if match.start() > cursor:
            prefix = content[cursor:match.start()]
            if partial:
                prefix = _trim_partial_gemma_suffix(prefix)
            if prefix:
                clean_parts.append(prefix)

        channel = match.group(1)
        segment_start = match.end()
        segment_end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        segment = content[segment_start:segment_end]

        if channel != "final" and segment.endswith("<|end|>"):
            segment = segment[:-7]
        elif channel != "final":
            segment = segment.replace("<|end|>", "")

        if partial:
            segment = _trim_partial_gemma_suffix(segment)

        if not segment:
            cursor = segment_end
            continue

        if channel == "analysis":
            thinking_parts.append(segment)
        else:
            clean_parts.append(segment)

        cursor = segment_end

    clean = "".join(clean_parts)
    thinking = "".join(thinking_parts)
    return clean, thinking, True


# ---------------------------------------------------------------------------
# Gemma 4 extractor (asymmetric channels) — new in April 2026
# ---------------------------------------------------------------------------


def _extract_gemma4_channels(
    content: str, *, partial: bool = False
) -> tuple[str, str, bool]:
    """Extract Gemma 4 ``<|channel>thought…<channel|>`` reasoning.

    Gemma 4 emits exactly one reasoning segment per assistant turn (unlike
    Gemma 3 which interleaves analysis/commentary/final). The opener may
    repeat due to llama.cpp PR #21760 edge cases — we treat a second opener
    within the reasoning block as literal content, not a nested channel.

    Args:
        content: Raw buffered text.
        partial: True during streaming, False at end of stream. When True,
            we hold a partial trailing marker back for the next chunk.

    Returns ``(clean_content, thinking_text, matched)`` where ``matched`` is
    True iff at least one opener was seen (closer may still be pending on
    truncated streams).
    """
    opener = _GEMMA4_CHANNEL_OPEN_RE.search(content)
    if not opener:
        if partial:
            # Potential opener being assembled across chunks — hold any
            # matching suffix back so it doesn't leak as content. Catches
            # short prefixes (``<|chan``) that the old-format check missed.
            return _trim_partial_gemma4_suffix(content), "", False
        return content, "", False

    prefix = content[: opener.start()]
    after_open = content[opener.end() :]

    # Closer search. If truncated / missing, everything after the opener is
    # reasoning-in-progress (partial=True) or reasoning-final (partial=False).
    close_idx = after_open.find(_GEMMA4_CHANNEL_CLOSE)
    if close_idx == -1:
        thinking = after_open
        if partial:
            thinking = _trim_partial_gemma4_suffix(thinking)
        return prefix, thinking, True

    thinking = after_open[:close_idx]
    clean_suffix = after_open[close_idx + len(_GEMMA4_CHANNEL_CLOSE) :]
    if partial:
        clean_suffix = _trim_partial_gemma4_suffix(clean_suffix)
    return prefix + clean_suffix, thinking, True


# ---------------------------------------------------------------------------
# Post-hoc normalization (non-streaming)
# ---------------------------------------------------------------------------


def normalize_thinking(
    content: str,
    thinking_field: str | None = None,
    *,
    family: str | None = None,
    model: str | None = None,
    thinking_enabled: bool | None = None,
    preserve_thinking: bool = False,
    salvage_empty_content: bool = False,
) -> tuple[str, str]:
    """Extract thinking content from a complete message.

    Args:
        content: The raw message content.
        thinking_field: Native side-channel value (e.g. Ollama ``thinking``,
            OpenAI ``reasoning_content``). Takes priority when present.
        family: Optional GGUF ``general.architecture`` value — lets the
            function skip format parsers that can't match.
        model: Optional model id/path — used as a fallback signal when
            ``family`` is not provided.
        thinking_enabled: Per-request flag from the UI's thinking toggle.
            When ``False``, suppresses the asymmetric-closer treatment for
            GLM-style families (their template puts ``</think>`` in the
            prompt prefix when thinking is off, so the response is plain
            content with no reasoning to extract). ``None`` means "treat as
            on" — matches the behavior before the per-request flag existed.
        preserve_thinking: Per-request flag for Qwen 3.6's
            ``preserve_thinking`` chat-template kwarg. When True the chat
            template doesn't inject the leading ``<think>\\n`` prefix on
            the current turn (prior reasoning is preserved in history
            instead), so the response no longer starts INSIDE a think
            block. Disables the asymmetric-closer assumption to prevent
            the entire visible answer from routing into reasoning.
        salvage_empty_content: When True, if the extracted content is
            empty but reasoning is non-empty, promote reasoning → content
            so the user sees an answer rather than a mute "Thought for
            Ns" bubble. Handles the asymmetric-closer flake (GLM-5.2
            through NVIDIA NIM, GLM-4.7-Flash) where the model routes
            100% of its output into reasoning without ever emitting
            visible content. Opt-in because it inverts the "no closer
            arrived → route to reasoning" contract used by
            ``_STARTS_THINKING_FAMILIES``.

    Returns ``(clean_content, thinking_text)``.
    """
    if family is None:
        family = detect_reasoning_family(model=model, arch=None)
    parsers = _resolve_active_parsers(family)

    inline_blocks: list[str] = []
    clean = content

    if "think" in parsers:
        # Pick the right regex + delimiter strings for the family. Magistral
        # uses [THINK]/[/THINK]; everything else uses <think>/</think>.
        if family in _MAGISTRAL_FAMILIES:
            think_pattern = _MAGISTRAL_THINK_PATTERN
            open_tag = "[THINK]"
            close_tag = "[/THINK]"
        else:
            think_pattern = _THINK_PATTERN
            open_tag = "<think>"
            close_tag = "</think>"

        # Asymmetric handling for families like GLM whose chat template puts
        # the opener in the prompt prefix. The response starts inside a think
        # block; only the closer arrives in the visible stream. Treat
        # everything before the first closer as reasoning. If the closer
        # never appears, the entire response is reasoning (visible content
        # empty) — same as Ollama's GLM47Parser. Skip when the user has
        # thinking off (template puts the closer in the prompt instead →
        # response is plain content with no reasoning to carve out).
        # Magistral is symmetric and never enters this branch.
        #
        # Also skip when the caller already supplied a native reasoning
        # side-channel (``thinking_field``): the cloud DeepSeek API and
        # llama-server's ``--reasoning-format deepseek`` both deliver
        # ``content`` already cleaned of CoT, with reasoning in a separate
        # field. Applying the asymmetric-closer transform on already-clean
        # content would route the entire visible answer into reasoning,
        # since no ``</think>`` ever appears in the cleaned text.
        starts_thinking = (
            family in _STARTS_THINKING_FAMILIES
            and thinking_enabled is not False
            and not thinking_field
            and not preserve_thinking
        )
        # Asymmetric-closer handling. We treat a response as having "started
        # inside a think block" in two situations:
        #   1. Known starts-thinking family — the chat template guarantees the
        #      opener is in the prompt prefix, so route to reasoning even if
        #      no closer arrives (truncated mid-thought).
        #   2. Data-driven ORPHAN CLOSER — a ``</think>`` appears with NO
        #      ``<think>`` anywhere. The opener must have lived in the prompt
        #      prefix, so this is an asymmetric model that slipped past family
        #      detection. This happens for custom-named finetunes whose GGUF
        #      arch resolves correctly (e.g. ``qwen35``) but whose model NAME
        #      matches no needle (e.g. "Qwythos-9B-Claude-Mythos") — and for
        #      any backend that doesn't plumb the arch hint through. We only
        #      act when a closer is actually present, so a plain model that
        #      never emits ``</think>`` is unaffected.
        # Both honor the same request-flag gating (thinking on, no native
        # side-channel, not preserve_thinking).
        eligible = (
            thinking_enabled is not False
            and not thinking_field
            and not preserve_thinking
        )
        if eligible and open_tag not in clean:
            close_idx = clean.find(close_tag)
            if close_idx != -1:
                inline_blocks.append(clean[:close_idx])
                clean = clean[close_idx + len(close_tag):]
            elif starts_thinking:
                # Known asymmetric family, no closer — model never finished
                # thinking. Route everything to reasoning rather than leaking.
                inline_blocks.append(clean)
                clean = ""

        # Standard symmetric extraction for any remaining open/close pairs
        # (covers the case where the model also emits opening tags).
        inline_blocks.extend(think_pattern.findall(clean))
        clean = think_pattern.sub("", clean)

    gemma_thinking = ""
    if "gemma_channel" in parsers:
        clean, gemma_thinking, _ = _extract_gemma_channels(clean, partial=False)

    gemma4_thinking = ""
    if "gemma4_channel" in parsers:
        clean, gemma4_thinking, _ = _extract_gemma4_channels(clean, partial=False)

    clean = clean.strip()

    if thinking_field:
        parts = [thinking_field]
        if inline_blocks:
            parts.extend(inline_blocks)
        if gemma_thinking:
            parts.append(gemma_thinking)
        if gemma4_thinking:
            parts.append(gemma4_thinking)
        thinking_out = "\n".join(parts)
        if salvage_empty_content and not clean and thinking_out:
            return thinking_out, ""
        return clean, thinking_out

    parts: list[str] = []
    if gemma4_thinking:
        parts.append(gemma4_thinking)
    if gemma_thinking:
        parts.append(gemma_thinking)
    if inline_blocks:
        parts.extend(inline_blocks)
    if parts:
        thinking_out = "\n".join(parts)
        if salvage_empty_content and not clean and thinking_out:
            return thinking_out, ""
        return clean, thinking_out

    return clean, ""


# ---------------------------------------------------------------------------
# Streaming buffer
# ---------------------------------------------------------------------------


class ThinkingStreamBuffer:
    """Streaming buffer that separates reasoning from content deltas.

    Runs the set of parsers appropriate for the model family in parallel.
    Multiple formats can coexist (some models emit both ``<think>`` and
    Gemma-style channels), so we try each relevant parser per chunk and
    emit the union.

    Handles tags split across deltas via:

    * A character-level state machine for ``<think>...</think>``.
    * Regex re-parsing of accumulated raw text for Gemma / Gemma 4
      channels, emitting only the unseen suffix each call. Partial trailing
      markers are held back until the next chunk completes them.

    No-hint construction (``ThinkingStreamBuffer()``) is backward compatible:
    all parsers run. Pass ``family=`` or ``model=`` to scope parsing for a
    known family (skips false-positive checks, slightly faster).

    ``local_engine`` (default ``True``) gates the asymmetric "starts inside a
    think block" assumption used by GLM-4.x / DeepSeek-V4 / Qwen3 / MiniMax /
    EXAONE (see ``_STARTS_THINKING_FAMILIES``). That assumption is only valid
    for a local llama-server whose ``--jinja`` template injects the bare
    opener into the prompt prefix. Cloud OpenAI-compat backends must pass
    ``local_engine=is_local_engine_url(base_url)`` or a plain-content reply
    from a matched model is silently routed entirely into reasoning.

    ``salvage_empty_content`` (default ``False``) enables the end-of-stream
    safety net for the asymmetric-closer flake: when a model routes the
    entire response into reasoning without ever emitting visible content
    (GLM-5.2 through NVIDIA NIM, GLM-4.7-Flash, DeepSeek V3.2 sometimes),
    ``salvage()`` returns the accumulated reasoning as content so the user
    sees the answer rather than a mute "Thought for 7s" bubble. Opt-in
    because it inverts the "no closer arrived → route to reasoning"
    contract that ``_STARTS_THINKING_FAMILIES`` established.
    """

    def __init__(
        self,
        *,
        family: str | None = None,
        model: str | None = None,
        thinking_enabled: bool | None = None,
        preserve_thinking: bool = False,
        local_engine: bool = True,
        salvage_empty_content: bool = False,
    ) -> None:
        if family is None:
            family = detect_reasoning_family(model=model, arch=None)
        self._active = _resolve_active_parsers(family)
        self._family = family

        # Think-marker delimiters. Most families use `<think>...</think>`,
        # but Magistral uses `[THINK]...[/THINK]` special tokens. The
        # streaming parser uses the same overlap-and-trim shape for both
        # — only the delimiter strings change.
        if family in _MAGISTRAL_FAMILIES:
            self._think_open = "[THINK]"
            self._think_close = "[/THINK]"
        else:
            self._think_open = "<think>"
            self._think_close = "</think>"

        # For families whose chat template puts the opener in the prompt
        # prefix (GLM-4.x, DeepSeek V3.2/V4, MiniMax M2.x, EXAONE 4.x — see
        # _STARTS_THINKING_FAMILIES), the response stream starts INSIDE a
        # thinking block. Initialize the state machine accordingly so we
        # route content to thinking until the closer arrives. Mirrors
        # Ollama's GLM47Parser.Init in model/parsers/glm47.go.
        #
        # Skip when thinking_enabled=False: those templates put the closer
        # in the prompt prefix instead, so the response is plain content
        # and starting "inside think" would route it all to reasoning.
        #
        # Also skip when preserve_thinking is True (Qwen 3.6 only): that
        # kwarg changes the chat template so the leading ``<think>\n``
        # prefix is NOT injected on the current turn (prior reasoning is
        # carried in history instead). Without that prefix the response
        # is plain content and the asymmetric assumption would route the
        # entire visible answer into reasoning.
        #
        # CRITICAL — the ``local_engine`` gate (cloud OpenAI-compat hosts:
        # NVIDIA NIM, Fireworks, Together, Z.AI, OpenRouter, …). The
        # prompt-prefix opener injection is a property of *our own*
        # llama-server ``--jinja`` template; a cloud host templates
        # server-side and returns either a native ``reasoning_content``
        # side-channel (handled below) or a clean content stream. If such a
        # host returns plain content (no leading ``</think>``, no native
        # reasoning), assuming we start INSIDE a think block routes the
        # entire visible answer into the thinking channel and empties the
        # response — issue #17.
        #
        # EXCEPTION — some cloud/proxy endpoints for these same asymmetric
        # families DO stream the CoT inline in ``content``, terminated by a
        # lone ``</think>`` (opener still in the server-side prompt prefix).
        # For those, refusing to start inside-think leaks the entire
        # reasoning block into visible content before the orphan closer
        # arrives (observed: cloud DeepSeek-V4 dumping ~16 KB of planning
        # monologue ahead of the real answer). The streaming state machine
        # cannot un-emit already-streamed content, so a purely reactive
        # orphan-closer branch (below) can't recover the cross-chunk bulk —
        # we must start inside-think from chunk 1.
        #
        # We reconcile the two by gating cloud start-inside-think on
        # ``salvage_empty_content``: the caller has opted into the
        # end-of-stream safety net, so the #17 clean-content case (all output
        # wrongly routed to reasoning, visible stream empty) is caught by
        # ``salvage()`` and promoted back to content on flush. The native
        # side-channel flip in ``_process_impl`` further disarms us the moment
        # ``reasoning_content`` arrives, covering side-channel cloud endpoints.
        # Callers WITHOUT the salvage net (e.g. the soft-trigger probe) keep
        # the strict local-only behavior.
        self._inside_think = (
            "think" in self._active
            and family in _STARTS_THINKING_FAMILIES
            and thinking_enabled is not False
            and not preserve_thinking
            and (local_engine or salvage_empty_content)
        )
        # Data-driven orphan-closer state for the streaming ``think`` parser.
        # Mirrors the non-streaming ``normalize_thinking`` branch: a lone
        # ``</think>`` with no ``<think>`` ever seen means the opener lived in
        # the prompt prefix (asymmetric family that slipped past family
        # detection — custom-named finetunes whose GGUF arch resolves but whose
        # NAME matches no needle, or backends not plumbing the arch hint). When
        # that first orphan closer arrives we route the text before it to
        # reasoning rather than leaking it. Gated on the same request flags.
        self._orphan_eligible = (
            "think" in self._active
            and thinking_enabled is not False
            and not preserve_thinking
        )
        self._seen_open_tag = False
        self._orphan_consumed = False
        self._tag_buffer = ""
        # Gemma 3 / GPT-OSS accumulator
        self._gemma_active = False
        self._gemma_raw = ""
        self._gemma_clean_emitted = 0
        self._gemma_thinking_emitted = 0
        # Gemma 4 accumulator
        self._gemma4_active = False
        self._gemma4_raw = ""
        self._gemma4_clean_emitted = 0
        self._gemma4_thinking_emitted = 0
        # End-of-stream salvage accounting. See ``salvage()`` docstring.
        self._salvage_enabled = salvage_empty_content
        self._content_emitted_bytes = 0
        self._thinking_accumulator: list[str] = []

    # ----- public API ----------------------------------------------------

    def _account(self, content: str, thinking: str) -> None:
        """Record the byte counts and reasoning text for end-of-stream salvage.

        Called at every public-method return with the emitted delta so
        ``salvage()`` can decide whether the visible content stream stayed
        empty across the entire response — see the class docstring's
        ``salvage_empty_content`` paragraph.
        """
        if content:
            self._content_emitted_bytes += len(content)
        if thinking and self._salvage_enabled:
            self._thinking_accumulator.append(thinking)

    def salvage(self) -> str:
        """Return accumulated reasoning as content when the visible stream
        stayed empty across the whole response. Empty string otherwise.

        Handles the asymmetric-closer flake where a hybrid reasoning model
        routes 100% of its output into ``reasoning_content`` (or into
        ``<think>…`` without ever emitting ``</think>``) and never writes
        visible content. Without salvage the user sees a mute reasoning
        bubble and no answer. Idempotent — subsequent calls return "".

        Only active when the buffer was constructed with
        ``salvage_empty_content=True``.
        """
        if (
            not self._salvage_enabled
            or self._content_emitted_bytes > 0
            or not self._thinking_accumulator
        ):
            return ""
        salvaged = "".join(self._thinking_accumulator)
        self._thinking_accumulator = []
        # Guard against re-salvage on double-flush.
        self._content_emitted_bytes = len(salvaged)
        return salvaged

    def process(
        self, content_delta: str, thinking_delta: str = ""
    ) -> tuple[str, str]:
        """Process a streaming chunk.

        Args:
            content_delta: Text from the backend's content stream.
            thinking_delta: Backend native reasoning delta (passes through).

        Returns ``(clean_content_delta, thinking_delta)``.
        """
        clean, thinking = self._process_impl(content_delta, thinking_delta)
        self._account(clean, thinking)
        return clean, thinking

    def flush(self) -> tuple[str, str]:
        """Flush any pending buffered content at end of stream — see
        ``_flush_impl`` for the parser-specific details."""
        clean, thinking = self._flush_impl()
        self._account(clean, thinking)
        return clean, thinking

    def _process_impl(
        self, content_delta: str, thinking_delta: str = ""
    ) -> tuple[str, str]:
        """Parser-specific streaming logic — public entry is ``process``."""
        result_thinking = thinking_delta

        # If the backend is sending reasoning via the native side-channel
        # (cloud DeepSeek API ``reasoning_content`` chunks, llama-server's
        # ``--reasoning-format deepseek`` extraction, etc.), disable the
        # asymmetric-closer "starts inside think" assumption. The content
        # stream is already clean — routing it into reasoning would empty
        # the visible answer entirely.
        if thinking_delta:
            self._inside_think = False

        if not content_delta:
            return "", result_thinking

        # -- Gemma 4 asymmetric channels (checked before Gemma 3).
        # Two trigger modes depending on whether Gemma 3 is also active:
        #
        # * Gemma 4 only (explicit ``family='gemma4'``): trigger greedily on
        #   any opener/closer prefix (``<|chan``, ``<chan``) since there's
        #   no other parser racing for the same content.
        # * Gemma 4 alongside Gemma 3 (no hint, both active): trigger only
        #   on the unambiguous disambiguator ``<|channel>``. Ambiguous
        #   prefixes go to Gemma 3's buffer; if they later resolve to
        #   Gemma 4, we hand the buffer off.
        if "gemma4_channel" in self._active:
            pooled = (
                self._gemma_raw + content_delta
                if self._gemma_active
                else content_delta
            )
            gemma3_active = "gemma_channel" in self._active
            if gemma3_active:
                trigger = self._gemma4_active or "<|channel>" in pooled
            else:
                # No parser races; be greedy on any plausible prefix.
                trigger = (
                    self._gemma4_active
                    or "<|chan" in pooled
                    or "<chan" in pooled
                )
            if trigger:
                if self._gemma_active and not self._gemma4_active:
                    # Handoff from the Gemma 3 accumulator (which was
                    # holding the ambiguous ``<|chan…`` prefix before we
                    # could tell which format was emerging).
                    carry = self._gemma_raw
                    self._gemma_active = False
                    self._gemma_raw = ""
                    self._gemma_clean_emitted = 0
                    self._gemma_thinking_emitted = 0
                    self._gemma4_raw = carry
                    self._gemma4_active = True
                clean_delta, think_delta, consumed = self._process_gemma4(content_delta)
                if consumed:
                    if think_delta:
                        result_thinking = (
                            result_thinking + think_delta if result_thinking else think_delta
                        )
                    return clean_delta, result_thinking

        # -- Gemma 3 / GPT-OSS symmetric channels
        if "gemma_channel" in self._active:
            trigger = self._gemma_active or "<|" in content_delta
            if trigger:
                clean_delta, think_delta, consumed = self._process_gemma(content_delta)
                if consumed:
                    if think_delta:
                        result_thinking = (
                            result_thinking + think_delta if result_thinking else think_delta
                        )
                    return clean_delta, result_thinking

        # -- <think>/</think> character-level state machine
        if "think" in self._active:
            clean_content, inline_thinking = self._process_think(content_delta)
            if inline_thinking:
                result_thinking = (
                    result_thinking + inline_thinking if result_thinking else inline_thinking
                )
            return clean_content, result_thinking

        # No parsers active — pass content through unchanged.
        return content_delta, result_thinking

    def _flush_impl(self) -> tuple[str, str]:
        """Parser-specific end-of-stream flush — public entry is ``flush``.

        Called when the backend closes the stream. Flushes pending partial
        markers and any unfinished reasoning block (truncated streams get
        their in-progress reasoning routed to the thinking channel rather
        than leaked as content).
        """
        # Gemma 4 flush — if opener was seen but closer wasn't, everything
        # after opener is reasoning.
        if self._gemma4_active:
            clean_total, think_total, _ = _extract_gemma4_channels(
                self._gemma4_raw, partial=False
            )
            clean_delta = clean_total[self._gemma4_clean_emitted :]
            think_delta = think_total[self._gemma4_thinking_emitted :]
            self._gemma4_active = False
            self._gemma4_raw = ""
            self._gemma4_clean_emitted = 0
            self._gemma4_thinking_emitted = 0
            return clean_delta, think_delta

        # Gemma 3 flush
        if self._gemma_active:
            clean_total, think_total, _ = _extract_gemma_channels(
                self._gemma_raw, partial=False
            )
            clean_delta = clean_total[self._gemma_clean_emitted :]
            think_delta = think_total[self._gemma_thinking_emitted :]
            self._gemma_active = False
            self._gemma_raw = ""
            self._gemma_clean_emitted = 0
            self._gemma_thinking_emitted = 0
            return clean_delta, think_delta

        # <think> flush — any text still in tag_buffer. If we were inside a
        # <think> block, route to thinking; otherwise content. Note: for
        # ``_STARTS_THINKING_FAMILIES`` a truncated stream legitimately
        # ends without ``</think>`` (model got cut off mid-reasoning), so
        # routing the tail to thinking is correct. The ``preserve_thinking``
        # bug — where the leading opener prefix is missing and the whole
        # response routes to thinking — is prevented at init via the
        # ``not preserve_thinking`` guard on ``_inside_think``, not here.
        if not self._tag_buffer:
            return "", ""
        text = self._tag_buffer
        self._tag_buffer = ""
        if self._inside_think:
            return "", text
        return text, ""

    # ----- per-format helpers -------------------------------------------

    def _process_gemma4(self, content_delta: str) -> tuple[str, str, bool]:
        """Gemma 4 streaming parse — returns (clean, think, consumed)."""
        self._gemma4_active = True
        self._gemma4_raw += content_delta
        clean_total, think_total, matched = _extract_gemma4_channels(
            self._gemma4_raw, partial=True
        )

        # Engaged if either a full opener matched or the extractor held
        # something back as a partial marker (which manifests as
        # len(clean_total) + len(think_total) < len(raw)).
        held_back = (len(clean_total) + len(think_total)) < len(self._gemma4_raw)
        if matched or held_back:
            clean_delta = clean_total[self._gemma4_clean_emitted :]
            think_delta = think_total[self._gemma4_thinking_emitted :]
            self._gemma4_clean_emitted = len(clean_total)
            self._gemma4_thinking_emitted = len(think_total)
            return clean_delta, think_delta, True

        # False alarm — trigger fired but neither a complete opener nor a
        # valid partial was assembled. Release accumulated text as clean
        # content so nothing is lost.
        released = self._gemma4_raw
        self._gemma4_active = False
        self._gemma4_raw = ""
        self._gemma4_clean_emitted = 0
        self._gemma4_thinking_emitted = 0
        return released, "", True

    def _process_gemma(self, content_delta: str) -> tuple[str, str, bool]:
        """Gemma 3 / GPT-OSS streaming parse — returns (clean, think, consumed)."""
        self._gemma_active = True
        self._gemma_raw += content_delta
        clean_total, think_total, matched = _extract_gemma_channels(
            self._gemma_raw, partial=True
        )

        if matched or "<|" in self._gemma_raw:
            clean_delta = clean_total[self._gemma_clean_emitted :]
            think_delta = think_total[self._gemma_thinking_emitted :]
            self._gemma_clean_emitted = len(clean_total)
            self._gemma_thinking_emitted = len(think_total)
            return clean_delta, think_delta, True

        released = self._gemma_raw
        self._gemma_active = False
        self._gemma_raw = ""
        self._gemma_clean_emitted = 0
        self._gemma_thinking_emitted = 0
        return released, "", True

    def _process_think(self, content_delta: str) -> tuple[str, str]:
        """Streaming parse for symmetric think markers. Returns (clean, think).

        Marker strings are family-scoped via ``self._think_open`` /
        ``self._think_close`` (set in ``__init__``). Default markers are
        ``<think>`` / ``</think>``; Magistral uses ``[THINK]`` /
        ``[/THINK]``.

        Algorithm: append the delta to the persistent ``_tag_buffer``,
        extract every complete tag in order, then hold back only the
        trailing suffix that could still be the prefix of a partial
        tag — everything else emits now. This guarantees ``_tag_buffer``
        stays bounded by ``len(longest_marker) - 1`` regardless of input,
        so degenerate sequences like ``<<<<<<<<<<<<<<x`` no longer
        accumulate without bound.

        Mirrors the bound-and-trim pattern Ollama uses in
        ``model/parsers/glm46.go::eat()`` + ``parsers.go::overlap()``.
        Replaces an earlier per-character state machine that had a
        latent unbounded-growth window when ``trigger`` lived in the
        buffer past ``max_buf`` chars.
        """
        if not content_delta and not self._tag_buffer:
            return "", ""

        self._tag_buffer += content_delta

        open_tag = self._think_open
        close_tag = self._think_close
        open_len = len(open_tag)
        close_len = len(close_tag)

        clean_parts: list[str] = []
        think_parts: list[str] = []

        # Drain every complete tag in order. ``find`` is O(n) but the
        # buffer is bounded; in practice we exit after at most a few
        # iterations per call.
        while self._tag_buffer:
            open_idx = self._tag_buffer.find(open_tag)
            close_idx = self._tag_buffer.find(close_tag)
            if open_idx < 0 and close_idx < 0:
                break

            # Pick whichever complete tag appears first in the buffer.
            if open_idx < 0:
                tag_idx, tag_len, will_be_inside = close_idx, close_len, False
            elif close_idx < 0 or open_idx < close_idx:
                tag_idx, tag_len, will_be_inside = open_idx, open_len, True
            else:
                tag_idx, tag_len, will_be_inside = close_idx, close_len, False

            is_close = not will_be_inside

            # Data-driven orphan closer: a ``</think>`` reached while NOT inside
            # a think block, with no ``<think>`` ever seen, means the opener was
            # in the prompt prefix — the text before it was reasoning. Route it
            # to thinking instead of leaking it. Only the FIRST such closer
            # triggers (``_orphan_consumed``), matching the non-streaming path;
            # a later stray ``</think>`` in genuine content stays content. This
            # is best-effort within the current buffer — for a KNOWN asymmetric
            # family ``_inside_think`` is already True from init, so the whole
            # block routes correctly regardless of chunk boundaries.
            route_before_to_think = self._inside_think
            if (
                is_close
                and not self._inside_think
                and not self._seen_open_tag
                and self._orphan_eligible
                and not self._orphan_consumed
            ):
                route_before_to_think = True
                self._orphan_consumed = True

            before = self._tag_buffer[:tag_idx]
            if before:
                (think_parts if route_before_to_think else clean_parts).append(before)

            if not is_close:
                self._seen_open_tag = True
            self._inside_think = will_be_inside
            self._tag_buffer = self._tag_buffer[tag_idx + tag_len:]

        # No more complete tags. Keep only the trailing suffix that
        # could still complete a marker; emit the rest now so the buffer
        # stays bounded by ``len(longest_marker) - 1``.
        if self._tag_buffer:
            safe_emit = _trim_partial_suffix(
                self._tag_buffer, (open_tag, close_tag)
            )
            if safe_emit:
                (think_parts if self._inside_think else clean_parts).append(safe_emit)
                self._tag_buffer = self._tag_buffer[len(safe_emit):]

        return "".join(clean_parts), "".join(think_parts)
