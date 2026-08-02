"""Context-adaptive prompt budgets for the companion.

The companion's prompt budgets (tool-roster char budget, chat transcript
window, over-budget log ceiling) were hardcoded for 4-8k-context models.
A user running a 32k / 131k / 262k model was silently rationed to the same
tiny budget — capability and memory left on the table. See
``docs/superpowers/specs/2026-07-16-companion-prompt-budget-scaling-design.md``.

Principle: budgets DEFAULT to a fraction of the loaded model's actual
context (minus a reserve), and fall back to the legacy fixed values when
``companion_prompt_budget_auto`` is off OR the context window is unknown
(``0``). The fixed fallback is the current behaviour, so this is
zero-regression until a real window is known.

Only the ENFORCED levers scale here — the roster char budget and the chat
transcript window (``prompt_compose.LAYER_CAPS`` is documentary, not a trim
pass). Voice keeps its lean window: voice budget is a prefill-LATENCY bound,
not a context bound, and scaling it needs TTFB measurement first (phase C).

Mirrors the reserve math already proven in ``coder/context_tokens.py``.
"""
from __future__ import annotations

import contextlib
from typing import Any

from augmentum.config import settings
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# ── Fixed fallbacks (the legacy values — the "safe option") ──────────
# These mirror the constants still declared in prompt_compose.py / tools.py
# so turning auto OFF, or an unknown window, reproduces today's behaviour.
_FIXED_ROSTER_CHARS = 1200
_FIXED_TRANSCRIPT_TURNS_CHAT = 14
_FIXED_CEILING_CHAT = 3200

# ── Scaling bounds ───────────────────────────────────────────────────
# The roster's real ceiling is "fit the whole tool catalogue" (~55 tools),
# not the context — beyond that, more budget just wastes prefill and dilutes
# selection. ~4000 chars comfortably fits the full roster.
_MAX_ROSTER_CHARS = 4000
# A real conversational memory on a large-context model, bounded so a 262k
# window doesn't stuff hundreds of turns (prefill + relevance both degrade).
_MAX_TRANSCRIPT_TURNS_CHAT = 60
# Context (usable tokens) at which each lever reaches its max. Modest — most
# modern local models clear 32k, so they get the full roster + rich memory.
_ROSTER_FULL_AT_USABLE = 30_000
_TRANSCRIPT_FULL_AT_USABLE = 60_000


def _reserve_pct() -> float:
    """Live-tunable reserve fraction (output/tool/reasoning headroom).

    Reads ``settings.companion_prompt_context_reserve_pct``; falls back to
    0.10. Bounded to (0.02, 0.40) so a misconfigured setting can't starve
    the model or shrink the usable window into uselessness. Same shape as
    coder's ``coder_context_reserve_pct``.
    """
    default = 0.10
    try:
        raw = getattr(settings, "companion_prompt_context_reserve_pct", default)
        pct = float(raw if raw is not None else default)
    except Exception:  # noqa: BLE001 — never break prompt assembly
        return default
    return max(0.02, min(0.40, pct))


def _usable_tokens(context_length: int) -> int:
    """Model window minus reserve headroom. 0 if the window is unknown."""
    try:
        ctx = int(context_length or 0)
    except (TypeError, ValueError):
        return 0
    if ctx <= 0:
        return 0
    reserve = max(2_048, int(ctx * _reserve_pct()))
    reserve = min(reserve, 32_768, max(1_024, ctx // 3))
    return max(0, ctx - reserve)


def _auto_on() -> bool:
    try:
        return bool(getattr(settings, "companion_prompt_budget_auto", True))
    except Exception:  # noqa: BLE001
        return True


def _lerp(usable: int, full_at: int, floor: int, ceil: int) -> int:
    """Linear ramp from ``floor`` (at 0 usable) to ``ceil`` (at ``full_at``)."""
    if usable >= full_at:
        return ceil
    frac = max(0.0, usable / full_at)
    return int(floor + frac * (ceil - floor))


def derive_roster_char_budget(context_length: int) -> int:
    """Tool-roster char budget, scaled from the loaded window.

    Fixed fallback (1200) when auto is off or the window is unknown. Ramps
    to the full-catalogue budget (~4000) by ~30k usable tokens, so any 32k+
    model gets the whole roster (no more youtube/media.play deferral).
    """
    usable = _usable_tokens(context_length)
    if not _auto_on() or usable <= 0:
        return _FIXED_ROSTER_CHARS
    return _lerp(usable, _ROSTER_FULL_AT_USABLE, _FIXED_ROSTER_CHARS, _MAX_ROSTER_CHARS)


def derive_transcript_turns_chat(context_length: int) -> int:
    """Chat transcript window (turns), scaled from the loaded window.

    Fixed fallback (14) when auto is off or window unknown. Ramps to 60 by
    ~60k usable tokens — real continuity instead of goldfish memory. VOICE is
    deliberately NOT scaled here (prefill latency; phase C).
    """
    usable = _usable_tokens(context_length)
    if not _auto_on() or usable <= 0:
        return _FIXED_TRANSCRIPT_TURNS_CHAT
    return _lerp(
        usable, _TRANSCRIPT_FULL_AT_USABLE,
        _FIXED_TRANSCRIPT_TURNS_CHAT, _MAX_TRANSCRIPT_TURNS_CHAT,
    )


def derive_ceiling_chat(context_length: int) -> int:
    """Over-budget LOG ceiling for chat, scaled so growing the transcript
    doesn't spam ``becca_prompt_over_budget``. This is a log threshold, not a
    trim (``_enforce_total_budget`` is log-only), so it's advisory — but keep
    it honest against the usable window.
    """
    usable = _usable_tokens(context_length)
    if not _auto_on() or usable <= 0:
        return _FIXED_CEILING_CHAT
    # Chat may use most of the usable window; keep the fixed value as a floor.
    return max(_FIXED_CEILING_CHAT, usable)


def cached_context_length(runtime: Any, model_name: str = "") -> int:
    """Synchronous read of the per-model context length cached by a prior
    :func:`resolve_context_length` call this turn. Returns 0 when not yet
    resolved (→ fixed fallback). For sync call sites (e.g. the voice
    first-hop tool attach) that run after ``_gather_ctx`` warmed the cache.
    """
    cache = getattr(runtime, "_companion_ctx_len_cache", None)
    if not isinstance(cache, dict) or not cache:
        return 0
    if model_name and model_name in cache:
        return cache[model_name]
    # Single-model common case: return the only entry.
    if len(cache) == 1:
        return next(iter(cache.values()))
    return 0


async def resolve_context_length(runtime: Any) -> int:
    """Loaded primary-model context window (tokens); 0 if unknown.

    Resolves via the same tier facade Becca speaks through
    (``tiers.primary`` → ``backend.get_context_length``) and caches per
    model name on the runtime so we probe once per model, not per turn.
    Never raises — returns 0 (→ fixed fallback) on any failure.
    """
    try:
        from augmentum.companion_runtime import tiers

        backend, model_name = await tiers.primary(runtime)
    except Exception as exc:  # noqa: BLE001
        log.debug("companion_ctx_len_resolve_failed", error=str(exc))
        return 0

    cache = getattr(runtime, "_companion_ctx_len_cache", None)
    if cache is None:
        cache = {}
        with contextlib.suppress(AttributeError, TypeError):
            runtime._companion_ctx_len_cache = cache  # noqa: SLF001
    if isinstance(cache, dict) and model_name in cache:
        return cache[model_name]

    try:
        ctx_len = int(await backend.get_context_length(model_name) or 0)
    except Exception as exc:  # noqa: BLE001
        log.debug("companion_ctx_len_probe_failed", model=model_name, error=str(exc))
        ctx_len = 0
    if isinstance(cache, dict):
        cache[model_name] = ctx_len
    return ctx_len
