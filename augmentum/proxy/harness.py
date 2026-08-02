"""Harness briefing — make any external IDE coding agent project-aware.

When a user drives Augmentum's ``/v1`` proxy from a terminal/IDE coding agent
(OpenCode, Claude Code, Cursor, Aider, Cline, Continue, Zed, …), this layer
injects, on every turn, the user's **isolated, accumulated coding memory** plus
their **professional working conventions** — so the agent benefits from
Augmentum's memory regardless of which harness the user runs, and completes the
task to the user's standards.

This is the READ/inject half (spec P1). It is:
  * **Harness-agnostic** — one seam serves every tool (header or User-Agent).
  * **Scope-isolated** — reads only the ``harness`` memory scope; the
    companion/chat/narrative scopes are physically unreachable here (and vice
    versa). Matches the "isolate, don't pool" decision.
  * **Budgeted** — a hard token cap so the harness's own context is never blown.
  * **Fail-open** — any error logs a warning and forwards the request unchanged;
    enrichment never breaks a coding turn.
  * **Tool-safe** — injects CONTEXT only, never tools. The harness's own
    read/write/bash loop is untouched.

See ``docs/superpowers/specs/2026-06-28-harness-memory-design.md``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.memory.models import MemoryType
from augmentum.training.capture import capture_harness_observation
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request

    from augmentum.models.base import InternalChatRequest

log = get_logger(__name__)

# Global harness scope: holds only the seeded default conventions (generic,
# harness- and project-agnostic). Isolated from companion/chat/etc.
HARNESS_SCOPE = "harness"
# Learned/harvested memories live in per-project sub-scopes so conventions
# learned in one project never bleed into another:
#   harness:<harness>:<project>   — project identity from X-Augmentum-Project
#   harness:default               — shared fallback when no project header is
#                                   sent (also where pre-scoping flat memories
#                                   were migrated — see migration 315)
# All harness:* sub-scopes are isolated from the general pool via the scope
# prefix handling in memory/store.py (is_isolated_scope/_isolation_sql).
_HARNESS_HEADER = "x-augmentum-harness"
_PROJECT_HEADER = "x-augmentum-project"
_SCOPE_TOKEN_RE = re.compile(r"[^a-z0-9._-]+")


def detect_project(request: Request) -> str:
    """Sanitized project id from ``X-Augmentum-Project`` (the claude-aug
    wrappers send the cwd basename), or "" when absent."""
    raw = (request.headers.get(_PROJECT_HEADER) or "").strip().lower()
    if not raw:
        return ""
    return _SCOPE_TOKEN_RE.sub("-", raw)[:64].strip("-.")


def harness_memory_scope(harness: str, project: str) -> str:
    """The memory scope learned/harvested memories read and write.

    With a project id: ``harness:<harness>:<project>`` (per-harness,
    per-project isolation). Without one: the shared ``harness:default``
    pool — graceful degradation for clients that don't send the header,
    and where legacy flat-scope memories were migrated.
    """
    project = _SCOPE_TOKEN_RE.sub("-", (project or "").lower())[:64].strip("-.")
    if not project:
        return f"{HARNESS_SCOPE}:default"
    h = _SCOPE_TOKEN_RE.sub("-", (harness or "default").lower())[:32].strip("-.") or "default"
    return f"{HARNESS_SCOPE}:{h}:{project}"

# Known IDE coding-agent User-Agents → canonical harness id. Extend freely to
# support a new tool; detection is also satisfied by the explicit
# ``X-Augmentum-Harness`` header (which we set in our own configs).
_HARNESS_UA: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"opencode", re.I), "opencode"),
    # pi (pi.dev / earendil-works) — our own extension sets the explicit
    # X-Augmentum-Harness: pi header (which wins below), so the UA match is
    # belt-and-suspenders; "earendil"/"pi-coding-agent" are distinctive enough
    # not to false-positive. See pi/extensions/augmentum.ts.
    (re.compile(r"pi-coding-agent|earendil", re.I), "pi"),
    (re.compile(r"claude[-_ ]?code|claude[-_ ]?cli", re.I), "claude_code"),
    (re.compile(r"cursor", re.I), "cursor"),
    (re.compile(r"aider", re.I), "aider"),
    (re.compile(r"\bcline\b|continue\.?dev|continuedev", re.I), "continue"),
    (re.compile(r"windsurf|codeium", re.I), "windsurf"),
    (re.compile(r"\bzed\b", re.I), "zed"),
    (re.compile(r"\bgoose\b", re.I), "goose"),
    (re.compile(r"\bcrush\b|charmbracelet", re.I), "crush"),
    (re.compile(r"\bcodex\b", re.I), "codex"),
]

_BLOCK_HEADER = (
    "[Augmentum memory — facts and conventions distilled from this developer's "
    "earlier statements across sessions, retrieved as background for the current "
    "request. Treat as the developer's own prior guidance; apply what's relevant.]"
)
_BLOCK_FOOTER = "[end Augmentum memory]"

# Day-1 professional working conventions, seeded once per user into the harness
# PROCEDURAL scope (editable/removable afterwards via the memory store). Generic,
# universally-good coding discipline — NOT project-specific (the project's own
# CLAUDE.md/AGENTS.md already rides in the harness's system prompt). Gives every
# harness a baseline "work professionally" signal even before anything is
# learned. Disable via ``harness_seed_defaults``.
_DEFAULT_CONVENTIONS: tuple[str, ...] = (
    "Before claiming a task is done, run the project's tests/build and report "
    "the real result — never assume it passes.",
    "Fix the root cause / general class of a bug, not just the exact reported "
    "instance.",
    "Read neighboring code first; match the surrounding style, naming, and "
    "patterns rather than imposing your own.",
    "Prefer explicit, readable solutions over clever ones; keep the change "
    "surface small and reversible.",
    "State what a change breaks, orphans, or disconnects — not only what it "
    "adds.",
)


def detect_harness(request: Request) -> str:
    """Canonical harness id ("opencode"/"claude_code"/…) when the caller is an
    external IDE coding agent, else "".

    Header (``X-Augmentum-Harness``) wins — we set it in our own configs and it
    needs no User-Agent guessing. Otherwise match the User-Agent against the
    known-harness table. The in-app web UI (session cookie + ``X-Augmentum-
    Session`` header, browser UA) and internal callers never match.
    """
    explicit = (request.headers.get(_HARNESS_HEADER) or "").strip().lower()
    if explicit:
        return explicit
    ua = request.headers.get("user-agent", "") or ""
    for pattern, name in _HARNESS_UA:
        if pattern.search(ua):
            return name
    return ""


def _budget_lines(lines: list[str], budget_tokens: int) -> list[str]:
    """Take whole lines until the (approx ¼-char-per-token) budget is spent."""
    out: list[str] = []
    used = 0
    for raw in lines:
        line = " ".join((raw or "").split())
        if not line:
            continue
        cost = max(1, len(line) // 4)
        if used + cost > budget_tokens:
            break
        out.append(line)
        used += cost
    return out


async def ensure_harness_seed(app_state, user_id: str) -> None:
    """Seed the default professional conventions into the user's harness scope
    once (idempotent — store() dedups). Best-effort; never raises into a turn."""
    if not user_id or not getattr(settings, "harness_seed_defaults", True):
        return
    store = getattr(app_state, "memory_store", None)
    if store is None:
        return
    try:
        existing = await store.recall(
            query="working conventions discipline standards",
            user_id=user_id,
            limit=1,
            memory_types=[MemoryType.PROCEDURAL],
            scope=HARNESS_SCOPE, scope_strict=True,
        )
        if existing:
            return  # already seeded (or user-authored conventions exist)
        for conv in _DEFAULT_CONVENTIONS:
            await store.store(
                conv,
                MemoryType.PROCEDURAL,
                user_id=user_id,
                importance=0.7,
                source_type="system",
                scope=HARNESS_SCOPE,
            )
        log.info("harness_conventions_seeded", user_id=user_id, count=len(_DEFAULT_CONVENTIONS))
    except Exception:
        log.warning("harness_seed_failed", user_id=user_id, exc_info=True)


async def _build_briefing(
    app_state, *, user_id: str, query: str, proc_budget: int, fact_budget: int,
    project_scope: str, include_conventions: bool = True,
) -> str:
    store = getattr(app_state, "memory_store", None)
    if store is None or not user_id:
        return ""
    parts: list[str] = []

    # 1. Procedural conventions: global seeds (scope "harness") + conventions
    #    learned in THIS project's scope. ``include_conventions=False`` is the
    #    injection-frequency tamer (harness_conventions_mode="first_turn"):
    #    local models over-index on a block repeated every turn. recall()
    #    excludes the PROVISIONAL tier, so unvalidated captures never surface.
    if include_conventions and getattr(settings, "harness_enrich_procedural", True):
        conv_query = "working conventions standards discipline how we do things"
        procedural = []
        for conv_scope in (HARNESS_SCOPE, project_scope):
            try:
                procedural.extend(await store.recall(
                    query=conv_query,
                    user_id=user_id, limit=8,
                    memory_types=[MemoryType.PROCEDURAL],
                    scope=conv_scope, scope_strict=True,
                ))
            except Exception:
                log.warning("harness_procedural_recall_failed",
                            scope=conv_scope, exc_info=True)
        seen: set[str] = set()
        contents = []
        for m in procedural:
            key = " ".join((m.content or "").split()).lower()
            if key and key not in seen:
                seen.add(key)
                contents.append(m.content)
        lines = _budget_lines(contents, proc_budget)
        if lines:
            parts.append(
                "Working conventions (apply throughout):\n"
                + "\n".join(f"- {ln}" for ln in lines)
            )

    # 2. Relevant accumulated memory — similarity-gated to the current ask,
    #    read from the PROJECT scope only (never other projects/harnesses).
    if getattr(settings, "harness_enrich_memory", True) and query:
        try:
            # No min_score floor: recall's score is a PRODUCT of several [0,1]
            # factors (rrf × strength × importance × tier × source × surprise),
            # so even a strong match lands well below ~0.01 — a floor here
            # silently drops legitimate hits. Rely on recall's own relevance
            # ranking + the limit + the token budget to bound what's injected.
            facts = await store.recall(
                query=query, user_id=user_id, limit=8,
                memory_types=[
                    MemoryType.FACT, MemoryType.PREFERENCE, MemoryType.ENTITY,
                    MemoryType.SKILL,
                ],
                scope=project_scope, scope_strict=True,
            )
        except Exception:
            log.warning("harness_fact_recall_failed", exc_info=True)
            facts = []
        lines = _budget_lines([m.content for m in facts], fact_budget)
        if lines:
            parts.append(
                "Relevant project memory:\n" + "\n".join(f"- {ln}" for ln in lines)
            )

    if not parts:
        return ""
    return f"{_BLOCK_HEADER}\n" + "\n\n".join(parts) + f"\n{_BLOCK_FOOTER}"


_WORKFLOW_HEADER = "[Augmentum saved workflow — a procedure you saved that fits this task]"
_WORKFLOW_FOOTER = "[End saved workflow — adapt it; it's guidance, not a script]"


async def _build_workflow_briefing(
    app_state, *, user_id: str, query: str, harness: str, project: str,
) -> str:
    """Retrieve the single best-matching self-saved workflow (FTS on its
    when_to_use trigger) for the current ask and format it for injection.

    Subtractive by design: only the top match, only when the trigger
    actually matches the query — no "inject everything" echo chamber.
    """
    if not getattr(settings, "harness_workflow_inject_enabled", True):
        return ""
    try:
        from augmentum.tools import workflow_store
    except Exception:
        return ""
    own = harness_memory_scope(harness, project)
    scopes = [own] if own == f"{HARNESS_SCOPE}:default" else [own, f"{HARNESS_SCOPE}:default"]
    rows = await workflow_store.search_workflows(
        app_state, user_id=user_id, scopes=scopes, query=query, limit=1,
    )
    if not rows:
        return ""
    w = rows[0]
    body = (
        f"{w['name']}: {w['when_to_use']}\n{(w.get('steps') or '')[:1200]}"
    )
    return f"{_WORKFLOW_HEADER}\n{body}\n{_WORKFLOW_FOOTER}"


async def inject_harness_context(
    internal_req: InternalChatRequest, app_state, *, user_id: str, harness: str,
    project: str = "",
) -> bool:
    """Prepend a scope-isolated Augmentum-memory block to the current turn.

    Mutates ``internal_req.messages`` (the last user turn — the volatile tail,
    so the stable prefix stays cacheable). Returns True iff anything was
    injected. Fail-open and tool-safe: errors leave the request untouched and
    ``internal_req.tools`` is never modified.
    """
    if not getattr(settings, "harness_enrich_enabled", True):
        return False
    if not user_id or not internal_req.messages:
        return False
    try:
        await ensure_harness_seed(app_state, user_id)

        # Query = the last user message (the current ask).
        last_user_idx = None
        for i in range(len(internal_req.messages) - 1, -1, -1):
            if internal_req.messages[i].role == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            return False
        last = internal_req.messages[last_user_idx]
        query = last.content if isinstance(last.content, str) else str(last.content)

        budget = int(getattr(settings, "harness_inject_token_budget", 800) or 800)
        proc_budget = min(budget // 2, 300)
        fact_budget = max(budget - proc_budget, 0)

        # Injection-frequency tamer: "first_turn" (default) injects the full
        # conventions block only on the FIRST turn of a session (no assistant
        # message in the transcript yet) and facts-only afterwards — repeated
        # every-turn conventions make small local models parrot them.
        # "always" restores the historical per-turn behavior.
        mode = str(getattr(settings, "harness_conventions_mode", "first_turn") or "first_turn")
        first_turn = not any(m.role == "assistant" for m in internal_req.messages)
        include_conventions = mode != "first_turn" or first_turn

        block = await _build_briefing(
            app_state, user_id=user_id, query=query,
            proc_budget=proc_budget, fact_budget=fact_budget,
            project_scope=harness_memory_scope(harness, project),
            include_conventions=include_conventions,
        )

        # Soft procedural memory: surface the model's own best-matching
        # workflow (FTS on when_to_use) so it doesn't re-derive a procedure
        # it already saved — Hermes/AWM-style, at zero per-turn tool cost.
        wf_block = await _build_workflow_briefing(
            app_state, user_id=user_id, query=query, harness=harness, project=project,
        )

        combined = "\n\n".join(b for b in (block, wf_block) if b)
        if not combined:
            return False

        internal_req.messages[last_user_idx] = dataclasses.replace(
            last, content=f"{combined}\n\n{query}",
        )
        log.info(
            "harness_context_injected",
            harness=harness, project=project or "default", user_id=user_id,
            conventions=include_conventions,
            block_chars=len(combined), items=block.count("\n- "),
            workflow=bool(wf_block),
        )
        return True
    except Exception:
        log.warning("harness_briefing_failed", harness=harness, exc_info=True)
        return False


# --- Capture (learn-out) ---------------------------------------------------
# Learn durable knowledge from harness turns into the isolated harness scope.
# Background + fire-and-forget (never blocks the turn), gated by a cheap
# teaching-signal pre-filter, then a CODING-TUNED extraction on Slot C (the
# resident utility model — free, local). A regex extractor mangles coding facts
# (it truncates "deploy via ./start.sh" at the dot); the LLM preserves commands
# and paths and distinguishes durable knowledge from one-off task requests.
# Tier is left to store(): durable/stated rules → EXPLICIT → ACTIVE (injectable
# next turn); incidental → EXTRACTED → PROVISIONAL until corroborated.

_BG_TASKS: set[asyncio.Task] = set()
# One in-flight capture per user — coalesces rapid harness turns so the
# background Slot C extraction can't flood the model shared with voice/classifier.
_CAPTURE_INFLIGHT: set[str] = set()

# Cheap pre-filter: only spend a (free, background, Slot-C) LLM call on turns
# that plausibly TEACH/CORRECT something durable. The LLM is the real gate (it
# returns [] for non-teachings), so this is deliberately GENEROUS — better to
# burn a cheap extraction than miss a correction. Pure action turns ("fix this
# bug", "add a function") still match nothing and are skipped.
_TEACH_SIGNALS = re.compile(
    r"\b("
    r"remember|always|never|don'?t|do not|stop using|instead|actually|"
    r"correct|correction|corrected|deprecat\w*|no longer|not anymore|from now on|"
    r"going forward|by default|for future|keep in mind|note that|important|make sure|"
    r"fyi|heads.?up|prefer|convention|standard|the rule|the way we|"
    r"our (rule|convention|standard|way)|should (always|never|use|be)|"
    r"must (always|never|use|be)|switch(ed)? to|"
    r"we (use|deploy|run|prefer|call|keep|store|don'?t|never|always|now)"
    r")\b",
    re.I,
)

# Strong imperative/correction markers. A message that is only a QUESTION
# ("how do we deploy?") must never write memory — a read must not mutate state.
# We skip capture for interrogatives UNLESS they also carry a strong marker
# (e.g. "remember: ... , right?").
_STRONG_TEACH = re.compile(
    r"\b(remember|always|never|don'?t|do not|correct|correction|deprecat\w*|"
    r"instead|from now on|we now|stop using|use .{1,30} not)\b",
    re.I,
)

# Drop any candidate that looks like it carries a secret — never persist creds.
# Backstop, not a guarantee: broad coverage of common token shapes + key=value
# secrets + scheme://user:pass@ connection strings. The extractor is also told
# not to capture credentials.
_SECRET = re.compile(
    r"("
    r"sk-[A-Za-z0-9_-]{16,}|sk_(live|test)_[A-Za-z0-9]{16,}|"            # OpenAI / Stripe
    r"AKIA[0-9A-Z]{12,}|ASIA[0-9A-Z]{12,}|"                             # AWS access key
    r"AIza[0-9A-Za-z_-]{30,}|"                                          # Google API key
    r"xox[baprs]-[0-9A-Za-z-]{8,}|"                                     # Slack token
    r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"         # GitHub PAT(s)
    r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{6,}|"      # JWT
    r"[a-z][a-z0-9+.\-]*://[^\s:@/]+:[^\s:@/]+@|"                       # scheme://user:pass@
    r"api[_-]?key\s*[=:]\s*\S+|password\s*[=:]\s*\S+|secret\s*[=:]\s*\S+|"
    r"token\s*[=:]\s*[A-Za-z0-9._-]{12,}|bearer\s+[A-Za-z0-9._-]{16,}|"
    r"-----BEGIN"
    r")",
    re.I,
)

_EXTRACTION_SYSTEM = (
    "You extract DURABLE, reusable knowledge from a developer's message to a "
    "coding agent, and reconcile it against existing memories. Capture ONLY "
    "things worth remembering across sessions: stated conventions/rules, "
    "decisions, corrections, and project facts (commands, file locations, "
    "architecture, tooling). Do NOT capture transient task requests, code to "
    "write, or one-off instructions. IGNORE questions — if the message is only "
    "ASKING about or requesting something (not stating a durable "
    "fact/rule/correction), return an empty list. Preserve commands and file "
    "paths VERBATIM. "
    "You are given EXISTING memories as 'id: text'. If a new memory REPLACES or "
    "CONTRADICTS an existing one (a changed command, a reversed decision, a "
    'deprecation), set its "supersedes" to that existing id. Reply with ONLY '
    "compact JSON and nothing else: "
    '{"memories":[{"kind":"convention|fact|preference","text":"...","durable":true,"supersedes":null}]}'
    ". Empty list if nothing durable. durable=true for rules/corrections the "
    "developer clearly wants kept. supersedes is an existing id (integer) or null."
)


def _fire(coro) -> None:
    """Fire-and-forget a coroutine without blocking the turn, holding a strong
    reference so it isn't garbage-collected mid-flight."""
    try:
        task = asyncio.create_task(coro)
    except RuntimeError:  # no running loop (shouldn't happen on the request path)
        return
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)


def _parse_memories(raw: str) -> list[dict]:
    import json

    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except Exception:
        return []
    out: list[dict] = []
    for item in (data.get("memories") or []) if isinstance(data, dict) else []:
        if not isinstance(item, dict):
            continue
        text = " ".join((item.get("text") or "").split())
        if len(text) < 4 or _SECRET.search(text):
            continue
        sup = item.get("supersedes")
        out.append({
            "kind": (item.get("kind") or "fact").strip().lower(),
            "text": text,
            "durable": bool(item.get("durable")),
            "supersedes": sup if isinstance(sup, int) and not isinstance(sup, bool) else None,
        })
    return out[:6]


async def _llm_extract(
    app_state, user_message: str, existing: list[tuple[int, str]] | None = None,
) -> list[dict]:
    """Coding-tuned extraction + reconcile on Slot C (utility role). ``existing``
    is the in-scope memories (int id, text) the new facts may supersede. Returns
    [] on any failure — capture is best-effort."""
    registry = getattr(app_state, "provider_registry", None)
    if registry is None:
        return []
    try:
        backend, model = await registry.resolve_model_for_role(
            "utility",
            override=getattr(settings, "memory_llm_extraction_model", "") or "",
            settings=settings,
        )
    except Exception:
        return []
    if backend is None:
        return []
    from augmentum.models.base import InternalChatRequest, Message

    block = ""
    if existing:
        block = "EXISTING memories:\n" + "\n".join(f"{i}: {t}" for i, t in existing) + "\n\n"
    req = InternalChatRequest(
        model=model,
        messages=[
            Message(role="system", content=_EXTRACTION_SYSTEM),
            Message(role="user", content=f"{block}New developer message:\n{user_message[:3500]}"),
        ],
        max_tokens=400,
        temperature=0.0,
        stream=False,
    )
    try:
        resp = await backend.chat(req)
        return _parse_memories(resp.message.content or "")
    except Exception:
        log.warning("harness_llm_extract_failed", exc_info=True)
        return []


async def _capture(
    app_state, user_id: str, user_message: str, *,
    harness: str = "", model: str = "", session_id: str = "", project: str = "",
) -> None:
    """STAGE durable-knowledge CANDIDATES from a harness turn for later review.

    Observation-only: harness turns NEVER mutate live memory anymore. The cheap
    Slot-C extraction surfaces *what's worth harvesting*; the candidates land in
    the harness-harvest staging feed (folded into training-capture), where a
    deliberate pass filters the good ones into the baseline. The baseline only
    grows when you say so — no auto-accumulation, no read-that-mutates class.
    """
    if not user_id or not (user_message or "").strip():
        return
    if not _TEACH_SIGNALS.search(user_message):
        return  # no teaching signal — skip the LLM call entirely
    if user_message.rstrip().endswith("?") and not _STRONG_TEACH.search(user_message):
        return  # a pure question must never produce a harvest candidate
    try:
        from augmentum.memory.models import MemoryType

        # Read-only pull of the current BASELINE (seed + already-harvested) so a
        # candidate that would CHANGE the baseline can be FLAGGED for review.
        # We never act on it here — no store / no supersede. INTEGER ids given
        # to the extractor are anti-hallucination context only.
        store = getattr(app_state, "memory_store", None)
        project_scope = harness_memory_scope(harness, project)
        existing_mems: list = []
        if store is not None:
            # Baseline = global seeds + THIS project's scope — never another
            # project's memories.
            for base_scope in (HARNESS_SCOPE, project_scope):
                try:
                    existing_mems.extend(await store.recall(
                        query=user_message, user_id=user_id, limit=6,
                        scope=base_scope, scope_strict=True,
                        memory_types=[
                            MemoryType.FACT, MemoryType.PREFERENCE, MemoryType.PROCEDURAL,
                            MemoryType.ENTITY, MemoryType.SKILL,
                        ],
                    ))
                except Exception:
                    log.warning("harness_capture_recall_failed",
                                scope=base_scope, exc_info=True)
        # De-dupe across the two scope pulls by memory id.
        seen_ids: set = set()
        existing_mems = [
            m for m in existing_mems
            if getattr(m, "id", None) not in seen_ids
            and not seen_ids.add(getattr(m, "id", None))
        ]
        id_map = {i: m for i, m in enumerate(existing_mems)}
        existing_pairs = [(i, m.content) for i, m in enumerate(existing_mems)]

        raw = await _llm_extract(app_state, user_message, existing_pairs)
        if not raw:
            return
        candidates: list[dict] = []
        for cand in raw:
            sup = cand.get("supersedes")
            base = id_map.get(sup) if isinstance(sup, int) else None
            candidates.append({
                "kind": cand["kind"],
                "text": cand["text"],
                "durable": cand["durable"],
                # Read-only flag: this candidate would change baseline entry X.
                "supersedes_baseline_id": getattr(base, "id", None) if base else None,
                "supersedes_baseline_text": getattr(base, "content", None) if base else None,
                # Which scope the baseline entry lives in — promotion must
                # supersede within the SAME scope (a seed lives in the global
                # scope; project memories in the project scope).
                "supersedes_baseline_scope": getattr(base, "scope", None) if base else None,
                # Where a promoted candidate is written.
                "target_scope": project_scope,
            })
        if not candidates:
            return
        capture_harness_observation(
            user_id=user_id, session_id=session_id, harness=harness, model=model,
            source_message=user_message, candidates=candidates,
        )
        log.info("harness_harvest_staged", user_id=user_id, count=len(candidates))
    except Exception:
        log.warning("harness_capture_failed", user_id=user_id, exc_info=True)


def schedule_harness_capture(
    internal_req: InternalChatRequest, app_state, *, user_id: str, harness: str = "",
    project: str = "",
) -> None:
    """Fire-and-forget STAGING of harvest candidates from this harness turn.

    Reads the last user message; if it carries a teaching signal, a coding-tuned
    extraction runs on Slot C in the background and STAGES candidates into the
    harness-harvest feed (observation-only — never written to live memory).
    Non-blocking; fail-open. Gated by ``harness_capture_enabled``.
    """
    if not user_id or not getattr(settings, "harness_capture_enabled", True):
        return
    if user_id in _CAPTURE_INFLIGHT:
        return  # a capture for this user is already running — coalesce
    user_message = ""
    for msg in reversed(internal_req.messages):
        if msg.role == "user":
            user_message = msg.content if isinstance(msg.content, str) else str(msg.content)
            break
    if not user_message.strip():
        return
    model = getattr(internal_req, "model", "") or ""
    session_id = getattr(internal_req, "session_id", "") or ""
    _CAPTURE_INFLIGHT.add(user_id)
    _fire(_capture_guarded(
        app_state, user_id, user_message,
        harness=harness, model=model, session_id=session_id, project=project,
    ))


async def _capture_guarded(
    app_state, user_id: str, user_message: str, **kwargs,
) -> None:
    try:
        await _capture(app_state, user_id, user_message, **kwargs)
    finally:
        _CAPTURE_INFLIGHT.discard(user_id)


# --- Harvest → baseline (deliberate, human-gated promotion) ----------------
# The ONLY path that writes harness baseline memory. Staging never mutates the
# baseline; a human reviews the staged candidates and promotes the keepers here.


async def promote_candidate(
    app_state, user_id: str, obs_id: str, candidate_index: int,
) -> dict:
    """Promote ONE staged harvest candidate into the baseline (harness scope).

    A candidate flagged as contradicting the baseline supersedes that entry
    (invalidate-not-delete); otherwise it's stored fresh as EXPLICIT/ACTIVE
    (injectable next turn). Records the verdict in the harvest ledger.
    """
    from augmentum.memory.models import SourceType
    from augmentum.training.capture import get_harness_record, record_harvest_decision

    store = getattr(app_state, "memory_store", None)
    if store is None:
        return {"status": "error", "error": "no memory store"}
    rec = get_harness_record(user_id, obs_id)
    if rec is None:
        return {"status": "error", "error": "record not found"}
    cands = rec.get("candidates") or []
    if not (0 <= candidate_index < len(cands)):
        return {"status": "error", "error": "candidate index out of range"}
    cand = cands[candidate_index]
    text = (cand.get("text") or "").strip()
    if not text:
        return {"status": "error", "error": "empty candidate"}

    kind = (cand.get("kind") or "fact").lower()
    mtype = (
        MemoryType.PROCEDURAL if kind == "convention"
        else MemoryType.PREFERENCE if kind == "preference"
        else MemoryType.FACT
    )
    base_id = cand.get("supersedes_baseline_id")
    # Scope routing: fresh candidates land in the project scope recorded at
    # staging time; a supersede targets the scope its baseline entry lives in
    # (a seed → global scope, a project memory → that project's scope).
    # Records staged before per-project scoping carry neither field and fall
    # back to the shared default project scope (where the legacy flat-scope
    # memories were migrated — see migration 315).
    target_scope = cand.get("target_scope") or harness_memory_scope("", "")
    supersede_scope = cand.get("supersedes_baseline_scope") or target_scope
    try:
        if base_id:
            new_id = await store.supersede(
                base_id, text, user_id=user_id, memory_type=mtype,
                scope=supersede_scope, scope_strict=True,
                source_type=SourceType.EXPLICIT, importance=0.7, confidence=0.9,
            )
        else:
            new_id = await store.store(
                text, mtype, user_id=user_id, scope=target_scope,
                scope_strict=True, source_type=SourceType.EXPLICIT,
                importance=0.7, confidence=0.9,
            )
    except Exception:
        log.warning("harness_promote_failed", user_id=user_id, obs_id=obs_id, exc_info=True)
        return {"status": "error", "error": "store write failed"}

    record_harvest_decision(
        obs_id=obs_id, idx=candidate_index, action="promote",
        user_id=user_id, baseline_id=str(new_id or ""),
    )
    log.info(
        "harness_candidate_promoted", user_id=user_id, obs_id=obs_id,
        idx=candidate_index, baseline_id=new_id, superseded=bool(base_id),
    )
    return {"status": "promoted", "baseline_id": new_id, "superseded": bool(base_id)}


def dismiss_candidate(user_id: str, obs_id: str, candidate_index: int) -> dict:
    """Mark a staged candidate dismissed (no baseline write) so it drops out of
    the pending review queue."""
    from augmentum.training.capture import get_harness_record, record_harvest_decision

    # Ownership gate (parity with promote_candidate): record_harvest_decision
    # keys the ledger by obs_id alone, so without this check any user could
    # dismiss another user's candidate out of their review queue. obs_ids are
    # unguessable, but defense-in-depth — confirm the record is ours first.
    rec = get_harness_record(user_id, obs_id)
    if rec is None:
        return {"status": "error", "error": "record not found"}
    cands = rec.get("candidates") or []
    if not (0 <= candidate_index < len(cands)):
        return {"status": "error", "error": "candidate index out of range"}

    record_harvest_decision(
        obs_id=obs_id, idx=candidate_index, action="dismiss", user_id=user_id,
    )
    return {"status": "dismissed"}
