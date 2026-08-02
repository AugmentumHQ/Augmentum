"""Shared per-turn workspace grounding for coder plan/act phases."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from augmentum.models.base import InternalChatRequest
from augmentum.modes.coder.runtime_truth import RuntimeTruth, build_runtime_truth

if TYPE_CHECKING:
    from augmentum.modes.coder.handler import CoderHandler

log = structlog.get_logger(__name__)


def _clip_context(text: str, max_chars: int) -> str:
    """Clip a context block while making truncation visible to the model."""
    text = (text or "").strip()
    if not text or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return (
        text[:max_chars].rstrip()
        + f"\n... [truncated {omitted} chars; inspect workspace/tools for details]"
    )


def _latest_user_content(request: InternalChatRequest) -> str:
    """Return the latest raw user message content for repo-map/search ranking."""
    for msg in reversed(request.messages):
        if getattr(msg, "role", "") != "user":
            continue
        content = (getattr(msg, "content", "") or "").strip()
        if content:
            return content
    return ""


@dataclass(slots=True)
class TurnContext:
    """One-per-user-turn shared workspace grounding."""

    latest_input: str
    user_goal: str
    user_query: str
    digest: str | None = None
    snapshot_tree: str | None = None
    fallback_context: str | None = None
    repo_map_block: str | None = None
    workspace_profile_block: str | None = None
    tree_is_authoritative: bool = False
    runtime_truth: RuntimeTruth | None = None
    recalled_block: str = ""
    _workspace_id: str = ""
    _semantic_hits: list[Any] = field(default_factory=list)
    _semantic_loaded: bool = False
    _semantic_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _render_base(self, *, include_fallback: bool) -> str:
        base = ""
        if self.digest:
            base = self.digest
        else:
            parts: list[str] = []
            if self.snapshot_tree:
                parts.append(self.snapshot_tree)
            if self.repo_map_block:
                parts.append(self.repo_map_block)
            if parts:
                base = "\n\n".join(parts)
            elif include_fallback and self.fallback_context:
                base = self.fallback_context
        if self.workspace_profile_block:
            return (
                f"{self.workspace_profile_block}\n\n{base}"
                if base else self.workspace_profile_block
            )
        return base

    def to_plan_context(self) -> str:
        """Render the plan-phase workspace context."""
        return self._with_recall(self._render_base(include_fallback=True))

    def _with_recall(self, body: str) -> str:
        """Append the auto-recalled past-turns block to a rendered context.

        The block is already bounded + HISTORICAL-framed by
        ``_render_recalled_block``; appending keeps it visually distinct
        from current-state grounding. No-op when recall is empty.
        """
        if not self.recalled_block:
            return body
        return f"{body}\n\n{self.recalled_block}" if body else self.recalled_block

    def _render_project_grounding(self, *, include_fallback: bool) -> str:
        """Render project grounding without profile/runtime overlays."""
        if self.digest:
            return self.digest
        parts: list[str] = []
        if self.snapshot_tree:
            parts.append(self.snapshot_tree)
        if self.repo_map_block:
            parts.append(self.repo_map_block)
        if parts:
            return "\n\n".join(parts)
        if include_fallback and self.fallback_context:
            return self.fallback_context
        return ""

    def to_native_context(
        self,
        *,
        prior_turns: str = "",
        runtime_truth_block: str = "",
        max_chars: int = 6000,
    ) -> str:
        """Render a bounded grounding prelude for the native strategy.

        Native mode intentionally skips the full hybrid scaffold, but it
        still benefits from concise facts that prevent needless
        rediscovery across turns. Keep this bounded and non-nudgy:
        orientation only, with no task-list/sticky-reminder machinery.
        """
        runtime = runtime_truth_block.strip()
        if not runtime and self.runtime_truth is not None:
            runtime = self.runtime_truth.render_block()

        sections: list[str] = []
        if runtime:
            sections.append("## Runtime Truth\n" + _clip_context(runtime, 1200))
        if prior_turns:
            sections.append(_clip_context(prior_turns, 1800))
        if self.workspace_profile_block:
            sections.append(
                "## Workspace Profile\n"
                + _clip_context(self.workspace_profile_block, 1600)
            )
        # recalled_block intentionally NOT rendered here — it re-ranks
        # per query and would mutate the system prefix every turn,
        # killing the slot cache. It rides the runtime carrier instead
        # (see ``to_native_dynamic_context``).

        project = self._render_project_grounding(include_fallback=True)
        if project:
            sections.append(
                "## Workspace Grounding\n" + _clip_context(project, 2800)
            )

        if not sections:
            return ""

        body = "\n\n".join(sections)
        body = _clip_context(body, max_chars)
        return (
            "## Native Context Prelude\n"
            "Brief Augmentum grounding for this turn. Use it as orientation; "
            "inspect files or tools when exact current state matters.\n\n"
            f"{body}"
        )

    def to_native_dynamic_context(self) -> str:
        """Per-turn dynamic grounding for native's runtime carrier.

        Native skips semantic search by design, so its dynamic half is
        the auto-recall block only. Same cache rationale as
        ``to_act_dynamic_context``.
        """
        if not self.recalled_block:
            return ""
        return _clip_context(self.recalled_block, 900)

    async def _load_semantic_hits(self) -> list[Any]:
        if self._semantic_loaded:
            return self._semantic_hits

        async with self._semantic_lock:
            if self._semantic_loaded:
                return self._semantic_hits
            if not self.user_query:
                self._semantic_loaded = True
                return self._semantic_hits

            try:
                from augmentum.coder.indexer import search_index

                results = await search_index(
                    self._workspace_id,
                    self.user_query,
                    limit=5,
                )
                self._semantic_hits = list(results or [])
            except Exception:
                log.warning("coder_semantic_search_failed", exc_info=True)
                self._semantic_hits = []
            self._semantic_loaded = True
            return self._semantic_hits

    async def to_act_context(self) -> str:
        """Render the act-phase workspace context — STABLE parts only.

        This string lands in the system message, which must stay
        byte-identical across turns for the llama-server slot
        prefix-cache to hit (see ``CoderHandler._build_messages``).
        Query-dependent content (semantic hits, auto-recall) mutates
        every turn and therefore rides the per-turn runtime carrier
        instead — measured live 2026-07-02: recall in the system
        prompt caused contract=violated at message 0, re-prefilling
        the entire context (~7KB LCP into system, then 100% cold).
        Fetch that half via ``to_act_dynamic_context``.
        """
        return self._render_base(include_fallback=False)

    async def to_act_dynamic_context(self) -> str:
        """Render the per-turn DYNAMIC grounding for the runtime carrier.

        Semantic "Relevant Code" hits (re-ranked per user query) + the
        auto-recall block (per-query ranking). Both change across turns
        by nature, so they live in the carrier at the tail of the
        message list where their churn only invalidates the newest
        user message, never the long stable prefix.
        """
        parts: list[str] = []
        results = await self._load_semantic_hits()
        if results:
            hit_parts = ["## Relevant Code"]
            for r in results:
                content = r.content
                if content.startswith("File:"):
                    content = (
                        content.split("\n", 1)[-1]
                        if "\n" in content
                        else content
                    )
                hit_parts.append(
                    f"\n{r.file_path}:{r.start_line}-{r.end_line} "
                    f"(score {r.score:.2f})",
                )
                hit_parts.append(content[:1000])
            parts.append("\n".join(hit_parts))
        if self.recalled_block:
            parts.append(self.recalled_block)
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Auto-recall — surface the top semantically-relevant PAST turns from the
# durable archive into each turn's context, so the deep archive informs work
# without the model having to call the ``recall`` tool. The write path
# (turn_archive append → embed) is always live; this is the read path that
# wakes it up. Bounded, deduped against the in-prompt <prior_turns> ring,
# HISTORICAL-framed, and entirely best-effort (any failure → no block).
# ---------------------------------------------------------------------------

# Char ceiling for the rendered block — ~200 tokens. Kept small because this
# rides every turn alongside <prior_turns>; recall is orientation, not the
# task. Tighter than the recall *tool* (which is on-demand and can be richer).
_RECALL_BUDGET_CHARS = 800


def _recall_at(event_time: int) -> str:
    """Absolute local timestamp ('Jul 2 14:32') for a unix timestamp.

    Absolute — NOT relative ('7m ago') — on purpose: relative phrasing
    re-renders differently every minute, so a byte-identical recall hit
    would still mutate the prompt prefix and invalidate the llama-server
    slot cache (measured live 2026-07-02, contract=violated at message 0).
    The model gets "now" from the <current_time> block that rides at the
    END of the runtime carrier, outside the stable zone, and can compute
    recency itself. '' when unknown.
    """
    if not event_time:
        return ""
    from datetime import datetime

    from augmentum.utils.datetime_context import _get_local_tz

    dt = datetime.fromtimestamp(event_time, _get_local_tz())
    return f"{dt.strftime('%b')} {dt.day} {dt.strftime('%H:%M')}"


def _render_recalled_block(hits: list[Any]) -> str:
    """Render auto-recalled past turns as a compact, HISTORICAL-framed block.

    Returns ``""`` when ``hits`` is empty or nothing fits the budget — the
    caller treats that as "no recall this turn". The framing mirrors the
    recall tool's header so the model treats returned file/outcome state as
    stale, not current.
    """
    if not hits:
        return ""
    lines = [
        "<recalled_context>",
        "Older work in THIS workspace that may relate to the current task. "
        "HISTORICAL — already completed or abandoned. Use it to avoid "
        "re-discovering past findings; do NOT treat returned file state as "
        "current (re-read files) or past outcomes as still true (re-verify).",
    ]
    used = sum(len(s) + 1 for s in lines) + len("</recalled_context>") + 1
    added = False
    for h in hits:
        at = _recall_at(getattr(h, "event_time", 0))
        head_bits = [f"Turn {getattr(h, 'turn_index', 0)}"]
        if at:
            head_bits.append(at)
        outcome = (getattr(h, "outcome", "") or "").strip()
        if outcome:
            head_bits.append(outcome)
        goal = (getattr(h, "user_goal", "") or "").strip().replace("\n", " ")
        summary = (getattr(h, "summary", "") or "").strip().replace("\n", " ")
        detail = goal or summary
        if len(detail) > 160:
            detail = detail[:160] + "…"
        head = " · ".join(head_bits)
        line = f"• {head} — {detail}" if detail else f"• {head}"
        if used + len(line) + 1 > _RECALL_BUDGET_CHARS:
            break
        lines.append(line)
        used += len(line) + 1
        added = True
    if not added:
        return ""
    lines.append("</recalled_context>")
    return "\n".join(lines)


async def _build_recalled_block(
    handler: CoderHandler, *, user_goal: str, user_query: str,
) -> str:
    """Compute the auto-recall block for this turn (or ``""``).

    Best-effort throughout: a missing conn, disabled setting, empty query,
    embedder hiccup, or search error all return ``""`` so the turn proceeds
    without recall. Deduped against the turn-summary ring so we never repeat
    a turn already shown verbatim in <prior_turns>.
    """
    from augmentum.config import settings

    if not getattr(settings, "coder_auto_recall_enabled", True):
        return ""
    if not getattr(settings, "coder_archive_enabled", True):
        return ""

    query = (user_goal or user_query or "").strip()
    if not query:
        return ""

    sm = getattr(handler, "_state_manager", None)
    conn = getattr(getattr(sm, "backend", None), "conn", None) if sm else None
    user_id = getattr(handler, "_user_id", "") or ""
    workspace_id = getattr(handler, "_workspace_id", "") or ""
    if conn is None or not user_id or not workspace_id:
        return ""

    k = max(1, min(int(getattr(settings, "coder_auto_recall_k", 3) or 3), 10))
    max_distance = float(getattr(settings, "coder_auto_recall_max_distance", 0.0) or 0.0)

    # Turns already shown verbatim in the <prior_turns> ring — don't repeat
    # them; recall earns its tokens only by surfacing OLDER relevant work.
    recent_indices: set[int] = set()
    state = getattr(handler, "_state", None)
    for s in getattr(state, "turn_summaries", None) or []:
        try:
            recent_indices.add(int(s.get("turn_idx", 0)))
        except (AttributeError, TypeError, ValueError):
            continue

    try:
        from augmentum.coder.turn_archive_embed import search_similar

        hits = await search_similar(
            conn,
            user_id=user_id,
            workspace_id=workspace_id,
            query=query,
            k=k + len(recent_indices),
            similarity_threshold=max_distance,
        )
    except Exception:
        log.warning("coder_auto_recall_failed", exc_info=True)
        return ""

    fresh = [h for h in hits if h.turn_index not in recent_indices][:k]
    block = _render_recalled_block(fresh)
    if block:
        log.info(
            "coder.auto_recall",
            workspace_id=workspace_id,
            hits=len(fresh),
            top_turn=fresh[0].turn_index,
            top_distance=round(fresh[0].distance, 4),
        )
    return block


async def build_turn_context(
    *,
    handler: CoderHandler,
    request: InternalChatRequest,
) -> TurnContext:
    """Assemble shared per-turn workspace grounding for plan and act."""
    from augmentum.modes.coder import handler as _handler

    await handler._get_workspace_guide()
    runtime_truth = await build_runtime_truth(handler=handler)

    latest_input, user_goal = _handler._extract_goal_split(request.messages)
    user_query = _latest_user_content(request)[:300]

    digest: str | None = None
    snapshot_tree: str | None = None
    fallback_context: str | None = None
    repo_map_block: str | None = None
    workspace_profile_block: str | None = None
    tree_is_authoritative = False

    try:
        sm = getattr(handler, "_state_manager", None)
        conn = getattr(getattr(sm, "backend", None), "conn", None) if sm else None
        user_id = getattr(handler, "_user_id", "") or ""
        workspace_id = getattr(handler, "_workspace_id", "") or ""
        if conn is not None and user_id and workspace_id:
            from augmentum.coder.profile import (
                CoderProfileStore,
                observe_workspace_profile,
                render_profile_block,
            )

            store = CoderProfileStore(conn)
            await observe_workspace_profile(
                store,
                user_id=user_id,
                workspace_id=workspace_id,
                container_manager=handler._container_manager,
            )
            entries = await store.query_for_workspace(
                user_id=user_id,
                workspace_id=workspace_id,
            )
            workspace_profile_block = render_profile_block(
                entries,
                query=user_query,
                max_entries=8,
            )
    except Exception:
        log.debug("turn_context_profile_failed", exc_info=True)

    if handler._container_manager:
        digest = None
        try:
            from augmentum.coder.digest import build_project_digest

            digest_kwargs: dict[str, Any] = {}
            try:
                import inspect

                if "token_budget" in inspect.signature(
                    build_project_digest,
                ).parameters:
                    digest_kwargs["token_budget"] = getattr(
                        handler, "_coder_digest_token_budget", None,
                    )
            except (TypeError, ValueError):
                digest_kwargs["token_budget"] = getattr(
                    handler, "_coder_digest_token_budget", None,
                )

            digest = await build_project_digest(
                handler._container_manager,
                handler._workspace_id,
                **digest_kwargs,
            )
            if digest:
                tree_is_authoritative = True
        except Exception:
            log.debug("turn_context_digest_failed", exc_info=True)

        if handler._workspace_snapshot is not None and not digest:
            try:
                await handler._workspace_snapshot.refresh_if_stale(force=True)
                tree = handler._workspace_snapshot.render()
                if tree:
                    snapshot_tree = tree
                    tree_is_authoritative = True
            except Exception:
                log.debug("turn_context_snapshot_failed", exc_info=True)

        if not digest:
            try:
                from augmentum.coder.repomap import build_repo_map

                repo_map_block = await build_repo_map(
                    handler._container_manager,
                    handler._workspace_id,
                    query=user_query,
                    skip_file_listing=tree_is_authoritative,
                )
            except Exception:
                log.debug("turn_context_repo_map_failed", exc_info=True)

    if not digest and not snapshot_tree and not repo_map_block:
        fallback_context = await handler._get_workspace_context()

    recalled_block = await _build_recalled_block(
        handler, user_goal=user_goal, user_query=user_query,
    )

    return TurnContext(
        latest_input=latest_input,
        user_goal=user_goal,
        user_query=user_query,
        digest=digest,
        snapshot_tree=snapshot_tree,
        fallback_context=fallback_context,
        repo_map_block=repo_map_block,
        workspace_profile_block=workspace_profile_block,
        tree_is_authoritative=tree_is_authoritative,
        runtime_truth=runtime_truth,
        recalled_block=recalled_block,
        _workspace_id=handler._workspace_id,
    )
