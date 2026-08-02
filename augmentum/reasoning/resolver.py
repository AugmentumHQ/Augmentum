"""Flow resolution — selects the active reasoning flow for a request."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from augmentum.reasoning.models import ReasoningFlow
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.reasoning.store import FlowStore

log = get_logger(__name__)

# Minimum keyword/domain score for auto-routing to pick a flow over the
# UARF fallback. Shared with the routing-preview endpoint so what the
# editor shows is exactly what dispatch will do.
MIN_AUTO_ROUTE_SCORE = 2


def _kw_in_query(keyword: str, query_lower: str) -> bool:
    """Word-boundary keyword match.

    Bare substring matching routed real queries wrong: Application's
    "form" fired on "transFORMers", "app" on "hAPPen". Word boundaries
    keep multi-word phrases working ("current events", "how to") while
    stopping mid-word hits.
    """
    kw = keyword.strip().lower()
    if not kw:
        return False
    return re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", query_lower) is not None


def score_flow_for_query(flow: ReasoningFlow, query: str) -> tuple[int, list[str]]:
    """Keyword/domain routing score for one flow against one query.

    The single source of truth for auto-routing relevance — used by the
    live resolver (step 4 of :func:`resolve_flow`) AND the
    ``/flows/routing-preview`` endpoint. Returns ``(score,
    matched_keywords)``; keywords score 2, domains 1.
    """
    query_lower = query.lower()
    score = 0
    matched: list[str] = []
    for kw in flow.trigger_keywords:
        if _kw_in_query(kw, query_lower):
            score += 2
            matched.append(kw)
    for domain in flow.trigger_domains:
        if _kw_in_query(domain, query_lower):
            score += 1
            matched.append(domain)
    return score, matched


async def resolve_flow(
    store: FlowStore | None,
    *,
    model: str = "",
    query: str = "",
    explicit_flow_id: str = "",
    user_id: str = "",
) -> ReasoningFlow | None:
    """Select the best reasoning flow for a request.

    Priority: explicit selection > model pinning > auto routing > default.

    When the default flow is "Auto Routing", keyword/domain matching selects
    the best flow.  If nothing matches, returns None so the handler falls
    through to the full UARF pipeline.  When the user sets a specific flow as
    default (e.g. Quick Answer), that flow is always used — keyword matching
    is skipped.

    `user_id` scopes every store lookup to the requesting user (plus the
    NULL-owner builtins). Without it, the FlowStore falls back to "any
    matching row across all owners", which leaks one user's flows into
    another user's auto-routing keyword/default-flow selection.
    """
    if not store:
        return None

    # 1. Explicit selection (skip if it's the Auto Routing meta-flow —
    #    that should fall through to keyword matching at step 4)
    if explicit_flow_id:
        flow = await store.get_flow(explicit_flow_id, user_id=user_id)
        if flow and flow.name != "Auto Routing":
            log.info("flow_resolved", method="explicit", flow=flow.name)
            return flow

    # 2. Model pinning — check all flows for pinned_models match
    if model:
        flows_with_counts = await store.list_flows(user_id=user_id)
        model_lower = model.lower()
        for flow_summary, _ in flows_with_counts:
            full_flow = await store.get_flow(flow_summary.id, user_id=user_id)
            if not full_flow or not full_flow.auto_select:
                continue
            if any(m.lower() in model_lower or model_lower in m.lower()
                   for m in full_flow.pinned_models):
                log.info("flow_resolved", method="model_pin", flow=full_flow.name, model=model)
                return full_flow

    # 3. Check the default flow
    default = await store.get_default_flow(user_id=user_id)

    # If the user set a specific flow as default (not Auto Routing),
    # respect their choice — skip keyword matching entirely.
    if default and default.name != "Auto Routing":
        log.info("flow_resolved", method="default", flow=default.name)
        return default

    # 4. Auto Routing — keyword/domain matching across all flows
    if query:
        flows_with_counts = await store.list_flows(user_id=user_id)
        best_match: ReasoningFlow | None = None
        best_score = 0

        for flow_summary, _ in flows_with_counts:
            if flow_summary.name == "Auto Routing":
                continue
            full_flow = await store.get_flow(flow_summary.id, user_id=user_id)
            if not full_flow or not full_flow.auto_select:
                continue

            score, _matched = score_flow_for_query(full_flow, query)

            if score > best_score or (
                score == best_score
                and score > 0
                and best_match is not None
                and "agentic" in [d.lower() for d in best_match.trigger_domains]
                and "agentic" not in [d.lower() for d in full_flow.trigger_domains]
            ):
                best_score = score
                best_match = full_flow

        if best_match and best_score >= MIN_AUTO_ROUTE_SCORE:
            log.info("flow_resolved", method="auto_routing", flow=best_match.name, score=best_score)
            return best_match

    # 5. No keyword match — fall through to UARF pipeline (return None)
    # The handler will run the full hardcoded UARF phases (ASSESS → IDENTIFY →
    # RELEVANT → APPLY → VERIFY → CONCLUDE) which provides structured reasoning
    # with verification that Quick Answer's single-step can't match.
    log.info("flow_resolved", method="uarf_fallback", reason="no_keyword_match")
    return None


async def _find_flow_by_name(
    store: FlowStore, name: str, *, user_id: str = ""
) -> ReasoningFlow | None:
    """Find a flow by exact name (scoped to ``user_id`` + builtins)."""
    flows_with_counts = await store.list_flows(user_id=user_id)
    for flow_summary, _ in flows_with_counts:
        if flow_summary.name == name:
            return await store.get_flow(flow_summary.id, user_id=user_id)
    return None
