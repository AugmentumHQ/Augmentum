"""Shared KV / prompt-cache reuse audit.

Extracted verbatim from ``LlamaCppBackend`` so the OpenAI-compatible
backend can run the SAME request-side prefix-stability contract joined
with the response-side cache telemetry. The logic is provider-neutral —
it reads ``request.messages``, ``kv_session_key`` and two token counts —
so the only backend-specific seam is ``_kv_reuse_trackable`` (below).

For llama-server the counts come from the ``timings`` block; for a remote
provider they are ``cache_miss_tokens`` / ``cache_hit_tokens`` off the
normalised usage. A byte-stable payload whose provider still re-charged
the whole prompt classifies as ``server_void`` on both — for the bundled
engine that means a slot/restore miss, for a remote endpoint it means the
endpoint ignored the ``prompt_cache_key`` we sent (the codex-proxy bridge
case). ``cold_expected`` still suppresses first-turn noise on either.
"""
from __future__ import annotations

from collections import OrderedDict

from augmentum.models.base import InternalChatRequest
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class KvReuseAuditMixin:
    """Prefix-stability contract + response-side reuse audit.

    Backends mix this in and call ``_init_kv_audit()`` from ``__init__``.
    ``track_prefix_stability`` runs on the request path (before dispatch);
    ``_audit_kv_reuse`` runs when the response's token counts arrive and
    returns an ``augmentum``-payload dict (or None when nothing to judge).
    """

    def _init_kv_audit(self) -> None:
        """Allocate the per-session request-side tracking state.

        Two bounded LRUs keyed by ``kv_session_key``: the last payload seen
        (for turn-to-turn diffing) and the joined contract verdict the
        response-side audit consumes. Call once from the owning backend's
        ``__init__``."""
        self._prefix_track: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
        self._kv_contract: OrderedDict[str, dict] = OrderedDict()

    def _kv_reuse_trackable(self) -> bool:
        """Whether reuse auditing applies to this backend/request.

        Default: always (remote providers with normalised cache telemetry).
        ``LlamaCppBackend`` overrides it to require a live slot manager, so a
        remote llama.cpp backend keeps skipping the audit exactly as before.
        """
        return True

    _PREFIX_TRACK_MAX_SESSIONS = 8

    @staticmethod
    def _prefix_track_messages(messages) -> list[tuple[str, str]]:
        """Flatten request messages to comparable (role, content) pairs."""
        out: list[tuple[str, str]] = []
        for m in messages:
            content = m.content if isinstance(m.content, str) else str(m.content or "")
            out.append((m.role or "", content))
        return out

    def track_prefix_stability(
        self, request: InternalChatRequest, *, source: str = "turn",
    ) -> None:
        """Measure where this turn's payload diverges from the previous one.

        The stable-prefix contract: everything before the dynamic suffix
        (the final user message + its per-turn injections) must be
        byte-identical turn-to-turn, because every KV reuse mechanism —
        in-slot prefix match, RAM prompt-cache restore (f_keep >= 0.25
        gate), and hybrid-model checkpoint validity — is bounded by the
        first divergent token. Divergence at the previous payload's LAST
        message is expected (that's the suffix moving forward); anything
        earlier means a mode mutated history mid-prefix and forfeits
        reuse — logged as a violation with the offending message index
        and role so the owning surface is identifiable from telemetry.

        ``source="prewarm"`` records prepare_stable_checkpoint content
        under a sibling key; the next real turn is compared against BOTH
        entries, so prewarm/turn mismatch (a prewarm that prefills
        unmatchable content) is measured live rather than assumed.
        """
        base_key = (getattr(request, "kv_session_key", "") or "").strip()
        if not base_key:
            return
        key = f"{base_key}::prewarm" if source == "prewarm" else base_key
        msgs = self._prefix_track_messages(request.messages)

        prev = self._prefix_track.get(base_key)
        prev_prewarm = self._prefix_track.get(f"{base_key}::prewarm")

        self._prefix_track[key] = msgs
        self._prefix_track.move_to_end(key)
        while len(self._prefix_track) > self._PREFIX_TRACK_MAX_SESSIONS * 2:
            self._prefix_track.popitem(last=False)

        if source != "turn":
            return

        # Prewarm first: when the prewarmed checkpoint bridges the turn-to-
        # turn divergence (its whole prefix matches this turn), a turn-
        # baseline violation is expected — narrative's injections around
        # the last user message make raw consecutive payloads diverge by
        # design, and the prewarm exists precisely to bridge that. Demote
        # those to info so warnings mean "reuse is actually broken".
        prewarm_bridges = False
        verdicts: dict[str, dict] = {}
        for label, baseline in (("prewarm", prev_prewarm), ("turn", prev)):
            if baseline is None:
                continue
            stable = 0
            for (pr, pc), (cr, cc) in zip(baseline, msgs, strict=False):
                if pr != cr or pc != cc:
                    break
                stable += 1
            div_lcp = 0
            div_role = ""
            if stable < min(len(baseline), len(msgs)):
                div_role = msgs[stable][0]
                for a, b in zip(baseline[stable][1], msgs[stable][1], strict=False):
                    if a != b:
                        break
                    div_lcp += 1
            cur_chars = sum(len(c) for _, c in msgs)
            stable_chars = sum(len(msgs[i][1]) for i in range(stable)) + div_lcp
            # Expected divergence boundary differs by baseline:
            # - turn baseline: the dynamic suffix is the last USER message
            #   plus everything after it (trailing injections — jailbreak,
            #   author's note). Divergence at/after that region is normal
            #   suffix movement; divergence BEFORE it means history was
            #   mutated mid-prefix.
            # - prewarm baseline: the checkpoint IS the stable prefix, so
            #   the next turn should extend it exactly — any divergence
            #   inside it means the prewarm prefilled unmatchable content.
            if label == "prewarm":
                violation = stable < len(baseline)
                if not violation:
                    prewarm_bridges = True
            else:
                last_user_idx = 0
                for i in range(len(baseline) - 1, -1, -1):
                    if baseline[i][0] == "user":
                        last_user_idx = i
                        break
                violation = stable < last_user_idx
            bridged = violation and label == "turn" and prewarm_bridges
            emit = log.warning if (violation and not bridged) else log.info
            # On violation, include the text on each side of the divergence
            # point so the offending injection names itself in telemetry —
            # index/role alone proved insufficient to identify the culprit
            # (2026-07-02 audit).
            div_baseline_snip = div_current_snip = None
            if violation and div_role:
                if stable < len(baseline):
                    div_baseline_snip = baseline[stable][1][div_lcp : div_lcp + 120]
                if stable < len(msgs):
                    div_current_snip = msgs[stable][1][div_lcp : div_lcp + 120]
            stable_pct = round(stable_chars / max(1, cur_chars), 3)
            verdicts[label] = {
                "violation": violation,
                "stable_pct": stable_pct,
                "divergent_role": div_role or None,
                "div_current_snip": div_current_snip,
            }
            emit(
                "kv_prefix_stability",
                baseline=label,
                session=base_key[-12:],
                mode=getattr(request, "kv_mode", "") or "",
                prev_msgs=len(baseline),
                cur_msgs=len(msgs),
                stable_msgs=stable,
                divergent_index=stable if div_role else None,
                divergent_role=div_role or None,
                char_lcp_in_divergent=div_lcp if div_role else None,
                stable_pct=stable_pct,
                div_baseline_snip=div_baseline_snip,
                div_current_snip=div_current_snip,
                contract=(
                    "violated_bridged" if bridged
                    else "violated" if violation else "ok"
                ),
            )

            # DIAGNOSTIC (opt-in via AUGMENTUM_KV_DUMP): on a real turn-
            # baseline violation, persist BOTH payloads being compared so
            # the exact divergent message can be diffed offline. A single
            # dump is self-contained (baseline + current). Remove once the
            # coder prefix-instability root cause is pinned (2026-07-03).
            if violation and not bridged and label == "turn":
                self._dump_prefix_divergence(
                    base_key, baseline, msgs, stable, div_lcp,
                )

        # Persist the joined verdict for the response-side reuse audit
        # (_audit_kv_reuse). ``expected_pct`` is the best reuse fraction
        # ANY baseline permits — the server's LCP matcher gets to pick
        # whichever prefix matches longest, so a turn-baseline violation
        # bridged by a prewarm still expects full reuse. Contract is
        # "ok" when at least one baseline is violation-free.
        if verdicts:
            any_ok = any(not v["violation"] for v in verdicts.values())
            best = max(verdicts.values(), key=lambda v: v["stable_pct"])
            culprit = verdicts.get("turn") or best
            self._kv_contract[base_key] = {
                "contract": "ok" if any_ok else "violated",
                "expected_pct": best["stable_pct"],
                "divergent_role": culprit.get("divergent_role"),
                "div_current_snip": culprit.get("div_current_snip"),
            }
        else:
            self._kv_contract[base_key] = {
                "contract": "first_turn", "expected_pct": 0.0,
            }
        self._kv_contract.move_to_end(base_key)
        while len(self._kv_contract) > self._PREFIX_TRACK_MAX_SESSIONS * 2:
            self._kv_contract.popitem(last=False)

    _KV_AUDIT_SLACK = 0.15  # char→token skew + moving suffix tolerance
    _KV_REUSE_FLOOR = 0.05  # below this, reuse is effectively zero

    def _audit_kv_reuse(
        self,
        request: InternalChatRequest,
        *,
        evaluated_n: int,
        cache_n: int,
        endpoint: str,
    ) -> dict | None:
        """Judge actual KV reuse against what the payload permitted.

        Joins the three per-turn signals at the only point they coexist
        (response timings arrival):

        1. contract — did Augmentum keep the payload prefix byte-stable?
           (request side, stored by track_prefix_stability)
        2. kv_tier  — what did slot management decide? (hot / restore /
           cold, bound to this task's ContextVar by _manage_slot)
        3. cache_n / prompt_n — what did llama-server ACTUALLY reuse?

        Classification:
          hot                — reuse matched expectation; working as designed
          partial_reuse      — some reuse, but well short of what the
                               stable prefix permitted
          payload_divergence — Augmentum voided the contract (culprit
                               already named by kv_prefix_stability)
          server_void        — payload was reusable but the server cold-
                               prefilled anyway; ``cause`` narrows it via
                               the tier decision
          cold_expected      — first turn / untracked session; cold is
                               correct, not a defect

        Returns aug-payload fields for the terminal stream chunk (rides
        the existing ``augmentum`` dict through streaming.py to every
        surface) or None when there's nothing to judge.
        """
        base_key = (getattr(request, "kv_session_key", "") or "").strip()
        total = max(0, int(evaluated_n)) + max(0, int(cache_n))
        if not base_key or total <= 0 or not self._kv_reuse_trackable():
            return None
        from augmentum.proxy.status_bus import kv_tier_var, request_id_var

        verdict = self._kv_contract.get(base_key) or {}
        contract = verdict.get("contract") or "untracked"
        expected_pct = float(verdict.get("expected_pct") or 0.0)
        actual_pct = round(cache_n / total, 3)
        tier = kv_tier_var.get() or ""
        # Tokens re-evaluated that a stable, matched prefix would have
        # served from cache — the direct cost of the void in prefill work.
        wasted = max(0, int(round(expected_pct * total)) - int(cache_n))

        cause = None
        if contract in ("first_turn", "untracked"):
            cls = "cold_expected"
        elif contract == "violated":
            cls = "payload_divergence"
        elif actual_pct + self._KV_AUDIT_SLACK >= expected_pct:
            cls = "hot"
        elif actual_pct <= self._KV_REUSE_FLOOR:
            cls = "server_void"
            # The tier decision names the failing mechanism: a "hot"
            # tier with zero reuse means occupancy tracking and the
            # slot's real KV disagree; a restore tier means the restore
            # ran but didn't produce matchable tokens.
            cause = {
                "hot": "slot_kv_mismatch",
                "cold_with_checkpoint": "restore_ineffective",
                "cold_replay_warmed": "replay_ineffective",
                "cold_no_checkpoint": "no_checkpoint",
            }.get(tier, "unmanaged" if not tier or tier == "unmanaged" else tier)
        else:
            cls = "partial_reuse"

        emit = (
            log.warning
            if cls in ("payload_divergence", "server_void")
            else log.info
        )
        emit(
            "kv_reuse_audit",
            session=base_key[-12:],
            mode=getattr(request, "kv_mode", "") or "",
            endpoint=endpoint,
            classification=cls,
            cause=cause,
            kv_tier=tier,
            contract=contract,
            cache_hit_tokens=int(cache_n),
            evaluated_tokens=int(evaluated_n),
            actual_reuse_pct=actual_pct,
            expected_reuse_pct=round(expected_pct, 3),
            wasted_prefill_tokens=wasted,
            divergent_role=(
                verdict.get("divergent_role")
                if cls == "payload_divergence" else None
            ),
            div_current_snip=(
                verdict.get("div_current_snip")
                if cls == "payload_divergence" else None
            ),
            request_id=request_id_var.get() or "",
        )
        out = {
            "kv_reuse": cls,
            "kv_reuse_pct": actual_pct,
            "kv_expected_pct": round(expected_pct, 3),
            "kv_tier": tier,
        }
        if cause:
            out["kv_void_cause"] = cause
        return out

    def _dump_prefix_divergence(
        self, base_key, baseline, msgs, stable, div_lcp,
    ) -> None:
        """Write both compared payloads to /data/kv_debug for offline diff.

        Self-gated on a sentinel file so it can be armed/disarmed with a
        touch/rm and no container recreate. Capped so an armed sentinel
        left in place can't flood the disk. Diagnostic only — remove once
        the coder prefix-instability root cause is pinned (2026-07-03).
        """
        try:
            import json
            import pathlib
            import time

            d = pathlib.Path("/data/kv_debug")
            if not (d / ".enable").exists():
                return
            if len(list(d.glob("*.json"))) >= 8:
                return
            d.mkdir(parents=True, exist_ok=True)
            stamp = int(time.time() * 1000)
            out = d / f"{base_key[-12:]}_{stamp}_{len(msgs)}.json"
            out.write_text(
                json.dumps(
                    {
                        "session": base_key[-12:],
                        "stable_index": stable,
                        "char_lcp_in_divergent": div_lcp,
                        "prev_msgs": len(baseline),
                        "cur_msgs": len(msgs),
                        "baseline": [
                            {"i": i, "role": r, "len": len(c), "content": c}
                            for i, (r, c) in enumerate(baseline)
                        ],
                        "current": [
                            {"i": i, "role": r, "len": len(c), "content": c}
                            for i, (r, c) in enumerate(msgs)
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            log.warning("kv_dump_failed", error=str(exc))
