"""Structured run ledger for Coder turns.

The ledger records the controller-visible shape of a Coder turn without
requiring the UI to replay raw chat deltas. It is intentionally fed from
the same ``augmentum`` stream metadata the frontend consumes, so the API,
logs, and UI all describe the same run.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import aiosqlite

from augmentum.coder import citations, oracle_telemetry
from augmentum.coder.reasoning_repetition import ReasoningRepetitionMeter


def _json(data: Any) -> str:
    return json.dumps(data, default=str, sort_keys=True)


def _dedupe_append(items: list[str], value: str, *, limit: int = 80) -> None:
    clean = (value or "").strip()
    if not clean or clean in items:
        return
    items.append(clean[:500])
    if len(items) > limit:
        del items[: len(items) - limit]


class CoderTurnLedgerStore:
    """SQLite CRUD for ``coder_turn_runs`` and ``coder_turn_events``."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def create_run(
        self,
        *,
        run_id: str,
        user_id: str,
        workspace_id: str,
        session_id: str,
        model: str,
        provider: str = "",
        strategy: str = "",
        prompt_profile: str = "",
        tooling_profile: str = "",
    ) -> None:
        now = time.time()
        await self._conn.execute(
            """
            INSERT INTO coder_turn_runs
                (id, user_id, project_id, session_id, strategy, model,
                 provider, prompt_profile, tooling_profile, status,
                 started_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)
            """,
            (
                run_id,
                user_id or "",
                # Python kwarg stays `workspace_id` for now; column is
                # `project_id` post-migration 200. Caller passes the
                # checkout id, which equals the project id thanks to
                # the migration-200 backfill convention. Renaming the
                # Python parameter is deferred to a housekeeping PR.
                workspace_id or "",
                session_id or "",
                strategy or "",
                model or "",
                provider or "",
                prompt_profile or "",
                tooling_profile or "",
                now,
                now,
            ),
        )
        await self._conn.commit()

    async def record_event(
        self,
        *,
        run_id: str,
        seq: int,
        event_type: str,
        phase: str = "",
        status: str = "",
        payload: dict[str, Any] | None = None,
        user_id: str = "",
    ) -> None:
        now = time.time()
        await self._conn.execute(
            """
            INSERT INTO coder_turn_events
                (run_id, seq, timestamp, type, phase, status, payload, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                seq,
                now,
                event_type or status or "event",
                phase or "",
                status or "",
                _json(payload or {}),
                user_id,
            ),
        )
        await self._conn.commit()

    async def finish_run(
        self,
        *,
        run_id: str,
        status: str,
        summary: dict[str, Any],
    ) -> None:
        now = time.time()
        await self._conn.execute(
            """
            UPDATE coder_turn_runs
            SET status = ?,
                strategy = COALESCE(NULLIF(?, ''), strategy),
                completed_at = ?,
                updated_at = ?,
                first_event_at = ?,
                first_useful_action_at = ?,
                iterations = ?,
                tool_calls = ?,
                parallel_waves = ?,
                retries = ?,
                no_response_events = ?,
                empty_native_content = ?,
                malformed_tool_calls = ?,
                commands_run = ?,
                files_touched = ?,
                tests_run = ?,
                browser_checks = ?,
                finish_reason = ?,
                fallback_reason = ?,
                checkpoint_id = ?,
                changed_files = ?,
                closeout_json = ?,
                metrics_json = ?,
                priming_telemetry = ?,
                input_cost_usd = ?,
                output_cost_usd = ?,
                cost_model_id = ?
            WHERE id = ?
            """,
            (
                status,
                str(summary.get("strategy") or ""),
                now,
                now,
                summary.get("first_event_at"),
                summary.get("first_useful_action_at"),
                int(summary.get("iterations") or 0),
                int(summary.get("tool_calls") or 0),
                int(summary.get("parallel_waves") or 0),
                int(summary.get("retries") or 0),
                int(summary.get("no_response_events") or 0),
                int(summary.get("empty_native_content") or 0),
                int(summary.get("malformed_tool_calls") or 0),
                _json(summary.get("commands_run") or []),
                _json(summary.get("files_touched") or []),
                _json(summary.get("tests_run") or []),
                _json(summary.get("browser_checks") or []),
                str(summary.get("finish_reason") or ""),
                str(summary.get("fallback_reason") or ""),
                str(summary.get("checkpoint_id") or ""),
                _json(summary.get("changed_files") or []),
                _json(summary.get("closeout") or {}),
                _json(summary.get("metrics") or {}),
                _json(summary.get("priming_telemetry") or {}),
                float(summary.get("input_cost_usd") or 0.0),
                float(summary.get("output_cost_usd") or 0.0),
                str(summary.get("cost_model_id") or ""),
                run_id,
            ),
        )
        await self._conn.commit()

    async def get_run(self, run_id: str, *, user_id: str = "") -> dict[str, Any] | None:
        self._conn.row_factory = aiosqlite.Row
        query = "SELECT * FROM coder_turn_runs WHERE id = ?"
        params: list[Any] = [run_id]
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        query += " LIMIT 1"
        cursor = await self._conn.execute(query, params)
        row = await cursor.fetchone()
        if row is None:
            return None
        data = dict(row)
        list_keys = {
            "commands_run",
            "files_touched",
            "tests_run",
            "browser_checks",
            "changed_files",
        }
        for key in (
            "commands_run",
            "files_touched",
            "tests_run",
            "browser_checks",
            "changed_files",
            "closeout_json",
            "metrics_json",
        ):
            raw = data.get(key)
            try:
                data[key] = json.loads(raw) if raw else ([] if key in list_keys else {})
            except Exception:
                data[key] = raw
        return data

    async def oracle_stats(
        self,
        *,
        user_id: str,
        project_id: str = "",
        limit: int = 500,
    ) -> dict[str, Any]:
        """Aggregate verification-spine telemetry over recent finished runs.

        The target metric is ``no_oracle_done_rate`` — the fraction of
        write-turns that shipped without any oracle after the last write.
        Only runs that carry the ``oracle`` metrics block (post-spine)
        contribute; older rows are counted in ``runs_without_telemetry``
        so the denominator is honest rather than silently shrunk.
        """
        self._conn.row_factory = aiosqlite.Row
        query = (
            "SELECT model, strategy, metrics_json FROM coder_turn_runs "
            "WHERE user_id = ? AND status != 'running'"
        )
        params: list[Any] = [user_id]
        if project_id:
            query += " AND project_id = ?"
            params.append(project_id)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(max(1, min(int(limit or 500), 2000)))
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        stats: dict[str, Any] = {
            "runs": 0,
            "runs_without_telemetry": 0,
            "write_runs": 0,
            "no_oracle_done": 0,
            "no_oracle_done_rate": None,
            "kinds": {},
            "last_outcomes": {},
            "per_model": {},
        }
        for row in rows:
            try:
                metrics = json.loads(row["metrics_json"] or "{}")
            except Exception:
                metrics = {}
            oracle = metrics.get("oracle") if isinstance(metrics, dict) else None
            if not isinstance(oracle, dict):
                stats["runs_without_telemetry"] += 1
                continue
            stats["runs"] += 1
            model = str(row["model"] or "")
            per_model = stats["per_model"].setdefault(
                model, {"runs": 0, "write_runs": 0, "no_oracle_done": 0},
            )
            per_model["runs"] += 1
            for kind in oracle.get("kinds") or []:
                stats["kinds"][kind] = stats["kinds"].get(kind, 0) + 1
            if oracle.get("wrote"):
                stats["write_runs"] += 1
                per_model["write_runs"] += 1
                outcome = str(oracle.get("last_outcome") or "none")
                stats["last_outcomes"][outcome] = (
                    stats["last_outcomes"].get(outcome, 0) + 1
                )
                if oracle.get("no_oracle_done"):
                    stats["no_oracle_done"] += 1
                    per_model["no_oracle_done"] += 1
        if stats["write_runs"]:
            stats["no_oracle_done_rate"] = round(
                stats["no_oracle_done"] / stats["write_runs"], 3,
            )
        return stats

    async def list_events(
        self,
        run_id: str,
        *,
        user_id: str = "",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        if user_id:
            owned = await self.get_run(run_id, user_id=user_id)
            if owned is None:
                return []
        self._conn.row_factory = aiosqlite.Row
        cursor = await self._conn.execute(
            """
            SELECT seq, timestamp, type, phase, status, payload
            FROM coder_turn_events
            WHERE run_id = ?
            ORDER BY seq ASC
            LIMIT ?
            """,
            (run_id, max(1, min(int(limit or 500), 2000))),
        )
        rows = await cursor.fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["payload"] = json.loads(item.get("payload") or "{}")
            except Exception:
                item["payload"] = {}
            out.append(item)
        return out


@dataclass(slots=True)
class CoderTurnLedger:
    """In-memory accumulator for one Coder turn."""

    store: CoderTurnLedgerStore
    run_id: str
    model: str = ""
    strategy: str = ""
    user_id: str = ""
    started_at: float = field(default_factory=time.time)
    seq: int = 0
    first_event_at: float | None = None
    first_useful_action_at: float | None = None
    iterations: int = 0
    tool_calls: int = 0
    parallel_waves: int = 0
    retries: int = 0
    no_response_events: int = 0
    empty_native_content: int = 0
    malformed_tool_calls: int = 0
    commands_run: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    tests_run: list[str] = field(default_factory=list)
    browser_checks: list[str] = field(default_factory=list)
    checkpoint_id: str = ""
    changed_files: list[str] = field(default_factory=list)
    # Verification-spine telemetry (oracle_telemetry.py): every classified
    # oracle call {seq, kind, tool, outcome}, and the seq of the last
    # SUCCESSFUL write — summarize() folds these at finish.
    oracle_calls: list[dict[str, Any]] = field(default_factory=list)
    last_write_seq: int = -1
    # Citation ledger (citations.py): one evidence row per write / oracle
    # tool result, accumulated in memory and bulk-persisted at finish() —
    # the claim→proof spine the brief drills into and the gate judges over.
    workspace_id: str = ""
    citations: list[Any] = field(default_factory=list)
    finish_reason: str = ""
    fallback_reason: str = ""
    active_tool_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    token_snapshots: list[dict[str, Any]] = field(default_factory=list)
    max_prompt_tokens: int = 0
    last_prompt_tokens: int = 0
    compactions: int = 0
    # Reasoning-stream instrument (reasoning_repetition.py) — accumulates the
    # turn's reasoning text and reports token volume + repetition signals at
    # finish. Phase 1a of the adaptive-supervision spec: measure reasoning-loop
    # frequency before building the interrupt-and-salvage detector.
    reasoning_meter: ReasoningRepetitionMeter = field(default_factory=ReasoningRepetitionMeter)
    # Usage tallies — accumulated from ``chunk.usage`` when the backend
    # populates it. Some backends only emit usage on the final chunk
    # (Anthropic, OpenAI), others stream it (llama-cpp final delta);
    # we treat the latest non-zero values as authoritative per call and
    # sum across calls in a single turn (the agent loop may invoke the
    # model multiple times via tool-result follow-ups). Cost is computed
    # in ``finish`` from these tallies × cost_table lookup.
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    # Sticky per-call snapshots so a backend that emits incremental
    # usage doesn't double-count: we track the latest seen (prompt,
    # completion) tuple and only roll the delta into the totals.
    _last_seen_prompt: int = 0
    _last_seen_completion: int = 0

    @classmethod
    async def start(
        cls,
        store: CoderTurnLedgerStore,
        *,
        user_id: str,
        workspace_id: str,
        session_id: str,
        model: str,
        provider: str = "",
        strategy: str = "",
        prompt_profile: str = "",
        tooling_profile: str = "",
    ) -> CoderTurnLedger:
        run_id = "ctr_" + uuid.uuid4().hex[:18]
        await store.create_run(
            run_id=run_id,
            user_id=user_id,
            workspace_id=workspace_id,
            session_id=session_id,
            model=model,
            provider=provider,
            strategy=strategy,
            prompt_profile=prompt_profile,
            tooling_profile=tooling_profile,
        )
        return cls(store=store, run_id=run_id, model=model, strategy=strategy,
                   user_id=user_id, workspace_id=workspace_id)

    async def observe_chunk(self, chunk: Any) -> None:
        # Cost-side accounting: every chunk may carry a ``usage`` field
        # populated by the backend's usage parser (Anthropic/OpenAI emit
        # it on the final chunk, llama-server on the inline usage line).
        # We roll the delta into running totals so multi-call turns (the
        # agent loop calls .chat() repeatedly between tool results) sum
        # correctly. See _record_usage_delta for the dedup logic.
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self._record_usage_delta(usage)

        aug = getattr(chunk, "augmentum", None) or {}
        if not isinstance(aug, dict):
            return
        phase = str(aug.get("phase") or "")
        status = str(aug.get("status") or "")
        if not phase and not status:
            return
        # High-frequency live reasoning relay (several chunks/second while
        # the model thinks — see chat_egress.ReasoningRelay). Persisting
        # each one would be an INSERT+COMMIT per chunk on the live DB and
        # would flood the replay ledger; the full reasoning trace already
        # round-trips via message history, so nothing durable is lost.
        if status == "reasoning_delta":
            # Not persisted (would be a SQLite write per token), but the
            # reasoning text IS metered here for the loop instrument before
            # we drop it — the only place the full reasoning stream passes
            # through the ledger. See reasoning_repetition.py.
            self.reasoning_meter.feed(str(aug.get("thinking") or ""))
            return

        now = time.time()
        if self.first_event_at is None:
            self.first_event_at = now

        payload = dict(aug)
        payload.pop("mode", None)
        self._accumulate(payload, phase=phase, status=status, now=now)

        self.seq += 1
        await self.store.record_event(
            run_id=self.run_id,
            seq=self.seq,
            event_type=status or phase or "event",
            phase=phase,
            status=status,
            payload=payload,
            user_id=self.user_id,
        )

    def _record_usage_delta(self, usage: Any) -> None:
        """Roll a single usage report into the cumulative totals.

        Backends emit usage in two patterns:
          * Final-chunk only (Anthropic / OpenAI) — one shot per call.
          * Incremental + final (some llama-server builds) — growing
            counters within a single call.

        We treat a usage report as defining the "current call's" tallies.
        A non-monotonic transition (current < last_seen) signals the
        start of a new call → we flush ``last_seen`` into the totals
        and reset the snapshot. This handles both patterns without
        backend-specific logic.
        """
        try:
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion = int(getattr(usage, "completion_tokens", 0) or 0)
        except (TypeError, ValueError):
            return
        if prompt <= 0 and completion <= 0:
            return
        # Detect "new call" — if either counter went down, the previous
        # call's totals are now sealed; bank them and reset.
        if prompt < self._last_seen_prompt or completion < self._last_seen_completion:
            self.prompt_tokens_total += self._last_seen_prompt
            self.completion_tokens_total += self._last_seen_completion
            self._last_seen_prompt = 0
            self._last_seen_completion = 0
        # Always store the latest non-decreasing snapshot.
        if prompt >= self._last_seen_prompt:
            self._last_seen_prompt = prompt
        if completion >= self._last_seen_completion:
            self._last_seen_completion = completion

    def _seal_pending_usage(self) -> None:
        """Bank the current call's tallies before finalizing the run."""
        if self._last_seen_prompt or self._last_seen_completion:
            self.prompt_tokens_total += self._last_seen_prompt
            self.completion_tokens_total += self._last_seen_completion
            self._last_seen_prompt = 0
            self._last_seen_completion = 0

    def _mark_useful(self, now: float) -> None:
        if self.first_useful_action_at is None:
            self.first_useful_action_at = now

    def _accumulate(self, payload: dict[str, Any], *, phase: str, status: str, now: float) -> None:
        if payload.get("strategy") and status == "strategy":
            self.strategy = str(payload.get("strategy") or "")

        if status in {"tool_call", "tool_result", "streaming", "responding"}:
            self._mark_useful(now)
        if status == "compaction":
            self.compactions += 1
        if status == "budget":
            tokens = payload.get("tokens")
            if isinstance(tokens, dict):
                count = int(tokens.get("tokens") or 0)
                self.last_prompt_tokens = count
                self.max_prompt_tokens = max(self.max_prompt_tokens, count)
                snapshot = {
                    "scope": str(tokens.get("scope") or ""),
                    "tokens": count,
                    "limit": int(tokens.get("limit") or 0),
                    "ratio": tokens.get("ratio"),
                    "iteration": tokens.get("iteration"),
                    "compacted": bool(tokens.get("compacted")),
                }
                self.token_snapshots.append(snapshot)
                if len(self.token_snapshots) > 80:
                    del self.token_snapshots[: len(self.token_snapshots) - 80]
        if status == "empty_model_stop_retry":
            self.empty_native_content += 1
            self.no_response_events += 1
            self.retries += 1
        if status in {"tool_error", "rate_limited"}:
            self.retries += 1
        if status == "fallback_summary":
            self.fallback_reason = str(payload.get("termination_reason") or "fallback_summary")

        if status in {
            "max_iterations_reached",
            "tasks_completed",
            "finish_task_called",
            "repeat_stopped",
            "no_progress",
            "validation_error_break",
            "test_failure_streak_break",
            "same_file_edit_break",
            "action_stagnation_break",
            "inspection_loop_break",
            "no_write_progress_break",
        }:
            self.finish_reason = status

        if status == "complete":
            self.finish_reason = str(payload.get("termination_reason") or self.finish_reason or "complete")
            self.iterations = max(self.iterations, int(payload.get("iterations_used") or 0))
            if payload.get("strategy"):
                self.strategy = str(payload.get("strategy") or self.strategy)

        tc = payload.get("tool_call")
        if isinstance(tc, dict):
            self.tool_calls += 1
            tool_id = str(tc.get("id") or "")
            tool = str(tc.get("tool") or tc.get("name") or "")
            tool_input = tc.get("input") if isinstance(tc.get("input"), dict) else {}
            if tool_id:
                self.active_tool_inputs[tool_id] = {"tool": tool, "input": dict(tool_input)}
            if tool in {"shell_exec", "shell_read"}:
                _dedupe_append(self.commands_run, str(tool_input.get("command") or ""))
            if tool == "test_run":
                _dedupe_append(self.tests_run, str(tool_input.get("command") or "test_run"))
            if tool.startswith("browser_"):
                _dedupe_append(self.browser_checks, tool)

        tr = payload.get("tool_result")
        if isinstance(tr, dict):
            tool_id = str(tr.get("id") or "")
            tool = str(tr.get("tool") or "")
            remembered = self.active_tool_inputs.get(tool_id, {})
            tool_input = remembered.get("input") if isinstance(remembered, dict) else {}
            if tr.get("checkpoint"):
                self.checkpoint_id = str(tr.get("checkpoint") or "")
            # Verification-spine telemetry. Oracle calls are classified on
            # the RESULT (not the call) so the outcome rides along; a
            # failed write is not a claim, so last_write_seq only advances
            # on success-or-unreported.
            oracle_kind = oracle_telemetry.classify_oracle_kind(
                tool, tool_input if isinstance(tool_input, dict) else {},
            )
            outcome = ""
            if oracle_kind:
                outcome = oracle_telemetry.classify_outcome(
                    success=tr.get("success"),
                    output_preview=str(tr.get("output_preview") or ""),
                )
                self.oracle_calls.append({
                    "seq": self.seq,
                    "kind": oracle_kind,
                    "tool": tool,
                    "outcome": outcome,
                })
                if len(self.oracle_calls) > 200:
                    del self.oracle_calls[: len(self.oracle_calls) - 200]
            # Citation ledger: one evidence row per write / oracle tool result
            # (claim→proof provenance the brief drills into). Pure classifier;
            # a failed write is not a claim so it yields nothing.
            new_cites = citations.citations_from_tool_result(
                seq=self.seq,
                tool=tool,
                tool_input=tool_input if isinstance(tool_input, dict) else {},
                success=tr.get("success"),
                checkpoint=str(tr.get("checkpoint") or ""),
                oracle_kind=oracle_kind,
                outcome=outcome,
            )
            if new_cites:
                self.citations.extend(new_cites)
                if len(self.citations) > 300:
                    del self.citations[: len(self.citations) - 300]
            if tool in oracle_telemetry.WRITE_TOOLS and tr.get("success") is not False:
                self.last_write_seq = self.seq
            if tool in {"file_write", "code_edit", "code_edit_batch", "apply_patch"}:
                for key in ("path", "file_path"):
                    if isinstance(tool_input, dict) and tool_input.get(key):
                        _dedupe_append(self.files_touched, str(tool_input.get(key)))
                        _dedupe_append(self.changed_files, str(tool_input.get(key)))
                edits = tool_input.get("edits") if isinstance(tool_input, dict) else None
                if isinstance(edits, list):
                    for edit in edits:
                        if isinstance(edit, dict) and edit.get("path"):
                            _dedupe_append(self.files_touched, str(edit.get("path")))
                            _dedupe_append(self.changed_files, str(edit.get("path")))
            if tool.startswith("browser_"):
                _dedupe_append(self.browser_checks, tool)
            preview = str(tr.get("output_preview") or "")
            if "not valid JSON" in preview:
                self.malformed_tool_calls += 1

    async def finish(
        self,
        *,
        status: str = "completed",
        priming_telemetry: dict[str, Any] | None = None,
    ) -> None:
        # Seal any pending per-call usage delta before we read the
        # totals. The last chunk of the last call may have been the
        # only place usage was reported.
        self._seal_pending_usage()

        # Cost lookup — local-hosted models return (0.0, 0.0) so the
        # row records $0 cost; cloud models surface real spend. We
        # lazy-import to keep the ledger module light during tests.
        try:
            from augmentum.fabric.cost_table import lookup_cost
            in_per_tok, out_per_tok = lookup_cost(self.model)
        except Exception:
            in_per_tok, out_per_tok = 0.0, 0.0
        input_cost_usd = self.prompt_tokens_total * in_per_tok
        output_cost_usd = self.completion_tokens_total * out_per_tok

        # Verification-spine summary (observational — never blocks). One
        # durable event so replay/inspector surfaces see it inline, plus
        # the same dict in metrics_json for cheap aggregation across runs.
        oracle_summary = oracle_telemetry.summarize(
            wrote=bool(self.changed_files),
            last_write_seq=self.last_write_seq,
            oracle_calls=self.oracle_calls,
        )
        try:
            self.seq += 1
            await self.store.record_event(
                run_id=self.run_id,
                seq=self.seq,
                event_type="oracle_summary",
                phase="closeout",
                status="oracle_summary",
                payload=oracle_summary,
                user_id=self.user_id,
            )
        except Exception:
            # Telemetry must never break turn close; the metrics_json
            # copy below still lands via finish_run.
            pass

        # Citation ledger: bulk-persist the turn's evidence rows (claim→proof
        # provenance). Best-effort inside save_citations — never breaks close.
        if self.citations:
            await citations.save_citations(
                self.store._conn,  # noqa: SLF001 — same-module store owns the conn
                turn_run_id=self.run_id,
                user_id=self.user_id,
                workspace_id=self.workspace_id,
                citations=list(self.citations),
            )

        await self.store.finish_run(
            run_id=self.run_id,
            status=status,
            summary={
                "strategy": self.strategy,
                "first_event_at": self.first_event_at,
                "first_useful_action_at": self.first_useful_action_at,
                "iterations": self.iterations,
                "tool_calls": self.tool_calls,
                "parallel_waves": self.parallel_waves,
                "retries": self.retries,
                "no_response_events": self.no_response_events,
                "empty_native_content": self.empty_native_content,
                "malformed_tool_calls": self.malformed_tool_calls,
                "commands_run": self.commands_run,
                "files_touched": self.files_touched,
                "tests_run": self.tests_run,
                "browser_checks": self.browser_checks,
                "finish_reason": self.finish_reason,
                "fallback_reason": self.fallback_reason,
                "checkpoint_id": self.checkpoint_id,
                "changed_files": self.changed_files,
                "priming_telemetry": priming_telemetry or {},
                "input_cost_usd": input_cost_usd,
                "output_cost_usd": output_cost_usd,
                "cost_model_id": self.model,
                "closeout": {
                    "changed_files": self.changed_files,
                    "commands_run": self.commands_run,
                    "tests_run": self.tests_run,
                    "browser_checks": self.browser_checks,
                    "checkpoint_id": self.checkpoint_id,
                    "known_gaps": [],
                },
                "metrics": {
                    "visible_answer": self.first_useful_action_at is not None,
                    "first_useful_action_latency_ms": (
                        int((self.first_useful_action_at - self.started_at) * 1000)
                        if self.first_useful_action_at is not None
                        else None
                    ),
                    "tool_call_parse_success_rate": (
                        1.0
                        if self.tool_calls and not self.malformed_tool_calls
                        else (
                            max(0.0, (self.tool_calls - self.malformed_tool_calls) / self.tool_calls)
                            if self.tool_calls
                            else None
                        )
                    ),
                    "verification_coverage": bool(self.tests_run or self.browser_checks),
                    "oracle": oracle_summary,
                    "max_prompt_tokens": self.max_prompt_tokens,
                    "last_prompt_tokens": self.last_prompt_tokens,
                    "compactions": self.compactions,
                    "token_snapshots": self.token_snapshots,
                    "prompt_tokens_total": self.prompt_tokens_total,
                    "completion_tokens_total": self.completion_tokens_total,
                    "reasoning": self.reasoning_meter.summary(),
                },
            },
        )
