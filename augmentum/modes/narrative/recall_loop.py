"""Iterative tool-execution loop for narrative recall.

Wraps a backend's ``chat_stream`` so the model can call ``recall_*``
verbs mid-turn. Pure orchestration — has no opinion about which tools
exist, only knows how to:

1. Stream one inner backend response, buffering content + tool_call deltas
2. When the inner stream ends, decide:
   - if no recall tool_calls were emitted → flush buffered content as
     final output to the outer consumer and terminate
   - if recall tool_calls were emitted → execute each one against the
     persistence layer, append an assistant message (with the
     tool_calls) and one tool message per result to the working history,
     emit a meta chunk per tool execution so the UI can render
     "checking my notes…", then re-call chat_stream with the updated
     history
3. Cap at ``max_iters`` to prevent runaway recall-only loops

Why a separate module (vs. inlining in narrative/handler.py): keeping
the loop pure makes it testable with a mock backend without needing to
spin up the full handler / engine stack. ``handler.py`` decides WHEN to
use the loop (feature flag + dispatcher injection); this module owns HOW
the loop runs.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from augmentum.models.base import (
    InternalChatRequest,
    InternalStreamChunk,
    Message,
)
from augmentum.modes.narrative.recall_schemas import (
    RECALL_TOOL_NAMES,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class _TurnMetrics:
    """Per-turn accounting so the loop can emit one structured summary
    on termination. Mirrors the ``engine_perf`` event shape used by
    llama_server_manager so dashboards parse both the same way.
    """

    iterations: int = 0
    tool_calls_total: int = 0
    tool_calls_per_name: dict[str, int] = field(default_factory=dict)
    tool_latency_ms_total: int = 0
    tool_latency_ms_max: int = 0
    tool_result_chars_total: int = 0
    tool_dispatch_errors: int = 0
    non_recall_tool_calls_dropped: int = 0
    max_iters_hit: bool = False
    finish_reason: str = ""
    pregate_drops: int = 0
    pregate_dropped_chars: int = 0
    # Loop-health ladder (adapted from coder mode's guard suite) —
    # narrative's failure is the inverse of coder's: a tiny tool budget
    # burned on reads so the STORY never gets written. Rungs:
    #   read_dup_nudges       — identical read re-issued this turn (in-band note)
    #   final_iter_synthesized — last budgeted call ran tool-free (forced prose)
    read_dup_nudges: int = 0
    final_iter_synthesized: bool = False

    def record_call(self, name: str, latency_ms: int, result_chars: int, errored: bool) -> None:
        self.tool_calls_total += 1
        self.tool_calls_per_name[name] = self.tool_calls_per_name.get(name, 0) + 1
        self.tool_latency_ms_total += latency_ms
        if latency_ms > self.tool_latency_ms_max:
            self.tool_latency_ms_max = latency_ms
        self.tool_result_chars_total += result_chars
        if errored:
            self.tool_dispatch_errors += 1

    def as_payload(self) -> dict:
        avg_latency = (
            self.tool_latency_ms_total // self.tool_calls_total
            if self.tool_calls_total else 0
        )
        return {
            "iterations":               self.iterations,
            "tool_calls":               self.tool_calls_total,
            "tool_calls_per_name":      dict(self.tool_calls_per_name),
            "tool_latency_ms_avg":      avg_latency,
            "tool_latency_ms_max":      self.tool_latency_ms_max,
            "tool_result_chars_total":  self.tool_result_chars_total,
            "tool_dispatch_errors":     self.tool_dispatch_errors,
            "non_recall_dropped":       self.non_recall_tool_calls_dropped,
            "max_iters_hit":            self.max_iters_hit,
            "finish_reason":            self.finish_reason,
            "pregate_drops":            self.pregate_drops,
            "pregate_dropped_chars":    self.pregate_dropped_chars,
            "read_dup_nudges":          self.read_dup_nudges,
            "final_iter_synthesized":   self.final_iter_synthesized,
        }


# A dispatcher takes (tool_name, raw_arguments) and returns the tool's
# string output. Provided by the handler via a closure that binds
# persistence + session_id + user_id so this module stays decoupled
# from the persistence layer.
RecallDispatcher = Callable[[str, str | dict | None], Awaitable[str]]

# Pre-call gate budget (chars). The first slice of each iteration's CONTENT
# is held until we know whether this pass is heading into an internal tool
# call. Models with a strong agentic register (DeepSeek especially) announce
# their tool plan in visible prose BEFORE the call — "Let me check whether
# there's a lorebook entry about X…", sometimes a whole paragraph of it —
# and since content otherwise streams immediately (the 2026-06-30 anti-blob
# fix), that meta paragraph lands in the story pane irreversibly and gets
# saved into history (Matt's live report, 2026-07-15).
#
# Disambiguation is by CALL TYPE, not length: prose followed by a READ call
# (recall_*, lorebook check/search) is plan narration — dropped; prose
# followed by a WRITE call (lorebook create/update/delete) is story that
# just established the fact being recorded — flushed. Past this budget the
# gate opens and everything streams live exactly as before (a fixed budget
# can't tell a long preamble from a story, so beyond it the conduct
# directive owns the behavior). Suppression is always surfaced via a meta
# chunk + metrics, never silent.
_PREGATE_MAX_CHARS = 600


@dataclass(slots=True)
class _ToolCallAcc:
    """Accumulator for one tool_call emitted by the model (streamed in pieces)."""

    id: str = ""
    name: str = ""
    arguments: str = ""

    def merge_delta(self, delta: dict) -> None:
        """Merge one tool_calls delta dict from a stream chunk.

        Streaming providers send tool_call args char-by-char in
        successive deltas; only ``id`` and ``name`` are set on the
        first delta. Everything else is incremental.
        """
        if not self.id and delta.get("id"):
            self.id = str(delta["id"])
        fn = delta.get("function") or {}
        if not self.name and fn.get("name"):
            self.name = str(fn["name"])
        chunk_args = fn.get("arguments")
        if chunk_args:
            self.arguments += str(chunk_args)

    def to_assistant_part(self) -> dict:
        """Render as the dict shape that goes onto Message.tool_calls."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments or "{}",
            },
        }


def _coalesce_tool_call_deltas(
    accumulators: dict[int, _ToolCallAcc],
    deltas: list[dict],
) -> None:
    """Fold a list of tool_call delta dicts into the per-index accumulators.

    OpenAI's streaming protocol identifies which tool_call a delta
    belongs to via ``index``. Most local backends preserve that, but
    some streams omit it after the first chunk — fall back to the only
    index seen so far in that case.
    """
    for delta in deltas:
        if not isinstance(delta, dict):
            continue
        idx_raw = delta.get("index")
        try:
            idx = int(idx_raw) if idx_raw is not None else 0
        except (TypeError, ValueError):
            idx = 0
        acc = accumulators.setdefault(idx, _ToolCallAcc())
        acc.merge_delta(delta)


def _synthesis_directive(metrics: _TurnMetrics) -> Message:
    """Build the tool-free synthesis directive appended on the final
    budgeted iteration (and the post-cap fallback).

    Seeded from what the turn actually gathered so the model knows its
    lookups are done. Deliberately terse and non-meta — the model must
    write STORY, not narrate that it finished retrieving.
    """
    per = ", ".join(
        f"{name}×{count}"
        for name, count in metrics.tool_calls_per_name.items()
    ) or "none"
    body = (
        "SYNTHESIS STEP — no tools are available on this call. You have "
        f"already gathered what you need this turn ({metrics.tool_calls_total} "
        f"tool call(s): {per}). Do not mention tools, lookups, or your "
        "process. Write the next part of the story now, as prose, using the "
        "context you gathered."
    )
    return Message(role="system", content=body)


def _read_signature(name: str, arguments: str | None) -> str:
    """Canonical (tool, args) key for windowed duplicate-read detection.

    Whitespace-normalized so ``{"name":"Ana"}`` and ``{"name": "Ana"}``
    collapse to one signature.
    """
    return f"{name}::{' '.join((arguments or '').split())}"


async def stream_with_recall_tools(
    request: InternalChatRequest,
    *,
    backend,
    dispatcher: RecallDispatcher,
    max_iters: int = 3,
    internal_tool_names: frozenset[str] | None = None,
    internal_write_names: frozenset[str] | None = None,
) -> AsyncIterator[InternalStreamChunk]:
    """Run the iterative narrative-tool loop.

    Yields stream chunks for the caller (handler) to forward to its
    own client. Internal tool_calls (recall + lorebook) are intercepted
    inside the loop and NOT yielded as-is; instead they surface as
    ``augmentum`` meta chunks so the UI can render status affordances.

    Non-internal tool_calls (e.g. if the request also carried other
    tools) are passed through unchanged.

    Termination:
      - Inner response ends with no internal tool_calls → flush
        buffered content + done chunk → return
      - Inner response ends with internal tool_calls → execute them,
        append messages, recurse
      - ``max_iters`` reached while still emitting internal calls →
        force-terminate with a meta warning so the user/log sees it,
        then yield whatever final content was emitted on the last
        iteration
    """
    # Resolve the set of tool names this loop intercepts.
    _internal_names = internal_tool_names if internal_tool_names is not None else RECALL_TOOL_NAMES
    # Mutating internal tools (lorebook create/update/delete). Prose held by
    # the pre-call gate is FLUSHED when the pass ends in one of these (story
    # that established the fact) and DROPPED for read-only passes (plan
    # narration). None → empty set → every internal call counts as a read.
    _write_names = internal_write_names or frozenset()

    # Work on a shallow copy of the messages list so we can append
    # assistant + tool messages without mutating the caller's request.
    messages: list[Message] = list(request.messages)
    metrics = _TurnMetrics()
    turn_started_ns = time.monotonic_ns()
    # Ladder state (windowed within the turn).
    seen_read_sigs: set[str] = set()

    for iteration in range(max_iters):
        metrics.iterations = iteration + 1
        # On the FINAL budgeted iteration, strip tools entirely
        # (tool_choice="none") and append a synthesis directive so the
        # model physically cannot spend its last call shopping for tools —
        # a read-loop is forced to produce prose instead of ending the turn
        # empty (the tool_loop_cap / 0-content failure seen with GLM +
        # inkling). Only when there are >=2 iterations, so a max_iters=1
        # config still gets one real tool pass. See the loop-health mapping
        # in docs/recurring-regressions.md.
        is_final_iter = max_iters >= 2 and iteration == max_iters - 1
        if is_final_iter:
            synth_messages = list(messages)
            synth_messages.append(_synthesis_directive(metrics))
            inner_request = _replace_messages(
                request, synth_messages, strip_tools=True,
            )
            metrics.final_iter_synthesized = True
        else:
            # Prepare the inner request — same as the outer except messages
            # may have grown by tool-message rounds.
            inner_request = _replace_messages(request, messages)

        content_parts: list[str] = []
        tool_acc: dict[int, _ToolCallAcc] = {}
        last_finish_reason: str | None = None
        # Chunks that carry tool_call deltas are held back (the model is
        # dictating a tool call, not visible content). Everything else
        # streams to the user IMMEDIATELY — mirrors the passthrough NATIVE
        # resolver (_resolve_tool_calls_streaming): content_delta and
        # tool_calls arrive as separate deltas, so we never buffer prose
        # to "wait and see". (2026-06-30: was buffer-then-flush, which
        # dumped the whole narrative turn as one blob at end-of-stream.)
        # EXCEPTION (2026-07-15): the first _PREGATE_MAX_CHARS of content
        # per iteration are held until we know they aren't a tool-plan
        # preamble — see the constant's comment for the defect class.
        toolcall_chunks: list[InternalStreamChunk] = []
        pregate_open = True
        pregate_chunks: list[InternalStreamChunk] = []
        pregate_chars = 0

        async for chunk in backend.chat_stream(inner_request):
            # Pull tool_call deltas out of the augmentum side-channel.
            tc_deltas = None
            if chunk.augmentum and isinstance(chunk.augmentum, dict):
                tc_deltas = chunk.augmentum.get("tool_calls")

            # Always track content for the final assistant message history.
            if chunk.content_delta:
                content_parts.append(chunk.content_delta)
            if chunk.finish_reason:
                last_finish_reason = chunk.finish_reason

            if tc_deltas:
                _coalesce_tool_call_deltas(tool_acc, tc_deltas)
                # Hold the tool-call mechanics (and any content bundled in
                # the same chunk) back rather than streaming it. In the
                # recall pattern the model emits the recall call BEFORE any
                # prose, so there is normally no visible content here.
                toolcall_chunks.append(chunk)
                continue

            if pregate_open:
                # Stream-end markers ride with the gated content so a
                # flush can't emit content AFTER a done chunk; pure
                # thinking/status chunks pass through live.
                if chunk.content_delta or chunk.done or chunk.finish_reason:
                    pregate_chunks.append(chunk)
                    pregate_chars += len(chunk.content_delta)
                    if pregate_chars >= _PREGATE_MAX_CHARS:
                        # Real prose, not a preamble — open the gate.
                        for held in pregate_chunks:
                            yield held
                        pregate_chunks = []
                        pregate_open = False
                    continue
                yield chunk
                continue

            # Gate already open — stream it NOW.
            yield chunk

        # Inner stream complete — decide path.
        recall_calls = [
            acc for acc in tool_acc.values()
            if acc.name in _internal_names
        ]
        other_calls = [
            acc for acc in tool_acc.values()
            if acc.name and acc.name not in _internal_names
        ]

        if recall_calls and pregate_chunks:
            if any(acc.name in _write_names for acc in recall_calls):
                # WRITE call — the held prose is story that just
                # established the fact being recorded. Flush it, with
                # stream-end markers stripped since the loop continues
                # into the tool round.
                from dataclasses import replace as _chunk_replace
                for held in pregate_chunks:
                    yield _chunk_replace(held, done=False, finish_reason=None)
            else:
                # READ-only pass — the held content was a tool-plan
                # preamble ("Let me check the lorebook…"). Drop it from
                # the visible stream and the saved story. It stays in the
                # WORKING history below (content_parts) so the provider
                # sees a faithful assistant turn. Re-emit any thinking
                # that rode in the held chunks; suppression is surfaced
                # via meta + metrics, never silent.
                dropped = sum(len(c.content_delta) for c in pregate_chunks)
                metrics.pregate_drops += 1
                metrics.pregate_dropped_chars += dropped
                for held in pregate_chunks:
                    if held.thinking_delta:
                        yield InternalStreamChunk(thinking_delta=held.thinking_delta)
                yield InternalStreamChunk(
                    content_delta="",
                    augmentum={
                        "status": "tool_preamble_suppressed",
                        "chars": dropped,
                        "iteration": iteration + 1,
                    },
                )
            pregate_chunks = []

        if not recall_calls:
            # Short reply that never filled the gate — deliver it before
            # any other terminal handling (it is the story).
            for held in pregate_chunks:
                yield held
            pregate_chunks = []
            # Content already streamed live above. If the model ALSO made
            # non-recall tool_calls, forward those held-back chunks so the
            # outer handler / client still sees them (prior behavior).
            if other_calls:
                for buf in toolcall_chunks:
                    yield buf
            else:
                # No tool calls of any kind survived (e.g. name-less delta
                # fragments where the name delta was lost). Don't leak partial
                # tool_call deltas to the client, but DO recover any prose that
                # rode alongside them — this is a terminal path (we return
                # below), so otherwise that content is silently lost.
                for buf in toolcall_chunks:
                    if buf.content_delta:
                        yield InternalStreamChunk(content_delta=buf.content_delta)
            metrics.finish_reason = last_finish_reason or "stop"
            _emit_turn_summary(metrics, turn_started_ns)
            return

        # We have at least one recall tool_call. Emit a meta chunk
        # PER call so the UI can render a status hint, then execute.
        # We deliberately do NOT yield the buffered chunks here —
        # tool_call deltas are internal mechanics from the consumer's
        # POV and content alongside them is usually empty (a model
        # that emits prose + tool_calls in the same chunk has decided
        # to do both, and the prose will be re-emitted after the tool
        # round in the model's next pass anyway).

        # Append assistant message with the tool_calls — needed so the
        # next backend.chat_stream sees the right history shape.
        assistant_msg_content = "".join(content_parts)
        assistant_msg = Message(
            role="assistant",
            content=assistant_msg_content,
            tool_calls=[acc.to_assistant_part() for acc in tool_acc.values() if acc.name],
        )
        messages.append(assistant_msg)

        # Execute each recall, append a tool message per result, emit
        # the meta chunk for UI visibility.
        for acc in recall_calls:
            errored = False
            call_started_ns = time.monotonic_ns()
            try:
                # Parsed args for the meta surface; if invalid, the
                # dispatcher itself will return a clean error string.
                try:
                    args_preview = json.loads(acc.arguments) if acc.arguments else {}
                except (json.JSONDecodeError, TypeError):
                    args_preview = {"_raw": (acc.arguments or "")[:120]}

                tool_output = await dispatcher(acc.name, acc.arguments)
            except Exception as exc:  # noqa: BLE001
                # Dispatcher failure shouldn't kill the turn; feed the
                # error back to the model as the tool result so it can
                # decide what to do (recover, apologize, etc.).
                errored = True
                log.warning(
                    "narrative_recall_dispatch_failed",
                    tool=acc.name, error=str(exc)[:200],
                )
                tool_output = f"recall_tool_error: {type(exc).__name__}: {str(exc)[:200]}"
                args_preview = {"_dispatch_error": True}

            call_elapsed_ms = max(
                0, (time.monotonic_ns() - call_started_ns) // 1_000_000,
            )
            metrics.record_call(
                acc.name, call_elapsed_ms, len(tool_output), errored,
            )

            # Read-dup rung (adapted from coder's duplicate_call_nudge):
            # an identical READ re-issued this turn can't surface anything
            # new. Append an in-band redirect to the tool result (no extra
            # message, no KV churn) so the model stops re-checking and
            # writes. Writes are exempt — recording the same fact twice is
            # a lorebook concern the dispatcher dedupes, not a loop stall.
            is_write = acc.name in _write_names
            sig = _read_signature(acc.name, acc.arguments)
            if not is_write and not errored:
                if sig in seen_read_sigs:
                    metrics.read_dup_nudges += 1
                    tool_output = (
                        tool_output.rstrip()
                        + "\n\n[Loop note: you already retrieved this exact "
                        "information earlier this turn. Re-checking cannot "
                        "produce anything new — use what you have and "
                        "continue the story.]"
                    )
                seen_read_sigs.add(sig)

            yield InternalStreamChunk(
                content_delta="",
                augmentum={
                    "status": "narrative_tool_used",
                    "tool": acc.name,
                    "args": args_preview,
                    "result_preview": tool_output[:400],
                    "iteration": iteration + 1,
                    "latency_ms": call_elapsed_ms,
                },
            )

            messages.append(Message(
                role="tool",
                content=tool_output,
                tool_call_id=acc.id,
            ))

        # If the model also called other (non-recall) tools, surface
        # them via meta so the outer client knows they were dropped —
        # otherwise debugging the silence is opaque.
        if other_calls:
            metrics.non_recall_tool_calls_dropped += len(other_calls)
            yield InternalStreamChunk(
                content_delta="",
                augmentum={
                    "status": "non_recall_tool_call_ignored",
                    "tools": [
                        {"name": c.name, "id": c.id} for c in other_calls
                    ],
                },
            )

        # Loop continues — next iteration re-streams with the updated
        # message history.

    # Reached only in the max_iters<2 edge (single-shot config) or if the
    # final tool-free pass itself still emitted a recall call — do one last
    # tool-free synthesis so the turn never ends empty. Surface it.
    log.warning(
        "narrative_recall_max_iters_reached",
        max_iters=max_iters,
        final_message_count=len(messages),
    )
    metrics.max_iters_hit = True

    # One forced tool-free synthesis call so the turn never ends empty.
    synth_messages = list(messages)
    synth_messages.append(_synthesis_directive(metrics))
    fallback_request = _replace_messages(
        request, synth_messages, strip_tools=True,
    )
    metrics.final_iter_synthesized = True
    produced = False
    yield InternalStreamChunk(
        content_delta="",
        augmentum={
            "status": "recall_tool_loop_synthesizing",
            "max_iters": max_iters,
            "note": (
                "Reached the recall-tool budget without a written reply — "
                "forcing a tool-free pass to write the story."
            ),
        },
    )
    async for chunk in backend.chat_stream(fallback_request):
        # Strip any stray tool_call deltas (tools are off; ignore leakage).
        if chunk.augmentum and isinstance(chunk.augmentum, dict) and chunk.augmentum.get("tool_calls"):
            continue
        if chunk.content_delta:
            produced = True
        yield chunk

    metrics.finish_reason = (
        "tool_loop_cap_synthesized" if produced else "tool_loop_cap"
    )
    _emit_turn_summary(metrics, turn_started_ns)
    if not produced:
        # Even the tool-free pass wrote nothing — genuine model failure.
        # Emit the terminal marker so the client isn't left hanging.
        yield InternalStreamChunk(
            content_delta="",
            augmentum={
                "status": "recall_tool_loop_max_iters",
                "max_iters": max_iters,
                "note": (
                    "The model produced no visible reply even with tools "
                    "disabled. Increase narrative_recall_tools_max_iters or "
                    "simplify the turn."
                ),
            },
            done=True,
            finish_reason="tool_loop_cap",
        )


def _emit_turn_summary(metrics: _TurnMetrics, turn_started_ns: int) -> None:
    """One structured log event per turn — the source of truth for the
    measurement plan ('do recall tools actually help?'). Mirrors the
    engine_perf event shape so the same parsers handle both.
    """
    total_ms = max(0, (time.monotonic_ns() - turn_started_ns) // 1_000_000)
    payload = metrics.as_payload()
    payload["turn_total_ms"] = total_ms
    log.info("narrative_recall_turn", **payload)


def _replace_messages(
    request: InternalChatRequest,
    messages: list[Message],
    *,
    strip_tools: bool = False,
) -> InternalChatRequest:
    """Return a copy of ``request`` with a new messages list.

    Cheap structural clone — all the per-request fields are forwarded
    so the inner call carries the same model/temperature/tools/etc as
    the outer. With ``strip_tools=True`` the copy also drops the tool
    schemas and sets ``tool_choice="none"`` — used for the forced
    synthesis pass so the model can only write prose.
    """
    # InternalChatRequest is a dataclass; shallow attribute copy with
    # the new message list is enough.
    from dataclasses import fields, replace as dataclass_replace

    changes: dict = {"messages": messages}
    if strip_tools:
        changes["tools"] = None
        changes["tool_choice"] = "none"

    try:
        return dataclass_replace(request, **changes)
    except TypeError:
        # Conservative fallback if a future version of the dataclass
        # adds non-init fields that ``replace`` can't handle.
        kwargs = {f.name: getattr(request, f.name) for f in fields(request) if f.init}
        kwargs.update(changes)
        return type(request)(**kwargs)
