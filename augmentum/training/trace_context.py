"""Turn-scoped capture context for backend-boundary training capture.

Why this exists
---------------
Training-trace capture used to live only at the proxy layer
(``augmentum/training/capture.py``), which reads the ORIGINAL pre-handler
``request.messages``. That misses the truth for most modes:

* **analytical** runs its tool loop on local per-phase copies,
* **narrative** builds a separate augmented request via ``dataclass_replace``
  (see the ``InternalChatRequest`` docstring in ``models/base.py``),
* **coder** / **self-edit** run their own agent loops that bypass the proxy
  hook entirely (``run_broker`` / ``selfedit.native_loop``).

The ONE place every mode converges on a fully-resolved request — composed
system prompt, full message interleave, tools — is ``ModelBackend.chat`` /
``chat_stream`` (``models/base.py:758``). This module holds the per-TURN
context that correlates the individual backend calls of one logical turn into
a single trace.

Flow
----
1. A turn entry point opens a :func:`capture_turn` scope.
2. The ``CapturingBackend`` wrapper (wired in a later step) records each
   resolved request into the active context via :meth:`CaptureContext.record`.
3. The scope writes one turn trace on exit.

Gating
------
Inert unless ``settings.training_capture_enabled`` (and, when
``training_capture_user_id`` is set, the user matches). When off,
:func:`capture_turn` never sets a context, so :func:`current_capture_context`
returns ``None`` and the wrapper is a zero-cost passthrough.

Snapshot semantics
------------------
:func:`serialize_request` MUST be called at backend-invocation time, because
several modes (companion native loop, passthrough ``_execute_and_append``)
mutate ``request.messages`` IN PLACE between iterations — capturing a reference
would record the final state N times instead of each call's real input.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from augmentum.prompts.primer import tag_for
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.base import InternalChatRequest

log = get_logger(__name__)

# On-disk turn-trace shape version. Bump when the JSONL shape changes so the
# downstream transform (trace -> skeleton -> row) can branch on it. v1 was the
# proxy-layer single-request shape in capture.py; v2 was the raw multi-call
# turn; v3 adds training-ready fields (flattened chain, system_prompt, tools).
TRACE_SCHEMA_VERSION = 3


# --------------------------------------------------------------------------
# Message serialization (self-contained; mirrors capture.py so the on-disk
# message shape stays consistent across both capture paths during migration).
# --------------------------------------------------------------------------

# Tool-loop synthesis prompts injected by the harness — strip from traces since
# they're internals, not real conversation turns.
_SYNTH_MARKERS = (
    "synthesize the tool",
    "synthesize the above",
    "synthesize the results",
    "now respond to the user",
    "based on the tool results",
    "use the information above",
    "use the tool results above",
    "incorporate the tool results",
    "do not repeat the raw tool output",
)


def _is_synthesis_message(content: str) -> bool:
    if not content:
        return False
    head = content[:120].lower()
    return any(marker in head for marker in _SYNTH_MARKERS)


def _serialize_message(msg: object) -> dict | None:
    """Convert one ``Message`` into a clean trace dict, or ``None`` to drop it.

    Preserves the fields a training row needs: ``tool_calls`` (with parsed
    arguments), ``thinking``, and ``tool_call_id`` for tool-result turns.
    """
    role = getattr(msg, "role", None)
    if role is None:
        return None
    content = getattr(msg, "content", None) or ""

    # Drop harness synthesis prompts (they ride as user turns).
    if role == "user" and _is_synthesis_message(content):
        return None

    entry: dict = {"role": role, "content": content}

    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        cleaned: list[dict] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
            name = func.get("name", "") or tc.get("name", "")
            args_raw = func.get("arguments", tc.get("arguments", "{}"))
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except (json.JSONDecodeError, TypeError):
                args = args_raw
            cleaned.append({"id": tc.get("id", ""), "name": name, "arguments": args})
        if cleaned:
            entry["tool_calls"] = cleaned

    thinking = getattr(msg, "thinking", None)
    if thinking:
        entry["thinking"] = thinking

    tool_call_id = getattr(msg, "tool_call_id", None)
    if tool_call_id:
        entry["tool_call_id"] = tool_call_id

    return entry


def serialize_request(messages: list) -> tuple[str, list[dict]]:
    """Split a resolved message list into ``(system_prompt, chain)``.

    The FIRST system message becomes the full (un-hashed) ``system_prompt`` —
    this is where the memory / state / tool injections actually live, and the
    whole reason this path exists. Any *additional* system messages (mid-
    conversation injections, e.g. narrative STATE blocks) are kept in the
    chain so nothing is lost. Everything else is serialized in order.

    Call this at backend-invocation time — see the module docstring's snapshot
    note.
    """
    system_prompt = ""
    chain: list[dict] = []
    for msg in messages:
        role = getattr(msg, "role", None)
        if role == "system":
            content = getattr(msg, "content", "") or ""
            if not system_prompt:
                system_prompt = content
                continue  # first system → the primary primer/system block
            # subsequent system messages: keep them in the chain (real
            # mid-conversation injections), don't silently drop.
        entry = _serialize_message(msg)
        if entry is not None:
            chain.append(entry)
    return system_prompt, chain


# --------------------------------------------------------------------------
# Turn context
# --------------------------------------------------------------------------


@dataclass
class CaptureContext:
    """One logical turn's worth of backend calls, accumulated in order."""

    turn_id: str
    user_id: str
    session_id: str
    mode: str
    surface_tag: str
    is_background: bool = False
    calls: list[dict] = field(default_factory=list)
    # Tool schemas the turn made available OUTSIDE the normal request.tools
    # path — specifically the passthrough direct-invoke path, which runs a tool
    # deterministically and synthesizes its result without ever setting
    # request.tools. Noted here so the trace still records what was available.
    noted_tools: list[dict] = field(default_factory=list)
    # Final response/thinking supplied by a handler when the backend-boundary
    # hook can't see it — e.g. the voice path's native loop emits its answer as
    # loop EVENTS, not backend content, so the captured call records no text.
    # Used as the trace's final_response only when the flattened chain is empty.
    noted_final_response: str = ""
    noted_final_thinking: str = ""

    def record(
        self,
        *,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict] | None,
        response_text: str,
        response_thinking: str,
        model: str,
        sampling: dict | None = None,
        flags: dict | None = None,
    ) -> None:
        """Append one backend call's resolved request + response."""
        entry: dict[str, Any] = {
            "seq": len(self.calls) + 1,
            "model": model or "",
            "system_prompt": system_prompt,
            "messages": messages,
            "tools": tools,
            "response_text": response_text or "",
            "response_thinking": response_thinking or "",
        }
        if sampling:
            entry["sampling"] = sampling
        if flags:
            entry["flags"] = flags
        self.calls.append(entry)


_CAPTURE_CTX: ContextVar[CaptureContext | None] = ContextVar(
    "augmentum_capture_ctx", default=None
)


def current_capture_context() -> CaptureContext | None:
    """The capture context for the current async task, or ``None`` when capture
    is off / not inside a :func:`capture_turn` scope."""
    return _CAPTURE_CTX.get()


def note_tool_schemas(schemas: list[dict] | None) -> None:
    """Record tool schemas that won't ride on ``request.tools``.

    The passthrough direct-invoke path executes a tool deterministically and
    synthesizes its output into the prompt, so the backend call carries no
    ``tools`` — the capture hook would otherwise see zero schemas for a turn
    that very much used a tool. Call this with the direct-invoked tools'
    native schemas so the written trace's ``tool_schemas`` / ``tools_used``
    reflect reality. No-op when capture is inactive.
    """
    if not schemas:
        return
    ctx = _CAPTURE_CTX.get()
    if ctx is None:
        return
    for s in schemas:
        if isinstance(s, dict):
            ctx.noted_tools.append(s)


def note_final_response(text: str, thinking: str = "") -> None:
    """Supply the turn's final answer when the backend hook can't capture it.

    The voice path runs the companion's native FC loop, which emits its final
    text as loop events rather than as backend content — so the captured
    backend call has an empty response. The route already has the assembled
    reply; noting it here lets the trace record a real final_response. No-op
    when capture is inactive; only used if the flattened chain is empty.
    """
    ctx = _CAPTURE_CTX.get()
    if ctx is None:
        return
    if text:
        ctx.noted_final_response = text
    if thinking:
        ctx.noted_final_thinking = thinking


def _capture_enabled(user_id: str) -> bool:
    """Synchronous gate. Reads the live settings singleton — no DB / event-loop
    reentry (unlike the legacy capture.py bootstrap)."""
    from augmentum.config import settings

    if not getattr(settings, "training_capture_enabled", False):
        return False
    want = getattr(settings, "training_capture_user_id", "") or ""
    # When a training-gen user is pinned, only capture that user's turns.
    return not (want and user_id and user_id != want)


def _publish_surface(mode: str) -> None:
    """Publish the turn's mode for native-primer serving — BEFORE the capture
    gate, so primer routing works with capture disabled."""
    try:
        from augmentum.prompts.native_serve import set_current_surface

        set_current_surface(mode)
    except Exception:  # pragma: no cover - never block a turn on this
        log.debug("surface_publish_failed", exc_info=True)


def _new_turn_id() -> str:
    return f"turn_{uuid.uuid4().hex[:16]}"


@contextlib.contextmanager
def capture_turn(
    *,
    user_id: str,
    session_id: str = "",
    mode: str = "",
    surface_tag: str = "",
    is_background: bool = False,
) -> Iterator[CaptureContext | None]:
    """Open a turn-scoped capture context.

    Yields the :class:`CaptureContext` when capture is active, or ``None`` (a
    pure no-op) when disabled, filtered out, or background. On exit, writes one
    turn trace iff any backend calls were recorded — so a turn that produced no
    model calls leaves no empty file noise.
    """
    _publish_surface(mode)
    if is_background or not _capture_enabled(user_id):
        yield None
        return

    ctx = CaptureContext(
        turn_id=_new_turn_id(),
        user_id=user_id,
        session_id=session_id or "",
        mode=mode or "",
        surface_tag=surface_tag or tag_for(mode or ""),
    )
    token = _CAPTURE_CTX.set(ctx)
    error = ""
    try:
        yield ctx
    except BaseException as exc:  # record the failure, then re-raise unchanged
        error = type(exc).__name__
        raise
    finally:
        _CAPTURE_CTX.reset(token)
        if ctx.calls:
            try:
                _write_turn_trace(ctx, error=error)
            except Exception:
                log.warning("training_turn_trace_write_failed", exc_info=True)


def _extract_sampling(request: object) -> dict | None:
    """Pull training-relevant sampling params from the request."""
    s: dict = {}
    for key in ("temperature", "top_p", "top_k", "max_tokens", "seed"):
        val = getattr(request, key, None)
        if val is not None:
            s[key] = val
    raw = getattr(request, "raw_options", None) or {}
    for key in ("reasoning_effort", "repeat_penalty", "frequency_penalty", "presence_penalty"):
        val = raw.get(key) if isinstance(raw, dict) else getattr(request, key, None)
        if val is not None:
            s[key] = val
    return s or None


def _extract_flags(request: object) -> dict | None:
    """Pull mode-trigger flags that affect output structure."""
    f: dict = {}
    think = getattr(request, "think", None)
    if think is not None:
        f["think"] = think
    if getattr(request, "voice_input", False):
        f["voice_input"] = True
    if getattr(request, "continue_last_assistant", False):
        f["continue_last_assistant"] = True
    kwargs = getattr(request, "chat_template_kwargs", None) or {}
    if isinstance(kwargs, dict):
        et = kwargs.get("enable_thinking")
        if et is not None:
            f["enable_thinking"] = et
    return f or None


def record_backend_call(
    request: InternalChatRequest,
    *,
    response_text: str = "",
    response_thinking: str = "",
    request_snapshot: tuple[str, list[dict]] | None = None,
) -> None:
    """Record one backend call into the active turn context (no-op if none).

    ``request_snapshot`` is the ``(system_prompt, chain)`` captured at
    backend-invocation time. Streaming callers MUST snapshot at entry (before
    awaiting) and pass it here at completion, because the loop mutates
    ``request.messages`` in place after the call. Non-streaming callers may
    omit it and let this serialize ``request.messages`` directly (nothing
    mutates it between call and return).
    """
    ctx = current_capture_context()
    if ctx is None:
        return
    # Background model calls (memory refresh, dream, extraction) ride the same
    # backend; exclude them from the user-turn trace.
    if getattr(request, "is_background_task", False):
        return

    if request_snapshot is not None:
        system_prompt, chain = request_snapshot
    else:
        system_prompt, chain = serialize_request(getattr(request, "messages", []) or [])

    sampling = _extract_sampling(request)
    flags = _extract_flags(request)

    ctx.record(
        system_prompt=system_prompt,
        messages=chain,
        tools=getattr(request, "tools", None),
        response_text=response_text,
        response_thinking=response_thinking,
        model=getattr(request, "model", "") or "",
        sampling=sampling,
        flags=flags,
    )


def _collect_tools_used(calls: list[dict]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for call in calls:
        for msg in call.get("messages", []):
            for tc in msg.get("tool_calls", []):
                name = tc.get("name", "")
                if name and name not in seen:
                    seen.add(name)
                    out.append(name)
    return out


def _collect_tools_available(calls: list[dict]) -> list[str]:
    """Extract tool names from schemas available across all calls."""
    seen: set[str] = set()
    out: list[str] = []
    for call in calls:
        for t in call.get("tools") or []:
            if isinstance(t, dict):
                func = t.get("function", t)
                name = func.get("name", "")
                if name and name not in seen:
                    seen.add(name)
                    out.append(name)
    return out


def _collect_tool_schemas(calls: list[dict]) -> list[dict]:
    """Return full tool schema objects from all calls (deduplicated).

    Multi-phase modes (analytical, coder) may inject different tools per
    call.  We collect the union so ``assemble_dataset.py`` sees every
    tool the model had access to during the turn.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for call in calls:
        for t in call.get("tools") or []:
            if not isinstance(t, dict):
                continue
            func = t.get("function", t)
            name = func.get("name", "")
            if name and name not in seen:
                seen.add(name)
                out.append(t)
    return out


def _merge_field(calls: list[dict], key: str) -> dict | None:
    """Return the named dict field from the first call that has one."""
    for call in calls:
        val = call.get(key)
        if val:
            return val
    return None


def _flatten_to_chain(calls: list[dict]) -> tuple[str, list[dict], str, str]:
    """Derive the training-ready message chain from the raw calls array.

    Returns ``(system_prompt, chain, final_response, final_thinking)`` where
    ``chain`` is the chronological message list (no system) and
    ``final_response`` / ``final_thinking`` are the last call's output.

    The last call's ``messages`` contains the accumulated chain BEFORE that
    call was made (handlers append each response before the next iteration).
    We append the last call's response to complete the sequence.
    """
    if not calls:
        return "", [], "", ""

    system_prompt = calls[0].get("system_prompt", "")
    # The user-facing final answer is the last call that actually produced
    # text/thinking. Multi-phase modes append calls that can return empty —
    # analytical's CONCLUDE is followed by a fire-and-forget auto-verify pass;
    # a tool loop's terminal iteration can be a bare tool-call with no text —
    # and using calls[-1] blindly would blank the trace's final_response. Fall
    # back to the literal last call only when nothing produced output at all.
    last = next(
        (
            c for c in reversed(calls)
            if (c.get("response_text") or "").strip()
            or (c.get("response_thinking") or "").strip()
        ),
        calls[-1],
    )
    chain = list(last.get("messages", []))
    resp = last.get("response_text", "")
    think = last.get("response_thinking", "")

    if resp or think:
        final_msg: dict[str, Any] = {"role": "assistant", "content": resp}
        if think:
            final_msg["thinking"] = think
        chain.append(final_msg)

    return system_prompt, chain, resp, think


# Training LANES (2026-07-03, Matt) — the seed model trains on CAPABILITIES,
# not surfaces, so the 16 surface tags fold into 5 training lanes:
#
#   :C  chat       — general assistance + reasoning (also absorbs analytical/
#                    agentic/builder/knowledge/cast: same capability, different
#                    harness scaffold, which the harness re-supplies at runtime)
#   :-  coder      — interleaved tool loops, channel discipline, verify habits
#   :N  narrative  — long-horizon persona endurance
#   :V  voice      — spoken register (short turns, no markdown)
#   :U  you        — autonomy: companion/tick/initiative (absorbs system + xr;
#                    the only lane whose data cannot come from reactive chat)
#
# :G game and :L stream stay OUTSIDE the 5 — they're the separate ``:G``/``:L``
# config lanes from the seed design, not seed-core data. :D direct stays
# excluded from every lane (external-harness pass-through, deliberately never
# companion/seed data).
#
# The fine-grained ``tag`` + raw ``mode`` remain on every trace, so lane
# assignment is always reversible if the fold changes.
_LANE_FOR_TAG: dict[str, str] = {
    ":C": ":C", ":A": ":C", ":T": ":C", ":W": ":C", ":K": ":C", ":R": ":C",
    ":-": ":-",
    ":N": ":N",
    ":V": ":V", ":Vp": ":V",
    ":B": ":U", ":S": ":U", ":X": ":U",
    ":G": ":G",
    ":L": ":L",
    ":D": ":D",
}

# Filesystem dir per LANE. chat/coder/narrative/voice keep their existing
# dirs (no corpus split); companion traces move to ``you/``; the folded
# surfaces' old dirs (analytical/agentic/builder/knowledge/xr/cast/system)
# simply stop growing — existing traces stay where they are.
_LANE_DIRS: dict[str, str] = {
    ":C": "chat",
    ":-": "coder",
    ":N": "narrative",
    ":V": "voice",
    ":U": "you",
    ":G": "game",
    ":L": "stream",
    ":D": "direct",
}


def _lane_for(tag: str) -> str:
    return _LANE_FOR_TAG.get(tag, ":C")


def _tag_dir(tag: str) -> str:
    return _LANE_DIRS.get(_lane_for(tag), "other")


def _write_turn_trace(ctx: CaptureContext, *, error: str = "") -> None:
    """Write one turn trace organized by tag for training-ready output.

    Writes to ``{training_capture_dir}/{tag_dir}/{date}.jsonl``.
    Each line is self-contained: the flattened ``chain`` is directly usable
    by ``assemble_dataset.py`` without needing to reconstruct from ``calls``.
    """
    from augmentum.config import settings

    now = datetime.now(UTC)
    tag = tag_for(ctx.mode)
    lane = _lane_for(tag)
    system_prompt, chain, final_response, final_thinking = _flatten_to_chain(
        ctx.calls
    )

    # Handler-noted fallback (voice native loop): the answer arrived as loop
    # events, so the captured backend call has no content. Use the noted reply
    # only when the chain produced nothing of its own.
    if not final_response and ctx.noted_final_response:
        final_response = ctx.noted_final_response
        final_thinking = final_thinking or ctx.noted_final_thinking
        fmsg: dict[str, Any] = {"role": "assistant", "content": final_response}
        if final_thinking:
            fmsg["thinking"] = final_thinking
        chain = [*chain, fmsg]

    chain_depth = sum(
        1 for msg in chain
        if msg.get("role") == "assistant" and msg.get("tool_calls")
    )

    # Merge request.tools-derived schemas with any directly-noted ones (the
    # direct-invoke path) so a deterministic-tool turn still records what ran.
    tool_schemas = _collect_tool_schemas(ctx.calls)
    _seen_schema = {
        (s.get("function", s) if isinstance(s, dict) else {}).get("name", "")
        for s in tool_schemas
    }
    for s in ctx.noted_tools:
        name = (s.get("function", s) if isinstance(s, dict) else {}).get("name", "")
        if name and name not in _seen_schema:
            _seen_schema.add(name)
            tool_schemas.append(s)
    tools_available = [
        (s.get("function", s) if isinstance(s, dict) else {}).get("name", "")
        for s in tool_schemas
    ]
    tools_available = [n for n in tools_available if n]

    trace: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "turn_id": ctx.turn_id,
        "timestamp": now.isoformat(),
        "user_id": ctx.user_id,
        "session_id": ctx.session_id,
        "mode": ctx.mode,
        "tag": tag,
        # Training LANE — the coarse capability bucket the seed trains on
        # (5 lanes fold the 16 surface tags; see _LANE_FOR_TAG). ``tag`` +
        # ``mode`` stay recorded so the fold is always reversible.
        "lane": lane,

        # Training-ready fields — usable directly by assemble_dataset.py
        "system_prompt": system_prompt,
        "chain": chain,
        "final_response": final_response,
        "final_thinking": final_thinking,
        "tools_available": tools_available,
        "tool_schemas": tool_schemas,
        "tools_used": _collect_tools_used(ctx.calls),
        "chain_depth": chain_depth,
        "model": ctx.calls[-1]["model"] if ctx.calls else "",
        "num_calls": len(ctx.calls),

        # Sampling & flags from the first backend call (mode defaults applied)
        "sampling": _merge_field(ctx.calls, "sampling"),
        "flags": _merge_field(ctx.calls, "flags"),

        # Raw calls preserved for debugging / multi-phase analysis
        "calls": ctx.calls,
        "error": error,
    }

    base_dir = Path(settings.training_capture_dir)
    tag_path = base_dir / _tag_dir(tag)
    tag_path.mkdir(parents=True, exist_ok=True)
    trace_file = tag_path / f"{now.strftime('%Y-%m-%d')}.jsonl"
    with open(trace_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")

    log.info(
        "training_turn_trace_captured",
        turn_id=ctx.turn_id,
        mode=ctx.mode,
        tag=tag,
        lane=lane,
        dir=_tag_dir(tag),
        num_calls=len(ctx.calls),
        chain_depth=chain_depth,
        tools=trace["tools_used"],
    )


# --------------------------------------------------------------------------
# Backend interception
# --------------------------------------------------------------------------
#
# We record at ``ModelBackend.chat`` / ``chat_stream`` — the one boundary every
# mode converges on — but DO NOT wrap the backend in a subclass: the codebase
# does ``isinstance(backend, OpenAIBackend)`` / ``LlamaCppBackend`` on the
# RESOLVED generation backend (analytical/tool_calling.py, passthrough/
# handler.py, model_manager.py) to pick the tool-call tier. A wrapper subclass
# would silently fail those checks. Instead we replace the two bound methods on
# the instance, leaving its class — and therefore isinstance — untouched.

_HOOK_SENTINEL = "_augmentum_capture_hooked"
_HOOK_ORIG = "_augmentum_capture_orig"


def install_capture_hook(backend: object) -> None:
    """Install training-trace capture on a backend instance (idempotent).

    Inert when no :func:`capture_turn` scope is active: the hot path is a single
    ``current_capture_context()`` read, so non-training calls and the
    latency-sensitive classifier slot are unaffected. Reversible via
    :func:`uninstall_capture_hook`.
    """
    if getattr(backend, _HOOK_SENTINEL, False):
        return
    orig_chat = backend.chat  # bound methods, captured before replacement
    orig_chat_stream = backend.chat_stream

    async def _chat(request: Any) -> Any:
        ctx = current_capture_context()
        if ctx is None or getattr(request, "is_background_task", False):
            return await orig_chat(request)
        # Snapshot BEFORE delegating — loops mutate request.messages in place.
        snapshot = serialize_request(getattr(request, "messages", []) or [])
        resp = await orig_chat(request)
        try:
            msg = getattr(resp, "message", None)
            record_backend_call(
                request,
                response_text=getattr(msg, "content", "") or "",
                response_thinking=getattr(msg, "thinking", "") or "",
                request_snapshot=snapshot,
            )
        except Exception:
            log.warning("capture_hook_chat_record_failed", exc_info=True)
        return resp

    async def _chat_stream(request: Any) -> Any:
        ctx = current_capture_context()
        if ctx is None or getattr(request, "is_background_task", False):
            async for chunk in orig_chat_stream(request):
                yield chunk
            return
        snapshot = serialize_request(getattr(request, "messages", []) or [])
        text_parts: list[str] = []
        think_parts: list[str] = []
        try:
            async for chunk in orig_chat_stream(request):
                delta = getattr(chunk, "content_delta", "")
                if delta:
                    text_parts.append(delta)
                tdelta = getattr(chunk, "thinking_delta", "")
                if tdelta:
                    think_parts.append(tdelta)
                yield chunk
        finally:
            try:
                record_backend_call(
                    request,
                    response_text="".join(text_parts),
                    response_thinking="".join(think_parts),
                    request_snapshot=snapshot,
                )
            except Exception:
                log.warning("capture_hook_stream_record_failed", exc_info=True)

    backend.chat = _chat  # type: ignore[attr-defined]
    backend.chat_stream = _chat_stream  # type: ignore[attr-defined]
    setattr(backend, _HOOK_ORIG, (orig_chat, orig_chat_stream))
    setattr(backend, _HOOK_SENTINEL, True)


def begin_capture(
    *,
    user_id: str,
    session_id: str = "",
    mode: str = "",
    is_background: bool = False,
    force_new: bool = False,
) -> tuple[CaptureContext | None, Any]:
    """Open a capture scope without a ``with`` block.

    Returns ``(ctx, token)`` — pass both to :func:`end_capture` in a
    ``finally`` block.  When capture is off or filtered, returns
    ``(None, None)`` and :func:`end_capture` is a no-op.

    Reentrant by default: a nested scope REUSES the active one (one trace per
    logical turn). Pass ``force_new=True`` ONLY when the nested work is a
    genuine SUB-AGENT whose output is not the parent's reasoning — e.g. an
    artifact creator's internal generation, which must be its OWN ``:W`` row
    and kept OUT of the orchestrator's chain (else the document body leaks in
    as if the orchestrator wrote it inline). Use :func:`begin_capture` for the
    same-turn engine (native loop), ``force_new`` for the tool's own LLM work.
    """
    _publish_surface(mode)
    if is_background or not _capture_enabled(user_id):
        return None, None
    # Reentrant (default): when a turn scope is already active (e.g. the voice
    # route opened one and the companion native loop nests inside), REUSE it
    # instead of shadowing. One trace per logical turn, owned by the OUTERMOST
    # scope — so its mode/session/notes (note_final_response) win, and the inner
    # begin/end are no-ops. ``token=None`` marks a non-owner. ``force_new``
    # bypasses this for sub-agent generation (see docstring).
    existing = _CAPTURE_CTX.get()
    if existing is not None and not force_new:
        return existing, None
    ctx = CaptureContext(
        turn_id=_new_turn_id(),
        user_id=user_id,
        session_id=session_id or "",
        mode=mode or "",
        surface_tag=tag_for(mode or ""),
        is_background=is_background,
    )
    token = _CAPTURE_CTX.set(ctx)
    return ctx, token


def end_capture(
    ctx: CaptureContext | None,
    token: Any,
    *,
    error: str = "",
) -> None:
    """Close a capture scope opened by :func:`begin_capture`.

    Writes the turn trace if any backend calls were recorded.  Safe to
    call with ``(None, None)`` — no-op.
    """
    if ctx is None:
        return
    # token=None => a reused (inner) scope; the outermost owner writes + resets.
    if token is None:
        return
    _CAPTURE_CTX.reset(token)
    if ctx.calls:
        try:
            _write_turn_trace(ctx, error=error)
        except Exception:
            log.warning("training_turn_trace_write_failed", exc_info=True)


def uninstall_capture_hook(backend: object) -> None:
    """Restore a backend's original ``chat`` / ``chat_stream`` (idempotent)."""
    if not getattr(backend, _HOOK_SENTINEL, False):
        return
    orig = getattr(backend, _HOOK_ORIG, None)
    if orig:
        backend.chat, backend.chat_stream = orig  # type: ignore[attr-defined]
    for attr in (_HOOK_SENTINEL, _HOOK_ORIG):
        if hasattr(backend, attr):
            delattr(backend, attr)
