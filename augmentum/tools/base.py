"""Tool base class and result types for the Augmentum tool framework."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Voice exposure tiers — mirrors the bucket vocabulary in
# ``augmentum/intent/manifest.py``. ``None`` means the tool is not
# voice-reachable. Pick the most conservative tier the primitive fits.
VoiceLevel = Literal["core", "interactive", "disruptive", "costly"]


@dataclass(frozen=True)
class SurfaceExposure:
    """Where a Tool is reachable from. Declared once by the Tool;
    every surface registry (chat handler, voice route, coder agent,
    Artifact Studio, auto HTTP route, contextual right-click) derives
    its tool list from this declaration.

    Default exposure (``chat=True, coder=True``) matches the historical
    behavior of any Tool registered in :class:`ToolRegistry`. Adding a
    Tool with default surfaces does not silently widen reach.

    Fields:
        chat: Exposed as an LLM function-call in the chat handler.
        voice: Voice exposure tier, or ``None`` to hide from voice.
            Tier choice gates the ambient-mode policy filter — see
            ``augmentum/intent/manifest.py``.
        coder: Exposed to the coder agent's tool tree.
        companion: Exposed via companion (Becca) dispatch.
        artifact_studio: Surfaced as an Artifact Studio button or
            action on artifacts whose MIME type matches
            ``file_context_menu``.
        file_context_menu: List of MIME globs (``image/*``) and/or
            file extensions (``.pdf``) that trigger this tool in
            contextual right-click / drag-target menus.
        http_route: When set, the proxy startup hook auto-registers
            ``POST {route}`` for this tool. Leave ``None`` to skip
            auto-registration (or to keep an existing hand-written
            route).
        voice_capability_line: One-line capability summary populated
            into ``VOICE_TOOL_CAPABILITIES`` when ``voice`` is set.
        flow: Offered inside reasoning-flow steps (the flow editor's
            tools grid and the flow executor's category expansion).
            Data-returning capabilities (search/fetch/verify/compute/
            artifact) belong here; conversational surface actions
            (notifications, schedulers, media/device verbs) do not —
            a flow step firing ``device.set_alarm`` mid-pipeline is
            never what the flow author meant. Explicit per-step
            ``tool_names`` pins still override this.
    """

    chat: bool = True
    voice: VoiceLevel | None = None
    coder: bool = True
    companion: bool = False
    artifact_studio: bool = False
    file_context_menu: tuple[str, ...] = ()
    http_route: str | None = None
    voice_capability_line: str = ""
    flow: bool = True


class ToolCategory(str, Enum):
    SEARCH = "search"
    FETCH = "fetch"
    EXECUTE = "execute"
    VERIFY = "verify"
    FILE = "file"
    IMAGE = "image"
    ARTIFACT = "artifact"
    CODE = "code"
    SHELL = "shell"


# ── Companion verbs architecture — core verb metadata ────────────────


class CoreVerbSafetyClass(str, Enum):
    """Same axis as management verbs' SafetyClass — keeps the two
    halves of the verb taxonomy speaking the same vocabulary."""
    READ = "READ"
    WRITE_SELF = "WRITE_SELF"
    WRITE_USER = "WRITE_USER"
    NO_USER_FACING = "NO_USER_FACING"


class CoreVerbAutonomyClass(str, Enum):
    """How autonomous the verb is allowed to be — gated by the
    presence-mode autonomy floor at dispatch time."""
    EXPLICIT = "EXPLICIT"        # user must explicitly request
    SUGGESTED = "SUGGESTED"      # surfaces as a recommendation
    BACKGROUND = "BACKGROUND"    # fires without user-initiated trigger


@dataclass(frozen=True)
class CostEnvelope:
    """Soft caps on resource cost per invocation. Dispatcher checks
    these against actual usage and logs ``budget_exceeded`` outcomes
    when blown — protects against runaway tool calls."""
    max_wallclock_ms: int = 60_000
    max_db_ops: int = 200


@dataclass(frozen=True)
class CoreVerbMetadata:
    """Companion verbs architecture — core verb declaration.

    Tools that declare ``core_verb`` participate in the verb registry
    alongside management verbs (companion_runtime/event_bus.py).
    Management verbs read ``can_invoke_core`` allowlists to decide
    which core verbs they're permitted to call.

    The metadata is INTENTIONALLY separate from SurfaceExposure: a
    Tool can be chat-only without being a core verb, and a core
    verb can be reachable from multiple surfaces. Declaring
    ``core_verb`` is the explicit opt-in.
    """
    safety_class: CoreVerbSafetyClass = CoreVerbSafetyClass.READ
    autonomy_class: CoreVerbAutonomyClass = CoreVerbAutonomyClass.EXPLICIT
    cost_envelope: CostEnvelope = field(default_factory=CostEnvelope)
    # Whether the verb's response must cite its own substrate writes
    # so downstream chain-of-thought consumers can reconstruct what
    # changed. Required for all WRITE_* verbs by default.
    cite_self_required: bool = False
    # When True, the verb participates in chain-depth limits — its
    # invocation counts toward the 2-step ceiling that prevents
    # runaway cascades. Default True; turn off only for terminal
    # surface verbs (notify) where chain depth is meaningless.
    counts_in_chain_depth: bool = True


@dataclass
class ToolResult:
    success: bool
    output: str = ""
    error: str = ""
    metadata: dict = field(default_factory=dict)
    validation_error: bool = False  # True when the call itself was malformed
    # WHY the call failed, when it did. Empty on success.
    #
    #   "invalid_input"  — the model's arguments didn't fit the schema
    #   "internal_error" — the tool raised; it never really ran
    #   "upstream_error" — a dependency (SearXNG, executor, API) failed
    #
    # This exists because "the tool ran and found nothing" and "the tool
    # crashed" are opposite signals that were previously both scored as
    # ``empty`` — which made the synthesis layer tell the model the
    # search came back empty and to answer from its own knowledge. The
    # model then doubted well-sourced material it had already cited.
    # Anything that distinguishes "no results" from "no run" must key
    # off this, not off ``success`` alone.
    failure_kind: str = ""
    # Non-fatal degradations the tool absorbed — e.g. "chapter 3
    # illustration failed", "2 of 5 rows had ragged column counts and
    # were padded". The handler should mention these in the chat
    # response so the user isn't surprised by a silently degraded
    # artifact; the frontend can also surface them on the result card.
    # Tools MUST append to this list rather than swallowing degradations
    # with log.warning alone.
    warnings: list[str] = field(default_factory=list)
    # Structured presentation envelope for the frontend. When set, the UI
    # renders a typed ToolCard (preview/edit/download) instead of dumping
    # ``output`` as plain markdown. ``output`` should then be a short
    # 1-3 sentence summary the LLM can chain on, not the full payload.
    #
    # Schema::
    #
    #   {
    #     "kind": "artifact" | "image" | "search" | "article" |
    #             "code_exec" | "calc" | "data",
    #     "title": str,
    #     "subtitle": str,
    #     "summary": str,
    #     "preview": dict,           # kind-specific preview data
    #     "actions": [               # buttons rendered on the card
    #         {"label": str, "href"?: str, "event"?: str,
    #          "icon"?: str, "payload"?: dict}
    #     ],
    #     "artifact_id"?: str,       # for edit/preview/download routing
    #     "source_url"?: str,        # for fetch/search citations
    #   }
    card: dict | None = None


async def invoke_tool(tool: object, params: dict) -> ToolResult:
    """Call ``tool`` with coercion + fan-out, tolerating non-Tool objects.

    Not everything in a registry is a :class:`Tool` subclass — MCP-bridged
    tools, adapters and test doubles are duck-typed and expose only
    ``execute``. Calling ``.invoke`` on those raises ``AttributeError``,
    which is exactly the class of failure this whole change set exists to
    remove; a hardening pass must not introduce a new crash of its own.

    Prefer this helper at any dispatch site whose registry can hold
    objects it didn't construct.
    """
    # isinstance, NOT hasattr: a MagicMock fabricates any attribute you
    # ask it for, so ``getattr(tool, "invoke")`` succeeds on every mock
    # and would silently route around the real ``execute`` the caller
    # is asserting on. Only a genuine Tool subclass has the real thing.
    if isinstance(tool, Tool):
        return await tool.invoke(params)
    return await tool.execute(**params)  # type: ignore[attr-defined]


def format_output_with_warnings(output: str, warnings: list[str]) -> str:
    """Append a human-readable warnings section to ``output`` for the
    chat-facing summary. Tools should use this to ensure degradations
    are visible in the chat stream, not just in ``ToolResult.warnings``
    (which is primarily for frontend cards and telemetry).
    """
    if not warnings:
        return output
    joined = "\n".join(f"- {w}" for w in warnings)
    suffix = f"\n\nWarnings:\n{joined}"
    return (output or "") + suffix


def make_artifact_card(
    info: dict,
    *,
    kind: str,
    title: str,
    subtitle: str = "",
    summary: str = "",
    preview: dict | None = None,
) -> dict:
    """Build a standard artifact ToolCard with preview/edit/download actions.

    ``info`` is the dict returned by ``ArtifactStore.save`` — must contain
    ``id`` and ``download_url``. The frontend dispatches ``artifact:preview``
    and ``artifact:edit`` events with ``payload.artifact_id`` so existing
    editors (Artifact Studio etc.) can be reused as the card's targets.
    """
    artifact_id = info.get("id", "")
    download_url = info.get("download_url", "")
    return {
        "kind": kind,
        "title": title,
        "subtitle": subtitle,
        "summary": summary,
        "preview": preview or {},
        "artifact_id": artifact_id,
        "actions": [
            {
                "label": "Preview",
                "event": "artifact:preview",
                "payload": {"artifact_id": artifact_id},
                "icon": "eye",
            },
            {
                "label": "Edit",
                "event": "artifact:edit",
                "payload": {"artifact_id": artifact_id},
                "icon": "edit",
            },
            {
                "label": "Download",
                "href": download_url,
                "icon": "download",
            },
        ],
    }


class Tool(ABC):
    """Base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def category(self) -> ToolCategory: ...

    @property
    def input_schema(self) -> dict:
        """JSON Schema for tool input (for LLM tool-use)."""
        return {}

    @property
    def surfaces(self) -> SurfaceExposure:
        """Where this tool is reachable from.

        Default exposure is ``chat=True, coder=True`` — matches the
        historical behavior of any Tool already in the registry.
        Override to add voice / companion / artifact-studio reach or
        to auto-register an HTTP route.
        """
        return SurfaceExposure()

    @property
    def core_verb(self) -> CoreVerbMetadata | None:
        """Companion verbs architecture — optional core-verb declaration.

        When set, the Tool participates in the companion verb registry
        alongside management verbs. Management verbs read these
        declarations to decide which core verbs they're permitted to
        invoke (via ``can_invoke_core`` allowlists).

        Default ``None``: the Tool is just a regular Tool with no
        companion-verb semantics. Declaring metadata is the explicit
        opt-in to the two-class verb taxonomy.
        """
        return None

    @property
    def timeout(self) -> float:
        """Per-tool execution timeout in seconds."""
        return 45.0

    @property
    def cacheable(self) -> bool:
        """Whether results can be cached for identical inputs."""
        return True

    @property
    def cache_ttl(self) -> float:
        """Cache time-to-live in seconds.  0 = never expires."""
        return 300.0

    # ------------------------------------------------------------------
    # v2 — quality, composability, and resilience metadata
    # ------------------------------------------------------------------

    @property
    def error_hints(self) -> dict[str, str]:
        """Map of error substring → recovery guidance for the LLM.

        When a tool execution fails, the error message is scanned against
        these patterns. The first matching hint is appended to help the
        LLM self-correct on retry.

        Example::

            {"No results": "Try broader search terms or different keywords."}
        """
        return {}

    @property
    def produces(self) -> list[str]:
        """Output types this tool produces.

        Used by the chain planner to validate step dependencies.
        Common types: "text", "image_url", "artifact_url", "structured_data".
        """
        return ["text"]

    @property
    def consumes(self) -> list[str]:
        """Input types this tool can accept from other tools.

        For example, ``create_ebook`` consumes ``"image_url"`` from
        ``image_generation``.  The chain planner uses this to ensure
        prerequisite steps are planned.
        """
        return []

    @property
    def requires_services(self) -> list[str]:
        """External services this tool needs to function.

        Checked at schema-injection time and by the tools dropdown API.
        Known service names: "searxng", "executor", "image_pipeline".
        """
        return []

    @property
    def auto_invoke_when_enabled(self) -> bool:
        """Whether the passthrough handler should call this tool directly
        when the user has explicitly enabled it in the tool picker,
        bypassing the model's tool-calling decision.

        Use this for tools where the button IS the user's intent signal
        (YouTube, build_application) — the user wouldn't have turned
        them on if they didn't want them invoked. The query then becomes
        the tool input, and the model only synthesizes the result.

        Leave False for tools that depend on query shape (calculator,
        datetime, convert) — those should stay model-driven so they
        don't fire on unrelated prompts.
        """
        return False

    @property
    def long_running(self) -> bool:
        """Whether this tool runs in the background and drives its own
        progress UI. Only applies when ``auto_invoke_when_enabled`` is
        True — the passthrough handler spawns a background task for it,
        emits a kickoff chunk, and returns immediately so the user can
        keep chatting.

        Long-running tools are expected to:
          - accept ``_progress_callback`` in their ``execute`` kwargs
          - push updates to a client-polled state dict (e.g.
            ``ACTIVE_BUILDS`` for app_builder) whose shape the UI
            already understands
          - tolerate cancellation signals through that state
        """
        return False

    @property
    def model_hint(self) -> str:
        """Extra usage guidance appended to ``description`` for small models.

        When the system detects a model below ~14B parameters, this hint
        is appended to the tool description in the schema to compensate
        for weaker instruction following.  Empty string = description is
        sufficient on its own.
        """
        return ""

    # Error fragments that mean "the CALL was malformed", as opposed to "the
    # tool ran and something went wrong". These are the ones a schema reminder
    # can actually fix, so the generic fallback is scoped to them — appending
    # parameter lists to a network timeout would be noise.
    _SHAPE_ERROR_MARKERS: tuple[str, ...] = (
        "invalid arguments",
        "unexpected keyword argument",
        "required positional argument",
        "missing 1 required",
        "invalid input",
        "validation error",
        "is required",
        "are required",
        "must be",
        "expected",
        "not a valid",
        "no such parameter",
    )

    # Opening of the generated schema reminder. Doubles as the marker
    # ``enrich_error`` looks for to stay idempotent — keep the two in sync by
    # keeping them the same constant.
    _SCHEMA_REMINDER_STEM: str = "{name} accepts — "

    def enrich_error(self, error: str, params: dict) -> str:
        """Enrich a raw error message with recovery guidance for the LLM.

        Two layers, in priority order:

        1. ``self.error_hints`` — the tool's own substring→guidance map. First
           match wins and short-circuits, because a hand-written hint is always
           better targeted than the generic fallback.
        2. A schema reminder, appended only when the error looks like a
           MALFORMED CALL (see ``_SHAPE_ERROR_MARKERS``): the accepted parameter
           names, which of them are required, and any the model just sent that
           the tool does not accept.

        Layer 2 exists because most tools define no ``error_hints`` at all, so
        the model previously saw a bare ``invalid arguments: execute() got an
        unexpected keyword argument 'query_string'`` with no indication of what
        the right name was — unrecoverable without guessing. This docstring
        promised the schema behaviour for a long time while the implementation
        only ever did layer 1; this makes the promise true for every tool
        instead of the seventeen that hand-wrote hints.

        Never raises: a malformed ``input_schema`` must degrade to the raw
        error, never replace a tool failure with an enrichment failure.

        **Idempotent.** ``Tool.invoke`` already enriches before returning, and
        several dispatch paths (passthrough, coder, chain) then enrich
        ``result.error`` again on their way to building the tool message. That
        is double-calling by design — those call sites also handle results from
        surfaces that never went through ``invoke`` — so re-appending a hint the
        text already carries would show the model the same paragraph twice and
        burn context teaching it nothing. Both layers below no-op when their
        guidance is already present.

        Override for custom enrichment logic.
        """
        for pattern, hint in self.error_hints.items():
            if pattern.lower() in error.lower():
                return error if hint in error else f"{error}\n\nHint: {hint}"

        low = error.lower()
        if not any(marker in low for marker in self._SHAPE_ERROR_MARKERS):
            return error
        try:
            guidance = self._schema_reminder(params)
        except Exception:  # noqa: BLE001 — enrichment must never mask the error
            log.debug("enrich_error_schema_reminder_failed", tool=self.name)
            return error
        # Match on the STEM, not the whole string. The two calls see different
        # ``params`` — ``invoke`` enriches with the coerced dict (strays already
        # stripped), the outer dispatch site with the model's raw one — so the
        # trailing "Not accepted: …" clause differs and an equality check would
        # let the duplicate through.
        if not guidance or self._SCHEMA_REMINDER_STEM.format(name=self.name) in error:
            return error
        return f"{error}\n\nHint: {guidance}"

    def _schema_reminder(self, params: dict) -> str:
        """Describe this tool's accepted parameters for a mis-formed call.

        Names only, not the full JSON Schema — the schema was already sent with
        the tool definition, so repeating it wastes context. What the model
        actually lacks is which name it got wrong.
        """
        schema = self.input_schema or {}
        props = schema.get("properties") or {}
        if not isinstance(props, dict) or not props:
            return ""
        required = [r for r in (schema.get("required") or []) if isinstance(r, str)]
        optional = [n for n in props if n not in required]

        parts: list[str] = []
        if required:
            parts.append(f"required: {', '.join(sorted(required))}")
        if optional:
            parts.append(f"optional: {', '.join(sorted(optional))}")
        guidance = f"{self._SCHEMA_REMINDER_STEM.format(name=self.name)}{'; '.join(parts)}."

        # The single most useful line: the name they invented.
        unknown = sorted(
            k for k in params
            # ``_context``/``_user_id`` are injected by the runtime, never by
            # the model, so naming them would send it chasing a phantom.
            if not k.startswith("_") and k not in props
        )
        if unknown:
            guidance += (
                f" Not accepted: {', '.join(unknown)} — "
                "use the names above, and send arguments as a JSON object."
            )
        missing = sorted(r for r in required if r not in params)
        if missing:
            guidance += f" Missing: {', '.join(missing)}."
        return guidance

    def health_check(self) -> bool:
        """Fast check whether this tool can execute right now.

        Called by the tools dropdown API to grey out unavailable tools.
        Default returns True.  Override for tools with external service
        dependencies (executor, SearXNG, image pipeline).
        """
        return True

    @staticmethod
    def extract_user_id(kwargs: dict) -> str:
        """Canonical user_id extraction for tools.

        Two historical patterns collide here:

        - chain.py passes ``_user_id`` as a top-level kwarg (and since the
          propagation fix also nests it under ``_context``).
        - passthrough/orchestrator calls pass ``_context={"user_id": ...}``
          directly, without the top-level kwarg.
        - Direct handler calls (agentic, coder) may pass neither.

        Every tool that persists to a user-scoped table should route its
        user_id through this helper rather than reinventing the
        extraction. Returns empty string if nothing is present — callers
        decide whether that's a hard error or a skip.
        """
        uid = kwargs.get("_user_id") or ""
        if uid:
            return str(uid)
        ctx = kwargs.get("_context") or {}
        if isinstance(ctx, dict):
            return str(ctx.get("user_id") or "")
        return ""

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult: ...

    async def invoke(self, params: dict) -> ToolResult:
        """Run this tool from an LLM-supplied parameter dict.

        **Every dispatch path should call this, not ``execute``.** It is
        the single seam where model-supplied arguments are reconciled
        with the declared schema, so a new dispatch path cannot forget
        the reconciliation the way six existing ones did.

        Three things happen here that ``execute`` must not have to
        re-implement per tool:

        1. **Coercion** — scalar type fixes and envelope unwrapping.
        2. **Fan-out** — a list supplied for a string param becomes one
           call per element (see :mod:`augmentum.tools.params`). A model
           asking for three searches gets three searches instead of an
           ``AttributeError``.
        3. **Failure typing** — anything that goes wrong is returned as
           a ``ToolResult`` with ``failure_kind`` set, never raised. A
           tool that never ran is a fundamentally different signal from
           a tool that ran and found nothing, and the synthesis layer
           has to be able to tell them apart or it will tell the model
           to answer from memory when the truth is "we broke".
        """
        from augmentum.tools.params import coerce_params, split_fanout

        stripped: list[str] = []
        try:
            params = coerce_params(self, dict(params), stripped_out=stripped)
        except Exception:  # noqa: BLE001 — coercion is best-effort
            log.debug("tool_param_coerce_failed", tool=self.name, exc_info=True)

        fan_key, fan_values, fan_error = split_fanout(self, params)
        if fan_error:
            return ToolResult(
                success=False, error=fan_error,
                validation_error=True, failure_kind="invalid_input",
            )
        if fan_key:
            result = await self._invoke_fanout(params, fan_key, fan_values)
        else:
            result = await self._invoke_once(params)
        return self._note_ignored_params(result, stripped)

    def _note_ignored_params(
        self, result: ToolResult, stripped: list[str],
    ) -> ToolResult:
        """Surface model-supplied arguments that were stripped as unknown.

        ``coerce_params`` deletes any argument not in the tool's schema so
        ``execute(**params)`` won't raise on it. That deletion used to be
        logged at ``debug`` only — invisible to the model, which then acted
        on a result that silently ignored what it asked for: a dropped
        ``cwd`` ran in the wrong directory, a dropped case-insensitive flag
        turned a real hit into a false "no results". Appending the ignored
        names to the tool's own output turns a silent drop into a
        self-correcting signal, without failing an otherwise-good call.

        Applies to every tool and every future param — the systemic
        counterpart to declaring the specific high-traffic ones.
        """
        if not stripped or not isinstance(result, ToolResult):
            return result
        names = ", ".join(sorted(set(stripped)))
        note = (
            f"[Ignored unsupported argument(s): {names}. This tool does not "
            "accept them, so the result above does NOT reflect them. Check "
            "this tool's parameters and retry if they mattered.]"
        )
        # Append to whichever text field the model actually reads. Never
        # clobber existing content, and don't flip success — the call may
        # well have succeeded; it just ignored an extra arg.
        if result.output:
            result.output = f"{result.output}\n\n{note}"
        elif result.error:
            result.error = f"{result.error}\n\n{note}"
        else:
            result.output = note
        result.metadata = {
            **(result.metadata or {}),
            "ignored_params": sorted(set(stripped)),
        }
        result.warnings = [*result.warnings, f"ignored unsupported args: {names}"]
        return result

    def _drop_unaccepted_context(self, params: dict) -> dict:
        """Strip ``_``-prefixed context keys this tool's ``execute`` can't take.

        Dispatch layers inject runtime context (``_user_id``, ``_context``,
        ``_session_id``…) that only *some* tools declare. Callers can't know
        which without introspecting every tool, and guessing by category (as
        ``chain.py`` does) misses tools outside those categories — a
        ``_user_id`` added to fix user-scoped tools then breaks every tool
        with a strict signature ("unexpected keyword argument '_user_id'").

        Ask the signature instead: a tool declaring ``**kwargs`` takes
        anything; otherwise only its named parameters get through.
        """
        if not any(k.startswith("_") for k in params):
            return params
        try:
            sig = inspect.signature(self.execute)
        except (TypeError, ValueError):  # C-implemented / exotic callables
            return params
        if any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in sig.parameters.values()):
            return params
        allowed = set(sig.parameters)
        return {
            k: v for k, v in params.items()
            if not k.startswith("_") or k in allowed
        }

    async def _invoke_once(self, params: dict) -> ToolResult:
        """Execute once, converting an exception into a typed result."""
        params = self._drop_unaccepted_context(params)
        try:
            return await self.execute(**params)
        except TypeError as exc:
            # Wrong/missing kwargs — the model's call didn't match the
            # signature. Recoverable if we say so in its language.
            return ToolResult(
                success=False,
                error=self.enrich_error(f"invalid arguments: {exc}", params),
                validation_error=True, failure_kind="invalid_input",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("tool_execute_raised", tool=self.name, exc_info=True)
            return ToolResult(
                success=False,
                error=self.enrich_error(str(exc), params),
                failure_kind="internal_error",
            )

    async def _invoke_fanout(
        self, params: dict, key: str, values: list,
    ) -> ToolResult:
        """Run once per value and merge, preserving the model's intent.

        Runs concurrently — these are independent calls, and serializing
        four web searches would turn a 3s turn into 12s.
        """
        import asyncio

        log.info("tool_param_fanout", tool=self.name, param=key, count=len(values))
        results = await asyncio.gather(*[
            self._invoke_once({**params, key: value}) for value in values
        ])

        sections: list[str] = []
        warnings: list[str] = []
        merged_meta: dict = {}
        for value, res in zip(values, results, strict=False):
            body = res.output if res.success else f"(failed: {res.error})"
            sections.append(f'### {key} = "{value}"\n{body}')
            warnings.extend(res.warnings)
            if res.success and isinstance(res.metadata, dict):
                merged_meta.update(res.metadata)

        ok = [r for r in results if r.success]
        if not ok:
            # Every branch failed — report as a real failure rather than
            # a "successful" result whose body happens to be error text,
            # which would be scored as usable content downstream.
            first = results[0]
            return ToolResult(
                success=False,
                error=first.error or "all calls failed",
                failure_kind=first.failure_kind or "internal_error",
                warnings=warnings,
            )
        if len(ok) < len(results):
            warnings.append(
                f"{len(results) - len(ok)} of {len(results)} {self.name} "
                f"calls failed; results below cover the rest."
            )
        merged_meta["fanout"] = {"param": key, "values": list(values)}
        return ToolResult(
            success=True,
            output="\n\n".join(sections),
            metadata=merged_meta,
            warnings=warnings,
        )

    def validate_input(self, **kwargs) -> bool:
        """Basic input validation. Override for custom checks.

        NOTE: this is advisory only and is NOT called by any dispatch
        path — enforcement lives in :meth:`invoke` / ``params.py``,
        which every surface shares. Do not add a guard here expecting
        it to run in production.
        """
        return True
