"""Coder delegation verb — hand a build/fix task to a background coder agent.

The companion works headless-first (docs/superpowers/specs/
2026-06-10-companion-headless-agency-design.md): when the user asks her to
BUILD or FIX code in one of their projects, she doesn't do it inline — she
queues a ``coder_background_run`` job against the right workspace and reports
the verified result later through the brief (Part B → brief-panel.js). This
verb is the entry point.

Two never-auto-select seams (CLAUDE.md rule #2) are load-bearing here:

* **Workspace** — the resolver (``coder/workspace_resolver.py``) either returns
  a confident single match (which we ANNOUNCE — "building in <name>") or a set
  of candidates the user picks from (tap-or-say, via the companion-candidates
  dock). We never silently pick one.
* **Model** — the run uses the user's PRIMARY chat model (Slot A), NOT whatever
  small utility model happened to drive the companion turn. When the user has
  pinned a heavyweight model in the model manager, we surface it as an optional
  "⚡ heavyweight" escalation on the card (and answerable by voice, "use the
  heavyweight one") — but primary is the default; we never invent a model.

Tier-3 only + kept OUT of ``CORE_TOOL_NAMES`` on purpose: the verb only enters
the companion's roster when the turn scores relevant to it, so it can't
over-reach on ordinary chat. ``stakes=costly`` keeps it out of the always-on
ambient widget's default policy (it fires from a foreground voice call or chat,
never silently), and ``companion_initiatable=False`` blocks self-started runs.
"""
from __future__ import annotations

import re
import time
from typing import Any

from augmentum.intent.action import ActionFanout, ActionResult, SessionContext
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Name-derivation stopwords — the verbs/glue of a build request carry no
# workspace-naming signal ("build me a dark-mode toggle" → "dark-mode-toggle").
_NAME_STOP = frozenset({
    "build", "make", "create", "add", "implement", "fix", "update", "change",
    "the", "a", "an", "in", "on", "to", "for", "my", "me", "please", "some",
    "new", "into", "with", "and", "of", "app", "page", "feature", "bug", "code",
    "project", "workspace", "repo", "this", "that", "it", "go", "set", "up",
    "write", "refactor", "want", "need", "just", "then", "let", "lets", "begin",
    "start", "research", "task", "about",
})


def _derive_ws_name(prompt: str) -> str:
    """A short, human-editable workspace name from the build request.

    Keeps the first few content words (the user renames freely afterward — the
    name was never the lock-in). Always returns something non-empty."""
    words = [
        w for w in re.findall(r"[a-z0-9]+", (prompt or "").lower())
        if w not in _NAME_STOP and len(w) > 1
    ]
    return ("-".join(words[:4]) or "new-project")[:40]


def _profile_arg_schema() -> dict[str, Any]:
    """Build the ``profile`` arg schema from the live profile catalog so the
    model chooses deliberately (labels + descriptions inline), and the enum
    can never drift from ``profiles.all_profiles()``."""
    from augmentum.coder.profiles import all_profiles

    profs = all_profiles()
    catalog = "; ".join(f"'{p.id}' ({p.label}): {p.description}" for p in profs)
    return {
        "type": "string",
        "enum": [p.id for p in profs],
        "description": (
            "Tooling profile for a NEW workspace — pick the best fit for the "
            f"task. Options: {catalog}. Omit to use the default. The agent's "
            "later installs (apt/pip/npm) persist across restarts, so this is a "
            "starting point, not a lock-in — pick the closest and it'll grow."
        ),
    }

_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)
# coder.delegate is the SOLE path from the companion to "actually build/fix code
# in a workspace" — the one verb tonight's "launch a coder run" needed. It must
# never be budget-clipped out of the roster (a clipped tool is invisible AND
# uncallable), so it declares itself load-bearing.
_TIER3_ALWAYS = ActionFanout(tier1=False, tier2=False, tier3=True, always_offer=True)


def _models(tier: str) -> tuple[str, str, bool, str]:
    """Resolve (primary, heavyweight, heavyweight_available, chosen).

    Reads the server-configured chat models — NOT the companion turn's model.
    ``chosen`` honours the requested tier, falling back to primary when a
    heavyweight isn't configured.
    """
    from augmentum.config import settings as _s

    primary = (getattr(_s, "primary_chat_model", "") or "").strip()
    heavy = (getattr(_s, "heavyweight_model", "") or "").strip()
    heavyweight_available = bool(heavy and heavy != primary)
    chosen = heavy if (tier == "heavyweight" and heavy) else primary
    return primary, heavy, heavyweight_available, chosen


async def _enqueue(
    app_state: Any, *, user_id: str, workspace_id: str, prompt: str, model: str,
) -> str | None:
    """Queue a coder_background_run job. Mirrors the queue_background_run
    route (coder_routes.py) so the job handler + brief perception are reused
    verbatim. Returns the job id, or None if the queue isn't wired."""
    jobs_store = getattr(app_state, "jobs_store", None)
    job_runner = getattr(app_state, "job_runner", None)
    if jobs_store is None or job_runner is None:
        return None
    job_id = await jobs_store.create(
        user_id=user_id,
        job_type="coder_background_run",
        payload={
            "workspace_id": workspace_id,
            "prompt": prompt,
            "model": model,
            "coder_strategy": "",
        },
        priority=5,
        max_attempts=2,
    )
    job_runner.wake()
    log.info(
        "coder_delegate_queued",
        user_id=user_id, workspace_id=workspace_id, job_id=job_id, model=model,
    )
    return job_id


def _queued_result(workspace_id: str, workspace_title: str, tier: str) -> ActionResult:
    tier_note = " with your heavyweight model" if tier == "heavyweight" else ""
    return ActionResult(
        short_circuit=True, fulfilled=True,
        speak=(
            f"On it — I'll build that in {workspace_title}{tier_note} in the "
            "background, and bring you the result when it's done."
        ),
        toast=f"Queued in {workspace_title}"[:80],
        # Routed as an acknowledgment (no panel — the async run's result opens
        # the brief later). Its one job on the voice path: a "she did it" tick
        # that dismisses any lingering candidate dock (becca:verb-fired).
        surface_emit={
            "channel": "coder.delegate",
            "payload": {"workspace_id": workspace_id, "title": workspace_title},
        },
    )


def _speak_for_offer(candidates: list[Any]) -> str:
    names = [c.title for c in candidates[:3]]
    if not names:
        return "I can build that — set up a workspace for it and I'll start."
    if len(names) == 1:
        return f"I can build that in {names[0]}, or a new workspace — which?"
    listed = ", ".join(names[:-1]) + f", or {names[-1]}"
    return f"Which workspace should I build in — {listed}, or a new one?"


async def _delegate(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    if not session.user_id:
        return ActionResult(
            short_circuit=True, fulfilled=False,
            speak="I can't start a coding task for a signed-out session.",
        )
    app_state = getattr(session, "app_state", None)
    prompt = (args.get("prompt") or "").strip() or (text or "").strip()
    if not prompt:
        # Park the ask so the user's next line fills it (not a bare re-derive).
        refs = getattr(session, "referents", None)
        if refs is not None:
            refs.pending_intent = {
                "action_id": "coder.delegate", "args": {},
                "missing": ["prompt"], "question": "What would you like me to build?",
                "asked_at": time.time(),
            }
        return ActionResult(
            short_circuit=True, fulfilled=False,
            speak="What would you like me to build?",
        )

    tier = str(args.get("tier") or "primary").strip().lower()
    if tier not in ("primary", "heavyweight"):
        tier = "primary"
    primary, heavy, heavyweight_available, model = _models(tier)
    if not model:
        return ActionResult(
            short_circuit=True, fulfilled=False,
            speak=(
                "You haven't picked a chat model yet — choose one in "
                "Settings → Models and I'll use it to build."
            ),
        )

    # An explicit workspace pick reached us (a follow-up card tap, or the model
    # driving the turn filled the slot).
    workspace_id = str(args.get("workspace_id") or "").strip()
    if workspace_id == "__new__":
        # Hands-off create: provision a real workspace with the companion's
        # chosen profile + a derived (editable) name and start the build — NOT
        # a route to the empty creator. Matt 2026-07-28: if the user asks the
        # companion to create AND hand off, dumping them on a blank form defeats
        # the point; she should use judgement. Profile/name are both editable
        # after, and deps now persist across recreate (apt-persistence layer),
        # so an imperfect profile is cheap, not a cage.
        return await _create_and_delegate(
            app_state, session, prompt=prompt, model=model, tier=tier,
            profile=str(args.get("profile") or "").strip().lower(),
            workspace_name=str(args.get("workspace_name") or "").strip(),
        )

    # A model-supplied workspace_id is UNTRUSTED. A card tap sends the real id,
    # but the LLM driving a companion turn fills this slot with a NAME it heard
    # ("this inference workspace") rather than the id — which then queues a
    # coder_background_run against an id no container/DB row has, and every op
    # dies with `KeyError: Workspace <name> not found` deep in the container
    # manager (observed 2026-07-28, workspace_id="inference"; the run "starts"
    # then does nothing). Only dispatch a workspace_id that's a REAL owned
    # workspace; otherwise fold the string into the resolver as a naming HINT
    # (confident → announce, else → offer picks) so a bogus id can never reach
    # the job queue.
    from augmentum.coder.workspace_resolver import _owned_workspaces, resolve_workspace

    resolve_hint = ""
    if workspace_id:
        owned_ids = {w.id for w in await _owned_workspaces(app_state, session.user_id)}
        if workspace_id in owned_ids:
            job_id = await _enqueue(
                app_state, user_id=session.user_id,
                workspace_id=workspace_id, prompt=prompt, model=model,
            )
            if not job_id:
                return ActionResult(
                    short_circuit=True, fulfilled=False,
                    speak="Background jobs aren't available right now, so I can't start that.",
                )
            title = await _workspace_title(app_state, session.user_id, workspace_id)
            _clear_offer(session)
            _record_trail(session, workspace_id, title)
            return _queued_result(workspace_id, title, tier)
        # Not a real id — the model guessed a name. Keep it as a resolver hint
        # instead of dispatching it into the void.
        log.info("coder_delegate_untrusted_workspace_id", supplied=workspace_id[:80])
        resolve_hint = workspace_id

    # No (valid) workspace named — resolve, folding in any name hint the model
    # supplied so "the inference workspace" still lands on the right one.
    result = await resolve_workspace(
        app_state, user_id=session.user_id,
        request_text=f"{prompt} {resolve_hint}".strip(),
    )

    if result.decision == "confident" and result.top is not None:
        job_id = await _enqueue(
            app_state, user_id=session.user_id,
            workspace_id=result.top.workspace_id, prompt=prompt, model=model,
        )
        if not job_id:
            return ActionResult(
                short_circuit=True, fulfilled=False,
                speak="Background jobs aren't available right now, so I can't start that.",
            )
        _clear_offer(session)
        _record_trail(session, result.top.workspace_id, result.top.title)
        return _queued_result(result.top.workspace_id, result.top.title, tier)

    # Ambiguous (or the user has no workspaces) — offer picks + a new tile.
    from augmentum.coder.workspace_resolver import WorkspaceCandidate

    picks = list(result.candidates)
    payloads = [c.to_payload() for c in picks]
    payloads.append(WorkspaceCandidate(
        workspace_id="__new__", title="New workspace",
        subtitle="Start fresh", is_new=True,
    ).to_payload())

    refs = getattr(session, "referents", None)
    if refs is not None:
        now = time.time()
        refs.pending_candidates = payloads
        refs.pending_candidates_at = now
        # Generic offer metadata so the architect router can resolve a spoken
        # "the second one" to coder.delegate(workspace_id=<id>) — see the
        # router's offered-candidate block (no longer media-hardcoded).
        refs.pending_candidates_intent = "coder.delegate"
        refs.pending_candidates_id_field = "workspace_id"
        refs.pending_intent = {
            "action_id": "coder.delegate",
            "args": {"prompt": prompt, "tier": tier},
            "missing": ["workspace_id"],
            "question": "Which workspace should I build in?",
            "asked_at": now,
        }

    return ActionResult(
        short_circuit=True, fulfilled=False,
        speak=_speak_for_offer(picks),
        surface_emit={
            "channel": "companion.candidates",
            "payload": {
                "intent": "coder.delegate",
                "candidates": payloads,
                # The dock enqueues on tap; it needs the build request + the
                # model strings (primary always; heavyweight only if pinned).
                "delegation": {
                    "prompt": prompt,
                    "model": primary,
                    "heavyweight_model": heavy,
                    "tier": tier,
                    "heavyweight_available": heavyweight_available,
                },
            },
        },
    )


async def _create_and_delegate(
    app_state: Any, session: SessionContext, *,
    prompt: str, model: str, tier: str, profile: str, workspace_name: str,
) -> ActionResult:
    """Provision a new owned workspace (companion-chosen profile + name) and
    queue the build against it. Everything is editable afterward; a bad profile
    guess is recoverable because deps persist across recreate."""
    mgr = getattr(app_state, "container_manager", None)
    if mgr is None:
        return ActionResult(
            short_circuit=True, fulfilled=False,
            speak="I can't set up a workspace right now — the coder service isn't available.",
        )
    from augmentum.coder import profiles as _profiles

    # Model's profile choice when valid; otherwise "" so create_workspace falls
    # through to coder_default_tooling_profile (never hard-fail on a guess).
    if profile and not _profiles.has_profile(profile):
        log.info("coder_delegate_unknown_profile", supplied=profile[:40])
        profile = ""
    name = workspace_name or _derive_ws_name(prompt)
    try:
        info = await mgr.create_workspace(
            name=name, tooling_profile=profile, user_id=session.user_id,
            kind="regular",
        )
    except Exception as exc:
        log.warning("coder_delegate_create_failed", error=str(exc)[:200])
        return ActionResult(
            short_circuit=True, fulfilled=False,
            speak="I couldn't set up a new workspace for that — let's try again in a moment.",
        )
    job_id = await _enqueue(
        app_state, user_id=session.user_id,
        workspace_id=info.id, prompt=prompt, model=model,
    )
    if not job_id:
        return ActionResult(
            short_circuit=True, fulfilled=False,
            speak=(
                f"I set up {name}, but background jobs aren't available right "
                "now — so I can't start the build yet."
            ),
        )
    _clear_offer(session)
    _record_trail(session, info.id, name)
    prof_label = ""
    try:
        prof_id = _profiles.resolve(profile).id if profile else ""
        prof_label = f" on the {_profiles.resolve(prof_id).label} profile" if prof_id else ""
    except Exception:
        prof_label = ""
    tier_note = " with your heavyweight model" if tier == "heavyweight" else ""
    return ActionResult(
        short_circuit=True, fulfilled=True,
        speak=(
            f"Set up {name}{prof_label} — I'll build that{tier_note} in the "
            "background and bring you the result when it's done."
        ),
        toast=f"Created {name}"[:80],
        surface_emit={
            "channel": "coder.delegate",
            "payload": {"workspace_id": info.id, "title": name, "created": True},
        },
    )


def _record_trail(session: SessionContext, workspace_id: str, title: str) -> None:
    """Append a 'coder_run' position to her trail so "take me there" jumps to
    the workspace she's building in. Explicit (not the native-loop _TRAIL_KINDS
    auto-append) because the destination is the workspace_id, which the tool's
    call args don't carry. Never raises."""
    refs = getattr(session, "referents", None)
    if refs is None or not workspace_id:
        return
    try:
        trail = getattr(refs, "trail", None)
        if trail is None:
            return
        trail.append({
            "kind": "coder_run",
            "label": (title or workspace_id)[:160],
            "ref": workspace_id[:300],
            "ts": time.time(),
        })
        del trail[:-20]  # TRAIL_CAP — mirror native_loop
    except Exception:
        log.debug("coder_delegate_trail_append_failed", exc_info=True)


def _clear_offer(session: SessionContext) -> None:
    refs = getattr(session, "referents", None)
    if refs is None:
        return
    refs.pending_candidates = []
    refs.pending_candidates_at = 0.0
    refs.pending_candidates_intent = ""
    refs.pending_candidates_id_field = ""


async def _workspace_title(app_state: Any, user_id: str, workspace_id: str) -> str:
    """Best-effort human name for a workspace id (falls back to the id)."""
    try:
        from augmentum.coder.workspace_resolver import _owned_workspaces
        for w in await _owned_workspaces(app_state, user_id):
            if w.id == workspace_id:
                return getattr(w, "name", "") or workspace_id
    except Exception:
        log.debug("coder_delegate_title_lookup_failed", exc_info=True)
    return workspace_id


register_action(
    id="coder.delegate",
    summary=(
        "Delegate a CODING task (build / fix / implement / refactor) to a "
        "background coder agent working in one of the user's own project "
        "workspaces. Use ONLY when the user asks to change or create code in a "
        "project — NOT for questions about code, explanations, or anything you "
        "can answer yourself. The result comes back later as a reviewable brief."
    ),
    examples=[
        "go build the settings page in my ui project",
        "fix the login bug in my api workspace",
        "implement a dark-mode toggle",
    ],
    arg_schema={
        "prompt": {
            "type": "string",
            "description": "The coding task to carry out, in the user's words.",
        },
        "workspace_id": {
            "type": "string",
            "description": (
                "Target workspace id (the REAL id, not a name). Omit to let the "
                "user pick when it's ambiguous. Use '__new__' to create a fresh "
                "workspace for this task — then also set 'profile' and optionally "
                "'workspace_name'. If unsure which existing workspace, omit this "
                "rather than guessing a name."
            ),
        },
        "profile": _profile_arg_schema(),
        "workspace_name": {
            "type": "string",
            "description": (
                "Optional short name for a NEW workspace (only with "
                "workspace_id='__new__'). Omit to auto-derive from the task; "
                "the user can rename it afterward."
            ),
        },
        "tier": {
            "type": "string",
            "enum": ["primary", "heavyweight"],
            "description": (
                "Which model tier to run. Default 'primary' (the user's chat "
                "model); 'heavyweight' only when the user asks for the stronger "
                "model and one is pinned."
            ),
        },
    },
    required=["prompt"],
    surfaces=["becca", "chat"],
    fanout=_TIER3_ALWAYS,
    stakes="costly",
    delivery="verbal",
    companion_initiatable=False,
    handler=_delegate,
)
