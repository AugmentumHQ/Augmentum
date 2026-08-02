"""Builder facade — run an autonomous build-test-fix loop in a workspace.

This is the entry point for the Builder-on-the-coder-harness model (spec:
docs/superpowers/specs/2026-06-15-builder-profiles-system-synthesizer-design.md).

It does NOT reimplement an agent loop. It assembles the existing primitives:

  * a coder workspace (``ContainerManager.create_workspace``),
  * the coder tool set + the builder resource tools (filtered to a build-
    relevant allowlist),
  * the Frontend App Builder Power rendered into the system prompt (the
    definition-of-done + pull-don't-guess guidance),

and drives them with ``augmentum.agents.loop.run_subagent`` — the same bounded
autonomous loop bug_finder uses. The agent writes the app, starts a dev server,
drives it with Playwright, asserts behavior, fixes what's broken, and finishes.
Then the facade snapshots the workspace into a library artifact.

Heavy imports (coder tools, the agent loop) are deferred into ``run_build`` so
the pure helpers below — file collection, source_json assembly, progress
mapping — can be imported and unit-tested without the backend stack.
"""

from __future__ import annotations

import io
import json
import tarfile
import uuid
import zipfile
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from augmentum.builds.contract import (
    derive_behaviors,
    render_behaviors_for_build,
    render_failures_for_fix,
)
from augmentum.builds.quality import (
    behavior_quality_summary,
    behavior_verdict,
    judge_tool_names,
    quality_summary,
)
from augmentum.builds.store import legacy_status
from augmentum.builds.verify import gate_summary, run_behavior_gate
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:  # type-only; avoids importing the heavy backend at module load
    from augmentum.agents.loop import SubagentProgress

log = get_logger(__name__)


# Tools the build agent gets, beyond the three builder_* resource tools.
# Deliberately scoped: write/inspect code, run a dev server, drive a browser,
# and finish. No git/test/bug-finder/subagent machinery — a build is a tight
# write→serve→drive→verify→fix loop, not a full dev session.
BUILDER_CODER_TOOL_NAMES: frozenset[str] = frozenset({
    "file_read", "file_write", "file_list", "dir_tree",
    "code_edit", "code_edit_batch", "apply_patch",
    "code_grep", "code_glob",
    "shell_exec", "shell_read", "env_info",
    "service_start", "service_list", "service_logs", "service_stop", "service_probe",
    "browser_open", "browser_snapshot", "browser_click", "browser_type",
    "browser_screenshot", "browser_evaluate", "browser_verify",
    "browser_wait", "browser_extract", "browser_fill_form",
    "finish_task",
})

# Directory prefixes never published into the artifact (dep/build/VCS noise).
_EXCLUDED_PREFIXES = (
    ".git/", "node_modules/", ".venv/", "venv/", "__pycache__/",
    "dist/", "build/", ".next/", ".cache/", ".augmentum/",
    ".augmentum_gate/",  # the behavior-gate's staged script + assertions
)
_MAX_PUBLISH_FILE_BYTES = 512 * 1024  # skip anything bigger; apps are text+small assets


# ---------------------------------------------------------------------------
# Pure helpers (no Docker / no backend — unit-testable)
# ---------------------------------------------------------------------------

def _normalize_member_path(name: str) -> str:
    """Normalize a tar member path to a workspace-relative POSIX path.

    ``workspace_archive_stream`` tars ``/workspace``; members can arrive as
    ``workspace/app.js``, ``./app.js`` or ``app.js`` depending on how the tar
    was built. Strip a leading ``./`` and a single leading ``workspace/``
    segment so the published paths are clean and relative.
    """
    n = (name or "").lstrip("/")
    while n.startswith("./"):
        n = n[2:]
    for prefix in ("workspace/", "app/workspace/"):
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    return n.strip()


def _is_excluded_path(rel: str) -> bool:
    if not rel or rel.endswith("/"):
        return True
    return any(rel == p.rstrip("/") or rel.startswith(p) for p in _EXCLUDED_PREFIXES)


def extract_text_files_from_targz(data: bytes, *, max_file_bytes: int = _MAX_PUBLISH_FILE_BYTES) -> list[dict]:
    """Extract publishable text files from a gzipped tar of a workspace.

    Returns ``[{"path", "content", "isEntrypoint"}]`` sorted by path. Binary
    files (anything that isn't valid UTF-8), oversized files, and dep/build/VCS
    paths are dropped — the published artifact is the app's source, not its
    node_modules. The shallowest ``index.html`` is flagged ``isEntrypoint``.
    """
    files: list[dict] = []
    if not data:
        return files
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            rel = _normalize_member_path(member.name)
            if _is_excluded_path(rel):
                continue
            if member.size > max_file_bytes:
                continue
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            try:
                content = fobj.read().decode("utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable — skip
            files.append({"path": rel, "content": content})
    files.sort(key=lambda f: f["path"])
    _mark_entrypoint(files)
    return files


def _mark_entrypoint(files: list[dict]) -> None:
    """Flag the shallowest index.html as the entrypoint (best-effort)."""
    entry = None
    for f in files:
        base = f["path"].rsplit("/", 1)[-1].lower()
        if base == "index.html":
            depth = f["path"].count("/")
            if entry is None or depth < entry[0]:
                entry = (depth, f)
    if entry is not None:
        entry[1]["isEntrypoint"] = True


def build_source_json(
    *,
    name: str,
    files: list[dict],
    profile_id: str = "static",
    target: str = "inline",
    capabilities: list[str] | None = None,
) -> str:
    """Assemble the artifact ``source_json`` string.

    Keeps ``type == "application"`` so every existing consumer (library list,
    preview route, open-in-code, AI-edit) works unchanged; profile/target/
    capabilities are additive.
    """
    return json.dumps(
        {
            "type": "application",
            "name": name or "Built App",
            "profile": profile_id or "static",
            "target": target or "inline",
            "capabilities": list(capabilities or []),
            "files": files,
        },
        separators=(",", ":"),
    )


def zip_files(files: list[dict]) -> bytes:
    """Zip the published files into a downloadable artifact."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.writestr(f["path"], f.get("content", ""))
    return buf.getvalue()


def map_progress(progress: SubagentProgress) -> dict:
    """Translate a SubagentProgress into the build-monitor progress shape.

    Kept flat + JSON-serializable so it can ride the build_runs SSE stream and
    the legacy ``project_progress`` UI without bespoke conversion.
    """
    return {
        "phase": progress.phase,
        "iteration": progress.iteration,
        "tool": progress.tool_name,
        "preview": progress.text_preview,
        "tokens_in": progress.tokens_in,
        "tokens_out": progress.tokens_out,
        "elapsed_ms": progress.wallclock_ms,
    }


def _default_system_prompt() -> str:
    """Fallback build guidance when the Frontend App Power isn't installed."""
    return (
        "You are an autonomous web-app builder working in a Linux workspace. "
        "Build the app the user describes, then PROVE it works before finishing: "
        "write the files, start a static dev server (e.g. `python3 -m http.server`), "
        "open it with browser_open (it returns the page snapshot), drive every "
        "control — browser_fill_form for forms, browser_click/browser_type for "
        "single controls, browser_wait for anything async (never a setTimeout "
        "sleep) — assert results with browser_extract or browser_evaluate, and "
        "check the console is clean. Fix anything that fails and re-check. "
        "Pull working patterns with builder_reference, a palette with "
        "builder_design_system, and verified API signatures with builder_api_refs "
        "instead of guessing. No stubs, no placeholders. Call finish_task only when "
        "every behavior is verified and the console is clean."
    )


# The workspace ships Python 3, node, and a static server; browser automation
# runs on the shared browser sidecar service (the browser_* tools), not in
# the workspace. Weak models otherwise burn iterations (and spike host load)
# trying to ``pip``/``apt-get install playwright`` — which was observed
# crashing builds. State the substrate up front so they don't.
_SUBSTRATE_NOTE = (
    "\n\nSUBSTRATE: This workspace ALREADY has Python 3, node, and a static "
    "file server installed and ready. The browser_* tools drive a real "
    "browser that runs OUTSIDE the workspace — NEVER run pip / apt-get / npm "
    "to install a browser, Playwright, or a server. Drive the app with the "
    "browser_* tools and serve it with service_start."
)


def build_system_prompt(power_registry: Any | None, power_id: str = "frontend-app") -> str:
    """Render the Builder Power into a system prompt, with a safe fallback."""
    if power_registry is not None:
        try:
            manifest = power_registry.get_power(power_id)
            if manifest is not None:
                block = manifest.render_prompt_block()
                if block and block.strip():
                    return (
                        "You are an autonomous web-app builder. Follow the active "
                        "Power's definition of done exactly.\n\n" + block + _SUBSTRATE_NOTE
                    )
        except Exception:  # noqa: BLE001 — never let power lookup break a build
            log.warning("builder.power_render_failed", power_id=power_id, exc_info=True)
    return _default_system_prompt() + _SUBSTRATE_NOTE


def build_initial_message(objective: str, *, project_name: str) -> str:
    """The build task handed to the agent."""
    return (
        f"Build this application: {objective}\n\n"
        f"Project name: {project_name}\n"
        "Workspace root is /workspace — write all files there. When you're done "
        "AND have verified every behavior in a real browser, call finish_task "
        "with a one-line summary of what you built and which checks passed."
    )


# How each prior stop reason is explained to the agent on resume — so a
# budget-exhausted build is told it was mid-work (continue), not that it failed.
_STOP_REASON_EXPLANATIONS = {
    "budget": (
        "You ran out of your iteration/token budget before finishing — you were "
        "most likely mid-build or mid-verification, not actually done."
    ),
    "stuck": "You got stuck repeating the same action and the loop was halted.",
    "error": "A backend error interrupted the run before you finished.",
    "cancelled": "The run was cancelled before you finished.",
    "canceled": "The run was cancelled before you finished.",
    "complete": (
        "Your previous build finished and was published. The user wants to "
        "change or extend it — keep what works and make the requested changes."
    ),
}


def _steps_digest(prior_steps: list | None, *, limit: int = 15) -> str:
    """Compact 'what you just did' digest from the persisted tool trail."""
    if not prior_steps:
        return ""
    lines: list[str] = []
    for s in prior_steps[-limit:]:
        if not isinstance(s, dict):
            continue
        tool = str(s.get("tool") or "").strip()
        if not tool:
            continue
        preview = str(s.get("preview") or "").replace("\n", " ").strip()[:80]
        lines.append(f"  - {tool}{(': ' + preview) if preview else ''}")
    return "\n".join(lines)


def build_resume_message(
    objective: str,
    *,
    project_name: str,
    instructions: str = "",
    prior_stop_reason: str = "",
    prior_steps: list | None = None,
) -> str:
    """The continuation task handed to the agent on resume.

    The workspace IS the checkpoint: the agent's prior files are already on
    disk, so the message tells it to orient by reading the workspace and the
    running app rather than starting over. Carries why the prior run stopped,
    a digest of recent steps, and any new user instructions (re-prompt)."""
    reason = _STOP_REASON_EXPLANATIONS.get(
        (prior_stop_reason or "").lower(), "The previous run did not finish."
    )
    parts: list[str] = [
        "You are CONTINUING work on an app you already started in THIS "
        "workspace (/workspace) — your previous files are still here. Do NOT "
        "start over or recreate the project from scratch.",
        "",
        f"Original objective: {objective}",
        f"Project name: {project_name}",
        "",
        f"Why the previous run stopped: {reason}",
    ]
    digest = _steps_digest(prior_steps)
    if digest:
        parts += ["", "The most recent things you did:", digest]
    if (instructions or "").strip():
        parts += [
            "",
            "New instructions from the user — apply these as part of "
            f"continuing:\n{instructions.strip()}",
        ]
    else:
        parts += [
            "",
            "No new instructions — finish the original objective: complete "
            "anything unfinished and verify every behavior.",
        ]
    parts += [
        "",
        "Start by orienting yourself: run dir_tree and read the key files to "
        "see the current state, then service_start (if it isn't already "
        "serving) and browser_open the app to see what works right now. Then "
        "continue the build-test-fix loop. Drive every behavior in a real "
        "browser and assert it before you call finish_task.",
    ]
    return "\n".join(parts)


# How many times a build re-enters the loop to fix behaviors the gate found
# broken before we publish what we have (marked unverified with the failures).
_MAX_FIX_ROUNDS = 2

# Stop reasons that are NOT failures — the build hit a checkpoint (budget) or
# went in circles (stuck), but the workspace persists and the user is offered a
# continue (resume) or stop. Only genuine errors / cancellation are terminal.
# A hard limit is one bad task away from being a bug; a checkpoint never is.
PAUSE_STOP_REASONS = frozenset({"budget", "stuck"})


def build_fix_message(objective: str, *, project_name: str, behaviors: list[dict]) -> str:
    """The fix-loop task: an independent browser check found specific behaviors
    broken — fix the root cause and they'll be re-checked."""
    return (
        "An automated browser check just ran against your app in this workspace "
        "(/workspace) and found real defects. Your files are already here.\n\n"
        f"Original objective: {objective}\n"
        f"Project name: {project_name}\n\n"
        + render_failures_for_fix(behaviors)
        + "\n\nFix the ROOT CAUSE of each failing behavior in the code (not a "
        "workaround), then re-run the app in the browser yourself to confirm the "
        "fix before calling finish_task. The same checks will run again."
    )


# ---------------------------------------------------------------------------
# Orchestration (Docker + backend — integration path)
# ---------------------------------------------------------------------------

async def run_build(
    *,
    objective: str,
    user_id: str,
    backend: Any,
    model: str,
    container_manager: Any,
    artifact_store: Any,
    build_run_store: Any,
    power_registry: Any | None = None,
    profile_id: str = "static",
    session_id: str = "",
    build_id: str = "",
    event_sink: Callable[[str, dict, bool], None] | None = None,
    budget: Any | None = None,
    tooling_profile: str = "browser",
    reuse_workspace_id: str = "",
    resume: bool = False,
    instructions: str = "",
    prior_steps: list | None = None,
    prior_stop_reason: str = "",
    kind: str = "",
    create_row: bool = True,
) -> dict:
    """Run one autonomous build to completion and publish it to the library.

    Returns a result dict: ``{build_id, workspace_id, artifact_id, status,
    stop_reason, files, name, verdict, qualityStatus}``. Never raises under
    normal operation — failures are captured into the build_run row and the
    returned dict.

    Resume / re-prompt (workspace-as-checkpoint): pass ``resume=True`` with
    ``reuse_workspace_id`` (a workspace the caller has already ensured is
    live — the existing one, or one rebuilt from the artifact) to continue an
    existing ``build_runs`` row instead of creating a fresh build. The caller
    is responsible for flipping the row back to ``running`` (via
    ``BuildRunStore.begin_resume``) before calling. ``instructions`` carries
    the user's new asks (re-prompt); ``prior_steps`` / ``prior_stop_reason``
    seed the continuation prompt.

    Parameters mirror bug_finder's orchestrator: the caller resolves the
    ``backend``/``model`` and passes the stores in, so this stays decoupled
    from ``app.state`` and testable in integration.
    """
    # Deferred heavy imports — keep the pure helpers above import-light.
    from augmentum.agents.budget import SubagentBudget
    from augmentum.agents.loop import SubagentSpec, run_subagent
    from augmentum.agents.tools import filter_tools
    from augmentum.builds.budgets import build_budget
    from augmentum.coder.builder_tools import create_builder_tools
    from augmentum.coder.state import CoderState
    from augmentum.coder.tools import create_coder_tools
    from augmentum.tools.artifact_application import derive_project_name

    if not user_id:
        raise ValueError("run_build requires user_id")

    bid = build_id or f"build_{uuid.uuid4().hex[:16]}"
    project_name = derive_project_name(objective)
    quality_kind = (kind or profile_id or "static").strip().lower()

    def _emit(event_kind: str, payload: dict, terminal: bool = False) -> None:
        if event_sink is None:
            return
        try:
            event_sink(event_kind, {"build_id": bid, **payload}, terminal)
        except Exception:  # noqa: BLE001 — a misbehaving sink must not break the build
            log.debug("builder.event_sink_failed", kind=event_kind, exc_info=True)

    # 1. Persist the build run up front so the monitor/library show it building.
    #    A resume reuses the existing row (already flipped to running by the
    #    caller via begin_resume) — do NOT create a second row.
    if not resume and create_row:
        await build_run_store.create(
            user_id=user_id, build_id=bid, session_id=session_id,
            kind="application", status="running", name=project_name,
            request={"objective": objective, "model": model},
            profile_id=profile_id, target="inline",
        )
    _emit("stage", {"stage": "resuming" if resume else "starting",
                    "progress": 0.02, "name": project_name})

    workspace_id = ""
    artifact_id = ""
    files: list[dict] = []
    stop_reason = "error"
    error_text = ""

    try:
        # 2. Get the workspace: reuse the caller-provided live one (resume /
        #    rebuilt-from-artifact), else create a fresh one (cold start).
        if reuse_workspace_id:
            workspace_id = reuse_workspace_id
        else:
            info = await container_manager.create_workspace(
                name=project_name, publish_ports=True,
                tooling_profile=tooling_profile, user_id=user_id,
            )
            workspace_id = getattr(info, "id", "") or ""
        await build_run_store.update(bid, user_id=user_id, workspace_id=workspace_id)
        _emit("stage", {"stage": "workspace_ready", "progress": 0.08,
                        "workspace_id": workspace_id})

        # 3. Assemble the tool set: filtered coder tools + builder resource tools.
        state = CoderState(session_id=session_id or bid, workspace_id=workspace_id)
        coder_tools = filter_tools(
            create_coder_tools(container_manager, workspace_id, state, user_id=user_id),
            BUILDER_CODER_TOOL_NAMES,
        )
        tools = tuple(coder_tools) + tuple(create_builder_tools())

        # 4. Progress bridge: SSE event + a coarse build_run progress snapshot.
        async def _on_progress(progress: SubagentProgress) -> None:
            snap = map_progress(progress)
            _emit("build_progress", snap)
            if progress.phase in ("tool_call", "done"):
                try:
                    await build_run_store.update(bid, user_id=user_id, progress=snap)
                except Exception:  # noqa: BLE001
                    log.debug("builder.progress_persist_failed", exc_info=True)

        # 4b. Derive the behavior contract from the OBJECTIVE up front (frozen,
        #     spec-grounded — the anti-gaming anchor). The agent builds toward
        #     these, and the gate checks the SAME ones. Best-effort: an empty
        #     contract falls back to the trail-based verdict.
        _emit("stage", {"stage": "planning_checks", "progress": 0.10})
        behaviors = await derive_behaviors(
            backend, model=model, objective=objective, kind=quality_kind,
        )

        # 5. Drive the autonomous loop.
        if resume:
            initial_message = build_resume_message(
                objective, project_name=project_name, instructions=instructions,
                prior_stop_reason=prior_stop_reason, prior_steps=prior_steps,
            )
        else:
            initial_message = build_initial_message(objective, project_name=project_name)
        contract_block = render_behaviors_for_build(behaviors)
        if contract_block:
            initial_message = initial_message + "\n\n" + contract_block
        spec = SubagentSpec(
            role="builder",
            model=model,
            system_prompt=build_system_prompt(power_registry, "frontend-app"),
            initial_user_message=initial_message,
            tools=tools,
            # A full build-test-fix loop is far heavier than the bug-finder
            # probe SubagentBudget defaults (200k tok) were tuned for: each
            # iteration carries file contents + browser snapshots, and a
            # thinking model spends reasoning tokens on top. 200k trips at
            # ~18 iterations mid-fix; a thorough agent on a stateful app (kanban)
            # hit max_iterations=50 with 41 assertions still mid-verify and never
            # reached finish_task. Give a comprehensive build room to actually
            # finish + self-certify; remote flash models make this cheap.
            budget=budget or build_budget(model, "build"),
            instance_id=bid,
            progress_callback=_on_progress,
        )
        _emit("stage", {"stage": "building", "progress": 0.15})
        result = await run_subagent(spec, backend=backend)
        stop_reason = result.stop_reason
        if stop_reason != "complete":
            error_text = result.stop_detail or result.recovery_hint or stop_reason

        # 5b. Outcome gate + bounded fix loop. Run the spec-derived behavior
        #     contract against the REAL running app; on failures, re-enter the
        #     loop with concrete defect feedback and re-check. Runs BEFORE the
        #     archive so the published artifact reflects the fixes. Skipped when
        #     no contract was derived or the build didn't finish — then the
        #     trail-based verdict is the fallback.
        gate_ran = False
        if behaviors and stop_reason == "complete":
            for fix_round in range(_MAX_FIX_ROUNDS + 1):
                _emit("stage", {"stage": "verifying", "progress": 0.70,
                                "behaviors": behaviors})
                behaviors, gate_ran = await run_behavior_gate(
                    container_manager=container_manager, workspace_id=workspace_id,
                    backend=backend, model=model, behaviors=behaviors,
                )
                summary = gate_summary(behaviors)
                _emit("build_progress", {
                    "phase": "tool_call", "tool": "behavior_gate", "iteration": fix_round,
                    "preview": f"{summary['passed']}/{summary['checked']} behaviors passing",
                })
                if not gate_ran or summary["all_passed"] or fix_round == _MAX_FIX_ROUNDS:
                    break
                _emit("stage", {"stage": "fixing", "progress": 0.78,
                                "behaviors": behaviors})
                fix_spec = SubagentSpec(
                    role="builder", model=model,
                    system_prompt=build_system_prompt(power_registry, "frontend-app"),
                    initial_user_message=build_fix_message(
                        objective, project_name=project_name, behaviors=behaviors,
                    ),
                    tools=tools,
                    budget=SubagentBudget(
                        max_iterations=40, max_wallclock_seconds=1200.0,
                        max_tokens=1_500_000,
                    ),
                    instance_id=f"{bid}_fix{fix_round + 1}",
                    progress_callback=_on_progress,
                )
                # The app was already complete; a fix attempt's stop_reason
                # doesn't downgrade the build — the gate decides quality.
                result = await run_subagent(fix_spec, backend=backend)

        # 6. Snapshot the workspace into a publishable file list.
        _emit("stage", {"stage": "collecting", "progress": 0.85})
        buf = bytearray()
        async for chunk in container_manager.workspace_archive_stream(workspace_id, excludes=None):
            buf += chunk
        files = extract_text_files_from_targz(bytes(buf))

        # 7. Publish to the library (artifact + source_json), even on a partial
        #    build — the workspace stays live for "Open in Code" continuation.
        if files:
            _emit("stage", {"stage": "publishing", "progress": 0.95})
            source_json = build_source_json(
                name=project_name, files=files,
                profile_id=profile_id, target="inline",
            )
            saved = await artifact_store.save(
                data=zip_files(files),
                filename=f"{_safe_filename(project_name)}.zip",
                fmt="zip",
                display_name=project_name,
                source_json=source_json,
                user_id=user_id,
                metadata={"build_id": bid, "profile": profile_id, "workspace_id": workspace_id},
            )
            artifact_id = saved.get("id", "") if isinstance(saved, dict) else ""

        # Budget/stuck stops PAUSE (resumable checkpoint), they don't fail —
        # the workspace persists and the user decides continue vs stop. Only
        # genuine errors / cancellation are terminal-negative.
        if stop_reason == "complete":
            final_status = "completed"
        elif stop_reason in PAUSE_STOP_REASONS:
            final_status = "paused"
        else:
            final_status = "failed"

        # 8. Quality verdict. When the behavior gate ran, "done" is defined by
        #    how many spec-derived behaviors ACTUALLY PASSED in a real browser
        #    (outcome). When it couldn't run (no contract / no playwright), fall
        #    back to grading the agent's tool trail against the Power's floor.
        tool_names = [
            getattr(e, "tool", "") for e in (getattr(result, "tool_call_log", None) or [])
        ]
        if gate_ran:
            verdict = behavior_verdict(
                behaviors, status=final_status, artifact_ok=bool(artifact_id),
            )
            quality = behavior_quality_summary(verdict)
        else:
            verdict = judge_tool_names(
                tool_names, status=final_status,
                artifact_ok=bool(artifact_id), kind=quality_kind,
                has_files=bool(files),
            )
            quality = quality_summary(verdict, final_status=final_status)
        quality_status = quality["qualityStatus"]

        await build_run_store.update(
            bid, user_id=user_id, status=final_status,
            artifact_id=artifact_id or None, error=error_text or None,
            result={
                "artifact_id": artifact_id, "workspace_id": workspace_id,
                "file_count": len(files), "stop_reason": stop_reason,
                "iterations": getattr(result, "iterations", 0),
                "verdict": verdict,
                "qualityStatus": quality_status,
                "behaviors": behaviors,
                "project": {
                    "name": project_name,
                    "artifactId": artifact_id, "artifact_id": artifact_id,
                    "workspaceId": workspace_id,
                    "status": legacy_status(final_status),
                    "qualityStatus": quality_status,
                    "quality_status": quality_status,
                    "warnings": quality["warnings"],
                    "blockingErrors": quality["blockingErrors"],
                    "behaviors": behaviors,
                },
            },
        )
        _emit("stage", {"stage": "complete", "progress": 1.0, "status": final_status,
                        "stop_reason": stop_reason,
                        "awaiting_continue": final_status == "paused",
                        "artifact_id": artifact_id, "workspace_id": workspace_id,
                        "verdict": verdict, "qualityStatus": quality_status,
                        "warnings": quality["warnings"], "behaviors": behaviors,
                        "blockingErrors": quality["blockingErrors"]}, terminal=True)

        return {
            "build_id": bid, "workspace_id": workspace_id, "artifact_id": artifact_id,
            "status": final_status, "stop_reason": stop_reason, "name": project_name,
            "files": [f["path"] for f in files],
            "verdict": verdict, "qualityStatus": quality_status, "behaviors": behaviors,
        }

    except Exception as exc:  # noqa: BLE001 — surface as a failed build, never crash the caller
        log.warning("builder.run_failed", build_id=bid, error=str(exc), exc_info=True)
        try:
            await build_run_store.update(
                bid, user_id=user_id, status="failed", error=str(exc),
                workspace_id=workspace_id or None,
            )
        except Exception:  # noqa: BLE001
            log.debug("builder.failure_persist_failed", exc_info=True)
        _emit("stage", {"stage": "error", "error": str(exc),
                        "workspace_id": workspace_id}, terminal=True)
        return {
            "build_id": bid, "workspace_id": workspace_id, "artifact_id": artifact_id,
            "status": "failed", "stop_reason": "error", "error": str(exc),
            "name": project_name, "files": [],
        }


def _safe_filename(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_ " else "" for c in (name or "app"))
    return (keep.strip().replace(" ", "-").lower() or "app")[:60]
