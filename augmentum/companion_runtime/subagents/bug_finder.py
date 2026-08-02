"""BugFinderSubagent — wraps ``augmentum.bug_finder.orchestrator.run_bug_finder``.

Eight-stage pipeline (intake → workspace prep → plan → detect → verify
→ fix → report). Adapter resolves runtime dependencies from
``ctx.runtime._app_state`` the same way other subagents do (see
``CoderSubagent``), builds a config from the intent's ``workspace_id``
metadata + the runtime's primary model, and drives the orchestrator
inline. No job-queue indirection — Becca's invocation runs synchronously
inside the dispatch turn. The REST API path is the queued route.

Returns a clean ``SubagentResult.error`` when the caller didn't supply
a workspace_id rather than silently picking one. Surfaces are
responsible for resolving the workspace before dispatching.
"""

from __future__ import annotations

from augmentum.companion_runtime.subagents.base import (
    SubagentBase,
    SubagentContext,
    SubagentResult,
)
from augmentum.companion_runtime.subagents.registry import SubagentRegistry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class BugFinderSubagent(SubagentBase):
    name = "bug_finder"
    description = (
        "Eight-stage bug-finding pipeline (intake/plan/detect/verify/"
        "fix/report). Best when intent describes a defect to track down "
        "or audit. Requires intent.metadata['workspace_id']."
    )
    role_affinity = ("collaborator",)
    focus_affinity = ("owner", "world")
    state_affinity = ("working",)

    async def invoke(self, ctx: SubagentContext) -> SubagentResult:
        try:
            from augmentum.bug_finder.orchestrator import (
                BugFinderIntake,
                BugFinderRunConfig,
                run_bug_finder,
            )
            from augmentum.bug_finder.role_models import RoleModelConfig
            from augmentum.companion_runtime import tiers
        except Exception as exc:
            return SubagentResult(
                content="", handled_by=self.name,
                error=f"bug_finder_import_failed: {exc!s}",
            )

        app_state = getattr(ctx.runtime, "_app_state", None)
        if app_state is None:
            return SubagentResult(
                content="", handled_by=self.name,
                error="bug_finder_unavailable: runtime has no app_state binding",
            )

        container_manager = getattr(app_state, "container_manager", None)
        provider_registry = getattr(app_state, "provider_registry", None)
        if container_manager is None or provider_registry is None:
            return SubagentResult(
                content="", handled_by=self.name,
                error=(
                    "bug_finder_unavailable: container_manager or "
                    "provider_registry not initialized"
                ),
            )

        workspace_id = str(
            ctx.intent.metadata.get("workspace_id") or "",
        ).strip()
        if not workspace_id:
            return SubagentResult(
                content="", handled_by=self.name,
                error=(
                    "bug_finder_requires_workspace: intent.metadata must "
                    "include 'workspace_id' (the coder workspace to audit). "
                    "Surfaces should resolve the active workspace before "
                    "dispatching."
                ),
            )

        try:
            _backend, primary_model = await tiers.primary(ctx.runtime)
        except Exception as exc:
            return SubagentResult(
                content="", handled_by=self.name,
                error=f"bug_finder_no_primary_model: {exc!s}",
            )

        verifier_model = str(
            ctx.intent.metadata.get("verifier_model") or "",
        ).strip()
        if not verifier_model:
            try:
                from augmentum.config import settings as _settings
                verifier_model = (
                    getattr(_settings, "heavyweight_model", "") or ""
                ).strip()
            except Exception:
                verifier_model = ""

        focus_paths_raw = ctx.intent.metadata.get("focus_paths") or []
        if isinstance(focus_paths_raw, str):
            focus_paths_raw = [focus_paths_raw]
        focus_paths = tuple(
            str(p).strip() for p in focus_paths_raw if str(p).strip()
        )
        threat_model = str(
            ctx.intent.metadata.get("threat_model") or "",
        ).strip()

        try:
            role_models = RoleModelConfig.from_primary(
                primary_model, verifier=verifier_model,
            )
        except ValueError as exc:
            return SubagentResult(
                content="", handled_by=self.name,
                error=f"bug_finder_invalid_role_models: {exc!s}",
            )

        intake = BugFinderIntake(
            workspace_id=workspace_id,
            focus_paths=focus_paths,
            threat_model=threat_model,
        )
        config = BugFinderRunConfig(intake=intake, role_models=role_models)

        async def _resolve_backend(model_name: str):
            return await provider_registry.resolve_backend_with_fabric(model_name)

        await ctx.bus.publish_topic(
            "subagent.invoked",
            {"name": self.name, "invocation_id": ctx.invocation_id},
            source_companion_id=ctx.companion_id,
        )
        try:
            report = await run_bug_finder(
                config,
                resolve_backend=_resolve_backend,
                container_manager=container_manager,
                user_id=ctx.intent.user_id,
                job_ctx=None,
            )
        except Exception as exc:
            log.exception("bug_finder_failed", error=str(exc))
            return SubagentResult(
                content="", handled_by=self.name,
                error=f"bug_finder_failed: {exc!s}",
            )
        finally:
            await ctx.bus.publish_topic(
                "subagent.completed",
                {"name": self.name, "invocation_id": ctx.invocation_id},
                source_companion_id=ctx.companion_id,
            )

        confirmed = sum(1 for f in report.findings if f.status == "confirmed")
        fixed = sum(1 for f in report.findings if f.status == "fixed")
        fix_failed = sum(
            1 for f in report.findings if f.status == "fix_failed"
        )
        unconfirmable = sum(
            1 for f in report.findings if f.status == "unconfirmable"
        )
        summary_lines = [
            f"Bug Finder run {report.run_id}: {report.stop_reason}.",
            (
                f"Findings: {len(report.findings)} total — "
                f"{fixed} fixed, {confirmed} confirmed, "
                f"{fix_failed} fix-failed, {unconfirmable} unconfirmable."
            ),
        ]
        if report.same_model_self_verification:
            summary_lines.append(
                "Note: single-model self-verification "
                "(verifier == fixer). Read findings with care.",
            )
        if report.stop_detail:
            summary_lines.append(f"Detail: {report.stop_detail}")

        return SubagentResult(
            content="\n".join(summary_lines),
            handled_by=self.name,
            metadata={
                "run_id": report.run_id,
                "stop_reason": report.stop_reason,
                "findings_total": len(report.findings),
                "findings_fixed": fixed,
                "findings_confirmed": confirmed,
                "findings_fix_failed": fix_failed,
                "findings_unconfirmable": unconfirmable,
                "same_model_self_verification": (
                    report.same_model_self_verification
                ),
            },
        )


SubagentRegistry.register(BugFinderSubagent)
