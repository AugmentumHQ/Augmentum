"""Background chain execution manager.

Runs flow chains as background asyncio tasks, stores results, and pushes
notifications to subscribed SSE connections and voice WebSockets.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.tools.chain import build_synthesis_prompt, execute_chain
from augmentum.tools.custom_flows import flow_to_plan
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.base import InternalChatRequest, ModelBackend
    from augmentum.tools.registry import ToolRegistry

log = get_logger(__name__)

_TASK_TTL_SECONDS = 3600  # 1 hour


@dataclass
class BackgroundTask:
    """State of a single background chain execution."""

    task_id: str
    flow_name: str
    flow_id: str
    session_id: str
    query: str
    user_id: str = ""
    status: str = "running"  # running | completed | failed
    created_at: float = field(default_factory=time.monotonic)
    completed_at: float | None = None
    result_summary: str = ""
    error: str = ""
    injected: bool = False  # Whether result has been injected into context


class BackgroundChainManager:
    """Manages background chain execution, result storage, and notification dispatch."""

    def __init__(
        self,
        max_per_session: int = 5,
        max_total: int = 50,
        provider_registry: object | None = None,
    ) -> None:
        self._tasks: dict[str, BackgroundTask] = {}
        # Cache keys include user_id so two tenants can't collide on a
        # shared session_id. See CLAUDE.md multi-tenancy rules.
        self._session_tasks: dict[tuple[str, str], list[str]] = {}
        self._async_tasks: dict[str, asyncio.Task] = {}  # task_id -> asyncio.Task
        self._notification_queues: dict[tuple[str, str], list[asyncio.Queue]] = {}
        self._max_per_session = max_per_session
        self._max_total = max_total
        self._provider_registry = provider_registry
        self._lock = asyncio.Lock()

    def _sweep_completed(self) -> None:
        """Remove completed/failed tasks older than TTL."""
        now = time.monotonic()
        expired = [
            tid for tid, task in self._tasks.items()
            if task.status in ("completed", "failed")
            and task.completed_at is not None
            and (now - task.completed_at) > _TASK_TTL_SECONDS
        ]
        for tid in expired:
            self._tasks.pop(tid, None)
            self._async_tasks.pop(tid, None)
        # Clean session index
        for key in list(self._session_tasks):
            self._session_tasks[key] = [
                t for t in self._session_tasks[key] if t in self._tasks
            ]
            if not self._session_tasks[key]:
                del self._session_tasks[key]
        if expired:
            log.info("background_tasks_swept", count=len(expired))

    def shutdown(self) -> None:
        """Cancel all running tasks on server shutdown."""
        for atask in self._async_tasks.values():
            if not atask.done():
                atask.cancel()
        self._async_tasks.clear()
        self._tasks.clear()
        self._session_tasks.clear()
        self._notification_queues.clear()
        log.info("background_chain_manager_shutdown")

    async def launch(
        self,
        flow: dict,
        query: str,
        session_id: str,
        *,
        user_id: str = "",
        backend: ModelBackend | None = None,
        tool_registry: ToolRegistry | None = None,
        request_context: InternalChatRequest | None = None,
    ) -> str:
        """Launch a flow as a background task. Returns task_id.

        Raises ValueError if limits are exceeded.
        """
        async with self._lock:
            self._sweep_completed()
            key = (user_id, session_id)
            # Enforce per-session limit (per-user slice)
            session_task_ids = self._session_tasks.get(key, [])
            active = [
                tid for tid in session_task_ids
                if tid in self._tasks and self._tasks[tid].status == "running"
            ]
            if len(active) >= self._max_per_session:
                raise ValueError(
                    f"Max background tasks per session ({self._max_per_session}) reached"
                )

            # Enforce total limit
            total_active = sum(
                1 for t in self._tasks.values() if t.status == "running"
            )
            if total_active >= self._max_total:
                raise ValueError(
                    f"Max total background tasks ({self._max_total}) reached"
                )

            task_id = uuid.uuid4().hex[:12]
            task = BackgroundTask(
                task_id=task_id,
                flow_name=flow.get("name", "unknown"),
                flow_id=flow.get("id", ""),
                session_id=session_id,
                user_id=user_id,
                query=query,
            )
            self._tasks[task_id] = task

            if key not in self._session_tasks:
                self._session_tasks[key] = []
            self._session_tasks[key].append(task_id)

        # Launch the background asyncio task
        atask = asyncio.create_task(
            self._run_chain(task, flow, backend, tool_registry, request_context)
        )
        self._async_tasks[task_id] = atask

        log.info(
            "background_chain_launched",
            task_id=task_id,
            flow=flow.get("name"),
            session=session_id,
        )
        return task_id

    async def _run_chain(
        self,
        task: BackgroundTask,
        flow: dict,
        backend: ModelBackend | None,
        tool_registry: ToolRegistry | None,
        request_context: InternalChatRequest | None,
    ) -> None:
        """Execute the chain, update task state, push notification."""
        from augmentum.models.base import InternalChatRequest, Message

        try:
            # Resolve through the provider registry whenever available so a
            # blank request model inherits the user's primary chat model
            # before touching any backend-default model.
            resolved_model = (request_context.model if request_context else "") or ""
            if self._provider_registry:
                try:
                    resolved_be, resolved_model = await self._provider_registry.resolve_backend_with_fabric(  # type: ignore[union-attr]
                        resolved_model,
                    )
                    if resolved_be:
                        backend = resolved_be
                except Exception as exc:
                    log.debug("background_chain_model_resolve_failed", error=str(exc))  # fall through to default backend

            # If model is still empty, ask the backend for its first model
            # so internal LLM calls (replan, arg resolution, synthesis) send
            # a valid model name instead of "".
            if not resolved_model and backend:
                try:
                    models = await backend.list_models()
                    if models:
                        resolved_model = models[0].name
                except Exception as exc:
                    log.debug("background_chain_list_models_failed", error=str(exc))  # best-effort — some backends don't list models

            if not backend or not tool_registry:
                raise RuntimeError("Backend or tool registry not available")

            plan = flow_to_plan(flow)

            # Inject {{query}}
            for step in plan.steps:
                if step.input:
                    for k, v in step.input.items():
                        if isinstance(v, str) and "{{query}}" in v:
                            step.input[k] = v.replace("{{query}}", task.query)

            ctx = request_context or InternalChatRequest(
                model=resolved_model or "",
                messages=[Message(role="user", content=task.query)],
                stream=False,
            )
            # Ensure the context carries a valid model name for internal calls
            if resolved_model and not ctx.model:
                ctx.model = resolved_model

            # Constrain replan to the plan's tool set — see flow_routes.py
            # for the analogous fix in the synchronous flow runner.
            _bg_allowed = {s.tool for s in plan.steps if getattr(s, "tool", None)}
            results = await asyncio.wait_for(
                execute_chain(
                    plan, backend, tool_registry,
                    request_context=ctx,
                    allowed_tool_names=_bg_allowed or None,
                ),
                timeout=settings.passthrough_chain_timeout,
            )

            # Synthesize results
            synth_prompt = build_synthesis_prompt(
                plan, results, user_query=task.query,
            )
            synth_request = InternalChatRequest(
                model=ctx.model,
                messages=[
                    Message(role="user", content=task.query),
                    Message(role="user", content=synth_prompt),
                ],
                stream=False,
            )
            try:
                synth_response = await asyncio.wait_for(
                    backend.chat(synth_request),
                    timeout=settings.passthrough_chain_synthesis_timeout,
                )
                summary = synth_response.message.content if synth_response.message else ""
            except TimeoutError:
                log.warning(
                    "background_chain_synthesis_timeout",
                    task_id=task.task_id,
                    timeout=settings.passthrough_chain_synthesis_timeout,
                )
                parts = ["*(Background chain completed but synthesis timed out. Raw results:)*"]
                for r in results.values():
                    if hasattr(r, "success") and r.success and hasattr(r, "output"):
                        tool_name = getattr(r, "tool_name", "step")
                        parts.append(f"**{tool_name}:** {r.output[:500]}")
                summary = "\n\n".join(parts)

            # Cap result size
            max_chars = settings.passthrough_chain_bg_result_max_chars
            if len(summary) > max_chars:
                summary = summary[:max_chars] + "…[truncated]"

            task.status = "completed"
            task.completed_at = time.monotonic()
            task.result_summary = summary

            log.info(
                "background_chain_completed",
                task_id=task.task_id,
                flow=task.flow_name,
                elapsed=f"{task.completed_at - task.created_at:.1f}s",
            )

            await self._push_notification(task.user_id, task.session_id, {
                "type": "flow_complete",
                "task_id": task.task_id,
                "flow_name": task.flow_name,
                "status": "completed",
                "summary": summary,
                "elapsed_seconds": round(task.completed_at - task.created_at, 1),
            })

        except Exception as exc:
            task.status = "failed"
            task.completed_at = time.monotonic()
            task.error = str(exc)

            log.warning(
                "background_chain_failed",
                task_id=task.task_id,
                flow=task.flow_name,
                error=str(exc),
            )

            await self._push_notification(task.user_id, task.session_id, {
                "type": "flow_failed",
                "task_id": task.task_id,
                "flow_name": task.flow_name,
                "status": "failed",
                "error": str(exc),
            })

        finally:
            self._async_tasks.pop(task.task_id, None)

    async def _push_notification(self, user_id: str, session_id: str, event: dict) -> None:
        """Push an event to all notification queues for a (user, session)."""
        queues = self._notification_queues.get((user_id, session_id), [])
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("notification_queue_full", session=session_id)

    def subscribe(self, session_id: str, *, user_id: str = "") -> asyncio.Queue:
        """Create and return a notification queue for a (user, session)."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        key = (user_id, session_id)
        if key not in self._notification_queues:
            self._notification_queues[key] = []
        self._notification_queues[key].append(queue)
        return queue

    def unsubscribe(
        self, session_id: str, queue: asyncio.Queue, *, user_id: str = "",
    ) -> None:
        """Remove a specific notification queue."""
        key = (user_id, session_id)
        queues = self._notification_queues.get(key, [])
        if queue in queues:
            queues.remove(queue)
        if not queues:
            self._notification_queues.pop(key, None)

    def get_tasks(self, session_id: str, *, user_id: str = "") -> list[BackgroundTask]:
        """List all tasks for a (user, session)."""
        task_ids = self._session_tasks.get((user_id, session_id), [])
        return [self._tasks[tid] for tid in task_ids if tid in self._tasks]

    def get_task(self, task_id: str, *, user_id: str = "") -> BackgroundTask | None:
        """Fetch a task by id — returns None when the authenticated user is
        not the owner. Pass ``user_id=""`` only from trusted callers that have
        already verified ownership."""
        task = self._tasks.get(task_id)
        if task is None:
            return None
        if user_id and task.user_id != user_id:
            return None
        return task

    def get_pending_results(
        self, session_id: str, *, user_id: str = "",
    ) -> list[BackgroundTask]:
        """Get completed tasks that haven't been injected into context yet."""
        return [
            t for t in self.get_tasks(session_id, user_id=user_id)
            if t.status == "completed" and not t.injected
        ]

    def mark_injected(self, task_id: str, *, user_id: str = "") -> None:
        """Mark a task's results as injected into conversation context."""
        task = self._tasks.get(task_id)
        if task and (not user_id or task.user_id == user_id):
            task.injected = True

    def cleanup_session(self, session_id: str, *, user_id: str = "") -> None:
        """Cancel running tasks and clean up for a (user, session)."""
        key = (user_id, session_id)
        task_ids = self._session_tasks.pop(key, [])
        for tid in task_ids:
            atask = self._async_tasks.pop(tid, None)
            if atask and not atask.done():
                atask.cancel()
            self._tasks.pop(tid, None)
        self._notification_queues.pop(key, None)
