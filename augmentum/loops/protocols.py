"""Dependency-injection protocols for the LoopRunner.

The runner accepts each capability as a Protocol so the coder + agentic
modes can adapt their existing classes without inheritance changes.
None of these protocols is concrete here — they describe what the
runner expects to call. PR-2.4 onward implements the actual wiring.

Why protocols and not abstract classes
--------------------------------------
The coder has a rich :class:`ContainerManager` that already implements
``file_read``, ``file_write``, ``shell_exec`` etc. against a Docker
container. The agentic mode has no container; its file ops resolve to
whatever the LLM emitted in earlier steps. Both can satisfy the same
``FileIO`` protocol without sharing a base class. Duck-typing keeps the
mode-specific code unchanged.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class FileIO(Protocol):
    """Read/write the substrate the loop operates against.

    For coder, this is the workspace container's `/workspace/` tree.
    For agentic, this could be an in-memory artifact tree or a
    project's bare repo (mounted/cloned per PR-1.2).
    """

    async def file_read(self, path: str) -> str:
        """Return the file's contents as text. Raise ``FileNotFoundError``
        when missing — the runner translates that into a recorded tool
        failure rather than a fatal error."""
        ...

    async def file_write(self, path: str, content: str) -> None:
        """Replace the file's contents. Parent dirs are created as
        needed. Raise on permission errors."""
        ...

    async def file_exists(self, path: str) -> bool:
        """Cheap probe used by the verify-gate before reading."""
        ...


@runtime_checkable
class ToolExecutor(Protocol):
    """Dispatch a single tool call.

    The runner emits tool invocations parsed from the model's response
    (whether native function-calls or text-extracted) and delegates the
    actual execution here. Returning ``ToolResult``-shaped dicts keeps
    the runner agnostic to the mode's specific tool registry.
    """

    async def execute(
        self,
        *,
        name: str,
        args: dict[str, Any],
        call_id: str,
    ) -> dict[str, Any]:
        """Execute one tool call.

        Return shape (informal): ``{"ok": bool, "result": ..., "error":
        str, "metadata": {...}}``. The runner records non-ok results
        into the observation ledger (PR-2.2) and applies soft breakers
        (PR-2.3) based on the failure signature.
        """
        ...

    def list_available(self) -> list[dict[str, Any]]:
        """Return the JSON schema list to advertise to the model.
        Same shape the OpenAI/Anthropic tool definitions expect."""
        ...


@runtime_checkable
class PermissionGate(Protocol):
    """Authorise (or veto) a sensitive tool call before execution.

    Coder uses this for shell_exec on production paths; the agentic
    handler uses it for approval-gated steps. Returning False halts
    the call; the runner records a "denied" outcome in the ledger.
    """

    async def authorize(
        self,
        *,
        tool: str,
        args: dict[str, Any],
    ) -> bool:
        ...


@runtime_checkable
class QuestionAsker(Protocol):
    """Surface an ``ask_user`` mid-loop question to the user.

    The runner suspends the iteration loop while awaiting the answer.
    Modes wire this to their existing streaming-question machinery —
    the coder's question-bubble UI, the agentic mode's pending-approval
    pattern, etc.
    """

    async def ask(self, *, question: str, context: dict[str, Any]) -> str:
        ...


@runtime_checkable
class ChunkEmitter(Protocol):
    """Sink for streaming chunks back to the caller.

    The runner pushes incremental updates (model deltas, tool starts /
    completes, sticky reminders) here. The mode's HTTP route forwards
    them to the client. The protocol is async-iterator-shaped so the
    run-broker can subscribe + replay.
    """

    async def emit(self, chunk: dict[str, Any]) -> None:
        ...

    def iterate(self) -> AsyncIterator[dict[str, Any]]:
        ...


__all__ = [
    "ChunkEmitter",
    "FileIO",
    "PermissionGate",
    "QuestionAsker",
    "ToolExecutor",
]
