"""LLM-based consistency checker tool."""

from __future__ import annotations

from augmentum.models.base import InternalChatRequest, Message, ModelBackend
from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_CONSISTENCY_SYSTEM_PROMPT = """\
You are a rigorous logical consistency checker. You will be given a list of \
statements and optional context. Your job is to:

1. Identify any logical contradictions between the statements.
2. Identify any statements that are internally inconsistent.
3. Identify any statements that contradict the provided context.

Respond in the following structured format:

CONTRADICTIONS_FOUND: <yes or no>
NUM_CONTRADICTIONS: <integer>

For each contradiction found:
CONTRADICTION_<N>:
- STATEMENT_A: <text of first conflicting statement>
- STATEMENT_B: <text of second conflicting statement>
- EXPLANATION: <why these contradict>

If no contradictions are found, state:
NO_CONTRADICTIONS: The statements are logically consistent.
"""


class ConsistencyCheckTool(Tool):
    """Check a set of statements for logical consistency using an LLM."""

    @property
    def name(self) -> str:
        return "consistency_check"

    @property
    def description(self) -> str:
        return "Check a set of statements for logical consistency"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "statements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of statements to check for consistency",
                },
                "context": {
                    "type": "string",
                    "description": "Optional context for the statements",
                    "default": "",
                },
            },
            "required": ["statements"],
        }

    def __init__(self, backend: ModelBackend, model: str = "llama3.1:8b") -> None:
        self._backend = backend
        self._model = model

    def validate_input(self, **kwargs) -> bool:
        statements = kwargs.get("statements")
        return isinstance(statements, list) and len(statements) > 0

    async def execute(
        self,
        *,
        statements: list[str],
        context: str = "",
    ) -> ToolResult:
        """Run a consistency check over the provided statements."""
        if not statements:
            return ToolResult(success=False, error="No statements provided")

        # Build the user prompt.
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(statements))
        user_prompt = f"Statements to check:\n{numbered}"
        if context:
            user_prompt += f"\n\nContext:\n{context}"

        request = InternalChatRequest(
            model=self._model,
            messages=[
                Message(role="system", content=_CONSISTENCY_SYSTEM_PROMPT),
                Message(role="user", content=user_prompt),
            ],
            stream=False,
        )

        try:
            response = await self._backend.chat(request)
        except Exception as exc:
            log.warning("consistency_check_failed", error=str(exc))
            return ToolResult(success=False, error=f"Backend call failed: {exc}")

        output = response.message.content if response.message else ""

        # Parse structured fields from the LLM response.
        has_contradictions = "CONTRADICTIONS_FOUND: yes" in output.lower().replace(
            "contradictions_found: yes", "CONTRADICTIONS_FOUND: yes"
        )
        # A simpler detection: look for the literal string.
        has_contradictions = "contradictions_found: yes" in output.lower()

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "num_statements": len(statements),
                "has_contradictions": has_contradictions,
            },
        )
