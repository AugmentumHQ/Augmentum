"""Cross-step working memory for agentic execution.

Accumulates tool chain results and generative step outputs across the
lifecycle of an agentic task, providing formatted context for subsequent
steps.  This is what allows a Draft step to reference what the Research
chain discovered, without the model needing to hold it all in context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.tools.chain import StepResult

log = get_logger(__name__)


class WorkingMemory:
    """Accumulates cross-step context during agentic task execution.

    Two kinds of step output are tracked:

    * **Chain results** — ``dict[int, StepResult]`` from ``execute_chain()``.
      These are tool execution outputs with success/failure status.
    * **Generative outputs** — raw LLM text from draft/review/deliver steps.

    The memory provides formatted context blocks for injection into
    subsequent LLM calls, keeping each call focused while aware of
    everything the task has accomplished so far.
    """

    def __init__(self, goal: str, plan_md: str = "") -> None:
        self.goal = goal
        self.plan_md = plan_md

        # Ordered record of all step outputs
        self._steps: list[tuple[str, str, str]] = []  # (name, kind, summary)

        # Detailed chain results (for rich context when needed)
        self._chain_results: dict[str, dict[int, StepResult]] = {}

        # Generative outputs (full text, truncated on retrieval)
        self._generative_outputs: dict[str, str] = {}

        # Artifacts produced
        self._artifacts: list[dict] = []

        # Structured review verdicts (step_name -> {verdict, reasoning, issues})
        self._review_verdicts: dict[str, dict] = {}

        # Counters
        self.total_tool_calls: int = 0
        self.total_llm_calls: int = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_chain_results(
        self, step_name: str, results: dict[int, StepResult],
    ) -> None:
        """Record tool chain results for a step."""
        self._chain_results[step_name] = results

        # Build summary
        parts = []
        for r in results.values():
            status = "OK" if r.success else "FAILED"
            preview = r.output[:300].replace("\n", " ")
            parts.append(f"[{r.tool_name} ({status})]: {preview}")
            self.total_tool_calls += 1

        summary = "\n".join(parts)
        self._steps.append((step_name, "chain", summary))
        log.info(
            "wmem_chain_recorded",
            step=step_name,
            tools=len(results),
            success=sum(1 for r in results.values() if r.success),
        )

    def record_generative_output(self, step_name: str, output: str) -> None:
        """Record a generative LLM step's output."""
        self._generative_outputs[step_name] = output
        summary = output[:500].replace("\n", " ")
        self._steps.append((step_name, "generative", summary))
        self.total_llm_calls += 1

    def record_review_verdict(
        self,
        step_name: str,
        verdict: str,
        reasoning: str = "",
        issues: list[str] | None = None,
    ) -> None:
        """Record a structured verdict from a review step's submit_review tool call."""
        self._review_verdicts[step_name] = {
            "verdict": verdict.upper().strip(),
            "reasoning": reasoning,
            "issues": list(issues or []),
        }

    def get_review_verdict(self, step_name: str) -> dict | None:
        """Return a recorded structured review verdict, or None if absent."""
        return self._review_verdicts.get(step_name)

    def record_artifact(self, artifact_meta: dict) -> None:
        """Record metadata about a created artifact."""
        self._artifacts.append(artifact_meta)

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def build_context_for_step(
        self, current_step: str, max_chars: int = 6000,
    ) -> str:
        """Build a context block from all prior steps for injection into an LLM call.

        Budget is allocated **proportionally to content size** — a 10,000-char
        draft gets far more space than a 200-char review output.  Each step is
        capped at 50% of the total budget to prevent a single step from starving
        others, with a 200-char floor so nothing is completely dropped.
        """
        if not self._steps:
            return ""

        # Compute actual content lengths for proportional allocation
        content_lengths: list[int] = []
        for name, kind, summary in self._steps:
            if kind == "chain":
                length = len(self.build_chain_context(name))
            else:
                length = len(self._generative_outputs.get(name, summary))
            content_lengths.append(max(length, 1))

        total_content = sum(content_lengths)
        _MIN_BUDGET = 200
        _MAX_SHARE = 0.5
        max_per_step = int(max_chars * _MAX_SHARE)

        step_budgets: list[int] = []
        for length in content_lengths:
            share = length / total_content
            budget = int(max_chars * share)
            budget = max(_MIN_BUDGET, min(budget, max_per_step))
            step_budgets.append(budget)

        # Scale down if total exceeds max_chars
        total_allocated = sum(step_budgets)
        if total_allocated > max_chars:
            scale = max_chars / total_allocated
            step_budgets = [max(_MIN_BUDGET, int(b * scale)) for b in step_budgets]

        sections: list[str] = []
        for i, (name, kind, summary) in enumerate(self._steps):
            budget = step_budgets[i] if i < len(step_budgets) else 200

            if kind == "chain":
                # Use full chain results up to budget
                full_context = self.build_chain_context(name)
                if len(full_context) > budget:
                    full_context = full_context[:budget] + "\n...[truncated]"
                sections.append(f"### {name} (tool results)\n{full_context}")
            else:
                # Use full generative output up to budget
                output = self._generative_outputs.get(name, summary)
                if len(output) > budget:
                    output = output[:budget] + "\n...[truncated]"
                sections.append(f"### {name}\n{output}")

        text = "## Prior Results\n\n" + "\n\n".join(sections)

        # Hard cap
        if len(text) > max_chars:
            text = "## Prior Results (truncated)\n\n..." + text[-(max_chars - 50):]

        return text

    def build_chain_context(self, step_name: str) -> str:
        """Build focused context from chain results for a specific prior step.

        Returns tool outputs with metadata URLs surfaced. Each result is
        capped at 1500 chars to keep total context manageable across
        multi-step flows. Duplicate outputs (same tool called multiple
        times) are deduplicated by content hash.
        """
        results = self._chain_results.get(step_name)
        if not results:
            return self._generative_outputs.get(step_name, "")

        parts = []
        seen_hashes: set[int] = set()
        for r in results.values():
            # Deduplicate: skip if output content is identical to a prior result
            content_hash = hash(r.output[:500])
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            status = "OK" if r.success else "FAILED"
            output = r.output[:1500]
            # Surface image/artifact URLs so downstream steps can reference them
            if r.metadata:
                if r.metadata.get("url"):
                    output += f"\nImage URL: {r.metadata['url']}"
                if r.metadata.get("prompt"):
                    output += f"\nPrompt: {r.metadata['prompt'][:200]}"
                if r.metadata.get("download_url"):
                    output += f"\nDownload: {r.metadata['download_url']}"
            parts.append(f"## {r.tool_name} ({status})\n{output}")
        return "\n\n".join(parts)

    def to_synthesis_context(self) -> str:
        """Build the final context block for the synthesis/deliver step.

        Includes all step outputs with full detail for the most recent
        chain results, suitable for injection into the final LLM call.
        """
        parts = [f"Task goal: {self.goal}\n"]

        for name, kind, _ in self._steps:
            if kind == "chain":
                results = self._chain_results.get(name, {})
                chain_parts = []
                for r in results.values():
                    status = "OK" if r.success else "FAILED"
                    chain_parts.append(
                        f"  [{r.tool_name} ({status})]: {r.output[:1500]}"
                    )
                parts.append(f"## {name}\n" + "\n".join(chain_parts))
            else:
                output = self._generative_outputs.get(name, "")
                parts.append(f"## {name}\n{output[:2000]}")

        if self._artifacts:
            artifact_lines = [
                f"  - {a.get('display_name', a.get('filename', '?'))} ({a.get('format', '?')})"
                for a in self._artifacts
            ]
            parts.append("## Artifacts Created\n" + "\n".join(artifact_lines))

        return "\n\n".join(parts)

    @property
    def last_output(self) -> str:
        """Return the output of the most recent step."""
        if not self._steps:
            return ""
        name, kind, _ = self._steps[-1]
        if kind == "generative":
            return self._generative_outputs.get(name, "")
        # For chain steps, return the summary
        return self._steps[-1][2]

    @property
    def all_step_names(self) -> list[str]:
        """Return names of all recorded steps in order."""
        return [name for name, _, _ in self._steps]

    def get_step_output(self, step_name: str) -> str:
        """Get the output of a specific step by name."""
        if step_name in self._generative_outputs:
            return self._generative_outputs[step_name]
        # Check chain results
        results = self._chain_results.get(step_name)
        if results:
            parts = []
            for r in results.values():
                if r.success:
                    parts.append(r.output[:1000])
            return "\n".join(parts)
        return ""

    def format_for_plan_context(self) -> str:
        """Compact format for injection alongside the plan attention anchor."""
        if not self._steps:
            return ""
        lines = ["Completed:"]
        for name, _kind, summary in self._steps:
            preview = summary[:100].replace("\n", " ")
            lines.append(f"  - {name}: {preview}")
        return "\n".join(lines)

    def format_for_create_context(self, max_chars: int = 24000) -> str:
        """Rich context for artifact-creation steps (Create Book, etc.).

        Provides the full generative draft and all image URLs so the chain
        planner can construct complete tool calls.  Uses a larger budget
        than ``format_for_plan_context`` because artifact creation needs
        the full content to work correctly.
        """
        if not self._steps:
            return ""

        parts: list[str] = []

        for name, kind, _summary in self._steps:
            if kind == "chain":
                # Include full image URLs from chain results
                results = self._chain_results.get(name, {})
                chain_lines: list[str] = []
                for r in results.values():
                    status = "OK" if r.success else "FAILED"
                    line = f"[{r.tool_name} ({status})]: {r.output[:200]}"
                    if r.metadata:
                        if r.metadata.get("url"):
                            line += f"\nImage URL: {r.metadata['url']}"
                        if r.metadata.get("prompt"):
                            line += f"\nPrompt: {r.metadata['prompt'][:150]}"
                    chain_lines.append(line)
                parts.append(f"## {name}\n" + "\n\n".join(chain_lines))
            else:
                # Include full generative output (draft text)
                output = self._generative_outputs.get(name, "")
                parts.append(f"## {name}\n{output}")

        text = "\n\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return text
