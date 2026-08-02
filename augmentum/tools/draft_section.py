"""Draft section tool — generates one focused section of a document.

Used by the chain executor to parallelize document drafting.  Each
section gets its own LLM call with targeted research context and source
attribution, producing deeper content than a single-shot full-document
generation.

The chain executor's ``_resolve_args_via_llm`` populates the input
parameters (section title, key points, research context) from the
working memory of prior steps.  The tool makes a focused LLM call
with a content-generation system prompt and returns the section text
with inline source citations.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.base import ModelBackend
    from augmentum.models.provider_registry import ProviderRegistry

log = get_logger(__name__)


_SECTION_SYSTEM_PROMPT = """\
You are writing one section of a larger document. Write ONLY this section — \
do not include a document title, introduction to the full document, or conclusion \
for the full document.

Guidelines:
- Write substantive, detailed paragraphs (not bullet-point summaries)
- Include specific facts, figures, and data from the research provided
- Cite sources inline using [Source Name] or [URL domain] format
- Use clear topic sentences and logical paragraph transitions
- Target approximately {word_target} words
- Do NOT add meta-commentary ("In this section we will discuss...")
- Do NOT repeat information from other sections (see document outline)

Document outline (for context — write ONLY section {section_number}):
{document_outline}

Research relevant to this section:
{research_context}\
"""


class DraftSectionTool(Tool):
    """Generate one section of a document with focused LLM call.

    Designed to be called multiple times in parallel by the chain
    executor — once per document section.  Each call gets targeted
    research context so the model produces deeper, more specific
    content than a single-shot full-document generation.
    """

    def __init__(
        self,
        backend: ModelBackend,
        provider_registry: ProviderRegistry | None = None,
    ) -> None:
        self._default_backend = backend
        self._provider_registry = provider_registry

    @property
    def name(self) -> str:
        return "draft_section"

    @property
    def description(self) -> str:
        return (
            "Write one section of a document. Produces detailed, well-sourced "
            "content for a specific section based on research findings. "
            "Call once per section — multiple calls run in parallel for speed. "
            "Returns the section text with inline source citations."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.ARTIFACT

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "section_title": {
                    "type": "string",
                    "description": "The heading for this section",
                },
                "key_points": {
                    "type": "string",
                    "description": (
                        "Bullet list of key points this section must cover. "
                        "Include specific facts and data from research."
                    ),
                },
                "research_context": {
                    "type": "string",
                    "description": (
                        "Research findings relevant to THIS section (from prior "
                        "search/fetch steps). Include source URLs for citation."
                    ),
                },
                "document_outline": {
                    "type": "string",
                    "description": (
                        "Full document outline (all section titles) so this "
                        "section avoids repeating content from other sections."
                    ),
                },
                "section_number": {
                    "type": "integer",
                    "description": "This section's position (1-based) in the document",
                    "default": 1,
                },
                "total_sections": {
                    "type": "integer",
                    "description": "Total number of sections in the document",
                    "default": 1,
                },
                "word_target": {
                    "type": "integer",
                    "description": "Target word count for this section (default: 400)",
                    "default": 400,
                },
                "style": {
                    "type": "string",
                    "description": "Writing style (formal, conversational, technical)",
                    "default": "formal",
                },
            },
            "required": ["section_title", "key_points"],
        }

    @property
    def timeout(self) -> float:
        return 120.0

    async def execute(
        self,
        *,
        section_title: str = "",
        key_points: str = "",
        research_context: str = "",
        document_outline: str = "",
        section_number: int = 1,
        total_sections: int = 1,
        word_target: int = 400,
        style: str = "formal",
        task_id: str = "",
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        if not section_title:
            return ToolResult(success=False, error="No section_title provided")

        # Build the focused system prompt
        system = _SECTION_SYSTEM_PROMPT.format(
            word_target=word_target,
            section_number=section_number,
            document_outline=document_outline or "(not provided)",
            research_context=research_context[:8000] or "(no research provided — use your knowledge)",
        )

        if style != "formal":
            system += f"\n\nWriting style: {style}"

        # Build the user prompt
        user = f"## Section {section_number}: {section_title}\n\n"
        if key_points:
            user += f"Key points to cover:\n{key_points}\n\n"
        user += "Write this section now."

        from augmentum.models.base import InternalChatRequest, Message

        # Resolve the correct backend for the user's current model.
        # The tool is registered once at startup with a default backend,
        # but the user may be chatting with a different provider (e.g.
        # LM Studio instead of DeepSeek).  Use the provider registry
        # to resolve the right backend from the request context's model.
        _request_context = kwargs.pop("_request_context", None)
        model = ""
        backend = self._default_backend

        if _request_context and hasattr(_request_context, "model"):
            model = _request_context.model or ""
            if model and self._provider_registry:
                try:
                    resolved_be, resolved_model = (
                        await self._provider_registry.resolve_backend_with_fabric(model)
                    )
                    backend = resolved_be
                    model = resolved_model
                except Exception:
                    log.debug("draft_section_backend_resolve_failed", model=model)

        request = InternalChatRequest(
            model=model,
            messages=[
                Message(role="system", content=system),
                Message(role="user", content=user),
            ],
            stream=False,
        )

        try:
            response = await asyncio.wait_for(
                backend.chat(request),
                timeout=self.timeout,
            )
            content = response.message.content if response.message else ""

            if not content or len(content.strip()) < 50:
                return ToolResult(
                    success=False,
                    error=f"Section '{section_title}' produced insufficient content",
                )

            # Extract source citations from the content for metadata
            sources = _extract_citations(content)

            word_count = len(content.split())
            log.info(
                "section_drafted",
                section=section_title,
                words=word_count,
                sources=len(sources),
            )

            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "section_title": section_title,
                    "section_number": section_number,
                    "word_count": word_count,
                    "sources": sources,
                },
            )

        except TimeoutError:
            return ToolResult(
                success=False,
                error=f"Section '{section_title}' timed out after {self.timeout}s",
            )
        except Exception as e:
            log.error("section_draft_failed", section=section_title, error=str(e))
            return ToolResult(success=False, error=f"Draft failed: {e}")


def _extract_citations(text: str) -> list[str]:
    """Extract inline source citations from section text.

    Looks for patterns like [NASA], [Source Name], [example.com],
    [https://...] and returns unique citation strings.
    """
    import re

    citations: list[str] = []
    seen: set[str] = set()

    # [Source Name] or [URL]
    for m in re.finditer(r"\[([^\[\]]{3,80})\]", text):
        cite = m.group(1).strip()
        # Skip common markdown patterns that aren't citations
        if cite.startswith(("x", "X", " ")) or cite in ("source", "citation"):
            continue
        if cite.lower() not in seen:
            seen.add(cite.lower())
            citations.append(cite)

    return citations[:20]  # cap at 20 unique citations
