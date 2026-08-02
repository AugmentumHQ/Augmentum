"""Tool registry — discovers and manages available tools.

Includes fuzzy name resolution so that small models that can't perfectly
reproduce tool names (e.g. ``search`` instead of ``web_search``) still
trigger the correct tool.
"""

from __future__ import annotations

import re

from augmentum.tools.base import Tool, ToolCategory
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Mapping from UARF phase names to the tool categories available during that phase.
_PHASE_CATEGORIES: dict[str, list[ToolCategory]] = {
    "assess": [],
    "identify": [],
    # GATHER is merged IDENTIFY+RELEVANT for moderate — full tool access
    "gather": [
        ToolCategory.SEARCH,
        ToolCategory.FETCH,
        ToolCategory.EXECUTE,
        ToolCategory.VERIFY,
        ToolCategory.FILE,
        ToolCategory.IMAGE,
    ],
    "relevant": [ToolCategory.SEARCH, ToolCategory.FETCH],
    "apply": [
        ToolCategory.SEARCH,
        ToolCategory.FETCH,
        ToolCategory.EXECUTE,
        ToolCategory.VERIFY,
        ToolCategory.FILE,
        ToolCategory.ARTIFACT,
        ToolCategory.IMAGE,
    ],
    "verify": [ToolCategory.VERIFY, ToolCategory.EXECUTE],
    "conclude": [],
    # RESPOND is the merged simple path — same tool access as APPLY
    "respond": [
        ToolCategory.SEARCH,
        ToolCategory.FETCH,
        ToolCategory.EXECUTE,
        ToolCategory.VERIFY,
        ToolCategory.FILE,
        ToolCategory.ARTIFACT,
        ToolCategory.IMAGE,
    ],
}

# Common aliases / misspellings that small models produce.
# Maps normalised alias → canonical tool name.
_TOOL_ALIASES: dict[str, str] = {
    "search": "web_search",
    "websearch": "web_search",
    "web-search": "web_search",
    "searxng": "web_search",
    "google": "web_search",
    "fetch": "web_fetch",
    "webfetch": "web_fetch",
    "web-fetch": "web_fetch",
    "python": "python_exec",
    "pythonexec": "python_exec",
    "python-exec": "python_exec",
    "exec": "python_exec",
    "execute": "python_exec",
    "code": "python_exec",
    "code_run": "python_exec",
    "run_code": "python_exec",
    "run_python": "python_exec",
    "math": "math_verify",
    "mathverify": "math_verify",
    "math-verify": "math_verify",
    # NOTE: the bare "verify_math" alias is claimed further down by the flow
    # tool (-> "flow_verify_math"); that later entry wins. Kept single here to
    # preserve that long-standing resolution. Flip if the math tool should own it.
    "sympy": "math_verify",
    "calc": "calculator",
    "calculate": "calculator",
    "file": "file_ops",
    "fileops": "file_ops",
    "file-ops": "file_ops",
    "files": "file_ops",
    "read_file": "file_ops",
    "write_file": "file_ops",
    # Companion catalogue key (TOOL_CATALOG "files_read") — "search your
    # indexed files by name or keyword", which is search_files, not the
    # path read/write file_ops. Aliased so the native FC loop can actually
    # expose it instead of logging tool_resolve_failed.
    "files_read": "search_files",
    "search_file": "search_files",
    "find_files": "search_files",
    "date": "datetime",
    "time": "datetime",
    "date_time": "datetime",
    "datetime_tool": "datetime",
    "units": "unit_converter",
    "unitconverter": "unit_converter",
    "unit-converter": "unit_converter",
    "convert": "unit_converter",
    "convert_units": "unit_converter",
    "text": "text_analysis",
    "textanalysis": "text_analysis",
    "text-analysis": "text_analysis",
    "analyze_text": "text_analysis",
    "json": "json_tool",
    "jsontool": "json_tool",
    "json-tool": "json_tool",
    "parse_json": "json_tool",
    "hash_tool": "hash",
    "hashtool": "hash",
    "hash-tool": "hash",
    "memory": "memory_recall",
    "memory_recall": "memory_recall",
    "recall": "memory_recall",
    "remember": "memory_recall",
    "memories": "memory_recall",
    "image": "image_generation",
    "image_gen": "image_generation",
    "imagegeneration": "image_generation",
    "image-generation": "image_generation",
    "generate_image": "image_generation",
    "draw": "image_generation",
    "paint": "image_generation",
    "picture": "image_generation",
    "img": "image_generation",
    "img_gen": "image_generation",
    "dalle": "image_generation",
    "stable_diffusion": "image_generation",
    "sd": "image_generation",
    # Artifact tools
    "document": "create_document",
    "create_doc": "create_document",
    "doc": "create_document",
    "pdf": "create_document",
    "docx": "create_document",
    "write_document": "create_document",
    "presentation": "create_presentation",
    "pptx": "create_presentation",
    "powerpoint": "create_presentation",
    "slides": "create_presentation",
    "create_slides": "create_presentation",
    "spreadsheet": "create_spreadsheet",
    "xlsx": "create_spreadsheet",
    "excel": "create_spreadsheet",
    "create_excel": "create_spreadsheet",
    "table": "create_spreadsheet",
    "ebook": "create_ebook",
    "epub": "create_ebook",
    "storybook": "create_ebook",
    "create_book": "create_ebook",
    # Unified web tool
    "web_lookup": "web",
    "lookup_url": "web",
    "browse": "web",
    "internet": "web",
    "web_tool": "web",
    # Wikipedia tool
    "wiki": "wikipedia",
    "wiki_search": "wikipedia",
    "wikipedia_search": "wikipedia",
    "wiki_lookup": "wikipedia",
    "lookup": "wikipedia",
    "encyclopedia": "wikipedia",
    # YouTube transcript tool
    "youtube_transcript": "youtube",
    "yt": "youtube",
    "yt_transcript": "youtube",
    "video_transcript": "youtube",
    "captions": "youtube",
    "subtitles": "youtube",
    "transcript": "youtube",
    # Document parsing tool
    "parse": "document_parse",
    "parse_document": "document_parse",
    "parse_doc": "document_parse",
    "read_document": "document_parse",
    "read_doc": "document_parse",
    "read_pdf": "document_parse",
    "extract_text": "document_parse",
    "parse_pdf": "document_parse",
    "parse_docx": "document_parse",
    "parse_xlsx": "document_parse",
    "parse_pptx": "document_parse",
    # Flow tool aliases (models may omit the flow_ prefix)
    "deep_research": "flow_deep_research",
    "fact_check": "flow_fact_check",
    "video_summary": "flow_video_summary",
    "analyze_document": "flow_analyze_document",
    "verify_math": "flow_verify_math",
    "research_illustrate": "flow_research___illustrate",
    "data_pipeline": "flow_data_pipeline",
    "compare_sources": "flow_compare_sources",
    # Chart tool
    "chart": "create_chart",
    "create_graph": "create_chart",
    "graph": "create_chart",
    "plot": "create_chart",
    "visualization": "create_chart",
    "visualize": "create_chart",
    "bar_chart": "create_chart",
    "line_chart": "create_chart",
    "pie_chart": "create_chart",
}

# Regex to strip markdown / punctuation that small models wrap tool names in.
# NOTE: underscores are preserved (they are meaningful in tool names like web_search).
_CLEAN_NAME_RE = re.compile(r"[`*\"\'\[\](){}]")


def _normalise(name: str) -> str:
    """Normalise a tool name for fuzzy matching.

    Dots and dashes both fold to underscore. OpenAI-compatible function
    names forbid dots, so a dotted registry id (``my.taste``,
    ``media.play``) is sent to the model as-is but comes BACK sanitized
    to ``my_taste`` / ``media_play`` (llama-server / the model rewrites
    it). Without folding ``.`` here, ``resolve("my_taste")`` never matches
    the registered ``my.taste`` and the model's correctly-chosen verb is
    rejected as "Unknown tool" (live: passthrough_tool_unresolved, every
    voice turn 2026-06-18). Mirrors ``_tool_name_collision_key``'s fold so
    resolution and dedup agree on identity.
    """
    return (
        _CLEAN_NAME_RE.sub("", name).strip().lower()
        .replace("-", "_").replace(".", "_").replace(" ", "_")
    )


class ToolMetrics:
    """Lightweight per-tool call tracking — no LLM cost, just counters."""

    def __init__(self) -> None:
        self._calls: dict[str, int] = {}
        self._successes: dict[str, int] = {}
        self._failures: dict[str, int] = {}
        self._cache_hits: dict[str, int] = {}
        self._total_ms: dict[str, float] = {}

    def record(self, tool_name: str, *, success: bool, elapsed_ms: float, cached: bool = False) -> None:
        self._calls[tool_name] = self._calls.get(tool_name, 0) + 1
        if cached:
            self._cache_hits[tool_name] = self._cache_hits.get(tool_name, 0) + 1
        elif success:
            self._successes[tool_name] = self._successes.get(tool_name, 0) + 1
        else:
            self._failures[tool_name] = self._failures.get(tool_name, 0) + 1
        self._total_ms[tool_name] = self._total_ms.get(tool_name, 0.0) + elapsed_ms

    def snapshot(self) -> dict[str, dict]:
        """Return metrics for all tracked tools."""
        all_tools = set(self._calls)
        result = {}
        for name in sorted(all_tools):
            calls = self._calls.get(name, 0)
            result[name] = {
                "calls": calls,
                "successes": self._successes.get(name, 0),
                "failures": self._failures.get(name, 0),
                "cache_hits": self._cache_hits.get(name, 0),
                "avg_ms": round(self._total_ms.get(name, 0) / max(calls, 1), 1),
            }
        return result


class ToolRegistry:
    """Discovers and manages available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self.metrics = ToolMetrics()

    def register(self, tool: Tool) -> None:
        """Register a tool by its name."""
        if tool.name in self._tools:
            log.warning("tool_already_registered", name=tool.name)
        self._tools[tool.name] = tool
        log.info("tool_registered", name=tool.name, category=tool.category.value)

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry. Returns True if it existed."""
        if name in self._tools:
            del self._tools[name]
            log.info("tool_unregistered", name=name)
            return True
        return False

    def get(self, name: str) -> Tool | None:
        """Look up a tool by exact name."""
        return self._tools.get(name)

    def resolve(self, raw_name: str) -> Tool | None:
        """Fuzzy-resolve a tool name to a registered tool.

        Resolution order:
        1. Exact match
        2. Normalised exact match (lowercase, strip markdown/punctuation)
        3. Alias table lookup
        4. Substring / suffix match (e.g. ``search`` matches ``web_search``)
        """
        # 1. Exact
        tool = self._tools.get(raw_name)
        if tool:
            return tool

        normalised = _normalise(raw_name)

        # 2. Normalised exact match against registered names
        for registered_name, t in self._tools.items():
            if _normalise(registered_name) == normalised:
                log.debug("tool_resolved_normalised", raw=raw_name, resolved=registered_name)
                return t

        # 3. Alias table
        canonical = _TOOL_ALIASES.get(normalised)
        if canonical and canonical in self._tools:
            log.debug("tool_resolved_alias", raw=raw_name, alias=normalised, resolved=canonical)
            return self._tools[canonical]

        # 4. Substring / suffix — e.g. "search" matches "web_search"
        candidates: list[tuple[str, Tool]] = []
        for registered_name, t in self._tools.items():
            reg_norm = _normalise(registered_name)
            if normalised in reg_norm or reg_norm.endswith(normalised):
                candidates.append((registered_name, t))

        if len(candidates) == 1:
            log.debug(
                "tool_resolved_substring",
                raw=raw_name,
                resolved=candidates[0][0],
            )
            return candidates[0][1]

        if len(candidates) > 1:
            log.warning(
                "tool_resolve_ambiguous",
                raw=raw_name,
                candidates=[c[0] for c in candidates],
            )
            # Return first match (prefer shortest name = closest match)
            candidates.sort(key=lambda c: len(c[0]))
            return candidates[0][1]

        log.warning("tool_resolve_failed", raw=raw_name, normalised=normalised)
        return None

    def list_tools(self, category: ToolCategory | None = None) -> list[Tool]:
        """Return all registered tools, optionally filtered by category."""
        if category is None:
            return list(self._tools.values())
        return [t for t in self._tools.values() if t.category == category]

    def get_for_surface(
        self,
        surface: str,
        *,
        voice_level: str | None = None,
    ) -> list[Tool]:
        """Return tools exposed on a given surface.

        Args:
            surface: One of ``"chat"``, ``"voice"``, ``"coder"``,
                ``"companion"``, ``"artifact_studio"``, or ``"http"``.
                For ``"http"``, returns tools that opted into the
                auto-route registration (``http_route`` is set).
            voice_level: When ``surface == "voice"`` and this is set,
                filter to tools whose voice tier matches exactly
                (``"core" | "interactive" | "disruptive" | "costly"``).
                ``None`` returns all voice-exposed tools regardless
                of tier.

        Surface gating is orthogonal to :meth:`get_for_phase` — chat
        handlers should intersect ``get_for_surface("chat")`` with
        ``get_for_phase(phase)`` and the user's textbox allowlist.
        """
        out: list[Tool] = []
        for t in self._tools.values():
            s = t.surfaces
            if surface == "chat" and s.chat:
                out.append(t)
            elif surface == "voice" and s.voice is not None:
                if voice_level is None or s.voice == voice_level:
                    out.append(t)
            elif surface == "coder" and s.coder or surface == "companion" and s.companion or surface == "artifact_studio" and s.artifact_studio or surface == "http" and s.http_route:
                out.append(t)
        return out

    def get_for_phase(
        self,
        phase: str,
        *,
        exclude: frozenset[str] | None = None,
        allowed_names: frozenset[str] | None = None,
    ) -> list[Tool]:
        """Return tools available for a given UARF phase.

        Phase-to-category mapping (the *capability boundary* — what the phase
        is structurally permitted to touch):
        - RELEVANT: search, fetch
        - APPLY/RESPOND: search, fetch, execute, verify, file, artifact, image
        - VERIFY: verify, execute
        - Other phases: no tools

        Args:
            phase: The UARF phase name.
            exclude: Optional set of tool names to exclude (e.g. when
                auto-search has already handled ``web_search``).
            allowed_names: Optional set of tool names the *user* selected
                via the textbox tool selector (``X-Augmentum-Tools`` header).
                When provided, the result is the intersection of the phase's
                category capability and this user selection — the selector
                becomes the source of truth, the phase map only restricts.
                Pass ``None`` to use the full category set; pass an empty
                frozenset to return no tools (user explicitly chose "none").
        """
        categories = _PHASE_CATEGORIES.get(phase.lower(), [])
        if not categories:
            return []
        tools = [t for t in self._tools.values() if t.category in categories]
        if exclude:
            tools = [t for t in tools if t.name not in exclude]
        if allowed_names is not None:
            tools = [t for t in tools if t.name in allowed_names]
        return tools
