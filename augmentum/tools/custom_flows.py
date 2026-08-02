"""Custom flow store — CRUD, trigger matching, and template resolution for
user-defined tool chain flows.
"""

from __future__ import annotations

import json
import re
import uuid

import aiosqlite

from augmentum.config import settings
from augmentum.tools.chain import ChainPlan, ChainStep
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default flows — seeded on first run when the store is empty
# ---------------------------------------------------------------------------

_DEFAULT_FLOWS: list[dict] = [
    {
        "name": "Deep Research",
        "description": "Search the web, fetch the top result, then fetch a second source for cross-reference",
        "trigger_pattern": "",
        "steps": [
            {"id": 1, "tool": "web_search", "input": {"query": "{{query}}"}, "needs": [], "reason": "Search the web for relevant results"},
            {"id": 2, "tool": "wikipedia", "input": {"query": "{{query}}"}, "needs": [], "reason": "Get Wikipedia overview for background context"},
            {"id": 3, "tool": "web_fetch", "input": {"url": "{{step.1.metadata.urls.0}}"}, "needs": [1], "reason": "Fetch the top search result for detailed content"},
        ],
    },
    {
        "name": "Fact Check",
        "description": "Cross-reference a claim across web, Wikipedia, and math verification",
        "trigger_pattern": r"(?:fact.?check|is\s+it\s+true|verify.*claim)",
        "steps": [
            {"id": 1, "tool": "web_search", "input": {"query": "{{query}} fact check"}, "needs": [], "reason": "Search for fact-checking sources"},
            {"id": 2, "tool": "wikipedia", "input": {"query": "{{query}}"}, "needs": [], "reason": "Check Wikipedia for established facts"},
            {"id": 3, "tool": "web_fetch", "input": {"url": "{{step.1.metadata.urls.0}}"}, "needs": [1], "reason": "Fetch the top fact-check source"},
        ],
    },
    {
        "name": "Video Summary",
        "description": "Fetch a YouTube transcript and get readability stats on the content",
        "trigger_pattern": r"(?:summarize|summary\s+of|recap).*(?:youtube|video|watch)",
        "steps": [
            {"id": 1, "tool": "youtube_transcript", "input": {"video": "{{query}}"}, "needs": [], "reason": "Fetch the video transcript"},
            {"id": 2, "tool": "text_analysis", "input": {"text": "{{step.1.output}}"}, "needs": [1], "reason": "Analyze transcript length, complexity, and reading time"},
        ],
    },
    {
        "name": "Analyze Document",
        "description": "Parse a PDF, DOCX, or other document and analyze its content",
        "trigger_pattern": r"(?:analyze|parse|read).*(?:document|pdf|docx|file)",
        "steps": [
            {"id": 1, "tool": "document_parse", "input": {"path": "{{query}}"}, "needs": [], "reason": "Parse the document into text"},
            {"id": 2, "tool": "text_analysis", "input": {"text": "{{step.1.output}}"}, "needs": [1], "reason": "Analyze word count, readability, and structure"},
        ],
    },
    {
        "name": "Verify Math",
        "description": "Compute an expression numerically, then cross-check with symbolic math",
        "trigger_pattern": r"(?:verify|check|prove).*(?:math|calculation|formula|equation)",
        "steps": [
            {"id": 1, "tool": "calculator", "input": {"expression": "{{query}}"}, "needs": [], "reason": "Compute the numeric result"},
            {"id": 2, "tool": "math_verify", "input": {"expression": "{{query}}", "expected": "{{step.1.output}}"}, "needs": [1], "reason": "Cross-check with SymPy symbolic engine"},
        ],
    },
    {
        "name": "Research & Illustrate",
        "description": "Research a topic, then generate an image based on findings",
        "trigger_pattern": r"(?:illustrate|visualize|picture\s+of|draw|depict)",
        "steps": [
            {"id": 1, "tool": "web_search", "input": {"query": "{{query}}"}, "needs": [], "reason": "Research the topic to inform the image"},
            {"id": 2, "tool": "image_generation", "needs": [1], "reason": "Generate an image based on the research findings"},
        ],
    },
    {
        "name": "Data Pipeline",
        "description": "Fetch data from a URL and process it with Python",
        "trigger_pattern": "",
        "steps": [
            {"id": 1, "tool": "web_fetch", "input": {"url": "{{query}}"}, "needs": [], "reason": "Fetch raw data from the URL"},
            {"id": 2, "tool": "python_exec", "needs": [1], "reason": "Write Python code to parse and analyze the fetched data"},
        ],
    },
    {
        "name": "Compare Sources",
        "description": "Search web and Wikipedia in parallel, then fetch the best source for depth",
        "trigger_pattern": "",
        "steps": [
            {"id": 1, "tool": "web_search", "input": {"query": "{{query}}"}, "needs": [], "reason": "Search the web for current information"},
            {"id": 2, "tool": "wikipedia", "input": {"query": "{{query}}"}, "needs": [], "reason": "Get the Wikipedia perspective"},
            {"id": 3, "tool": "web_fetch", "input": {"url": "{{step.1.metadata.urls.0}}"}, "needs": [1], "reason": "Fetch full article from top web result for detailed comparison"},
        ],
    },
]

# ReDoS protection — reject patterns with nested quantifiers
# Only catches truly dangerous patterns like (.+)+, (a*)*, not safe ones like (this\s+)?
_NESTED_QUANTIFIER_RE = re.compile(
    r"\([^)]*[+*][^)]*\)[+*]"  # e.g. (.+)+, (a*)* — but NOT (x+)?
    r"|"
    r"\([^)]*\)\{[0-9,]+\}[+*]"  # e.g. (a){2,}+
)
_MAX_TRIGGER_PATTERN_LEN = 200


def _validate_regex_safe(pattern: str) -> None:
    """Validate that a regex pattern is safe from ReDoS.

    Raises ValueError if the pattern is too long or contains nested quantifiers.
    """
    if len(pattern) > _MAX_TRIGGER_PATTERN_LEN:
        raise ValueError(
            f"Trigger pattern too long ({len(pattern)} chars, max {_MAX_TRIGGER_PATTERN_LEN})"
        )
    if _NESTED_QUANTIFIER_RE.search(pattern):
        raise ValueError(
            "Trigger pattern contains nested quantifiers (ReDoS risk)"
        )


def validate_flow_tools(
    steps: list[dict],
    tool_registry: object | None,
) -> list[str]:
    """Return warnings for steps referencing unregistered tools."""
    if not tool_registry:
        return []
    warnings: list[str] = []
    for step in steps:
        tool_name = step.get("tool", "")
        if tool_name and not tool_registry.resolve(tool_name):  # type: ignore[union-attr]
            warnings.append(
                f"Step {step.get('id', '?')}: tool '{tool_name}' is not currently registered"
            )
    return warnings


def flow_to_plan(flow: dict) -> ChainPlan:
    """Convert a stored flow dict to a ChainPlan."""
    steps_raw = json.loads(flow["steps_json"]) if isinstance(flow["steps_json"], str) else flow["steps_json"]
    steps = []
    for s in steps_raw:
        steps.append(ChainStep(
            id=s["id"],
            tool=s["tool"],
            input=s.get("input"),
            needs=s.get("needs", []),
            reason=s.get("reason", ""),
        ))
    return ChainPlan(steps=steps, source=f"custom:{flow['id']}")


def match_trigger(query: str, flows: list[dict]) -> dict | None:
    """Find the first enabled flow whose trigger_pattern matches the query.

    Returns the flow dict or None.
    """
    for flow in flows:
        if not flow.get("enabled", True):
            continue
        pattern = flow.get("trigger_pattern", "")
        if not pattern:
            continue
        try:
            if re.search(pattern, query, re.IGNORECASE):
                return flow
        except re.error:
            log.warning("flow_trigger_regex_error", flow=flow["id"], pattern=pattern)
    return None


class CustomFlowStore:
    """SQLite-backed CRUD for custom tool chain flows."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def list_flows(self, *, enabled_only: bool = False, user_id: str = "") -> list[dict]:
        """List all flows, optionally only enabled ones."""
        conditions: list[str] = []
        params: list = []
        if enabled_only:
            conditions.append("enabled = 1")
        if user_id:
            conditions.append("(user_id = ? OR user_id IS NULL)")
            params.append(user_id)
        sql = "SELECT * FROM custom_flows"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY name"
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row, strict=False)) for row in rows]

    async def get_flow(self, flow_id: str, *, user_id: str = "") -> dict | None:
        """Get a single flow by ID."""
        query = "SELECT * FROM custom_flows WHERE id = ?"
        params: list = [flow_id]
        if user_id:
            query += " AND (user_id = ? OR user_id IS NULL)"
            params.append(user_id)
        cursor = await self._db.execute(query, params)
        row = await cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        return dict(zip(cols, row, strict=False))

    async def create_flow(
        self,
        name: str,
        steps: list[dict],
        *,
        description: str = "",
        trigger_pattern: str = "",
        user_id: str = "",
    ) -> dict:
        """Create a new custom flow. Returns the created flow dict."""
        if not user_id:
            raise ValueError("custom_flows insert requires user_id")
        # Enforce max flows limit
        cursor = await self._db.execute("SELECT COUNT(*) FROM custom_flows")
        (count,) = await cursor.fetchone()
        if count >= settings.passthrough_chain_max_flows:
            raise ValueError(
                f"Maximum number of flows reached ({settings.passthrough_chain_max_flows})"
            )

        flow_id = uuid.uuid4().hex[:12]
        steps_json = json.dumps(steps)

        # Validate trigger pattern if provided
        if trigger_pattern:
            _validate_regex_safe(trigger_pattern)
            try:
                re.compile(trigger_pattern)
            except re.error as exc:
                raise ValueError(f"Invalid trigger pattern: {exc}") from exc

        await self._db.execute(
            "INSERT INTO custom_flows "
            "(id, name, description, trigger_pattern, steps_json, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [flow_id, name, description, trigger_pattern, steps_json, user_id],
        )
        await self._db.commit()
        log.info("flow_created", id=flow_id, name=name, steps=len(steps))
        return await self.get_flow(flow_id, user_id=user_id)  # type: ignore[return-value]

    async def update_flow(self, flow_id: str, *, user_id: str = "", **fields: object) -> dict | None:
        """Update a flow. Supported fields: name, description, trigger_pattern, steps, enabled."""
        if not user_id:
            raise ValueError("custom_flows update requires user_id")
        sets = []
        params: list[object] = []

        if "name" in fields:
            sets.append("name = ?")
            params.append(fields["name"])
        if "description" in fields:
            sets.append("description = ?")
            params.append(fields["description"])
        if "trigger_pattern" in fields:
            pattern = fields["trigger_pattern"]
            if pattern:
                try:
                    re.compile(str(pattern))
                except re.error as exc:
                    raise ValueError(f"Invalid trigger pattern: {exc}") from exc
            sets.append("trigger_pattern = ?")
            params.append(pattern)
        if "steps" in fields:
            sets.append("steps_json = ?")
            params.append(json.dumps(fields["steps"]))
        if "enabled" in fields:
            sets.append("enabled = ?")
            params.append(1 if fields["enabled"] else 0)

        if not sets:
            return await self.get_flow(flow_id, user_id=user_id)

        sets.append("updated_at = datetime('now')")
        params.extend([flow_id, user_id])

        await self._db.execute(
            f"UPDATE custom_flows SET {', '.join(sets)} "
            "WHERE id = ? AND (user_id = ? OR user_id IS NULL)",
            tuple(params),
        )
        await self._db.commit()
        log.info("flow_updated", id=flow_id, fields=list(fields.keys()))
        return await self.get_flow(flow_id, user_id=user_id)

    async def delete_flow(self, flow_id: str, *, user_id: str = "") -> bool:
        """Delete a flow. Returns True if it existed."""
        if not user_id:
            raise ValueError("custom_flows delete requires user_id")
        cursor = await self._db.execute(
            "DELETE FROM custom_flows "
            "WHERE id = ? AND (user_id = ? OR user_id IS NULL)",
            (flow_id, user_id),
        )
        await self._db.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            log.info("flow_deleted", id=flow_id)
        return deleted

    async def match_query(self, query: str, *, user_id: str = "") -> dict | None:
        """Find the first enabled flow whose trigger matches the query."""
        flows = await self.list_flows(enabled_only=True, user_id=user_id)
        return match_trigger(query, flows)

    async def fuzzy_find(self, name_query: str, *, user_id: str = "") -> dict | None:
        """Find a flow by fuzzy name match (for /flow command)."""
        flows = await self.list_flows(user_id=user_id)
        query_lower = name_query.lower().strip()
        # Exact match first
        for f in flows:
            if f["name"].lower() == query_lower:
                return f
        # Prefix match
        for f in flows:
            if f["name"].lower().startswith(query_lower):
                return f
        # Substring match
        for f in flows:
            if query_lower in f["name"].lower():
                return f
        return None

    async def export_all(self, *, user_id: str = "") -> list[dict]:
        """Export all flows as a list of dicts."""
        flows = await self.list_flows(user_id=user_id)
        for f in flows:
            f["steps_json"] = json.loads(f["steps_json"]) if isinstance(f["steps_json"], str) else f["steps_json"]
        return flows

    async def import_flows(self, flows_data: list[dict], *, user_id: str = "") -> int:
        """Import flows from a list of dicts. Returns count imported."""
        count = 0
        for f in flows_data:
            steps = f.get("steps_json") or f.get("steps", [])
            if isinstance(steps, str):
                steps = json.loads(steps)
            await self.create_flow(
                name=f["name"],
                steps=steps,
                description=f.get("description", ""),
                trigger_pattern=f.get("trigger_pattern", ""),
                user_id=user_id,
            )
            count += 1
        return count

    async def seed_defaults(self, *, user_id: str = "") -> int:
        """Seed default flows for ``user_id``, updating when definitions change.

        Per-user (Tier 0 multi-tenant rollout) — call once per user when
        they first open the flows panel. Returns total count of created +
        updated flows.
        """
        if not user_id:
            raise ValueError("seed_defaults requires user_id")
        existing = await self.list_flows(user_id=user_id)
        existing_by_name = {f["name"]: f for f in existing}
        count = 0
        for f in _DEFAULT_FLOWS:
            try:
                old = existing_by_name.get(f["name"])
                if old:
                    # Update if steps or description changed
                    old_steps_raw = old.get("steps_json") or old.get("steps") or "[]"
                    if isinstance(old_steps_raw, str):
                        old_steps_parsed = json.loads(old_steps_raw)
                    else:
                        old_steps_parsed = old_steps_raw
                    old_steps = json.dumps(old_steps_parsed, sort_keys=True)
                    new_steps = json.dumps(f["steps"], sort_keys=True)
                    if old_steps != new_steps or old.get("description", "") != f.get("description", ""):
                        await self.update_flow(
                            old["id"],
                            user_id=user_id,
                            steps=f["steps"],
                            description=f.get("description", ""),
                            trigger_pattern=f.get("trigger_pattern", ""),
                        )
                        log.info("default_flow_updated", name=f["name"])
                        count += 1
                else:
                    await self.create_flow(
                        name=f["name"],
                        steps=f["steps"],
                        description=f.get("description", ""),
                        trigger_pattern=f.get("trigger_pattern", ""),
                        user_id=user_id,
                    )
                    count += 1
            except Exception:
                log.warning("seed_default_flow_failed", name=f["name"], exc_info=True)
        if count:
            log.info("default_flows_seeded", count=count, user_id=user_id)
        return count


# ---------------------------------------------------------------------------
# AI flow generator — describe a workflow in natural language, get a flow
# ---------------------------------------------------------------------------

_GENERATE_SYSTEM = """\
You are a tool-chain flow designer. The user will describe a workflow in plain \
language. You must produce a JSON flow definition that chains the available tools.

Available tools (name — description — required params):
{tool_list}

Template variables you can use in step inputs:
- {{{{query}}}} — the user's input when running the flow
- {{{{step.N.output}}}} — output text from step N
- {{{{step.N.metadata.KEY}}}} — metadata field from step N (e.g. urls.0)

Rules:
- Step IDs start at 1, sequential.
- "needs" lists step IDs that must complete first. Steps with no shared \
dependencies run in parallel.
- If a step's best arguments depend on prior step results in a creative way \
(e.g. writing Python code based on fetched data, or crafting an image prompt \
from research), OMIT the "input" field entirely — the AI will determine \
arguments at runtime.
- Only use "input" when the arguments are clearly deterministic from templates.
- trigger_pattern is an optional regex for auto-triggering. Keep it simple. \
Set to "" if unsure.
- Respond with ONLY a JSON object, no explanation. Schema:

{{
  "name": "string",
  "description": "string",
  "trigger_pattern": "string or empty",
  "steps": [
    {{"id": 1, "tool": "tool_name", "input": {{"param": "value"}} or null, "needs": [], "reason": "why"}}
  ]
}}"""


async def generate_flow_via_llm(
    description: str,
    backend: object,
    tool_registry: object,
    *,
    model: str = "",
) -> dict:
    """Generate a flow definition from a natural language description.

    Uses the configured LLM backend to translate user intent into a
    structured flow. Returns a dict suitable for ``CustomFlowStore.create_flow``.

    Raises ValueError if the LLM response can't be parsed.
    """
    from augmentum.models.base import InternalChatRequest, Message

    # Build tool catalog for the prompt
    tool_lines = []
    for tool in tool_registry.list_tools():  # type: ignore[union-attr]
        schema = tool.input_schema or {}
        props = schema.get("properties", {})
        required = schema.get("required", [])
        params = []
        for k, v in props.items():
            req = " (required)" if k in required else ""
            params.append(f"{k}: {v.get('type', 'string')}{req}")
        tool_lines.append(f"- {tool.name} — {tool.description} — {', '.join(params)}")

    from augmentum.utils.datetime_context import get_datetime_context

    system = (
        f"{get_datetime_context()}\n\n"
        + _GENERATE_SYSTEM.format(tool_list="\n".join(tool_lines))
    )

    request = InternalChatRequest(
        model=model,
        messages=[
            Message(role="system", content=system),
            Message(role="user", content=description),
        ],
        stream=False,
    )

    response = await backend.chat(request)  # type: ignore[union-attr]
    text = (response.message.content if response.message else "").strip()

    # --- Robust JSON extraction (handles preamble, fences, trailing text) ---
    # Strip markdown fences first
    cleaned = text
    if cleaned.startswith("```"):
        first_nl = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
        cleaned = cleaned[first_nl + 1:]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
        cleaned = cleaned.strip()

    # Find the JSON object by bracket scanning
    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}")
    if json_start == -1 or json_end == -1 or json_end <= json_start:
        raise ValueError(
            "LLM response contains no JSON object. Raw response starts with: "
            + repr(text[:200])
        )

    try:
        flow = json.loads(cleaned[json_start : json_end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse LLM JSON: {exc}") from exc

    # --- Structure validation ---
    if not isinstance(flow, dict):
        raise ValueError("LLM returned non-object JSON")
    if not flow.get("name"):
        raise ValueError("Generated flow has no name")
    steps = flow.get("steps", [])
    if not steps:
        raise ValueError("Generated flow has no steps")

    step_ids = set()
    warnings: list[str] = []

    for i, s in enumerate(steps):
        if "id" not in s or "tool" not in s:
            raise ValueError(f"Step {i + 1} missing 'id' or 'tool'")
        step_ids.add(s["id"])

    # --- Tool name validation ---
    registered = {t.name for t in tool_registry.list_tools()}  # type: ignore[union-attr]
    for s in steps:
        if s["tool"] not in registered:
            warnings.append(f"Step {s['id']}: tool '{s['tool']}' is not registered")

    # --- DAG validation (cycle detection via topological sort) ---
    for s in steps:
        for dep_id in s.get("needs", []):
            if dep_id not in step_ids:
                raise ValueError(
                    f"Step {s['id']} depends on non-existent step {dep_id}"
                )

    # Simple cycle check: topological sort
    remaining = {s["id"]: set(s.get("needs", [])) for s in steps}
    sorted_ids: list[int] = []
    while remaining:
        ready = [sid for sid, deps in remaining.items() if not deps]
        if not ready:
            cycle_ids = ", ".join(str(s) for s in remaining)
            raise ValueError(f"Circular dependency among steps: {cycle_ids}")
        for sid in ready:
            sorted_ids.append(sid)
            del remaining[sid]
        for deps in remaining.values():
            deps -= set(ready)

    log.info("flow_generated_via_llm", name=flow["name"], steps=len(steps),
             warnings=len(warnings))
    flow["_warnings"] = warnings
    return flow
