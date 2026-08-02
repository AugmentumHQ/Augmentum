"""SQLite-backed storage for reasoning flows and steps."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

from augmentum.reasoning.models import FlowStep, ReasoningFlow
from augmentum.reasoning.templates import BUILTIN_TEMPLATES
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


def _id() -> str:
    return uuid.uuid4().hex[:16]


class FlowStore:
    """CRUD operations for reasoning flows backed by SQLite."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Seed built-in templates on first run
    # ------------------------------------------------------------------

    async def seed_builtins(self) -> int:
        """Insert built-in flow templates if they don't already exist.

        Also ensures any new built-in templates are added on upgrade,
        and updates existing builtins when their template changes.
        Returns the number of templates seeded or updated.
        """
        # Find which builtins already exist by name → id
        cursor = await self._db.execute(
            "SELECT name, id FROM reasoning_flows WHERE is_builtin = 1"
        )
        existing = {row[0]: row[1] for row in await cursor.fetchall()}

        count = 0
        for name, factory in BUILTIN_TEMPLATES.items():
            flow = factory()
            if flow.name in existing:
                # Update existing builtin — replace steps
                old_id = existing[flow.name]
                flow.id = old_id
                flow.is_builtin = True
                # Delete old steps and re-insert
                await self._db.execute(
                    "DELETE FROM reasoning_flow_steps WHERE flow_id = ?",
                    (old_id,),
                )
                # Update the flow record
                await self._db.execute(
                    "UPDATE reasoning_flows SET description = ?, icon = ?, "
                    "auto_select = ?, trigger_domains = ?, trigger_keywords = ?, "
                    "auto_search = ?, autonomy_level = ?, max_tool_calls_per_step = ? "
                    "WHERE id = ?",
                    (
                        flow.description, flow.icon,
                        int(flow.auto_select),
                        json.dumps(flow.trigger_domains or []),
                        json.dumps(flow.trigger_keywords or []),
                        int(flow.auto_search),
                        flow.autonomy_level or 0,
                        flow.max_tool_calls_per_step or 0,
                        old_id,
                    ),
                )
                # Re-insert steps
                for i, step in enumerate(flow.steps):
                    step.id = _id()
                    step.flow_id = old_id
                    step.sort_order = i
                    await self._insert_step(step)
                count += 1
                continue
            flow.id = _id()
            flow.is_builtin = True
            for i, step in enumerate(flow.steps):
                step.id = _id()
                step.flow_id = flow.id
                step.sort_order = i
            await self._insert_flow(flow)
            count += 1

        if count:
            # Ensure exactly one default exists; prefer Auto Routing
            dc = await self._db.execute(
                "SELECT COUNT(*) FROM reasoning_flows WHERE is_default = 1"
            )
            default_count = (await dc.fetchone())[0]
            if default_count == 0:
                ar = await self._db.execute(
                    "SELECT id FROM reasoning_flows WHERE name = 'Auto Routing' AND is_builtin = 1"
                )
                ar_row = await ar.fetchone()
                if ar_row:
                    await self._db.execute(
                        "UPDATE reasoning_flows SET is_default = 1 WHERE id = ?",
                        (ar_row[0],),
                    )

            await self._db.commit()
            log.info("reasoning_flows_seeded", count=count)
        return count

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def list_flows(self, *, user_id: str = "") -> list[tuple[ReasoningFlow, int]]:
        """List all flows (without step details).

        Step counts come back in the same query via a correlated subquery
        so this is O(1) round-trips instead of N+1 (one SELECT COUNT per
        flow, which was the prior shape).
        """
        step_count_sql = (
            "(SELECT COUNT(*) FROM reasoning_flow_steps s WHERE s.flow_id = f.id) "
            "AS _step_count"
        )
        if user_id:
            cursor = await self._db.execute(
                f"SELECT f.*, {step_count_sql} FROM reasoning_flows f "
                "WHERE f.user_id = ? OR f.user_id IS NULL "
                "ORDER BY f.is_default DESC, f.name ASC",
                (user_id,),
            )
        else:
            cursor = await self._db.execute(
                f"SELECT f.*, {step_count_sql} FROM reasoning_flows f "
                "ORDER BY f.is_default DESC, f.name ASC"
            )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        flows: list[tuple[ReasoningFlow, int]] = []
        for row in rows:
            d = dict(zip(cols, row))
            step_count = int(d.pop("_step_count", 0) or 0)
            flow = self._row_to_flow(d)
            flow.steps = []  # no details in list view
            flows.append((flow, step_count))
        return flows

    async def get_flow(self, flow_id: str, *, user_id: str = "") -> ReasoningFlow | None:
        """Get a flow with all its steps."""
        query = "SELECT * FROM reasoning_flows WHERE id = ?"
        params: list = [flow_id]
        if user_id:
            query += " AND (user_id = ? OR user_id IS NULL)"
            params.append(user_id)
        cursor = await self._db.execute(query, params)
        row = await cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        flow = self._row_to_flow(dict(zip(cols, row)))
        flow.steps = await self._get_steps(flow_id)
        return flow

    async def get_default_flow(self, *, user_id: str = "") -> ReasoningFlow | None:
        """Get the flow marked as default."""
        if user_id:
            cursor = await self._db.execute(
                "SELECT * FROM reasoning_flows WHERE is_default = 1 AND (user_id = ? OR user_id IS NULL) LIMIT 1",
                (user_id,),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM reasoning_flows WHERE is_default = 1 LIMIT 1"
            )
        row = await cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        flow = self._row_to_flow(dict(zip(cols, row)))
        flow.steps = await self._get_steps(flow.id)
        return flow

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create_flow(self, flow: ReasoningFlow, *, user_id: str = "") -> ReasoningFlow:
        """Insert a new flow with steps."""
        if not flow.id:
            flow.id = _id()
        for i, step in enumerate(flow.steps):
            if not step.id:
                step.id = _id()
            step.flow_id = flow.id
            step.sort_order = i

        await self._insert_flow(flow, user_id=user_id)
        await self._db.commit()
        return flow

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    async def update_flow(self, flow_id: str, updates: dict, *, user_id: str = "") -> ReasoningFlow | None:
        """Update flow fields and optionally replace all steps."""
        if not user_id:
            raise ValueError("reasoning_flows update requires user_id")
        existing = await self.get_flow(flow_id, user_id=user_id)
        if not existing:
            return None
        if existing.is_builtin:
            return None  # builtins are clone-only

        steps = updates.pop("steps", None)

        if updates:
            set_parts = []
            values = []
            for key, val in updates.items():
                if key in ("id", "is_builtin", "created_at"):
                    continue
                if isinstance(val, (list, dict)):
                    val = json.dumps(val)
                elif isinstance(val, bool):
                    val = int(val)
                set_parts.append(f"{key} = ?")
                values.append(val)

            if set_parts:
                set_parts.append("updated_at = datetime('now')")
                set_parts.append("version = version + 1")
                values.extend([flow_id, user_id])
                sql = (
                    f"UPDATE reasoning_flows SET {', '.join(set_parts)} "
                    "WHERE id = ? AND (user_id = ? OR user_id IS NULL)"
                )
                await self._db.execute(sql, values)

        if steps is not None:
            await self._replace_steps(flow_id, steps)

        await self._db.commit()
        return await self.get_flow(flow_id, user_id=user_id)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_flow(self, flow_id: str, *, user_id: str = "") -> bool:
        """Delete a flow (not builtins). Returns True if deleted."""
        if not user_id:
            raise ValueError("reasoning_flows delete requires user_id")
        cursor = await self._db.execute(
            "SELECT is_builtin FROM reasoning_flows "
            "WHERE id = ? AND (user_id = ? OR user_id IS NULL)",
            (flow_id, user_id),
        )
        row = await cursor.fetchone()
        if not row or row[0]:
            return False

        await self._db.execute(
            "DELETE FROM reasoning_flow_steps WHERE flow_id = ?", (flow_id,)
        )
        await self._db.execute(
            "DELETE FROM reasoning_flows "
            "WHERE id = ? AND (user_id = ? OR user_id IS NULL)",
            (flow_id, user_id),
        )
        await self._db.commit()
        return True

    # ------------------------------------------------------------------
    # Set default
    # ------------------------------------------------------------------

    async def set_default(self, flow_id: str, *, user_id: str = "") -> bool:
        """Set a flow as the default (unsets any previous default)."""
        if not user_id:
            raise ValueError("reasoning_flows set_default requires user_id")
        cursor = await self._db.execute(
            "SELECT id FROM reasoning_flows "
            "WHERE id = ? AND (user_id = ? OR user_id IS NULL)",
            (flow_id, user_id),
        )
        if not await cursor.fetchone():
            return False

        await self._db.execute(
            "UPDATE reasoning_flows SET is_default = 0 "
            "WHERE user_id = ? OR user_id IS NULL",
            (user_id,),
        )
        await self._db.execute(
            "UPDATE reasoning_flows SET is_default = 1 "
            "WHERE id = ? AND (user_id = ? OR user_id IS NULL)",
            (flow_id, user_id),
        )
        await self._db.commit()
        return True

    # ------------------------------------------------------------------
    # Clone
    # ------------------------------------------------------------------

    async def clone_flow(self, flow_id: str, new_name: str = "", *, user_id: str = "") -> ReasoningFlow | None:
        """Clone a flow (including builtins). Returns the new flow."""
        source = await self.get_flow(flow_id, user_id=user_id)
        if not source:
            return None

        clone = source.model_copy(deep=True)
        clone.id = _id()
        clone.name = new_name or f"{source.name} (copy)"
        clone.is_builtin = False
        clone.is_default = False
        clone.version = 1

        for step in clone.steps:
            step.id = _id()
            step.flow_id = clone.id

        await self._insert_flow(clone, user_id=user_id)
        await self._db.commit()
        return clone

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    async def export_flow(self, flow_id: str, *, user_id: str = "") -> dict | None:
        """Export a flow as a JSON-serializable dict."""
        flow = await self.get_flow(flow_id, user_id=user_id)
        if not flow:
            return None
        return flow.model_dump()

    async def import_flow(self, data: dict, *, user_id: str = "") -> ReasoningFlow:
        """Import a flow from a JSON dict. Assigns new IDs."""
        flow = ReasoningFlow(**data)
        flow.id = _id()
        flow.is_builtin = False
        flow.is_default = False

        for i, step in enumerate(flow.steps):
            step.id = _id()
            step.flow_id = flow.id
            step.sort_order = i

        await self._insert_flow(flow, user_id=user_id)
        await self._db.commit()
        return flow

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def list_all(self, *, user_id: str = "") -> list[ReasoningFlow]:
        """List all flows with their steps (for agentic flow resolution)."""
        if user_id:
            cursor = await self._db.execute(
                "SELECT * FROM reasoning_flows WHERE user_id = ? OR user_id IS NULL ORDER BY name ASC",
                (user_id,),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM reasoning_flows ORDER BY name ASC"
            )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        flows = []
        for row in rows:
            flow = self._row_to_flow(dict(zip(cols, row)))
            flow.steps = await self._get_steps(flow.id)
            flows.append(flow)
        return flows

    async def get(self, flow_id: str, *, user_id: str = "") -> ReasoningFlow | None:
        """Alias for get_flow (used by agentic handler)."""
        return await self.get_flow(flow_id, user_id=user_id)

    async def find_by_name(self, name_query: str, *, user_id: str = "") -> ReasoningFlow | None:
        """Find a flow by fuzzy name match (for /flow command).

        Tries: exact match > case-insensitive > substring > word overlap.
        """
        flows_with_counts = await self.list_flows(user_id=user_id)
        query_lower = name_query.lower().strip()
        if not query_lower:
            return None

        # Exact match
        for flow_summary, _ in flows_with_counts:
            if flow_summary.name == name_query:
                return await self.get_flow(flow_summary.id, user_id=user_id)

        # Case-insensitive exact
        for flow_summary, _ in flows_with_counts:
            if flow_summary.name.lower() == query_lower:
                return await self.get_flow(flow_summary.id, user_id=user_id)

        # Substring match (query in name or name in query)
        for flow_summary, _ in flows_with_counts:
            name_lower = flow_summary.name.lower()
            if query_lower in name_lower or name_lower in query_lower:
                return await self.get_flow(flow_summary.id, user_id=user_id)

        # Word overlap (best score wins)
        query_words = set(query_lower.split())
        best = None
        best_score = 0
        for flow_summary, _ in flows_with_counts:
            name_words = set(flow_summary.name.lower().split())
            overlap = len(query_words & name_words)
            if overlap > best_score:
                best_score = overlap
                best = flow_summary
        if best and best_score > 0:
            return await self.get_flow(best.id, user_id=user_id)

        return None

    async def greedy_match(self, text: str, *, user_id: str = "") -> tuple[ReasoningFlow | None, str]:
        """Greedy prefix matching: find the longest flow name prefix in text.

        Given text like "Deep Research world war 2", tries progressively
        shorter prefixes against known flow names:
          "Deep Research world war 2" → no match
          "Deep Research world war" → no match
          "Deep Research world" → no match
          "Deep Research" → match! remainder = "world war 2"

        Returns (matched_flow, remainder_query). If no match is found
        at any prefix length, falls back to find_by_name on the full text.
        """
        words = text.strip().split()
        if not words:
            return None, ""

        # Build a case-insensitive lookup of flow names
        flows_with_counts = await self.list_flows(user_id=user_id)
        name_map: dict[str, str] = {}  # lowercase name → flow id
        for flow_summary, _ in flows_with_counts:
            name_map[flow_summary.name.lower()] = flow_summary.id

        # Try longest prefix first, down to single word (exact name match)
        for n in range(len(words), 0, -1):
            candidate = " ".join(words[:n]).lower()
            if candidate in name_map:
                flow = await self.get_flow(name_map[candidate], user_id=user_id)
                if flow:
                    remainder = " ".join(words[n:]).strip()
                    return flow, remainder

        # No exact prefix matched — fall back to fuzzy find_by_name
        flow = await self.find_by_name(text, user_id=user_id)
        if flow:
            return flow, ""
        return None, text

    async def _insert_flow(self, flow: ReasoningFlow, *, user_id: str = "") -> None:
        if not user_id and not flow.is_builtin:
            raise ValueError("reasoning_flows insert requires user_id (non-builtin)")
        cols = ("id, name, description, icon, version, is_default, is_builtin,"
                " auto_select, trigger_domains, trigger_keywords, pinned_models,"
                " auto_search, max_tool_calls_per_step, autonomy_level")
        placeholders = "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
        params: list = [
            flow.id, flow.name, flow.description, flow.icon, flow.version,
            int(flow.is_default), int(flow.is_builtin), int(flow.auto_select),
            json.dumps(flow.trigger_domains), json.dumps(flow.trigger_keywords),
            json.dumps(flow.pinned_models), int(flow.auto_search),
            flow.max_tool_calls_per_step, flow.autonomy_level,
        ]
        if user_id:
            cols += ", user_id"
            placeholders += ", ?"
            params.append(user_id)
        await self._db.execute(
            f"INSERT INTO reasoning_flows ({cols}) VALUES ({placeholders})",
            params,
        )
        for step in flow.steps:
            await self._insert_step(step)

    async def _insert_step(self, step: FlowStep) -> None:
        await self._db.execute(
            """INSERT INTO reasoning_flow_steps
               (id, flow_id, sort_order, name, system_prompt, user_template,
                role, tool_categories, tool_names, complexity_gate,
                stream_to_user, output_cap, enabled, model_override,
                tool_choice)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                step.id, step.flow_id, step.sort_order, step.name,
                step.system_prompt, step.user_template, step.role,
                json.dumps(step.tool_categories), json.dumps(step.tool_names),
                json.dumps(step.complexity_gate), int(step.stream_to_user),
                step.output_cap, int(step.enabled), step.model_override,
                step.tool_choice,
            ),
        )

    async def _get_steps(self, flow_id: str) -> list[FlowStep]:
        cursor = await self._db.execute(
            "SELECT * FROM reasoning_flow_steps WHERE flow_id = ? ORDER BY sort_order",
            (flow_id,),
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [self._row_to_step(dict(zip(cols, row))) for row in rows]

    async def _replace_steps(self, flow_id: str, steps: list) -> None:
        await self._db.execute(
            "DELETE FROM reasoning_flow_steps WHERE flow_id = ?", (flow_id,)
        )
        for i, step_data in enumerate(steps):
            if isinstance(step_data, dict):
                step = FlowStep(**step_data)
            else:
                step = step_data
            step.id = step.id or _id()
            step.flow_id = flow_id
            step.sort_order = i
            await self._insert_step(step)

    @staticmethod
    def _parse_json_list(raw: str | None, fallback: str = "[]") -> list:
        """Safely parse a JSON list from a DB column.

        Handles legacy comma-separated strings and malformed JSON gracefully.
        """
        if not raw:
            return json.loads(fallback)
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return result
            return [result] if result else []
        except (json.JSONDecodeError, TypeError):
            # Legacy: comma-separated string like "agentic" or "search,fetch"
            return [s.strip() for s in raw.split(",") if s.strip()]

    @staticmethod
    def _row_to_flow(d: dict) -> ReasoningFlow:
        pjl = FlowStore._parse_json_list
        return ReasoningFlow(
            id=d["id"],
            name=d["name"],
            description=d.get("description", ""),
            icon=d.get("icon", ""),
            version=d.get("version", 1),
            is_default=bool(d.get("is_default", 0)),
            is_builtin=bool(d.get("is_builtin", 0)),
            auto_select=bool(d.get("auto_select", 1)),
            trigger_domains=pjl(d.get("trigger_domains", "[]")),
            trigger_keywords=pjl(d.get("trigger_keywords", "[]")),
            pinned_models=pjl(d.get("pinned_models", "[]")),
            auto_search=bool(d.get("auto_search", 1)),
            max_tool_calls_per_step=d.get("max_tool_calls_per_step", 3),
            autonomy_level=d.get("autonomy_level") or 2,
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )

    @staticmethod
    def _row_to_step(d: dict) -> FlowStep:
        pjl = FlowStore._parse_json_list
        return FlowStep(
            id=d["id"],
            flow_id=d["flow_id"],
            sort_order=d["sort_order"],
            name=d["name"],
            system_prompt=d.get("system_prompt", ""),
            user_template=d.get("user_template", ""),
            role=d.get("role", "analyze"),
            tool_categories=pjl(d.get("tool_categories", "[]")),
            tool_names=pjl(d.get("tool_names", "[]")),
            complexity_gate=pjl(d.get("complexity_gate", "[]")),
            stream_to_user=bool(d.get("stream_to_user", 0)),
            output_cap=d.get("output_cap", 800),
            enabled=bool(d.get("enabled", 1)),
            model_override=d.get("model_override", ""),
            tool_choice=d.get("tool_choice", "") or "",
        )
