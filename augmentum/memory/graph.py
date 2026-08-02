"""Knowledge graph — schema-free entity-relationship graph with bi-temporal edges.

Stores nodes (entities) and edges (relationships) discovered from conversation.
Uses SQLite recursive CTEs for graph traversal — no Neo4j dependency.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from augmentum.memory.embeddings import EmbeddingService
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.state.backends.sqlite import SQLiteBackend

log = get_logger(__name__)

# Cosine similarity threshold for fuzzy entity resolution
ENTITY_RESOLUTION_THRESHOLD = 0.85


@dataclass
class GraphNode:
    """A node in the knowledge graph."""

    id: str
    label: str
    kind: str = "thing"
    properties: dict = field(default_factory=dict)
    chat_id: str | None = None
    user_id: str = "default"
    memory_id: str | None = None
    mentions: int = 1
    created_at: str = ""
    updated_at: str = ""


@dataclass
class GraphEdge:
    """A directed, weighted, bi-temporal edge in the knowledge graph."""

    id: int = 0
    source_id: str = ""
    target_id: str = ""
    relation: str = ""
    weight: float = 0.5
    evidence: str = ""
    chat_id: str | None = None
    valid_from: int | None = None
    valid_until: int | None = None
    message_idx: int | None = None
    created_at: str = ""
    updated_at: str = ""


class KnowledgeGraph:
    """Schema-free knowledge graph backed by SQLite.

    Provides CRUD, entity resolution, graph traversal, and temporal queries.
    """

    def __init__(
        self,
        backend: SQLiteBackend,
        entity_resolution_threshold: float = ENTITY_RESOLUTION_THRESHOLD,
    ) -> None:
        self._backend = backend
        self._resolution_threshold = entity_resolution_threshold

    @property
    def _conn(self):
        return self._backend.conn

    @property
    def _vec_enabled(self) -> bool:
        return self._backend.vec_enabled

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    async def upsert_node(
        self,
        label: str,
        kind: str = "thing",
        chat_id: str | None = None,
        user_id: str = "default",
        properties: dict | None = None,
        memory_id: str | None = None,
    ) -> GraphNode:
        """Insert or merge a node via fuzzy entity resolution.

        If a node with similar label already exists (cosine sim >= threshold),
        merges into it. Otherwise creates a new node.
        """
        # Try entity resolution first
        existing = await self.find_node(label, chat_id=chat_id, user_id=user_id)
        if existing:
            # Merge: increment mentions, update properties
            merged_props = {**existing.properties, **(properties or {})}
            now = datetime.now(UTC).isoformat()
            await self._conn.execute(
                "UPDATE kg_nodes SET mentions = mentions + 1, properties = ?, "
                "updated_at = ? WHERE id = ?",
                (json.dumps(merged_props), now, existing.id),
            )
            await self._conn.commit()
            existing.mentions += 1
            existing.properties = merged_props
            existing.updated_at = now
            log.debug("kg_node_merged", label=label, into=existing.id)
            return existing

        # Create new node — wrap in transaction so node + vec are atomic
        node_id = str(uuid.uuid4())[:16]
        embedding = EmbeddingService.embed_one(label)
        blob = EmbeddingService.to_blob(embedding)
        now = datetime.now(UTC).isoformat()

        try:
            await self._conn.execute(
                "INSERT INTO kg_nodes (id, label, kind, properties, chat_id, user_id, "
                "memory_id, embedding, mentions, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                (
                    node_id, label, kind, json.dumps(properties or {}),
                    chat_id, user_id, memory_id, blob, now, now,
                ),
            )

            # Insert into vec table for future resolution
            if self._vec_enabled:
                await self._conn.execute(
                    "INSERT INTO kg_nodes_vec (node_id, embedding) VALUES (?, ?)",
                    (node_id, blob),
                )

            await self._conn.commit()
        except Exception:
            await self._conn.rollback()
            raise
        log.debug("kg_node_created", id=node_id, label=label, kind=kind)

        return GraphNode(
            id=node_id, label=label, kind=kind,
            properties=properties or {},
            chat_id=chat_id, user_id=user_id,
            memory_id=memory_id, mentions=1,
            created_at=now, updated_at=now,
        )

    async def find_node(
        self,
        label: str,
        chat_id: str | None = None,
        user_id: str = "default",
    ) -> GraphNode | None:
        """Find a node by fuzzy label match (embedding similarity).

        Searches within the same chat scope. Also checks global nodes (chat_id IS NULL).
        """
        if not self._vec_enabled:
            # Fallback: exact label match
            cursor = await self._conn.execute(
                "SELECT * FROM kg_nodes WHERE label = ? AND user_id = ? "
                "AND (chat_id = ? OR chat_id IS NULL) LIMIT 1",
                (label, user_id, chat_id),
            )
            row = await cursor.fetchone()
            return self._row_to_node(dict(row)) if row else None

        embedding = await asyncio.to_thread(EmbeddingService.embed_query, label)
        blob = EmbeddingService.to_blob(embedding)

        try:
            cursor = await self._conn.execute(
                "SELECT node_id, distance FROM kg_nodes_vec "
                "WHERE embedding MATCH ? AND k = 10 ORDER BY distance",
                (blob,),
            )
            rows = await cursor.fetchall()
        except Exception:
            log.debug("kg_vec_search_failed", label=label, exc_info=True)
            return None

        for row in rows:
            similarity = 1.0 - row[1]
            if similarity < self._resolution_threshold:
                continue

            node = await self.get_node(row[0])
            if node and node.user_id == user_id:
                # Check scope: same chat or global
                if node.chat_id == chat_id or node.chat_id is None:
                    return node

        return None

    async def get_node(self, node_id: str) -> GraphNode | None:
        """Get a node by ID."""
        cursor = await self._conn.execute(
            "SELECT * FROM kg_nodes WHERE id = ?", (node_id,),
        )
        row = await cursor.fetchone()
        return self._row_to_node(dict(row)) if row else None

    async def get_nodes_by_chat(
        self,
        chat_id: str | None = None,
        user_id: str = "default",
        kind: str | None = None,
        limit: int = 50,
    ) -> list[GraphNode]:
        """List nodes for a chat (including global nodes).

        When chat_id is None, returns ALL nodes (not just global).
        """
        conditions = ["user_id = ?"]
        params: list = [user_id]

        if chat_id is not None:
            conditions.append("(chat_id = ? OR chat_id IS NULL)")
            params.append(chat_id)

        if kind:
            conditions.append("kind = ?")
            params.append(kind)

        where = " AND ".join(conditions)
        params.append(limit)

        cursor = await self._conn.execute(
            f"SELECT * FROM kg_nodes WHERE {where} ORDER BY mentions DESC LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_node(dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    async def upsert_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 0.5,
        evidence: str = "",
        chat_id: str | None = None,
        message_idx: int | None = None,
    ) -> GraphEdge:
        """Insert or update an edge. On conflict, reinforces weight."""
        now = datetime.now(UTC).isoformat()

        # Try update first (UNIQUE constraint on source, target, relation, chat)
        cursor = await self._conn.execute(
            "UPDATE kg_edges SET weight = MIN(1.0, weight + 0.1), "
            "evidence = ?, updated_at = ?, message_idx = ? "
            "WHERE source_id = ? AND target_id = ? AND relation = ? "
            "AND (chat_id = ? OR (chat_id IS NULL AND ? IS NULL)) "
            "AND valid_until IS NULL",
            (evidence, now, message_idx, source_id, target_id, relation, chat_id, chat_id),
        )

        if cursor.rowcount > 0:
            await self._conn.commit()
            log.debug("kg_edge_reinforced", source=source_id, target=target_id, relation=relation)
            # Fetch the updated edge
            cursor2 = await self._conn.execute(
                "SELECT * FROM kg_edges WHERE source_id = ? AND target_id = ? "
                "AND relation = ? AND (chat_id = ? OR (chat_id IS NULL AND ? IS NULL)) "
                "AND valid_until IS NULL",
                (source_id, target_id, relation, chat_id, chat_id),
            )
            row = await cursor2.fetchone()
            return self._row_to_edge(dict(row)) if row else GraphEdge()

        # Insert new
        await self._conn.execute(
            "INSERT INTO kg_edges (source_id, target_id, relation, weight, evidence, "
            "chat_id, valid_from, message_idx, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (source_id, target_id, relation, weight, evidence,
             chat_id, message_idx, message_idx, now, now),
        )
        await self._conn.commit()
        log.debug("kg_edge_created", source=source_id, target=target_id, relation=relation)

        return GraphEdge(
            source_id=source_id, target_id=target_id, relation=relation,
            weight=weight, evidence=evidence, chat_id=chat_id,
            valid_from=message_idx, message_idx=message_idx,
            created_at=now, updated_at=now,
        )

    async def get_edges(
        self,
        node_id: str,
        direction: str = "outgoing",
        chat_id: str | None = None,
        active_only: bool = True,
        limit: int = 20,
    ) -> list[GraphEdge]:
        """Get edges connected to a node."""
        col = "source_id" if direction == "outgoing" else "target_id"
        conditions = [f"{col} = ?"]
        params: list = [node_id]

        if active_only:
            conditions.append("valid_until IS NULL")

        conditions.append("(chat_id = ? OR chat_id IS NULL)")
        params.append(chat_id)
        params.append(limit)

        where = " AND ".join(conditions)
        cursor = await self._conn.execute(
            f"SELECT * FROM kg_edges WHERE {where} ORDER BY weight DESC LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_edge(dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    async def neighborhood(
        self,
        node_id: str,
        depth: int = 1,
        chat_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[GraphNode, GraphEdge]]:
        """Get the N-hop neighborhood of a node.

        Returns (node, connecting_edge) pairs for all reachable nodes.
        """
        if depth < 1:
            return []

        # Use recursive CTE for multi-hop traversal
        cursor = await self._conn.execute(
            """
            WITH RECURSIVE connected(node_id, edge_id, depth, visited) AS (
                SELECT e.target_id, e.id, 1,
                       e.source_id || ',' || e.target_id
                FROM kg_edges e
                WHERE e.source_id = ? AND e.valid_until IS NULL
                  AND (e.chat_id = ? OR e.chat_id IS NULL)
                UNION ALL
                SELECT e.target_id, e.id, c.depth + 1,
                       c.visited || ',' || e.target_id
                FROM kg_edges e
                JOIN connected c ON e.source_id = c.node_id
                WHERE c.depth < ?
                  AND e.valid_until IS NULL
                  AND (e.chat_id = ? OR e.chat_id IS NULL)
                  AND c.visited NOT LIKE '%' || e.target_id || '%'
            )
            SELECT DISTINCT node_id, edge_id FROM connected
            LIMIT ?
            """,
            (node_id, chat_id, depth, chat_id, limit),
        )
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            node = await self.get_node(row[0])
            edge_cursor = await self._conn.execute(
                "SELECT * FROM kg_edges WHERE id = ?", (row[1],),
            )
            edge_row = await edge_cursor.fetchone()
            if node and edge_row:
                results.append((node, self._row_to_edge(dict(edge_row))))

        return results

    async def shortest_path(
        self,
        source_id: str,
        target_id: str,
        chat_id: str | None = None,
        max_depth: int = 4,
    ) -> list[tuple[GraphNode, GraphEdge]] | None:
        """Find shortest path between two nodes. Returns None if no path."""
        cursor = await self._conn.execute(
            """
            WITH RECURSIVE paths(node_id, depth, path, edge_path) AS (
                SELECT e.target_id, 1,
                       ? || ',' || e.target_id,
                       CAST(e.id AS TEXT)
                FROM kg_edges e
                WHERE e.source_id = ? AND e.valid_until IS NULL
                  AND (e.chat_id = ? OR e.chat_id IS NULL)
                UNION ALL
                SELECT e.target_id, p.depth + 1,
                       p.path || ',' || e.target_id,
                       p.edge_path || ',' || CAST(e.id AS TEXT)
                FROM kg_edges e
                JOIN paths p ON e.source_id = p.node_id
                WHERE p.depth < ?
                  AND e.valid_until IS NULL
                  AND (e.chat_id = ? OR e.chat_id IS NULL)
                  AND p.path NOT LIKE '%' || e.target_id || '%'
            )
            SELECT path, edge_path FROM paths
            WHERE node_id = ?
            ORDER BY depth
            LIMIT 1
            """,
            (source_id, source_id, chat_id, max_depth, chat_id, target_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        node_ids = row[0].split(",")
        edge_ids = row[1].split(",")

        results = []
        for nid, eid in zip(node_ids[1:], edge_ids):
            node = await self.get_node(nid)
            edge_cursor = await self._conn.execute(
                "SELECT * FROM kg_edges WHERE id = ?", (int(eid),),
            )
            edge_row = await edge_cursor.fetchone()
            if node and edge_row:
                results.append((node, self._row_to_edge(dict(edge_row))))

        return results

    # ------------------------------------------------------------------
    # Decay and maintenance
    # ------------------------------------------------------------------

    async def decay_edges(
        self,
        chat_id: str | None = None,
        factor: float = 0.95,
        prune_threshold: float = 0.1,
    ) -> dict[str, int]:
        """Apply decay to all active edges and prune weak ones.

        Returns stats: {decayed, pruned}.
        """
        now = datetime.now(UTC).isoformat()

        # Decay all active edges
        cursor = await self._conn.execute(
            "UPDATE kg_edges SET weight = weight * ?, updated_at = ? "
            "WHERE valid_until IS NULL AND (chat_id = ? OR chat_id IS NULL)",
            (factor, now, chat_id),
        )
        decayed = cursor.rowcount

        # Prune edges below threshold
        cursor2 = await self._conn.execute(
            "UPDATE kg_edges SET valid_until = -1 "
            "WHERE valid_until IS NULL AND weight < ? "
            "AND (chat_id = ? OR chat_id IS NULL)",
            (prune_threshold, chat_id),
        )
        pruned = cursor2.rowcount

        await self._conn.commit()
        if decayed or pruned:
            log.info("kg_decay_applied", decayed=decayed, pruned=pruned)

        return {"decayed": decayed, "pruned": pruned}

    async def promote_to_global(self, node_id: str) -> bool:
        """Promote a chat-scoped node to global scope."""
        now = datetime.now(UTC).isoformat()
        cursor = await self._conn.execute(
            "UPDATE kg_nodes SET chat_id = NULL, updated_at = ? WHERE id = ?",
            (now, node_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Context generation
    # ------------------------------------------------------------------

    async def build_context_summary(
        self,
        chat_id: str | None = None,
        user_id: str = "default",
        max_chars: int = 800,
    ) -> str:
        """Build a concise text summary of the knowledge graph for context injection."""
        nodes = await self.get_nodes_by_chat(chat_id=chat_id, user_id=user_id, limit=30)
        if not nodes:
            return ""

        parts = []
        chars = 0

        # Group nodes by kind
        by_kind: dict[str, list[GraphNode]] = {}
        for node in nodes:
            by_kind.setdefault(node.kind, []).append(node)

        for kind, kind_nodes in by_kind.items():
            labels = [n.label for n in kind_nodes[:10]]
            line = f"{kind.title()}s: {', '.join(labels)}"
            if chars + len(line) > max_chars:
                break
            parts.append(line)
            chars += len(line)

        # Add top relationships
        for node in nodes[:10]:
            edges = await self.get_edges(node.id, chat_id=chat_id, limit=5)
            for edge in edges:
                target = await self.get_node(edge.target_id)
                if target:
                    strength = "strong" if edge.weight > 0.7 else "moderate" if edge.weight > 0.4 else "weak"
                    line = f"{node.label} --{edge.relation}({strength})--> {target.label}"
                    if chars + len(line) > max_chars:
                        break
                    parts.append(line)
                    chars += len(line)
            if chars >= max_chars:
                break

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def stats(
        self,
        chat_id: str | None = None,
        user_id: str = "default",
    ) -> dict[str, int]:
        """Get graph statistics.

        When chat_id is None, counts ALL nodes/edges (not just global).
        """
        if chat_id is not None:
            node_cursor = await self._conn.execute(
                "SELECT COUNT(*) FROM kg_nodes WHERE user_id = ? "
                "AND (chat_id = ? OR chat_id IS NULL)",
                (user_id, chat_id),
            )
            edge_cursor = await self._conn.execute(
                "SELECT COUNT(*) FROM kg_edges WHERE valid_until IS NULL "
                "AND (chat_id = ? OR chat_id IS NULL)",
                (chat_id,),
            )
        else:
            node_cursor = await self._conn.execute(
                "SELECT COUNT(*) FROM kg_nodes WHERE user_id = ?",
                (user_id,),
            )
            edge_cursor = await self._conn.execute(
                "SELECT COUNT(*) FROM kg_edges WHERE valid_until IS NULL",
            )
        node_count = (await node_cursor.fetchone())[0]
        edge_count = (await edge_cursor.fetchone())[0]

        return {"nodes": node_count, "edges": edge_count}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_node(row: dict) -> GraphNode:
        props = row.get("properties", "{}")
        if isinstance(props, str):
            try:
                props = json.loads(props)
            except (json.JSONDecodeError, TypeError):
                props = {}

        return GraphNode(
            id=row["id"],
            label=row["label"],
            kind=row.get("kind", "thing"),
            properties=props,
            chat_id=row.get("chat_id"),
            user_id=row.get("user_id", "default"),
            memory_id=row.get("memory_id"),
            mentions=row.get("mentions", 1),
            created_at=row.get("created_at", ""),
            updated_at=row.get("updated_at", ""),
        )

    @staticmethod
    def _row_to_edge(row: dict) -> GraphEdge:
        return GraphEdge(
            id=row.get("id", 0),
            source_id=row.get("source_id", ""),
            target_id=row.get("target_id", ""),
            relation=row.get("relation", ""),
            weight=row.get("weight", 0.5),
            evidence=row.get("evidence", ""),
            chat_id=row.get("chat_id"),
            valid_from=row.get("valid_from"),
            valid_until=row.get("valid_until"),
            message_idx=row.get("message_idx"),
            created_at=row.get("created_at", ""),
            updated_at=row.get("updated_at", ""),
        )
