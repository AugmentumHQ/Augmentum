"""Persistence layer for load balancer configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite


@dataclass
class BalancerConfig:
    """Configuration for a load balancer."""

    id: str
    name: str
    strategy: str = "round_robin"
    fallback_enabled: bool = False
    enabled: bool = True
    created_at: str = ""
    updated_at: str = ""


@dataclass
class BalancerMember:
    """A model member within a load balancer pool."""

    id: int = 0
    balancer_id: str = ""
    model_name: str = ""
    backend_key: str = ""
    weight: float = 1.0
    priority: int = 0
    enabled: bool = True
    last_used_at: str | None = None


@dataclass
class ABVoteStats:
    """Aggregated A/B vote statistics for a single model."""

    model_name: str
    backend_key: str
    up: int = 0
    down: int = 0

    @property
    def total(self) -> int:
        return self.up + self.down

    @property
    def score(self) -> float:
        return self.up / self.total if self.total > 0 else 0.5


class BalancerStore:
    """CRUD operations for load balancers in SQLite."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    # ------------------------------------------------------------------
    # Balancer CRUD
    # ------------------------------------------------------------------

    async def list_balancers(self) -> list[BalancerConfig]:
        cursor = await self._conn.execute(
            "SELECT * FROM load_balancers ORDER BY created_at"
        )
        rows = await cursor.fetchall()
        return [self._row_to_config(row) for row in rows]

    async def get_balancer(self, balancer_id: str) -> BalancerConfig | None:
        cursor = await self._conn.execute(
            "SELECT * FROM load_balancers WHERE id = ?", (balancer_id,)
        )
        row = await cursor.fetchone()
        return self._row_to_config(row) if row else None

    async def create_balancer(self, config: BalancerConfig) -> BalancerConfig:
        await self._conn.execute(
            """INSERT INTO load_balancers (id, name, strategy, fallback_enabled, enabled)
               VALUES (?, ?, ?, ?, ?)""",
            (config.id, config.name, config.strategy,
             1 if config.fallback_enabled else 0,
             1 if config.enabled else 0),
        )
        await self._conn.commit()
        return (await self.get_balancer(config.id)) or config

    async def update_balancer(
        self, balancer_id: str, **fields: str | bool | None
    ) -> BalancerConfig | None:
        existing = await self.get_balancer(balancer_id)
        if not existing:
            return None

        updates = ["updated_at = datetime('now')"]
        params: list = []

        for key in ("name", "strategy"):
            if key in fields:
                updates.append(f"{key} = ?")
                params.append(fields[key])

        if "fallback_enabled" in fields:
            updates.append("fallback_enabled = ?")
            params.append(1 if fields["fallback_enabled"] else 0)

        if "enabled" in fields:
            updates.append("enabled = ?")
            params.append(1 if fields["enabled"] else 0)

        params.append(balancer_id)
        await self._conn.execute(
            f"UPDATE load_balancers SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        await self._conn.commit()
        return await self.get_balancer(balancer_id)

    async def delete_balancer(self, balancer_id: str) -> bool:
        cursor = await self._conn.execute(
            "DELETE FROM load_balancers WHERE id = ?", (balancer_id,)
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Members
    # ------------------------------------------------------------------

    async def list_members(self, balancer_id: str) -> list[BalancerMember]:
        cursor = await self._conn.execute(
            "SELECT * FROM load_balancer_members WHERE balancer_id = ? ORDER BY priority",
            (balancer_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_member(row) for row in rows]

    async def add_member(
        self,
        balancer_id: str,
        model_name: str,
        backend_key: str,
        weight: float = 1.0,
        priority: int = 0,
    ) -> BalancerMember:
        cursor = await self._conn.execute(
            """INSERT INTO load_balancer_members
               (balancer_id, model_name, backend_key, weight, priority)
               VALUES (?, ?, ?, ?, ?)""",
            (balancer_id, model_name, backend_key, weight, priority),
        )
        await self._conn.commit()
        member_id = cursor.lastrowid
        return BalancerMember(
            id=member_id or 0,
            balancer_id=balancer_id,
            model_name=model_name,
            backend_key=backend_key,
            weight=weight,
            priority=priority,
        )

    async def update_member(self, member_id: int, **fields) -> bool:
        updates = []
        params: list = []
        for key in ("weight", "priority", "enabled"):
            if key in fields:
                updates.append(f"{key} = ?")
                val = fields[key]
                if key == "enabled":
                    val = 1 if val else 0
                params.append(val)
        if not updates:
            return False
        params.append(member_id)
        cursor = await self._conn.execute(
            f"UPDATE load_balancer_members SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def remove_member(self, member_id: int) -> bool:
        cursor = await self._conn.execute(
            "DELETE FROM load_balancer_members WHERE id = ?", (member_id,)
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def touch_last_used(self, member_id: int) -> None:
        await self._conn.execute(
            "UPDATE load_balancer_members SET last_used_at = datetime('now') WHERE id = ?",
            (member_id,),
        )
        await self._conn.commit()

    # ------------------------------------------------------------------
    # A/B Votes
    # ------------------------------------------------------------------

    async def record_vote(
        self,
        balancer_id: str,
        model_name: str,
        backend_key: str,
        vote: str,
        session_id: str | None = None,
        *,
        user_id: str = "",
    ) -> None:
        await self._conn.execute(
            """INSERT INTO ab_test_votes
               (balancer_id, model_name, backend_key, vote, session_id, user_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (balancer_id, model_name, backend_key, vote, session_id, user_id),
        )
        await self._conn.commit()

    async def get_vote_stats(
        self, balancer_id: str, *, user_id: str = "",
    ) -> list[ABVoteStats]:
        if user_id:
            cursor = await self._conn.execute(
                """SELECT model_name, backend_key,
                          SUM(CASE WHEN vote = 'up' THEN 1 ELSE 0 END) AS up,
                          SUM(CASE WHEN vote = 'down' THEN 1 ELSE 0 END) AS down
                   FROM ab_test_votes
                   WHERE balancer_id = ? AND user_id = ?
                   GROUP BY model_name, backend_key""",
                (balancer_id, user_id),
            )
        else:
            cursor = await self._conn.execute(
                """SELECT model_name, backend_key,
                          SUM(CASE WHEN vote = 'up' THEN 1 ELSE 0 END) AS up,
                          SUM(CASE WHEN vote = 'down' THEN 1 ELSE 0 END) AS down
                   FROM ab_test_votes
                   WHERE balancer_id = ?
                   GROUP BY model_name, backend_key""",
                (balancer_id,),
            )
        rows = await cursor.fetchall()
        return [
            ABVoteStats(
                model_name=dict(r)["model_name"],
                backend_key=dict(r)["backend_key"],
                up=dict(r)["up"],
                down=dict(r)["down"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_config(row: aiosqlite.Row) -> BalancerConfig:
        d = dict(row)
        return BalancerConfig(
            id=d["id"],
            name=d["name"],
            strategy=d.get("strategy", "round_robin"),
            fallback_enabled=bool(d.get("fallback_enabled", 0)),
            enabled=bool(d.get("enabled", 1)),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )

    @staticmethod
    def _row_to_member(row: aiosqlite.Row) -> BalancerMember:
        d = dict(row)
        return BalancerMember(
            id=d["id"],
            balancer_id=d["balancer_id"],
            model_name=d["model_name"],
            backend_key=d["backend_key"],
            weight=d.get("weight", 1.0),
            priority=d.get("priority", 0),
            enabled=bool(d.get("enabled", 1)),
            last_used_at=d.get("last_used_at"),
        )
