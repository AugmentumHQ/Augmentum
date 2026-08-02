"""Cross-project coder profile store — Phase 2.3 of the coder foundation.

Records observations the agent makes about *how* the user works:
preferred conventions per language, per-project quirks, edit-pattern
accept/reject signals, tool preferences. Workspace-local entries
override global entries via :meth:`CoderProfileStore.query_for_workspace`.

This phase ships the data layer only. Population happens later in the
retro loop (Phase 7); the LLM-side surfacing (turning a profile entry
into a system-prompt nudge) lands when cross-modal context (Phase 5)
needs it.

Multi-tenancy: every method requires ``user_id``. There's no legitimate
"system-owned" coder preference — preferences belong to the user.

Workspace scoping: ``workspace_id=""`` (or ``None``) = global; any other
value = workspace-local. The empty-string sentinel is used in SQL
(rather than NULL) so the ``UNIQUE(user_id, workspace_id, category, key)``
constraint treats global rows as deduplicating correctly.
"""
from __future__ import annotations

import json
import shlex
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

import aiosqlite

ACTIVE_PROFILE_CATEGORIES = frozenset({
    "runtime",
    "project",
    "command",
    "convention",
    "failure",
    "preference",
    # Backward-compatible categories from the original profile store.
    "language",
    "pattern",
    "tool",
})


@dataclass(frozen=True, slots=True)
class ProfileEntry:
    """One observation in the coder profile.

    Attributes
    ----------
    id:
        Stable per-row UUID.
    user_id:
        Owning user. Required.
    workspace_id:
        Workspace this preference scopes to, or ``""`` for global.
    category:
        Coarse bucket — ``language`` / ``project`` / ``pattern`` /
        ``tool`` / ``convention`` are the conventional values, but the
        store doesn't enforce a closed set so future categories can
        slot in without a migration.
    key:
        Specific preference key within the category. Examples:
        ``python.return_type_style``, ``naming.case``,
        ``edit_format.prefer_replace_block``.
    value:
        JSON-decoded payload. Free-form so callers can store strings,
        numbers, lists, or small structured objects without a schema
        explosion.
    confidence:
        Heuristic strength of the observation. 0.0-1.0. Bumped by
        repeated observations; should be down-weighted by accept/reject
        signals once those land.
    observation_count:
        How many times the entry has been bumped. Used both as a
        confidence proxy and as a freshness signal.
    last_observed_at:
        Epoch seconds of the most recent observation.
    created_at:
        Epoch seconds when first recorded.
    """

    id: str
    user_id: str
    workspace_id: str
    category: str
    key: str
    value: Any
    confidence: float
    observation_count: int
    last_observed_at: float
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self)}


def _row_to_entry(row: aiosqlite.Row | dict) -> ProfileEntry:
    """Coerce a SQLite row into a typed entry.

    Tolerates both ``aiosqlite.Row`` and a plain dict (the latter
    keeps in-memory test fixtures simple). The stored ``value`` column
    is JSON; non-JSON legacy rows fall back to the raw string.
    """
    raw_value = row["value"] if "value" in row.keys() else ""
    try:
        decoded: Any = json.loads(raw_value) if raw_value else ""
    except (json.JSONDecodeError, TypeError):
        decoded = raw_value
    return ProfileEntry(
        id=row["id"],
        user_id=row["user_id"],
        workspace_id=row["workspace_id"] or "",
        category=row["category"],
        key=row["key"],
        value=decoded,
        confidence=float(row["confidence"]),
        observation_count=int(row["observation_count"]),
        last_observed_at=float(row["last_observed_at"]),
        created_at=float(row["created_at"]),
    )


class CoderProfileStore:
    """CRUD for the ``coder_profile`` table.

    Used by:
      - the retro loop (Phase 7) for population on each turn close
      - the context assembler (Phase 5) for cross-modal system-prompt
        injection
      - tests + the eval harness for snapshot inspection

    Not used by the act loop directly — the store is async and the
    sticky reminder must stay synchronous. The handler reads a
    pre-built block once per turn before entering the loop.
    """

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    @staticmethod
    def _normalize_workspace(workspace_id: str | None) -> str:
        """``None`` and ``""`` both mean global. Stored as empty string
        so the UNIQUE constraint deduplicates correctly."""
        return (workspace_id or "").strip()

    # ── Writes ────────────────────────────────────────────────────────

    async def upsert(
        self,
        *,
        user_id: str,
        category: str,
        key: str,
        value: Any,
        workspace_id: str | None = None,
        confidence: float = 0.5,
    ) -> ProfileEntry:
        """Record (or refresh) an observation.

        First call creates the row with ``observation_count=1`` and the
        provided ``confidence``. Subsequent calls increment
        ``observation_count`` and update ``last_observed_at``; they
        also adopt the new ``value`` and ``confidence`` (assumption: a
        fresher observation is more correct than an older one).

        Returns the entry as it now stands in the DB.
        """
        if not user_id:
            raise ValueError("CoderProfileStore.upsert requires user_id")
        if not category or not key:
            raise ValueError("CoderProfileStore.upsert requires category + key")
        ws = self._normalize_workspace(workspace_id)
        now = time.time()
        encoded = json.dumps(value, default=str)

        # Try update first. If rowcount==0 we insert.
        cursor = await self._conn.execute(
            """UPDATE coder_profile
               SET value = ?, confidence = ?,
                   observation_count = observation_count + 1,
                   last_observed_at = ?
               WHERE user_id = ? AND workspace_id = ?
                 AND category = ? AND key = ?""",
            (encoded, confidence, now, user_id, ws, category, key),
        )
        if cursor.rowcount == 0:
            await self._conn.execute(
                """INSERT INTO coder_profile
                   (id, user_id, workspace_id, category, key, value,
                    confidence, observation_count, last_observed_at,
                    created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                (
                    uuid.uuid4().hex[:16],
                    user_id, ws, category, key, encoded,
                    confidence, now, now,
                ),
            )
        await self._conn.commit()

        # Re-read for the canonical row (gives us the actual
        # observation_count after the upsert).
        return await self._require_get(
            user_id=user_id, workspace_id=ws, category=category, key=key,
        )

    async def delete(
        self,
        *,
        user_id: str,
        category: str,
        key: str,
        workspace_id: str | None = None,
    ) -> bool:
        """Drop a single (user, workspace, category, key) row.

        Returns True iff a row was deleted.
        """
        if not user_id:
            raise ValueError("CoderProfileStore.delete requires user_id")
        ws = self._normalize_workspace(workspace_id)
        cursor = await self._conn.execute(
            """DELETE FROM coder_profile
               WHERE user_id = ? AND workspace_id = ?
                 AND category = ? AND key = ?""",
            (user_id, ws, category, key),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    # ── Reads ─────────────────────────────────────────────────────────

    async def get(
        self,
        *,
        user_id: str,
        category: str,
        key: str,
        workspace_id: str | None = None,
    ) -> ProfileEntry | None:
        """Look up a single entry by exact (user, workspace, category, key)."""
        if not user_id:
            raise ValueError("CoderProfileStore.get requires user_id")
        ws = self._normalize_workspace(workspace_id)
        self._conn.row_factory = aiosqlite.Row
        cursor = await self._conn.execute(
            """SELECT * FROM coder_profile
               WHERE user_id = ? AND workspace_id = ?
                 AND category = ? AND key = ?""",
            (user_id, ws, category, key),
        )
        row = await cursor.fetchone()
        return _row_to_entry(row) if row else None

    async def _require_get(
        self,
        *,
        user_id: str,
        workspace_id: str,
        category: str,
        key: str,
    ) -> ProfileEntry:
        entry = await self.get(
            user_id=user_id, workspace_id=workspace_id,
            category=category, key=key,
        )
        if entry is None:
            raise RuntimeError(
                "coder_profile upsert succeeded but row missing on re-read "
                f"(user={user_id!r}, workspace={workspace_id!r}, "
                f"category={category!r}, key={key!r})"
            )
        return entry

    async def query_for_workspace(
        self,
        *,
        user_id: str,
        workspace_id: str,
        category: str | None = None,
    ) -> list[ProfileEntry]:
        """Workspace-aware query: workspace-local rows shadow global rows.

        For each (category, key) pair, returns the workspace-local row
        if one exists, otherwise the global row. Useful for "give me
        the agent's view of preferences for this workspace" — global
        defaults plus per-workspace overrides, with no caller-side
        merge.

        ``workspace_id`` must be a non-empty workspace identifier.
        Use :meth:`query_global` to fetch only the global slice.
        """
        if not user_id:
            raise ValueError("CoderProfileStore.query_for_workspace requires user_id")
        if not workspace_id:
            raise ValueError(
                "query_for_workspace needs a real workspace_id; use query_global "
                "for the global slice"
            )
        self._conn.row_factory = aiosqlite.Row
        sql = (
            "SELECT * FROM coder_profile "
            "WHERE user_id = ? AND (workspace_id = ? OR workspace_id = '')"
        )
        params: list[Any] = [user_id, workspace_id]
        if category:
            sql += " AND category = ?"
            params.append(category)
        # Sort so workspace-local rows come before global per (category, key).
        sql += (
            " ORDER BY category, key, "
            "CASE WHEN workspace_id = '' THEN 1 ELSE 0 END"
        )
        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()

        seen: set[tuple[str, str]] = set()
        result: list[ProfileEntry] = []
        for row in rows:
            ck = (row["category"], row["key"])
            if ck in seen:
                continue   # global shadowed by a workspace-local row
            seen.add(ck)
            result.append(_row_to_entry(row))
        return result

    async def query_global(
        self,
        *,
        user_id: str,
        category: str | None = None,
    ) -> list[ProfileEntry]:
        """All global (workspace-agnostic) entries for a user."""
        if not user_id:
            raise ValueError("CoderProfileStore.query_global requires user_id")
        self._conn.row_factory = aiosqlite.Row
        sql = "SELECT * FROM coder_profile WHERE user_id = ? AND workspace_id = ''"
        params: list[Any] = [user_id]
        if category:
            sql += " AND category = ?"
            params.append(category)
        sql += " ORDER BY category, key"
        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_entry(row) for row in rows]

    async def query_workspace_only(
        self,
        *,
        user_id: str,
        workspace_id: str,
        category: str | None = None,
    ) -> list[ProfileEntry]:
        """Only workspace-local entries (no global merge).

        Useful when introspecting how the per-workspace memory has
        diverged from defaults — e.g. for the retro loop's "this
        workspace prefers X over the user's usual Y" surfacing.
        """
        if not user_id or not workspace_id:
            raise ValueError(
                "query_workspace_only requires both user_id and workspace_id"
            )
        self._conn.row_factory = aiosqlite.Row
        sql = (
            "SELECT * FROM coder_profile "
            "WHERE user_id = ? AND workspace_id = ?"
        )
        params: list[Any] = [user_id, workspace_id]
        if category:
            sql += " AND category = ?"
            params.append(category)
        sql += " ORDER BY category, key"
        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_entry(row) for row in rows]


def _value_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, sort_keys=True)


def _entry_relevance(entry: ProfileEntry, query: str) -> float:
    q_terms = {
        term.lower()
        for term in (query or "").replace("_", " ").replace("-", " ").split()
        if len(term) > 2
    }
    haystack = " ".join([
        entry.category,
        entry.key,
        _value_text(entry.value),
    ]).lower()
    overlap = sum(1 for term in q_terms if term in haystack)
    category_boost = {
        "command": 1.2,
        "runtime": 1.0,
        "project": 0.9,
        "failure": 0.8,
        "convention": 0.7,
        "preference": 0.6,
    }.get(entry.category, 0.4)
    workspace_boost = 0.4 if entry.workspace_id else 0.0
    freshness = min(0.5, max(0.0, entry.observation_count - 1) * 0.08)
    return (
        float(entry.confidence or 0.0)
        + overlap * 0.6
        + category_boost
        + workspace_boost
        + freshness
    )


def render_profile_block(
    entries: list[ProfileEntry],
    *,
    query: str = "",
    max_entries: int = 8,
) -> str:
    """Render concise workspace memory for prompt injection.

    The profile table can accumulate many observations. The agent loop
    should only see the highest-signal facts for the current request, so
    this function scores by category, confidence, workspace locality, and
    rough lexical overlap with the latest user message.
    """
    candidates = [
        entry for entry in entries
        if entry.category in ACTIVE_PROFILE_CATEGORIES
    ]
    ranked = sorted(
        candidates,
        key=lambda entry: _entry_relevance(entry, query),
        reverse=True,
    )[: max(1, min(int(max_entries or 8), 16))]
    if not ranked:
        return ""
    lines = ["## Workspace Profile", "Use these concise learned facts when relevant:"]
    for entry in ranked:
        value = _value_text(entry.value)
        if len(value) > 160:
            value = value[:157] + "..."
        scope = "workspace" if entry.workspace_id else "global"
        lines.append(f"- [{scope}] {entry.category}.{entry.key}: {value}")
    return "\n".join(lines)


async def upsert_profile_if_changed(
    store: CoderProfileStore,
    *,
    user_id: str,
    workspace_id: str,
    category: str,
    key: str,
    value: Any,
    confidence: float = 0.7,
) -> ProfileEntry | None:
    """Upsert a workspace fact only when it is new or materially changed."""
    if not user_id or not workspace_id:
        return None
    try:
        existing = await store.get(
            user_id=user_id,
            workspace_id=workspace_id,
            category=category,
            key=key,
        )
        if existing is not None and existing.value == value:
            return existing
        return await store.upsert(
            user_id=user_id,
            workspace_id=workspace_id,
            category=category,
            key=key,
            value=value,
            confidence=confidence,
        )
    except Exception:
        return None


async def observe_workspace_profile(
    store: CoderProfileStore,
    *,
    user_id: str,
    workspace_id: str,
    container_manager,
) -> None:
    """Best-effort inference of project runtime facts.

    This intentionally records only stable, low-noise facts from project
    manifests. It does not infer personal preferences or failures; those
    should come from accepted completions and controller validation.
    """
    if not user_id or not workspace_id or container_manager is None:
        return
    script = (
        "import json,os\n"
        "files=['package.json','pnpm-lock.yaml','yarn.lock','package-lock.json',"
        "'bun.lockb','pyproject.toml','requirements.txt','pytest.ini',"
        "'vite.config.js','vite.config.ts','next.config.js','go.mod','Cargo.toml']\n"
        "out={'files':[f for f in files if os.path.exists(f)]}\n"
        "if os.path.exists('package.json'):\n"
        "    try:\n"
        "        pkg=json.load(open('package.json',encoding='utf-8'))\n"
        "        out['package_scripts']=pkg.get('scripts') or {}\n"
        "        deps={**(pkg.get('dependencies') or {}), **(pkg.get('devDependencies') or {})}\n"
        "        out['js_deps']=list(deps.keys())[:200]\n"
        "    except Exception as exc:\n"
        "        out['package_error']=str(exc)\n"
        "print(json.dumps(out,sort_keys=True))\n"
    )
    try:
        output = await container_manager.run_command(
            workspace_id,
            ["bash", "-lc", f"python3 -c {shlex.quote(script)}"],
            timeout=5.0,
        )
        data = json.loads((output or "{}").strip().splitlines()[-1])
    except Exception:
        return

    files = set(data.get("files") or [])
    if not files:
        return

    package_manager = ""
    if "pnpm-lock.yaml" in files:
        package_manager = "pnpm"
    elif "yarn.lock" in files:
        package_manager = "yarn"
    elif "bun.lockb" in files:
        package_manager = "bun"
    elif "package-lock.json" in files or "package.json" in files:
        package_manager = "npm"
    if package_manager:
        await upsert_profile_if_changed(
            store,
            user_id=user_id,
            workspace_id=workspace_id,
            category="runtime",
            key="package_manager",
            value=package_manager,
            confidence=0.85,
        )

    scripts = data.get("package_scripts") if isinstance(data.get("package_scripts"), dict) else {}
    command_keys = {
        "dev": "dev_command",
        "test": "test_command",
        "lint": "lint_command",
        "build": "build_command",
    }
    for script_name, key in command_keys.items():
        if script_name in scripts:
            runner = package_manager or "npm"
            cmd = (
                f"{runner} {script_name}"
                if runner in {"yarn", "bun"}
                else f"{runner} run {script_name}"
            )
            if script_name == "test" and runner == "npm":
                cmd = "npm test"
            await upsert_profile_if_changed(
                store,
                user_id=user_id,
                workspace_id=workspace_id,
                category="command",
                key=key,
                value=cmd,
                confidence=0.8,
            )

    deps = set(data.get("js_deps") or [])
    framework = ""
    if "next" in deps or "next.config.js" in files:
        framework = "Next.js"
    elif "vite" in deps or "vite.config.js" in files or "vite.config.ts" in files:
        framework = "Vite"
    elif "react" in deps:
        framework = "React"
    elif "vue" in deps:
        framework = "Vue"
    elif "svelte" in deps:
        framework = "Svelte"
    elif "pyproject.toml" in files:
        framework = "Python"
    elif "go.mod" in files:
        framework = "Go"
    elif "Cargo.toml" in files:
        framework = "Rust"
    if framework:
        await upsert_profile_if_changed(
            store,
            user_id=user_id,
            workspace_id=workspace_id,
            category="project",
            key="framework",
            value=framework,
            confidence=0.75,
        )
