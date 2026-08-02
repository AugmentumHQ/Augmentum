"""Discover, load, and cache subagent role definitions.

Sources, in precedence order (workspace > user > built-in):

1. **Workspace-local**: ``<workspace>/.augmentum/agents/*.md`` — roles
   that travel with the project (version-controlled).
2. **User-global**: ``~/.augmentum/agents/*.md`` — the user's personal
   role library, shared across workspaces.
3. **Built-ins**: presets defined in ``presets.py``. Always present.

Each file is YAML frontmatter + Markdown body. The frontmatter declares
``name``, ``model.preferred``, ``model.fallbacks``, ``tools``,
``budget``, ``context``, etc.; the body is the role's system prompt.

Hot reload: ``AgentRegistry.refresh_if_stale()`` re-scans when ANY
discovered file's mtime changed. The dispatcher calls this once per
``task_dispatch`` invocation — discovery is cheap (filesystem walk +
mtime cmp), parsing only re-runs for changed files.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from augmentum.agents.budget import SubagentBudget
from augmentum.agents.spec import CONTEXT_MODES, AgentRole
from augmentum.agents.tools import resolve_tool_spec
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


_FRONTMATTER_DELIM = "---"


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from Markdown body.

    Tolerates: missing frontmatter (returns ``{}, text``), CRLF line
    endings, leading whitespace before the opening ``---``.

    Yaml parsed via PyYAML if available; otherwise a tiny key:value
    parser handles the common-case role file (string values + nested
    one-level dicts). Role files SHOULD use yaml-clean shapes.
    """
    stripped = text.lstrip()
    if not stripped.startswith(_FRONTMATTER_DELIM):
        return {}, text

    rest = stripped[len(_FRONTMATTER_DELIM):].lstrip("\n").lstrip("\r\n")
    end = rest.find(f"\n{_FRONTMATTER_DELIM}")
    if end == -1:
        return {}, text

    front_text = rest[:end]
    body = rest[end + len(_FRONTMATTER_DELIM) + 1:].lstrip("\n")

    try:
        import yaml  # type: ignore
        parsed = yaml.safe_load(front_text) or {}
    except ImportError:
        parsed = _parse_simple_yaml(front_text)
    if not isinstance(parsed, dict):
        return {}, body
    return parsed, body


def _parse_simple_yaml(text: str) -> dict:
    """Tiny fallback YAML parser for role files without PyYAML.

    Handles: top-level ``key: value``, top-level ``key:`` followed by
    indented ``- item`` (list) or ``  subkey: value`` (dict). Strings
    stay strings, ``true``/``false`` become bool, integers become int.
    """
    out: dict = {}
    cur_key: str | None = None
    cur_kind: str | None = None  # "list" or "dict"
    cur_indent = 0

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if indent == 0:
            cur_kind = None
            if ":" in stripped:
                k, _, v = stripped.partition(":")
                k = k.strip()
                v = v.strip()
                if not v:
                    cur_key = k
                    out[k] = None
                else:
                    out[k] = _coerce_scalar(v)
                    cur_key = None
            continue
        # Indented under cur_key.
        if cur_key is None:
            continue
        if stripped.startswith("- "):
            cur_kind = "list"
            if not isinstance(out[cur_key], list):
                out[cur_key] = []
            out[cur_key].append(_coerce_scalar(stripped[2:].strip()))
        elif ":" in stripped:
            cur_kind = "dict"
            if not isinstance(out[cur_key], dict):
                out[cur_key] = {}
            sk, _, sv = stripped.partition(":")
            sk = sk.strip()
            sv = sv.strip()
            if sv:
                out[cur_key][sk] = _coerce_scalar(sv)
            else:
                out[cur_key][sk] = None
        cur_indent = indent
        _ = cur_indent  # silence "unused"

    return out


def _coerce_scalar(v: str):
    s = v.strip().strip("'\"")
    if s.lower() in ("true", "yes"):
        return True
    if s.lower() in ("false", "no"):
        return False
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_coerce_scalar(p) for p in inner.split(",")]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _role_from_frontmatter(
    front: dict,
    body: str,
    *,
    source: str,
    file_path: str,
    mtime: float,
) -> AgentRole | None:
    """Build an ``AgentRole`` from parsed frontmatter + body.

    Returns ``None`` if mandatory fields are missing (logs a warning).
    """
    name = str(front.get("name") or "").strip()
    if not name:
        log.warning("agent_role_missing_name", path=file_path)
        return None

    model_block = front.get("model") or {}
    if isinstance(model_block, str):
        preferred = model_block.strip()
        fallbacks: list[str] = []
    elif isinstance(model_block, dict):
        preferred = str(model_block.get("preferred") or "").strip()
        fb_raw = model_block.get("fallbacks") or []
        fallbacks = [str(x).strip() for x in fb_raw if str(x).strip()]
    else:
        preferred = ""
        fallbacks = []

    tool_spec = front.get("tools", "read_only")
    tools = tuple(sorted(resolve_tool_spec(tool_spec)))

    budget_block = front.get("budget") or {}
    if isinstance(budget_block, dict):
        budget = SubagentBudget(
            max_iterations=int(budget_block.get("iterations", 30)),
            max_wallclock_seconds=float(budget_block.get("wallclock_s", 600.0)),
            max_tokens=int(budget_block.get("tokens", 200_000)),
        )
    else:
        budget = SubagentBudget()

    ctx_block = front.get("context") or {}
    if isinstance(ctx_block, dict):
        ctx_mode = str(ctx_block.get("mode") or "workspace").strip()
    else:
        ctx_mode = str(ctx_block).strip() or "workspace"
    if ctx_mode not in CONTEXT_MODES:
        log.warning(
            "agent_role_invalid_context_mode",
            path=file_path,
            mode=ctx_mode,
        )
        ctx_mode = "workspace"

    vis_block = front.get("visibility") or {}
    if not isinstance(vis_block, dict):
        vis_block = {}
    perms_block = front.get("permissions") or {}
    if not isinstance(perms_block, dict):
        perms_block = {}
    par_block = front.get("parallelism") or {}
    if not isinstance(par_block, dict):
        par_block = {}

    return AgentRole(
        name=name,
        description=str(front.get("description") or "").strip(),
        system_prompt=body.strip() or str(front.get("system_prompt") or "").strip(),
        preferred_model=preferred,
        fallback_models=tuple(fallbacks),
        tools=tools,
        budget=budget,
        tool_guard=str(front.get("tool_guard") or "detector"),
        context_mode=ctx_mode,
        stream_to_parent=bool(vis_block.get("stream_to_parent", True)),
        visible_in_ui=bool(vis_block.get("visible_in_ui", True)),
        log_persistence=bool(vis_block.get("log_persistence", True)),
        can_spawn_subagents=bool(perms_block.get("can_spawn_subagents", False)),
        max_concurrent=int(par_block.get("max_concurrent", 4)),
        source=source,
        file_path=file_path,
        mtime=mtime,
        extra={k: v for k, v in front.items() if k not in {
            "name", "description", "system_prompt",
            "model", "tools", "budget", "tool_guard",
            "context", "visibility", "permissions", "parallelism",
        }},
    )


def _scan_dir(directory: Path, *, source: str) -> Iterable[AgentRole]:
    if not directory.exists() or not directory.is_dir():
        return
    for path in sorted(directory.glob("*.md")):
        try:
            mtime = path.stat().st_mtime
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("agent_role_read_failed", path=str(path), error=str(exc))
            continue
        front, body = _split_frontmatter(text)
        role = _role_from_frontmatter(
            front, body, source=source, file_path=str(path), mtime=mtime,
        )
        if role is not None:
            yield role


class AgentRegistry:
    """Cache of available roles + change detection.

    Construct with the optional workspace path; call ``get(name)`` to
    look up a role (raises ``KeyError`` on miss after refresh).
    """

    def __init__(
        self,
        *,
        workspace_dir: str | None = None,
        user_dir: str | None = None,
        builtins: dict[str, AgentRole] | None = None,
    ) -> None:
        self._workspace_dir = Path(workspace_dir) if workspace_dir else None
        self._user_dir = Path(user_dir) if user_dir else Path.home() / ".augmentum" / "agents"
        self._builtins = dict(builtins or {})
        self._roles: dict[str, AgentRole] = {}
        self._mtimes: dict[str, float] = {}
        self._loaded = False

    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-scan all directories from scratch."""
        roles: dict[str, AgentRole] = {}
        mtimes: dict[str, float] = {}

        # 3rd precedence first so later overrides win.
        for name, role in self._builtins.items():
            roles[name] = role

        if self._user_dir:
            for role in _scan_dir(self._user_dir, source="user"):
                roles[role.name] = role
                mtimes[role.file_path] = role.mtime

        if self._workspace_dir:
            ws_agents = self._workspace_dir / ".augmentum" / "agents"
            for role in _scan_dir(ws_agents, source="workspace"):
                roles[role.name] = role
                mtimes[role.file_path] = role.mtime

        self._roles = roles
        self._mtimes = mtimes
        self._loaded = True

    def refresh_if_stale(self) -> bool:
        """Refresh only if a discovered file mtime changed.

        Returns True if anything changed (caller may want to log).
        """
        if not self._loaded:
            self.refresh()
            return True

        # Cheap mtime walk over the SAME paths we last saw.
        for path, last_mtime in list(self._mtimes.items()):
            try:
                cur = os.path.getmtime(path)
            except OSError:
                # File deleted — definitely stale.
                self.refresh()
                return True
            if cur != last_mtime:
                self.refresh()
                return True

        # New files appeared? Cheap scan: just count *.md files in each dir.
        new_seen: set[str] = set()
        if self._user_dir and self._user_dir.exists():
            new_seen |= {str(p) for p in self._user_dir.glob("*.md")}
        if self._workspace_dir:
            ws_agents = self._workspace_dir / ".augmentum" / "agents"
            if ws_agents.exists():
                new_seen |= {str(p) for p in ws_agents.glob("*.md")}

        if new_seen != set(self._mtimes.keys()):
            self.refresh()
            return True

        return False

    # ------------------------------------------------------------------

    def get(self, name: str) -> AgentRole:
        if not self._loaded:
            self.refresh()
        if name not in self._roles:
            raise KeyError(f"agent role {name!r} is not registered")
        return self._roles[name]

    def list(self) -> list[AgentRole]:
        if not self._loaded:
            self.refresh()
        return sorted(self._roles.values(), key=lambda r: (r.source != "workspace", r.name))

    def names(self) -> list[str]:
        return [r.name for r in self.list()]

    # ------------------------------------------------------------------

    def add_builtin(self, role: AgentRole) -> None:
        """Programmatic registration — used by ``presets.py`` to seed
        the built-in roster after the registry is constructed."""
        self._builtins[role.name] = replace(role, source="builtin")
        if self._loaded:
            self._roles[role.name] = self._builtins[role.name]
