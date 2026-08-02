"""Subagent role offer catalog.

Each entry corresponds to a built-in role from
``augmentum/agents/presets.py::BUILTIN_ROLES``. "Install" semantics:
write the role's frontmatter + system prompt to
``~/.augmentum/agents/<name>.md`` so the user has an editable copy
they can customize. The built-in stays as the fallback if the file
is missing or unparseable.

Why per-user files instead of workspace-local: the offer can come
from any surface (chat / coder / companion). There's no guaranteed
workspace at offer time. ``~/.augmentum/agents`` is the
``AgentRegistry`` user-global location — picked up by every
workspace automatically.

Admin-scope: the user-global dir is shared across the deployment,
so a regular user can't write into it. Solo deployments (one user)
behave the same as admin from this offer's perspective.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from augmentum.agents.presets import BUILTIN_ROLES
from augmentum.offers.catalog.base import (
    CatalogEntry,
    OfferPreview,
    register_kind,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request

    from augmentum.agents.spec import AgentRole


log = get_logger(__name__)


KIND: str = "subagent_role"


def _user_agents_dir() -> Path:
    return Path.home() / ".augmentum" / "agents"


def _role_file_path(name: str) -> Path:
    return _user_agents_dir() / f"{name}.md"


def _role_frontmatter(role: AgentRole) -> str:
    """Serialize the role's key fields as YAML frontmatter.

    Built deliberately by hand rather than via PyYAML — the role
    file's expected shape is well-known and we want consistent
    formatting users can read + diff easily. Anything we don't
    serialize (defaults) stays at the registry's defaults.
    """

    lines: list[str] = ["---", f"name: {role.name}"]
    if role.description:
        # Multi-line description as a single quoted string — YAML
        # tolerates this without further escaping for the descriptions
        # the built-ins ship with.
        desc = role.description.replace('"', '\\"')
        lines.append(f'description: "{desc}"')
    if role.preferred_model:
        lines.append("model:")
        lines.append(f"  preferred: {role.preferred_model}")
        if role.fallback_models:
            lines.append("  fallbacks:")
            for m in role.fallback_models:
                lines.append(f"    - {m}")
    if role.tools:
        lines.append("tools:")
        for tool in role.tools:
            lines.append(f"  - {tool}")
    if role.context_mode and role.context_mode != "workspace":
        lines.append(f"context: {role.context_mode}")
    if role.can_spawn_subagents:
        lines.append(f"can_spawn_subagents: {str(role.can_spawn_subagents).lower()}")
    if role.max_concurrent:
        lines.append(f"max_concurrent: {role.max_concurrent}")
    budget_lines: list[str] = []
    b = role.budget
    if b:
        if b.max_iterations and b.max_iterations != 30:
            budget_lines.append(f"  max_iterations: {b.max_iterations}")
        if b.max_wallclock_seconds and b.max_wallclock_seconds != 180:
            budget_lines.append(f"  max_wallclock_seconds: {b.max_wallclock_seconds}")
        if b.max_tokens:
            budget_lines.append(f"  max_tokens: {b.max_tokens}")
    if budget_lines:
        lines.append("budget:")
        lines.extend(budget_lines)
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _make_entry(role: AgentRole) -> CatalogEntry:
    name = role.name
    description = role.description or ""

    async def _preview(_target_id: str, _user_id: str) -> OfferPreview | None:
        path = _role_file_path(name)
        if path.exists():
            return None  # already installed
        return OfferPreview(
            label=f"{name} (built-in subagent role)",
            hint=description[:160] + ("…" if len(description) > 160 else ""),
            details={
                "scope": "admin",
                "name": name,
                "tools": list(role.tools)[:6],
                "context_mode": role.context_mode,
                "writes_to": str(_role_file_path(name)),
            },
        )

    async def _accept(payload: dict[str, Any], request: Request) -> dict[str, Any]:
        path = _role_file_path(name)
        if path.exists():
            return {
                "ok": True,
                "already_installed": True,
                "path": str(path),
                "next_step": (
                    f"Edit {path} to customize the {name} role. "
                    "Augmentum reloads on file change."
                ),
            }

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            content = _role_frontmatter(role) + (role.system_prompt or "")
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return {
                "ok": False,
                "error": "write_failed",
                "detail": str(exc)[:200],
            }

        log.info("offer_subagent_role_cloned", role=name, path=str(path))
        return {
            "ok": True,
            "role": name,
            "path": str(path),
            "next_step": (
                f"Editable copy written to {path}. Open it to tweak "
                f"the system prompt, tool allowlist, or budget."
            ),
        }

    return CatalogEntry(
        kind=KIND,
        target_id=name,
        title=f"Customize the {name} subagent role?",
        scope="admin",
        build_preview=_preview,
        accept=_accept,
        # Subagent role files are consumed by coder's task_dispatch
        # tool. Offering from elsewhere creates a file the user can
        # accept but never see take effect.
        allowed_modes=("coder",),
    )


ENTRIES: list[CatalogEntry] = [_make_entry(r) for r in BUILTIN_ROLES.values()]


if ENTRIES:
    register_kind(KIND, ENTRIES)
