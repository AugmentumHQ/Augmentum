"""Normalized runtime models for Augmentum Powers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


@dataclass(slots=True)
class PowerHealth:
    """Resolved dependency status for a Power in the current runtime."""

    status: str
    blocked_reasons: list[str] = field(default_factory=list)
    missing_bins: list[str] = field(default_factory=list)
    missing_env: list[str] = field(default_factory=list)
    missing_mcp_servers: list[str] = field(default_factory=list)
    missing_tools: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "blocked_reasons": list(self.blocked_reasons),
            "missing_bins": list(self.missing_bins),
            "missing_env": list(self.missing_env),
            "missing_mcp_servers": list(self.missing_mcp_servers),
            "missing_tools": list(self.missing_tools),
        }


@dataclass(slots=True)
class PowerActivation:
    """Per-workspace active Power selection."""

    power_id: str
    workspace_id: str
    source: str = "manual"
    scope: str = "workspace"
    reason: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "power_id": self.power_id,
            "workspace_id": self.workspace_id,
            "source": self.source,
            "scope": self.scope,
            "reason": self.reason,
        }


@dataclass(slots=True)
class PowerManifest:
    """Normalized filesystem-backed Power definition."""

    id: str
    slug: str
    display_name: str
    description: str
    source_kind: str
    package_dir: Path
    manifest_path: Path
    manifest_name: str
    body_markdown: str
    instruction_excerpt: str
    kind: str = "guidance"
    activation_policy: str = "manual"
    activation_windows: list[str] = field(default_factory=list)
    mode_scope: list[str] = field(default_factory=list)
    invocation: str = "manual"
    triggers: list[str] = field(default_factory=list)
    preferred_tools: list[str] = field(default_factory=list)
    blocked_tools: list[str] = field(default_factory=list)
    required_mcp_servers: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    required_bins: list[str] = field(default_factory=list)
    required_env: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    emoji: str = "◆"
    homepage: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def relative_files(self) -> list[str]:
        files: list[str] = []
        for path in sorted(self.package_dir.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            files.append(path.name + ("/" if path.is_dir() else ""))
        return files

    def render_prompt_block(self, *, max_chars: int = 3200) -> str:
        """Return the system-prompt block injected into coder mode."""
        tool_bias = []
        if self.preferred_tools:
            tool_bias.append(f"Prefer tools: {', '.join(self.preferred_tools[:8])}")
        if self.blocked_tools:
            tool_bias.append(f"Avoid tools: {', '.join(self.blocked_tools[:8])}")
        if self.required_mcp_servers:
            tool_bias.append(
                f"Related MCP servers: {', '.join(self.required_mcp_servers[:6])}",
            )
        guidance = _clip(self.instruction_excerpt or self.body_markdown, max_chars)
        parts = [
            f'<active_power id="{self.id}" source="{self.source_kind}" invocation="{self.invocation}">',
            f"Power: {self.display_name}",
        ]
        if self.description:
            parts.append(f"Summary: {self.description}")
        parts.append(f"Kind: {self.kind}")
        if self.mode_scope:
            parts.append(f"Modes: {', '.join(self.mode_scope)}")
        parts.extend(tool_bias)
        if guidance:
            parts.append("Guidance:")
            parts.append(guidance)
        parts.append(
            "Use this capability pack when it matches the current task. "
            "Do not force it when it is irrelevant.",
        )
        parts.append("</active_power>")
        return "\n".join(parts)

    def to_summary_dict(
        self,
        *,
        enabled: bool,
        health: PowerHealth,
        active: PowerActivation | None = None,
    ) -> dict[str, Any]:
        status = "disabled" if not enabled else health.status
        return {
            "id": self.id,
            "slug": self.slug,
            "display_name": self.display_name,
            "description": self.description,
            "source_kind": self.source_kind,
            "source_label": "Native" if self.source_kind == "native" else "Compatibility",
            "manifest_name": self.manifest_name,
            "kind": self.kind,
            "activation_policy": self.activation_policy,
            "activation_windows": list(self.activation_windows),
            "modes": list(self.mode_scope),
            "invocation": self.invocation,
            "triggers": list(self.triggers),
            "preferred_tools": list(self.preferred_tools),
            "blocked_tools": list(self.blocked_tools),
            "required_mcp_servers": list(self.required_mcp_servers),
            "required_tools": list(self.required_tools),
            "required_bins": list(self.required_bins),
            "required_env": list(self.required_env),
            "tags": list(self.tags),
            "emoji": self.emoji,
            "homepage": self.homepage,
            "status": status,
            "enabled": enabled,
            "active": bool(active),
            "active_scope": active.scope if active else "",
            "active_source": active.source if active else "",
            "active_reason": active.reason if active else "",
            "files": self.relative_files(),
            "instruction_excerpt": _clip(self.instruction_excerpt, 1600),
            "health": health.to_dict(),
        }

    def to_detail_dict(
        self,
        *,
        enabled: bool,
        health: PowerHealth,
        active: PowerActivation | None = None,
    ) -> dict[str, Any]:
        data = self.to_summary_dict(enabled=enabled, health=health, active=active)
        data.update(
            {
                "manifest_path": str(self.manifest_path),
                "package_dir": str(self.package_dir),
                "body_markdown": self.body_markdown,
                "metadata": dict(self.metadata),
            },
        )
        return data
