"""Filesystem manifest parsing for Augmentum Powers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from augmentum.powers.models import PowerManifest

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_H1_RE = re.compile(r"^#\s+([^\n]+)", re.MULTILINE)


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return value.strip().strip("\"'")


def _parse_frontmatter_block(block: str) -> dict[str, Any]:
    """Parse a small YAML-like frontmatter subset without PyYAML."""
    data: dict[str, Any] = {}
    lines = block.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in line:
            i += 1
            continue
        if line.startswith((" ", "\t")):
            i += 1
            continue
        key, _, raw_value = line.partition(":")
        key = key.strip()
        value = raw_value.strip()
        if value in {">", "|"}:
            j = i + 1
            block_lines: list[str] = []
            while j < len(lines) and (lines[j].startswith((" ", "\t")) or not lines[j].strip()):
                child = lines[j].strip()
                if child:
                    block_lines.append(child)
                j += 1
            data[key] = (" ".join(block_lines) if value == ">" else "\n".join(block_lines)).strip()
            i = j
            continue
        if value == "":
            j = i + 1
            items: list[str] = []
            while j < len(lines) and (lines[j].startswith((" ", "\t")) or not lines[j].strip()):
                child = lines[j].strip()
                if child.startswith("- "):
                    items.append(child[2:].strip())
                elif child:
                    items.append(child)
                j += 1
            data[key] = items if items else ""
            i = j
            continue
        data[key] = _coerce_scalar(value)
        i += 1
    return data


def _humanize_slug(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.replace("_", "-").split("-") if part)


def _clean_markdown_excerpt(text: str) -> str:
    text = re.sub(r"^#.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_`>#-]{1,3}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _first_paragraph(text: str) -> str:
    for chunk in re.split(r"\n\s*\n", text or ""):
        stripped = chunk.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cleaned = _clean_markdown_excerpt(stripped)
        if cleaned:
            return cleaned[:800]
    return ""


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[\n,]", value)
        return [part.strip() for part in parts if part.strip()]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                if name:
                    out.append(name)
        return out
    if isinstance(value, dict):
        return [str(k).strip() for k, enabled in value.items() if enabled and str(k).strip()]
    return []


def _pick(meta: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in meta:
            return meta[key]
    return None


def _normalize_modes(raw: Any) -> list[str]:
    modes = [m.lower() for m in _normalize_list(raw)]
    if not modes:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for mode in modes:
        if mode in seen:
            continue
        seen.add(mode)
        out.append(mode)
    return out


def _normalize_choice(raw: Any, *, allowed: set[str], default: str) -> str:
    value = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    return value if value in allowed else default


def _normalize_invocation(meta: dict[str, Any], triggers: list[str]) -> str:
    user_invocable = _pick(meta, "user-invocable", "user_invocable")
    manual = True if user_invocable is None else bool(user_invocable)
    auto = bool(triggers or _pick(meta, "auto", "auto_invoke", "auto-invoke"))
    if manual and auto:
        return "hybrid"
    if auto:
        return "auto"
    return "manual"


def _normalize_kind(meta: dict[str, Any]) -> str:
    return _normalize_choice(
        _pick(meta, "kind", "power_kind", "power-kind"),
        allowed={"guidance", "verifier", "workflow", "integration", "bridge"},
        default="guidance",
    )


def _normalize_activation_policy(meta: dict[str, Any], *, kind: str) -> str:
    explicit_only = _pick(meta, "explicit_only", "explicit-only")
    if explicit_only is True:
        return "explicit_only"
    raw = _pick(meta, "activation_policy", "activation-policy", "activation")
    if raw is not None:
        return _normalize_choice(
            raw,
            allowed={"manual", "controller", "model_request", "explicit_only"},
            default="manual",
        )
    if kind == "bridge":
        return "explicit_only"
    return "manual"


def _normalize_activation_windows(raw: Any) -> list[str]:
    allowed = {
        "pre_plan",
        "implementation",
        "post_write",
        "verify_failed",
        "pre_finish",
    }
    out: list[str] = []
    seen: set[str] = set()
    for item in _normalize_list(raw):
        normalized = item.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in allowed and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def discover_manifest_file(package_dir: Path) -> Path | None:
    """Return the preferred manifest file for a package directory."""
    for candidate in ("POWER.md", "SKILL.md"):
        path = package_dir / candidate
        if path.exists():
            return path
    return None


def parse_power_manifest(manifest_path: Path, *, source_kind: str) -> PowerManifest:
    """Parse a filesystem Power/Skill manifest into a normalized model."""
    raw = manifest_path.read_text(encoding="utf-8")
    frontmatter: dict[str, Any] = {}
    body = raw
    match = _FRONTMATTER_RE.match(raw)
    if match:
        frontmatter = _parse_frontmatter_block(match.group(1))
        body = raw[match.end():]

    heading = _H1_RE.search(body)
    slug = manifest_path.parent.name
    display_name = (
        str(_pick(frontmatter, "display_name", "name") or "").strip()
        or (heading.group(1).strip() if heading else "")
        or _humanize_slug(slug)
    )
    description = (
        str(_pick(frontmatter, "description", "summary") or "").strip()
        or _first_paragraph(body)
    )

    kind = _normalize_kind(frontmatter)
    activation_policy = _normalize_activation_policy(frontmatter, kind=kind)
    activation_windows = _normalize_activation_windows(
        _pick(
            frontmatter,
            "activation_windows",
            "activation-windows",
            "windows",
            "checkpoints",
        ),
    )
    triggers = _normalize_list(
        _pick(frontmatter, "triggers", "trigger", "trigger_pattern", "trigger-pattern"),
    )
    mode_scope = _normalize_modes(
        _pick(frontmatter, "modes", "mode_scope", "mode-scope", "mode"),
    )
    preferred_tools = _normalize_list(
        _pick(
            frontmatter,
            "preferred_tools",
            "preferred-tools",
            "recommended_tools",
            "recommended-tools",
        ),
    )
    blocked_tools = _normalize_list(
        _pick(
            frontmatter,
            "blocked_tools",
            "blocked-tools",
            "disallowed_tools",
            "disallowed-tools",
        ),
    )
    required_mcp_servers = _normalize_list(
        _pick(
            frontmatter,
            "required_mcp_servers",
            "required-mcp-servers",
            "mcp_servers",
            "mcp-servers",
            "mcp",
        ),
    )
    required_tools = _normalize_list(
        _pick(frontmatter, "required_tools", "required-tools"),
    )
    required_bins = _normalize_list(
        _pick(frontmatter, "required_bins", "required-bins", "bins", "binaries"),
    )
    required_env = _normalize_list(
        _pick(frontmatter, "required_env", "required-env", "env", "env_vars", "env-vars"),
    )
    tags = _normalize_list(_pick(frontmatter, "tags"))
    invocation = _normalize_invocation(frontmatter, triggers)

    excerpt = body.strip()
    return PowerManifest(
        id=slug,
        slug=slug,
        display_name=display_name,
        description=description[:600],
        source_kind=source_kind,
        package_dir=manifest_path.parent,
        manifest_path=manifest_path,
        manifest_name=manifest_path.name,
        body_markdown=body.strip(),
        instruction_excerpt=excerpt[:8000],
        kind=kind,
        activation_policy=activation_policy,
        activation_windows=activation_windows,
        mode_scope=mode_scope,
        invocation=invocation,
        triggers=triggers,
        preferred_tools=preferred_tools,
        blocked_tools=blocked_tools,
        required_mcp_servers=required_mcp_servers,
        required_tools=required_tools,
        required_bins=required_bins,
        required_env=required_env,
        tags=tags,
        emoji=str(_pick(frontmatter, "emoji") or "◆").strip() or "◆",
        homepage=str(_pick(frontmatter, "homepage", "website") or "").strip(),
        metadata=frontmatter,
    )
