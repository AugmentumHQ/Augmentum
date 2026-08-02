"""Registry and compatibility normalization for Augmentum Powers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from augmentum.powers.manifest import discover_manifest_file, parse_power_manifest
from augmentum.powers.models import PowerHealth, PowerManifest
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class PowerRegistry:
    """Discover and normalize repo-local Power packages."""

    def __init__(self, *, search_roots: list[tuple[Path, str]] | None = None) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        if search_roots is None:
            from augmentum.config import settings as _settings

            data_dir = Path(getattr(_settings, "data_dir", "/data"))
            search_roots = [
                (repo_root / ".augmentum" / "powers", "native"),
                (repo_root / ".claude" / "skills", "compat"),
                # Community-installed powers — dropped here by
                # augmentum/proxy/community_routes.py after the user
                # confirms an "Open in Augmentum" install. Lives under
                # the data dir so it persists across container restarts
                # and survives image rebuilds. Source-kind "community"
                # lets the UI distinguish community packs from shipped
                # ones if it wants to surface a badge.
                (data_dir / "community-powers", "community"),
            ]
        self._search_roots = search_roots
        self._powers: dict[str, PowerManifest] = {}
        self.rescan()

    def rescan(self) -> dict[str, PowerManifest]:
        found: dict[str, PowerManifest] = {}
        for root, source_kind in self._search_roots:
            if not root.exists() or not root.is_dir():
                continue
            for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
                if not child.is_dir():
                    continue
                manifest_path = discover_manifest_file(child)
                if manifest_path is None:
                    continue
                try:
                    manifest = parse_power_manifest(manifest_path, source_kind=source_kind)
                except Exception as exc:
                    log.warning(
                        "power_manifest_parse_failed",
                        path=str(manifest_path),
                        error=str(exc),
                    )
                    continue
                if manifest.id in found:
                    log.info(
                        "power_manifest_shadowed",
                        power_id=manifest.id,
                        kept=str(found[manifest.id].manifest_path),
                        skipped=str(manifest.manifest_path),
                    )
                    continue
                found[manifest.id] = manifest
        self._powers = found
        return dict(self._powers)

    def list_powers(self) -> list[PowerManifest]:
        return sorted(self._powers.values(), key=lambda item: item.display_name.lower())

    def get_power(self, power_id: str) -> PowerManifest | None:
        return self._powers.get(power_id)

    def evaluate_health(
        self,
        manifest: PowerManifest,
        *,
        mcp_client=None,
        tool_registry=None,
    ) -> PowerHealth:
        missing_bins = [name for name in manifest.required_bins if not shutil.which(name)]
        missing_env = [name for name in manifest.required_env if not os.environ.get(name)]
        connected_servers = set(getattr(mcp_client, "connected_servers", []) or [])
        missing_mcp = [
            name for name in manifest.required_mcp_servers
            if name and name not in connected_servers
        ]
        missing_tools: list[str] = []
        if tool_registry is not None:
            for name in manifest.required_tools:
                if not tool_registry.get(name):
                    missing_tools.append(name)
        reasons: list[str] = []
        if missing_bins:
            reasons.append(f"Missing binaries: {', '.join(missing_bins)}")
        if missing_env:
            reasons.append(f"Missing env vars: {', '.join(missing_env)}")
        if missing_mcp:
            reasons.append(f"MCP not connected: {', '.join(missing_mcp)}")
        if missing_tools:
            reasons.append(f"Missing tools: {', '.join(missing_tools)}")
        return PowerHealth(
            status="needs_setup" if reasons else "ready",
            blocked_reasons=reasons,
            missing_bins=missing_bins,
            missing_env=missing_env,
            missing_mcp_servers=missing_mcp,
            missing_tools=missing_tools,
        )
