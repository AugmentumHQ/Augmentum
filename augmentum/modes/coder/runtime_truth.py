"""Observed-vs-designed runtime truth for coder turns."""
from __future__ import annotations

from dataclasses import dataclass, field

import structlog

log = structlog.get_logger(__name__)

_PREBAKED_BASELINE_FAMILIES = (
    "Python + pip/venv",
    "Node + npm",
    "Go",
    "common build/search tools",
    "basic DB clients",
    "headless GUI support",
    "Python and Node dev tooling",
)
_FALLBACK_BASELINE_FAMILIES = (
    "Python + pip/venv",
    "Node + npm",
    "Go",
    "common build/search tools",
    "basic DB clients",
    "headless GUI support",
    "guide-critical Python and Node dev tooling",
)
_EXPECTED_BASELINE_RUNTIMES = ("python3", "node", "go")
_EXPECTED_BASELINE_PACKAGE_MANAGERS = ("pip", "npm")
_OPTIONAL_RUNTIME_KEYS = ("rustc", "java")
_OPTIONAL_PACKAGE_MANAGERS = ("cargo",)
_PROFILE_PACKAGE_MANAGERS = {
    "power": ("uv", "pipx", "pnpm", "yarn"),
    "browser": ("uv", "pipx", "pnpm", "yarn"),
}
_PROFILE_FAMILIES = {
    "power": (
        "process/network inspection",
        "modern Python package tooling",
        "modern JS package managers",
        "native build/debug helpers",
    ),
    "browser": (
        "Power profile extras",
        "Playwright",
        "Chromium browser runtime",
        "headless browser test dependencies",
    ),
}


def _looks_missing(value: str | None) -> bool:
    return not value or value.strip().lower() == "missing"


@dataclass(slots=True)
class RuntimeTruth:
    """Canonical per-turn environment truth used for prompt grounding."""

    workspace_mode: str = "unknown"
    workspace_image: str | None = None
    observed_runtimes: dict[str, str] = field(default_factory=dict)
    observed_package_managers: dict[str, str] = field(default_factory=dict)
    probe_succeeded: bool = False
    intended_baseline: tuple[str, ...] = _PREBAKED_BASELINE_FAMILIES
    tooling_profile: str = "standard"

    @property
    def missing_baseline(self) -> tuple[str, ...]:
        missing: list[str] = []
        for name in _EXPECTED_BASELINE_RUNTIMES:
            if _looks_missing(self.observed_runtimes.get(name)):
                missing.append(name)
        for name in _EXPECTED_BASELINE_PACKAGE_MANAGERS:
            if _looks_missing(self.observed_package_managers.get(name)):
                missing.append(name)
        return tuple(missing)

    @property
    def missing_optional(self) -> tuple[str, ...]:
        missing: list[str] = []
        for name in _OPTIONAL_RUNTIME_KEYS:
            if _looks_missing(self.observed_runtimes.get(name)):
                missing.append(name)
        for name in _OPTIONAL_PACKAGE_MANAGERS:
            if _looks_missing(self.observed_package_managers.get(name)):
                missing.append(name)
        for name in _PROFILE_PACKAGE_MANAGERS.get(self.tooling_profile, ()):
            if _looks_missing(self.observed_package_managers.get(name)):
                missing.append(name)
        return tuple(missing)

    def render_block(self) -> str:
        """Render a concise system block separating observed from intended state."""
        lines = ["<runtime_truth>", "Workspace mode:"]

        if self.workspace_mode == "prebaked":
            lines.append(
                f"- prebaked workspace image: {self.workspace_image or 'augmentum-workspace'}",
            )
            lines.append(
                "- intended runtime shape: full augmentum-workspace baseline",
            )
        elif self.workspace_mode == "fallback":
            lines.append(
                f"- fallback image: {self.workspace_image or 'ubuntu:24.04'}",
            )
            lines.append(
                "- intended runtime shape: ubuntu fallback plus guide-critical bootstrap subset",
            )
        else:
            lines.append("- workspace mode unavailable")
            lines.append(
                "- intended runtime shape: coder baseline unknown; treat the workspace guide as design intent only",
            )

        lines.append("Observed now (direct probe):")
        lines.append(f"- tooling profile: {self.tooling_profile or 'standard'}")

        if self.observed_runtimes:
            for name in ("python3", "node", "go", "rustc", "java"):
                if name in self.observed_runtimes:
                    lines.append(f"- {name}: {self.observed_runtimes[name]}")
        else:
            lines.append("- probe unavailable")

        observed_manager_lines = [
            f"- {name}: {self.observed_package_managers[name]}"
            for name in ("pip", "npm", "cargo", "uv", "pipx", "pnpm", "yarn")
            if (
                name in self.observed_package_managers
                and not _looks_missing(self.observed_package_managers[name])
            )
        ]
        if observed_manager_lines:
            lines.append("Package managers observed:")
            lines.extend(observed_manager_lines)

        if self.missing_baseline:
            lines.append(
                "Not observed now (treat as unavailable until verified or installed):",
            )
            for name in self.missing_baseline:
                lines.append(f"- {name}")

        if self.missing_optional:
            lines.append("Optional extras not observed:")
            for name in self.missing_optional:
                lines.append(f"- {name}")

        lines.extend([
            "Intended baseline:",
            (
                "- coder workspaces target "
                + ", ".join(self.intended_baseline)
                + "; ubuntu fallback bootstraps the guide-critical subset when possible"
            ),
        ])
        profile_families = _PROFILE_FAMILIES.get(self.tooling_profile)
        if profile_families:
            lines.append(
                "- selected profile target: " + ", ".join(profile_families)
            )
        lines.extend([
            (
                "- treat only the observed lines above as confirmed present right now; "
                "the workspace guide describes the intended baseline, not proof"
            ),
            (
                "- if the task depends on environment details, run env_info once for "
                "a fuller snapshot (disk, memory, project files, and versions)"
            ),
            "</runtime_truth>",
        ])
        return "\n".join(lines)


async def _detect_workspace_identity(*, handler) -> tuple[str, str | None, str]:
    """Best-effort workspace mode detection from the live container image."""
    cm = handler._container_manager
    if cm is None:
        return "unknown", None, "standard"

    get_workspace = getattr(cm, "_get_workspace", None)
    docker = getattr(cm, "_docker", None)
    if get_workspace is None or docker is None:
        return "unknown", None, "standard"

    try:
        info = await get_workspace(handler._workspace_id)
        tooling_profile = getattr(info, "tooling_profile", None) or "standard"
        container_id = getattr(info, "container_id", None)
        if not container_id:
            return "unknown", None, tooling_profile
        container = await docker.containers.get(container_id)
        details = await container.show()
        image = ((details.get("Config") or {}).get("Image") or "").strip() or None
        if image == "ubuntu:24.04":
            return "fallback", image, tooling_profile
        if image:
            return "prebaked", image, tooling_profile
    except Exception:
        log.debug("coder_runtime_truth_mode_probe_failed", exc_info=True)

    return "unknown", None, "standard"


async def build_runtime_truth(*, handler) -> RuntimeTruth:
    """Probe the current workspace runtime without paying for full env_info."""
    if handler._container_manager is None:
        return RuntimeTruth()

    workspace_mode, workspace_image, tooling_profile = await _detect_workspace_identity(
        handler=handler,
    )

    script = r"""
printf 'python3\t'; (python3 --version 2>/dev/null || echo "missing")
printf 'node\t'; (node --version 2>/dev/null || echo "missing")
printf 'go\t'; (go version 2>/dev/null || echo "missing")
printf 'rustc\t'; (rustc --version 2>/dev/null || echo "missing")
printf 'java\t'; ((java -version 2>&1 | head -1) || echo "missing")
printf 'pip\t'; (pip --version 2>/dev/null || echo "missing")
printf 'npm\t'; (npm --version 2>/dev/null || echo "missing")
printf 'cargo\t'; (cargo --version 2>/dev/null || echo "missing")
printf 'uv\t'; (uv --version 2>/dev/null || echo "missing")
printf 'pipx\t'; (pipx --version 2>/dev/null || echo "missing")
printf 'pnpm\t'; (pnpm --version 2>/dev/null || echo "missing")
printf 'yarn\t'; (yarn --version 2>/dev/null || echo "missing")
"""
    try:
        output = await handler._container_manager._run_command(
            handler._workspace_id,
            ["bash", "-c", script],
            timeout=5.0,
        )
    except Exception:
        log.debug("coder_runtime_truth_probe_failed", exc_info=True)
        return RuntimeTruth()

    observed_runtimes: dict[str, str] = {}
    observed_package_managers: dict[str, str] = {}
    for raw_line in (output or "").splitlines():
        if "\t" not in raw_line:
            continue
        key, value = raw_line.split("\t", 1)
        key = key.strip().lower()
        value = value.strip() or "missing"
        if key in {"python3", "node", "go", "rustc", "java"}:
            observed_runtimes[key] = value
        elif key in {"pip", "npm", "cargo", "uv", "pipx", "pnpm", "yarn"}:
            observed_package_managers[key] = value

    return RuntimeTruth(
        workspace_mode=workspace_mode,
        workspace_image=workspace_image,
        observed_runtimes=observed_runtimes,
        observed_package_managers=observed_package_managers,
        probe_succeeded=bool(observed_runtimes or observed_package_managers),
        intended_baseline=(
            _FALLBACK_BASELINE_FAMILIES
            if workspace_mode == "fallback"
            else _PREBAKED_BASELINE_FAMILIES
        ),
        tooling_profile=tooling_profile,
    )
