"""Data models for Coder mode."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContainerInfo:
    """Workspace container metadata."""
    id: str
    name: str
    container_id: str | None = None
    status: str = "stopped"
    template_id: str | None = None
    git_url: str | None = None
    created_at: float = 0.0
    last_active: float | None = None
    resources_cpu: float = 2.0
    resources_memory: str = "2g"
    tooling_profile: str = "browser"
    # Owning tenant. Set at create time so the INSERT in
    # ContainerManager._persist_workspace stamps it atomically. Empty when
    # the row predates the multi-tenant rollout.
    user_id: str = ""
    # Per-workspace toggle for the hybrid-loop soft circuit-breakers
    # (action_stagnation, test_failure_streak, same_file_edit, etc.).
    # True (default) preserves existing behavior; False bypasses the
    # soft breakers for strong models that legitimately run long. The
    # hard iteration ceiling stays in place either way.
    safeguards_enabled: bool = True
    # Workspace type — ``regular`` (default) or ``bug_finder``. Chosen
    # at creation, fixed for the life of the workspace. Bug Finder
    # workspaces surface an additional workbench tab in coder for
    # autonomous audit runs; regular workspaces look identical to
    # today.
    kind: str = "regular"
    # Optional per-workspace verifier model override for Bug Finder
    # audits. Empty string means single-model self-verification (the
    # local-hardware default). When set, this model is used for the
    # verifier role instead of the user's currently-selected model.
    bug_finder_verifier_model: str = ""
    # Owning Project ID (migration 199/200). Empty for legacy rows that
    # predate Phase 1's bare-repo substrate; new checkouts always link
    # to a Project. The Project owns the durable bare repo at
    # ``{data_dir}/projects/{user_id}/{project_id}.git/`` from which
    # this checkout was cloned.
    project_id: str = ""
    # Per-workspace coder planning mode (migration 207 + 208).
    # Cycles via Shift+Tab in the composer; persists across restarts.
    #
    #   "auto" (DEFAULT) — model runs freely, no permission modals.
    #                      The "trust the model" mode used by Cursor
    #                      Composer / Aider out of the box. Operator
    #                      policy rules in .augmentum/permissions.toml
    #                      still apply for explicit deny entries.
    #   "default"        — per-tool permission prompts on mutations.
    #                      The original safety mode; UI labels this
    #                      as "approve" but the string is kept for
    #                      back-compat with persisted rows.
    #   "plan"           — soft planning guidance. Strong system-prompt
    #                      addendum nudges the model to outline its
    #                      approach before editing. Tools remain
    #                      available; the model decides when to
    #                      propose-and-wait vs proceed. Hard tool
    #                      filtering retired post-208 — natural-mode
    #                      collaboration over forced read-only.
    planning_mode: str = "auto"
    # Container lifecycle policy (migration 211).
    #   False (DEFAULT for new rows) — workspace participates in the
    #     idle reaper. Container auto-stops after CODER_IDLE_TIMEOUT
    #     seconds of no activity. DB row + volume survive; user can
    #     restart instantly. Right for transient "try a thing"
    #     workspaces.
    #   True — workspace is exempt from the reaper. Container stays
    #     running across browser sessions. Use for workspaces hosting
    #     a dev server / daemon / long-running test harness the user
    #     wants to keep up while they step away. Still disposable
    #     via the explicit "Stop now" button or delete.
    # Backfill flipped every pre-migration row to True so legacy
    # workspaces preserve current "never auto-stop" behavior.
    always_on: bool = False
    # LAN accessibility (migration 303).
    #   False (DEFAULT) — published ports bind to 127.0.0.1 (loopback).
    #   True — published ports bind to 0.0.0.0 (LAN-reachable). When
    #     gate_domain is configured, listening ports also get a Caddy
    #     reverse-proxy snippet at <workspace-slug>.<gate_domain> with
    #     HTTPS + Augmentum auth. Deliberate user action only.
    lan_accessible: bool = False


@dataclass
class FileEntry:
    """File or directory in a workspace."""
    name: str
    path: str
    is_dir: bool
    size: int = 0
    modified: float = 0.0


@dataclass
class WorkspaceConfig:
    """Configuration for creating a workspace."""
    name: str
    base_image: str = "augmentum-workspace"
    tooling_profile: str = "browser"
    packages: list[str] = field(default_factory=list)
    git_url: str | None = None
    cpu: float = 2.0
    memory: str = "2g"
    pids: int = 256
    # Owning Project ID. Empty string means "auto-create a new Project
    # for this checkout"; non-empty links the checkout to an existing
    # Project (and clones its bare repo into the workspace volume).
    project_id: str = ""
