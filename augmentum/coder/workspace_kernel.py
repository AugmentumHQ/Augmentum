"""Workspace-kernel substrate for coder mode.

The kernel maintains ``/workspace/.augmentum/`` — a scratch directory
the model can read on demand instead of having content re-framed into
the message stream every iteration. See the design doc at
``docs/superpowers/specs/2026-05-16-workspace-kernel-design.md`` for
the full rationale (two-kind scaffolding rule, tier-conditional
writes, quiet annotations).

This module ships the layout + writer surface. Per-file population
lands incrementally — the first migration (this PR) only ensures the
directory exists; later PRs add ``recent_failures.md``, ``world.md``,
``recall/``, ``profile.md``, etc. as their corresponding sticky-reminder
sections are migrated out of the message stream.

Why a single module rather than per-file modules
------------------------------------------------
Each ``.augmentum/`` file is a thin view over state the coder already
maintains (turn summaries, failure ledger, workspace snapshot, profile
store). A single ``WorkspaceKernel`` knows how to render any of them,
so future migrations add a render method here rather than a new
module + new wiring per file.

Design contract
---------------
* **Best-effort.** Every kernel call swallows exceptions and logs at
  debug. A kernel failure must never block a turn — the model's
  contract is "files might be there", not "files are guaranteed".
* **Tier-conditional.** Effort scales with task tier. REFLEX writes
  nothing; PROJECT writes the full set. The tier classifier (Phase 1)
  is the source of truth.
* **Idempotent.** Repeated calls produce the same files. Safe to call
  every turn-start.
* **Read paths return ``""`` on miss.** Callers (sticky reminder,
  annotations) can treat empty as "nothing to surface" without a
  separate existence check.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.modes.coder.intent import Tier
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.coder.containers import ContainerManager

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Canonical paths — single source of truth for every kernel-managed file.
# Used by both the kernel writer and the readers in handler.py /
# phase_act.py so a path change is one edit.
# ---------------------------------------------------------------------------

KERNEL_ROOT = "/workspace/.augmentum"

PLAN_MD = f"{KERNEL_ROOT}/plan.md"
RECENT_FAILURES_MD = f"{KERNEL_ROOT}/recent_failures.md"
WORLD_MD = f"{KERNEL_ROOT}/world.md"
PROFILE_MD = f"{KERNEL_ROOT}/profile.md"
IDENTITY_TOML = f"{KERNEL_ROOT}/identity.toml"
OBJECTIVE_MD = f"{KERNEL_ROOT}/objective.md"
OBSERVATIONS_JSONL = f"{KERNEL_ROOT}/observations.jsonl"
RECALL_DIR = f"{KERNEL_ROOT}/recall"
SCRATCH_DIR = f"{KERNEL_ROOT}/scratch"
TRACE_DIR = f"{KERNEL_ROOT}/trace"


# Minimum length for auto-seeding objective.md from a user message.
# Below this we assume the message is conversational ("hi", "thanks",
# "continue") rather than a real ask. The kernel still surfaces an
# empty objective gracefully — no seed isn't catastrophic.
#
# 2026-05-31: lowered from 30 → 12. Real user asks like "fix the bug",
# "make this faster", "add tests" land in the 10-20 char range; the
# previous 30-char floor was rejecting half of them as conversational.
# The trade-off: "ok continue" / "thanks" still skip (under 12), but
# "fix bug" / "add tests" / "deploy this" now seed correctly.
_OBJECTIVE_SEED_MIN_CHARS = 12


# Tier → set of files the kernel maintains. REFLEX intentionally
# empty: a one-shot edit doesn't justify any kernel I/O. PROJECT gets
# the full set including recall (Phase 5) and profile (Phase 7) once
# those migrations land.
#
# IDENTITY_TOML (2026-05-28) is included at SURGICAL+ because identity
# detection is cheap (a handful of file existence probes + small file
# reads) and the manifest is what gives the agent project orientation
# without a discovery round-trip. REFLEX still skips — a one-shot
# edit doesn't need the manifest.
#
# OBSERVATIONS_JSONL (2026-05-28) is COMPOSED+ — durable cross-session
# memory only pays off on multi-step / multi-session work. SURGICAL
# tasks don't accumulate enough learnings to justify the ledger. The
# file is never *written* by refresh (it's append-only via the
# ``observe`` tool); inclusion here just declares "this is part of
# the kernel surface at this tier" for visibility audit purposes.
_FILES_BY_TIER: dict[Tier, frozenset[str]] = {
    Tier.REFLEX:   frozenset(),
    Tier.SURGICAL: frozenset({PLAN_MD, IDENTITY_TOML, OBJECTIVE_MD}),
    Tier.COMPOSED: frozenset({
        PLAN_MD, RECENT_FAILURES_MD, WORLD_MD, IDENTITY_TOML,
        OBJECTIVE_MD, OBSERVATIONS_JSONL,
    }),
    Tier.PROJECT:  frozenset({
        PLAN_MD, RECENT_FAILURES_MD, WORLD_MD, PROFILE_MD, IDENTITY_TOML,
        OBJECTIVE_MD, OBSERVATIONS_JSONL,
    }),
}


class WorkspaceKernel:
    """Maintains ``/workspace/.augmentum/`` for the coder agent.

    One instance per (workspace, handler). Cheap to construct — no
    I/O at init. Construct, then call :meth:`refresh` at turn-start
    with the resolved tier + current state.

    Read methods are stateless conveniences; the kernel writes files
    but the model is the primary author of ``plan.md``, so reads must
    not assume the kernel wrote what's there.
    """

    def __init__(
        self,
        container_manager: ContainerManager | None,
        workspace_id: str,
    ) -> None:
        self._container_manager = container_manager
        self._workspace_id = workspace_id

    # ── Public API ────────────────────────────────────────────────────

    def files_for_tier(self, tier: Tier) -> frozenset[str]:
        """Return the canonical-path set the kernel maintains for ``tier``.

        Used by tests + the migration code that needs to know which
        sticky-reminder sections to suppress at a given tier.
        """
        return _FILES_BY_TIER.get(tier, frozenset())

    async def refresh(self, *, tier: Tier) -> None:
        """Ensure ``.augmentum/`` exists and any tier-relevant files
        are present. Per-file population lands in subsequent
        migrations; this initial slice only guarantees the directory.

        REFLEX skips everything — no directory creation, no writes.
        The model on a REFLEX task is doing one edit and stopping;
        creating a scratch dir for it is wasted I/O.
        """
        if self._container_manager is None or not self._workspace_id:
            return
        if tier == Tier.REFLEX:
            return
        # mkdir -p so repeated calls are no-ops. Subdirs (recall/,
        # scratch/, trace/) are created lazily by the migrations that
        # need them — Surgical/Composed don't write to them so creating
        # them here would be premature.
        try:
            await self._container_manager._run_command(
                self._workspace_id, ["mkdir", "-p", KERNEL_ROOT],
            )
        except Exception:
            log.debug(
                "workspace_kernel_mkdir_failed",
                workspace_id=self._workspace_id,
                exc_info=True,
            )

        # Refresh tier-relevant files. Identity is included from
        # SURGICAL up — detection is cheap and the manifest is what
        # gives the model project orientation. Per-file population for
        # other files (recent_failures.md, world.md, profile.md) is
        # not yet wired here — those land in later migrations.
        files = _FILES_BY_TIER.get(tier, frozenset())
        if IDENTITY_TOML in files:
            await self.refresh_identity()

    async def read_plan(self) -> str:
        """Best-effort read of ``plan.md``. ``""`` on any miss/error.

        Mirrors the pre-v2 ``CoderHandler._read_plan_md`` contract so
        the migration in ``phase_act.py`` can switch source modules
        without changing call-site behavior.
        """
        return await self._read(PLAN_MD)

    async def refresh_identity(self) -> None:
        """Detect project identity facts and persist to ``identity.toml``.

        Three-section merge contract (see :mod:`augmentum.coder.identity`):

        * ``[detected]`` is replaced wholesale by current detector output.
        * ``[asserted]`` (user-edited) survives untouched.
        * ``[discovered]`` (model-appended) survives untouched.

        Best-effort: any failure (container down, detector crash,
        write error) is logged at debug and swallowed. The agent's
        contract is "the file might be there" — never "the file is
        guaranteed."

        Idempotent: called every turn-start at SURGICAL+ tiers. The
        cost is a handful of file_read probes + one file_write —
        cheap enough to redo each turn so the manifest reflects
        recently-added project files (e.g. a new package.json that
        appeared since the last turn).
        """
        if self._container_manager is None or not self._workspace_id:
            return
        try:
            from augmentum.coder.identity import (
                detect_identity,
                merge_refresh,
                parse_manifest,
                serialize_manifest,
            )

            # Read existing (if any) so asserted + discovered are
            # preserved across refresh cycles.
            existing_text = await self._read(IDENTITY_TOML)
            existing = parse_manifest(existing_text) if existing_text else None

            fresh = await detect_identity(
                self._container_manager, self._workspace_id,
            )

            if existing is not None:
                merged = merge_refresh(existing, fresh.detected)
            else:
                merged = fresh

            await self._container_manager.file_write(
                self._workspace_id,
                IDENTITY_TOML,
                serialize_manifest(merged),
            )
        except Exception:
            log.debug(
                "workspace_kernel_identity_refresh_failed",
                workspace_id=self._workspace_id,
                exc_info=True,
            )

    async def read_identity(self):
        """Best-effort read + parse of ``identity.toml``.

        Returns an :class:`IdentityManifest` (possibly empty on miss).
        Import is deferred to avoid eager-loading the identity module
        in tests that don't exercise this path.
        """
        from augmentum.coder.identity import IdentityManifest, parse_manifest

        text = await self._read(IDENTITY_TOML)
        if not text:
            return IdentityManifest()
        return parse_manifest(text)

    async def record_observation(
        self,
        *,
        category: str,
        fact: str,
        source: str,
        confidence: str = "confirmed",
    ) -> bool:
        """Append one observation to the cross-session ledger.

        Thin wrapper over :func:`augmentum.coder.observations.append_observation`
        so callers (the ``observe`` tool, future auto-extraction paths)
        stay decoupled from the on-disk format. Returns True on
        successful persist, False on any failure — best-effort
        contract; never raises.
        """
        if self._container_manager is None or not self._workspace_id:
            return False
        from augmentum.coder.observations import (
            CATEGORIES,
            CONFIDENCES,
            Observation,
            append_observation,
        )

        # Defensive validation. The tool layer should already gate on
        # these, but a kernel call site that bypasses the tool (e.g.
        # auto-extraction) shouldn't be able to write garbage either.
        cat = category if category in CATEGORIES else "other"
        conf = confidence if confidence in CONFIDENCES else "confirmed"
        fact_text = (fact or "").strip()
        if not fact_text:
            return False

        import time as _time
        obs = Observation(
            ts=_time.time(),
            category=cat,
            fact=fact_text,
            source=(source or "").strip() or "unknown",
            confidence=conf,
        )
        return await append_observation(
            self._container_manager,
            self._workspace_id,
            obs,
            path=OBSERVATIONS_JSONL,
        )

    async def read_observations(self) -> list:
        """Read the full observation ledger. ``[]`` on miss.

        Returns a list of :class:`Observation`. Empty when the file
        doesn't exist (fresh workspace) or can't be read. Callers can
        feed the result into ``query_observations`` for filtering.
        """
        if self._container_manager is None or not self._workspace_id:
            return []
        from augmentum.coder.observations import read_ledger

        return await read_ledger(
            self._container_manager,
            self._workspace_id,
            path=OBSERVATIONS_JSONL,
        )

    async def read_objective(self) -> str:
        """Best-effort read of ``objective.md``. ``""`` on miss.

        The objective is user-curated; we don't parse it. The full
        text is what the user wrote (with the seeded header stripped
        on render).
        """
        return await self._read(OBJECTIVE_MD)

    async def seed_objective_if_missing(self, text: str) -> bool:
        """Write ``objective.md`` from the user's first substantive ask.

        Idempotent: if the file already exists, nothing happens (user
        edits or prior seeds win). Gated on text length so a casual
        first message ("hi") doesn't pollute the anchor.

        Returns True iff a fresh seed was written.

        The seeded body uses a small wrapping header explaining the
        contract — that way a user who discovers the file by accident
        understands what it is without consulting docs:

            # Session Objective
            <!-- This is what you (the user) want from this session.   -->
            <!-- The agent reads this on every turn. Edit it directly  -->
            <!-- to correct course. The agent will not edit this file  -->
            <!-- without explicit permission.                           -->

            <user-provided seed text>
        """
        if self._container_manager is None or not self._workspace_id:
            return False
        body = (text or "").strip()
        if len(body) < _OBJECTIVE_SEED_MIN_CHARS:
            return False
        try:
            existing = await self._read(OBJECTIVE_MD)
            if existing:
                # File already present — user edits or prior seeds
                # take precedence. Never silently overwrite.
                return False
            # mkdir -p is idempotent — repeated seeds across a session
            # do the right thing.
            try:
                await self._container_manager._run_command(
                    self._workspace_id,
                    ["bash", "-c", "mkdir -p /workspace/.augmentum"],
                    timeout=3.0,
                )
            except Exception:
                log.debug("workspace_kernel.mkdir_kernel_root_failed", workspace_id=self._workspace_id, exc_info=True)

            seeded = (
                "# Session Objective\n"
                "<!-- This is what you (the user) want from this session. -->\n"
                "<!-- The agent reads this on every turn. Edit it directly -->\n"
                "<!-- to correct course. The agent will not edit this file -->\n"
                "<!-- without explicit permission. -->\n"
                "\n"
                f"{body}\n"
            )
            await self._container_manager.file_write(
                self._workspace_id, OBJECTIVE_MD, seeded,
            )
            return True
        except Exception:
            log.debug(
                "workspace_kernel_objective_seed_failed",
                workspace_id=self._workspace_id,
                exc_info=True,
            )
            return False

    @staticmethod
    def _strip_objective_header(text: str) -> str:
        """Extract the user-meaningful body, stripping our seeded header.

        Looks for the first non-blank, non-``#``, non-``<!--`` line
        and returns from there. Keeps a hand-edited file (where the
        user wrote arbitrary content) intact; only filters the
        scaffolding we wrote ourselves.
        """
        if not text:
            return ""
        lines = text.splitlines()
        body: list[str] = []
        started = False
        for line in lines:
            stripped = line.strip()
            if not started:
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    continue
                if stripped.startswith("<!--"):
                    continue
                started = True
            body.append(line)
        return "\n".join(body).strip()

    async def render_facts_block(
        self,
        *,
        budget_chars: int = 400,
        identity_budget: int = 120,
        objective_budget: int = 300,
    ) -> str:
        """Composite render of identity + observations for system-prompt inclusion.

        The output is a ``<workspace_facts>`` block:

            <workspace_facts>
            Project: python (uv) · test=pytest · deploy=fly.io
            Established:
              [constraint] node 18 is locked; do not require node 20+
              [gotcha] tests need SQLITE_PATH=/tmp/test.db
            </workspace_facts>

        Best-effort: returns ``""`` on container failure, empty
        manifest+ledger, or zero budget. Caller can unconditionally
        include the block — empty string is benign.

        ``budget_chars`` is the TOTAL budget; identity takes
        ``identity_budget`` (default 120) and the remainder goes to
        observations. The split favours identity because the
        "project shape" line is the cheapest, highest-leverage
        orientation — a model that knows "this is a Python uv +
        pytest project" can avoid a half-dozen discovery shell
        commands on its own.

        Why not unconditionally render: the 2026-05-16 kernel design
        ethos is "supplemental scaffolding lives in files, not in
        message injection." This method honors that contract by
        rendering a *summary* (compact, slow-changing) rather than
        the full files. The model can still ``cat`` the full
        identity.toml / observations.jsonl for detail.
        """
        if budget_chars <= 0:
            return ""
        if self._container_manager is None or not self._workspace_id:
            return ""

        # Objective render — TOP priority. The user-pinned anchor is
        # always-on so the model can never lose the original ask, even
        # at turn 50. Read takes precedence over identity/observations
        # because the budget MUST cover the objective if it exists.
        objective_text = ""
        try:
            raw_obj = await self.read_objective()
            if raw_obj:
                # Strip our seeded scaffolding (#-comment header +
                # HTML comments) so the rendered objective is just
                # the user's meaningful text.
                stripped = self._strip_objective_header(raw_obj)
                if stripped:
                    # Clip to objective_budget so a user who wrote
                    # paragraphs doesn't blow the whole facts budget.
                    if len(stripped) > objective_budget:
                        stripped = stripped[: max(0, objective_budget - 1)].rstrip() + "…"
                    objective_text = stripped
        except Exception:
            log.debug(
                "workspace_kernel_objective_render_failed",
                workspace_id=self._workspace_id,
                exc_info=True,
            )

        # Identity render — defensive on read failure. Module import
        # is deferred so this file stays importable without the
        # identity module loaded (mirrors read_identity/refresh_identity).
        identity_text = ""
        try:
            manifest = await self.read_identity()
            from augmentum.coder.identity import render_identity_summary

            identity_text = render_identity_summary(
                manifest, budget_chars=identity_budget,
            )
        except Exception:
            log.debug(
                "workspace_kernel_identity_render_failed",
                workspace_id=self._workspace_id,
                exc_info=True,
            )

        # Observations render — gets whatever budget remains after
        # objective + identity. The priority order is intentional:
        # the objective is the user's pinned ask (must survive),
        # identity is project orientation (cheap), observations
        # are durable facts (compress easily under pressure).
        remaining = budget_chars - len(objective_text) - len(identity_text) - 60
        obs_budget = max(0, remaining)
        observations_text = ""
        try:
            observations = await self.read_observations()
            if observations and obs_budget > 0:
                from augmentum.coder.observations import render_for_prompt

                observations_text = render_for_prompt(
                    observations, budget_chars=obs_budget,
                )
                # ``render_for_prompt`` wraps in its own
                # ``<observations>`` tags; we re-wrap inside our
                # composite block, so strip those.
                observations_text = observations_text.replace("<observations>", "")
                observations_text = observations_text.replace("</observations>", "")
                observations_text = observations_text.strip()
        except Exception:
            log.debug(
                "workspace_kernel_observations_render_failed",
                workspace_id=self._workspace_id,
                exc_info=True,
            )

        if not objective_text and not identity_text and not observations_text:
            return ""

        lines: list[str] = ["<workspace_facts>"]
        if objective_text:
            lines.append("Objective (user-pinned, see objective.md):")
            for raw in objective_text.split("\n"):
                if raw.strip():
                    lines.append(f"  {raw}")
            lines.append("")
        if identity_text:
            lines.append(identity_text)
        if observations_text:
            lines.append("Established (durable facts from prior work):")
            for raw in observations_text.split("\n"):
                if raw.strip():
                    lines.append(raw)
        # Trailing directive — the model has to know these are
        # canonical so it doesn't re-discover them via shell.
        lines.append("")
        # Closing directive — what the model should DO with this
        # block. Mentions the objective contract iff one's present;
        # otherwise just the trust-and-act language. Keeping the
        # objective rule visible at the bottom of the block (right
        # where the model's attention exits the block) is load-
        # bearing — a separate sys-prompt directive about
        # objective.md elsewhere would be easier to overlook.
        if objective_text:
            lines.append(
                "These facts are known true. Don't re-verify them "
                "via shell commands — trust the block and act on "
                "the objective. Do NOT edit objective.md without "
                "explicit user permission — it's the user's pinned "
                "anchor; ask first if you think it needs revision."
            )
        else:
            lines.append(
                "These facts are known true. Don't re-verify them "
                "via shell commands — trust the block and act on it."
            )
        lines.append("</workspace_facts>")
        return "\n".join(lines)

    async def render_orientation(
        self,
        *,
        budget_chars: int = 240,
        identity_budget: int = 100,
    ) -> str:
        """Minimal objective + project-shape anchor for slim-context subagents.

        Unlike :meth:`render_facts_block`, this deliberately drops the
        observations ledger — it carries only the two cheapest, highest-
        leverage orientation signals:

        * the user-pinned **objective** (so the child knows what the
          session is actually FOR, even when the lead's freehand prompt
          under-specifies), and
        * the **project-shape** identity line (so a ``research`` answer
          fits THIS stack — Python/FastAPI/SQLite — instead of drifting
          generic, and an ``audit_zone`` reviewer knows what the system
          IS before judging its code).

        Handed to every context mode including ``slim``. Kept tiny
        (≤ ~240 chars) so the 10-way ``audit_zone`` fan-out doesn't pay
        a real token cost. Best-effort: returns ``""`` on any read
        failure or empty manifest — callers can include it
        unconditionally.
        """
        if budget_chars <= 0:
            return ""
        if self._container_manager is None or not self._workspace_id:
            return ""

        objective_text = ""
        try:
            raw_obj = await self.read_objective()
            if raw_obj:
                stripped = self._strip_objective_header(raw_obj)
                if stripped:
                    obj_budget = max(0, budget_chars - identity_budget)
                    if len(stripped) > obj_budget:
                        stripped = stripped[: max(0, obj_budget - 1)].rstrip() + "…"
                    objective_text = stripped
        except Exception:
            log.debug(
                "workspace_kernel_orientation_objective_failed",
                workspace_id=self._workspace_id,
                exc_info=True,
            )

        identity_text = ""
        try:
            manifest = await self.read_identity()
            from augmentum.coder.identity import render_identity_summary

            identity_text = render_identity_summary(
                manifest, budget_chars=identity_budget,
            )
        except Exception:
            log.debug(
                "workspace_kernel_orientation_identity_failed",
                workspace_id=self._workspace_id,
                exc_info=True,
            )

        if not objective_text and not identity_text:
            return ""

        lines: list[str] = ["<orientation>"]
        if objective_text:
            lines.append(f"Objective: {objective_text}")
        if identity_text:
            lines.append(identity_text)
        lines.append("</orientation>")
        return "\n".join(lines)

    @staticmethod
    def hint_text(*, enabled: bool | None = None) -> str:
        """One-sentence system-prompt block describing the kernel directory.

        Tells the model that ``/workspace/.augmentum/`` is a scratch
        directory it can read on demand for orientation. Replaces
        per-iteration sticky-reminder injection of plan content with
        on-demand ``file_read`` — strong models ignore the directory,
        weak models read it, same surface serves both.

        Static so callsites can render the hint without a live kernel
        instance — the legacy handler constructs ``_workspace_kernel``
        only when a container_manager is present, but the system-
        prompt block should still appear in unit-test contexts where
        no container exists.

        Returns ``""`` when ``coder_kernel_v2`` is disabled OR when the
        caller explicitly passes ``enabled=False``. Centralised here
        rather than inlined at every callsite so the wording is
        single-sourced (handler's _build_messages + phase_act's
        native sys_text both render the same hint).

        Parameters
        ----------
        enabled:
            Override the global ``coder_kernel_v2`` flag check. When
            ``None`` (default), read the live setting. Passing an
            explicit boolean is rare — useful only for callers that
            have already gated on the flag and want to avoid a second
            settings import.
        """
        if enabled is None:
            from augmentum.config import settings as _settings
            enabled = bool(_settings.coder_kernel_v2)
        if not enabled:
            return ""
        return (
            "<workspace_kernel>\n"
            "Your workspace includes `/workspace/.augmentum/` — a "
            "scratch directory the system maintains. Run "
            "`ls /workspace/.augmentum/` to see what's there; read any "
            "file (e.g. `plan.md`) when you need orientation. The "
            "kernel writes these files between turns so you don't have "
            "to keep state in your head across iterations.\n"
            "</workspace_kernel>"
        )

    # ── Internals ─────────────────────────────────────────────────────

    async def _read(self, path: str) -> str:
        """Best-effort container file read returning trimmed content
        or ``""`` on miss. Centralised so every reader has the same
        error-swallowing contract."""
        if self._container_manager is None or not self._workspace_id:
            return ""
        try:
            raw = await self._container_manager.file_read(
                self._workspace_id, path,
            )
        except Exception:
            return ""
        return (raw or "").strip()
