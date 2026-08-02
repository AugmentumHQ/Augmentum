"""The foundry loop — the closed ring that makes generation self-improving.

    concept ─▶ [3d: Blender asset + visual-verify] ─▶ generate game code
            ─▶ deploy ─▶ game_agent plays it ─▶ progress.py score
            ─▶ defect relay ─▶ regenerate (next pass)

The MVP acceptance test: run two passes with NO human in the loop and see the
pass-2 score exceed pass-1 (the relay actually made the game more playable).

Design: the loop is written against **injected stage callables** so its
control flow — pass iteration, relay threading, score-delta — is unit-testable
with fakes (see tests). ``wire_default_stages`` binds the real implementations
(coder loop dispatch, Blender tool, VisionRouter, game-agent session + play).
The real stages need the running stack (job runner, workspace, orchestrator,
headless browser host); the loop logic here does not, so it is verified
independently of them.
"""
from __future__ import annotations

import base64
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from augmentum.coder.foundry.contract import GameBuildSpec
from augmentum.game_agent.defect_relay import defects_from_progress, relay_brief
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Emitter signature: (event_type, **data) -> None. No-op default so the loop
# runs headless (tests, cron) without a theater attached.
EventEmitter = Callable[..., None]


def _noop_emit(_type: str, **_data: Any) -> None:
    pass


# ── Stage callable contracts (all async) ──────────────────────────────
# Each returns plain data so the loop can thread it without knowing the
# implementation. Any stage may raise; the loop records the failure on the
# pass and continues (a broken pass is data, not a crash).

# (spec) -> {"glb_asset": str, "render_png_bytes": bytes|None}  (3d only)
AssetStage = Callable[[GameBuildSpec], Awaitable[dict]]
# (image_bytes, objective) -> list[str] visual-defect notes
VerifyStage = Callable[[bytes, str], Awaitable[list]]
# (spec) -> {"slug": str, "files": dict[str,str], "violations": list[str], "run_id": str}
GenerateStage = Callable[[GameBuildSpec], Awaitable[dict]]
# (slug, files, spec, play_seconds) -> SessionEndPayload.progress dict | None
PlayStage = Callable[[str, dict, GameBuildSpec, int], Awaitable[dict]]


@dataclass
class PassResult:
    """One generation→play→score pass."""

    index: int
    slug: str = ""
    run_id: str = ""
    violations: list[str] = field(default_factory=list)
    progress: dict | None = None          # ProgressScore.to_dict() or None
    vision_notes: list[str] = field(default_factory=list)
    relay: str = ""                        # brief fed to the NEXT pass
    error: str = ""

    @property
    def score(self) -> float:
        return float((self.progress or {}).get("score", 0.0) or 0.0)

    @property
    def score_per_min(self) -> float:
        return float((self.progress or {}).get("score_per_min", 0.0) or 0.0)


@dataclass
class FoundryResult:
    """Outcome of a full foundry run."""

    passes: list[PassResult] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        """True when a later pass beat the first on the honest metric.

        Uses ``score`` when the runs are comparable-length, else
        ``score_per_min`` (the normalized metric progress.py recommends for
        unequal durations). Requires at least two scored passes.
        """
        scored = [p for p in self.passes if p.progress is not None]
        if len(scored) < 2:
            return False
        first, last = scored[0], scored[-1]
        # Prefer score_per_min — robust to unequal play durations.
        if last.score_per_min != first.score_per_min:
            return last.score_per_min > first.score_per_min
        return last.score > first.score

    def summary(self) -> str:
        lines = ["Foundry run:"]
        for p in self.passes:
            if p.error:
                lines.append(f"  pass {p.index}: ERROR {p.error}")
            elif p.progress is None:
                lines.append(f"  pass {p.index}: {p.slug} — no score (play failed)")
            else:
                lines.append(
                    f"  pass {p.index}: {p.slug} — score={p.score:.3f} "
                    f"score_per_min={p.score_per_min:.3f} "
                    f"defects={len(p.violations)}+{len(p.vision_notes)}"
                )
        lines.append(f"  improved: {self.improved}")
        return "\n".join(lines)


async def run_foundry(
    spec: GameBuildSpec,
    *,
    generate: GenerateStage,
    play: PlayStage,
    asset: AssetStage | None = None,
    verify: VerifyStage | None = None,
    passes: int = 2,
    play_seconds: int = 90,
    on_event: EventEmitter | None = None,
) -> FoundryResult:
    """Drive the closed loop for ``passes`` iterations, threading feedback.

    ``asset`` + ``verify`` are only used for 3d specs (Blender stage). The
    ``spec`` is mutated pass-to-pass with the latest ``relay`` + ``glb_asset``
    so each regeneration sees the prior playtest's defects. ``on_event`` is the
    theater feed — called with structured stage events (see events.py).
    """
    emit = on_event or _noop_emit
    result = FoundryResult()
    total = max(1, passes)
    emit("run_start", passes=total, dimension=spec.dimension, title=spec.title)

    for i in range(1, total + 1):
        pr = PassResult(index=i)
        emit("pass_start", index=i)
        try:
            # 1. Asset stage (3d only) + visual verify of the render.
            if spec.dimension == "3d" and asset is not None:
                emit("asset_building", index=i)
                asset_out = await asset(spec)
                spec.glb_asset = asset_out.get("glb_asset", spec.glb_asset)
                render = asset_out.get("render_png_bytes")
                if render:
                    emit("asset_render", index=i,
                         image="data:image/png;base64," + base64.b64encode(render).decode())
                    if verify is not None:
                        pr.vision_notes = await verify(render, spec.objective) or []
                        for note in pr.vision_notes:
                            emit("observation", index=i, text=f"render check: {note}")

            # 2. Generate the game code (contract-validated by the stage).
            emit("generating", index=i)
            gen = await generate(spec)
            pr.slug = gen.get("slug", "")
            pr.run_id = gen.get("run_id", "")
            pr.violations = list(gen.get("violations", []))
            files = gen.get("files", {}) or {}
            emit("generated", index=i, slug=pr.slug, violations=pr.violations,
                 file_count=len(files))
            if pr.violations:
                # Contract not met — the build isn't playable. Record it, turn
                # the violations into next-pass feedback, and skip play (there
                # is nothing gradable). "designed ≠ applied": never pretend.
                log.warning("foundry_contract_violations", slug=pr.slug,
                            violations=pr.violations)
                pr.relay = _violations_relay(pr.violations)
                spec.relay = pr.relay
                emit("pass_scored", index=i, score=None, score_per_min=None,
                     defects=[{"kind": "contract", "severity": "blocker", "detail": v}
                              for v in pr.violations])
                result.passes.append(pr)
                if i < total:
                    emit("regenerating", index=i)
                continue

            # 3. Deploy + autonomous play → external score.
            emit("play_start", index=i)
            pr.progress = await play(pr.slug, files, spec, play_seconds)

            # 4. Defect relay (worst-first, from the unfakeable score).
            defects = defects_from_progress(
                pr.progress, duration_ms=(pr.progress or {}).get("duration_ms"),
                vision_notes=pr.vision_notes,
            )
            pr.relay = relay_brief(defects, pr.progress)
            spec.relay = pr.relay
            emit("pass_scored", index=i, score=pr.score,
                 score_per_min=pr.score_per_min,
                 defects=[asdict(d) for d in defects])
            if defects and i < total:
                emit("regenerating", index=i)

        except Exception as exc:  # a broken pass is data, not a crash
            pr.error = str(exc)
            emit("observation", index=i, text=f"pass failed: {exc}")
            log.warning("foundry_pass_failed", index=i, error=str(exc), exc_info=True)

        result.passes.append(pr)
        log.info("foundry_pass_done", index=i, slug=pr.slug,
                 score=pr.score, improved_so_far=result.improved)

    emit("done", improved=result.improved,
         passes=[asdict(p) for p in result.passes])
    return result


def _violations_relay(violations: list[str]) -> str:
    """Turn contract violations into a next-pass feedback block."""
    lines = ["## Build contract not satisfied — the game could not be playtested.",
             "Fix these so the game is agent-playable:"]
    for i, v in enumerate(violations, 1):
        lines.append(f"{i}. {v}")
    return "\n".join(lines)
