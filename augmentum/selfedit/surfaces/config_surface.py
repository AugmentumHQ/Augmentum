"""The config / Adaptation surface — the first concrete adapter and the
reachable-NOW one.

Adaptation = per-user config/state (layout, density, which panels, theme,
surfaced features, shortcuts). It's DATA, not code: no build, no worktree, no
restart — instant and reversible. Crucially it carries the **cheapest possible
oracle**: setting a value to X and reading back X *is* the intent confirmation, a
mechanical check. So an Adaptation change earns the VERIFIED tier honestly and
auto-applies — which is exactly why "the app rearranged itself for me" can be
instant and safe.

The settings store is injected (two async callables + a revert ledger dict), so
this adapter is pure and testable with a plain dict. The real wiring passes the
per-user settings store; writes MUST be scoped by ``change.actor`` (user_id) — the
adapter refuses an empty actor so it can never write the anon/shared row.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from augmentum.selfedit.surfaces.base import (
    CLASS_ADAPTATION,
    CaptureArtifact,
    ReshapeChange,
    ReshapeOutcome,
    SurfaceAdapter,
)
from augmentum.selfedit.verifier import (
    FAIL,
    ORACLE_MECHANICAL,
    PASS,
    Verifier,
    VerifierResult,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

SURFACE_CONFIG = "config"

# Injected store seams: read a user's setting, write it.
ReadSetting = Callable[[str, str], Awaitable[object]]        # (user_id, key) -> value | None
WriteSetting = Callable[[str, str, object], Awaitable[None]]  # (user_id, key, value)


def _payload(change: ReshapeChange) -> tuple[str, str, object]:
    p = change.payload
    return change.actor, str(p.get("key", "")), p.get("value")


def build_config_surface(*, read: ReadSetting, write: WriteSetting,
                         revert_ledger: dict | None = None) -> SurfaceAdapter:
    """Construct the config surface bound to a (read, write) settings store. The
    revert ledger (a dict; the caller may persist it) holds prior values keyed by
    a deterministic token so a change can be undone exactly."""
    ledger: dict = revert_ledger if revert_ledger is not None else {}

    async def apply(change: ReshapeChange) -> ReshapeOutcome:
        user_id, key, value = _payload(change)
        if not user_id:
            return ReshapeOutcome(False, detail="refusing config write with empty actor")
        if not key:
            return ReshapeOutcome(False, detail="no config key in payload")
        prior = await read(user_id, key)
        # Deterministic, unique token (no RNG/clock): ledger size is monotonic.
        token = f"{user_id}|{key}|{len(ledger)}"
        ledger[token] = {"user_id": user_id, "key": key, "prior": prior}
        await write(user_id, key, value)
        log.info("reshape_config_applied", key=key, user_id=user_id)
        return ReshapeOutcome(True, revert_token=token, detail=f"set {key}")

    async def revert(token: str) -> bool:
        rec = ledger.get(token)
        if rec is None:
            return False
        await write(rec["user_id"], rec["key"], rec["prior"])
        log.info("reshape_config_reverted", key=rec["key"], user_id=rec["user_id"])
        return True

    def make_verifier(change: ReshapeChange) -> Verifier:
        """Mechanical oracle: read the value back and confirm it equals what was
        asked. Read-back == intended IS the intent confirmation for a config set."""
        _, key, value = _payload(change)

        async def run(_ctx: dict) -> VerifierResult:
            current = await read(change.actor, key)
            ok = current == value
            return VerifierResult(
                name=f"config:{key}", oracle=ORACLE_MECHANICAL,
                status=PASS if ok else FAIL, confirms_intent=True,
                score=1.0 if ok else 0.0, required=True,
                detail=f"read-back {'==' if ok else '!='} intended for {key}")

        return Verifier(f"config:{key}", ORACLE_MECHANICAL, run,
                        intent_classes=(CLASS_ADAPTATION,), confirms_intent=True, cost=1)

    async def capture(change: ReshapeChange) -> CaptureArtifact:
        _, key, _value = _payload(change)
        current = await read(change.actor, key)
        return CaptureArtifact(kind="state", ref=key, summary=f"{key} = {current!r}")

    return SurfaceAdapter(
        name=SURFACE_CONFIG, change_classes=(CLASS_ADAPTATION,),
        apply=apply, revert=revert, make_verifier=make_verifier, capture=capture,
        note="per-user config; mechanical read-back oracle → VERIFIED tier, "
             "instant + reversible. The reachable-now surface.")
