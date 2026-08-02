"""End-to-end smoke for Phase 3c translation-layer chain.

Publishes a synthetic ``state.delta_threshold_crossed`` event and
verifies that:

  1. ``narrate_state_to_user`` runs and (if gated on) posts to the
     notification hub.
  2. ``propose_action`` runs and emits ``companion.action_proposed``.
  3. ``enqueue_proposed_action`` runs and writes a row to
     ``companion_initiative_queue``.

The three kill switches are toggled True in-process so the test
doesn't depend on DB-side overrides.

Run from inside the container::

    docker exec augmentum-augmentum-1 python -m tests._smoke_phase3c_chain
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from augmentum.companion_runtime.bus import PresenceBus
from augmentum.companion_runtime.event_bus import VerbDispatcher
from augmentum.companion_runtime.verbs import VerbRegistry
from augmentum.state.backends.sqlite import SQLiteBackend

_OWNER = "usr_test_owner"
_COMPANION = "cmp_test"


async def _wait_for(predicate, *, timeout=2.0, step=0.05):
    """Poll ``predicate`` until True or timeout. Returns final value."""
    waited = 0.0
    while waited < timeout:
        if predicate():
            return True
        await asyncio.sleep(step)
        waited += step
    return predicate()


async def main() -> int:
    backend = SQLiteBackend(":memory:")
    await backend.connect()

    # Tables are created by SQLiteBackend's migration suite during
    # connect(). We just seed a drive_state row so propose_action has
    # urgencies to read.
    await backend.conn.execute(
        "INSERT OR REPLACE INTO companion_drive_state "
        "(user_id, companion_id, curiosity_level, competence_level, "
        "connection_level, rest_level) VALUES (?, ?, ?, ?, ?, ?)",
        (_OWNER, _COMPANION, 0.9, 0.4, 0.7, 0.3),
    )
    await backend.conn.commit()

    # Flip the kill switches in-process so we exercise the live path.
    # Also lift presence_mode to "engaged" — the dispatcher's autonomy
    # gate reads presence_mode and rejects anything WRITE_USER while the
    # mode is "silent" (the secure default), and the gate fires once per
    # fanout pass before per-verb dispatch.
    from augmentum.config import settings
    object.__setattr__(settings, "companion_narrate_state_enabled", True)
    object.__setattr__(settings, "companion_propose_action_enabled", True)
    object.__setattr__(settings, "companion_action_enqueue_enabled", True)
    object.__setattr__(settings, "companion_presence_mode", "engaged")

    bus = PresenceBus()
    runtime = SimpleNamespace(
        bus=bus,
        backend=backend,
        companion_id=_COMPANION,
        owner_user_id=_OWNER,
        _app_state=SimpleNamespace(notification_hub=None),
        _last_pad=None,
        memory=SimpleNamespace(_backend=backend),
        state=SimpleNamespace(snapshot=lambda: {"role": "active:1.0|passive:0.0"}),
    )

    dispatcher = VerbDispatcher(runtime=runtime)
    await dispatcher.start()

    # Register just the three Phase 3c verbs we care about. The shared
    # registry is process-wide so all 12 are already loaded.
    target_names = {"narrate_state_to_user", "propose_action", "enqueue_proposed_action"}
    for v in VerbRegistry.all():
        if v.name in target_names:
            dispatcher.register(v)
    print(f"registered {len(target_names)} translation verbs")

    # Synthetic threshold-crossed event matching the new payload shape
    # (carries absolutes plus deltas).
    payload = {
        "field": "affect.pad",
        "valence": 0.42,
        "arousal": 0.71,
        "dominance": 0.30,
        "valence_delta": 0.21,
        "arousal_delta": 0.18,
    }

    # Publish with empty source — the dispatcher self-filters events from
    # its own companion_id (so a verb's emit() doesn't re-enter the same
    # dispatcher). An external synthetic publish needs a non-matching source.
    await bus.publish_topic(
        "state.delta_threshold_crossed",
        payload,
        source_companion_id="",
    )

    # Wait for fanout to drain. The verbs do small async work; 1s is
    # plenty in a clean process.
    await asyncio.sleep(1.0)

    # ── Read the verb_log to see what fired ───────────────────────────
    async with backend.conn.execute(
        "SELECT verb_name, outcome, latency_ms FROM companion_verb_log "
        "ORDER BY fired_at ASC"
    ) as cur:
        rows = await cur.fetchall()

    print("\nverb_log after synthetic publish:")
    for r in rows:
        print(f"  {r[0]:<32} {r[1]:<22} {r[2]}ms")

    # The enqueue verb is event-driven on companion.action_proposed, which
    # propose_action emits. Wait once more for the second-hop.
    await asyncio.sleep(0.5)

    async with backend.conn.execute(
        "SELECT kind, payload, score, status FROM companion_initiative_queue"
    ) as cur:
        queue_rows = await cur.fetchall()

    print("\ncompanion_initiative_queue rows:")
    for r in queue_rows:
        print(f"  kind={r[0]}  score={r[2]}  status={r[3]}")
        print(f"  payload={r[1]}")

    await dispatcher.stop()
    await backend.close()

    # ── Verdict ───────────────────────────────────────────────────────
    fired = {r[0] for r in rows}
    expected = {"narrate_state_to_user", "propose_action", "enqueue_proposed_action"}
    missing = expected - fired
    if missing:
        print(f"\nFAIL: verbs missing from log: {missing}")
        return 1
    if not queue_rows:
        print("\nFAIL: enqueue_proposed_action didn't write to initiative_queue")
        return 1
    print("\n=== Phase 3c chain verified ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
