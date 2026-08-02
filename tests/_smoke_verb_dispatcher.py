"""End-to-end smoke for the verb dispatcher substrate (Phase 2).

NOT part of the regular test suite — runs directly via `python -m
tests._smoke_verb_dispatcher` to validate the dispatcher + verb log +
cooldown behave correctly against a real PresenceBus.

To convert into a pytest test: wrap each block in a test function, use
pytest-asyncio fixtures for the backend. Phase 5 will land that as
part of the observability test suite.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from augmentum.companion_runtime.bus import PresenceBus
from augmentum.companion_runtime.event_bus import (
    DispatchClass,
    SafetyClass,
    VerbDispatcher,
    verb,
)
from augmentum.state.backends.sqlite import SQLiteBackend


async def main() -> int:
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    bus = PresenceBus()
    runtime = SimpleNamespace(
        bus=bus, backend=backend,
        companion_id="becca", owner_user_id="test_user",
    )

    fired: list[str] = []

    @verb(
        "time.tick(60s)",
        name="tick_test_verb",
        dispatch_class=DispatchClass.TICK_ALIGNED,
        safety_class=SafetyClass.READ,
        cooldown_ms=5000,
    )
    async def tick_test_verb(event, ctx) -> None:
        fired.append(event.topic)
        ctx.cite("companion_drive_state", row_id=1)

    dispatcher = VerbDispatcher(runtime)
    dispatcher.register(tick_test_verb)
    await dispatcher.start()
    print("dispatcher started, verb registered")

    # 1. First fire — should run
    await bus.publish_topic(
        "time.tick(60s)", {"interval_s": 60, "label": "60s"},
        source_companion_id="becca",
    )
    await asyncio.sleep(0.3)
    print(f"after 1st publish: fired={len(fired)} (expected 1)")
    assert len(fired) == 1, f"expected 1 fire, got {len(fired)}"

    # 2. Second fire within cooldown — should record cooldown_skipped
    await bus.publish_topic(
        "time.tick(60s)", {"interval_s": 60, "label": "60s"},
        source_companion_id="becca",
    )
    await asyncio.sleep(0.3)
    print(f"after 2nd publish: fired={len(fired)} (expected still 1)")
    assert len(fired) == 1, "cooldown should block 2nd fire"

    # 3. Non-matching topic — should NOT touch the verb
    await bus.publish_topic(
        "affect.pad", {"valence": 0.5, "arousal": 0.3, "dominance": 0.1},
        source_companion_id="becca",
    )
    await asyncio.sleep(0.3)
    print(f"after non-match publish: fired={len(fired)} (expected still 1)")
    assert len(fired) == 1, "non-match shouldn't fire"

    # 4. Inspect verb_log
    cur = await backend.conn.execute(
        "SELECT outcome, cited_substrate, latency_ms, error "
        "FROM companion_verb_log WHERE verb_name='tick_test_verb' ORDER BY id",
    )
    rows = await cur.fetchall()
    print()
    print(f"verb_log rows: {len(rows)}")
    for r in rows:
        print(f"  outcome={r[0]:18s} cited={r[1][:60]:60s} latency_ms={r[2]}")
    outcomes = [r[0] for r in rows]
    assert outcomes.count("ok") == 1, f"expected 1 ok, got {outcomes}"
    assert outcomes.count("cooldown_skipped") == 1, (
        f"expected 1 cooldown_skipped, got {outcomes}")

    ok_row = next(r for r in rows if r[0] == "ok")
    assert "companion_drive_state" in ok_row[1], "cite_self trail lost"

    # 5. Snapshot
    snap = dispatcher.snapshot()
    print()
    print(f"dispatcher.snapshot(): {snap}")
    assert snap["verbs_registered"] == 1
    assert snap["running"] is True

    await dispatcher.stop()
    await backend.close()
    print()
    print("=== Phase 2 substrate end-to-end verified ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
