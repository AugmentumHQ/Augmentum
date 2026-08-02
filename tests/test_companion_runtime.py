"""Companion runtime substrate contracts.

These tests intentionally sit at the subsystem level. The coverage
scanner treats ``test_companion_runtime.py`` as the shared anchor for
the growing runtime package, and future verb/avatar/autonomy tests can
extend this file instead of creating one-off coverage shims.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_presence_bus_promotes_critical_event_over_idle_backpressure():
    from augmentum.companion_runtime.bus import PresenceBus

    bus = PresenceBus()
    sub = await bus.subscribe("**", queue_size=1, slice_key="test")

    await bus.publish_topic("surface.idle", {"pose": "rest"})
    await bus.publish_topic("state.transition", {"state": "focused"})

    event = await sub.queue.get()
    assert event.topic == "state.transition"
    assert event.payload == {"state": "focused"}
    assert bus.snapshot()["dropped_total"] == 1

    await bus.unsubscribe(sub)


@pytest.mark.asyncio
async def test_emit_chat_turn_completed_uses_mode_propagation_default():
    from augmentum.companion_runtime.bus import (
        PROP_FACTUAL_ONLY,
        PresenceBus,
        emit_chat_turn_completed,
    )

    bus = PresenceBus()
    runtime = SimpleNamespace(bus=bus, companion_id="becca")
    app_state = SimpleNamespace(companion_runtime=runtime)
    sub = await bus.subscribe("chat.*", slice_key="test")

    await emit_chat_turn_completed(
        app_state,
        mode="coder",
        user_id="usr_test",
        session_id="sess_1",
        wire_format="openai",
        stream=True,
        user_text="",
        assistant_text="",
    )

    event = await sub.queue.get()
    assert event.topic == "chat.turn_completed"
    assert event.propagation == PROP_FACTUAL_ONLY
    assert event.payload["mode"] == "coder"
    assert event.payload["session_id"] == "sess_1"

    await bus.unsubscribe(sub)


def test_verb_event_strips_dispatch_metadata_and_preserves_propagation():
    from augmentum.companion_runtime.bus import PROP_AFFECT_ONLY, PresenceEvent
    from augmentum.companion_runtime.event_bus import VerbEvent

    event = PresenceEvent(
        topic="behavior.pose",
        payload={
            "pose": "wave",
            "_chain_depth": 2,
            "_parent_verb_log_id": 41,
        },
        source_companion_id="becca",
        propagation=PROP_AFFECT_ONLY,
    )

    wrapped = VerbEvent.from_presence(
        event,
        owner_user_id="usr_test",
        companion_id="becca",
    )

    assert wrapped.topic == "behavior.pose"
    assert wrapped.payload == {"pose": "wave"}
    assert wrapped.chain_depth == 2
    assert wrapped.parent_verb_log_id == 41
    assert wrapped.source_companion_id == "becca"
    assert wrapped.propagation == PROP_AFFECT_ONLY


def test_dispatch_policy_defaults_back_management_verb_contract():
    from augmentum.companion_runtime.dispatch_policy import (
        DEFAULT_CHAIN_DEPTH_LIMIT,
        DEFAULT_COALESCE_WINDOW_S,
        DEFAULT_MAX_CONSECUTIVE_ERRORS,
    )
    from augmentum.companion_runtime.event_bus import (
        DispatchClass,
        ManagementVerb,
        SafetyClass,
    )

    async def handler(event, ctx):
        return None

    verb = ManagementVerb(
        name="pose_wave",
        handler=handler,
        subscribes_to=("behavior.pose",),
    )

    assert DEFAULT_MAX_CONSECUTIVE_ERRORS == 5
    assert DEFAULT_CHAIN_DEPTH_LIMIT >= 3
    assert DEFAULT_COALESCE_WINDOW_S > 0
    assert verb.safety_class is SafetyClass.READ
    assert verb.dispatch_class is DispatchClass.TICK_ALIGNED
    assert verb.matches("behavior.pose")


def test_embody_event_maps_action_proposal_to_animation_intent():
    from augmentum.companion_runtime.verbs.embody_event import (
        animation_intent_for_event,
    )

    intent = animation_intent_for_event(
        "companion.action_proposed",
        {"kind": "reach_out", "urgency": 0.8},
    )

    assert intent is not None
    assert intent["roles"] == ["attention-seek", "reach-out"]
    assert intent["pose_verb"] == "reach_out"
    assert intent["priority"] == "situational"
    assert intent["source"] == "verb:embody_event:action:reach_out"


def test_embody_event_maps_pad_delta_to_body_language():
    from augmentum.companion_runtime.verbs.embody_event import (
        animation_intent_for_event,
    )

    intent = animation_intent_for_event(
        "state.delta_threshold_crossed",
        {"field": "affect.pad", "valence": -0.42, "arousal": 0.72},
    )

    assert intent is not None
    assert intent["roles"] == ["react-negative", "mirror-anger"]
    assert intent["pose_verb"] == "boundary"
    assert intent["emotion"]["energy"] == 0.72
    assert intent["reason"] == "state.delta_threshold_crossed"


@pytest.mark.asyncio
async def test_embody_event_emits_behavior_animation_intent_contract():
    from augmentum.companion_runtime.bus import PROP_AFFECT_ONLY
    from augmentum.companion_runtime.verbs.embody_event import embody_event

    emitted = []

    class _Ctx:
        async def emit(self, topic, payload, *, propagation="full"):
            emitted.append({
                "topic": topic,
                "payload": payload,
                "propagation": propagation,
            })

    event = SimpleNamespace(
        topic="companion.action_proposed",
        payload={"kind": "creation", "urgency": 0.66},
        propagation=PROP_AFFECT_ONLY,
    )

    await embody_event.handler(event, _Ctx())

    assert emitted == [{
        "topic": "behavior.animation_intent",
        "payload": {
            "roles": ["think", "ponder"],
            "pose_verb": "thinking",
            "emotion": {
                "warmth": 0.52,
                "energy": 0.36,
                "openness": 0.58,
                "focus": 0.9,
            },
            "source": "verb:embody_event:action:creation",
            "priority": "situational",
            "pose_duration_ms": 6810,
            "reason": "companion.action_proposed",
            "explicit": False,
        },
        "propagation": PROP_AFFECT_ONLY,
    }]


def test_embody_event_is_registered_with_management_verbs():
    from augmentum.companion_runtime.verbs import VerbRegistry

    assert "embody_event" in VerbRegistry.names()


@pytest.mark.asyncio
async def test_energy_spend_never_drops_below_floor():
    from augmentum.companion_runtime import energy

    class _Cursor:
        def __init__(self, row=None):
            self._row = row

        async def fetchone(self):
            return self._row

        async def close(self):
            return None

    class _Conn:
        def __init__(self):
            self.updates = []

        async def execute(self, sql, params=()):
            if sql.lstrip().upper().startswith("SELECT"):
                return _Cursor((0.06, energy.DEFAULT_BASELINE, None, None))
            self.updates.append((sql, params))
            return _Cursor()

        async def commit(self):
            return None

    conn = _Conn()
    runtime = SimpleNamespace(
        companion_id="becca",
        backend=SimpleNamespace(conn=conn),
    )

    await energy.spend(runtime, user_id="usr_test", amount=1.0)

    assert conn.updates
    assert conn.updates[-1][1][0] == energy.LEVEL_FLOOR


def test_safety_floor_surface_thresholds_and_classifier_hook():
    from augmentum.companion_runtime import safety_floor

    safety_floor.set_classifier(lambda text: 0.75)
    try:
        free_chat = safety_floor.classify("plain test", surface="free_chat")
        coder = safety_floor.classify("plain test", surface="coder")
    finally:
        safety_floor.set_classifier(safety_floor._regex_score)

    assert free_chat.fired is True
    assert free_chat.threshold_used < coder.threshold_used
    assert coder.fired is False


def test_tool_protocol_scans_structured_action_tags():
    from augmentum.companion_runtime.tool_protocol import scan

    calls = scan(
        "one beat <tool:image_generate prompt='moonlit desk' /> "
        "<handoff:coder brief=\"inspect runtime\" />"
    )

    assert [call.kind for call in calls] == ["tool", "handoff"]
    assert calls[0].name == "image_generate"
    assert calls[0].args == {"prompt": "moonlit desk"}
    assert calls[1].name == "coder"
    assert calls[1].args == {"brief": "inspect runtime"}
