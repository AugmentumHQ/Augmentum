# Companion Embodiment Reuse Note

Date: 2026-06-06
Status: Living implementation note

## Principle

Prefer one central reusable outlet over many direct hooks.

When companion/autonomy behavior needs to affect the avatar, route the intent
through the existing embodiment surfaces first. Do not add direct animator calls,
one-off event listeners, or bespoke animation selectors unless the current
surfaces cannot express the behavior.

The desired shape is:

1. A situation becomes a companion verb or pose verb.
2. The central router maps that verb to existing posture, gesture, emotion, or
   movement-conductor intent.
3. The avatar loop/arbitrator applies it with priority, expiry, and recovery.
4. Pose families and slerp/via rules keep motion safe and reusable.

This lets one change in the central map fan out cleanly across chat, PTT, TTS,
presence-bus events, hosting mode, idle mode, and future autonomous behaviors.

## Current Central Surfaces

- `ui/scripts/companion-animation-router.js`
  - Central companion event/verb to animation and pose-intent adapter.
  - Add new situational mappings here before wiring another caller.
  - Accepts explicit `behavior.animation_intent` payloads, including `pose_verb`
    and `pose_family`.

- `ui/scripts/avatar-pose-presets.js`
  - Single source of truth for approved static pose families.
  - Promote existing custom JSON poses into this catalog after visual review.
  - Keep families as safe slerp basins; do not mix cross-body or foot-shifted
    poses into a family unless direct family drift is visually safe.

- `ui/scripts/avatar-pose-orchestrator.js`
  - Existing family drift, slerp, sequence, and `via` waypoint engine.
  - Prefer adding `via` waypoints/family metadata over creating a new transition
    mechanism.

- `ui/scripts/avatar.js`
  - Final live avatar pose arbitration.
  - Owns recovery from transient companion pose intents back to flow-driven idle.

- `ui/scripts/avatar-pose-trigger.js`, `ui/scripts/movement-conductor.js`,
  and `ui/scripts/anim-atlas.js`
  - Existing path for gesture/clip selection, cooldowns, energy budget, ratings,
    and explicit pose/animation requests.

## Reuse Checklist

Before adding companion embodiment behavior:

1. Can this be represented as a `pose_verb`, existing conductor roles, or an
   explicit atlas id?
2. Can an existing pose family represent the body language?
3. If not, is there an existing `poses/*.json` export that can be promoted into
   `POSE_PRESETS`?
4. Does the new pose need its own family to preserve safe slerp drift?
5. Should transitions route through existing neutral waypoints with `via`?
6. Does the router set an expiry so the avatar returns to idle/hosting flow?
7. Did tests cover the central mapping rather than only one caller?

## Current Direction

The production path should continue using `PoseOrchestrator`,
`PresenceEngine`, `PoseTriggerEngine`, and `MovementConductor`.

`MotionEngine`, `EmbodimentEngine`, and `PoseResolver` remain valuable future
substrate, but they should enter production through a narrow adapter or shadow
mode first. Do not replace the live stack wholesale while the current stack can
express the behavior.

