/**
 * avatar-spatial-director.js — Proxemic orchestrator (Sprint 1 / Unit C).
 *
 * Subscribes to the conversational PresenceEngine and the avatar FSM, maps
 * continuous affect signals (`presence`, `flow`, `temperature`, `resonance`)
 * into a schema-shaped *spatial intent*, drives FSM posture transitions,
 * and prepares the per-frame `ctx` consumed by `fformation.js::update()`.
 *
 * Phase 1 (this file) is **heuristic-only** — no SLM. The heuristic rule
 * table is encoded verbatim from the Wave 2 sprint plan's Unit C section.
 * Phase 3 will blend an SLM-emitted intent on top of the heuristic; the
 * `_slmIntent` slot is reserved for that, but production logic is out of
 * scope for Sprint 1 and intentionally not implemented.
 *
 * Hot-path discipline (matches `fformation.js` conventions):
 *   - No allocations inside `update()`. Both returned sub-objects
 *     (`intent` and `fformationCtx`) are stable scratch instances owned
 *     by the director — callers must NOT retain them across frames.
 *   - `dt` is clamped to `[0, 0.25]` to defend against backgrounded-tab
 *     multi-second frames (same convention as `fformation.js`).
 *   - No `console.log`. `console.warn` only on truly unexpected input
 *     (e.g. missing PresenceEngine reference at construction).
 *
 * THREE.js dependency — bare 'three' specifier resolved via the HTML
 * import map (`index.html`, `avatar-testbench.html`, etc.) that maps
 * 'three' → `ui/lib/three/three.module.min.js`. Matches every other
 * `ui/scripts/avatar-*.js` module.
 *
 * @see docs/superpowers/specs/2026-05-14-embodied-presence-design.md
 * @see docs/superpowers/plans/2026-05-14-embodied-presence-sprint-plan.md
 */

import * as THREE from 'three';

import { FORMATION, INTENT } from './fformation.js';
import { FSM_STATES } from './avatar-fsm.js';

// ─── Heuristic thresholds (Unit C table) ──────────────────────────────
//
// Encoded verbatim from the Wave 2 sprint plan's "Heuristic rules to
// encode" table. Tunable post-deploy; collected here so a single edit
// covers every branch below.

const THRESH = Object.freeze({
  TEMP_LEAN_IN:       0.7,   // temperature > X with high resonance → lean_in
  TEMP_POSITIVE:      0.85,  // proxy for positive valence inside lean_in
  RESO_LEAN_IN:       0.6,   // pairs with TEMP_LEAN_IN
  RESO_EMPHATIC:      0.5,   // pairs with FLOW_EMPHATIC
  FLOW_EMPHATIC:      0.6,   // high outgoing flow → vis_a_vis lean_in
  TEMP_SETTLE:        0.3,   // temperature < X with low presence → settle
  PRES_SETTLE:        0.4,   // presence < X
  SILENCE_SETTLE_S:   25,    // silence > X seconds → settle
  PRES_IDLE_DWELL:    0.3,   // long-silence dwell guard
  SILENCE_IDLE_S:     60,    // silence > X seconds → idle dwell (noop)
});

// ─── TTL / confidence defaults ────────────────────────────────────────

const TTL = Object.freeze({
  LEAN_IN_MS:    3000,
  SETTLE_MS:     4000,
  IDLE_DWELL_MS: 6000,
  HOLD_MS:       1500,
});

const CONFIDENCE = Object.freeze({
  LEAN_IN: 0.75,
  SETTLE:  0.7,
  DWELL:   0.6,
  HOLD:    0.5,
});

// Cap on the telemetry `reason` field per the spatial intent schema.
const REASON_MAX = 48;

/**
 * Truncate a telemetry reason to the schema's 48-char ceiling without
 * allocating beyond what `String.prototype.slice` already does on a
 * short input. Defensive only — callers should pass short literals.
 * @param {string} s
 * @returns {string}
 */
function _capReason(s) {
  if (typeof s !== 'string') return '';
  return s.length <= REASON_MAX ? s : s.slice(0, REASON_MAX);
}

/**
 * Clamp a number to [lo, hi]; substitutes `fallback` when input is not
 * a finite number. Mirrors `_clamp01` in `fformation.js`.
 * @param {number|undefined|null} v
 * @param {number} lo
 * @param {number} hi
 * @param {number} fallback
 * @returns {number}
 */
function _clamp(v, lo, hi, fallback) {
  if (!Number.isFinite(v)) return fallback;
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

// ─── ProxemicDirector ─────────────────────────────────────────────────

/**
 * Phase-1 heuristic proxemic director.
 *
 * Owns: the spatial-intent scratch object, the fformation-ctx scratch
 * object, and a reference to (but no introspection of) the FSM and the
 * fformation controller. The director is stateless across runs aside
 * from the scratch buffers and a small bookkeeping field for the last
 * emitted verb (used only for telemetry continuity — not for behavior).
 *
 * Phase 3 hook: `_slmIntent` is reserved for a future SLM-emitted intent
 * blend. Not implemented in Phase 1; setter is a no-op stub.
 */
export class ProxemicDirector {
  /**
   * @param {{
   *   presence: object,
   *   fsm: object,
   *   fformation: object,
   * }} opts
   *   - `presence`: a `PresenceEngine` instance exposing `.presence`,
   *     `.flow`, `.temperature`, `.resonance` (read every frame).
   *   - `fsm`:      an `AvatarFSM` instance; the director calls
   *                 `fsm.current()` to read state and (caller-side)
   *                 routes `fsmRequest` back into `fsm.requestTransition()`.
   *   - `fformation`: the `FFormationController` instance whose
   *                 `update(ctx, dt)` the host will call. Held opaquely
   *                 in Phase 1 — the director only fills its `ctx`.
   */
  constructor(opts) {
    if (!opts || !opts.presence) {
      console.warn('[avatar-spatial-director] missing PresenceEngine; intents will default to hold');
    }
    this._presence   = opts?.presence   ?? null;
    this._fsm        = opts?.fsm        ?? null;
    this._fformation = opts?.fformation ?? null;

    // Reserved for Phase 3 SLM blend. Intentionally inert in Phase 1.
    this._slmIntent = null;

    // Last-emitted verb / formation — telemetry continuity only.
    this._lastVerb      = 'hold';
    this._lastFormation = FORMATION.NONE;

    // Stable scratch intent object. Returned by `update()`; callers must
    // NOT retain it across frames (will be overwritten next tick).
    // Shape mirrors the spec's "Spatial intent schema" exactly.
    this._scratchIntent = {
      v:        1,
      noop:     false,
      formation: FORMATION.NONE,
      side:     'either',
      zone:     'interpersonal',
      verb:     'hold',
      gaze: {
        target:    'user_eyes',
        mutual_ms: 0,
      },
      gesture_sync: {
        mode: 'none',
        role: 'none',
      },
      anchor:    null,
      seat_id:   null,
      urgency:   0.0,
      confidence: CONFIDENCE.HOLD,
      ttl_ms:    TTL.HOLD_MS,
      reason:    'init',
    };

    // Stable scratch fformation-ctx object. Shape matches the keys read
    // by `FFormationController.update()` (see fformation.js JSDoc).
    // We pre-allocate the nested `self` / `user` / `anchor` / `affect`
    // sub-objects and the THREE.Vector3 slots so the hot path performs
    // only field writes, never `new`.
    this._scratchCtx = {
      self: {
        pos: new THREE.Vector3(),
        yaw: 0,
        fwd: new THREE.Vector3(),
      },
      user: {
        headPos: new THREE.Vector3(),
        headFwd: new THREE.Vector3(),
      },
      // Anchor uses its own scratch Vector3 so callers can pass in any
      // THREE.Vector3-ish and we never alias their storage. When the
      // caller passes `anchor: null`, we set `_scratchCtx.anchor = null`
      // by replacing the field; the scratch vector is retained on
      // `_anchorVec` so we can re-attach next frame without allocating.
      anchor: null,
      _anchorVec: new THREE.Vector3(),
      affect: {
        presence:    0,
        flow:        0,
        temperature: 0,
        resonance:   0,
      },
      intent: null,
      intentOverride: false,
    };
  }

  /**
   * Reset internal bookkeeping. Safe to call mid-session — invoked by
   * the host when the proxemic flag toggles or the scene resets.
   * Does NOT touch the FSM / PresenceEngine / fformation references
   * those are owned by the caller and reset on their own lifecycle.
   */
  reset() {
    this._lastVerb      = 'hold';
    this._lastFormation = FORMATION.NONE;
    this._slmIntent     = null;
    // Reset scratch intent to the "hold" default so a stale verb cannot
    // leak into the first post-reset frame's telemetry.
    const i = this._scratchIntent;
    i.noop      = false;
    i.formation = FORMATION.NONE;
    i.side      = 'either';
    i.zone      = 'interpersonal';
    i.verb      = 'hold';
    i.gaze.target    = 'user_eyes';
    i.gaze.mutual_ms = 0;
    i.gesture_sync.mode = 'none';
    i.gesture_sync.role = 'none';
    i.anchor     = null;
    i.seat_id    = null;
    i.urgency    = 0.0;
    i.confidence = CONFIDENCE.HOLD;
    i.ttl_ms     = TTL.HOLD_MS;
    i.reason     = 'reset';
  }

  /**
   * Phase-3 hook (NOT used in Phase 1).
   *
   * Reserved entry point for the SLM intent blender. In Phase 1 this is
   * a no-op stub so callers wired against the eventual API don't crash;
   * Phase 3 will validate, version-check, and blend with the heuristic
   * path. Documenting the seam here is deliberate — it keeps the public
   * surface stable across the phase boundary.
   *
   * @param {object|null} _intent  schema-shaped intent emitted by the SLM
   */
  setSlmIntent(_intent) {
    // Phase 3 stub. Intentionally unimplemented in Sprint 1; the
    // heuristic path is the only source of truth in this phase.
    this._slmIntent = null;
  }

  /**
   * Run one tick of the director.
   *
   * The returned `intent` and `fformationCtx` are references to internal
   * scratch buffers; the caller MUST consume them within the same tick
   * (typically: pass `fformationCtx` directly to `fformation.update()`
   * and read `intent` for telemetry / FSM routing) and MUST NOT retain
   * them across frames. The third returned field, `fsmRequest`, is
   * either `null` (no transition requested this tick) or a plain object
   * `{to, opts}` — the caller is expected to pass these to
   * `fsm.requestTransition(to, opts)`.
   *
   * Phase 1 explicitly does NOT request the `Locomoting` state — that
   * transition becomes available once plane-aware navigation lands in
   * a later phase (see design spec §"Phased roadmap"). The director
   * deliberately stays within the seated posture lattice for now.
   *
   * @param {{
   *   self: {position: THREE.Vector3, forward: THREE.Vector3},
   *   user: {position: THREE.Vector3, forward: THREE.Vector3},
   *   anchor: THREE.Vector3 | null,
   *   silenceSeconds: number,
   *   emotion: string | null,
   * }} ctx
   * @param {number} dt  seconds since last tick (clamped internally to [0, 0.25])
   * @returns {{
   *   intent: object,
   *   fformationCtx: object,
   *   fsmRequest: {to: string, opts?: object} | null,
   * }}
   *   `intent` and `fformationCtx` are scratch references owned by the
   *   director — do not retain across frames.
   */
  update(ctx, dt) {
    // Clamp dt — paused-tab defense, mirrors fformation.js.
    if (!Number.isFinite(dt) || dt < 0) dt = 0;
    if (dt > 0.25) dt = 0.25;

    const intent       = this._scratchIntent;
    const fformationCtx = this._scratchCtx;

    // Read presence scalars (default to neutral when engine missing).
    const p  = this._presence;
    const presence    = _clamp(p?.presence,    0, 1,  0.5);
    const flow        = _clamp(p?.flow,       -1, 1,  0.0);
    const temperature = _clamp(p?.temperature, 0, 1,  0.2);
    const resonance   = _clamp(p?.resonance,   0, 1,  0.0);

    const silenceSeconds = Number.isFinite(ctx?.silenceSeconds)
      ? Math.max(0, ctx.silenceSeconds)
      : 0;

    // ── Apply heuristic rule table (first match wins) ─────────────────

    if (temperature > THRESH.TEMP_LEAN_IN && resonance > THRESH.RESO_LEAN_IN) {
      // High warmth + high resonance → lean in.
      // Positive-valence proxy: temperature > TEMP_POSITIVE picks vis-à-vis;
      // otherwise the gentler l-shape.
      const positive = temperature > THRESH.TEMP_POSITIVE;
      this._fillLeanIn(
        intent,
        positive ? FORMATION.VIS_A_VIS : FORMATION.L_SHAPE,
        'warm+resonant',
      );
    } else if (flow > THRESH.FLOW_EMPHATIC && resonance > THRESH.RESO_EMPHATIC) {
      // Emphatic affirmation — strong outgoing flow plus resonance.
      this._fillLeanIn(intent, FORMATION.VIS_A_VIS, 'emphatic affirm');
    } else if (
      temperature < THRESH.TEMP_SETTLE
      && presence < THRESH.PRES_SETTLE
      && silenceSeconds > THRESH.SILENCE_SETTLE_S
    ) {
      // Cool + low presence + sustained silence → settle, give space.
      this._fillSettle(intent, 'cool+silent');
    } else if (
      silenceSeconds > THRESH.SILENCE_IDLE_S
      && presence > THRESH.PRES_IDLE_DWELL
    ) {
      // Long silence with the user still notionally present → idle dwell
      // (no-op; previous formation holds).
      this._fillIdleDwell(intent, 'idle dwell');
    } else {
      // Default — hold posture / formation.
      this._fillHold(intent, 'hold');
    }

    this._lastVerb      = intent.verb;
    this._lastFormation = intent.formation;

    // ── Fill the fformation ctx (zero-allocation field writes) ────────
    this._fillFformationCtx(
      fformationCtx,
      ctx,
      presence,
      flow,
      temperature,
      resonance,
      intent,
    );

    // ── Decide whether to request an FSM transition ───────────────────
    const fsmRequest = this._deriveFsmRequest(intent);

    return { intent, fformationCtx, fsmRequest };
  }

  // ─── Private fill helpers (each mutates `intent` in place) ───────────

  /**
   * Populate `intent` for a `lean_in` emission.
   * @param {object} intent  scratch intent object
   * @param {string} formation  one of FORMATION values
   * @param {string} reason  short telemetry tag (≤ 48 chars)
   */
  _fillLeanIn(intent, formation, reason) {
    intent.noop      = false;
    intent.formation = formation;
    intent.side      = 'either';
    intent.zone      = 'interpersonal';
    intent.verb      = 'lean_in';
    intent.gaze.target    = 'user_eyes';
    intent.gaze.mutual_ms = 1500;
    intent.gesture_sync.mode = 'matching';
    intent.gesture_sync.role = 'responder';
    intent.anchor    = null;
    intent.seat_id   = null;
    intent.urgency   = 0.55;
    intent.confidence = CONFIDENCE.LEAN_IN;
    intent.ttl_ms    = TTL.LEAN_IN_MS;
    intent.reason    = _capReason(reason);
  }

  /**
   * Populate `intent` for a `settle` emission (low engagement, give space).
   * @param {object} intent
   * @param {string} reason
   */
  _fillSettle(intent, reason) {
    intent.noop      = false;
    intent.formation = FORMATION.SIDE_BY_SIDE;
    intent.side      = 'either';
    intent.zone      = 'interpersonal';
    intent.verb      = 'settle';
    intent.gaze.target    = 'ambient';
    intent.gaze.mutual_ms = 0;
    intent.gesture_sync.mode = 'none';
    intent.gesture_sync.role = 'none';
    intent.anchor    = null;
    intent.seat_id   = null;
    intent.urgency   = 0.2;
    intent.confidence = CONFIDENCE.SETTLE;
    intent.ttl_ms    = TTL.SETTLE_MS;
    intent.reason    = _capReason(reason);
  }

  /**
   * Populate `intent` for an idle-dwell no-op (long silence; hold prior).
   * Sets `noop=true` per the schema — consumers preserve the previous
   * formation and skip transition requests.
   * @param {object} intent
   * @param {string} reason
   */
  _fillIdleDwell(intent, reason) {
    intent.noop      = true;
    // Keep the last formation visible so callers reading intent.formation
    // for telemetry see a stable value across the dwell.
    intent.formation = this._lastFormation || FORMATION.NONE;
    intent.side      = 'either';
    intent.zone      = 'interpersonal';
    intent.verb      = 'hold';
    intent.gaze.target    = 'ambient';
    intent.gaze.mutual_ms = 0;
    intent.gesture_sync.mode = 'none';
    intent.gesture_sync.role = 'none';
    intent.anchor    = null;
    intent.seat_id   = null;
    intent.urgency   = 0.05;
    intent.confidence = CONFIDENCE.DWELL;
    intent.ttl_ms    = TTL.IDLE_DWELL_MS;
    intent.reason    = _capReason(reason);
  }

  /**
   * Populate `intent` for the default `hold` emission.
   * @param {object} intent
   * @param {string} reason
   */
  _fillHold(intent, reason) {
    intent.noop      = false;
    intent.formation = FORMATION.NONE;
    intent.side      = 'either';
    intent.zone      = 'interpersonal';
    intent.verb      = 'hold';
    intent.gaze.target    = 'user_eyes';
    intent.gaze.mutual_ms = 0;
    intent.gesture_sync.mode = 'none';
    intent.gesture_sync.role = 'none';
    intent.anchor    = null;
    intent.seat_id   = null;
    intent.urgency   = 0.1;
    intent.confidence = CONFIDENCE.HOLD;
    intent.ttl_ms    = TTL.HOLD_MS;
    intent.reason    = _capReason(reason);
  }

  /**
   * Write the per-frame fformation ctx from the host-supplied poses and
   * the just-emitted spatial intent. Mutates `fformationCtx` in place;
   * never allocates.
   *
   * Maps our public `ctx.self/user` shape (`position` + `forward`) onto
   * the keys consumed by `FFormationController.update()`
   * (`self.pos/yaw/fwd`, `user.headPos/headFwd`).
   *
   * @param {object} fformationCtx  scratch ctx object
   * @param {object} ctx  host-supplied frame ctx (see `update` JSDoc)
   * @param {number} presence
   * @param {number} flow
   * @param {number} temperature
   * @param {number} resonance
   * @param {object} intent  freshly-filled scratch intent
   */
  _fillFformationCtx(fformationCtx, ctx, presence, flow, temperature, resonance, intent) {
    const sPos = ctx?.self?.position;
    const sFwd = ctx?.self?.forward;
    const uPos = ctx?.user?.position;
    const uFwd = ctx?.user?.forward;

    // Self pose. Defensive copies so a caller mutating their input on
    // the next tick can't desync our scratch.
    if (sPos) {
      fformationCtx.self.pos.set(sPos.x, sPos.y, sPos.z);
    } else {
      fformationCtx.self.pos.set(0, 0, 0);
    }
    if (sFwd) {
      fformationCtx.self.fwd.set(sFwd.x, sFwd.y, sFwd.z);
      // Derive yaw on the XZ ground plane from forward. atan2(x, z) so
      // yaw=0 faces +Z, matching fformation.js's convention
      // (see _detectFormation / desiredYaw computation there).
      fformationCtx.self.yaw = Math.atan2(sFwd.x, sFwd.z);
    } else {
      fformationCtx.self.fwd.set(0, 0, 1);
      fformationCtx.self.yaw = 0;
    }

    // User head pose.
    if (uPos) {
      fformationCtx.user.headPos.set(uPos.x, uPos.y, uPos.z);
    } else {
      fformationCtx.user.headPos.set(0, 0, 0);
    }
    if (uFwd) {
      fformationCtx.user.headFwd.set(uFwd.x, uFwd.y, uFwd.z);
    } else {
      fformationCtx.user.headFwd.set(0, 0, -1);
    }

    // Anchor (optional). Copy into our owned scratch vector and point
    // `anchor` at the canonical `{pos}` shape fformation expects.
    if (ctx?.anchor) {
      fformationCtx._anchorVec.set(ctx.anchor.x, ctx.anchor.y, ctx.anchor.z);
      // Reuse a stable wrapper object so we don't allocate every frame.
      if (!fformationCtx.anchor || fformationCtx.anchor.pos !== fformationCtx._anchorVec) {
        fformationCtx.anchor = { pos: fformationCtx._anchorVec };
      }
    } else {
      fformationCtx.anchor = null;
    }

    // Affect block — direct field writes, no allocation.
    fformationCtx.affect.presence    = presence;
    fformationCtx.affect.flow        = flow;
    fformationCtx.affect.temperature = temperature;
    fformationCtx.affect.resonance   = resonance;

    // Intent → fformation's symbolic-intent string (subset). `noop` and
    // `hold` map to no override; otherwise we translate by formation.
    if (intent.noop || intent.verb === 'hold') {
      fformationCtx.intent = null;
      fformationCtx.intentOverride = false;
    } else {
      fformationCtx.intent = this._formationToSymbolicIntent(intent.formation);
      // We override only when we actually have a non-NONE formation; lets
      // fformation's detection do its job in ambiguous cases.
      fformationCtx.intentOverride =
        intent.formation !== FORMATION.NONE
        && fformationCtx.intent != null;
    }
  }

  /**
   * Translate a FORMATION enum into the closest INTENT enum string that
   * `fformation.mapIntentToFormation()` can round-trip. Returns `null`
   * for FORMATION.NONE (no override) so the detector picks the geometry.
   *
   * @param {string} formation
   * @returns {string|null}
   */
  _formationToSymbolicIntent(formation) {
    switch (formation) {
      case FORMATION.VIS_A_VIS:    return INTENT.INTIMATE;
      case FORMATION.L_SHAPE:      return INTENT.COLLABORATIVE;
      case FORMATION.SIDE_BY_SIDE: return INTENT.COMPANIONABLE;
      default:                     return null;
    }
    // Note: this is the inverse of fformation.js's
    // `mapIntentToFormation` — the seam is kept explicit on both sides.
  }

  /**
   * Decide whether the host should request an FSM posture transition
   * given the just-emitted spatial intent. Returns `null` when no
   * request is warranted (the common case).
   *
   * Rules (Phase 1):
   *   - `verb=lean_in` && current FSM state is `SeatedDefault`
   *       → request `SeatedLeaning`.
   *   - `verb=settle` && current FSM state is NOT already a seated-back
   *       posture and NOT the default → request `SeatedBack`.
   *   - `intent.noop`         → never request (preserve current state).
   *   - otherwise             → no request.
   *
   * Phase 1 deliberately does NOT request `Locomoting`; the seated
   * posture lattice is the entire transition surface until plane-aware
   * navigation lands.
   *
   * @param {object} intent  freshly-filled scratch intent
   * @returns {{to: string, opts?: object} | null}
   */
  _deriveFsmRequest(intent) {
    if (!this._fsm || intent.noop) return null;

    const cur = this._fsm.current?.();
    if (!cur || typeof cur.name !== 'string') return null;
    const name = cur.name;

    if (intent.verb === 'lean_in' && name === FSM_STATES.SEATED_DEFAULT) {
      return {
        to: FSM_STATES.SEATED_LEANING,
        opts: { reason: 'director:lean_in' },
      };
    }

    if (
      intent.verb === 'settle'
      && name !== FSM_STATES.SEATED_DEFAULT
      && name !== FSM_STATES.SEATED_BACK
    ) {
      return {
        to: FSM_STATES.SEATED_BACK,
        opts: { reason: 'director:settle' },
      };
    }

    // Phase 1: Locomoting is intentionally NOT a director-emitted target.
    // Plane-aware navigation (see design spec, Phase 4) gates that path.

    return null;
  }

  /**
   * Diagnostics snapshot for the avatar-pose-harness testbench. Allocates
   * freely; do NOT call from the hot path.
   * @returns {object}
   */
  _dbg() {
    return {
      lastVerb:      this._lastVerb,
      lastFormation: this._lastFormation,
      hasPresence:   !!this._presence,
      hasFsm:        !!this._fsm,
      hasFformation: !!this._fformation,
      slmIntent:     this._slmIntent ? '<set>' : null,
      thresholds:    { ...THRESH },
    };
  }
}

// Re-export the FORMATION / FSM_STATES references the director consumes,
// so a caller wiring up the director can grab everything from one import.
export { FORMATION, FSM_STATES };
