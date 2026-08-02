/**
 * avatar-xr-contact.js — production XR/desktop integration of ContactReactor.
 *
 * Self-contained module wired into avatar-xr.js's session lifecycle AND
 * avatar-body-physics-desktop.js's render-loop wiring. When the user is
 * in VR/MR, their controllers or tracked hands feed the ContactReactor;
 * on desktop/mobile, contact-desktop-fallback.js feeds the reactor via
 * pointer events instead. Pass `inputMode: 'desktop'` to init to skip
 * the XR controller/hand binding (the reactor is then fed externally
 * via reactor.setUserHand()).
 *
 * The reactor queries the AI VRM's BodyAtlas + BodyMesh to detect
 * approach / hover / contact; contact events fire expression spikes
 * on the VRM's expression manager AND optionally drive the AI's arm
 * via AvatarIK to reach toward the user.
 *
 * Lifecycle (called from avatar-xr.js):
 *
 *   initXRContact({ three, vrm, renderer, ik? })   // on session start, after _prepareVrmSceneForXR
 *   tickXRContact(dtMs)                            // every frame inside _updateXrInteractions
 *   teardownXRContact()                            // on session end
 *
 * Decoupled from production's existing _updateHandInputs / hand-state
 * machinery — we read controllers/hands directly off renderer.xr so
 * both can coexist. The reactor reads vrm.__augmentumBodyMesh /
 * vrm.__augmentumBodyAtlas which avatar.js already populates at load.
 *
 * If `ik` is omitted at init, the reactor still runs but won't drive
 * reach-toward-user motion — only expression spikes fire on contact.
 * Production can pass its own AvatarIK instance for full behavior.
 */

import { ContactReactor } from './contact-reactor.js';
import { AvatarIK } from './avatar-ik.js';
import { handleAudioContact } from './avatar-audio-reactions.js';

let _state = null;

// Visual-feedback gating: `body_physics_visual_feedback_enabled` controls
// the layered region-specific morphs added on top of the base spike calls.
// Defaults to enabled; the fetch is fire-and-forget so it never blocks
// init or contact handling — pre-fetch responses are treated as enabled.
let _visualFeedbackEnabled = true;
let _visualFeedbackFetched = false;

function _ensureVisualFeedbackSetting() {
  if (_visualFeedbackFetched) return;
  _visualFeedbackFetched = true;
  try {
    fetch('/api/config/ui')
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (!data) return;
        const raw = data['body_physics_visual_feedback_enabled'];
        if (raw === undefined || raw === null) return;
        if (raw === false || raw === 'false' || raw === '0' || raw === 0) {
          _visualFeedbackEnabled = false;
        } else {
          _visualFeedbackEnabled = true;
        }
      })
      .catch(() => { /* leave default (enabled) */ });
  } catch { /* leave default (enabled) */ }
}

function _hasExpression(vrm, name) {
  try { return !!vrm?.expressionManager?.getExpressionTrackName?.(name); }
  catch { return false; }
}

/**
 * Initialize contact reactor for an XR session. Idempotent — calling
 * again replaces the previous reactor (e.g. after VRM swap mid-session).
 *
 * @param {object} opts
 * @param {object} opts.three            THREE namespace
 * @param {object} opts.vrm              VRM with BodyMesh + (optional) BodyAtlas
 * @param {object} opts.renderer         WebGLRenderer with xr.enabled
 * @param {object} [opts.ik]             AvatarIK instance; if absent, one is
 *                                       constructed automatically (best-effort)
 * @returns {ContactReactor|null}        the reactor, or null if init failed
 */
export function initXRContact(opts) {
  // Kick off the visual-feedback setting fetch on first init. It runs
  // async; until it resolves we treat the feature as enabled (default).
  _ensureVisualFeedbackSetting();
  const { three, vrm, renderer } = opts;
  const inputMode = opts.inputMode === 'desktop' ? 'desktop' : 'xr';
  if (!three || !vrm || !renderer) {
    console.warn('[xr-contact] init missing required deps (three/vrm/renderer)');
    return null;
  }
  if (!vrm.__augmentumBodyMesh) {
    console.warn('[xr-contact] vrm has no BodyMesh — contact detection disabled');
    return null;
  }
  // Tear down any prior state — important when VRM swaps mid-session.
  teardownXRContact();

  // Reuse a passed-in IK; otherwise auto-construct one unless the caller
  // explicitly opts out via `disableReach: true`. When reach is disabled,
  // only the soft response stack fires (compliance indent + rapier sway +
  // expression spikes + audio cues) — no arm IK targeting the toucher.
  //
  // Why disabling reach is sometimes correct: ContactReactor's reach drives
  // the avatar's nearest arm toward the user-hand world position via IK.
  // When the target is INSIDE the avatar's own body footprint (e.g. user
  // pokes the stomach), no valid IK pose exists — the shoulder/elbow chain
  // crunches into an anatomically-broken silhouette. VRChat PhysBones,
  // VRoid spring bones, Resonite ragdolls, and the ECA literature all
  // converge on the same answer: don't auto-reach toward the toucher,
  // just let the body respond softly + animate region-keyed flinches.
  // Desktop path opts out by default; XR retains reach for now (distance
  // reaches still make sense when the user's hand is across the room).
  let ik = opts.ik || null;
  if (!ik && !opts.disableReach) {
    try {
      ik = new AvatarIK({
        three, vrm,
        poleHint: { L: [-1, -0.5, +1], R: [+1, -0.5, +1] },
      });
    } catch (err) {
      console.warn('[xr-contact] AvatarIK construction failed (reach disabled):', err?.message);
    }
  }

  // Bind controllers + hands (XR mode only). In desktop mode the reactor
  // is fed externally via reactor.setUserHand() from contact-desktop-
  // fallback.js's pointer events.
  let controllers = [];
  let hands = [];
  const detachers = [];
  if (inputMode === 'xr') {
    controllers = [
      renderer.xr.getController(0),
      renderer.xr.getController(1),
    ];
    hands = [
      renderer.xr.getHand(0),
      renderer.xr.getHand(1),
    ];
    for (const c of [...controllers, ...hands]) {
      c.userData.handedness = c.userData.handedness || null;
      const onConnected = (e) => {
        c.userData.handedness = e.data?.handedness || null;
        console.debug('[xr-contact] input connected:', c.userData.handedness, e.data?.targetRayMode);
      };
      const onDisconnected = () => { c.userData.handedness = null; };
      c.addEventListener('connected', onConnected);
      c.addEventListener('disconnected', onDisconnected);
      detachers.push(() => {
        c.removeEventListener('connected', onConnected);
        c.removeEventListener('disconnected', onDisconnected);
      });
    }

    // Late-bind: if the session is already presenting when init runs (e.g.
    // VRM swapped mid-session, or session started before BodyMesh was ready),
    // the 'connected' events have already fired. Read current input sources
    // out of the active session and seed handedness manually.
    try {
      const session = renderer.xr.getSession?.();
      if (session?.inputSources) {
        for (const src of session.inputSources) {
          const idx = src.targetRayMode === 'tracked-pointer'
            ? Array.from(session.inputSources).indexOf(src)
            : -1;
          if (idx < 0 || idx > 1) continue;
          if (src.handedness === 'left' || src.handedness === 'right') {
            controllers[idx].userData.handedness = src.handedness;
            if (src.hand) hands[idx].userData.handedness = src.handedness;
            console.debug('[xr-contact] late-bound input:', src.handedness, src.targetRayMode);
          }
        }
      }
    } catch (err) { console.debug('[xr-contact] late-bind failed:', err?.message); }
  }

  console.debug('[xr-contact] initialized with', {
    inputMode,
    hasBodyMesh: !!vrm.__augmentumBodyMesh,
    hasBodyAtlas: !!vrm.__augmentumBodyAtlas,
    hasIK: !!ik,
    hasExpressionManager: !!vrm.expressionManager,
    triCount: vrm.__augmentumBodyMesh?.triangleCount,
  });

  const recentSpikes = new Map();

  const reactor = new ContactReactor({
    three, vrm,
    bodyMesh: vrm.__augmentumBodyMesh,
    bodyAtlas: vrm.__augmentumBodyAtlas,
    ik,
    embodiment: {
      onContactEvent: (evt) => {
        console.debug('[xr-contact] contact event:', evt.state, evt.region, `(${evt.userSide})`, evt.released ? 'released' : '');
        _handleContact(evt, vrm, recentSpikes);
        // Audio cues: region-keyed WebAudio reactions. Own debounce + gate
        // inside avatar-audio-reactions.js; no-op if the module isn't init'd.
        try { handleAudioContact(evt); }
        catch (err) { console.debug('[xr-contact] audio reaction failed:', err?.message); }
      },
    },
    onLog: (e) => {
      // State transitions only (debounced) — every frame would spam.
      if (e.state === 'approach' || e.state === 'contact') {
        console.debug('[xr-contact] state', e.userSide, e.prev, '→', e.state, e.region);
      }
    },
  });

  _state = {
    three, vrm, renderer, reactor, ik,
    controllers, hands, detachers, recentSpikes,
    inputMode,
    tmpVec: new three.Vector3(),
  };
  return reactor;
}

/**
 * Tick the reactor. In XR mode, reads controller/hand world positions and
 * feeds them in. In desktop mode, user hands are already set externally
 * via reactor.setUserHand(); we just tick the reactor and decay spikes.
 * Always safe to call — no-ops when no state, or when XR mode requires
 * a presenting session that isn't active.
 *
 * @param {number} dtMs   frame delta in milliseconds
 */
export function tickXRContact(dtMs) {
  if (!_state) return;
  const { renderer, controllers, hands, reactor, tmpVec, recentSpikes, vrm, ik, inputMode } = _state;

  // XR mode: read controller/hand positions and feed the reactor. Skipped
  // in desktop mode — contact-desktop-fallback.js feeds user hands directly
  // via reactor.setUserHand() from pointer events.
  let leftPos = null, rightPos = null;
  if (inputMode === 'xr') {
    if (!renderer.xr?.isPresenting) return;
    // Read wrist world positions per handedness. Controllers take
    // priority; hand-tracking joints fill in any side controllers
    // didn't provide (e.g. user dropped one controller).
    for (const c of controllers) {
      const h = c.userData.handedness;
      if (h !== 'left' && h !== 'right') continue;
      c.getWorldPosition(tmpVec);
      const pos = [tmpVec.x, tmpVec.y, tmpVec.z];
      if (h === 'left')  leftPos  = pos;
      else               rightPos = pos;
    }
    for (const hand of hands) {
      const h = hand.userData.handedness;
      if (h !== 'left' && h !== 'right') continue;
      const wrist = hand.joints?.['wrist'];
      if (!wrist) continue;
      wrist.getWorldPosition(tmpVec);
      const pos = [tmpVec.x, tmpVec.y, tmpVec.z];
      if (h === 'left' && !leftPos)   leftPos  = pos;
      if (h === 'right' && !rightPos) rightPos = pos;
    }
    reactor.setUserHands([leftPos, rightPos]);
  }

  reactor.tick(dtMs);

  // Sampled diagnostic: log once per second so the activity stream
  // shows the reactor is alive + what state it's seeing.
  _state.diagFrameCount = (_state.diagFrameCount || 0) + 1;
  if (_state.diagFrameCount >= 60) {
    _state.diagFrameCount = 0;
    const inspect = reactor.inspect();
    console.debug('[xr-contact] tick', {
      L: leftPos ? `(${leftPos.map(v => v.toFixed(2)).join(',')})` : null,
      R: rightPos ? `(${rightPos.map(v => v.toFixed(2)).join(',')})` : null,
      state: inspect.userState,
      lastContact: {
        L: inspect.lastContact.L ? `${inspect.lastContact.L.region} ${inspect.lastContact.L.distance.toFixed(2)}m` : null,
        R: inspect.lastContact.R ? `${inspect.lastContact.R.region} ${inspect.lastContact.R.distance.toFixed(2)}m` : null,
      },
    });
  }

  // Drive IK update when the reactor is actively reaching so the arm
  // bones reflect the new target this frame. AvatarIK writes normalized
  // humanoid bones; production's vrm.update later pushes those to the
  // raw skeleton during render.
  if (ik && reactor.isActivelyReaching) {
    ik.update?.(Math.min(0.05, dtMs / 1000));
  }

  _decaySpikes(vrm, recentSpikes);
}

/**
 * Tear down on session end. Resets any in-flight expression spikes,
 * detaches controller listeners, drops references.
 */
export function teardownXRContact() {
  if (!_state) return;
  // Cancel any pending blink-close timers so they don't fire into a
  // torn-down or swapped VRM after teardown completes.
  for (const t of _pendingBlinkTimers) {
    try { clearTimeout(t); } catch {}
  }
  _pendingBlinkTimers.clear();
  // Reset any in-flight expression spikes so we don't leave the
  // avatar with a half-applied morph after the session ends.
  if (_state.vrm?.expressionManager) {
    for (const name of _state.recentSpikes.keys()) {
      try { _state.vrm.expressionManager.setValue(name, 0); } catch {}
    }
  }
  for (const detach of _state.detachers) detach();
  _state = null;
}

/** Get the active reactor for introspection (debug HUDs, telemetry). */
export function getXRContactReactor() {
  return _state?.reactor || null;
}

// ─── Internal: contact → expression mapping ─────────────────────────────
function _handleContact(evt, vrm, recentSpikes) {
  if (evt.released) return;
  const region = evt.region || '';
  // Region-keyed reaction policy — matches the bench's embodiment.onContactEvent
  // for consistency, but writes directly to expressionManager without going
  // through the full mood-vector decay model (production keeps the existing
  // narrative/mood pipeline as the long-term emotional state).
  if (/^(cheek|forehead|chin|mouth|temple|jaw|nose)/.test(region)) {
    _spike(vrm, recentSpikes, 'surprised', 0.50, 1500);
    _spike(vrm, recentSpikes, 'happy',     0.35, 2500);
    // Visual feedback (body_physics_visual_feedback_enabled): layered region-specific morphs.
    if (_visualFeedbackEnabled) {
      // Quick blink: snap to 0.8, then schedule a hard reset 200ms later.
      // The decay loop also pulls it down, but the chained spike guarantees
      // a clean close even if decay timing drifts.
      if (_hasExpression(vrm, 'blink')) {
        _spike(vrm, recentSpikes, 'blink', 0.8, 200);
        _scheduleBlinkClose(vrm, recentSpikes, 200);
      }
      if (_hasExpression(vrm, 'blush')) {
        _spike(vrm, recentSpikes, 'blush', 0.5, 2500);
      }
    }
  } else if (/^shoulder_/.test(region)) {
    // Shoulder touch = warm/relaxed. No mouth shape — the previous draft
    // mixed `relaxed` with chest's `aa` for every body region which made
    // every poke look like the same gasp regardless of where you touched.
    _spike(vrm, recentSpikes, 'happy', 0.30, 2000);
    if (_visualFeedbackEnabled) {
      if (_hasExpression(vrm, 'relaxed')) {
        _spike(vrm, recentSpikes, 'relaxed', 0.25, 1500);
      }
    }
  } else if (/^hand_/.test(region)) {
    // Hand touch = warm smile. Closed-mouth — opening the mouth on a
    // hand touch reads as weird/inappropriate. Dropped the `aa`/`oh`
    // viseme that the previous draft added.
    _spike(vrm, recentSpikes, 'happy', 0.40, 2500);
  } else if (/^(chest_|sternum|back_upper)/.test(region)) {
    // Chest touch = stronger surprise. The `surprised` morph on most VRMs
    // already opens the mouth slightly; explicit `aa` on top fights it
    // and produced the "same gasp regardless of region" issue. Drop the
    // explicit viseme — let the emotion morph carry the mouth shape.
    _spike(vrm, recentSpikes, 'surprised', 0.55, 1500);
  } else if (/^(hip_|belly|navel|thigh_|side_)/.test(region)) {
    // Hip/belly touch = surprise + flash of annoyance — distinctly NOT
    // a gasp. The `angry` morph pulls the mouth closed and lowers the
    // brows, differentiating this from a chest poke.
    _spike(vrm, recentSpikes, 'surprised', 0.45, 2000);
    _spike(vrm, recentSpikes, 'angry',     0.25, 2000);
  } else {
    _spike(vrm, recentSpikes, 'surprised', 0.30, 1500);
  }
}

// Track any pending blink-close timers so teardown can clear them and
// we don't leave a stale setValue(0) firing after the VRM was swapped.
const _pendingBlinkTimers = new Set();

function _scheduleBlinkClose(vrm, recentSpikes, delayMs) {
  let timer;
  timer = setTimeout(() => {
    _pendingBlinkTimers.delete(timer);
    // Only fire if the same VRM is still the active one — teardown
    // clears _state, so guard against firing into a torn-down session.
    if (!_state || _state.vrm !== vrm) return;
    _spike(vrm, recentSpikes, 'blink', 0, 200);
  }, delayMs);
  _pendingBlinkTimers.add(timer);
}

function _spike(vrm, recentSpikes, name, value, durationMs) {
  if (!vrm?.expressionManager) return;
  const now = performance.now();
  recentSpikes.set(name, {
    value, startAt: now, expireAt: now + durationMs,
  });
  try { vrm.expressionManager.setValue(name, value); } catch {}
}

function _decaySpikes(vrm, recentSpikes) {
  if (!vrm?.expressionManager) return;
  const now = performance.now();
  for (const [name, spike] of recentSpikes) {
    if (now >= spike.expireAt) {
      try { vrm.expressionManager.setValue(name, 0); } catch {}
      recentSpikes.delete(name);
      continue;
    }
    const t = (now - spike.startAt) / (spike.expireAt - spike.startAt);
    const decayed = spike.value * (1 - t);
    try { vrm.expressionManager.setValue(name, decayed); } catch {}
  }
}
