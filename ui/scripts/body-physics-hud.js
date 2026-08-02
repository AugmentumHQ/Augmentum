/**
 * body-physics-hud.js — developer-facing debug overlay for the hybrid
 * body-physics stack (SDF compliance + Rapier ragdoll + contact reactor),
 * plus an adaptive quality controller that auto-tunes channel parameters
 * when frame time slips under load.
 *
 * Pairs with avatar-xr-compliance.js, avatar-xr-rapier.js,
 * avatar-xr-contact.js, and body-physics-coordinator.js. Reads only —
 * never mutates channel state directly. When adaptive quality fires, it
 * steers the coordinator's HUD overrides (`setOverride`) rather than
 * poking channel knobs, so server-side settings stay the source of
 * truth and overrides are cleared automatically when load subsides.
 *
 * Hidden by default; toggle with Ctrl+Shift+B. The 200ms render loop is
 * suspended when hidden. The rAF frame sampler runs continuously so the
 * sampler is already warm when the developer opens the HUD.
 */

import { getXRCompliance } from './avatar-xr-compliance.js';
import { getXRContactReactor } from './avatar-xr-contact.js';
import { escapeHtml } from './app.js';

// ─── Late-bound peer accessors ───────────────────────────────────────────────
// avatar-xr-rapier.js and body-physics-coordinator.js may not exist in every
// build. Each accessor is resolved once via dynamic import; if the import
// fails the accessor stays a stub returning null.
/** @type {() => any} */ let _getXRRapier = () => null;
/** @type {() => any} */ let _getBodyPhysicsCoordinator = () => null;

(async () => {
  try {
    const mod = await import('./avatar-xr-rapier.js');
    if (typeof mod?.getXRRapier === 'function') _getXRRapier = mod.getXRRapier;
  } catch { /* compliance-only build */ }
  try {
    const mod = await import('./body-physics-coordinator.js');
    if (typeof mod?.getBodyPhysicsCoordinator === 'function') {
      _getBodyPhysicsCoordinator = mod.getBodyPhysicsCoordinator;
    }
  } catch { /* no coordinator available */ }
})();

// ─── Tunables ────────────────────────────────────────────────────────────────
const POLL_MS              = 200;
const FRAME_HISTORY        = 60;
const STRESS_MS_THRESHOLD  = 18;   // > 18ms avg ⇒ under 55fps
const RELIEF_MS_THRESHOLD  = 14;   // < 14ms avg ⇒ above ~71fps
const STRESS_HOLD_MS       = 1000; // sustained for 1s before reducing
const RELIEF_HOLD_MS       = 2000; // sustained for 2s before recovering
const ADJUST_INTERVAL_MS   = 1000; // one step per second while in reducing/recovering
const COMPLY_FLOOR         = 0.3;
const RAPIER_WEIGHT_FLOOR  = 0.2;
const STEP_DOWN_FACTOR     = 0.9;
const STEP_UP_FACTOR       = 1.05;
const NOMINAL_COMPLY_GAIN  = 1.0;
const NOMINAL_RAPIER_WT    = 0.6;
const HOTKEY = { code: 'KeyB', ctrl: true, shift: true };

// ─── Module-singleton instance ───────────────────────────────────────────────
/** @type {BodyPhysicsHUD|null} */
let _hud = null;

/**
 * Continuous frame-time sampler. A self-rescheduling rAF closure feeds a
 * 60-deep ring buffer; `snapshot()` returns the rolling average. Module-
 * scope so the buffer is warm by the time the HUD opens.
 */
const _frameSampler = (() => {
  const buf = new Float32Array(FRAME_HISTORY);
  let idx = 0;
  let filled = 0;
  let last = (typeof performance !== 'undefined') ? performance.now() : Date.now();
  let rafId = 0;

  function tick(now) {
    const dt = now - last;
    last = now;
    if (dt > 0 && dt < 500) {  // ignore tab-throttle gaps that would skew avg
      buf[idx] = dt;
      idx = (idx + 1) % FRAME_HISTORY;
      if (filled < FRAME_HISTORY) filled++;
    }
    rafId = requestAnimationFrame(tick);
  }
  if (typeof requestAnimationFrame === 'function') {
    rafId = requestAnimationFrame(tick);
  }

  return {
    /** @returns {{ avgMs:number, fps:number }} */
    snapshot() {
      if (!filled) return { avgMs: 0, fps: 0 };
      let sum = 0;
      for (let i = 0; i < filled; i++) sum += buf[i];
      const avgMs = sum / filled;
      const fps = avgMs > 0 ? 1000 / avgMs : 0;
      return { avgMs, fps };
    },
    /** For teardown — stop scheduling further rAFs. The buffer is GC'd with the closure. */
    stop() { if (rafId && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(rafId); },
  };
})();

// ─── Adaptive quality controller ─────────────────────────────────────────────
/**
 * State machine over the coordinator's HUD overrides. Listens to the frame
 * sampler; when avg frame-time crosses thresholds for sustained windows, steps
 * `compliance_gain` and `rapier_weight` down (under load) or back up (relief),
 * clearing overrides entirely once both are back to nominal so the coordinator
 * resumes server-managed values.
 */
class AdaptiveQualityController {
  constructor() {
    /** 'nominal' | 'reducing' | 'recovering' */
    this.mode = 'nominal';
    this._stressSince = 0;
    this._reliefSince = 0;
    this._lastAdjustAt = 0;
  }

  /** @returns {boolean} true if there is an active HUD override on either key */
  _hasOverrides(coord) {
    if (!coord) return false;
    const ovs = coord.overrides || {};
    return ('body_physics_compliance_gain' in ovs)
        || ('body_physics_rapier_weight' in ovs);
  }

  /** Current effective value (override > server). Returns NaN if not resolvable. */
  _effective(coord, key, fallback) {
    if (!coord) return fallback;
    try {
      const ov = coord.overrides?.[key];
      if (ov !== undefined && ov !== null) return Number(ov);
      return Number(coord.settings?.[key] ?? fallback);
    } catch {
      return fallback;
    }
  }

  /**
   * Per-tick controller step. Driven from the HUD poll loop, so adaptive
   * quality only runs while the HUD is open (developer affordance).
   *
   * @param {number} avgMs   current rolling avg frame ms
   * @param {number} nowMs   high-res clock for hold-window bookkeeping
   */
  step(avgMs, nowMs) {
    const coord = _getBodyPhysicsCoordinator();
    if (!coord || !avgMs) return;

    const underStress = avgMs > STRESS_MS_THRESHOLD;
    const inRelief    = avgMs < RELIEF_MS_THRESHOLD;

    // Stress hold-window: only enter "reducing" after STRESS_HOLD_MS of sustained slow frames.
    if (underStress) {
      if (!this._stressSince) this._stressSince = nowMs;
      this._reliefSince = 0;
      if (this.mode === 'recovering') this.mode = 'nominal';   // abort recovery on new stress
      if (nowMs - this._stressSince >= STRESS_HOLD_MS) this.mode = 'reducing';
    } else if (inRelief) {
      if (!this._reliefSince) this._reliefSince = nowMs;
      this._stressSince = 0;
      // Only enter recovering if we actually have overrides to walk back.
      if (this._hasOverrides(coord) && nowMs - this._reliefSince >= RELIEF_HOLD_MS) {
        this.mode = 'recovering';
      } else if (!this._hasOverrides(coord) && this.mode !== 'nominal') {
        this.mode = 'nominal';
      }
    } else {
      // Neither stressed nor relieved — drift back to nominal if no overrides remain.
      this._stressSince = 0;
      this._reliefSince = 0;
      if (this.mode !== 'nominal' && !this._hasOverrides(coord)) this.mode = 'nominal';
    }

    // Throttle adjustment cadence to one step per ADJUST_INTERVAL_MS.
    if (this.mode === 'nominal') return;
    if (nowMs - this._lastAdjustAt < ADJUST_INTERVAL_MS) return;
    this._lastAdjustAt = nowMs;

    if (this.mode === 'reducing') {
      const gain   = this._effective(coord, 'body_physics_compliance_gain', NOMINAL_COMPLY_GAIN);
      const weight = this._effective(coord, 'body_physics_rapier_weight',  NOMINAL_RAPIER_WT);
      const nextGain   = Math.max(COMPLY_FLOOR,        gain   * STEP_DOWN_FACTOR);
      const nextWeight = Math.max(RAPIER_WEIGHT_FLOOR, weight * STEP_DOWN_FACTOR);
      if (nextGain   !== gain)   coord.setOverride('body_physics_compliance_gain', nextGain);
      if (nextWeight !== weight) coord.setOverride('body_physics_rapier_weight',   nextWeight);
      return;
    }

    if (this.mode === 'recovering') {
      const gain   = this._effective(coord, 'body_physics_compliance_gain', NOMINAL_COMPLY_GAIN);
      const weight = this._effective(coord, 'body_physics_rapier_weight',  NOMINAL_RAPIER_WT);
      const serverGain   = Number(coord.settings?.body_physics_compliance_gain ?? NOMINAL_COMPLY_GAIN);
      const serverWeight = Number(coord.settings?.body_physics_rapier_weight  ?? NOMINAL_RAPIER_WT);

      const nextGain   = Math.min(serverGain,   gain   * STEP_UP_FACTOR);
      const nextWeight = Math.min(serverWeight, weight * STEP_UP_FACTOR);

      // Snap-to-server when we're within 1% — float drift would otherwise keep us
      // permanently in "recovering" with an irrelevant 0.999× override.
      const gainAtTarget   = Math.abs(nextGain   - serverGain)   / Math.max(1e-6, serverGain)   < 0.01;
      const weightAtTarget = Math.abs(nextWeight - serverWeight) / Math.max(1e-6, serverWeight) < 0.01;

      if (gainAtTarget)   coord.setOverride('body_physics_compliance_gain', null);
      else                coord.setOverride('body_physics_compliance_gain', nextGain);
      if (weightAtTarget) coord.setOverride('body_physics_rapier_weight',   null);
      else                coord.setOverride('body_physics_rapier_weight',   nextWeight);

      if (!this._hasOverrides(coord)) this.mode = 'nominal';
    }
  }
}

// ─── HUD DOM + render ────────────────────────────────────────────────────────
/**
 * DOM-owning HUD instance. One per call to initBodyPhysicsHUD(); subsequent
 * calls return the existing instance.
 */
class BodyPhysicsHUD {
  constructor() {
    this.root = null;
    this._pollHandle = null;
    this._hotkeyListener = null;
    this._visible = false;
    this._controller = new AdaptiveQualityController();
  }

  /** Build the DOM, install the hotkey, leave the HUD hidden. */
  mount() {
    if (this.root) return;
    const root = document.createElement('div');
    root.id = 'body-physics-hud';
    root.setAttribute('aria-label', 'Body physics debug HUD');
    Object.assign(root.style, {
      position: 'fixed',
      top: '12px',
      right: '12px',
      width: '280px',
      maxHeight: 'calc(100vh - 24px)',
      overflowY: 'auto',
      padding: '10px 12px',
      background: 'rgba(12, 14, 18, 0.86)',
      color: '#cfd6df',
      font: '11px/1.45 ui-monospace, Menlo, Consolas, monospace',
      borderRadius: '8px',
      boxShadow: '0 6px 24px rgba(0, 0, 0, 0.5)',
      zIndex: '999999',
      pointerEvents: 'none',     // never intercept clicks — purely informational
      whiteSpace: 'pre',
      display: 'none',           // HIDDEN at boot per spec
    });
    document.body.appendChild(root);
    this.root = root;

    this._hotkeyListener = (ev) => {
      if (ev.code === HOTKEY.code && ev.ctrlKey === HOTKEY.ctrl && ev.shiftKey === HOTKEY.shift) {
        ev.preventDefault();
        this.toggle();
      }
    };
    window.addEventListener('keydown', this._hotkeyListener);
  }

  toggle() {
    this._visible = !this._visible;
    if (!this.root) return;
    this.root.style.display = this._visible ? 'block' : 'none';
    if (this._visible) {
      this._render();
      this._startPolling();
    } else {
      this._stopPolling();
    }
  }

  _startPolling() {
    if (this._pollHandle) return;
    this._pollHandle = setInterval(() => {
      try { this._render(); } catch (err) {
        // Don't let a render hiccup tear the whole HUD down.
        console.debug('[body-physics-hud] render error:', err?.message);
      }
    }, POLL_MS);
  }

  _stopPolling() {
    if (this._pollHandle) {
      clearInterval(this._pollHandle);
      this._pollHandle = null;
    }
  }

  /**
   * Snapshot each peer module, build a single innerHTML string. Each peer is
   * wrapped in its own try/catch so a missing or half-initialized peer shows
   * "-" instead of crashing the whole render.
   */
  _render() {
    const { avgMs, fps } = _frameSampler.snapshot();
    // Drive adaptive quality from the same clock used for hold-windows.
    this._controller.step(avgMs, (typeof performance !== 'undefined') ? performance.now() : Date.now());

    const lines = [];
    lines.push(`<b>[BODY PHYSICS HUD]</b>`);
    lines.push(`─────────────────────────`);
    lines.push(`fps  : ${fps.toFixed(1).padStart(5)}   ms: ${avgMs.toFixed(1)}`);
    lines.push(`─────────────────────────`);
    lines.push(this._renderCompliance());
    lines.push(`─────────────────────────`);
    lines.push(this._renderRapier());
    lines.push(`─────────────────────────`);
    lines.push(this._renderReactor());
    lines.push(`─────────────────────────`);
    lines.push(this._renderWeights());
    lines.push(`─────────────────────────`);
    lines.push(this._renderMode());
    this.root.innerHTML = lines.join('\n');
  }

  _renderCompliance() {
    try {
      const c = getXRCompliance();
      if (!c) return `COMPLIANCE   -\n max disp   : -\n active     : -\n stiffness  : -\n recover    : -`;
      const on = c.enabled ? 'on' : 'off';
      // Walk the per-bone state map to find max displacement + active count.
      let maxDisp = 0;
      let maxBone = '-';
      let active = 0;
      const bones = c._bones || new Map();
      const total = bones.size || 0;
      for (const [name, state] of bones) {
        const m = state?.current?.length?.() || 0;
        if (m > 1e-4) active++;
        if (m > maxDisp) { maxDisp = m; maxBone = name; }
      }
      const maxMm = (maxDisp * 1000).toFixed(1);
      const stiff = Number(c.stiffness ?? 0).toFixed(2);
      const rec   = Number(c.recoverHz ?? 0).toFixed(1);
      return [
        `COMPLIANCE   ${escapeHtml(on)}`,
        ` max disp   : ${maxMm.padStart(5)} mm    [${escapeHtml(maxBone)}]`,
        ` active     : ${active} / ${total}`,
        ` stiffness  : ${stiff}`,
        ` recover    : ${rec} Hz`,
      ].join('\n');
    } catch {
      return `COMPLIANCE   -\n max disp   : -\n active     : -\n stiffness  : -\n recover    : -`;
    }
  }

  _renderRapier() {
    try {
      const r = _getXRRapier();
      if (!r) return `RAPIER       -   w=-\n bones sim  : -\n max angle  : -`;
      const on = r.enabled ? 'on' : 'off';
      const w  = Number(r.weight ?? 0).toFixed(2);
      let bonesSim = 0;
      let maxAngleDeg = 0;
      let maxBone = '-';
      try {
        const deltas = r.getBoneDeltas?.();
        if (deltas && typeof deltas.forEach === 'function') {
          deltas.forEach((entry, name) => {
            bonesSim++;
            const q = entry?.quatDelta;
            if (!q) return;
            // quaternion → angle: 2 * acos(|w|), clamped for float drift.
            const wq = Math.min(1, Math.max(-1, Math.abs(q.w ?? 1)));
            const ang = 2 * Math.acos(wq) * 180 / Math.PI;
            if (ang > maxAngleDeg) { maxAngleDeg = ang; maxBone = name; }
          });
        }
      } catch { /* deltas unavailable mid-init — leave at 0 */ }
      const angStr = maxAngleDeg.toFixed(1);
      return [
        `RAPIER       ${escapeHtml(on)}  w=${w}`,
        ` bones sim  : ${bonesSim}`,
        ` max angle  : ${angStr}°       [${escapeHtml(maxBone)}]`,
      ].join('\n');
    } catch {
      return `RAPIER       -   w=-\n bones sim  : -\n max angle  : -`;
    }
  }

  _renderReactor() {
    try {
      const reactor = getXRContactReactor();
      if (!reactor) return `REACTOR      -\n L: -            -\n R: -            -`;
      const snap = reactor.inspect?.() || {};
      const stateL = snap.userState?.L || 'idle';
      const stateR = snap.userState?.R || 'idle';
      const last = snap.lastContact || {};
      const rowFor = (side) => {
        const ent = last[side];
        if (!ent || !ent.region) return ` ${side}: -            -`;
        const region = String(ent.region);
        const dist   = (typeof ent.distance === 'number') ? `${ent.distance.toFixed(2)}m` : '-';
        // Pad region to a fixed 13-char column so the distance lines up.
        const regionPad = region.length >= 13 ? region.slice(0, 13) : region.padEnd(13);
        return ` ${side}: ${escapeHtml(regionPad)} ${dist}`;
      };
      return [
        `REACTOR      L:${escapeHtml(String(stateL))} R:${escapeHtml(String(stateR))}`,
        rowFor('L'),
        rowFor('R'),
      ].join('\n');
    } catch {
      return `REACTOR      -\n L: -            -\n R: -            -`;
    }
  }

  _renderWeights() {
    try {
      const coord = _getBodyPhysicsCoordinator();
      if (!coord) return `WEIGHTS (live)\n comply gain : -\n rapier wt   : -\n recover hz  : -`;
      const snap = coord.inspect?.() || {};
      const gain    = Number(snap.compliance_gain ?? 0).toFixed(2);
      const weight  = Number(snap.rapier_weight   ?? 0).toFixed(2);
      const recover = Number(snap.recover_hz      ?? 0).toFixed(1);
      return [
        `WEIGHTS (live)`,
        ` comply gain : ${gain}`,
        ` rapier wt   : ${weight}`,
        ` recover hz  : ${recover}`,
      ].join('\n');
    } catch {
      return `WEIGHTS (live)\n comply gain : -\n rapier wt   : -\n recover hz  : -`;
    }
  }

  _renderMode() {
    const m = this._controller.mode;
    return `[adaptive: ${escapeHtml(m)}]`;
  }

  /** Tear down DOM, listeners, polling. */
  unmount() {
    this._stopPolling();
    if (this._hotkeyListener) {
      window.removeEventListener('keydown', this._hotkeyListener);
      this._hotkeyListener = null;
    }
    if (this.root?.parentNode) this.root.parentNode.removeChild(this.root);
    this.root = null;
    this._visible = false;
  }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/**
 * Initialize the HUD: creates DOM nodes, registers Ctrl+Shift+B hotkey, leaves
 * the overlay HIDDEN. Idempotent — repeated calls return the existing instance.
 *
 * @returns {BodyPhysicsHUD}
 */
export function initBodyPhysicsHUD() {
  if (_hud) return _hud;
  _hud = new BodyPhysicsHUD();
  _hud.mount();
  return _hud;
}

/** Show/hide the HUD. Same effect as pressing Ctrl+Shift+B. */
export function toggleBodyPhysicsHUD() {
  if (!_hud) initBodyPhysicsHUD();
  _hud.toggle();
}

/**
 * Tear down: removes DOM, clears the poll interval, removes the hotkey
 * listener, and stops the rAF sampler. The sampler does not re-arm on
 * subsequent init() calls in the same page lifetime, so this is intended
 * for test or full-page teardown rather than per-session toggling.
 */
export function teardownBodyPhysicsHUD() {
  if (!_hud) return;
  _hud.unmount();
  _hud = null;
  try { _frameSampler.stop(); } catch { /* nothing to stop */ }
}

/** @returns {BodyPhysicsHUD|null} */
export function getBodyPhysicsHUD() {
  return _hud;
}
