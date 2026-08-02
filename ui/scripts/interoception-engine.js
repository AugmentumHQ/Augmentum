/**
 * interoception-engine.js
 *
 * Synthetic physiology + affective state substrate for the companion
 * avatar. Maintains six unit-range state variables with homeostatic
 * set points and exponential drift toward them; reacts to browser
 * events (TTS, pointerdown, mode change, audio-bus, idle, dance) by
 * applying additive impulses. Output coupling is read-on-demand —
 * other modules call ``getBreathModifier()`` / ``getPhysiology()`` /
 * ``getAffect()`` and decide what to do with the values.
 *
 * v0 ships one observable wire: ``breath_rate`` → AvatarAnimator's
 * existing ``setBreathModifier()``. PresenceEngine's own breath
 * modifier is preserved and multiplexed on top so this is purely
 * additive — at default state (`breath_rate = 0.5`) the multiplier
 * is identity and behavior matches today.
 *
 * Design doc: docs/superpowers/specs/2026-05-16-interoception-engine-design.md
 *
 * Lifecycle:
 *   const intero = new InteroceptionEngine();
 *   // ...per animate frame:
 *   intero.update(dt);
 *   const { rate, depth } = intero.getBreathModifier();
 *   // ...on teardown:
 *   intero.dispose();
 */

// ── Tuning constants ─────────────────────────────────────────────
//
// Set points are universal defaults for v0. v3 (personality
// parameterization) will read these per-character from
// personality.embodiment. All values in [0,1].

const SETPOINTS = Object.freeze({
  heart_rate:     0.40,
  breath_rate:    0.50,
  muscle_tension: 0.30,
  arousal:        0.35,
  valence:        0.60,
});

// Time constants (seconds) — how fast each variable returns to its
// set point in the absence of impulses. valence is intentionally
// slow: a positive interaction is "felt" for minutes, not seconds.
const TAU = Object.freeze({
  heart_rate:     60,
  breath_rate:    45,
  muscle_tension: 90,
  arousal:        75,
  valence:        240,
});

// Driver impulse table — see design doc §"Drivers" for rationale on
// magnitudes. The point is *accumulation*: small events combine into
// visible state.
const IMPULSES = Object.freeze({
  tts_start:      { breath_rate: +0.20, heart_rate: +0.15, muscle_tension: +0.05, arousal: +0.18 },
  tts_end:        { valence:     +0.04 },
  media_active:   { breath_rate: +0.05, heart_rate: +0.05, arousal: +0.06, valence: +0.02 },
  mode_changed:   { breath_rate: +0.04, heart_rate: +0.05, muscle_tension: +0.04, arousal: +0.10 },
  pointer_down:   { breath_rate: +0.02, heart_rate: +0.03, muscle_tension: +0.02, arousal: +0.04 },
  pointer_sustained: { heart_rate: +0.01, muscle_tension: +0.01, arousal: +0.02, valence: +0.01 },
  user_idle:      { breath_rate: -0.05, heart_rate: -0.04, muscle_tension: -0.06, arousal: -0.08 },
  dance_active:   { breath_rate: +0.04, heart_rate: +0.06, muscle_tension: -0.02, arousal: +0.06, valence: +0.03 },
  wake_flash:     { breath_rate: +0.10, heart_rate: +0.10, muscle_tension: +0.04, arousal: +0.15, valence: +0.06 },
});

// Breath-modifier mapping. At ``breath_rate = 0.5`` the modifier is
// identity {1,1} so we layer onto the existing breath system without
// changing default behavior. Gains chosen so:
//   breath_rate = 0.9  →  rate ≈ 1.4,  depth ≈ 1.15  (post-TTS, excited)
//   breath_rate = 0.2  →  rate ≈ 0.7,  depth ≈ 0.92  (deep idle settle)
const BREATH_RATE_GAIN = 1.0;     // rate slope from baseline
const BREATH_DEPTH_GAIN = 0.5;    // depth slope from baseline (half — depth shifts less than rate)

// Sustained-pointer-activity integrator: a rolling counter that ticks
// up while pointermove fires and decays otherwise. Above the trigger
// threshold we emit one ``pointer_sustained`` impulse per cooldown
// window.
const POINTER_ACTIVITY_TRIGGER = 3.0;      // seconds-equivalent
const POINTER_ACTIVITY_COOLDOWN_MS = 8000; // gap between repeat impulses
const POINTER_ACTIVITY_DECAY = 0.6;        // per-second decay when no movement

// Habituation — high-frequency drivers (TTS, pointer, media-active)
// lose magnitude on repeat so prolonged interaction doesn't pin the
// physiology at maximum. Novel/rare events (wake_flash, mode_changed,
// user_idle, tts_end, dance_active) bypass habituation entirely — they
// represent real state changes that should retain their full impact.
// The fire-count for each habituated key decays with HABITUATION_TAU
// so an absence of stimulus releases the desensitization.
const HABITUATION_TAU_SEC = 25;
const HABITUATION_GAIN = 0.45;            // larger = stronger suppression
const HABITUATED_KEYS = Object.freeze(new Set([
  'tts_start', 'media_active', 'pointer_down', 'pointer_sustained',
]));

// User-idle threshold (no pointer events for this long → idle impulse,
// once per cooldown window).
const USER_IDLE_THRESHOLD_MS = 120 * 1000;
const USER_IDLE_COOLDOWN_MS = 60 * 1000;

// Internal tick cadence ceiling — drift integration uses min(dt,
// MAX_DT) to defend against backgrounded-tab multi-second frames.
const MAX_DT_SEC = 0.5;

// Telemetry cadence — console.debug snapshot for tuning.
const TELEMETRY_INTERVAL_MS = 5000;

export class InteroceptionEngine {
  constructor(opts = {}) {
    /** Active state. Initialized to set points so first frame is at rest. */
    this.state = { ...SETPOINTS };
    /** Override hook for set points (future: personality parameterization). */
    this.setpoints = Object.assign({}, SETPOINTS, opts.setpoints || {});
    this.tau = Object.assign({}, TAU, opts.tau || {});

    // Pointer-activity integrator (sustained interaction → arousal)
    this._pointerActivity = 0;
    this._lastPointerActivityImpulseAt = 0;
    this._lastPointerMoveAt = 0;

    // Habituation fire-counts per impulse key. Decays in update().
    this._habituation = {};

    // Idle watchdog
    this._lastInputAt = performance.now();
    this._lastIdleImpulseAt = 0;

    // Listener handles — stored so dispose() can remove cleanly.
    this._listeners = [];
    this._attachListeners();

    // Telemetry
    this._lastTelemetryAt = 0;
    this._telemetryEnabled = opts.telemetry !== false;

    // Expose a global handle for inspection. Mirrors the
    // __beccaAudioRole pattern other modules use for read-only state.
    try { window.__beccaInteroception = this; } catch (_) {}
  }

  /**
   * Apply an event impulse by key. ``key`` is one of IMPULSES.
   * Magnitudes are looked up from the table; ``scale`` multiplies
   * the whole impulse (default 1). Out-of-table keys are a no-op.
   *
   * @param {string} key  one of IMPULSES
   * @param {number} [scale=1.0]
   */
  trigger(key, scale = 1.0) {
    const imp = IMPULSES[key];
    if (!imp) return;
    let effective = scale;
    if (HABITUATED_KEYS.has(key)) {
      const count = this._habituation[key] || 0;
      effective *= 1 / (1 + count * HABITUATION_GAIN);
      this._habituation[key] = count + 1;
    }
    for (const v in imp) {
      this.state[v] = _clamp01(this.state[v] + imp[v] * effective);
    }
  }

  /**
   * Pulled by the animate() loop each frame. Integrates drift toward
   * set points; processes the pointer-activity / idle watchdogs.
   *
   * @param {number} dt  seconds since last update
   */
  update(dt) {
    const ddt = Math.min(Math.max(0, dt), MAX_DT_SEC);
    // First-order drift toward set point per variable.
    for (const k in this.state) {
      const tau = this.tau[k] || 60;
      this.state[k] += (this.setpoints[k] - this.state[k]) * (ddt / tau);
    }

    // Pointer-activity integrator decays toward zero in absence of moves.
    this._pointerActivity = Math.max(0,
      this._pointerActivity - POINTER_ACTIVITY_DECAY * ddt);

    // Habituation fire-counts decay exponentially — quiet windows
    // re-sensitize the engine so a TTS phrase after a long silence
    // hits with full impact again.
    const habDecay = Math.exp(-ddt / HABITUATION_TAU_SEC);
    for (const k in this._habituation) {
      const next = this._habituation[k] * habDecay;
      if (next < 0.01) delete this._habituation[k];
      else this._habituation[k] = next;
    }

    // Idle watchdog. Single impulse per cooldown window.
    const now = performance.now();
    const idleFor = now - this._lastInputAt;
    if (idleFor > USER_IDLE_THRESHOLD_MS
        && now - this._lastIdleImpulseAt > USER_IDLE_COOLDOWN_MS) {
      this.trigger('user_idle');
      this._lastIdleImpulseAt = now;
    }

    // Telemetry tick — cheap snapshot for tuning.
    if (this._telemetryEnabled
        && now - this._lastTelemetryAt > TELEMETRY_INTERVAL_MS) {
      this._lastTelemetryAt = now;
      console.debug('[interoception]',
        Object.fromEntries(Object.entries(this.state)
          .map(([k, v]) => [k, +v.toFixed(2)])));
    }
  }

  /**
   * Breath modifier derived from ``breath_rate``. Multiplexed onto
   * PresenceEngine's modifier in avatar.js. At default state this
   * returns identity {1, 1} so behavior matches today.
   *
   * @returns {{rate: number, depth: number}}
   */
  getBreathModifier() {
    const dev = this.state.breath_rate - SETPOINTS.breath_rate;
    return {
      rate:  1.0 + dev * BREATH_RATE_GAIN,
      depth: 1.0 + dev * BREATH_DEPTH_GAIN,
    };
  }

  /** Snapshot of physiological vars (for PresenceEngine consumption v1). */
  getPhysiology() {
    return {
      heart_rate:     this.state.heart_rate,
      breath_rate:    this.state.breath_rate,
      muscle_tension: this.state.muscle_tension,
    };
  }

  /** Snapshot of affective vars. */
  getAffect() {
    return {
      arousal: this.state.arousal,
      valence: this.state.valence,
    };
  }

  // ── Listeners ────────────────────────────────────────────────

  _attachListeners() {
    const on = (target, evt, fn) => {
      target.addEventListener(evt, fn, { passive: true });
      this._listeners.push({ target, evt, fn });
    };

    // TTS playback lifecycle. chat/tts.js emits {active: boolean} —
    // active=true marks the start of a phrase, active=false the end.
    on(window, 'augmentum:tts-playback', (e) => {
      const active = e?.detail?.active;
      if (active === true) this.trigger('tts_start');
      else if (active === false) this.trigger('tts_end');
      this._touchInput();
    });

    // Audio-bus state — media-tier active = music/video playing
    on(window, 'augmentum:audio-bus-state', (e) => {
      const kinds = e?.detail?.activeKinds || [];
      if (kinds.includes('music') || kinds.includes('video')) {
        this.trigger('media_active', 0.1);  // small per-tick; fires often
      }
    });

    // Mode change — dispatched on document (not window) by app.js.
    on(document, 'augmentum:mode-changed', () => {
      this.trigger('mode_changed');
      this._touchInput();
    });

    // Wake-word flash — install a wrapper around the existing hook so
    // we don't fight the widget's own listener.
    const origWake = window.__beccaFlashWake;
    const wakeWrap = (phrase) => {
      this.trigger('wake_flash');
      this._touchInput();
      if (typeof origWake === 'function') {
        try { origWake(phrase); } catch (_) {}
      }
    };
    if (origWake !== wakeWrap) {
      try {
        window.__beccaFlashWake = wakeWrap;
        this._wakeWrapOriginal = origWake;
        this._wakeWrapInstalled = wakeWrap;
      } catch (_) { /* readonly global in some sandboxes — wake-flash stays default */ }
    }

    // Pointerdown — small arousal jolt
    on(window, 'pointerdown', () => {
      this.trigger('pointer_down');
      this._touchInput();
    });

    // Pointermove — feeds the sustained-activity integrator
    on(window, 'pointermove', () => {
      this._touchInput();
      this._pointerActivity = Math.min(10, this._pointerActivity + 0.05);
      if (this._pointerActivity > POINTER_ACTIVITY_TRIGGER) {
        const now = performance.now();
        if (now - this._lastPointerActivityImpulseAt > POINTER_ACTIVITY_COOLDOWN_MS) {
          this.trigger('pointer_sustained');
          this._lastPointerActivityImpulseAt = now;
        }
      }
    });

    // Keystrokes count as input for idle-watchdog purposes
    on(window, 'keydown', () => this._touchInput());
  }

  /** Mark that we just saw user input (resets idle watchdog clock). */
  _touchInput() {
    this._lastInputAt = performance.now();
  }

  /**
   * Remove all listeners and global handles. Idempotent.
   */
  dispose() {
    for (const { target, evt, fn } of this._listeners) {
      try { target.removeEventListener(evt, fn); } catch (_) {}
    }
    this._listeners = [];
    // Restore wake-flash hook only if we still own the slot. If a
    // later layer replaced our wrapper, we leave it alone — they
    // may have wrapped us in turn.
    try {
      if (window.__beccaFlashWake === this._wakeWrapInstalled) {
        window.__beccaFlashWake = this._wakeWrapOriginal;
      }
    } catch (_) { /* readonly global — best-effort uninstall */ }
    try {
      if (window.__beccaInteroception === this) {
        delete window.__beccaInteroception;
      }
    } catch (_) { /* readonly global — best-effort cleanup */ }
  }
}

function _clamp01(x) {
  return x < 0 ? 0 : (x > 1 ? 1 : x);
}
