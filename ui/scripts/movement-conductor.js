/**
 * movement-conductor.js — runtime layer that turns intents into
 * actual avatar motion via the tagged atlas.
 *
 * The conductor:
 *   - holds runtime state (energy budget, recency, last-played, mode)
 *   - takes intents from any source (lifecycle hooks, sentence
 *     classifiers, explicit user requests, idle escalation)
 *   - selects a candidate from anim-atlas via select()
 *   - dispatches to playVrma (which routes VRMA + BVH internally)
 *   - records the decision so future selections respect cooldowns,
 *     variety, and the back-to-back-theatrical suppression budget
 *
 * Energy budget:
 *   Each cost-N play subtracts N from the budget. The budget recovers
 *   linearly via tick(dt) at BUDGET_RECOVERY_PER_SEC, so a max-cost
 *   theatrical (≈0.95) drops the budget low enough to gate any other
 *   theatrical for ~25s while micros (≈0.12) still pass.
 *
 *   This is the structural mechanism that prevents "back to back big
 *   animations" — there's no formal tier system, just one threshold
 *   per animation.
 *
 * Mode awareness:
 *   The conductor's `mode` field gates which atlas entries are
 *   eligible at all (modes tag must match). chat-call vs narrative
 *   etc. are different selection populations from the same atlas.
 *
 * Adaptation hook (stub):
 *   recordReaction(animId, signal) shifts per-id bias for future
 *   selections. Wire it from a transcript observer when ready.
 */

import { select as selectAnim, getAnim } from './anim-atlas.js';
import { avatarState, playVrma, stopVrma, currentVrmaName } from './avatar.js';

const BUDGET_RECOVERY_PER_SEC = 0.04;   // full recovery in ~25s from empty
const RECENT_CAP = 5;                   // last-N played for variety penalty
const MIN_BUDGET_TO_FIRE = 0.05;        // floor; below this nothing plays

// Safety-release caps. User-uploaded VRMAs/BVHs introduce variance the
// bundled curation never had — clips whose 'finished' event never fires
// (malformed tracks), wrong atlas durations (uploads default to 0), and
// loop:true entries played outside the host-rotation context that would
// normally replace them. The watchdog force-releases the action so the
// avatar always returns to the procedural idle instead of freezing on
// a stray final frame or looping forever.
const LOOP_SAFETY_CAP_MS = 120_000;     // loop plays nobody rotated away
const ONESHOT_FALLBACK_CAP_MS = 30_000; // one-shots with unknown duration
const ONESHOT_GRACE_MS = 5_000;         // beyond expected end before forcing

// Per-rating bias multipliers. Applied to ``bias[animId]`` on top of any
// adaptation deltas from recordReaction. 'broken' is a hard veto:
// _applyRatings sets bias to 0 so selection skips the entry entirely.
const _RATING_BIAS = {
  like:    1.6,
  dislike: 0.35,
  broken:  0.0,
};

// localStorage cache key for ratings. Server (POST/GET /api/dance/ratings)
// is authoritative; this key only seeds the in-memory map on cold load
// so the timeline renders something instantly before the server
// roundtrip lands.
const _RATINGS_KEY = 'becca.dance.ratings';

export class MovementConductor {
  /**
   * @param {object} [opts]
   * @param {string}  [opts.mode='chat-call']
   * @param {Object<string, number>} [opts.bias]   per-id multiplier
   * @param {() => number}           [opts.now]   test seam
   */
  constructor(opts = {}) {
    this.mode = opts.mode || 'chat-call';
    this.bias = { ...(opts.bias || {}) };
    this._now = opts.now || (() => Date.now());

    this.energyBudget = 1.0;
    this.recent = [];                   // ids; most-recent at end
    this.lastPlayed = new Map();        // id → ms timestamp
    /** Energy of the most-recently-picked clip (0..1). Threaded into the
     *  selector's context so candidates whose energy delta exceeds ~0.4
     *  get scored down — prevents jarring "full-energy → languid →
     *  full-energy" jumps in continuous-rotation contexts. Null until
     *  the first dispatch. */
    this.lastPickedEnergy = null;

    // Telemetry — the most-recent decision result, useful for HUDs.
    this.lastResult = null;             // { intent, picked, reason }

    // Per-id curation ratings: { animId: { kind, slotBonusSec, ts } }
    // Loaded from localStorage on construction so curation accumulates
    // across reloads. ``kind`` is one of 'like' | 'dislike' | 'broken';
    // ``slotBonusSec`` is the per-id slot extension from "longer" rating
    // (additive, not replacing — multiple "longer" clicks stack).
    this._ratings = {};
    this._loadRatings();
    this._applyRatings();

    // Active loop constraint (Phase C). When non-null, the selector
    // filters its candidate pool to members of this Set BEFORE scoring
    // — bundled and user-uploaded ids mix freely. null means "no loop
    // active, full atlas eligible". Set on widget mount via the
    // /api/dance/loops fetch and updated when the user activates a
    // loop from the loops overlay.
    this.activeLoopIds = null;

    // Stuck-state watchdog handle — see _armSafetyRelease.
    this._safetyTimer = null;

    // Quiet mode — when true, ``play()`` no-ops for any non-explicit
    // dispatch. The procedural layer (breathing, sway, head-track,
    // look-at-user) keeps running; only the VRMA layer is muted.
    //
    // Default true: by default Becca is a still avatar that looks at
    // the user, NOT a perpetual animator. The presence widget flips
    // this off when she should actively animate — currently only when
    // hosting media (dance-with-music). Voice-call PoseTriggerEngine
    // fires also pass through `play()` and are gated by this; an
    // active call should unmute the conductor explicitly so call
    // animations work.
    //
    // Explicit dispatches (``play(intent, {explicit:true})`` and
    // ``playById(id)``) ALWAYS bypass this gate — user clicks on
    // animations in the panel should always play.
    this.quietMode = true;
  }

  // ─── Public API ────────────────────────────────────────────────

  setMode(mode) { this.mode = mode; }

  /**
   * Mute or unmute auto-selected animations. See ``quietMode`` for
   * semantics. Explicit dispatches are unaffected.
   *
   * @param {boolean} enabled
   */
  setQuietMode(enabled) {
    this.quietMode = !!enabled;
  }

  /**
   * Per-id bias adjustment. >1 increases selection odds, <1 decreases.
   * Used by recordReaction and per-character config.
   */
  setBias(bias) { this.bias = { ...bias }; }

  /**
   * Constrain the selector to a specific set of animation ids. Pass
   * ``null`` (or omit) to clear the constraint and re-enable the full
   * atlas. The Set wraps both bundled ATLAS ids and user-animation ids
   * indistinguishably — membership is the only signal.
   *
   * @param {Iterable<string>|null} animIds
   */
  setActiveLoop(animIds) {
    if (animIds == null) {
      this.activeLoopIds = null;
      return;
    }
    this.activeLoopIds = new Set(animIds);
  }

  /** Per-frame energy regen. Call from the animation loop with dt. */
  tick(dt) {
    if (dt > 0 && this.energyBudget < 1) {
      this.energyBudget = Math.min(1, this.energyBudget + BUDGET_RECOVERY_PER_SEC * dt);
    }
  }

  /**
   * Estimated ms until energyBudget reaches ``target``. Returns 0 if
   * already at or above target. Useful for callers that retry on the
   * 'budget-floor' suppression reason — instead of polling on a fixed
   * timer, schedule exactly long enough for the budget to refill.
   *
   * @param {number} [target=0.55]  budget level callers want to hit;
   *                                 default ≈ cheapest theatrical cost
   *                                 after the 2026-05-19 recalibration
   * @returns {number} ms (0 = ready now)
   */
  estimatedBudgetRefillMs(target = 0.55) {
    if (this.energyBudget >= target) return 0;
    const deltaSec = (target - this.energyBudget) / BUDGET_RECOVERY_PER_SEC;
    return Math.max(0, Math.round(deltaSec * 1000));
  }

  /**
   * Try to play an animation matching the intent. Returns the picked
   * atlas entry on success, null when nothing fit (caller should treat
   * null as "procedural carries — no animation this beat").
   *
   * @param {object} intent           { roles?, emotion? }
   * @param {object} [options]
   * @param {boolean} [options.explicit=false]   user-initiated request:
   *                                              bypass cooldown / recency,
   *                                              allow explicitOnly entries
   * @param {Function} [options.onFinish]        chained next-intent hook
   * @returns {object|null}
   */
  async play(intent, options = {}) {
    // Quiet-mode gate — kill auto-fires (PoseTriggerEngine idle
    // escalation, companion-animation-router topic intents) when the
    // presence layer hasn't explicitly unmuted the conductor. The
    // procedural pose/breath/sway layer is unaffected; only VRMA
    // dispatch is gated here.
    if (this.quietMode && !options.explicit) {
      this.lastResult = { intent, picked: null, reason: 'quiet-mode' };
      return null;
    }
    if (this.energyBudget < MIN_BUDGET_TO_FIRE && !options.explicit) {
      this.lastResult = { intent, picked: null, reason: 'budget-floor' };
      return null;
    }

    const ctx = {
      mode: this.mode,
      energyBudget: options.explicit ? 1.0 : this.energyBudget,
      recent: options.explicit ? [] : this.recent,
      lastPlayed: options.explicit ? new Map() : this.lastPlayed,
      bias: this.bias,
      includeExplicitOnly: !!options.explicit,
      // Active-loop constraint. Skipped on explicit dispatches — if the
      // user asks for "do a peace sign" we honor the request even if
      // peace-sign isn't in the active loop. Auto-selection respects
      // the loop.
      activeLoopIds: options.explicit ? null : this.activeLoopIds,
      // Skip energy-delta penalty on explicit requests — user-initiated
      // picks should land what was asked for without softening.
      lastEnergy: options.explicit ? null : this.lastPickedEnergy,
      now: this._now(),
    };

    const anim = selectAnim(intent, ctx);
    if (!anim) {
      this.lastResult = { intent, picked: null, reason: 'no-fit' };
      return null;
    }

    this._record(anim);
    await this._dispatch(anim, options);
    this.lastResult = { intent, picked: anim, reason: 'played' };
    return anim;
  }

  /**
   * Direct play by id — bypasses selection. Used for explicit
   * user-initiated requests that pattern-matched a specific clip
   * (e.g. "do a peace sign" → playById('peace-sign')).
   */
  async playById(id, options = {}) {
    const anim = getAnim(id);
    if (!anim) {
      console.warn('[movement-conductor] unknown id:', id);
      this.lastResult = { intent: { id }, picked: null, reason: 'unknown-id' };
      return null;
    }
    this._record(anim);
    await this._dispatch(anim, { ...options, explicit: true });
    this.lastResult = { intent: { id }, picked: anim, reason: 'played-explicit' };
    return anim;
  }

  /** Stop whatever's playing. Procedural animator resumes. */
  stop() {
    if (this._safetyTimer) {
      clearTimeout(this._safetyTimer);
      this._safetyTimer = null;
    }
    stopVrma();
  }

  /** Currently-playing animation name, or null if none. */
  currentName() {
    return currentVrmaName();
  }

  /**
   * Adaptation stub. Reactions feed back into bias so the deployed
   * vocabulary drifts toward what THIS user enjoys. Wire this from a
   * transcript observer when ready; for now it's a no-op-friendly hook.
   *
   * @param {string} animId
   * @param {'positive'|'negative'} signal
   */
  recordReaction(animId, signal) {
    const cur = this.bias[animId] ?? 1.0;
    const delta = signal === 'positive' ? 0.05 : -0.05;
    this.bias[animId] = Math.max(0.1, Math.min(3.0, cur + delta));
  }

  /**
   * Explicit user curation. Stronger signal than recordReaction —
   * a user pressing a button on the timeline panel.
   *
   * @param {string} animId
   * @param {'like'|'dislike'|'broken'|'longer'|'clear'} kind
   *
   * 'like'    → boost selection bias (×1.6)
   * 'dislike' → suppress selection bias (×0.35) but don't veto
   * 'broken'  → hard veto: bias = 0, never picked until cleared
   * 'longer'  → +8 seconds added to per-id slot bonus (stacks if
   *             clicked again, capped at +60s)
   * 'clear'   → remove any rating for this id (un-break, un-like, etc.)
   *
   * Persisted to localStorage; reload picks up where they left off.
   */
  recordRating(animId, kind) {
    if (!animId) return;
    const existing = this._ratings[animId] || {};
    let incrementSec = null;
    if (kind === 'clear') {
      delete this._ratings[animId];
    } else if (kind === 'longer') {
      const cur = existing.slotBonusSec || 0;
      incrementSec = 8;
      this._ratings[animId] = {
        ...existing,
        slotBonusSec: Math.min(60, cur + incrementSec),
        ts: this._now(),
      };
    } else if (kind === 'like' || kind === 'dislike' || kind === 'broken') {
      this._ratings[animId] = {
        ...existing,
        kind,
        ts: this._now(),
      };
    } else {
      return;
    }
    this._saveRatings();
    this._applyRatings();
    // Fire-and-forget server sync. Failure leaves the local cache
    // ahead of the server, which the next refresh resolves.
    this._postRating(animId, kind, incrementSec);
  }

  /** Per-id slot extension from "longer" ratings. Returns seconds (0 if unset). */
  getSlotBonus(animId) {
    return this._ratings[animId]?.slotBonusSec || 0;
  }

  /** Current rating record for an id ({kind?, slotBonusSec?, ts?}) or null. */
  getRating(animId) {
    return this._ratings[animId] || null;
  }

  /** All ratings — for UI display of the timeline panel. */
  getAllRatings() {
    return { ...this._ratings };
  }

  // ─── Ratings persistence + application ────────────────────────────
  //
  // Server (/api/dance/ratings) is authoritative. localStorage caches
  // the last known state so the UI has something to render before the
  // server fetch returns on cold load — and lets curation degrade
  // gracefully if the user is offline. Reconciliation flow:
  //
  //   constructor → _loadRatings() reads localStorage (instant)
  //                ↓
  //   refreshRatingsFromServer() (called by widget after construction)
  //                ↓ GET /api/dance/ratings — replaces in-memory map
  //                ↓ overwrites localStorage cache to match
  //
  // Every recordRating() POSTs the change to the server, then mirrors
  // it into localStorage. A failed POST leaves the change in
  // localStorage so the next successful refresh wins (last-write-wins
  // by the user's most recent device).

  _loadRatings() {
    try {
      const raw = (typeof localStorage !== 'undefined')
        ? localStorage.getItem(_RATINGS_KEY) : null;
      if (raw) this._ratings = JSON.parse(raw) || {};
    } catch (_) { this._ratings = {}; }
  }

  _saveRatings() {
    try {
      if (typeof localStorage !== 'undefined') {
        localStorage.setItem(_RATINGS_KEY, JSON.stringify(this._ratings));
      }
    } catch (_) { /* quota / private mode — best effort */ }
  }

  /**
   * Fetch the authoritative ratings map from the server and merge it
   * into in-memory state. Called by the widget once at mount and after
   * sign-in changes. Failure is silent — localStorage cache stays.
   */
  async refreshRatingsFromServer() {
    try {
      const resp = await fetch('/api/dance/ratings', {
        credentials: 'same-origin',
      });
      if (!resp.ok) return;
      const data = await resp.json();
      this._ratings = data?.ratings || {};
      this._saveRatings();
      this._applyRatings();
    } catch (_) { /* offline / unauthed — keep cache */ }
  }

  /**
   * POST a single rating change to the server. Best-effort: the
   * in-memory + localStorage update has already happened by the time
   * we get here, so a failure just means the device-local cache is
   * ahead of the server until the next successful refresh.
   */
  async _postRating(animId, kind, incrementSec) {
    try {
      const body = { kind };
      if (kind === 'longer' && incrementSec != null) {
        body.increment_sec = incrementSec;
      }
      await fetch(`/api/dance/ratings/${encodeURIComponent(animId)}`, {
        method: 'PUT',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    } catch (_) { /* keep local — refresh will reconcile */ }
  }

  /** Re-derive ``this.bias`` from ratings + carry-over from recordReaction. */
  _applyRatings() {
    for (const [id, rec] of Object.entries(this._ratings)) {
      const kind = rec?.kind;
      const mul = _RATING_BIAS[kind];
      if (mul !== undefined) {
        this.bias[id] = mul;  // ratings are absolute, overwriting reaction-deltas
      }
    }
  }

  /** Diagnostic snapshot for HUDs / logs. */
  getDebugState() {
    return {
      mode: this.mode,
      energyBudget: this.energyBudget,
      recent: [...this.recent],
      lastResult: this.lastResult,
      biasCount: Object.keys(this.bias).length,
    };
  }

  // ─── Internal ──────────────────────────────────────────────────

  _record(anim) {
    this.energyBudget = Math.max(0, this.energyBudget - (anim.cost ?? 0));
    this.recent.push(anim.id);
    if (this.recent.length > RECENT_CAP) this.recent.shift();
    this.lastPlayed.set(anim.id, this._now());
    // Track this pick's energy for the next selection's transition-
    // smoothing — falls back to neutral 0.5 if the atlas entry has no
    // emotion vector (defensive; all dance entries do).
    this.lastPickedEnergy = anim?.emotion?.energy ?? 0.5;
  }

  async _dispatch(anim, options) {
    // Build playVrma options from atlas entry + caller overrides.
    // Caller-provided options take precedence so explicit requests can
    // tweak speed / framing without editing the atlas entry.
    //
    // Speed jitter (±8%) — applied per-play for replay variance on long
    // clips. Gated so it never overrides:
    //   - an explicit caller `options.speed` (user-initiated requests)
    //   - an explicit atlas `anim.speed` (intentional per-clip tuning
    //     like dance-28's 0.75× which was authored because full speed
    //     reads frenetic)
    //   - short clips (duration < 8s) — these feel novel each play
    //     already via the cooldown spacing; jitter on a 2s dab adds
    //     nothing visible
    // Skipped on explicit dispatches so user-initiated picks land at
    // exactly the speed implied by the atlas entry.
    let speed = options.speed ?? anim.speed ?? 1.0;
    if (options.speed == null
        && anim.speed == null
        && (anim.duration ?? 0) >= 8.0
        && !options.explicit) {
      // Uniform ±8% (range 0.92..1.08). Math.random() is fine — same
      // dance played twice gets two different speeds, which is the
      // entire point.
      const jitter = 0.92 + Math.random() * 0.16;
      speed = jitter;
    }
    const playOpts = {
      name: anim.id,
      loop: options.loop ?? anim.loop ?? false,
      speed,
      framing: options.framing ?? anim.framing ?? null,
      framingOffset: options.framingOffset ?? anim.framingOffset ?? null,
      onFinish: options.onFinish || null,
    };
    // Trim: prefer explicit seconds; fall back to the fraction form if
    // duration is known (so the conductor handles either notation).
    if (anim.trimStart != null) {
      playOpts.trimStart = anim.trimStart;
    } else if (anim.trimStartFrac != null && anim.duration > 0) {
      playOpts.trimStart = anim.duration * anim.trimStartFrac;
    }
    if (anim.trimEnd != null) playOpts.trimEnd = anim.trimEnd;

    try {
      const ok = await playVrma(anim.source, playOpts);
      if (ok !== false) this._armSafetyRelease(anim, playOpts, options);
    } catch (err) {
      console.warn('[movement-conductor] dispatch failed:', anim.id, err);
    }
  }

  /**
   * Arm the stuck-state watchdog for the play that just started.
   *
   * One-shots already release via the mixer's 'finished' listener in
   * avatar.js; this is the backstop for when that event never fires
   * (malformed user uploads) plus the cap for loop:true plays that no
   * rotation context will ever replace (preview, verb-driven picks).
   * Release = stopVrma(), which recenters the hips and hands the body
   * back to the procedural idle (breath / sway / look-at-user) — the
   * "normal idle" guarantee.
   *
   * ``options.safetyCapMs === 0`` disables the watchdog — the host
   * dance rotation owns its replacement schedule (including user
   * "longer" slot bonuses) and must not be cut mid-slot.
   */
  _armSafetyRelease(anim, playOpts, options = {}) {
    if (this._safetyTimer) {
      clearTimeout(this._safetyTimer);
      this._safetyTimer = null;
    }
    const capOverride = options.safetyCapMs;
    if (capOverride === 0) return;

    let capMs;
    if (playOpts.loop) {
      capMs = capOverride ?? LOOP_SAFETY_CAP_MS;
    } else {
      // Real loaded duration beats the atlas guess — uploads default
      // their duration metadata to 0, which would undershoot badly.
      const realDur = avatarState.vrmaCurrentDuration > 0
        ? avatarState.vrmaCurrentDuration
        : (anim.duration || 0);
      if (realDur > 0) {
        const trim = playOpts.trimStart || 0;
        const speed = playOpts.speed || 1.0;
        const expectedMs = (Math.max(0.5, realDur - trim) / speed) * 1000;
        capMs = capOverride ?? (expectedMs + ONESHOT_GRACE_MS);
      } else {
        capMs = capOverride ?? ONESHOT_FALLBACK_CAP_MS;
      }
    }

    this._safetyTimer = setTimeout(() => {
      this._safetyTimer = null;
      // Only force-release if OUR play is still the active action —
      // anything that replaced it owns its own lifecycle.
      if (currentVrmaName() === anim.id) {
        console.info('[movement-conductor] safety release → idle', anim.id);
        try { stopVrma(); } catch (_) { /* already torn down */ }
      }
    }, capMs);
  }
}
