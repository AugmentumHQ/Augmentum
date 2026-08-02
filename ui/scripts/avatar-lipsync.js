/**
 * avatar-lipsync.js — Volume-gated shape cycling lip sync
 *
 * Uses the same approach as VSeeFace / VTube Studio / Animaze:
 * detect speech volume → cycle through 3 mouth shapes on a timer
 * with smooth blending. No FFT vowel detection (that produces fish-mouth).
 *
 * Visemes (VRM standard):
 *   aa — open mouth (あ)    — used as primary open shape
 *   oh — open rounded (お)  — secondary shape (variety)
 *   ee — wide/teeth (え)    — tertiary shape (variety)
 *   ih — smile/teeth (い)   — occasional flavor
 *   ou — rounded lips (う)  — occasional flavor
 *   jaw — overall open amount (synthetic, from volume)
 */

// Shape selection pool with English phonotactic-frequency weights.
// Real English vowel distribution (approximate, from large speech
// corpora — see Preston Blair's animation tables / VTube Studio
// Advanced Lipsync calibration data):
//   ~30% aa-class (open vowels: /æ/ /ɑ/ /ʌ/)
//   ~25% ih-class (lax neutrals: /ɪ/ /ɛ/ /ə/)
//   ~20% E-class  (front spread: /i/ /eɪ/)
//   ~13% oh-class (open round: /oʊ/ /ɔ/)
//   ~12% ou-class (close round: /u/ /ʊ/)
//
// Pre-PA configuration cycled aa/ee/ih round-robin, which weights
// each at 33% — overrepresenting the wide /ee/ shape. Weighted
// random pulled toward those frequencies looks more like real speech
// rhythm and breaks the eye-locking metronome that uniform cycling
// produces. Rounded shapes (oh/ou) ride in as a smaller flavor pool
// so they show up as genuine variety, not steady-state pucker.
//
// Earlier dominant configuration that triggered the user-reported
// fish-bowl: PRIMARY=['aa','oh','ee'], FLAVOR=['ih','ou']. Putting
// 'oh' (open + rounded) in primary kept the mouth in O-position
// ~33% of cycles regardless of what was being said.
const PRIMARY_SHAPE_WEIGHTS = [
  { shape: 'aa', weight: 0.40 },
  { shape: 'ih', weight: 0.33 },
  { shape: 'ee', weight: 0.27 },
];
// Cumulative array for fast weighted random pick. Each entry is
// { shape, threshold } where threshold is the cumulative weight up
// to and including that shape. The picker rolls Math.random() and
// returns the first entry whose threshold >= roll.
const _PRIMARY_CUM = (() => {
  let acc = 0;
  return PRIMARY_SHAPE_WEIGHTS.map((entry) => {
    acc += entry.weight;
    return { shape: entry.shape, threshold: acc };
  });
})();
const FLAVOR_SHAPES = ['oh', 'ou'];
const ALL_SHAPES = ['aa', 'ih', 'ou', 'ee', 'oh'];

// Timing
const MIN_SHAPE_DURATION = 0.12;  // seconds — fastest shape change (~8/sec for fast speech)
const MAX_SHAPE_DURATION = 0.25;  // seconds — slowest (~4/sec for slow speech)
const SILENCE_TIMEOUT = 0.6;      // seconds — long enough to bridge inter-sentence pauses
// Flavor (rounded) shapes appear ~8% of cycles. Lower than the prior
// 15% so puckered shapes are genuine variety, not a steady-state
// quarter of the visual.
const FLAVOR_CHANCE = 0.08;
// Insert a "rest" frame every ~4 cycles. Real speech has frequent
// natural micro-closures from bilabials (M/B/P); without them the
// mouth reads as pumping-without-pause. The rest frame is brief
// (one cycle's worth of duration) so the pacing stays natural, but
// its mere presence breaks up the continuous-open look that
// volume-driven cycling otherwise produces.
const REST_INSERT_EVERY = 4;
// Compound target for the rest frame. Pre-PA the rest used a uniform
// 0.05 across all 5 shapes which read as "all visemes at near-zero"
// — flat and slightly off. Real "mouth at rest mid-conversation"
// looks like a tiny relaxed open (small `ih`) with subtle teeth
// showing (small `ee`); using the 5-viseme basis to approximate
// that compound shape gives a natural micro-pause that bilabial
// closures (M/B/P) actually produce in real speech without needing
// the audio data to know they happened.
const REST_BLEND = {
  aa: 0.00,
  ih: 0.15,  // primary: relaxed small open
  ee: 0.05,  // secondary: tiny teeth show
  oh: 0.00,
  ou: 0.00,
};

// Smoothing — amplitude mode (legacy timer-cycling)
const OPEN_LERP = 0.25;   // how fast mouth opens (0-1, higher = snappier)
const CLOSE_LERP = 0.10;  // how fast mouth closes (slower = more natural hold)
const SHAPE_LERP = 0.20;  // how fast shapes blend into each other
// Fast-collapse lerp used when speech transitions to silence. VTube
// Studio's docs explicitly recommend driving all viseme weights to
// zero on the silence transition rather than freezing on the last
// shape; ~30 ms time-constant at 60 fps maps to lerp ~0.50.
const SILENCE_COLLAPSE_LERP = 0.50;

// Smoothing — schedule mode. Faster than amplitude because the schedule
// itself encodes phoneme timing; slow close-lerps would smear visemes
// across each other instead of replacing them at the right moments.
const SCHEDULE_OPEN_LERP = 0.45;
const SCHEDULE_CLOSE_LERP = 0.35;

// Volume
const SPEECH_THRESHOLD = 0.008;   // RMS below this = silence (lowered — TTS can be quiet)
const VOLUME_SCALE = 5.0;         // multiplier on RMS to get 0-1 intensity

// Dry-pulse mode — used when TTS is audibly playing but its audio does
// NOT flow through the analyser (iOS native-playback path; a failed
// MediaElementSource bind). The analyser reads silence, so amplitude
// detection can't drive the mouth — instead synthesize a gentle
// intensity envelope and let the normal shape-cycling/rest machinery
// run on top of it. Deliberately understated: a slow wobble around
// mid-low intensity reads as relaxed natural speech; anything punchier
// would look canned because it can't follow the actual audio. The
// per-viseme ceilings and rest frames apply unchanged, so the visual
// register matches the calibrated amplitude path.
const DRY_PULSE_BASE = 0.45;      // center of the synthetic envelope
const DRY_PULSE_WOBBLE = 0.15;    // ± slow sinusoidal variation
const DRY_PULSE_HZ = 0.45;        // wobble speed (full cycles per second)

// Per-viseme intensity ceilings. The 5 VRM visemes are NOT
// equally "visually loud" — `aa` (open neutral) reads as natural
// even at full opening (it's the everyday speaking shape), while
// `ee` (wide spread + teeth) and `ou` (puckered) look exaggerated
// at the same weight because each adds a second axis of motion
// (spread / pucker) on top of jaw open. Calibrate per shape so
// each lands in its own natural range:
//
//   aa  1.00 — open neutral, the most common English vowel; no cap.
//   ih  0.95 — small lax shape; near-full because the shape itself
//              is already understated (a small open with relaxed lips).
//   oh  0.75 — open + rounded; compound shape, dampen moderately.
//   ee  0.70 — wide spread / teeth visible; the "smile" — visually
//              loudest, dampen most.
//   ou  0.70 — puckered; reads strongly even at low weights.
//
// Production tools (VTube Studio docs explicitly: "vowel parameters
// never peak at 1.0 simultaneously"; Kalidokit's typical output:
// dominant shape ~0.6-0.8) calibrate similarly. These values match
// observed natural-speech weight ranges in JALI's reference curves.
const VISEME_INTENSITY = {
  aa: 1.00,
  ih: 0.95,
  ee: 0.70,
  oh: 0.75,
  ou: 0.70,
};

// Global lip-sync intensity scale. Multiplies every per-viseme
// ceiling above for a uniform additional dampener — useful per-VRM
// calibration without rewriting the shape table. 1.0 = the per-shape
// calibration above is the actual peak; 0.85 = scaled-down global
// (everything 15% softer); 1.15 = extra-energetic. Composes with
// _emotionDampen (situational, e.g. 0.6 for sad).
const LIPSYNC_INTENSITY_SCALE = 1.0;

export class AvatarLipSync {
  constructor() {
    this._timeData = null;
    this._rmsScratch = null;

    // Volume tracking
    this._smoothedRms = 0;
    this._peakRms = 0.03;

    // Shape cycling state
    this._currentShape = 'aa';
    this._nextShapeIn = 0.18;
    this._isSpeaking = false;
    this._silenceTimer = 0;
    // Cycle counter for periodic rest-frame insertion. Reset whenever
    // a rest frame is inserted so cadence stays consistent.
    this._cycleSinceRest = 0;
    // Latched flag for "was speaking last frame" — used to detect the
    // edge transition into silence so we can drive viseme weights to
    // zero once instead of every frame.
    this._wasSpeaking = false;

    // Output weights (smoothed)
    this._weights = { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0, jaw: 0 };

    // Targets (set by shape cycling, smoothed into _weights)
    this._targets = { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };

    this._emotionDampen = 1.0;
    this._forceSpeaking = false;
    // Dry-pulse: synthesize intensity when the active clip's audio
    // bypasses the analyser (iOS / failed bind). Set per-clip by the
    // tts-playback event's analyserDry flag; cleared on turn end.
    this._dryPulse = false;

    // Phoneme-driven schedule (Phase 1). When set, takes precedence over
    // amplitude shape-cycling. Falls back to amplitude when null.
    //   _schedule: array of {t, v, w} sorted by t (ms from clock origin)
    //   _scheduleClock: () => number — returns current playback ms
    //   _scheduleEndMs: when the schedule terminates (clear after this)
    this._schedule = null;
    this._scheduleClock = null;
    this._scheduleEndMs = 0;
  }

  setEmotionDampen(factor) {
    this._emotionDampen = Math.max(0.3, Math.min(1.0, factor));
  }

  /**
   * Force speaking state on/off. Use when the voice system knows
   * TTS is active but audio may have brief gaps between sentences.
   */
  setForceSpeaking(speaking) {
    this._forceSpeaking = speaking;
  }

  /**
   * Dry-pulse on/off. When on (and no viseme schedule is active),
   * update() drives the mouth from a synthetic intensity envelope
   * instead of analyser RMS — for clips whose audio bypasses the
   * analyser entirely. A phoneme schedule still takes precedence.
   */
  setDryPulse(active) {
    this._dryPulse = !!active;
  }

  /**
   * Attach a phoneme-driven viseme schedule. While active, update() drives
   * mouth shapes from the schedule instead of amplitude cycling.
   *
   * @param {{duration_ms: number, events: Array<{t:number,v:string,w:number}>}} schedule
   * @param {() => number} clockFn — returns elapsed ms since the schedule's
   *   audio playback began. For PCM (BufferSource), use
   *   `() => (audioCtx.currentTime - bufferStartCtxTime) * 1000`.
   *   For an HTMLAudioElement, use `() => audioEl.currentTime * 1000`.
   *   Returning a value past schedule.duration_ms ends the schedule cleanly.
   */
  setVisemeSchedule(schedule, clockFn) {
    if (!schedule || !Array.isArray(schedule.events) || !schedule.events.length) {
      this.clearVisemeSchedule();
      return;
    }
    if (typeof clockFn !== 'function') {
      this.clearVisemeSchedule();
      return;
    }
    // Defensive: ensure events are sorted by t.
    const events = schedule.events.slice().sort((a, b) => (a.t || 0) - (b.t || 0));
    this._schedule = events;
    this._scheduleClock = clockFn;
    this._scheduleEndMs = schedule.duration_ms || events[events.length - 1].t || 0;
    // Prime the amplitude speech-state machine to "speaking" while the
    // schedule drives. update() returns early on the schedule path, so these
    // fields keep whatever value they held before the schedule attached
    // (typically _isSpeaking=false). When the schedule self-terminates mid-
    // audio (its clock passes the end), control falls back to amplitude and
    // a stale _isSpeaking=false makes the mouth snap shut for a frame before
    // speech is re-detected. Priming here makes the handoff seamless;
    // _wasSpeaking=true avoids a spurious transitioning-to-silence collapse.
    this._isSpeaking = true;
    this._wasSpeaking = true;
    this._silenceTimer = 0;
  }

  /** Detach the active schedule and revert to amplitude cycling. */
  clearVisemeSchedule() {
    this._schedule = null;
    this._scheduleClock = null;
    this._scheduleEndMs = 0;
    // Keep the silence edge-detector consistent: sync _wasSpeaking to the
    // current speaking state so the first amplitude frame after an explicit
    // detach doesn't compute a spurious transitioning-to-silence collapse
    // from a stale _wasSpeaking (update() never touched it while the
    // schedule was active). A genuine transition to silence still fires
    // once amplitude actually reads silence and flips _isSpeaking false.
    this._wasSpeaking = this._isSpeaking;
  }

  /** True iff a schedule is currently driving mouth shapes. */
  hasActiveSchedule() {
    return this._schedule !== null;
  }

  /**
   * Call once per animation frame.
   * @param {AnalyserNode} analyser
   * @param {number} now — performance.now() / 1000
   * @returns {{ aa, ih, ou, ee, oh, jaw }} weights 0–1
   */
  update(analyser, now) {
    // Schedule path takes precedence when active. The schedule self-terminates
    // when the clock advances past its end, dropping us back to amplitude.
    if (this._schedule !== null) {
      const visemes = this._updateFromSchedule();
      if (visemes !== null) return visemes;
      // Schedule ended — fall through to amplitude (mouth has already closed
      // because the trailing sil event drove all shapes to zero).
    }

    // No analyser available (e.g. unit test, schedule-only client) — return
    // current weights without driving them, UNLESS dry-pulse is active:
    // the synthetic envelope needs no audio data.
    const hasAnalyser = !!analyser
      && typeof analyser.getFloatTimeDomainData === 'function';
    if (!hasAnalyser && !this._dryPulse) {
      return { ...this._weights };
    }

    let intensity = 0;
    let aboveThreshold = false;

    if (hasAnalyser) {
      // Get time-domain data for RMS calculation (more reliable than frequency data for volume)
      if (!this._timeData || this._timeData.length !== analyser.fftSize) {
        this._timeData = new Float32Array(analyser.fftSize);
      }
      analyser.getFloatTimeDomainData(this._timeData);

      // Compute RMS
      let sum = 0;
      for (let i = 0; i < this._timeData.length; i++) {
        sum += this._timeData[i] * this._timeData[i];
      }
      const rawRms = Math.sqrt(sum / this._timeData.length);

      // Adaptive peak tracking
      if (rawRms > this._peakRms) {
        this._peakRms = rawRms;
      } else {
        this._peakRms *= 0.9995;
        this._peakRms = Math.max(0.01, this._peakRms);
      }

      // Smooth RMS
      const rmsLerp = rawRms > this._smoothedRms ? 0.35 : 0.15;
      this._smoothedRms += (rawRms - this._smoothedRms) * rmsLerp;

      // Normalized intensity (0–1)
      intensity = Math.min(1.0, (this._smoothedRms / (this._peakRms * 0.6)) * VOLUME_SCALE);
      aboveThreshold = this._smoothedRms > SPEECH_THRESHOLD;
    }

    // Dry-pulse override — the clip is audibly playing but its audio
    // bypasses the analyser, so RMS reads silence. Synthesize a gentle
    // envelope; everything downstream (shape cycling, rest frames,
    // per-viseme ceilings) runs unchanged so the look matches the
    // calibrated amplitude path, just steadier.
    if (this._dryPulse) {
      const wobble = Math.sin(now * DRY_PULSE_HZ * 2 * Math.PI);
      intensity = DRY_PULSE_BASE + DRY_PULSE_WOBBLE * wobble;
      aboveThreshold = true;
    }

    // Frame delta (approximate from analyser timing)
    const dt = 1 / 60; // assume 60fps — close enough for shape timing

    // Speech state machine
    // _forceSpeaking keeps lip sync alive during inter-sentence gaps
    if (aboveThreshold || this._forceSpeaking) {
      this._silenceTimer = 0;
      if (!this._isSpeaking) {
        this._isSpeaking = true;
        this._nextShapeIn = 0.05; // start first shape quickly
      }
    } else {
      this._silenceTimer += dt;
      if (this._silenceTimer > SILENCE_TIMEOUT) {
        this._isSpeaking = false;
      }
    }

    if (this._isSpeaking) {
      // Count down to next shape change
      this._nextShapeIn -= dt;
      if (this._nextShapeIn <= 0) {
        this._advanceShape(intensity);
        // Next shape timing: faster when louder, slower when quieter
        this._nextShapeIn = MIN_SHAPE_DURATION + (1 - intensity) * (MAX_SHAPE_DURATION - MIN_SHAPE_DURATION);
        // Add small random jitter (±20%) for natural feel
        this._nextShapeIn *= 0.8 + Math.random() * 0.4;
      }

      // Set target: current shape gets intensity (capped by its
      // per-viseme ceiling), others decay. The '__rest__' sentinel
      // from _advanceShape applies REST_BLEND — a compound shape
      // that approximates a relaxed-mouth-mid-conversation pose
      // using fractional weights on multiple visemes. Per-viseme
      // ceilings on active speaking keep the visually loud shapes
      // (ee, ou) from over-dominating while letting natural shapes
      // (aa) reach full opening.
      const isRest = this._currentShape === '__rest__';
      for (const key of ALL_SHAPES) {
        if (isRest) {
          this._targets[key] = REST_BLEND[key] ?? 0;
        } else if (key === this._currentShape) {
          const cap = VISEME_INTENSITY[key] ?? 1.0;
          this._targets[key] = intensity * cap * LIPSYNC_INTENSITY_SCALE;
        } else {
          this._targets[key] = 0;
        }
      }
    } else {
      // Silence: all targets zero
      for (const key of ALL_SHAPES) {
        this._targets[key] = 0;
      }
    }

    // Edge-detect transition into silence — when isSpeaking flips
    // false this frame, force a fast collapse of all viseme weights
    // toward 0 instead of the slow CLOSE_LERP. Without this the
    // mouth holds the last vowel for the slow close window after
    // every sentence, which reads as "frozen on the last word."
    // VTube Studio's lip-sync wiki documents this as the canonical
    // gate behavior for amplitude-driven systems.
    const transitioningToSilence = this._wasSpeaking && !this._isSpeaking;
    this._wasSpeaking = this._isSpeaking;

    // Smooth weights toward targets
    for (const key of ALL_SHAPES) {
      const target = this._targets[key] * this._emotionDampen;
      const current = this._weights[key];
      const isOpening = target > current;
      let effectiveLerp;
      if (transitioningToSilence) {
        // Snap-to-rest collapse on the silence transition.
        effectiveLerp = SILENCE_COLLAPSE_LERP;
      } else if (key === this._currentShape) {
        // Active shape uses the asymmetric open/close lerps.
        effectiveLerp = isOpening ? OPEN_LERP : CLOSE_LERP;
      } else {
        // Other shapes blend at the shape-blend rate.
        effectiveLerp = SHAPE_LERP;
      }
      this._weights[key] += (target - current) * effectiveLerp;
      // Clamp
      if (this._weights[key] < 0.005) this._weights[key] = 0;
      if (this._weights[key] > 1.0) this._weights[key] = 1.0;
    }

    // Jaw: directly from smoothed intensity (not shape-cycling).
    // The 0.7 multiplier is the jaw-vs-shape balance (jaw never
    // hits full while a viseme could); LIPSYNC_INTENSITY_SCALE then
    // scales the whole pair so the jaw and viseme open in lockstep.
    const jawTarget = this._isSpeaking
      ? Math.min(1.0, intensity * 0.7) * LIPSYNC_INTENSITY_SCALE * this._emotionDampen
      : 0;
    this._weights.jaw += (jawTarget - this._weights.jaw) * (jawTarget > this._weights.jaw ? OPEN_LERP : CLOSE_LERP);

    return { ...this._weights };
  }

  /**
   * Drive viseme weights from the active schedule.
   * Returns null when the schedule has ended (caller falls through to amplitude).
   */
  _updateFromSchedule() {
    let elapsedMs;
    try {
      elapsedMs = this._scheduleClock();
    } catch {
      this.clearVisemeSchedule();
      return null;
    }
    if (typeof elapsedMs !== 'number' || !isFinite(elapsedMs)) {
      this.clearVisemeSchedule();
      return null;
    }

    // Schedule ended — let amplitude take over on next frame.
    if (elapsedMs > this._scheduleEndMs + 50) {
      this.clearVisemeSchedule();
      return null;
    }

    // Binary search would be faster for very long schedules, but with
    // ~30-100 events per sentence and 60 fps the linear scan is trivial.
    const events = this._schedule;
    let active = events[0];
    for (let i = 0; i < events.length; i++) {
      if (events[i].t <= elapsedMs) {
        active = events[i];
      } else {
        break;
      }
    }

    const v = active.v;
    const w = (typeof active.w === 'number') ? active.w : 0;

    // Set targets: only the active viseme is non-zero; everything else
    // decays. Per-viseme ceiling caps each shape at its natural
    // visual range, then LIPSYNC_INTENSITY_SCALE applies the global
    // dampener. Schedule path uses the same calibration as amplitude
    // so flipping engine modes via voice_lipsync_engine doesn't
    // produce a visual jump in mouth amplitude.
    const cap = VISEME_INTENSITY[v] ?? 1.0;
    const scaledW = w * cap * LIPSYNC_INTENSITY_SCALE;
    for (const key of ALL_SHAPES) {
      this._targets[key] = (key === v) ? scaledW : 0;
    }

    // Smooth weights toward targets — schedule mode uses faster lerps so
    // phoneme-rate transitions don't smear into each other.
    for (const key of ALL_SHAPES) {
      const target = this._targets[key] * this._emotionDampen;
      const current = this._weights[key];
      const isOpening = target > current;
      const lerp = isOpening ? SCHEDULE_OPEN_LERP : SCHEDULE_CLOSE_LERP;
      this._weights[key] += (target - current) * lerp;
      if (this._weights[key] < 0.005) this._weights[key] = 0;
      if (this._weights[key] > 1.0) this._weights[key] = 1.0;
    }

    // Jaw tracks the dominant viseme weight — closed during sil, open
    // during vowels. Same intensity scale + emotion dampen as the
    // viseme targets so jaw and visemes stay in lockstep.
    const jawTarget = (v === 'sil')
      ? 0
      : Math.min(1.0, w * 0.7) * LIPSYNC_INTENSITY_SCALE * this._emotionDampen;
    this._weights.jaw += (jawTarget - this._weights.jaw)
      * (jawTarget > this._weights.jaw ? SCHEDULE_OPEN_LERP : SCHEDULE_CLOSE_LERP);

    return { ...this._weights };
  }

  _advanceShape(intensity) {
    // Periodic rest insert — every Nth cycle the mouth ducks toward
    // a near-closed state for one cycle. Real speech has frequent
    // micro-closures from bilabials (M/B/P) that volume-driven
    // cycling can't see; this approximates the visual cadence.
    this._cycleSinceRest += 1;
    if (this._cycleSinceRest >= REST_INSERT_EVERY) {
      this._cycleSinceRest = 0;
      // 'rest' isn't a real viseme key — set _currentShape to a
      // sentinel that the target loop in update() treats as
      // "all viseme targets near-zero." We use 'aa' but the
      // intensity multiplier collapses the weight via REST_WEIGHT
      // applied below in the next update cycle.
      this._currentShape = '__rest__';
      return;
    }

    // Flavor pool (rounded shapes) wins ~FLAVOR_CHANCE of cycles
    // when the speaker is loud enough that an emphatic shape reads
    // naturally. Below the intensity threshold we stick to flat-lipped
    // primaries so quiet speech doesn't pucker.
    if (Math.random() < FLAVOR_CHANCE && intensity > 0.3) {
      const pool = FLAVOR_SHAPES.filter(s => s !== this._currentShape);
      this._currentShape = pool[Math.floor(Math.random() * pool.length)];
      return;
    }

    // Primary pick: weighted random across {aa: 40%, ih: 33%,
    // ee: 27%} matching English vowel frequency. Avoid immediate
    // repeats by re-rolling once if the picked shape equals the
    // current one — same-shape transitions are visually a no-op
    // and waste a cycle.
    const next = this._weightedPickPrimary();
    if (next === this._currentShape) {
      this._currentShape = this._weightedPickPrimary();
    } else {
      this._currentShape = next;
    }
  }

  _weightedPickPrimary() {
    const roll = Math.random();
    for (let i = 0; i < _PRIMARY_CUM.length; i++) {
      if (roll <= _PRIMARY_CUM[i].threshold) {
        return _PRIMARY_CUM[i].shape;
      }
    }
    // Floating-point fallthrough: return last bucket.
    return _PRIMARY_CUM[_PRIMARY_CUM.length - 1].shape;
  }

  getRMS(analyser) {
    if (!analyser || typeof analyser.getFloatTimeDomainData !== 'function') {
      return this._smoothedRms;
    }

    const size = analyser.fftSize || 0;
    if (!size) return this._smoothedRms;

    if (!this._rmsScratch || this._rmsScratch.length !== size) {
      this._rmsScratch = new Float32Array(size);
    }

    analyser.getFloatTimeDomainData(this._rmsScratch);
    let sum = 0;
    for (let i = 0; i < this._rmsScratch.length; i++) {
      sum += this._rmsScratch[i] * this._rmsScratch[i];
    }
    return Math.sqrt(sum / this._rmsScratch.length);
  }

  getSmoothedRMS() {
    return this._smoothedRms;
  }

  reset() {
    for (const key of [...ALL_SHAPES, 'jaw']) {
      this._weights[key] = 0;
      this._targets[key] = 0;
    }
    this._smoothedRms = 0;
    this._isSpeaking = false;
    this._silenceTimer = 0;
    this._peakRms = 0.03;
    this._dryPulse = false;
    this.clearVisemeSchedule();
  }
}
