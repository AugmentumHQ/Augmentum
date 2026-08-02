/**
 * embodiment-engine.js — autonomous behavior layer on top of MotionEngine.
 *
 * Gives the avatar internal state (mood, attention, energy, speaking state)
 * that decays/drifts over time and reacts to external events. The engine
 * runs on a per-frame tick and decides:
 *
 *   - What pose primitive to be in (sampled from the loaded library,
 *     scored by mood + tag affinity + personality + recency)
 *   - Where to look (gaze target — user / self / world / specific landmark)
 *   - What expression to wear (mood vector → blendshape weights)
 *   - How energetically to perform transitions (energy → curve modulation)
 *
 * The avatar stops being a puppet and starts being an autonomous presence.
 *
 * Architecture:
 *
 *   external events                       per-frame
 *   ───────────────                       ─────────
 *   onMessage(text, sentiment)            tick(dt)
 *   onUserActivity(type)                    │
 *   onSpeakingState(state)                  │
 *   onWorldEvent(name, data)                ▼
 *                                       ┌──────────────────┐
 *           │                           │ decay mood,      │
 *           ▼                           │ drift energy,    │
 *      ┌──────────┐                     │ pick idle pose,  │
 *      │ event    │  ──── feeds ────►   │ update gaze,     │
 *      │ queue    │                     │ update expression│
 *      └──────────┘                     └──────────────────┘
 *                                              │
 *                                              ▼
 *                                       motion engine
 *                                       (pose / gaze / expr channels)
 *
 * No three.js dependency on the engine itself. Reads the motion engine
 * and writes via its channels.
 */

const DEFAULT_PERSONALITY = Object.freeze({
  name: 'default',
  baselineEnergy: 0.5,             // 0..1, where energy drifts to
  // Resting state: zero on each face emotion (neutral expression) +
  // mild internal energy. The expression mapping subtracts baseline
  // before amplifying, so any non-zero baseline here would manifest as
  // a permanent slight expression — usually unwanted.
  baselineMood: Object.freeze({ intensity: 0.20 }),
  gestureFrequencyMs: 9000,        // mean ms between idle gestures
  gestureJitterMs: 4000,           // ± randomization on that interval
  emotionalReactivity: 1.0,        // 0..2 multiplier on mood bumps from events
  energyReactivity: 1.0,           // 0..2 multiplier on energy bumps from events
  preferredFamilies: ['idle_standing', 'thinking'],
  habitualPrimitives: [],          // names this personality prefers
  avoidFamilies: [],
  attentionPersistMs: 4000,        // how long to hold attention before drifting
  blinkRateHz: 0.2,                // baseline blinks per second (0.2 = 1 every 5s)
});

// Mood decay half-lives, in seconds. Different emotions persist differently.
const MOOD_HALF_LIFE_SEC = Object.freeze({
  happy:     45,   // joy lingers
  sad:       60,   // sadness sticks around
  angry:     20,   // anger burns hot but fades faster
  surprised:  5,   // spike-and-fade
  intensity: 25,   // overall arousal returns to baseline in ~half a minute
});

const EMOTION_TO_EXPRESSION = Object.freeze({
  happy:     'happy',
  sad:       'sad',
  angry:     'angry',
  surprised: 'surprised',
});

const SPEAKING_STATES = Object.freeze({
  IDLE:      'idle',
  LISTENING: 'listening',
  THINKING:  'thinking',
  SPEAKING:  'speaking',
});

export class EmbodimentEngine {
  /**
   * @param {object} opts
   * @param {object} opts.motionEngine        MotionEngine instance
   * @param {object} [opts.personality]       PersonalityConfig (see DEFAULT_PERSONALITY)
   * @param {Iterable<object>} [opts.primitives]  pose primitive JSONs to make available
   * @param {function} [opts.onLog]           callback for diagnostic logging
   */
  constructor(opts) {
    if (!opts?.motionEngine) throw new Error('EmbodimentEngine requires motionEngine');
    this.motionEngine = opts.motionEngine;
    this.personality = { ...DEFAULT_PERSONALITY, ...(opts.personality || {}) };
    this.onLog = opts.onLog || (() => {});

    // ─── State (the "mind") ────────────────────────────────────────────
    this.mood = { happy: 0, sad: 0, angry: 0, surprised: 0, intensity: this.personality.baselineMood.intensity };
    Object.assign(this.mood, this.personality.baselineMood);
    this.energy = this.personality.baselineEnergy;
    this.speakingState = SPEAKING_STATES.IDLE;
    this.attention = { target: 'forward', worldPos: null, setAtMs: 0 };
    this.lastIdleGestureMs = 0;
    this.lastEventMs = 0;
    this._nextIdleAtMs = performance.now() + this.personality.gestureFrequencyMs;
    this._currentPose = null;
    this._recentPoseNames = [];   // recency cache to avoid immediate repeats

    // ─── Pose library ─────────────────────────────────────────────────
    /** name → primitive */
    this._primitives = new Map();
    if (opts.primitives) {
      for (const p of opts.primitives) this.registerPrimitive(p);
    }

    // ─── Internal tracking ─────────────────────────────────────────────
    this._eventQueue = [];
    this._tickCount = 0;
    this._enabled = true;
  }

  registerPrimitive(primitive) {
    if (primitive?.name) this._primitives.set(primitive.name, primitive);
  }

  setEnabled(b) { this._enabled = !!b; }

  // ─── External event hooks ────────────────────────────────────────────
  /**
   * Incoming message from the conversation. Sentiment derived elsewhere
   * (e.g. from the narrative engine).
   * @param {object} sentiment  { valence: -1..1, intensity: 0..1, topics?: string[] }
   */
  onMessage(sentiment = {}) {
    const r = this.personality.emotionalReactivity;
    const er = this.personality.energyReactivity;
    const valence = sentiment.valence ?? 0;
    const intensity = sentiment.intensity ?? 0.3;
    // Positive valence → happy bumps; negative → sad/angry split by intensity
    if (valence > 0) {
      this.mood.happy = clamp01(this.mood.happy + valence * 0.4 * r);
      this.mood.sad = clamp01(this.mood.sad - valence * 0.15 * r);
    } else if (valence < 0) {
      const neg = -valence;
      // High-intensity negative → angry; low-intensity negative → sad
      const angryShare = clamp01(intensity);
      this.mood.angry = clamp01(this.mood.angry + neg * 0.35 * r * angryShare);
      this.mood.sad   = clamp01(this.mood.sad   + neg * 0.4  * r * (1 - angryShare));
      this.mood.happy = clamp01(this.mood.happy - neg * 0.2 * r);
    }
    // Intensity scales overall arousal + energy
    this.mood.intensity = clamp01(this.mood.intensity + intensity * 0.35 * r);
    this.energy = clamp01(this.energy + intensity * 0.25 * er);
    this.lastEventMs = performance.now();
    // High-intensity events surface as a surprise spike too
    if (intensity > 0.6) {
      this.mood.surprised = clamp01(this.mood.surprised + (intensity - 0.5) * 0.6);
    }
    // Attention shifts to user briefly on every message
    this.lookAt('user');
    this._eventQueue.push({ type: 'message', sentiment });
  }

  /**
   * Contact event from a ContactReactor: user touched (or just left) a
   * region of the avatar's body. Region-specific reactions:
   *
   *   - face (cheek_L/R, forehead, chin, mouth): blink + look at hand
   *     + spike happy/surprised mood, gentle inward head tilt
   *   - shoulder_L/R: glance toward hand, slight engagement
   *   - chest_L/R, sternum: recoil mood (surprised), gaze averted briefly
   *   - hand_L/R: warm — happy mood, sustained gaze, no recoil
   *   - intimate regions (belly low, hip inward): privacy reaction
   *     (gaze away, slight backward lean via mood)
   *   - other: generic mild reaction
   */
  onContactEvent(evt) {
    const region = evt.region || '';
    const released = !!evt.released;
    if (released) {
      // Light touch off — gaze can drift back to user, no mood spike.
      this.lookAt('user');
      return;
    }
    const reactivity = this.personality.emotionalReactivity;
    // Face touches — affectionate / surprising contact
    if (/^(cheek|forehead|chin|mouth|temple|jaw|nose)/.test(region)) {
      this.mood.happy     = clamp01(this.mood.happy + 0.35 * reactivity);
      this.mood.surprised = clamp01(this.mood.surprised + 0.4 * reactivity);
      this.mood.intensity = clamp01(this.mood.intensity + 0.3 * reactivity);
      this.lookAt('user');
      this.onLog?.({ kind: 'contact-react', region, response: 'face-touch' });
      return;
    }
    // Shoulder — friendly engagement
    if (/^shoulder_/.test(region)) {
      this.mood.happy     = clamp01(this.mood.happy + 0.25 * reactivity);
      this.mood.intensity = clamp01(this.mood.intensity + 0.2 * reactivity);
      this.lookAt('user');
      this.onLog?.({ kind: 'contact-react', region, response: 'shoulder-engage' });
      return;
    }
    // Hand-on-hand — warm, sustained
    if (/^hand_/.test(region)) {
      this.mood.happy     = clamp01(this.mood.happy + 0.4 * reactivity);
      this.mood.intensity = clamp01(this.mood.intensity + 0.25 * reactivity);
      this.lookAt('user');
      this.onLog?.({ kind: 'contact-react', region, response: 'hand-warm' });
      return;
    }
    // Chest / sternum — privacy/surprise
    if (/^(chest_|sternum|back_upper)/.test(region)) {
      this.mood.surprised = clamp01(this.mood.surprised + 0.5 * reactivity);
      this.mood.intensity = clamp01(this.mood.intensity + 0.4 * reactivity);
      this.lookAt('forward');   // gaze averted briefly
      this.onLog?.({ kind: 'contact-react', region, response: 'chest-startle' });
      return;
    }
    // Intimate / lower body — strong privacy reaction
    if (/^(hip_|belly|navel|thigh_|side_)/.test(region)) {
      this.mood.surprised = clamp01(this.mood.surprised + 0.55 * reactivity);
      this.mood.angry     = clamp01(this.mood.angry + 0.2 * reactivity);
      this.lookAt('forward');
      this.onLog?.({ kind: 'contact-react', region, response: 'privacy-recoil' });
      return;
    }
    // Default — mild surprise
    this.mood.surprised = clamp01(this.mood.surprised + 0.3 * reactivity);
    this.mood.intensity = clamp01(this.mood.intensity + 0.2 * reactivity);
    this.onLog?.({ kind: 'contact-react', region, response: 'generic' });
  }

  /** User typed / moved / clicked. Adjusts attention + energy slightly. */
  onUserActivity(type = 'mouse') {
    this.lookAt('user');
    // Slight energy bump on engagement
    this.energy = clamp01(this.energy + 0.04 * this.personality.energyReactivity);
    this.lastEventMs = performance.now();
  }

  onSpeakingState(state) {
    if (!Object.values(SPEAKING_STATES).includes(state)) return;
    const prev = this.speakingState;
    this.speakingState = state;
    if (state === SPEAKING_STATES.SPEAKING && prev !== SPEAKING_STATES.SPEAKING) {
      // Started speaking — slight energy bump, look at user
      this.energy = clamp01(this.energy + 0.1 * this.personality.energyReactivity);
      this.lookAt('user');
    }
    if (state === SPEAKING_STATES.LISTENING) {
      this.lookAt('user');
    }
    if (state === SPEAKING_STATES.THINKING) {
      this.lookAt('thinking');
    }
  }

  /** Generic world event. Anyone can fire these. */
  onWorldEvent(name, data = {}) {
    this._eventQueue.push({ type: 'world', name, data });
  }

  /** Set the gaze target by symbolic name or world position. */
  lookAt(target, worldPos = null) {
    this.attention = { target, worldPos, setAtMs: performance.now() };
  }

  // ─── Per-frame tick ──────────────────────────────────────────────────
  tick(dtMs, nowMs = performance.now()) {
    if (!this._enabled) return;
    const dtSec = dtMs / 1000;
    this._tickCount++;

    // 1. Mood decay — exponential with per-emotion half-life
    for (const [emotion, halfLife] of Object.entries(MOOD_HALF_LIFE_SEC)) {
      const baseline = this.personality.baselineMood[emotion] ?? 0;
      const k = Math.pow(0.5, dtSec / halfLife);
      this.mood[emotion] = baseline + (this.mood[emotion] - baseline) * k;
    }

    // 2. Energy drift toward personality baseline
    const eBaseline = this.personality.baselineEnergy;
    this.energy += (eBaseline - this.energy) * Math.min(1, dtSec * 0.25);

    // 3. Attention decay — after persistMs, drift back toward 'forward'
    const attentionAge = nowMs - this.attention.setAtMs;
    if (this.attention.target !== 'forward' && attentionAge > this.personality.attentionPersistMs) {
      // Soft decay: hold a bit longer if recent event, otherwise drift
      if (nowMs - this.lastEventMs > this.personality.attentionPersistMs) {
        this.attention = { target: 'forward', worldPos: null, setAtMs: nowMs };
      }
    }

    // 4. Idle gesture: when in IDLE speaking state and overdue, pick one
    if (this.speakingState === SPEAKING_STATES.IDLE && nowMs >= this._nextIdleAtMs) {
      this._performIdleGesture();
      this._scheduleNextIdle(nowMs);
    }

    // 5. Update gaze channel (read attention → world position)
    this._updateGaze();

    // 6. Update expression channel from mood
    this._updateExpression();
  }

  _scheduleNextIdle(nowMs) {
    const jitter = (Math.random() - 0.5) * 2 * this.personality.gestureJitterMs;
    this._nextIdleAtMs = nowMs + Math.max(2000, this.personality.gestureFrequencyMs + jitter);
  }

  // ─── Pose selection ──────────────────────────────────────────────────
  _performIdleGesture() {
    if (this._primitives.size === 0) return;
    const pick = this._pickMoodAppropriate({ topN: 5 });
    if (pick) this._applyPose(pick);
  }

  /**
   * Pick a pose primitive that fits the current mood + state, without
   * applying it. Useful for callers that want to know what would happen.
   * @param {object} [opts]
   * @param {number} [opts.topN]      pick weighted-random from top N (default 5)
   * @param {string[]} [opts.bias]    tag list — primitives matching these get a score bonus
   * @returns {object|null}           primitive object, or null
   */
  pickMoodAppropriate(opts = {}) {
    return this._pickMoodAppropriate(opts);
  }

  _pickMoodAppropriate({ topN = 5, bias = null } = {}) {
    if (this._primitives.size === 0) return null;
    const candidates = [...this._primitives.values()];
    const scored = candidates.map(p => {
      let s = this._scorePrimitive(p);
      if (bias && p.tags?.some(t => bias.includes(t))) s *= 1.5;
      return { p, s };
    });
    scored.sort((a, b) => b.s - a.s);
    const N = Math.min(topN, scored.length);
    const top = scored.slice(0, N);
    const totalW = top.reduce((s, x) => s + Math.max(0.01, x.s), 0);
    let r = Math.random() * totalW;
    let pick = top[0].p;
    for (const x of top) {
      r -= Math.max(0.01, x.s);
      if (r <= 0) { pick = x.p; break; }
    }
    return pick;
  }

  /**
   * Called by the consumer (bench / orchestrator) when a VRMA animation
   * just finished. Picks a mood-appropriate landing primitive and
   * transitions into it from the current bone state. The PoseTransitionChannel
   * snapshots whatever frame the VRMA ended on, so the handoff is smooth.
   *
   * @param {object} [opts]
   * @param {number} [opts.duration]   transition length ms (default 900 — quicker than idle)
   * @param {string[]} [opts.bias]     tag bias for the landing pose ('idle', 'casual', etc.)
   * @returns {object|null}            the primitive that was applied, or null
   */
  performLandingGesture({ duration = 900, bias = ['idle'] } = {}) {
    const pick = this._pickMoodAppropriate({ topN: 4, bias });
    if (!pick) return null;
    const ch = this.motionEngine.channel('pose_transition');
    if (!ch) return null;
    // Make sure pose transitions are enabled to take over from VRMA
    if (!ch.enabled) ch.enabled = true;
    ch.setTarget(pick, { duration, energy: this.energy });
    this._currentPose = pick;
    this._recentPoseNames.unshift(pick.name);
    if (this._recentPoseNames.length > 6) this._recentPoseNames.pop();
    this.lastIdleGestureMs = performance.now();
    this._scheduleNextIdle(performance.now());   // reset idle clock so it doesn't fire immediately
    this.onLog?.({ kind: 'landing', name: pick.name, energy: this.energy, mood: { ...this.mood } });
    return pick;
  }

  _scorePrimitive(p) {
    if (!p) return 0;
    let s = 1.0;
    // Recency penalty
    const recentIdx = this._recentPoseNames.indexOf(p.name);
    if (recentIdx >= 0) s *= Math.max(0.05, 0.3 * recentIdx / 5);   // very recent = much lower
    // Personality habitual bonus
    if (this.personality.habitualPrimitives?.includes(p.name)) s *= 1.4;
    // Family preferences
    if (this.personality.preferredFamilies?.includes(p.family)) s *= 1.3;
    if (this.personality.avoidFamilies?.includes(p.family)) s *= 0.3;
    // Mood-tag affinity
    const tags = p.tags || [];
    if (this.mood.happy > 0.4 && tags.some(t => ['casual','confident','open','greeting','playful'].includes(t))) s *= 1.4;
    if (this.mood.happy > 0.4 && tags.some(t => ['defensive','closed','sad','contemplative'].includes(t))) s *= 0.5;
    if (this.mood.sad > 0.4 && tags.some(t => ['contemplative','self-soothing','nervous','closed'].includes(t))) s *= 1.4;
    if (this.mood.sad > 0.4 && tags.some(t => ['confident','power','assertive','greeting'].includes(t))) s *= 0.4;
    if (this.mood.angry > 0.4 && tags.some(t => ['assertive','emphatic','sincere','confident'].includes(t))) s *= 1.3;
    if (this.mood.angry > 0.4 && tags.some(t => ['casual','formal','polite'].includes(t))) s *= 0.5;
    if (this.mood.surprised > 0.4 && tags.some(t => ['greeting','gesture','open'].includes(t))) s *= 1.2;
    // High intensity prefers emphatic
    if (this.mood.intensity > 0.6 && tags.some(t => ['emphatic','assertive','expression'].includes(t))) s *= 1.3;
    if (this.mood.intensity < 0.3 && tags.some(t => ['idle','neutral','attentive'].includes(t))) s *= 1.2;
    // Single-arm atoms are less appropriate as full idle states
    if (tags.includes('single-arm')) s *= 0.6;
    // Per-arm filtering: only pick per_arm if intensity high enough (small fidgets)
    if (p.family === 'per_arm' && this.mood.intensity < 0.4) s *= 0.3;
    return s;
  }

  _applyPose(primitive) {
    const ch = this.motionEngine.channel('pose_transition');
    if (!ch) return;
    // Duration biased by energy (high energy = snappier)
    const baseDuration = 1100;
    const durationJitter = (Math.random() - 0.5) * 400;
    const duration = baseDuration + durationJitter;
    ch.setTarget(primitive, { duration, energy: this.energy });
    this._currentPose = primitive;
    this._recentPoseNames.unshift(primitive.name);
    if (this._recentPoseNames.length > 6) this._recentPoseNames.pop();
    this.lastIdleGestureMs = performance.now();
    this.onLog?.({ kind: 'pose-applied', name: primitive.name, energy: this.energy, mood: { ...this.mood } });
  }

  // ─── Gaze ────────────────────────────────────────────────────────────
  _updateGaze() {
    const gazeCh = this.motionEngine.channel('gaze');
    if (!gazeCh) return;
    const tgt = this.attention.target;
    let pos = null;
    if (tgt === 'user') {
      // Heuristic: user is at the camera position. Caller can override
      // via setUserWorldPosition().
      pos = this._userWorldPos || [0, 1.4, 1.6];
    } else if (tgt === 'forward') {
      // Look slightly down to "rest" gaze
      pos = [0, 1.3, 1.6];
    } else if (tgt === 'thinking') {
      // Look up-left to "thinking" stance (classic deliberation pose)
      pos = [-0.3, 1.6, 0.5];
    } else if (tgt === 'self') {
      // Look down at chest/hand area
      pos = this._selfWorldPos || [0, 1.0, 0.3];
    } else if (this.attention.worldPos) {
      pos = this.attention.worldPos;
    }
    if (pos) {
      if (!gazeCh.enabled) gazeCh.enabled = true;
      gazeCh.setTargetWorld(pos);
    }
  }

  setUserWorldPosition(pos) { this._userWorldPos = pos; }
  setSelfWorldPosition(pos) { this._selfWorldPos = pos; }

  // ─── Expression ──────────────────────────────────────────────────────
  //
  // Mood → expression weights. Three things matter for realism:
  //
  //   1. AMPLIFICATION: VRM morph weights need to reach ~0.5+ to be
  //      perceptually visible at typical viewing distance. Small mood
  //      values (0.1-0.3) should still produce readable expressions.
  //      We use a power curve with exponent 0.4 (more aggressive than
  //      sqrt at 0.5), which lifts small mood values toward visible
  //      while still saturating gracefully at 1.
  //
  //   2. COMPETITION: When multiple emotions are active (e.g. happy
  //      AND sad), letting them sum to >1 produces a muddy blended look.
  //      Instead, we let the DOMINANT emotion (highest mood value) take
  //      most of the weight budget, with secondary emotions scaled down.
  //      This is closer to how real faces work — a face doesn't show
  //      smile and frown at full intensity simultaneously.
  //
  //   3. THRESHOLDING: Very small mood values (<0.05) produce zero
  //      expression rather than a barely-visible flicker. Prevents
  //      noise from creating "always slightly twitching" face.
  // ─────────────────────────────────────────────────────────────────────
  _updateExpression() {
    const exprCh = this.motionEngine.channel('expression');
    if (!exprCh) return;
    if (!exprCh.enabled) exprCh.enabled = true;

    // Build amplified weights. Three principles for "subtle resting,
    // visible on events":
    //   1. Subtract baseline so resting mood = no expression. Only the
    //      delta from baseline gets visualized.
    //   2. Threshold below ~0.08 mood-delta → zero weight. Avoids
    //      barely-visible twitching from decay tail.
    //   3. Gentle amplification (pow 0.6) and cap at 0.75 — VRoid
    //      morphs read clearly at 0.5, going higher just opens the
    //      mouth disproportionately on most rigs.
    const raw = [];
    for (const [emotion, expr] of Object.entries(EMOTION_TO_EXPRESSION)) {
      const m = this.mood[emotion] ?? 0;
      const baseline = this.personality.baselineMood[emotion] ?? 0;
      const delta = Math.max(0, m - baseline);
      const amplified = delta < 0.08 ? 0 : Math.min(0.75, Math.pow(delta, 0.6));
      raw.push({ emotion, expr, mood: m, amplified });
    }
    raw.sort((a, b) => b.amplified - a.amplified);

    // Dominant takes full amplified value. Secondary gets up to 25% of its
    // own amplified value scaled by how close it is to the dominant.
    // Anything below the dominant by >0.25 is suppressed entirely.
    const weights = {};
    for (let i = 0; i < raw.length; i++) {
      const r = raw[i];
      if (r.amplified <= 0) { weights[r.expr] = 0; continue; }
      if (i === 0) {
        weights[r.expr] = r.amplified;
      } else {
        const lead = raw[0].amplified;
        const gap = lead - r.amplified;
        const compete = Math.max(0, 1 - gap / 0.25);
        weights[r.expr] = r.amplified * 0.25 * compete;
      }
    }
    exprCh.setExpressions(weights);
  }

  // ─── Inspector / diagnostics ─────────────────────────────────────────
  inspect() {
    return {
      mood: { ...this.mood },
      energy: this.energy,
      speakingState: this.speakingState,
      attention: { ...this.attention },
      currentPose: this._currentPose?.name || null,
      recentPoses: [...this._recentPoseNames],
      nextIdleInMs: Math.max(0, this._nextIdleAtMs - performance.now()),
      personality: this.personality.name,
      tickCount: this._tickCount,
      primitivesLoaded: this._primitives.size,
    };
  }
}

function clamp01(v) { return Math.max(0, Math.min(1, v)); }

// Pre-canned personality profiles. Pass to EmbodimentEngine constructor.
export const PERSONALITY_PROFILES = Object.freeze({
  default:  { ...DEFAULT_PERSONALITY },
  energetic: {
    ...DEFAULT_PERSONALITY,
    name: 'energetic',
    baselineEnergy: 0.7,
    baselineMood: { happy: 0.25, intensity: 0.5 },
    gestureFrequencyMs: 6000,
    emotionalReactivity: 1.4,
    energyReactivity: 1.3,
    preferredFamilies: ['idle_standing', 'thinking'],
    habitualPrimitives: ['confident_both_hips', 'making_a_point', 'palms_open_low'],
  },
  reserved: {
    ...DEFAULT_PERSONALITY,
    name: 'reserved',
    baselineEnergy: 0.35,
    baselineMood: { happy: 0.05, intensity: 0.2 },
    gestureFrequencyMs: 12000,
    emotionalReactivity: 0.7,
    preferredFamilies: ['formal', 'formal_behind', 'thinking'],
    habitualPrimitives: ['hands_clasped_front', 'idle_hands_behind', 'thoughtful_chin_touch'],
    attentionPersistMs: 6000,
  },
  thoughtful: {
    ...DEFAULT_PERSONALITY,
    name: 'thoughtful',
    baselineEnergy: 0.42,
    baselineMood: { happy: 0.08, intensity: 0.32 },
    gestureFrequencyMs: 10000,
    preferredFamilies: ['thinking', 'idle_standing'],
    habitualPrimitives: ['thoughtful_chin_touch', 'stomach_hold', 'palms_open_low'],
    attentionPersistMs: 5000,
  },
});

export const SPEAKING_STATE_NAMES = SPEAKING_STATES;
