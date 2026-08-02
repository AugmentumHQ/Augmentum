/**
 * avatar-pose-trigger.js — voice/transcript event router
 *
 * Translates voice + transcript events into INTENTS and emits them to
 * the MovementConductor. The conductor handles selection, cooldowns,
 * energy budget, and dispatch — this module just classifies and routes.
 *
 * Intents are { roles, emotion } pairs. The atlas (anim-atlas.js)
 * holds the pool of candidates each intent can match.
 *
 * Wiring (consumed from voice.js):
 *   const engine = new PoseTriggerEngine({ conductor, character });
 *   engine.start();
 *   engine.onCallOpened();
 *   engine.onUserTranscriptFinal(text);
 *   engine.onUserStartedSpeaking();
 *   engine.onResponseStarted(text);
 *   engine.onResponseEndedSpeaking();
 *   engine.dispose();
 */

// ─── Sentiment & intent detection (regex v1) ──────────────────────────
//
// Cheap keyword/regex classifier. Returns one of:
//   'positive' | 'emphatic' | 'thoughtful' | 'question' | 'neutral'
function classifySentiment(text) {
  if (!text) return 'neutral';
  const t = text.trim();
  if (/[!]{2,}|^WOW|amazing|incredible|awesome|wonderful|brilliant|fantastic/i.test(t)) return 'emphatic';
  if (/\b(love|happy|glad|excited|great|excellent|perfect|exactly|yes!)\b/i.test(t)) return 'positive';
  if (/\b(hmm|let me think|interesting|i wonder|consider|perhaps|maybe)\b/i.test(t)) return 'thoughtful';
  if (/\?\s*$/.test(t) || /^(why|how|what|when|where|who|which|do you|are you|can you|would you)\b/i.test(t)) return 'question';
  return 'neutral';
}

function detectGreeting(text) {
  if (!text) return false;
  return /^(hi|hey|hello|yo|good (morning|afternoon|evening)|howdy)\b/i.test(text.trim());
}

function detectFarewell(text) {
  if (!text) return false;
  return /\b(goodbye|bye|good night|going to (sleep|bed)|see you (later|tomorrow|soon)|gotta go|talk (to you )?(later|soon)|ttyl|catch you later|signing off|have a good (night|day))\b/i.test(text);
}

function detectGoodNews(text) {
  if (!text) return false;
  return /\b(i (got|won|finished|completed|aced|nailed)|great news|good news|i'?m (so )?(happy|excited|thrilled)|finally( did)?|made it)\b/i.test(text);
}

// ─── Sentiment → intent map ──────────────────────────────────────────
// Each AI-response sentiment maps to a roles + emotion vector. The
// atlas's tagged candidates score against these via select(), so
// adding a new animation that fits these tags automatically joins
// the eligible pool — no edit here required.
const SENTIMENT_INTENT = {
  positive: {
    roles: ['celebrate', 'agree-strong', 'react-positive'],
    emotion: { warmth: 0.8, energy: 0.75, openness: 0.7, focus: 0.65 },
  },
  emphatic: {
    roles: ['emphasize', 'react-positive', 'excitement-peak'],
    emotion: { warmth: 0.7, energy: 0.85, openness: 0.7, focus: 0.75 },
  },
  thoughtful: {
    roles: ['think', 'ponder'],
    emotion: { warmth: 0.5, energy: 0.3, openness: 0.4, focus: 0.85 },
  },
  question: {
    roles: ['question', 'curiosity'],
    emotion: { warmth: 0.6, energy: 0.5, openness: 0.7, focus: 0.75 },
  },
  // 'neutral' deliberately omitted — silence is a valid response,
  // procedural breathing/sway carries.
};

// ─── Explicit pose requests ──────────────────────────────────────────
// User-initiated direct asks bypass cooldown + variety. Each entry
// either targets a specific atlas id (precise gesture) or emits an
// intent (lets the conductor pick from a tag-matched pool).
const POSE_REQUEST_PATTERNS = [
  { pattern: /\b(wave( at me| hello| hi| to me)?|say hi|greet me|wave back)\b/i,
    id: 'hello' },
  { pattern: /\b(jump and wave|excited wave|big wave)\b/i,
    id: 'wave-jump' },
  { pattern: /\b(peace sign|do peace|v[- ]sign|peace fingers|throw up (a |the )?peace)\b/i,
    id: 'peace-sign' },
  { pattern: /\b(spin|twirl|turn around|do a spin|pirouette)\b/i,
    id: 'spin' },
  { pattern: /\b(dance|do a dance|show me (some )?dance moves|let'?s dance)\b/i,
    intent: { roles: ['dance', 'show-off'] } },
  { pattern: /\b(model pose|strike a pose|pose for me|^pose$|fashion pose)\b/i,
    id: 'model-pose' },
  { pattern: /\b(shoot|gun (gesture|fingers)|pew pew|finger guns?|aim)\b/i,
    id: 'shoot' },
  { pattern: /\b(cheer (me on)?|encourage me|hype me up|root for me|celebrate)\b/i,
    intent: { roles: ['celebrate', 'agree-strong'] } },
  { pattern: /\b(check (your )?phone|look at (your )?phone|use (your )?phone|smartphone)\b/i,
    id: 'smartphone' },
  { pattern: /\b(deep bow|dogeza|prostrate|kneel down|^bow$|take a bow)\b/i,
    id: 'dogeza' },
  // Note: show-body, drink-water, squat patterns from the previous
  // version are dropped — the underlying VRMAs (VRMA_01, drinkwater,
  // VRMA_07) are not in the v1 atlas. Re-add as explicit-only atlas
  // entries if/when those clips are wanted back.
];

const NEGATION_RE = /\b(don'?t|do not|stop|never|no\s+\w+\s+please|please don'?t|quit)\b/i;

function parseExplicitPoseRequest(text) {
  if (!text) return null;
  for (const entry of POSE_REQUEST_PATTERNS) {
    const match = text.match(entry.pattern);
    if (!match) continue;
    const before = text.slice(Math.max(0, match.index - 30), match.index);
    if (NEGATION_RE.test(before)) continue;
    return entry;
  }
  return null;
}

// ─── PoseTriggerEngine ────────────────────────────────────────────────

export class PoseTriggerEngine {
  /**
   * @param {object} opts
   * @param {MovementConductor} opts.conductor   from avatar.js (avatarState.conductor)
   * @param {object} [opts.character]            avatar profile (for per-id bias)
   * @param {Function} [opts.classifier]         text → sentiment override
   */
  constructor(options = {}) {
    if (!options.conductor) {
      throw new Error('PoseTriggerEngine requires opts.conductor');
    }
    this.conductor = options.conductor;
    this.character = options.character || {};
    this.classifier = options.classifier || classifySentiment;

    this.callStartedAt = 0;
    this.lastUserActivityAt = 0;
    this.lastIdleEscalationAt = 0;
    this.callPhase = 'opening';   // 'opening' | 'flowing' | 'closing'
    this._idleTimer = null;
    this._disposed = false;

    // Apply per-character bias if the profile carries one.
    if (this.character.pose_bias) {
      this.conductor.setBias(this.character.pose_bias);
    }
  }

  start() {
    if (this._disposed) return;
    this._idleTimer = setInterval(() => this._idleTick(), 5000);
  }

  dispose() {
    this._disposed = true;
    if (this._idleTimer) {
      clearInterval(this._idleTimer);
      this._idleTimer = null;
    }
  }

  // ─── Event hooks (called by voice.js) ──────────────────────────────

  onCallOpened() {
    this.callStartedAt = Date.now();
    this.lastUserActivityAt = Date.now();
    this.lastIdleEscalationAt = Date.now();
    this.callPhase = 'opening';
    this.conductor.play({
      roles: ['greet'],
      emotion: { warmth: 0.85, energy: 0.7, openness: 0.7, focus: 0.6 },
    });
  }

  onUserStartedSpeaking() {
    this.lastUserActivityAt = Date.now();
    this.lastIdleEscalationAt = Date.now();
    if (this.callPhase === 'opening') this.callPhase = 'flowing';
    // Subtle attentive shift — most calls won't fire because
    // `lean-in`'s 90s cooldown will gate it. That's the point: gate
    // by cooldown, not by probability. The conductor's recency filter
    // also prevents back-to-back same-id fires.
    this.conductor.play({
      roles: ['listen', 'attentive', 'idle-shift'],
      emotion: { warmth: 0.65, energy: 0.4, openness: 0.65, focus: 0.8 },
    });
  }

  onUserStoppedSpeaking() {
    this.lastUserActivityAt = Date.now();
  }

  onUserTranscriptFinal(text) {
    this.lastUserActivityAt = Date.now();
    if (!text) return;

    // Explicit pose request takes precedence — bypasses cooldown / variety.
    const explicit = parseExplicitPoseRequest(text);
    if (explicit) {
      if (explicit.id) {
        this.conductor.playById(explicit.id);
      } else if (explicit.intent) {
        this.conductor.play(explicit.intent, { explicit: true });
      }
      return;
    }

    // First few seconds + greeting = explicit greeting trigger
    const fromCallStart = (Date.now() - this.callStartedAt) / 1000;
    if (fromCallStart < 12 && detectGreeting(text)) {
      this.onCallOpened();   // re-roll the greeting
    }
    if (detectFarewell(text)) {
      this.callPhase = 'closing';
      this.conductor.play({
        roles: ['farewell'],
        emotion: { warmth: 0.8, energy: 0.5, openness: 0.7, focus: 0.6 },
      });
    }
    if (detectGoodNews(text)) {
      this.conductor.play({
        roles: ['celebrate', 'gratitude', 'react-positive'],
        emotion: { warmth: 0.85, energy: 0.7, openness: 0.75, focus: 0.65 },
      });
    }
  }

  onResponseStarted(text) {
    this.lastUserActivityAt = Date.now();
    const sentiment = this.classifier(text);
    const intent = SENTIMENT_INTENT[sentiment];
    if (intent) this.conductor.play(intent);
  }

  onResponseEndedSpeaking() {
    this.lastUserActivityAt = Date.now();
    this.lastIdleEscalationAt = Date.now();
  }

  // ─── Internal: idle escalation ─────────────────────────────────────

  _idleTick() {
    if (this.callPhase === 'closing') return;
    const idleMs = Date.now() - this.lastUserActivityAt;
    const sinceLastEscalation = Date.now() - this.lastIdleEscalationAt;
    let intent = null;

    if (idleMs >= 90000 && sinceLastEscalation >= 90000) {
      // Long idle: dance / show-off (kebab/28/25 dances, plus jump-vv).
      intent = { roles: ['dance', 'show-off'] };
    } else if (idleMs >= 45000 && sinceLastEscalation >= 45000) {
      // Mid idle: distracted activities (smartphone, model pose, etc.).
      intent = { roles: ['idle-fill', 'idle-distracted'] };
    } else if (idleMs >= 15000 && sinceLastEscalation >= 15000) {
      // Short idle: subtle shift (lean-in, lookAround, model-pose).
      intent = { roles: ['idle-shift', 'idle-attentive'] };
    }

    if (intent) {
      this.lastIdleEscalationAt = Date.now();
      this.conductor.play(intent);
    }
  }

  // ─── Adaptation passthrough ────────────────────────────────────────
  // For wiring a future reaction observer (transcript-driven). The
  // conductor's bias map is the durable surface; we just route here.
  recordReaction(animId, signal) {
    this.conductor.recordReaction(animId, signal);
  }
}
