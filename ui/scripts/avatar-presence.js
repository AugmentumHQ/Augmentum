/**
 * avatar-presence.js — Conversational Presence Engine
 *
 * Replaces keyword→gesture mapping with physics-based conversational energy.
 * Four continuous variables drive all avatar behavior:
 *   - presence  (0-1): How "here" the avatar is
 *   - flow      (-1 to 1): Energy direction (negative=receiving, positive=giving)
 *   - temperature (0-1): Emotional intensity
 *   - resonance (0-1): Conversational sync level
 *
 * Animation emerges from these variables — nothing is explicitly triggered.
 */

// ---------------------------------------------------------------------------
// Smooth interpolation helpers
// ---------------------------------------------------------------------------

/** Exponential decay toward target. */
function _decay(current, target, halflife, dt) {
  if (halflife <= 0) return target;
  return target + (current - target) * Math.exp(-0.6931472 * dt / halflife);
}

/** Clamp value to range. */
function _clamp(v, min, max) { return v < min ? min : v > max ? max : v; }

function _nowSeconds() {
  if (typeof performance !== 'undefined' && performance.now) return performance.now() / 1000;
  return Date.now() / 1000;
}

// ---------------------------------------------------------------------------
// Energy Model
// ---------------------------------------------------------------------------

const ENERGY_DEFAULTS = {
  // Halflife controls how fast each variable responds (seconds)
  presenceHalflife: 1.5,       // slow rise/fall — feels like attention shifting
  flowHalflife: 0.4,           // medium — tracks turn-taking
  temperatureHalflife: 2.0,    // slow — emotional state is sticky
  resonanceHalflife: 3.0,      // slowest — sync builds over conversation

  // Presence dynamics
  presencePerUserWord: 0.015,  // user speech pumps presence
  presencePerAiSentence: 0.04, // AI completing a sentence pumps presence
  presenceDecayIdle: 0.08,     // presence loss per second when nobody speaks
  presenceFloor: 0.15,         // never fully "gone"

  // Flow dynamics
  flowUserSpeaking: -0.8,      // target when user speaks
  flowAiSpeaking: 0.7,         // target when AI speaks
  // Target when ANOTHER AI character (the peer in a group call) speaks.
  // Less intense than flowUserSpeaking — listening to a peer is still
  // receiving energy, but a user holding the floor commands more attention.
  flowPeerSpeaking: -0.6,
  flowIdle: 0.0,               // neutral at rest

  // Temperature dynamics
  temperatureDecay: 0.03,      // per-second decay toward baseline
  temperatureBaseline: 0.2,    // resting temperature (not zero — always some life)
  temperatureImpulse: 0.3,     // spike per emotional shift

  // Resonance dynamics
  resonancePerExchange: 0.08,  // rises when turns alternate
  resonanceDecayPerSecond: 0.02,
  resonanceDecayLongSilence: 0.06, // faster decay after 10s silence
};

export class PresenceEngine {
  constructor(config = {}) {
    this._config = { ...ENERGY_DEFAULTS, ...config };

    // --- Energy state ---
    this.presence = 0.3;     // start slightly present
    this.flow = 0.0;
    this.temperature = this._config.temperatureBaseline;
    this.resonance = 0.0;

    // --- Conversation tracking ---
    this._lastSpeaker = null;  // 'user' | 'ai' | null
    this._silenceTime = 0;
    this._turnCount = 0;
    this._lastTurnTime = 0;

    // --- Emotion momentum ---
    this._emotionValence = 0;     // -1 (negative) to +1 (positive)
    this._emotionArousal = 0;     // 0 (calm) to 1 (intense)
    this._prevEmotion = 'neutral';
    this._emotionMomentum = 0;    // rate of change
    this._sentimentWindow = [];   // recent sentiment scores for arc tracking

    // --- Behavior output (read by avatar.js each frame) ---
    this.gesture = null;           // gesture name to trigger (consumed after read)
    this.emotion = 'neutral';      // current derived emotion
    // Override state — set by setEmotionOverride() when the companion
    // runtime's bus bridges interior affect/state into the visual
    // layer. Empty + expired = falls through to audio-derived path.
    this._emotionOverride = '';
    this._emotionOverrideExpiresAt = 0;
    this.breathModifier = { rate: 1.0, depth: 1.0 };
    this.idleAction = null;        // idle action to trigger (consumed after read)

    // --- Idle behavior ---
    this._idleTimer = 0;
    this._idleEscalation = 0;     // increases with sustained idle
    this._nextIdleAt = 8 + Math.random() * 8;

    // --- Gesture cadence ---
    this._gestureCooldown = 0;
    this._sentenceBoundaryCount = 0;

    // --- User listening ---
    this._userWordCount = 0;
    this._userSentenceCount = 0;
    this._listeningNodTimer = 0;
    this._nextNodAt = 0;

    // --- Semantic action planning ---
    this._avatarProfile = config.avatarProfile || null;
    this._semanticBuffer = '';
    this._semanticQueue = [];
  }

  setAvatarProfile(profile) {
    this._avatarProfile = profile || null;
  }

  // -------------------------------------------------------------------------
  // Frame update — called every animation frame
  // -------------------------------------------------------------------------

  update(dt) {
    const C = this._config;

    // --- Presence decay ---
    if (this._lastSpeaker === null) {
      this._silenceTime += dt;
      this.presence = _decay(this.presence, C.presenceFloor, C.presenceHalflife * 2, dt);
    } else {
      this._silenceTime = 0;
    }
    this.presence = _clamp(this.presence, 0, 1);

    // --- Flow toward target ---
    let flowTarget = C.flowIdle;
    if (this._lastSpeaker === 'user') flowTarget = C.flowUserSpeaking;
    else if (this._lastSpeaker === 'ai') flowTarget = C.flowAiSpeaking;
    else if (this._lastSpeaker === 'peer') flowTarget = C.flowPeerSpeaking;
    this.flow = _decay(this.flow, flowTarget, C.flowHalflife, dt);
    this.flow = _clamp(this.flow, -1, 1);

    // --- Temperature decay toward baseline ---
    this.temperature = _decay(this.temperature, C.temperatureBaseline, C.temperatureHalflife, dt);
    this.temperature = _clamp(this.temperature, 0, 1);

    // --- Resonance decay ---
    const resoDecay = this._silenceTime > 10 ? C.resonanceDecayLongSilence : C.resonanceDecayPerSecond;
    this.resonance = Math.max(0, this.resonance - resoDecay * dt);

    // --- Sentiment decay toward neutral (emotions fade if not reinforced) ---
    const sentDecayRate = 0.12; // per-second decay
    if (this._emotionValence > 0) {
      this._emotionValence = Math.max(0, this._emotionValence - sentDecayRate * dt);
    } else if (this._emotionValence < 0) {
      this._emotionValence = Math.min(0, this._emotionValence + sentDecayRate * dt);
    }
    this._emotionArousal = Math.max(0, this._emotionArousal - sentDecayRate * 0.8 * dt);
    this._emotionMomentum = Math.max(0, this._emotionMomentum - 0.3 * dt);

    // --- Derive breath modifier from energy ---
    this.breathModifier.rate = 0.7 + this.temperature * 0.8 + Math.abs(this.flow) * 0.3;
    this.breathModifier.depth = 0.8 + (1 - this.temperature) * 0.4 + this.presence * 0.3;

    // --- Idle behavior generation ---
    this._updateIdle(dt);

    // --- Listening behavior generation ---
    this._updateListening(dt);

    // --- Speaking gesture cadence ---
    this._gestureCooldown = Math.max(0, this._gestureCooldown - dt);

    // --- Derive emotion from valence + arousal ---
    this._deriveEmotion();
  }

  // -------------------------------------------------------------------------
  // Input events — called by avatar.js / voice.js
  // -------------------------------------------------------------------------

  /** Call when voice state changes (listening, recording, processing, speaking). */
  onStateChange(state) {
    switch (state) {
      case 'recording':
        this._lastSpeaker = 'user';
        this._bumpPresence(0.05);
        break;
      case 'processing':
        this._lastSpeaker = null; // brief gap
        break;
      case 'speaking':
        if (this._lastSpeaker === 'user') {
          // Turn exchange — boost resonance
          this._turnCount++;
          this.resonance = _clamp(this.resonance + this._config.resonancePerExchange, 0, 1);
          this._lastTurnTime = performance.now() / 1000;
          // Dampen carryover sentiment from user's turn so AI starts fresh
          this._emotionValence *= 0.3;
          this._emotionArousal *= 0.4;
        }
        this._lastSpeaker = 'ai';
        this._bumpPresence(0.06);
        this._sentenceBoundaryCount = 0;
        break;
      case 'listening':
        // If coming from speaking, the AI just finished
        if (this._lastSpeaker === 'ai') {
          this._lastSpeaker = null;
        }
        break;
      case 'peer_speaking':
        // Another character (not us, not the user) is holding the floor.
        // Maps _lastSpeaker to 'peer' so the flow target picks up
        // flowPeerSpeaking and _updateListening fires (its gate is
        // flow <= -0.2, which -0.6 satisfies).
        this._lastSpeaker = 'peer';
        this._bumpPresence(0.04);
        this._sentenceBoundaryCount = 0;
        break;
    }
  }

  /** Call with user transcript text (partial or final). */
  onUserTranscript(text, isFinal) {
    if (!text) return;
    const words = text.trim().split(/\s+/).length;
    this._userWordCount += words;
    this._bumpPresence(this._config.presencePerUserWord * words);

    if (isFinal) {
      this._userSentenceCount++;
      // Question detection — raises temperature slightly (user is engaged)
      if (/\?$/.test(text.trim())) {
        this.temperature = _clamp(this.temperature + 0.05, 0, 1);
      }
    }
  }

  /**
   * Call with user's mic RMS (0-1) each frame.
   * Drives presence + temperature from voice energy, not just text.
   */
  onUserAudioRMS(rms) {
    if (rms < 0.01) return;
    // Loud speaking boosts temperature (energetic user)
    if (rms > 0.05) {
      this.temperature = _clamp(this.temperature + rms * 0.02, 0, 1);
    }
    // Any audible speech maintains presence
    this._bumpPresence(rms * 0.005);

    // Reactive backchannel — loud user audio while listening triggers
    // a brief attentive lean / acknowledge gesture as the body-level
    // equivalent of "mm" / "yeah." Cooldown keeps it sparse so sustained
    // loud speech doesn't bounce her head non-stop. Sets this.gesture
    // directly rather than queueing — the semantic queue only drains
    // during speaking flow, so listening-side reactions need the direct
    // path that consumeGesture() reads each frame.
    if (rms > 0.15 && this.flow < -0.2 && this._gestureCooldown <= 0) {
      const now = _nowSeconds();
      // 3.5 → 5.5s spacing (2026-06-11): combined with the listening
      // timer's shared budget this caps total acks at roughly one per
      // 5s of sustained speech, instead of one every ~2s.
      if (now - (this._lastReactiveAt || 0) > 5.5) {
        this._lastReactiveAt = now;
        const raw = this.resonance > 0.5 ? 'call_attentive_lean' : 'call_acknowledge';
        const gesture = _normalizeGestureName(raw);
        if (gesture) {
          this.gesture = gesture;
          this._gestureCooldown = 1.5;
        }
      }
    }
  }

  /**
   * Call with the ambient media RMS (0-1) each frame — music she's
   * "with", a narration playing, etc. Distinct from user audio:
   * media is environmental, not addressed to her. Maintains presence
   * (she's engaged with what's playing) and modestly raises
   * temperature for energetic content (music).
   *
   * Call site: avatar.js animate loop synthesizes a target from
   * AudioBus's per-kind activity and lerps toward it.
   */
  onMediaAudioRMS(rms) {
    if (rms < 0.01) return;
    // Half the user-audio gain — she's listening, not being addressed.
    if (rms > 0.05) {
      this.temperature = _clamp(this.temperature + rms * 0.01, 0, 1);
    }
    this._bumpPresence(rms * 0.0025);
  }

  /** Call with each LLM delta chunk. */
  onLLMDelta(text) {
    if (!text) return;

    const completedSentences = this._collectCompletedSentences(text);
    for (const sentence of completedSentences) {
      this._queueSemanticActionForSentence(sentence);
    }

    // Sentence boundary detection
    const sentences = text.match(/[.!?]+/g);
    if (sentences) {
      this._sentenceBoundaryCount += sentences.length;
      this._bumpPresence(this._config.presencePerAiSentence * sentences.length);
      this._maybeGesture();
    }

    // Punctuation-based temperature
    const exclamations = (text.match(/!/g) || []).length;
    const questions = (text.match(/\?/g) || []).length;
    const ellipsis = (text.match(/\.\.\./g) || []).length;
    if (exclamations) this.temperature = _clamp(this.temperature + 0.04 * exclamations, 0, 1);
    if (questions) this.temperature = _clamp(this.temperature + 0.02 * questions, 0, 1);
    if (ellipsis) this.temperature = _clamp(this.temperature - 0.02, 0, 1); // contemplative

    // Sentiment analysis — lightweight keyword scoring
    this._analyzeSentiment(text);
  }

  /** Call with [gesture:name] tags extracted from LLM output. */
  onExplicitGesture(name) {
    // Explicit tags are high-confidence — bypass cadence
    const gesture = _normalizeGestureName(name, { explicit: true });
    if (!gesture) return;
    this.gesture = gesture;
    this._gestureCooldown = Math.max(this._gestureCooldown, 0.75);
  }

  /** Consume the current gesture (call after reading). Returns name or null. */
  consumeGesture() {
    const g = this.gesture;
    this.gesture = null;
    return g;
  }

  /** Consume the current idle action. Returns name or null. */
  consumeIdleAction() {
    const a = this.idleAction;
    this.idleAction = null;
    return a;
  }

  // -------------------------------------------------------------------------
  // Internal: Sentiment & emotion
  // -------------------------------------------------------------------------

  /** Lightweight sentiment scoring — tracks momentum, not just state. */
  _analyzeSentiment(text) {
    const lower = text.toLowerCase();
    let score = 0;
    let arousal = 0;

    // Negation window — words near "not/don't/no/never/isn't/aren't" flip polarity
    const hasNegation = /\b(not|n't|no|never|neither|nor|hardly|barely|without)\b/.test(lower);

    // Positive signals
    const positiveWords = ['happy', 'glad', 'smile', 'laugh', 'joy', 'love', 'wonderful',
      'great', 'amazing', 'incredible', 'fantastic', 'awesome', 'beautiful', 'thank',
      'excited', 'perfect', 'brilliant', 'delighted', 'pleased', 'excellent'];
    // Negative signals
    const negativeWords = ['sad', 'sorry', 'unfortunately', 'regret', 'fear', 'angry',
      'terrible', 'horrible', 'awful', 'worried', 'anxious', 'disappointed', 'upset',
      'frustrated', 'painful', 'grief', 'loss', 'fail', 'wrong', 'bad'];
    // High-arousal signals
    const arousalWords = ['wow', 'incredible', 'amazing', 'unbelievable', 'shocked',
      'thrilling', 'intense', 'urgent', 'critical', 'danger', 'exciting', 'furious'];
    // Low-arousal signals
    const calmWords = ['calm', 'peace', 'gentle', 'quiet', 'soft', 'slow', 'rest',
      'consider', 'reflect', 'contemplate', 'perhaps', 'maybe'];

    // Score per word reduced (0.08 not 0.15) so emotions need sustained signal
    const posWeight = hasNegation ? -0.04 : 0.08; // negation flips positive → mildly negative
    const negWeight = hasNegation ? 0.03 : -0.08;  // negation dampens negative → slightly positive
    for (const w of positiveWords) { if (lower.includes(w)) score += posWeight; }
    for (const w of negativeWords) { if (lower.includes(w)) score += negWeight; }
    for (const w of arousalWords) { if (lower.includes(w)) arousal += 0.10; }
    for (const w of calmWords) { if (lower.includes(w)) arousal -= 0.06; }

    if (score === 0 && arousal === 0) return;

    // Update momentum (rate of change matters more than absolute)
    const prevValence = this._emotionValence;
    this._emotionValence = _clamp(this._emotionValence + score, -1, 1);
    this._emotionArousal = _clamp(this._emotionArousal + arousal, 0, 1);

    // Momentum = how much valence changed (transitions are expressive)
    const signedDelta = this._emotionValence - prevValence;
    this._emotionMomentum = Math.abs(signedDelta);

    // Momentum spikes temperature (transitions create energy)
    if (this._emotionMomentum > 0.1) {
      this.temperature = _clamp(
        this.temperature + this._config.temperatureImpulse * this._emotionMomentum,
        0, 1
      );
    }

    // Strong signed valence flips deserve a deliberate beat. Inject a
    // high-confidence semantic action so the next gesture window expresses
    // the transition — light laugh for a positive jump, thoughtful pause
    // for a sobering one. The queue's same-action dedup keeps this from
    // spamming when several sentiment-laden sentences land in a row.
    if (signedDelta > 0.15) {
      this._queueSyntheticAction('call_light_laugh', 0.9);
    } else if (signedDelta < -0.15) {
      this._queueSyntheticAction('call_thoughtful_pause', 0.9);
    }

    // Track sentiment window for arc detection
    this._sentimentWindow.push({ valence: this._emotionValence, arousal: this._emotionArousal });
    if (this._sentimentWindow.length > 20) this._sentimentWindow.shift();
  }

  /** Map continuous valence + arousal to discrete emotion name. */
  _deriveEmotion() {
    // External override (from the companion runtime's affect/state bus
    // events) takes priority while it's fresh. The override decays
    // linearly so the avatar drifts back to audio-derived emotion
    // smoothly rather than snapping at the deadline.
    if (this._emotionOverride && Date.now() < this._emotionOverrideExpiresAt) {
      this.emotion = this._emotionOverride;
      this._prevEmotion = this._emotionOverride;
      return;
    }

    const v = this._emotionValence;
    const a = this._emotionArousal;

    let emotion;
    if (a > 0.5 && v > 0.35) emotion = 'excited';
    else if (a > 0.5 && v < -0.35) emotion = 'angry';
    else if (a > 0.45) emotion = 'surprised';
    else if (v > 0.4) emotion = 'happy';
    else if (v < -0.4 && a < 0.3) emotion = 'sad';
    else if (v < -0.25) emotion = 'nervous';
    else if (a < 0.2 && Math.abs(v) < 0.15) emotion = 'relaxed';
    else if (this._emotionMomentum > 0.08) emotion = 'curious';
    else emotion = 'neutral';

    if (emotion !== this._prevEmotion) {
      this._prevEmotion = emotion;
    }
    this.emotion = emotion;
  }

  /**
   * Push an interior-state-driven emotion onto the avatar's face for a
   * limited duration. Used by the companion-runtime bus bridge — when
   * affect.changed or state.transition arrives, the bridge maps the
   * payload to an emotion name and calls this so the visible expression
   * follows interior reality instead of being computed entirely from
   * local audio cues.
   *
   * @param {string} emotion   - one of the names _deriveEmotion produces
   *                              ('curious', 'happy', 'sad', 'relaxed',
   *                               'nervous', 'neutral', 'excited',
   *                               'surprised', 'angry'). Unknown names
   *                              fall through silently.
   * @param {number} ms        - how long the override remains active.
   *                              The audio-derived value resumes after.
   */
  setEmotionOverride(emotion, ms = 6000) {
    const valid = new Set([
      'curious', 'happy', 'sad', 'relaxed', 'nervous',
      'neutral', 'excited', 'surprised', 'angry',
    ]);
    if (!emotion || !valid.has(emotion)) return;
    this._emotionOverride = emotion;
    this._emotionOverrideExpiresAt = Date.now() + Math.max(500, Math.min(60000, ms));
    // Apply immediately so the next render sees it without waiting
    // for the periodic _deriveEmotion() cycle.
    this.emotion = emotion;
    this._prevEmotion = emotion;
  }

  // -------------------------------------------------------------------------
  // Internal: Behavior generation
  // -------------------------------------------------------------------------

  /** Presence bump (clamped). */
  _bumpPresence(amount) {
    this.presence = _clamp(this.presence + amount, 0, 1);
  }

  /** Speaking gesture cadence — gestures emerge at sentence rhythm. */
  _maybeGesture() {
    if (this._gestureCooldown > 0) return;
    if (this.flow < 0.2) return; // not in speaking mode

    // Explicit sentence-driven actions (wrap_up, key_point, etc.) carry deliberate
    // intent; they bypass the probability gate so RNG can't suppress a planned beat.
    const topSemantic = this._peekSemanticAction();
    const hasHighConfidence = topSemantic && topSemantic.score >= 0.8;

    if (!hasHighConfidence) {
      const hasSemanticAction = this._semanticQueue.length > 0;
      const prob = hasSemanticAction
        ? Math.min(0.88, 0.45 + this.temperature * 0.25 + this.presence * 0.1)
        : 0.15 + this.temperature * 0.3 + this.presence * 0.1;
      if (Math.random() > prob) return;
    }

    // Select gesture based on emotional state
    const gesture = this._selectSpeakingGesture();
    if (gesture) {
      this.gesture = gesture;
      // Cooldown scales inversely with temperature (energetic = more gestures)
      this._gestureCooldown = 2.0 + (1 - this.temperature) * 3.0;
    }
  }

  /** Pick a contextually appropriate gesture for speaking. */
  _selectSpeakingGesture() {
    const planned = this._consumeSemanticAction();
    if (planned) return planned;

    const e = this.emotion;
    const pool = _SPEAKING_GESTURE_POOLS[e] || _SPEAKING_GESTURE_POOLS.neutral;
    const safePool = pool.map(name => _normalizeGestureName(name)).filter(Boolean);
    if (!safePool.length) return null;
    return safePool[Math.floor(Math.random() * safePool.length)];
  }

  /** Idle behavior escalation. */
  _updateIdle(dt) {
    // Only generate idle when presence is low and nobody is speaking
    if (this.presence > 0.4 || this._lastSpeaker !== null) {
      this._idleTimer = 0;
      this._idleEscalation = 0;
      this._nextIdleAt = 8 + Math.random() * 8;
      return;
    }

    this._idleTimer += dt;
    if (this._idleTimer < this._nextIdleAt) return;

    // Escalating idle actions
    this._idleEscalation++;

    // Predictability breaker: sometimes she just... doesn't. A skipped
    // beat reads as stillness, and stillness is what makes the moves
    // that DO happen feel alive instead of scheduled (2026-06-11:
    // "very predictable and constantly goes off").
    const skipped = Math.random() < 0.3;
    if (!skipped) {
      const actions = _IDLE_ACTIONS[Math.min(this._idleEscalation - 1, _IDLE_ACTIONS.length - 1)];
      const safeActions = actions
        .map(name => _normalizeGestureName(name))
        .filter(Boolean)
        .filter(name => name !== this._lastIdleAction || actions.length === 1);
      if (safeActions.length) {
        this.idleAction = safeActions[Math.floor(Math.random() * safeActions.length)];
        this._lastIdleAction = this.idleAction;
      }
    }

    // Next idle interval LENGTHENS with escalation — long-idle settles
    // toward calm (~45-90s between motions), not a 7s metronome. The
    // old formula (16 - escalation*2, floor 7) was written for short
    // quiet stretches in calls; on the always-on widget it ratcheted
    // to the floor within a minute and stayed there for hours.
    const baseInterval = Math.min(45, 14 + this._idleEscalation * 6);
    this._nextIdleAt = this._idleTimer + baseInterval + Math.random() * baseInterval;
  }

  /** Listening micro-behaviors. */
  _updateListening(dt) {
    // Only when receiving (user speaking)
    if (this.flow > -0.2 || this.presence < 0.2) {
      this._listeningNodTimer = 0;
      return;
    }

    this._listeningNodTimer += dt;
    // Nod interval based on resonance (high resonance = more frequent
    // nods). 2026-06-11 recalibrated: the old 1.5-2.5s cadence at high
    // resonance — PLUS the independent reactive backchannel — stacked
    // into a nod every ~2s, bobblehead territory. Real listeners ack
    // sparsely; the reactive path already covers emphasis moments.
    if (this._nextNodAt === 0) {
      this._nextNodAt = 5.0 + (1 - this.resonance) * 4.0;
    }

    if (this._listeningNodTimer >= this._nextNodAt) {
      // Shared nod budget with the reactive backchannel — a timer nod
      // within ~4s of a reactive ack reads as a tic, not attention.
      const now = _nowSeconds();
      const sinceReactive = now - (this._lastReactiveAt || 0);
      if (this._gestureCooldown <= 0 && sinceReactive > 4.0) {
        const gesture = this.resonance > 0.58 ? 'call_attentive_lean' : 'call_acknowledge';
        this.gesture = _normalizeGestureName(gesture);
        this._gestureCooldown = 1.2;
        // Stamp the shared budget both ways — a reactive ack right
        // after a timer nod is the same tic in the other order.
        this._lastReactiveAt = now;
      }
      this._listeningNodTimer = 0;
      this._nextNodAt = 4.0 + (1 - this.resonance) * 4.0 + Math.random() * 2.0;
    }
  }

  /** Reset on deactivation. */
  dispose() {
    this.presence = 0.3;
    this.flow = 0.0;
    this.temperature = this._config.temperatureBaseline;
    this.resonance = 0.0;
    this._lastSpeaker = null;
    this._sentimentWindow = [];
    this._semanticBuffer = '';
    this._semanticQueue = [];
  }

  _collectCompletedSentences(text) {
    this._semanticBuffer = `${this._semanticBuffer}${text}`;
    const completed = [];
    const sentenceRe = /[^.!?]*[.!?]+/g;
    let match;
    let lastEnd = 0;
    while ((match = sentenceRe.exec(this._semanticBuffer)) !== null) {
      const sentence = match[0].trim();
      if (sentence) completed.push(sentence);
      lastEnd = sentenceRe.lastIndex;
    }
    if (lastEnd > 0) {
      this._semanticBuffer = this._semanticBuffer.slice(lastEnd);
    }
    if (this._semanticBuffer.length > 420) {
      this._semanticBuffer = this._semanticBuffer.slice(-220);
    }
    return completed;
  }

  _queueSemanticActionForSentence(sentence) {
    const plan = _planSemanticAction(sentence);
    if (!plan) return;
    this._queueSyntheticAction(plan.action, plan.score);
  }

  _queueSyntheticAction(rawAction, score) {
    const action = this._normalizePlannedAction(rawAction);
    if (!action) return;
    if (this._semanticQueue.some((item) => item.action === action)) return;

    this._semanticQueue.push({
      action,
      score,
      expiresAt: _nowSeconds() + 8,
    });
    this._semanticQueue.sort((a, b) => b.score - a.score);
    this._semanticQueue = this._semanticQueue.slice(0, 4);
  }

  _peekSemanticAction() {
    const now = _nowSeconds();
    this._semanticQueue = this._semanticQueue.filter((item) => item.expiresAt > now);
    return this._semanticQueue[0] || null;
  }

  _consumeSemanticAction() {
    const now = _nowSeconds();
    this._semanticQueue = this._semanticQueue.filter((item) => item.expiresAt > now);
    const next = this._semanticQueue.shift();
    return next?.action || null;
  }

  _normalizePlannedAction(action) {
    const normalized = _normalizeGestureName(action);
    if (!normalized) return null;
    if (this._profileAllowsAction(normalized)) return normalized;
    return _normalizeGestureName('call_acknowledge');
  }

  _profileAllowsAction(action) {
    const compat = this._avatarProfile?.callActions?.speaking?.[action];
    if (!compat) return true;
    return compat.canAuto || (compat.canPlay && compat.score >= 0.86);
  }
}

// ---------------------------------------------------------------------------
// Gesture pools — selected by emotion during speaking
// ---------------------------------------------------------------------------
const _SPEAKING_GESTURE_POOLS = {
  neutral:   ['call_acknowledge', 'call_clarify_question'],
  happy:     ['call_acknowledge', 'call_light_laugh'],
  sad:       ['call_thoughtful_pause', 'call_acknowledge'],
  excited:   ['call_key_point', 'call_acknowledge'],
  angry:     ['call_clarify_question', 'call_acknowledge'],
  surprised: ['call_clarify_question', 'call_acknowledge'],
  curious:   ['call_clarify_question', 'call_acknowledge'],
  nervous:   ['call_thoughtful_pause', 'call_acknowledge'],
  thinking:  ['call_thoughtful_pause', 'call_acknowledge'],
  relaxed:   ['call_acknowledge', 'call_clarify_question'],
};

// ---------------------------------------------------------------------------
// Idle actions — escalating tiers
// ---------------------------------------------------------------------------
const _IDLE_ACTIONS = [
  // Tier 0: barely-there signs of life
  ['blink_slow', 'weight_shift', 'look_away'],
  // Tier 1: quiet attention shifts
  ['call_grounding_breath', 'head_tilt', 'posture_adjust'],
  // Tier 2: visible, still conversational
  ['look_around', 'posture_adjust', 'sigh'],
  // Tier 3: sustained idle, but avoid theatrical poses
  ['lean_back', 'look_away', 'call_grounding_breath'],
];

const _PRODUCTION_CALL_ACTIONS = new Set([
  'call_acknowledge',
  'call_attentive_lean',
  'call_clarify_question',
  'call_key_point',
  'call_thoughtful_pause',
  'call_light_laugh',
  'call_grounding_breath',
  'call_wrap_up',
]);

const _AUTO_GESTURE_ALLOWLIST = new Set([
  'nod',
  'open_palms',
  'head_tilt',
  'look_away',
  'lean_forward',
  'lean_back',
  'shrug',
  'shake',
  'laugh',
  'deep_breath',
  'posture_adjust',
  'blink_slow',
  'weight_shift',
  'sigh',
  ..._PRODUCTION_CALL_ACTIONS,
]);

const _EXPLICIT_GESTURE_ALLOWLIST = new Set([
  ..._AUTO_GESTURE_ALLOWLIST,
  'bow',
  'point_up',
  'surprise',
  'wave',
  'look_around',
  'stretch_subtle',
]);

const _GESTURE_ALIASES = {
  call_acknowledge: 'call_acknowledge',
  call_greeting_wave: 'call_acknowledge',
  call_attentive_lean: 'call_attentive_lean',
  call_clarify_question: 'call_clarify_question',
  call_reassure_open: 'call_acknowledge',
  call_key_point: 'call_key_point',
  call_precise_detail: 'call_key_point',
  call_compare_options: 'call_acknowledge',
  call_side_reference: 'call_key_point',
  call_thoughtful_pause: 'call_thoughtful_pause',
  call_soft_shrug: 'call_acknowledge',
  call_light_laugh: 'call_light_laugh',
  call_gentle_no: 'call_acknowledge',
  call_grounding_breath: 'call_grounding_breath',
  call_wrap_up: 'call_wrap_up',
  think: 'call_thoughtful_pause',
  hand_to_chin: 'head_tilt',
  cross_arms: 'look_away',
};

function _planSemanticAction(sentence) {
  const text = String(sentence || '').trim().toLowerCase();
  if (!text) return null;

  if (/\b(to summarize|in summary|in short|overall|finally|lastly|bottom line|next steps?|wrap(?:ping)? up)\b/.test(text)) {
    return { action: 'call_wrap_up', score: 0.96 };
  }
  if (/\b(take a breath|breathe|no rush|slow down|take a moment|ground(?:ing)?|it'?s okay)\b/.test(text)) {
    return { action: 'call_grounding_breath', score: 0.92 };
  }
  if (/[?]\s*$/.test(text) || /\b(what|why|how|which|could|would|should|can you|do you want)\b/.test(text) && /[?]/.test(text)) {
    return { action: 'call_clarify_question', score: 0.9 };
  }
  if (/\b(key|important|main thing|crucial|remember|first|second|third|because|therefore|so the point)\b/.test(text)) {
    return { action: 'call_key_point', score: 0.86 };
  }
  if (/\b(haha|lol|funny|glad|nice|great)\b/.test(text) || /!\s*$/.test(text) && /\b(great|nice|perfect|love|happy)\b/.test(text)) {
    return { action: 'call_light_laugh', score: 0.82 };
  }
  if (/\b(maybe|perhaps|i think|let me think|consider|roughly|probably|hmm|hm)\b/.test(text) || /\.{3}/.test(text)) {
    return { action: 'call_thoughtful_pause', score: 0.8 };
  }

  return null;
}

function _normalizeGestureName(name, { explicit = false } = {}) {
  const clean = String(name || '').trim();
  if (!clean) return null;

  const normalized = _GESTURE_ALIASES[clean] || clean;
  const allowlist = explicit ? _EXPLICIT_GESTURE_ALLOWLIST : _AUTO_GESTURE_ALLOWLIST;
  return allowlist.has(normalized) ? normalized : null;
}
