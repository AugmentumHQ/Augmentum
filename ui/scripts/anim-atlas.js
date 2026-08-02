/**
 * anim-atlas.js — single tagged registry for VRMA, BVH, and (eventually)
 * static-pose animations. The whole movement system selects from this.
 *
 * Replaces the event-bucketed TRIGGERS map in avatar-pose-trigger.js
 * with a tagged-query model. Adding a new animation = one entry; adding
 * a new trigger context = one call to `select()` with new intent tags.
 *
 * ─── Schema (one shape for all types) ───────────────────────────────
 *
 *   id:        string         unique key. Prefer the curation alias;
 *                              fall back to filename without extension.
 *   type:      'vrma' | 'bvh' | 'pose'
 *   source:    string         URL for VRMA/BVH, preset name for pose
 *
 *   roles:     string[]       2-4 conversational functions this serves.
 *                              OPEN VOCABULARY — describe what it
 *                              expresses, not what fires it. New roles
 *                              emerge naturally from new animations.
 *
 *   emotion: { warmth, energy, openness, focus }   each in [0, 1]
 *                              — warmth   receptive ↔ guarded
 *                              — energy   calm ↔ animated
 *                              — openness closed ↔ exposed
 *                              — focus    diffuse ↔ direct
 *
 *   modes:     string[]       which surfaces this fits. One or more of:
 *                              'chat-call', 'chat-passive', 'narrative',
 *                              'passthrough', or '*' for all.
 *
 *   cost:      number [0..1]  0=micro (slerp pose), 1=theatrical (long
 *                              full-body VRMA). Drives static-vs-animated
 *                              routing AND back-to-back-theatrical
 *                              suppression via the conductor's energy
 *                              budget.
 *
 *   duration:  number         seconds. Useful for cost/budget reasoning;
 *                              0 for static poses (time-invariant).
 *   cooldown:  number         seconds before this id is eligible to fire
 *                              again. Stricter for unique gestures
 *                              (peace, dogeza), looser for micro-reactions.
 *
 *   ── Optional ─────────────────────────────────────────────────────
 *   framing?:        string   camera preset (e.g. 'fullBody')
 *   trimStartFrac?:  number   [0..1] skip opening N% of clip
 *   trimStart?:      number   seconds — alternative to trimStartFrac
 *   trimEnd?:        number   seconds — trim end-of-clip noise
 *   speed?:          number   playback rate multiplier (default 1.0)
 *   loop?:           boolean  default false; ambient/idle clips set true
 *   explicitOnly?:   boolean  if true, only fires on direct user request
 *                             (skip auto-selection). Use for showy
 *                             gestures the user must opt into.
 *   impliesAfter?:   string   role to consider next when this finishes
 *   notes?:          string   curation notes (origin, behavior, gotchas)
 *
 * ─── Selection ──────────────────────────────────────────────────────
 *
 *   const anim = select({ roles: ['celebrate'], emotion: aiAffect },
 *                       { mode: 'chat-call', energyBudget, recent,
 *                         lastPlayed, bias });
 *   if (anim) play(anim);
 *
 * Filters by mode + cost ≤ budget + not-recent + cooldown-elapsed,
 * scores by role-overlap + emotion-distance + per-id bias + jitter,
 * returns top-K weighted-random pick. Returns null when nothing fits;
 * caller treats that as "procedural carries, no animation this beat."
 */

// ─── Atlas — ~147 entries: 2026-05-04 VRMA curation + BVH library ───
// Durations are measured from the files (anim_audit.py 2026-06-10),
// not estimates. Quarantined-with-comment entries failed that audit.

export const ATLAS = [
  // ─── A: Re-tagged from the previously-wired VRMAs ──────────────
  {
    id: 'hello',
    type: 'vrma',
    source: '/ui/lib/animations/004_hello_1.vrma',
    roles: ['greet', 'wave'],
    emotion: { warmth: 0.85, energy: 0.65, openness: 0.7, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.6,
    duration: 15.8,
    cooldown: 7200,
    trimStartFrac: 0.4,
    framing: 'fullBody',
    notes: 'Alias: hello-turnedaround,spinthenwave. Trim 40% skips the '
         + 'spin-and-turn intro before the actual wave begins.',
  },
  {
    id: 'wave-jump',
    type: 'vrma',
    source: '/ui/lib/animations/VRMA_02.vrma',
    roles: ['greet', 'wave', 'celebrate'],
    emotion: { warmth: 0.85, energy: 0.95, openness: 0.85, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.85,
    duration: 7.3,
    cooldown: 86400,
    trimStart: 3.0,
    framing: 'fullBody',
    notes: 'Alias: traditionalwave. Big jump-up wave — once-a-day-max '
         + 'cooldown so it stays special.',
  },
  {
    id: 'peace-sign',
    type: 'vrma',
    source: '/ui/lib/animations/VRMA_03.vrma',
    roles: ['agree', 'agree-strong', 'farewell'],
    emotion: { warmth: 0.7, energy: 0.55, openness: 0.6, focus: 0.65 },
    modes: ['chat-call', 'narrative'],
    cost: 0.35,
    duration: 11.7,
    cooldown: 600,
    framing: 'fullBody',
    notes: 'Alias: peace-sign. Was over-firing across 4 trigger paths '
         + 'pre-curation; cooldown raised to 10min so it punctuates '
         + 'rather than bookmarks every positive beat.',
  },
  {
    id: 'shoot',
    type: 'vrma',
    source: '/ui/lib/animations/VRMA_04.vrma',
    roles: ['emphasize', 'point'],
    emotion: { warmth: 0.55, energy: 0.75, openness: 0.5, focus: 0.85 },
    modes: ['chat-call', 'narrative'],
    cost: 0.45,
    duration: 9.6,
    cooldown: 120,
    notes: 'Finger-guns / shoot-emphasis — punctuates a key point with '
         + 'directional energy. Pairs well with response-emphatic.',
  },
  {
    id: 'spin',
    type: 'vrma',
    source: '/ui/lib/animations/VRMA_05.vrma',
    roles: ['celebrate', 'react-positive', 'show-off'],
    emotion: { warmth: 0.8, energy: 0.95, openness: 0.85, focus: 0.5 },
    modes: ['chat-call', 'narrative'],
    cost: 0.85,
    duration: 9.3,
    cooldown: 240,
    framing: 'fullBody',
    notes: 'Spin/twirl. High cost; reserved for genuine excitement peaks.',
  },
  {
    id: 'model-pose',
    type: 'vrma',
    source: '/ui/lib/animations/VRMA_06.vrma',
    roles: ['ponder', 'pose-display', 'idle-shift'],
    emotion: { warmth: 0.6, energy: 0.45, openness: 0.55, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.5,
    duration: 7.5,
    cooldown: 180,
    framing: 'fullBody',
    notes: 'Contemplative model pose. Used both for response-thoughtful '
         + '(when AI is pondering) and as idle filler.',
  },
  {
    id: 'lean-in',
    type: 'vrma',
    source: '/ui/lib/animations/001_motion_pose.vrma',
    roles: ['listen', 'idle-shift', 'attentive'],
    emotion: { warmth: 0.7, energy: 0.4, openness: 0.65, focus: 0.8 },
    modes: ['chat-call', 'narrative'],
    cost: 0.3,
    duration: 20.0,
    cooldown: 90,
    notes: 'Alias: leaninhandonknees-leftandright-spin. Subtle attentive '
         + 'shift — was over-firing on every speech start at 10%; '
         + 'cooldown is the gate, not the probability.',
  },
  {
    id: 'smartphone',
    type: 'vrma',
    source: '/ui/lib/animations/005_smartphone.vrma',
    roles: ['idle-fill', 'idle-distracted'],
    emotion: { warmth: 0.4, energy: 0.3, openness: 0.4, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4,
    duration: 10.4,
    cooldown: 300,
    loop: true,
    impliesAfter: 'idle-distracted',
    notes: 'Loops while user stays silent. impliesAfter chains it to '
         + 'continue-phone-style follow-ups for sustained idle.',
  },
  {
    id: 'kebab-dance',
    type: 'vrma',
    source: '/ui/lib/animations/dance_kebab.vrma',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.85, energy: 0.95, openness: 0.85, focus: 0.5 },
    modes: ['chat-call', 'narrative'],
    cost: 0.75,
    duration: 20.1,
    cooldown: 600,
    loop: true,
    framing: 'fullBody',
    notes: 'Original VRMA dance — cooldown was 1800s from pre-loop authoring; '
         + 'aligned to 600s so it cycles with the rest of the pool. Cost '
         + 'dropped from 0.9 → 0.75 (energy budget rebalance).',
  },
  {
    id: 'dance-28',
    type: 'vrma',
    source: '/ui/lib/animations/dance_28.vrma',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.8, energy: 0.9, openness: 0.8, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.7,
    duration: 57.6,
    cooldown: 600,
    loop: true,
    speed: 0.75,
    framing: 'fullBody',
    notes: 'Alias: dance-28-75speed. Plays at 0.75× — full speed reads '
         + 'frenetic; slowed down it feels intentional. Cooldown aligned to '
         + 'pool default (was 1800s pre-loop); cost dropped 0.9 → 0.7.',
  },
  {
    id: 'dance-25',
    type: 'vrma',
    source: '/ui/lib/animations/dance_25.vrma',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.75, energy: 0.85, openness: 0.75, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.65,
    duration: 22.9,
    cooldown: 600,
    loop: true,
    trimEnd: 0.8,
    framing: 'fullBody',
    notes: 'Trim the loop seam — last 0.8s has glitchy frames. Cooldown '
         + 'aligned to pool default (was 1800s pre-loop); cost dropped '
         + '0.9 → 0.65.',
  },
  {
    id: 'dogeza',
    type: 'vrma',
    source: '/ui/lib/animations/002_dogeza.vrma',
    roles: ['bow-deep', 'apology'],
    emotion: { warmth: 0.6, energy: 0.4, openness: 0.3, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.7,
    duration: 7.3,
    cooldown: 600,
    framing: 'fullBody',
    explicitOnly: true,
    notes: 'Alias: traditionalbow-dogeza. Explicit-only — never auto. '
         + 'User asks for "bow" / "dogeza" / "deep bow" → fires.',
  },

  // ─── B: vrm-viewer pack (10 of 11; Clapping rejected for finger clip) ─
  {
    id: 'surprised-vv',
    type: 'vrma',
    source: '/ui/lib/animations/Surprised.vrma',
    roles: ['react-surprise', 'react-positive'],
    emotion: { warmth: 0.6, energy: 0.7, openness: 0.75, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.3,
    duration: 3.9,
    cooldown: 90,
    notes: 'Reactive "oh!" rather than celebratory. Fires on unexpected '
         + 'user statements — distinct role from cheer-tier.',
  },
  {
    id: 'blush-vv',
    type: 'vrma',
    source: '/ui/lib/animations/Blush.vrma',
    roles: ['shy', 'react-compliment', 'soften'],
    emotion: { warmth: 0.85, energy: 0.4, openness: 0.5, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.25,
    duration: 3.9,
    cooldown: 180,
    notes: 'Receiving-compliment / shy reaction. Cooldown long because '
         + 'overuse cheapens the gesture.',
  },
  {
    id: 'lookaround-vv',
    type: 'vrma',
    source: '/ui/lib/animations/LookAround.vrma',
    roles: ['idle-shift', 'idle-attentive'],
    emotion: { warmth: 0.55, energy: 0.45, openness: 0.65, focus: 0.5 },
    modes: ['chat-call', 'chat-passive', 'narrative'],
    cost: 0.2,
    duration: 3.9,
    cooldown: 60,
    notes: 'Subtle look-around between turns. Lower cost than model-idle '
         + '— small enough to fire freely.',
  },
  {
    id: 'goodbye-vv',
    type: 'vrma',
    source: '/ui/lib/animations/Goodbye.vrma',
    roles: ['farewell', 'wave'],
    emotion: { warmth: 0.85, energy: 0.6, openness: 0.7, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.45,
    duration: 3.9,
    cooldown: 1800,
    notes: 'Alternative farewell to peace-goodbye / wave-goodbye. More '
         + 'understated; good for soft call closings.',
  },
  {
    id: 'thinking-vv',
    type: 'vrma',
    source: '/ui/lib/animations/Thinking.vrma',
    roles: ['think', 'ponder'],
    emotion: { warmth: 0.5, energy: 0.3, openness: 0.4, focus: 0.85 },
    modes: ['chat-call', 'chat-passive', 'narrative'],
    cost: 0.3,
    duration: 3.9,
    cooldown: 90,
    notes: 'Alternative to model-pose for response-thoughtful. Lower cost '
         + '+ shorter cooldown = more freely fires when AI deliberates.',
  },
  {
    id: 'relax-vv',
    type: 'vrma',
    source: '/ui/lib/animations/Relax.vrma',
    roles: ['idle-relaxed', 'idle-fill'],
    emotion: { warmth: 0.7, energy: 0.3, openness: 0.6, focus: 0.45 },
    modes: ['chat-call', 'chat-passive', 'narrative'],
    cost: 0.25,
    duration: 3.9,
    cooldown: 120,
    notes: 'Mid-call atmospheric idle. Calmer than lookAround.',
  },
  {
    id: 'sad-vv',
    type: 'vrma',
    source: '/ui/lib/animations/Sad.vrma',
    roles: ['sympathy', 'mirror-sad', 'soften'],
    emotion: { warmth: 0.7, energy: 0.2, openness: 0.4, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4,
    duration: 3.9,
    cooldown: 240,
    notes: 'Acknowledges user shared something heavy. NEW role coverage — '
         + 'production has no current trigger for "user-shared-bad-news."',
  },
  {
    id: 'sleepy-vv',
    type: 'vrma',
    source: '/ui/lib/animations/Sleepy.vrma',
    roles: ['idle-low-energy', 'late-call', 'idle-fill'],
    emotion: { warmth: 0.65, energy: 0.15, openness: 0.5, focus: 0.3 },
    modes: ['chat-call', 'narrative'],
    cost: 0.3,
    duration: 3.9,
    cooldown: 600,
    notes: 'Late-call / low-energy idle. Cooldown long; fires only when '
         + 'session has been going a while.',
  },
  {
    id: 'angry-vv',
    type: 'vrma',
    source: '/ui/lib/animations/Angry.vrma',
    roles: ['frustration', 'disagree-strong'],
    emotion: { warmth: 0.2, energy: 0.85, openness: 0.4, focus: 0.85 },
    modes: ['narrative'],
    cost: 0.6,
    duration: 3.9,
    cooldown: 600,
    notes: 'Narrative-only. Strong frustration / disagreement is rarely '
         + 'right tone for chat-call. Reserved for storytelling beats.',
  },
  {
    id: 'jump-vv',
    type: 'vrma',
    source: '/ui/lib/animations/Jump.vrma',
    roles: ['celebrate', 'react-positive', 'excitement-peak'],
    emotion: { warmth: 0.85, energy: 0.95, openness: 0.85, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.75,
    duration: 3.9,
    cooldown: 240,
    framing: 'fullBody',
    notes: 'Excitement peak — alternative to wave-jump for non-greeting '
         + 'celebration moments.',
  },

  // ─── C: Cheer-tier BVH (the gekirei replacements — ALL kept) ───
  {
    id: 'excitement3',
    type: 'bvh',
    source: '/bvh-library/animation/excitement3.bvh',
    roles: ['celebrate', 'agree-strong', 'react-positive'],
    emotion: { warmth: 0.85, energy: 0.85, openness: 0.7, focus: 0.65 },
    modes: ['chat-call', 'narrative'],
    cost: 0.5,
    duration: 2.93,
    cooldown: 90,
    notes: 'Punchy "yes!" without theatrics. Primary gekirei replacement.',
  },
  {
    id: 'joy3',
    type: 'bvh',
    source: '/bvh-library/animation/joy3.bvh',
    roles: ['celebrate', 'joy', 'react-positive'],
    emotion: { warmth: 0.9, energy: 0.8, openness: 0.75, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.55,
    duration: 3.83,
    cooldown: 90,
    notes: 'Warmer/longer than excitement3. Pairs with shared-joy '
         + 'beats — user-shared-good-news, response-positive with high warmth.',
  },
  {
    id: 'pride',
    type: 'bvh',
    source: '/bvh-library/animation/pride.bvh',
    roles: ['pride', 'agree-strong', 'celebrate'],
    emotion: { warmth: 0.7, energy: 0.65, openness: 0.55, focus: 0.75 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4,
    duration: 2.37,
    cooldown: 120,
    notes: '"We did it" register — distinct from raw celebration. '
         + 'Good for accomplishment beats.',
  },
  {
    id: 'pride2',
    type: 'bvh',
    source: '/bvh-library/animation/pride2.bvh',
    roles: ['pride', 'agree-strong', 'celebrate'],
    emotion: { warmth: 0.7, energy: 0.7, openness: 0.6, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.45,
    duration: 2.93,
    cooldown: 120,
    notes: 'Variant of pride. Same role set so they substitute via the '
         + 'recency penalty when fired close together.',
  },
  {
    id: 'gratitude',
    type: 'bvh',
    source: '/bvh-library/animation/gratitude.bvh',
    roles: ['gratitude', 'agree', 'soften'],
    emotion: { warmth: 0.85, energy: 0.5, openness: 0.65, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.35,
    duration: 3.03,
    cooldown: 120,
    notes: 'Receiving-warmth — distinct from celebrate. "Thank you" / '
         + '"I appreciate that" beats.',
  },
  {
    id: 'optimism',
    type: 'bvh',
    source: '/bvh-library/animation/optimism.bvh',
    roles: ['celebrate', 'agree', 'forward-looking'],
    emotion: { warmth: 0.75, energy: 0.65, openness: 0.7, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4,
    duration: 2.87,
    cooldown: 120,
    notes: 'Alias: salute/optimism. Forward-looking / "this is going to '
         + 'work out" energy. Salute-like gesture per curation.',
  },
  {
    id: 'relief',
    type: 'bvh',
    source: '/bvh-library/animation/relief.bvh',
    roles: ['relief', 'react-positive', 'soften'],
    emotion: { warmth: 0.75, energy: 0.45, openness: 0.65, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.3,
    duration: 3.03,
    cooldown: 90,
    notes: '"Phew" — for resolved tension / problem-solved beats.',
  },
  {
    id: 'approval',
    type: 'bvh',
    source: '/bvh-library/animation/approval.bvh',
    roles: ['agree', 'affirm', 'emphasize'],
    emotion: { warmth: 0.7, energy: 0.6, openness: 0.6, focus: 0.75 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4,
    duration: 4.73,
    cooldown: 45,
    notes: 'Direct affirmation — middle ground between subtle nod and '
         + 'theatrical cheer. The "I agree" workhorse.',
  },

  // ─── D: Subtle micro-reactions (3 of 7 kept) ───────────────────
  {
    id: 'amusement2',
    type: 'bvh',
    source: '/bvh-library/animation/amusement2.bvh',
    roles: ['amusement', 'react-positive', 'micro-laugh'],
    emotion: { warmth: 0.7, energy: 0.55, openness: 0.6, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.12,
    duration: 1.17,
    cooldown: 20,
    notes: 'Brief smile/haha. Micro-tier — fires often without disrupting flow.',
  },
  {
    id: 'amusement3',
    type: 'bvh',
    source: '/bvh-library/animation/amusement3.bvh',
    roles: ['amusement', 'react-positive', 'micro-laugh'],
    emotion: { warmth: 0.7, energy: 0.6, openness: 0.65, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.15,
    duration: 1.70,
    cooldown: 25,
    notes: 'Slightly longer amusement variant. Same role set as amusement2 '
         + 'so they alternate naturally via recency.',
  },
  {
    id: 'realization',
    type: 'bvh',
    source: '/bvh-library/animation/realization.bvh',
    roles: ['realization', 'react-positive', 'oh-i-see'],
    emotion: { warmth: 0.65, energy: 0.55, openness: 0.65, focus: 0.85 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4,
    duration: 5.87,
    cooldown: 120,
    notes: '"Oh I see" — slower beat. Connecting-the-dots moment. '
         + 'Longer cooldown because the gesture reads as a moment.',
  },

  // ─── E: Conversational reactions (NEW user-state coverage) ─────
  {
    id: 'curiosity',
    type: 'bvh',
    source: '/bvh-library/animation/curiosity.bvh',
    roles: ['curiosity', 'question', 'mirror-curious'],
    emotion: { warmth: 0.6, energy: 0.5, openness: 0.7, focus: 0.75 },
    modes: ['chat-call', 'narrative'],
    cost: 0.3,
    duration: 4.27,
    cooldown: 60,
    notes: 'Mirrors user curiosity. Production has zero coverage today; '
         + 'fires when user signals genuine interest in a topic.',
  },
  {
    id: 'confusion',
    type: 'bvh',
    source: '/bvh-library/animation/confusion.bvh',
    roles: ['confusion', 'mirror-confused', 'question'],
    emotion: { warmth: 0.5, energy: 0.55, openness: 0.55, focus: 0.65 },
    modes: ['chat-call', 'narrative'],
    cost: 0.6,
    duration: 13.37,
    cooldown: 180,
    notes: '"Wait, what?" — long form (13s). Heavy cost + 3min cooldown '
         + 'because holding the screen this long needs to read as a moment.',
  },
  {
    id: 'nervousness',
    type: 'bvh',
    source: '/bvh-library/animation/nervousness.bvh',
    roles: ['nervousness', 'mirror-nervous', 'hesitant'],
    emotion: { warmth: 0.55, energy: 0.5, openness: 0.4, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4,
    duration: 6.30,
    cooldown: 90,
    notes: 'Hesitant beat — softens delivery of uncertain answers / '
         + 'mirrors user nervousness around vulnerable topics.',
  },
  {
    id: 'embarrassment',
    type: 'bvh',
    source: '/bvh-library/animation/embarrassment.bvh',
    roles: ['embarrassment', 'mirror-embarrassed', 'soften'],
    emotion: { warmth: 0.7, energy: 0.55, openness: 0.4, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.55,
    duration: 11.03,
    cooldown: 180,
    notes: '"Oh no, I messed up" cringe — long form (11s). Pairs with AI '
         + 'self-correction or mirrors user embarrassment.',
  },
  {
    id: 'disappointment',
    type: 'bvh',
    source: '/bvh-library/animation/disappointment.bvh',
    roles: ['disappointment', 'mirror-disappointed', 'soften'],
    emotion: { warmth: 0.55, energy: 0.3, openness: 0.4, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.3,
    duration: 4.23,
    cooldown: 120,
    notes: '"Oh" deflate. Mirrors user disappointment in something not '
         + 'going their way.',
  },
  {
    id: 'caring',
    type: 'bvh',
    source: '/bvh-library/animation/caring.bvh',
    roles: ['comfort', 'sympathy', 'mirror-sad', 'soften'],
    emotion: { warmth: 0.95, energy: 0.3, openness: 0.7, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.55,
    duration: 10.03,
    cooldown: 180,
    notes: 'Warm sympathy for "user shared something hard" — long form '
         + '(10s). Pairs with response-thoughtful + low-energy AI affect.',
  },

  // ─── F: BVH dance library (sillytavern-pack/animation/dance_*) ───
  // BVH dances joining the VRMA dances. Same roles ['dance',
  // 'idle-fill', 'show-off'] so the selector treats them as peers.
  // Durations across the whole atlas were ground-truthed from the
  // actual files on 2026-06-10 (.augmentum-dev-cache/anim_audit.py).
  // bvh-dance-1 quarantined 2026-06-10 — loop-tagged but end-to-start
  // pose gap is 11.4 deg/joint (visible pop on every loop). Re-add
  // once trimmed to a clean cycle.
  {
    id: 'bvh-dance-2',
    type: 'bvh',
    source: '/bvh-library/animation/dance_2.bvh',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.72, energy: 0.88, openness: 0.75, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.65, duration: 20.4, cooldown: 600, loop: true,
    notes: 'BVH dance — variant of dance_1 in same family.',
  },
  {
    id: 'bvh-dance-backup',
    type: 'bvh',
    source: '/bvh-library/animation/dance_backup.bvh',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.7, energy: 0.8, openness: 0.7, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.55, duration: 25.4, cooldown: 600, loop: true,
    notes: 'BVH dance — backup-dancer-style sway. Lower intensity peak.',
  },
  {
    id: 'bvh-dance-dab',
    type: 'bvh',
    source: '/bvh-library/animation/dance_dab.bvh',
    roles: ['dance', 'show-off', 'celebrate'],
    emotion: { warmth: 0.6, energy: 0.95, openness: 0.75, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.7, duration: 7.7, cooldown: 1800,
    notes: 'Dab — meme gesture, one-shot, longer cooldown so it punctuates '
         + 'rather than repeats. Pairs with strong celebrate intent.',
  },
  {
    id: 'bvh-dance-gangnam',
    type: 'bvh',
    source: '/bvh-library/animation/dance_gangnam_style.bvh',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.7, energy: 0.95, openness: 0.8, focus: 0.5 },
    modes: ['chat-call', 'narrative'],
    cost: 0.9, duration: 12.4, cooldown: 1800, loop: true,
    notes: 'Gangnam-style horse-riding dance. High-energy showpiece.',
  },
  {
    id: 'bvh-dance-headdrop',
    type: 'bvh',
    source: '/bvh-library/animation/dance_headdrop.bvh',
    roles: ['dance', 'show-off'],
    emotion: { warmth: 0.55, energy: 0.85, openness: 0.6, focus: 0.85 },
    modes: ['chat-call', 'narrative'],
    cost: 0.65, duration: 15.7, cooldown: 900,
    notes: 'Headbang-style head drop. Performance-coded.',
  },
  {
    id: 'bvh-dance-maraschino',
    type: 'bvh',
    source: '/bvh-library/animation/dance_marachinostep.bvh',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.78, energy: 0.78, openness: 0.78, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.55, duration: 3.3, cooldown: 600, loop: true,
    notes: 'Maraschino step — vintage Broadway routine. Warm + playful.',
  },
  {
    id: 'bvh-dance-northern-soul-spin',
    type: 'bvh',
    source: '/bvh-library/animation/dance_northern_soul_spin.bvh',
    roles: ['dance', 'show-off'],
    emotion: { warmth: 0.7, energy: 0.92, openness: 0.85, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.85, duration: 8.9, cooldown: 1200,
    notes: 'Northern soul spin — full body rotation, end position is '
         + 'non-neutral; relies on motion engine to settle.',
  },
  {
    id: 'bvh-dance-ontop',
    type: 'bvh',
    source: '/bvh-library/animation/dance_ontop.bvh',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.7, energy: 0.85, openness: 0.75, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.7, duration: 24.3, cooldown: 600, loop: true,
    notes: 'On-top dance — elevated/showy posture.',
  },
  {
    id: 'bvh-dance-pushback',
    type: 'bvh',
    source: '/bvh-library/animation/dance_pushback.bvh',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.6, energy: 0.82, openness: 0.7, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.6, duration: 17.8, cooldown: 600, loop: true,
    notes: 'Pushback dance — assertive sway.',
  },
  {
    id: 'bvh-dance-rumba',
    type: 'bvh',
    source: '/bvh-library/animation/dance_rumba.bvh',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.88, energy: 0.7, openness: 0.85, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.55, duration: 2.3, cooldown: 600, loop: true,
    notes: 'Rumba — sultry slower dance. Pairs with warm music in widget '
         + 'host mode.',
  },

  // ─── G: BVH idle library (sit/laying/kneel/neutral variants) ─────
  // 11 idle BVHs to break the "stuck in one pose" feel during quiet
  // stretches. Low cost so they layer alongside breathing without
  // dominating the energy budget. ``loop: true`` — atlas selector
  // can sit on one for minutes before swapping.
  {
    id: 'bvh-idle-kneel',
    type: 'bvh',
    source: '/bvh-library/animation/kneel_idle.bvh',
    roles: ['idle-low-energy', 'idle-fill', 'settled'],
    emotion: { warmth: 0.55, energy: 0.25, openness: 0.45, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.15, duration: 3.3, cooldown: 60, loop: true,
    notes: 'Kneeling idle — calm contemplative posture.',
  },
  // bvh-idle-kneel-2 quarantined 2026-06-10 — loop seam gap 11.0
  // deg/joint pops on repeat. kneel_idle (v1) loops clean and stays.
  {
    id: 'bvh-idle-laying',
    type: 'bvh',
    source: '/bvh-library/animation/laying_idle.bvh',
    roles: ['idle-low-energy', 'idle-relaxed', 'late-call'],
    emotion: { warmth: 0.6, energy: 0.18, openness: 0.55, focus: 0.4 },
    modes: ['chat-call', 'narrative'],
    cost: 0.15, duration: 10.3, cooldown: 120, loop: true,
    notes: 'Laying-down idle. Casual late-call register.',
  },
  {
    id: 'bvh-idle-laying-2',
    type: 'bvh',
    source: '/bvh-library/animation/laying_idle2.bvh',
    roles: ['idle-low-energy', 'idle-relaxed', 'late-call'],
    emotion: { warmth: 0.6, energy: 0.18, openness: 0.55, focus: 0.4 },
    modes: ['chat-call', 'narrative'],
    cost: 0.15, duration: 0.8, cooldown: 120, loop: true,
    notes: 'Variant of laying_idle.',
  },
  {
    id: 'bvh-idle-laying-3',
    type: 'bvh',
    source: '/bvh-library/animation/laying_idle3.bvh',
    roles: ['idle-low-energy', 'idle-relaxed', 'late-call'],
    emotion: { warmth: 0.6, energy: 0.2, openness: 0.55, focus: 0.4 },
    modes: ['chat-call', 'narrative'],
    cost: 0.15, duration: 19.8, cooldown: 120, loop: true,
    notes: 'Variant of laying_idle.',
  },
  {
    id: 'bvh-idle-neutral',
    type: 'bvh',
    source: '/bvh-library/animation/neutral_idle.bvh',
    roles: ['idle-fill', 'idle-relaxed'],
    emotion: { warmth: 0.55, energy: 0.4, openness: 0.55, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.12, duration: 8.8, cooldown: 45, loop: true,
    notes: 'Neutral standing/sitting idle — baseline.',
  },
  {
    id: 'bvh-idle-neutral-2',
    type: 'bvh',
    source: '/bvh-library/animation/neutral_idle2.bvh',
    roles: ['idle-fill', 'idle-relaxed'],
    emotion: { warmth: 0.55, energy: 0.42, openness: 0.55, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.12, duration: 19.9, cooldown: 45, loop: true,
    notes: 'Variant of neutral_idle.',
  },
  // bvh-idle-sit / -2 / -3 quarantined 2026-06-10 — all three sitting
  // idles loop with end-to-start gaps of 8.4-12.7 deg/joint (visible
  // pop each cycle). sit_idle4 measured clean and stays.
  {
    id: 'bvh-idle-sit-4',
    type: 'bvh',
    source: '/bvh-library/animation/sit_idle4.bvh',
    roles: ['idle-fill', 'idle-relaxed', 'settled'],
    emotion: { warmth: 0.55, energy: 0.35, openness: 0.5, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.12, duration: 6.4, cooldown: 45, loop: true,
    notes: 'Sitting idle variant.',
  },

  // ─── H: BVH action library (greetings, gestures, locomotion) ─────
  // Most are explicit-request driven — user says "wave" / "jump".
  // Locomotion variants (jog/run/walk) translate the avatar through
  // world space; flagged explicitOnly so they never auto-fire in the
  // chromeless widget where she'd walk off-screen.
  {
    id: 'bvh-action-greeting',
    type: 'bvh',
    source: '/bvh-library/animation/action_greeting.bvh',
    roles: ['greet', 'wave'],
    emotion: { warmth: 0.85, energy: 0.65, openness: 0.7, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.5, duration: 5.5, cooldown: 600,
    notes: 'BVH greeting — alternate take to the VRMA hello family.',
  },
  {
    id: 'bvh-action-greeting-1',
    type: 'bvh',
    source: '/bvh-library/animation/action_greeting1.bvh',
    roles: ['greet', 'wave'],
    emotion: { warmth: 0.85, energy: 0.7, openness: 0.7, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.5, duration: 4.8, cooldown: 600,
    notes: 'Greeting variant for natural alternation.',
  },
  {
    id: 'bvh-action-jump',
    type: 'bvh',
    source: '/bvh-library/animation/action_jump.bvh',
    roles: ['celebrate', 'excitement-peak', 'react-positive'],
    emotion: { warmth: 0.7, energy: 0.95, openness: 0.85, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.7, duration: 3.3, cooldown: 1800,
    notes: 'Jump — high-energy celebrate moment. Long cooldown.',
  },
  {
    id: 'bvh-action-pat',
    type: 'bvh',
    source: '/bvh-library/animation/action_pat.bvh',
    roles: ['comfort', 'sympathy', 'soften'],
    emotion: { warmth: 0.9, energy: 0.4, openness: 0.7, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4, duration: 4.4, cooldown: 600,
    notes: 'Patting gesture — comforting beat.',
  },
  {
    id: 'bvh-action-pickingup',
    type: 'bvh',
    source: '/bvh-library/animation/action_pickingup.bvh',
    roles: ['picking-up', 'reach'],
    emotion: { warmth: 0.55, energy: 0.55, openness: 0.5, focus: 0.85 },
    modes: ['chat-call', 'narrative'],
    cost: 0.45, duration: 4.7, cooldown: 1800,
    explicitOnly: true,
    notes: 'Picking-up — needs a prop in scene for the gesture to read. '
         + 'Explicit-only.',
  },
  // bvh-action-attention-seeking was cut 2026-06-10 — the retargeted
  // clip reads badly on the VRM rig. The source .bvh stays in
  // /bvh-library if it's ever worth re-adding as a trimmed upload.
  {
    id: 'bvh-action-standup',
    type: 'bvh',
    source: '/bvh-library/animation/action_standup.bvh',
    roles: ['posture-shift'],
    emotion: { warmth: 0.55, energy: 0.55, openness: 0.55, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4, duration: 3.5, cooldown: 1800,
    explicitOnly: true,
    notes: 'Standing-up transition. Mostly useful when followed by an '
         + 'action; explicit-only for now.',
  },
  {
    id: 'bvh-action-laydown',
    type: 'bvh',
    source: '/bvh-library/animation/action_laydown.bvh',
    roles: ['posture-shift', 'wind-down'],
    emotion: { warmth: 0.5, energy: 0.2, openness: 0.4, focus: 0.4 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4, duration: 5.8, cooldown: 1800,
    explicitOnly: true,
    notes: 'Lay-down transition. Pairs with laying_idle*.',
  },
  {
    id: 'bvh-action-crouch',
    type: 'bvh',
    source: '/bvh-library/animation/action_crouch.bvh',
    roles: ['posture-shift'],
    emotion: { warmth: 0.5, energy: 0.45, openness: 0.4, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.35, duration: 4.1, cooldown: 1800,
    explicitOnly: true,
    notes: 'Crouch — needs prop/object alignment to read cleanly.',
  },
  {
    id: 'bvh-action-gaming',
    type: 'bvh',
    source: '/bvh-library/animation/action_gaming.bvh',
    roles: ['hobby', 'playful'],
    emotion: { warmth: 0.7, energy: 0.55, openness: 0.6, focus: 0.85 },
    modes: ['chat-call', 'narrative'],
    cost: 0.45, duration: 6.7, cooldown: 1800,
    explicitOnly: true,
    notes: 'Gaming-controller pose — needs prop to read. Explicit-only.',
  },
  // Locomotion — all explicit-only, all translate world position
  {
    id: 'bvh-action-jog',
    type: 'bvh',
    source: '/bvh-library/animation/action_jog.bvh',
    roles: ['locomotion'],
    emotion: { warmth: 0.55, energy: 0.75, openness: 0.55, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.5, duration: 2.8, cooldown: 1800,
    explicitOnly: true,
    notes: 'Locomotion — translates avatar through world. Voice-call '
         + 'experience mode only; off in widget (would walk off-screen).',
  },
  {
    id: 'bvh-action-run',
    type: 'bvh',
    source: '/bvh-library/animation/action_run.bvh',
    roles: ['locomotion'],
    emotion: { warmth: 0.55, energy: 0.9, openness: 0.55, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.55, duration: 0.7, cooldown: 1800,
    explicitOnly: true,
    notes: 'Locomotion — translates avatar. See bvh-action-jog.',
  },
  {
    id: 'bvh-action-walk',
    type: 'bvh',
    source: '/bvh-library/animation/action_walk.bvh',
    roles: ['locomotion'],
    emotion: { warmth: 0.55, energy: 0.5, openness: 0.55, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.45, duration: 1.4, cooldown: 1800,
    explicitOnly: true,
    notes: 'Locomotion — translates avatar. See bvh-action-jog.',
  },
  {
    id: 'bvh-action-crawling',
    type: 'bvh',
    source: '/bvh-library/animation/action_crawling.bvh',
    roles: ['locomotion'],
    emotion: { warmth: 0.5, energy: 0.5, openness: 0.3, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.5, duration: 1.9, cooldown: 1800,
    explicitOnly: true,
    notes: 'Locomotion — translates avatar; ground-level. Explicit-only.',
  },

  // ─── I: Missing emotion primaries (filling gaps from sillytavern) ─
  // Atlas has 17 BVH emotions; this adds 11 base variants where the
  // primary emotion isn't yet represented. Numbered alternates (joy2,
  // sadness2, etc.) deferred — single canonical entry per emotion is
  // enough until non-repetition becomes a felt issue.
  {
    id: 'bvh-emotion-anger',
    type: 'bvh',
    source: '/bvh-library/animation/anger.bvh',
    roles: ['react-negative', 'angry', 'mirror-anger'],
    emotion: { warmth: 0.15, energy: 0.85, openness: 0.4, focus: 0.85 },
    modes: ['chat-call', 'narrative'],
    cost: 0.55, duration: 19.2, cooldown: 240,
    notes: 'Anger expression — clipped energy. Pairs with mirroring '
         + 'user frustration (sparingly; over-firing reads as aggressive).',
  },
  {
    id: 'bvh-emotion-fear',
    type: 'bvh',
    source: '/bvh-library/animation/fear.bvh',
    roles: ['react-negative', 'startled', 'apprehensive'],
    emotion: { warmth: 0.3, energy: 0.7, openness: 0.25, focus: 0.85 },
    modes: ['chat-call', 'narrative'],
    cost: 0.5, duration: 20.0, cooldown: 300,
    notes: 'Apprehension / fear — recoil + protective body language.',
  },
  {
    id: 'bvh-emotion-surprise',
    type: 'bvh',
    source: '/bvh-library/animation/surprise.bvh',
    roles: ['react-surprise', 'startled', 'react-positive'],
    emotion: { warmth: 0.6, energy: 0.85, openness: 0.85, focus: 0.9 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4, duration: 4.0, cooldown: 120,
    notes: 'Surprise — sharp react. Works for both delighted and '
         + 'concerned surprise depending on follow-up.',
  },
  {
    id: 'bvh-emotion-love',
    type: 'bvh',
    source: '/bvh-library/animation/love.bvh',
    roles: ['affection', 'warmth', 'react-positive'],
    emotion: { warmth: 0.95, energy: 0.5, openness: 0.85, focus: 0.75 },
    modes: ['chat-call', 'narrative'],
    cost: 0.5, duration: 4.6, cooldown: 600,
    notes: 'Warmth / affection — heart-felt beat. Cooldown long because '
         + 'overuse cheapens the gesture.',
  },
  {
    id: 'bvh-emotion-joy',
    type: 'bvh',
    source: '/bvh-library/animation/joy.bvh',
    roles: ['react-positive', 'celebrate'],
    emotion: { warmth: 0.85, energy: 0.8, openness: 0.85, focus: 0.65 },
    modes: ['chat-call', 'narrative'],
    cost: 0.5, duration: 8.5, cooldown: 180,
    notes: 'Joy — base variant. Atlas already has joy3 (louder). '
         + 'This is the medium-energy take.',
  },
  {
    id: 'bvh-emotion-sadness',
    type: 'bvh',
    source: '/bvh-library/animation/sadness.bvh',
    roles: ['react-negative', 'sad', 'mirror-sad'],
    emotion: { warmth: 0.4, energy: 0.25, openness: 0.4, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.5, duration: 3.1, cooldown: 240,
    notes: 'Sadness — slump + downward gaze.',
  },
  {
    id: 'bvh-emotion-grief',
    type: 'bvh',
    source: '/bvh-library/animation/grief.bvh',
    roles: ['react-negative', 'mirror-sad', 'sympathy'],
    emotion: { warmth: 0.45, energy: 0.18, openness: 0.3, focus: 0.5 },
    modes: ['chat-call', 'narrative'],
    cost: 0.55, duration: 6.8, cooldown: 900,
    notes: 'Grief — heavier than sadness. Reserved for "they shared '
         + 'something deeply hard" beats; long cooldown.',
  },
  {
    id: 'bvh-emotion-desire',
    type: 'bvh',
    source: '/bvh-library/animation/desire.bvh',
    roles: ['longing', 'curious'],
    emotion: { warmth: 0.7, energy: 0.55, openness: 0.7, focus: 0.85 },
    modes: ['chat-call', 'narrative'],
    cost: 0.45, duration: 3.8, cooldown: 600,
    notes: 'Longing / desire — leaning-toward beat.',
  },
  {
    id: 'bvh-emotion-disgust',
    type: 'bvh',
    source: '/bvh-library/animation/disgust.bvh',
    roles: ['react-negative', 'recoil'],
    emotion: { warmth: 0.2, energy: 0.6, openness: 0.3, focus: 0.8 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4, duration: 4.4, cooldown: 600,
    notes: 'Disgust — recoil with negative valence. Rare beat.',
  },
  {
    id: 'bvh-emotion-remorse',
    type: 'bvh',
    source: '/bvh-library/animation/remorse.bvh',
    roles: ['react-negative', 'apologetic', 'sad'],
    emotion: { warmth: 0.55, energy: 0.3, openness: 0.4, focus: 0.65 },
    modes: ['chat-call', 'narrative'],
    cost: 0.45, duration: 2.4, cooldown: 600,
    notes: 'Remorse / apologetic — when AI misses or has to walk something '
         + 'back. Pairs with response-thoughtful.',
  },
  {
    id: 'bvh-emotion-neutral',
    type: 'bvh',
    source: '/bvh-library/animation/neutral.bvh',
    roles: ['idle-fill', 'react-neutral'],
    emotion: { warmth: 0.55, energy: 0.4, openness: 0.55, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.15, duration: 8.4, cooldown: 60,
    notes: 'Neutral baseline emotion clip. Low-cost filler.',
  },

  // ─── J: Hit reactions — body-atlas region triggers (future) ──────
  // 9 hit-reactions from sillytavern-pack. All explicit-only for now;
  // the contact-reactor.js path that will auto-fire them on body-atlas
  // region collision isn't wired yet (task #128). Tagging now so the
  // wiring becomes a one-line consumer rather than discovery work.
  {
    id: 'bvh-hit-butt',
    type: 'bvh',
    source: '/bvh-library/animation/hitarea_butt.bvh',
    roles: ['react-contact', 'startled-low'],
    emotion: { warmth: 0.5, energy: 0.6, openness: 0.4, focus: 0.85 },
    modes: ['chat-call', 'narrative'],
    cost: 0.35, duration: 19.8, cooldown: 60,
    explicitOnly: true,
    notes: 'Contact region: butt. Wired by contact-reactor (future). '
         + 'See task #128.',
  },
  {
    id: 'bvh-hit-chest',
    type: 'bvh',
    source: '/bvh-library/animation/hitarea_chest.bvh',
    roles: ['react-contact', 'startled-mid'],
    emotion: { warmth: 0.4, energy: 0.7, openness: 0.35, focus: 0.85 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4, duration: 3.3, cooldown: 60,
    explicitOnly: true,
    notes: 'Contact region: chest.',
  },
  {
    id: 'bvh-hit-foot',
    type: 'bvh',
    source: '/bvh-library/animation/hitarea_foot.bvh',
    roles: ['react-contact', 'startled-low'],
    emotion: { warmth: 0.5, energy: 0.5, openness: 0.4, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.3, duration: 1.6, cooldown: 60,
    explicitOnly: true,
    notes: 'Contact region: foot.',
  },
  {
    id: 'bvh-hit-groin',
    type: 'bvh',
    source: '/bvh-library/animation/hitarea_groin.bvh',
    roles: ['react-contact', 'startled-high', 'defensive'],
    emotion: { warmth: 0.3, energy: 0.8, openness: 0.25, focus: 0.9 },
    modes: ['chat-call', 'narrative'],
    cost: 0.45, duration: 3.3, cooldown: 120,
    explicitOnly: true,
    notes: 'Contact region: groin. Higher arousal recoil.',
  },
  {
    id: 'bvh-hit-hands',
    type: 'bvh',
    source: '/bvh-library/animation/hitarea_hands.bvh',
    roles: ['react-contact', 'engagement', 'warm-touch'],
    emotion: { warmth: 0.8, energy: 0.5, openness: 0.7, focus: 0.75 },
    modes: ['chat-call', 'narrative'],
    cost: 0.3, duration: 6.6, cooldown: 30,
    explicitOnly: true,
    notes: 'Contact region: hands. Friendly tier (no recoil).',
  },
  {
    id: 'bvh-hit-head',
    type: 'bvh',
    source: '/bvh-library/animation/hitarea_head.bvh',
    roles: ['react-contact', 'startled-high', 'defensive'],
    emotion: { warmth: 0.4, energy: 0.75, openness: 0.3, focus: 0.85 },
    modes: ['chat-call', 'narrative'],
    cost: 0.45, duration: 3.0, cooldown: 120,
    explicitOnly: true,
    notes: 'Contact region: head.',
  },
  {
    id: 'bvh-hit-leg',
    type: 'bvh',
    source: '/bvh-library/animation/hitarea_leg.bvh',
    roles: ['react-contact', 'startled-low'],
    emotion: { warmth: 0.5, energy: 0.55, openness: 0.4, focus: 0.75 },
    modes: ['chat-call', 'narrative'],
    cost: 0.35, duration: 18.0, cooldown: 60,
    explicitOnly: true,
    notes: 'Contact region: leg.',
  },
  {
    id: 'bvh-reaction-groinhit',
    type: 'bvh',
    source: '/bvh-library/animation/reaction_groinhit.bvh',
    roles: ['react-contact', 'pain', 'startled-high'],
    emotion: { warmth: 0.2, energy: 0.85, openness: 0.2, focus: 0.95 },
    modes: ['chat-call', 'narrative'],
    cost: 0.55, duration: 4.2, cooldown: 600,
    explicitOnly: true,
    notes: 'Stronger pain reaction. Reserved.',
  },
  {
    id: 'bvh-reaction-headshot',
    type: 'bvh',
    source: '/bvh-library/animation/reaction_headshot.bvh',
    roles: ['react-contact', 'pain', 'startled-high'],
    emotion: { warmth: 0.2, energy: 0.85, openness: 0.2, focus: 0.95 },
    modes: ['chat-call', 'narrative'],
    cost: 0.55, duration: 3.2, cooldown: 600,
    explicitOnly: true,
    notes: 'Strong impact reaction. Reserved.',
  },

  // ─── K: BOOTH dance pack (creator-licensed J-pop / Vocaloid VRMAs) ─
  // 11 dance VRMAs sourced from booth.pm (free / creator-licensed).
  // Each zip ships with a /vrma/ folder; we extract that and rename
  // the file to an ASCII slug. Same roles + emotion profile as the
  // existing dance pool so they're peers in conductor selection.
  // Durations approximate (unverified — refine per-clip after preview).
  {
    id: 'dance-doumo-tensai',
    type: 'vrma',
    source: '/ui/lib/animations/doumo_tensai_desu.vrma',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.72, energy: 0.88, openness: 0.78, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.65, duration: 15.2, cooldown: 600, loop: true,
    framing: 'fullBody',
    notes: 'どーも、天才です ("Hi, I\'m a Genius"). BOOTH creator-licensed.',
  },
  {
    id: 'dance-hitamuki-cinderella',
    type: 'vrma',
    source: '/ui/lib/animations/hitamuki_cinderella.vrma',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.82, energy: 0.82, openness: 0.82, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.6, duration: 17.1, cooldown: 600, loop: true,
    framing: 'fullBody',
    notes: 'ひたむきシンデレラ. BOOTH creator-licensed.',
  },
  {
    id: 'dance-kokoro-yoho',
    type: 'vrma',
    source: '/ui/lib/animations/kokoro_yoho.vrma',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.75, energy: 0.78, openness: 0.78, focus: 0.65 },
    modes: ['chat-call', 'narrative'],
    cost: 0.55, duration: 15.5, cooldown: 600, loop: true,
    framing: 'fullBody',
    notes: '心予報 (Heart Forecast — Vocaloid). BOOTH creator-licensed.',
  },
  {
    id: 'dance-countach',
    type: 'vrma',
    source: '/ui/lib/animations/countach.vrma',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.7, energy: 0.9, openness: 0.78, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.75, duration: 21.7, cooldown: 600, loop: true,
    framing: 'fullBody',
    notes: 'Countach. BOOTH creator-licensed.',
  },
  {
    id: 'dance-dokidoki-kyun',
    type: 'vrma',
    source: '/ui/lib/animations/dokidoki_kyun.vrma',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.82, energy: 0.92, openness: 0.85, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.75, duration: 15.5, cooldown: 600, loop: true,
    framing: 'fullBody',
    notes: 'どきどきキュンで大暴走 ("Doki Doki Kyun"). High energy.',
  },
  {
    id: 'dance-kiss-kitsune',
    type: 'vrma',
    source: '/ui/lib/animations/kiss_kitsune.vrma',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.85, energy: 0.8, openness: 0.85, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.6, duration: 16.6, cooldown: 600, loop: true,
    framing: 'fullBody',
    notes: 'キスキツネ ("Kiss Fox"). BOOTH creator-licensed.',
  },
  {
    id: 'dance-ai-scream',
    type: 'vrma',
    source: '/ui/lib/animations/ai_scream.vrma',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.78, energy: 0.92, openness: 0.85, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.9, duration: 32.7, cooldown: 600, loop: true,
    framing: 'fullBody',
    notes: '愛スクリーム ("Love Scream"). Long form; biggest file in pack.',
  },
  {
    id: 'dance-tetris',
    type: 'vrma',
    source: '/ui/lib/animations/tetris.vrma',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.6, energy: 0.85, openness: 0.7, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.7, duration: 15.9, cooldown: 600, loop: true,
    framing: 'fullBody',
    notes: 'テトリス (Tetris-themed dance). BOOTH creator-licensed.',
  },
  {
    id: 'dance-shikanoko',
    type: 'vrma',
    source: '/ui/lib/animations/shikanoko.vrma',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.85, energy: 0.85, openness: 0.85, focus: 0.5 },
    modes: ['chat-call', 'narrative'],
    cost: 0.7, duration: 5.2, cooldown: 600, loop: true,
    framing: 'fullBody',
    notes: 'しかのこのこのここしたんたん (viral meme dance). Short / loopable.',
  },
  {
    id: 'dance-ui-mugibatake',
    type: 'vrma',
    source: '/ui/lib/animations/ui_mugibatake.vrma',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.78, energy: 0.78, openness: 0.78, focus: 0.65 },
    modes: ['chat-call', 'narrative'],
    cost: 0.55, duration: 26.8, cooldown: 600, loop: true,
    framing: 'fullBody',
    notes: 'うい麦畑でつかまえて. BOOTH creator-licensed.',
  },
  {
    id: 'dance-soiree',
    type: 'vrma',
    source: '/ui/lib/animations/soiree.vrma',
    roles: ['dance', 'idle-fill', 'show-off'],
    emotion: { warmth: 0.78, energy: 0.85, openness: 0.8, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.7, duration: 28.7, cooldown: 600, loop: true,
    framing: 'fullBody',
    notes: 'ソワレ (Soirée). Longer-form; second-biggest in pack.',
  },

  // ─── K: Adopted variants (2026-06-10) ────────────────────────────
  // Files present on disk but never wired. Most are numbered variants
  // of existing emotion BVHs (anger2/anger3 vs anger.bvh) — adopting
  // them gives per-emotion variety so the selector doesn't lock in on
  // the single base take. Tags cloned from the corresponding base
  // entry above; cooldowns nudged so siblings don't all fire in the
  // same window. Durations are best-effort estimates from file size
  // (BVH ≈ 30 KB/sec at 60fps × small skeleton); the avatar pipeline
  // auto-corrects via vrmaCurrentDuration on first play.
  //
  // Annotation key (notes):
  //   v2 / v3 — alternate take of the same emotion
  //   activity — non-emotion behavioral clip (exercise, distraction)

  // Emotion variants — anger
  {
    id: 'bvh-emotion-anger-v2',
    type: 'bvh',
    source: '/bvh-library/animation/anger2.bvh',
    roles: ['react-negative', 'angry', 'mirror-anger'],
    emotion: { warmth: 0.15, energy: 0.9, openness: 0.35, focus: 0.85 },
    modes: ['chat-call', 'narrative'],
    cost: 0.55, duration: 20.8, cooldown: 360,
    notes: 'Anger v2 — larger movement. Cooldown raised vs base so the '
         + 'pair doesn\'t fire back-to-back.',
  },
  {
    id: 'bvh-emotion-anger-v3',
    type: 'bvh',
    source: '/bvh-library/animation/anger3.bvh',
    roles: ['react-negative', 'angry', 'mirror-anger'],
    emotion: { warmth: 0.18, energy: 0.85, openness: 0.4, focus: 0.8 },
    modes: ['chat-call', 'narrative'],
    cost: 0.6, duration: 28.2, cooldown: 360,
    notes: 'Anger v3.',
  },

  // Annoyance — milder than anger
  {
    id: 'bvh-emotion-annoyance',
    type: 'bvh',
    source: '/bvh-library/animation/annoyance1.bvh',
    roles: ['react-negative', 'mirror-anger', 'idle-shift'],
    emotion: { warmth: 0.35, energy: 0.55, openness: 0.45, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.35, duration: 2.9, cooldown: 180,
    notes: 'Annoyance — milder than anger. Sigh-shaped.',
  },

  // Admiration variants
  {
    id: 'bvh-emotion-admiration-v2',
    type: 'bvh',
    source: '/bvh-library/animation/admiration2.bvh',
    roles: ['react-positive', 'agree', 'gratitude'],
    emotion: { warmth: 0.82, energy: 0.55, openness: 0.75, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.35, duration: 1.2, cooldown: 240,
    notes: 'Admiration v2.',
  },
  {
    id: 'bvh-emotion-admiration-v3',
    type: 'bvh',
    source: '/bvh-library/animation/admiration3.bvh',
    roles: ['react-positive', 'agree', 'gratitude'],
    emotion: { warmth: 0.85, energy: 0.5, openness: 0.78, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4, duration: 1.7, cooldown: 240,
    notes: 'Admiration v3.',
  },

  // Approval variants
  {
    id: 'bvh-emotion-approval-v2',
    type: 'bvh',
    source: '/bvh-library/animation/approval2.bvh',
    roles: ['agree', 'react-positive', 'gratitude'],
    emotion: { warmth: 0.75, energy: 0.55, openness: 0.7, focus: 0.65 },
    modes: ['chat-call', 'narrative'],
    cost: 0.3, duration: 1.2, cooldown: 180,
    notes: 'Approval v2.',
  },
  {
    id: 'bvh-emotion-approval-v3',
    type: 'bvh',
    source: '/bvh-library/animation/approval3.bvh',
    roles: ['agree', 'react-positive', 'gratitude'],
    emotion: { warmth: 0.78, energy: 0.5, openness: 0.72, focus: 0.65 },
    modes: ['chat-call', 'narrative'],
    cost: 0.35, duration: 1.7, cooldown: 180,
    notes: 'Approval v3.',
  },

  // Caring variant
  {
    id: 'bvh-emotion-caring-v1',
    type: 'bvh',
    source: '/bvh-library/animation/caring1.bvh',
    roles: ['affection', 'warmth', 'react-positive'],
    emotion: { warmth: 0.88, energy: 0.45, openness: 0.8, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.45, duration: 22.4, cooldown: 360,
    notes: 'Caring v1 — extended take.',
  },

  // Confusion variants
  {
    id: 'bvh-emotion-confusion-v2',
    type: 'bvh',
    source: '/bvh-library/animation/confusion2.bvh',
    roles: ['react-confused', 'question'],
    emotion: { warmth: 0.5, energy: 0.4, openness: 0.55, focus: 0.5 },
    modes: ['chat-call', 'narrative'],
    cost: 0.35, duration: 4.8, cooldown: 240,
    notes: 'Confusion v2.',
  },
  {
    id: 'bvh-emotion-confusion-v3',
    type: 'bvh',
    source: '/bvh-library/animation/confusion3.bvh',
    roles: ['react-confused', 'question'],
    emotion: { warmth: 0.5, energy: 0.42, openness: 0.55, focus: 0.5 },
    modes: ['chat-call', 'narrative'],
    cost: 0.35, duration: 4.5, cooldown: 240,
    notes: 'Confusion v3.',
  },

  // Curiosity variants
  {
    id: 'bvh-emotion-curiosity-v2',
    type: 'bvh',
    source: '/bvh-library/animation/curiosity2.bvh',
    roles: ['curiosity', 'question', 'mirror-curious'],
    emotion: { warmth: 0.6, energy: 0.5, openness: 0.75, focus: 0.78 },
    modes: ['chat-call', 'narrative'],
    cost: 0.35, duration: 5.8, cooldown: 240,
    notes: 'Curiosity v2.',
  },
  {
    id: 'bvh-emotion-curiosity-v3',
    type: 'bvh',
    source: '/bvh-library/animation/curiosity3.bvh',
    roles: ['curiosity', 'question', 'mirror-curious'],
    emotion: { warmth: 0.6, energy: 0.5, openness: 0.75, focus: 0.78 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4, duration: 6.3, cooldown: 240,
    notes: 'Curiosity v3.',
  },

  // Desire variants
  {
    id: 'bvh-emotion-desire-v1',
    type: 'bvh',
    source: '/bvh-library/animation/desire1.bvh',
    roles: ['longing', 'curious'],
    emotion: { warmth: 0.72, energy: 0.55, openness: 0.7, focus: 0.85 },
    modes: ['chat-call', 'narrative'],
    cost: 0.45, duration: 15.3, cooldown: 720,
    notes: 'Desire v1.',
  },
  {
    id: 'bvh-emotion-desire-v2',
    type: 'bvh',
    source: '/bvh-library/animation/desire2.bvh',
    roles: ['longing', 'curious'],
    emotion: { warmth: 0.72, energy: 0.55, openness: 0.72, focus: 0.85 },
    modes: ['chat-call', 'narrative'],
    cost: 0.5, duration: 21.9, cooldown: 720,
    notes: 'Desire v2 — longest take.',
  },

  // Disappointment variant
  {
    id: 'bvh-emotion-disappointment-v2',
    type: 'bvh',
    source: '/bvh-library/animation/disappointment2.bvh',
    roles: ['react-negative', 'wind-down', 'idle-shift'],
    emotion: { warmth: 0.4, energy: 0.3, openness: 0.4, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.35, duration: 5.5, cooldown: 360,
    notes: 'Disappointment v2.',
  },

  // Disapproval (variant naming has a typo in source: disaproval1.bvh)
  {
    id: 'bvh-emotion-disapproval-v1',
    type: 'bvh',
    source: '/bvh-library/animation/disaproval1.bvh',
    roles: ['react-negative', 'mirror-anger'],
    emotion: { warmth: 0.3, energy: 0.5, openness: 0.4, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4, duration: 8.6, cooldown: 360,
    notes: 'Disapproval v1. Source filename is mis-spelled upstream '
         + '(disaproval1.bvh, missing second p).',
  },

  // Disgust variants
  {
    id: 'bvh-emotion-disgust-v1',
    type: 'bvh',
    source: '/bvh-library/animation/disgust1.bvh',
    roles: ['react-negative', 'recoil'],
    emotion: { warmth: 0.2, energy: 0.55, openness: 0.3, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.35, duration: 3.3, cooldown: 240,
    notes: 'Disgust v1.',
  },
  {
    id: 'bvh-emotion-disgust-v2',
    type: 'bvh',
    source: '/bvh-library/animation/disgust2.bvh',
    roles: ['react-negative', 'recoil'],
    emotion: { warmth: 0.2, energy: 0.6, openness: 0.3, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.5, duration: 38.9, cooldown: 600,
    notes: 'Disgust v2 — extended take (3.4MB source).',
  },

  // Excitement variant (base + v3 already wired)
  {
    id: 'bvh-emotion-excitement-v2',
    type: 'bvh',
    source: '/bvh-library/animation/excitement2.bvh',
    roles: ['celebrate', 'react-positive', 'show-off'],
    emotion: { warmth: 0.85, energy: 0.82, openness: 0.7, focus: 0.65 },
    modes: ['chat-call', 'narrative'],
    cost: 0.5, duration: 5.3, cooldown: 120,
    notes: 'Excitement v2 — between base and v3.',
  },

  // Fear variants
  {
    id: 'bvh-emotion-fear-v2',
    type: 'bvh',
    source: '/bvh-library/animation/fear2.bvh',
    roles: ['react-negative', 'startled', 'apprehensive'],
    emotion: { warmth: 0.3, energy: 0.7, openness: 0.25, focus: 0.85 },
    modes: ['chat-call', 'narrative'],
    cost: 0.5, duration: 3.9, cooldown: 360,
    notes: 'Fear v2.',
  },
  {
    id: 'bvh-emotion-fear-v3',
    type: 'bvh',
    source: '/bvh-library/animation/fear3.bvh',
    roles: ['react-negative', 'startled', 'apprehensive'],
    emotion: { warmth: 0.3, energy: 0.7, openness: 0.25, focus: 0.85 },
    modes: ['chat-call', 'narrative'],
    cost: 0.5, duration: 3.8, cooldown: 360,
    notes: 'Fear v3.',
  },

  // Joy variant (base + joy3 already wired)
  {
    id: 'bvh-emotion-joy-v2',
    type: 'bvh',
    source: '/bvh-library/animation/joy2.bvh',
    roles: ['react-positive', 'celebrate'],
    emotion: { warmth: 0.85, energy: 0.8, openness: 0.85, focus: 0.65 },
    modes: ['chat-call', 'narrative'],
    cost: 0.5, duration: 8.2, cooldown: 240,
    notes: 'Joy v2 — between base and joy3.',
  },

  // Love variants
  {
    id: 'bvh-emotion-love-v2',
    type: 'bvh',
    source: '/bvh-library/animation/love2.bvh',
    roles: ['affection', 'warmth', 'react-positive'],
    emotion: { warmth: 0.92, energy: 0.5, openness: 0.85, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.45, duration: 10.9, cooldown: 480,
    notes: 'Love v2.',
  },
  {
    id: 'bvh-emotion-love-v3',
    type: 'bvh',
    source: '/bvh-library/animation/love3.bvh',
    roles: ['affection', 'warmth', 'react-positive'],
    emotion: { warmth: 0.9, energy: 0.5, openness: 0.85, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.35, duration: 4.2, cooldown: 240,
    notes: 'Love v3 — shorter take.',
  },

  // Nervousness variants (note: nervousnes3 is mis-spelled upstream)
  {
    id: 'bvh-emotion-nervousness-v2',
    type: 'bvh',
    source: '/bvh-library/animation/nervousness2.bvh',
    roles: ['apprehensive', 'startled', 'idle-shift'],
    emotion: { warmth: 0.4, energy: 0.55, openness: 0.4, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.35, duration: 3.9, cooldown: 240,
    notes: 'Nervousness v2.',
  },
  {
    id: 'bvh-emotion-nervousness-v3',
    type: 'bvh',
    source: '/bvh-library/animation/nervousnes3.bvh',
    roles: ['apprehensive', 'startled', 'idle-shift'],
    emotion: { warmth: 0.4, energy: 0.55, openness: 0.4, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.35, duration: 3.8, cooldown: 240,
    notes: 'Nervousness v3. Source filename mis-spelled upstream '
         + '(nervousnes3.bvh, missing second s).',
  },

  // Neutral variants — useful as low-cost idle fillers
  {
    id: 'bvh-emotion-neutral-v2',
    type: 'bvh',
    source: '/bvh-library/animation/neutral2.bvh',
    roles: ['idle-fill', 'react-neutral'],
    emotion: { warmth: 0.55, energy: 0.4, openness: 0.55, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.2, duration: 17.5, cooldown: 90,
    notes: 'Neutral v2 — extended take.',
  },
  {
    id: 'bvh-emotion-neutral-v3',
    type: 'bvh',
    source: '/bvh-library/animation/neutral3.bvh',
    roles: ['idle-fill', 'react-neutral'],
    emotion: { warmth: 0.55, energy: 0.4, openness: 0.55, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.15, duration: 6.0, cooldown: 90,
    notes: 'Neutral v3.',
  },
  {
    id: 'bvh-emotion-neutral-v4',
    type: 'bvh',
    source: '/bvh-library/animation/neutral4.bvh',
    roles: ['idle-fill', 'react-neutral'],
    emotion: { warmth: 0.55, energy: 0.4, openness: 0.55, focus: 0.55 },
    modes: ['chat-call', 'narrative'],
    cost: 0.15, duration: 6.2, cooldown: 90,
    notes: 'Neutral v4.',
  },

  // Relief variant
  {
    id: 'bvh-emotion-relief-v1',
    type: 'bvh',
    source: '/bvh-library/animation/relief1.bvh',
    roles: ['settle', 'wind-down', 'react-positive'],
    emotion: { warmth: 0.72, energy: 0.4, openness: 0.65, focus: 0.5 },
    modes: ['chat-call', 'narrative'],
    cost: 0.35, duration: 10.5, cooldown: 240,
    notes: 'Relief v1.',
  },

  // Remorse variants
  {
    id: 'bvh-emotion-remorse-v2',
    type: 'bvh',
    source: '/bvh-library/animation/remorse2.bvh',
    roles: ['react-negative', 'apology', 'wind-down'],
    emotion: { warmth: 0.5, energy: 0.35, openness: 0.45, focus: 0.5 },
    modes: ['chat-call', 'narrative'],
    cost: 0.35, duration: 6.9, cooldown: 360,
    notes: 'Remorse v2.',
  },
  {
    id: 'bvh-emotion-remorse-v3',
    type: 'bvh',
    source: '/bvh-library/animation/remorse3.bvh',
    roles: ['react-negative', 'apology', 'wind-down'],
    emotion: { warmth: 0.5, energy: 0.35, openness: 0.45, focus: 0.5 },
    modes: ['chat-call', 'narrative'],
    cost: 0.3, duration: 9.7, cooldown: 360,
    notes: 'Remorse v3 — shorter take.',
  },

  // Sadness variant
  {
    id: 'bvh-emotion-sadness-v2',
    type: 'bvh',
    source: '/bvh-library/animation/sadness2.bvh',
    roles: ['react-negative', 'wind-down'],
    emotion: { warmth: 0.45, energy: 0.3, openness: 0.4, focus: 0.5 },
    modes: ['chat-call', 'narrative'],
    cost: 0.3, duration: 1.5, cooldown: 360,
    notes: 'Sadness v2 — quick take.',
  },

  // Surprise variant
  {
    id: 'bvh-emotion-surprise-v2',
    type: 'bvh',
    source: '/bvh-library/animation/surprise2.bvh',
    roles: ['react-surprise', 'startled', 'react-positive'],
    emotion: { warmth: 0.55, energy: 0.8, openness: 0.65, focus: 0.85 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4, duration: 3.7, cooldown: 240,
    notes: 'Surprise v2.',
  },

  // Exercise / activity BVHs — not emotion-tagged. Used when the
  // companion needs visible-busy behavior (idle escalation > 45s,
  // user-requested workout demo, narrative gym scenes).
  {
    id: 'bvh-activity-crunch',
    type: 'bvh',
    source: '/bvh-library/animation/exercise_crunch.bvh',
    roles: ['idle-fill', 'idle-distracted', 'show-off'],
    emotion: { warmth: 0.55, energy: 0.7, openness: 0.55, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4, duration: 2.4, cooldown: 900, explicitOnly: true,
    framing: 'fullBody',
    notes: 'Single crunch. Explicit-only — fullBody framing breaks '
         + 'most chat-call camera framings.',
  },
  {
    id: 'bvh-activity-crunches',
    type: 'bvh',
    source: '/bvh-library/animation/exercise_crunches.bvh',
    roles: ['idle-fill', 'idle-distracted', 'show-off'],
    emotion: { warmth: 0.55, energy: 0.75, openness: 0.55, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4, duration: 1.1, cooldown: 900, explicitOnly: true,
    framing: 'fullBody',
    notes: 'Crunch reps loop.',
  },
  {
    id: 'bvh-activity-jogging',
    type: 'bvh',
    source: '/bvh-library/animation/exercise_jogging.bvh',
    roles: ['idle-fill', 'idle-distracted', 'show-off'],
    emotion: { warmth: 0.55, energy: 0.78, openness: 0.6, focus: 0.65 },
    modes: ['chat-call', 'narrative'],
    cost: 0.45, duration: 2.6, cooldown: 1200, explicitOnly: true,
    framing: 'fullBody',
    notes: 'Jogging-in-place. Explicit-only.',
  },
  {
    id: 'bvh-activity-jumping-jacks',
    type: 'bvh',
    source: '/bvh-library/animation/exercise_jumping_jacks.bvh',
    roles: ['idle-fill', 'idle-distracted', 'show-off'],
    emotion: { warmth: 0.55, energy: 0.85, openness: 0.65, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.5, duration: 1.1, cooldown: 1200, explicitOnly: true,
    framing: 'fullBody',
    notes: 'Jumping jacks. Explicit-only.',
  },

  // ─── VRMAs from animation_nitral-fork — alternates of wired set ──
  // All explicit-only initially. They're alternate takes of clips we
  // already wired from ui/lib/animations/; promote out of explicit-
  // only once you've eye-tested each one. Served via the /bvh-library
  // mount (→ poses/external/sillytavern-pack/). The original /poses/...
  // URLs 404'd — that mount's allowlist rejects 2+-level nesting —
  // so these nine never played until the 2026-06-10 ground-truth
  // audit caught it.
  {
    id: 'nitral-greeting',
    type: 'vrma',
    source: '/bvh-library/animation_nitral-fork/greeting.vrma',
    roles: ['greet', 'wave'],
    emotion: { warmth: 0.82, energy: 0.6, openness: 0.7, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4, duration: 7.3, cooldown: 3600, explicitOnly: true,
    framing: 'fullBody',
    notes: 'Alternate greeting from nitral-fork. Explicit-only — '
         + 'unproven mount path; test before auto-firing.',
  },
  {
    id: 'nitral-greeting-v2',
    type: 'vrma',
    source: '/bvh-library/animation_nitral-fork/greeting2.vrma',
    roles: ['greet', 'wave'],
    emotion: { warmth: 0.8, energy: 0.6, openness: 0.7, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.35, duration: 5.2, cooldown: 3600, explicitOnly: true,
    framing: 'fullBody',
    notes: 'Greeting v2 — shorter alternate.',
  },
  {
    id: 'nitral-hello',
    type: 'vrma',
    source: '/bvh-library/animation_nitral-fork/hello.vrma',
    roles: ['greet', 'wave'],
    emotion: { warmth: 0.82, energy: 0.6, openness: 0.7, focus: 0.6 },
    modes: ['chat-call', 'narrative'],
    cost: 0.35, duration: 15.8, cooldown: 3600, explicitOnly: true,
    framing: 'fullBody',
    notes: 'Alternate hello from nitral-fork.',
  },
  {
    id: 'nitral-model-pose',
    type: 'vrma',
    source: '/bvh-library/animation_nitral-fork/model pose.vrma',
    roles: ['ponder', 'pose-display', 'idle-shift'],
    emotion: { warmth: 0.6, energy: 0.45, openness: 0.55, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.5, duration: 7.5, cooldown: 600, explicitOnly: true,
    framing: 'fullBody',
    notes: 'Alternate model-pose.',
  },
  {
    id: 'nitral-motion-pose',
    type: 'vrma',
    source: '/bvh-library/animation_nitral-fork/motion_pose.vrma',
    roles: ['ponder', 'pose-display', 'idle-shift'],
    emotion: { warmth: 0.6, energy: 0.5, openness: 0.55, focus: 0.7 },
    modes: ['chat-call', 'narrative'],
    cost: 0.5, duration: 20.0, cooldown: 600, explicitOnly: true,
    framing: 'fullBody',
    notes: 'Alternate motion-pose.',
  },
  {
    id: 'nitral-peace-sign',
    type: 'vrma',
    source: '/bvh-library/animation_nitral-fork/peace sign.vrma',
    roles: ['agree', 'agree-strong', 'farewell'],
    emotion: { warmth: 0.7, energy: 0.55, openness: 0.6, focus: 0.65 },
    modes: ['chat-call', 'narrative'],
    cost: 0.4, duration: 11.7, cooldown: 600, explicitOnly: true,
    framing: 'fullBody',
    notes: 'Alternate peace-sign.',
  },
  {
    id: 'nitral-shoot',
    type: 'vrma',
    source: '/bvh-library/animation_nitral-fork/shoot.vrma',
    roles: ['emphasize', 'point'],
    emotion: { warmth: 0.55, energy: 0.75, openness: 0.5, focus: 0.85 },
    modes: ['chat-call', 'narrative'],
    cost: 0.45, duration: 9.6, cooldown: 240, explicitOnly: true,
    notes: 'Alternate shoot/finger-guns.',
  },
  {
    id: 'nitral-show-full-body',
    type: 'vrma',
    source: '/bvh-library/animation_nitral-fork/show full body.vrma',
    roles: ['pose-display', 'show-off', 'idle-shift'],
    emotion: { warmth: 0.65, energy: 0.55, openness: 0.65, focus: 0.65 },
    modes: ['chat-call', 'narrative'],
    cost: 0.55, duration: 11.8, cooldown: 1800, explicitOnly: true,
    framing: 'fullBody',
    notes: 'Full-body display pose. Explicit-only.',
  },
  {
    id: 'nitral-spin',
    type: 'vrma',
    source: '/bvh-library/animation_nitral-fork/spin.vrma',
    roles: ['celebrate', 'react-positive', 'show-off'],
    emotion: { warmth: 0.8, energy: 0.92, openness: 0.85, focus: 0.5 },
    modes: ['chat-call', 'narrative'],
    cost: 0.8, duration: 9.3, cooldown: 240, explicitOnly: true,
    framing: 'fullBody',
    notes: 'Alternate spin/twirl.',
  },
];

// id → entry index for O(1) lookup
const _BY_ID = Object.fromEntries(ATLAS.map((e) => [e.id, e]));

// ─── User-uploaded animations ─────────────────────────────────────────
//
// Bundled ATLAS entries are stable code. Uploads live server-side in
// the user_animations table and are fetched on widget mount, then
// registered here so select() and getAnim() see a unified atlas.
//
// User ids are always 'user:<ts>_<hex>' so they can never collide with
// bundled ids. Registration is idempotent — the most recent call wins
// (lets the widget refresh after a rename without an explicit clear).
const _USER_ANIMS = [];
const _USER_BY_ID = new Map();

/**
 * Replace the user-animations registry with ``entries`` (atlas-shape
 * objects from /api/animations/list). Idempotent — call after every
 * server refresh.
 *
 * @param {object[]} entries
 */
export function registerUserAnimations(entries) {
  _USER_ANIMS.length = 0;
  _USER_BY_ID.clear();
  for (const e of (entries || [])) {
    if (!e?.id) continue;
    _USER_ANIMS.push(e);
    _USER_BY_ID.set(e.id, e);
  }
  _invalidatePool();
}

/** Number of user-registered entries currently in the atlas. */
export function userAnimationCount() {
  return _USER_ANIMS.length;
}

// ─── Bundled-entry overrides (user_atlas_overrides, server-side) ──
//
// Per-user customization of the BUNDLED entries above: ``disabled``
// removes an entry from auto-selection; ``patch`` shallow-merges
// edited metadata (roles, emotion, cost, ...) over the code-defined
// entry. Uploads are edited directly via PUT /api/animations/{id} and
// never appear here.

const _OVERRIDES = new Map();   // atlas_id → {disabled, patch}

/**
 * Replace the override registry with rows from
 * ``GET /api/animations/overrides``. Idempotent, like
 * registerUserAnimations — most recent call wins.
 *
 * @param {Array<{atlas_id: string, disabled: boolean, patch: object}>} rows
 */
export function registerAtlasOverrides(rows) {
  _OVERRIDES.clear();
  for (const r of (rows || [])) {
    if (!r?.atlas_id) continue;
    _OVERRIDES.set(r.atlas_id, {
      disabled: !!r.disabled,
      patch: (r.patch && typeof r.patch === 'object') ? r.patch : {},
    });
  }
  _invalidatePool();
}

/** The override for one bundled id, or null. */
export function getOverride(id) {
  return _OVERRIDES.get(id) ?? null;
}

function _patched(entry) {
  const ov = _OVERRIDES.get(entry.id);
  if (!ov || !Object.keys(ov.patch).length) return entry;
  return { ...entry, ...ov.patch };
}

// Effective selection pool: bundled entries with patches applied and
// disabled ones dropped, plus uploads. Memoized — select() runs every
// beat; rebuild only when a registry changes.
let _POOL = null;
function _invalidatePool() { _POOL = null; }
function _effectivePool() {
  if (!_POOL) {
    _POOL = [
      ...ATLAS.filter((e) => !_OVERRIDES.get(e.id)?.disabled).map(_patched),
      ..._USER_ANIMS,
    ];
  }
  return _POOL;
}

/**
 * Every entry — bundled (patched) + uploads — INCLUDING disabled ones,
 * for management UIs. Each row gains:
 *   bundled:  true for code-defined entries (no delete, only disable)
 *   disabled: current per-user disable state (always false for uploads)
 */
export function listEffectiveEntries() {
  return [
    ...ATLAS.map((e) => ({
      ..._patched(e),
      bundled: true,
      disabled: !!_OVERRIDES.get(e.id)?.disabled,
    })),
    ..._USER_ANIMS.map((e) => ({ ...e, bundled: false, disabled: false })),
  ];
}

/**
 * Distinct role vocabulary across the effective pool (disabled entries
 * excluded — a role only reachable through a disabled entry isn't
 * callable). Sorted for stable display + tool schemas.
 */
export function listRoles() {
  const roles = new Set();
  for (const e of _effectivePool()) {
    for (const r of (e.roles || [])) roles.add(r);
  }
  return [...roles].sort();
}

// ─── Category families ───────────────────────────────────────────
//
// A coarse grouping over the open role vocabulary so the library UI
// can sort ~150 entries into browsable buckets. Derived, NOT stored:
// an entry's family follows from its roles, so re-tagging a clip
// moves it between families automatically and user uploads slot in
// with zero extra metadata. Dominant role (roles[0]) is tried first
// so e.g. ['celebrate', 'greet'] lands in emotion, not greetings;
// then any-role match in rule order. Unknown/custom roles → 'other'.
const _FAMILY_RULES = [
  ['dance', new Set(['dance', 'show-off', 'pose-display', 'hobby'])],
  ['greetings', new Set(['greet', 'wave', 'farewell', 'bow-deep',
    'late-call'])],
  ['idle', new Set(['idle-fill', 'idle-shift', 'idle-attentive',
    'idle-relaxed', 'idle-distracted', 'idle-low-energy', 'settle',
    'settled', 'wind-down', 'posture-shift', 'locomotion'])],
  ['reactions', new Set(['react-positive', 'react-negative',
    'react-neutral', 'react-surprise', 'react-confused',
    'react-compliment', 'react-contact', 'startled', 'startled-low',
    'startled-mid', 'startled-high', 'recoil', 'realization',
    'oh-i-see', 'relief', 'defensive'])],
  ['mirroring', new Set(['mirror-sad', 'mirror-anger', 'mirror-curious',
    'mirror-confused', 'mirror-nervous', 'mirror-embarrassed',
    'mirror-disappointed'])],
  ['conversation', new Set(['agree', 'agree-strong', 'disagree-strong',
    'affirm', 'emphasize', 'point', 'question', 'listen', 'think',
    'ponder', 'attentive', 'engagement', 'apologetic', 'apology',
    'attention-seek'])],
  ['affection', new Set(['affection', 'warmth', 'warm-touch', 'comfort',
    'soften', 'sympathy', 'gratitude', 'longing', 'reach-out', 'reach',
    'picking-up'])],
  ['emotion', new Set(['joy', 'celebrate', 'excitement-peak',
    'amusement', 'micro-laugh', 'playful', 'pride', 'sad', 'angry',
    'frustration', 'disappointment', 'embarrassment', 'nervousness',
    'shy', 'hesitant', 'apprehensive', 'confusion', 'curiosity',
    'curious', 'pain', 'forward-looking'])],
];

export const FAMILY_ORDER = [
  ..._FAMILY_RULES.map(([name]) => name), 'other',
];

export function familyOf(entry) {
  const roles = entry?.roles || [];
  if (roles.length) {
    for (const [family, members] of _FAMILY_RULES) {
      if (members.has(roles[0])) return family;
    }
    for (const [family, members] of _FAMILY_RULES) {
      if (roles.some((r) => members.has(r))) return family;
    }
  }
  return 'other';
}

/** Families present in the effective pool (disabled included — the
 *  library lists those too), with entry counts, in FAMILY_ORDER. */
export function listFamilies() {
  const counts = new Map();
  for (const e of listEffectiveEntries()) {
    const f = familyOf(e);
    counts.set(f, (counts.get(f) || 0) + 1);
  }
  return FAMILY_ORDER
    .filter((f) => counts.has(f))
    .map((f) => ({ family: f, count: counts.get(f) }));
}

export function getAnim(id) {
  const bundled = _BY_ID[id];
  if (bundled) return _patched(bundled);
  return _USER_BY_ID.get(id) ?? null;
}

export function listIds() {
  return _effectivePool().map((e) => e.id);
}

// ─── Selection ─────────────────────────────────────────────────────

/**
 * Select the best-fit animation for an intent + context, or null.
 *
 * @param {object} intent
 * @param {string[]} [intent.roles]    desired roles to match
 * @param {object}   [intent.emotion]  target emotion {warmth,energy,openness,focus}
 *
 * @param {object} context
 * @param {string}                   context.mode           current mode
 * @param {number}                  [context.energyBudget=1] [0..1] remaining
 * @param {string[]}                [context.recent=[]]    last N played ids
 * @param {Map<string, number>}     [context.lastPlayed]   id → ms timestamp
 * @param {Object<string, number>}  [context.bias]         per-id multiplier
 * @param {boolean}                 [context.includeExplicitOnly=false]
 *                                  when true, allows entries flagged
 *                                  explicitOnly. Set true for direct
 *                                  user requests, false for auto-selection.
 * @param {number}                  [context.lastEnergy]   energy of the
 *                                  previously-played pick (0..1). When
 *                                  set, candidates whose energy delta
 *                                  exceeds ~0.4 from the last pick are
 *                                  scored down — prevents jarring "full-
 *                                  energy → languid → full-energy" jumps
 *                                  in continuous-rotation contexts
 *                                  (hosting-mode dance loop).
 * @param {number}                  [context.now]          test seam
 *
 * @returns {object|null}
 */
export function select(intent = {}, context = {}) {
  const now = context.now ?? Date.now();
  const energyBudget = context.energyBudget ?? 1.0;
  const recent = context.recent ?? [];
  const lastPlayed = context.lastPlayed ?? new Map();
  const bias = context.bias ?? {};

  // Unified pool: bundled (per-user patches applied, disabled entries
  // dropped) + user-uploaded. Uploads carry the same shape as bundled
  // entries (the route's _row_to_atlas_entry shapes them) so the
  // filter + score logic works identically.
  const pool = _effectivePool();
  // Active-loop constraint (Phase C). When the caller passes
  // ``context.activeLoopIds`` (a Set of animation ids), only members
  // of the loop are eligible. Empty Set = no loop active = entire
  // pool eligible. Empty-array-typed-as-Set is treated as "no loop"
  // too so a misconfigured loop with zero animations doesn't stall
  // the conductor.
  const loopIds = context.activeLoopIds;
  const useLoop = loopIds && typeof loopIds.has === 'function'
                  && loopIds.size > 0;
  const candidates = pool.filter((a) => {
    if (useLoop && !loopIds.has(a.id)) return false;
    if (a.explicitOnly && !context.includeExplicitOnly) return false;
    if (!_modeMatches(a.modes, context.mode)) return false;
    if (a.cost > energyBudget) return false;
    if (recent.includes(a.id)) return false;
    const lastAt = lastPlayed.get(a.id) ?? 0;
    if ((now - lastAt) < (a.cooldown ?? 0) * 1000) return false;
    return true;
  });

  if (!candidates.length) return null;

  const scored = candidates
    .map((a) => ({ anim: a, score: _scoreFor(a, intent, bias, context) }))
    .filter((s) => s.score > 0);

  if (!scored.length) return null;
  scored.sort((a, b) => b.score - a.score);
  return _weightedPick(scored.slice(0, 3))?.anim ?? null;
}

// ─── Helpers ───────────────────────────────────────────────────────

function _modeMatches(modes, ctxMode) {
  if (!modes?.length) return true;
  if (modes.includes('*')) return true;
  return modes.includes(ctxMode);
}

function _scoreFor(anim, intent, bias, context = {}) {
  let s = 0;
  // Role overlap is the dominant signal — a hard match on tags should
  // dominate over an emotion-vector closeness, since tags express the
  // function the caller asked for.
  if (intent.roles?.length) {
    const overlap = intent.roles.filter((r) => anim.roles?.includes(r)).length;
    s += overlap * 10;
  }
  // Emotion fit — secondary. Closer = better, capped at 5.
  if (intent.emotion && anim.emotion) {
    const dist = _emotionDistance(anim.emotion, intent.emotion);
    s += Math.max(0, 5 - dist * 5);
  }
  // Transition-aware energy penalty: when the caller passes the
  // previously-played pick's energy (continuous-rotation contexts like
  // the hosting-mode dance loop), penalize candidates that jump too
  // far in energy. Delta of 0.4 = penalty 2; delta of 1.0 = penalty 5
  // (caps at emotion-fit scale so it can rebalance the secondary
  // signal but doesn't override role overlap).
  if (context.lastEnergy != null
      && anim.emotion?.energy != null) {
    const delta = Math.abs(anim.emotion.energy - context.lastEnergy);
    if (delta > 0.4) {
      s -= Math.min(5, (delta - 0.4) * 8);
    }
  }
  // Per-id bias from observed reactions / character preferences.
  s *= (bias[anim.id] ?? 1.0);
  // Tiny jitter so equal-scored candidates rotate across calls.
  s += Math.random() * 0.3;
  return s;
}

function _emotionDistance(a, b) {
  const dw = (a.warmth ?? 0.5) - (b.warmth ?? 0.5);
  const de = (a.energy ?? 0.5) - (b.energy ?? 0.5);
  const dox = (a.openness ?? 0.5) - (b.openness ?? 0.5);
  const df = (a.focus ?? 0.5) - (b.focus ?? 0.5);
  return Math.sqrt(dw * dw + de * de + dox * dox + df * df);
}

function _weightedPick(scored) {
  const total = scored.reduce((s, x) => s + x.score, 0);
  if (total <= 0) return scored[0] ?? null;
  let r = Math.random() * total;
  for (const x of scored) {
    r -= x.score;
    if (r <= 0) return x;
  }
  return scored[scored.length - 1];
}
