/**
 * VRMA Library — curated catalog of approved animations the production
 * avatar pipeline (and the future PresenceDirector) is allowed to play.
 *
 * Authored 2026-05-02 from a hands-on audit of all 26 .vrma files in
 * `ui/lib/animations/`. Files that aren't approved (drinkwater needing
 * a prop + environment, sitting/floor poses that look broken in the
 * void, locomotion that doesn't fit a sit-down voice call) are
 * intentionally absent. The director queries this map; anything missing
 * is invisible to runtime.
 *
 * Field reference:
 *   url:           full path the player loads
 *   label:         human-readable name (used in pickers / debug logs)
 *   category:      'dance' | 'sustained_pose' | 'idle' | 'greeting' |
 *                  'reactive' | 'display' | 'exercise'
 *   loops:         whether it makes sense as a continuous loop. dances
 *                  & sustained poses loop; greetings & reactive don't.
 *   intensity:     'low' | 'medium' | 'high' — drives PIP-vs-main
 *                  eligibility (you don't want the active speaker
 *                  spinning during their TTS turn).
 *   duration_ms:   approximate visible duration (informational)
 *   suitable_for:  array of roles where this VRMA makes sense.
 *                  'main' = active speaker, 'pip' = listener,
 *                  'solo' = single-character call.
 *   trim_start_ms: optional. Skip this many ms of intro on play AND on
 *                  every loop wrap. Useful for clips with a dead intro
 *                  (VRMA_02 jumps in air for 3s before waving;
 *                  005_smartphone takes 5.9s to pick up the phone).
 *   trim_end_ms:   optional. Shorten the playable clip duration by this
 *                  many ms — cuts off glitchy final frames (VRMA_05
 *                  spin flips upside down in the last 0.8s).
 *   default_speed: optional, default 1.0. Playback rate multiplier.
 *                  Some choreography reads better slower (VRMA_05 spin
 *                  at 0.75x).
 *   prop:          optional. Phone, etc. Director loads the GLB and
 *                  parents to the named bone with offset/rotation.
 *   moodTrigger:   optional. Names the mood/emotion this VRMA expresses.
 *                  ('happy', 'sad', 'angry', 'surprised', 'tired',
 *                  'contemplative', 'shy', 'excited', 'curious'). Used
 *                  by EmbodimentEngine + ProceduralLoop to pick a VRMA
 *                  that matches the avatar's current internal state.
 *                  When absent, the mood is inferred from category +
 *                  intensity (dance + high → happy/excited; sustained
 *                  + low → neutral/calm; reactive + high → emphatic).
 *   moodAffinity:  optional. Explicit { emotion: weight, ... } dot
 *                  product target. Overrides the inferred affinity.
 *   endsCleanly:   optional, default true. When false, the avatar ends
 *                  the clip in a non-neutral position (legs splayed,
 *                  body rotated, arms up). The motion engine triggers
 *                  a full-body restoreToNatural transition after the
 *                  clip completes, BEFORE the mood-appropriate landing
 *                  gesture, to bring the avatar back to baseline.
 *   notes:         author's commentary
 */

const ANIM_PATH = '/ui/lib/animations';

/**
 * Freeze profiles for VRMA playback. The VRMAChannel locks specified
 * bones at their pre-play state so cumulative drift (hip yaw on loop
 * boundaries, head-pitch tracks that don't return cleanly) doesn't
 * compound. Trade-off: locked bones lose their authored animation.
 *
 *   TRUNK   - Locks the entire trunk + head. Body stays planted, only
 *             arms + legs + fingers animate. Right for dances where
 *             the clip drifts but you want the arms to swing. WRONG
 *             for emotion clips where the head/spine ARE the expression.
 *   MINIMAL - Just hips. Anchors the body in place but lets the spine,
 *             head, arms, legs all animate. Right for cleanly-authored
 *             clips that just need an anchor (most emotion + greeting
 *             + sustained-pose clips).
 *   NONE    - Nothing locked. Use only for clips that explicitly need
 *             full body translation (locomotion future work).
 *
 * Per-entry override: set `freezeBones: ['hips','spine','head']` etc.
 * for fine control. Default is TRUNK unless the entry overrides.
 */
export const FREEZE_PROFILES = Object.freeze({
  TRUNK:   ['hips', 'spine', 'chest', 'upperChest', 'neck', 'head'],
  MINIMAL: ['hips'],
  NONE:    [],
});

export const VRMA_LIBRARY = {
  // ─────────────────────────────────────────────────────────────────
  // Dances — high energy, full-body, loop-friendly. PIP-only by
  // default: the active speaker dancing during their TTS turn would
  // be jarring. Director uses these for "passing time" idles and
  // "PIP starts dancing → main reacts/joins" scenarios.
  // ─────────────────────────────────────────────────────────────────
  motion_pose: {
    url: `${ANIM_PATH}/001_motion_pose.vrma`,
    label: 'Motion pose',
    category: 'dance',
    loops: true,
    intensity: 'medium',
    duration_ms: 6000,
    suitable_for: ['main', 'pip', 'solo'],
    moodTrigger: 'happy',
    endsCleanly: false,
    notes: 'Hands on legs, bends each way, shuffles, spins. Dance-flavored idle. Director should gate on "not currently TTS-speaking" before triggering on main.',
  },
  dance_25: {
    url: `${ANIM_PATH}/dance_25.vrma`,
    label: 'Dance 25',
    category: 'dance',
    loops: false,
    intensity: 'high',
    duration_ms: 12000,
    suitable_for: ['main', 'pip', 'solo'],
    moodTrigger: 'happy',
    endsCleanly: false,
    notes: 'Full dance routine — high energy, big motion. Director gates on speech state.',
  },
  dance_28: {
    url: `${ANIM_PATH}/dance_28.vrma`,
    label: 'Dance 28',
    category: 'dance',
    loops: false,
    intensity: 'high',
    duration_ms: 16000,
    suitable_for: ['main', 'pip', 'solo'],
    moodTrigger: 'happy',
    endsCleanly: false,
    notes: 'Full dance routine — high energy, big motion. Director gates on speech state.',
  },
  dance_kebab: {
    url: `${ANIM_PATH}/dance_kebab.vrma`,
    label: 'Kebab Dance',
    category: 'dance',
    loops: false,
    intensity: 'high',
    duration_ms: 8000,
    suitable_for: ['main', 'pip', 'solo'],
    moodTrigger: 'happy',
    endsCleanly: false,
    notes: 'Comedic dance, big silly motion. Director gates on speech state.',
  },

  // ─────────────────────────────────────────────────────────────────
  // Sustained poses — held body shape with breathing animation only,
  // 0 hip range. The "attentive listener / contemplative / settled"
  // gaps from the design doc map here. Body shape per file is TBD
  // until visually labeled.
  // ─────────────────────────────────────────────────────────────────
  pose_world_smiles: {
    url: `${ANIM_PATH}/pose_world_smiles.vrma`,
    label: 'Pose · smiles',
    category: 'sustained_pose',
    loops: true,
    intensity: 'low',
    duration_ms: 3000,
    suitable_for: ['main', 'pip', 'solo'],
    notes: 'Static breathing-only pose. Body shape TBD on visual review.',
  },
  pose_world_lovely: {
    url: `${ANIM_PATH}/pose_world_lovely.vrma`,
    label: 'Pose · lovely',
    category: 'sustained_pose',
    loops: true,
    intensity: 'low',
    duration_ms: 3000,
    suitable_for: ['main', 'pip', 'solo'],
    notes: 'Static breathing-only pose. Body shape TBD on visual review.',
  },
  pose_world_sparkle: {
    url: `${ANIM_PATH}/pose_world_sparkle.vrma`,
    label: 'Pose · sparkle',
    category: 'sustained_pose',
    loops: true,
    intensity: 'low',
    duration_ms: 3000,
    suitable_for: ['main', 'pip', 'solo'],
    notes: 'Static breathing-only pose. Body shape TBD on visual review.',
  },
  pose_world_connected: {
    url: `${ANIM_PATH}/pose_world_connected.vrma`,
    label: 'Pose · connected',
    category: 'sustained_pose',
    loops: true,
    intensity: 'low',
    duration_ms: 3000,
    suitable_for: ['main', 'pip', 'solo'],
    notes: 'Static breathing-only pose. Body shape TBD on visual review.',
  },
  vrma_06_model: {
    url: `${ANIM_PATH}/VRMA_06.vrma`,
    label: 'Model pose',
    category: 'sustained_pose',
    loops: true,
    intensity: 'low',
    duration_ms: 4000,
    suitable_for: ['main', 'pip', 'solo'],
    notes: 'Display pose — useful as held shape for "presenting."',
  },

  // ─────────────────────────────────────────────────────────────────
  // Idle activities — character is doing a low-key something while
  // waiting. PIP's go-to during long silences.
  // ─────────────────────────────────────────────────────────────────
  smartphone: {
    url: `${ANIM_PATH}/005_smartphone.vrma`,
    label: 'Smartphone',
    category: 'idle',
    loops: true,
    intensity: 'low',
    duration_ms: 5000,
    suitable_for: ['main', 'pip', 'solo'],
    prop: {
      // Calibrated in scene-test 2026-04-30. Phone GLB parents to the
      // VRM's left-hand spot with this offset/rotation. The future
      // director must reuse the same calibration — don't reinvent.
      url: '/ui/lib/props/phone_3d_model.glb',
      intendedSize: 0.14,
      attachPhases: [
        { fromTime: 0, spot: 'leftPalm',
          offset:   [0.080, -0.015, 0.025],
          rotation: [-3.140, 1.210, -1.490] },
      ],
    },
    trim_start_ms: 5900,    // first 5.9s is the pickup motion — skip to where phone is already in hand
    notes: 'Phone scrolling. trimStart skips pickup; phone starts in hand. Prop must be bound to leftPalm with the calibrated offset.',
  },

  // ─────────────────────────────────────────────────────────────────
  // Greetings — one-shot, plays once, holds final frame. Wired to
  // "user joins call" / "speaker swap to this character" events.
  // ─────────────────────────────────────────────────────────────────
  hello_wave: {
    url: `${ANIM_PATH}/004_hello_1.vrma`,
    label: 'Hello / wave',
    category: 'greeting',
    loops: false,
    intensity: 'medium',
    duration_ms: 7000,
    suitable_for: ['main', 'pip', 'solo'],
    notes: 'Starts turned around, spins ~3.5s in, double-hand wave to user.',
  },
  wave_jump: {
    url: `${ANIM_PATH}/VRMA_02.vrma`,
    label: 'Wave (jump up)',
    category: 'greeting',
    loops: false,
    intensity: 'medium',
    duration_ms: 4000,
    trim_start_ms: 3000,
    suitable_for: ['main', 'pip', 'solo'],
    notes: 'Trimmed: skips the crouch-and-spring intro (jump removed); plays remaining wave.',
  },

  // ─────────────────────────────────────────────────────────────────
  // Reactive / display micro-content — short, expressive, plays once.
  // ─────────────────────────────────────────────────────────────────
  encourage: {
    url: `${ANIM_PATH}/007_gekirei.vrma`,
    label: 'Encouragement',
    category: 'reactive',
    loops: false,
    intensity: 'medium',
    duration_ms: 4000,
    suitable_for: ['main', 'pip', 'solo'],
    notes: 'Hands down symmetric, right hand up, button-pressing motion.',
  },
  show_full_body: {
    url: `${ANIM_PATH}/VRMA_01.vrma`,
    label: 'Show full body',
    category: 'display',
    loops: false,
    intensity: 'medium',
    duration_ms: 5000,
    suitable_for: ['main', 'solo'],
    notes: 'Sweeps horizontally to display full body — physical layout constraint: needs horizontal canvas space, would crop in PIP corner. This is a hardware not contextual restriction.',
  },
  peace_sign: {
    url: `${ANIM_PATH}/VRMA_03.vrma`,
    label: 'Peace sign',
    category: 'reactive',
    loops: false,
    intensity: 'low',
    duration_ms: 3000,
    suitable_for: ['main', 'pip', 'solo'],
    notes: 'Quick peace-sign gesture.',
  },
  shoot: {
    url: `${ANIM_PATH}/VRMA_04.vrma`,
    label: 'Shoot',
    category: 'reactive',
    loops: false,
    intensity: 'medium',
    duration_ms: 3000,
    suitable_for: ['main', 'pip', 'solo'],
    notes: 'Finger-gun / point gesture.',
  },
  spin: {
    url: `${ANIM_PATH}/VRMA_05.vrma`,
    label: 'Spin',
    category: 'display',
    loops: false,
    intensity: 'high',
    duration_ms: 4000,
    suitable_for: ['main', 'pip', 'solo'],
    endsCleanly: false,
    trim_end_ms: 800,         // last 0.8s flips upside down — glitch
    default_speed: 0.75,      // 25% slower — choreography reads better at this pace
    notes: 'Full body spin. trimEnd 0.8s skips end-flip glitch; defaultSpeed 0.75x reads better. Director should gate on "not currently TTS-speaking" — too distracting during active speech.',
  },

  // ─────────────────────────────────────────────────────────────────
  // Exercise / active idle — physical motion, fits "stretching /
  // restless / killing time" idle scenarios.
  // ─────────────────────────────────────────────────────────────────
  squat: {
    url: `${ANIM_PATH}/VRMA_07.vrma`,
    label: 'Squat',
    category: 'exercise',
    loops: true,
    intensity: 'medium',
    duration_ms: 3000,
    suitable_for: ['pip'],
    endsCleanly: false,
    notes: 'Squat down + back up. Vertical motion only.',
  },

  // ─────────────────────────────────────────────────────────────────
  // Emotion expression — short, mood-themed clips. Each maps cleanly
  // to one embodiment-engine mood axis. PRIMARY way the embodiment
  // system surfaces mood as visible motion. moodTrigger drives the
  // procedural picker's selection: when mood.<trigger> is high, this
  // clip's score boosts.
  //
  // First-pass categorization (2026-05-14 — fine-tune from review).
  // ─────────────────────────────────────────────────────────────────
  angry: {
    url: `${ANIM_PATH}/Angry.vrma`,
    label: 'Angry',
    category: 'emotion_expression',
    loops: false,
    intensity: 'high',
    duration_ms: 3500,
    suitable_for: ['main', 'pip', 'solo'],
    moodTrigger: 'angry',
    endsCleanly: true,
    notes: 'Mood-driven gesture. Fires when mood.angry high. Minimal freeze: anger lives in head shake + body tension.',
  },
  sad: {
    url: `${ANIM_PATH}/Sad.vrma`,
    label: 'Sad',
    category: 'emotion_expression',
    loops: false,
    intensity: 'low',
    duration_ms: 4000,
    suitable_for: ['main', 'pip', 'solo'],
    moodTrigger: 'sad',
    endsCleanly: true,
    notes: 'Mood-driven gesture. Fires when mood.sad high. Minimal freeze: sadness lives in head-down + slumped spine.',
  },
  surprised: {
    url: `${ANIM_PATH}/Surprised.vrma`,
    label: 'Surprised',
    category: 'emotion_expression',
    loops: false,
    intensity: 'high',
    duration_ms: 2500,
    suitable_for: ['main', 'pip', 'solo'],
    moodTrigger: 'surprised',
    endsCleanly: true,
    notes: 'Mood-driven spike gesture. Fires on high surprise intensity. Minimal freeze: surprise needs head jerk + spine react.',
  },
  blush: {
    url: `${ANIM_PATH}/Blush.vrma`,
    label: 'Blush',
    category: 'emotion_expression',
    loops: false,
    intensity: 'low',
    duration_ms: 3000,
    suitable_for: ['main', 'pip', 'solo'],
    moodTrigger: 'shy',
    moodAffinity: { happy: 0.4, surprised: 0.3 },
    endsCleanly: true,
    notes: 'Shy/embarrassed gesture. Compose with happy mood. Minimal freeze: head-down + averted gaze are the gesture.',
  },
  thinking: {
    url: `${ANIM_PATH}/Thinking.vrma`,
    label: 'Thinking',
    category: 'sustained_pose',
    loops: false,
    intensity: 'low',
    duration_ms: 4500,
    suitable_for: ['main', 'pip', 'solo'],
    moodTrigger: 'contemplative',
    moodAffinity: { intensity: 0.3 },
    endsCleanly: true,
    notes: 'Hand-to-chin contemplative gesture. Use when speakingState is "thinking" or for deliberation pauses. Minimal freeze: head tilt is part of the read.',
  },
  sleepy: {
    url: `${ANIM_PATH}/Sleepy.vrma`,
    label: 'Sleepy',
    category: 'emotion_expression',
    loops: false,
    intensity: 'low',
    duration_ms: 4000,
    suitable_for: ['main', 'pip', 'solo'],
    moodTrigger: 'tired',
    moodAffinity: { intensity: -0.5 },
    endsCleanly: true,
    notes: 'Low-energy yawn/stretch. Use during long idle windows. Minimal freeze: yawn IS spine + head + jaw.',
  },
  relax: {
    url: `${ANIM_PATH}/Relax.vrma`,
    label: 'Relax',
    category: 'sustained_pose',
    loops: true,
    intensity: 'low',
    duration_ms: 5000,
    suitable_for: ['main', 'pip', 'solo'],
    moodTrigger: 'calm',
    moodAffinity: { intensity: -0.3, happy: 0.2 },
    endsCleanly: true,
    notes: 'Calm relaxed pose. Compose with low-intensity moods.',
  },
  look_around: {
    url: `${ANIM_PATH}/LookAround.vrma`,
    label: 'Look around',
    category: 'idle',
    loops: false,
    intensity: 'low',
    duration_ms: 3500,
    suitable_for: ['main', 'pip', 'solo'],
    moodTrigger: 'curious',
    endsCleanly: true,
    notes: 'Head/eye saccade pattern. Use during user-silent windows or scene change. Minimal freeze: head turn IS the gesture.',
  },
  clapping: {
    url: `${ANIM_PATH}/Clapping.vrma`,
    label: 'Clapping',
    category: 'reactive',
    loops: false,
    intensity: 'medium',
    duration_ms: 3000,
    suitable_for: ['main', 'pip', 'solo'],
    moodTrigger: 'happy',
    moodAffinity: { intensity: 0.5 },
    endsCleanly: true,
    notes: 'Applause / celebration. Fire on positive user events.',
  },
  jump: {
    url: `${ANIM_PATH}/Jump.vrma`,
    label: 'Jump',
    category: 'reactive',
    loops: false,
    intensity: 'high',
    duration_ms: 2500,
    suitable_for: ['main', 'pip', 'solo'],
    moodTrigger: 'excited',
    moodAffinity: { happy: 0.7, intensity: 0.9 },
    endsCleanly: false,
    notes: 'High excitement jump. Use on big positive events. Leaves the avatar slightly offset (endsCleanly=false).',
  },
  goodbye: {
    url: `${ANIM_PATH}/Goodbye.vrma`,
    label: 'Goodbye',
    category: 'greeting',
    loops: false,
    intensity: 'medium',
    duration_ms: 3500,
    suitable_for: ['main', 'pip', 'solo'],
    moodTrigger: null,
    endsCleanly: true,
    notes: 'Farewell wave. Wired to user departure events. Minimal freeze: wave needs body lean.',
  },
};

/** Stable order array — useful for ordered iteration (e.g. picking a
 *  random idle from a category, with weighted selection). */
export const VRMA_KEYS = Object.keys(VRMA_LIBRARY);

/** Files in `ui/lib/animations/` that are explicitly NOT approved.
 *  Listed here so the audit history doesn't get lost — re-evaluating
 *  them later (e.g. when an environment scene exists) starts here. */
export const VRMA_DEFERRED = {
  '002_dogeza.vrma':       'Floor pose. Looks broken in the void — needs a floor environment.',
  '003_humidai.vrma':      'Step-up motion lifts the foot into empty air — reads as jarring without a step/box prop or environment to stand on.',
  '006_drinkwater.vrma':   'Needs water-bottle prop AND environment (table to set down on).',
  '008_gatan.vrma':        'Sitting → standing → sitting transition. Needs a chair to sit on.',
  'Reading (Loop).vrma':   'Sustained sitting pose for VTuber-style streams. Needs chair + book.',
  'CC0-walk.vrma':         'Locomotion — not used in sit-down voice calls.',
  'CC0-slowrun.vrma':      'Locomotion — not used in sit-down voice calls.',
  'CC0-run.vrma':          'Locomotion — not used in sit-down voice calls.',
};

/** Quick category lookup — `byCategory.dance` returns the subset of
 *  approved entries with that category. Iterating a single bucket is
 *  the common access pattern (e.g. "pick a random dance for PIP"). */
export const byCategory = (() => {
  const buckets = {};
  for (const [key, entry] of Object.entries(VRMA_LIBRARY)) {
    if (!buckets[entry.category]) buckets[entry.category] = [];
    buckets[entry.category].push({ key, ...entry });
  }
  return buckets;
})();

/** Filter entries by role suitability. Pass 'main' / 'pip' / 'solo'. */
export function vrmaSuitableFor(role) {
  return Object.entries(VRMA_LIBRARY)
    .filter(([, entry]) => entry.suitable_for.includes(role))
    .map(([key, entry]) => ({ key, ...entry }));
}

/**
 * Derive a default mood-affinity vector from a VRMA's category +
 * intensity when it doesn't declare one explicitly. The procedural
 * picker uses this to score VRMAs against current embodiment mood.
 *
 * @param {object} entry  a VRMA_LIBRARY entry
 * @returns {object}      { happy, sad, angry, surprised, intensity }
 */
export function inferMoodAffinity(entry) {
  if (entry.moodAffinity) return entry.moodAffinity;
  const out = { happy: 0, sad: 0, angry: 0, surprised: 0, intensity: 0 };
  const intensityWeight = entry.intensity === 'high' ? 0.8
                        : entry.intensity === 'medium' ? 0.5 : 0.2;
  switch (entry.category) {
    case 'dance':            out.happy = 0.9; out.intensity = intensityWeight; break;
    case 'sustained_pose':   out.intensity = -0.2; break;
    case 'idle':             out.intensity = -0.3; break;
    case 'greeting':         out.happy = 0.4; out.intensity = 0.3; break;
    case 'reactive':         out.intensity = intensityWeight; break;
    case 'display':          out.intensity = intensityWeight * 0.6; break;
    case 'exercise':         out.intensity = intensityWeight; break;
    case 'emotion_expression':
      if (entry.moodTrigger && entry.moodTrigger in out) out[entry.moodTrigger] = 1.0;
      out.intensity = intensityWeight;
      break;
  }
  return out;
}

/**
 * Score a VRMA against current mood + recency cache. Higher = better
 * pick. Negative is allowed (deprioritized).
 *
 * @param {object} entry      a VRMA_LIBRARY entry
 * @param {object} mood       { happy, sad, angry, surprised, intensity } 0..1
 * @param {string[]} recent   recent VRMA keys for recency penalty
 */
export function scoreVRMA(entry, mood, recent = []) {
  const affinity = inferMoodAffinity(entry);
  let score = 0;
  for (const [emotion, weight] of Object.entries(affinity)) {
    score += (mood[emotion] ?? 0) * weight;
  }
  // Baseline so VRMAs without strong mood match still get sampled occasionally
  score += 0.15;
  // Recency penalty
  const recentIdx = recent.indexOf(entry.key || entry.label);
  if (recentIdx >= 0) score *= 0.25;
  return score;
}
