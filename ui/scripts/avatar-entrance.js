/**
 * avatar-entrance.js — the "First Breath" entrance choreographer.
 *
 * The avatar's living idle (breathing, gaze, blink, sway) is rich, but her
 * FIRST impression was flat: she snapped straight into idle with no arrival,
 * no acknowledgement that *you* showed up. This plays one short choreographed
 * beat the moment she first appears, then dissolves into the idle she already
 * runs:
 *
 *   Beat 1 — Arrival (t=0): a grounding inhale + a warm emotion bloom.
 *   Beat 2 — Recognition (t≈MICRO_MS): a micro head-tilt double-take.
 *   Beat 3 — Greeting (t≈GREETING_MS): a wave (rare, high-impact) or a gentle
 *            open-palms (frequent reopen), keyed to context.
 *   Settle: the existing breathing/sway idle takes back over; camera reverts.
 *
 * Escalating warmth so it never wears out: a big wave is reserved for the
 * first-ever visit and long-absence returns; frequent reopens get only the
 * gentle bloom; and a cooldown means a page refresh doesn't re-trigger it.
 *
 * Pure decision logic lives in selectEntrance() (unit-tested); playEntrance()
 * executes the plan against the live avatar systems, with every call guarded so
 * a missing subsystem (or any throw) can never break avatar activation.
 *
 * No imports — same dependency-free style as body-atlas.js.
 */

// Tuning. Beat timings are deliberately short; the whole thing is < 1s of
// explicit motion before the idle reclaims her.
export const ENTRANCE_COOLDOWN_MS = 20 * 60 * 1000;   // within this since last entrance → skip entirely
export const RETURN_GAP_MS        = 10 * 60 * 60 * 1000; // gap longer than this → "welcome back"
export const WARMTH_MS            = 5000;             // how long the warm emotion override holds
export const MICRO_MS             = 480;              // recognition head-tilt
export const GREETING_MS          = 700;              // the wave / open-palms beat

const LAST_SEEN_KEY     = 'augmentum.avatar.lastSeen';
const LAST_ENTRANCE_KEY = 'augmentum.avatar.lastEntrance';

/**
 * Decide what (if anything) the entrance should do, purely from context.
 *
 * @param {object} ctx
 * @param {number} ctx.now                 current epoch ms
 * @param {number|null} ctx.lastSeenMs     epoch ms she was last seen (null = first ever)
 * @param {number|null} ctx.lastEntranceMs epoch ms the last entrance played (null = never)
 * @param {number} [ctx.cooldownMs]        override ENTRANCE_COOLDOWN_MS (tests)
 * @param {number} [ctx.returnGapMs]       override RETURN_GAP_MS (tests)
 * @param {boolean} [ctx.canFullBody=false] true when full-body framing is
 *        available (an adaptive camera that can frame the cinematic VRMA wave);
 *        false → portrait-safe procedural wave that reads in head/shoulders.
 * @param {boolean} [ctx.enabled=true]     master switch
 * @returns {object|null} a plan, or null to skip the entrance entirely
 */
export function selectEntrance(ctx) {
  const {
    now,
    lastSeenMs = null,
    lastEntranceMs = null,
    cooldownMs = ENTRANCE_COOLDOWN_MS,
    returnGapMs = RETURN_GAP_MS,
    canFullBody = false,
    enabled = true,
  } = ctx || {};

  if (!enabled || typeof now !== 'number') return null;
  // Refresh / re-mount guard: don't replay a full entrance right after one.
  if (lastEntranceMs != null && (now - lastEntranceMs) < cooldownMs) return null;

  const hour = new Date(now).getHours();
  const firstTime = lastSeenMs == null;
  const gapMs = firstTime ? Infinity : Math.max(0, now - lastSeenMs);
  const longGap = gapMs >= returnGapMs;
  const evening = hour >= 20 || hour < 6;          // wind-down hours → calmer
  const morning = hour >= 5 && hour < 12;

  // Big, high-impact greeting is RARE: first sighting or a real absence.
  // Frequent same-session reopens get a gentle, warm acknowledgement only.
  if (firstTime || longGap) {
    // Cinematic full-body VRMA wave when we can frame it; otherwise a procedural
    // upper-body wave that reads cleanly in a portrait/head-shoulders crop.
    const greeting = canFullBody ? { kind: 'vrma', id: 'hello' } : { kind: 'gesture', name: 'wave' };
    return {
      flavor: firstTime ? 'first-time' : 'welcome-back',
      emotion: 'happy',
      warmthMs: WARMTH_MS,
      micro: 'head_tilt',
      greeting,
      camera: canFullBody ? 'fullBody' : null,   // frame the wave (reverts itself)
    };
  }
  return {
    flavor: evening ? 'fresh-evening' : (morning ? 'fresh-morning' : 'fresh-day'),
    emotion: evening ? 'relaxed' : 'curious',
    warmthMs: WARMTH_MS,
    micro: 'head_tilt',
    greeting: { kind: 'gesture', name: 'nod' },  // gentle, arm-neutral (no arm raise)
    camera: null,                                       // keep current framing
  };
}

/**
 * Play the entrance. Always stamps "last seen"; stamps "last entrance" only when
 * a beat actually plays. Returns a controller with cancel() so the caller can
 * abbreviate the beat if the user immediately engages (starts talking/typing).
 *
 * @param {object} deps
 * @param {object} [deps.animator]      AvatarAnimator (triggerInhale, setEmotion)
 * @param {object} [deps.presence]      PresenceEngine (setEmotionOverride, onExplicitGesture)
 * @param {object} [deps.conductor]     MovementConductor (playById, currentName)
 * @param {object} [deps.adaptiveCamera] adaptive camera (setPreset)
 * @param {object} [deps.storage]       { get(k), set(k,v) } — defaults to localStorage
 * @param {Function} [deps.setTimeout]  injectable for tests
 * @param {Function} [deps.clearTimeout]
 * @param {object} ctx                  passed to selectEntrance (minus the stored stamps)
 * @returns {{ plan: object|null, cancel: Function }}
 */
export function playEntrance(deps = {}, ctx = {}) {
  const storage = deps.storage || _defaultStorage();
  const setT = deps.setTimeout || ((typeof setTimeout !== 'undefined') ? setTimeout : null);
  const clearT = deps.clearTimeout || ((typeof clearTimeout !== 'undefined') ? clearTimeout : null);

  const lastSeenMs = _readNum(storage, LAST_SEEN_KEY);
  const lastEntranceMs = _readNum(storage, LAST_ENTRANCE_KEY);
  const plan = selectEntrance({ ...ctx, lastSeenMs, lastEntranceMs });

  // Always remember we saw the user now (so the NEXT visit can read the gap),
  // even when the entrance is skipped by cooldown.
  _write(storage, LAST_SEEN_KEY, String(ctx.now));
  if (plan) _write(storage, LAST_ENTRANCE_KEY, String(ctx.now));

  if (!plan) return { plan: null, cancel() {} };

  const { animator, presence, conductor, adaptiveCamera } = deps;
  const timers = [];
  let cancelled = false;
  const safe = (fn) => { try { fn(); } catch { /* entrance must never break activation */ } };
  const after = (ms, fn) => {
    if (!setT) { safe(fn); return; }
    timers.push(setT(() => { if (!cancelled) safe(fn); }, ms));
  };

  // Beat 1 — Arrival: grounding breath + warm bloom (immediate).
  safe(() => animator?.triggerInhale?.());
  safe(() => animator?.setEmotion?.(plan.emotion));
  safe(() => presence?.setEmotionOverride?.(plan.emotion, plan.warmthMs));
  if (plan.camera) safe(() => adaptiveCamera?.setPreset?.(plan.camera));

  // Beat 2 — Recognition: a micro head-tilt double-take.
  if (plan.micro) after(MICRO_MS, () => presence?.onExplicitGesture?.(plan.micro));

  // Beat 3 — Greeting: wave (don't stomp a VRMA already playing) or gentle palms.
  after(GREETING_MS, () => {
    if (plan.greeting.kind === 'vrma') {
      if (!conductor?.currentName?.()) {
        conductor?.playById?.(plan.greeting.id, { explicit: true });
      }
    } else {
      presence?.onExplicitGesture?.(plan.greeting.name);
    }
  });

  return {
    plan,
    cancel() {
      if (cancelled) return;
      cancelled = true;
      if (clearT) for (const id of timers) clearT(id);
    },
  };
}

// ─── internals ──────────────────────────────────────────────────────────────
function _defaultStorage() {
  try {
    if (typeof localStorage !== 'undefined') {
      return { get: (k) => localStorage.getItem(k), set: (k, v) => localStorage.setItem(k, v) };
    }
  } catch { /* storage blocked (private mode) — fall through to a no-op */ }
  return { get: () => null, set: () => {} };
}
function _readNum(storage, key) {
  try {
    const v = storage.get(key);
    if (v == null) return null;
    const n = parseInt(v, 10);
    return Number.isFinite(n) ? n : null;
  } catch { return null; }
}
function _write(storage, key, value) {
  try { storage.set(key, value); } catch { /* ignore */ }
}
