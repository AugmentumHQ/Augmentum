/**
 * Node regression for the "First Breath" entrance choreographer.
 *
 * Run by hand:  node tests/js/test_avatar_entrance.mjs
 *
 * Covers the pure decision logic (escalating warmth, cooldown, time-of-day,
 * returning-user) and the playEntrance gating/stamping/cancel contract — the
 * choreography TIMING itself is tuned on-device, but the WHAT/WHEN-to-fire and
 * the never-break-activation guards are locked here.
 */

let pass = 0, fail = 0;
function ok(cond, msg) { if (cond) pass++; else { fail++; console.error(`FAIL: ${msg}`); } }
function eq(a, b, msg) {
  if (JSON.stringify(a) === JSON.stringify(b)) pass++;
  else { fail++; console.error(`FAIL: ${msg}\n   expected ${JSON.stringify(b)}\n   got      ${JSON.stringify(a)}`); }
}

const url = new URL('../../ui/scripts/avatar-entrance.js', import.meta.url);
const m = await import(url);
const { selectEntrance, playEntrance, ENTRANCE_COOLDOWN_MS, RETURN_GAP_MS } = m;

// A fixed clock helper: build epoch ms for a given local hour today.
function atHour(h) { const d = new Date(2026, 5, 20, h, 0, 0); return d.getTime(); }

// ── selectEntrance: first-ever visit, full-body framing → cinematic VRMA wave.
{
  const p = selectEntrance({ now: atHour(14), lastSeenMs: null, lastEntranceMs: null, canFullBody: true });
  ok(p && p.flavor === 'first-time', 'first ever → first-time flavor');
  eq(p.greeting, { kind: 'vrma', id: 'hello' }, 'first time + full-body → VRMA hello');
  ok(p.camera === 'fullBody', 'first time frames full body for the wave');
  ok(p.emotion === 'happy', 'first time is happy');
}

// ── First-ever, PORTRAIT framing → procedural upper-body wave (no camera move).
{
  const p = selectEntrance({ now: atHour(14), lastSeenMs: null, lastEntranceMs: null, canFullBody: false });
  ok(p && p.flavor === 'first-time', 'first ever (portrait) → first-time');
  eq(p.greeting, { kind: 'gesture', name: 'wave' }, 'portrait first time → procedural wave');
  ok(p.camera === null, 'portrait wave does not move the camera');
}

// ── Returning after a long absence → welcome-back wave. ─────────────────────
{
  const now = atHour(14);
  const p = selectEntrance({ now, lastSeenMs: now - (RETURN_GAP_MS + 1000), lastEntranceMs: now - (RETURN_GAP_MS + 1000), canFullBody: true });
  ok(p && p.flavor === 'welcome-back', 'long gap → welcome-back');
  eq(p.greeting, { kind: 'vrma', id: 'hello' }, 'welcome-back + full-body → VRMA wave');
}

// ── Frequent reopen (short gap, daytime) → gentle bloom, NO big wave. ────────
{
  const now = atHour(14);
  // 40 min since last seen (> cooldown so not skipped), but same session-ish.
  const p = selectEntrance({ now, lastSeenMs: now - 40 * 60 * 1000, lastEntranceMs: now - 40 * 60 * 1000 });
  ok(p && p.flavor === 'fresh-day', 'short daytime gap → fresh-day');
  ok(p.greeting.kind === 'gesture' && p.greeting.name === 'nod', 'fresh reopen is a gentle arm-neutral nod, not a wave');
  ok(p.camera === null, 'gentle greeting keeps the current framing');
  ok(p.emotion === 'curious', 'daytime fresh is curious');
}

// ── Evening reopen → calmer (relaxed). ──────────────────────────────────────
{
  const now = atHour(22);
  const p = selectEntrance({ now, lastSeenMs: now - 40 * 60 * 1000, lastEntranceMs: now - 40 * 60 * 1000 });
  ok(p && p.flavor === 'fresh-evening', 'evening → fresh-evening');
  ok(p.emotion === 'relaxed', 'evening is relaxed');
  ok(p.greeting.kind === 'gesture', 'evening greeting stays gentle');
}

// ── Cooldown: just greeted → skip entirely. ─────────────────────────────────
{
  const now = atHour(14);
  const p = selectEntrance({ now, lastSeenMs: now - 1000, lastEntranceMs: now - 1000 });
  ok(p === null, 'within cooldown since last entrance → no entrance');
}

// ── Master switch off → skip. ───────────────────────────────────────────────
{
  ok(selectEntrance({ now: atHour(14), lastSeenMs: null, enabled: false }) === null, 'enabled:false → null');
}

// ── playEntrance: stamps storage, fires beats, honors cooldown. ─────────────
function fakeStorage(initial = {}) {
  const map = new Map(Object.entries(initial));
  return { map, get: (k) => (map.has(k) ? map.get(k) : null), set: (k, v) => map.set(k, v) };
}
function fakeClock() {
  let seq = 1; const jobs = new Map();
  return {
    setTimeout: (fn, ms) => { const id = seq++; jobs.set(id, { fn, ms }); return id; },
    clearTimeout: (id) => jobs.delete(id),
    run: () => { for (const { fn } of [...jobs.values()]) fn(); },
    pending: () => jobs.size,
  };
}
function recordingDeps() {
  const calls = [];
  return {
    calls,
    animator: { triggerInhale: () => calls.push('inhale'), setEmotion: (e) => calls.push(`setEmotion:${e}`) },
    presence: {
      setEmotionOverride: (e, ms) => calls.push(`override:${e}:${ms}`),
      onExplicitGesture: (g) => calls.push(`gesture:${g}`),
    },
    conductor: { currentName: () => null, playById: (id, o) => calls.push(`vrma:${id}:${o && o.explicit}`) },
    adaptiveCamera: { setPreset: (p) => calls.push(`camera:${p}`) },
  };
}

// First-time play → bloom immediately, micro + wave after timers, storage stamped.
{
  const st = fakeStorage();
  const clk = fakeClock();
  const d = recordingDeps();
  const now = atHour(14);
  const ctrl = playEntrance({ ...d, storage: st, setTimeout: clk.setTimeout, clearTimeout: clk.clearTimeout }, { now, canFullBody: true });
  ok(ctrl.plan && ctrl.plan.flavor === 'first-time', 'play returns the first-time plan');
  // Beat 1 fired synchronously:
  ok(d.calls.includes('inhale'), 'beat1: inhale fired immediately');
  ok(d.calls.includes('setEmotion:happy'), 'beat1: emotion set immediately');
  ok(d.calls.some((c) => c.startsWith('override:happy:')), 'beat1: presence override immediately');
  ok(d.calls.includes('camera:fullBody'), 'beat1: camera framed immediately');
  ok(!d.calls.includes('vrma:hello:true'), 'wave has NOT fired before timers run');
  clk.run();
  ok(d.calls.includes('gesture:head_tilt'), 'beat2: recognition head-tilt fired on timer');
  ok(d.calls.includes('vrma:hello:true'), 'beat3: wave fired explicit on timer');
  ok(st.get('augmentum.avatar.lastSeen') === String(now), 'lastSeen stamped');
  ok(st.get('augmentum.avatar.lastEntrance') === String(now), 'lastEntrance stamped');
}

// Cooldown play → no beats, but lastSeen still advances (so the gap stays honest).
{
  const now = atHour(14);
  const st = fakeStorage({
    'augmentum.avatar.lastSeen': String(now - 1000),
    'augmentum.avatar.lastEntrance': String(now - 1000),
  });
  const clk = fakeClock();
  const d = recordingDeps();
  const ctrl = playEntrance({ ...d, storage: st, setTimeout: clk.setTimeout, clearTimeout: clk.clearTimeout }, { now });
  ok(ctrl.plan === null, 'cooldown → no plan');
  ok(d.calls.length === 0, 'cooldown → zero motion calls');
  ok(st.get('augmentum.avatar.lastSeen') === String(now), 'lastSeen still advances under cooldown');
  ok(st.get('augmentum.avatar.lastEntrance') === String(now - 1000), 'lastEntrance NOT advanced under cooldown');
}

// cancel() before timers → wave is abbreviated away (only the bloom happened).
{
  const st = fakeStorage();
  const clk = fakeClock();
  const d = recordingDeps();
  const ctrl = playEntrance({ ...d, storage: st, setTimeout: clk.setTimeout, clearTimeout: clk.clearTimeout }, { now: atHour(14) });
  ctrl.cancel();
  clk.run();
  ok(!d.calls.includes('vrma:hello:true'), 'cancel() abbreviates the wave');
  ok(!d.calls.includes('gesture:head_tilt'), 'cancel() abbreviates the micro-gesture');
  ok(d.calls.includes('setEmotion:happy'), 'the immediate warm bloom still happened');
  ok(clk.pending() === 0, 'cancel() cleared pending timers');
}

// Guard: a throwing subsystem never propagates out of playEntrance.
{
  const st = fakeStorage();
  const d = {
    animator: { triggerInhale: () => { throw new Error('boom'); }, setEmotion: () => { throw new Error('boom'); } },
    presence: { setEmotionOverride: () => { throw new Error('boom'); }, onExplicitGesture: () => { throw new Error('boom'); } },
    conductor: { currentName: () => { throw new Error('boom'); }, playById: () => { throw new Error('boom'); } },
    adaptiveCamera: { setPreset: () => { throw new Error('boom'); } },
    storage: st,
  };
  let threw = false;
  try { const c = playEntrance(d, { now: atHour(14) }); c.cancel(); } catch { threw = true; }
  ok(!threw, 'a throwing subsystem never breaks activation');
}

// Missing subsystems entirely (deps absent) → no throw, storage still stamped.
{
  const st = fakeStorage();
  let threw = false;
  try { playEntrance({ storage: st, setTimeout: (f) => f(), clearTimeout: () => {} }, { now: atHour(14) }); }
  catch { threw = true; }
  ok(!threw, 'absent subsystems are tolerated');
  ok(st.get('augmentum.avatar.lastSeen') !== null, 'lastSeen stamped even with no subsystems');
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
