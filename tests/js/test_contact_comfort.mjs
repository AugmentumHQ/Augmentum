/**
 * Node regression for comfort-gated contact (touchability-aware reach).
 *
 * Run by hand:  node tests/js/test_contact_comfort.mjs
 *
 * Locks in the contract: the body atlas's per-region touchability prior
 * modulates how the avatar meets a user hand — guarded zones (low touchability:
 * eyes/mouth/neck) keep MORE distance and reach more hesitantly; welcoming zones
 * (high: cheek/chest/shoulder) lean in; and with no atlas / gating off the
 * reach is byte-for-byte the legacy behavior (STOP_SHORT_M = 8cm).
 */

let pass = 0, fail = 0;
function ok(cond, msg) { if (cond) pass++; else { fail++; console.error(`FAIL: ${msg}`); } }
function close(a, b, eps, msg) {
  if (Math.abs(a - b) <= eps) pass++;
  else { fail++; console.error(`FAIL: ${msg}\n   |${a} - ${b}| = ${Math.abs(a - b)} > ${eps}`); }
}

// ── Minimal THREE.Vector3 (only what ContactReactor touches). ───────────────
class Vector3 {
  constructor(x = 0, y = 0, z = 0) { this.x = x; this.y = y; this.z = z; }
  set(x, y, z) { this.x = x; this.y = y; this.z = z; return this; }
}
const THREE = { Vector3 };

// Atlas stub: just the two fields the comfort map reads. Subset of the real
// regionTable + touchabilityDefaults (values are the authored bake defaults).
const atlasStub = {
  regionTable:        ['cheek_L', 'shoulder_L', 'neck', 'hand_L', 'other'],
  touchabilityDefaults: [240,        180,          40,     80,       40],   // /255
};

// VRM stub: hands fixed in world; hips present for intent's hips proxy.
function makeVrm(handPos = [1, 1, 0]) {
  const node = (pos) => ({ getWorldPosition: (v) => v.set(pos[0], pos[1], pos[2]) });
  const bones = {
    leftHand: node(handPos), rightHand: node([2, 1, 0]),
    hips: node([0, 0.9, 0]),
  };
  return { humanoid: { getNormalizedBoneNode: (n) => bones[n] || null } };
}

// bodyMesh stub: returns a chosen region + distance for the user hand.
function makeMesh(region, distance, point = [0.5, 1, 0]) {
  return { closestPoint: () => ({ region, distance, point }) };
}

// IK stub: captures the last world target so we can measure the stop-short gap.
function makeIk() {
  return { last: null, setHandPositionWorld(side, p) { this.last = [...p]; } };
}

const url = new URL('../../ui/scripts/contact-reactor.js', import.meta.url);
const { ContactReactor } = await import(url);

// Drive a steady reach in the HOVER state until the smoothed target settles,
// then return the residual gap = ‖userHand - avatarTarget‖.
function settledGap({ region, comfortGating = true, atlas = atlasStub }) {
  const userHand = [1, 1, 0.6];        // 0.6 m straight out from the avatar hand at [1,1,0]
  const ik = makeIk();
  const r = new ContactReactor({
    three: THREE,
    vrm: makeVrm([1, 1, 0]),
    bodyMesh: makeMesh(region, 0.10),  // 0.10 m → 'hover' band → reach engages
    bodyAtlas: atlas,
    ik,
    velocityAware: false,
    comfortGating,
  });
  r.setUserHand('R', userHand);
  for (let i = 0; i < 600; i++) r.tick(16);   // ~10 s of 60fps reach → fully settled
  ok(ik.last !== null, `[${region}] reach produced an IK target`);
  const d = ik.last ? Math.hypot(userHand[0] - ik.last[0], userHand[1] - ik.last[1], userHand[2] - ik.last[2]) : NaN;
  return d;
}

const STOP_SHORT = 0.08;

// ── 1. Legacy: gating off → exactly STOP_SHORT_M regardless of region. ───────
{
  const g = settledGap({ region: 'neck', comfortGating: false });
  close(g, STOP_SHORT, 1e-3, 'gating off → legacy 8cm gap');
}

// ── 2. No atlas → legacy gap (back-compat for user VRMs). ────────────────────
{
  const g = settledGap({ region: 'neck', atlas: null });
  close(g, STOP_SHORT, 1e-3, 'no atlas → legacy 8cm gap');
}

// ── 3. Neutral zone (shoulder, c=0.706, between pivots) → unchanged 8cm. ─────
{
  const g = settledGap({ region: 'shoulder_L' });
  close(g, STOP_SHORT, 1e-3, 'neutral zone keeps the base gap');
}

// ── 4. Guarded zone (neck, c=0.157 < 0.40) → wider gap (keeps distance). ─────
{
  const g = settledGap({ region: 'neck' });
  // expected: 0.08 + ((0.40-0.157)/0.40)*0.22 = 0.08 + 0.1337 = 0.2137 m
  close(g, 0.08 + ((0.40 - 40 / 255) / 0.40) * 0.22, 2e-3, 'guarded neck → ~21cm standoff');
  ok(g > STOP_SHORT + 0.05, 'guarded zone keeps materially more distance than legacy');
}

// ── 5. Welcoming zone (cheek, c=0.941 > 0.75) → narrower gap (leans in). ─────
{
  const g = settledGap({ region: 'cheek_L' });
  // expected: 0.08 * (1 - ((0.941-0.75)/0.25)*0.5) = 0.08 * (1 - 0.3835) = 0.0493 m
  close(g, STOP_SHORT * (1 - ((240 / 255 - 0.75) / 0.25) * 0.5), 2e-3, 'welcoming cheek → leans in (~5cm)');
  ok(g < STOP_SHORT - 0.01, 'welcoming zone closes nearer than legacy');
}

// ── 6. Monotonic: guarded > neutral > welcome. ───────────────────────────────
{
  const guarded = settledGap({ region: 'neck' });
  const neutral = settledGap({ region: 'shoulder_L' });
  const welcome = settledGap({ region: 'cheek_L' });
  ok(guarded > neutral && neutral > welcome, `gap is monotonic by comfort (${guarded.toFixed(3)} > ${neutral.toFixed(3)} > ${welcome.toFixed(3)})`);
}

// ── 7. Contact payload carries comfort + comfortBand to embodiment. ──────────
{
  const events = [];
  const r = new ContactReactor({
    three: THREE,
    vrm: makeVrm([1, 1, 0]),
    bodyMesh: makeMesh('neck', 0.01),   // 0.01 m → 'contact'
    bodyAtlas: atlasStub,
    embodiment: { onContactEvent: (e) => events.push(e) },
    velocityAware: false,
  });
  r.setUserHand('R', [1, 1, 0.01]);
  r.tick(16);
  ok(events.length >= 1, 'contact fired an embodiment event');
  const e = events[0];
  close(e.comfort, 40 / 255, 1e-6, 'payload.comfort == neck touchability default');
  ok(e.comfortBand === 'guarded', "payload.comfortBand == 'guarded' for neck");
  ok(e.region === 'neck', 'payload still carries region');
}

// ── 8. Unknown region (not in table) → null comfort, legacy reach. ───────────
{
  const g = settledGap({ region: 'kneecap_definitely_not_a_region' });
  close(g, STOP_SHORT, 1e-3, 'unknown region → legacy 8cm gap');
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
