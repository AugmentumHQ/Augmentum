/**
 * Node regression for BodyFrame — the world↔bake similarity transform that
 * keeps body-atlas collision/compliance/IK correct when the avatar is rotated
 * (vrm.scene.rotation.y) or uniformly scaled (wrapper.scale.setScalar).
 *
 * Run by hand:  node tests/js/test_body_atlas_frame.mjs
 *
 * The contract under test: a query point at a FIXED location on the body must
 * report the SAME signed distance (up to runtime scale) no matter how the
 * avatar is posed. That's exactly the property the raw `atlas.sdf([worldXYZ])`
 * calls were silently violating once she turned.
 */

let pass = 0, fail = 0;
function ok(cond, msg) {
  if (cond) { pass++; } else { fail++; console.error(`FAIL: ${msg}`); }
}
function close(a, b, eps, msg) {
  const d = Math.abs(a - b);
  if (d <= eps) { pass++; } else { fail++; console.error(`FAIL: ${msg}\n   |${a} - ${b}| = ${d} > ${eps}`); }
}
function vclose(a, b, eps, msg) {
  const d = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  if (d <= eps) { pass++; } else { fail++; console.error(`FAIL: ${msg}\n   ‖${JSON.stringify(a)} - ${JSON.stringify(b)}‖ = ${d} > ${eps}`); }
}

const url = new URL('../../ui/scripts/body-atlas.js', import.meta.url);
const { BodyAtlas, BodyFrame } = await import(url);

// ── Build a tiny synthetic atlas. sdf(p) ≈ p.x (a planar ramp), so we can
//    reason about expected distances analytically. 10³ grid spanning [-0.5,0.5].
function makeAtlas({ hips } = {}) {
  const dim = 10, h = 0.1, origin = [-0.5, -0.5, -0.5];
  const total = dim * dim * dim;
  const indices = [], sdf = [], region = [], touchability = [], flags = [], normal = [];
  for (let k = 0; k < dim; k++) for (let j = 0; j < dim; j++) for (let i = 0; i < dim; i++) {
    const idx = (k * dim + j) * dim + i;
    const x = origin[0] + (i + 0.5) * h;
    indices.push(idx);
    sdf.push(x);                 // planar ramp along world/bake x
    region.push(0);
    touchability.push(128);
    flags.push(0x08);            // mark all as surface-band so surfacePoints() returns some
    normal.push(1, 0, 0);        // +x outward
  }
  const data = {
    schema: 'augmentum.body-atlas.v1',
    regionTable: ['torso'],
    touchabilityDefaults: {},
    bbox: { dims: [dim, dim, dim], voxelSize: h, origin },
    skeletonHeight: 1.6,
    voxels: { indices, sdf, region, touchability, flags, normal },
    anchors: [{ point: [0.1, 0.0, 0.0], region: 'torso', curvature: 0.5 }],
  };
  if (hips) data.hips = hips;
  return new BodyAtlas(data);
}

// Bake hips at a non-origin pose with a 90° yaw, to make the transform non-trivial.
const SQRT1_2 = Math.SQRT1_2;
const bakeHips = { worldPos: [0, 0.9, 0.0], worldQuat: [0, SQRT1_2, 0, SQRT1_2] }; // +90° about Y
const atlas = makeAtlas({ hips: bakeHips });

// ── 1. Identity frame: current hips == bake hips, scale 1 → pure passthrough.
{
  const f = atlas.frame(bakeHips.worldPos, bakeHips.worldQuat, 1);
  const p = [0.12, 0.05, -0.03];
  vclose(f.toBake(p), p, 1e-9, 'identity toBake is passthrough');
  close(f.sdf(p), atlas.sdf(p), 1e-9, 'identity sdf matches raw atlas sdf');
  vclose(f.toWorld(f.toBake(p)), p, 1e-9, 'identity round-trips');
}

// ── 2. Round-trip under an arbitrary rotation + translation + scale.
{
  // Avatar turned 37° about Y, moved, and scaled to 1.4x.
  const a = (37 * Math.PI) / 180;
  const curQuat = [0, Math.sin(a / 2), 0, Math.cos(a / 2)];
  const curPos = [0.5, 1.1, -0.25];
  const s = 1.4;
  const f = atlas.frame(curPos, curQuat, s);
  for (const pB of [[0.1, 0, 0], [-0.2, 0.15, 0.05], [0.0, -0.1, 0.2]]) {
    const pW = f.toWorld(pB);
    vclose(f.toBake(pW), pB, 1e-6, `round-trip bake→world→bake for ${JSON.stringify(pB)}`);
  }
}

// ── 3. THE bug: SDF invariance. A hand at the SAME body location reads the
//    same penetration (×scale) regardless of how the avatar is posed.
{
  const cases = [
    { pos: [0, 0.9, 0], quat: bakeHips.worldQuat, s: 1 },        // at bake
    { pos: [0, 0.9, 0], quat: [0, Math.sin(0.6), 0, Math.cos(0.6)], s: 1 },   // turned
    { pos: [0.3, 1.2, 0.1], quat: [0, Math.sin(-0.9), 0, Math.cos(-0.9)], s: 1.6 }, // turned+moved+scaled
  ];
  const pB = [0.13, 0.04, -0.06];      // fixed body-frame point
  const sBake = atlas.sdf(pB);          // its true (bake) signed distance
  for (const c of cases) {
    const f = atlas.frame(c.pos, c.quat, c.s);
    const pW = f.toWorld(pB);           // where that body point appears now
    close(f.sdf(pW), c.s * sBake, 1e-5,
      `sdf invariant under pose (quat=${JSON.stringify(c.quat)}, s=${c.s})`);
  }
}

// ── 4. gradient comes back in WORLD space and stays unit-length.
//    Use an IDENTITY-bake atlas so the world direction is exactly R_y(a)·(+x)
//    with no bake offset to reason around.
{
  const a = (50 * Math.PI) / 180;
  const atlasI = makeAtlas({ hips: { worldPos: [0, 0.9, 0], worldQuat: [0, 0, 0, 1] } });
  const f = atlasI.frame([0.1, 0.95, 0], [0, Math.sin(a / 2), 0, Math.cos(a / 2)], 1.2);
  const g = f.gradient([0.1, 0.0, 0.0]);
  close(Math.hypot(g[0], g[1], g[2]), 1.0, 1e-6, 'world gradient is unit length');
  // Bake gradient here is +x ([1,0,0]); after +a yaw the world dir is R_y(a)·x.
  const expected = [Math.cos(a), 0, -Math.sin(a)];   // R_y(+a) applied to +x
  vclose(g, expected, 1e-5, 'gradient rotated into world frame matches yaw');
}

// ── 5. pushOutsideBody respects clearance in WORLD meters after scaling.
{
  const s = 1.5;
  const f = atlas.frame([0, 0.9, 0], bakeHips.worldQuat, s);
  // A point just inside (bake sdf<0): bake x=-0.05 → sdf=-0.05.
  const pW = f.toWorld([-0.05, 0.0, 0.0]);
  const out = f.pushOutsideBody(pW, 0.03, 8);   // want ≥3cm world clearance
  ok(f.sdf(out) >= 0.03 - 1e-4, 'pushOutsideBody reaches world-meter clearance');
}

// ── 6. surfacePoints / convexAnchors come back in WORLD space.
{
  const a = (90 * Math.PI) / 180;
  const f = atlas.frame([0.2, 1.0, 0.1], [0, Math.sin(a / 2), 0, Math.cos(a / 2)], 1.0);
  const anchors = f.convexAnchors('torso');
  ok(anchors.length === 1, 'one torso anchor');
  // Anchor baked at [0.1,0,0]; its world position = frame.toWorld of that.
  vclose(anchors[0].point, f.toWorld([0.1, 0, 0]), 1e-9, 'anchor point is world-space');
  const pts = f.surfacePoints({ n: 5 });
  ok(pts.length === 5, 'surfacePoints returns world-space samples');
}

// ── 7. Back-compat: an atlas WITHOUT a bake hips block → identity framing.
{
  const noFrame = makeAtlas();              // no hips
  ok(noFrame.hasBakeFrame === false, 'atlas without hips reports hasBakeFrame=false');
  const f = noFrame.frame([5, 5, 5], [0, 1, 0, 0], 3);   // bogus live pose ignored
  const p = [0.1, 0.2, -0.1];
  vclose(f.toBake(p), p, 1e-9, 'no-bake-frame falls back to identity (no regression)');
  close(f.sdf(p), noFrame.sdf(p), 1e-9, 'no-bake-frame sdf == raw sdf');
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
