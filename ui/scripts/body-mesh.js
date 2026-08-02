/**
 * body-mesh.js — millimeter-precision mesh substrate for VRM avatars.
 *
 * Layer-1 of the BodySubstrate architecture (see body-atlas.js for layer-2,
 * the centimeter-precision voxel atlas). The mesh substrate gives:
 *
 *   - Exact closest-point queries on the avatar's actual surface mesh,
 *     resolved via a BVH built at avatar load time.
 *   - Per-triangle region labels (chin, cheek_L, sternum, hip_L, knee_R,
 *     etc.) consistent with the atlas's regionTable so the two layers
 *     speak the same vocabulary.
 *   - Named anatomical landmarks (head_top, chin, nose_tip, eye_outer_L,
 *     knuckle_R, etc.) resolved from VRM humanoid bones and surface-snapped
 *     to the nearest mesh point.
 *   - Region-filtered closest-point ("which triangle is closest to my
 *     fingertip AMONG triangles tagged 'forehead'?") for precise targeting.
 *
 * Built per-VRM at avatar load; no on-disk artifact required (BVH is
 * derived from the mesh which is already loaded). Construction is
 * ~100ms for a typical 30-50K-triangle VRM and runs once per avatar.
 *
 * Querying is sub-microsecond via the same BVH algorithm the atlas
 * generator uses; closest-point with surface normal returned in O(log n).
 *
 * Companion module: body-atlas.js for cm-scale path planning + free-space
 * queries. The two compose via avatar.js::loadVRM stashing both on the VRM:
 *   vrm.__augmentumBodyMesh   (this module)
 *   vrm.__augmentumBodyAtlas  (existing voxel atlas)
 *
 * Phase 1 (this file): BVH + closestPoint + triangle region labels +
 * bone-derived landmarks.
 * Phase 2 (deferred): mesh-curvature anchors, blendshape-derived
 * landmarks (lip corners, eyelid points, etc.), runtime-skinned
 * landmark positions for dynamic queries.
 */

// Region taxonomy — MUST match body-atlas-generator.html's REGIONS list
// so triangle region indices interop with atlas voxel region indices.
const REGIONS = [
  // 0–13 head sub-regions
  'forehead', 'temple_L', 'temple_R', 'ear_L', 'ear_R',
  'cheek_L', 'cheek_R', 'eye_L', 'eye_R', 'mouth',
  'jaw', 'chin', 'head_top', 'head_back',
  // 14 neck
  'neck',
  // 15–24 torso
  'shoulder_L', 'shoulder_R', 'chest_L', 'chest_R', 'sternum',
  'side_L', 'side_R', 'belly', 'navel', 'back_upper',
  // 25–27 hips
  'hip_L', 'hip_R', 'lower_back',
  // 28–35 arms
  'upper_arm_L', 'upper_arm_R', 'elbow_L', 'elbow_R',
  'lower_arm_L', 'lower_arm_R', 'hand_L', 'hand_R',
  // 36–45 legs
  'thigh_L', 'thigh_R', 'knee_L', 'knee_R',
  'shin_L', 'shin_R', 'ankle_L', 'ankle_R', 'foot_L', 'foot_R',
  // 46 unclassified
  'other',
];
const REGION_INDEX = Object.fromEntries(REGIONS.map((r, i) => [r, i]));
const OTHER_IDX = REGION_INDEX.other;

const HUMANOID_TO_REGION = {
  head: 'head_top',
  neck: 'neck',
  leftShoulder: 'shoulder_L',
  rightShoulder: 'shoulder_R',
  leftUpperArm: 'upper_arm_L',
  rightUpperArm: 'upper_arm_R',
  leftLowerArm: 'lower_arm_L',
  rightLowerArm: 'lower_arm_R',
  leftHand: 'hand_L',
  rightHand: 'hand_R',
  chest: 'chest_L',
  upperChest: 'chest_L',
  spine: 'belly',
  hips: 'lower_back',
  leftUpperLeg: 'thigh_L',
  rightUpperLeg: 'thigh_R',
  leftLowerLeg: 'shin_L',
  rightLowerLeg: 'shin_R',
  leftFoot: 'foot_L',
  rightFoot: 'foot_R',
  leftToes: 'foot_L',
  rightToes: 'foot_R',
};

// ─────────────────────────────────────────────────────────────────────────────
// Triangle extraction — skins each SkinnedMesh to T-pose world frame and
// produces a flat triangle buffer with face normals, areas, and dominant
// humanoid-bone influence per triangle.
//
// Ported from body-atlas-generator.html (Phase 1a). Stays in sync with that
// implementation; refactoring both to a shared module is future work.
// ─────────────────────────────────────────────────────────────────────────────
function extractTriangles(THREE, vrm) {
  const trisOut = { verts: [], normals: [], areas: [], humanoidBoneIdx: [] };
  const humanoidBoneNames = [];
  const humanoidBoneLookup = new Map();

  if (vrm.humanoid?.humanBones) {
    for (const [hname, info] of Object.entries(vrm.humanoid.humanBones)) {
      const node = info.node;
      if (node) humanoidBoneLookup.set(node, hname);
    }
  }

  function ensureBoneIdx(name) {
    let idx = humanoidBoneNames.indexOf(name);
    if (idx < 0) { idx = humanoidBoneNames.length; humanoidBoneNames.push(name); }
    return idx;
  }
  const otherBoneIdx = ensureBoneIdx('__other__');

  let totalTris = 0;
  vrm.scene.updateMatrixWorld(true);
  vrm.scene.traverse((obj) => {
    if (!obj.isSkinnedMesh) return;
    const geom = obj.geometry;
    if (!geom?.attributes?.position) return;
    const pos = geom.attributes.position;
    const skinIndex = geom.attributes.skinIndex;
    const skinWeight = geom.attributes.skinWeight;
    const idx = geom.index;
    const skel = obj.skeleton;
    if (!skel) return;

    // Dominant humanoid bone per vertex
    const vertHbone = new Int32Array(pos.count);
    for (let i = 0; i < pos.count; i++) {
      let maxW = -1, maxBoneIdx = -1;
      for (let k = 0; k < 4; k++) {
        const w = skinWeight ? skinWeight.array[i*4+k] : 0;
        if (w > maxW) { maxW = w; maxBoneIdx = skinIndex ? skinIndex.array[i*4+k] : 0; }
      }
      if (maxBoneIdx < 0 || maxBoneIdx >= skel.bones.length) {
        vertHbone[i] = otherBoneIdx;
      } else {
        const rawBone = skel.bones[maxBoneIdx];
        const hname = humanoidBoneLookup.get(rawBone) || '__other__';
        vertHbone[i] = ensureBoneIdx(hname);
      }
    }

    // T-pose world positions via boneTransform
    const v = new THREE.Vector3();
    const worldPos = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i);
      if (typeof obj.applyBoneTransform === 'function') {
        obj.applyBoneTransform(i, v);
      } else if (typeof obj.boneTransform === 'function') {
        obj.boneTransform(i, v);
      }
      v.applyMatrix4(obj.matrixWorld);
      worldPos[i*3+0] = v.x; worldPos[i*3+1] = v.y; worldPos[i*3+2] = v.z;
    }

    const triCount = idx ? idx.count / 3 : pos.count / 3;
    const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
    const n = new THREE.Vector3(), e1 = new THREE.Vector3(), e2 = new THREE.Vector3();
    for (let t = 0; t < triCount; t++) {
      const i0 = idx ? idx.array[t*3+0] : t*3+0;
      const i1 = idx ? idx.array[t*3+1] : t*3+1;
      const i2 = idx ? idx.array[t*3+2] : t*3+2;
      a.set(worldPos[i0*3], worldPos[i0*3+1], worldPos[i0*3+2]);
      b.set(worldPos[i1*3], worldPos[i1*3+1], worldPos[i1*3+2]);
      c.set(worldPos[i2*3], worldPos[i2*3+1], worldPos[i2*3+2]);
      e1.subVectors(b, a); e2.subVectors(c, a);
      n.crossVectors(e1, e2);
      const area2 = n.length();
      if (area2 < 1e-12) continue;
      n.divideScalar(area2);
      trisOut.verts.push(a.x,a.y,a.z, b.x,b.y,b.z, c.x,c.y,c.z);
      trisOut.normals.push(n.x, n.y, n.z);
      trisOut.areas.push(area2 * 0.5);
      trisOut.humanoidBoneIdx.push(vertHbone[i0]);
      totalTris++;
    }
  });

  return {
    verts: new Float32Array(trisOut.verts),
    normals: new Float32Array(trisOut.normals),
    areas: new Float32Array(trisOut.areas),
    humanoidBoneIdx: new Uint16Array(trisOut.humanoidBoneIdx),
    humanoidBoneNames,
    triCount: totalTris,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// BVH over triangles — top-down longest-axis median split, flat-array layout.
// Same construction as body-atlas-generator.html (Phase 1b); included inline
// so this module is self-contained at runtime.
// ─────────────────────────────────────────────────────────────────────────────
function buildBVH(tris) {
  const { verts, triCount } = tris;
  const triOrder = new Int32Array(triCount);
  for (let i = 0; i < triCount; i++) triOrder[i] = i;

  const triCentroids = new Float32Array(triCount * 3);
  const triAABBmin = new Float32Array(triCount * 3);
  const triAABBmax = new Float32Array(triCount * 3);
  for (let t = 0; t < triCount; t++) {
    const ax = verts[t*9+0], ay = verts[t*9+1], az = verts[t*9+2];
    const bx = verts[t*9+3], by = verts[t*9+4], bz = verts[t*9+5];
    const cx = verts[t*9+6], cy = verts[t*9+7], cz = verts[t*9+8];
    triCentroids[t*3+0] = (ax+bx+cx)/3;
    triCentroids[t*3+1] = (ay+by+cy)/3;
    triCentroids[t*3+2] = (az+bz+cz)/3;
    triAABBmin[t*3+0] = Math.min(ax,bx,cx); triAABBmax[t*3+0] = Math.max(ax,bx,cx);
    triAABBmin[t*3+1] = Math.min(ay,by,cy); triAABBmax[t*3+1] = Math.max(ay,by,cy);
    triAABBmin[t*3+2] = Math.min(az,bz,cz); triAABBmax[t*3+2] = Math.max(az,bz,cz);
  }

  const LEAF_SIZE = 8;
  const maxNodes = triCount * 2;
  const nAABBmin = new Float32Array(maxNodes * 3);
  const nAABBmax = new Float32Array(maxNodes * 3);
  const nLeft = new Int32Array(maxNodes).fill(-1);
  const nRight = new Int32Array(maxNodes).fill(-1);
  const nFirstTri = new Int32Array(maxNodes);
  const nTriCount = new Int32Array(maxNodes);
  let nodeCount = 0;

  function build2(start, end) {
    const idx = nodeCount++;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = start; i < end; i++) {
      const t = triOrder[i];
      if (triAABBmin[t*3+0] < minX) minX = triAABBmin[t*3+0];
      if (triAABBmin[t*3+1] < minY) minY = triAABBmin[t*3+1];
      if (triAABBmin[t*3+2] < minZ) minZ = triAABBmin[t*3+2];
      if (triAABBmax[t*3+0] > maxX) maxX = triAABBmax[t*3+0];
      if (triAABBmax[t*3+1] > maxY) maxY = triAABBmax[t*3+1];
      if (triAABBmax[t*3+2] > maxZ) maxZ = triAABBmax[t*3+2];
    }
    nAABBmin[idx*3+0]=minX; nAABBmin[idx*3+1]=minY; nAABBmin[idx*3+2]=minZ;
    nAABBmax[idx*3+0]=maxX; nAABBmax[idx*3+1]=maxY; nAABBmax[idx*3+2]=maxZ;

    const count = end - start;
    if (count <= LEAF_SIZE) {
      nLeft[idx] = -1; nRight[idx] = -1;
      nFirstTri[idx] = start; nTriCount[idx] = count;
    } else {
      const dx = maxX-minX, dy = maxY-minY, dz = maxZ-minZ;
      const axis = dx >= dy && dx >= dz ? 0 : (dy >= dz ? 1 : 2);
      const subset = Array.from(triOrder.slice(start, end));
      subset.sort((a, b) => triCentroids[a*3+axis] - triCentroids[b*3+axis]);
      for (let i = 0; i < subset.length; i++) triOrder[start+i] = subset[i];
      const mid = start + Math.floor(count/2);
      nLeft[idx] = build2(start, mid);
      nRight[idx] = build2(mid, end);
      nFirstTri[idx] = -1; nTriCount[idx] = 0;
    }
    return idx;
  }
  build2(0, triCount);

  return {
    nAABBmin, nAABBmax, nLeft, nRight, nFirstTri, nTriCount,
    triOrder, nodeCount, tris,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Closest-point on BVH — returns distance + nearest triangle ID + the
// actual closest point on that triangle (not just distance). The atlas
// generator version returns only {d, tri}; we extend it to also resolve
// the closest point coordinates via a second pass on the winning triangle.
// ─────────────────────────────────────────────────────────────────────────────
function distancePointAABB(qx, qy, qz, minX, minY, minZ, maxX, maxY, maxZ) {
  const dx = Math.max(0, Math.max(minX - qx, qx - maxX));
  const dy = Math.max(0, Math.max(minY - qy, qy - maxY));
  const dz = Math.max(0, Math.max(minZ - qz, qz - maxZ));
  return Math.sqrt(dx*dx + dy*dy + dz*dz);
}

// Closest point + barycentric coords on triangle ABC.
// Eberly 1999 formulation; returns {d2, px, py, pz}.
function closestPointTriangle(qx, qy, qz, ax, ay, az, bx, by, bz, cx, cy, cz) {
  const ABx = bx-ax, ABy = by-ay, ABz = bz-az;
  const ACx = cx-ax, ACy = cy-ay, ACz = cz-az;
  const APx = qx-ax, APy = qy-ay, APz = qz-az;
  const d1 = ABx*APx + ABy*APy + ABz*APz;
  const d2 = ACx*APx + ACy*APy + ACz*APz;
  if (d1 <= 0 && d2 <= 0) {
    return { d2: APx*APx+APy*APy+APz*APz, px: ax, py: ay, pz: az };
  }
  const BPx = qx-bx, BPy = qy-by, BPz = qz-bz;
  const d3 = ABx*BPx + ABy*BPy + ABz*BPz;
  const d4 = ACx*BPx + ACy*BPy + ACz*BPz;
  if (d3 >= 0 && d4 <= d3) {
    return { d2: BPx*BPx+BPy*BPy+BPz*BPz, px: bx, py: by, pz: bz };
  }
  const vc = d1*d4 - d3*d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    const px = ax + v*ABx, py = ay + v*ABy, pz = az + v*ABz;
    const dx = qx-px, dy = qy-py, dz = qz-pz;
    return { d2: dx*dx+dy*dy+dz*dz, px, py, pz };
  }
  const CPx = qx-cx, CPy = qy-cy, CPz = qz-cz;
  const d5 = ABx*CPx + ABy*CPy + ABz*CPz;
  const d6 = ACx*CPx + ACy*CPy + ACz*CPz;
  if (d6 >= 0 && d5 <= d6) {
    return { d2: CPx*CPx+CPy*CPy+CPz*CPz, px: cx, py: cy, pz: cz };
  }
  const vb = d5*d2 - d1*d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    const px = ax + w*ACx, py = ay + w*ACy, pz = az + w*ACz;
    const dx = qx-px, dy = qy-py, dz = qz-pz;
    return { d2: dx*dx+dy*dy+dz*dz, px, py, pz };
  }
  const va = d3*d6 - d5*d4;
  if (va <= 0 && (d4-d3) >= 0 && (d5-d6) >= 0) {
    const t = (d4-d3) / ((d4-d3) + (d5-d6));
    const px = bx + t*(cx-bx), py = by + t*(cy-by), pz = bz + t*(cz-bz);
    const dx = qx-px, dy = qy-py, dz = qz-pz;
    return { d2: dx*dx+dy*dy+dz*dz, px, py, pz };
  }
  const denom = 1/(va + vb + vc);
  const v = vb * denom, w = vc * denom;
  const px = ax + ABx*v + ACx*w;
  const py = ay + ABy*v + ACy*w;
  const pz = az + ABz*v + ACz*w;
  const dx = qx-px, dy = qy-py, dz = qz-pz;
  return { d2: dx*dx+dy*dy+dz*dz, px, py, pz };
}

function closestPointBVH(bvh, qx, qy, qz, triRegionFilter) {
  const { nAABBmin, nAABBmax, nLeft, nRight, nFirstTri, nTriCount,
          triOrder, tris } = bvh;
  const verts = tris.verts;
  let bestD2 = Infinity;
  let bestTri = -1;
  let bestPx = 0, bestPy = 0, bestPz = 0;
  const stack = [0];
  while (stack.length) {
    const idx = stack.pop();
    const ad = distancePointAABB(qx, qy, qz,
      nAABBmin[idx*3+0], nAABBmin[idx*3+1], nAABBmin[idx*3+2],
      nAABBmax[idx*3+0], nAABBmax[idx*3+1], nAABBmax[idx*3+2]);
    if (ad*ad >= bestD2) continue;
    if (nLeft[idx] < 0) {
      const start = nFirstTri[idx], cnt = nTriCount[idx];
      for (let i = 0; i < cnt; i++) {
        const t = triOrder[start+i];
        if (triRegionFilter && !triRegionFilter(t)) continue;
        const r = closestPointTriangle(qx, qy, qz,
          verts[t*9+0], verts[t*9+1], verts[t*9+2],
          verts[t*9+3], verts[t*9+4], verts[t*9+5],
          verts[t*9+6], verts[t*9+7], verts[t*9+8]);
        if (r.d2 < bestD2) {
          bestD2 = r.d2; bestTri = t;
          bestPx = r.px; bestPy = r.py; bestPz = r.pz;
        }
      }
    } else {
      const L = nLeft[idx], R = nRight[idx];
      const dL = distancePointAABB(qx, qy, qz,
        nAABBmin[L*3+0], nAABBmin[L*3+1], nAABBmin[L*3+2],
        nAABBmax[L*3+0], nAABBmax[L*3+1], nAABBmax[L*3+2]);
      const dR = distancePointAABB(qx, qy, qz,
        nAABBmin[R*3+0], nAABBmin[R*3+1], nAABBmin[R*3+2],
        nAABBmax[R*3+0], nAABBmax[R*3+1], nAABBmax[R*3+2]);
      if (dL < dR) { stack.push(R); stack.push(L); }
      else         { stack.push(L); stack.push(R); }
    }
  }
  return {
    d: Math.sqrt(bestD2),
    tri: bestTri,
    point: bestTri >= 0 ? [bestPx, bestPy, bestPz] : null,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Body landmarks from VRM humanoid bones — mirrors the generator's
// computeHeadLandmarks + new torso/joint extensions so triangle region
// labels match what the atlas produces voxel-by-voxel.
// ─────────────────────────────────────────────────────────────────────────────
function computeBodyLandmarks(THREE, vrm) {
  const h = vrm.humanoid;
  const get = (n) => h.getNormalizedBoneNode(n)?.getWorldPosition(new THREE.Vector3());
  const head = get('head');
  const neck = get('neck');
  const jaw = get('jaw');
  const lEye = get('leftEye');
  const rEye = get('rightEye');
  const hips = get('hips');
  const chest = get('chest') || get('upperChest') || get('spine');
  const lShoulder = get('leftShoulder');
  const rShoulder = get('rightShoulder');
  const lUpperLeg = get('leftUpperLeg');
  const rUpperLeg = get('rightUpperLeg');
  const lLowerArm = get('leftLowerArm');
  const rLowerArm = get('rightLowerArm');
  const lLowerLeg = get('leftLowerLeg');
  const rLowerLeg = get('rightLowerLeg');
  const lFoot = get('leftFoot');
  const rFoot = get('rightFoot');
  const lHand = get('leftHand');
  const rHand = get('rightHand');

  const headRadius = neck ? Math.max(0.10, (head.y - neck.y) * 0.85) : 0.13;
  let faceForwardZ = +1;
  if (lEye || rEye) {
    const eyeZ = ((lEye?.z ?? head.z) + (rEye?.z ?? head.z)) / 2;
    faceForwardZ = (eyeZ > head.z) ? +1 : -1;
  }
  const faceCenter = head.clone();
  if (neck) {
    const headSize = head.y - neck.y;
    faceCenter.y -= headSize * 0.15;
    faceCenter.z += faceForwardZ * headSize * 0.40;
  } else {
    faceCenter.z += faceForwardZ * 0.05;
  }
  const eyeY = (lEye && rEye) ? (lEye.y + rEye.y) * 0.5 : faceCenter.y + 0.01;
  const mouthY = lEye ? lEye.y - (lEye.y - (jaw?.y ?? head.y - headRadius * 0.5)) * 0.5
                      : faceCenter.y - headRadius * 0.30;
  const chinY = jaw?.y ?? (head.y - headRadius * 0.55);

  const shoulderHalfWidth = lShoulder && rShoulder
    ? Math.max(0.08, Math.abs(lShoulder.x - rShoulder.x) * 0.5)
    : 0.16;
  const sternumX = shoulderHalfWidth * 0.22;
  const sideX = shoulderHalfWidth * 0.78;
  const chestZ = chest ? chest.z : (head.z * 0.5 + (hips?.z ?? 0) * 0.5);

  const hipHalfWidth = lUpperLeg && rUpperLeg
    ? Math.max(0.07, Math.abs(lUpperLeg.x - rUpperLeg.x) * 0.5)
    : 0.12;
  const hipSplitX = hipHalfWidth * 0.55;
  const hipY = hips ? hips.y : 0.0;
  const torsoY = chest ? chest.y : (head.y - 0.25);
  const navelYTop = hipY + (torsoY - hipY) * 0.45;
  const navelYBottom = hipY + (torsoY - hipY) * 0.05;

  return {
    head, neck, jaw, lEye, rEye, hips, chest,
    lShoulder, rShoulder, lUpperLeg, rUpperLeg,
    lHand, rHand, lFoot, rFoot,
    faceCenter,
    hipsZ: hips ? hips.z : 0,
    headRadius,
    eyeY, mouthY, chinY,
    faceForwardZ,
    chestZ, sternumX, sideX, backUpperZMargin: 0.05,
    hipSplitX, navelYTop, navelYBottom,
    elbowL: lLowerArm, elbowR: rLowerArm,
    kneeL:  lLowerLeg, kneeR:  rLowerLeg,
    ankleL: lFoot,     ankleR: rFoot,
    jointRadiusSq: {
      elbow: 0.045 * 0.045,
      knee:  0.055 * 0.055,
      ankle: 0.045 * 0.045,
    },
  };
}

function isInHeadRegion(wx, wy, wz, lm) {
  if (!lm.head) return false;
  const dx = wx - lm.head.x;
  const dz = wz - lm.head.z;
  const minY = lm.chinY - 0.02;
  const maxY = lm.head.y + lm.headRadius * 2.0;
  if (wy < minY || wy > maxY) return false;
  return Math.sqrt(dx*dx + dz*dz) < lm.headRadius * 2.0;
}

function subdivideHead(wx, wy, wz, lm) {
  const h = lm.head;
  const fc = lm.faceCenter;
  const r = lm.headRadius;
  const f = lm.faceForwardZ;
  const dx = wx - h.x;
  const dy = wy - h.y;
  const dzFace = (wz - fc.z) * f;
  const dzHead = (wz - h.z)  * f;

  if (dy > r * 0.45) return 'head_top';
  if (dzHead < -r * 0.5) return 'head_back';
  if (Math.abs(dx) > r * 0.70 && wy > lm.chinY - 0.01 && wy < lm.eyeY + 0.04 && dzHead < r * 0.20) {
    return dx > 0 ? 'ear_L' : 'ear_R';
  }
  if (wy < lm.chinY) {
    if (Math.abs(dx) < 0.035 && dzFace > 0) return 'chin';
    return 'jaw';
  }
  if (dzFace > -r * 0.30) {
    if (wy > lm.eyeY + 0.020) {
      if (Math.abs(dx) > r * 0.40) return dx > 0 ? 'temple_L' : 'temple_R';
      return 'forehead';
    }
    if (Math.abs(wy - lm.eyeY) < 0.030) {
      if (Math.abs(dx) < 0.020) return 'forehead';
      if (Math.abs(dx) < r * 0.40) return dx > 0 ? 'eye_L' : 'eye_R';
      return dx > 0 ? 'temple_L' : 'temple_R';
    }
    if (wy > lm.mouthY) {
      if (Math.abs(dx) < 0.025) return 'forehead';
      return dx > 0 ? 'cheek_L' : 'cheek_R';
    }
    if (wy > lm.chinY) {
      if (Math.abs(dx) < 0.030) return 'mouth';
      return dx > 0 ? 'cheek_L' : 'cheek_R';
    }
  }
  if (Math.abs(dx) > r * 0.30) {
    if (wy > lm.eyeY - 0.01) return dx > 0 ? 'temple_L' : 'temple_R';
    return dx > 0 ? 'cheek_L' : 'cheek_R';
  }
  return dy > 0 ? 'head_top' : 'head_back';
}

// Region label for a single point — mirrors the generator's Phase 3 logic
// applied to triangle centroids. Same refinements: head subdivision,
// chest/shoulder x-sign, lower_back/belly z-sign, sternum/side/back_upper,
// hip_L/R, navel, joint regions.
function regionForPoint(wx, wy, wz, hboneName, lm) {
  if (isInHeadRegion(wx, wy, wz, lm)) {
    return subdivideHead(wx, wy, wz, lm);
  }
  let regionName = HUMANOID_TO_REGION[hboneName] || 'other';
  if (regionName === 'chest_L' && wx < 0) regionName = 'chest_R';
  if (regionName === 'shoulder_L' && wx < 0) regionName = 'shoulder_R';
  if (regionName === 'lower_back') {
    if ((wz - lm.hipsZ) * lm.faceForwardZ > 0.02) regionName = 'belly';
  }
  if (regionName === 'chest_L' || regionName === 'chest_R') {
    const dzChest = (wz - lm.chestZ) * lm.faceForwardZ;
    if (dzChest < -lm.backUpperZMargin) {
      regionName = 'back_upper';
    } else if (Math.abs(wx) < lm.sternumX) {
      regionName = 'sternum';
    } else if (Math.abs(wx) > lm.sideX) {
      regionName = wx > 0 ? 'side_L' : 'side_R';
    }
  }
  if (regionName === 'lower_back' && Math.abs(wx) > lm.hipSplitX) {
    regionName = wx > 0 ? 'hip_L' : 'hip_R';
  }
  if (regionName === 'belly'
      && wy >= lm.navelYBottom
      && wy <= lm.navelYTop
      && Math.abs(wx) < 0.05) {
    regionName = 'navel';
  }
  // Joint regions
  const checkJoint = (jointPos, radiusSq, name) => {
    if (!jointPos) return null;
    const dx_ = wx - jointPos.x, dy_ = wy - jointPos.y, dz_ = wz - jointPos.z;
    if (dx_*dx_ + dy_*dy_ + dz_*dz_ < radiusSq) return name;
    return null;
  };
  if (regionName === 'upper_arm_L' || regionName === 'lower_arm_L') {
    regionName = checkJoint(lm.elbowL, lm.jointRadiusSq.elbow, 'elbow_L') || regionName;
  } else if (regionName === 'upper_arm_R' || regionName === 'lower_arm_R') {
    regionName = checkJoint(lm.elbowR, lm.jointRadiusSq.elbow, 'elbow_R') || regionName;
  } else if (regionName === 'thigh_L' || regionName === 'shin_L') {
    regionName = checkJoint(lm.kneeL, lm.jointRadiusSq.knee, 'knee_L') || regionName;
  } else if (regionName === 'thigh_R' || regionName === 'shin_R') {
    regionName = checkJoint(lm.kneeR, lm.jointRadiusSq.knee, 'knee_R') || regionName;
  }
  if (regionName === 'shin_L' || regionName === 'foot_L') {
    regionName = checkJoint(lm.ankleL, lm.jointRadiusSq.ankle, 'ankle_L') || regionName;
  } else if (regionName === 'shin_R' || regionName === 'foot_R') {
    regionName = checkJoint(lm.ankleR, lm.jointRadiusSq.ankle, 'ankle_R') || regionName;
  }
  return regionName;
}

function classifyTriangles(tris, lm) {
  const triCount = tris.triCount;
  const out = new Uint8Array(triCount);
  const verts = tris.verts;
  const hboneIdx = tris.humanoidBoneIdx;
  const hboneNames = tris.humanoidBoneNames;
  for (let t = 0; t < triCount; t++) {
    const ax = verts[t*9+0], ay = verts[t*9+1], az = verts[t*9+2];
    const bx = verts[t*9+3], by = verts[t*9+4], bz = verts[t*9+5];
    const cx = verts[t*9+6], cy = verts[t*9+7], cz = verts[t*9+8];
    const wx = (ax+bx+cx)/3, wy = (ay+by+cy)/3, wz = (az+bz+cz)/3;
    const hname = hboneNames[hboneIdx[t]];
    const regionName = regionForPoint(wx, wy, wz, hname, lm);
    out[t] = REGION_INDEX[regionName] ?? OTHER_IDX;
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Region-aware geometric primitives used by landmark refinement. These walk
// the triangle-region assignments and pull out anatomically meaningful points
// directly from the labeled mesh — replacing fixed bone-offset heuristics
// that varied widely across VRMs (see poses/landmark-cross-vrm-stats.json
// for the empirical reliability tiers).
// ─────────────────────────────────────────────────────────────────────────────
function regionTriangleCentroid(triRegions, tris, regionName) {
  const target = REGION_INDEX[regionName];
  if (target === undefined) return null;
  const verts = tris.verts;
  let sx = 0, sy = 0, sz = 0, count = 0;
  for (let t = 0; t < triRegions.length; t++) {
    if (triRegions[t] !== target) continue;
    const cx = (verts[t*9+0] + verts[t*9+3] + verts[t*9+6]) / 3;
    const cy = (verts[t*9+1] + verts[t*9+4] + verts[t*9+7]) / 3;
    const cz = (verts[t*9+2] + verts[t*9+5] + verts[t*9+8]) / 3;
    sx += cx; sy += cy; sz += cz;
    count++;
  }
  if (count === 0) return null;
  return { point: [sx/count, sy/count, sz/count], triangleCount: count };
}

function regionTriangleExtremum(triRegions, tris, regionName, axis, dir) {
  // axis: 0/1/2 = x/y/z; dir: +1 = max, -1 = min. Returns the triangle
  // centroid that's most extreme along that axis × direction. Used for
  // "highest point of head_top" / "most-forward point in centerline" etc.
  const target = REGION_INDEX[regionName];
  if (target === undefined) return null;
  const verts = tris.verts;
  let bestVal = -Infinity * dir;
  let bestPos = null;
  for (let t = 0; t < triRegions.length; t++) {
    if (triRegions[t] !== target) continue;
    const cx = (verts[t*9+0] + verts[t*9+3] + verts[t*9+6]) / 3;
    const cy = (verts[t*9+1] + verts[t*9+4] + verts[t*9+7]) / 3;
    const cz = (verts[t*9+2] + verts[t*9+5] + verts[t*9+8]) / 3;
    const val = axis === 0 ? cx : (axis === 1 ? cy : cz);
    if (dir > 0 ? val > bestVal : val < bestVal) {
      bestVal = val;
      bestPos = [cx, cy, cz];
    }
  }
  return bestPos;
}

// ─────────────────────────────────────────────────────────────────────────────
// Named landmark resolution — for each interesting anatomical point, get
// an estimate position (preferring region-centroid for low-reliability
// landmarks, falling back to bone-anchored heuristics), then surface-snap
// via BVH closestPoint so the result is an actual exterior mesh point.
//
// Phase 1: bone-derived + region-centroid landmarks. Phase 2 will add
// blendshape-derived points (lip ring from MouthOpen morph, etc.) and
// runtime-skinned current positions.
//
// Refinement history (2026-05-14): the 6 low-reliability landmarks
// identified by poses/landmark-cross-vrm-stats.json (head_top, belly_center,
// elbow_L/R, collarbone_L/R) were moved from fixed bone offsets to region-
// centroid resolution. Where the region exists in the bake, the centroid
// IS the anatomical center; no hand-tuned offset can beat that.
// ─────────────────────────────────────────────────────────────────────────────
function resolveLandmarks(THREE, vrm, lm, bvh, triRegions) {
  const out = {};
  const tris = bvh.tris;

  // Accepts either a THREE.Vector3 or [x,y,z]. Surface-snaps via BVH,
  // optionally restricted to triangles in a preferred region.
  const def = (name, position, preferredRegion) => {
    if (!position) return;
    const ax = position.x ?? position[0];
    const ay = position.y ?? position[1];
    const az = position.z ?? position[2];
    if (ax === undefined) return;
    const filter = preferredRegion
      ? (t) => REGIONS[triRegions[t]] === preferredRegion
      : null;
    let snapped = closestPointBVH(bvh, ax, ay, az, filter);
    if (snapped.tri < 0) {
      snapped = closestPointBVH(bvh, ax, ay, az, null);
    }
    if (snapped.tri < 0) return;
    const t = snapped.tri;
    const region = REGIONS[triRegions[t]];
    const normals = bvh.tris.normals;
    out[name] = {
      point: snapped.point,
      normal: [normals[t*3+0], normals[t*3+1], normals[t*3+2]],
      region,
      triangleId: t,
      boneAnchor: [ax, ay, az],
    };
  };

  // Refined: prefer the region centroid (or extremum) over a bone offset
  // when the region exists. The centroid IS the mesh-anatomical center;
  // surface-snap then projects it to a representative triangle. If the
  // region has no triangles on this VRM, fall through to the bone fallback.
  const defFromRegion = (name, regionName, fallback) => {
    const centroid = regionTriangleCentroid(triRegions, tris, regionName);
    if (centroid && centroid.triangleCount >= 4) {
      def(name, centroid.point, regionName);
      return;
    }
    if (fallback) def(name, fallback, regionName);
  };
  const defFromRegionExtremum = (name, regionName, axis, dir, fallback) => {
    const extr = regionTriangleExtremum(triRegions, tris, regionName, axis, dir);
    if (extr) {
      def(name, extr, regionName);
      return;
    }
    if (fallback) def(name, fallback, regionName);
  };

  const fwd = lm.faceForwardZ;
  // Head/face
  if (lm.head) {
    // head_top: max-Y point in head_top region — finds actual top of
    // hair/skull regardless of how tall the VRM's hair extends.
    // Fallback: bone-offset estimate for VRMs without head_top triangles.
    const fallbackTop = lm.head.clone(); fallbackTop.y += lm.headRadius * 0.95;
    defFromRegionExtremum('head_top', 'head_top', 1, +1, fallbackTop);

    const back = lm.head.clone(); back.z -= fwd * lm.headRadius * 0.95;
    def('head_back', back, 'head_back');
  }
  if (lm.jaw) {
    const chin = lm.jaw.clone(); chin.z += fwd * 0.03;
    def('chin', chin, 'chin');
  }
  if (lm.faceCenter) {
    // Nose tip — between eyes, forward of face center
    const noseTip = lm.faceCenter.clone();
    noseTip.y = (lm.eyeY + lm.mouthY) * 0.5;
    noseTip.z += fwd * 0.04;
    def('nose_tip', noseTip);
    // Forehead center
    const forehead = lm.faceCenter.clone();
    forehead.y = lm.eyeY + 0.05;
    def('forehead_center', forehead, 'forehead');
    // Mouth center
    const mouth = lm.faceCenter.clone();
    mouth.y = lm.mouthY;
    def('mouth_center', mouth, 'mouth');
  }
  if (lm.lEye) {
    const outer = lm.lEye.clone(); outer.x += 0.025; outer.z += fwd * 0.02;
    def('eye_outer_L', outer, 'eye_L');
    const inner = lm.lEye.clone(); inner.x -= 0.015; inner.z += fwd * 0.02;
    def('eye_inner_L', inner, 'eye_L');
  }
  if (lm.rEye) {
    const outer = lm.rEye.clone(); outer.x -= 0.025; outer.z += fwd * 0.02;
    def('eye_outer_R', outer, 'eye_R');
    const inner = lm.rEye.clone(); inner.x += 0.015; inner.z += fwd * 0.02;
    def('eye_inner_R', inner, 'eye_R');
  }
  if (lm.head) {
    // Cheek bones (approx midway between eye and jaw, outward)
    if (lm.lEye) {
      const cheekL = lm.lEye.clone();
      cheekL.y = (lm.eyeY + lm.chinY) * 0.5;
      cheekL.x += 0.015;
      def('cheek_L', cheekL, 'cheek_L');
    }
    if (lm.rEye) {
      const cheekR = lm.rEye.clone();
      cheekR.y = (lm.eyeY + lm.chinY) * 0.5;
      cheekR.x -= 0.015;
      def('cheek_R', cheekR, 'cheek_R');
    }
    // Ear approximations
    const earL = lm.head.clone(); earL.x += lm.headRadius * 0.9; earL.y = lm.eyeY;
    def('ear_L', earL, 'ear_L');
    const earR = lm.head.clone(); earR.x -= lm.headRadius * 0.9; earR.y = lm.eyeY;
    def('ear_R', earR, 'ear_R');
    // Temples
    const templeL = lm.head.clone(); templeL.x += lm.headRadius * 0.85; templeL.y = lm.eyeY + 0.03;
    def('temple_L', templeL, 'temple_L');
    const templeR = lm.head.clone(); templeR.x -= lm.headRadius * 0.85; templeR.y = lm.eyeY + 0.03;
    def('temple_R', templeR, 'temple_R');
  }
  // Neck
  if (lm.neck) {
    const front = lm.neck.clone(); front.z += fwd * 0.04;
    def('neck_front', front, 'neck');
    const back = lm.neck.clone(); back.z -= fwd * 0.04;
    def('neck_back', back, 'neck');
  }
  // Torso landmarks
  if (lm.chest) {
    const sternum = lm.chest.clone(); sternum.x = 0; sternum.z += fwd * 0.06;
    def('sternum', sternum, 'sternum');
    // collarbone: max-Y point in chest_L/R (closest to shoulder boundary).
    // Stable across VRMs because we use the mesh's own chest-top edge
    // rather than interpolating chest+shoulder bone positions.
    if (lm.lShoulder) {
      const fallbackL = lm.chest.clone();
      fallbackL.x = lm.lShoulder.x * 0.6;
      fallbackL.y = (lm.chest.y + lm.lShoulder.y) * 0.5;
      fallbackL.z += fwd * 0.04;
      defFromRegionExtremum('collarbone_L', 'chest_L', 1, +1, fallbackL);
    }
    if (lm.rShoulder) {
      const fallbackR = lm.chest.clone();
      fallbackR.x = lm.rShoulder.x * 0.6;
      fallbackR.y = (lm.chest.y + lm.rShoulder.y) * 0.5;
      fallbackR.z += fwd * 0.04;
      defFromRegionExtremum('collarbone_R', 'chest_R', 1, +1, fallbackR);
    }
  }
  // Belly + navel
  if (lm.hips && lm.chest) {
    // belly_center: region centroid of the 'belly' triangles.
    const fallbackBelly = lm.hips.clone();
    fallbackBelly.y = (lm.hips.y + lm.chest.y) * 0.6;
    fallbackBelly.z += fwd * 0.08;
    defFromRegion('belly_center', 'belly', fallbackBelly);

    const navel = lm.hips.clone();
    navel.y = lm.hips.y + (lm.chest.y - lm.hips.y) * 0.22;
    navel.z += fwd * 0.07;
    def('navel', navel, 'navel');
  }
  // Hips
  if (lm.lUpperLeg) {
    const hipL = lm.lUpperLeg.clone(); hipL.x += 0.04;
    def('hip_L', hipL, 'hip_L');
  }
  if (lm.rUpperLeg) {
    const hipR = lm.rUpperLeg.clone(); hipR.x -= 0.04;
    def('hip_R', hipR, 'hip_R');
  }
  // Shoulders
  if (lm.lShoulder) def('shoulder_L', lm.lShoulder, 'shoulder_L');
  if (lm.rShoulder) def('shoulder_R', lm.rShoulder, 'shoulder_R');
  // Elbows: prefer region centroid (Phase 3 joint sphere produces a
  // localized blob of triangles at the actual elbow surface). The bone
  // position alone snapped inconsistently across VRMs because arm
  // diameters vary; region centroid is robust to that.
  if (lm.elbowL) defFromRegion('elbow_L', 'elbow_L', lm.elbowL);
  if (lm.elbowR) defFromRegion('elbow_R', 'elbow_R', lm.elbowR);
  // Wrists / hands
  if (lm.lHand) {
    def('wrist_L', lm.lHand, 'hand_L');
    const knuckle = lm.lHand.clone(); knuckle.x += 0.06;
    def('knuckle_L', knuckle, 'hand_L');
  }
  if (lm.rHand) {
    def('wrist_R', lm.rHand, 'hand_R');
    const knuckle = lm.rHand.clone(); knuckle.x -= 0.06;
    def('knuckle_R', knuckle, 'hand_R');
  }
  // Knees / ankles
  if (lm.kneeL) def('knee_L', lm.kneeL, 'knee_L');
  if (lm.kneeR) def('knee_R', lm.kneeR, 'knee_R');
  if (lm.ankleL) def('ankle_L', lm.ankleL, 'ankle_L');
  if (lm.ankleR) def('ankle_R', lm.ankleR, 'ankle_R');

  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Public class — what avatar.js stashes on the VRM.
// ─────────────────────────────────────────────────────────────────────────────
export class BodyMesh {
  static REGIONS = REGIONS;

  /**
   * Build a BodyMesh from a loaded VRM. Construction is synchronous-ish:
   * triangle extraction + BVH build + region classification + landmark
   * resolution. ~100ms for a typical VRM; do it once per avatar load.
   *
   * @param {object} opts
   * @param {object} opts.three  THREE namespace
   * @param {object} opts.vrm    VRM with humanoid
   * @returns {BodyMesh}
   */
  static create({ three, vrm }) {
    if (!three) throw new Error('BodyMesh.create requires opts.three');
    if (!vrm?.humanoid) throw new Error('BodyMesh.create requires opts.vrm with humanoid');
    const t0 = performance.now();
    const tris = extractTriangles(three, vrm);
    const bvh = buildBVH(tris);
    const landmarks = computeBodyLandmarks(three, vrm);
    const triRegions = classifyTriangles(tris, landmarks);
    const resolvedLandmarks = resolveLandmarks(three, vrm, landmarks, bvh, triRegions);
    return new BodyMesh({
      three, vrm, tris, bvh, landmarks, triRegions, resolvedLandmarks,
      buildMs: performance.now() - t0,
    });
  }

  constructor({ three, vrm, tris, bvh, landmarks, triRegions, resolvedLandmarks, buildMs }) {
    this._three = three;
    this._vrm = vrm;
    this.tris = tris;
    this.bvh = bvh;
    this.bodyLandmarks = landmarks;       // raw bone-anchored points
    this.triRegions = triRegions;         // Uint8Array, region idx per triangle
    this._namedLandmarks = resolvedLandmarks;  // surface-snapped + region-tagged
    this.regionTable = REGIONS;
    this.buildMs = buildMs;
  }

  /**
   * Closest point on the avatar surface to a world-space query point.
   * Optionally restricted to triangles in a named region.
   *
   * @param {[number,number,number]} worldPos
   * @param {object} [opts]
   * @param {string} [opts.region]  restrict to this region (e.g. 'chin')
   * @returns {{point:[number,number,number], normal:[number,number,number],
   *           region:string, distance:number, triangleId:number}|null}
   */
  closestPoint(worldPos, opts = {}) {
    const region = opts.region;
    const filter = region
      ? (t) => REGIONS[this.triRegions[t]] === region
      : null;
    const r = closestPointBVH(this.bvh, worldPos[0], worldPos[1], worldPos[2], filter);
    if (r.tri < 0) return null;
    const t = r.tri;
    const normals = this.tris.normals;
    return {
      point: r.point,
      normal: [normals[t*3+0], normals[t*3+1], normals[t*3+2]],
      region: REGIONS[this.triRegions[t]],
      distance: r.d,
      triangleId: t,
    };
  }

  /**
   * Named anatomical landmark — surface-snapped position + region + normal.
   * Phase 1: bone-derived, baked at construction; returns rest-pose
   * position. Phase 2 will add runtime-skinned current-position queries.
   *
   * @param {string} name
   * @returns {{point:[number,number,number], normal:[number,number,number],
   *           region:string, triangleId:number, boneAnchor:[number,number,number]}|null}
   */
  landmark(name) {
    return this._namedLandmarks[name] || null;
  }

  /** All available landmark names (sorted). */
  listLandmarks() {
    return Object.keys(this._namedLandmarks).sort();
  }

  /** All triangle IDs in a named region. Useful for region-mask visualization. */
  trianglesInRegion(regionName) {
    const target = REGION_INDEX[regionName];
    if (target === undefined) return [];
    const out = [];
    for (let t = 0; t < this.triRegions.length; t++) {
      if (this.triRegions[t] === target) out.push(t);
    }
    return out;
  }

  /** Per-region triangle counts. Diagnostic helper. */
  regionStats() {
    const counts = new Uint32Array(REGIONS.length);
    for (let t = 0; t < this.triRegions.length; t++) counts[this.triRegions[t]]++;
    const out = {};
    for (let i = 0; i < REGIONS.length; i++) out[REGIONS[i]] = counts[i];
    return out;
  }

  /** Total triangle count. */
  get triangleCount() { return this.tris.triCount; }
}
