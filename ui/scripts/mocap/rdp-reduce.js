// ui/scripts/mocap/rdp-reduce.js
// Ramer-Douglas-Peucker keyframe reduction for motion data

function pointLineDistance(p, a, b) {
  const ab = a.map((v, i) => b[i] - v);
  const ap = a.map((v, i) => p[i] - v);
  const abLen2 = ab.reduce((s, v) => s + v * v, 0);
  if (abLen2 < 1e-12) {
    return Math.sqrt(ap.reduce((s, v) => s + v * v, 0));
  }
  const t = Math.max(0, Math.min(1, ap.reduce((s, v, i) => s + v * ab[i], 0) / abLen2));
  const proj = a.map((v, i) => v + t * ab[i]);
  const diff = p.map((v, i) => v - proj[i]);
  return Math.sqrt(diff.reduce((s, v) => s + v * v, 0));
}

function rdpIndices(points, epsilon) {
  if (points.length <= 2) return points.map((_, i) => i);

  let maxDist = 0;
  let maxIdx = 0;
  const first = points[0];
  const last = points[points.length - 1];

  for (let i = 1; i < points.length - 1; i++) {
    const d = pointLineDistance(points[i], first, last);
    if (d > maxDist) { maxDist = d; maxIdx = i; }
  }

  if (maxDist > epsilon) {
    const left = rdpIndices(points.slice(0, maxIdx + 1), epsilon);
    const right = rdpIndices(points.slice(maxIdx), epsilon).map(i => i + maxIdx);
    return [...left.slice(0, -1), ...right];
  }
  return [0, points.length - 1];
}

export function reduceFrames(frames, epsilon = 2) {
  if (frames.length <= 2) {
    return frames.map((f, i) => ({
      t: frames.length === 1 ? 0 : i,
      bones: { ...f.bones },
    }));
  }

  const duration = frames[frames.length - 1].timestamp_ms - frames[0].timestamp_ms;
  if (duration <= 0) return [{ t: 0, bones: { ...frames[0].bones } }];

  const allBones = new Set();
  for (const f of frames) {
    for (const name of Object.keys(f.bones)) allBones.add(name);
  }

  const keptIndices = new Set([0, frames.length - 1]);

  for (const boneName of allBones) {
    const points = frames.map(f => {
      const r = f.bones[boneName];
      return r ? [r[0], r[1], r[2]] : [0, 0, 0];
    });

    const indices = rdpIndices(points, epsilon);
    for (const idx of indices) keptIndices.add(idx);
  }

  const sorted = [...keptIndices].sort((a, b) => a - b);

  return sorted.map(idx => {
    const f = frames[idx];
    const t = (f.timestamp_ms - frames[0].timestamp_ms) / duration;
    return {
      t: Math.round(t * 1000) / 1000,
      bones: { ...f.bones },
    };
  });
}
