/**
 * body-atlas.js — runtime querier for Augmentum Body Atlas
 *
 * Loads a precomputed .body-atlas JSON file and exposes spatial queries:
 *   sdf, region, normal, gradient, contactCheck, surface point sampling,
 *   anchor lookup, world↔body frame conversion.
 *
 * Sign convention: SDF < 0 inside the body, SDF > 0 outside.
 * ∇SDF (gradient) points OUTWARD — i.e. the direction to push a joint
 * out of the body if it has clipped in.
 *
 * Storage on disk is sparse — only voxels within ±activeBand of the surface
 * carry data. The querier hydrates those into a dense Float32Array on load
 * (faster random-access lookups, ~10MB peak resident for a Becca-scale atlas).
 *
 * Frame convention:
 *   - Atlas was baked at the VRM's T-pose with the avatar at native rotation.
 *   - bbox.origin and voxel positions are stored in the WORLD frame at bake.
 *   - At runtime, queries arrive in world space; we transform through the
 *     hips bone's current pose to recover the bake-time frame.
 *
 *   query_in_bake_frame = Q_hips_inverse * (query_world - hips_world)
 *                          + hips_world_at_bake
 *
 * THREE is injected for vector/quaternion math (same pattern as avatar-ik.js).
 */

export class BodyAtlas {
  /** url -> Promise<BodyAtlas>. Shared across every avatar surface so the
   *  large atlas is fetched + parsed once, not once per surface. */
  static _loadCache = new Map();

  /**
   * Load and parse a .body-atlas JSON file.
   *
   * Deduped + cached per URL: these atlases are large (50-60 MB raw) and the
   * SAME file is requested by every avatar surface that mounts a given VRM
   * (chat, presence, voice, XR, animator, …) — without coalescing, 4+ surfaces
   * each fire a full concurrent download before the browser HTTP cache
   * populates (observed live: /poses/body-atlas-becca.json fetched 4x @ ~3.5s).
   * The atlas is a read-only spatial querier, so one parsed instance is safely
   * shared across all consumers. Failures aren't cached (transient errors can
   * retry; a 404 just re-404s cheaply). Wire transfer is further cut ~4x by the
   * precompressed .json.gz the server serves (scripts/bake_body_atlases.py).
   * @param {string} url
   * @returns {Promise<BodyAtlas>}
   */
  static async load(url) {
    const cached = BodyAtlas._loadCache.get(url);
    if (cached) return cached;
    const p = BodyAtlas._fetchAndParse(url);
    BodyAtlas._loadCache.set(url, p);
    // Don't cache a rejection — let the next caller retry.
    p.catch(() => {
      if (BodyAtlas._loadCache.get(url) === p) BodyAtlas._loadCache.delete(url);
    });
    return p;
  }

  static async _fetchAndParse(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`body-atlas fetch failed: ${res.status} (${url})`);
    const data = await res.json();
    return new BodyAtlas(data);
  }

  /**
   * Construct from an already-parsed atlas object (e.g. tests).
   */
  constructor(data) {
    if (data.schema !== 'augmentum.body-atlas.v1') {
      console.warn(`[body-atlas] unexpected schema: ${data.schema}`);
    }
    this.raw = data;
    this.regionTable = data.regionTable;
    this.touchabilityDefaults = data.touchabilityDefaults;
    this.bbox = data.bbox;
    this.dimX = data.bbox.dims[0];
    this.dimY = data.bbox.dims[1];
    this.dimZ = data.bbox.dims[2];
    this.h = data.bbox.voxelSize;
    this.origin = data.bbox.origin;
    this.skeletonHeight = data.skeletonHeight;
    this.bakeHipsPos = data.hips?.worldPos || [0, 0, 0];
    this.bakeHipsQuat = data.hips?.worldQuat || [0, 0, 0, 1];
    // Whether the bake recorded a trustworthy hips pose. Older atlases baked
    // before the hips block existed fall back to identity framing (legacy
    // raw-world queries) so nothing regresses. See frame() / BodyFrame.
    this.hasBakeFrame = !!(data.hips && data.hips.worldPos && data.hips.worldQuat);

    // Hydrate sparse voxel arrays into dense lookup tables.
    // For voxels outside the active band, we keep an "outside marker" sdf
    // value that's beyond any meaningful threshold so contact / collision
    // queries automatically miss.
    const total = this.dimX * this.dimY * this.dimZ;
    this.OUTSIDE_FAR = 999.0;   // sentinel for inactive voxels
    this.sdfDense = new Float32Array(total);
    this.sdfDense.fill(this.OUTSIDE_FAR);
    this.regionDense = new Uint8Array(total);
    this.touchabilityDense = new Uint8Array(total);
    this.flagsDense = new Uint8Array(total);
    this.normalXDense = new Int8Array(total);
    this.normalYDense = new Int8Array(total);
    this.normalZDense = new Int8Array(total);
    const otherIdx = this.regionTable.indexOf('other');
    this.regionDense.fill(otherIdx >= 0 ? otherIdx : 0);

    const v = data.voxels;
    for (let i = 0; i < v.indices.length; i++) {
      const idx = v.indices[i];
      this.sdfDense[idx] = v.sdf[i];
      this.regionDense[idx] = v.region[i];
      this.touchabilityDense[idx] = v.touchability[i];
      this.flagsDense[idx] = v.flags[i];
      // Pack normal components into int8 (-127..127). Source is float -1..1.
      this.normalXDense[idx] = Math.round(Math.max(-1, Math.min(1, v.normal[i*3+0])) * 127);
      this.normalYDense[idx] = Math.round(Math.max(-1, Math.min(1, v.normal[i*3+1])) * 127);
      this.normalZDense[idx] = Math.round(Math.max(-1, Math.min(1, v.normal[i*3+2])) * 127);
    }

    // Anchors with region resolution
    this.anchors = (data.anchors || []).map((a) => ({
      point: a.point,
      region: a.region,
      curvature: a.curvature || 0,
    }));

    // Index anchors by region for fast lookup
    this.anchorsByRegion = {};
    for (const a of this.anchors) {
      if (!this.anchorsByRegion[a.region]) this.anchorsByRegion[a.region] = [];
      this.anchorsByRegion[a.region].push(a);
    }
  }

  /**
   * Linear voxel index for grid coords (i, j, k).
   * Layout matches generator: z * dimY * dimX + y * dimX + x.
   */
  _linIdx(i, j, k) {
    return (k * this.dimY + j) * this.dimX + i;
  }

  /**
   * Convert world point to integer voxel coords.
   * @returns {[i, j, k] | null} null if outside grid
   */
  _worldToVoxel(p) {
    const i = Math.floor((p[0] - this.origin[0]) / this.h);
    const j = Math.floor((p[1] - this.origin[1]) / this.h);
    const k = Math.floor((p[2] - this.origin[2]) / this.h);
    if (i < 0 || i >= this.dimX || j < 0 || j >= this.dimY || k < 0 || k >= this.dimZ) {
      return null;
    }
    return [i, j, k];
  }

  /**
   * Trilinear-interpolated SDF at a world point.
   * @param {[number, number, number]} p
   * @returns {number} signed distance in meters, OUTSIDE_FAR if out of grid
   */
  sdf(p) {
    const fx = (p[0] - this.origin[0]) / this.h;
    const fy = (p[1] - this.origin[1]) / this.h;
    const fz = (p[2] - this.origin[2]) / this.h;
    const i0 = Math.floor(fx), j0 = Math.floor(fy), k0 = Math.floor(fz);
    if (i0 < 0 || i0 >= this.dimX - 1
     || j0 < 0 || j0 >= this.dimY - 1
     || k0 < 0 || k0 >= this.dimZ - 1) {
      return this.OUTSIDE_FAR;
    }
    const tx = fx - i0, ty = fy - j0, tz = fz - k0;
    const i1 = i0 + 1, j1 = j0 + 1, k1 = k0 + 1;
    const s = this.sdfDense;
    const c000 = s[this._linIdx(i0, j0, k0)];
    const c100 = s[this._linIdx(i1, j0, k0)];
    const c010 = s[this._linIdx(i0, j1, k0)];
    const c110 = s[this._linIdx(i1, j1, k0)];
    const c001 = s[this._linIdx(i0, j0, k1)];
    const c101 = s[this._linIdx(i1, j0, k1)];
    const c011 = s[this._linIdx(i0, j1, k1)];
    const c111 = s[this._linIdx(i1, j1, k1)];
    // Trilinear interp
    const c00 = c000 * (1 - tx) + c100 * tx;
    const c01 = c001 * (1 - tx) + c101 * tx;
    const c10 = c010 * (1 - tx) + c110 * tx;
    const c11 = c011 * (1 - tx) + c111 * tx;
    const c0 = c00 * (1 - ty) + c10 * ty;
    const c1 = c01 * (1 - ty) + c11 * ty;
    return c0 * (1 - tz) + c1 * tz;
  }

  /**
   * SDF gradient via central differences on the voxel grid.
   * Returns the OUTWARD direction (gradient of sdf, which increases outward).
   * Length is ~1.0 for grid-aligned queries; query at finer-than-h-resolution
   * gives sub-voxel gradient via finite differences over h.
   *
   * @returns {[number, number, number]}
   */
  gradient(p) {
    const h = this.h;
    const sx0 = this.sdf([p[0] - h, p[1], p[2]]);
    const sx1 = this.sdf([p[0] + h, p[1], p[2]]);
    const sy0 = this.sdf([p[0], p[1] - h, p[2]]);
    const sy1 = this.sdf([p[0], p[1] + h, p[2]]);
    const sz0 = this.sdf([p[0], p[1], p[2] - h]);
    const sz1 = this.sdf([p[0], p[1], p[2] + h]);
    const gx = (sx1 - sx0) / (2 * h);
    const gy = (sy1 - sy0) / (2 * h);
    const gz = (sz1 - sz0) / (2 * h);
    const m = Math.sqrt(gx*gx + gy*gy + gz*gz);
    if (m < 1e-6) return [0, 1, 0];   // degenerate near medial axis; default up
    return [gx / m, gy / m, gz / m];
  }

  /**
   * Region label at a world point. Falls back to 'other' if outside grid.
   * Uses nearest-voxel (no interpolation; regions are categorical).
   */
  region(p) {
    const v = this._worldToVoxel(p);
    if (!v) return 'other';
    const idx = this._linIdx(v[0], v[1], v[2]);
    return this.regionTable[this.regionDense[idx]] || 'other';
  }

  /**
   * Touchability prior at a world point. 0..1.
   */
  touchability(p) {
    const v = this._worldToVoxel(p);
    if (!v) return 0;
    return this.touchabilityDense[this._linIdx(v[0], v[1], v[2])] / 255;
  }

  /**
   * Read back the precomputed surface normal at the nearest voxel (int8 decoded).
   * Note: this is the NORMAL stored at gen time (=∇SDF). For runtime gradient
   * computation use gradient() — it's smoother for queries between voxels.
   */
  normal(p) {
    const v = this._worldToVoxel(p);
    if (!v) return [0, 1, 0];
    const idx = this._linIdx(v[0], v[1], v[2]);
    return [
      this.normalXDense[idx] / 127,
      this.normalYDense[idx] / 127,
      this.normalZDense[idx] / 127,
    ];
  }

  /**
   * Full record for a world point: sdf + region + normal + touchability + flags.
   */
  query(p) {
    return {
      sdf: this.sdf(p),
      region: this.region(p),
      normal: this.gradient(p),         // smoother than stored normal
      touchability: this.touchability(p),
      flags: (() => {
        const v = this._worldToVoxel(p);
        if (!v) return 0;
        return this.flagsDense[this._linIdx(v[0], v[1], v[2])];
      })(),
    };
  }

  /**
   * Is the point on the body surface within `eps`?
   * @param {Vec3} p
   * @param {number} eps  meters (default 2cm)
   */
  contactCheck(p, eps = 0.02) {
    const sdf = this.sdf(p);
    if (Math.abs(sdf) > eps) return { inContact: false, distance: sdf };
    return {
      inContact: true,
      distance: sdf,
      region: this.region(p),
      normal: this.gradient(p),
    };
  }

  /**
   * Returns true if the point is INSIDE the body (sdf < 0).
   */
  isInside(p) {
    return this.sdf(p) < 0;
  }

  /**
   * Sample N world points from the surface band (|sdf| < 2cm), optionally
   * restricted to a region and/or filtered by minimum touchability.
   */
  surfacePoints(opts = {}) {
    const { region, minTouchability = 0, n = 20 } = opts;
    const wantRegionIdx = region ? this.regionTable.indexOf(region) : -1;
    const out = [];
    const v = this.raw.voxels;
    const minT255 = Math.round(minTouchability * 255);
    for (let i = 0; i < v.indices.length && out.length < n; i++) {
      if ((v.flags[i] & 0x08) === 0) continue;   // not in surface band
      if (wantRegionIdx >= 0 && v.region[i] !== wantRegionIdx) continue;
      if (v.touchability[i] < minT255) continue;
      const idx = v.indices[i];
      const x = idx % this.dimX;
      const y = Math.floor(idx / this.dimX) % this.dimY;
      const z = Math.floor(idx / (this.dimX * this.dimY));
      out.push([
        this.origin[0] + (x + 0.5) * this.h,
        this.origin[1] + (y + 0.5) * this.h,
        this.origin[2] + (z + 0.5) * this.h,
      ]);
    }
    return out;
  }

  /**
   * Get convex anchors, optionally filtered by region.
   * @returns {Array<{point, region, curvature}>}
   */
  convexAnchors(region = null) {
    if (region) return this.anchorsByRegion[region] || [];
    return this.anchors;
  }

  /**
   * Project a velocity onto the tangent plane of the surface at `p`.
   * Used for dwell motion — keeps a hand sliding ON the body, not into/out of it.
   */
  projectTangent(p, velocity) {
    const n = this.gradient(p);
    const dot = velocity[0]*n[0] + velocity[1]*n[1] + velocity[2]*n[2];
    return [
      velocity[0] - dot * n[0],
      velocity[1] - dot * n[1],
      velocity[2] - dot * n[2],
    ];
  }

  /**
   * Push a point out of the body to a target clearance.
   * If sdf(p) < clearance, walk along the gradient until sdf >= clearance.
   * Used by IK collision response and path planner.
   *
   * @param {Vec3} p   query point
   * @param {number} clearance  desired minimum sdf (≥0)
   * @param {number} maxIters
   * @returns {Vec3} the pushed-out point
   */
  pushOutsideBody(p, clearance = 0.0, maxIters = 8) {
    let curr = [p[0], p[1], p[2]];
    for (let i = 0; i < maxIters; i++) {
      const s = this.sdf(curr);
      if (s >= clearance) break;
      const g = this.gradient(curr);
      const step = clearance - s + this.h * 0.5;
      curr = [curr[0] + g[0]*step, curr[1] + g[1]*step, curr[2] + g[2]*step];
    }
    return curr;
  }

  /**
   * Compute a free-space spline path from `start` to `end` that stays outside
   * the body at clearance ≥ `opts.clearance`. Returns waypoints + a sample()
   * function for parametric access.
   *
   * Algorithm:
   *   1. Generate N straight-line waypoints
   *   2. For each waypoint, push outside body via gradient
   *   3. Iterate until no segment violates clearance (or max iterations)
   *   4. Fit Catmull-Rom spline through waypoints; expose sample(t)
   */
  planPath(start, end, opts = {}) {
    const clearance = opts.clearance ?? 0.05;
    const numWaypoints = opts.waypoints ?? 12;
    const maxIters = opts.maxIters ?? 4;
    let waypoints = [];
    for (let i = 0; i <= numWaypoints; i++) {
      const t = i / numWaypoints;
      waypoints.push([
        start[0] + (end[0] - start[0]) * t,
        start[1] + (end[1] - start[1]) * t,
        start[2] + (end[2] - start[2]) * t,
      ]);
    }
    // Iteratively push waypoints out
    for (let iter = 0; iter < maxIters; iter++) {
      let allOk = true;
      for (let i = 1; i < waypoints.length - 1; i++) {
        const w = waypoints[i];
        const s = this.sdf(w);
        if (s < clearance) {
          waypoints[i] = this.pushOutsideBody(w, clearance);
          allOk = false;
        }
      }
      if (allOk) break;
    }
    // Wrap with Catmull-Rom sampling
    const path = waypoints;
    return {
      waypoints,
      sample: (t) => catmullRom(path, t),
      length: () => {
        let L = 0;
        for (let i = 1; i < path.length; i++) {
          const dx = path[i][0] - path[i-1][0];
          const dy = path[i][1] - path[i-1][1];
          const dz = path[i][2] - path[i-1][2];
          L += Math.sqrt(dx*dx + dy*dy + dz*dz);
        }
        return L;
      },
    };
  }

  /**
   * Build a per-VRM {@link BodyFrame} that maps live world-space queries into
   * this atlas's bake frame (and results back out to world space).
   *
   * The voxel grid is frozen in the avatar's BAKE world pose — hips at
   * `data.hips.worldPos/worldQuat`, scale 1. At runtime the avatar rotates
   * (`vrm.scene.rotation.y`), translates, and is uniformly scaled
   * (`wrapper.scale.setScalar`), so a raw `atlas.sdf([worldXYZ])` lands in the
   * WRONG voxels the moment she turns or is resized. A BodyFrame applies the
   * inverse similarity (rotation + translation + uniform scale) so
   * collision / compliance / IK stay correct in any pose.
   *
   * IMPORTANT: the atlas instance is SHARED across surfaces (the load cache),
   * so the live frame must NOT be stored on the atlas. Each consumer holds its
   * own BodyFrame and rebuilds it each tick from its VRM's current hips.
   *
   * Pass the avatar's CURRENT NORMALIZED-hips world pose (the bake used the
   * normalized hips too — see body-atlas-generator.html) and uniform world
   * scale:
   *
   *   const hips = vrm.humanoid.getNormalizedBoneNode('hips');
   *   const frame = atlas.frame(
   *     hips.getWorldPosition(tmpV).toArray(),
   *     hips.getWorldQuaternion(tmpQ).toArray(),
   *     hips.getWorldScale(tmpS).x);
   *
   * @param {[number,number,number]} curHipsPos      current hips world position
   * @param {[number,number,number,number]} curHipsQuat current hips world quat [x,y,z,w]
   * @param {number} [scale=1] current uniform world scale (bake scale assumed 1)
   * @returns {BodyFrame}
   */
  frame(curHipsPos, curHipsQuat, scale = 1) {
    return new BodyFrame(this, curHipsPos, curHipsQuat, scale);
  }

  /**
   * @deprecated Use {@link frame} — these are identity passthroughs kept only
   * so any old caller doesn't break. The real transform lives on BodyFrame.
   */
  worldToBody(p) { return [...p]; }
  bodyToWorld(p) { return [...p]; }
}

// ─── Quaternion helpers (no THREE dependency — body-atlas.js stays import-free) ──
// Quaternions are [x, y, z, w]; vectors are [x, y, z]. Hamilton product.
function _qnorm(q) {
  const m = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / m, q[1] / m, q[2] / m, q[3] / m];
}
function _qconj(q) { return [-q[0], -q[1], -q[2], q[3]]; }
function _qmul(a, b) {
  const ax = a[0], ay = a[1], az = a[2], aw = a[3];
  const bx = b[0], by = b[1], bz = b[2], bw = b[3];
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}
// Rotate vector v by unit quaternion q (v' = v + 2*qw*(qxyz×v) + 2*qxyz×(qxyz×v)).
function _qrotVec(q, v) {
  const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  const tx = 2 * (qy * v[2] - qz * v[1]);
  const ty = 2 * (qz * v[0] - qx * v[2]);
  const tz = 2 * (qx * v[1] - qy * v[0]);
  return [
    v[0] + qw * tx + (qy * tz - qz * ty),
    v[1] + qw * ty + (qz * tx - qx * tz),
    v[2] + qw * tz + (qx * ty - qy * tx),
  ];
}

/**
 * BodyFrame — a per-VRM view of a (shared, bake-frame) BodyAtlas that accepts
 * and returns WORLD-space coordinates/directions for the avatar's CURRENT pose.
 *
 * Mirrors the atlas query surface (sdf, gradient, region, touchability, normal,
 * contactCheck, isInside, query, pushOutsideBody, projectTangent, surfacePoints,
 * convexAnchors, planPath) but every position in/out is world space and every
 * direction out is world space. SDF distances are returned in world meters
 * (scaled by the avatar's runtime scale) so existing world-meter thresholds
 * (elbowClearance, ACTIVE_RANGE_M, …) keep their meaning.
 *
 * Transform (bake hips R_b,t_b at scale 1; current hips R_n,t_n at scale s):
 *   toBake(p)  = t_b + (1/s) * (R_b * R_n⁻¹) * (p - t_n)
 *   toWorld(q) = t_n +  s     * (R_n * R_b⁻¹) * (q - t_b)
 *   dirToWorld(d) = (R_n * R_b⁻¹) * d           (rotation only)
 */
export class BodyFrame {
  constructor(atlas, curHipsPos, curHipsQuat, scale = 1) {
    this.atlas = atlas;
    if (!atlas.hasBakeFrame) {
      // No trustworthy bake pose → behave exactly like legacy raw-world
      // queries (identity) so an un-rebaked atlas never regresses.
      curHipsPos = atlas.bakeHipsPos;
      curHipsQuat = atlas.bakeHipsQuat;
      scale = 1;
    }
    this._tNow = [curHipsPos[0], curHipsPos[1], curHipsPos[2]];
    this._tBake = atlas.bakeHipsPos;
    this._s = (scale && scale > 1e-6) ? scale : 1;
    this._invS = 1 / this._s;
    const Rnow = _qnorm(curHipsQuat);
    const Rbake = _qnorm(atlas.bakeHipsQuat);
    this._qToBake = _qmul(Rbake, _qconj(Rnow));   // (world - tNow) → bake frame
    this._qToWorld = _qconj(this._qToBake);        // bake → world rotation
  }

  /** World point → bake-frame point (for grid lookup). */
  toBake(p) {
    const v = [p[0] - this._tNow[0], p[1] - this._tNow[1], p[2] - this._tNow[2]];
    const r = _qrotVec(this._qToBake, v);
    return [
      this._tBake[0] + r[0] * this._invS,
      this._tBake[1] + r[1] * this._invS,
      this._tBake[2] + r[2] * this._invS,
    ];
  }

  /** Bake-frame point → world point. */
  toWorld(p) {
    const v = [
      (p[0] - this._tBake[0]) * this._s,
      (p[1] - this._tBake[1]) * this._s,
      (p[2] - this._tBake[2]) * this._s,
    ];
    const r = _qrotVec(this._qToWorld, v);
    return [this._tNow[0] + r[0], this._tNow[1] + r[1], this._tNow[2] + r[2]];
  }

  /** Bake-frame direction → world direction (rotation only; preserves length). */
  dirToWorld(d) { return _qrotVec(this._qToWorld, d); }

  /** Signed distance in WORLD meters (scaled by runtime scale). */
  sdf(p) { return this._s * this.atlas.sdf(this.toBake(p)); }

  /** Outward unit gradient in WORLD space. */
  gradient(p) { return this.dirToWorld(this.atlas.gradient(this.toBake(p))); }

  region(p) { return this.atlas.region(this.toBake(p)); }
  touchability(p) { return this.atlas.touchability(this.toBake(p)); }
  normal(p) { return this.dirToWorld(this.atlas.normal(this.toBake(p))); }
  isInside(p) { return this.atlas.sdf(this.toBake(p)) < 0; }

  query(p) {
    const pb = this.toBake(p);
    const q = this.atlas.query(pb);
    return {
      sdf: this._s * q.sdf,
      region: q.region,
      normal: this.dirToWorld(q.normal),
      touchability: q.touchability,
      flags: q.flags,
    };
  }

  contactCheck(p, eps = 0.02) {
    const sdf = this.sdf(p);
    if (Math.abs(sdf) > eps) return { inContact: false, distance: sdf };
    return { inContact: true, distance: sdf, region: this.region(p), normal: this.gradient(p) };
  }

  /** Push a world point out of the body to a world-meter clearance. */
  pushOutsideBody(p, clearance = 0.0, maxIters = 8) {
    const pushed = this.atlas.pushOutsideBody(this.toBake(p), clearance * this._invS, maxIters);
    return this.toWorld(pushed);
  }

  /** Project a world velocity onto the body's tangent plane at world point p. */
  projectTangent(p, velocity) {
    const n = this.gradient(p);
    const dot = velocity[0] * n[0] + velocity[1] * n[1] + velocity[2] * n[2];
    return [velocity[0] - dot * n[0], velocity[1] - dot * n[1], velocity[2] - dot * n[2]];
  }

  /** Surface sample points, returned in WORLD space. */
  surfacePoints(opts = {}) {
    return this.atlas.surfacePoints(opts).map((pt) => this.toWorld(pt));
  }

  /** Convex anchors with their `point` in WORLD space. */
  convexAnchors(region = null) {
    return this.atlas.convexAnchors(region).map((a) => ({ ...a, point: this.toWorld(a.point) }));
  }

  /** Plan a collision-free path between two WORLD points; world-space output. */
  planPath(start, end, opts = {}) {
    const plan = this.atlas.planPath(this.toBake(start), this.toBake(end), opts);
    return {
      waypoints: plan.waypoints.map((w) => this.toWorld(w)),
      sample: (t) => this.toWorld(plan.sample(t)),
      length: () => this._s * plan.length(),
    };
  }
}

// ─── Internal: Catmull-Rom spline sampling ─────────────────────────────────
function catmullRom(points, t) {
  // t in [0, 1] over the whole path
  const segs = points.length - 1;
  const u = t * segs;
  const seg = Math.min(Math.floor(u), segs - 1);
  const localT = u - seg;
  const p0 = points[Math.max(0, seg - 1)];
  const p1 = points[seg];
  const p2 = points[seg + 1];
  const p3 = points[Math.min(segs, seg + 2)];
  return [
    catmullRom1D(p0[0], p1[0], p2[0], p3[0], localT),
    catmullRom1D(p0[1], p1[1], p2[1], p3[1], localT),
    catmullRom1D(p0[2], p1[2], p2[2], p3[2], localT),
  ];
}
function catmullRom1D(p0, p1, p2, p3, t) {
  const t2 = t*t, t3 = t2*t;
  return 0.5 * (
    (2*p1) +
    (-p0 + p2) * t +
    (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
    (-p0 + 3*p1 - 3*p2 + p3) * t3
  );
}
