/**
 * pose-resolver.js — resolves semantic pose primitives against a VRM's
 * body substrate (BodyMesh + BodyAtlas) to produce concrete bone state
 * + IK targets + finger styles.
 *
 * Input contract: pose primitive JSON per
 *   docs/superpowers/specs/2026-05-14-pose-primitive-schema.md
 *
 * Output contract:
 *   {
 *     bones: { boneName: [eulerX, eulerY, eulerZ] }   // radians
 *     handTargets: { L: [x,y,z]|null, R: [x,y,z]|null } // world-space
 *     fingerStyles: { L: string|null, R: string|null }
 *     diagnostics: { valid, anchors, issues }
 *   }
 *
 * The resolver is the foundation for cross-VRM portability: it queries
 * the live VRM's mesh + atlas to find anatomy-relative target positions,
 * so the same primitive produces correct results on every VRM.
 *
 * THREE is injected via constructor — same pattern as other substrate
 * modules. No module-level dependencies.
 */

const DEG2RAD = Math.PI / 180;

// Axis convention for body rotations (pitch/yaw/roll → bone Euler).
// pitch = X (positive = forward nod), yaw = Y (positive = turn left),
// roll = Z (positive = right-ear-down tilt). Matches the convention
// used by POSE_PRESETS in avatar-pose-presets.js.
function bodyRotationToEuler(rot) {
  const pitch = (rot.pitch ?? 0) * DEG2RAD;
  const yaw   = (rot.yaw   ?? 0) * DEG2RAD;
  const roll  = (rot.roll  ?? 0) * DEG2RAD;
  return [pitch, yaw, roll];
}

// Spine convenience: forwardLean → pitch, twist → yaw.
function spineRotationToEuler(rot) {
  const pitch = ((rot.pitch ?? rot.forwardLean ?? 0)) * DEG2RAD;
  const yaw   = ((rot.yaw   ?? rot.twist       ?? 0)) * DEG2RAD;
  const roll  = ((rot.roll  ?? 0)) * DEG2RAD;
  return [pitch, yaw, roll];
}

export class PoseResolver {
  /**
   * @param {object} opts
   * @param {object} opts.three   THREE namespace
   * @param {object} opts.vrm     VRM with humanoid + augmentumBodyMesh
   * @param {object} [opts.bodyMesh]   override (defaults to vrm.__augmentumBodyMesh)
   * @param {object} [opts.bodyAtlas]  override (defaults to vrm.__augmentumBodyAtlas)
   */
  constructor({ three, vrm, bodyMesh = null, bodyAtlas = null }) {
    if (!three) throw new Error('PoseResolver requires opts.three');
    if (!vrm?.humanoid) throw new Error('PoseResolver requires opts.vrm with humanoid');
    this.three = three;
    this.vrm = vrm;
    this.bodyMesh = bodyMesh || vrm.__augmentumBodyMesh;
    this.bodyAtlas = bodyAtlas || vrm.__augmentumBodyAtlas;
    if (!this.bodyMesh) {
      console.warn('[pose-resolver] no BodyMesh on VRM — resolution will fail for region anchors');
    }
  }

  /**
   * Resolve a pose primitive against the bound VRM.
   *
   * @param {object} primitive    pose primitive JSON
   * @returns {object}            resolved pose (see file header)
   */
  resolve(primitive) {
    const result = {
      bones: {},
      handTargets: { L: null, R: null },
      fingerStyles: { L: null, R: null },
      diagnostics: { valid: true, anchors: {}, issues: [], primitive: primitive?.name },
    };

    if (!primitive || primitive.schema !== 'augmentum.pose-primitive.v1') {
      result.diagnostics.valid = false;
      result.diagnostics.issues.push('missing or wrong schema');
      return result;
    }

    // ─── Validity gates ────────────────────────────────────────────────
    // Required-bone checks are SOFT by default: a missing optional bone
    // (jaw, leftEye, rightEye, toes) usually means landmark/region
    // resolution falls back to a head-relative estimate, and the
    // affordance vocabulary doesn't depend on those bones. So we surface
    // missing bones as warnings rather than invalidating the resolution.
    // Primitives that genuinely won't work without a bone should declare
    // it with the `strict` modifier (e.g. ["jaw!"] — trailing '!').
    const requires = primitive.validity?.requires || [];
    for (const entry of requires) {
      if (entry === 'humanoid') continue;
      const strict = entry.endsWith('!');
      const boneName = strict ? entry.slice(0, -1) : entry;
      const node = this.vrm.humanoid.getNormalizedBoneNode?.(boneName);
      if (!node) {
        if (strict) {
          result.diagnostics.issues.push(`VRM missing required bone (strict): ${boneName}`);
          result.diagnostics.valid = false;
        } else {
          result.diagnostics.issues.push(`VRM missing optional bone (soft): ${boneName} — using fallback`);
        }
      }
    }
    const profileGate = primitive.validity?.vrmAxisProfiles;
    if (profileGate) {
      const profile = this.vrm.__augmentumCompatibilityProfile?.armAxisProfile;
      if (profile && !profileGate.includes(profile)) {
        result.diagnostics.issues.push(`VRM arm profile '${profile}' not in allowed list ${JSON.stringify(profileGate)}`);
        // Soft-fail: don't kill the resolve, just flag
      }
    }

    // ─── Anchors ──────────────────────────────────────────────────────
    for (const [anchorKey, anchor] of Object.entries(primitive.anchors || {})) {
      const side = anchorKey === 'leftHand' ? 'L'
                 : anchorKey === 'rightHand' ? 'R'
                 : null;
      if (!side) {
        result.diagnostics.issues.push(`unknown anchor key: ${anchorKey} (expected leftHand/rightHand)`);
        continue;
      }
      const resolved = this._resolveAnchor(anchor, side);
      result.diagnostics.anchors[anchorKey] = resolved;
      if (resolved?.position) {
        result.handTargets[side] = resolved.position;
      }
      // Per-anchor finger style override (anchor wins over top-level)
      if (anchor.fingerStyle) {
        result.fingerStyles[side] = anchor.fingerStyle;
      }
    }

    // ─── Top-level finger styles ─────────────────────────────────────
    if (primitive.fingerStyles) {
      for (const side of ['L', 'R']) {
        if (primitive.fingerStyles[side] && !result.fingerStyles[side]) {
          result.fingerStyles[side] = primitive.fingerStyles[side];
        }
      }
    }

    // ─── Body rotations ──────────────────────────────────────────────
    const body = primitive.body || {};
    if (body.head)       result.bones.head       = bodyRotationToEuler(body.head);
    if (body.neck)       result.bones.neck       = bodyRotationToEuler(body.neck);
    if (body.spine)      result.bones.spine      = spineRotationToEuler(body.spine);
    if (body.chest)      result.bones.chest      = spineRotationToEuler(body.chest);
    if (body.upperChest) result.bones.upperChest = spineRotationToEuler(body.upperChest);
    if (body.hips)       result.bones.hips       = bodyRotationToEuler(body.hips);

    return result;
  }

  /**
   * Resolve a single anchor to a world-space target position + region tag.
   * Dispatches by anchor.type.
   * @private
   */
  _resolveAnchor(anchor, side) {
    if (!anchor || !anchor.type) {
      return { error: 'anchor missing type', anchor };
    }
    switch (anchor.type) {
      case 'region-contact': return this._resolveRegionContact(anchor, side);
      case 'rest':           return this._resolveRest(anchor, side);
      case 'landmark-relative': return this._resolveLandmark(anchor, side);
      case 'world-target':   return this._resolveWorldTarget(anchor, side);
      case 'free-space':     return { position: null, kind: 'free-space' };
      default:
        return { error: `unknown anchor type: ${anchor.type}`, anchor };
    }
  }

  _resolveRegionContact(anchor, side) {
    if (!anchor.region) return { error: 'region-contact missing region' };
    const lm = this._regionPoint(anchor.region);
    if (!lm) return { error: `region '${anchor.region}' not resolvable on this VRM` };
    const pos = this._applyOffset(lm.point, lm.normal, anchor.offset);
    return {
      position: pos,
      region: lm.region || anchor.region,
      normal: lm.normal,
      surfacePoint: lm.point,
      tolerance: anchor.tolerance ?? 0.02,
      palmFacing: anchor.palmFacing || null,
      kind: 'region-contact',
    };
  }

  _resolveRest(anchor, side) {
    const region = anchor.region || (side === 'L' ? 'hip_L' : 'hip_R');
    const lm = this._regionPoint(region);
    if (!lm) {
      // Best-effort fallback: hip bone position with downward offset
      const hipBoneName = side === 'L' ? 'leftUpperLeg' : 'rightUpperLeg';
      const node = this.vrm.humanoid.getNormalizedBoneNode?.(hipBoneName);
      if (!node) return { error: `rest region '${region}' not resolvable, no hip bone fallback` };
      const p = node.getWorldPosition(new this.three.Vector3());
      return {
        position: [p.x, p.y, p.z],
        region: 'fallback',
        tolerance: anchor.tolerance ?? 0.04,
        kind: 'rest',
      };
    }
    const pos = this._applyOffset(lm.point, lm.normal, anchor.offset);
    return {
      position: pos,
      region: lm.region || region,
      normal: lm.normal,
      surfacePoint: lm.point,
      tolerance: anchor.tolerance ?? 0.04,
      kind: 'rest',
    };
  }

  _resolveLandmark(anchor, side) {
    if (!anchor.landmark) return { error: 'landmark-relative missing landmark' };
    const lm = this.bodyMesh?.landmark(anchor.landmark);
    if (!lm) return { error: `landmark '${anchor.landmark}' not present on this VRM` };
    const pos = this._applyOffset(lm.point, lm.normal, anchor.offset);
    return {
      position: pos,
      region: lm.region,
      normal: lm.normal,
      surfacePoint: lm.point,
      tolerance: anchor.tolerance ?? 0.015,
      kind: 'landmark-relative',
    };
  }

  _resolveWorldTarget(anchor, side) {
    if (!anchor.position || anchor.position.length !== 3) {
      return { error: 'world-target requires position [x,y,z]' };
    }
    let pos = [...anchor.position];
    if (anchor.frame === 'hipsRelative') {
      const hips = this.vrm.humanoid.getNormalizedBoneNode?.('hips')
        ?.getWorldPosition(new this.three.Vector3());
      if (hips) {
        pos = [pos[0] + hips.x, pos[1] + hips.y, pos[2] + hips.z];
      }
    }
    return {
      position: pos,
      region: null,
      tolerance: anchor.tolerance ?? 0.03,
      kind: 'world-target',
    };
  }

  /**
   * Get a representative surface point for a region name. Tries the
   * named landmark first (exact, surface-snapped), then falls back to
   * the region's triangle centroid via BodyMesh.
   * @private
   */
  _regionPoint(regionName) {
    if (!this.bodyMesh) return null;
    // Try named landmark first
    const lm = this.bodyMesh.landmark(regionName);
    if (lm) return lm;
    // Fall back to region centroid
    const triIds = this.bodyMesh.trianglesInRegion(regionName);
    if (!triIds.length) return null;
    const verts = this.bodyMesh.tris.verts;
    const normals = this.bodyMesh.tris.normals;
    let sx = 0, sy = 0, sz = 0, nx = 0, ny = 0, nz = 0;
    for (const t of triIds) {
      sx += (verts[t*9+0] + verts[t*9+3] + verts[t*9+6]) / 3;
      sy += (verts[t*9+1] + verts[t*9+4] + verts[t*9+7]) / 3;
      sz += (verts[t*9+2] + verts[t*9+5] + verts[t*9+8]) / 3;
      nx += normals[t*3+0]; ny += normals[t*3+1]; nz += normals[t*3+2];
    }
    const n = triIds.length;
    const nMag = Math.hypot(nx, ny, nz) || 1;
    return {
      point: [sx/n, sy/n, sz/n],
      normal: [nx/nMag, ny/nMag, nz/nMag],
      region: regionName,
      triangleCount: n,
    };
  }

  /**
   * Apply an offset to a surface point. By convention, offset[2] is
   * interpreted as a normal-direction push (outward from body surface);
   * offset[0]/[1] are absolute world deltas. Future versions may add a
   * tangent-plane offset mode.
   * @private
   */
  _applyOffset(point, normal, offset) {
    if (!offset) return [...point];
    const dx = offset[0] || 0;
    const dy = offset[1] || 0;
    const dn = offset[2] || 0;   // interpreted as along-normal when normal is present
    if (normal && dn !== 0) {
      return [
        point[0] + dx + normal[0] * dn,
        point[1] + dy + normal[1] * dn,
        point[2] + normal[2] * dn,
      ];
    }
    return [point[0] + dx, point[1] + dy, point[2] + (offset[2] || 0)];
  }

  /**
   * Convenience: resolve + apply in one call. Modifies the VRM directly.
   * Returns the resolved pose dict for inspection.
   *
   * @param {object} primitive
   * @param {object} opts
   * @param {object} opts.ik             AvatarIK instance (optional)
   * @param {object} opts.applier        AvatarAffordanceApplier instance (optional)
   * @param {object} opts.restQuats      bind-rest quaternions (vrm.__augmentumBoneRestQuats)
   * @returns {object}                   resolved pose
   */
  apply(primitive, { ik = null, applier = null, restQuats = null } = {}) {
    const resolved = this.resolve(primitive);

    // Body bone rotations: compose as rest * delta
    const rest = restQuats || this.vrm.__augmentumBoneRestQuats;
    const tmpEuler = new this.three.Euler();
    const tmpQuat = new this.three.Quaternion();
    for (const [boneName, eulerArr] of Object.entries(resolved.bones)) {
      const node = this.vrm.humanoid.getNormalizedBoneNode?.(boneName);
      if (!node) continue;
      tmpEuler.set(eulerArr[0], eulerArr[1], eulerArr[2], 'XYZ');
      tmpQuat.setFromEuler(tmpEuler);
      if (rest?.[boneName]) {
        node.quaternion.copy(rest[boneName]).multiply(tmpQuat);
      } else {
        node.quaternion.copy(tmpQuat);
      }
    }

    // Per-side arm application. The affordance vocabulary entries are
    // FULL arm-chain poses (shoulder → fingers, 18 bones), not finger-
    // curl overlays. So if a side has an assigned fingerStyle, that
    // affordance IS the arm motion — apply it directly. IK is only a
    // fallback for sides with anchors but no affordance (rare in v1).
    //
    // The anchor's `region` field is semantic metadata for the affordance
    // (for VRM-portable affordance selection: "find an affordance that
    // lands in region X"). It is NOT a runtime IK target unless no
    // affordance is provided.
    for (const side of ['L', 'R']) {
      const style = resolved.fingerStyles[side];
      if (style && applier && typeof applier.setHandPose === 'function') {
        try {
          applier.setHandPose(side, style);
        } catch (err) {
          resolved.diagnostics.issues.push(`affordance '${style}' failed: ${err.message}`);
          // Fall through to IK below
        }
        continue;
      }
      // IK fallback: only fires when an anchor specified a hand target
      // but no affordance was assigned. The wrist lands at the resolved
      // surface point; hand orientation is uncontrolled. Useful only
      // for free-space/world-target anchors in v1.
      if (ik) {
        const t = resolved.handTargets[side];
        if (t && typeof ik.setHandPositionWorld === 'function') {
          ik.setHandPositionWorld(side, t);
        }
      }
    }

    return resolved;
  }
}
