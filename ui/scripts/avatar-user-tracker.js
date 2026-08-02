/**
 * avatar-user-tracker.js — User-awareness substrate for VR.
 *
 * Publishes the user's head position + orientation + seat-relative
 * derivatives each frame so avatar subsystems (lookAt, reach IK,
 * proximity triggers, gaze reactions) can subscribe without re-deriving
 * from camera/gizmo state.
 *
 * Two source modes:
 *   - VR: head pose from the XR camera (which the headset writes each
 *     frame via local-floor reference space). Head forward is the live
 *     gaze direction.
 *   - Desktop / authoring: head pose from a seat-anchored gizmo. No real
 *     head-tracking data, so head orientation is approximated as the
 *     seat's facing direction.
 *
 * Designed to be the single read API for "where is the user." Subscribers
 * poll per-frame; no event system — the animation loop already polls
 * everything else.
 *
 * Derived signals (foundation for Phase 6 proximity reactions):
 *   - headRelativeToSeat: head world position minus seat anchor.
 *   - leanDelta: projection of headRelativeToSeat onto seat-forward.
 *     Positive = leaning forward, negative = leaning back.
 *   - seatLateralDelta: projection onto seat-right. Positive = leaning
 *     toward seatmate (when seatmate is to the right of the user).
 *
 * THREE namespace is dependency-injected because consumers live in
 * different module-resolution worlds (CDN three for ui/mockups/scene-
 * test.html, vendored three for production ui/scripts/avatar.js).
 */

export class UserHeadTracker {
  constructor(THREE) {
    if (!THREE) throw new Error('UserHeadTracker requires THREE');
    this.THREE = THREE;

    // Published per-frame state. Consumers should read these AFTER
    // calling one of the update* methods this frame.
    this.headPosition = new THREE.Vector3();
    this.headQuaternion = new THREE.Quaternion();
    this.headForward = new THREE.Vector3(0, 0, -1);

    this.seatPosition = new THREE.Vector3();
    this.seatForward = new THREE.Vector3(0, 0, -1);
    this.seatRight = new THREE.Vector3(1, 0, 0);

    this.headRelativeToSeat = new THREE.Vector3();

    this.isVR = false;
    this.isActive = false;
    this.lastUpdateClock = 0; // performance.now() at last update; debug aid

    // Reusable scratch — avoid per-frame allocations.
    this._tmpV = new THREE.Vector3();
    this._tmpQ = new THREE.Quaternion();
    this._upAxis = new THREE.Vector3(0, 1, 0);
    this._forwardLocal = new THREE.Vector3(0, 0, -1);
    this._rightLocal = new THREE.Vector3(1, 0, 0);
  }

  /**
   * Update from an XR camera (which the headset drives each frame).
   * Call once per frame while a VR session is active.
   *
   * @param {Object3D} camera - the XR-driven camera; getWorldPosition /
   *   getWorldQuaternion / getWorldDirection must be valid.
   * @param {{x:number,y:number,z:number,rotY:number}} seat - the seat
   *   anchor (the same coords used to build the XR rig).
   */
  updateFromXR(camera, seat) {
    camera.getWorldPosition(this.headPosition);
    camera.getWorldQuaternion(this.headQuaternion);
    camera.getWorldDirection(this.headForward);

    this._populateSeatFrame(seat);
    this.headRelativeToSeat.copy(this.headPosition).sub(this.seatPosition);

    this.isVR = true;
    this.isActive = true;
    this.lastUpdateClock = performance.now();
  }

  /**
   * Update from a desktop-mode seat gizmo. Used by the bench for pose
   * authoring with the user's position visible as a proxy. No real
   * head-tracking data, so head orientation == seat orientation.
   *
   * @param {Object3D} seatGroup - the seat anchor group (rotated by seat
   *   rotY).
   * @param {Object3D} headMesh - the head proxy mesh inside the group.
   * @param {{x:number,y:number,z:number,rotY:number}} seat
   */
  updateFromGizmo(seatGroup, headMesh, seat) {
    headMesh.getWorldPosition(this.headPosition);
    seatGroup.getWorldQuaternion(this.headQuaternion);

    this._populateSeatFrame(seat);
    // Without real head-tracking, head forward == seat forward.
    this.headForward.copy(this.seatForward);
    this.headRelativeToSeat.copy(this.headPosition).sub(this.seatPosition);

    this.isVR = false;
    this.isActive = true;
    this.lastUpdateClock = performance.now();
  }

  /** Mark inactive — call when neither source is providing data. */
  setInactive() {
    this.isActive = false;
  }

  /** Distance from user head to a given world point. */
  distanceTo(worldPoint) {
    return this.headPosition.distanceTo(worldPoint);
  }

  /**
   * How far the user's head has leaned forward of the seat anchor along
   * the seat's facing direction.
   *
   *   > 0   leaning forward (toward whatever the seat faces)
   *   = 0   neutral
   *   < 0   leaning back into the couch
   *
   * Foundation for Phase 6 proximity reactions ("user leans in →
   * avatar leans in").
   */
  leanDelta() {
    return this.headRelativeToSeat.dot(this.seatForward);
  }

  /**
   * Lateral lean along the seat's right axis. Positive = leaned to seat-
   * right (toward seatmate on the right), negative = seat-left.
   */
  lateralDelta() {
    return this.headRelativeToSeat.dot(this.seatRight);
  }

  /**
   * Did the user's head forward vector pass within `radius` of a target
   * world point? Cheap "is the user looking at this?" check — projects
   * the target onto the head-forward ray and tests distance.
   *
   * @returns {boolean}
   */
  isGazingAt(targetWorld, radius = 0.20) {
    this._tmpV.copy(targetWorld).sub(this.headPosition);
    const along = this._tmpV.dot(this.headForward);
    if (along <= 0) return false; // target is behind the head
    // Perpendicular distance from target to the head-forward ray.
    this._tmpV.addScaledVector(this.headForward, -along);
    return this._tmpV.lengthSq() <= radius * radius;
  }

  // ── Internals ──────────────────────────────────────────────────────
  _populateSeatFrame(seat) {
    this.seatPosition.set(seat.x, seat.y, seat.z);
    // Seat forward: -Z rotated around the world up axis by rotY. Stable
    // regardless of head tilt (so lean detection isn't polluted by the
    // user nodding).
    this.seatForward.copy(this._forwardLocal).applyAxisAngle(this._upAxis, seat.rotY);
    this.seatRight.copy(this._rightLocal).applyAxisAngle(this._upAxis, seat.rotY);
  }
}
