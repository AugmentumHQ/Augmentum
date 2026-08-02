/**
 * avatar-thumbnail-render.js — offscreen VRM headshot renderer.
 *
 * Spins up a dedicated WebGL canvas (preserveDrawingBuffer: true), loads
 * the VRM, frames head+shoulders, takes one render pass, returns a PNG
 * blob. Used by:
 *
 *   - settings.js avatar grid (tile thumbnails)
 *   - avatar-thumbnail.js (companion summon pip portrait)
 *
 * Independent of any live avatar pipeline — the shared avatar.js
 * renderer runs preserveDrawingBuffer:false for perf, which means
 * sampling it post-frame yields a transparent buffer most of the time.
 * Renders here are guaranteed-readable because we own the GL context.
 */

const SIZE = 160;
const _OPTS_DEFAULT = Object.freeze({});

/**
 * Render a head-framed PNG thumbnail of a VRM.
 *
 * @param {string} vrmUrl       URL of the .vrm file.
 * @param {object} [opts]       Per-avatar render hints.
 * @param {number} [opts.faceRotationY]  Explicit Y rotation; falls
 *   back to the arm-axis mirror heuristic when omitted.
 * @returns {Promise<Blob|null>}  PNG blob, or null on failure.
 */
export async function renderVRMThumbnail(vrmUrl, opts = _OPTS_DEFAULT) {
  const [THREE, { GLTFLoader }, { VRMLoaderPlugin, VRMUtils }] = await Promise.all([
    import('../lib/three/three.module.min.js'),
    import('../lib/three/GLTFLoader.js'),
    import('../lib/three-vrm/three-vrm.module.min.js'),
  ]);

  const scene = new THREE.Scene();
  scene.background = null;

  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);

  const canvas = document.createElement('canvas');
  canvas.width = SIZE;
  canvas.height = SIZE;
  const gl = canvas.getContext('webgl2', { alpha: true, antialias: true, preserveDrawingBuffer: true })
          || canvas.getContext('webgl', { alpha: true, antialias: true, preserveDrawingBuffer: true });
  const renderer = new THREE.WebGLRenderer({ canvas, context: gl, alpha: true, antialias: true });
  renderer.setSize(SIZE, SIZE);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  // 3-point lighting (matches voice call scene).
  const keyLight = new THREE.DirectionalLight(0xfff5e6, 1.0);
  keyLight.position.set(1, 1.5, -1);
  scene.add(keyLight);
  const fillLight = new THREE.DirectionalLight(0xe6f0ff, 0.4);
  fillLight.position.set(-1, 1, -0.5);
  scene.add(fillLight);
  const rimLight = new THREE.DirectionalLight(0xffeedd, 0.6);
  rimLight.position.set(0, 0.5, 1);
  scene.add(rimLight);
  scene.add(new THREE.AmbientLight(0xffffff, 0.15));

  const loader = new GLTFLoader();
  loader.register(p => new VRMLoaderPlugin(p));

  let vrm;
  try {
    vrm = await new Promise((resolve, reject) => {
      loader.load(vrmUrl, gltf => {
        const v = gltf.userData.vrm;
        if (!v) { reject(new Error('No VRM data')); return; }
        if (VRMUtils) {
          try { VRMUtils.removeUnnecessaryVertices(gltf.scene); } catch { /* ok */ }
          try { VRMUtils.removeUnnecessaryJoints(gltf.scene); } catch { /* ok */ }
        }
        try { v.humanoid?.resetNormalizedPose?.(); } catch { /* ok */ }
        try {
          v.scene.updateMatrixWorld(true);
          // mannerisms.face_rotation_y is an explicit override — when set,
          // that exact value is the rotation. Otherwise fall back to the
          // arm-axis heuristic. (Stacking the two produced 2π = 0 for
          // avatars where both fire.)
          let rotY;
          if (typeof opts.faceRotationY === 'number' && Number.isFinite(opts.faceRotationY)) {
            rotY = opts.faceRotationY;
          } else {
            rotY = _hasMirroredAvatarArmAxis(v, THREE) ? Math.PI : 0;
          }
          if (rotY) v.scene.rotation.y = rotY;
        } catch { /* ok */ }
        resolve(v);
      }, undefined, reject);
    });
  } catch (err) {
    try { renderer.dispose(); } catch { /* ok */ }
    try { gl?.getExtension('WEBGL_lose_context')?.loseContext(); } catch { /* ok */ }
    throw err;
  }

  scene.add(vrm.scene);
  vrm.update(0);

  // Auto-frame head + shoulders.
  const box = new THREE.Box3().setFromObject(vrm.scene);
  const bSize = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(bSize);
  box.getCenter(center);
  const headY = box.max.y;
  const frameTop = headY + bSize.y * 0.05;
  const frameBottom = headY - bSize.y * 0.35;
  const frameCenterY = (frameTop + frameBottom) / 2;
  const fovRad = camera.fov * Math.PI / 180;
  const dist = ((frameTop - frameBottom) / 2) / Math.tan(fovRad / 2) * 1.1;
  camera.position.set(center.x, frameCenterY, center.z - dist);
  camera.lookAt(center.x, frameCenterY, center.z);

  renderer.render(scene, camera);
  const blob = await new Promise(r => canvas.toBlob(r, 'image/png'));

  scene.remove(vrm.scene);
  vrm.scene.traverse(obj => {
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
      mats.forEach(m => { if (m.map) m.map.dispose(); m.dispose(); });
    }
  });
  renderer.dispose();
  try { gl?.getExtension('WEBGL_lose_context')?.loseContext(); } catch { /* ok */ }

  return blob;
}

// VRMs from different exporters disagree on the arm-axis convention.
// When `leftLower.localToParent(leftUpper)` has positive X and the
// right side has negative X, the VRM faces backwards relative to our
// camera anchor — rotate Y by π so head-framing actually shows the face.
function _hasMirroredAvatarArmAxis(vrm, THREE) {
  const humanoid = vrm?.humanoid;
  if (!humanoid) return false;

  const leftUpper = humanoid.getNormalizedBoneNode?.('leftUpperArm')
    || humanoid.getRawBoneNode?.('leftUpperArm');
  const leftLower = humanoid.getNormalizedBoneNode?.('leftLowerArm')
    || humanoid.getRawBoneNode?.('leftLowerArm');
  const rightUpper = humanoid.getNormalizedBoneNode?.('rightUpperArm')
    || humanoid.getRawBoneNode?.('rightUpperArm');
  const rightLower = humanoid.getNormalizedBoneNode?.('rightLowerArm')
    || humanoid.getRawBoneNode?.('rightLowerArm');
  if (!leftUpper || !leftLower || !rightUpper || !rightLower) return false;

  const left = leftUpper.worldToLocal(leftLower.getWorldPosition(new THREE.Vector3()));
  const right = rightUpper.worldToLocal(rightLower.getWorldPosition(new THREE.Vector3()));
  return left.x > 0.001 && right.x < -0.001;
}
