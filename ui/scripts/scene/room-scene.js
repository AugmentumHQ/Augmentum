/**
 * room-scene.js — Shared environment layer for Three.js scenes.
 *
 * One source of truth for the modern-room GLB load, starfield, and
 * environment registry. Consumed by ui/mockups/scene-test.html (director),
 * and (planned) the WebXR session module + production avatar.js voice
 * scene. Extracting this means each consumer constructs its own scene/
 * camera/renderer/lighting and lets RoomScene own the environment toggle.
 *
 * Dependency-free at import time. Callers inject THREE + loader classes
 * because consumers live in different module-resolution worlds: scene-test
 * uses CDN three@0.155 via importmap; production avatar.js uses the local
 * vendored bundle at /ui/lib/three/. One file, two import roots.
 */

const ROOM_GLB_URL = '/ui/lib/scenes/modern-room.glb';

// Augmentum's CSP whitelists jsdelivr in connect-src. unpkg works for
// module imports but is blocked for fetch — the Draco decoder uses fetch
// to pull the .wasm/.js worker, so jsdelivr is the only viable host.
const DRACO_DECODER_URL = 'https://cdn.jsdelivr.net/npm/three@0.155.0/examples/jsm/libs/draco/';

/**
 * Environment registry. Each environment is a self-contained authoring
 * preset declaring background, scene props, avatar anchor, recommended
 * camera framing, and the affordances it provides.
 *
 * `provides` is a list of capability tags ('standing', 'sit-couch',
 * 'floor-clearance'). VRMA filtering and pose-affordance gating consume
 * these to only surface motions the active environment supports.
 *
 * `anchors` is the future hook for SpatialSurfaceRegistry — wall/table
 * slots where spatial windows can mount in WebXR mode. Empty today;
 * populated once anchor positions are measured against the GLB.
 */
export const ENVIRONMENTS = {
  none: {
    label: 'None (void)',
    provides: ['standing'],
    backgroundColor: 0x06060e,
    starfield: true,
    avatarDefault: [0, -0.24, 0],
    avatarRotY: Math.PI,
    cameraPosition: [0, 1.45, -2.2],
    cameraTarget:   [0, 1.10,  0],
    ignoreVrmaAvatarPosition: true,
    anchors: [],
  },
  'modern-room': {
    label: 'Modern Room',
    provides: ['standing', 'sit-couch', 'floor-clearance'],
    backgroundColor: 0x06060e,
    starfield: false,
    avatarDefault: [0.75, -0.24, 1.95],
    avatarRotY: 3.11,
    cameraPosition: [-0.76, 1.59, -1.64],
    cameraTarget:   [0, 1.20, 0],
    ignoreVrmaAvatarPosition: false,
    anchors: [],
  },
};

const STARFIELD_DEFAULT_COUNT = 1500;

/**
 * Procedural starfield — random points on a spherical shell, biased
 * upward so they read above the floor plane in the void environment.
 */
export function createStarfield(THREE, { starCount = STARFIELD_DEFAULT_COUNT } = {}) {
  const geo = new THREE.BufferGeometry();
  const positions = new Float32Array(starCount * 3);
  for (let i = 0; i < starCount; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(Math.random() * 2 - 1);
    const r = 40 + Math.random() * 30;
    positions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
    positions[i * 3 + 1] = r * Math.cos(phi) + 1;
    positions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
  }
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const mat = new THREE.PointsMaterial({
    color: 0xffffff,
    size: 0.15,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.85,
  });
  return new THREE.Points(geo, mat);
}

/**
 * RoomScene — owns the environment layer of an existing Three.js scene.
 *
 * Loads the modern-room GLB once, manages a starfield, and exposes
 * setEnvironment(name) to toggle background colour, starfield presence,
 * and room-mesh visibility. Caller retains ownership of scene/camera/
 * renderer/lighting/avatar — RoomScene mutates `scene` in place but
 * does not construct it.
 *
 * Camera framing presets and avatar default placement are surfaced via
 * the returned environment object (setEnvironment returns the env), so
 * callers can apply them however they want (snap, tween, ignore).
 */
export class RoomScene {
  constructor({
    THREE,
    GLTFLoader,
    DRACOLoader,
    scene,
    dracoDecoderUrl = DRACO_DECODER_URL,
    roomGlbUrl = ROOM_GLB_URL,
  }) {
    if (!THREE) throw new Error('RoomScene: THREE is required');
    if (!GLTFLoader) throw new Error('RoomScene: GLTFLoader is required');
    if (!DRACOLoader) throw new Error('RoomScene: DRACOLoader is required');
    if (!scene) throw new Error('RoomScene: scene is required');

    this.THREE = THREE;
    this.GLTFLoader = GLTFLoader;
    this.DRACOLoader = DRACOLoader;
    this.scene = scene;
    this.dracoDecoderUrl = dracoDecoderUrl;
    this.roomGlbUrl = roomGlbUrl;

    this.roomGroup = null;
    this.starfield = null;
    this.activeEnvironment = null;
    this._draco = null;
  }

  /**
   * Load the room GLB. The mesh is added to the scene immediately but
   * hidden — call setEnvironment('modern-room') to reveal it.
   *
   * onProgress receives the GLTFLoader xhr event ({ loaded, total }).
   */
  async load({ onProgress } = {}) {
    const dracoLoader = new this.DRACOLoader();
    dracoLoader.setDecoderPath(this.dracoDecoderUrl);
    dracoLoader.setDecoderConfig({ type: 'js' });
    this._draco = dracoLoader;

    const loader = new this.GLTFLoader();
    loader.setDRACOLoader(dracoLoader);

    const gltf = await new Promise((resolve, reject) => {
      loader.load(this.roomGlbUrl, resolve, onProgress, reject);
    });

    this.roomGroup = gltf.scene;
    this.roomGroup.scale.setScalar(1);
    this.roomGroup.position.set(0, 0, 0);
    this.roomGroup.visible = false;
    this.scene.add(this.roomGroup);
    return this;
  }

  /**
   * Switch active environment. Updates scene background, toggles
   * starfield, toggles room-mesh visibility. Returns the environment
   * config so callers can read avatarDefault / cameraPosition / etc.
   */
  setEnvironment(name) {
    const env = ENVIRONMENTS[name];
    if (!env) return null;
    this.activeEnvironment = name;

    this.scene.background = new this.THREE.Color(env.backgroundColor);

    if (env.starfield && !this.starfield) {
      this.starfield = createStarfield(this.THREE);
      this.scene.add(this.starfield);
    } else if (!env.starfield && this.starfield) {
      this.scene.remove(this.starfield);
      this.starfield.geometry.dispose();
      this.starfield.material.dispose();
      this.starfield = null;
    }

    if (this.roomGroup) this.roomGroup.visible = (name === 'modern-room');

    return env;
  }

  getEnvironment(name = this.activeEnvironment) {
    return ENVIRONMENTS[name] ?? null;
  }

  /** Wall/table anchor slots for the active environment. */
  getAnchors() {
    return this.getEnvironment()?.anchors ?? [];
  }

  dispose() {
    if (this.starfield) {
      this.scene.remove(this.starfield);
      this.starfield.geometry.dispose();
      this.starfield.material.dispose();
      this.starfield = null;
    }
    if (this.roomGroup) {
      this.scene.remove(this.roomGroup);
      this.roomGroup = null;
    }
    if (this._draco) {
      this._draco.dispose();
      this._draco = null;
    }
  }
}
