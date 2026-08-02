/**
 * avatar.js — Avatar orchestrator
 *
 * Manages Three.js scene, VRM loading, animation loop,
 * cinematic experience layer, and voice.js integration.
 */
import { AvatarAnimator } from './avatar-animator.js';
import { AvatarLipSync } from './avatar-lipsync.js';
import { segmentPortrait, Portrait2DRenderer } from './avatar-2d.js';
import { app, escapeHtml, showToast } from './app.js';
import { AtmosphereEngine } from './avatar-atmosphere.js';
import { SubtitleRenderer } from './avatar-subtitle.js';
import { DrawerManager } from './avatar-drawer.js';
import { PresenceEngine } from './avatar-presence.js';
import { playEntrance } from './avatar-entrance.js';
import {
  createAvatarCompatibilityProfile,
  armAxisSignFromProfile,
  fingerAxisSignFromProfile,
} from './avatar-vrm-profile.js';
import { applyPosePreset, POSE_PRESETS } from './avatar-pose-presets.js';
import { MovementConductor } from './movement-conductor.js';
import { BodyAtlas } from './body-atlas.js';
import { BodyMesh } from './body-mesh.js';
import { InteroceptionEngine } from './interoception-engine.js';
import {
  initDesktopBodyPhysics,
  tickDesktopBodyPhysics,
  teardownDesktopBodyPhysics,
} from './avatar-body-physics-desktop.js';
import { bus } from './activity-bus.js';

let THREE = null;
let VRMModule = null;

// ---- Public state ----
export const avatarState = {
  active: false,
  loading: false,
  scene: null,
  renderer: null,
  camera: null,
  vrm: null,
  animator: null,
  lipSync: null,
  animFrameId: null,
  analyserNode: null,
  // Group chat
  secondaryVrm: null,
  secondaryAnimator: null,
  avatarProfile: null,
  secondaryAvatarProfile: null,
  groupMembers: null,
  activeSpeaker: null,
  // Group narrative generation mode — drives PIP tap behavior. In
  // 'manual' mode, tapping the PIP both swaps the visual focus AND
  // pins that character to respond next. In every other mode
  // ('round_robin' / 'random' / 'llm_decide'), tapping is view-only:
  // the engine still picks the speaker, the user just glances at
  // the listener's reaction.
  groupMode: '',
  // Split-pane group rendering: each character gets its own scene/camera/
  // renderer/canvas, parked in its own DOM pane. The .main pane fills the
  // viewport (active speaker); the .pip pane is the small overlay
  // (non-speaker). Speaker swap = swap .main/.pip CSS classes between
  // panes, no shared-scene collisions, each VRM has its full coordinate
  // space for VRMA gestures.
  pipScene: null,
  pipCamera: null,
  pipRenderer: null,
  paneA: null,   // DOM element, character-A canvas host
  paneB: null,   // DOM element, character-B canvas host
  // 2D portrait
  renderer2D: null,  // Portrait2DRenderer instance
  mode: null,        // '2d' | 'vrm' | null
  callMode: null,
  characterId: '',
  avatarId: '',
  avatarName: '',          // resolved from /api/avatar/for-session, exposed for VR HUD + future surfaces
  _currentEmotion: null,
  _contextLost: false,
  xrFrameHandler: null,
  // Experience layers
  atmosphere: null,
  subtitle: null,
  drawer: null,
  adaptiveCamera: null,
  experienceMode: false,
  _sentenceBuffer: '',
  presence: null,             // PresenceEngine instance — always points at the active speaker
  secondaryPresence: null,    // PresenceEngine for the non-active group character; swaps with presence on speaker switch (null in solo calls)
  _zoomAbortController: null,
  // BVH playback — populated by _playBvh when a .bvh URL is loaded via
  // playVrma. The vrmaMixer/Action fields hold the BVH skeleton's mixer
  // (parallel to the VRMA path), and bvhSkeleton holds the parsed
  // skeleton whose bone quaternions get name-match copied onto the VRM
  // humanoid every frame after mixer.update. Null when no BVH active.
  bvhSkeleton: null,
  // MovementConductor — owns runtime selection (atlas query + dispatch
  // + energy budget + cooldown). Instantiated module-level so it
  // survives across VRM swaps; mode/bias get refreshed on character
  // change. PoseTriggerEngine and any future caller emit intents into
  // this rather than constructing playVrma calls themselves.
  conductor: null,
};

// Module-level conductor singleton — survives VRM swaps. Tick() runs
// from the animation loop. PoseTriggerEngine reads this off avatarState.
export const movementConductor = new MovementConductor();
avatarState.conductor = movementConductor;

// ── Media-audio amplitude bridge ────────────────────────────────────
//
// AudioBus dispatches ``augmentum:audio-bus-state`` whenever a source
// claims/releases the bus — but it carries activity flags only, not
// real-time amplitude. v1 of #132 synthesizes a per-kind target RMS
// from those events; the animate loop lerps the current value toward
// the target and feeds it to PresenceEngine.onMediaAudioRMS each
// frame so her body "breathes with" whatever's playing.
//
// v2 (later): tap an AnalyserNode on the active media element for
// real amplitude reactivity. Synthetic gets us 80% of the felt
// experience without coupling to per-source players.
const _MEDIA_KIND_TARGET_RMS = Object.freeze({
  music:     0.18,  // energetic; bumps temperature
  narration: 0.08,  // attentive; presence only
  dialogue:  0.10,
  mixed:     0.10,
  ambient:   0.04,  // very gentle
  speech:    0.00,  // handled by TTS lipsync path, don't double up
  sfx:       0.00,
  unknown:   0.05,
});
let _mediaTargetRms = 0;
let _mediaCurrentRms = 0;
// Lerp time-constant — seconds to halve the gap to target. ~0.8s
// feels musical without being twitchy.
const _MEDIA_RMS_TAU = 0.8;

window.addEventListener('augmentum:audio-bus-state', (e) => {
  const kinds = e?.detail?.activeKinds || [];
  let target = 0;
  for (const k of kinds) {
    const v = _MEDIA_KIND_TARGET_RMS[k] || 0;
    if (v > target) target = v;
  }
  _mediaTargetRms = target;
});

// Pre-speech inhale — when TTS starts playing, snap the breath cycle
// to a deep inhale so the visible chest rise overlaps the first
// phoneme. Reads as "she's gathering breath to speak" instead of the
// audio starting from a still body. No-op if she's already mid-inhale.
window.addEventListener('augmentum:tts-playback', (e) => {
  if (!e?.detail?.active) return;
  avatarState.animator?.triggerInhale?.();
});

// ---- Lazy-load Three.js + VRM ----
async function ensureLibsLoaded() {
  if (THREE) return;
  // three-vrm-animation comes in alongside the base modules so loadVRM
  // can attach VRMLookAtQuaternionProxy on EVERY VRM at load time
  // (scene-test's pattern). Without it, the first VRMA play would warn
  // about an auto-created lookAt proxy and gaze could pop when the
  // animation starts. Module is ~50KB — cheap to bundle into the
  // initial load.
  [THREE, VRMModule, VrmaModule] = await Promise.all([
    import('../lib/three/three.module.min.js'),
    import('../lib/three-vrm/three-vrm.module.min.js'),
    import('../lib/three-vrm/three-vrm-animation.module.min.js'),
  ]);
}

// ─────────────────────────────────────────────────────────────────────
// VRMA (VRM Animation) playback infrastructure
// ─────────────────────────────────────────────────────────────────────
// Lazy-loaded animation module; cached after first use.
let VrmaModule = null;
let VrmaGltfLoader = null;

// Active mixer + action (single concurrent VRMA — new playback stops prior)
avatarState.vrmaMixer = null;
avatarState.vrmaAction = null;
avatarState.vrmaCurrentName = null;     // logical name for cooldown / event reporting
avatarState.vrmaOnFinish = null;        // optional callback when LoopOnce completes

// Spec-version patcher — older Booth VRMAs ship without specVersion in the
// VRMC_vrm_animation extension. The current loader bails on undefined.
// We unpack the GLB JSON chunk, inject specVersion: '1.0', and feed the
// patched bytes to the loader as a Blob URL. Transparent for files that
// already have specVersion (no-op early-return).
async function _fetchTolerantVrmaUrl(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`fetch failed: ${response.status}`);
  const buffer = await response.arrayBuffer();
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== 0x46546C67) return url;  // not a GLB; pass through
  const totalLen = view.getUint32(8, true);
  const jsonLen = view.getUint32(12, true);
  if (view.getUint32(16, true) !== 0x4E4F534A) return url;
  const jsonBytes = new Uint8Array(buffer, 20, jsonLen);
  const json = JSON.parse(new TextDecoder().decode(jsonBytes));
  const ext = json.extensions?.VRMC_vrm_animation;
  if (!ext || ext.specVersion) return url;
  ext.specVersion = '1.0';
  const newJsonStr = JSON.stringify(json);
  const newJsonBytes = new TextEncoder().encode(newJsonStr);
  const padLen = Math.ceil(newJsonBytes.length / 4) * 4;
  const paddedJson = new Uint8Array(padLen);
  paddedJson.set(newJsonBytes);
  for (let i = newJsonBytes.length; i < padLen; i++) paddedJson[i] = 0x20;
  let binChunk = null;
  const binStart = 20 + jsonLen;
  if (binStart < totalLen) {
    const binLen = view.getUint32(binStart, true);
    if (view.getUint32(binStart + 4, true) === 0x004E4942) {
      binChunk = new Uint8Array(buffer, binStart + 8, binLen);
    }
  }
  const outLen = 12 + 8 + paddedJson.length + (binChunk ? 8 + binChunk.length : 0);
  const out = new ArrayBuffer(outLen);
  const oView = new DataView(out);
  const oBytes = new Uint8Array(out);
  oView.setUint32(0, 0x46546C67, true);
  oView.setUint32(4, 2, true);
  oView.setUint32(8, outLen, true);
  oView.setUint32(12, paddedJson.length, true);
  oView.setUint32(16, 0x4E4F534A, true);
  oBytes.set(paddedJson, 20);
  if (binChunk) {
    const bs = 20 + paddedJson.length;
    oView.setUint32(bs, binChunk.length, true);
    oView.setUint32(bs + 4, 0x004E4942, true);
    oBytes.set(binChunk, bs + 8);
  }
  return URL.createObjectURL(new Blob([out], { type: 'model/gltf-binary' }));
}

async function _ensureVrmaLoader() {
  if (VrmaModule && VrmaGltfLoader) return;
  if (!GLTFLoaderClass) {
    const gltfModule = await import('../lib/three/GLTFLoader.js');
    GLTFLoaderClass = gltfModule.GLTFLoader;
  }
  VrmaModule = await import('../lib/three-vrm/three-vrm-animation.module.min.js');
  VrmaGltfLoader = new GLTFLoaderClass();
  VrmaGltfLoader.register((parser) => new VrmaModule.VRMAnimationLoaderPlugin(parser));
  // VRMLookAtQuaternionProxy is created on-demand by createVRMAnimationClip
  // when the VRM doesn't ship one. We could pre-create here for cleanliness,
  // but the warning is harmless and the proxy lifecycle ties to the VRM.
}

// Lazy-load BVHLoader from the vendored three.js addon. Used by
// _playBvh — the BVH path is a peer of the VRMA path (same mixer/
// action surface in avatarState), so callers don't need to know which
// format they got. BVHLoader.js is vendored alongside GLTFLoader.js
// because production runs from local files only (no CDN at runtime).
let BvhLoaderInstance = null;
async function _ensureBvhLoader() {
  if (BvhLoaderInstance) return;
  const mod = await import('../lib/three/BVHLoader.js');
  BvhLoaderInstance = new mod.BVHLoader();
}

/**
 * Play a VRMA file on the active VRM avatar.
 *
 * Stops any currently-playing VRMA. Returns a promise that resolves when
 * the action is queued (NOT when playback finishes). If `loop` is false,
 * the optional `onFinish` callback fires when the action completes.
 *
 * @param {string} url - Path to .vrma file (typically /ui/lib/animations/...)
 * @param {object} [options]
 * @param {boolean} [options.loop=false] - Loop the animation indefinitely
 * @param {number} [options.speed=1.0] - Playback speed multiplier
 * @param {number} [options.trimStart=0] - Skip first N seconds of clip
 * @param {number} [options.trimEnd=0] - Trim last N seconds (loop wraps earlier)
 * @param {string} [options.name] - Logical name for cooldown / event reporting
 * @param {string} [options.framing] - Camera preset to use during playback ('fullBody' for dances/spins/jumps)
 * @param {{x?: number, y?: number}} [options.framingOffset] - Pan camera + lookAt by this delta (asymmetric poses)
 * @param {Function} [options.onFinish] - Called when LoopOnce action completes
 */
export async function playVrma(url, options = {}) {
  if (!avatarState.active || !avatarState.vrm) return false;
  // Dispatch by extension. .bvh files take a separate parsing path
  // (BVHLoader → mixer on bvh skeleton + per-frame retarget) but
  // populate the same vrmaMixer/Action surface so callers, the
  // animator's `vrmaActive` gate, and stopVrma all keep working
  // uniformly.
  if (url.toLowerCase().endsWith('.bvh')) {
    return _playBvh(url, options);
  }
  await _ensureVrmaLoader();
  let patchedUrl = '';
  try {
    patchedUrl = await _fetchTolerantVrmaUrl(url);
    const gltf = await VrmaGltfLoader.loadAsync(patchedUrl);
    const vrmAnimations = gltf.userData.vrmAnimations;
    if (!vrmAnimations?.length) {
      console.warn('[avatar] VRMA has no animations:', url);
      return false;
    }
    const clip = VrmaModule.createVRMAnimationClip(vrmAnimations[0], avatarState.vrm);

    // Apply trims
    if (options.trimEnd > 0) {
      clip.duration = Math.max(0.1, clip.duration - options.trimEnd);
    }
    const trimStart = (options.trimStart > 0) ? options.trimStart : 0;

    // Stop any prior VRMA before starting new one
    stopVrma();

    avatarState.vrmaMixer = new THREE.AnimationMixer(avatarState.vrm.scene);
    avatarState.vrmaAction = avatarState.vrmaMixer.clipAction(clip);
    avatarState.vrmaCurrentName = options.name || url.split('/').pop();
    avatarState.vrmaOnFinish = options.onFinish || null;
    // Surface the real clip duration so callers can time rotations
    // against the actual animation length rather than guessing via
    // the atlas's informational ``duration`` field.
    avatarState.vrmaCurrentDuration = clip.duration;

    const loop = options.loop !== false;
    avatarState.vrmaAction.setLoop(
      loop ? THREE.LoopRepeat : THREE.LoopOnce,
      loop ? Infinity : 1,
    );
    avatarState.vrmaAction.clampWhenFinished = !loop;
    avatarState.vrmaAction.setEffectiveTimeScale(options.speed ?? 1.0);
    // NOTE: deliberately no fadeIn. AnimationMixer.fadeIn ramps the action
    // weight up from 0, and at weight 0 a lone action shows the model's BIND
    // pose — a T-pose (arms straight out) on most VRMs — so fading a greeting
    // in made her arms swing through a "Y" before the clip. Snapping straight
    // to the clip's first frame avoids that. A true ease would require a
    // crossfade FROM the live procedural pose (arms-down), not from bind.
    avatarState.vrmaAction.play();

    // Camera framing — pull back to fullBody for dances/spins/jumps so the
    // avatar doesn't dip in and out of viewport. AdaptiveCamera holds the
    // preset while vrmaActive is true (see update()), then auto-reverts.
    // framingOffset lets per-VRMA pan the camera (e.g. waves that extend
    // one arm asymmetrically and would otherwise clip out of frame).
    if (options.framing && avatarState.adaptiveCamera) {
      avatarState.adaptiveCamera.setPreset(options.framing, options.framingOffset);
    }

    if (trimStart > 0) {
      avatarState.vrmaAction.time = trimStart;
      avatarState.vrmaMixer.addEventListener('loop', (e) => {
        if (e.action === avatarState.vrmaAction) avatarState.vrmaAction.time = trimStart;
      });
    }

    if (!loop) {
      // Always attach a finished listener for one-shots so the avatar
      // doesn't get stuck at the action's last frame. Holds the end
      // pose briefly (so the gesture reads), then clears the action so
      // the procedural animator resumes — the procedural's own per-frame
      // easing smooths the bones back to neutral over ~10-20 frames.
      const finishedAction = avatarState.vrmaAction;
      avatarState.vrmaMixer.addEventListener('finished', (e) => {
        if (e.action !== finishedAction) return;
        // Fire user callback first (e.g. impliesAfter chaining)
        if (avatarState.vrmaOnFinish) {
          const cb = avatarState.vrmaOnFinish;
          avatarState.vrmaOnFinish = null;
          try { cb(avatarState.vrmaCurrentName); } catch (err) { console.error(err); }
        }
        // Hold-then-release. Guard against the action having been
        // replaced by another playVrma call during the hold window.
        setTimeout(() => {
          if (avatarState.vrmaAction === finishedAction) stopVrma();
        }, 600);
      });
    }

    return true;
  } catch (err) {
    console.error('[avatar] playVrma failed:', url, err);
    return false;
  } finally {
    if (patchedUrl && patchedUrl !== url && patchedUrl.startsWith('blob:')) {
      URL.revokeObjectURL(patchedUrl);
    }
  }
}

/**
 * BVH playback path. Mirrors playVrma's lifecycle (loop, trim, framing,
 * onFinish, hold-then-release for one-shots) but uses a BVHLoader-parsed
 * skeleton+clip and copies bone quaternions onto the VRM humanoid each
 * frame in the animation loop.
 *
 * Works because the sillytavern-pack BVHs use VRM humanoid bone names
 * natively (hips, spine, leftUpperArm, etc.). Future packs that use
 * different naming will need a BONE_ALIASES mapping; today's packs
 * don't, so retargeting is a name-match copy.
 */
async function _playBvh(url, options = {}) {
  await _ensureBvhLoader();
  try {
    const res = await fetch(url);
    if (!res.ok) {
      console.warn('[avatar] BVH fetch failed:', res.status, url);
      return false;
    }
    const text = await res.text();
    const result = BvhLoaderInstance.parse(text);
    const { skeleton, clip } = result;
    if (!skeleton?.bones?.length) {
      console.warn('[avatar] BVH has no bones:', url);
      return false;
    }

    // Drop position tracks — most BVH files only put position on the
    // root (Hips), values are in BVH units (often cm) which don't match
    // the VRM scale. Rotation-only gives a clean in-place animation.
    const rotOnlyTracks = clip.tracks.filter((t) => t.name.endsWith('.quaternion'));
    let duration = clip.duration;
    if (options.trimEnd > 0) {
      duration = Math.max(0.1, duration - options.trimEnd);
    }
    const cleanedClip = new THREE.AnimationClip(clip.name, duration, rotOnlyTracks);
    const trimStart = (options.trimStart > 0) ? options.trimStart : 0;

    // Stop any prior animation (VRMA or BVH) before starting new one
    stopVrma();

    // Mixer targets bones[0] (the BVH root). The mixer walks its
    // children and binds tracks by bone name. We do NOT add the
    // skeleton to any scene — only its local quaternions matter; the
    // retargeter copies them onto the VRM each frame.
    avatarState.vrmaMixer = new THREE.AnimationMixer(skeleton.bones[0]);
    avatarState.vrmaAction = avatarState.vrmaMixer.clipAction(cleanedClip);
    avatarState.vrmaCurrentName = options.name || url.split('/').pop();
    avatarState.vrmaOnFinish = options.onFinish || null;
    avatarState.bvhSkeleton = skeleton;   // signals retargeter in animate()
    // Surface the real clip duration (post-trim) so dance / pose
    // callers can rotate based on actual length, not atlas guesses.
    avatarState.vrmaCurrentDuration = cleanedClip.duration;

    const loop = options.loop !== false;
    avatarState.vrmaAction.setLoop(
      loop ? THREE.LoopRepeat : THREE.LoopOnce,
      loop ? Infinity : 1,
    );
    avatarState.vrmaAction.clampWhenFinished = !loop;
    avatarState.vrmaAction.setEffectiveTimeScale(options.speed ?? 1.0);
    avatarState.vrmaAction.play();

    if (options.framing && avatarState.adaptiveCamera) {
      avatarState.adaptiveCamera.setPreset(options.framing, options.framingOffset);
    }

    if (trimStart > 0) {
      avatarState.vrmaAction.time = trimStart;
      avatarState.vrmaMixer.addEventListener('loop', (e) => {
        if (e.action === avatarState.vrmaAction) avatarState.vrmaAction.time = trimStart;
      });
    }

    if (!loop) {
      const finishedAction = avatarState.vrmaAction;
      avatarState.vrmaMixer.addEventListener('finished', (e) => {
        if (e.action !== finishedAction) return;
        if (avatarState.vrmaOnFinish) {
          const cb = avatarState.vrmaOnFinish;
          avatarState.vrmaOnFinish = null;
          try { cb(avatarState.vrmaCurrentName); } catch (err) { console.error(err); }
        }
        setTimeout(() => {
          if (avatarState.vrmaAction === finishedAction) stopVrma();
        }, 600);
      });
    }

    return true;
  } catch (err) {
    console.error('[avatar] _playBvh failed:', url, err);
    return false;
  }
}

/** Stop any currently-playing VRMA or BVH. Procedural animator resumes naturally. */
export function stopVrma() {
  if (avatarState.vrmaMixer) {
    avatarState.vrmaMixer.stopAllAction();
    // Only VRMAs are bound to vrm.scene; BVH mixer binds to the BVH
    // skeleton's root bone. uncacheRoot of vrm.scene is a no-op for
    // BVH and harmless to call unconditionally.
    avatarState.vrmaMixer.uncacheRoot(avatarState.vrm?.scene);
  }
  avatarState.vrmaMixer = null;
  avatarState.vrmaAction = null;
  avatarState.vrmaCurrentName = null;
  avatarState.vrmaCurrentDuration = 0;
  avatarState.vrmaOnFinish = null;
  avatarState.bvhSkeleton = null;   // GC the BVH skeleton if any was active
  // Recenter hips — VRMAs may translate/rotate the hips bone (dances,
  // spins, jumps), and stopAllAction leaves the bone wherever the last
  // frame landed. Without this restore, repeated VRMAs compound the drift.
  const rest = avatarState.vrm?.__augmentumHipsRest;
  if (rest) {
    const hips = avatarState.vrm.humanoid?.getNormalizedBoneNode?.('hips');
    if (hips) {
      hips.position.copy(rest.position);
      hips.quaternion.copy(rest.quaternion);
    }
  }
}

/** Returns the name of the currently-playing VRMA, or null if none active. */
export function currentVrmaName() {
  return avatarState.vrmaCurrentName;
}

// ---- Loading Indicator ----
function _showLoadingIndicator(container, name) {
  const el = document.createElement('div');
  el.className = 'avatar-loading';
  el.innerHTML = `
    <div class="avatar-loading-shimmer"></div>
    <div class="avatar-loading-content">
      <div class="avatar-loading-spinner"></div>
      <div class="avatar-loading-text">Loading ${name}...</div>
    </div>`;
  container.appendChild(el);
}

function _hideLoadingIndicator(container) {
  const el = container.querySelector('.avatar-loading');
  if (!el) return;
  el.classList.add('fade-out');
  setTimeout(() => el.remove(), 300);
}

/**
 * Build a tiny procedural scene that PMREMGenerator bakes into the
 * environment cube map. A 20m inverted box with subtly tinted walls plus
 * two emissive panels (warm overhead, cool fill) gives MeshStandard
 * surfaces (eyes, cornea, accessories) coherent specular reflections
 * without shipping an HDR asset. Disposed after the one-shot bake.
 */
function _buildEnvScene(THREE) {
  const envScene = new THREE.Scene();

  const wallMat = new THREE.MeshStandardMaterial({
    color: 0x33363c, side: THREE.BackSide, roughness: 0.92, metalness: 0,
  });
  envScene.add(new THREE.Mesh(new THREE.BoxGeometry(20, 20, 20), wallMat));

  const _emit = (color, intensity) => {
    const m = new THREE.MeshBasicMaterial({ color });
    if (intensity != null) m.toneMapped = false;
    return m;
  };
  const panelGeo = new THREE.PlaneGeometry(6, 6);

  const top = new THREE.Mesh(panelGeo, _emit(0xfff0e0, 1));
  top.position.set(0, 9, 0); top.rotation.x = Math.PI / 2;
  envScene.add(top);

  const rim = new THREE.Mesh(panelGeo, _emit(0xffe6cc, 1));
  rim.position.set(0, 2, 9.5);
  envScene.add(rim);

  const fill = new THREE.Mesh(panelGeo, _emit(0xc8d8ff, 1));
  fill.position.set(-9, 1.5, 0); fill.rotation.y = Math.PI / 2;
  envScene.add(fill);

  return envScene;
}

// ---- Scene Setup ----
function createScene(container, opts = {}) {
  const {
    Scene, PerspectiveCamera, WebGLRenderer, DirectionalLight, AmbientLight,
    HemisphereLight, PointLight, SRGBColorSpace, ACESFilmicToneMapping,
    PMREMGenerator,
  } = THREE;

  // Quality knob. 'standard' preserves the historic call-mode budget.
  // 'high' is for the persistent companion widget — the surface is
  // small, GL is the only renderer in flight, and the user is staring
  // at a 360×480-ish viewport so the extra fidelity actually shows.
  // pixelRatio stays at 2 (3 looked over-sharp on stylized VRMs);
  // anisotropy in the load path carries the texture-clarity win.
  const quality = opts.quality === 'high' ? 'high' : 'standard';
  const maxPixelRatio = 2;

  const scene = new Scene();
  scene.background = null; // Transparent — overlay glow shows through

  // Camera: frame head and shoulders (VRM faces +Z, camera at -Z to see front)
  const camera = new PerspectiveCamera(30, container.clientWidth / container.clientHeight, 0.1, 100);
  camera.position.set(0, 1.35, -1.5);
  camera.lookAt(0, 1.2, 0);

  // Renderer
  const canvas = document.createElement('canvas');
  // Presence surfaces (live wallpaper / lock-screen, ?presence=1) freeze the
  // render loop when idle to save battery — but a WebGL canvas clears its
  // drawing buffer after each composite, so a frozen frame would go BLACK.
  // preserveDrawingBuffer:true keeps the last frame painted after the loop
  // stops. Scoped to presence only (small perf cost) so the interactive chat
  // avatar is unaffected.
  const _preserve = (() => {
    try { return new URLSearchParams(location.search).get('presence') === '1'; }
    catch { return false; }
  })();
  const gl = canvas.getContext('webgl2', { alpha: true, antialias: true, preserveDrawingBuffer: _preserve })
          || canvas.getContext('webgl', { alpha: true, antialias: true, preserveDrawingBuffer: _preserve });
  const renderer = new WebGLRenderer({ canvas, context: gl, alpha: true, antialias: true, preserveDrawingBuffer: _preserve });
  // Enable WebXR. No effect on desktop rendering — only matters once an
  // immersive session is requested via avatar-xr.js. Set at construction
  // so the flag is true before any frame is drawn.
  renderer.xr.enabled = true;
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, maxPixelRatio));
  renderer.outputColorSpace = SRGBColorSpace || THREE.SRGBColorSpace;
  // High-quality standalone uses ACES tone mapping at reduced exposure
  // (1.0 was visibly hot on stylized VRMs — skin and key-light hotspots
  // bloomed). 0.8 lifts blacks slightly without washing midtones. The
  // call path keeps tone mapping disabled — see FREEZE-FIX below.
  if (quality === 'high') {
    const tm = ACESFilmicToneMapping || THREE.ACESFilmicToneMapping;
    if (tm !== undefined) {
      renderer.toneMapping = tm;
      renderer.toneMappingExposure = 0.8;
    }
  }
  // FREEZE-FIX 2026-05-14: ACES tone mapping disabled pending MR
  // investigation. Suspected to interact with Quest Browser's MR
  // passthrough alpha-blending path and stall the rAF loop. Re-enable
  // once we've confirmed the freeze cause and either: (a) gate on a
  // setting that's off in MR, (b) clear-color-mask the alpha after
  // tone-mapping, or (c) use CineonToneMapping instead.
  // renderer.toneMapping = ACESFilmicToneMapping || THREE.ACESFilmicToneMapping;
  // renderer.toneMappingExposure = 1.0;
  container.appendChild(renderer.domElement);

  // Snapshot fallback — 2D canvas that shows last good frame when WebGL dies
  const fallback = document.createElement('canvas');
  fallback.className = 'avatar-vrm-fallback';
  fallback.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:none;z-index:0;';
  container.appendChild(fallback);
  let _snapshotTimer = 0;

  // WebGL context loss recovery — mobile browsers kill GL during audio
  // playback. ``_ownsState`` guards against stale listeners from a prior
  // disposed scene firing AFTER a fresh activation replaced this canvas.
  // deactivateAvatar's forceContextLoss() fires the lost event async, so
  // by the time it lands a new activation may already own avatarState —
  // without the guard, the stale write would mark the live scene's
  // _contextLost=true and the heartbeat would churn forever.
  const _ownsState = () => avatarState.renderer === renderer;
  canvas.addEventListener('webglcontextlost', (e) => {
    e.preventDefault();
    if (avatarState._disposing) return;
    if (!_ownsState()) return;
    console.warn('[avatar] WebGL context lost — showing frozen snapshot');
    avatarState._contextLost = true;
    // Show the last snapshot
    fallback.style.display = 'block';
    canvas.style.visibility = 'hidden';
  });
  canvas.addEventListener('webglcontextrestored', () => {
    if (!_ownsState()) return;
    console.debug('[avatar] WebGL context restored — resuming render');
    avatarState._contextLost = false;
    fallback.style.display = 'none';
    canvas.style.visibility = '';
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  });

  // Periodically snapshot the WebGL canvas to the 2D fallback (every 2s)
  avatarState._snapshotInterval = setInterval(() => {
    if (avatarState._contextLost || !avatarState.active) return;
    // Copying a WebGL canvas into a 2D canvas can force a GPU sync/readback.
    // During immersive XR that shows up as a tiny but visible headset hitch,
    // and the fallback is irrelevant because the headset owns presentation.
    if (_isRendererPresentingXR(renderer)) return;
    // Same hitch shows up during chat streaming — the GPU sync lands on
    // the main thread mid-token-flush. Skip while a stream is active;
    // the fallback is only consulted on WebGL context loss, so a snapshot
    // that's stream-duration stale is fine in the rare crash case.
    if (bus.state.chat_streaming) return;
    try {
      fallback.width = canvas.width;
      fallback.height = canvas.height;
      const ctx2d = fallback.getContext('2d');
      if (ctx2d) ctx2d.drawImage(canvas, 0, 0);
    } catch { /* ignore */ }
  }, 2000);

  // 3-point cinematic lighting (camera at -Z, model faces +Z)
  const keyLight = new DirectionalLight(0xfff5e6, 1.0);  // warm key — front-right
  keyLight.position.set(1, 1.5, -1);
  scene.add(keyLight);

  const fillLight = new DirectionalLight(0xe6f0ff, 0.4);  // cool fill — front-left
  fillLight.position.set(-1, 1, -0.5);
  scene.add(fillLight);

  const rimLight = new DirectionalLight(0xffeedd, 0.6);  // warm rim — behind model
  rimLight.position.set(0, 0.5, 1);
  scene.add(rimLight);

  // HemisphereLight: sky tint above + ground bounce below. Cheaper than
  // an env map for general fill and gives top/bottom directional cues that
  // a flat AmbientLight can't. Kept low so it doesn't wash out the 3-point
  // setup. AmbientLight retained for backwards-compatible base lift.
  const hemi = new HemisphereLight(0xe6f0ff, 0x33302a, 0.25);
  hemi.position.set(0, 1, 0);
  scene.add(hemi);

  const ambient = new AmbientLight(0xffffff, 0.10);
  scene.add(ambient);

  // FREEZE-FIX 2026-05-14: PMREM env map bake disabled pending MR
  // investigation. Suspected to leak WebGL framebuffer state that the
  // subsequent WebXR layer bind can't recover from, manifesting as a
  // frozen-frame stall on MR entry. Re-enable after the bake is verified
  // to fully restore the GL state machine, OR move the bake to fire
  // strictly post-XR-session-end so the two paths never share a frame.
  // try {
  //   const pmrem = new PMREMGenerator(renderer);
  //   pmrem.compileEquirectangularShader();
  //   const envScene = _buildEnvScene(THREE);
  //   const envTarget = pmrem.fromScene(envScene, 0.04);
  //   scene.environment = envTarget.texture;
  //   avatarState._envRenderTarget = envTarget;
  //   avatarState._pmrem = pmrem;
  //   envScene.traverse((o) => {
  //     if (o.isMesh) { o.geometry.dispose(); o.material.dispose(); }
  //   });
  // } catch (err) {
  //   console.debug('[avatar] env IBL bake skipped:', err?.message);
  // }

  // Eye catch light (small point light at camera height)
  const catchLight = new PointLight(0xffffff, 0.3, 3);
  catchLight.position.copy(camera.position);
  scene.add(catchLight);

  return { scene, camera, renderer };
}

// ---- Auto-framing: detect VRM bounding box and position camera ----
function _isRendererPresentingXR(renderer = avatarState.renderer) {
  return !!renderer?.xr?.isPresenting;
}

// Walk a Three.js subtree and crank every texture to the renderer's
// max anisotropy. Idempotent — safe to re-run after a hot-swap. Used
// by the high-quality standalone path; the call path leaves textures
// at default (bilinear) to keep the per-frame budget tight.
function _applyMaxAnisotropy(root, renderer) {
  if (!root || !renderer?.capabilities?.getMaxAnisotropy) return;
  const max = renderer.capabilities.getMaxAnisotropy();
  if (!max || max <= 1) return;
  const slots = ['map', 'normalMap', 'emissiveMap', 'metalnessMap',
                 'roughnessMap', 'aoMap', 'specularMap', 'matcap',
                 'shadeMultiplyTexture', 'outlineWidthMultiplyTexture',
                 'rimMultiplyTexture', 'uvAnimationMaskTexture'];
  root.traverse((obj) => {
    if (!obj.material) return;
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
    for (const m of mats) {
      for (const k of slots) {
        const t = m[k];
        if (t && t.isTexture && t.anisotropy !== max) {
          t.anisotropy = max;
        }
      }
    }
  });
}

function _autoFrameVRM(vrm, camera) {
  if (_isRendererPresentingXR()) return;
  const { Box3, Vector3 } = THREE;
  const box = new Box3().setFromObject(vrm.scene);
  const size = new Vector3();
  const center = new Vector3();
  box.getSize(size);
  box.getCenter(center);

  const modelHeight = size.y;
  const modelWidth = size.x;  // T-pose silhouette width — used by fullBody preset below
  const headY = box.max.y;
  const feetY = box.min.y;

  // Aspect-driven framing intent:
  //   - Landscape (aspect ≥ 0.85, desktop / tablet horizontal): head +
  //     upper chest. Shows expression and body language tightly.
  //   - Portrait (aspect < 0.85, mobile / phone vertical): full body,
  //     head to feet. Mobile viewports have tons of vertical real
  //     estate to spend and almost no horizontal — full body fills
  //     the screen cinematically and matches the user's mental model
  //     of "I'm in a video call with this character."
  //
  // The earlier aspect-aware approach used `modelWidth + padding` for
  // frameWidth, but `modelWidth` from Box3 captures the T-POSE
  // silhouette (~1.5 units wide). The springs settle to arms-at-sides
  // within ~1s, so the actual idle silhouette is closer to 0.55 of
  // model height. Using T-pose width forced the camera way too far
  // back on portrait (avatar small in lower half of viewport, shoes
  // cut off at the bottom). Now we use a fixed nominal width based on
  // the idle silhouette, which lets distance be height-driven on
  // portrait — which is what we want for full body framing anyway.
  const fovRad = camera.fov * Math.PI / 180;
  const tanHalfFov = Math.tan(fovRad / 2);
  const aspect = camera.aspect || 1;
  const isPortrait = aspect < 0.85;

  let frameTop, frameBottom, frameWidth;
  if (isPortrait) {
    // Full body — tight margin above head and below feet so the avatar
    // fills the canvas. Earlier 6%/4% margins left visible space
    // between the avatar's head and the listening-status header.
    frameTop = headY + modelHeight * 0.02;
    frameBottom = feetY - modelHeight * 0.02;
    // Idle silhouette is shoulder-span at most. Padding leaves room
    // for breathing/sway and the occasional gesture without cropping.
    frameWidth = modelHeight * 0.65;
  } else {
    // Head + upper chest (existing landscape framing).
    frameTop = headY + modelHeight * 0.05;
    frameBottom = headY - modelHeight * 0.35;
    frameWidth = modelHeight * 0.55;
  }
  const frameCenterY = (frameTop + frameBottom) / 2;
  const frameHeight = frameTop - frameBottom;

  const distY = (frameHeight / 2) / tanHalfFov;
  const distX = (frameWidth / 2) / (tanHalfFov * aspect);
  const dist = Math.max(distY, distX);

  // VRM faces +Z, camera at -Z
  // Portrait uses a tighter breathing-room multiplier so the avatar
  // reads as filling the screen; landscape keeps the comfortable 10%
  // padding because the avatar is smaller-by-design there.
  const camZ = center.z - dist * (isPortrait ? 1.04 : 1.1);

  // Update camera presets based on this model's proportions
  const basePos = [center.x, frameCenterY, camZ];
  const baseLookAt = [center.x, frameCenterY - modelHeight * 0.02, center.z];

  // Store base framing for adaptive camera
  avatarState._autoFrame = { basePos, baseLookAt, modelHeight, headY, center };

  // Update all presets relative to this model
  const dz = dist * 0.05; // subtle zoom offsets
  const dy = modelHeight * 0.02;
  CAMERA_PRESETS.default =    { pos: [...basePos],                                      lookAt: [...baseLookAt] };
  CAMERA_PRESETS.speaking =   { pos: [basePos[0], basePos[1] - dy * 0.3, basePos[2] - dz],     lookAt: [baseLookAt[0], baseLookAt[1] - dy * 0.2, baseLookAt[2]] };
  CAMERA_PRESETS.gesture =    { pos: [basePos[0], basePos[1] - dy * 2, basePos[2] - dz * 4],    lookAt: [baseLookAt[0], baseLookAt[1] - dy * 1.5, baseLookAt[2]] };
  CAMERA_PRESETS.thinking =   { pos: [basePos[0], basePos[1] + dy, basePos[2] + dz],            lookAt: [baseLookAt[0], baseLookAt[1] + dy * 0.5, baseLookAt[2]] };
  CAMERA_PRESETS.drawerOpen = { pos: [basePos[0] - 0.15, basePos[1], basePos[2]],               lookAt: [baseLookAt[0] - 0.08, baseLookAt[1], baseLookAt[2]] };
  CAMERA_PRESETS.group =      { pos: [center.x, center.y + modelHeight * 0.3, camZ - dist * 0.3], lookAt: [center.x, center.y + modelHeight * 0.2, center.z] };
  // Full-body framing — used by VRMA dances/spins/jumps/waves. Aspect-
  // aware so a portrait viewport doesn't crop the dance horizontally.
  const fullFrameHeight = modelHeight * 1.35;
  const fullFrameWidth = modelWidth * 1.6 + modelHeight * 0.15;  // arms can swing
  const fullDistY = (fullFrameHeight / 2) / tanHalfFov;
  const fullDistX = (fullFrameWidth / 2) / (tanHalfFov * aspect);
  const fullDist = Math.max(fullDistY, fullDistX);
  const fullCamZ = center.z - fullDist;
  const fullLookY = center.y;
  CAMERA_PRESETS.fullBody =   { pos: [center.x, fullLookY + modelHeight * 0.05, fullCamZ], lookAt: [center.x, fullLookY, center.z] };

  // Apply default framing immediately
  camera.position.set(...basePos);
  camera.lookAt(...baseLookAt);

  console.debug('[avatar] Auto-framed VRM:', {
    height: modelHeight.toFixed(2),
    headY: headY.toFixed(2),
    camDist: dist.toFixed(2),
    frameCenter: frameCenterY.toFixed(2),
  });
}

/**
 * (Retired with split-pane group rendering — kept temporarily for git
 * blame context; safe to delete on next pass.)
 *
 * Frame the camera for a group two-shot.
 *
 * Approach borrowed from the cinematic mockup
 * (ui/mockups/voice-mode-cinematic.html, renderGroupVRM, ~line 1826):
 *   - look at face level (head world Y of vrm1)
 *   - camera positioned slightly BELOW face level (subtle upward tilt
 *     reads as flattering / engaged; level or above reads as flat)
 *   - distance computed from the union width with a baseline of ~2.0
 *     world units, with aspect-aware pullback so half-windowed desktops
 *     don't crop the pair
 *   - point eyes at the camera so both characters "look at" the viewer
 */
function _autoFrameGroup(vrm1, vrm2, camera, viewport) {
  const { Box3, Vector3 } = THREE;
  // Walk up to the layout wrapper (or fall back to vrm.scene) and
  // refresh world matrices from there — Box3.setFromObject and
  // bone.getWorldPosition both need current matrixWorld on every
  // ancestor, and we just mutated wrapper.position/rotation so the
  // chain is dirty.
  const root1 = vrm1.__augmentumLayoutWrapper || vrm1.scene;
  const root2 = vrm2.__augmentumLayoutWrapper || vrm2.scene;
  root1.updateMatrixWorld(true);
  root2.updateMatrixWorld(true);
  const box = new Box3();
  box.makeEmpty();
  box.expandByObject(root1);
  box.expandByObject(root2);

  const size = new Vector3();
  const center = new Vector3();
  box.getSize(size);
  box.getCenter(center);

  // Look at face level — find the head world Y of vrm1 (both VRMs are
  // height-normalised so their heads land at very close Y values).
  let faceY = box.max.y - size.y * 0.08;  // fallback: a bit below crown
  try {
    const headBone = vrm1.humanoid?.getNormalizedBoneNode?.('head');
    if (headBone) {
      const headWorld = new Vector3();
      headBone.getWorldPosition(headWorld);
      faceY = headWorld.y - 0.03;  // slight downward bias to face center
    }
  } catch { /* fallback above */ }

  // Width-driven distance: enough room around the pair on this aspect.
  // Mockup uses fov 28 + dist 2.0 at aspect 1.6. Production camera is
  // wider (likely 30+), so we compute the distance per aspect rather
  // than copying a magic number.
  const fovRad = camera.fov * Math.PI / 180;
  const aspect = (viewport?.clientWidth || 1) / (viewport?.clientHeight || 1) || 1;
  // Frame width is FIXED — the avatars start in T-pose (arms straight
  // out) and the spring-driven idle takes a couple seconds to settle
  // them to arms-at-sides. If we computed frameWidthX from the current
  // Box3 we'd frame for the T-pose silhouette and the camera would
  // stay parked too far back forever. Two height-normalised characters
  // at ±0.42 with relaxed arms span ≈ 1.3 units; +0.4 padding = 1.7.
  const frameWidthX = 1.7;
  // Frame height: head + upper-body. ~1.05 unit visible vertical.
  const frameHeightY = 1.05;
  const distY = (frameHeightY / 2) / Math.tan(fovRad / 2);
  const distX = (frameWidthX / 2) / (Math.tan(fovRad / 2) * aspect);
  const dist = Math.max(distY, distX) * 1.05;

  // Camera sits a touch below face Y (~0.18 below per mockup) so the
  // viewer is looking up slightly. Eyes at face height read flat.
  const camY = faceY - 0.18;
  const camZ = center.z - dist;
  const basePos = [center.x, camY, camZ];
  const baseLookAt = [center.x, faceY, center.z];

  // Lock all the adaptive-camera presets to this framing — otherwise
  // transitions on speaking/thinking/gesture would snap to stale
  // single-VRM positions that the auto-frame computed at first load.
  CAMERA_PRESETS.default = { pos: [...basePos], lookAt: [...baseLookAt] };
  CAMERA_PRESETS.group = { pos: [...basePos], lookAt: [...baseLookAt] };
  CAMERA_PRESETS.speaking = { pos: [...basePos], lookAt: [...baseLookAt] };
  CAMERA_PRESETS.thinking = { pos: [...basePos], lookAt: [...baseLookAt] };
  CAMERA_PRESETS.gesture = { pos: [...basePos], lookAt: [...baseLookAt] };
  CAMERA_PRESETS.drawerOpen = {
    pos: [basePos[0] - 0.15, basePos[1], basePos[2]],
    lookAt: [baseLookAt[0] - 0.08, baseLookAt[1], baseLookAt[2]],
  };
  CAMERA_PRESETS.fullBody = {
    pos: [center.x, faceY - 0.05, center.z - dist * 1.5],
    lookAt: [center.x, center.y, center.z],
  };

  camera.position.set(...basePos);
  camera.lookAt(...baseLookAt);

  // Eyes track camera — both characters look at the viewer (mockup
  // detail). Without this, eyes default to forward gaze and the pair
  // looks past the camera into the void.
  try {
    if (vrm1.lookAt) vrm1.lookAt.target = camera;
    if (vrm2.lookAt) vrm2.lookAt.target = camera;
  } catch { /* not all VRMs have lookAt */ }

  avatarState._autoFrame = {
    basePos, baseLookAt,
    modelHeight: size.y,
    headY: box.max.y,
    center,
  };

  console.debug('[avatar] Auto-framed group:', {
    aspect: aspect.toFixed(2),
    unionWidth: size.x.toFixed(2),
    faceY: faceY.toFixed(2),
    camDist: dist.toFixed(2),
    distLimitedBy: distX > distY ? 'width' : 'height',
  });
}

// ---- User zoom controls (pinch + scroll wheel) ----
function _setupZoomControls(container, camera) {
  avatarState._zoomAbortController?.abort();
  const zoomController = typeof AbortController !== 'undefined'
    ? new AbortController()
    : null;
  avatarState._zoomAbortController = zoomController;
  const withZoomSignal = (options) => zoomController
    ? { ...options, signal: zoomController.signal }
    : options;
  let userZoom = 0; // offset from auto-framed position (negative = closer)

  // Scroll wheel
  container.addEventListener('wheel', (e) => {
    e.preventDefault();
    userZoom += e.deltaY * 0.003;
    userZoom = Math.max(-1.5, Math.min(2.0, userZoom)); // clamp range
    _applyZoom(camera, userZoom);
  }, withZoomSignal({ passive: false }));

  // Pinch-to-zoom (touch)
  let _pinchStartDist = 0;
  let _pinchStartZoom = 0;

  container.addEventListener('touchstart', (e) => {
    if (e.touches.length === 2) {
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      _pinchStartDist = Math.sqrt(dx * dx + dy * dy);
      _pinchStartZoom = userZoom;
    }
  }, withZoomSignal({ passive: true }));

  container.addEventListener('touchmove', (e) => {
    if (e.touches.length === 2) {
      e.preventDefault();
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const scale = _pinchStartDist / Math.max(1, dist);
      userZoom = _pinchStartZoom + (scale - 1) * 3;
      userZoom = Math.max(-1.5, Math.min(2.0, userZoom));
      _applyZoom(camera, userZoom);
    }
  }, withZoomSignal({ passive: false }));

  // Double-tap to reset zoom
  let _lastTap = 0;
  container.addEventListener('touchend', (e) => {
    if (e.touches.length === 0) {
      const now = Date.now();
      if (now - _lastTap < 300) {
        userZoom = 0;
        _applyZoom(camera, 0);
      }
      _lastTap = now;
    }
  }, withZoomSignal({ passive: true }));

  avatarState._userZoom = 0;
  avatarState._setZoom = (z) => { userZoom = z; _applyZoom(camera, z); };
}

function _applyZoom(camera, zoomOffset) {
  if (_isRendererPresentingXR()) return;
  const frame = avatarState._autoFrame;
  if (!frame) return;
  // Move camera along its look direction (Z axis for VRM)
  const pos = frame.basePos;
  camera.position.set(pos[0], pos[1], pos[2] - zoomOffset);
  avatarState._userZoom = zoomOffset;
}

// ---- Adaptive Camera ----
const CAMERA_PRESETS = {
  default:    { pos: [0, 1.35, -1.5],     lookAt: [0, 1.2, 0] },
  gesture:    { pos: [0, 1.28, -1.8],     lookAt: [0, 1.15, 0] },
  thinking:   { pos: [0, 1.37, -1.4],     lookAt: [0, 1.22, 0] },
  speaking:   { pos: [0, 1.34, -1.55],    lookAt: [0, 1.19, 0] },
  drawerOpen: { pos: [-0.15, 1.35, -1.5], lookAt: [-0.08, 1.2, 0] },
  group:      { pos: [0, 1.25, -2.2],     lookAt: [0, 1.1, 0] },
  // Full-body framing for the no-history landing state. Camera pulled
  // back + lowered to chest-height with the lookAt at mid-torso so the
  // entire standing avatar (~1.6m tall) fits with comfortable headroom.
  fullBody:   { pos: [0, 1.05, -2.6],     lookAt: [0, 0.85, 0] },
};

// Micro-gestures that must NEVER move the camera. Idle beats and
// listening acks fire every few seconds during quiet/listening
// stretches; coupling each to the 'gesture' preset produced a
// constant zoom-out/zoom-in pump (2026-06-11). Deliberate gestures
// (waves, semantic actions, dances) keep their cinematography.
const _SUBTLE_CAMERA_GESTURES = new Set([
  'blink_slow', 'weight_shift', 'look_away', 'look_around',
  'call_grounding_breath', 'head_tilt', 'posture_adjust', 'sigh',
  'lean_back', 'deep_breath', 'nod',
  'call_acknowledge', 'call_attentive_lean',
]);

class AdaptiveCamera {
  constructor(camera, THREE) {
    this._camera = camera;
    this._THREE = THREE;
    this._halflife = 1.4;  // slower, more cinematic transitions

    this._current = [...CAMERA_PRESETS.default.pos, ...CAMERA_PRESETS.default.lookAt];
    this._velocity = [0, 0, 0, 0, 0, 0];
    this._target = [...this._current];
    this._targetPreset = 'default';
    this._revertTimer = 0;
    this._revertDelay = 4.0;  // hold position longer before reverting
    // Dynamic centering offset — added to both camera position AND lookAt
    // so the view "pans" without changing distance/angle. Driven each frame
    // by the hips bone delta during VRMA playback. Smoothed independently
    // of the spring (lerped 85/15) so it follows the body without lag.
    this._dynamicOffset = [0, 0];
    // NaN sentinels guarantee the first frame applies the camera write.
    this._lastAppliedOx = NaN;
    this._lastAppliedOy = NaN;
  }

  setDynamicOffset(x, y) {
    this._dynamicOffset[0] = x;
    this._dynamicOffset[1] = y;
  }

  setPreset(name, offset = null) {
    const p = CAMERA_PRESETS[name];
    if (!p) return;
    // Apply user zoom offset to the Z position
    const zoom = avatarState._userZoom || 0;
    // Per-VRMA framing offset — shifts both camera position and lookAt by
    // the same delta so the camera "pans" without changing distance/angle.
    // Used by waves and other asymmetric one-armed gestures so the
    // extended arm stays inside the portrait frame.
    const ox = offset?.x || 0;
    const oy = offset?.y || 0;
    this._target = [
      p.pos[0] + ox, p.pos[1] + oy, p.pos[2] - zoom,
      p.lookAt[0] + ox, p.lookAt[1] + oy, p.lookAt[2],
    ];
    this._targetPreset = name;
    this._revertTimer = 0;
  }

  update(dt, state) {
    if (this._targetPreset !== 'default' && this._targetPreset !== 'drawerOpen') {
      this._revertTimer += dt;
      if (this._revertTimer > this._revertDelay) {
        const shouldHold = (this._targetPreset === 'speaking' && state.speaking)
          || (this._targetPreset === 'thinking' && state.processing)
          || (this._targetPreset === 'gesture' && state.gesturing)
          || (this._targetPreset === 'fullBody' && state.vrmaActive);
        if (!shouldHold) {
          this.setPreset(state.drawerOpen ? 'drawerOpen' : 'default');
        }
      }
    }

    const d = (4.0 * 0.6931472) / this._halflife;
    const eydt = Math.exp(-d * dt);
    const jdt = d * dt;

    // Track whether the spring is at rest. Pos eps ≈ 0.01 mm in scene
    // units; vel eps prevents micro-jitter from a near-zero velocity
    // tail from re-triggering the write every frame.
    const POS_EPS = 1e-5;
    const VEL_EPS = 1e-4;
    let settled = true;
    for (let i = 0; i < 6; i++) {
      const c = this._current[i] - this._target[i];
      const j0 = this._velocity[i] + c * d;
      this._current[i] = this._target[i] + (c + j0 * dt) * eydt;
      this._velocity[i] = (this._velocity[i] - j0 * jdt) * eydt;
      if (settled
          && (Math.abs(this._current[i] - this._target[i]) > POS_EPS
              || Math.abs(this._velocity[i]) > VEL_EPS)) {
        settled = false;
      }
    }

    const ox = this._dynamicOffset[0];
    const oy = this._dynamicOffset[1];
    // Skip the camera write entirely when the spring has settled AND the
    // dynamic offset hasn't moved measurably since the last apply. Three's
    // position.set + lookAt each recompute matrices and orthogonalize the
    // up-vector basis — the most expensive per-frame CPU work the camera
    // does during quiet windows (the common case during chat streaming).
    // Visual fidelity is unchanged: we're already at target.
    if (!settled
        || ox !== this._lastAppliedOx
        || oy !== this._lastAppliedOy) {
      this._camera.position.set(this._current[0] + ox, this._current[1] + oy, this._current[2]);
      this._camera.lookAt(this._current[3] + ox, this._current[4] + oy, this._current[5]);
      this._lastAppliedOx = ox;
      this._lastAppliedOy = oy;
    }
  }
}

// ---- VRM Loading ----
let GLTFLoaderClass = null;

/** Pull render-time options out of an avatar payload from
 *  `/api/avatar/for-session`. mannerisms can come over the wire as a
 *  JSON string or as a parsed object — both shapes are tolerated. */
function _vrmLoadOpts(avatarData) {
  let m = avatarData?.mannerisms;
  if (typeof m === 'string') {
    try { m = JSON.parse(m); } catch { m = null; }
  }
  if (!m || typeof m !== 'object') return {};
  const opts = {};
  if (typeof m.face_rotation_y === 'number' && Number.isFinite(m.face_rotation_y)) {
    opts.faceRotationY = m.face_rotation_y;
  }
  return opts;
}

async function loadVRM(url, opts = {}) {
  // GLTFLoader is a Three.js addon, not part of three-vrm
  if (!GLTFLoaderClass) {
    const gltfModule = await import('../lib/three/GLTFLoader.js');
    GLTFLoaderClass = gltfModule.GLTFLoader;
  }
  const { VRMLoaderPlugin, VRMUtils } = VRMModule;

  const loader = new GLTFLoaderClass();
  loader.register((parser) => new VRMLoaderPlugin(parser));

  return new Promise((resolve, reject) => {
    loader.load(
      url,
      (gltf) => {
        const vrm = gltf.userData.vrm;
        if (!vrm) {
          reject(new Error('No VRM data found in GLTF'));
          return;
        }
        if (VRMUtils) {
          try { VRMUtils.removeUnnecessaryVertices(gltf.scene); } catch { /* ignore */ }
          try { VRMUtils.removeUnnecessaryJoints(gltf.scene); } catch { /* ignore */ }
        }
        // Disable frustum culling on every mesh. SkinnedMesh bounding spheres
        // are computed once at the rest pose and never re-updated for bone
        // animation, so small sub-meshes (eyes, eyelashes, mouth interior)
        // get culled the moment a VRMA translates the hips or rotates the
        // head far enough that their original bound falls outside the
        // frustum — the avatar appears see-through into her own skull.
        // Per-mesh culling cost on a single avatar is negligible.
        vrm.scene.traverse((obj) => {
          if (obj.isMesh || obj.isSkinnedMesh) obj.frustumCulled = false;
        });
        // Normalize rest pose so animator math (tuned for T-pose) lands the
        // same way on VRM 0.x (T-pose default) and VRM 1.0 (A-pose default).
        try { vrm.humanoid?.resetNormalizedPose?.(); } catch { /* ignore */ }
        let compatibilityProfile = null;
        try {
          vrm.scene.updateMatrixWorld(true);
          compatibilityProfile = createAvatarCompatibilityProfile(THREE, vrm, {
            gltfJson: gltf.parser?.json,
            label: url,
          });
          vrm.__augmentumCompatibilityProfile = compatibilityProfile;
        } catch (error) {
          console.debug('[avatar] VRM compatibility profiling failed', error);
        }
        // Auto-correct facing for exports whose humanoid arm axis is mirrored
        // relative to the older bundled VRMs. mannerisms.face_rotation_y is
        // an explicit override that REPLACES the heuristic — when set, that
        // exact value is the rotation, full stop. (Earlier "stack on top"
        // logic produced 2π = 0 for Vance because the heuristic was already
        // firing.)
        //
        // NOTE — facing correction migration deferred. scene-test calls
        // VRMUtils.rotateVRM0(vrm) (a one-shot bone-level rebake) instead
        // of mutating vrm.scene.rotation.y. The two mechanisms are NOT
        // equivalent: rotateVRM0 is version-gated (no-op on VRM 1.0,
        // π on VRM 0.x) while our heuristic flips on armAxisProfile ===
        // 'mirrored' (a property of bone authoring, not of declared VRM
        // version). The bundled roster has both VRM 1.0 declarations
        // AND mirrored arm axes — they need the π rotation, but
        // rotateVRM0 alone would no-op them and the avatar would face
        // away from the camera. Migrating cleanly requires per-VRM
        // testing and likely a small heuristic on top of rotateVRM0,
        // not a straight replacement. Leaving the existing path
        // unchanged until that audit is run.
        try {
          vrm.scene.updateMatrixWorld(true);
          let rotY;
          if (typeof opts.faceRotationY === 'number' && Number.isFinite(opts.faceRotationY)) {
            rotY = opts.faceRotationY;
          } else {
            rotY = compatibilityProfile?.facingCorrection === 'rotateY180' ? Math.PI : 0;
          }
          if (rotY) vrm.scene.rotation.y = rotY;
          // Stash the base facing rotation so layout passes (e.g. group
          // inward-turn) can compose on top without compounding when
          // they're called more than once (e.g. on speaker switch).
          vrm.__augmentumBaseRotationY = rotY;
        } catch { /* ignore */ }
        // Capture clean BIND rest quats BEFORE applying any pose,
        // while the normalized humanoid is still at its post-reset bind
        // orientation. Two consumers downstream:
        //   - AvatarAnimator._restQuats — the spring channel composes
        //     IDLE_ARM_POSE / breath / sway deltas as `rest * delta`.
        //     If rest is captured AFTER applyPosePreset(natural) writes
        //     arms, the composition adds the natural pose's -77° Z to
        //     the spring's +80° target = +3° (arms straight up). Reading
        //     from this stash keeps `rest` at the canonical bind.
        //   - AvatarAnimator._fingerRestQuats — same logic for fingers.
        // Covers every spring-controlled body bone plus all fingers; the
        // bone list mirrors VRM_BONE_NAMES + finger suffixes.
        try {
          const restQuats = {};
          const humanoid = vrm.humanoid;
          if (humanoid) {
            const BODY_BONES = [
              'hips', 'spine', 'chest', 'upperChest', 'neck', 'head',
              'leftShoulder', 'rightShoulder',
              'leftUpperArm', 'rightUpperArm',
              'leftLowerArm', 'rightLowerArm',
              'leftHand', 'rightHand',
              'leftUpperLeg', 'rightUpperLeg',
              'leftLowerLeg', 'rightLowerLeg',
              'leftFoot', 'rightFoot',
              'leftEye', 'rightEye',
            ];
            for (const name of BODY_BONES) {
              const node = humanoid.getNormalizedBoneNode?.(name)
                        || humanoid.getRawBoneNode?.(name);
              if (node) restQuats[name] = node.quaternion.clone();
            }
            for (const side of ['left', 'right']) {
              for (const finger of ['Thumb', 'Index', 'Middle', 'Ring', 'Little']) {
                for (const joint of ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']) {
                  const name = `${side}${finger}${joint}`;
                  const node = humanoid.getNormalizedBoneNode?.(name)
                            || humanoid.getRawBoneNode?.(name);
                  if (node) restQuats[name] = node.quaternion.clone();
                }
              }
            }
          }
          // Split into two consumer-facing maps so animator constructor
          // code stays readable. They reference the same Quaternion
          // instances (no extra clone) — animator clones on read.
          vrm.__augmentumBoneRestQuats = restQuats;
          vrm.__augmentumFingerRestQuats = restQuats;
        } catch { /* fall through; animator will fall back to current-state snapshot */ }
        // Apply the natural standing pose as the at-load baseline (scene-
        // test's behavior). Without this, every VRM starts in T-pose
        // (VRM 0.x) or A-pose (VRM 1.0) and only settles to the relaxed
        // standing pose once the procedural animator's spring layers
        // kick in. With this, the very first render frame already shows
        // the avatar in their natural pose with hands curled.
        //
        // Note: the animator's IDLE_ARM_POSE springs will continue to
        // drive arms toward their target values; POSES.natural and
        // IDLE_ARM_POSE are tuned to land the same arms-down standing
        // look, so there's no perceptible slerp jolt. Head/spine/chest
        // get the natural tilt for the first ~1s before easing back to
        // the spring's identity baseline — softens the appearance of a
        // statue-stiff first frame.
        try {
          applyPosePreset(THREE, vrm, POSE_PRESETS.natural, {
            armAxisSign: armAxisSignFromProfile(compatibilityProfile?.armAxisProfile, 'mirrored'),
            fingerAxisSign: fingerAxisSignFromProfile(compatibilityProfile?.fingerAxisProfile),
            // Skip internal reset — resetNormalizedPose ran above for
            // the compatibility profile probe and the rest pose is
            // still clean here.
            reset: false,
          });
        } catch (err) { console.debug('[avatar] applyPosePreset(natural) failed', err); }
        // Snapshot the hips rest transform AFTER facing correction +
        // pose application so the world position reflects any 180°
        // rotation AND the natural-pose hip translation. stopVrma()
        // restores local position+quaternion (rotation-invariant); the
        // camera dynamic-centering uses worldPosition for correct
        // world-space deltas.
        try {
          const hips = vrm.humanoid?.getNormalizedBoneNode?.('hips');
          if (hips) {
            vrm.scene.updateMatrixWorld(true);
            const worldRest = new THREE.Vector3();
            hips.getWorldPosition(worldRest);
            vrm.__augmentumHipsRest = {
              position: hips.position.clone(),
              quaternion: hips.quaternion.clone(),
              worldPosition: worldRest,
            };
          }
        } catch { /* ignore */ }
        // Attach a VRMLookAtQuaternionProxy so VRMAs and other gaze
        // consumers find a quaternion-driven lookAt target on the VRM
        // up front instead of three-vrm-animation auto-creating one on
        // first use (which warns and can cause a one-frame gaze pop).
        // Same pattern scene-test.html establishes — see :872. Idempotent:
        // we check for an existing proxy under the well-known name so
        // the second activation of the same VRM doesn't attach twice.
        try {
          if (vrm.lookAt && VrmaModule?.VRMLookAtQuaternionProxy) {
            const existing = vrm.scene.getObjectByName?.('VRMLookAtQuaternionProxy');
            if (!existing) {
              const proxy = new VrmaModule.VRMLookAtQuaternionProxy(vrm.lookAt);
              proxy.name = 'VRMLookAtQuaternionProxy';
              vrm.scene.add(proxy);
            }
          }
        } catch (err) { console.debug('[avatar] VRMLookAtQuaternionProxy attach failed', err); }
        // BodyMesh — millimeter-precision mesh substrate. Built
        // synchronously from the live VRM mesh at avatar load time
        // (~100ms one-shot, no on-disk artifact). Stashed regardless
        // of whether the per-VRM voxel atlas is also available; the
        // mesh substrate is universally derivable from any VRM with
        // a humanoid rig. See ui/scripts/body-mesh.js for the API.
        vrm.__augmentumBodyMesh = null;
        try {
          vrm.__augmentumBodyMesh = BodyMesh.create({ three: THREE, vrm });
          console.debug(`[avatar] body mesh built in ${vrm.__augmentumBodyMesh.buildMs.toFixed(0)}ms (${vrm.__augmentumBodyMesh.triangleCount.toLocaleString()} tris)`);
        } catch (err) {
          console.warn('[avatar] BodyMesh build failed:', err.message);
        }

        // Best-effort body-atlas load. Atlas files are baked per-VRM by
        // scripts/bake_body_atlases.py and live at /poses/body-atlas-
        // <slug>.json (slug = VRM filename stem, lowercased). When a
        // consumer eventually reads vrm.__augmentumBodyAtlas they get
        // the substrate for region-targeted poses, body-aware IK
        // collision, surface-tangent dwell, etc. — see body-atlas.js.
        //
        // Fire-and-forget: VRM load resolves immediately. Atlas arrives
        // ~1-3s later for the few VRMs that have one baked. 404 is the
        // expected response for user-imported VRMs and silenced.
        vrm.__augmentumBodyAtlas = null;
        try {
          const slugMatch = String(url).match(/\/([^\/?#]+)\.vrm(?:\?|#|$)/i);
          let slug = slugMatch ? slugMatch[1].toLowerCase() : null;
          // Bundled avatars are served at /api/avatar/bundled_[fm]_<name>.vrm
          // but the bake script names atlas files by VRM filename stem
          // (e.g. Becca.vrm → body-atlas-becca.json). Strip the registry
          // prefix so the lookup finds the on-disk file.
          if (slug) slug = slug.replace(/^bundled_[fm]_/, '');
          if (slug) {
            const atlasUrl = `/poses/body-atlas-${slug}.json`;
            BodyAtlas.load(atlasUrl).then((atlas) => {
              vrm.__augmentumBodyAtlas = atlas;
              console.debug(`[avatar] body atlas loaded for ${slug}`);
            }).catch((err) => {
              // 404 is the common case (most VRMs aren't baked); only
              // surface other failures (corrupt JSON, schema mismatch).
              if (!String(err?.message || '').includes('404')) {
                console.debug(`[avatar] body atlas load failed for ${slug}:`, err.message);
              }
            });
          }
        } catch { /* never let atlas wiring block VRM load */ }
        resolve(vrm);
      },
      undefined,
      reject
    );
  });
}

// ---- Animation Loop ----
function startAnimationLoop() {
  const { Clock } = THREE;
  const clock = new Clock();

  function animate(timestamp, xrFrame) {
    // No recursive requestAnimationFrame — the loop is driven by
    // renderer.setAnimationLoop() registered at the bottom of this
    // function. Required for WebXR: rAF stops firing while an
    // immersive session is presenting.

    // Optional frame-rate cap for ambient presence (live wallpaper /
    // lock-screen idle): skip the heavy update+render on frames that arrive
    // sooner than the target interval. Checked BEFORE clock.getDelta() so the
    // clock keeps accumulating and the rendered frame still advances by the
    // true elapsed time (no slow-motion). Never caps an XR session — the
    // headset vsync owns pacing. Off (interval 0) for the interactive avatar.
    const _capMs = avatarState._targetFrameInterval;
    if (_capMs) {
      const _t = timestamp || performance.now();
      if (avatarState._lastRenderT && (_t - avatarState._lastRenderT) < _capMs &&
          !_isRendererPresentingXR()) {
        return;
      }
      avatarState._lastRenderT = _t;
    }

    const delta = clock.getDelta();
    // Clamp the dt the camera + liveliness consumers see, so a stall spike
    // (GC, a throttled coder frame) can't lurch the camera in one big step.
    // The bone springs already self-clamp to 0.1s; this brings the camera and
    // the liveliness EMA in line with them.
    const dtClamped = Math.min(delta, 0.1);
    // Liveliness (0..1). When the adaptive frame cap throttles us under load
    // (coder mode floors the avatar to ~8fps to give the main thread back to
    // the terminal/compile), damp the HIGH-frequency motion (gaze saccades,
    // micro-sway, weight-shift, tremor) so the few rendered frames read as a
    // calm hold instead of jitter. 1 at >=24fps (or uncapped); 0 at <=8fps.
    // EMA-smoothed so settle/wake eases over ~0.6s rather than popping.
    {
      const _capFps = avatarState._targetFrameInterval > 0
        ? 1000 / avatarState._targetFrameInterval : 60;
      const _llTarget = _capFps >= 24 ? 1 : (_capFps <= 8 ? 0 : (_capFps - 8) / 16);
      const _llCur = (avatarState._liveliness == null) ? 1 : avatarState._liveliness;
      avatarState._liveliness = _llCur + (_llTarget - _llCur) * (1 - Math.exp(-delta / 0.6));
    }
    const now = performance.now() / 1000;
    const xrPresenting = _isRendererPresentingXR();
    // Per-frame snapshot of speaking state from the ActivityBus. Every
    // downstream consumer in this frame uses these locals so the value
    // can't shift mid-frame if a voice/TTS callback fires between reads.
    const isSpeaking = !!bus.state.is_speaking;
    const ttsPlaying = !!bus.state.tts_playing;
    if (xrFrame && avatarState.xrFrameHandler) {
      try {
        avatarState.xrFrameHandler(timestamp || performance.now(), xrFrame);
      } catch (err) {
        console.warn('[avatar] XR frame handler failed:', err);
      }
    }

    // Lip sync follows actual TTS playback, not the broader UI speaking state.
    let visemes = null;
    if (ttsPlaying && avatarState.lipSync && avatarState.analyserNode) {
      visemes = avatarState.lipSync.update(avatarState.analyserNode, now);
    } else if (avatarState.lipSync) {
      avatarState.lipSync.reset(); // zero out when not speaking
    }

    // Visemes are applied by the animator via _updateViseme() —
    // do NOT set them directly here to avoid double-writing which
    // pushes morph targets past 1.0 and makes the face mesh vanish.

    // User mic RMS — computed once per frame, fed to both presence
    // engines (primary + group secondary) so the non-speaker also
    // "hears" the user and reacts.
    let _userMicRms = 0;
    if (!isSpeaking && avatarState.analyserNode?._micAnalyser) {
      _userMicRms = avatarState.lipSync?.getRMS(avatarState.analyserNode._micAnalyser) || 0;
    }

    // Interoception substrate tick — runs before presence so presence
    // (and any future v1 consumers) can read fresh physiology. v0 only
    // exposes ``getBreathModifier()`` which is multiplexed below.
    if (avatarState.interoception) {
      avatarState.interoception.update(delta);
    }

    // Media RMS bridge — lerp current toward AudioBus-derived target,
    // feed each frame. Smoothed so a sudden start/stop of media
    // doesn't pop the avatar's energy.
    if (_mediaTargetRms > 0 || _mediaCurrentRms > 0) {
      const alpha = 1 - Math.exp(-delta / _MEDIA_RMS_TAU);
      _mediaCurrentRms += (_mediaTargetRms - _mediaCurrentRms) * alpha;
    }

    // Presence engine update — primary (active speaker in group, only VRM in solo)
    if (avatarState.presence) {
      avatarState.presence.update(delta);

      if (_userMicRms > 0.01) avatarState.presence.onUserAudioRMS(_userMicRms);
      if (_mediaCurrentRms > 0.01 && avatarState.presence.onMediaAudioRMS) {
        avatarState.presence.onMediaAudioRMS(_mediaCurrentRms);
      }

      // Consume gesture (if any). Subtle micro-gestures (listening
      // acks like call_acknowledge) are marked so the adaptive camera
      // ignores them — see _SUBTLE_CAMERA_GESTURES.
      const gesture = avatarState.presence.consumeGesture();
      if (gesture && avatarState.animator?.triggerGesture) {
        avatarState.animator.triggerGesture(gesture);
        if (_SUBTLE_CAMERA_GESTURES.has(gesture)) {
          avatarState._subtleGestureActive = true;
        }
      }

      // Consume idle action (if any). ALL idle micro-gestures are
      // camera-subtle — a weight shift or corner glance must not ride
      // a camera dolly (the zoom-out/in pump on every idle beat,
      // 2026-06-11). Cleared when the animator's gesture slot empties.
      const idleAction = avatarState.presence.consumeIdleAction();
      if (idleAction && avatarState.animator?.triggerGesture) {
        avatarState.animator.triggerGesture(idleAction);
        avatarState._subtleGestureActive = true;
      }

      // Apply breath modifier — multiplexes interoception's
      // physiological breath modifier onto PresenceEngine's
      // emotion-driven one. At interoception's default state the
      // physiological modifier is identity {1,1} so this is purely
      // additive — no behavior change when no events have fired.
      if (avatarState.animator?.setBreathModifier) {
        const interoBreath = avatarState.interoception?.getBreathModifier?.()
                             || { rate: 1, depth: 1 };
        const presenceBreath = avatarState.presence.breathModifier
                               || { rate: 1, depth: 1 };
        avatarState.animator.setBreathModifier({
          rate:  interoBreath.rate  * presenceBreath.rate,
          depth: interoBreath.depth * presenceBreath.depth,
        });
      }

      // Affect coupling — arousal/valence from InteroceptionEngine bias
      // posture (spine upright, shoulder height, chest open) and
      // expression (continuous happy/sad weight). Subtle by design:
      // this is felt-state on top of PresenceEngine's categorical
      // emotion, not a replacement for it.
      if (avatarState.animator?.setAffectModifier) {
        const affect = avatarState.interoception?.getAffect?.()
                       || { arousal: 0.5, valence: 0.5 };
        avatarState.animator.setAffectModifier(affect);
      }

      // Physiology coupling — heart_rate biases blink cadence,
      // muscle_tension biases sway amplitude. Defaults to setpoints
      // (identity multipliers) when no interoception source is wired.
      if (avatarState.animator?.setPhysiology) {
        const phys = avatarState.interoception?.getPhysiology?.()
                     || { heart_rate: 0.40, muscle_tension: 0.30 };
        avatarState.animator.setPhysiology(phys);
      }

      // Pose-intent coupling — pick the idle pose family from conversational
      // flow so the orchestrator drifts among "engaged" lean-in poses while
      // listening (user speaking) and relaxed standing variants when Becca
      // is speaking. Asymmetric thresholds give the lean-in mode stickiness:
      // it takes a clear listening signal (flow < -0.3) to switch in, but
      // only modest neutralization (flow > -0.1) to release.
      //
      // ``bus.state.becca_conversation === 'processing'`` overrides flow
      // and routes to the thinking family — the visible "she's considering"
      // beat between user-input-end and first-phoneme.
      if (avatarState.animator?.setPoseIntent) {
        const conv = bus.state.becca_conversation;
        const flow = avatarState.presence.flow ?? 0;
        const prev = avatarState._poseIntent || 'idle_standing';
        const companionPose = avatarState._companionPoseIntent;
        const companionPoseActive = companionPose?.family
          && (!companionPose.expiresAt || Date.now() < companionPose.expiresAt);
        if (companionPose && !companionPoseActive) {
          avatarState._companionPoseIntent = null;
        }
        let next = prev;
        if (conv === 'processing') {
          next = 'thinking';
        } else if (companionPoseActive) {
          next = companionPose.family;
        } else if (prev === 'thinking') {
          // Release from thinking once we're no longer processing.
          next = flow < -0.3 ? 'idle_engaged' : 'idle_standing';
        } else if (prev === 'idle_standing' && flow < -0.3) {
          next = 'idle_engaged';
        } else if (prev === 'idle_engaged' && flow > -0.1) {
          next = 'idle_standing';
        } else if (prev !== 'idle_standing' && prev !== 'idle_engaged') {
          // Transient companion pose verbs can leave us in richer
          // families (talking, formal, closed, etc.). Once their
          // expiry passes, return to the existing flow-driven idle
          // contract instead of pinning that family forever.
          next = flow < -0.3 ? 'idle_engaged' : 'idle_standing';
        }
        if (next !== prev) {
          avatarState._poseIntent = next;
          avatarState.animator.setPoseIntent(next);
        }
      }

      // Apply emotion
      if (avatarState.animator?.setEmotion) {
        avatarState.animator.setEmotion(avatarState.presence.emotion);
      }
      avatarState._currentEmotion = avatarState.presence.emotion;

      // Lip sync dampening from emotion
      if (avatarState.lipSync?.setEmotionDampen) {
        avatarState.lipSync.setEmotionDampen(avatarState.presence.emotion === 'sad' ? 0.6 : 1.0);
      }
    }

    // Presence engine update — secondary (group only). Mirrors the
    // primary pipeline but feeds the secondary animator instead, so the
    // listener has driven breath/emotion/gestures/idle actions just
    // like the active speaker. Skipped when no secondaryAnimator exists
    // (i.e. solo calls — keeps the solo render path byte-identical).
    if (avatarState.secondaryAnimator && avatarState.secondaryPresence) {
      avatarState.secondaryPresence.update(delta);

      if (_userMicRms > 0.01) avatarState.secondaryPresence.onUserAudioRMS(_userMicRms);

      const gesture = avatarState.secondaryPresence.consumeGesture();
      if (gesture && avatarState.secondaryAnimator.triggerGesture) {
        avatarState.secondaryAnimator.triggerGesture(gesture);
      }

      const idleAction = avatarState.secondaryPresence.consumeIdleAction();
      if (idleAction && avatarState.secondaryAnimator.triggerGesture) {
        avatarState.secondaryAnimator.triggerGesture(idleAction);
      }

      if (avatarState.secondaryAnimator.setBreathModifier) {
        avatarState.secondaryAnimator.setBreathModifier(avatarState.secondaryPresence.breathModifier);
      }
      if (avatarState.secondaryAnimator.setEmotion) {
        avatarState.secondaryAnimator.setEmotion(avatarState.secondaryPresence.emotion);
      }
    }

    // Conductor tick — recovers energy budget over time so back-to-back
    // theatricals get suppressed. No-op when budget is full.
    movementConductor.tick(delta);

    // VRMA mixer tick — runs BEFORE the procedural animator so the
    // animator can layer subtle motion on top of the canned animation.
    // (When no VRMA is active, this is a no-op.)
    if (avatarState.vrmaMixer) {
      avatarState.vrmaMixer.update(delta);
      // BVH retargeter — when a .bvh is the active "VRMA," the mixer
      // just wrote new local quaternions to the BVH skeleton's bones.
      // Copy them onto the matching VRM humanoid bones by name. The
      // sillytavern-pack BVHs already use VRM bone names so this is a
      // straight name-match (see _playBvh comment for the alias caveat).
      if (avatarState.bvhSkeleton && avatarState.vrm?.humanoid) {
        for (const bone of avatarState.bvhSkeleton.bones) {
          const vrmBone = avatarState.vrm.humanoid.getNormalizedBoneNode(bone.name);
          if (!vrmBone) continue;
          vrmBone.quaternion.copy(bone.quaternion);
        }
      }
    }

    // Procedural animation
    if (avatarState.animator) {
      avatarState.animator.update(delta, {
        speaking: isSpeaking,
        liveliness: avatarState._liveliness,   // <1 when throttled → calm the high-freq motion layers

        rms: visemes?.jaw || 0,
        visemes: visemes,  // full { aa, ih, ou, ee, oh, jaw } object
        emotion: null,
        awareness: null,
        userSpeaking: avatarState.presence ? avatarState.presence.flow < -0.3 : false,
        vrmaActive: !!avatarState.vrmaAction,   // animator can dampen its own writes when true
        // Body physics tick — runs AFTER pose channels, BEFORE vrm.update,
        // so compliance bone-quat deltas land this frame. No-op when no
        // desktop body-physics instance is active (XR path ticks via its
        // own session-bound frame handler).
        onPreVrmUpdate: !xrPresenting ? tickDesktopBodyPhysics : null,
      });
    }

    // Secondary avatar (group chat). Speaking/visemes stay off — those
    // belong to the active speaker — but userSpeaking is driven by the
    // secondary's own presence engine, so its listening behavior (gaze
    // intent, attentive arm pose, etc.) can respond when the user or
    // the peer character is holding the floor.
    if (avatarState.secondaryAnimator) {
      avatarState.secondaryAnimator.update(delta, {
        speaking: false,
        rms: 0,
        emotion: null,
        awareness: null,
        userSpeaking: avatarState.secondaryPresence
          ? avatarState.secondaryPresence.flow < -0.3
          : false,
        vrmaActive: false,
      });
    }

    // VRM internal update (spring bones) is handled inside AvatarAnimator.update()
    // Only update secondary VRM here (not driven by primary animator)
    if (avatarState.secondaryVrm && !avatarState.secondaryAnimator) {
      avatarState.secondaryVrm.update(delta);
    }

    // Render. In group mode there are TWO independent rendering stacks
    // — the active speaker's (.scene/.camera/.renderer) and the
    // non-speaker's (.pipScene/.pipCamera/.pipRenderer).
    //
    // Adaptive frame rate (perf pass 2026-06-06): when no live activity
    // is driving change (speech / VRMA / gesture / call action / XR),
    // throttle the GPU render to IDLE_FPS. Animator + presence math
    // still runs every rAF tick — cheap CPU — but renderer.render
    // (shader fill + browser composite) is the expensive part and
    // skipping it during long quiet windows takes 60→30fps idle work
    // off the GPU without any visible difference (breath/blink cadence
    // stays smooth at 30fps; the avatar is barely moving anyway).
    //
    // PiP gating: the secondary scene only matters in group mode. Solo
    // mode is the default, so skipping the second renderer.render is
    // an easy ~half of the per-frame GPU work in normal use.
    const _IDLE_FPS = 30;
    const _IDLE_FRAME_MS = 1000 / _IDLE_FPS;
    const isActive = xrPresenting
      || isSpeaking
      || !!avatarState.vrmaAction
      || !!avatarState.animator?._activeGesture
      || !!avatarState.animator?._activeCallAction;
    const nowMs = timestamp || performance.now();
    if (!avatarState._lastRenderAt) avatarState._lastRenderAt = 0;
    const sinceLastRender = nowMs - avatarState._lastRenderAt;
    const shouldRender = isActive || sinceLastRender >= _IDLE_FRAME_MS;

    if (shouldRender && avatarState.renderer && avatarState.scene && avatarState.camera && !avatarState._contextLost) {
      avatarState.renderer.render(avatarState.scene, avatarState.camera);
      avatarState._lastRenderAt = nowMs;
    }
    // Skip PiP entirely when no secondary VRM is mounted. The renderer
    // exists for group-mode handoffs but rendering an empty scene
    // every frame is pure waste in solo mode (the default).
    if (shouldRender && !xrPresenting && avatarState.secondaryVrm
        && avatarState.pipRenderer && avatarState.pipScene && avatarState.pipCamera
        && !avatarState._contextLost) {
      avatarState.pipRenderer.render(avatarState.pipScene, avatarState.pipCamera);
    }

    // Experience layers
    if (avatarState.adaptiveCamera && !xrPresenting) {
      const gesturing = avatarState.animator?._activeGesture != null
        || avatarState.animator?._activeCallAction != null;
      // Subtle micro-gestures (idle beats, listening acks) don't get
      // cinematography — only deliberate gestures (speaking pools,
      // semantic actions, big call actions) move the camera. The flag
      // clears once the gesture slot empties.
      if (!gesturing) avatarState._subtleGestureActive = false;
      const cinematicGesturing = gesturing && !avatarState._subtleGestureActive;

      // Dynamic centering — track hips world-position delta from rest so
      // the camera follows dances/jumps/translating animations. Smoothed
      // (lerp 0.85/0.15 ≈ 200ms half-life) so it feels subtle, not jerky.
      // Decays back to 0 when no VRMA is active so the camera re-centers.
      // World coords (not local) so 180° facing-corrected VRMs follow correctly.
      let targetX = 0;
      let targetY = 0;
      const restWorld = avatarState.vrm?.__augmentumHipsRest?.worldPosition;
      if (avatarState.vrmaAction && restWorld) {
        const hips = avatarState.vrm.humanoid?.getNormalizedBoneNode?.('hips');
        if (hips) {
          if (!avatarState._tmpHipsWorld) avatarState._tmpHipsWorld = new THREE.Vector3();
          hips.getWorldPosition(avatarState._tmpHipsWorld);
          targetX = avatarState._tmpHipsWorld.x - restWorld.x;
          targetY = (avatarState._tmpHipsWorld.y - restWorld.y) * 0.4;  // dampen Y
        }
      }
      const prev = avatarState.adaptiveCamera._dynamicOffset;
      avatarState.adaptiveCamera.setDynamicOffset(
        prev[0] * 0.85 + targetX * 0.15,
        prev[1] * 0.85 + targetY * 0.15,
      );

      avatarState.adaptiveCamera.update(dtClamped, {
        speaking: isSpeaking,
        processing: !isSpeaking && avatarState._currentEmotion === 'thinking',
        gesturing: cinematicGesturing,
        drawerOpen: avatarState.drawer?._open,
        vrmaActive: !!avatarState.vrmaAction,
      });

      if (isSpeaking && avatarState.adaptiveCamera._targetPreset === 'default') {
        avatarState.adaptiveCamera.setPreset('speaking');
      }
      if (cinematicGesturing && avatarState.adaptiveCamera._targetPreset !== 'gesture') {
        avatarState.adaptiveCamera.setPreset('gesture');
      }
    }

    if (!xrPresenting && avatarState.atmosphere) {
      avatarState.atmosphere.update(delta, _getAtmosphereFrameState(visemes));
    }

    if (!xrPresenting && avatarState.subtitle) {
      avatarState.subtitle.update(delta, isSpeaking);
    }
  }

  // setAnimationLoop drives at rAF rate on desktop and headset vsync in
  // XR. Single source of truth for the 3D loop's lifecycle — start here,
  // stop in deactivateAvatar via setAnimationLoop(null).
  //
  // Save the animate fn + clock on avatarState so pauseAvatarRender /
  // resumeAvatarRender can pause-and-resume the loop without disposing
  // any state. The pause path is hot — invoked from visibilitychange
  // every time the user switches tabs — so reconstruction of the
  // animate closure on every resume would be wasteful.
  avatarState._animate = animate;
  avatarState._clock = clock;
  avatarState._renderPaused = false;
  avatarState.renderer.setAnimationLoop(animate);
}

// ---------------------------------------------------------------------------
// Render-loop pause/resume — used by the companion widget and other
// surfaces that want to stop the 3D render loop when their host UI is
// hidden (tab inactive, widget collapsed) WITHOUT tearing down the
// avatar. Soft pause: VRM stays loaded, animator state persists, the
// renderer just stops calling animate() until resumed.
//
// Distinct from deactivateAvatar (which fully disposes). Pause/resume
// is cheap; deactivate is the heavy unmount path.
// ---------------------------------------------------------------------------

/**
 * Pause the 3D render loop without disposing the avatar.
 *
 * Idempotent. No-op when no renderer is active OR the loop is already
 * paused. Designed to be called from visibilitychange handlers — every
 * tab switch should not produce dispose/reload cycles.
 */
export function pauseAvatarRender() {
  if (!avatarState.renderer) return;
  if (avatarState._renderPaused) return;
  avatarState._renderPaused = true;
  try { avatarState.renderer.setAnimationLoop(null); } catch { /* ignore */ }
}

/**
 * Resume the 3D render loop. Idempotent (no-op if not paused).
 *
 * Resets the internal Clock so the first post-resume frame doesn't see
 * a wall-clock-sized delta jump (which would manifest as the VRM
 * snapping forward several seconds of animation in one frame).
 */
export function resumeAvatarRender() {
  if (!avatarState.renderer || !avatarState._animate) return;
  if (!avatarState._renderPaused) return;
  avatarState._renderPaused = false;
  // ``Clock.start()`` resets the internal lastTime to now, so the
  // very next getDelta() returns ≈0. Without this, hours of paused
  // wall-clock would arrive as a single 3600-second delta on resume,
  // and the animator would integrate that into one giant pose step.
  try { avatarState._clock?.start(); } catch { /* ignore */ }
  try { avatarState.renderer.setAnimationLoop(avatarState._animate); } catch { /* ignore */ }
}

/**
 * Set (or clear) the render-loop frame-rate cap at runtime.
 *
 * The interactive widget normally renders uncapped (full rAF rate). When
 * the companion is only an ambient presence — e.g. the user switched to
 * coder mode in the same tab — there's no reason to pay 60fps for idle
 * breathing/blinking. Callers throttle to 30fps (plenty for idle + lipsync)
 * or lower without tearing the avatar down. ``fps <= 0`` restores uncapped.
 *
 * The cap is honored by the animate loop (see _targetFrameInterval); XR
 * sessions ignore it (the headset owns pacing). Resets the cadence anchor
 * so a step UP in rate takes effect on the very next frame instead of
 * waiting out the old (longer) interval. Idempotent — no-op when the cap
 * is already at the requested value.
 */
export function setAvatarFrameCap(fps) {
  const interval = fps > 0 ? 1000 / fps : 0;
  if (avatarState._targetFrameInterval === interval) return;
  avatarState._targetFrameInterval = interval;
  avatarState._lastRenderT = 0;
}

// Speaking state lives on the ActivityBus:
//   bus.state.voice_state  — string ('idle'|'listening'|'recording'|'speaking'
//                                    |'processing'|'composing'|'armed')
//   bus.state.tts_playing  — boolean (real decoded TTS audio actively playing)
//   bus.state.is_speaking  — derived: voice_state === 'speaking' || tts_playing
//
// _syncSpeakingState recomputes ``is_speaking`` whenever an input changes.
// Voice state handles posture; playback activity handles late/queued TTS
// segments that can arrive after the UI briefly returns idle.

function _syncSpeakingState() {
  const ttsPlaying = !!bus.state.tts_playing;
  const isSpeaking = bus.state.voice_state === 'speaking' || ttsPlaying;
  bus.set('is_speaking', isSpeaking);

  // Mouth cycling should be re-armed by actual playback. During sentence gaps,
  // analyser RMS can fall naturally while the avatar keeps an engaged posture.
  if (avatarState.lipSync?.setForceSpeaking) {
    avatarState.lipSync.setForceSpeaking(ttsPlaying);
  }
}

function _getAtmosphereFrameState(visemes) {
  const isSpeaking = !!bus.state.is_speaking;
  const voiceState = bus.state.voice_state || (isSpeaking ? 'speaking' : 'idle');
  return {
    voiceState,
    speaking: isSpeaking,
    recording: voiceState === 'recording',
    listening: voiceState === 'listening',
    processing: voiceState === 'processing',
    composing: voiceState === 'composing',
    userSpeaking: avatarState.presence ? avatarState.presence.flow < -0.3 : false,
    rms: visemes?.jaw || 0,
    emotion: avatarState._currentEmotion,
    emotionIntensity: 0.6,
  };
}


// ---- Group Chat Helpers ----

/**
 * Build two pane DOM elements inside the avatar viewport for split-pane
 * group rendering. Each pane will host its own canvas/scene/renderer.
 * Returns { paneA, paneB }. Existing children of the viewport are not
 * removed — caller is responsible for clearing first when needed.
 */
function _buildGroupPanes(viewport) {
  const paneA = document.createElement('div');
  paneA.className = 'voice-avatar-pane main';
  paneA.dataset.role = 'main';

  const paneB = document.createElement('div');
  paneB.className = 'voice-avatar-pane pip';
  paneB.dataset.role = 'pip';

  // role/tabindex/aria are added by activateAvatar after wiring click
  // handlers on both panes (only the .pip-classed pane is interactive
  // at any given moment, but the role flips on every speaker swap).

  viewport.appendChild(paneA);
  viewport.appendChild(paneB);
  viewport.classList.add('group-split');
  return { paneA, paneB };
}

/**
 * Set which pane shows the active speaker (.main) and which shows the
 * non-speaker (.pip). speakerIndex 0 → A in main, 1 → B in main.
 *
 * Uses FLIP (First-Last-Invert-Play) to animate the swap: snapshot
 * each pane's rect before the class flip, then animate from the
 * inverted-to-old transform back to identity. The CSS transition on
 * .voice-avatar-pane can't smoothly interpolate between
 * `inset: 0` (main) and `bottom/right + width + aspect-ratio` (pip)
 * because they're different positioning paradigms — FLIP sidesteps
 * that by using transforms which CAN interpolate.
 */
function _setActivePaneClasses(speakerIndex) {
  if (!avatarState.paneA || !avatarState.paneB) return;

  // FIRST: measure pre-swap rects.
  const aFirst = avatarState.paneA.getBoundingClientRect();
  const bFirst = avatarState.paneB.getBoundingClientRect();

  // LAST: apply the class swap so the panes reflow to their new state.
  const aIsMain = speakerIndex === 0;
  avatarState.paneA.classList.toggle('main', aIsMain);
  avatarState.paneA.classList.toggle('pip', !aIsMain);
  avatarState.paneA.dataset.role = aIsMain ? 'main' : 'pip';
  avatarState.paneB.classList.toggle('main', !aIsMain);
  avatarState.paneB.classList.toggle('pip', aIsMain);
  avatarState.paneB.dataset.role = aIsMain ? 'pip' : 'main';

  // Read new rects after the layout settles, then run FLIP.
  const aLast = avatarState.paneA.getBoundingClientRect();
  const bLast = avatarState.paneB.getBoundingClientRect();
  _flipPane(avatarState.paneA, aFirst, aLast);
  _flipPane(avatarState.paneB, bFirst, bLast);
}

/** Animate a pane from its previous bounding rect to its new one. */
function _flipPane(pane, first, last) {
  // Skip if dimensions are degenerate (e.g., pane invisible at start).
  if (!first || !last || first.width === 0 || last.width === 0) return;
  const dx = first.left - last.left;
  const dy = first.top - last.top;
  const sw = first.width / last.width;
  const sh = first.height / last.height;
  // Skip if there's effectively no change (sub-pixel jitter).
  if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5
      && Math.abs(sw - 1) < 0.005 && Math.abs(sh - 1) < 0.005) return;
  if (typeof pane.animate !== 'function') return;
  pane.animate(
    [
      { transform: `translate(${dx}px, ${dy}px) scale(${sw}, ${sh})`, transformOrigin: '0 0' },
      { transform: 'translate(0, 0) scale(1, 1)', transformOrigin: '0 0' },
    ],
    {
      duration: 280,
      easing: 'cubic-bezier(0.16, 1, 0.3, 1)',
      // composite:'replace' is the default — once the animation ends,
      // the pane sits at its actual CSS-driven layout, so there's no
      // residual transform to clean up.
    },
  );
}

function _positionGroupAvatars(speakerIndex) {
  const vrms = [avatarState.vrm, avatarState.secondaryVrm];
  if (!vrms[0] || !vrms[1]) return;

  // Symmetric two-shot, framing borrowed from the cinematic mockup
  // (ui/mockups/voice-mode-cinematic.html, renderGroupVRM):
  //   - spread of ±0.42 (closer than ±0.55 — reads as conversational,
  //     not posed apart)
  //   - subtle inward rotation (~10°) so each character is angling
  //     toward the other, "talking together" feel rather than two
  //     people staring straight at the camera
  //   - equal scale (set by _normalizeVRMHeight); active speaker is
  //     tracked for lipsync but the layout doesn't shuffle per turn
  // Layout target: the scaling wrapper if present (group mode),
  // otherwise the VRM scene itself. Wrappers isolate the VRM from
  // position/rotation/scale concerns so the animator's IK and the
  // spring-bone math don't trip over our layout transforms.
  const leftRoot = vrms[0].__augmentumLayoutWrapper || vrms[0].scene;
  const rightRoot = vrms[1].__augmentumLayoutWrapper || vrms[1].scene;

  leftRoot.position.set(-0.42, 0, 0);
  rightRoot.position.set(0.42, 0, 0);
  // Inward rotation goes on the wrapper, NOT the vrm scene — vrm.scene
  // already carries the face_rotation_y override and the animator may
  // touch it. Layout rotation on the wrapper composes cleanly.
  leftRoot.rotation.y = 0.18;   // turn slightly toward right
  rightRoot.rotation.y = -0.18; // turn slightly toward left

  avatarState.activeSpeaker = speakerIndex;
}

/**
 * Wrap a VRM scene in a scaling Group so different VRM exports read
 * as comparable size in a group two-shot, without mutating the VRM
 * scene's own transform.
 *
 * Why a wrapper instead of `vrm.scene.scale.setScalar(...)` directly:
 * the VRM has internal contracts about its own scene transform — the
 * animator's IK targets, the spring-bone math, the VRMA's hip rest
 * snapshot, all assume the scene's own scale is unchanged. Mutating
 * `vrm.scene.scale` directly causes the animator to target stale
 * world positions and the legs bend awkwardly trying to reach them
 * (the "weird sitting motion" failure mode).
 *
 * The wrapper holds the scale; vrm.scene stays at its natural transform.
 * Group-mode positioning and rotation also go on the wrapper, so the
 * VRM is fully isolated from layout concerns.
 *
 * Returns the wrapper, or null if the VRM has no measurable height.
 */
function _wrapVRMForGroupLayout(vrm, targetHeight = 1.6) {
  const { Box3, Vector3, Group } = THREE;
  vrm.scene.updateMatrixWorld(true);
  const box = new Box3().setFromObject(vrm.scene);
  const size = new Vector3();
  box.getSize(size);
  if (size.y < 0.01) return null;

  const factor = targetHeight / size.y;
  const wrapper = new Group();
  wrapper.scale.setScalar(factor);
  wrapper.add(vrm.scene);
  vrm.__augmentumLayoutWrapper = wrapper;
  return wrapper;
}

// ---- Shared Experience Layer (cinematic mode for both VRM and 2D) ----

function _activateExperienceLayer(viewport) {
  avatarState.experienceMode = true;
  viewport.classList.add('avatar-experience');
  const overlay = document.getElementById('voice-overlay');
  if (overlay) overlay.classList.add('avatar-mode-active', 'avatar-thread-mode');

  // Atmosphere
  const atmoCanvas = document.getElementById('avatar-atmosphere-canvas');
  if (atmoCanvas && overlay) {
    avatarState.atmosphere = new AtmosphereEngine(
      atmoCanvas,
      overlay,
      null
    );
  }

  // Subtitles
  const subtitleContainer = document.getElementById('avatar-subtitle-container');
  if (subtitleContainer) {
    avatarState.subtitle = new SubtitleRenderer(subtitleContainer);
  }

  // Drawer
  const drawerContainer = document.getElementById('avatar-drawer-container');
  if (drawerContainer) {
    avatarState.drawer = new DrawerManager(drawerContainer, (open) => {
      if (avatarState.adaptiveCamera) {
        avatarState.adaptiveCamera.setPreset(open ? 'drawerOpen' : 'default');
      }
      viewport.classList.toggle('drawer-open', open);
    });
  }

  // Hide transcript divs — subtitles handle text display in avatar mode.
  // Keep `.voice-transcript-log` visible as the avatar-mode message thread.
  const transcriptUser = document.querySelector('.voice-transcript-user');
  const transcriptAi = document.querySelector('.voice-transcript-ai');
  const transcriptLog = document.querySelector('.voice-transcript-log');
  if (transcriptUser) transcriptUser.classList.add('avatar-experience-hidden');
  if (transcriptAi) transcriptAi.classList.add('avatar-experience-hidden');
  if (transcriptLog) transcriptLog.classList.remove('avatar-experience-hidden');

  // No-history-visible mode: trigger full-body framing whenever the user
  // can't see message history — either because the conversation is empty
  // OR because they collapsed the transcript drawer. Two observers feed
  // the same `updateHistoryState`:
  //   - childList on transcriptLog → catches first message arrival
  //   - attributes (class) on overlay → catches collapse/expand toggle
  if (transcriptLog && overlay) {
    const updateHistoryState = () => {
      const isCollapsed = overlay.classList.contains('avatar-transcript-collapsed');
      const hasMessages = transcriptLog.children.length > 0;
      const noHistoryVisible = isCollapsed || !hasMessages;
      const wasNoHistory = overlay.classList.contains('no-history');
      if (noHistoryVisible === wasNoHistory) return;
      overlay.classList.toggle('no-history', noHistoryVisible);
      // Camera preset: on WIDE desktop (≥1280px) the avatar always renders
      // full-body even with the transcript visible (side-by-side layout).
      // On narrow / mobile, camera matches the size — fullBody when
      // no-history-visible, default head-and-shoulders otherwise.
      if (avatarState.adaptiveCamera) {
        const wideDesktop = window.innerWidth >= 1280;
        const useFullBody = noHistoryVisible || wideDesktop;
        avatarState.adaptiveCamera.setPreset(useFullBody ? 'fullBody' : 'default');
      }
    };
    updateHistoryState();   // initial state
    const childObs = new MutationObserver(updateHistoryState);
    childObs.observe(transcriptLog, { childList: true });
    const classObs = new MutationObserver(updateHistoryState);
    classObs.observe(overlay, { attributes: true, attributeFilter: ['class'] });
    avatarState._historyObserver = childObs;
    avatarState._collapseObserver = classObs;
  } else {
    console.warn('[avatar] no-history wiring SKIPPED: missing element',
      { overlay: !!overlay, transcriptLog: !!transcriptLog });
  }
}

// ---- 2D Portrait Activation ----

async function _activate2DAvatar(avatarData, analyserNode, viewport) {
    // Get or run segmentation
    let segData = null;
    if (avatarData.segmentation_data) {
        try {
            segData = typeof avatarData.segmentation_data === 'string'
                ? JSON.parse(avatarData.segmentation_data)
                : avatarData.segmentation_data;
        } catch { segData = null; }
    }

    // Run browser-side MediaPipe if no cached data
    if (!segData) {
        try {
            segData = await segmentPortrait(avatarData.portrait_url);
            // Cache result on server for next time
            fetch(`/api/avatar/${avatarData.avatar_id}/segmentation`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(segData),
            }).catch(() => {});
        } catch (e) {
            console.warn('[avatar] Browser MediaPipe segmentation failed:', e.message);
        }
    }

    if (!segData) {
        showAvatarToast('Failed to analyze portrait');
        avatarState.loading = false;
        return false;
    }

    // Hide orb, show viewport
    const orbWrap = document.querySelector('.voice-orb-wrap');
    if (orbWrap) orbWrap.style.display = 'none';
    // Flip the orb-group container into avatar layout mode. CSS keys
    // sizing/anchoring rules off [data-mode="avatar"] vs [data-mode="orb"]
    // so plain voice calls and avatar calls don't fight over the same
    // margin rules on the shared parent.
    document.querySelector('.voice-orb-group')?.setAttribute('data-mode', 'avatar');
    viewport.style.display = '';
    viewport.innerHTML = '';
    _showLoadingIndicator(viewport, avatarData.name || 'portrait');

    // Create 2D renderer
    const renderer2D = new Portrait2DRenderer(viewport, avatarData.portrait_url, segData);
    await renderer2D.init();
    _hideLoadingIndicator(viewport);

    avatarState.renderer2D = renderer2D;
    avatarState.lipSync = new AvatarLipSync();
    avatarState.analyserNode = analyserNode;
    avatarState.mode = '2d';
    avatarState.active = true;
    avatarState.loading = false;

    // Start 2D animation loop
    _start2DAnimLoop();

    // UI updates
    const toggleBtn = document.getElementById('voice-avatar-toggle');
    if (toggleBtn) toggleBtn.classList.add('active');

    // --- Experience layer (cinematic mode) ---
    _activateExperienceLayer(viewport);

    // Resize 2D canvas after experience layer changes container dimensions
    // Use rAF to let the CSS layout settle before measuring
    requestAnimationFrame(() => {
      if (avatarState.renderer2D) avatarState.renderer2D.resize();
    });

    // Resize 2D canvas on window resize. rAF-coalesce so drag-resize
    // doesn't trigger 60 canvas resets per second.
    if (avatarState._resizeHandler) {
      window.removeEventListener('resize', avatarState._resizeHandler);
    }
    let _avatar2dResizePending = false;
    avatarState._resizeHandler = () => {
      if (_avatar2dResizePending) return;
      _avatar2dResizePending = true;
      requestAnimationFrame(() => {
        _avatar2dResizePending = false;
        if (avatarState.renderer2D) avatarState.renderer2D.resize();
      });
    };
    window.addEventListener('resize', avatarState._resizeHandler);

    return true;
}

function _start2DAnimLoop() {
    let lastTime = performance.now() / 1000;

    function animate() {
        avatarState.animFrameId = requestAnimationFrame(animate);
        const now = performance.now() / 1000;
        const delta = now - lastTime;
        lastTime = now;

        // Per-frame ActivityBus snapshot — see 3D animate() for rationale.
        const isSpeaking = !!bus.state.is_speaking;
        const ttsPlaying = !!bus.state.tts_playing;

        // Lip sync follows actual TTS playback, not the broader UI speaking state.
        let visemes = null;
        if (ttsPlaying && avatarState.lipSync && avatarState.analyserNode) {
            visemes = avatarState.lipSync.update(avatarState.analyserNode, now);
        } else if (avatarState.lipSync) {
            avatarState.lipSync.reset();
        }

        // Update 2D renderer
        if (avatarState.renderer2D) {
            avatarState.renderer2D.update(delta, visemes, {
                emotion: avatarState._currentEmotion,
                speaking: isSpeaking,
            });
        }

        // Feed user mic RMS into presence (2D path)
        if (!isSpeaking && avatarState.presence && avatarState.analyserNode?._micAnalyser) {
            const rms = avatarState.lipSync?.getRMS(avatarState.analyserNode._micAnalyser) || 0;
            if (rms > 0.01) avatarState.presence.onUserAudioRMS(rms);
        }

        // Experience layers (atmosphere + subtitles) — shared with VRM loop
        if (avatarState.atmosphere) {
            avatarState.atmosphere.update(delta, _getAtmosphereFrameState(visemes));
        }
        if (avatarState.subtitle) {
            avatarState.subtitle.update(delta, isSpeaking);
        }

    }

    animate();
}

// ---- Public API ----

export async function activateAvatar(analyserNode, sessionInfo) {
  // Standalone-companion mode (Becca widget) may already have the
  // avatar pipeline up. Tear it down so the call can activate cleanly.
  // deactivateAvatar no longer auto-reactivates the widget (that lives
  // in _teardownVoiceCall now), so this is a plain sync teardown.
  if (avatarState.active && avatarState._standalone) {
    try { deactivateAvatar(); } catch (_) {}
  }
  if (avatarState.active || avatarState.loading) return false;
  avatarState.loading = true;
  // Clear any stale context-lost flag from a prior session. deactivateAvatar
  // calls forceContextLoss() which fires the OLD canvas's webglcontextlost
  // listener AFTER the in-place reset, leaving the flag stuck true. Without
  // this, the new render loop and snapshot interval both early-return and
  // the swapped avatar shows a blank viewport.
  avatarState._contextLost = false;

  try {
    // Resolve avatar for this session (BEFORE loading libs — 2D doesn't need Three.js)
    const mode = sessionInfo?.mode || app.state.mode || 'passthrough';
    const characterId = sessionInfo?.characterId || '';
    const avatarId = sessionInfo?.avatarId || '';
    const params = new URLSearchParams({ mode });
    if (characterId) params.set('character_id', characterId);
    if (avatarId) params.set('avatar_id', avatarId);
    const resp = await fetch(`/api/avatar/for-session?${params.toString()}`);
    if (!resp.ok) {
      showAvatarToast('Set up an avatar in character settings');
      avatarState.loading = false;
      return false;
    }
    const avatarData = await resp.json();
    if (!avatarData.avatar_id) {
      showAvatarToast('Set up an avatar in character settings');
      avatarState.loading = false;
      return false;
    }

    // Get or create viewport container
    const viewport = document.getElementById('voice-avatar-viewport');
    if (!viewport) { avatarState.loading = false; return false; }

    avatarState.callMode = mode;
    avatarState.characterId = characterId;
    avatarState.avatarId = avatarData.avatar_id;
    avatarState.avatarName = avatarData.name || '';

    // Type selector: 2D portrait vs VRM
    if (avatarData.type === 'portrait') {
        return await _activate2DAvatar(avatarData, analyserNode, viewport);
    }
    // Else fall through to existing VRM path...

    // Show loading indicator while VRM downloads (can be 10-50MB)
    const orbWrap = document.querySelector('.voice-orb-wrap');
    if (orbWrap) orbWrap.style.display = 'none';
    // Flip the orb-group container into avatar layout mode. See the matching
    // comment in the 2D-portrait path above.
    document.querySelector('.voice-orb-group')?.setAttribute('data-mode', 'avatar');
    viewport.style.display = '';
    viewport.innerHTML = '';
    _showLoadingIndicator(viewport, avatarData.name || 'avatar');

    await ensureLibsLoaded(); // Only load Three.js for VRM mode

    // Guard: >2 group members — fall through to single-avatar mode
    if (!avatarId && sessionInfo.groupMembers?.length > 2) {
      console.warn('[avatar] Group chat with >2 members — showing active speaker only');
    }

    const isGroup = !avatarId && sessionInfo.groupMembers?.length === 2;

    // Group chat: split-pane rendering. Each character gets its own
    // scene/camera/renderer/canvas inside its own DOM pane. The .main
    // pane fills the viewport (active speaker), the .pip pane is the
    // small overlay (non-speaker). No shared scene = no animation
    // collisions when one character VRMA-translates into the other's
    // space, and each VRM gets its own auto-framed camera.
    if (isGroup) {
      const [avatar1Data, avatar2Data] = await Promise.all([
        fetch(`/api/avatar/for-session?mode=${encodeURIComponent(mode)}&character_id=${encodeURIComponent(sessionInfo.groupMembers[0].id)}`).then(r => r.json()),
        fetch(`/api/avatar/for-session?mode=${encodeURIComponent(mode)}&character_id=${encodeURIComponent(sessionInfo.groupMembers[1].id)}`).then(r => r.json()),
      ]);

      const [vrm1, vrm2] = await Promise.all([
        loadVRM(avatar1Data.vrm_url, _vrmLoadOpts(avatar1Data)),
        loadVRM(avatar2Data.vrm_url, _vrmLoadOpts(avatar2Data)),
      ]);

      _hideLoadingIndicator(viewport);

      // Build the split-pane DOM structure (paneA + paneB) and create
      // two independent rendering stacks — one per pane. Character A is
      // initially in main, character B in pip; the speaker-swap path
      // re-points avatarState.scene/camera/renderer between the two
      // stacks rather than moving canvases around.
      const { paneA, paneB } = _buildGroupPanes(viewport);
      avatarState.paneA = paneA;
      avatarState.paneB = paneB;

      const stackA = createScene(paneA);
      const stackB = createScene(paneB);

      stackA.scene.add(vrm1.scene);
      stackB.scene.add(vrm2.scene);

      // Initial speaker = character A. avatarState.scene/camera/renderer
      // always point at the ACTIVE SPEAKER's stack; .pipScene/.pipCamera/
      // .pipRenderer point at the non-speaker. onSpeakerSwitch swaps
      // these references AND the .main/.pip pane classes so the visible
      // hierarchy follows the active speaker.
      avatarState.scene = stackA.scene;
      avatarState.camera = stackA.camera;
      avatarState.renderer = stackA.renderer;
      avatarState.pipScene = stackB.scene;
      avatarState.pipCamera = stackB.camera;
      avatarState.pipRenderer = stackB.renderer;

      avatarState.vrm = vrm1;
      avatarState.secondaryVrm = vrm2;
      avatarState.avatarProfile = vrm1.__augmentumCompatibilityProfile || null;
      avatarState.secondaryAvatarProfile = vrm2.__augmentumCompatibilityProfile || null;

      const mannerisms1 = typeof avatar1Data.mannerisms === 'string'
        ? JSON.parse(avatar1Data.mannerisms) : (avatar1Data.mannerisms || {});
      const mannerisms2 = typeof avatar2Data.mannerisms === 'string'
        ? JSON.parse(avatar2Data.mannerisms) : (avatar2Data.mannerisms || {});

      // Frame each character in its OWN scene with its OWN camera —
      // no compromise framing for the pair. Solo's auto-frame works
      // unchanged because each VRM is alone in its own scene.
      _autoFrameVRM(vrm1, stackA.camera);
      _autoFrameVRM(vrm2, stackB.camera);

      // Animators are constructed AFTER VRMs are added to scenes so the
      // foot-lock IK captures correct world-position anchors.
      avatarState.animator = new AvatarAnimator(THREE, vrm1);
      avatarState.secondaryAnimator = new AvatarAnimator(THREE, vrm2);

      // Per-VRM presence engine for the secondary character so the
      // listener has its own breath / emotion / idle-action / listening
      // micro-behaviors. The primary engine is created below at line
      // ~1863 (shared codepath with solo); only the secondary is wired
      // here. Swaps with avatarState.presence on speaker switch so the
      // pointer-by-role contract (.presence == active speaker's) holds.
      avatarState.secondaryPresence = new PresenceEngine({
        avatarProfile: avatarState.secondaryAvatarProfile,
      });

      avatarState.groupMembers = sessionInfo.groupMembers;
      avatarState.groupMode = sessionInfo.groupMode || '';
      avatarState.activeSpeaker = 0;
      _setActivePaneClasses(0);

      _setupZoomControls(paneA, stackA.camera);

      // Tap the PIP-classed pane. Both panes carry the listener because
      // their roles flip on every speaker switch — at any given moment,
      // whichever pane has .pip is the tappable one.
      //
      // Behavior is mode-agnostic: tap always BOTH swaps the visual
      // immediately AND pins the next-turn speaker via the voice WS.
      // The narrative handler's _resolve_group_speaker treats
      // speaker_override as priority-1 (wins over rotation, random, and
      // llm_decide), and it's one-shot — only the immediately-next turn
      // is pinned. After that, round_robin keeps rotating, random keeps
      // randomising, llm_decide keeps deciding. So a tap always means
      // "you, next turn" without permanently overriding the mode.
      const _activatePip = async () => {
        const target = avatarState.activeSpeaker === 0 ? 1 : 0;
        const member = avatarState.groupMembers?.[target];
        if (!member?.name) return;
        // Visual swap fires first so the tap feels responsive.
        onSpeakerSwitch(member.name);
        try {
          const v = await import('./voice.js');
          v.sendVoiceSpeakerOverride?.(member.name);
        } catch { /* voice module not loaded — view-swap still happened */ }
      };
      const _onPaneTap = (e) => {
        const pane = e.currentTarget;
        if (!pane.classList.contains('pip')) return;
        _activatePip();
      };
      const _onPaneKey = (e) => {
        if (e.key !== 'Enter' && e.key !== ' ') return;
        const pane = e.currentTarget;
        if (!pane.classList.contains('pip')) return;
        e.preventDefault();
        _activatePip();
      };
      for (const pane of [paneA, paneB]) {
        pane.addEventListener('click', _onPaneTap);
        pane.addEventListener('keydown', _onPaneKey);
        pane.setAttribute('role', 'button');
        pane.setAttribute('tabindex', '0');
        pane.setAttribute('aria-label', 'Switch active speaker');
      }

      viewport.classList.add('group');
    } else {
      // Single-avatar mode (default) — one scene, one camera, one renderer.
      const { scene, camera, renderer } = createScene(viewport);
      avatarState.scene = scene;
      avatarState.camera = camera;
      avatarState.renderer = renderer;

      const vrm = await loadVRM(avatarData.vrm_url, _vrmLoadOpts(avatarData));
      _hideLoadingIndicator(viewport);
      scene.add(vrm.scene);
      avatarState.vrm = vrm;
      avatarState.avatarProfile = vrm.__augmentumCompatibilityProfile || null;
      avatarState.secondaryAvatarProfile = null;

      // Auto-frame: compute bounding box and position camera to show head+shoulders
      _autoFrameVRM(vrm, camera);

      // User zoom controls (pinch + scroll wheel + double-tap reset)
      _setupZoomControls(viewport, camera);

      avatarState.animator = new AvatarAnimator(THREE, vrm);
    }
    avatarState.presence = new PresenceEngine({
      avatarProfile: avatarState.avatarProfile,
    });

    // Embodied presence: opt-in FSM. When voice_xr_proxemics_enabled is off,
    // avatarState.avatarFsm stays null and _stampLockedSitPose() applies the
    // seated lock unconditionally (identical to today's behavior).
    try {
      const _cfgResp = await fetch('/api/config/tools', { credentials: 'same-origin' });
      const _cfg = _cfgResp.ok ? await _cfgResp.json() : {};
      if (_cfg.voice_xr_proxemics_enabled) {
        const { AvatarFSM } = await import('./avatar-fsm.js');
        avatarState.avatarFsm = new AvatarFSM({ presence: avatarState.presence });
      }
    } catch (err) {
      console.warn('[avatar] voice_xr_proxemics_enabled probe failed; FSM disabled', err);
    }

    // Create lip sync
    avatarState.lipSync = new AvatarLipSync();
    avatarState.analyserNode = analyserNode;

    // Start animation loop
    startAnimationLoop();

    // Desktop/mobile body physics — pointer-driven contact + compliance +
    // rapier chain. Init lives here (after renderer/scene/camera/vrm are
    // wired) so the avatar starts responsive to touch the moment it's
    // visible. Idempotent; safe across VRM swaps.
    initDesktopBodyPhysics({
      three: THREE,
      vrm: avatarState.vrm,
      renderer: avatarState.renderer,
      camera: avatarState.camera,
    }).catch((err) => console.debug('[avatar] desktop body physics init failed:', err?.message));

    avatarState.mode = 'vrm';
    avatarState.active = true;
    avatarState.loading = false;

    // Update UI
    const toggleBtn = document.getElementById('voice-avatar-toggle');
    if (toggleBtn) toggleBtn.classList.add('active');

    // --- Experience layer (cinematic mode) ---
    _activateExperienceLayer(viewport);

    // Adaptive camera (VRM only — 2D doesn't have a Three.js camera).
    // Tracks the active speaker's camera; in group mode the PIP scene's
    // camera stays static (foundation behavior — per-camera adaptive
    // control can layer on later).
    avatarState.adaptiveCamera = new AdaptiveCamera(avatarState.camera, THREE);

    // Resize Three.js renderer(s) when viewport dimensions change.
    // Catches: experience mode activation, window resize, orientation change.
    // Solo mode has one stack sized to the viewport; group mode has two
    // stacks each sized to their own pane (the .main pane fills the
    // viewport, the .pip pane is a fixed-size overlay).
    const _resizeStack = (renderer, camera, w, h) => {
      if (!renderer || !w || !h) return;
      renderer.setSize(w, h);
      if (camera) {
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      }
    };
    const _doResize = () => {
      // Desktop-pet mode: the avatar canvas lives in the fixed-size voice
      // pill, not this viewport. Re-fit to the pill instead of the (hidden)
      // full-screen overlay.
      if (avatarState.petMode) { _resizePet(); return; }
      const xrPresenting = _isRendererPresentingXR();
      if (xrPresenting) return;
      if (avatarState.paneA && avatarState.paneB) {
        // Group mode: each renderer's canvas lives inside its own pane.
        // The renderer references can swap on speaker change, but the
        // canvas-pane association is stable; size each canvas to the
        // pane it actually lives in.
        for (const renderer of [avatarState.renderer, avatarState.pipRenderer]) {
          if (!renderer) continue;
          const pane = renderer.domElement.parentElement;
          if (!pane) continue;
          const camera = renderer === avatarState.renderer
            ? avatarState.camera
            : avatarState.pipCamera;
          _resizeStack(renderer, camera, pane.clientWidth, pane.clientHeight);
        }
        // Re-frame each character in their own scene now that the
        // aspect has changed. Without this, _autoFrameVRM only ran at
        // activation: rotating phone or snapping the window mid-call
        // leaves the camera at its original aspect, which crops or
        // floats the character. Cheap to recompute (one Box3 + math).
        if (avatarState.vrm && avatarState.camera) {
          _autoFrameVRM(avatarState.vrm, avatarState.camera);
        }
        if (avatarState.secondaryVrm && avatarState.pipCamera) {
          _autoFrameVRM(avatarState.secondaryVrm, avatarState.pipCamera);
        }
      } else {
        // Solo mode
        _resizeStack(avatarState.renderer, avatarState.camera, viewport.clientWidth, viewport.clientHeight);
        if (avatarState.vrm && avatarState.camera) {
          _autoFrameVRM(avatarState.vrm, avatarState.camera);
        }
      }
    };

    // Stash the resize fn so onSpeakerSwitch can trigger it explicitly
    // for an instant crisp upscale on swap.
    avatarState._doResize = _doResize;

    // Observe each pane in group mode so a CSS class flip (.main ↔ .pip
    // changes pane size) automatically resizes the renderer inside.
    // Solo mode falls back to viewport observation.
    //
    // rAF-gate the callback identically to the window-resize handler
    // below. _doResize calls renderer.setSize() which mutates the canvas
    // intrinsic size; without coalescing, a CSS-driven layout pass that
    // observers fire on every frame can pile up sync Three.js work and
    // — worst case — trigger a ResizeObserver loop when canvas resize
    // feeds back into pane sizing.
    let _avatarROPending = false;
    const _doResizeRAFGated = () => {
      if (_avatarROPending) return;
      _avatarROPending = true;
      requestAnimationFrame(() => {
        _avatarROPending = false;
        _doResize();
      });
    };
    avatarState._resizeObserver = new ResizeObserver(_doResizeRAFGated);
    if (avatarState.paneA && avatarState.paneB) {
      avatarState._resizeObserver.observe(avatarState.paneA);
      avatarState._resizeObserver.observe(avatarState.paneB);
    } else {
      avatarState._resizeObserver.observe(viewport);
    }

    // Keep window resize as fallback. rAF-coalesce so a drag-resize burst
    // doesn't trigger 60 Three.js setSize() calls per second.
    if (avatarState._resizeHandler) {
      window.removeEventListener('resize', avatarState._resizeHandler);
    }
    let _avatar3dResizePending = false;
    avatarState._resizeHandler = () => {
      if (_avatar3dResizePending) return;
      _avatar3dResizePending = true;
      requestAnimationFrame(() => {
        _avatar3dResizePending = false;
        _doResize();
      });
    };
    window.addEventListener('resize', avatarState._resizeHandler);

    return true;
  } catch (err) {
    console.error('[avatar] activation failed:', err);
    showAvatarToast('Avatar failed to load');
    // Restore orb on failure
    const orbWrap = document.querySelector('.voice-orb-wrap');
    if (orbWrap) orbWrap.style.display = '';
    document.querySelector('.voice-orb-group')?.setAttribute('data-mode', 'orb');
    const viewport = document.getElementById('voice-avatar-viewport');
    if (viewport) {
      viewport.style.display = 'none';
      viewport.innerHTML = '';
      viewport.classList.remove('avatar-experience', 'drawer-open', 'group');
    }
    document.getElementById('voice-overlay')?.classList.remove('avatar-mode-active', 'avatar-thread-mode');
    avatarState.loading = false;
    return false;
  }
}

// ---- Companion-ambient mode ----------------------------------------------
// Standalone activation for the persistent Becca presence widget. Same scene/
// camera/renderer/VRM/animator/presence pipeline as a voice call — minus the
// session fetch, group plumbing, audio-analyser binding, transcript bindings,
// and the cinema experience layers (atmosphere/subtitle/drawer). The VRM
// breathes, blinks, micro-gestures, and follows the pose-preset system from
// PresenceEngine without any audio source. When a voice call starts later,
// the same canvas can be reparented into the call viewport and the analyser
// can be bound at that point — no re-instantiation.

/**
 * Activate the avatar in companion-ambient mode (no voice call binding).
 *
 * @param {Object} opts
 * @param {HTMLElement} opts.host    DOM element to mount the canvas into.
 * @param {string} opts.vrmUrl       VRM model URL.
 * @param {Object} [opts.vrmLoadOpts] Optional loader overrides (e.g. faceRotationY).
 * @param {boolean} [opts.skipExperience=true] Skip atmosphere/subtitle/drawer.
 * @returns {Promise<boolean>} true on success.
 */
export async function activateAvatarStandalone(opts = {}) {
  const { host, vrmUrl, vrmLoadOpts = {}, skipExperience = true, targetFps = 0 } = opts;
  if (!host || !vrmUrl) {
    console.warn('[avatar] activateAvatarStandalone: host and vrmUrl required');
    return false;
  }
  if (avatarState.active || avatarState.loading) return false;
  avatarState.loading = true;
  avatarState._contextLost = false;

  try {
    await ensureLibsLoaded();

    const { scene, camera, renderer } = createScene(host, { quality: 'high' });
    avatarState.scene = scene;
    avatarState.camera = camera;
    avatarState.renderer = renderer;

    const vrm = await loadVRM(vrmUrl, vrmLoadOpts);
    scene.add(vrm.scene);
    avatarState.vrm = vrm;
    avatarState.avatarProfile = vrm.__augmentumCompatibilityProfile || null;
    avatarState.secondaryAvatarProfile = null;

    // Bump every VRM texture to max anisotropy. The 3-point lighting +
    // free camera framing means a lot of textures end up at oblique
    // angles (hairline, cheek, side of nose, eyelashes) where bilinear
    // filtering turns to mush. Anisotropic filtering is essentially
    // free once textures have mipmaps, which the VRM loader already
    // generates for us.
    _applyMaxAnisotropy(vrm.scene, renderer);

    _autoFrameVRM(vrm, camera);

    avatarState.animator = new AvatarAnimator(THREE, vrm);
    avatarState.presence = new PresenceEngine({
      avatarProfile: avatarState.avatarProfile,
    });

    // Interoception substrate — synthetic physiology + affective state
    // peer to PresenceEngine. Standalone-only in v0: the call path's
    // visible behavior is dominated by lipsync and doesn't need an
    // interior. Design doc:
    //   docs/superpowers/specs/2026-05-16-interoception-engine-design.md
    avatarState.interoception = new InteroceptionEngine();

    // FSM probe — matches the call path so XR-style proxemics are available
    // when the setting is on. Cheap; one fetch.
    try {
      const _cfgResp = await fetch('/api/config/tools', { credentials: 'same-origin' });
      const _cfg = _cfgResp.ok ? await _cfgResp.json() : {};
      if (_cfg.voice_xr_proxemics_enabled) {
        const { AvatarFSM } = await import('./avatar-fsm.js');
        avatarState.avatarFsm = new AvatarFSM({ presence: avatarState.presence });
      }
    } catch (err) {
      console.warn('[avatar] standalone: FSM probe failed', err);
    }

    // LipSync is instantiated so a later analyser bind (call start) is a
    // one-line swap. Mouth stays neutral until analyserNode is non-null.
    avatarState.lipSync = new AvatarLipSync();
    avatarState.analyserNode = null;

    startAnimationLoop();

    // Desktop/mobile body physics — pointer-driven contact + compliance +
    // rapier chain. Mirrors the call in the voice-call activate path.
    initDesktopBodyPhysics({
      three: THREE,
      vrm: avatarState.vrm,
      renderer: avatarState.renderer,
      camera: avatarState.camera,
    }).catch((err) => console.debug('[avatar] desktop body physics init failed:', err?.message));

    avatarState.mode = 'vrm';
    avatarState.active = true;
    avatarState.loading = false;
    avatarState._standalone = true;
    avatarState._standaloneHost = host;
    // Frame-rate cap for ambient presence surfaces (live wallpaper / lock-screen
    // idle). 0 = uncapped (the floating widget + interactive avatar stay full-rate).
    avatarState._targetFrameInterval = targetFps > 0 ? 1000 / targetFps : 0;
    avatarState._lastRenderT = 0;

    if (!skipExperience) {
      _activateExperienceLayer(host);
    }

    avatarState.adaptiveCamera = new AdaptiveCamera(avatarState.camera, THREE);

    // "First Breath" entrance — the moment she first appears (the standalone
    // companion/chat view, which otherwise snapped straight to idle with no
    // greeting), play a short arrival → recognition → greeting beat, then let
    // the living idle reclaim her. canFullBody=FALSE: she greets from her
    // natural arms-DOWN posture (procedural gestures layer on the idle pose, so
    // her arms never splay). A full-body VRMA wave was tried but its mixer fades
    // in from the model's BIND pose (T-pose, arms out), so her arms swung
    // through a "Y" before the wave — not worth it. Gated by a cooldown +
    // escalating warmth so a refresh doesn't re-trigger it. Fully guarded
    // internally — it can never break activation. The controller is stashed so
    // the user engaging immediately can abbreviate the beat. (The voice-call
    // path greets separately via PoseTriggerEngine.onCallOpened.)
    try {
      avatarState._entranceController = playEntrance({
        animator: avatarState.animator,
        presence: avatarState.presence,
        conductor: avatarState.conductor,
        adaptiveCamera: avatarState.adaptiveCamera,
      }, { now: Date.now(), canFullBody: false });
    } catch (err) {
      console.debug('[avatar] entrance skipped:', err?.message);
    }

    // Resize plumbing — observe the host instead of the call viewport.
    const _doResize = () => {
      if (avatarState.petMode) { _resizePet(); return; }
      const w = host.clientWidth;
      const h = host.clientHeight;
      if (!avatarState.renderer || !w || !h) return;
      avatarState.renderer.setSize(w, h);
      if (avatarState.camera) {
        avatarState.camera.aspect = w / h;
        avatarState.camera.updateProjectionMatrix();
      }
      if (avatarState.vrm && avatarState.camera) {
        _autoFrameVRM(avatarState.vrm, avatarState.camera);
      }
    };
    avatarState._doResize = _doResize;

    let _roPending = false;
    avatarState._resizeObserver = new ResizeObserver(() => {
      if (_roPending) return;
      _roPending = true;
      requestAnimationFrame(() => { _roPending = false; _doResize(); });
    });
    avatarState._resizeObserver.observe(host);

    if (avatarState._resizeHandler) {
      window.removeEventListener('resize', avatarState._resizeHandler);
    }
    let _winPending = false;
    avatarState._resizeHandler = () => {
      if (_winPending) return;
      _winPending = true;
      requestAnimationFrame(() => { _winPending = false; _doResize(); });
    };
    window.addEventListener('resize', avatarState._resizeHandler);

    // Initial frame to settle aspect after canvas append.
    _doResize();

    return true;
  } catch (err) {
    console.error('[avatar] standalone activation failed:', err);
    avatarState.loading = false;
    return false;
  }
}


// ---- Desktop-pet mode -----------------------------------------------------
// When a solo VRM call is minimized, the live avatar canvas is reparented out
// of the (now-hidden) #voice-avatar-viewport and into the floating voice pill,
// which grows into a small portrait frame. The WebGL context is untouched —
// only the DOM parent and render size change — so the avatar keeps idling and
// gesturing with no reload or warm-up flicker. expandVoiceCall() reverses it.

export function canEnterPetMode() {
  // Groups are supported now: we reparent only the active speaker's canvas
  // (the inactive VRM stays attached to its hidden pane). Speaker swaps
  // crossfade the canvas in the pet host — see onSpeakerSwitch's pet path.
  return !!(avatarState.active
    && avatarState.mode === 'vrm'
    && avatarState.vrm
    && avatarState.renderer);
}

function _petCanvases() {
  // The renderer canvas plus its sibling 2D context-loss fallback.
  const out = [];
  const c = avatarState.renderer?.domElement;
  if (c) out.push(c);
  const fb = c?.parentElement?.querySelector('.avatar-vrm-fallback');
  if (fb) out.push(fb);
  return out;
}

function _resizePet() {
  const host = avatarState._petHost;
  const r = avatarState.renderer;
  if (!host || !r) return;
  const w = host.clientWidth || 132;
  const h = host.clientHeight || 150;
  r.setSize(w, h);
  if (avatarState.camera) {
    avatarState.camera.aspect = w / h;
    avatarState.camera.updateProjectionMatrix();
  }
  if (!_isRendererPresentingXR() && avatarState.vrm && avatarState.camera) {
    _autoFrameVRM(avatarState.vrm, avatarState.camera);
  }
}

/** Detach inline styles applied during a pet-mode crossfade so a canvas
 *  is layout-neutral when reparented back to its pane / viewport. */
function _clearPetTransitionStyles(canvas) {
  if (!canvas) return;
  canvas.style.position = '';
  canvas.style.inset = '';
  canvas.style.width = '';
  canvas.style.height = '';
  canvas.style.opacity = '';
  canvas.style.transition = '';
  canvas.style.pointerEvents = '';
}

export function enterPetMode(hostEl) {
  if (!hostEl || avatarState.petMode || !canEnterPetMode()) return false;
  // Group: reparent only the active speaker's canvas. The inactive VRM
  // stays bound to its pane (invisible while the overlay is minimized);
  // its renderer keeps drawing into a canvas that isn't on screen, same
  // as today's full-screen-but-PIP behavior, so no new GPU surprise.
  // Solo: reparent both the renderer canvas and any 2D fallback sibling.
  if (avatarState.paneA && avatarState.paneB) {
    const activeCanvas = avatarState.renderer?.domElement;
    if (activeCanvas) hostEl.appendChild(activeCanvas);
  } else {
    for (const el of _petCanvases()) hostEl.appendChild(el);
  }
  avatarState.petMode = true;
  avatarState._petHost = hostEl;
  void hostEl.offsetHeight;  // flush layout so clientWidth/Height are real
  _resizePet();
  return true;
}

export function exitPetMode() {
  if (!avatarState.petMode) return;
  // Group: return the active canvas to ITS pane (not the viewport root)
  // so the split-pane structure is intact when the overlay re-expands.
  // The inactive canvas never left its pane; nothing to do for it.
  // Solo: canvases go back into the single viewport.
  if (avatarState.paneA && avatarState.paneB) {
    const activeCanvas = avatarState.renderer?.domElement;
    _clearPetTransitionStyles(activeCanvas);
    const targetPane = avatarState.activeSpeaker === 1 ? avatarState.paneB : avatarState.paneA;
    if (activeCanvas && targetPane) targetPane.appendChild(activeCanvas);
  } else {
    const viewport = document.getElementById('voice-avatar-viewport');
    if (viewport) {
      for (const el of _petCanvases()) viewport.appendChild(el);
    }
  }
  avatarState.petMode = false;
  avatarState._petHost = null;
  // Restore full-size framing once the viewport is visible/laid-out again.
  if (typeof avatarState._doResize === 'function') {
    requestAnimationFrame(() => avatarState._doResize());
  }
}

export function deactivateAvatar() {
  if (!avatarState.active) return;
  // Mark teardown in flight so the deliberate forceContextLoss() below
  // doesn't trigger the snapshot-fallback path in the webglcontextlost
  // listener.
  avatarState._disposing = true;
  avatarState.petMode = false;
  avatarState._petHost = null;

  // Tear down desktop body physics FIRST — its pointer listeners are
  // attached to the renderer's domElement which we're about to dispose,
  // and Rapier's async teardown wants the world before dispose() lands.
  // Fire-and-forget: the Rapier await is capped at 500ms internally.
  teardownDesktopBodyPhysics().catch((err) => {
    console.debug('[avatar] desktop body physics teardown error:', err?.message);
  });

  try {

  // Dispose experience layers
  if (avatarState.atmosphere) {
    avatarState.atmosphere.dispose();
    avatarState.atmosphere = null;
  }
  if (avatarState.subtitle) {
    avatarState.subtitle.dispose();
    avatarState.subtitle = null;
  }
  if (avatarState.drawer) {
    avatarState.drawer.dispose();
    avatarState.drawer = null;
  }
  avatarState.adaptiveCamera = null;
  avatarState.experienceMode = false;
  avatarState._sentenceBuffer = '';

  // Restore transcript divs
  const transcriptUser = document.querySelector('.voice-transcript-user');
  const transcriptAi = document.querySelector('.voice-transcript-ai');
  const transcriptLog = document.querySelector('.voice-transcript-log');
  if (transcriptUser) transcriptUser.classList.remove('avatar-experience-hidden');
  if (transcriptAi) transcriptAi.classList.remove('avatar-experience-hidden');
  if (transcriptLog) transcriptLog.classList.remove('avatar-experience-hidden');

  const viewport = document.getElementById('voice-avatar-viewport');
  if (viewport) viewport.classList.remove('avatar-experience', 'drawer-open', 'group', 'group-split');
  document.getElementById('voice-overlay')?.classList.remove('avatar-mode-active', 'avatar-thread-mode');

  // Stop animation loop. The 3D path uses renderer.setAnimationLoop;
  // the 2D portrait path stays on requestAnimationFrame (animFrameId).
  // Handle both. If an XR session is presenting, end it first so
  // renderer disposal below runs cleanly — three.js's setSession
  // teardown depends on the session being out of presenting state.
  if (avatarState.renderer) {
    if (avatarState.renderer.xr?.isPresenting) {
      try { avatarState.renderer.xr.getSession()?.end(); } catch { /* ignore */ }
    }
    try { avatarState.renderer.setAnimationLoop(null); } catch { /* ignore */ }
  }
  // Clear pause/resume bookkeeping so the next activateAvatar starts
  // from a known-clean state. Without this, a paused-then-deactivated
  // sequence would leave ``_renderPaused=true`` on the singleton and
  // the next activation's first resume call would noop incorrectly.
  avatarState._animate = null;
  avatarState._clock = null;
  avatarState._renderPaused = false;
  avatarState._targetFrameInterval = 0;
  avatarState._lastRenderT = 0;
  if (avatarState.animFrameId) {
    cancelAnimationFrame(avatarState.animFrameId);
    avatarState.animFrameId = null;
  }

  // Stop any in-flight VRMA — releases mixer references to the VRM scene
  // BEFORE we dispose the scene contents below.
  stopVrma();

  // Dispose 2D renderer if active
  if (avatarState.renderer2D) {
    avatarState.renderer2D.dispose();
    avatarState.renderer2D = null;
  }
  avatarState.mode = null;

  // Dispose Three.js resources
  if (avatarState._snapshotInterval) {
    clearInterval(avatarState._snapshotInterval);
    avatarState._snapshotInterval = null;
  }
  avatarState._contextLost = false;

  // CRITICAL: dispose all GPU resources BEFORE renderer.dispose().
  // Without this, geometries/materials/textures from the VRM leak each
  // open/close cycle. After ~16 cycles the browser hits its WebGL
  // context limit and the entire app renders as RGB noise.
  // Use VRMUtils.deepDispose if available (canonical VRM cleanup);
  // fall back to manual scene traversal otherwise.
  const _disposeMaterial = (m) => {
    if (!m) return;
    Object.keys(m).forEach((key) => {
      const v = m[key];
      if (v && typeof v === 'object' && 'minFilter' in v) v.dispose?.();   // texture
    });
    m.dispose?.();
  };
  const _disposeSceneContents = (root) => {
    if (!root) return;
    root.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose?.();
      if (obj.material) {
        if (Array.isArray(obj.material)) obj.material.forEach(_disposeMaterial);
        else _disposeMaterial(obj.material);
      }
      if (obj.skeleton) obj.skeleton.dispose?.();
    });
  };
  if (avatarState.vrm?.scene) {
    if (VRMModule?.VRMUtils?.deepDispose) {
      VRMModule.VRMUtils.deepDispose(avatarState.vrm.scene);
    } else {
      _disposeSceneContents(avatarState.vrm.scene);
    }
  }
  if (avatarState.secondaryVrm?.scene) {
    if (VRMModule?.VRMUtils?.deepDispose) {
      VRMModule.VRMUtils.deepDispose(avatarState.secondaryVrm.scene);
    } else {
      _disposeSceneContents(avatarState.secondaryVrm.scene);
    }
  }
  _disposeSceneContents(avatarState.scene);

  // Dispose both rendering stacks. In solo mode .pipRenderer is null
  // so the second pass is a no-op. WebGL context limit is ~16, and
  // group mode uses 2 contexts at once, so explicit forceContextLoss
  // matters even more here.
  const _disposeRenderer = (renderer) => {
    if (!renderer) return;
    // PMREM environment target: dispose the render target + the PMREM
    // generator before the renderer goes, so neither leaks a GL slot when
    // we forceContextLoss(). Both are one-shot resources from createScene.
    try { avatarState._envRenderTarget?.dispose?.(); } catch { /* ignore */ }
    try { avatarState._pmrem?.dispose?.(); } catch { /* ignore */ }
    avatarState._envRenderTarget = null;
    avatarState._pmrem = null;
    renderer.dispose();
    try {
      renderer.forceContextLoss?.();
      const ctx = renderer.getContext?.();
      const loseExt = ctx?.getExtension?.('WEBGL_lose_context');
      loseExt?.loseContext?.();
    } catch (e) { /* swallow — best effort */ }
    renderer.domElement?.remove();
  };
  _disposeRenderer(avatarState.renderer);
  avatarState.renderer = null;
  if (avatarState.pipRenderer) {
    if (avatarState.pipScene) _disposeSceneContents(avatarState.pipScene);
    _disposeRenderer(avatarState.pipRenderer);
    avatarState.pipRenderer = null;
    avatarState.pipScene = null;
    avatarState.pipCamera = null;
  }
  // Remove the split-pane DOM elements — leaving them behind would
  // cause the next solo activation to render its canvas alongside
  // stale group panes.
  if (avatarState.paneA) { avatarState.paneA.remove(); avatarState.paneA = null; }
  if (avatarState.paneB) { avatarState.paneB.remove(); avatarState.paneB = null; }
  if (avatarState.presence) {
    avatarState.presence.dispose();
    avatarState.presence = null;
  }
  if (avatarState.interoception) {
    avatarState.interoception.dispose();
    avatarState.interoception = null;
  }
  if (avatarState.secondaryPresence) {
    avatarState.secondaryPresence.dispose();
    avatarState.secondaryPresence = null;
  }
  if (avatarState.animator) {
    avatarState.animator.dispose?.();
    avatarState.animator = null;
  }
  if (avatarState.secondaryAnimator) {
    avatarState.secondaryAnimator.dispose?.();
    avatarState.secondaryAnimator = null;
  }
  avatarState.scene = null;
  avatarState.camera = null;
  avatarState.vrm = null;
  avatarState.xrFrameHandler = null;
  avatarState.secondaryVrm = null;
  avatarState.avatarProfile = null;
  avatarState.secondaryAvatarProfile = null;
  avatarState.lipSync = null;
  avatarState.analyserNode = null;

  // Show orb, hide avatar viewport
  const orbWrap = document.querySelector('.voice-orb-wrap');
  if (orbWrap) orbWrap.style.display = '';
  document.querySelector('.voice-orb-group')?.setAttribute('data-mode', 'orb');
  if (viewport) { viewport.style.display = 'none'; viewport.innerHTML = ''; }

  // Remove resize observers
  if (avatarState._resizeObserver) {
    avatarState._resizeObserver.disconnect();
    avatarState._resizeObserver = null;
  }
  if (avatarState._resizeHandler) {
    window.removeEventListener('resize', avatarState._resizeHandler);
    avatarState._resizeHandler = null;
  }
  if (avatarState._zoomAbortController) {
    avatarState._zoomAbortController.abort();
    avatarState._zoomAbortController = null;
  }
  // Disconnect the no-history class observers (added in _activateExperienceLayer)
  if (avatarState._historyObserver) {
    avatarState._historyObserver.disconnect();
    avatarState._historyObserver = null;
  }
  if (avatarState._collapseObserver) {
    avatarState._collapseObserver.disconnect();
    avatarState._collapseObserver = null;
  }
  // Clear any no-history class so it doesn't persist for next session
  document.getElementById('voice-overlay')?.classList.remove('no-history');

  avatarState.active = false;
  // Reset speaking-state bus keys atomically — anything subscribed will
  // see the deactivated state in one fan-out, not three.
  bus.setBatch({
    voice_state: 'idle',
    tts_playing: false,
    is_speaking: false,
  });
  avatarState.callMode = null;
  avatarState.characterId = '';
  avatarState.avatarId = '';
  avatarState.avatarName = '';

  // Update UI
  const toggleBtn = document.getElementById('voice-avatar-toggle');
  if (toggleBtn) toggleBtn.classList.remove('active');

  } catch (err) {
    console.warn('[avatar] deactivate threw mid-disposal:', err);
  } finally {
    // GUARANTEED visual+state cleanup. Runs even if any disposal step in
    // the body above threw. Without this, a single failure (half-loaded
    // VRM, double-freed three.js material, presence engine in an
    // intermediate state) used to leave the avatar viewport painted on
    // screen with the last frozen frame plus the fallback snapshot, and
    // avatarState.active pinned true so a fresh activate was blocked.
    try {
      const orbWrap = document.querySelector('.voice-orb-wrap');
      if (orbWrap) orbWrap.style.display = '';
      // Reset the orb-group's data-mode so CSS reverts to orb layout.
      // If we skip this, a future activate that throws before flipping
      // it would leave the parent stuck in avatar-mode sizing.
      document.querySelector('.voice-orb-group')?.setAttribute('data-mode', 'orb');
    } catch { /* document gone */ }
    try {
      const viewport = document.getElementById('voice-avatar-viewport');
      if (viewport) { viewport.style.display = 'none'; viewport.innerHTML = ''; }
    } catch { /* document gone */ }
    try {
      const toggleBtn = document.getElementById('voice-avatar-toggle');
      if (toggleBtn) toggleBtn.classList.remove('active');
    } catch { /* document gone */ }
    avatarState.active = false;
    avatarState._disposing = false;
    avatarState._contextLost = false;
    avatarState._standalone = false;
    avatarState._standaloneHost = null;

    // Reactivation of the companion-ambient standalone path lives at the
    // ONE place that actually owns "the call is over": _teardownVoiceCall
    // in voice.js. We do NOT auto-reactivate here, because deactivate
    // fires in many contexts that should NOT bring the widget back —
    // mid-call avatar toggle off, avatar-picker switching, switchAvatar,
    // call-mode swaps. The old eager-reactivate caused a feedback loop:
    // every mid-call deactivate woke the widget, the widget's fresh
    // WebGL context fought the call for GL slots, and once the browser's
    // ~16-context limit was hit every new context spawned already-lost.
  }
}

/**
 * Switch to a different avatar mid-call.
 * Tears down the current avatar and activates the new one.
 * @param {string} characterId — character ID to resolve avatar for (empty = default)
 */
export async function switchAvatar(selection = {}) {
  if (avatarState.loading) return;
  const analyserNode = avatarState.analyserNode;
  if (!analyserNode) return;

  const characterId = typeof selection === 'string'
    ? selection
    : (selection.characterId || '');
  const avatarId = typeof selection === 'object'
    ? (selection.avatarId || '')
    : '';
  const mode = typeof selection === 'object' && selection.mode
    ? selection.mode
    : (avatarState.callMode || app.state.mode || 'passthrough');

  // Tear down current avatar (but keep the overlay open)
  deactivateAvatar();

  // Small delay so the DOM cleans up
  await new Promise(r => setTimeout(r, 100));

  await activateAvatar(analyserNode, {
    mode,
    characterId,
    avatarId,
  });
}

/**
 * Show an avatar picker popover with all available avatars.
 * Tapping one calls switchAvatar.
 */
export async function showAvatarPicker() {
  // Remove existing picker if open
  document.querySelector('.avatar-picker-overlay')?.remove();

  let avatars = [];
  try {
    const resp = await fetch('/api/avatar/list');
    if (resp.ok) {
      const data = await resp.json();
      avatars = data.avatars || [];
    }
  } catch { return; }

  if (!avatars.length) {
    try { showToast('No avatars available', 'info'); } catch {}
    return;
  }

  const overlay = document.createElement('div');
  overlay.className = 'avatar-picker-overlay';

  let html = '<div class="avatar-picker-panel">';
  html += '<div class="avatar-picker-header">Switch Avatar</div>';
  html += '<div class="avatar-picker-grid">';
  for (const a of avatars) {
    const avatarId = a.id || a.avatar_id || '';
    const thumb = a.type === 'portrait' && a.portrait_url
      ? a.portrait_url
      : (a.thumbnail_url || a.portrait_url || '');
    const name = a.name || (a.type === 'portrait' ? '2D Portrait' : 'VRM Avatar');
    const charId = a.character_id || '';
    html += `<button class="avatar-picker-item" data-character-id="${escapeHtml(charId)}" data-avatar-id="${escapeHtml(avatarId)}">`;
    html += thumb
      ? `<img src="${escapeHtml(thumb)}" alt="" class="avatar-picker-thumb" loading="lazy">`
      : `<div class="avatar-picker-thumb avatar-picker-placeholder">?</div>`;
    html += `<span class="avatar-picker-name">${escapeHtml(name)}</span>`;
    html += '</button>';
  }
  html += '</div></div>';
  overlay.innerHTML = html;

  // Click handler
  overlay.addEventListener('click', (e) => {
    const item = e.target.closest('.avatar-picker-item');
    if (item) {
      const characterId = item.dataset.characterId || '';
      const avatarId = item.dataset.avatarId || '';
      overlay.remove();
      switchAvatar({ avatarId, characterId });
      return;
    }
    // Click outside panel closes
    if (!e.target.closest('.avatar-picker-panel')) {
      overlay.remove();
    }
  });

  document.body.appendChild(overlay);
}

export function onStateChange(state) {
  bus.set('voice_state', state);
  _syncSpeakingState();

  // Abbreviate the "First Breath" entrance the moment the user actually engages
  // (starts talking / the AI starts replying) — she cuts the greeting short and
  // gets straight to attentive rather than waving over a live exchange.
  if (state === 'recording' || state === 'speaking') {
    avatarState._entranceController?.cancel?.();
  }

  // Feed state to presence engine. The primary engine sees the state
  // verbatim. The secondary (group only) sees 'speaking' as
  // 'peer_speaking' — from its perspective, someone ELSE is holding the
  // floor, not "an AI" — which gates _updateListening on.
  if (avatarState.presence) {
    avatarState.presence.onStateChange(state);
  }
  if (avatarState.secondaryPresence) {
    const peerState = state === 'speaking' ? 'peer_speaking' : state;
    avatarState.secondaryPresence.onStateChange(peerState);
  }

  // Reset sentence buffer on new response
  if (state === 'speaking') {
    avatarState._sentenceBuffer = '';
  }

  // Diagnostic: log viewport visibility state on every transition
  if (avatarState.active) {
    const vp = document.getElementById('voice-avatar-viewport');
    const root = vp?.querySelector('.avatar-2d-root');
    const baseImg = vp?.querySelector('.avatar-2d-base');
    const vpRect = vp?.getBoundingClientRect();
    console.debug('[avatar-debug] state:', state,
      'vp.display:', vp?.style.display,
      'vp.classes:', vp?.className,
      'vp.rect:', vpRect ? `${Math.round(vpRect.width)}x${Math.round(vpRect.height)}` : 'null',
      'root:', !!root,
      'baseImg:', !!baseImg,
      'baseImg.naturalW:', baseImg?.naturalWidth,
      'baseImg.complete:', baseImg?.complete);
  }
}

export function onTtsPlaybackChange(active) {
  bus.set('tts_playing', !!active);
  _syncSpeakingState();

  // If real audio starts after the visible state has fallen back to listening,
  // presence should still behave like the AI is speaking. Mirror to the
  // secondary engine with role mapping (peer_speaking while audio is live).
  const primaryState = bus.state.tts_playing
    ? 'speaking'
    : (bus.state.voice_state || 'idle');
  if (avatarState.presence) {
    avatarState.presence.onStateChange(primaryState);
  }
  if (avatarState.secondaryPresence) {
    const peerState = primaryState === 'speaking' ? 'peer_speaking' : primaryState;
    avatarState.secondaryPresence.onStateChange(peerState);
  }
}

/** Fan-out helper: feed user transcript text to every active presence
 *  engine. The user is speaking to all characters in the call, so both
 *  the primary and (in groups) the secondary should react to it. */
export function onUserTranscript(text, isFinal) {
  if (avatarState.presence) avatarState.presence.onUserTranscript(text, isFinal);
  if (avatarState.secondaryPresence) avatarState.secondaryPresence.onUserTranscript(text, isFinal);
}

/** Fan-out helper: feed user mic RMS to every active presence engine.
 *  Used by voice/STT paths that don't go through the animate-loop RMS
 *  pickup (e.g. an explicit external publisher). */
export function onUserAudioRMS(rms) {
  if (avatarState.presence) avatarState.presence.onUserAudioRMS(rms);
  if (avatarState.secondaryPresence) avatarState.secondaryPresence.onUserAudioRMS(rms);
}

export function onLLMDelta(text) {
  if (!avatarState.presence && !avatarState.animator && !avatarState.renderer2D) return;

  // Extract [gesture:name] tags — high-confidence explicit signals
  const gestureTagRe = /\[gesture:(\w+)\]/g;
  let match;
  let cleanText = text;
  while ((match = gestureTagRe.exec(text)) !== null) {
    if (avatarState.presence) {
      avatarState.presence.onExplicitGesture(match[1]);
    } else if (avatarState.animator?.triggerGesture) {
      avatarState.animator.triggerGesture(match[1]);
    }
    cleanText = cleanText.replace(match[0], '');
  }

  // Subtitle sentence detection
  if (avatarState.subtitle) {
    if (!avatarState._sentenceBuffer) avatarState._sentenceBuffer = '';
    avatarState._sentenceBuffer += cleanText;
    const sentenceRe = /[^.!?]*[.!?]+/g;
    let sMatch;
    while ((sMatch = sentenceRe.exec(avatarState._sentenceBuffer)) !== null) {
      avatarState.subtitle.setAISentence(sMatch[0].trim());
    }
    const lastEnd = avatarState._sentenceBuffer.lastIndexOf('.') + 1
      || avatarState._sentenceBuffer.lastIndexOf('!') + 1
      || avatarState._sentenceBuffer.lastIndexOf('?') + 1;
    if (lastEnd > 0) {
      avatarState._sentenceBuffer = avatarState._sentenceBuffer.slice(lastEnd);
    }
  }

  // Feed to presence engine (replaces keyword matching)
  if (avatarState.presence) {
    avatarState.presence.onLLMDelta(cleanText);
  }
}

/**
 * Signal that a chat stream is active (true) or has finished (false).
 *
 * Thin wrapper over ``bus.set('chat_streaming', …)``. Kept as a named
 * export so external surfaces that aren't activity-bus-aware (extensions,
 * tests) have a stable entry point. Internal consumers should read
 * ``bus.state.chat_streaming`` directly.
 */
export function notifyChatStreaming(active) {
  bus.set('chat_streaming', !!active);
}

export function onSpeakerSwitch(characterName) {
  if (!avatarState.groupMembers || avatarState.groupMembers.length < 2) return;

  const idx = avatarState.groupMembers.findIndex(m => m.name === characterName);
  if (idx < 0 || idx === avatarState.activeSpeaker) return;

  // Pet-mode pre-pass: capture canvas refs and pre-size the incoming
  // renderer to pet host dims, so the crossfade after the state swap
  // doesn't show a jarring size jump. Skipped when not in pet-mode —
  // the existing FLIP-based pane swap runs unchanged then.
  const inPet = avatarState.petMode && avatarState._petHost;
  let _petOutgoingCanvas = null;
  let _petIncomingCanvas = null;
  if (inPet) {
    _petOutgoingCanvas = avatarState.renderer?.domElement || null;
    _petIncomingCanvas = avatarState.pipRenderer?.domElement || null;
    const host = avatarState._petHost;
    const w = host.clientWidth || 132;
    const h = host.clientHeight || 150;
    // Resize the incoming renderer and force a single synchronous render
    // so the first frame inside the pet host is correctly sized — without
    // this the user sees one frame at the small PIP backing-store size.
    if (avatarState.pipRenderer) {
      try { avatarState.pipRenderer.setSize(w, h); } catch { /* ignore */ }
    }
    if (avatarState.pipCamera) {
      avatarState.pipCamera.aspect = w / h;
      avatarState.pipCamera.updateProjectionMatrix();
    }
    if (avatarState.secondaryVrm && avatarState.pipCamera) {
      _autoFrameVRM(avatarState.secondaryVrm, avatarState.pipCamera);
    }
    if (avatarState.pipRenderer && avatarState.pipScene && avatarState.pipCamera) {
      try { avatarState.pipRenderer.render(avatarState.pipScene, avatarState.pipCamera); }
      catch { /* renderer may be mid-disposal during a race; the crossfade still proceeds */ }
    }
  }

  // Swap which pane wears .main / .pip CSS classes — the visible
  // hierarchy follows the active speaker.
  _setActivePaneClasses(idx);

  // Swap stack pointers so .scene/.camera/.renderer always refer to
  // the active speaker's stack and .pipScene/.pipCamera/.pipRenderer
  // to the non-speaker's. The DOM canvases don't move (each renderer
  // owns the canvas inside its own pane); only the avatarState refs
  // re-aim so downstream lipsync/animator/render code keeps working.
  // Plain temp-var swaps — the destructuring-assignment one-liner
  // form is correct ES, but easier to mistrust at a glance.
  const _swap = (k1, k2) => {
    const t = avatarState[k1];
    avatarState[k1] = avatarState[k2];
    avatarState[k2] = t;
  };
  _swap('vrm', 'secondaryVrm');
  _swap('animator', 'secondaryAnimator');
  _swap('avatarProfile', 'secondaryAvatarProfile');
  _swap('scene', 'pipScene');
  _swap('camera', 'pipCamera');
  _swap('renderer', 'pipRenderer');
  // Presence engine follows the character via the role pointer swap —
  // each VRM keeps its own engine, the pointer just re-aims. After the
  // swap, .presence is the new speaker's engine; .secondaryPresence is
  // the new listener's.
  _swap('presence', 'secondaryPresence');

  avatarState.activeSpeaker = idx;

  // Re-aim profile references on each engine (they tag along through
  // the swap, but the swap may have crossed avatarProfile too — so
  // re-bind explicitly to keep "this engine speaks for THIS profile"
  // unambiguous for any subscriber that reads engine._avatarProfile).
  avatarState.presence?.setAvatarProfile?.(avatarState.avatarProfile);
  avatarState.secondaryPresence?.setAvatarProfile?.(avatarState.secondaryAvatarProfile);

  // Announce role change to each engine so their flow targets retarget
  // immediately rather than drifting through the decay over ~halflife.
  // New primary: about to speak. New secondary: peer is now speaking.
  avatarState.presence?.onStateChange('speaking');
  avatarState.secondaryPresence?.onStateChange('peer_speaking');

  // Release the now-secondary animator's speaker-state: cancels any
  // active call action (which would have left an explicit hand pose
  // frozen on the inactive character) and resets the hand-pose channel
  // to 'relaxed' baseline. Without this, a character who was mid-"let
  // me explain" call action when the speaker swapped would visibly hold
  // that hand shape until they next become the speaker again — the
  // root cause of the "weird hand positioning" symptom in groups.
  // In-progress short gestures (nod, surprise) are left to finish.
  avatarState.secondaryAnimator?.releaseActiveSpeakerState?.();

  if (inPet && _petOutgoingCanvas && _petIncomingCanvas) {
    _petCrossfadeCanvas(_petOutgoingCanvas, _petIncomingCanvas, idx);
  } else if (avatarState._doResize) {
    // Resize both renderers — the canvas that just got promoted from .pip
    // to .main has a small backing store and needs to upscale, otherwise
    // the new main view is blurry. Conversely the demoted canvas can
    // shrink to save GPU time. ResizeObservers on each pane normally
    // catch this, but we trigger explicitly so the swap looks crisp on
    // the very first frame after the click.
    avatarState._doResize();
  }

  if (avatarState.presence?.setAvatarProfile) {
    avatarState.presence.setAvatarProfile(avatarState.avatarProfile);
  }
}

/** Crossfade swap of the visible canvas inside the pet pill.
 *
 *  Both canvases briefly co-exist in the pet host, layered with absolute
 *  positioning. The outgoing fades to 0 and the incoming fades to 1 over
 *  ~250ms; on transitionend the outgoing canvas is returned to its pane
 *  (paneA or paneB based on the post-swap activeSpeaker — the outgoing
 *  is now the "pip" half, so it belongs to whichever pane is NOT the new
 *  active speaker's). Falls back to a timeout in case transitionend
 *  doesn't fire (browser tab throttled, transition cancelled). */
function _petCrossfadeCanvas(outgoing, incoming, newActiveIdx) {
  const host = avatarState._petHost;
  if (!host) return;

  // Both canvases positioned absolutely so they stack rather than flowing
  // side-by-side and re-laying-out the host mid-swap.
  for (const c of [outgoing, incoming]) {
    c.style.position = 'absolute';
    c.style.inset = '0';
    c.style.width = '100%';
    c.style.height = '100%';
    c.style.pointerEvents = 'none';
  }
  incoming.style.opacity = '0';
  incoming.style.transition = 'opacity 250ms ease';
  outgoing.style.transition = 'opacity 250ms ease';

  host.appendChild(incoming);
  // Flush the initial opacity=0 paint before flipping it to 1, otherwise
  // the browser batches both and skips the transition.
  void incoming.offsetHeight;
  incoming.style.opacity = '1';
  outgoing.style.opacity = '0';

  let done = false;
  const finalize = () => {
    if (done) return;
    done = true;
    // Outgoing's pane is the one NOT owning the new active speaker.
    // (After the state swap, paneA still hosts canvas-0 and paneB still
    // hosts canvas-1 in DOM ownership terms — but we've taken one of them
    // into the pet host. Put it back in the pane it came from.)
    const outgoingPane = newActiveIdx === 0 ? avatarState.paneB : avatarState.paneA;
    _clearPetTransitionStyles(outgoing);
    _clearPetTransitionStyles(incoming);
    if (outgoingPane) outgoingPane.appendChild(outgoing);
  };

  outgoing.addEventListener('transitionend', finalize, { once: true });
  // Safety net: if transitionend never fires (visibility change, GC, etc.)
  // run cleanup at ~1.5× the transition so we're not stuck composited.
  setTimeout(finalize, 380);
}

export function dispose() {
  deactivateAvatar();
}

function showAvatarToast(msg) {
  try {
    showToast(msg, 'warning');
  } catch {
    console.warn('[avatar]', msg);
  }
}
