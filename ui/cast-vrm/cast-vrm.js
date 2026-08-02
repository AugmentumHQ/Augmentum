/**
 * cast-vrm.js — VRM companion surface for the cast receiver.
 *
 * Slim Three.js + three-vrm scene. Renders one bundled VRM in soft
 * idle (gentle breathing + blink + look-around). Designed to drop
 * into the receiver's iframe slot — postMessage protocol from the
 * receiver shell drives state patches (emotion, animation, look_at).
 *
 * Deliberately decoupled from the main UI's avatar.js orchestrator:
 *   - avatar.js pulls in lipsync / body atlas / IK / movement
 *     conductor / drawer / atmosphere / etc. — all coupled to the
 *     main app state and DOM. Not portable to a fullscreen surface.
 *   - This file is ~250 LOC, no app dependencies, runs anywhere
 *     /ui/lib/three + /ui/lib/three-vrm can load.
 *
 * Future phases plug into the postMessage receivers below:
 *   - lipsync feed from a voice surface (Phase H)
 *   - body atlas reactions from companion runtime (Phase G+)
 *   - VRMA animation playback for richer idle / gesture (Phase G+)
 */

const stage = document.getElementById('stage');
const fallback = document.getElementById('fallback');

let THREE = null;
let VRMModule = null;
let GLTFLoaderClass = null;

let scene = null;
let camera = null;
let renderer = null;
let currentVRM = null;
let clock = null;

let surfaceId = '';
let initialAvatarId = new URLSearchParams(location.search).get('avatar_id') || '';
let voiceSessionId = new URLSearchParams(location.search).get('voice_session_id') || '';

// Voice consumer state — populated when voice_session_id is set.
let voiceWS = null;
let voiceCtx = null;
let voiceAnalyser = null;
let voiceLipFrame = 0;

// Idle behaviour timers
let nextBlinkAt = 0;
let nextLookAt = 0;
let blinkPhase = 0;             // 0 = open, 1 = closing, 2 = closed, 3 = opening
let blinkProgress = 0;
let lookTarget = { x: 0, y: 0 };
let lookSmoothed = { x: 0, y: 0 };


function showFallback(msg, isError = false) {
  if (!fallback) return;
  fallback.textContent = msg;
  fallback.classList.toggle('err', !!isError);
  fallback.style.display = '';
}


function hideFallback() {
  if (fallback) fallback.style.display = 'none';
}


function postParent(payload) {
  try { window.parent.postMessage(payload, '*'); } catch {}
}


/* ── three.js scene setup ─────────────────────────────────────── */


async function ensureLibs() {
  if (THREE && VRMModule && GLTFLoaderClass) return;
  console.log('[cast-vrm] importing three.js');
  THREE = await import('/ui/lib/three/three.module.min.js');
  console.log('[cast-vrm] importing three-vrm');
  VRMModule = await import('/ui/lib/three-vrm/three-vrm.module.min.js');
  console.log('[cast-vrm] importing GLTFLoader');
  const gltfMod = await import('/ui/lib/three/GLTFLoader.js');
  GLTFLoaderClass = gltfMod.GLTFLoader;
  console.log('[cast-vrm] libs ready');
}


function makeScene() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05060a);

  camera = new THREE.PerspectiveCamera(28, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(0, 1.45, 2.8);
  camera.lookAt(0, 1.2, 0);

  // Performance budget — many TV boxes are ARM 32-bit with weak Mali
  // GPUs (Onn S905, Amlogic budget chips). MToon + 4K + 2x DPR will
  // OOM-kill the WebView process within seconds. Detect TV-class
  // hardware via the UA hint our APK shell sets and clamp render
  // quality aggressively. On a real GPU the high-quality branch lights
  // up the same scene without the budget knobs.
  const isLowPowerTv = /AugmentumTVReceiver/i.test(navigator.userAgent);
  renderer = new THREE.WebGLRenderer({
    antialias: !isLowPowerTv,
    alpha: false,
    powerPreference: 'low-power',
  });
  renderer.setPixelRatio(isLowPowerTv ? 1 : Math.min(window.devicePixelRatio || 1, 2));
  // Cap internal render size on TV — output is upscaled by the WebView.
  // 1280x720 is plenty for a VRM idle scene and quarters the fragment
  // load vs. native 4K.
  if (isLowPowerTv && (window.innerWidth > 1280 || window.innerHeight > 720)) {
    const targetW = 1280;
    const targetH = Math.round(targetW * (window.innerHeight / window.innerWidth));
    renderer.setSize(targetW, targetH, false);
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
  } else {
    renderer.setSize(window.innerWidth, window.innerHeight, false);
  }
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  stage.appendChild(renderer.domElement);

  // Soft, flattering lighting — feel rather than fidelity.
  const hemi = new THREE.HemisphereLight(0xffffff, 0x404060, 0.85);
  scene.add(hemi);

  const key = new THREE.DirectionalLight(0xffffff, 1.2);
  key.position.set(1.2, 2.0, 1.4);
  scene.add(key);

  const fill = new THREE.DirectionalLight(0xc0d8ff, 0.45);
  fill.position.set(-1.5, 1.4, 0.8);
  scene.add(fill);

  // Subtle ground hint (catches the eye, anchors the figure).
  const groundGeo = new THREE.CircleGeometry(2.6, 64);
  const groundMat = new THREE.MeshBasicMaterial({
    color: 0x0e1018,
    transparent: true,
    opacity: 0.85,
  });
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = 0;
  scene.add(ground);

  clock = new THREE.Clock();

  window.addEventListener('resize', () => {
    if (!camera || !renderer) return;
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight, false);
  });
}


/* ── VRM loading ──────────────────────────────────────────────── */


async function loadVRMByAvatarId(avatarId) {
  if (!avatarId) {
    console.log('[cast-vrm] resolving avatar via /api/avatar/for-session');
    try {
      const r = await fetch('/api/avatar/for-session', { credentials: 'same-origin' });
      if (r.ok) {
        const body = await r.json();
        if (body.avatar_id) {
          avatarId = body.avatar_id;
          console.log('[cast-vrm] for-session resolved', avatarId, 'source=', body.source);
        } else {
          console.warn('[cast-vrm] for-session returned no avatar', body);
        }
      } else {
        console.warn('[cast-vrm] for-session failed', r.status);
      }
    } catch (err) {
      console.warn('[cast-vrm] for-session threw', err);
    }
  }
  if (!avatarId) {
    console.log('[cast-vrm] falling back to /api/avatar/bundled');
    try {
      const r = await fetch('/api/avatar/bundled', { credentials: 'same-origin' });
      if (r.ok) {
        const body = await r.json();
        const first = (body.avatars || [])[0];
        if (first?.id) {
          avatarId = first.id;
          console.log('[cast-vrm] bundled fallback', avatarId);
        }
      }
    } catch { /* ignore */ }
  }
  if (!avatarId) {
    showFallback('No avatar available — sign in on this device, then re-cast', true);
    return null;
  }
  const url = `/api/avatar/${encodeURIComponent(avatarId)}.vrm`;
  return loadVRMFromUrl(url);
}


async function loadVRMFromUrl(url) {
  console.log('[cast-vrm] downloading VRM from', url);
  const loader = new GLTFLoaderClass();
  loader.register((parser) => new VRMModule.VRMLoaderPlugin(parser));
  let gltf;
  try {
    gltf = await loader.loadAsync(url, (xhr) => {
      if (xhr && xhr.total) {
        const pct = Math.round((xhr.loaded / xhr.total) * 100);
        if (pct % 25 === 0) console.log('[cast-vrm] VRM download', pct + '%');
      }
    });
    console.log('[cast-vrm] VRM parsed, has vrm data:', !!gltf?.userData?.vrm);
  } catch (err) {
    console.warn('[cast-vrm] VRM load failed', err);
    showFallback(`Failed to load companion: ${String(err.message || err).slice(0, 80)}`, true);
    return null;
  }
  const vrm = gltf.userData?.vrm;
  if (!vrm) {
    showFallback('Companion file has no VRM data', true);
    return null;
  }
  // Performance: combine skeletons so each frame doesn't pay the
  // many-mesh skin overhead. removeUnnecessaryJoints is deprecated
  // in newer three-vrm; combineSkeletons is the modern path.
  if (VRMModule.VRMUtils?.combineSkeletons) {
    try { VRMModule.VRMUtils.combineSkeletons(vrm.scene); } catch {}
  }

  // Low-power TV bypass: strip MToon shaders + outline meshes + the
  // VRM update tick. MToon's multi-pass toon shading torches budget
  // Mali GPUs (Onn S905 class) inside 10 seconds; spring bones add
  // CPU on top. Trade visual fidelity for survival — the companion
  // becomes flat-shaded with no outline, but it actually renders.
  if (_isLowPowerTvRuntime && vrm.scene) {
    let stripped = 0;
    let removed = 0;
    const toRemove = [];
    vrm.scene.traverse((obj) => {
      // MToon emits outline meshes as siblings of the body meshes
      // (look for *_MToonOutline naming convention or isOutline flag).
      if (obj.isMesh && (obj.name?.includes('Outline') || obj.userData?.isOutline)) {
        toRemove.push(obj);
        return;
      }
      if (obj.isMesh && obj.material) {
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
        const swapped = mats.map((m) => {
          const next = new THREE.MeshBasicMaterial({
            map: m.map || null,
            color: m.color ? m.color.clone() : new THREE.Color(0xffffff),
            transparent: !!m.transparent,
            alphaTest: m.alphaTest || 0,
            side: m.side ?? THREE.FrontSide,
          });
          stripped += 1;
          return next;
        });
        obj.material = Array.isArray(obj.material) ? swapped : swapped[0];
      }
    });
    for (const m of toRemove) {
      m.parent?.remove(m);
      removed += 1;
    }
    console.log('[cast-vrm] low-power simplification:', stripped, 'materials swapped,', removed, 'outline meshes removed');
    // Skip VRM.update() entirely below — flagged in the render loop.
    vrm.__skipUpdate = true;
  }
  if (currentVRM) {
    scene.remove(currentVRM.scene);
    currentVRM = null;
  }
  scene.add(vrm.scene);
  // Per-VRM facing correction. The desktop avatar mounts the same
  // bundled roster correctly because it probes the arm local-X axes
  // and only applies the π rotation when armAxisProfile = 'mirrored'
  // (the VRoid Studio 2.x convention). The earlier hardcoded π here
  // worked for half the bundled roster and spun the other half to
  // face away from the camera. Reuse the same probe (lightweight —
  // four bone lookups + a Vector3) so cast and desktop agree.
  try {
    const profMod = await import('/ui/scripts/avatar-vrm-profile.js');
    const prof = profMod.createAvatarCompatibilityProfile(THREE, vrm);
    if (prof?.facingCorrection === 'rotateY180') {
      vrm.scene.rotation.y = Math.PI;
    }
  } catch (err) {
    // If the probe fails (module-load error, malformed VRM), fall back
    // to the previous behavior so the common roster (VRM 1.0 + mirrored
    // arms) stays correct.
    console.warn('[cast-vrm] facing-profile probe failed, using legacy π', err);
    vrm.scene.rotation.y = Math.PI;
  }
  // Position so feet rest at y=0.
  const box = new THREE.Box3().setFromObject(vrm.scene);
  vrm.scene.position.y -= box.min.y;
  currentVRM = vrm;
  hideFallback();
  postParent({
    event: 'vrm.loaded',
    data: { avatar_id_attempted: url },
  });
  return vrm;
}


/* ── Idle behaviour ───────────────────────────────────────────── */


function tickIdle(dt, tNow) {
  if (!currentVRM) return;

  // Amplitude-driven lipsync when voice is connected. Cheap to call
  // every frame even when no voice — analyser returns zeros so the
  // mouth stays closed.
  if (voiceWS && voiceWS.readyState === WebSocket.OPEN) {
    voiceLipFrame += 1;
    // Every ~2 frames is enough — mouth doesn't need to update at
    // 60Hz, that just stresses the GPU.
    if (voiceLipFrame % 2 === 0) _drainAmplitude();
  }

  // Breathing — gentle scale on chest. VRM ExpressionManager doesn't
  // have a chest preset, so we nudge the spine bone Y rotation a hair.
  const breath = Math.sin(tNow * 1.1) * 0.012;
  const chestBone = currentVRM.humanoid?.getNormalizedBoneNode?.('chest')
    || currentVRM.humanoid?.getNormalizedBoneNode?.('spine');
  if (chestBone) {
    chestBone.rotation.x = breath;
  }

  // Blinks. Whole cycle ~150ms.
  if (tNow >= nextBlinkAt && blinkPhase === 0) {
    blinkPhase = 1;
    blinkProgress = 0;
  }
  if (blinkPhase > 0) {
    blinkProgress += dt;
    const phaseDur = 0.05;  // 50ms per phase
    const exp = currentVRM.expressionManager;
    if (exp) {
      let weight = 0;
      if (blinkPhase === 1) {
        weight = Math.min(1, blinkProgress / phaseDur);
      } else if (blinkPhase === 2) {
        weight = 1;
      } else if (blinkPhase === 3) {
        weight = Math.max(0, 1 - blinkProgress / phaseDur);
      }
      try { exp.setValue('blink', weight); } catch {}
      if (blinkProgress >= phaseDur) {
        blinkPhase = (blinkPhase + 1) % 4;
        blinkProgress = 0;
        if (blinkPhase === 0) {
          // Next blink in 3-7 seconds.
          nextBlinkAt = tNow + 3 + Math.random() * 4;
          try { exp.setValue('blink', 0); } catch {}
        }
      }
    } else {
      blinkPhase = 0;
      nextBlinkAt = tNow + 3 + Math.random() * 4;
    }
  }

  // Look-around. New target every 4-8 seconds.
  if (tNow >= nextLookAt) {
    lookTarget = {
      x: (Math.random() - 0.5) * 0.4,
      y: (Math.random() - 0.5) * 0.15,
    };
    nextLookAt = tNow + 4 + Math.random() * 4;
  }
  lookSmoothed.x += (lookTarget.x - lookSmoothed.x) * Math.min(1, dt * 1.2);
  lookSmoothed.y += (lookTarget.y - lookSmoothed.y) * Math.min(1, dt * 1.2);

  const lookAt = currentVRM.lookAt;
  if (lookAt) {
    // Pseudo target in world space.
    const x = lookSmoothed.x;
    const y = 1.45 + lookSmoothed.y;
    const tgt = new THREE.Object3D();
    tgt.position.set(x, y, 1.2);
    lookAt.target = tgt;
  }
}


/* ── Animation loop ───────────────────────────────────────────── */


// Per-frame budget on TV hardware. 30fps halves GPU work and the
// idle VRM scene is visually indistinguishable from 60fps on a
// living-room screen. Browser tabs keep 60fps.
const _isLowPowerTvRuntime = /AugmentumTVReceiver/i.test(navigator.userAgent);
const _frameIntervalMs = _isLowPowerTvRuntime ? 1000 / 30 : 0;
let _lastFrameAt = 0;

function animate(now) {
  requestAnimationFrame(animate);
  if (!scene || !renderer) return;
  if (_frameIntervalMs > 0 && now && now - _lastFrameAt < _frameIntervalMs) return;
  _lastFrameAt = now || 0;
  const dt = clock.getDelta();
  const tNow = clock.getElapsedTime();
  // Skip idle behaviour (blink/breathing/look-around) on low-power TV
  // — the per-frame matrix math is small but it adds up alongside
  // skinning. Static companion still reads as "there".
  if (!_isLowPowerTvRuntime) tickIdle(dt, tNow);
  // VRM.update() runs spring bones, lookAt, expression manager. Heavy
  // on Mali; flagged off in loadVRMFromUrl for low-power.
  if (currentVRM?.update && !currentVRM.__skipUpdate) currentVRM.update(dt);
  renderer.render(scene, camera);
}


/* ── Voice consumer (Phase H Day 1) ──────────────────────────── */


async function _ensureAudioCtx() {
  if (voiceCtx) return voiceCtx;
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) {
    console.warn('[cast-vrm] Web Audio unavailable; voice playback disabled');
    return null;
  }
  voiceCtx = new Ctx();
  voiceAnalyser = voiceCtx.createAnalyser();
  voiceAnalyser.fftSize = 256;
  voiceAnalyser.smoothingTimeConstant = 0.6;
  voiceAnalyser.connect(voiceCtx.destination);
  return voiceCtx;
}


function _amplitudeToMouth(amp) {
  // Map [0, 1] amplitude to VRM 'aa' weight. Slight bias so quiet
  // syllables still show a mouth shape; clamp at 0.85 so the avatar
  // never looks like it's screaming during loud frames.
  return Math.min(0.85, Math.max(0, amp * 1.8));
}


function _drainAmplitude() {
  if (!voiceAnalyser || !currentVRM?.expressionManager) return;
  const buf = new Uint8Array(voiceAnalyser.frequencyBinCount);
  voiceAnalyser.getByteTimeDomainData(buf);
  // Compute RMS-ish amplitude (centred around 128).
  let sum = 0;
  for (let i = 0; i < buf.length; i++) {
    const v = (buf[i] - 128) / 128;
    sum += v * v;
  }
  const rms = Math.sqrt(sum / buf.length);
  try {
    currentVRM.expressionManager.setValue('aa', _amplitudeToMouth(rms));
  } catch {}
}


async function _playAudioChunk(arrayBuffer) {
  const ctx = await _ensureAudioCtx();
  if (!ctx) return;
  try {
    const audioBuf = await ctx.decodeAudioData(arrayBuffer.slice(0));
    const src = ctx.createBufferSource();
    src.buffer = audioBuf;
    src.connect(voiceAnalyser);
    src.start();
    src.onended = () => {
      // Mouth closes when audio ends — small grace tick so the
      // amplitude drain doesn't leave the mouth half-open.
      try { currentVRM?.expressionManager?.setValue('aa', 0); } catch {}
    };
  } catch (err) {
    console.warn('[cast-vrm] audio decode failed', err);
  }
}


function _connectVoiceStream(sessionId) {
  if (!sessionId) return;
  const proto = (location.protocol === 'https:') ? 'wss:' : 'ws:';
  const url = `${proto}//${location.host}/api/voice/sessions/${encodeURIComponent(sessionId)}/stream`;
  try {
    voiceWS = new WebSocket(url);
  } catch (err) {
    console.warn('[cast-vrm] voice WS open failed', err);
    return;
  }
  voiceWS.binaryType = 'arraybuffer';

  voiceWS.addEventListener('open', () => {
    postParent({ event: 'voice.connected', data: { voice_session_id: sessionId } });
  });

  voiceWS.addEventListener('message', async (ev) => {
    if (ev.data instanceof ArrayBuffer) {
      await _playAudioChunk(ev.data);
      return;
    }
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (!msg || typeof msg !== 'object') return;
    // Surface emotion shifts emitted by the voice pipeline.
    if (msg.type === 'emotion' && msg.emotion && currentVRM?.expressionManager) {
      try {
        for (const name of ['happy', 'sad', 'angry', 'surprised', 'relaxed', 'neutral']) {
          if (name !== msg.emotion) currentVRM.expressionManager.setValue(name, 0);
        }
        currentVRM.expressionManager.setValue(msg.emotion, 1);
      } catch {}
    }
    // Bubble up other JSON for telemetry/diagnostics.
    postParent({ event: 'voice.message', data: { type: msg.type || 'unknown' } });
  });

  voiceWS.addEventListener('close', () => {
    postParent({ event: 'voice.disconnected', data: {} });
    voiceWS = null;
  });
}


/* ── postMessage protocol from receiver shell ─────────────────── */


window.addEventListener('message', async (ev) => {
  const msg = ev.data;
  if (!msg || typeof msg !== 'object') return;

  if (msg.type === 'augmentum.surface_init') {
    surfaceId = String(msg.surface_id || '');
    const state = msg.state || {};
    if (state.avatar_id && state.avatar_id !== initialAvatarId) {
      await loadVRMByAvatarId(state.avatar_id);
    }
    // Voice binding — when a voice_session_id is in the init state,
    // subscribe to the fanout stream. The surface starts playing
    // audio + driving lipsync as soon as the voice WS comes up.
    const incomingVoice = state.voice_session_id || '';
    if (incomingVoice && incomingVoice !== voiceSessionId) {
      voiceSessionId = incomingVoice;
      _connectVoiceStream(voiceSessionId);
    }
    postParent({
      event: 'vrm.ready',
      data: {
        surface_id: surfaceId,
        avatar_id: initialAvatarId || state.avatar_id || '',
        voice_session_id: voiceSessionId,
      },
    });
    return;
  }

  if (msg.type === 'augmentum.surface_state') {
    const patch = msg.patch || {};
    if (patch.avatar_id) {
      await loadVRMByAvatarId(patch.avatar_id);
    }
    if (patch.emotion && currentVRM?.expressionManager) {
      try {
        // Clear other expression presets to keep the face coherent.
        for (const name of ['happy', 'sad', 'angry', 'surprised', 'relaxed', 'neutral']) {
          if (name !== patch.emotion) currentVRM.expressionManager.setValue(name, 0);
        }
        currentVRM.expressionManager.setValue(patch.emotion, 1);
      } catch {}
    }
    if (patch.look_at && typeof patch.look_at === 'object') {
      lookTarget = {
        x: Number(patch.look_at.x) || 0,
        y: Number(patch.look_at.y) || 0,
      };
      nextLookAt = clock.getElapsedTime() + 10;  // hold longer when commanded
    }
    postParent({
      event: 'vrm.patched',
      data: { applied: Object.keys(patch) },
    });
  }
});


/* ── Boot ─────────────────────────────────────────────────────── */


(async () => {
  console.log('[cast-vrm] boot start, avatar_id param =', initialAvatarId || '(none)');
  try {
    await ensureLibs();
    console.log('[cast-vrm] building scene');
    makeScene();
    console.log('[cast-vrm] loading avatar');
    await loadVRMByAvatarId(initialAvatarId);
    console.log('[cast-vrm] starting render loop');
    animate();
  } catch (err) {
    console.error('[cast-vrm] boot failed', err);
    showFallback(`Companion failed to start: ${String(err.message || err).slice(0, 80)}`, true);
  }
})();
