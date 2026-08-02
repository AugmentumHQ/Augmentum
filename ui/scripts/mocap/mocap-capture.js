// ui/scripts/mocap/mocap-capture.js
// MediaPipe initialization, webcam management, capture pipeline

import { createFilterBank } from './one-euro-filter.js';
import { solveBody, solveFace } from './pose-solver.js';

const MEDIAPIPE_VERSION = '0.10.34';

// Debug log collector — saves to localStorage, downloadable
const _logs = [];
const _rawLog = console.log.bind(console);
function mlog(...args) {
  const msg = args.map(a => { try { return typeof a === 'object' ? JSON.stringify(a) : String(a); } catch(e) { return '?'; } }).join(' ');
  _logs.push(`${new Date().toISOString()} ${msg}`);
  _rawLog('[mocap]', ...args);
  // Keep last 500 lines
  if (_logs.length > 500) _logs.shift();
  // Store for retrieval
  try { localStorage.setItem('mocap-debug-log', _logs.join('\n')); } catch(e) {}
}
// Expose for console access
window._mocapLogs = () => _logs.join('\n');
window._downloadMocapLog = () => {
  const blob = new Blob([_logs.join('\n')], {type:'text/plain'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'mocap-debug.log'; a.click();
};
const MEDIAPIPE_CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;

let poseLandmarker = null;
let faceLandmarker = null;
let videoEl = null;
let stream = null;
let running = false;
let rafId = null;

// Filter banks
let bodyFilters = null;
let faceFilters = null;

// Callbacks
let onFrame = null;
let onStatus = null;

// Filter params (tunable)
let _debugCount = 0;
let _lastPoseTs = -1;
let _lastFaceTs = -1;
const filterParams = {
  body: { minCutoff: 1.5, beta: 0.5 },
  face: { minCutoff: 1.0, beta: 0.3 },
};

async function initMediaPipe() {
  if (onStatus) onStatus('Loading MediaPipe...');

  const { FilesetResolver, PoseLandmarker, FaceLandmarker } = await import(
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`
  );

  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_CDN);

  try {
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
        delegate: 'CPU',
      },
      runningMode: 'VIDEO',
      numPoses: 1,
    });
    mlog('[mocap] PoseLandmarker created OK');
  } catch (e) {
    mlog('[mocap] ERROR: PoseLandmarker FAILED:', e);
    // Try lite model as fallback
    try {
      poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
          delegate: 'CPU',
        },
        runningMode: 'VIDEO',
        numPoses: 1,
      });
      mlog('[mocap] PoseLandmarker created OK (lite fallback)');
    } catch (e2) {
      mlog('[mocap] ERROR: PoseLandmarker lite ALSO FAILED:', e2);
    }
  }

  try {
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
        delegate: 'CPU',
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
    mlog('[mocap] FaceLandmarker created OK');
  } catch (e) {
    mlog('[mocap] ERROR: FaceLandmarker FAILED:', e);
  }

  if (onStatus) onStatus('MediaPipe ready');
}

export async function startCapture(opts) {
  videoEl = opts.video;
  onFrame = opts.onFrame;
  onStatus = opts.onStatus || null;

  if (!poseLandmarker) await initMediaPipe();

  bodyFilters = createFilterBank(33, 3, 30, filterParams.body.minCutoff, filterParams.body.beta);
  faceFilters = createFilterBank(52, 1, 30, filterParams.face.minCutoff, filterParams.face.beta);

  // List available cameras
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter(d => d.kind === 'videoinput');
  mlog('[mocap] Available cameras:', cameras.map(c => `${c.label} (${c.deviceId.slice(0,8)})`));

  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 } },
  });
  mlog('[mocap] Got stream, tracks:', stream.getVideoTracks().map(t => `${t.label} ${t.readyState}`));

  videoEl.srcObject = stream;
  await new Promise((resolve, reject) => {
    videoEl.onloadeddata = () => { mlog('[mocap] Video loadeddata event'); resolve(); };
    videoEl.onerror = (e) => { mlog('[mocap] ERROR: Video error:', e); reject(e); };
    videoEl.play().catch(e => mlog('[mocap] ERROR: Play failed:', e));
  });
  mlog('[mocap] Video playing:', videoEl.videoWidth, 'x', videoEl.videoHeight, 'readyState:', videoEl.readyState);

  // Draw video to PiP canvas (avoids WebGL context issues with video element)
  const pip = document.getElementById('mocap-pip');
  if (pip) {
    let pipCanvas = pip.querySelector('canvas');
    if (!pipCanvas) {
      pipCanvas = document.createElement('canvas');
      pipCanvas.style.cssText = 'width:100%;height:100%;transform:scaleX(-1);';
      pip.appendChild(pipCanvas);
    }
    pipCanvas.width = videoEl.videoWidth;
    pipCanvas.height = videoEl.videoHeight;
    const pipCtx = pipCanvas.getContext('2d');
    const drawPip = () => {
      if (!running) return;
      pipCtx.drawImage(videoEl, 0, 0);
      requestAnimationFrame(drawPip);
    };
    drawPip();
  }

  running = true;
  if (onStatus) onStatus('Capturing');
  processFrame();
}

function processFrame() {
  if (!running) return;
  rafId = requestAnimationFrame(processFrame);
  if (!videoEl || videoEl.readyState < 2) return;

  const nowMs = Math.round(performance.now());
  let bodyBones = {};
  let faceExpressions = {};

  // --- Pose detection ---
  if (poseLandmarker) {
    const poseTs = Math.max(nowMs, _lastPoseTs + 1);
    _lastPoseTs = poseTs;
    try {
      const poseResult = poseLandmarker.detectForVideo(videoEl, poseTs);
      if (_debugCount % 60 === 0) {
        mlog('[mocap] pose landmarks:', poseResult?.worldLandmarks?.length);
      }
      if (poseResult?.worldLandmarks?.length > 0) {
        const raw = poseResult.worldLandmarks[0];
        const flat = [];
        for (const lm of raw) flat.push(lm.x, lm.y, lm.z);
        const filtered = bodyFilters.filter(flat, nowMs / 1000);
        const smoothed = [];
        for (let i = 0; i < 33; i++) {
          smoothed.push({
            x: filtered[i * 3], y: filtered[i * 3 + 1], z: filtered[i * 3 + 2],
            visibility: raw[i].visibility,
          });
        }
        bodyBones = solveBody(smoothed);
      }
    } catch (e) {
      if (_debugCount % 60 === 0) mlog('[mocap] WARN: pose error:', e.message);
    }
  }

  // --- Face detection ---
  if (faceLandmarker) {
    const faceTs = Math.max(nowMs, _lastFaceTs + 1);
    _lastFaceTs = faceTs;
    try {
      const faceResult = faceLandmarker.detectForVideo(videoEl, faceTs);
      if (_debugCount % 60 === 0) {
        mlog('[mocap] face blendshapes:', faceResult?.faceBlendshapes?.length);
      }
      if (faceResult?.faceBlendshapes?.length > 0) {
        const shapes = {};
        for (const s of faceResult.faceBlendshapes[0].categories) shapes[s.categoryName] = s.score;
        const names = Object.keys(shapes);
        const vals = names.map(n => shapes[n]);
        const filteredVals = faceFilters.filter(vals, nowMs / 1000);
        const filteredShapes = {};
        names.forEach((n, i) => { filteredShapes[n] = filteredVals[i]; });
        faceExpressions = solveFace(filteredShapes);
      }
      // Head rotation from face transformation matrix (higher precision than body pose)
      if (faceResult?.facialTransformationMatrixes?.length > 0) {
        const matrix = faceResult.facialTransformationMatrixes[0].data;
        const m02 = matrix[8], m10 = matrix[1], m11 = matrix[5], m12 = matrix[9], m22 = matrix[10];
        bodyBones.head = [
          Math.round(Math.atan2(-m12, Math.sqrt(m02*m02 + m22*m22)) * (180/Math.PI) * 10) / 10,
          Math.round(Math.atan2(m02, m22) * (180/Math.PI) * 10) / 10,
          Math.round(Math.atan2(m10, m11) * (180/Math.PI) * 10) / 10,
        ];
      }
    } catch (e) {
      if (_debugCount % 60 === 0) mlog('[mocap] WARN: face error:', e.message);
    }
  }

  _debugCount++;

  if (onFrame) {
    onFrame({ timestamp_ms: nowMs, bones: bodyBones, blendShapes: faceExpressions });
  }
}

export function stopCapture() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  if (videoEl) {
    videoEl.srcObject = null;
  }
  bodyFilters?.reset();
  faceFilters?.reset();
  if (onStatus) onStatus('Stopped');
}

export function setFilterParams(region, minCutoff, beta) {
  filterParams[region] = { minCutoff, beta };
  if (region === 'body' && bodyFilters) bodyFilters.setParams(minCutoff, beta);
  if (region === 'face' && faceFilters) faceFilters.setParams(minCutoff, beta);
}

export function isCapturing() { return running; }
