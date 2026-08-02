/* camera.js — shared camera primitive.
 *
 * Reusable across surfaces that need a webcam: Connect (pre-call
 * preview, in-call switching, mid-call escalation), and the future
 * vision-language consumers (Gemini-Live-style responsive video where
 * frames are pulled off the stream on a timer and sent to a VL model).
 *
 * Why this lives at ui/scripts/camera.js and not under connect/:
 * the call site is Connect today, but the load-bearing reason for a
 * shared module is the cross-modal hand-off. A VL consumer wants a
 * camera stream + frame-capture without needing to import Connect's
 * call session. Putting this in a neutral location prevents that
 * future consumer from reaching back into connect/.
 *
 * What this module owns:
 *   - Enumerate available video inputs (with the labels-only-after-
 *     permission browser quirk handled).
 *   - Persist the user's preferred camera deviceId so subsequent
 *     getUserMedia calls pick the right camera by default.
 *   - Open a MediaStream with a deviceId constraint.
 *   - Capture a single frame off a stream as a JPEG/PNG Blob for the
 *     VL hand-off. Off-screen canvas; sized + quality-controlled.
 *
 * What this module deliberately doesn't own:
 *   - Anything WebRTC-specific (peer connections, SDP). Consumers
 *     compose the stream into their pipeline.
 *   - Microphone enumeration / output device switching. The audio
 *     output picker lives in ui/scripts/connect/ui.js because output
 *     routing is per-element (setSinkId) rather than a global
 *     primitive. Mic input is rarely user-pickable in the wild — most
 *     comms apps just use the system default — so we don't surface it
 *     either. Easy to extend here later if needed; the same
 *     enumerateDevices + localStorage shape applies.
 */

const PREF_KEY = 'augmentum.camera.preferredDeviceId';

// In-flight permission probe — if listVideoDevices() is called before
// any getUserMedia, labels come back empty in most browsers (privacy
// feature). We fire a one-shot getUserMedia({video: true}) then stop
// it immediately to unlock labels for the rest of the session.
let _labelsUnlocked = false;

/**
 * Enumerate available video input devices.
 *
 * Returns an array of `{ deviceId, label, kind: 'videoinput' }`. If
 * the user has not yet granted camera permission this session, labels
 * may be empty strings — we attempt a one-shot permission probe to
 * unlock them. If the probe fails (denied, no camera, etc.) we still
 * return the device list with whatever labels the browser gives.
 */
export async function listVideoDevices({ probeForLabels = true } = {}) {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  if (probeForLabels && !_labelsUnlocked) {
    await _unlockLabels();
  }
  let devices;
  try {
    devices = await navigator.mediaDevices.enumerateDevices();
  } catch (_) {
    return [];
  }
  return devices
    .filter((d) => d.kind === 'videoinput' && d.deviceId)
    .map((d) => ({
      deviceId: d.deviceId,
      label: d.label || '',
      kind: 'videoinput',
    }));
}

async function _unlockLabels() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    for (const t of stream.getTracks()) {
      try { t.stop(); } catch (_) {}
    }
    _labelsUnlocked = true;
  } catch (_) {
    // Permission denied or no camera — labels will stay empty but
    // the device list itself still returns. Caller falls back to a
    // generic "Camera 1 / Camera 2" rendering.
  }
}

/** Preferred deviceId from localStorage. Empty string if unset. */
export function getPreferredVideoDeviceId() {
  try {
    return String(localStorage.getItem(PREF_KEY) || '');
  } catch (_) {
    return '';
  }
}

/** Persist the user's choice. Pass '' to clear. */
export function setPreferredVideoDeviceId(deviceId) {
  try {
    if (deviceId) localStorage.setItem(PREF_KEY, String(deviceId));
    else localStorage.removeItem(PREF_KEY);
  } catch (_) { /* localStorage disabled — caller still has the choice in-memory */ }
}

/**
 * Resolve the deviceId to actually request. Picks the explicit arg
 * first, then the persisted preference, then '' (let the browser
 * pick its default — usually the system default camera).
 */
export function resolveVideoDeviceId(explicit) {
  if (explicit) return String(explicit);
  return getPreferredVideoDeviceId();
}

/**
 * Open a camera stream with optional deviceId constraint. Used both
 * for the dial-picker preview (audio: false) and the production call
 * stream (audio: true with the AEC/AGC/NS constraints mirrored from
 * Connect's existing call sites).
 *
 * On deviceId mismatch (e.g. the user yanked the USB camera between
 * preference-save and dial), we fall back to a generic video request
 * rather than failing. This keeps the call placement robust.
 */
export async function openCameraStream({
  deviceId = '',
  facingMode = '',
  audio = false,
  video = true,
} = {}) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('getUserMedia not supported');
  }
  if (!video && !audio) {
    throw new Error('openCameraStream: at least one of audio/video required');
  }

  // High-fidelity defaults: 720p ideal / 1080p max, 30fps ideal. The
  // browser negotiates down if the camera can't deliver, so this is a
  // ceiling not a floor. Without these hints, getUserMedia defaults to
  // 640x480 which makes the FaceTime-style fullscreen render look soft.
  // The encoder's bitrate ceiling in quality.js is sized to keep up.
  //
  // facingMode ('user' = front/selfie, 'environment' = back/world) is the
  // mobile front↔back selector — deviceId enumeration is unreliable before
  // permission, so the companion live-camera path passes facingMode and
  // lets the OS pick. An explicit deviceId still wins (desktop picker);
  // facingMode rides along as a hint when both are given. `ideal` (not
  // `exact`) keeps single-camera desktops from OverconstrainedError.
  const _size = {
    width: { ideal: 1280, max: 1920 },
    height: { ideal: 720, max: 1080 },
    frameRate: { ideal: 30, max: 30 },
  };
  const videoConstraint = video
    ? (deviceId
        ? {
            deviceId: { ideal: deviceId },
            ...(facingMode ? { facingMode: { ideal: facingMode } } : {}),
            ..._size,
          }
        : (facingMode
            ? { facingMode: { ideal: facingMode }, ..._size }
            : { ..._size }))
    : false;
  const audioConstraint = audio
    ? {
        echoCancellation: true,
        autoGainControl: true,
        noiseSuppression: true,
      }
    : false;

  try {
    return await navigator.mediaDevices.getUserMedia({
      audio: audioConstraint,
      video: videoConstraint,
    });
  } catch (err) {
    // If the deviceId constraint was the failure cause (camera
    // unplugged between save + use), retry without it. We don't catch
    // permission errors — those are the caller's problem to surface.
    if (deviceId && video && _isDeviceConstraintError(err)) {
      // Drop ONLY the deviceId — keep the resolution hints. Retrying with
      // a bare `video: true` discards them too, and getUserMedia then
      // defaults to 640x480: a stale saved deviceId silently downgraded
      // the whole call to a soft, low-res picture with nothing to explain
      // why. The size hints are `ideal`, so they can never be the thing
      // that made this call fail.
      console.warn(
        'connect: saved camera deviceId did not resolve — falling back to '
        + 'the default camera at full quality',
      );
      return await navigator.mediaDevices.getUserMedia({
        audio: audioConstraint,
        video: {
          ...(facingMode ? { facingMode: { ideal: facingMode } } : {}),
          ..._size,
        },
      });
    }
    throw err;
  }
}

function _isDeviceConstraintError(err) {
  const name = String(err?.name || '');
  return name === 'OverconstrainedError' || name === 'NotFoundError';
}

/** Stop every track on a stream. Safe to call with null/undefined. */
export function stopStream(stream) {
  if (!stream) return;
  try {
    for (const t of stream.getTracks()) {
      try { t.stop(); } catch (_) {}
    }
  } catch (_) { /* defensive — Safari has thrown here in the wild */ }
}

/**
 * Capture a single frame off the first video track of `stream`.
 *
 * Returns a `Blob`. Sized to fit `{maxWidth, maxHeight}` while keeping
 * the source aspect ratio. `format` is the canvas-toBlob mime —
 * 'image/jpeg' for VL hand-off (small payload, lossy is fine for
 * vision models), 'image/png' for fidelity-critical paths.
 *
 * Designed for the future VL/Gemini-Live consumer: an interval timer
 * calls captureFrame() on the call's outbound camera stream every
 * N ms, hands the blob to a backend vision endpoint, the response
 * threads back through the conversational layer. This module just
 * needs to keep the capture path cheap and allocation-light, so the
 * canvas is reused across calls (one per stream).
 */
export async function captureFrame(stream, {
  maxWidth = 1280,
  maxHeight = 720,
  format = 'image/jpeg',
  quality = 0.82,
} = {}) {
  if (!stream) throw new Error('captureFrame: stream required');
  const track = stream.getVideoTracks?.()[0];
  if (!track) throw new Error('captureFrame: no video track');

  // ImageCapture is the cheapest path when available — directly
  // pulls a frame from the track without routing through an
  // intermediate <video> element. Not in Safari yet (2026-06), so
  // we fall back to the canvas path there.
  if (typeof window.ImageCapture === 'function' && track.readyState === 'live') {
    try {
      const ic = new window.ImageCapture(track);
      const bitmap = await ic.grabFrame();
      try {
        return await _bitmapToBlob(bitmap, { maxWidth, maxHeight, format, quality });
      } finally {
        try { bitmap.close?.(); } catch (_) {}
      }
    } catch (_) {
      // Fall through to the <video> path — some platforms throw here
      // intermittently even when ImageCapture is defined (e.g. the
      // track is between frames).
    }
  }

  return await _videoElementCapture(stream, { maxWidth, maxHeight, format, quality });
}

async function _bitmapToBlob(bitmap, { maxWidth, maxHeight, format, quality }) {
  const { width: srcW, height: srcH } = bitmap;
  const { width: dstW, height: dstH } = _fitInto(srcW, srcH, maxWidth, maxHeight);
  const canvas = _scratchCanvas(dstW, dstH);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(bitmap, 0, 0, dstW, dstH);
  return await _canvasToBlob(canvas, format, quality);
}

async function _videoElementCapture(stream, { maxWidth, maxHeight, format, quality }) {
  // Reuse a single offscreen <video> per stream so the camera doesn't
  // re-warm every capture. WeakMap keyed by stream releases when the
  // caller drops their reference.
  let videoEl = _videoElementForStream.get(stream);
  if (!videoEl) {
    videoEl = document.createElement('video');
    videoEl.muted = true;
    videoEl.playsInline = true;
    videoEl.srcObject = stream;
    await videoEl.play().catch(() => {});
    _videoElementForStream.set(stream, videoEl);
  }
  const srcW = videoEl.videoWidth || 1280;
  const srcH = videoEl.videoHeight || 720;
  const { width: dstW, height: dstH } = _fitInto(srcW, srcH, maxWidth, maxHeight);
  const canvas = _scratchCanvas(dstW, dstH);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoEl, 0, 0, dstW, dstH);
  return await _canvasToBlob(canvas, format, quality);
}

const _videoElementForStream = new WeakMap();
let _scratch = null;

function _scratchCanvas(w, h) {
  if (!_scratch) _scratch = document.createElement('canvas');
  if (_scratch.width !== w) _scratch.width = w;
  if (_scratch.height !== h) _scratch.height = h;
  return _scratch;
}

function _fitInto(srcW, srcH, maxW, maxH) {
  if (srcW <= maxW && srcH <= maxH) return { width: srcW, height: srcH };
  const scale = Math.min(maxW / srcW, maxH / srcH);
  return {
    width: Math.max(1, Math.round(srcW * scale)),
    height: Math.max(1, Math.round(srcH * scale)),
  };
}

function _canvasToBlob(canvas, format, quality) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error('canvas.toBlob returned null'));
    }, format, quality);
  });
}

// ── Screen-share capture ────────────────────────────────────────
//
// Wraps getDisplayMedia. The browser drives the picker UI (which
// monitor / window / tab the user wants to share) — we just request
// the stream and hand it back. Consumers swap their camera track
// for the resulting video track via RTCRtpSender.replaceTrack so the
// SDP stays the same (no offer/answer round trip needed; peer just
// sees pixels change). When the user clicks the browser's native
// "Stop sharing" chrome, the track's 'ended' event fires; consumers
// listen for that and restore the camera track.

/**
 * Open a screen-share stream. Returns a MediaStream with one video
 * track (and optionally one audio track for tab-audio capture).
 *
 * Throws on user-cancel (NotAllowedError) — caller should distinguish
 * cancel from genuine errors via `err.name`.
 */
export async function openScreenStream({
  audio = false,
  video = true,
} = {}) {
  if (!navigator.mediaDevices?.getDisplayMedia) {
    throw new Error('getDisplayMedia not supported');
  }
  return await navigator.mediaDevices.getDisplayMedia({
    audio,
    video: video ? { frameRate: { ideal: 30 } } : false,
  });
}

/**
 * True when the runtime supports screen-share. UI surfaces this to
 * hide the screen-share affordance on Safari < 14 and similar.
 */
export function canShareScreen() {
  return !!navigator.mediaDevices?.getDisplayMedia;
}

// ── Device-change subscription ─────────────────────────────────
//
// Surfaces that want to live-update their camera list (e.g. the
// in-call switcher) can subscribe here. We multiplex the browser's
// single 'devicechange' event into multiple listeners so each surface
// doesn't have to attach + manage its own.
const _deviceChangeListeners = new Set();
let _deviceChangeWired = false;

export function onDeviceChange(fn) {
  if (typeof fn !== 'function') return () => {};
  _deviceChangeListeners.add(fn);
  if (!_deviceChangeWired && navigator.mediaDevices?.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', _fanOutDeviceChange);
    _deviceChangeWired = true;
  }
  return () => _deviceChangeListeners.delete(fn);
}

function _fanOutDeviceChange() {
  for (const fn of _deviceChangeListeners) {
    try { fn(); } catch (_) {}
  }
}
