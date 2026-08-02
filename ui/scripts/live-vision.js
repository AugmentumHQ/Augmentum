/**
 * live-vision.js — LiveVisionLoop
 *
 * The client half of the companion's live-camera ("see what I'm showing
 * it") capability. Pulls frames off an open camera stream on a low-rate
 * timer, JPEG-encodes them small, and hands each to a `send` callback —
 * the voice WS adapter forwards them as `video_frame` messages, the
 * server buffers the freshest set, and the next companion turn attaches
 * them so a VL primary reads them directly (or the sibling captioner
 * describes them for a text-only primary).
 *
 * Why a loop module and not just `setInterval(captureFrame)`:
 *   - **GPU budget is the real cost** (per-frame VL prefill competes with
 *     the chat model + TTS — see docs/.../project_hardware_tiers). So the
 *     default rate is deliberately slow (~1 frame / 1.5s) and the loop
 *     refuses to pile up: a capture/encode that overruns the interval is
 *     awaited, never stacked.
 *   - **Frame selection** — a near-duplicate of the last frame carries no
 *     new signal. A cheap luma-histogram diff drops static frames so we
 *     only spend bytes (and the server only spends GPU) when the scene
 *     actually changed. `minChangeRatio = 0` disables this.
 *   - **Liveness gate** — `shouldCapture()` lets the caller pause the loop
 *     while the companion is speaking or the tab is hidden, so we don't
 *     stream frames nobody asked about.
 *
 * Usage:
 *   const loop = new LiveVisionLoop({
 *     stream,                         // an open MediaStream with a video track
 *     send: (dataUrls) => ws.sendVideoFrames(dataUrls),
 *     shouldCapture: () => state === 'recording' || state === 'armed',
 *   });
 *   loop.start();
 *   ...
 *   loop.stop();                      // idempotent; safe on unmount
 *
 * This module is transport-agnostic: it never touches a WebSocket. The
 * caller owns the camera lifecycle (open/preview/stop) and the wire.
 */

import { captureFrame } from './camera.js';

const DEFAULT_INTERVAL_MS = 1500;   // ~0.66 fps — a spoken turn rarely needs more
const DEFAULT_MAX_WIDTH = 768;      // small enough for low visual-token budgets
const DEFAULT_MAX_HEIGHT = 768;
const DEFAULT_QUALITY = 0.7;        // lossy is fine for vision models
const DEFAULT_MIN_CHANGE = 0.06;    // ~6% luma-histogram delta to count as "new"

export class LiveVisionLoop {
  constructor(opts = {}) {
    this.stream = opts.stream || null;
    this._send = typeof opts.send === 'function' ? opts.send : () => {};
    this._shouldCapture = typeof opts.shouldCapture === 'function'
      ? opts.shouldCapture
      : () => true;
    this._onError = typeof opts.onError === 'function' ? opts.onError : () => {};

    this.intervalMs = opts.intervalMs || DEFAULT_INTERVAL_MS;
    this.maxWidth = opts.maxWidth || DEFAULT_MAX_WIDTH;
    this.maxHeight = opts.maxHeight || DEFAULT_MAX_HEIGHT;
    this.quality = opts.quality != null ? opts.quality : DEFAULT_QUALITY;
    this.minChangeRatio = opts.minChangeRatio != null
      ? opts.minChangeRatio
      : DEFAULT_MIN_CHANGE;

    this._running = false;
    this._timer = null;
    this._inFlight = false;
    this._lastHist = null;       // luma histogram of the last SENT frame
    this._lastSentAt = 0;
  }

  /** Swap the camera stream without restarting the loop (e.g. device change). */
  setStream(stream) {
    this.stream = stream || null;
    this._lastHist = null;       // force the next frame through the change gate
  }

  /** Begin the capture timer. Idempotent. */
  start() {
    if (this._running) return;
    if (!this.stream) { this._onError(new Error('live-vision: no stream')); return; }
    this._running = true;
    // Kick once immediately so the very first turn after enabling can see
    // something, then settle into the interval.
    this._tick();
    this._timer = setInterval(() => this._tick(), this.intervalMs);
  }

  /** Stop the loop and release per-loop state. Idempotent. */
  stop() {
    this._running = false;
    if (this._timer) { clearInterval(this._timer); this._timer = null; }
    this._lastHist = null;
    this._inFlight = false;
  }

  get running() { return this._running; }

  async _tick() {
    if (!this._running || this._inFlight || !this.stream) return;
    if (!this._shouldCapture()) return;
    this._inFlight = true;
    try {
      const blob = await captureFrame(this.stream, {
        maxWidth: this.maxWidth,
        maxHeight: this.maxHeight,
        format: 'image/jpeg',
        quality: this.quality,
      });
      if (!blob || !this._running) return;

      // Frame-selection: skip near-duplicates of the last sent frame.
      if (this.minChangeRatio > 0) {
        const hist = await _lumaHistogram(blob);
        if (hist && this._lastHist && _histDelta(this._lastHist, hist) < this.minChangeRatio) {
          return;  // scene didn't meaningfully change — save the bytes + GPU
        }
        this._lastHist = hist || this._lastHist;
      }

      const dataUrl = await _blobToDataUrl(blob);
      if (!dataUrl || !this._running) return;
      this._lastSentAt = Date.now();
      this._send([dataUrl]);
    } catch (err) {
      this._onError(err);
    } finally {
      this._inFlight = false;
    }
  }
}

function _blobToDataUrl(blob) {
  return new Promise((resolve) => {
    const fr = new FileReader();
    fr.onload = () => resolve(typeof fr.result === 'string' ? fr.result : '');
    fr.onerror = () => resolve('');
    fr.readAsDataURL(blob);
  });
}

// --- cheap frame-change detection (luma histogram over a downscaled copy) ---

let _diffCanvas = null;
const _DIFF_SIZE = 32;       // tiny — the histogram only needs gross structure
const _HIST_BINS = 16;

async function _lumaHistogram(blob) {
  try {
    const bitmap = await createImageBitmap(blob);
    try {
      if (!_diffCanvas) {
        _diffCanvas = document.createElement('canvas');
        _diffCanvas.width = _DIFF_SIZE;
        _diffCanvas.height = _DIFF_SIZE;
      }
      const ctx = _diffCanvas.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(bitmap, 0, 0, _DIFF_SIZE, _DIFF_SIZE);
      const { data } = ctx.getImageData(0, 0, _DIFF_SIZE, _DIFF_SIZE);
      const hist = new Float32Array(_HIST_BINS);
      const px = _DIFF_SIZE * _DIFF_SIZE;
      for (let i = 0; i < data.length; i += 4) {
        // Rec. 601 luma — integer-light, good enough for a change gate.
        const luma = (data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114);
        hist[Math.min(_HIST_BINS - 1, (luma / 256 * _HIST_BINS) | 0)] += 1;
      }
      for (let b = 0; b < _HIST_BINS; b++) hist[b] /= px;  // normalize
      return hist;
    } finally {
      try { bitmap.close?.(); } catch (_) {}
    }
  } catch (_) {
    return null;  // diff unavailable → caller treats every frame as "changed"
  }
}

function _histDelta(a, b) {
  // L1 distance halved → [0,1]; 0 identical, 1 fully disjoint.
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return sum / 2;
}
