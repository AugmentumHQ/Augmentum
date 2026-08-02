/**
 * becca-wake.js — BeccaWakeSession
 *
 * Always-on wake-word listening for the companion widget. Owns its own
 * mic + AudioWorkletNode (PCM16 16 kHz) + WebSocket to the server's
 * ``/ws/voice/wake`` detection endpoint. On a positive detection, calls
 * ``onWake(detection)`` so the host (the widget) can decide what to do
 * — typically open a voice call.
 *
 * Parallel to BeccaPttSession but with a different WS endpoint and no
 * TTS-playback path: detection is detection-only. The voice call that
 * follows a wake uses the existing call path.
 *
 * Pause/resume control lets the widget silence the listener while the
 * user is already in a call or holding PTT (so the wake doesn't fire
 * on the user's own input).
 *
 * Lifecycle:
 *   const wake = new BeccaWakeSession({
 *     avatarIds: ['wake-hey-samantha'],
 *     onWake: (d) => { ... },
 *     onStateChange: (s) => { ... },
 *   });
 *   await wake.start();    // acquire mic + open WS
 *   wake.pause();          // mute the worklet without tearing down
 *   wake.resume();         // unmute
 *   wake.dispose();        // full tear-down (on widget unmount)
 *
 * State machine emitted via onStateChange(state, detail?):
 *   'idle'        — not running
 *   'connecting'  — acquiring mic + opening WS
 *   'listening'   — streaming PCM, awaiting detections
 *   'paused'      — WS open but worklet gated off
 *   'error'       — failure; auto-resets to 'idle' after 2s
 */

import { getWsTicket } from './auth.js';

const PCM_TARGET_RATE = 16000;
const PCM_FRAME_SIZE = 512;
const ERROR_AUTO_RESET_MS = 2000;
const WS_RECONNECT_BACKOFF_MS = 2500;
// Warmup after start/resume — silence detections while the mic settles
// and any stale audio in the server-side rolling buffer flushes out.
// 2.5s comfortably covers the 1-second detection window plus a hop.
const WARMUP_MS = 2500;
// Floor between consecutive detections delivered to the host. Server-side
// refractory is 2s per source; the network round-trip plus client-side
// reaction adds enough latency that a longer client floor avoids
// double-fires on the call open path.
const DETECTION_FLOOR_MS = 5000;
// Live mic telemetry — log device + audio level periodically so the
// operator can see at a glance that the mic is wired and producing
// non-silence audio. Without this, a misconfigured mic looks identical
// to "wake just doesn't fire" — both produce no detections.
const TELEMETRY_INTERVAL_MS = 5000;

export class BeccaWakeSession {
  constructor(opts = {}) {
    this.avatarIds = Array.isArray(opts.avatarIds) && opts.avatarIds.length
      ? opts.avatarIds : ['wake-hey-samantha'];
    this.onWake = opts.onWake || (() => {});
    this.onStateChange = opts.onStateChange || (() => {});
    // Mirror of BeccaPttSession's echo-cancellation toggle. This is the
    // mic that's open while the companion PASSIVELY listens (idle/auto),
    // so it's the one that actually muffles playback via the browser's AEC
    // — the PTT mic only opens during a hold. Set at construction; the
    // widget restarts the session (dispose + recreate) to change it.
    this._echoCancelDisabled = !!opts.echoCancelDisabled;

    this._state = 'idle';
    this._ws = null;
    this._wsReady = false;
    this._audioContext = null;
    this._micStream = null;
    this._micSourceNode = null;
    this._pcmWorkletNode = null;
    this._streamingActive = false;
    this._disposed = false;
    this._reconnectTimer = null;
    // Detection-gating timestamps. ``_warmupUntil`` is the future
    // monotonic time before which incoming wake_detected frames are
    // silently dropped (set on start + resume so a fresh mic frame can't
    // immediately re-fire on stale buffer audio). ``_lastDetectionAt``
    // floors the gap between successful detections delivered to onWake
    // — belt-and-suspenders on top of the server-side refractory.
    this._warmupUntil = 0;
    this._lastDetectionAt = 0;
    // Mic telemetry — track frames sent + rolling RMS so a periodic
    // log can show "yes, mic is hearing X dBFS" without spamming.
    this._framesSent = 0;
    this._rmsSumSquared = 0;
    this._rmsSamples = 0;
    this._peak = 0;
    this._telemetryTimer = null;
  }

  /**
   * Synchronously create + resume the AudioContext from inside a user
   * gesture. iOS only unlocks audio from a genuine gesture and loses it
   * across any await, so this must run synchronously (no await). Idempotent;
   * a no-op once the context runs. See BeccaPttSession.primeAudioSync.
   */
  primeAudioSync() {
    try {
      if (this._disposed) return;
      if (!this._audioContext) {
        this._audioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (this._audioContext.state === 'suspended') {
        this._audioContext.resume().catch(() => {});
      }
    } catch (_) { /* best effort */ }
  }

  /** Acquire mic + open WS. Idempotent. Resolves when 'listening'. */
  async start() {
    if (this._disposed) return false;
    if (this._wsReady && this._streamingActive) return true;
    this._setState('connecting');
    try {
      // AudioContext + resume BEFORE getUserMedia — the iOS gesture that
      // unlocks audio is lost across the getUserMedia await (see
      // BeccaPttSession.ensureReady).
      if (!this._audioContext) {
        this._audioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (this._audioContext.state === 'suspended') {
        try { await this._audioContext.resume(); } catch (_) {}
      }
      if (!this._micStream) {
        // acquireMic centralizes the deviceId pick + per-device constraint
        // heuristic and logs the resolved label internally.
        const { acquireMic } = await import('./voice/mic-device.js');
        this._micStream = await acquireMic({
          usage: 'streaming',
          echoCancellation: this._echoCancelDisabled ? false : undefined,
        });
      }
      if (!this._micSourceNode) {
        this._micSourceNode = this._audioContext.createMediaStreamSource(this._micStream);
      }
      if (this._audioContext.state === 'suspended') {
        try { await this._audioContext.resume(); } catch (_) {}
      }
      if (!this._pcmWorkletNode) {
        await this._installPcmWorklet();
      }
      if (!this._wsReady) {
        await this._connectWs();
      }
      this._streamingActive = true;
      this._warmupUntil = Date.now() + WARMUP_MS;
      this._setState('listening');
      this._startTelemetry();
      return true;
    } catch (err) {
      console.warn('[becca-wake] start failed', err);
      this._setState('error', { message: err?.message || String(err) });
      this._scheduleErrorReset();
      return false;
    }
  }

  /** Stop streaming PCM but keep WS + mic alive. Cheap to undo. */
  pause() {
    if (this._disposed) return;
    if (!this._streamingActive) return;
    this._streamingActive = false;
    this._setState('paused');
  }

  /** Resume streaming PCM after a pause. */
  resume() {
    if (this._disposed) return;
    if (this._streamingActive) return;
    if (!this._wsReady) {
      // WS closed while paused; reconnect path will rearm.
      this.start();
      return;
    }
    this._streamingActive = true;
    this._warmupUntil = Date.now() + WARMUP_MS;
    this._setState('listening');
  }

  /** Full tear-down. Idempotent. */
  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    this._streamingActive = false;
    if (this._telemetryTimer) { clearInterval(this._telemetryTimer); this._telemetryTimer = null; }
    if (this._reconnectTimer) { clearTimeout(this._reconnectTimer); this._reconnectTimer = null; }
    if (this._pcmWorkletNode) {
      try { this._pcmWorkletNode.disconnect(); } catch (_) {}
      this._pcmWorkletNode = null;
    }
    if (this._micSourceNode) {
      try { this._micSourceNode.disconnect(); } catch (_) {}
      this._micSourceNode = null;
    }
    if (this._micStream) {
      this._micStream.getTracks().forEach(t => { try { t.stop(); } catch (_) {} });
      this._micStream = null;
    }
    if (this._audioContext) {
      try { this._audioContext.close(); } catch (_) {}
      this._audioContext = null;
    }
    if (this._ws) {
      try { this._ws.close(1000, 'wake session disposed'); } catch (_) {}
      this._ws = null;
    }
    this._wsReady = false;
    this._setState('idle');
  }

  get state() { return this._state; }

  // ── Internal ─────────────────────────────────────────────────────

  _setState(s, detail) {
    if (this._state === s) return;
    this._state = s;
    try { this.onStateChange(s, detail); } catch (_) {}
  }

  _scheduleErrorReset() {
    setTimeout(() => {
      if (!this._disposed && this._state === 'error') this._setState('idle');
    }, ERROR_AUTO_RESET_MS);
  }

  /** Begin periodic mic-level + frame-count logging. */
  _startTelemetry() {
    if (this._telemetryTimer) return;
    this._telemetryTimer = setInterval(() => {
      if (this._disposed) return;
      const frames = this._framesSent;
      const samples = this._rmsSamples;
      const sumSq = this._rmsSumSquared;
      const peak = this._peak;
      // Reset for next window so each log is fresh, not cumulative.
      this._framesSent = 0;
      this._rmsSamples = 0;
      this._rmsSumSquared = 0;
      this._peak = 0;
      if (samples === 0) {
        console.info('[becca-wake] mic-level', {
          state: this._state, frames, note: 'no audio samples this window',
        });
        return;
      }
      const rms = Math.sqrt(sumSq / samples);
      const rmsDb = rms > 0 ? 20 * Math.log10(rms) : -Infinity;
      const peakDb = peak > 0 ? 20 * Math.log10(peak) : -Infinity;
      const dbStr = (v) => Number.isFinite(v) ? `${v.toFixed(1)} dBFS` : '-inf';
      console.info('[becca-wake] mic-level', {
        state: this._state,
        frames_5s: frames,
        rms: dbStr(rmsDb),
        peak: dbStr(peakDb),
        ws: this._ws?.readyState === WebSocket.OPEN ? 'open' : 'closed',
      });
    }, TELEMETRY_INTERVAL_MS);
  }

  async _installPcmWorklet() {
    const nativeSampleRate = this._audioContext.sampleRate;
    const moduleCode = `
class WakePcmProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};
    this._targetRate = opts.targetSampleRate || ${PCM_TARGET_RATE};
    this._nativeRate = opts.nativeSampleRate || sampleRate;
    this._ratio = this._nativeRate / this._targetRate;
    this._frameSize = ${PCM_FRAME_SIZE};

    // Anti-aliasing low-pass at the NATIVE rate BEFORE decimation. Downsampling
    // to 16 kHz drops Nyquist to 8 kHz; without it, energy above 8 kHz folds
    // into the speech band as noise and degrades wake-word detection. Two
    // cascaded Butterworth biquads (~24 dB/oct), cutoff at 0.45*target.
    this._lp1 = this._makeLowpass(this._nativeRate, this._targetRate * 0.45);
    this._lp2 = this._makeLowpass(this._nativeRate, this._targetRate * 0.45);

    // Resampler state carried ACROSS render quanta — persists the continuous
    // fractional read position + the not-yet-consumed input tail so the stream
    // resamples seamlessly (old code reset phase + dropped samples each block).
    this._inBuf = new Float32Array(0);
    this._readPos = 0;
    this._outBuf = new Float32Array(0);
  }
  _makeLowpass(sr, fc) {
    // RBJ cookbook low-pass biquad, Q = 1/sqrt(2) (Butterworth, maximally flat).
    const w0 = 2 * Math.PI * (fc / sr);
    const cosw = Math.cos(w0);
    const alpha = Math.sin(w0) / (2 * Math.SQRT1_2);
    const a0 = 1 + alpha;
    return {
      b0: ((1 - cosw) / 2) / a0,
      b1: (1 - cosw) / a0,
      b2: ((1 - cosw) / 2) / a0,
      a1: (-2 * cosw) / a0,
      a2: (1 - alpha) / a0,
      x1: 0, x2: 0, y1: 0, y2: 0,
    };
  }
  _filterOne(st, x) {
    const y = st.b0 * x + st.b1 * st.x1 + st.b2 * st.x2 - st.a1 * st.y1 - st.a2 * st.y2;
    st.x2 = st.x1; st.x1 = x;
    st.y2 = st.y1; st.y1 = y;
    return y;
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) return true;
    const samples = input[0];

    // 1) Anti-alias low-pass at native rate (stateful across blocks).
    const filtered = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      filtered[i] = this._filterOne(this._lp2, this._filterOne(this._lp1, samples[i]));
    }

    // 2) Append to the persistent input buffer.
    const inBuf = new Float32Array(this._inBuf.length + filtered.length);
    inBuf.set(this._inBuf);
    inBuf.set(filtered, this._inBuf.length);
    this._inBuf = inBuf;

    // 3) Resample with a CONTINUOUS fractional cursor (phase carried across
    //    quanta), interpolating across the block seam.
    const out = [];
    let pos = this._readPos;
    while (pos + 1 < this._inBuf.length) {
      const lo = Math.floor(pos);
      const frac = pos - lo;
      out.push(this._inBuf[lo] * (1 - frac) + this._inBuf[lo + 1] * frac);
      pos += this._ratio;
    }

    // 4) Drop consumed input, keep the tail + sub-sample phase for next call.
    const consumed = Math.floor(pos);
    if (consumed > 0) this._inBuf = this._inBuf.slice(consumed);
    this._readPos = pos - consumed;

    if (out.length > 0) {
      // 5) Accumulate output and emit complete 512-sample PCM16 frames.
      const outBuf = new Float32Array(this._outBuf.length + out.length);
      outBuf.set(this._outBuf);
      outBuf.set(out, this._outBuf.length);
      this._outBuf = outBuf;

      while (this._outBuf.length >= this._frameSize) {
        const frame = this._outBuf.slice(0, this._frameSize);
        this._outBuf = this._outBuf.slice(this._frameSize);
        const pcm16 = new Int16Array(this._frameSize);
        for (let i = 0; i < this._frameSize; i++) {
          const s = Math.max(-1, Math.min(1, frame[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        this.port.postMessage(pcm16, [pcm16.buffer]);
      }
    }
    return true;
  }
}
registerProcessor('becca-wake-pcm-processor', WakePcmProcessor);
`;
    const blob = new Blob([moduleCode], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    try {
      await this._audioContext.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }
    this._pcmWorkletNode = new AudioWorkletNode(this._audioContext, 'becca-wake-pcm-processor', {
      processorOptions: { targetSampleRate: PCM_TARGET_RATE, nativeSampleRate },
    });
    this._pcmWorkletNode.port.onmessage = (e) => {
      if (!this._streamingActive) return;
      if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
      this._ws.send(e.data.buffer);
      // Accumulate RMS + peak for the telemetry tick. Frame is Int16Array;
      // normalize to [-1,1] for dB. Cheap O(N) per 32ms frame.
      const frame = e.data;
      let sumSq = 0;
      let peak = 0;
      for (let i = 0; i < frame.length; i++) {
        const s = frame[i] / 32768;
        sumSq += s * s;
        const a = Math.abs(s);
        if (a > peak) peak = a;
      }
      this._rmsSumSquared += sumSq;
      this._rmsSamples += frame.length;
      if (peak > this._peak) this._peak = peak;
      this._framesSent++;
    };
    this._micSourceNode.connect(this._pcmWorkletNode);
    this._pcmWorkletNode.connect(this._audioContext.destination);
  }

  async _connectWs() {
    const ticket = await getWsTicket();
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ids = encodeURIComponent(this.avatarIds.join(','));
    const url = `${proto}://${location.host}/ws/voice/wake`
              + `?ticket=${encodeURIComponent(ticket)}`
              + `&avatar_ids=${ids}`;
    await new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';
      let opened = false;
      ws.onopen = () => {
        opened = true;
        this._ws = ws;
        this._wsReady = true;
        this._wireWs();
        resolve();
      };
      ws.onerror = (err) => { if (!opened) reject(err); };
      setTimeout(() => { if (!opened) reject(new Error('wake ws open timeout')); }, 6000);
    });
  }

  _wireWs() {
    if (!this._ws) return;
    this._ws.onmessage = (e) => {
      // Detection WS sends JSON only — no binary frames downstream.
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === 'wake_detected') {
          const now = Date.now();
          if (now < this._warmupUntil) {
            console.info('[becca-wake] suppressed (warmup)', {
              remaining_ms: this._warmupUntil - now, ...msg,
            });
            return;
          }
          if (now - this._lastDetectionAt < DETECTION_FLOOR_MS) {
            console.info('[becca-wake] suppressed (detection floor)', {
              since_last_ms: now - this._lastDetectionAt, ...msg,
            });
            return;
          }
          this._lastDetectionAt = now;
          try { this.onWake(msg); } catch (err) {
            console.warn('[becca-wake] onWake callback failed', err);
          }
        } else if (msg.type === 'ready') {
          console.info('[becca-wake] ready', msg.avatar_ids);
        } else if (msg.type === 'error') {
          console.warn('[becca-wake] server error:', msg.message);
        }
      } catch (_) { /* ignore non-JSON */ }
    };
    this._ws.onclose = (ev) => {
      this._wsReady = false;
      this._ws = null;
      if (this._disposed) return;
      // Auto-reconnect with backoff. State stays 'listening' visually
      // because the mic + worklet are still alive — only the WS hop
      // is being re-established.
      if (this._reconnectTimer) clearTimeout(this._reconnectTimer);
      this._reconnectTimer = setTimeout(() => {
        if (!this._disposed) {
          this._connectWs().catch(err => {
            console.warn('[becca-wake] reconnect failed', err);
            this._setState('error', { message: 'connection lost' });
            this._scheduleErrorReset();
          });
        }
      }, WS_RECONNECT_BACKOFF_MS);
    };
    this._ws.onerror = (err) => {
      console.warn('[becca-wake] ws error', err);
    };
  }
}
