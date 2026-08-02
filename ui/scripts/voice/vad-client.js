/* ui/scripts/voice/vad-client.js
 *
 * Browser-side Silero VAD wrapper, emitting events the existing voice
 * WebSocket already understands. Mirrors the server VadProcessor's
 * event vocabulary:
 *
 *   speech_start    → WS frame {type: 'vad_speech_start'}
 *   speech_end      → WS frame {type: 'vad_speech_end'}
 *   speech_discard  → WS frame {type: 'vad_discard'}   (misfire)
 *
 * The server already has handlers for those three message types from
 * the legacy client-amplitude-VAD path (see augmentum/proxy/voice_routes.py
 * around line 3895–3920). Once the resolver routes the VAD component to
 * 'client:silero-wasm', the server-side Silero is skipped and the legacy
 * client-VAD handlers take over — driven now by real Silero events
 * instead of the older AudioWorklet amplitude trigger.
 *
 * Audio frame streaming is NOT this module's responsibility — it stays
 * on the existing AudioWorklet → WS binary path in voice.js. This
 * wrapper rides alongside, consuming the same MediaStream.
 */

import { loadSileroVAD, META as LOADER_META } from '/ui/lib/silero-vad/loader.js';

const DEFAULTS = Object.freeze({
  // Match server VadProcessor defaults so client/server behave the same.
  positiveSpeechThreshold: 0.6,    // server: speech_threshold = 0.6
  negativeSpeechThreshold: 0.45,   // small hysteresis below positive
  minSpeechFrames: 3,              // server: min_start_frames = 3 (~96ms)
  preSpeechPadFrames: 10,          // server: prefix_padding_ms = 300 (~10 × 32ms)
  redemptionFrames: 25,            // server: silence_duration_ms = 800 (~25 × 32ms)
});

/**
 * VadClient — drives @ricky0123/vad-web's MicVAD against a MediaStream
 * and forwards events to a callback (typically: send a WS frame).
 *
 * Usage:
 *   const vad = new VadClient({ ws, mediaStream, onEvent: (evt) => {...} });
 *   await vad.start();         // loads Silero, opens AudioWorklet
 *   vad.pause();               // stops processing (mic still open)
 *   vad.resume();
 *   await vad.destroy();       // tears down everything
 *
 * Events emitted via onEvent:
 *   { kind: 'speech_start', timestamp: <monotonic ms> }
 *   { kind: 'speech_end',   timestamp: <monotonic ms>, audio: Float32Array }
 *   { kind: 'speech_discard', timestamp: <monotonic ms>, reason: 'too_short' }
 *   { kind: 'ready' }   // emitted once after model loads
 *   { kind: 'error', error: Error }
 */
export class VadClient {
  /**
   * @param {object} opts
   * @param {MediaStream} [opts.mediaStream]  Existing mic stream to reuse.
   *                                           When omitted, MicVAD opens its own
   *                                           via getUserMedia.
   * @param {(evt: object) => void} opts.onEvent  Receives VAD events.
   * @param {Partial<typeof DEFAULTS>} [opts.config]  Override thresholds.
   */
  constructor(opts = {}) {
    this._stream = opts.mediaStream || null;
    this._onEvent = opts.onEvent || (() => {});
    this._config = { ...DEFAULTS, ...(opts.config || {}) };
    this._mic = null;
    this._started = false;
    this._destroyed = false;
  }

  /** Returns the Silero engine id the server advertises in client_caps. */
  static get capabilityId() { return 'silero-wasm'; }

  /** Returns the asset version pins, useful for telemetry. */
  static get versions() { return LOADER_META.versions; }

  /**
   * Load Silero, attach AudioWorklet, and begin emitting events.
   * Idempotent — calling start() twice is a no-op.
   *
   * Throws on load failure; callers should fall back to the legacy
   * amplitude-VAD path in voice.js (or to server-side VAD).
   */
  async start() {
    if (this._destroyed) throw new Error('VadClient is destroyed');
    if (this._started) return;

    const { vad } = await loadSileroVAD();

    const callbacks = {
      onSpeechStart: () => {
        this._onEvent({ kind: 'speech_start', timestamp: performance.now() });
      },
      onSpeechEnd: (audio) => {
        // ``audio`` is the Float32Array of the speech segment at 16kHz —
        // we don't forward it through the event by default since the
        // existing binary audio path already shipped these frames as
        // they were spoken. Callers that want offline endpointing can
        // read ``audio`` for STT, but the call-mode wiring won't.
        this._onEvent({ kind: 'speech_end', timestamp: performance.now(), audio });
      },
      onVADMisfire: () => {
        // Speech-was-detected-but-too-short. Server already has a
        // ``vad_discard`` handler — emit the matching shape.
        this._onEvent({
          kind: 'speech_discard',
          timestamp: performance.now(),
          reason: 'too_short',
        });
      },
      onFrameProcessed: undefined,  // not used; reduce postMessage chatter
    };

    const config = {
      positiveSpeechThreshold: this._config.positiveSpeechThreshold,
      negativeSpeechThreshold: this._config.negativeSpeechThreshold,
      minSpeechFrames: this._config.minSpeechFrames,
      preSpeechPadFrames: this._config.preSpeechPadFrames,
      redemptionFrames: this._config.redemptionFrames,
      // vad-web ≥0.0.18: model + worklet + ORT wasm are all resolved
      // from base paths (no explicit URLs). Pointing both to our
      // vendored same-origin folder keeps every internal dynamic
      // import on the same origin, which is required because the
      // bundle's internal ORT uses `import('./<x>.mjs')` to lazy-load
      // its WASM provider.
      baseAssetPath: LOADER_META.assetBase,
      onnxWASMBasePath: LOADER_META.assets.ortWasmBase,
      model: LOADER_META.versions.sileroModel,
      // When the caller provided a stream, reuse it. Otherwise MicVAD
      // opens its own — which costs an extra getUserMedia prompt the
      // user already approved for the streaming path.
      stream: this._stream || undefined,
    };

    try {
      this._mic = await vad.MicVAD.new({ ...config, ...callbacks });
      this._mic.start();
      this._started = true;
      this._onEvent({ kind: 'ready' });
    } catch (err) {
      this._onEvent({ kind: 'error', error: err });
      throw err;
    }
  }

  /** Stop processing audio but keep MicVAD allocated. Use for mute. */
  pause() {
    if (this._mic && !this._destroyed) this._mic.pause();
  }

  /** Resume after pause(). */
  resume() {
    if (this._mic && !this._destroyed) this._mic.start();
  }

  /** Tear down — releases AudioWorklet + ORT session. */
  async destroy() {
    this._destroyed = true;
    if (this._mic) {
      try {
        this._mic.destroy?.();
      } catch {
        // best-effort; ignore
      }
      this._mic = null;
    }
    this._started = false;
  }

  get isRunning() { return this._started && !this._destroyed; }
}

/**
 * Convenience helper — wires VadClient events directly into a
 * WebSocket as the protocol frames the server expects.
 *
 * Returns the VadClient instance so callers can pause/resume/destroy.
 *
 *   const vad = await attachVadToWebSocket(ws, mediaStream);
 *   // ... later
 *   await vad.destroy();
 */
export async function attachVadToWebSocket(ws, mediaStream, opts = {}) {
  const send = (msg) => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
    }
  };
  const client = new VadClient({
    mediaStream,
    onEvent: (evt) => {
      switch (evt.kind) {
        case 'speech_start':
          send({ type: 'vad_speech_start' });
          break;
        case 'speech_end':
          send({ type: 'vad_speech_end' });
          break;
        case 'speech_discard':
          send({ type: 'vad_discard', reason: evt.reason || 'misfire' });
          break;
        // 'ready' and 'error' don't go on the wire — caller logs.
        default:
          break;
      }
      opts.onEvent?.(evt);
    },
    config: opts.config,
  });
  await client.start();
  return client;
}
