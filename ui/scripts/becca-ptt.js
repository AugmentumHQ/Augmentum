/**
 * becca-ptt.js — BeccaPttSession
 *
 * No-overlay hold-to-talk. Owns its own WebSocket connection to /ws/voice
 * and its own mic capture via an AudioWorkletNode (same PCM16 16kHz frame
 * format the call path uses). The server's TTS response chunks are
 * accumulated into a Blob and played through the existing chat/tts.js
 * pipeline — that path emits ``augmentum:tts-playback`` with the analyser
 * the widget already binds for lipsync, so her body speaks the response
 * without a call overlay opening.
 *
 * Backend persona routing is wired: the WS query carries
 * ``persona_id=becca`` (see ``_connectWs`` below), and the server's
 * voice route hands the turn off to ``_run_becca_voice_turn`` →
 * ``BeccaVoice`` when the CompanionRuntime is ready. Falls back to the
 * legacy chat pipeline if the runtime isn't initialised yet. We use the
 * same /ws/voice endpoint the call path uses, just with a separate
 * session and input_mode=ptt.
 *
 * Design doc context:
 *   docs/superpowers/specs/2026-05-14-companion-runtime-design-v2.md (persona routing)
 *
 * Lifecycle:
 *   const session = new BeccaPttSession({ onStateChange: (s) => ... });
 *   await session.ensureReady();      // acquire mic + open WS (lazy)
 *   session.captureStart();            // pointerdown on PTT button
 *   session.captureStop();             // pointerup on PTT button
 *   session.dispose();                 // on widget unmount
 *
 * State machine emitted via onStateChange(state, detail?):
 *   'idle'        — no active session
 *   'connecting'  — acquiring mic + opening WS
 *   'armed'       — ready to capture (between turns)
 *   'recording'   — user is holding, PCM streaming
 *   'processing'  — release; awaiting transcript + LLM
 *   'speaking'    — server is sending TTS chunks
 *   'error'       — any failure; auto-resets to 'idle' after 2s
 */

import { ttsPlayBlob, ttsStopCurrent, ttsBeginExclusivePlayback } from './chat/tts.js';
import { getWsTicket } from './auth.js';
import { createUtteranceRecorder } from './voice/batch-stt.js';
import { LiveVisionLoop } from './live-vision.js';

const PCM_TARGET_RATE = 16000;
const PCM_FRAME_SIZE = 512;       // 32ms at 16kHz — matches Silero VAD frame
const ERROR_AUTO_RESET_MS = 2000;
const WS_RECONNECT_BACKOFF_MS = 1500;
// Mic-liveness watchdog. The capture graph keeps emitting frames (now full
// of digital silence) when the underlying MediaStreamTrack ends or is muted
// out from under us — the OS or another app grabbing the device, a device
// switch, or a tab-level revoke. Nothing in the streaming path notices, so
// the widget looks connected (ptt_open=True, frames flowing) while every
// sample is peak=0 forever. This interval re-verifies the track is live and
// rebuilds it if not. Matches the ~3s diagnostic cadence for fast recovery.
const MIC_WATCHDOG_INTERVAL_MS = 3000;

export class BeccaPttSession {
  constructor(opts = {}) {
    this.onStateChange = opts.onStateChange || (() => {});
    this.onTranscript = opts.onTranscript || (() => {});
    this.onLLMDelta = opts.onLLMDelta || (() => {});
    this.onTtsStart = opts.onTtsStart || (() => {});
    this.onTtsEnd = opts.onTtsEnd || (() => {});
    // Stage-manager draft mode. When set, a finished PTT utterance's
    // transcript is handed to this callback (into the compose box) instead
    // of auto-dispatching as a server turn — the user edits, then Sends via
    // sendText(). Null = legacy behaviour (speak → immediate turn).
    this.onDraftTranscript = opts.onDraftTranscript || null;
    // Server-side staging gate. When true we send input_mode:'staging' so the
    // voice route emits the STT transcript and WAITS for a stage_send instead
    // of running the turn. This is the ONLY interception point for the
    // streaming / always-listening path (auto mode), whose STT happens
    // server-side and never reaches _finishRecordingAndDispatch. The widget
    // flips it via setStaging() when the compose bar opens.
    this._staging = !!opts.onDraftTranscript;
    // Suppresses drafting the transcript ECHO the server sends back right
    // after a stage_send (the from_stage_send path re-emits the sent text) —
    // else the just-sent message would re-appear in the freshly-cleared box.
    this._sendEchoSuppress = false;
    this.sessionId = opts.sessionId || `becca-ptt-${Date.now()}`;
    this.mode = opts.mode || 'passthrough';
    // User toggle: when true the mic is acquired with echoCancellation:false
    // so the browser's AEC stops low-passing (muffling) music/media output
    // while the companion mic is open. Trade-off: the mic can then hear
    // playback, so wake-word/barge-in leans on server-side VAD. Flipped live
    // via setEchoCancelDisabled(); persisted by the widget, not here.
    this._echoCancelDisabled = !!opts.echoCancelDisabled;

    this._state = 'idle';
    this._ws = null;
    this._wsReady = false;
    this._audioContext = null;
    this._micStream = null;
    this._micSourceNode = null;
    this._pcmWorkletNode = null;
    this._ttsChunks = [];
    // Hold-to-talk recorder (manual PTT). Uses MediaRecorder → server batch
    // STT (local Moonshine), NOT the streaming PCM/VAD path — the button
    // press defines the utterance. null except while a manual hold is active.
    this._recorder = null;
    this._ttsBufferingFor = null;  // 'sentence' string when accumulating
    this._disposed = false;
    this._reconnectTimer = null;
    // Single-flight guard + periodic watchdog for mic self-healing. See
    // MIC_WATCHDOG_INTERVAL_MS — a dead/muted track must rebuild itself,
    // not silently stream zeros until the user reloads the page.
    this._healing = false;
    this._micWatchdog = null;
    // Wake-driven auto-capture state. When ``triggerWakeCapture()``
    // arms the session, server VAD's ``vad_state.speaking=false`` event
    // becomes the auto-stop trigger — no button release required.
    this._autoCaptureMode = false;
    this._autoCaptureMaxTimer = null;
    this._autoCaptureSilenceTimer = null;
    this._autoCaptureHeardSpeech = false;
    // Single-flight playback chain. Sentence-buffered TTS emits one
    // ``tts_start``/``tts_end`` pair per sentence; without a chain,
    // back-to-back ``tts_end`` events fire ``_flushTts`` concurrently
    // and ``ttsPlayBlob`` creates overlapping <audio> elements that
    // play simultaneously — the "doubled up" speech regression.
    this._ttsPlayChain = Promise.resolve();
    // Live-camera vision loop (opt-in). Null until a surface calls
    // startLiveVision() with an open camera stream; the loop sends
    // ``video_frame`` messages the server attaches to the next turn.
    this._liveVision = null;
    // Page-unload cleanup. The widget's unmount path only fires on
    // explicit DOM teardown; a page reload / tab close kills the JS
    // context without running it, so the WS stays open on the server
    // until the browser garbage-collects (minutes). Always-listening
    // remounts on the new page then spin up a fresh WS in parallel,
    // and the server sees two concurrent audio streams per user.
    // pagehide is the reliable "I'm leaving" hook on modern browsers
    // including iOS Safari; beforeunload is the older backstop.
    this._unloadHandler = () => { try { this.dispose(); } catch (_) {} };
    try { window.addEventListener('pagehide', this._unloadHandler); } catch (_) {}
    try { window.addEventListener('beforeunload', this._unloadHandler); } catch (_) {}
  }

  /**
   * Acquire mic + open WS lazily. Idempotent — safe to call before
   * every captureStart. Resolves when the session is in 'armed' state.
   */
  /**
   * Synchronously create + resume the AudioContext from inside a user
   * gesture. iOS Safari only unlocks audio from a genuine gesture, and the
   * unlock is LOST across any await — so this must run synchronously (no
   * await), typically from the first pointerdown/touch. Accepting the mic
   * permission prompt does NOT count as a gesture on iOS. Idempotent; a
   * no-op once the context is running. Without this, always-listening (which
   * auto-arms at page load with no gesture) leaves the context suspended on
   * iPad and no PCM frames ever flow — the mic looks open but nothing is heard.
   */
  primeAudioSync() {
    try {
      if (this._disposed) return;
      if (!this._audioContext) {
        this._audioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (this._audioContext.state === 'suspended') {
        // Fire-and-forget — awaiting would drop the user-gesture context.
        this._audioContext.resume().catch(() => {});
      }
    } catch (_) { /* best effort */ }
  }

  async ensureReady() {
    if (this._disposed) return false;
    // Ready only when BOTH the socket AND a LIVE mic track exist. A non-null
    // but dead/muted _micStream (the 'connected but peak=0 forever' bug) must
    // NOT short-circuit — tear the stale graph down so it's rebuilt below.
    if (this._wsReady && this._micIsLive()) return true;
    if (this._micStream && !this._micIsLive()) this._teardownAudioGraph();
    this._setState('connecting');
    try {
      // AudioContext FIRST, and resume BEFORE getUserMedia. On iOS the
      // gesture that unlocks audio is lost across the getUserMedia await, so
      // resuming after it (as we used to) left the context suspended on iPad.
      // primeAudioSync() does the real unlock from the first touch; this
      // reuses that context and is the unlock path on desktop.
      if (!this._audioContext) {
        this._audioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (this._audioContext.state === 'suspended') {
        try { await this._audioContext.resume(); } catch (_) {}
      }

      // Mic — shared acquireMic picks the user's deviceId + per-device
      // constraint heuristics.
      if (!this._micStream) {
        try {
          const { acquireMic } = await import('./voice/mic-device.js');
          this._micStream = await acquireMic({
            usage: 'streaming',
            echoCancellation: this._echoCancelDisabled ? false : undefined,
          });
        } catch (err) {
          this._micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        }
      }

      // Watch the track for loss. ``ended`` is permanent (device gone /
      // revoked) → heal immediately. ``mute`` can be a transient UA flap
      // that ``unmute`` reverses, so log it and let the watchdog confirm a
      // sustained dead state before rebuilding (avoids mute/unmute churn).
      const micTrack = this._micStream && this._micStream.getAudioTracks()[0];
      if (micTrack) {
        micTrack.onended = () => this._handleMicLost('ended');
        micTrack.onmute = () => console.warn('[becca-ptt] mic track muted');
      }

      // Source node needs both the context and the mic stream.
      if (!this._micSourceNode) {
        this._micSourceNode = this._audioContext.createMediaStreamSource(this._micStream);
      }
      // Some mobile browsers re-suspend across the getUserMedia await — resume
      // again before wiring the worklet.
      if (this._audioContext.state === 'suspended') {
        try { await this._audioContext.resume(); } catch (_) {}
      }

      // PCM worklet — registered once per context.
      if (!this._pcmWorkletNode) {
        await this._installPcmWorklet();
      }

      // WS — connect with a fresh ticket.
      if (!this._wsReady) {
        await this._connectWs();
      }

      this._startMicWatchdog();
      this._setState('armed');
      return true;
    } catch (err) {
      console.warn('[becca-ptt] ensureReady failed', err);
      this._setState('error', { message: err?.message || String(err) });
      this._scheduleErrorReset();
      return false;
    }
  }

  /**
   * Flip the echo-cancellation preference. If a mic is currently live we
   * tear the capture graph down and re-arm so the new constraint takes
   * effect immediately (getUserMedia constraints are fixed at acquire time,
   * and applyConstraints for AEC is honored unevenly across browsers, so a
   * clean re-acquire is the reliable path). If idle, the next ensureReady()
   * picks it up. No-op when the value is unchanged.
   */
  async setEchoCancelDisabled(disabled) {
    const next = !!disabled;
    if (next === this._echoCancelDisabled) return;
    this._echoCancelDisabled = next;
    if (this._disposed) return;
    // Only churn the mic if one is actually open. Recording mid-toggle is
    // rare, but tearing down under an active capture would drop the user's
    // utterance — defer to the next arm in that case.
    if (this._state === 'recording' || this._state === 'processing') return;
    if (!this._micIsLive()) return;
    try {
      this._teardownAudioGraph();
      await this.ensureReady();
    } catch (err) {
      console.warn('[becca-ptt] echo-cancel re-acquire failed', err);
    }
  }

  /**
   * User pressed PTT. Sends start_recording + opens the PCM gate.
   * If not yet armed, runs ensureReady() first.
   */
  async captureStart() {
    if (this._disposed) return;
    if (this._state === 'recording') return;
    if (this._state === 'processing' || this._state === 'speaking') {
      // A press mid-turn is the user TAKING THE FLOOR BACK. The old
      // no-op here held the mic hostage for the whole turn — a bad
      // transcript cost 20+ seconds of waiting before a retry was
      // even possible. Interrupt the server turn, cut local TTS, and
      // capture immediately.
      this._sendJson({ type: 'interrupt', played_sentences: 0 });
      try { ttsStopCurrent(); } catch (_) {}
    }
    // ensureReady() is idempotent and a cheap no-op when already armed with a
    // live mic — but it HEALS an armed-yet-dead mic, so call it on every press
    // (except mid-recording). Without this a press on a silently-dead mic
    // would build a recorder over a zero-sample stream.
    if (this._state !== 'recording') {
      const ok = await this.ensureReady();
      if (!ok || this._state === 'error') return;
    }
    // Hold-to-talk uses LOCAL BATCH STT: record the held utterance via
    // MediaRecorder and transcribe through the server's /v1/audio/
    // transcriptions endpoint (ffmpeg → Moonshine batch) on release. We do
    // NOT stream PCM or run server VAD for a manual hold — the button press
    // defines the boundary, so we skip the fragile streaming/VAD/resampler
    // path entirely (that path mangled utterances; the chat mic button,
    // which uses exactly this approach, did not). _streamingActive stays
    // false so the worklet never ships frames during a hold.
    if (!this._micStream) {
      const ok = await this.ensureReady();
      if (!ok || this._state === 'error') return;
    }
    try {
      this._recorder = createUtteranceRecorder(this._micStream);
      this._recorder.start();
    } catch (err) {
      console.warn('[becca-ptt] recorder start failed', err);
      this._recorder = null;
      this._setState('error', { message: err?.message || 'recorder failed' });
      this._scheduleErrorReset();
      return;
    }
    this._setState('recording');
  }

  /**
   * User released PTT. For a manual hold this stops the MediaRecorder,
   * transcribes via the batch endpoint, and injects the text into the
   * existing server turn (stage_send) — reply + TTS stream back over the
   * same WS. For the streaming auto-capture path (wake / always-listening)
   * it falls back to the legacy stop_recording handshake.
   */
  captureStop() {
    if (this._disposed) return;
    if (this._state !== 'recording') return;
    this._clearAutoCaptureTimers();
    this._autoCaptureMode = false;
    if (this._recorder) {
      // Hand the recorder to the dispatcher and clear the field BEFORE the
      // state transition, so the _setState force-close invariant (which
      // cancels an orphaned recorder) doesn't kill the capture we're about
      // to transcribe.
      const rec = this._recorder;
      this._recorder = null;
      this._setState('processing');
      this._finishRecordingAndDispatch(rec);
    } else {
      // Streaming/VAD auto-capture (triggerWakeCapture) — legacy handshake.
      this._streamingActive = false;
      this._sendJson({ type: 'stop_recording' });
      this._setState('processing');
    }
  }

  /**
   * Stop the hold recorder, transcribe via the local batch endpoint, and
   * hand the text to the server turn pipeline. TTS returns over the WS and
   * plays through the existing onmessage handlers.
   */
  async _finishRecordingAndDispatch(rec) {
    if (!rec) { this._setState('armed'); return; }
    let transcript = '';
    try {
      transcript = await rec.stop();
    } catch (err) {
      console.warn('[becca-ptt] batch STT failed', err);
    }
    if (this._disposed) return;
    if (transcript) {
      this._emitTranscript(transcript, true, 'final');
      if (this.onDraftTranscript) {
        // Stage-manager mode: draft the transcript into the compose box for
        // the user to edit and Send — do NOT dispatch a turn yet. Re-arm so
        // they can keep speaking to append more.
        try { this.onDraftTranscript(transcript); } catch (_) {}
        this._setState('armed');
        return;
      }
      // Inject into the existing server turn — runs the companion reply +
      // TTS with NO server-side STT. We stay in 'processing' until
      // tts_start / turn_complete arrive over the WS.
      this._sendJson({ type: 'stage_send', text: transcript });
    } else {
      // Nothing heard (empty/too-short hold or STT miss) — re-arm and let
      // the orchestrator know so it doesn't sit on the thinking pulse.
      this._setState('armed');
      document.dispatchEvent(new CustomEvent('becca-ptt:no-speech', {
        detail: { reason: 'stt_empty', auto_capture: false },
      }));
    }
  }

  /**
   * Open the turn WebSocket WITHOUT acquiring the mic — the text-input path
   * needs a turn channel but no capture graph. Idempotent; reuses a live
   * socket. Contrast ensureReady(), which also acquires + wires the mic.
   */
  async ensureWsReady() {
    if (this._disposed) return false;
    if (this._wsReady && this._ws && this._ws.readyState === WebSocket.OPEN) return true;
    this._setState('connecting');
    try {
      await this._connectWs();
      this._setState('armed');
      return true;
    } catch (err) {
      console.warn('[becca-ptt] ensureWsReady failed', err);
      this._setState('error', { message: err?.message || String(err) });
      this._scheduleErrorReset();
      return false;
    }
  }

  /**
   * Send typed (or staged-and-edited) text as a server turn — the same
   * stage_send path a finished utterance uses, minus STT. Reply + TTS
   * stream back over the WS and drive the widget through processing →
   * speaking exactly like a spoken turn. Mic is not required.
   */
  async sendText(text) {
    if (this._disposed) return false;
    const clean = (text || '').trim();
    if (!clean) return false;
    const ok = await this.ensureWsReady();
    if (!ok) return false;
    this._emitTranscript(clean, true, 'final');
    // The server re-emits this text as a 'transcript' on the stage_send path;
    // suppress drafting it back into the just-cleared box (mirror voice.js's
    // stage cooldown).
    this._sendEchoSuppress = true;
    setTimeout(() => { this._sendEchoSuppress = false; }, 1500);
    this._sendJson({ type: 'stage_send', text: clean });
    this._setState('processing');
    return true;
  }

  /**
   * Toggle the server-side staging gate live. In staging the voice route
   * emits the STT transcript and waits for stage_send rather than running the
   * turn — the ONLY interception point for the auto/streaming path (its STT is
   * server-side). Safe before the socket opens (applied on connect) or live.
   */
  setStaging(on) {
    this._staging = !!on;
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._sendJson({ type: 'config', input_mode: this._staging ? 'staging' : 'ptt' });
    }
  }

  /**
   * Wake-driven auto-capture. Called by the widget's wake-detection
   * handler instead of opening the voice-call modal. Same WS/mic as
   * normal PTT, but no button hold — server VAD's speech-end event
   * becomes the auto-release trigger so the user can just say their
   * wake phrase + question in one breath and get an inline reply.
   *
   * Safety guards:
   *  - Max capture window (12s) — caps runaway recordings.
   *  - Silence-without-speech (4s) — if VAD never sees speech-start,
   *    abort + go back to armed without burning a turn.
   *
   * Idempotent — calling while a capture is in flight is a no-op.
   */
  async triggerWakeCapture(opts = {}) {
    if (this._disposed) return false;
    if (this._state === 'recording' || this._autoCaptureMode) {
      console.info('[becca-ptt] wake-capture already in flight; ignoring');
      return false;
    }
    if (this._state !== 'armed') {
      const ok = await this.ensureReady();
      if (!ok || this._state === 'error') return false;
    }
    const maxMs = opts.maxMs || 12000;
    const silenceMs = opts.silenceMs || 4000;

    this._autoCaptureMode = true;
    this._autoCaptureHeardSpeech = false;
    // Capture provenance for the server's address gate. Default 'wake'
    // (the wake-word handler calls this with no source) — deliberate
    // addressing. The always-listening / follow-up re-arms pass
    // 'auto' / 'followup' so ambient gating still applies to them.
    const source = ['auto', 'followup'].includes(opts.source) ? opts.source : 'wake';
    this._sendJson({ type: 'start_recording', source });
    this._setState('recording');
    this._streamingActive = true;

    // Safety: cap total recording window so a stuck VAD can't keep us
    // recording forever.
    this._autoCaptureMaxTimer = setTimeout(() => {
      if (this._autoCaptureMode && this._state === 'recording') {
        console.info('[becca-ptt] wake-capture max window reached — stopping');
        this.captureStop();
      }
    }, maxMs);

    // Safety: if VAD never sees speech_start within ``silenceMs``, the
    // wake was likely a false-positive and the user didn't say anything.
    // Abort without sending stop_recording — just drop the stream.
    this._autoCaptureSilenceTimer = setTimeout(() => {
      if (this._autoCaptureMode && !this._autoCaptureHeardSpeech
          && this._state === 'recording') {
        console.info('[becca-ptt] wake-capture: no speech heard, aborting');
        this._streamingActive = false;
        this._clearAutoCaptureTimers();
        this._autoCaptureMode = false;
        this._sendJson({ type: 'stop_recording' });
        this._setState('armed');
        // Tell the orchestrator the capture timed out so it can
        // exit any follow-up window cleanly. Separate from
        // turn-complete because no turn actually happened.
        document.dispatchEvent(new CustomEvent('becca-ptt:turn-aborted', {
          detail: { reason: 'silence' },
        }));
      }
    }, silenceMs);

    console.info('[becca-ptt] wake-capture armed', { maxMs, silenceMs });
    return true;
  }

  _clearAutoCaptureTimers() {
    if (this._autoCaptureMaxTimer) {
      clearTimeout(this._autoCaptureMaxTimer);
      this._autoCaptureMaxTimer = null;
    }
    if (this._autoCaptureSilenceTimer) {
      clearTimeout(this._autoCaptureSilenceTimer);
      this._autoCaptureSilenceTimer = null;
    }
  }

  /**
   * Close out an in-flight auto-capture because the SERVER ended the turn
   * (it runs its own VAD + smart-turn and finalizes server-side — see
   * voice_routes.py _finalize_speech). We stop streaming frames and clear
   * the auto-capture flags so the next re-arm works, but we do NOT send a
   * stop_recording (the server already segmented the turn) and we leave the
   * state transition to the caller. Clearing _streamingActive before any
   * _setState also makes the 'leaving recording' force-close invariant a
   * no-op, so no spurious "force-closed" warning fires on the normal path.
   *
   * This is the heart of the seamless-turn fix: the client must NOT treat a
   * vad_state.speaking=false as "the user is done." The server emits that
   * same event while smart-turn is VETOING an endpoint (a mid-sentence
   * pause), so tearing down here cut the user off AND stranded the buffered
   * audio the server had parked → a null turn. We keep the mic open and let
   * the server's real finalization (processing / transcript / listening /
   * turn_complete) drive us, mirroring the voice-call pipeline.
   */
  _closeAutoCapture() {
    this._clearAutoCaptureTimers();
    this._autoCaptureMode = false;
    this._autoCaptureHeardSpeech = false;
    this._streamingActive = false;
  }

  /** Tear down WS + mic. Idempotent. */
  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    this._streamingActive = false;
    // Cut any in-flight spoken reply. TTS plays through chat/tts.js's own
    // <audio> element (not this session's AudioContext), so closing the WS
    // / context below does NOT stop it — without this, dismissing the widget
    // (× button → unmountBeccaPresence → dispose) leaves her still talking.
    // Also reset the play chain so a queued sentence flush can't restart it.
    try { ttsStopCurrent(); } catch (_) {}
    this._ttsPlayChain = Promise.resolve();
    this._ttsChunks = [];
    this.stopLiveVision();
    this._clearAutoCaptureTimers();
    this._autoCaptureMode = false;
    if (this._recorder) {
      try { this._recorder.cancel(); } catch (_) {}
      this._recorder = null;
    }
    if (this._unloadHandler) {
      try { window.removeEventListener('pagehide', this._unloadHandler); } catch (_) {}
      try { window.removeEventListener('beforeunload', this._unloadHandler); } catch (_) {}
      this._unloadHandler = null;
    }
    if (this._reconnectTimer) { clearTimeout(this._reconnectTimer); this._reconnectTimer = null; }
    if (this._micWatchdog) { clearInterval(this._micWatchdog); this._micWatchdog = null; }
    this._teardownAudioGraph();
    if (this._ws) {
      try { this._ws.close(1000, 'session disposed'); } catch (_) {}
      this._ws = null;
    }
    this._wsReady = false;
    this._setState('idle');
  }

  // ─── Internal ─────────────────────────────────────────────────

  _setState(s, detail) {
    if (this._state === s) return;
    // Invariant: leaving 'recording' by ANY path must tear the capture
    // down. captureStop() and the silence-abort clear these flags
    // before transitioning; every other exit (server 'processing' /
    // 'interrupted' / 'turn_complete' racing an open capture, a queued
    // TTS flush) used to leak _streamingActive=true +
    // _autoCaptureMode=true — mic frames streamed to the server
    // forever, the auto-capture max timer no-op'd (its guard needs
    // state==='recording'), and triggerWakeCapture refused every
    // re-arm ("already in flight") until page reload. Seen live
    // 2026-06-10 as the always-listening watchdog firing in a loop.
    if (this._state === 'recording' && (this._streamingActive || this._recorder)) {
      console.warn(`[becca-ptt] capture force-closed on '${s}' transition`);
      this._clearAutoCaptureTimers();
      this._autoCaptureMode = false;
      if (this._recorder) {
        // Manual hold abandoned by an out-of-band transition — drop it
        // (no stop_recording; the recorder path never sent start_recording).
        try { this._recorder.cancel(); } catch (_) {}
        this._recorder = null;
      } else {
        this._streamingActive = false;
        this._sendJson({ type: 'stop_recording' });
      }
    }
    this._state = s;
    try { this.onStateChange(s, detail); } catch (_) {}
  }

  _scheduleErrorReset() {
    setTimeout(() => {
      if (!this._disposed && this._state === 'error') {
        this._setState('idle');
      }
    }, ERROR_AUTO_RESET_MS);
  }

  /**
   * True only when the mic stream carries a LIVE, unmuted audio track. A
   * track that has ended (device unplugged, OS/another app grabbed the mic,
   * tab-level revoke) or been muted by the UA keeps feeding the capture graph
   * digital silence while the stream object stays non-null — the exact
   * 'connected but peak=0 forever' failure. Anything but live+unmuted is
   * treated as not-ready so it gets rebuilt.
   */
  _micIsLive() {
    if (!this._micStream) return false;
    const track = this._micStream.getAudioTracks()[0];
    return !!track && track.readyState === 'live' && !track.muted;
  }

  /**
   * Tear down the audio half of the session (worklet → source → stream →
   * context) WITHOUT touching the WS, unload handlers, or state. The next
   * ensureReady() rebuilds it from scratch. The context is closed too: a
   * rebuild re-runs addModule(), and registerProcessor() throws on a name
   * already registered on the same context — a fresh context sidesteps that.
   */
  _teardownAudioGraph() {
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
  }

  /** A mic track signalled permanent loss — rebuild unless mid-hold. */
  _handleMicLost(reason) {
    if (this._disposed) return;
    console.warn(`[becca-ptt] mic track ${reason} — rebuilding capture`);
    // Don't yank the graph out from under an in-flight manual hold; the
    // watchdog heals it once recording ends.
    if (this._state === 'recording') return;
    this._healMicIfNeeded();
  }

  /**
   * Rebuild the capture graph if the mic isn't live. Single-flight, and a
   * no-op while idle/disposed (don't re-prompt getUserMedia on a session the
   * user isn't using) or mid-recording (don't break an active hold). TTS
   * playback runs through its own <audio> element, so rebuilding our capture
   * context here never interrupts her speaking.
   */
  async _healMicIfNeeded() {
    if (this._disposed || this._healing) return;
    if (this._state === 'idle' || this._state === 'recording' || this._state === 'error') return;
    if (this._micIsLive()) return;
    this._healing = true;
    try {
      console.warn('[becca-ptt] mic not live — self-healing capture graph');
      this._teardownAudioGraph();
      await this.ensureReady();
    } catch (err) {
      console.warn('[becca-ptt] mic self-heal failed', err);
    } finally {
      this._healing = false;
    }
  }

  /** Periodic liveness check so a silently-dead mic recovers on its own. */
  _startMicWatchdog() {
    if (this._micWatchdog || this._disposed) return;
    this._micWatchdog = setInterval(() => {
      try { this._healMicIfNeeded(); } catch (_) {}
    }, MIC_WATCHDOG_INTERVAL_MS);
  }

  async _installPcmWorklet() {
    const nativeSampleRate = this._audioContext.sampleRate;
    const moduleCode = `
class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};
    this._targetRate = opts.targetSampleRate || ${PCM_TARGET_RATE};
    this._nativeRate = opts.nativeSampleRate || sampleRate;
    this._ratio = this._nativeRate / this._targetRate;
    this._frameSize = ${PCM_FRAME_SIZE};

    // Anti-aliasing low-pass at the NATIVE rate BEFORE decimation. Downsampling
    // to 16 kHz drops Nyquist to 8 kHz; without it, energy above 8 kHz (speech
    // sibilants s/f/sh/t live in 8-16 kHz) folds into the speech band as noise
    // and wrecks STT accuracy. Two cascaded Butterworth biquads (~24 dB/oct),
    // cutoff at 0.45*target (~7.2 kHz). Old code had NO anti-alias filter.
    this._lp1 = this._makeLowpass(this._nativeRate, this._targetRate * 0.45);
    this._lp2 = this._makeLowpass(this._nativeRate, this._targetRate * 0.45);

    // Resampler state carried ACROSS render quanta. Old code restarted its read
    // cursor at 0 every 128-sample process() call, dropping ~2 input samples
    // per block and resetting sub-sample phase at each seam (periodic
    // discontinuity). These persist the continuous read position + input tail.
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
registerProcessor('becca-ptt-pcm-processor', PcmCaptureProcessor);
`;
    const blob = new Blob([moduleCode], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    try {
      await this._audioContext.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }

    this._pcmWorkletNode = new AudioWorkletNode(this._audioContext, 'becca-ptt-pcm-processor', {
      processorOptions: { targetSampleRate: PCM_TARGET_RATE, nativeSampleRate },
    });

    this._pcmWorkletNode.port.onmessage = (e) => {
      if (!this._streamingActive) return;  // gate: only send while PTT held
      if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
      this._ws.send(e.data.buffer);
    };

    this._micSourceNode.connect(this._pcmWorkletNode);
    // Chrome optimizes away worklets with no connected output. Route through
    // the destination — the worklet emits silence.
    this._pcmWorkletNode.connect(this._audioContext.destination);
  }

  async _connectWs() {
    // Fresh ticket through the shared auth helper — same one voice.js uses.
    const ticket = await getWsTicket();

    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    // persona_id=becca routes the turn through BeccaVoice on the
    // backend (her own prompt + facets + tools), with a voice-channel
    // addendum so she keeps replies short and TTS-clean. Falls back to
    // the legacy chat pipeline if the CompanionRuntime isn't ready.
    const url = `${proto}://${location.host}/ws/voice`
              + `?ticket=${encodeURIComponent(ticket)}`
              + `&session_id=${encodeURIComponent(this.sessionId)}`
              + `&mode=${encodeURIComponent(this.mode)}`
              + `&persona_id=becca`;

    await new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';
      let opened = false;
      ws.onopen = () => {
        opened = true;
        this._ws = ws;
        this._wsReady = true;
        this._wireWs();
        // Tell the server our input mode. 'ptt' prevents server-side VAD from
        // auto-triggering on background noise; 'staging' (compose bar open)
        // makes it emit the transcript and wait for stage_send instead of
        // running the turn — the interception for the auto/streaming path.
        this._sendJson({ type: 'config', input_mode: this._staging ? 'staging' : 'ptt' });
        resolve();
      };
      ws.onerror = (err) => {
        if (!opened) reject(err);
      };
      // Safety: never hang forever.
      setTimeout(() => { if (!opened) reject(new Error('ws open timeout')); }, 6000);
    });
  }

  _wireWs() {
    if (!this._ws) return;

    this._ws.onmessage = (e) => {
      if (e.data instanceof ArrayBuffer) {
        // Binary frame = TTS audio chunk (MP3). Accumulate.
        if (!this._ttsChunks) this._ttsChunks = [];
        this._ttsChunks.push(new Uint8Array(e.data));
        return;
      }
      try {
        const msg = JSON.parse(e.data);
        this._handleServerMsg(msg);
      } catch (_) { /* ignore non-JSON text */ }
    };

    this._ws.onclose = (ev) => {
      this._wsReady = false;
      this._ws = null;
      if (this._disposed) return;
      // Auto-reconnect on unexpected close (not user-initiated dispose).
      if (this._reconnectTimer) clearTimeout(this._reconnectTimer);
      this._reconnectTimer = setTimeout(() => {
        if (!this._disposed) {
          this._connectWs().then(() => {
            // A backend restart drops only the WS — the mic graph survives
            // untouched. If the mic died independently while we were
            // disconnected, reconnecting the socket alone would resume
            // streaming SILENCE. Re-verify the mic is live too.
            this._healMicIfNeeded();
          }).catch(err => {
            console.warn('[becca-ptt] reconnect failed', err);
            this._setState('error', { message: 'connection lost' });
            this._scheduleErrorReset();
          });
        }
      }, WS_RECONNECT_BACKOFF_MS);
    };

    this._ws.onerror = (err) => {
      console.warn('[becca-ptt] ws error', err);
    };
  }

  _handleServerMsg(msg) {
    switch (msg.type) {
      case 'listening':
        // Server signal that the turn ended without TTS (empty STT,
        // unaddressed/ambient utterance, backchannel filter). Now that the
        // client no longer pre-transitions to 'processing' on vad_state, a
        // dropped turn arrives while we're still 'recording' — so we must
        // release from EITHER 'recording' or 'processing', else the capture
        // hangs (red) until the 30s max-window cap. Close the auto-capture,
        // re-arm, and tell the orchestrator the turn ended replyless so
        // always-listening re-arms now instead of waiting out its watchdog.
        if (this._state === 'recording' || this._state === 'processing') {
          const wasAuto = this._autoCaptureMode;
          if (this._autoCaptureMode) this._closeAutoCapture();
          this._setState('armed');
          // near_miss = she HEARD coherent, reply-shaped speech that landed
          // just under the addressing bar (server's call). The orchestrator
          // renders a faint, non-spoken "heard you" tell so a turn she heard
          // but chose not to answer is never a silent void.
          document.dispatchEvent(new CustomEvent('becca-ptt:turn-aborted', {
            detail: {
              reason: msg.near_miss ? 'near-miss' : 'no-reply',
              heard: !!msg.heard,
              near_miss: !!msg.near_miss,
              confidence: msg.confidence,
              auto_capture: wasAuto,
            },
          }));
        }
        break;
      case 'voice_decision':
        // The server's routing verdict for this turn (act / converse /
        // clarify / idle / drop) emitted the moment classification lands —
        // BEFORE any reply or dispatch. Purely informational at the PTT
        // layer; the presence dock renders a subtle per-goal tell and feeds
        // the opt-in decision HUD so the user can see what she decided
        // without reading logs. Fail-soft: a malformed payload just yields a
        // conservative 'drop'.
        document.dispatchEvent(new CustomEvent('becca-ptt:decision', {
          detail: {
            goal: msg.goal || 'drop',
            routerGoal: msg.router_goal || msg.goal || 'drop',
            addressed: !!msg.addressed,
            explicit: !!msg.explicit,
            nearMiss: !!msg.near_miss,
            confidence: typeof msg.confidence === 'number' ? msg.confidence : null,
            transcript: typeof msg.transcript === 'string' ? msg.transcript : '',
          },
        }));
        break;
      case 'vad_state':
        // ADVISORY ONLY — never a turn-end trigger. The server runs its own
        // VAD + smart-turn and finalizes the turn server-side; it emits
        // speaking=false BOTH on a real endpoint AND while smart-turn is
        // vetoing one (a mid-sentence pause it's waiting out). The client
        // can't tell them apart, so we must not end the capture here — doing
        // so cut the user off mid-sentence and stranded the parked audio,
        // producing a null turn. We keep streaming and let the server's real
        // finalization (processing / transcript / listening / turn_complete)
        // drive us. speaking=true still clears the no-speech abort so a slow
        // starter isn't dropped, and feeds the "heard you" UI.
        if (this._autoCaptureMode && msg.speaking === true) {
          this._autoCaptureHeardSpeech = true;
          if (this._autoCaptureSilenceTimer) {
            clearTimeout(this._autoCaptureSilenceTimer);
            this._autoCaptureSilenceTimer = null;
          }
        }
        break;
      case 'processing':
        // Server confirmed transcript + LLM in flight — this is the REAL
        // turn boundary for an auto-capture (the server's own VAD/smart-turn
        // decided the user finished). Close the capture cleanly before the
        // state flip so streaming stops and the next re-arm isn't blocked by
        // a stuck _autoCaptureMode flag.
        if (this._autoCaptureMode) this._closeAutoCapture();
        this._setState('processing');
        break;
      case 'transcript':
        if (msg.text && this.onDraftTranscript && this._staging && !this._sendEchoSuppress) {
          // Staging: the server did the STT and is WAITING for stage_send.
          // Draft the transcript into the compose box instead of a turn —
          // this is the interception for the auto/streaming path (server STT
          // never reaches _finishRecordingAndDispatch). Re-arm so always-
          // listening keeps capturing; the user speaks/edits then Sends.
          this._emitTranscript(msg.text, true, 'final');
          try { this.onDraftTranscript(msg.text); } catch (_) {}
          if (this._autoCaptureMode) this._closeAutoCapture();
          this._setState('armed');
          document.dispatchEvent(new CustomEvent('becca-ptt:turn-aborted', {
            detail: { reason: 'staged', heard: true, auto_capture: true },
          }));
        } else if (msg.text) {
          this._emitTranscript(msg.text, true, 'final');
        }
        break;
      // NB: the companion widget deliberately does NOT handle 'user_committed'
      // (which the call client, voice.js, uses to persist the user side of the
      // chat tree). The companion persists its conversation SERVER-side via the
      // becca runtime, not the client chat tree — so committing here would be
      // wrong (double-write / wrong store). Same /ws/voice protocol, different
      // persistence domain. Do not "fix" this by adding a user_committed case.
      case 'partial_transcript':
        if (msg.text) this._emitTranscript(msg.text, !!msg.is_final, 'partial');
        break;
      case 'llm_start':
        break;
      case 'llm_delta':
        if (msg.text) this._emitLLMDelta(msg.text);
        break;
      case 'tts_start':
        this._ttsChunks = [];
        this._ttsBufferingFor = msg.sentence || '';
        this._emitTtsStart(this._ttsBufferingFor);
        break;
      case 'tts_end':
        // Snapshot the current chunks so the next ``tts_start`` reset
        // can't clear them before this flush runs.
        {
          const chunks = this._ttsChunks;
          this._ttsChunks = [];
          this._ttsPlayChain = this._ttsPlayChain
            .catch(() => {})
            .then(() => this._flushTtsChunks(chunks))
            .catch(err => console.warn('[becca-ptt] tts play failed', err));
        }
        this._emitTtsEnd();
        break;
      case 'turn_complete':
        // The whole turn finished — back to armed for the next press.
        // Also broadcast a turn-complete event so the follow-up
        // orchestrator (becca-presence) can re-arm without having to
        // disambiguate the per-sentence armed transitions that fire
        // inside _flushTtsChunks during multi-sentence TTS.
        this._setState('armed');
        document.dispatchEvent(new CustomEvent('becca-ptt:turn-complete', {
          detail: { auto_capture: this._autoCaptureMode },
        }));
        break;
      case 'voice_no_speech':
        // STT heard nothing, or an explicit capture came out
        // incoherent. Re-arm immediately and surface the hint so the
        // user knows to retry instead of watching an endless thinking
        // pulse (this message used to fall through the switch
        // unhandled — 4 of 6 dropped turns in the 2026-06-10 session).
        {
          const wasAuto = this._autoCaptureMode;
          if (this._autoCaptureMode) this._closeAutoCapture();
          this._setState('armed');
          document.dispatchEvent(new CustomEvent('becca-ptt:no-speech', {
            detail: {
              reason: msg.reason || 'stt_empty',
              message: msg.message || '',
              auto_capture: wasAuto,
            },
          }));
        }
        break;
      case 'interrupted':
        this._ttsChunks = [];
        // Reset the play chain so any queued post-interrupt flushes
        // don't bleed onto the next turn. In-flight playback finishes
        // its current clip on its own; this just prevents the next
        // sentence's audio from queueing up behind it.
        this._ttsPlayChain = Promise.resolve();
        if (this._autoCaptureMode) this._closeAutoCapture();
        this._setState('armed');
        break;
      case 'error':
        console.warn('[becca-ptt] server error', msg.message);
        this._setState('error', { message: msg.message || 'server error' });
        this._scheduleErrorReset();
        break;
      case 'intent_action':
        // Server-side action registry fired — route via the shared
        // intent-action-router. The turn is already short-circuited
        // server-side; we just dispatch the surface effect.
        import('./intent-action-router.js')
          .then(m => m.dispatchIntentAction?.(msg))
          .catch(err => console.warn('[becca-ptt] intent dispatch failed', err));
        // Treat as turn-complete equivalent — the user issued a
        // valid utterance and got an action back, so follow-up
        // should re-arm just like a normal turn. The router emits
        // a separate ``conversation.close`` event for ``bye becca``
        // which forces follow-up to close.
        this._setState('armed');
        document.dispatchEvent(new CustomEvent('becca-ptt:turn-complete', {
          detail: { auto_capture: true, intent: msg?.action },
        }));
        break;
      default:
        // Unhandled message types are non-fatal.
        break;
    }
  }

  async _flushTtsChunks(chunks) {
    if (!chunks?.length) return;
    // Concatenate into a single Blob so the existing TTS pipeline
    // (chat/tts.js::ttsPlayBlob) handles playback + analyser binding.
    // ttsPlayBlob emits ``augmentum:tts-playback`` with the analyser,
    // which the widget's TTS listener already consumes for lipsync.
    const total = chunks.reduce((sum, c) => sum + c.byteLength, 0);
    const merged = new Uint8Array(total);
    let off = 0;
    for (const c of chunks) { merged.set(c, off); off += c.byteLength; }
    const blob = new Blob([merged], { type: 'audio/mpeg' });

    // Always-listening re-arms on turn_complete, but multi-sentence TTS
    // flushes are queued on _ttsPlayChain and can run AFTER the next
    // capture is already recording. Transitioning to 'speaking' here
    // would kill that capture (lipsync doesn't need PTT state — it's
    // driven by the augmentum:tts-playback event), so leave an active
    // recording alone and just play the audio.
    const captureActive = this._state === 'recording';
    if (!captureActive) this._setState('speaking');
    try {
      // Claim exclusive playback so the companion's voice supersedes any chat
      // auto-read in flight (and vice versa) — one voice at a time across
      // surfaces, not two overlapping <audio> elements.
      ttsBeginExclusivePlayback();
      await ttsPlayBlob(blob);
    } finally {
      // ``turn_complete`` will flip us to 'armed' next; if it doesn't
      // arrive (e.g. server didn't emit it after multi-sentence TTS),
      // fall back to 'armed' once playback finishes.
      if (this._state === 'speaking') this._setState('armed');
    }
  }

  _sendJson(obj) {
    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
    try { this._ws.send(JSON.stringify(obj)); } catch (_) {}
  }

  /**
   * Begin streaming low-rate camera frames to the server so the companion
   * can SEE what the user is showing it. The caller owns the camera
   * stream (and any preview UI); this only drives the capture→send loop.
   * Idempotent — swaps the stream if already running. No-op without a
   * stream. Honor the server-side ``companion_live_vision_enabled`` gate
   * at the caller: if it's off, frames are ignored server-side anyway,
   * but not starting the loop saves the client work.
   */
  startLiveVision(stream, opts = {}) {
    if (this._disposed || !stream) return;
    if (this._liveVision) { this._liveVision.setStream(stream); return; }
    this._liveVision = new LiveVisionLoop({
      stream,
      send: (frames) => this._sendVideoFrames(frames),
      // Don't burn GPU on frames the user isn't asking about: capture only
      // while connected and not mid-reply. Frames captured while 'armed'
      // (between turns) seed the next spoken turn.
      shouldCapture: () =>
        !this._disposed
        && !!this._ws && this._ws.readyState === WebSocket.OPEN
        && this._state !== 'speaking'
        && (typeof document === 'undefined' || !document.hidden),
      onError: (err) => console.debug('[becca-ptt] live-vision capture', err),
      ...opts,
    });
    this._liveVision.start();
  }

  /** Stop the camera frame loop. Idempotent. Does NOT stop the stream —
   *  the caller owns the camera lifecycle. */
  stopLiveVision() {
    if (this._liveVision) {
      try { this._liveVision.stop(); } catch (_) {}
      this._liveVision = null;
    }
  }

  get liveVisionActive() { return !!(this._liveVision && this._liveVision.running); }

  _sendVideoFrames(frames) {
    if (!Array.isArray(frames) || !frames.length) return;
    this._sendJson({ type: 'video_frame', frames });
  }

  _emitTranscript(text, final, source) {
    try { this.onTranscript(text, { final: !!final, source }); } catch (err) {
      console.warn('[becca-ptt] transcript hook failed', err);
    }
  }

  _emitLLMDelta(text) {
    try { this.onLLMDelta(text); } catch (err) {
      console.warn('[becca-ptt] llm hook failed', err);
    }
  }

  _emitTtsStart(sentence) {
    try { this.onTtsStart(sentence); } catch (err) {
      console.warn('[becca-ptt] tts-start hook failed', err);
    }
  }

  _emitTtsEnd() {
    try { this.onTtsEnd(); } catch (err) {
      console.warn('[becca-ptt] tts-end hook failed', err);
    }
  }
}
