/* connect/dialer.js — Caller-side Connect call state machine.
 *
 *   placeCall({ peerDid }) → CallSession
 *
 * The session walks through a narrow set of named states and emits
 * 'state-change' / 'remote-stream' / 'error' / 'ended' so the UI
 * overlay can render without coupling to wire verbs.
 *
 * State flow (happy path):
 *
 *   idle → connecting   (WS handshake + mic acquisition)
 *        → ringing      (INVITE routed, awaiting receiver decision)
 *        → negotiating  (EVENT_ACCEPT received; SDP offer/answer)
 *        → connected    (ICE connected, audio flowing)
 *        → ended        (hangup local/remote, or error)
 *
 * Why not a single onAcceptCreateOffer? Two reasons:
 *   1) States are how the UI labels itself — "Ringing…" vs
 *      "Connected" — and a single boolean would be lossy.
 *   2) Receiver-side flow (a future module) walks the same states in
 *      reverse order; a shared shape keeps the two halves coherent.
 *
 * Trickle batching: ICE candidates are pushed to a queue and flushed
 * either on a 100ms timer or when the queue hits 8 entries, whichever
 * comes first. Matches Matrix MSC2746's recommendation and avoids
 * the chatter that single-per-message would create.
 *
 * ── WebRTC jargon glossary (standard terms, not ours) ───────────────
 *   ICE  — Interactive Connectivity Establishment. The NAT-traversal
 *          handshake that finds a network path between two peers behind
 *          routers/firewalls. NOT an Augmentum name — it's the browser
 *          API (RTCIceCandidate, pc.iceConnectionState, …), so we keep it.
 *   candidate — one possible address a peer can be reached at (your LAN
 *          IP, your public IP, or a relay address). Both sides trade
 *          their candidates (the 'candidates' WS verbs) and probe pairs
 *          until one connects.
 *   STUN — the server that tells you your own public IP (for the direct
 *          path). TURN — the relay server used as a fallback when a
 *          direct path is blocked. Both arrive in `welcome.turn` /
 *          `iceServers`.
 *   SDP  — Session Description Protocol. The offer/answer blob that says
 *          "here are my codecs + media tracks"; exchanged once per call
 *          (and again on mid-call renegotiation, e.g. adding video).
 */

import {
  ensureConnected,
  getConnectState,
  getWelcome,
  on as onConnectEvent,
  onStateChange as onConnectStateChange,
  send as sendEnvelope,
  sendAndAwaitRouted,
} from './client.js';
import {
  openCameraStream,
  openScreenStream,
  resolveVideoDeviceId,
} from '../camera.js';
import { applyVideoQualityProfile, applyAudioQualityProfile } from './quality.js';
import { startRingback, stopRingback } from './ringback.js';

const ICE_FLUSH_MS = 100;
const ICE_FLUSH_BATCH = 8;
// Maximum time we'll sit in the RECONNECTING state before giving up
// and tearing the call down. The peer's connection may simply have
// gone away; this cap prevents an indefinite limbo state where the
// user thinks the call is "still trying".
const RECONNECT_TIMEOUT_MS = 20_000;
// Maximum time the CALLER will ring before giving up. The server arms a
// missed-call timer at DEFAULT_INVITE_LIFETIME_MS (60s) but doesn't push a
// teardown to the caller, so without this the dialer could sit in RINGING
// forever — wedging _activeSession and blocking the next call attempt. A hair
// past the server's window so the server marks "missed" first.
const RING_TIMEOUT_MS = 65_000;
// Maximum time we'll sit in NEGOTIATING (the callee accepted; we're doing the
// SDP offer/answer + ICE). If the connection never reaches CONNECTED — a peer
// whose ICE quietly stalls in 'checking' without ever firing 'failed', or a
// dropped ANSWER — this bounds the "Connecting…" spinner so it can't hang
// indefinitely. Generous (TURN relay allocation + ICE on a slow link can take
// a while) but finite, with a clean failure + escape instead of limbo.
const NEGOTIATE_TIMEOUT_MS = 30_000;

export const CALL_STATES = Object.freeze({
  IDLE: 'idle',
  CONNECTING: 'connecting',
  RINGING: 'ringing',
  NEGOTIATING: 'negotiating',
  CONNECTED: 'connected',
  // Mid-call transient — either ICE went 'disconnected' (network
  // blip / NAT rebinding) or the signaling WS dropped while the
  // media plane is still up. The overlay surfaces this so the user
  // knows the call hasn't died, just hiccupped. Returns to CONNECTED
  // when the underlying issue resolves, or transitions to ENDED if
  // the reconnect timer expires.
  RECONNECTING: 'reconnecting',
  ENDED: 'ended',
});

/**
 * Place a 1:1 audio call to a peer DID.
 *
 *   const session = placeCall({ peerDid: 'alice@home.alice.dev' });
 *   session.on('state-change', (s) => ...);
 *   session.on('remote-stream', (stream) => audioEl.srcObject = stream);
 *   await session.start();        // resolves once ringing OR rejects on error
 *   ...
 *   await session.hangup();
 */
export function placeCall({ peerDid, withVideo = false, videoDeviceId = '' }) {
  if (!peerDid || typeof peerDid !== 'string') {
    throw new Error('placeCall: peerDid required');
  }

  let modalities = withVideo ? 'audio,video' : 'audio';
  // The deviceId the caller asked for (explicit > persisted preference >
  // browser default). Resolved at call-start time; later switchVideoDevice
  // calls update it for subsequent re-acquires (e.g. video escalation
  // after starting audio-only).
  let activeVideoDeviceId = resolveVideoDeviceId(videoDeviceId);
  // Screen-share state. When sharing, we swap the camera track on the
  // existing video sender with a getDisplayMedia track via replaceTrack,
  // and stash the original camera track so stopScreenShare can put it
  // back. No SDP renegotiation — peer keeps decoding the same MID.
  let screenStream = null;
  let cameraTrackBeforeShare = null;
  let screenTrackEndedListener = null;
  let state = CALL_STATES.IDLE;
  let pc = null;
  let localStream = null;
  let remoteStream = null;
  let callId = null;
  let endReason = null;
  let lastError = null;
  const listeners = new Map(); // event → Set<fn>
  const unsubs = [];           // WS event unsubscribers
  const iceQueue = [];
  let iceFlushTimer = null;
  let ringTimer = null;        // caller-side "no answer" giveup timer
  let negotiateTimer = null;   // bounds the NEGOTIATING (SDP/ICE) phase
  let renegotiating = false;   // guard for overlapping renegotiations
  let answered = false;        // latch: first ANSWER wins (multi-device callee)
  let videoSender = null;      // RTCRtpSender for the active video track, if any
  // Buffer for remote ICE candidates that arrive before the ANSWER
  // has been processed (setRemoteDescription is async — the peer
  // starts trickling candidates immediately after setLocalDescription
  // on their side and can beat our setRemoteDescription on the wire).
  // Drained by _handleAnswer + _handleNegotiate.
  const pendingRemoteCandidates = [];
  // Reconnect bookkeeping. `reconnectTimer` is the deadline after
  // which we give up and fail. `stateBeforeReconnect` snapshots what
  // we were in (CONNECTED, usually) so we can restore on recovery.
  let reconnectTimer = null;
  let stateBeforeReconnect = null;
  // Quality-poll bookkeeping. Driven from setState transitions so
  // we sample pc.getStats() only while connected; emits a
  // 'quality-change' event the UI subscribes to for the live pill.
  let qualityTimer = null;
  let lastQualityBucket = '';

  // ── Event emitter ───────────────────────────────────────────
  function on(event, fn) {
    if (typeof fn !== 'function') return () => {};
    let set = listeners.get(event);
    if (!set) { set = new Set(); listeners.set(event, set); }
    set.add(fn);
    return () => set.delete(fn);
  }
  function emit(event, payload) {
    const set = listeners.get(event);
    if (!set) return;
    for (const fn of set) {
      try { fn(payload); } catch (err) { console.warn(`call listener (${event}) failed`, err); }
    }
  }
  function setState(next) {
    if (state === next) return;
    state = next;
    emit('state-change', next);
    // Ringback lifecycle: play while the callee is being rung
    // (state === RINGING), stop the moment we transition to anything
    // else. NEGOTIATING means they accepted and we're doing SDP;
    // ENDED means they declined, timed out, or we cancelled.
    if (next === CALL_STATES.RINGING) {
      startRingback();
      // Bound the ring so an unanswered call-back can't wedge in RINGING.
      if (ringTimer) clearTimeout(ringTimer);
      ringTimer = setTimeout(() => {
        if (state === CALL_STATES.RINGING) _fail('no_answer', new Error('ring timeout'));
      }, RING_TIMEOUT_MS);
    } else {
      stopRingback();
      if (ringTimer) { clearTimeout(ringTimer); ringTimer = null; }
    }
    // Bound the SDP/ICE negotiation so "Connecting…" can't hang forever if
    // ICE stalls without firing 'failed'. Armed on entering NEGOTIATING,
    // cleared the moment we leave it (→ CONNECTED, or a terminal state).
    if (next === CALL_STATES.NEGOTIATING) {
      if (negotiateTimer) clearTimeout(negotiateTimer);
      negotiateTimer = setTimeout(() => {
        if (state === CALL_STATES.NEGOTIATING) {
          _fail('negotiate_timeout', new Error('negotiation timeout'));
        }
      }, NEGOTIATE_TIMEOUT_MS);
    } else if (negotiateTimer) {
      clearTimeout(negotiateTimer);
      negotiateTimer = null;
    }
    // Quality poll runs only when actively connected. We keep
    // sampling through RECONNECTING (the UI tints the pill amber
    // while the call self-heals) and stop on terminal states.
    if (next === CALL_STATES.CONNECTED) {
      _startQualityPoll();
    } else if (next === CALL_STATES.ENDED) {
      _stopQualityPoll();
    }
    if (next === CALL_STATES.ENDED) emit('ended', { reason: endReason, error: lastError });
  }

  function _startQualityPoll() {
    if (qualityTimer) return;
    // Fire once immediately so the UI has a bucket to render.
    _sampleAndEmitQuality();
    qualityTimer = setInterval(_sampleAndEmitQuality, 2000);
  }

  function _stopQualityPoll() {
    if (qualityTimer) { clearInterval(qualityTimer); qualityTimer = null; }
    lastQualityBucket = '';
  }

  async function _sampleAndEmitQuality() {
    if (!pc || state === CALL_STATES.ENDED) return;
    let rtt = null;
    let lossFrac = null;
    try {
      const stats = await pc.getStats();
      for (const r of stats.values()) {
        if (r.type === 'candidate-pair' && r.nominated && r.state === 'succeeded') {
          if (typeof r.currentRoundTripTime === 'number') {
            // Pick the highest RTT across pairs — represents the
            // worst link the media currently uses.
            rtt = Math.max(rtt ?? 0, r.currentRoundTripTime);
          }
        }
        if (r.type === 'remote-inbound-rtp' && r.kind === 'audio') {
          if (typeof r.fractionLost === 'number') lossFrac = r.fractionLost;
        }
      }
    } catch (_) { /* getStats can throw on torn-down pc */ }
    const bucket = _qualityBucket({ rtt, lossFrac });
    const rttMs = rtt != null ? Math.round(rtt * 1000) : null;
    const lossPct = lossFrac != null ? Math.round(lossFrac * 100) : null;
    // Only emit on change to avoid re-render churn on the overlay.
    if (bucket !== lastQualityBucket) {
      lastQualityBucket = bucket;
      // Adapt the encoder before emitting — so any consumer that
      // reads encoder state in the emit handler sees a consistent
      // view. Fire-and-forget; the helper logs its own failures.
      if (videoSender) {
        applyVideoQualityProfile(videoSender, bucket).catch(() => {});
      }
      // Audio adapts on EVERY call, video-or-not — an audio-only call has
      // no videoSender, and that branch used to mean the encoder was left
      // entirely at browser defaults for the whole call.
      applyAudioQualityProfile(pc, bucket).catch(() => {});
      emit('quality-change', { bucket, rttMs, lossPct });
    }
  }

  function _qualityBucket({ rtt, lossFrac }) {
    // No samples yet — surface 'connecting' so the pill shows a
    // neutral state rather than (falsely) "excellent".
    if (rtt == null && lossFrac == null) return 'measuring';
    const rttMs = (rtt ?? 0) * 1000;
    const loss = lossFrac ?? 0;
    // Thresholds aligned with WebRTC conventional wisdom: under
    // 150ms RTT + sub-1% loss is "good for voice", 400ms is the
    // upper bound of conversational comfort.
    if (rttMs < 100 && loss < 0.01) return 'excellent';
    if (rttMs < 200 && loss < 0.03) return 'good';
    if (rttMs < 400 && loss < 0.08) return 'weak';
    return 'poor';
  }

  // ── Wire listeners (filter by callId so cross-call traffic is ignored) ──
  function wireWsListeners() {
    unsubs.push(onConnectEvent('accept', ({ data, peer }) => {
      if (!_matchesCall(data) || peer !== peerDid) return;
      _handleAccept().catch((err) => _fail('accept_failed', err));
    }));
    unsubs.push(onConnectEvent('decline', ({ data, peer }) => {
      if (!_matchesCall(data) || peer !== peerDid) return;
      endReason = 'declined';
      _tearDown();
    }));
    unsubs.push(onConnectEvent('answer', ({ data, peer }) => {
      if (!_matchesCall(data) || peer !== peerDid) return;
      _handleAnswer(data).catch((err) => _fail('answer_failed', err));
    }));
    unsubs.push(onConnectEvent('candidates', ({ data, peer }) => {
      if (!_matchesCall(data) || peer !== peerDid) return;
      _handleRemoteCandidates(data).catch((err) => _fail('candidates_failed', err));
    }));
    unsubs.push(onConnectEvent('negotiate', ({ data, peer }) => {
      if (!_matchesCall(data) || peer !== peerDid) return;
      _handleNegotiate(data).catch((err) => _fail('negotiate_failed', err));
    }));
    unsubs.push(onConnectEvent('hangup', ({ data, peer }) => {
      if (!_matchesCall(data) || peer !== peerDid) return;
      endReason = 'remote_hangup';
      _tearDown();
    }));
    unsubs.push(onConnectEvent('mute_state', ({ data, peer }) => {
      if (!_matchesCall(data) || peer !== peerDid) return;
      emit('peer-mute', { muted: !!data.muted });
    }));
    unsubs.push(onConnectEvent('video_state', ({ data, peer }) => {
      if (!_matchesCall(data) || peer !== peerDid) return;
      emit('peer-video', { videoEnabled: !!data.video_enabled });
    }));
    unsubs.push(onConnectEvent('__closed', () => {
      // Signaling socket died mid-call. WebRTC can survive a
      // signaling drop *after* connect (media flows direct over
      // ICE/UDP), so for an already-CONNECTED call enter
      // RECONNECTING and let the WS auto-reconnect loop restore us.
      // Pre-connected calls (still negotiating) are dead — fail.
      if (state === CALL_STATES.CONNECTED) {
        _enterReconnect('signaling_dropped');
      } else if (state === CALL_STATES.CONNECTING
              || state === CALL_STATES.RINGING
              || state === CALL_STATES.NEGOTIATING) {
        _fail('signaling_lost', new Error('signaling socket closed'));
      }
    }));
    // Restore from a WS-triggered reconnect once the socket comes
    // back up. ICE-triggered reconnects restore via
    // oniceconnectionstatechange; the WS path needs its own hook.
    unsubs.push(onConnectStateChange((wsState) => {
      if (wsState !== 'open') return;
      if (state !== CALL_STATES.RECONNECTING) return;
      // Only restore if ICE itself is still healthy; otherwise stay
      // RECONNECTING and let the ICE handler decide.
      const ice = pc && pc.iceConnectionState;
      if (ice === 'connected' || ice === 'completed') {
        _restoreFromReconnect();
      }
    }));
  }

  function _matchesCall(data) {
    if (!callId) return false;
    return data && data.call_id === callId;
  }

  // ── PeerConnection setup ────────────────────────────────────
  function _buildPeerConnection(iceServers) {
    const pcConfig = { iceServers: iceServers || [] };
    const conn = new RTCPeerConnection(pcConfig);

    conn.ontrack = (evt) => {
      // Audio-only Phase 1. ev.streams[0] is the canonical stream the
      // remote side passed to addTrack; if absent (older clients),
      // synthesize from the track itself so the audio el still works.
      const stream = (evt.streams && evt.streams[0]) || new MediaStream([evt.track]);
      remoteStream = stream;
      emit('remote-stream', stream);
    };

    conn.onicecandidate = (evt) => {
      if (!evt.candidate) {
        // End-of-gathering — send the sentinel empty batch so the
        // peer knows trickle is done (mirrors MSC2746).
        _flushIceQueue(/* sentinel */ true);
        return;
      }
      iceQueue.push({
        candidate: evt.candidate.candidate,
        sdpMid: evt.candidate.sdpMid,
        sdpMLineIndex: evt.candidate.sdpMLineIndex,
      });
      if (iceQueue.length >= ICE_FLUSH_BATCH) {
        _flushIceQueue(false);
      } else if (!iceFlushTimer) {
        iceFlushTimer = setTimeout(() => {
          iceFlushTimer = null;
          _flushIceQueue(false);
        }, ICE_FLUSH_MS);
      }
    };

    conn.oniceconnectionstatechange = () => {
      const ice = conn.iceConnectionState;
      if (ice === 'connected' || ice === 'completed') {
        if (state === CALL_STATES.NEGOTIATING) setState(CALL_STATES.CONNECTED);
        else if (state === CALL_STATES.RECONNECTING) _restoreFromReconnect();
      } else if (ice === 'failed') {
        // Hard failure — ICE gave up retrying on its own. Tear down.
        _clearReconnectTimer();
        _fail('ice_failed', new Error('ICE connection failed'));
      } else if (ice === 'disconnected') {
        // Transient: browser is re-checking connectivity. Surface
        // the RECONNECTING state so the user sees we're recovering
        // (instead of silent freeze) and start a deadline; if ICE
        // hasn't returned to connected by then, we fail explicitly.
        _enterReconnect('ice_disconnected');
      }
    };

    return conn;
  }

  function _enterReconnect(reason) {
    if (state === CALL_STATES.ENDED) return;
    if (state === CALL_STATES.RECONNECTING) return;  // idempotent
    if (state !== CALL_STATES.CONNECTED
        && state !== CALL_STATES.NEGOTIATING) {
      // Pre-connected states (CONNECTING, RINGING) have their own
      // failure paths — don't shadow them.
      return;
    }
    stateBeforeReconnect = state;
    setState(CALL_STATES.RECONNECTING);
    emit('reconnecting', { reason });
    _clearReconnectTimer();
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      if (state === CALL_STATES.RECONNECTING) {
        _fail('reconnect_timeout',
              new Error(`reconnect deadline exceeded (${reason})`));
      }
    }, RECONNECT_TIMEOUT_MS);
  }

  function _restoreFromReconnect() {
    if (state !== CALL_STATES.RECONNECTING) return;
    _clearReconnectTimer();
    const target = stateBeforeReconnect || CALL_STATES.CONNECTED;
    stateBeforeReconnect = null;
    setState(target);
    emit('reconnected', {});
  }

  function _clearReconnectTimer() {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  }

  function _flushIceQueue(sendSentinel) {
    if (iceFlushTimer) { clearTimeout(iceFlushTimer); iceFlushTimer = null; }
    if (!iceQueue.length && !sendSentinel) return;
    const batch = iceQueue.splice(0, iceQueue.length);
    if (state === CALL_STATES.ENDED) return;
    try {
      sendEnvelope({
        verb: 'candidates',
        peer: peerDid,
        data: {
          call_id: callId,
          candidates: batch,
        },
      });
    } catch (err) {
      // Lost signaling mid-trickle — surface it; the WebRTC connection
      // might still survive on already-exchanged candidates.
      console.warn('connect: candidate flush failed', err);
    }
  }

  async function _handleAccept() {
    if (state !== CALL_STATES.RINGING) return;
    setState(CALL_STATES.NEGOTIATING);
    answered = false;  // arm the answer latch for this negotiation
    const offer = await pc.createOffer({ offerToReceiveAudio: true });
    await pc.setLocalDescription(offer);
    sendEnvelope({
      verb: 'offer',
      peer: peerDid,
      data: {
        call_id: callId,
        sdp: offer.sdp,
      },
    });
  }

  async function _handleAnswer(data) {
    if (state !== CALL_STATES.NEGOTIATING) return;
    if (!data || !data.sdp) return;
    // First ANSWER wins. When the callee is signed in on multiple devices the
    // server fans the offer to all of them and each may answer; state only
    // leaves NEGOTIATING on ICE-connected, so without this latch a second
    // answer arriving in the pre-ICE window would call setRemoteDescription a
    // second time (signaling state is no longer have-local-offer) and throw,
    // failing an otherwise-good call. (The losing device's teardown via a
    // party-scoped select_answer is a separate, multi-device change.)
    if (answered) return;
    answered = true;
    await pc.setRemoteDescription({ type: 'answer', sdp: data.sdp });
    await _drainPendingRemoteCandidates();
  }

  async function _drainPendingRemoteCandidates() {
    if (!pc || !pc.remoteDescription) return;
    while (pendingRemoteCandidates.length) {
      const init = pendingRemoteCandidates.shift();
      try {
        await pc.addIceCandidate(init);
      } catch (err) {
        console.warn('connect: queued addIceCandidate failed', err);
      }
    }
  }

  async function _handleNegotiate(data) {
    // Mid-call SDP renegotiation routed from the peer. Two cases:
    //   1. We just sent a negotiate offer (caller-initiated escalate);
    //      this is the peer's answer — apply it.
    //   2. Peer initiated the renegotiation; this is their offer —
    //      setRemoteDescription, createAnswer, send back.
    if (!pc) return;
    if (state === CALL_STATES.ENDED) return;
    const desc = data && data.description;
    if (!desc || !desc.sdp || !desc.type) return;
    if (data.modalities) modalities = data.modalities;

    if (desc.type === 'answer') {
      try {
        await pc.setRemoteDescription({ type: 'answer', sdp: desc.sdp });
        await _drainPendingRemoteCandidates();
      } finally {
        renegotiating = false;
      }
      emit('negotiated', { modalities, initiatedLocally: true });
      return;
    }

    if (desc.type === 'offer') {
      await pc.setRemoteDescription({ type: 'offer', sdp: desc.sdp });
      await _drainPendingRemoteCandidates();
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      sendEnvelope({
        verb: 'negotiate',
        peer: peerDid,
        data: {
          call_id: callId,
          description: { type: 'answer', sdp: answer.sdp },
          modalities,
        },
      });
      emit('negotiated', { modalities, initiatedLocally: false });
    }
  }

  async function _handleRemoteCandidates(data) {
    if (!pc) return;
    const cands = Array.isArray(data.candidates) ? data.candidates : [];
    for (const c of cands) {
      if (!c || !c.candidate) continue; // empty == end-of-gathering sentinel
      const init = {
        candidate: c.candidate,
        sdpMid: c.sdpMid,
        sdpMLineIndex: c.sdpMLineIndex,
      };
      // Buffer if the peer's ANSWER hasn't been processed yet. The
      // peer starts trickling candidates immediately after their
      // setLocalDescription; depending on WS ordering and async tick
      // boundaries, those candidates can land here before _handleAnswer
      // has resolved setRemoteDescription. Drained on every
      // setRemoteDescription completion.
      if (!pc.remoteDescription) {
        pendingRemoteCandidates.push(init);
        continue;
      }
      try {
        await pc.addIceCandidate(init);
      } catch (err) {
        // A bad candidate from a buggy peer shouldn't kill the call;
        // there'll be more candidates trickling in. Log and move on.
        console.warn('connect: addIceCandidate failed', err);
      }
    }
  }

  function _fail(reason, err) {
    endReason = reason;
    lastError = err || null;
    emit('error', { reason, error: err });
    _tearDown();
  }

  function _tearDown() {
    if (state === CALL_STATES.ENDED) return;
    _clearReconnectTimer();
    _stopQualityPoll();
    if (iceFlushTimer) { clearTimeout(iceFlushTimer); iceFlushTimer = null; }
    if (ringTimer) { clearTimeout(ringTimer); ringTimer = null; }
    if (negotiateTimer) { clearTimeout(negotiateTimer); negotiateTimer = null; }
    iceQueue.length = 0;
    for (const off of unsubs) { try { off(); } catch (_) {} }
    unsubs.length = 0;
    if (pc) {
      try { pc.close(); } catch (_) {}
      pc = null;
    }
    if (localStream) {
      for (const t of localStream.getTracks()) { try { t.stop(); } catch (_) {} }
      localStream = null;
    }
    setState(CALL_STATES.ENDED);
  }

  // ── Public API ──────────────────────────────────────────────
  async function start() {
    if (state !== CALL_STATES.IDLE) {
      throw new Error(`placeCall: already ${state}`);
    }
    setState(CALL_STATES.CONNECTING);

    // 1) Make sure the signaling socket is up and welcomed.
    let welcome;
    try {
      welcome = await ensureConnected();
    } catch (err) {
      _fail('signaling_unavailable', err);
      throw err;
    }
    if (!welcome) welcome = getWelcome();

    // 2) Capture the local mic (+ camera if video escalation requested).
    //    We do this BEFORE INVITE so the browser permission prompt
    //    happens at click-time while the user gesture is still valid.
    //    Failing here aborts the call cleanly.
    //
    //    AEC/AGC/NS + deviceId constraint handling is centralized in
    //    camera.js — same primitive the pre-call preview + VL frame
    //    consumer use, so a single fix lands in every consumer.
    try {
      localStream = await openCameraStream({
        deviceId: withVideo ? activeVideoDeviceId : '',
        audio: true,
        video: withVideo,
      });
    } catch (err) {
      _fail('mic_denied', err);
      throw err;
    }

    // 3) Build the PC with the TURN cred from welcome envelope.
    const iceServers = welcome && welcome.turn ? [{
      urls: welcome.turn.urls,
      username: welcome.turn.username,
      credential: welcome.turn.credential,
    }] : [];
    pc = _buildPeerConnection(iceServers);
    for (const track of localStream.getTracks()) {
      const sender = pc.addTrack(track, localStream);
      // Capture the video sender so the quality poll can adapt its
      // encoding. Without this, a call that STARTS with video left
      // videoSender null → applyVideoQualityProfile never ran → the
      // encoder stayed on Chrome's conservative default bitrate and the
      // picture looked permanently low-res. (Mid-call addVideo() set it;
      // the start-with-video path didn't.)
      if (track.kind === 'video') videoSender = sender;
    }
    // Lift the encoder to the full-quality ceiling immediately rather than
    // waiting for the first 2s quality poll — best-effort (setParameters can
    // throw pre-negotiation; the helper swallows that and the poll retries).
    // Lift audio off the browser default immediately rather than waiting
    // for the first 2s quality poll — otherwise every call opens with a
    // couple of seconds of thin, ~32kbps voice before adaptation lands.
    applyAudioQualityProfile(pc, 'excellent').catch(() => {});
    if (videoSender) {
      applyVideoQualityProfile(videoSender, 'excellent').catch(() => {});
    }

    // 4) Wire WS event listeners BEFORE the INVITE so we don't miss an
    //    immediate decline / accept race.
    wireWsListeners();

    // 5) Mint a client-side call_id (Matrix MSC2746 pattern — initiator
    //    mints; both ends reconcile to the same id). The server returns
    //    its own call_id in the 'routed' ack; we adopt that to stay
    //    aligned with the server's persistence.
    try {
      const ack = await sendAndAwaitRouted({
        verb: 'invite',
        peer: peerDid,
        data: { modalities },
      });
      callId = ack.call_id || null;
      if (!callId) {
        throw new Error('connect: routed ack missing call_id');
      }
    } catch (err) {
      _fail('invite_failed', err);
      throw err;
    }

    setState(CALL_STATES.RINGING);
    return { callId, peerDid };
  }

  async function hangup(reason = 'local_hangup') {
    if (state === CALL_STATES.ENDED) return;
    if (callId) {
      try {
        sendEnvelope({
          verb: 'hangup',
          peer: peerDid,
          data: { call_id: callId, reason },
        });
      } catch (_) {
        // Best-effort — the receiver will time-out on its end if the
        // hangup didn't make it through.
      }
    }
    endReason = reason;
    _tearDown();
  }

  function setMicMuted(muted) {
    if (!localStream) return;
    for (const t of localStream.getAudioTracks()) {
      t.enabled = !muted;
    }
    // Announce to the peer so their UI can surface a "Peer is muted"
    // badge. Best-effort — if the WS is mid-flap the badge is just
    // out of date for a few seconds; the audio cue (silence) tells
    // them anyway. No retry queue for this.
    if (callId && state !== CALL_STATES.ENDED) {
      try {
        sendEnvelope({
          verb: 'mute_state',
          peer: peerDid,
          data: { call_id: callId, muted: !!muted },
        });
      } catch (_) { /* WS not open — accept the stale badge */ }
    }
  }

  function setVideoEnabled(enabled) {
    if (!localStream) return;
    for (const t of localStream.getVideoTracks()) {
      t.enabled = enabled;
    }
    // Announce, exactly as setMicMuted does. A disabled video track keeps
    // sending BLACK FRAMES rather than stopping, so unlike audio there is
    // no natural cue for the peer — without this signal their tile is a
    // black rectangle indistinguishable from a frozen or dropped call.
    // Best-effort, same as mute: a stale badge for a few seconds is an
    // acceptable failure mode, so no retry queue.
    if (callId && state !== CALL_STATES.ENDED) {
      try {
        sendEnvelope({
          verb: 'video_state',
          peer: peerDid,
          data: { call_id: callId, video_enabled: !!enabled },
        });
      } catch (_) { /* WS not open — accept the stale placeholder */ }
    }
  }

  /** True while the local camera track is live and enabled. */
  function isVideoEnabled() {
    const tracks = localStream?.getVideoTracks?.() || [];
    return tracks.length > 0 && tracks.some((t) => t.enabled);
  }

  /**
   * Mid-call escalate to video. Captures the camera, adds the track
   * to the existing PeerConnection, renegotiates via MSG_NEGOTIATE.
   *
   * Resolves once the renegotiation offer has been sent (the peer's
   * answer arrives async via _handleNegotiate). Throws on permission
   * denial. No-op when video is already attached.
   */
  async function addVideo() {
    if (state !== CALL_STATES.CONNECTED && state !== CALL_STATES.NEGOTIATING) {
      throw new Error(`addVideo: call is ${state}`);
    }
    if (!pc) throw new Error('addVideo: no peer connection');
    if (renegotiating) throw new Error('addVideo: renegotiation already in flight');
    if (localStream && localStream.getVideoTracks().length > 0 && videoSender) {
      return; // already attached
    }

    let camStream;
    try {
      camStream = await openCameraStream({
        deviceId: activeVideoDeviceId,
        audio: false,
        video: true,
      });
    } catch (err) {
      const e = new Error('camera_denied');
      e.cause = err;
      throw e;
    }
    const videoTrack = camStream.getVideoTracks()[0];
    if (!videoTrack) {
      throw new Error('camera_no_track');
    }

    // Merge the new track into the local stream so UIs that read
    // session.localStream see it without a separate handle.
    if (localStream) {
      try { localStream.addTrack(videoTrack); } catch (_) {}
    } else {
      localStream = camStream;
    }

    renegotiating = true;
    try {
      videoSender = pc.addTrack(videoTrack, localStream);
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      modalities = 'audio,video';
      sendEnvelope({
        verb: 'negotiate',
        peer: peerDid,
        data: {
          call_id: callId,
          description: { type: 'offer', sdp: offer.sdp },
          modalities,
        },
      });
      emit('local-stream-updated', { localStream, modalities });
    } catch (err) {
      // Clean up the track if we couldn't get the negotiate out — the
      // call should keep audio working even if escalation fails.
      try { videoTrack.stop(); } catch (_) {}
      try { if (videoSender) pc.removeTrack(videoSender); } catch (_) {}
      videoSender = null;
      renegotiating = false;
      throw err;
    }
  }

  /**
   * Mid-call de-escalate — drop the video track and renegotiate.
   * No-op when video isn't attached.
   */
  async function removeVideo() {
    if (!pc) return;
    if (renegotiating) throw new Error('removeVideo: renegotiation already in flight');
    if (!videoSender && (!localStream || localStream.getVideoTracks().length === 0)) {
      return;
    }

    if (localStream) {
      for (const t of localStream.getVideoTracks()) {
        try { t.stop(); } catch (_) {}
        try { localStream.removeTrack(t); } catch (_) {}
      }
    }
    try { if (videoSender) pc.removeTrack(videoSender); } catch (_) {}
    videoSender = null;

    renegotiating = true;
    try {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      modalities = 'audio';
      sendEnvelope({
        verb: 'negotiate',
        peer: peerDid,
        data: {
          call_id: callId,
          description: { type: 'offer', sdp: offer.sdp },
          modalities,
        },
      });
      emit('local-stream-updated', { localStream, modalities });
    } catch (err) {
      renegotiating = false;
      throw err;
    }
  }

  /**
   * Swap the active camera mid-call without renegotiating SDP.
   * RTCRtpSender.replaceTrack lets us hot-swap the source on the
   * existing send transceiver — peer keeps decoding from the same
   * MID, no offer/answer round trip needed. The peer's video element
   * just sees the pixels change.
   *
   * If video isn't currently attached (audio-only call), updates the
   * persisted preference so the next addVideo() picks up the new
   * camera; doesn't auto-escalate.
   */
  async function switchVideoDevice(newDeviceId) {
    const next = String(newDeviceId || '').trim();
    if (!next) throw new Error('switchVideoDevice: deviceId required');
    activeVideoDeviceId = next;

    // Audio-only call — nothing to swap right now. The new preference
    // will be honored the next time the caller escalates.
    if (!videoSender || !localStream
        || localStream.getVideoTracks().length === 0) {
      return { swapped: false };
    }

    let newStream;
    try {
      newStream = await openCameraStream({
        deviceId: next, audio: false, video: true,
      });
    } catch (err) {
      const e = new Error('camera_switch_failed');
      e.cause = err;
      throw e;
    }
    const newTrack = newStream.getVideoTracks()[0];
    if (!newTrack) {
      try { for (const t of newStream.getTracks()) t.stop(); } catch (_) {}
      throw new Error('camera_switch_no_track');
    }

    try {
      await videoSender.replaceTrack(newTrack);
    } catch (err) {
      try { newTrack.stop(); } catch (_) {}
      throw err;
    }

    // Stop + remove the OLD track, attach the new one to localStream
    // so the self-view <video> picks it up via the local-stream-updated
    // emit. We keep the same MediaStream object — UI elements that
    // already have it set as srcObject just see the new frames.
    for (const t of localStream.getVideoTracks()) {
      try { t.stop(); } catch (_) {}
      try { localStream.removeTrack(t); } catch (_) {}
    }
    try { localStream.addTrack(newTrack); } catch (_) {}
    emit('local-stream-updated', { localStream, modalities });
    return { swapped: true };
  }

  /**
   * Start sharing the screen. Replaces the camera track on the
   * existing video sender so the peer sees the screen on the same
   * MID — no SDP renegotiation, instant cutover.
   *
   * Auto-restores camera when the user clicks the browser's native
   * "Stop sharing" chrome (the screen track fires 'ended'). UI can
   * also call stopScreenShare() to end programmatically.
   *
   * If the call is currently audio-only, this throws — caller should
   * first call addVideo() to establish the video sender. We choose
   * not to silently escalate because the user explicitly requested
   * screen-share; surprising them with a video escalation is worse
   * than telling them to add video first.
   */
  async function startScreenShare() {
    if (state === CALL_STATES.ENDED) {
      throw new Error('startScreenShare: call is ended');
    }
    if (!videoSender) {
      const e = new Error('screen_share_requires_video');
      throw e;
    }
    if (screenStream) {
      // Already sharing — no-op rather than re-prompting the user
      // with the browser picker.
      return { started: false, alreadySharing: true };
    }

    let display;
    try {
      display = await openScreenStream({ video: true, audio: false });
    } catch (err) {
      if (String(err?.name) === 'NotAllowedError') {
        // User cancelled the picker — silent no-op (not an error
        // worth surfacing as a toast).
        const e = new Error('screen_share_cancelled');
        e.cause = err;
        throw e;
      }
      const e = new Error('screen_share_failed');
      e.cause = err;
      throw e;
    }
    const screenTrack = display.getVideoTracks()[0];
    if (!screenTrack) {
      try { for (const t of display.getTracks()) t.stop(); } catch (_) {}
      throw new Error('screen_share_no_track');
    }

    // Stash the camera track. We DON'T stop it — we just detach from
    // the sender — so we can put it back without a re-acquire (which
    // would trigger the browser's permission UI a second time).
    cameraTrackBeforeShare = videoSender.track || null;

    try {
      await videoSender.replaceTrack(screenTrack);
    } catch (err) {
      try { screenTrack.stop(); } catch (_) {}
      cameraTrackBeforeShare = null;
      throw err;
    }

    screenStream = display;
    // Browser-native "Stop sharing" chrome ends the track. Listen so
    // we can swap the camera back automatically and emit a state
    // change for the UI badge.
    screenTrackEndedListener = () => {
      // Fire-and-forget — the call shouldn't fail if restore fails.
      stopScreenShare().catch((err) => {
        console.warn('connect: screen-share auto-restore failed', err);
      });
    };
    screenTrack.addEventListener('ended', screenTrackEndedListener);

    emit('screen-share-changed', { sharing: true });
    return { started: true };
  }

  /**
   * Stop the screen share and put the camera track back on the
   * existing video sender. Safe to call when not sharing (no-op).
   */
  async function stopScreenShare() {
    if (!screenStream) return { stopped: false };

    const screenTrack = screenStream.getVideoTracks()[0];
    if (screenTrack && screenTrackEndedListener) {
      // Defensive — Safari has thrown on removeEventListener with
      // stale capture references; log so we know if it happens.
      try {
        screenTrack.removeEventListener('ended', screenTrackEndedListener);
      } catch (err) { console.debug('screen-track removeEventListener', err); }
    }
    screenTrackEndedListener = null;

    // Stop the screen track first so the OS share-indicator goes
    // away even if the camera-restore step fails below.
    for (const t of screenStream.getTracks()) {
      try { t.stop(); } catch (err) { console.debug('screen track stop', err); }
    }
    screenStream = null;

    // Restore the camera track. The stashed track is likely still
    // live (we never stopped it), so replaceTrack puts it right back
    // on the sender. If the user disabled video mid-share or the
    // track ended for some reason, we re-acquire from the camera.
    let restoreTrack = cameraTrackBeforeShare;
    cameraTrackBeforeShare = null;

    if (!restoreTrack || restoreTrack.readyState === 'ended') {
      // Re-acquire — same path addVideo uses, so AEC/AGC/NS + the
      // active deviceId are honored consistently.
      try {
        const cam = await openCameraStream({
          deviceId: activeVideoDeviceId,
          audio: false,
          video: true,
        });
        restoreTrack = cam.getVideoTracks()[0] || null;
        if (restoreTrack && localStream) {
          // Add to localStream so the self-view <video> shows it again.
          try { localStream.addTrack(restoreTrack); } catch (_) {}
        }
      } catch (err) {
        emit('screen-share-changed', { sharing: false });
        // Camera couldn't come back — the call survives but stays
        // video-disabled on our side. Caller's UI should show this.
        throw Object.assign(new Error('camera_restore_failed'), { cause: err });
      }
    }

    if (videoSender && restoreTrack) {
      try {
        await videoSender.replaceTrack(restoreTrack);
      } catch (err) {
        emit('screen-share-changed', { sharing: false });
        throw err;
      }
    }

    emit('screen-share-changed', { sharing: false });
    return { stopped: true };
  }

  return {
    on,
    start,
    hangup,
    setMicMuted,
    setVideoEnabled,
    addVideo,
    removeVideo,
    switchVideoDevice,
    startScreenShare,
    stopScreenShare,
    get state() { return state; },
    get callId() { return callId; },
    get peerDid() { return peerDid; },
    get modalities() { return modalities; },
    get withVideo() { return modalities.includes('video'); },
    // Single source of truth for the camera toggle's UI state — reading
    // the live track, so the chrome can never drift from the stream.
    get isVideoEnabled() { return isVideoEnabled(); },
    get localStream() { return localStream; },
    get remoteStream() { return remoteStream; },
    get lastError() { return lastError; },
    get videoDeviceId() { return activeVideoDeviceId; },
    get isScreenSharing() { return screenStream !== null; },
  };
}
