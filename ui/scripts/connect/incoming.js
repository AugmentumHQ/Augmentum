/* connect/incoming.js — receiver-side Connect call consumer.
 *
 * Mirrors dialer.js in reverse: where the caller mints the invite +
 * createOffer, the receiver waits for the user to accept (via a
 * notification banner action) and then completes the handshake by
 * consuming the caller's MSG_OFFER and producing an MSG_ANSWER.
 *
 * Wiring:
 *
 *   1) initIncomingCallConsumer() subscribes to the connect WS event
 *      EVENT_INVITE so the consumer always knows which call_ids are
 *      pending. The notification substrate already shows the banner;
 *      this module just shadows the pending-call map so it can react
 *      when the user clicks Accept.
 *
 *   2) When notifications.js fires the synchronous
 *      'augmentum:notification-action' event for a
 *      'connect.call.incoming' banner, this module checks if the
 *      action is "accept" and, if so, IMMEDIATELY calls
 *      navigator.mediaDevices.getUserMedia() — inside the click
 *      gesture so the browser's permission prompt fires normally.
 *      It then builds the RTCPeerConnection from the welcome TURN
 *      config and arms listeners for EVENT_OFFER / EVENT_CANDIDATES.
 *
 *   3) When EVENT_OFFER arrives (matching call_id), setRemoteDescription,
 *      createAnswer, MSG_ANSWER. Trickle ICE flows both ways.
 *
 *   4) The in-call overlay is reused from ui.js (same status pill +
 *      hangup affordance + remote audio playback).
 *
 * What this module deliberately doesn't do: handle hangup mid-call
 * differently from the caller side — the dialer's tearDown shape is
 * symmetric, so a thin shared session shape works for both sides.
 *
 * WebRTC jargon (ICE / candidate / STUN / TURN / SDP) is glossed in the
 * header of dialer.js — the same standard browser terms apply here.
 */

import {
  ensureConnected,
  getWelcome,
  on as onConnectEvent,
  onStateChange as onConnectStateChange,
  send as sendEnvelope,
} from './client.js';
import {
  openCameraStream,
  openScreenStream,
  resolveVideoDeviceId,
} from '../camera.js';
import { applyVideoQualityProfile, applyAudioQualityProfile } from './quality.js';

const ICE_FLUSH_MS = 100;
const ICE_FLUSH_BATCH = 8;
// Mirror of dialer.js's reconnect cap. Mid-call ICE blips and brief
// signaling-WS flaps surface as RECONNECTING instead of an immediate
// tear-down; after this deadline we fail cleanly.
const RECONNECT_TIMEOUT_MS = 20_000;
// Bounds the NEGOTIATING (SDP/ICE) phase so the receiver's "Connecting…"
// can't hang forever if ICE stalls without firing 'failed'. Symmetric with
// dialer.js (see the rationale there).
const NEGOTIATE_TIMEOUT_MS = 30_000;

const CALL_STATES = Object.freeze({
  PENDING: 'pending',          // invite received, user hasn't acted
  ACCEPTING: 'accepting',      // user clicked accept, mic+peer ramping up
  NEGOTIATING: 'negotiating',  // offer received, answer in flight
  CONNECTED: 'connected',
  // Mid-call transient — ICE disconnected or WS dropped while
  // CONNECTED. Returns to CONNECTED on recovery, ENDED on timeout.
  RECONNECTING: 'reconnecting',
  ENDED: 'ended',
});

// pending invites we know about (key: call_id)
const _pending = new Map();
// active receiver sessions (key: call_id) — once user accepts
const _sessions = new Map();
// UI callback so app.js / ui.js can render the overlay
let _onIncomingSession = null;

let _initialized = false;

// ── Public API ──────────────────────────────────────────────────

/**
 * Wire up the consumer. Safe to call multiple times — second + later
 * calls are no-ops. `onSession({callerDid, callId, kind, session})` is
 * called when the user accepts and an active CallSession is ready.
 */
export function initIncomingCallConsumer({ onSession } = {}) {
  if (_initialized) return;
  _initialized = true;
  _onIncomingSession = typeof onSession === 'function' ? onSession : null;

  // Track pending invites surfaced via WS so we have call_id ↔ caller
  // mapping at click time. Notifications give us the same info via
  // payload, but the WS path lets us decline-via-policy in the future
  // (e.g. auto-decline when on focus mode) without a banner click.
  onConnectEvent('invite', ({ data, peer }) => {
    const callId = String(data?.call_id || '');
    if (!callId) return;
    _pending.set(callId, {
      callId,
      callerDid: peer,
      modalities: String(data?.modalities || 'audio'),
      receivedAt: Date.now(),
    });
    // Garbage-collect older entries. Two minutes is generous —
    // typical invite lifetime is 60s but we cushion for clock skew.
    setTimeout(() => _pending.delete(callId), 120000);
  });

  // Synchronous handler — must call getUserMedia BEFORE the click
  // gesture is consumed, otherwise the permission prompt is denied.
  window.addEventListener(
    'augmentum:notification-action',
    _onActionGesture,
    // Capture so we run before the notification's own async POST.
    { capture: true },
  );
}

/** Best-effort introspection — useful for debug overlays. */
export function getPendingInvites() {
  return Array.from(_pending.values());
}

/** Test seam — clears in-memory state. */
export function resetIncomingConsumer() {
  _pending.clear();
  for (const sess of _sessions.values()) {
    try { sess.hangup('test_reset'); } catch (_) {}
  }
  _sessions.clear();
}

// ── Click-gesture entry point ───────────────────────────────────

function _onActionGesture(evt) {
  const detail = evt.detail || {};
  const n = detail.notification || {};
  const actionId = detail.actionId;
  if (!n.channel_id || !n.channel_id.startsWith('connect.call.')) return;
  if (actionId !== 'accept') return;

  const callId = String(n.payload?.call_id || '');
  if (!callId) return;
  if (_sessions.has(callId)) return; // duplicate click

  // Pull caller info preferentially from notification payload (always
  // present), with WS pending-map as a fallback for diagnostics.
  const callerDid = String(n.payload?.initiator_did || '');
  const pending = _pending.get(callId);
  const modalities = String(
    n.payload?.modalities || pending?.modalities || 'audio',
  );
  const wantsVideo = modalities.split(',').map((s) => s.trim()).includes('video');

  // Acquire mic+camera SYNCHRONOUSLY — must not await before this.
  // The browser tracks click-gesture in microtasks; an immediate call
  // to openCameraStream is in the same tick as the dispatchEvent
  // caller; camera.js itself doesn't await before getUserMedia.
  // Receiver-side device pick uses the persisted preference (set via
  // the dial picker's camera dropdown), with browser-default fallback.
  const initialDeviceId = wantsVideo ? resolveVideoDeviceId() : '';
  const mediaPromise = openCameraStream({
    deviceId: initialDeviceId,
    audio: true,
    video: wantsVideo,
  });

  // Build the receiver-side session and stash it. The actual peer
  // wiring happens asynchronously inside _runReceiverFlow.
  const session = _buildReceiverSession({
    callId,
    callerDid,
    modalities,
    mediaPromise,
    initialDeviceId,
  });
  _sessions.set(callId, session);

  if (_onIncomingSession) {
    try {
      _onIncomingSession({
        callId, callerDid, modalities, session,
      });
    } catch (err) {
      console.warn('connect: onIncomingSession callback failed', err);
    }
  }

  // Fire-and-forget — the action POST in notifications.js completes
  // independently and routes EVENT_ACCEPT back to the caller. We just
  // need to be ready for the resulting EVENT_OFFER.
  session._runReceiverFlow().catch((err) => {
    console.warn('connect: receiver flow failed', err);
  });
}

// ── Session shape (mirrors dialer.js for parity) ────────────────

function _buildReceiverSession({
  callId, callerDid, modalities: initialModalities, mediaPromise,
  initialDeviceId = '',
}) {
  let state = CALL_STATES.ACCEPTING;
  let pc = null;
  let localStream = null;
  let remoteStream = null;
  let endReason = null;
  let lastError = null;
  let modalities = initialModalities;
  let videoSender = null;
  let renegotiating = false;
  // Active video camera deviceId. Updated by switchVideoDevice for
  // hot-swaps; consulted by addVideo when escalating an audio-only
  // call so the user's preferred camera is honored.
  let activeVideoDeviceId = initialDeviceId;
  // Screen-share state. Symmetric with dialer.js — swap camera track
  // for screen track via replaceTrack, stash original to restore on stop.
  let screenStream = null;
  let cameraTrackBeforeShare = null;
  let screenTrackEndedListener = null;
  const listeners = new Map();
  const unsubs = [];
  const iceQueue = [];
  let iceFlushTimer = null;
  // Buffer for remote ICE candidates that arrive before
  // setRemoteDescription has been called. Without this, the
  // caller's first candidate batch typically races the OFFER
  // and addIceCandidate throws InvalidStateError
  // ("remote description was null"), which silently fails the
  // entire ICE gather → call sits at "Connecting" forever.
  // Drained by _handleOffer + _handleNegotiate once remoteDescription
  // resolves.
  const pendingRemoteCandidates = [];
  // The OFFER frame can arrive before pc is built (we have to wait
  // for getUserMedia + addTrack first). The caller sends MSG_OFFER
  // ~15-20ms after EVENT_ACCEPT reaches them; the receiver's
  // getUserMedia takes longer than that even when permissions are
  // already granted. We wire WS listeners EARLY (before mediaPromise)
  // so the frame lands here, then _replay_ the offer once pc is ready.
  let pendingOffer = null;
  // Reconnect bookkeeping — see dialer.js for shape.
  let reconnectTimer = null;
  let negotiateTimer = null;   // bounds the NEGOTIATING (SDP/ICE) phase
  let stateBeforeReconnect = null;
  // Quality-poll bookkeeping — same shape as dialer.js.
  let qualityTimer = null;
  let lastQualityBucket = '';

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
      try { fn(payload); }
      catch (err) { console.warn(`incoming listener (${event}) failed`, err); }
    }
  }
  function setState(next) {
    if (state === next) return;
    state = next;
    emit('state-change', next);
    if (next === CALL_STATES.CONNECTED) {
      _startQualityPoll();
    } else if (next === CALL_STATES.ENDED) {
      _stopQualityPoll();
    }
    // Bound the SDP/ICE negotiation — armed on entering NEGOTIATING, cleared
    // the moment we leave it. Mirrors dialer.js.
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
    if (next === CALL_STATES.ENDED) {
      emit('ended', { reason: endReason, error: lastError });
    }
  }

  function _startQualityPoll() {
    if (qualityTimer) return;
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
    if (bucket !== lastQualityBucket) {
      lastQualityBucket = bucket;
      // Match the dialer's adaptation curve — symmetric so both sides
      // throttle to the same profile under congestion.
      if (videoSender) {
        applyVideoQualityProfile(videoSender, bucket).catch(() => {});
      }
      // Symmetric with the dialer — both ends must run the same curve or
      // one side over-restricts while the other doesn't move.
      applyAudioQualityProfile(pc, bucket).catch(() => {});
      emit('quality-change', { bucket, rttMs, lossPct });
    }
  }

  function _qualityBucket({ rtt, lossFrac }) {
    if (rtt == null && lossFrac == null) return 'measuring';
    const rttMs = (rtt ?? 0) * 1000;
    const loss = lossFrac ?? 0;
    if (rttMs < 100 && loss < 0.01) return 'excellent';
    if (rttMs < 200 && loss < 0.03) return 'good';
    if (rttMs < 400 && loss < 0.08) return 'weak';
    return 'poor';
  }

  function _matchesCall(data) {
    return data && data.call_id === callId;
  }

  function _buildPeerConnection(iceServers) {
    const conn = new RTCPeerConnection({ iceServers: iceServers || [] });

    conn.ontrack = (evt) => {
      const stream = (evt.streams && evt.streams[0]) || new MediaStream([evt.track]);
      remoteStream = stream;
      emit('remote-stream', stream);
    };

    conn.onicecandidate = (evt) => {
      if (!evt.candidate) {
        _flushIceQueue(true);
        return;
      }
      iceQueue.push({
        candidate: evt.candidate.candidate,
        sdpMid: evt.candidate.sdpMid,
        sdpMLineIndex: evt.candidate.sdpMLineIndex,
      });
      if (iceQueue.length >= ICE_FLUSH_BATCH) _flushIceQueue(false);
      else if (!iceFlushTimer) {
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
        _clearReconnectTimer();
        _fail('ice_failed', new Error('ICE connection failed'));
      } else if (ice === 'disconnected') {
        _enterReconnect('ice_disconnected');
      }
    };

    return conn;
  }

  function _enterReconnect(reason) {
    if (state === CALL_STATES.ENDED) return;
    if (state === CALL_STATES.RECONNECTING) return;
    if (state !== CALL_STATES.CONNECTED
        && state !== CALL_STATES.NEGOTIATING) {
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
        peer: callerDid,
        data: { call_id: callId, candidates: batch },
      });
    } catch (err) {
      console.warn('connect: receiver candidate flush failed', err);
    }
  }

  function wireWsListeners() {
    unsubs.push(onConnectEvent('offer', ({ data, peer }) => {
      if (!_matchesCall(data) || peer !== callerDid) return;
      _handleOffer(data).catch((err) => _fail('offer_failed', err));
    }));
    unsubs.push(onConnectEvent('candidates', ({ data, peer }) => {
      if (!_matchesCall(data) || peer !== callerDid) return;
      _handleRemoteCandidates(data).catch((err) =>
        _fail('candidates_failed', err),
      );
    }));
    unsubs.push(onConnectEvent('negotiate', ({ data, peer }) => {
      if (!_matchesCall(data) || peer !== callerDid) return;
      _handleNegotiate(data).catch((err) => _fail('negotiate_failed', err));
    }));
    unsubs.push(onConnectEvent('hangup', ({ data, peer }) => {
      if (!_matchesCall(data) || peer !== callerDid) return;
      endReason = 'remote_hangup';
      _tearDown();
    }));
    unsubs.push(onConnectEvent('mute_state', ({ data, peer }) => {
      if (!_matchesCall(data) || peer !== callerDid) return;
      emit('peer-mute', { muted: !!data.muted });
    }));
    unsubs.push(onConnectEvent('__closed', () => {
      // WS drop mid-call: a CONNECTED call survives because media
      // flows directly over ICE; surface RECONNECTING and wait for
      // the WS to come back. Pre-connect states have no media path
      // yet and have to fail.
      if (state === CALL_STATES.CONNECTED) {
        _enterReconnect('signaling_dropped');
      } else if (state !== CALL_STATES.ENDED
              && state !== CALL_STATES.RECONNECTING) {
        _fail('signaling_lost', new Error('signaling socket closed'));
      }
    }));
    unsubs.push(onConnectStateChange((wsState) => {
      if (wsState !== 'open') return;
      if (state !== CALL_STATES.RECONNECTING) return;
      const ice = pc && pc.iceConnectionState;
      if (ice === 'connected' || ice === 'completed') {
        _restoreFromReconnect();
      }
    }));
  }

  async function _handleOffer(data) {
    if (state === CALL_STATES.ENDED) return;
    if (state !== CALL_STATES.ACCEPTING && state !== CALL_STATES.NEGOTIATING) {
      return;
    }
    // pc not built yet — buffer the offer and replay after pc is
    // constructed in _runReceiverFlow. The OFFER frame routinely beats
    // getUserMedia's resolution on a desktop browser; without this
    // buffer the receiver never sends an ANSWER and the call sits at
    // "Connecting" forever.
    if (!pc) {
      pendingOffer = data;
      return;
    }
    setState(CALL_STATES.NEGOTIATING);
    await pc.setRemoteDescription({ type: 'offer', sdp: data.sdp });
    await _drainPendingRemoteCandidates();
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
    sendEnvelope({
      verb: 'answer',
      peer: callerDid,
      data: { call_id: callId, sdp: answer.sdp },
    });
  }

  async function _drainPendingRemoteCandidates() {
    if (!pc || !pc.remoteDescription) return;
    while (pendingRemoteCandidates.length) {
      const init = pendingRemoteCandidates.shift();
      try {
        await pc.addIceCandidate(init);
      } catch (err) {
        console.warn('connect: receiver queued addIceCandidate failed', err);
      }
    }
  }

  async function _handleRemoteCandidates(data) {
    const cands = Array.isArray(data.candidates) ? data.candidates : [];
    for (const c of cands) {
      if (!c || !c.candidate) continue;
      const init = {
        candidate: c.candidate,
        sdpMid: c.sdpMid,
        sdpMLineIndex: c.sdpMLineIndex,
      };
      // Buffer if pc isn't built yet OR if remoteDescription hasn't
      // been set yet. Either way addIceCandidate would throw
      // InvalidStateError, every candidate in the batch would be
      // dropped, and ICE would never find a path → call stuck in
      // "Connecting" forever. Drained as soon as setRemoteDescription
      // resolves (see _drainPendingRemoteCandidates).
      if (!pc || !pc.remoteDescription) {
        pendingRemoteCandidates.push(init);
        continue;
      }
      try {
        await pc.addIceCandidate(init);
      } catch (err) {
        console.warn('connect: receiver addIceCandidate failed', err);
      }
    }
  }

  async function _handleNegotiate(data) {
    // Symmetric with dialer.js — caller-initiated negotiate offer is
    // the common case (caller wants to add video to an audio call);
    // receiver-initiated still works for completeness.
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
        peer: callerDid,
        data: {
          call_id: callId,
          description: { type: 'answer', sdp: answer.sdp },
          modalities,
        },
      });
      emit('negotiated', { modalities, initiatedLocally: false });
    }
  }

  async function addVideo() {
    if (state !== CALL_STATES.CONNECTED && state !== CALL_STATES.NEGOTIATING) {
      throw new Error(`addVideo: call is ${state}`);
    }
    if (!pc) throw new Error('addVideo: no peer connection');
    if (renegotiating) throw new Error('addVideo: renegotiation already in flight');
    if (localStream && localStream.getVideoTracks().length > 0 && videoSender) {
      return;
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
    if (!videoTrack) throw new Error('camera_no_track');

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
        peer: callerDid,
        data: {
          call_id: callId,
          description: { type: 'offer', sdp: offer.sdp },
          modalities,
        },
      });
      emit('local-stream-updated', { localStream, modalities });
    } catch (err) {
      try { videoTrack.stop(); } catch (_) {}
      try { if (videoSender) pc.removeTrack(videoSender); } catch (_) {}
      videoSender = null;
      renegotiating = false;
      throw err;
    }
  }

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
        peer: callerDid,
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
    if (negotiateTimer) { clearTimeout(negotiateTimer); negotiateTimer = null; }
    iceQueue.length = 0;
    for (const off of unsubs) { try { off(); } catch (_) {} }
    unsubs.length = 0;
    if (pc) { try { pc.close(); } catch (_) {} pc = null; }
    if (localStream) {
      for (const t of localStream.getTracks()) { try { t.stop(); } catch (_) {} }
      localStream = null;
    }
    _sessions.delete(callId);
    setState(CALL_STATES.ENDED);
  }

  async function _runReceiverFlow() {
    // Ensure the signaling socket is up; the WS may have dropped
    // since the invite landed in the notification feed.
    let welcome;
    try {
      welcome = await ensureConnected();
    } catch (err) {
      _fail('signaling_unavailable', err);
      throw err;
    }
    if (!welcome) welcome = getWelcome();
    const iceServers = welcome && welcome.turn ? [{
      urls: welcome.turn.urls,
      username: welcome.turn.username,
      credential: welcome.turn.credential,
    }] : [];

    // Wire WS listeners BEFORE awaiting media. EVENT_OFFER lands
    // ~15-20ms after MSG_ACCEPT (the caller fires createOffer
    // immediately on EVENT_ACCEPT) and beats getUserMedia even when
    // permissions are already granted. The handlers buffer (into
    // pendingOffer / pendingRemoteCandidates) when pc isn't built
    // yet; we replay after pc is constructed.
    wireWsListeners();

    // Wait for the mic/camera we kicked off inside the click gesture.
    try {
      localStream = await mediaPromise;
    } catch (err) {
      _fail('mic_denied', err);
      throw err;
    }

    pc = _buildPeerConnection(iceServers);
    for (const track of localStream.getTracks()) {
      const sender = pc.addTrack(track, localStream);
      // Capture the video sender so the quality poll adapts its encoding —
      // same fix as the dialer: answering a video call left videoSender null,
      // so the 4 Mbps / full-resolution ceiling was never applied and the
      // outbound picture sat on Chrome's conservative default bitrate.
      if (track.kind === 'video') videoSender = sender;
    }
    // Jump straight to the full-quality ceiling instead of waiting for the
    // first quality poll (best-effort; the helper is defensive pre-negotiation).
    // Lift audio off the browser default immediately rather than waiting
    // for the first 2s quality poll — otherwise every call opens with a
    // couple of seconds of thin, ~32kbps voice before adaptation lands.
    applyAudioQualityProfile(pc, 'excellent').catch(() => {});
    if (videoSender) {
      applyVideoQualityProfile(videoSender, 'excellent').catch(() => {});
    }

    // Replay any OFFER frame that arrived during the media wait.
    // _handleOffer set pendingOffer instead of throwing because pc
    // was null. Now that pc + tracks are wired, run the SDP exchange.
    if (pendingOffer) {
      const offerData = pendingOffer;
      pendingOffer = null;
      try {
        await _handleOffer(offerData);
      } catch (err) {
        _fail('offer_failed', err);
      }
    }
  }

  async function hangup(reason = 'local_hangup') {
    if (state === CALL_STATES.ENDED) return;
    try {
      sendEnvelope({
        verb: 'hangup',
        peer: callerDid,
        data: { call_id: callId, reason },
      });
    } catch (_) { /* hangup envelope is best-effort — teardown below proceeds */ }
    endReason = reason;
    _tearDown();
  }

  function setMicMuted(muted) {
    if (!localStream) return;
    for (const t of localStream.getAudioTracks()) t.enabled = !muted;
    if (callId && state !== CALL_STATES.ENDED) {
      try {
        sendEnvelope({
          verb: 'mute_state',
          peer: callerDid,
          data: { call_id: callId, muted: !!muted },
        });
      } catch (_) { /* best-effort; receiver tolerates stale badge */ }
    }
  }

  function setVideoEnabled(enabled) {
    if (!localStream) return;
    for (const t of localStream.getVideoTracks()) t.enabled = enabled;
  }

  /**
   * Hot-swap the camera mid-call — symmetric with dialer.js. Uses
   * RTCRtpSender.replaceTrack so the peer keeps decoding from the
   * same MID without any SDP renegotiation.
   */
  async function switchVideoDevice(newDeviceId) {
    const next = String(newDeviceId || '').trim();
    if (!next) throw new Error('switchVideoDevice: deviceId required');
    activeVideoDeviceId = next;

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

    for (const t of localStream.getVideoTracks()) {
      try { t.stop(); } catch (_) {}
      try { localStream.removeTrack(t); } catch (_) {}
    }
    try { localStream.addTrack(newTrack); } catch (_) {}
    emit('local-stream-updated', { localStream, modalities });
    return { swapped: true };
  }

  /**
   * Start sharing the screen — same shape as dialer.js. Swaps the
   * camera track on the existing video sender via replaceTrack so
   * the peer sees the screen on the same MID, no SDP renegotiation.
   */
  async function startScreenShare() {
    if (state === CALL_STATES.ENDED) {
      throw new Error('startScreenShare: call is ended');
    }
    if (!videoSender) {
      throw new Error('screen_share_requires_video');
    }
    if (screenStream) return { started: false, alreadySharing: true };

    let display;
    try {
      display = await openScreenStream({ video: true, audio: false });
    } catch (err) {
      if (String(err?.name) === 'NotAllowedError') {
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

    cameraTrackBeforeShare = videoSender.track || null;

    try {
      await videoSender.replaceTrack(screenTrack);
    } catch (err) {
      try { screenTrack.stop(); } catch (_) {}
      cameraTrackBeforeShare = null;
      throw err;
    }

    screenStream = display;
    screenTrackEndedListener = () => {
      stopScreenShare().catch((err) => {
        console.warn('connect: screen-share auto-restore failed', err);
      });
    };
    screenTrack.addEventListener('ended', screenTrackEndedListener);

    emit('screen-share-changed', { sharing: true });
    return { started: true };
  }

  async function stopScreenShare() {
    if (!screenStream) return { stopped: false };

    const screenTrack = screenStream.getVideoTracks()[0];
    if (screenTrack && screenTrackEndedListener) {
      try { screenTrack.removeEventListener('ended', screenTrackEndedListener); }
      catch (err) { console.debug('screen-track removeEventListener', err); }
    }
    screenTrackEndedListener = null;

    for (const t of screenStream.getTracks()) {
      try { t.stop(); } catch (err) { console.debug('screen track stop', err); }
    }
    screenStream = null;

    let restoreTrack = cameraTrackBeforeShare;
    cameraTrackBeforeShare = null;

    if (!restoreTrack || restoreTrack.readyState === 'ended') {
      try {
        const cam = await openCameraStream({
          deviceId: activeVideoDeviceId, audio: false, video: true,
        });
        restoreTrack = cam.getVideoTracks()[0] || null;
        if (restoreTrack && localStream) {
          try { localStream.addTrack(restoreTrack); } catch (_) {}
        }
      } catch (err) {
        emit('screen-share-changed', { sharing: false });
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
    hangup,
    setMicMuted,
    setVideoEnabled,
    addVideo,
    removeVideo,
    switchVideoDevice,
    startScreenShare,
    stopScreenShare,
    _runReceiverFlow,
    get state() { return state; },
    get callId() { return callId; },
    get peerDid() { return callerDid; },
    get modalities() { return modalities; },
    get withVideo() {
      return modalities.split(',').map((s) => s.trim()).includes('video');
    },
    get localStream() { return localStream; },
    get remoteStream() { return remoteStream; },
    get lastError() { return lastError; },
    get videoDeviceId() { return activeVideoDeviceId; },
    get isScreenSharing() { return screenStream !== null; },
  };
}
