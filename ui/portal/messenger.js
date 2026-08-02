/* messenger.js — the portal's self-contained comms gateway.
 *
 * Text (verified against the connect protocol): load the host contact +
 * thread, render history, send via POST, receive live over the signaling
 * WS. Calls (voice/video) use browser-native WebRTC P2P over the same
 * signaling verbs — inherently end-to-end, no SDK. The portal reaches only
 * the host (the existing guest ACL enforces that server-side).
 *
 * No dependency on the main app's modules — the guest mini-app stays light.
 */
const CONNECT = '/api/connect';
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const uuid = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`);

async function api(path, opts = {}) {
  const res = await fetch(CONNECT + path, {
    method: opts.method || 'GET',
    headers: opts.body ? { 'Content-Type': 'application/json' } : undefined,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
    credentials: 'same-origin',
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.status === 204 ? null : res.json();
}

export class PortalComms {
  constructor(root) {
    this.root = root;
    this.ws = null;
    this.host = null;       // {peer_did, peer_display_name}
    this.threadId = null;
    this.messages = [];
    this.handlers = new Map();
    this.call = null;       // active RTCPeerConnection wrapper
  }

  async start() {
    await this._loadHost();
    await this._loadThread();
    await this._loadMessages();
    this._render();
    this._connectWs().catch(() => {/* messaging still works via send + reload */});
  }

  async _loadHost() {
    const { contacts = [] } = await api('/contacts');
    // A portal guest has exactly one contact: their host.
    this.host = contacts[0] || { peer_did: '', peer_display_name: 'your host' };
  }

  async _loadThread() {
    const { threads = [] } = await api('/threads');
    const t = threads.find((x) => x.peer_did === this.host.peer_did);
    this.threadId = t ? t.thread_id : uuid(); // first send creates it server-side
  }

  async _loadMessages() {
    try {
      const { messages = [] } = await api(`/threads/${encodeURIComponent(this.threadId)}/messages`);
      this.messages = messages;
    } catch { this.messages = []; }
  }

  async sendText(text) {
    const body = (text || '').trim();
    if (!body) return;
    const message_id = uuid();
    // optimistic
    this.messages.push({ message_id, body, sender_did: 'me', sent_at: new Date().toISOString() });
    this._renderMessages();
    try {
      await api(`/threads/${encodeURIComponent(this.threadId)}/send`, {
        method: 'POST', body: { peer_did: this.host.peer_did, body, message_id },
      });
    } catch {
      this._toast("Couldn't send — check your connection.");
    }
  }

  /* ── live signaling WS ──────────────────────────────────────────── */
  async _connectWs() {
    const tkRes = await fetch('/api/auth/ws-ticket', { method: 'POST', credentials: 'same-origin' });
    if (!tkRes.ok) throw new Error('ws-ticket');
    const { ticket } = await tkRes.json();
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    this.ws = new WebSocket(`${proto}://${location.host}/api/connect/signaling?ticket=${encodeURIComponent(ticket)}`);
    this.ws.onmessage = (evt) => {
      let p; try { p = JSON.parse(evt.data); } catch { return; }
      if (p.type !== 'event') return;
      this._onEvent(p.event, p.data || {}, p.from || '');
    };
    this.ws.onclose = () => { setTimeout(() => this._connectWs().catch(() => {}), 3000); };
  }

  _wsSend(verb, data, to) {
    if (!this.ws || this.ws.readyState !== 1) return;
    this.ws.send(JSON.stringify({ type: 'msg', msg: verb, id: uuid(), to, data }));
  }

  _onEvent(verb, data, from) {
    if (verb === 'text_received') {
      this.messages.push({ message_id: data.message_id, body: data.body, sender_did: from || data.sender_did });
      this._renderMessages();
    } else if (['invite', 'offer', 'answer', 'candidates', 'accept', 'hangup', 'decline'].includes(verb)) {
      this._onCallSignal(verb, data, from);
    }
    (this.handlers.get(verb) || []).forEach((fn) => fn(data, from));
  }

  /* ── WebRTC P2P call (voice/video) ──────────────────────────────── */
  async startCall(withVideo) {
    if (this.call) return;
    const callId = uuid();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: !!withVideo });
    const pc = this._newPeer(callId);
    stream.getTracks().forEach((t) => pc.addTrack(t, stream));
    this.call = { pc, callId, localStream: stream, video: !!withVideo };
    this._renderCall('calling');
    this._wsSend('invite', { call_id: callId, modalities: withVideo ? 'audio,video' : 'audio' }, this.host.peer_did);
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    this._wsSend('offer', { call_id: callId, sdp: offer.sdp }, this.host.peer_did);
  }

  _newPeer(callId) {
    const pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
    pc.onicecandidate = (e) => {
      if (e.candidate) this._wsSend('candidates', { call_id: callId, candidate: e.candidate.toJSON() }, this.host.peer_did);
    };
    pc.ontrack = (e) => {
      const el = this.root.querySelector('[data-remote-media]');
      if (el && e.streams[0]) { el.srcObject = e.streams[0]; el.play?.().catch(() => {}); }
    };
    pc.onconnectionstatechange = () => {
      if (['failed', 'disconnected', 'closed'].includes(pc.connectionState)) this.endCall();
    };
    return pc;
  }

  async _onCallSignal(verb, data) {
    if (verb === 'answer' && this.call) {
      await this.call.pc.setRemoteDescription({ type: 'answer', sdp: data.sdp });
      this._renderCall('connected');
    } else if (verb === 'candidates' && this.call && data.candidate) {
      try { await this.call.pc.addIceCandidate(data.candidate); } catch {/* ignore */}
    } else if (verb === 'hangup' || verb === 'decline') {
      this.endCall();
    }
  }

  endCall() {
    if (!this.call) return;
    this._wsSend('hangup', { call_id: this.call.callId }, this.host.peer_did);
    try { this.call.localStream.getTracks().forEach((t) => t.stop()); } catch {/* */}
    try { this.call.pc.close(); } catch {/* */}
    this.call = null;
    this._renderCall(null);
  }

  on(verb, fn) { const a = this.handlers.get(verb) || []; a.push(fn); this.handlers.set(verb, a); }

  /* ── render ─────────────────────────────────────────────────────── */
  _render() {
    this.root.innerHTML = `
      <div class="pm-head">
        <div class="pm-host">${esc(this.host.peer_display_name || 'your host')}</div>
        <div class="pm-actions">
          <button class="pm-ic" data-call-audio title="Voice call">📞</button>
          <button class="pm-ic" data-call-video title="Video call">🎥</button>
        </div>
      </div>
      <div class="pm-log" data-log></div>
      <form class="pm-compose" data-compose>
        <input data-input placeholder="Message…" autocomplete="off" />
        <button class="pm-send" type="submit">Send</button>
      </form>
      <div class="pm-call" data-call hidden>
        <video data-remote-media autoplay playsinline></video>
        <div class="pm-call-state" data-call-state></div>
        <button class="pm-hang" data-hang>End call</button>
      </div>`;
    this.root.querySelector('[data-compose]').addEventListener('submit', (e) => {
      e.preventDefault();
      const inp = this.root.querySelector('[data-input]');
      this.sendText(inp.value); inp.value = '';
    });
    this.root.querySelector('[data-call-audio]').onclick = () => this.startCall(false).catch((x) => this._toast(x.message));
    this.root.querySelector('[data-call-video]').onclick = () => this.startCall(true).catch((x) => this._toast(x.message));
    this.root.querySelector('[data-hang]').onclick = () => this.endCall();
    this._renderMessages();
  }

  _renderMessages() {
    const log = this.root.querySelector('[data-log]');
    if (!log) return;
    log.innerHTML = this.messages.map((m) => {
      const cls = (m.sender_did === 'me') ? 'pm-mine' : 'pm-theirs';
      return `<div class="pm-msg ${cls}">${esc(m.body || '')}</div>`;
    }).join('');
    log.scrollTop = log.scrollHeight;
  }

  _renderCall(state) {
    const box = this.root.querySelector('[data-call]');
    if (!box) return;
    box.hidden = !state;
    const s = this.root.querySelector('[data-call-state]');
    if (s) s.textContent = state === 'calling' ? 'Calling…' : state === 'connected' ? 'Connected' : '';
  }

  _toast(msg) {
    const t = document.createElement('div');
    t.className = 'pm-toast'; t.textContent = msg;
    this.root.appendChild(t);
    setTimeout(() => t.remove(), 3500);
  }
}
