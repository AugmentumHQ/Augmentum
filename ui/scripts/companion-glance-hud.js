/**
 * companion-glance-hud.js
 * Sprint 5 — minimal at-a-glance HUD over the runtime's presence bus.
 *
 * Renders a small DOM overlay (works on flat and in dom-overlay XR
 * mode) showing:
 *   - state · role · focus (one line, glanceable)
 *   - tick activity (badge when behavior.activity_chosen fires)
 *   - dispatch decision (last winner subagent)
 *
 * The HUD does NOT call into the binding; it subscribes to the bus
 * itself so the two are independent. If the runtime is off or the WS
 * fails, the HUD shows "—" and gets out of the way.
 */

import { createReconnector } from './_ws-reconnect.js';

const POLL_MS = 200;            // poll cadence for the visible state line
const HOLD_BADGE_MS = 2200;     // how long a transient badge stays visible

function makeEl(tag, cls, text) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text != null) el.textContent = text;
  return el;
}

export class CompanionGlanceHUD {
  constructor(opts = {}) {
    this._wsUrl = opts.wsUrl || this._defaultWsUrl();
    this._root = null;
    this._stateLine = null;
    this._badge = null;
    this._ws = null;
    this._stopped = false;
    this._reconnector = null;
    this._badgeHideTimer = null;
    this._state = { state: '—', role_dominant: '—', focus: '—' };
  }

  _defaultWsUrl() {
    if (typeof location === 'undefined') return 'ws://localhost:8000/ws/companion/presence';
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${location.host}/ws/companion/presence?slice_key=glance_hud`;
  }

  attach(host) {
    const parent = host || document.body;
    if (this._root) return this._root;
    const root = makeEl('div', 'companion-glance-hud');
    root.setAttribute('aria-live', 'polite');
    root.style.cssText = [
      'position:fixed', 'right:12px', 'bottom:12px',
      'padding:6px 10px', 'border-radius:10px',
      'background:rgba(20,20,28,0.78)', 'color:#e8e8ee',
      'font:12px/1.3 system-ui,sans-serif',
      'pointer-events:none', 'z-index:2147483640',
      'backdrop-filter:blur(6px)',
    ].join(';');
    const stateLine = makeEl('div', 'cghud-state', '—');
    const badge = makeEl('div', 'cghud-badge', '');
    badge.style.cssText = 'margin-top:3px;opacity:0;transition:opacity 180ms ease-out;font-weight:500;';
    root.appendChild(stateLine);
    root.appendChild(badge);
    parent.appendChild(root);
    this._root = root;
    this._stateLine = stateLine;
    this._badge = badge;
    this._render();
    this._reconnector = createReconnector({
      connect: () => this._connect(),
      base: 1500, cap: 30000, name: 'glance_hud',
    });
    this._reconnector.start();
    return root;
  }

  detach() {
    this._stopped = true;
    if (this._reconnector) {
      try { this._reconnector.stop(); } catch (_) { /* ignore */ }
      this._reconnector = null;
    }
    if (this._ws) {
      try { this._ws.close(1000, 'hud_detach'); } catch (_) { /* ignore */ }
      this._ws = null;
    }
    if (this._badgeHideTimer) {
      clearTimeout(this._badgeHideTimer);
      this._badgeHideTimer = null;
    }
    if (this._root && this._root.parentNode) {
      this._root.parentNode.removeChild(this._root);
    }
    this._root = null;
  }

  _connect() {
    if (this._stopped) return null;
    let ws;
    try { ws = new WebSocket(this._wsUrl); }
    catch (e) { return null; }
    this._ws = ws;
    ws.addEventListener('message', (m) => {
      let evt;
      try { evt = JSON.parse(m.data); } catch (_) { return; }
      this._onEvent(evt);
    });
    ws.addEventListener('close', () => {
      this._ws = null;
      // Reconnector applies full-jitter backoff and resets on re-open.
      if (!this._stopped && this._reconnector) this._reconnector.schedule();
    });
    return ws;
  }

  _onEvent(evt) {
    if (!evt || !evt.topic) return;
    const p = evt.payload || {};
    if (evt.topic === 'state.transition') {
      this._state.state = p.to;
      this._render();
    } else if (evt.topic === 'role.transition') {
      this._state.role_dominant = p.to;
      this._render();
    } else if (evt.topic === 'focus.transition') {
      this._state.focus = (p.to && (p.to.value || p.to.kind)) || '—';
      this._render();
    } else if (evt.topic === 'behavior.activity_chosen') {
      this._showBadge(`▴ ${p.kind || 'activity'}`);
    } else if (evt.topic === 'dispatch.decided' && p.winner) {
      this._showBadge(`↦ ${p.winner}`);
    } else if (evt.topic === 'initiative.surfaced') {
      this._showBadge(`✦ ${p.kind || 'initiative'}`);
    }
  }

  _render() {
    if (!this._stateLine) return;
    const s = this._state;
    this._stateLine.textContent =
      `${s.state || '—'} · ${s.role_dominant || '—'} · ${s.focus || '—'}`;
  }

  _showBadge(text) {
    if (!this._badge) return;
    this._badge.textContent = text;
    this._badge.style.opacity = '1';
    if (this._badgeHideTimer) clearTimeout(this._badgeHideTimer);
    this._badgeHideTimer = setTimeout(() => {
      if (this._badge) this._badge.style.opacity = '0';
    }, HOLD_BADGE_MS);
  }
}

let _hud = null;

/** Lazy singleton. Pass a host element (defaults to document.body). */
export function getGlanceHUD(opts) {
  if (_hud) return _hud;
  _hud = new CompanionGlanceHUD(opts);
  return _hud;
}
