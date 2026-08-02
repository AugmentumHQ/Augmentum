/**
 * xr-companion-binding.js
 * Sprint 5 — bridge the server-resident CompanionRuntime presence bus
 * to the XR scene's avatar FSM, spatial director, and locomotion path.
 *
 * The binding does NOT own animation or rendering; it owns the
 * mapping from `bus event → FSM transition / pose target`. Render
 * paths stay where they are (`avatar-fsm.js`, `avatar-spatial-director.js`,
 * `avatar-locomotion.js`, `avatar-presence.js`).
 *
 * Subscribes to `/ws/companion/presence` with topic glob `**`. Events
 * arrive as JSON with `{topic, payload, source_companion_id, t}`.
 * When `settings.companion_xr_orchestrator` is off, the binding is a
 * no-op and the existing XR scene runs exactly as it does today.
 *
 * Reconnects with exponential back-off capped at 30s. Designed to be
 * imported once at XR session start; multiple imports are idempotent.
 */

const RECONNECT_BASE_MS = 1000;
const RECONNECT_CAP_MS = 30000;

/**
 * Map a (runtime_state, runtime_role) tuple to an FSM target state.
 * The mapping is intentionally small — six FSM states cover Becca's
 * observable poses at the avatar level. Sprint 6+ may add more.
 */
function mapStateToFSM(runtimeState, runtimeRole) {
  if (runtimeState === 'asleep')   return 'SeatedDefault';
  if (runtimeState === 'dormant')  return 'SeatedDefault';
  if (runtimeState === 'present') {
    if (runtimeRole === 'companion')    return 'SeatedLeaning';
    if (runtimeRole === 'collaborator') return 'SeatedForward';
    if (runtimeRole === 'host')         return 'StandingIdle';
    if (runtimeRole === 'observer')     return 'SeatedBack';
    return 'SeatedDefault';
  }
  return null;
}

export class XRCompanionBinding {
  /**
   * @param {{
   *   wsUrl?: string,
   *   fsm: import('./avatar-fsm.js').AvatarFSM,
   *   director?: object,
   *   locomotion?: object,
   *   presence?: object,
   *   onEvent?: (evt: object) => void,
   * }} opts
   */
  constructor(opts) {
    if (!opts || !opts.fsm) {
      throw new Error('XRCompanionBinding: fsm is required');
    }
    this._wsUrl = opts.wsUrl || this._defaultWsUrl();
    this._fsm = opts.fsm;
    this._director = opts.director || null;
    this._locomotion = opts.locomotion || null;
    this._presence = opts.presence || null;
    this._onEvent = typeof opts.onEvent === 'function' ? opts.onEvent : () => {};

    this._ws = null;
    this._reconnectAttempts = 0;
    this._stopped = false;
    this._latestState = { state: '', role_dominant: '', focus: null };
    this._unsubFns = [];
  }

  _defaultWsUrl() {
    if (typeof location === 'undefined') return 'ws://localhost:8000/ws/companion/presence';
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${location.host}/ws/companion/presence`;
  }

  start() {
    if (this._stopped) return;
    this._connect();
  }

  stop() {
    this._stopped = true;
    for (const fn of this._unsubFns) {
      try { fn(); } catch (_) { /* ignore */ }
    }
    this._unsubFns = [];
    if (this._ws) {
      try { this._ws.close(1000, 'binding_stop'); } catch (_) { /* ignore */ }
      this._ws = null;
    }
  }

  _connect() {
    if (this._stopped) return;
    let ws;
    try {
      ws = new WebSocket(this._wsUrl);
    } catch (err) {
      console.warn('xr-companion-binding: WebSocket construction failed', err);
      this._scheduleReconnect();
      return;
    }
    this._ws = ws;

    ws.addEventListener('open', () => {
      this._reconnectAttempts = 0;
      console.info('xr-companion-binding: connected', this._wsUrl);
    });

    ws.addEventListener('message', (msg) => {
      let evt;
      try { evt = JSON.parse(msg.data); }
      catch (_) { return; }
      this._dispatch(evt);
    });

    ws.addEventListener('close', () => {
      this._ws = null;
      if (!this._stopped) this._scheduleReconnect();
    });

    ws.addEventListener('error', (err) => {
      console.debug('xr-companion-binding: ws error', err);
      // close handler will trigger reconnect
    });
  }

  _scheduleReconnect() {
    if (this._stopped) return;
    const attempt = ++this._reconnectAttempts;
    const delay = Math.min(RECONNECT_CAP_MS, RECONNECT_BASE_MS * Math.pow(2, attempt - 1));
    setTimeout(() => { if (!this._stopped) this._connect(); }, delay);
  }

  /**
   * Route one inbound event. Touched topics:
   *   - state.transition  → cache + possibly FSM transition
   *   - role.transition   → cache + possibly FSM transition
   *   - focus.transition  → cache; director may re-aim gaze
   *   - behavior.activity_chosen → pose nudge per activity kind
   *   - behavior.reach_out → StandingIdle + locomotion target
   *   - scene.location_changed → director.setLocation if available
   *   - dispatch.decided → debug log
   *
   * Unknown topics are ignored (no warnings — the bus is broad).
   */
  _dispatch(evt) {
    this._onEvent(evt);
    const topic = evt && evt.topic;
    if (!topic) return;
    const payload = evt.payload || {};

    switch (topic) {
      case 'state.transition': {
        this._latestState.state = payload.to;
        this._applyFSM();
        break;
      }
      case 'role.transition': {
        this._latestState.role_dominant = payload.to;
        this._applyFSM();
        break;
      }
      case 'focus.transition': {
        this._latestState.focus = payload.to;
        if (this._director && typeof this._director.setFocus === 'function') {
          try { this._director.setFocus(payload.to); } catch (e) {
            console.debug('director.setFocus failed', e);
          }
        }
        break;
      }
      case 'behavior.activity_chosen': {
        this._applyActivity(payload);
        break;
      }
      case 'behavior.reach_out': {
        try {
          this._fsm.requestTransition('StandingIdle');
        } catch (e) { /* ignore */ }
        if (this._locomotion && typeof this._locomotion.moveTowardUser === 'function') {
          try { this._locomotion.moveTowardUser(); } catch (_) { /* ignore */ }
        }
        break;
      }
      case 'scene.location_changed': {
        if (this._director && typeof this._director.setLocation === 'function') {
          try { this._director.setLocation(payload.location); } catch (_) { /* ignore */ }
        }
        break;
      }
      case 'dream.invoked':
      case 'runtime.stopping':
      case 'runtime.started':
      case 'dispatch.decided':
      case 'dispatch.tiebreaker':
      case 'initiative.surfaced':
        // Observability events. The caller's onEvent hook gets them
        // for HUD rendering; we don't drive the avatar from them.
        break;
      default:
        break;
    }
  }

  _applyFSM() {
    const target = mapStateToFSM(
      this._latestState.state,
      this._latestState.role_dominant,
    );
    if (!target) return;
    try {
      this._fsm.requestTransition(target);
    } catch (e) {
      console.debug('fsm.requestTransition failed', e);
    }
  }

  _applyActivity(payload) {
    const kind = payload && payload.kind;
    if (!kind) return;
    let target = null;
    if (kind === 'journal' || kind === 'observation') {
      target = 'SeatedForward';
    } else if (kind === 'creation') {
      target = 'SeatedLeaning';
    } else if (kind === 'scene_update') {
      target = 'Locomoting';
    } else if (kind === 'dream_invocation') {
      target = 'SeatedBack';
    }
    if (target) {
      try { this._fsm.requestTransition(target); }
      catch (_) { /* ignore */ }
    }
  }

  /** Diagnostics. Used by the Glance HUD. */
  snapshot() {
    return {
      connected: !!(this._ws && this._ws.readyState === 1),
      wsUrl: this._wsUrl,
      reconnectAttempts: this._reconnectAttempts,
      latest: { ...this._latestState },
    };
  }
}

let _singleton = null;

/**
 * Lazy singleton. Calling `getXRCompanionBinding()` more than once
 * returns the same instance. `start()` must be called explicitly —
 * the binding does not auto-start so the XR boot path can decide
 * when WebSocket connection makes sense.
 */
export function getXRCompanionBinding(opts) {
  if (_singleton) return _singleton;
  _singleton = new XRCompanionBinding(opts);
  return _singleton;
}
