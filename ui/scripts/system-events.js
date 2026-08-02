/**
 * System events client.
 *
 * Opens a single EventSource to /api/system/events at app boot and
 * re-dispatches every server event as a DOM CustomEvent on `window`
 * with the name `system-event:<topic>` and `detail = { topic, data, ts, id }`.
 *
 * Subscribers in feature modules attach with:
 *   window.addEventListener('system-event:providers.added', (e) => {
 *     // e.detail.data carries the payload published server-side
 *   });
 *
 * Reconnect is automatic (EventSource handles it natively). We don't
 * replay history on reconnect — subscribers should re-fetch the
 * affected collection (provider list, model catalog, etc.) on event
 * receipt as the source of truth.
 */

let _source = null;
let _backoff = 1000;
const _BACKOFF_MAX = 30_000;

function _connect() {
  if (_source) return;
  try {
    _source = new EventSource('/api/system/events');
  } catch (err) {
    console.warn('[system-events] EventSource construct failed', err);
    _scheduleReconnect();
    return;
  }

  _source.addEventListener('open', () => {
    _backoff = 1000;
  });

  _source.addEventListener('error', () => {
    // EventSource auto-retries by default, but if the browser gives up
    // (state === CLOSED) we re-open with backoff.
    if (_source && _source.readyState === EventSource.CLOSED) {
      _source = null;
      _scheduleReconnect();
    }
  });

  _source.addEventListener('message', _dispatch);

  // Named-event handlers register lazily on first call.
  const wrapped = new Set();
  const orig = _source.addEventListener.bind(_source);
  _source.addEventListener = (type, fn, opts) => {
    if (type !== 'message' && type !== 'open' && type !== 'error' && !wrapped.has(type)) {
      wrapped.add(type);
      orig(type, _dispatch);
    }
    return orig(type, fn, opts);
  };

  // Pre-register handlers for known topic prefixes so the underlying
  // EventSource routes them. Server emits `event: <topic>` so we listen
  // per-topic; unknown topics still surface via the default 'message'
  // handler below.
  for (const topic of [
    'providers.added', 'providers.updated', 'providers.deleted',
    'models.installed', 'models.install_failed', 'models.changed',
    'voices.changed',
    'image_providers.added', 'image_providers.updated', 'image_providers.deleted',
    'image.models.changed',
    'characters.changed', 'personas.changed',
    'knowledge.changed',
    'sessions.changed',
  ]) {
    _source.addEventListener(topic, _dispatch);
  }
}

function _dispatch(ev) {
  if (!ev || !ev.data) return;
  let payload;
  try {
    payload = JSON.parse(ev.data);
  } catch {
    return;
  }
  const topic = payload.topic || ev.type || 'message';
  window.dispatchEvent(new CustomEvent(`system-event:${topic}`, {
    detail: payload,
  }));
}

function _scheduleReconnect() {
  setTimeout(() => {
    _backoff = Math.min(_backoff * 2, _BACKOFF_MAX);
    _connect();
  }, _backoff);
}

_connect();

export function close() {
  if (_source) {
    _source.close();
    _source = null;
  }
}
