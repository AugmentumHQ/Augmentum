import {
  getSurfaceReceiverUrl,
  normalizeSurfaceHandoff,
} from './surface-handoff.js?v=surface-handoff-20260512a';

export const SURFACE_CAST_NAMESPACE = 'urn:x-cast:com.augmentum.surface';
export const SURFACE_CAST_APP_ID_STORAGE_KEY = 'augmentum_cast_receiver_app_id';

const CAST_SENDER_URL = 'https://www.gstatic.com/cv/js/sender/v1/cast_sender.js?loadCastFramework=1';

let _castSdkPromise = null;
let _configuredAppId = '';

function _string(value) {
  return String(value || '').trim();
}

function _castErrorMessage(error) {
  const code = _string(error?.code || error?.errorCode || error?.description);
  const message = _string(error?.message || error);
  if (code === 'cancel') return 'Cast device selection was cancelled';
  if (code === 'timeout') return 'Cast device selection timed out';
  if (code === 'receiver_unavailable') return 'Cast receiver is unavailable';
  return message || code || 'Cast handoff failed';
}

export function getSurfaceCastAppId({ receiverApplicationId = '' } = {}) {
  const direct = _string(receiverApplicationId);
  if (direct) return direct;

  const globalId = _string(globalThis.AUGMENTUM_CAST_RECEIVER_APP_ID);
  if (globalId) return globalId;

  const metaId = _string(document.querySelector('meta[name="augmentum-cast-app-id"]')?.content);
  if (metaId) return metaId;

  try {
    return _string(localStorage.getItem(SURFACE_CAST_APP_ID_STORAGE_KEY));
  } catch {
    return '';
  }
}

export function setSurfaceCastAppId(appId) {
  const value = _string(appId);
  try {
    if (value) localStorage.setItem(SURFACE_CAST_APP_ID_STORAGE_KEY, value);
    else localStorage.removeItem(SURFACE_CAST_APP_ID_STORAGE_KEY);
  } catch { /* private browsing / quota — return value still drives in-memory state */ }
  return value;
}

export function surfaceCastConfigured(options = {}) {
  return !!getSurfaceCastAppId(options);
}

export function surfaceCastLikelyAvailable(options = {}) {
  return !!(
    globalThis.isSecureContext
    && surfaceCastConfigured(options)
    && typeof document !== 'undefined'
  );
}

export async function loadSurfaceCastSenderSdk({ timeoutMs = 10000 } = {}) {
  if (typeof window === 'undefined') {
    throw new Error('Google Cast is only available in a browser');
  }
  if (window.cast?.framework && window.chrome?.cast) return window.cast.framework;
  if (_castSdkPromise) return _castSdkPromise;

  _castSdkPromise = new Promise((resolve, reject) => {
    let settled = false;
    let timeout = null;
    const previous = window.__onGCastApiAvailable;

    const cleanup = () => {
      if (timeout) clearTimeout(timeout);
      if (window.__onGCastApiAvailable === onAvailable) {
        window.__onGCastApiAvailable = previous;
      }
    };

    const finish = (error = null) => {
      if (settled) return;
      settled = true;
      cleanup();
      if (error) {
        _castSdkPromise = null;
        reject(error);
        return;
      }
      resolve(window.cast.framework);
    };

    function onAvailable(available, ...rest) {
      try {
        if (typeof previous === 'function') previous(available, ...rest);
      } catch { /* prior handler threw — our resolve below is independent */ }
      if (available) finish();
      else finish(new Error('Google Cast SDK is unavailable'));
    }

    window.__onGCastApiAvailable = onAvailable;

    let script = document.querySelector('script[data-augmentum-cast-sdk="1"]');
    if (!script) {
      script = document.createElement('script');
      script.src = CAST_SENDER_URL;
      script.async = true;
      script.defer = true;
      script.dataset.augmentumCastSdk = '1';
      script.addEventListener('error', () => finish(new Error('Failed to load Google Cast SDK')));
      document.head.appendChild(script);
    }

    timeout = window.setTimeout(() => {
      if (window.cast?.framework && window.chrome?.cast) finish();
      else finish(new Error('Google Cast SDK did not become available'));
    }, Math.max(1000, Number(timeoutMs) || 10000));
  });

  return _castSdkPromise;
}

export async function configureSurfaceCastContext(options = {}) {
  if (!globalThis.isSecureContext) {
    throw new Error('Google Cast requires HTTPS or localhost');
  }
  const receiverApplicationId = getSurfaceCastAppId(options);
  if (!receiverApplicationId) {
    throw new Error('Surface Cast receiver app id is not configured');
  }
  await loadSurfaceCastSenderSdk(options);
  const context = window.cast?.framework?.CastContext?.getInstance?.();
  if (!context || !window.chrome?.cast) {
    throw new Error('Google Cast framework is unavailable');
  }
  if (_configuredAppId !== receiverApplicationId) {
    context.setOptions({
      receiverApplicationId,
      autoJoinPolicy: window.chrome.cast.AutoJoinPolicy?.ORIGIN_SCOPED,
      language: navigator.language || 'en-US',
    });
    _configuredAppId = receiverApplicationId;
  }
  return context;
}

export async function requestSurfaceCastSession(options = {}) {
  const context = await configureSurfaceCastContext(options);
  let session = context.getCurrentSession?.() || null;
  if (session) return session;
  await context.requestSession();
  session = context.getCurrentSession?.() || null;
  if (!session) throw new Error('No Cast session was started');
  return session;
}

function _castHandoffMessage(handoff, options = {}) {
  const normalized = normalizeSurfaceHandoff(handoff);
  const receiverUrl = getSurfaceReceiverUrl(normalized);
  return {
    v: 1,
    type: 'augmentum.surface.cast_handoff',
    sent_at: new Date().toISOString(),
    label: _string(options.label || 'Augmentum Surface'),
    receiver_url: receiverUrl,
    handoff: {
      version: normalized.version || 'augmentum.surface.handoff@1',
      transport: 'cast',
      handoff_id: normalized.handoff_id || '',
      ble_payload: normalized.ble_payload || {},
      ip: normalized.ip || {},
    },
  };
}

export async function sendHandoffOverCast(handoff, options = {}) {
  const namespace = _string(options.namespace || SURFACE_CAST_NAMESPACE);
  const session = options.session || await requestSurfaceCastSession(options);
  const message = _castHandoffMessage(handoff, options);
  if (typeof session.sendMessage !== 'function') {
    throw new Error('Cast session does not support custom messages');
  }
  try {
    await session.sendMessage(namespace, message);
  } catch (error) {
    const err = new Error(_castErrorMessage(error));
    err.cause = error;
    throw err;
  }
  const sessionObj = session.getSessionObj?.() || {};
  return {
    ok: true,
    transport: 'cast',
    namespace,
    receiverName: _string(
      sessionObj.receiver?.friendlyName
      || sessionObj.receiver?.label
      || 'Cast device',
    ),
    receiverUrl: message.receiver_url,
  };
}

export function installSurfaceCastGlobals(target = globalThis) {
  target.AugmentumSurfaceCast = {
    SURFACE_CAST_NAMESPACE,
    SURFACE_CAST_APP_ID_STORAGE_KEY,
    configureSurfaceCastContext,
    getSurfaceCastAppId,
    installSurfaceCastGlobals,
    loadSurfaceCastSenderSdk,
    requestSurfaceCastSession,
    sendHandoffOverCast,
    setSurfaceCastAppId,
    surfaceCastConfigured,
    surfaceCastLikelyAvailable,
  };
  return target.AugmentumSurfaceCast;
}

installSurfaceCastGlobals();
