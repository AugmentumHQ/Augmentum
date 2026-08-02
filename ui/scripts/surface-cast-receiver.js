import {
  startWithSurfaceHandoff,
  startWithSurfaceToken,
} from './surface-receiver.js';
import { SURFACE_CAST_NAMESPACE } from './surface-cast.js?v=surface-handoff-20260512a';

function _parseMessage(data) {
  if (typeof data === 'string') {
    try {
      return JSON.parse(data);
    } catch {
      return {};
    }
  }
  return data && typeof data === 'object' ? data : {};
}

function _handoffFromMessage(message) {
  if (message?.type === 'augmentum.surface.cast_handoff') {
    return message.handoff || message;
  }
  if (message?.type === 'augmentum.surface.handoff') {
    return message;
  }
  return null;
}

function _sendAck(context, senderId, body) {
  try {
    context.sendCustomMessage(SURFACE_CAST_NAMESPACE, senderId, {
      type: 'augmentum.surface.cast_ack',
      v: 1,
      ...body,
    });
  } catch { /* sender disconnected mid-ack — message is best-effort */ }
}

async function _handleCastMessage(context, event) {
  const message = _parseMessage(event?.data);
  const handoff = _handoffFromMessage(message);
  if (!handoff) {
    _sendAck(context, event?.senderId, { ok: false, error: 'unsupported_message' });
    return;
  }
  try {
    await startWithSurfaceHandoff(handoff, { source: 'cast' });
    _sendAck(context, event?.senderId, {
      ok: true,
      session_id: handoff?.ble_payload?.session_id || '',
      receiver_url: message?.receiver_url || handoff?.ip?.receiver_url || '',
    });
  } catch (error) {
    _sendAck(context, event?.senderId, {
      ok: false,
      error: String(error?.message || error || 'cast_handoff_failed'),
    });
  }
}

function startCastReceiver() {
  const framework = window.cast?.framework;
  const context = framework?.CastReceiverContext?.getInstance?.();
  if (!context) {
    startWithSurfaceToken('', { source: 'cast' });
    return;
  }
  context.addCustomMessageListener(SURFACE_CAST_NAMESPACE, (event) => {
    _handleCastMessage(context, event);
  });
  context.start({
    disableIdleTimeout: true,
  });
}

startCastReceiver();
