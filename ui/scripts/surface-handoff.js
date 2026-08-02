const DEFAULT_HANDOFF_TTL_S = 6 * 60 * 60;
const DEFAULT_BLUETOOTH_MTU = 185;
const MIN_BLUETOOTH_CHUNK_BYTES = 20;
const MAX_BLUETOOTH_CHUNK_BYTES = 512;
const BLUETOOTH_BLOCKED_KEY = 'augmentum_surface_bluetooth_blocked_v1';
export const SURFACE_HANDOFF_SERVICE_UUID = '9b7e0001-4d8f-4f42-9a7a-6f675f000001';
export const SURFACE_HANDOFF_PAYLOAD_CHARACTERISTIC_UUID = '9b7e0002-4d8f-4f42-9a7a-6f675f000001';

function _isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function _normalizeHandoff(value) {
  return value?.handoff || value || {};
}

async function _readResponse(response) {
  const text = await response.text().catch(() => '');
  if (!text) return { data: null, text: '' };
  try {
    return { data: JSON.parse(text), text };
  } catch {
    return { data: null, text };
  }
}

async function _apiJson(url, options = {}) {
  const response = await fetch(url, {
    cache: 'no-store',
    credentials: 'same-origin',
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
  });
  const { data, text } = await _readResponse(response);
  if (!response.ok) {
    const surfaceMissing = response.status === 404 && String(url).startsWith('/api/surfaces');
    const message = surfaceMissing
      ? 'Surface API is not available at this origin. Restart Augmentum/Caddy or confirm /api is proxied to the current backend.'
      : (data?.error || data?.detail || response.statusText || 'Request failed');
    const error = new Error(message);
    error.status = response.status;
    error.data = data;
    error.url = response.url || url;
    error.body = text.slice(0, 500);
    throw error;
  }
  return data || {};
}

function _number(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function _clamp01(value) {
  return Math.max(0, Math.min(1, _number(value, 0)));
}

function _comicTitle(file, title = '') {
  const meta = file?.source_metadata || {};
  const extra = meta.extra || {};
  return String(title || extra.series_name || file?.name || 'Comic').trim();
}

function _comicSubtitle(file) {
  const extra = file?.source_metadata?.extra || {};
  if (extra.chapter_name) return String(extra.chapter_name);
  if (extra.chapter_number != null) return `Chapter ${extra.chapter_number}`;
  if (extra.volume != null) return `Volume ${extra.volume}`;
  return '';
}

function _participantId(prefix = 'surface') {
  const cryptoObj = globalThis.crypto;
  if (cryptoObj?.randomUUID) return `${prefix}-${cryptoObj.randomUUID()}`;
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export function normalizeSurfaceHandoff(value) {
  return _normalizeHandoff(value);
}

export async function createSurfaceSession({
  kind = 'surface.generic',
  title = '',
  contentRef = {},
  state = {},
  participants = [],
} = {}) {
  return _apiJson('/api/surfaces/sessions', {
    method: 'POST',
    body: JSON.stringify({
      kind,
      title,
      content_ref: _isObject(contentRef) ? contentRef : {},
      state: _isObject(state) ? state : {},
      participants: Array.isArray(participants) ? participants : [],
    }),
  });
}

export async function getSurfaceApiStatus() {
  const response = await fetch('/api/surfaces/recipes', {
    cache: 'no-store',
    credentials: 'same-origin',
  });
  const { data, text } = await _readResponse(response);
  return {
    ok: response.ok,
    status: response.status,
    statusText: response.statusText,
    url: response.url,
    body: data || text.slice(0, 500),
  };
}

export async function issueSurfaceAccessToken(sessionId, {
  fileId = '',
  scopes = [],
  ttlS = DEFAULT_HANDOFF_TTL_S,
  lockToClient = false,
  queryParams = {},
} = {}) {
  if (!sessionId) throw new Error('Surface session id is required');
  return _apiJson(`/api/surfaces/sessions/${encodeURIComponent(sessionId)}/access-token`, {
    method: 'POST',
    body: JSON.stringify({
      file_id: String(fileId || ''),
      scopes: Array.isArray(scopes) ? scopes : [],
      ttl_s: Math.max(30, Math.round(_number(ttlS, DEFAULT_HANDOFF_TTL_S))),
      lock_to_client: !!lockToClient,
      query_params: _isObject(queryParams) ? queryParams : {},
    }),
  });
}

export async function issueSurfaceHandoff(sessionId, {
  fileId = '',
  scopes = [],
  ttlS = DEFAULT_HANDOFF_TTL_S,
  targetRole = 'display',
  targetLabel = '',
  targetIp = '',
  targetCapabilities = [],
  bluetoothMtu = DEFAULT_BLUETOOTH_MTU,
  queryParams = {},
} = {}) {
  if (!sessionId) throw new Error('Surface session id is required');
  return _apiJson(`/api/surfaces/sessions/${encodeURIComponent(sessionId)}/handoff`, {
    method: 'POST',
    body: JSON.stringify({
      file_id: String(fileId || ''),
      scopes: Array.isArray(scopes) ? scopes : [],
      ttl_s: Math.max(30, Math.round(_number(ttlS, DEFAULT_HANDOFF_TTL_S))),
      lock_to_client: false,
      query_params: _isObject(queryParams) ? queryParams : {},
      target_role: String(targetRole || 'display'),
      target_label: String(targetLabel || ''),
      target_ip: String(targetIp || ''),
      target_capabilities: Array.isArray(targetCapabilities) ? targetCapabilities : [],
      bluetooth_mtu: Math.round(_number(bluetoothMtu, DEFAULT_BLUETOOTH_MTU)),
    }),
  });
}

export async function patchSurfaceState(sessionId, {
  patch = {},
  baseRevision = null,
  sourceParticipantId = '',
  replace = false,
} = {}) {
  if (!sessionId) throw new Error('Surface session id is required');
  const body = {
    patch: _isObject(patch) ? patch : {},
    source_participant_id: String(sourceParticipantId || ''),
    replace: !!replace,
  };
  if (baseRevision != null) body.base_revision = Math.max(0, Math.round(_number(baseRevision, 0)));
  return _apiJson(`/api/surfaces/sessions/${encodeURIComponent(sessionId)}/state`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export async function createComicSurfaceHandoff({
  file = null,
  fileId = '',
  title = '',
  page = 1,
  pageCount = 0,
  scrollRatio = 0,
  mode = 'webtoon',
  direction = 'ltr',
  targetLabel = 'TV',
  targetIp = '',
  bluetoothMtu = DEFAULT_BLUETOOTH_MTU,
  participantId = '',
} = {}) {
  const resolvedFileId = String(fileId || file?.id || '').trim();
  if (!resolvedFileId) throw new Error('Comic file id is required');

  const controllerId = String(participantId || _participantId('comic-phone')).trim();
  const resolvedTitle = _comicTitle(file, title);
  const contentRef = {
    kind: 'comic',
    file_id: resolvedFileId,
    title: resolvedTitle,
    subtitle: _comicSubtitle(file),
    provider: String(file?.source_metadata?.provider || ''),
  };
  const state = {
    reader: {
      file_id: resolvedFileId,
      page: Math.max(1, Math.round(_number(page, 1))),
      page_count: Math.max(0, Math.round(_number(pageCount, 0))),
      scroll_ratio: _clamp01(scrollRatio),
      mode: String(mode || 'webtoon'),
      direction: String(direction || 'ltr'),
    },
  };
  const sessionResult = await createSurfaceSession({
    kind: 'comic.reader.webtoon',
    title: resolvedTitle,
    contentRef,
    state,
    participants: [{
      participant_id: controllerId,
      role: 'controller',
      label: 'Phone',
      capabilities: ['surface.follow_state@1', 'input.touch_scroll@1'],
      transport: 'browser',
      metadata: { surface: 'comic.reader' },
    }],
  });
  const session = sessionResult.session || {};
  const handoffResult = await issueSurfaceHandoff(session.id, {
    fileId: resolvedFileId,
    targetRole: 'display',
    targetLabel,
    targetIp,
    targetCapabilities: ['surface.follow_state@1', 'display.comic_read@1'],
    bluetoothMtu,
  });
  return {
    participantId: controllerId,
    session: handoffResult.session || session,
    handoff: handoffResult.handoff,
  };
}

export function getSurfaceReceiverUrl(handoff) {
  const normalized = _normalizeHandoff(handoff);
  return String(normalized?.ip?.receiver_url || normalized?.ble_payload?.receiver_url || '');
}

export function bluetoothHandoffAvailable() {
  try {
    if (sessionStorage.getItem(BLUETOOTH_BLOCKED_KEY) === '1') return false;
  } catch { /* sessionStorage unavailable — fall through to capability check */ }
  return !!(globalThis.isSecureContext && globalThis.navigator?.bluetooth?.requestDevice);
}

export function isBluetoothHandoffBlockedError(error) {
  const name = String(error?.name || '');
  const message = String(error?.message || error || '').toLowerCase();
  return name === 'SecurityError'
    || (name === 'NotFoundError' && message.includes('globally disabled'))
    || message.includes('bluetooth permission has been blocked')
    || message.includes('web bluetooth api globally disabled')
    || message.includes('blocked by permissions policy');
}

export function rememberBluetoothHandoffBlocked() {
  try { sessionStorage.setItem(BLUETOOTH_BLOCKED_KEY, '1'); } catch {}
}

export function splitBluetoothPayload(payloadJson, mtu = DEFAULT_BLUETOOTH_MTU) {
  const bytes = new TextEncoder().encode(String(payloadJson || ''));
  const rawMtu = Math.round(_number(mtu, DEFAULT_BLUETOOTH_MTU));
  const maxBytes = Math.max(
    MIN_BLUETOOTH_CHUNK_BYTES,
    Math.min(MAX_BLUETOOTH_CHUNK_BYTES, rawMtu - 3),
  );
  const chunks = [];
  for (let offset = 0; offset < bytes.length; offset += maxBytes) {
    chunks.push(bytes.slice(offset, offset + maxBytes));
  }
  return chunks.length ? chunks : [new Uint8Array()];
}

async function _writeCharacteristic(characteristic, chunk) {
  if (typeof characteristic.writeValueWithResponse === 'function') {
    await characteristic.writeValueWithResponse(chunk);
    return;
  }
  if (typeof characteristic.writeValue === 'function') {
    await characteristic.writeValue(chunk);
    return;
  }
  if (typeof characteristic.writeValueWithoutResponse === 'function') {
    await characteristic.writeValueWithoutResponse(chunk);
    return;
  }
  throw new Error('Bluetooth characteristic is not writable');
}

function _bluetoothRequestOptions({
  serviceUuid = SURFACE_HANDOFF_SERVICE_UUID,
  namePrefix = 'Augmentum',
  acceptAllDevices = false,
} = {}) {
  const service = String(serviceUuid || SURFACE_HANDOFF_SERVICE_UUID);
  if (acceptAllDevices) return { acceptAllDevices: true, optionalServices: [service] };
  return {
    filters: [{ services: [service] }, { namePrefix: String(namePrefix || 'Augmentum') }],
    optionalServices: [service],
  };
}

export async function requestSurfaceBluetoothTarget({
  serviceUuid = SURFACE_HANDOFF_SERVICE_UUID,
  characteristicUuid = SURFACE_HANDOFF_PAYLOAD_CHARACTERISTIC_UUID,
  namePrefix = 'Augmentum',
  acceptAllDevices = false,
} = {}) {
  if (!bluetoothHandoffAvailable()) {
    throw new Error('Web Bluetooth requires a secure Chrome or Edge browser and a user gesture');
  }
  const device = await navigator.bluetooth.requestDevice(_bluetoothRequestOptions({
    serviceUuid,
    namePrefix,
    acceptAllDevices,
  }));
  const server = await device.gatt.connect();
  const service = await server.getPrimaryService(serviceUuid);
  const characteristic = await service.getCharacteristic(characteristicUuid);
  return {
    device,
    service,
    characteristic,
    deviceName: device.name || '',
    disconnect() {
      if (device.gatt?.connected) device.gatt.disconnect();
    },
  };
}

export async function sendHandoffOverBluetooth(handoff, {
  namePrefix = 'Augmentum',
  acceptAllDevices = false,
  keepConnected = false,
  onProgress = null,
  target = null,
} = {}) {
  if (!target && !bluetoothHandoffAvailable()) {
    throw new Error('Web Bluetooth requires a secure Chrome or Edge browser and a user gesture');
  }
  const normalized = _normalizeHandoff(handoff);
  const bluetooth = normalized.bluetooth || {};
  const serviceUuid = String(bluetooth.service_uuid || '').trim();
  const characteristicUuid = String(bluetooth.payload_characteristic_uuid || '').trim();
  if (!serviceUuid || !characteristicUuid) {
    throw new Error('Handoff does not include Bluetooth service details');
  }
  const payloadJson = String(normalized.ble_payload_json || JSON.stringify(normalized.ble_payload || {}));
  const chunks = splitBluetoothPayload(payloadJson, bluetooth.mtu || DEFAULT_BLUETOOTH_MTU);
  let device = target?.device || null;
  let characteristic = target?.characteristic || null;
  let ownsConnection = false;
  if (!characteristic) {
    const requestOptions = _bluetoothRequestOptions({ serviceUuid, namePrefix, acceptAllDevices });
    device = await navigator.bluetooth.requestDevice(requestOptions);
    const server = await device.gatt.connect();
    const service = await server.getPrimaryService(serviceUuid);
    characteristic = await service.getCharacteristic(characteristicUuid);
    ownsConnection = true;
  }
  try {
    for (let index = 0; index < chunks.length; index += 1) {
      await _writeCharacteristic(characteristic, chunks[index]);
      if (typeof onProgress === 'function') {
        onProgress({ index: index + 1, total: chunks.length, bytes: chunks[index].byteLength });
      }
    }
  } finally {
    if (!keepConnected) {
      if (ownsConnection && device?.gatt?.connected) device.gatt.disconnect();
      else if (!ownsConnection) target?.disconnect?.();
    }
  }
  return {
    ok: true,
    deviceName: device?.name || target?.deviceName || '',
    chunks: chunks.length,
    bytes: new TextEncoder().encode(payloadJson).byteLength,
  };
}

export async function copySurfaceReceiverUrl(handoff) {
  const url = getSurfaceReceiverUrl(handoff);
  if (!url) throw new Error('Handoff does not include a receiver URL');
  if (!navigator.clipboard?.writeText) throw new Error('Clipboard is unavailable');
  await navigator.clipboard.writeText(url);
  return url;
}

export function openSurfaceReceiver(handoff, { target = '_blank' } = {}) {
  const url = getSurfaceReceiverUrl(handoff);
  if (!url) throw new Error('Handoff does not include a receiver URL');
  globalThis.open?.(url, target, 'noopener');
  return url;
}

export function installSurfaceHandoffGlobals(target = globalThis) {
  target.AugmentumSurfaceHandoff = {
    createSurfaceSession,
    createComicSurfaceHandoff,
    getSurfaceApiStatus,
    issueSurfaceAccessToken,
    issueSurfaceHandoff,
    patchSurfaceState,
    bluetoothHandoffAvailable,
    isBluetoothHandoffBlockedError,
    rememberBluetoothHandoffBlocked,
    requestSurfaceBluetoothTarget,
    sendHandoffOverBluetooth,
    splitBluetoothPayload,
    copySurfaceReceiverUrl,
    getSurfaceReceiverUrl,
    openSurfaceReceiver,
    normalizeSurfaceHandoff,
    SURFACE_HANDOFF_SERVICE_UUID,
    SURFACE_HANDOFF_PAYLOAD_CHARACTERISTIC_UUID,
  };
  return target.AugmentumSurfaceHandoff;
}

installSurfaceHandoffGlobals();
