/**
 * library/api.js — typed fetch helpers for /api/library/*.
 *
 * Every helper returns parsed JSON or throws. Routes are user-scoped
 * server-side, so the client is unauthenticated-looking on purpose;
 * the session cookie carries auth.
 *
 * Item-returning helpers stamp ``_type`` (and ``_isPublication``) via
 * ``classifyItem`` so downstream surfaces — the detail-pane open
 * dispatcher, menu actions, the cast launch path — can route to the
 * right viewer without re-classifying. Mirrors legacy library.js's
 * single classification pass.
 */

import { classifyItem, classifyItems } from './types.js';

async function _fetchJSON(url, init = {}) {
  const resp = await fetch(url, {
    credentials: 'same-origin',
    headers: { 'Accept': 'application/json', ...(init.headers || {}) },
    ...init,
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(`${init.method || 'GET'} ${url} -> ${resp.status} ${text}`);
  }
  return resp.json();
}

function _postJSON(url, body) {
  return _fetchJSON(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
}

function _putJSON(url, body) {
  return _fetchJSON(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
}

// ── Home dashboard ───────────────────────────────────────────────────

export async function fetchHome() {
  const payload = await _fetchJSON('/api/library/home');
  // Each items-bearing section gets classified so the sidebar /
  // main-pane can dispatch by ``_type`` without an extra pass.
  classifyItems(payload?.pinned);
  classifyItems(payload?.recent);
  classifyItems(payload?.continue);
  return payload;
}

// ── Items (filtered list) ────────────────────────────────────────────

export async function fetchItems({
  types = [], q = '', pinned = false, sort = 'recent',
  limit = 60, offset = 0,
} = {}) {
  const params = new URLSearchParams();
  if (types.length) params.set('types', types.join(','));
  if (q) params.set('q', q);
  if (pinned) params.set('pinned', '1');
  if (sort) params.set('sort', sort);
  params.set('limit', String(limit));
  params.set('offset', String(offset));
  const payload = await _fetchJSON(`/api/library/items?${params.toString()}`);
  classifyItems(payload?.items);
  return payload;
}

// ── Collections ──────────────────────────────────────────────────────

export function listCollections() {
  return _fetchJSON('/api/library/collections');
}

export function createCollection(body) {
  return _postJSON('/api/library/collections', body);
}

export async function getCollection(id) {
  const col = await _fetchJSON(`/api/library/collections/${encodeURIComponent(id)}`);
  classifyItems(col?.items);
  return col;
}

export function updateCollection(id, patch) {
  return _putJSON(
    `/api/library/collections/${encodeURIComponent(id)}`,
    patch,
  );
}

export function deleteCollection(id) {
  return _fetchJSON(
    `/api/library/collections/${encodeURIComponent(id)}`,
    { method: 'DELETE' },
  );
}

export function addCollectionItems(id, artifactIds) {
  return _postJSON(
    `/api/library/collections/${encodeURIComponent(id)}/items`,
    { artifact_ids: artifactIds },
  );
}

export function removeCollectionItem(id, artifactId) {
  return _fetchJSON(
    `/api/library/collections/${encodeURIComponent(id)}/items/`
      + encodeURIComponent(artifactId),
    { method: 'DELETE' },
  );
}

// ── Per-item ─────────────────────────────────────────────────────────

export function setPin(artifactId, pinned) {
  return _postJSON(
    `/api/library/items/${encodeURIComponent(artifactId)}/pin`,
    { pinned: Boolean(pinned) },
  );
}

export function recordActivity(artifactId, action, { surface = 'desktop', payload = {} } = {}) {
  return _postJSON(
    `/api/library/items/${encodeURIComponent(artifactId)}/activity`,
    { action, surface, payload },
  );
}

export function listActivity(artifactId) {
  return _fetchJSON(
    `/api/library/items/${encodeURIComponent(artifactId)}/activity`,
  );
}

export function setTags(artifactId, tags) {
  return _putJSON(
    `/api/library/items/${encodeURIComponent(artifactId)}/tags`,
    { tags },
  );
}

// ── Direct artifact operations (mirrors legacy library.js) ───────────

export async function importArtifact(file) {
  const form = new FormData();
  form.append('file', file);
  const resp = await fetch('/api/artifacts/import', {
    method: 'POST',
    body: form,
    credentials: 'same-origin',
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `Import failed: ${resp.status}`);
  }
  // Classify so the import-handler auto-open step (commit 4) can
  // dispatch by ``_type`` the same way the open dispatcher does.
  return classifyItem(await resp.json());
}

export async function deleteArtifact(artifactId) {
  const resp = await fetch(
    `/api/artifacts/${encodeURIComponent(artifactId)}`,
    { method: 'DELETE', credentials: 'same-origin' },
  );
  if (!resp.ok && resp.status !== 404) {
    throw new Error(`Delete failed: ${resp.status}`);
  }
  return true;
}

// Delete a library item by id, routing to the right namespace: publications
// (pub_ ids) go to the publications route; everything else to the artifact
// route. Used by bulk delete so mixed selections don't 404 the pub_ rows.
export async function deleteLibraryItem(itemId) {
  if (typeof itemId === 'string' && itemId.startsWith('pub_')) {
    const resp = await fetch(
      `/api/library/publications/${encodeURIComponent(itemId)}`,
      { method: 'DELETE', credentials: 'same-origin' },
    );
    if (!resp.ok && resp.status !== 404) {
      throw new Error(`Delete failed: ${resp.status}`);
    }
    return true;
  }
  return deleteArtifact(itemId);
}
