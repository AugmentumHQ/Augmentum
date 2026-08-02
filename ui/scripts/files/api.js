/**
 * Files panel — server talk. All REST endpoints live here so UI modules can
 * stay focused on rendering. Return shapes match the FastAPI routes.
 */

import {
  state, PAGE_SIZE, TAB_KIND, TAB_SOURCE, KIND_ALIAS,
  AUDIO_LIBRARY_CHIPS, VIDEO_CLOUD_CHIPS,
} from './state.js';

export const downloadUrl = (id) => `/api/files/download/${encodeURIComponent(id)}`;
export const renderUrl = (id) => `/api/files/render/${encodeURIComponent(id)}`;
// Sized thumbnail backed by the cacheable WebP route. Returns the
// full-size original when no thumb has been produced yet (route 404s,
// caller falls back via <img onerror>). ``size`` must be one of the
// server's ALLOWED_SIZES (150, 300, 800) — anything else 400s.
export const thumbUrl = (id, size = 300) =>
  `/api/files/thumb/${encodeURIComponent(id)}?size=${size}`;
// Same thumbnail pipeline keyed by source + source_id rather than
// file_index id. The image gallery tracks image_ids directly and
// doesn't carry file_ids, so this avoids a join round-trip.
export const thumbBySourceUrl = (source, sourceId, size = 300) =>
  `/api/files/thumb/by-source/${encodeURIComponent(source)}/${encodeURIComponent(sourceId)}?size=${size}`;
// Media rows (Audiobookshelf / Emby / ...) stream through a provider-aware
// proxy instead of the generic file download path. The proxy handles
// Range requests and upstream auth tokens so <audio controls> seeking
// works without exposing the user's server token to the browser.
export const mediaStreamUrl = (id, opts = {}) => {
  const params = new URLSearchParams();
  if (opts.episodeId) params.set('episode_id', String(opts.episodeId));
  if (opts.mediaSourceId) params.set('media_source_id', String(opts.mediaSourceId));
  if (Number.isFinite(opts.audioStreamIndex)) {
    params.set('audio_stream_index', String(opts.audioStreamIndex));
  }
  if (Number.isFinite(opts.subtitleStreamIndex)) {
    params.set('subtitle_stream_index', String(opts.subtitleStreamIndex));
  }
  if (Number.isFinite(opts.startTimeS) && Number(opts.startTimeS) > 0) {
    params.set('start_time_s', String(Math.max(0, Number(opts.startTimeS))));
  }
  const qs = params.toString();
  return `/api/media/stream/${encodeURIComponent(id)}${qs ? `?${qs}` : ''}`;
};
export const mediaCoverUrl = (id, opts = {}) => {
  const params = new URLSearchParams();
  if (opts.size) params.set('size', String(opts.size));
  const qs = params.toString();
  return `/api/media/cover/${encodeURIComponent(id)}${qs ? `?${qs}` : ''}`;
};
// Landscape backdrop — Jellyfin/Emby only. Falls back to 404 if the
// item has no backdrop; UI should gate the <img> on meta.has_backdrop
// so we avoid a network request we know will fail.
export const mediaBackdropUrl = (id) =>
  `/api/media/backdrop/${encodeURIComponent(id)}`;
// Actor/director headshot. file_id supplies server context; person_id
// is the provider's own person id (Jellyfin treats people as items).
export const personImageUrl = (fileId, personId) =>
  `/api/media/person/${encodeURIComponent(fileId)}/${encodeURIComponent(personId)}/image`;
// Full profile + filmography for a person. Returns JSON with works
// pre-resolved to our file_index so clicks route into Files.
export async function fetchPersonProfile(fileId, personId) {
  try {
    const resp = await fetch(
      `/api/media/person/${encodeURIComponent(fileId)}/${encodeURIComponent(personId)}`,
    );
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}
export const mediaSubtitleUrl = (id, opts = {}) => {
  const params = new URLSearchParams();
  if (opts.mediaSourceId) params.set('media_source_id', String(opts.mediaSourceId));
  if (Number.isFinite(opts.subtitleStreamIndex)) {
    params.set('subtitle_stream_index', String(opts.subtitleStreamIndex));
  }
  if (Number.isFinite(opts.startTimeS) && Number(opts.startTimeS) > 0) {
    params.set('start_time_s', String(Math.max(0, Number(opts.startTimeS))));
  }
  const qs = params.toString();
  return `/api/media/subtitle/${encodeURIComponent(id)}${qs ? `?${qs}` : ''}`;
};

export async function pushMediaProgress(
  id,
  { current_time_s, duration_s, is_finished = false, episode_id = '' },
) {
  try {
    const resp = await fetch(`/api/media/progress/${encodeURIComponent(id)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ current_time_s, duration_s, is_finished, episode_id }),
    });
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    // Transient network failures shouldn't disrupt playback — a later
    // progress tick retries, and the next catalog sync corrects drift.
    return null;
  }
}

export async function fetchMediaDetails(id, { episodeId = '' } = {}) {
  const params = new URLSearchParams();
  if (episodeId) params.set('episode_id', String(episodeId));
  const qs = params.toString();
  const resp = await fetch(`/api/media/details/${encodeURIComponent(id)}${qs ? `?${qs}` : ''}`);
  if (!resp.ok) return null;
  return resp.json();
}

export async function updateMediaPlaybackSelection(
  id,
  { media_source_id = '', audio_stream_index = null, subtitle_stream_index = null } = {},
) {
  try {
    const resp = await fetch(`/api/media/selection/${encodeURIComponent(id)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        media_source_id,
        audio_stream_index,
        subtitle_stream_index,
      }),
    });
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

/* ── Audiobook bookmarks ──────────────────────────────────────────── */

export async function fetchBookmarks(id, { episodeId = '' } = {}) {
  try {
    const params = new URLSearchParams();
    if (episodeId) params.set('episode_id', String(episodeId));
    const qs = params.toString();
    const resp = await fetch(
      `/api/media/bookmarks/${encodeURIComponent(id)}${qs ? `?${qs}` : ''}`,
      { credentials: 'same-origin' },
    );
    if (!resp.ok) return [];
    const body = await resp.json();
    return Array.isArray(body.bookmarks) ? body.bookmarks : [];
  } catch {
    return [];
  }
}

export async function addBookmark(id, { position_s, label = '', note = '', episode_id = '' }) {
  try {
    const resp = await fetch(`/api/media/bookmarks/${encodeURIComponent(id)}`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ position_s, label, note, episode_id }),
    });
    if (!resp.ok) return null;
    const body = await resp.json();
    return body.bookmark || null;
  } catch {
    return null;
  }
}

export async function deleteBookmark(bookmarkId) {
  try {
    const resp = await fetch(`/api/media/bookmarks/${encodeURIComponent(bookmarkId)}`, {
      method: 'DELETE',
      credentials: 'same-origin',
    });
    return resp.ok;
  } catch {
    return false;
  }
}

export async function fetchMediaOutputs(id) {
  try {
    const resp = await fetch(`/api/media/outputs/${encodeURIComponent(id)}`);
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

export async function fetchMediaReceiverProfiles() {
  try {
    const resp = await fetch('/api/media/receiver-profiles');
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

export async function fetchMediaReceiverLaunchPlan(id, receiverProfile = 'cast_video') {
  try {
    const params = new URLSearchParams();
    if (receiverProfile) params.set('receiver_profile', String(receiverProfile));
    const qs = params.toString();
    const resp = await fetch(
      `/api/media/outputs/${encodeURIComponent(id)}/launch-plan${qs ? `?${qs}` : ''}`,
    );
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

export async function fetchMediaCastLoad(id) {
  return fetchMediaReceiverLaunchPlan(id, 'cast_video');
}

export async function playMediaOnTransportReceiver(
  id,
  { transport = 'dlna', receiver_id = '', receiver_profile = 'dlna_generic_video' } = {},
) {
  try {
    const resp = await fetch(`/api/media/outputs/${encodeURIComponent(id)}/transport-play`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        transport,
        receiver_id,
        receiver_profile,
      }),
    });
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

export async function playMediaOnRemoteSession(
  id,
  {
    session_id,
    start_time_s = 0,
    play_command = 'PlayNow',
    media_source_id = '',
    audio_stream_index = null,
    subtitle_stream_index = null,
  } = {},
) {
  try {
    const resp = await fetch(`/api/media/outputs/${encodeURIComponent(id)}/remote-play`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id,
        start_time_s,
        play_command,
        media_source_id,
        audio_stream_index,
        subtitle_stream_index,
      }),
    });
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

export async function sendMediaRemoteCommand(
  id,
  { session_id, command, seek_position_s = null } = {},
) {
  try {
    const resp = await fetch(`/api/media/outputs/${encodeURIComponent(id)}/remote-command`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id,
        command,
        seek_position_s,
      }),
    });
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

export async function fetchMediaRemoteSession(serverId, sessionId) {
  try {
    const resp = await fetch(
      `/api/media/remote-sessions/${encodeURIComponent(serverId)}/${encodeURIComponent(sessionId)}`,
    );
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

export async function fetchMediaTransportSession(sessionId) {
  try {
    const resp = await fetch(`/api/media/transport-sessions/${encodeURIComponent(sessionId)}`);
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

export async function sendMediaRemoteSessionPlaystate(
  serverId,
  sessionId,
  { command, seek_position_s = null } = {},
) {
  try {
    const resp = await fetch(
      `/api/media/remote-sessions/${encodeURIComponent(serverId)}/${encodeURIComponent(sessionId)}/playstate`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          command,
          seek_position_s,
        }),
      },
    );
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

export async function sendMediaTransportSessionPlaystate(
  sessionId,
  { command, seek_position_s = null } = {},
) {
  try {
    const resp = await fetch(
      `/api/media/transport-sessions/${encodeURIComponent(sessionId)}/playstate`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          command,
          seek_position_s,
        }),
      },
    );
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

export async function sendMediaRemoteSessionGeneral(
  serverId,
  sessionId,
  { command, arguments: args = null } = {},
) {
  try {
    const resp = await fetch(
      `/api/media/remote-sessions/${encodeURIComponent(serverId)}/${encodeURIComponent(sessionId)}/general`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command,
          arguments: args,
        }),
      },
    );
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

export async function sendMediaTransportSessionGeneral(
  sessionId,
  { command, arguments: args = null } = {},
) {
  try {
    const resp = await fetch(
      `/api/media/transport-sessions/${encodeURIComponent(sessionId)}/general`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          command,
          arguments: args,
        }),
      },
    );
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

// --- Stats -------------------------------------------------------------

export async function fetchStats() {
  const resp = await fetch('/api/files/stats');
  if (!resp.ok) return null;
  return resp.json();
}

// --- List / search ----------------------------------------------------

export function buildLoadUrl(offset) {
  const params = new URLSearchParams();
  params.set('limit', String(PAGE_SIZE));
  params.set('offset', String(offset));
  params.set('sort', state.currentSort);
  const query = state.el.search?.value?.trim() || '';
  // Search and sort go on every endpoint. Favorites and Trash used to
  // early-return before `q` was set so the search box was silently a
  // no-op on those chips. Add `q` up-front now that both endpoints
  // accept it.
  if (query) params.set('q', query);
  // Favorites and Trash transcend the Local/Cloud scope — a user's
  // favorite list mixes both by design (the whole point of Favorites
  // is a curated pin list across everything). Trash similarly shouldn't
  // hide cloud items behind a scope the user isn't currently viewing.
  if (state.currentSource === 'trash')     return `/api/files/trash?${params}`;
  if (state.currentSource === 'favorites') return `/api/files/favorites?${params}`;
  // Scope: 'local' isolates uploaded/authored content from remote catalog
  // rows; 'cloud' shows only remote-server content. Sent on every list
  // call that isn't Favorites/Trash (which are scope-agnostic above).
  if (state.currentScope === 'local' || state.currentScope === 'cloud') {
    params.set('scope', state.currentScope);
  }
  const isMediaSource = AUDIO_LIBRARY_CHIPS.has(state.currentSource);
  if (state.currentSource && state.currentSource !== 'all') {
    if (TAB_KIND.has(state.currentSource)) {
      params.set('kind', KIND_ALIAS[state.currentSource] || state.currentSource);
    } else if (TAB_SOURCE.has(state.currentSource)) {
      params.set('source', state.currentSource);
    } else {
      params.set('source', state.currentSource);
    }
  }
  // Playback status filter only applies to media sources. Sending it
  // on non-media chips would silently hide everything because the
  // JSON path wouldn't exist, so we omit.
  if (isMediaSource && state.currentMediaStatus && state.currentMediaStatus !== 'all') {
    params.set('media_status', state.currentMediaStatus);
  }
  // Video chips reuse the same backend predicate (rows have
  // is_finished + progress_pct after Emby/Jellyfin sync). Track the
  // video watch-state in its own state slot so audiobooks and shows
  // can carry independent filter values.
  const isVideoCloud = VIDEO_CLOUD_CHIPS.has(state.currentSource);
  if (isVideoCloud && state.currentVideoStatus && state.currentVideoStatus !== 'all') {
    params.set('media_status', state.currentVideoStatus);
  }
  // Video genre + year range — video chips only, parallel to comic
  // filters. Backend ignores these params on non-video chips, so the
  // gate here is just bandwidth / param-cleanliness.
  if (isVideoCloud) {
    if (state.currentVideoGenre) {
      params.set('genre', state.currentVideoGenre);
    }
    if (state.currentVideoYearFrom) {
      params.set('year_from', String(state.currentVideoYearFrom));
    }
    if (state.currentVideoYearTo) {
      params.set('year_to', String(state.currentVideoYearTo));
    }
  }
  return `/api/files/search?${params}`;
}

export async function fetchListPage(offset) {
  const resp = await fetch(buildLoadUrl(offset));
  if (!resp.ok) return null;
  return resp.json();
}

export async function fetchFileEntry(id) {
  const resp = await fetch(`/api/files/entry/${encodeURIComponent(id)}`);
  if (!resp.ok) return null;
  return resp.json();
}

// --- Mutations --------------------------------------------------------

export async function patchName(id, newName) {
  return fetch(`/api/files/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: newName }),
  });
}

export async function deleteOne(id) {
  return fetch(`/api/files/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

export async function unpinLibrivox(id) {
  return fetch(`/api/media/pin/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

export async function bulkDeleteIds(ids) {
  return fetch('/api/files/bulk-delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ids }),
  });
}

export async function restoreOne(id) {
  return fetch(`/api/files/restore/${encodeURIComponent(id)}`, { method: 'POST' });
}

export async function bulkRestoreIds(ids) {
  return fetch('/api/files/bulk-restore', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ids }),
  });
}

export async function purgeTrash() {
  return fetch('/api/files/purge-trash', { method: 'POST' });
}

export async function toggleFavoriteApi(id) {
  const resp = await fetch(`/api/files/favorite/${encodeURIComponent(id)}`, { method: 'POST' });
  if (!resp.ok) return null;
  return resp.json();
}

export async function patchTags(id, tags) {
  const resp = await fetch(`/api/files/tags/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tags }),
  });
  if (!resp.ok) return null;
  return resp.json();
}

// Save an external video/article URL to the Files panel as a bookmark.
// Idempotent — same URL re-saves update the title rather than dupe.
export async function saveBookmark(payload) {
  const resp = await fetch('/api/files/bookmarks', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {}),
  });
  if (!resp.ok) return null;
  return resp.json();
}

// Tag autocomplete — backs the dropdown under the tag input. Returns
// existing tags ranked by use count, optionally filtered by prefix.
export async function suggestTags(prefix = '', limit = 20) {
  const params = new URLSearchParams();
  if (prefix) params.set('q', prefix);
  if (limit) params.set('limit', String(limit));
  const resp = await fetch(`/api/files/tags/suggest?${params}`);
  if (!resp.ok) return [];
  const data = await resp.json();
  return Array.isArray(data?.tags) ? data.tags : [];
}

// Apply a deterministic content transform (e.g. image-format conversion).
// `disposition` selects whether the response is the converted bytes
// (`download`) or a freshly-stored library file row (`new_file`).
export async function transformFile(id, operation, params, disposition = 'new_file') {
  const resp = await fetch(`/api/files/transform/${encodeURIComponent(id)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ operation, params: params || {}, disposition }),
  });
  if (!resp.ok) {
    let msg = `Transform failed (${resp.status})`;
    try { const err = await resp.json(); if (err.error) msg = err.error; } catch { /* ignore */ }
    throw new Error(msg);
  }
  return resp.json();
}

export async function summarize(id, model) {
  const resp = await fetch(`/api/files/summarize/${encodeURIComponent(id)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: model || '' }),
  });
  return resp;
}

export async function zipDownload(ids) {
  return fetch('/api/files/zip', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ids }),
  });
}

// --- Uploads ----------------------------------------------------------

export async function uploadFiles(fileList, onProgress) {
  if (!fileList || !fileList.length) return { uploaded: [], errors: [] };
  const form = new FormData();
  for (const file of fileList) form.append('files', file, file.name);

  // Prefer XHR for real upload progress; fetch doesn't expose it in Firefox
  // without ReadableStream upload (still flagged in some builds).
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/files/upload');
    if (xhr.upload && onProgress) {
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) onProgress(e.loaded / e.total);
      });
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try { resolve(JSON.parse(xhr.responseText)); }
        catch { resolve({ uploaded: [], errors: [{ error: 'bad response' }] }); }
      } else {
        reject(new Error(`upload failed: HTTP ${xhr.status}`));
      }
    };
    xhr.onerror = () => reject(new Error('upload network error'));
    xhr.send(form);
  });
}
