/**
 * Playlist — basic media queue that chains short items (YouTube + library
 * audio/video files) and dispatches each to the surface it natively plays
 * through. Auto-advances on the ended hook in grove-ambient.
 *
 * Item shapes (matched to backend):
 *   { type: 'youtube', videoId, title, channel?, thumbnail? }
 *   { type: 'file',    fileId,  name,  kind: 'audio'|'video', thumbnail? }
 *
 * Public API:
 *   init()                 — wire DOM + load playlists
 *   addItem(item)          — append to active playlist (creates one if none)
 *   notifyEnded()          — called by grove-ambient when current item ends.
 *                            Returns true if the playlist handled it (caller
 *                            should suppress its default fallback).
 *   isActive()             — is a playlist currently driving playback?
 */

import { loadVideo, loadMediaVideo, setEndedHook } from './grove-ambient.js';
import { escapeHtml, showToast } from './app.js';
import { downloadUrl, mediaStreamUrl, fetchFileEntry } from './files/api.js';
import { isMediaServerFile } from './files/helpers.js';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let _playlists = [];                  // [{id, name, items: [...], updated_at}]
let _activeId = null;                 // id of playlist currently selected in UI
let _playingIndex = -1;               // index of the item currently playing
let _shuffleOrder = null;             // [perm of indices] when shuffle on, else null
let _shufflePos = -1;                 // cursor into _shuffleOrder

const $ = (id) => document.getElementById(id);
let _dom = {};

// ---------------------------------------------------------------------------
// REST
// ---------------------------------------------------------------------------
async function _fetchAll() {
  try {
    const resp = await fetch('/api/playlists', { credentials: 'same-origin' });
    if (!resp.ok) return;
    const data = await resp.json();
    _playlists = Array.isArray(data.playlists) ? data.playlists : [];
  } catch (err) {
    console.warn('[Playlist] Load failed:', err);
  }
}

async function _createPlaylist(name) {
  try {
    const resp = await fetch('/api/playlists', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, items: [] }),
    });
    if (!resp.ok) return null;
    const data = await resp.json();
    if (data.playlist) {
      _playlists.unshift(data.playlist);
      return data.playlist;
    }
  } catch (err) {
    console.warn('[Playlist] Create failed:', err);
  }
  return null;
}

async function _patchPlaylist(id, patch) {
  try {
    const resp = await fetch(`/api/playlists/${encodeURIComponent(id)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
    if (!resp.ok) return null;
    const data = await resp.json();
    if (data.playlist) {
      const idx = _playlists.findIndex(p => p.id === id);
      if (idx >= 0) _playlists[idx] = data.playlist;
      return data.playlist;
    }
  } catch (err) {
    console.warn('[Playlist] Update failed:', err);
  }
  return null;
}

async function _deletePlaylist(id) {
  try {
    const resp = await fetch(`/api/playlists/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });
    if (resp.ok) {
      _playlists = _playlists.filter(p => p.id !== id);
      return true;
    }
  } catch (err) {
    console.warn('[Playlist] Delete failed:', err);
  }
  return false;
}

// ---------------------------------------------------------------------------
// Item key — used for dedupe + activity highlight
// ---------------------------------------------------------------------------
function _itemKey(item) {
  if (!item) return '';
  if (item.type === 'youtube') return `youtube:${item.videoId || ''}`;
  if (item.type === 'file') return `file:${item.fileId || ''}`;
  return '';
}

// ---------------------------------------------------------------------------
// Queue + dispatch
// ---------------------------------------------------------------------------
function _activePlaylist() {
  return _playlists.find(p => p.id === _activeId) || null;
}

function _resetShuffleOrder(len, startAt = 0) {
  const indices = Array.from({ length: len }, (_, i) => i);
  // Shuffle so the currently-playing index stays at position 0
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  if (startAt > 0 && startAt < indices.length) {
    const pos = indices.indexOf(startAt);
    if (pos > 0) [indices[0], indices[pos]] = [indices[pos], indices[0]];
  }
  _shuffleOrder = indices;
  _shufflePos = 0;
}

function _isShuffleOn() {
  return _shuffleOrder !== null;
}

function _nextIndex() {
  const pl = _activePlaylist();
  if (!pl || !pl.items.length) return -1;
  if (_isShuffleOn()) {
    _shufflePos = (_shufflePos + 1) % _shuffleOrder.length;
    return _shuffleOrder[_shufflePos];
  }
  return (_playingIndex + 1) % pl.items.length;
}

function _prevIndex() {
  const pl = _activePlaylist();
  if (!pl || !pl.items.length) return -1;
  if (_isShuffleOn()) {
    _shufflePos = (_shufflePos - 1 + _shuffleOrder.length) % _shuffleOrder.length;
    return _shuffleOrder[_shufflePos];
  }
  return (_playingIndex - 1 + pl.items.length) % pl.items.length;
}

async function _playItem(item, idx) {
  if (!item) return;
  _playingIndex = idx;

  if (item.type === 'youtube') {
    await loadVideo({
      videoId: item.videoId,
      title: item.title || 'Untitled',
      channel: item.channel || '',
      thumbnail: item.thumbnail || `https://i.ytimg.com/vi/${item.videoId}/mqdefault.jpg`,
      isLivestream: false,
    });
    _renderItems();
    return;
  }

  if (item.type === 'file') {
    const file = await fetchFileEntry(item.fileId);
    if (!file) {
      showToast('File not found — skipping.', 'error');
      // Auto-advance past missing items so a broken playlist still walks.
      _advance();
      return;
    }
    const url = isMediaServerFile(file)
      ? mediaStreamUrl(file.id, { episodeId: file.source_metadata?.selected_episode_id || '' })
      : downloadUrl(file.id);
    const el = document.createElement(item.kind === 'audio' ? 'audio' : 'video');
    el.src = url;
    el.playsInline = true;
    await loadMediaVideo({
      element: el,
      video: {
        title: item.name || file.name || 'Untitled',
        thumbnail: item.thumbnail || '',
        fileId: file.id,
        entityKind: 'playlist_item',
      },
      onClose: null,
    });
    _renderItems();
    return;
  }

  console.warn('[Playlist] Unknown item type:', item.type);
  _advance();
}

function _advance() {
  const idx = _nextIndex();
  const pl = _activePlaylist();
  if (idx < 0 || !pl) return;
  void _playItem(pl.items[idx], idx);
}

function _rewind() {
  const idx = _prevIndex();
  const pl = _activePlaylist();
  if (idx < 0 || !pl) return;
  void _playItem(pl.items[idx], idx);
}

// ---------------------------------------------------------------------------
// Public hooks
// ---------------------------------------------------------------------------
export function isActive() {
  return _playingIndex >= 0 && !!_activePlaylist();
}

export function notifyEnded() {
  if (!isActive()) return false;
  _advance();
  return true;
}

// ---------------------------------------------------------------------------
// Content-category families — the playlist boundary
// ---------------------------------------------------------------------------
// A playlist holds ONE family, so movies don't mix with music and music
// doesn't mix with audiobooks (the old audio/video binary lumped them).
// YouTube is flexible — it can join any family (music videos used as music,
// trailers in a watch list). An empty or youtube-only playlist is 'any'
// until a typed item pins it.
const _ENTITY_FAMILY = {
  // watch
  movie: 'watch', series: 'watch', season: 'watch', episode: 'watch',
  music_video: 'watch', live_program: 'watch',
  // music
  music: 'music', audio_track: 'music', music_album: 'music', track: 'music',
  // spoken
  book: 'spoken', podcast: 'spoken',
  // comics
  comic: 'comics', manga: 'comics',
};

const _FAMILY_LABEL = {
  watch: 'Watch', music: 'Music', spoken: 'Spoken', comics: 'Comics', any: 'Any',
};

function _itemFamily(item) {
  if (!item) return 'any';
  if (item.type === 'youtube') return 'youtube';            // flexible
  const ek = String(item.entityKind || item.entity_kind || '').toLowerCase();
  if (_ENTITY_FAMILY[ek]) return _ENTITY_FAMILY[ek];
  // Fallback by coarse kind when entity_kind is absent (older items).
  const k = String(item.kind || '').toLowerCase();
  if (k === 'video' || k === 'live_video') return 'watch';
  if (k === 'comic' || k === 'document') return 'comics';
  return 'spoken';   // unknown audio → spoken (audiobooks/podcasts dominate)
}

function _playlistFamily(p) {
  // The first non-youtube item pins the family; empty / youtube-only → 'any'.
  for (const it of (p.items || [])) {
    const fam = _itemFamily(it);
    if (fam !== 'youtube') return fam;
  }
  return 'any';
}

function _isCompatible(item, p) {
  const itemFam = _itemFamily(item);
  if (itemFam === 'youtube') return true;                  // fits anywhere
  const plFam = _playlistFamily(p);
  return plFam === 'any' || plFam === itemFam;
}

// Shared add — dedupe + persist. Used by the chooser and the programmatic
// addItem() entry point.
async function _commitItem(target, item) {
  const key = _itemKey(item);
  if (!key) {
    showToast('Cannot add — missing item id.', 'error');
    return false;
  }
  if (target.items.some(i => _itemKey(i) === key)) {
    showToast(`Already in "${target.name}"`);
    return true;
  }
  const updated = await _patchPlaylist(target.id, { items: [...target.items, item] });
  if (!updated) {
    showToast('Failed to save playlist.', 'error');
    return false;
  }
  showToast(`Added to "${updated.name}"`);
  _render();
  return true;
}

// ---------------------------------------------------------------------------
// Add-to-playlist chooser — "new vs existing", scoped to compatible family
// ---------------------------------------------------------------------------
let _chooserEl = null;

function _closeAddChooser() {
  if (_chooserEl && _chooserEl.parentNode) _chooserEl.parentNode.removeChild(_chooserEl);
  _chooserEl = null;
  document.removeEventListener('keydown', _onChooserKey, true);
}

function _onChooserKey(e) {
  if (e.key === 'Escape') { e.stopPropagation(); _closeAddChooser(); }
}

function _openAddChooser(item) {
  _closeAddChooser();
  const itemFam = _itemFamily(item);
  // Strict family boundary: only compatible existing playlists are offered.
  const compatible = _playlists.filter(p => _isCompatible(item, p));
  const famNote = itemFam === 'youtube'
    ? 'YouTube — fits any playlist'
    : `${_FAMILY_LABEL[itemFam] || 'Media'} playlist`;

  const overlay = document.createElement('div');
  overlay.className = 'playlist-add-overlay';
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-label', 'Add to playlist');
  overlay.innerHTML = `
    <div class="playlist-add-sheet" role="document">
      <div class="playlist-add-head">
        <span class="playlist-add-title">Add to playlist</span>
        <span class="playlist-add-fam">${escapeHtml(famNote)}</span>
      </div>
      <div class="playlist-add-new">
        <input type="text" class="playlist-add-name" placeholder="New playlist name…" maxlength="120" />
        <button type="button" class="playlist-add-create">Create &amp; add</button>
      </div>
      ${compatible.length ? `
        <div class="playlist-add-divider"><span>or add to</span></div>
        <ul class="playlist-add-list">
          ${compatible.map(p => `
            <li>
              <button type="button" class="playlist-add-existing" data-pl-id="${escapeHtml(p.id)}">
                <span class="playlist-add-pl-name">${escapeHtml(p.name)}</span>
                <span class="playlist-add-pl-count">${(p.items || []).length}</span>
              </button>
            </li>`).join('')}
        </ul>` : '<p class="playlist-add-empty">No compatible playlists yet — create one above.</p>'}
      <div class="playlist-add-foot">
        <button type="button" class="playlist-add-cancel">Cancel</button>
      </div>
    </div>`;

  const nameInput = overlay.querySelector('.playlist-add-name');
  const doCreate = async () => {
    const name = (nameInput.value || '').trim() || 'New Playlist';
    const created = await _createPlaylist(name);
    if (!created) { showToast('Could not create playlist.', 'error'); return; }
    _activeId = created.id;
    _renderSelect();
    await _commitItem(created, item);
    _closeAddChooser();
  };

  overlay.querySelector('.playlist-add-create')?.addEventListener('click', () => void doCreate());
  nameInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); void doCreate(); }
  });
  overlay.querySelectorAll('.playlist-add-existing').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const pl = _playlists.find(p => p.id === btn.dataset.plId);
      if (pl) await _commitItem(pl, item);
      _closeAddChooser();
    });
  });
  overlay.querySelector('.playlist-add-cancel')?.addEventListener('click', _closeAddChooser);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) _closeAddChooser(); });

  document.body.appendChild(overlay);
  _chooserEl = overlay;
  document.addEventListener('keydown', _onChooserKey, true);
  setTimeout(() => nameInput?.focus(), 30);
}

export async function addItem(item) {
  // Programmatic add — routes to the chooser so the family boundary +
  // "new vs existing" prompt apply consistently with the UI path.
  _openAddChooser(item);
  return true;
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------
function _renderSelect() {
  if (!_dom.select) return;
  if (_playlists.length === 0) {
    _dom.select.innerHTML = '<option value="">No playlists yet</option>';
    _dom.select.disabled = true;
    return;
  }
  _dom.select.disabled = false;
  _dom.select.innerHTML = _playlists.map(p => {
    // Provenance marker (persona-agnostic — OSS rule): companion-
    // created playlists carry a label suffix. The surface is a
    // dropdown, so the suffix IS the filter affordance for now.
    const originTag = p.origin === 'companion' ? ' · companion' : '';
    return `<option value="${escapeHtml(p.id)}"${p.id === _activeId ? ' selected' : ''}>${escapeHtml(p.name)}${originTag}</option>`;
  }).join('');
}

function _renderItems() {
  if (!_dom.items) return;
  const pl = _activePlaylist();
  if (!pl || !pl.items.length) {
    _dom.items.innerHTML = '<div class="grove-playlist-empty">No items yet — add YouTube videos or library audio/video files.</div>';
    return;
  }
  _dom.items.innerHTML = pl.items.map((item, idx) => {
    const playing = idx === _playingIndex ? ' playing' : '';
    const title = item.title || item.name || 'Untitled';
    const typeLabel = item.type === 'youtube' ? 'YT' : (item.kind === 'audio' ? 'AUDIO' : 'VIDEO');
    return `<div class="grove-playlist-item${playing}" data-idx="${idx}">
      <span class="grove-playlist-type">${typeLabel}</span>
      <span class="grove-playlist-title" title="${escapeHtml(title)}">${escapeHtml(title)}</span>
      <button class="grove-playlist-remove" data-remove-idx="${idx}" aria-label="Remove" title="Remove">&times;</button>
    </div>`;
  }).join('');
}

function _render() {
  _renderSelect();
  _renderItems();
  if (_dom.shuffle) {
    _dom.shuffle.classList.toggle('active', _isShuffleOn());
    _dom.shuffle.setAttribute('aria-pressed', _isShuffleOn() ? 'true' : 'false');
  }
}

async function _onNewClicked() {
  const name = prompt('Playlist name:', 'My Playlist');
  if (!name || !name.trim()) return;
  const created = await _createPlaylist(name.trim().slice(0, 80));
  if (created) {
    _activeId = created.id;
    _playingIndex = -1;
    _shuffleOrder = null;
    _render();
  }
}

async function _onDeleteClicked() {
  const pl = _activePlaylist();
  if (!pl) return;
  if (!confirm(`Delete playlist "${pl.name}"?`)) return;
  const ok = await _deletePlaylist(pl.id);
  if (ok) {
    _activeId = _playlists[0]?.id || null;
    _playingIndex = -1;
    _shuffleOrder = null;
    _render();
  }
}

function _onSelectChanged() {
  _activeId = _dom.select.value || null;
  _playingIndex = -1;
  _shuffleOrder = null;
  _render();
}

function _onShuffleClicked() {
  const pl = _activePlaylist();
  if (!pl || !pl.items.length) return;
  if (_isShuffleOn()) {
    _shuffleOrder = null;
    _shufflePos = -1;
  } else {
    _resetShuffleOrder(pl.items.length, Math.max(0, _playingIndex));
  }
  _render();
}

async function _onItemsClicked(e) {
  const removeBtn = e.target.closest('[data-remove-idx]');
  if (removeBtn) {
    e.stopPropagation();
    const idx = parseInt(removeBtn.dataset.removeIdx, 10);
    const pl = _activePlaylist();
    if (!pl || !Number.isFinite(idx)) return;
    const nextItems = pl.items.filter((_, i) => i !== idx);
    const updated = await _patchPlaylist(pl.id, { items: nextItems });
    if (updated) {
      if (idx === _playingIndex) _playingIndex = -1;
      else if (idx < _playingIndex) _playingIndex--;
      if (_isShuffleOn()) _resetShuffleOrder(updated.items.length, Math.max(0, _playingIndex));
      _render();
    }
    return;
  }
  const itemEl = e.target.closest('.grove-playlist-item');
  if (!itemEl) return;
  const idx = parseInt(itemEl.dataset.idx, 10);
  const pl = _activePlaylist();
  if (!pl || !Number.isFinite(idx) || !pl.items[idx]) return;
  if (_isShuffleOn()) _resetShuffleOrder(pl.items.length, idx);
  await _playItem(pl.items[idx], idx);
}

export async function init() {
  _dom = {
    section:  $('grove-playlist-section'),
    select:   $('grove-playlist-select'),
    items:    $('grove-playlist-items'),
    newBtn:   $('grove-playlist-new'),
    delBtn:   $('grove-playlist-delete'),
    shuffle:  $('grove-playlist-shuffle'),
    next:     $('grove-playlist-next'),
    prev:     $('grove-playlist-prev'),
  };
  if (!_dom.section) return;  // Grove panel not present on this surface

  await _fetchAll();
  if (_playlists.length > 0) _activeId = _playlists[0].id;
  _render();

  _dom.newBtn?.addEventListener('click', _onNewClicked);
  _dom.delBtn?.addEventListener('click', _onDeleteClicked);
  _dom.select?.addEventListener('change', _onSelectChanged);
  _dom.shuffle?.addEventListener('click', _onShuffleClicked);
  _dom.next?.addEventListener('click', _advance);
  _dom.prev?.addEventListener('click', _rewind);
  _dom.items?.addEventListener('click', _onItemsClicked);

  // External surfaces (yt-panel, files preview) dispatch this event.
  window.addEventListener('playlist:add-item', (e) => {
    if (e.detail) void addItem(e.detail);
  });

  // Register the auto-advance hook so grove-ambient calls us on 'ended'.
  setEndedHook(notifyEnded);
}
