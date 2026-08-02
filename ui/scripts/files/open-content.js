// Canonical off-surface content opener — THE entry point for any caller
// outside the Files grid lifecycle (companion candidate cards, intent
// verbs via intent-action-router, architect emissions, future panels)
// that wants to open/play a file_index item.
//
// Why this exists (2026-07-18): "hit Play on a companion movie card and
// it just goes away" turned out to be a CLASS of silent no-ops, not one
// bug. Off-surface callers were handing raw /api/files/entry rows (or
// worse, synthetic {id, name} stubs) straight to activateFile, which has
// two silent kill points for anything not living in the Files grid:
//
//   1. The media-server guard (actions.js): media-server rows without
//      source_metadata.stream_path / selected_episode_id return with no
//      error. The Files surface avoids this by synthesizing
//      selected_episode_id for leaf rows (continue-rail.js _resume).
//   2. Resolver misses (preview.js _resolvePreviewFile): openMediaPreview
//      re-resolves by id through state.detailOverrideFile → state.files →
//      registered extra sources. When Files isn't open, state.files is
//      empty and the click dies silently. Three surfaces had already
//      grown private workarounds (media.js _recentFiles LRU, the rail's
//      extra source, render.js next-up using openVideoPreviewById).
//
// This module consolidates those workarounds into one library-level
// helper with three guarantees:
//
//   - NEVER a silent dead end. Every path ends in playback, a viewer,
//     or an honest fallback (Files opened at the item + toast).
//   - NEVER auto-picks on the user's behalf. A series/season row needs
//     a human episode choice, so it deep-links to the item's detail in
//     Files (files:open-with-filter) instead of guessing an episode.
//   - Progress-safe audio. Audio goes to the media-player singleton
//     (background mini-player, resumes saved position) — never the
//     fullscreen preview overlay, which restarts from zero (the
//     audiobook-progress wipe documented in continue-rail.js).
//
// Returns a result string so callers/tests can assert the outcome:
// 'played' | 'opened' | 'detail' | 'fallback'.

import { fetchFileEntry } from './api.js';
import { isAudio, isMediaServerFile, isVideo } from './helpers.js';
import { activateFile } from './actions.js';
import { registerExtraFileSource } from './preview.js';

// Entries opened through this spine, registered as a preview resolver
// source so every downstream by-id hop (openMediaPreview, gallery,
// comic reader handoff) finds the full row even when Files is closed.
// Same LRU pattern as media.js _recentFiles — kept separate so the
// spine has no dependency on any surface being loaded.
const _openedFiles = new Map(); // file_id -> file_entry
const _OPENED_CAP = 64;
registerExtraFileSource(() => Array.from(_openedFiles.values()));

function _remember(file) {
  if (!file?.id) return;
  _openedFiles.delete(file.id);
  _openedFiles.set(file.id, file);
  if (_openedFiles.size > _OPENED_CAP) {
    _openedFiles.delete(_openedFiles.keys().next().value);
  }
}

function _toast(msg, tone = 'info') {
  try { window.__augmentum?.showToast?.(msg, tone, 3200); } catch { /* chrome not ready */ }
}

/**
 * Open the Files panel focused on the item (detail panel if the row
 * isn't in the current page) — the honest "I couldn't just play it,
 * here it is one click away" landing, and the deliberate landing for
 * rows that need a human choice (series/season episode pick).
 */
function _openInFiles(fileId, label) {
  window.dispatchEvent(new CustomEvent('files:open-with-filter', {
    detail: {
      search: label ? String(label).slice(0, 200) : '',
      fileId: fileId || '',
    },
  }));
}

function _entityKind(file) {
  return String(file?.source_metadata?.entity_kind || '').toLowerCase();
}

/**
 * Canonical open. Accepts a file_index entry object or a bare file id.
 *
 * opts.label — human title for fallback search/toast when the entry
 *              itself can't be fetched (e.g. stale reference).
 */
export async function openContent(fileOrId, opts = {}) {
  const label = opts.label || (typeof fileOrId === 'object' ? (fileOrId?.name || '') : '');
  let file = null;
  try {
    if (typeof fileOrId === 'string') {
      file = await fetchFileEntry(fileOrId);
    } else if (fileOrId && fileOrId.id) {
      // A row without source_metadata is a stub (synthetic caller shape,
      // lean cast-tile shape) — refetch so kind-helpers see the truth.
      file = fileOrId.source_metadata !== undefined
        ? fileOrId
        : (await fetchFileEntry(fileOrId.id)) || fileOrId;
    }
  } catch (err) {
    console.warn('[open-content] entry fetch failed', err);
  }

  if (!file || !file.id) {
    // Reference didn't resolve at all. If we have a title, Files search
    // is still one click from the item; otherwise say so and stop.
    if (label) {
      _openInFiles('', label);
      _toast(`Couldn't open "${label}" directly — showing it in Files.`);
    } else {
      _toast("Couldn't open that item — the reference didn't resolve.", 'error');
    }
    return 'fallback';
  }

  _remember(file);

  try {
    // Audio → background mini-player singleton (progress-safe, no
    // surface yank; the media.play contract).
    if (isAudio(file)) {
      const m = await import('../media-player.js');
      await m.play(file.id);
      return 'played';
    }

    const meta = file.source_metadata || {};
    if (isMediaServerFile(file) && !meta.stream_path && !meta.selected_episode_id) {
      const ek = _entityKind(file);
      if (ek === 'series' || ek === 'season') {
        // Which episode is the user's call — land on the item's detail
        // (episode list) rather than guessing.
        _openInFiles(file.id, file.name || label);
        return 'detail';
      }
      // Leaf row (movie / episode reached directly): the row itself is
      // the streamable unit. Same synthesis continue-rail.js does to
      // satisfy activateFile's media-server guard.
      file = {
        ...file,
        source_metadata: { ...meta, selected_episode_id: file.id },
      };
      _remember(file);
    }

    // Canonical kind-aware cascade: floating video player (cast
    // intercept, resume position, next-episode chain), comic reader,
    // gallery, pdf/epub/html/rendered previews, bookmark handoff.
    activateFile(file);
    return isVideo(file) ? 'played' : 'opened';
  } catch (err) {
    console.warn('[open-content] open failed', err);
    _openInFiles(file.id, file.name || label);
    _toast(`Couldn't start "${file.name || label || 'that'}" — showing it in Files.`, 'error');
    return 'fallback';
  }
}
