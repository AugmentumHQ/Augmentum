/**
 * library/open-item.js — THE "open this library item" dispatcher.
 *
 * Extracted 2026-07-19 from detail-pane._openItem during the companion
 * content work, so every caller — the detail pane's Open button, the
 * companion's game.launch channel, candidate cards, future panels —
 * runs ONE dispatch instead of a private copy (the same class-fix
 * discipline as files/open-content.js: off-surface callers reusing a
 * surface's inline logic is how silent drift starts).
 *
 * Dispatch table (mirrors the legacy library.js::_onCardClick):
 *   pub_* (publications)        → /api/library/play/{id} in a new tab
 *   metadata.kind=emulator_rom  → launch-picker → emulator-stage
 *   _type=app                   → workspace.openWorkspace(item, 'play')
 *   _type=game                  → game-surface.openGameSurface
 *   default                     → studio.openStudio
 *
 * Never a silent dead end: failures toast, and openLibraryItemById
 * resolves a bare artifact id to the full row first (game-surface and
 * emulator-stage need metadata.play_mode / embed_src / system off the
 * real artifact, not a stub).
 */

import { showToast } from '../app.js';
import { recordActivity } from './api.js';
import { classifyItem } from './types.js';

/**
 * Open a fully-loaded library item (artifact or publication row).
 * Returns true when a surface actually opened, false on failure/dismiss.
 */
export async function openLibraryItem(item, opts = {}) {
  if (!item || !item.id) return false;

  // Self-heal: ensure ``_type`` / ``_isPublication`` are stamped even
  // if the caller's fetch path forgot classifyItems (operator-observed
  // regression on home-dashboard tiles — unclassified items fell
  // through to studio for everything). Idempotent and cheap.
  if (item._type === undefined) classifyItem(item);
  console.log('[library.open]', {
    id: item.id,
    format: item.format,
    _type: item._type,
    _isPublication: item._isPublication,
    'metadata.kind': item.metadata?.kind,
  });

  // Last-opened tracking — two endpoints depending on the row's home
  // table. Best-effort; doesn't block the actual open. Publications
  // track opens via /launch; the artifact-scoped activity route 404s
  // for pub_* ids, so skip it rather than fire a doomed POST.
  if (item._isPublication) {
    fetch(`/api/library/publications/${encodeURIComponent(item.id)}/launch`, {
      method: 'POST', credentials: 'same-origin',
    }).catch(() => {});
  } else {
    fetch(`/api/artifacts/${encodeURIComponent(item.id)}/open`, {
      method: 'PATCH', credentials: 'same-origin',
    }).catch(() => {});
    try { recordActivity(item.id, 'open').catch(() => {}); } catch { /* best-effort */ }
  }

  if (item._isPublication) {
    window.open(
      `/api/library/play/${encodeURIComponent(item.id)}`,
      '_blank', 'noopener',
    );
    return true;
  }

  const kind = item.metadata?.kind || '';
  if (kind === 'emulator_rom') {
    try {
      const picker = await import('../agent/launch-picker.js');
      const choice = await picker.chooseLaunchMode({
        artifact: item, system: item.metadata?.system,
      });
      if (choice === null) return false;   // user dismissed
      const m = await import('../emulator-stage.js');
      await m.openEmulatorStage(item, {
        startWithAgent: choice.mode === 'partner',
        // characterId only meaningful in partner mode; null = anon
        // default companion.
        characterId: choice.mode === 'partner'
          ? (choice.characterId || null) : null,
      });
      return true;
    } catch (err) {
      console.error('[library] emulator stage open failed', err);
      showToast(
        `Failed to launch ROM: ${err?.message || 'Unknown error'}`,
        'error',
      );
      return false;
    }
  }

  try {
    if (item._type === 'app') {
      const m = await import('../workspace.js');
      await m.openWorkspace(item, 'play');
    } else if (item._type === 'game') {
      const m = await import('../game-surface.js');
      await m.openGameSurface(item);
    } else {
      const m = await import('../studio.js');
      await m.openStudio(item.id, { fromLibrary: true });
    }
    return true;
  } catch (err) {
    console.error('[library] open failed', err);
    showToast(`Failed to open: ${err?.message || 'Unknown error'}`, 'error');
    return false;
  }
}

/**
 * Resolve a bare artifact id to the full row, then open it. The
 * companion's launch channel carries only the id — the launchers need
 * the real metadata (play_mode, embed_src, source_json, system), so a
 * stub would silently mis-dispatch.
 */
export async function openLibraryItemById(artifactId, opts = {}) {
  const id = String(artifactId || '');
  if (!id) return false;
  try {
    const resp = await fetch(
      `/api/artifacts/${encodeURIComponent(id)}`,
      { credentials: 'same-origin' },
    );
    if (!resp.ok) throw new Error(`artifact fetch ${resp.status}`);
    const item = await resp.json();
    // Artifact rows can arrive with JSON-string metadata/tags when
    // fetched outside the library API layer (which decodes them
    // server-side) — normalize before classify.
    if (typeof item.metadata === 'string') {
      try { item.metadata = JSON.parse(item.metadata || '{}'); } catch { item.metadata = {}; }
    }
    if (typeof item.tags === 'string') {
      try { item.tags = JSON.parse(item.tags || '[]'); } catch { item.tags = []; }
    }
    return await openLibraryItem(item, opts);
  } catch (err) {
    console.warn('[library] open-by-id failed', err);
    showToast(
      `Couldn't open ${opts.label ? `"${opts.label}"` : 'that item'} — it may have been removed.`,
      'error',
    );
    return false;
  }
}
