/**
 * consumption/item-menu.js — right-click / long-press menu for media
 * tiles (Media surface now, cast-control later).
 *
 * Brings the Files grid's context-menu verbs to the consumption tiles:
 * play, open details, mark watched/unwatched (read/unread for comics),
 * reset progress, add to playlist, download, cast. Read-state writes
 * go through the same POST /api/media/progress contract every other
 * surface uses; playlist rides the global `playlist:add-item` event.
 */

import { escapeHtml, showToast } from '../app.js';
import { pushMediaProgress, downloadUrl } from '../files/api.js';

/**
 * @param item  tile shape from _entry_to_tile
 * @param ev    the triggering pointer event (menu anchors at cursor)
 * @param opts  { onPlay(item), onOpenDetail(item), onCast(item, anchor),
 *               onChanged(item) — fired after a progress write mutates
 *               the tile's is_finished/progress_pct }
 */
export function openItemMenu(item, ev, {
  onPlay,
  onOpenDetail,
  onCast,
  onChanged,
} = {}) {
  document.querySelector('.media-item-menu')?.remove();

  const isSeries = (item.play || {}).action === 'browse_series';
  const isComic = (item.kind || '') === 'comic';
  const leaf = !isSeries;
  const watchVerb = isComic ? 'read' : (item.kind === 'audio' ? 'listened' : 'watched');
  const hasProgress = (Number(item.progress_pct) || 0) > 0.5 || !!item.is_finished;

  const entries = [];
  if (leaf && onPlay) entries.push({ id: 'play', label: 'Play' });
  if (onOpenDetail) entries.push({ id: 'detail', label: isSeries ? 'Open' : 'View details' });
  if (leaf) {
    entries.push({
      id: 'toggle-watched',
      label: item.is_finished ? `Mark un${watchVerb}` : `Mark ${watchVerb}`,
    });
    if (hasProgress) entries.push({ id: 'reset', label: 'Reset progress' });
    if (item.kind === 'audio' || item.kind === 'video') {
      entries.push({ id: 'playlist', label: 'Add to playlist' });
    }
    entries.push({ id: 'download', label: 'Download' });
  }
  if (onCast && leaf) entries.push({ id: 'cast', label: 'Cast to TV…' });
  if (!entries.length) return;

  const menu = document.createElement('div');
  menu.className = 'media-item-menu';
  menu.setAttribute('role', 'menu');
  menu.innerHTML = `
    <div class="media-item-menu-title">${escapeHtml(item.title || 'Item')}</div>
    ${entries.map((e) =>
      `<button type="button" role="menuitem" data-menu="${e.id}">${escapeHtml(e.label)}</button>`,
    ).join('')}
  `;
  document.body.appendChild(menu);

  // Anchor at the pointer; clamp inside the viewport.
  const mw = menu.offsetWidth || 200;
  const mh = menu.offsetHeight || 200;
  const x = Math.min(Math.max(8, ev.clientX), window.innerWidth - mw - 8);
  const y = Math.min(Math.max(8, ev.clientY), window.innerHeight - mh - 8);
  menu.style.left = `${x}px`;
  menu.style.top = `${y}px`;

  const close = () => {
    menu.remove();
    document.removeEventListener('pointerdown', onAway, true);
    document.removeEventListener('keydown', onKey, true);
  };
  const onAway = (e) => { if (!menu.contains(e.target)) close(); };
  const onKey = (e) => {
    if (e.key === 'Escape') {
      e.stopPropagation();
      close();
    }
  };
  document.addEventListener('pointerdown', onAway, true);
  document.addEventListener('keydown', onKey, true);

  menu.addEventListener('click', async (e) => {
    const btn = e.target.closest('[data-menu]');
    if (!btn) return;
    const id = btn.dataset.menu;
    const anchor = ev.currentTarget || btn;
    close();
    switch (id) {
      case 'play':
        onPlay?.(item);
        break;
      case 'detail':
        onOpenDetail?.(item);
        break;
      case 'toggle-watched': {
        const target = !item.is_finished;
        const ok = await _writeProgress(item, {
          current_time_s: target ? (Number(item.duration_s) || 0) : 0,
          is_finished: target,
        });
        if (ok) {
          showToast(target ? `Marked ${watchVerb}.` : `Marked un${watchVerb}.`, 'info', 1600);
          onChanged?.(item);
        }
        break;
      }
      case 'reset': {
        const ok = await _writeProgress(item, { current_time_s: 0, is_finished: false });
        if (ok) {
          showToast('Progress reset.', 'info', 1600);
          onChanged?.(item);
        }
        break;
      }
      case 'playlist':
        window.dispatchEvent(new CustomEvent('playlist:add-item', {
          detail: {
            type: 'file',
            fileId: item.file_id,
            name: item.title || '',
            kind: item.kind === 'video' ? 'video' : 'audio',
            thumbnail: item.cover_url || '',
          },
        }));
        showToast('Added to playlist.', 'info', 1600);
        break;
      case 'download': {
        const a = document.createElement('a');
        a.href = downloadUrl(item.file_id);
        a.download = '';
        document.body.appendChild(a);
        a.click();
        a.remove();
        break;
      }
      case 'cast':
        onCast?.(item, anchor);
        break;
    }
  });
}

async function _writeProgress(item, { current_time_s, is_finished }) {
  const resp = await pushMediaProgress(item.file_id, {
    current_time_s,
    duration_s: Number(item.duration_s) || 0,
    is_finished,
  });
  if (!resp) {
    showToast("Couldn't update that — try again.", 'error', 2600);
    return false;
  }
  item.is_finished = is_finished;
  item.progress_pct = is_finished ? 100 : 0;
  return true;
}
