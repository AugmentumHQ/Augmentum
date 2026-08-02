// ui/scripts/connect/emoji-picker.js
//
// A lightweight, dependency-free emoji picker popover for the Connect
// composer. Curated categories (no keyword search / no external dataset)
// — enough to insert common emojis while typing, iMessage-style. Anchors
// above the trigger button; dismisses on outside-click / Escape / pick.

import { escapeHtml } from '../app.js';

const CATEGORIES = [
  { name: 'Smileys', emojis: ['😀','😃','😄','😁','😆','😅','😂','🤣','😊','😇','🙂','🙃','😉','😌','😍','🥰','😘','😗','😙','😚','😋','😛','😝','😜','🤪','🤨','🧐','🤓','😎','🥳','😏','😒','😞','😔','😟','😕','🙁','☹️','😣','😖','😫','😩','🥺','😢','😭','😤','😠','😡','🤬','🤯','😳','🥵','🥶','😱','😨','😰','😥','😓','🤗','🤔','🤭','🤫','🤥','😶','😐','😑','😬','🙄','😯','😦','😧','😮','😲','🥱','😴','🤤','😪','🤐','🥴','🤢','🤮','🤧','😷','🤒','🤕'] },
  { name: 'Gestures', emojis: ['👍','👎','👌','🤌','🤏','✌️','🤞','🤟','🤘','🤙','👈','👉','👆','👇','☝️','✋','🤚','🖐️','🖖','👋','🤝','🙏','✍️','💪','🦾','👏','🙌','👐','🤲','🤜','🤛','✊','👊'] },
  { name: 'Hearts', emojis: ['❤️','🧡','💛','💚','💙','💜','🖤','🤍','🤎','💔','❣️','💕','💞','💓','💗','💖','💘','💝','💟','♥️'] },
  { name: 'People', emojis: ['👶','🧒','👦','👧','🧑','👨','👩','🧔','👴','👵','🙋','🙆','🙅','💁','🙇','🤦','🤷','👮','🕵️','💂','👷','🤴','👸','🦸','🦹','🧙','🧚','🧛','🧟','💃','🕺','👯','🧖','🧗'] },
  { name: 'Animals', emojis: ['🐶','🐱','🐭','🐹','🐰','🦊','🐻','🐼','🐨','🐯','🦁','🐮','🐷','🐸','🐵','🐔','🐧','🐦','🐤','🦆','🦉','🦄','🐝','🦋','🐌','🐞','🐢','🐍','🐙','🦑','🦀','🐬','🐳','🐟','🐠'] },
  { name: 'Food', emojis: ['🍎','🍐','🍊','🍋','🍌','🍉','🍇','🍓','🫐','🍒','🍑','🥭','🍍','🥥','🥝','🍅','🥑','🍆','🥕','🌽','🌶️','🥔','🍟','🍔','🌭','🍕','🌮','🌯','🥗','🍝','🍜','🍣','🍤','🍰','🎂','🍦','🍩','🍪','☕','🍵','🍺','🍷','🥂'] },
  { name: 'Activities', emojis: ['⚽','🏀','🏈','⚾','🎾','🏐','🏉','🎱','🏓','🏸','🥅','🏒','🏑','⛳','🏹','🎣','🥊','🥋','⛸️','🎿','🏂','🏋️','🤸','⛹️','🏇','🧘','🏄','🏊','🚴','🎮','🎲','🎯','🎳','🎸','🎺','🎻','🎤','🎧','🎨'] },
  { name: 'Travel', emojis: ['🚗','🚕','🚙','🚌','🏎️','🚓','🚑','🚒','🚐','🚚','🏍️','🚲','✈️','🚀','🛸','🚁','⛵','🚤','🚉','🚆','🗺️','🗽','🗼','🏰','🏠','⛺','🏖️','🏝️','🌋','🗻','🌅','🌃','🌆'] },
  { name: 'Objects', emojis: ['⌚','📱','💻','⌨️','🖥️','🖨️','🕹️','📷','📸','🎥','📺','🔋','💡','🔦','📕','📚','📝','✏️','📌','📎','✂️','🔒','🔑','🔨','🔧','🧲','💎','🔔','🎁','🎈','🎉','🎊','✨'] },
  { name: 'Symbols', emojis: ['✅','❌','❗','❓','💯','🔥','⭐','🌟','💫','⚡','💥','💢','💦','💨','💬','💭','🎵','🎶','➕','➖','✔️','🔴','🟠','🟡','🟢','🔵','🟣','⚫','⚪'] },
];

let _picker = null;
let _onPick = null;
let _keyHandler = null;
let _outsideHandler = null;

export function openEmojiPicker(anchorEl, onPick) {
  if (_picker) { closeEmojiPicker(); return; }
  if (!anchorEl) return;
  _onPick = onPick;

  const el = document.createElement('div');
  el.className = 'connect-emoji-picker';
  el.setAttribute('role', 'dialog');
  el.setAttribute('aria-label', 'Emoji picker');
  const tabs = CATEGORIES.map((c, i) =>
    `<button class="connect-emoji-tab${i === 0 ? ' active' : ''}" type="button"
             data-cat="${i}" title="${escapeHtml(c.name)}" aria-label="${escapeHtml(c.name)}">${c.emojis[0]}</button>`,
  ).join('');
  const groups = CATEGORIES.map((c, i) => `
    <div class="connect-emoji-group" data-cat="${i}">
      <div class="connect-emoji-group-title">${escapeHtml(c.name)}</div>
      <div class="connect-emoji-grid">
        ${c.emojis.map((e) => `<button class="connect-emoji-cell" type="button" data-emoji="${e}" aria-label="${e}">${e}</button>`).join('')}
      </div>
    </div>`).join('');
  el.innerHTML = `
    <div class="connect-emoji-tabs">${tabs}</div>
    <div class="connect-emoji-scroll">${groups}</div>
  `;
  document.body.appendChild(el);
  _picker = el;

  // Anchor above the trigger, clamped to the viewport.
  const r = anchorEl.getBoundingClientRect();
  const w = el.offsetWidth || 320;
  el.style.left = `${Math.max(8, Math.min(r.left, window.innerWidth - w - 8))}px`;
  el.style.bottom = `${Math.max(8, window.innerHeight - r.top + 8)}px`;

  const scroll = el.querySelector('.connect-emoji-scroll');
  el.addEventListener('click', (ev) => {
    const cell = ev.target.closest('.connect-emoji-cell');
    if (cell) { try { _onPick?.(cell.dataset.emoji); } catch (_) {} return; }
    const tab = ev.target.closest('.connect-emoji-tab');
    if (tab) {
      for (const t of el.querySelectorAll('.connect-emoji-tab')) t.classList.toggle('active', t === tab);
      const grp = el.querySelector(`.connect-emoji-group[data-cat="${tab.dataset.cat}"]`);
      if (grp && scroll) scroll.scrollTop = grp.offsetTop;
    }
  });

  setTimeout(() => {
    _outsideHandler = (ev) => {
      if (!_picker) return;
      if (_picker.contains(ev.target)) return;
      if (anchorEl.contains(ev.target)) return;
      closeEmojiPicker();
    };
    document.addEventListener('pointerdown', _outsideHandler, true);
    _keyHandler = (ev) => { if (ev.key === 'Escape') closeEmojiPicker(); };
    document.addEventListener('keydown', _keyHandler);
  }, 0);
}

export function closeEmojiPicker() {
  if (_outsideHandler) { document.removeEventListener('pointerdown', _outsideHandler, true); _outsideHandler = null; }
  if (_keyHandler) { document.removeEventListener('keydown', _keyHandler); _keyHandler = null; }
  if (_picker) { _picker.remove(); _picker = null; }
  _onPick = null;
}
