/**
 * avatar-drawer.js — Slide-out panel for avatar experience
 *
 * Four tabs: Transcript, Tools, Character, Atmosphere.
 * Slides in from right edge. Auto-peeks on long responses or tool output.
 */

import { escapeHtml } from './app.js';

export class DrawerManager {
  constructor(container, onToggle) {
    this._container = container;
    this._onToggle = onToggle;
    this._open = false;
    this._activeTab = 'transcript';
    this._peekTimer = null;
    this._toolBadge = 0;

    this._buildDOM();
    this._bindEvents();
  }

  _buildDOM() {
    this._edge = document.createElement('div');
    this._edge.className = 'avatar-drawer-edge';
    this._container.appendChild(this._edge);

    this._panel = document.createElement('div');
    this._panel.className = 'avatar-drawer-panel';
    this._panel.innerHTML = `
      <div class="avatar-drawer-tabs">
        <button class="avatar-drawer-tab active" data-tab="transcript">Transcript</button>
        <button class="avatar-drawer-tab" data-tab="tools">Tools<span class="avatar-drawer-badge" hidden></span></button>
        <button class="avatar-drawer-tab" data-tab="character">Character</button>
        <button class="avatar-drawer-tab" data-tab="atmosphere">Atmosphere</button>
      </div>
      <div class="avatar-drawer-content">
        <div class="avatar-drawer-pane active" data-pane="transcript">
          <div class="avatar-drawer-transcript"></div>
        </div>
        <div class="avatar-drawer-pane" data-pane="tools">
          <div class="avatar-drawer-tools"></div>
        </div>
        <div class="avatar-drawer-pane" data-pane="character">
          <div class="avatar-drawer-character"></div>
        </div>
        <div class="avatar-drawer-pane" data-pane="atmosphere">
          <div class="avatar-drawer-atmo-controls">
            <label>Mood
              <select class="avatar-atmo-mood">
                <option value="auto">Auto</option>
                <option value="warm">Warm</option>
                <option value="cool">Cool</option>
                <option value="neutral">Neutral</option>
              </select>
            </label>
            <label>Particles
              <input type="range" class="avatar-atmo-particles" min="0" max="100" value="50">
            </label>
            <label>Glow
              <input type="range" class="avatar-atmo-glow" min="0" max="100" value="50">
            </label>
          </div>
        </div>
      </div>
    `;
    this._container.appendChild(this._panel);

    this._transcriptEl = this._panel.querySelector('.avatar-drawer-transcript');
    this._toolsEl = this._panel.querySelector('.avatar-drawer-tools');
    this._characterEl = this._panel.querySelector('.avatar-drawer-character');
    this._badgeEl = this._panel.querySelector('.avatar-drawer-badge');
  }

  _bindEvents() {
    this._edge.addEventListener('click', () => this.toggle());

    this._panel.querySelectorAll('.avatar-drawer-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        this._switchTab(tab.dataset.tab);
      });
    });

    document.addEventListener('keydown', this._onKey = (e) => {
      if (e.key === 'Escape' && this._open) {
        this.toggle(false);
      }
    });

    let startX = 0;
    this._panel.addEventListener('touchstart', (e) => { startX = e.touches[0].clientX; });
    this._panel.addEventListener('touchend', (e) => {
      const dx = e.changedTouches[0].clientX - startX;
      if (dx > 60) this.toggle(false);
    });
    this._edge.addEventListener('touchstart', (e) => { startX = e.touches[0].clientX; });
    this._edge.addEventListener('touchend', (e) => {
      const dx = e.changedTouches[0].clientX - startX;
      if (dx < -40) this.toggle(true);
    });
  }

  _switchTab(name) {
    this._activeTab = name;
    this._panel.querySelectorAll('.avatar-drawer-tab').forEach(t =>
      t.classList.toggle('active', t.dataset.tab === name));
    this._panel.querySelectorAll('.avatar-drawer-pane').forEach(p =>
      p.classList.toggle('active', p.dataset.pane === name));

    if (name === 'tools') {
      this._toolBadge = 0;
      this._badgeEl.hidden = true;
    }
  }

  toggle(forceState) {
    this._open = forceState !== undefined ? forceState : !this._open;
    this._panel.classList.toggle('open', this._open);
    this._edge.classList.toggle('open', this._open);
    if (this._onToggle) this._onToggle(this._open);
  }

  addMessage(role, text) {
    const div = document.createElement('div');
    div.className = `avatar-drawer-msg avatar-drawer-msg-${escapeHtml(role)}`;
    div.innerHTML = `<span class="avatar-drawer-msg-role">${escapeHtml(role)}</span> ${escapeHtml(text)}`;
    this._transcriptEl.appendChild(div);
    this._transcriptEl.scrollTop = this._transcriptEl.scrollHeight;

    if (role === 'ai' && text.length > 200 && !this._open) {
      this._autoPeek();
    }
  }

  addToolResult(toolName, content) {
    const div = document.createElement('div');
    div.className = 'avatar-drawer-tool-result';
    div.innerHTML = `<div class="avatar-drawer-tool-name">${escapeHtml(toolName)}</div><div class="avatar-drawer-tool-content">${escapeHtml(content)}</div>`;
    this._toolsEl.appendChild(div);
    this._toolsEl.scrollTop = this._toolsEl.scrollHeight;

    if (!this._open || this._activeTab !== 'tools') {
      this._toolBadge++;
      this._badgeEl.textContent = this._toolBadge;
      this._badgeEl.hidden = false;
    }
    if (!this._open) this._autoPeek();
  }

  setCharacterInfo(name, description, thumbnailUrl) {
    this._characterEl.innerHTML = `
      <div class="avatar-drawer-char-header">
        ${thumbnailUrl ? `<img src="${escapeHtml(thumbnailUrl)}" class="avatar-drawer-char-thumb" alt="">` : ''}
        <h3>${escapeHtml(name)}</h3>
      </div>
      <p>${escapeHtml(description || '')}</p>
    `;
  }

  _autoPeek() {
    this._panel.classList.add('peeking');
    clearTimeout(this._peekTimer);
    this._peekTimer = setTimeout(() => {
      if (!this._open) this._panel.classList.remove('peeking');
    }, 3000);
  }

  dispose() {
    document.removeEventListener('keydown', this._onKey);
    clearTimeout(this._peekTimer);
    this._edge.remove();
    this._panel.remove();
  }
}
