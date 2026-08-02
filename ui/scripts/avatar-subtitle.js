/**
 * avatar-subtitle.js — Cinematic subtitle renderer for avatar experience
 *
 * Shows the current spoken sentence in a frosted-glass bar at bottom center.
 * Crossfades between sentences. Auto-hides after speech ends.
 */

export class SubtitleRenderer {
  constructor(container) {
    this._container = container;
    this._currentText = '';
    this._hideTimer = 0;
    this._visible = false;

    // Create DOM elements
    this._bar = document.createElement('div');
    this._bar.className = 'avatar-subtitle-bar';
    this._bar.setAttribute('aria-live', 'polite');

    this._aiText = document.createElement('div');
    this._aiText.className = 'avatar-subtitle-ai';
    this._bar.appendChild(this._aiText);

    this._userText = document.createElement('div');
    this._userText.className = 'avatar-subtitle-user';

    container.appendChild(this._userText);
    container.appendChild(this._bar);
  }

  setAISentence(sentence) {
    if (!sentence || sentence === this._currentText) return;
    this._currentText = sentence;
    this._hideTimer = 0;

    this._aiText.classList.add('fading');
    setTimeout(() => {
      this._aiText.textContent = sentence;
      this._aiText.classList.remove('fading');
    }, 200);

    this._show();
  }

  setUserSpeech(text) {
    if (!text) {
      this._userText.classList.remove('visible');
      return;
    }
    this._userText.textContent = text;
    this._userText.classList.add('visible');
  }

  update(dt, isSpeaking) {
    if (!isSpeaking && this._visible) {
      this._hideTimer += dt;
      if (this._hideTimer >= 2.0) {
        this._hide();
      }
    } else if (isSpeaking) {
      this._hideTimer = 0;
      if (!this._visible && this._currentText) {
        this._show();
      }
    }
  }

  _show() {
    this._visible = true;
    this._bar.classList.add('visible');
  }

  _hide() {
    this._visible = false;
    this._bar.classList.remove('visible');
    this._userText.classList.remove('visible');
    this._currentText = '';
  }

  dispose() {
    this._bar.remove();
    this._userText.remove();
  }
}
