/**
 * onboarding.js — First-run welcome screen.
 *
 * Shows system detection results, discovered services, and feature highlights.
 * Dismissed state persisted to server via /api/config/ui (onboarding_completed).
 */
import { escapeHtml } from './app.js';

export async function checkOnboarding() {
  try {
    const res = await fetch('/api/config/ui');
    if (!res.ok) return;
    const data = await res.json();
    if (data.onboarding_completed === 'true' || data.onboarding_completed === true) return;
  } catch {
    return;
  }
  await _showOnboarding();
}

/**
 * Re-open the welcome / setup screen on demand (e.g. from Settings),
 * bypassing the ``onboarding_completed`` flag. Gives a user who dismissed
 * first-run setup a way back to the guided "connect a provider / open Model
 * Manager" path instead of being stranded.
 */
export async function reopenOnboarding() {
  await _showOnboarding();
}

async function _showOnboarding() {
  let hardware = {};
  let discovered = [];
  try {
    const [hwRes, discRes] = await Promise.all([
      fetch('/api/marketplace/hardware').catch(() => null),
      fetch('/api/models/status').catch(() => null),
    ]);
    if (hwRes?.ok) hardware = await hwRes.json();
    if (discRes?.ok) discovered = await discRes.json();
  } catch { /* proceed with empty data */ }

  const overlay = document.createElement('div');
  overlay.className = 'onboarding-overlay';
  overlay.innerHTML = `
    <div class="onboarding-card">
      <button class="onboarding-close" id="onboarding-close" title="Close">&times;</button>

      <div class="onboarding-logo">Augmentum</div>
      <div class="onboarding-tagline">Welcome to your private AI — chat, code, build, and play, in a way that's truly yours again.</div>

      <div class="onboarding-hw">
        <div class="onboarding-hw-title">Your System</div>
        ${hardware.gpu_available
          ? `<div class="onboarding-hw-row"><span>GPU</span><span>${escapeHtml(hardware.gpu_name || 'Detected')}</span></div>
             <div class="onboarding-hw-row"><span>VRAM</span><span>${hardware.gpu_vram_mb ? Math.round(hardware.gpu_vram_mb / 1024) + ' GB' : 'Unknown'}</span></div>`
          : '<div class="onboarding-hw-row"><span>GPU</span><span style="color:var(--text-secondary)">Not detected in container — configure via Marketplace</span></div>'
        }
        <div class="onboarding-hw-row"><span>Docker</span><span>${hardware.docker_available ? 'Connected' : 'Not available'}</span></div>
      </div>

      ${Array.isArray(discovered) && discovered.length ? `
        <div class="onboarding-discovered">
          <h3>Discovered Services</h3>
          ${discovered.map(s => `
            <div class="discovered-service">
              <span class="status-dot"></span>
              <div class="discovered-info">
                <div>${escapeHtml(s.name || s.id || 'Unknown')}</div>
                <div class="discovered-type">${escapeHtml(s.type || 'LLM Backend')} ${s.url ? '— ' + escapeHtml(s.url) : ''}</div>
              </div>
            </div>
          `).join('')}
        </div>
      ` : `
        <div class="onboarding-discovered">
          <h3>No AI services detected yet</h3>
          <p style="font-size:0.85rem;color:var(--text-secondary);margin-top:4px;">
            Already running Ollama, LM Studio, or another server? Connect it.
            Have GGUF models on disk, or want to download one? The Model
            Manager handles both.
          </p>
          <div class="onboarding-connect-actions">
            <button class="onboarding-btn" id="onboarding-add-provider">Connect a provider</button>
            <button class="onboarding-btn" id="onboarding-model-manager">Open Model Manager</button>
          </div>
        </div>
      `}

      <div class="onboarding-companion">
        <div class="onboarding-companion-title">A companion that does the legwork</div>
        <p class="onboarding-companion-body">
          Ask out loud or type, and it works the whole app for you — looks
          things up, digs into a question, writes up a doc or slide deck,
          makes images, finds something in your library. It can see what's on
          your screen, so you don't have to explain — then tells you what it
          found, in plain words. The memory stays on your hardware. Always yours.
        </p>
        <div class="onboarding-companion-examples">
          <span class="companion-example-chip">"Summarize this page"</span>
          <span class="companion-example-chip">"Make me a deck on solar power"</span>
          <span class="companion-example-chip">"Find something to watch"</span>
        </div>
        <div class="onboarding-companion-hint">Off by default — switch it on anytime in Settings → Companion.</div>
      </div>

      <div class="onboarding-features">
        <div class="feature-card">
          <strong>Memory</strong>
          Learns who you are across conversations. Nothing forgotten.
        </div>
        <div class="feature-card">
          <strong>Image Generation</strong>
          Background image gen while you chat. Local or cloud.
        </div>
        <div class="feature-card">
          <strong>Voice</strong>
          Real-time voice with speaker verification.
        </div>
        <div class="feature-card">
          <strong>Dream System</strong>
          AI reflects on conversations and evolves over time.
        </div>
        <div class="feature-card">
          <strong>Coding Agent</strong>
          Sandboxed workspaces with git checkpoints.
        </div>
        <div class="feature-card">
          <strong>Knowledge Packs</strong>
          Attach Wikipedia or custom knowledge to any chat.
        </div>
      </div>

      <div class="onboarding-actions">
        <button class="onboarding-btn primary" id="onboarding-start">Start Chatting</button>
      </div>
    </div>
  `;

  document.body.appendChild(overlay);

  // Multiple ways to dismiss
  overlay.querySelector('#onboarding-start').onclick = () => _dismiss(overlay);
  overlay.querySelector('#onboarding-close').onclick = () => _dismiss(overlay);

  // Empty-state shortcuts: land the user directly in the surface that
  // fixes "no model yet" — dismiss first so the overlay never stacks
  // under the opened panel. Buttons only exist when nothing was
  // discovered, hence the guards. Dynamic imports keep this lazy-loaded
  // module from dragging marketplace/models into the first paint.
  const addProviderBtn = overlay.querySelector('#onboarding-add-provider');
  if (addProviderBtn) {
    addProviderBtn.onclick = () => {
      _dismiss(overlay);
      import('./marketplace.js').then(m => m.openMarketplace()).catch(() => {});
    };
  }
  const modelManagerBtn = overlay.querySelector('#onboarding-model-manager');
  if (modelManagerBtn) {
    modelManagerBtn.onclick = () => {
      _dismiss(overlay);
      import('./models.js').then(m => m.openModelManager()).catch(() => {});
    };
  }
  document.addEventListener('keydown', function _escHandler(e) {
    if (e.key === 'Escape') {
      document.removeEventListener('keydown', _escHandler);
      _dismiss(overlay);
    }
  });
}

async function _dismiss(overlay) {
  if (overlay.classList.contains('dismissed')) return; // prevent double-dismiss
  overlay.classList.add('dismissed');
  setTimeout(() => overlay.remove(), 400);
  try {
    await fetch('/api/config/ui', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ onboarding_completed: 'true' }),
    });
  } catch { /* non-critical */ }
}
