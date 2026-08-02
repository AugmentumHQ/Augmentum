/**
 * NarrativeSurface — narrative chat with character context.
 *
 * Primary surface adopts existing DOM (backward compat).
 * Additional surfaces create their own DOM via the chat module classes.
 */

import { Surface } from '../surface.js';
import { SurfaceRegistry } from '../surface-registry.js';
import { app, escapeHtml, showToast } from '../app.js';
import { getSettings } from '../settings.js';

import { MessageRenderer } from '../chat/renderer.js';
import { ChatInput } from '../chat/input.js';
import { ChatStream } from '../chat/stream.js';
import { sessionStore } from '../chat/sessions.js';
import * as tree from '../chat/tree.js';
import { runImpersonate, fetchActivePersona } from '../chat/impersonate.js';

function getPrimaryInputArea() {
  return document.getElementById('chat-input')?.closest('.input-area') || null;
}

export class NarrativeSurface extends Surface {
  static type = 'narrative';

  constructor(id, config = {}) {
    super(id, config);
    this.characterId = config.characterId || '';
    this.characterName = config.characterName || '';
    this._sessionId = config.sessionId || null;
    this._isPrimary = config.primary || false;

    // Instance-scoped components
    this.renderer = null;
    this.input = null;
    this.stream = null;

    // Primary-only listener — mirrors the global active session while this
    // primary surface is visible so re-focusing the tab restores the user's
    // place. Removed on unmount.
    this._primarySessionListener = null;
    // Independent (non-primary) listener — picks up left-panel session
    // clicks so a focused non-primary narrative tab actually switches
    // which story the user is reading. Without it the surface stays
    // bound to whichever session it was created with.
    this._sessionChangedListener = null;
    // Set for the duration of a continue-in-place stream so the complete
    // handler appends onto the existing assistant node instead of branching.
    this._continuingNodeId = null;
  }

  mount(container) {
    super.mount(container);
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.height = '100%';
    container.style.overflow = 'hidden';

    if (this._isPrimary) {
      this._mountPrimary(container);
    } else {
      this._mountIndependent(container);
    }
  }

  _mountPrimary(container) {
    const chatScroll = document.getElementById('chat-scroll');
    const inputArea = getPrimaryInputArea();
    if (chatScroll && inputArea) {
      chatScroll.classList.remove('hidden');
      inputArea.classList.remove('hidden');
      container.appendChild(chatScroll);
      container.appendChild(inputArea);
    }

    // Mirror the active session + character while primary is visible so
    // switching away and back doesn't lose context.
    this._sessionId = sessionStore.getActiveId();
    this._primarySessionListener = (e) => {
      if (!this.isActive) return;
      this._sessionId = e.detail?.sessionId || null;
      // Pull character info so getTitle/getContext stay accurate
      if (this._sessionId) {
        const s = sessionStore.get(this._sessionId);
        if (s) {
          this.characterId = s.characterId || '';
          this.characterName = s.characterName || s.title || '';
          this.emit('surface:titleChanged', { title: this.getTitle() });
        }
      }
    };
    document.addEventListener('augmentum:session-changed', this._primarySessionListener);
  }

  _mountIndependent(container) {
    if (this._sessionId) {
      // Inherited session — backfill character fields onto the surface so
      // the tab title, left panel selection, and background all reflect
      // the adopted card. Without this the surface boots into the empty
      // character-grid state even though the session has content.
      const session = sessionStore.get(this._sessionId);
      if (session) {
        if (!this.characterId && session.characterId) this.characterId = session.characterId;
        if (!this.characterName) this.characterName = session.characterName || session.title || '';
      }
    } else {
      this._sessionId = sessionStore.create('narrative');
      const session = sessionStore.get(this._sessionId);
      if (session && this.characterId) {
        session.characterId = this.characterId;
        session.characterName = this.characterName;
        session.title = this.characterName || 'Story';
        sessionStore.save(session.id);
      }
    }

    this.renderer = new MessageRenderer({
      mode: 'narrative',
      onAction: (action, nodeId, data) => this._handleAction(action, nodeId, data),
      highlightHooks: {},
    });
    this.renderer.createDOM(container);

    // Pass `surface: this` — it's what ChatInput keys the per-surface
    // toolbar build on (without it story tabs shipped the bare composer),
    // and what ChatStream keys the skip-global-flow-headers gate on.
    this.input = new ChatInput({
      surface: this,
      onSend: (text, images, docs) => this._handleSend(text, images, docs),
      onStop: () => this.stream?.abort(),
    });
    this.input.createDOM(container);

    this.stream = new ChatStream({
      surface: this,
      onContent: (text) => this.renderer.appendToStreaming(text),
      onMeta: (meta) => this._handleMeta(meta),
      onComplete: (result) => this._handleStreamComplete(result),
      onError: (err) => {
        showToast('Stream error: ' + (err.message || 'Unknown'), 'error');
        this.input.setStreaming(false);
      },
    });

    const session = sessionStore.get(this._sessionId);
    if (session) {
      sessionStore.ensureLoaded(this._sessionId).then(() => {
        const loaded = sessionStore.get(this._sessionId);
        if (loaded) this.renderer.renderMessages(loaded);
      });
    }

    // Picks up left-panel session list clicks (and character grid clicks
    // that activate a session) so a focused non-primary narrative tab
    // re-renders for the new session instead of staying frozen on the
    // session it was opened with. isActive gate avoids stealing updates
    // from another focused narrative tab. Mode gate avoids switching
    // this surface to a non-narrative session if the user clicks a chat
    // session in the list while a narrative tab is focused.
    this._sessionChangedListener = (e) => {
      if (!this.isActive) return;
      const newId = e.detail?.sessionId;
      if (!newId || newId === this._sessionId) return;
      const s = sessionStore.get(newId);
      if (!s || (s.mode || 'passthrough') !== 'narrative') return;
      this._sessionId = newId;
      this.characterId = s.characterId || '';
      this.characterName = s.characterName || s.title || '';
      this.emit('surface:titleChanged', { title: this.getTitle() });
      sessionStore.ensureLoaded(newId).then(() => {
        const loaded = sessionStore.get(newId);
        if (loaded && this.renderer) this.renderer.renderMessages(loaded);
      });
    };
    document.addEventListener('augmentum:session-changed', this._sessionChangedListener);
  }

  activate() {
    super.activate();
    // Promote this tab's session to global-active so left panel (character
    // grid + recent chats strip) and inspector refresh to reflect this tab.
    // sessionStore.setActiveId no-ops on unchanged ids.
    if (this._sessionId) {
      sessionStore.setActiveId(this._sessionId);
    }
    if (!this._isPrimary && this.renderer && this._sessionId) {
      const session = sessionStore.get(this._sessionId);
      if (session) this.renderer.renderMessages(session);
    }
    // Primary narrative adopts the legacy #chat-scroll. The browser
    // drops its scroll position when the surface-container opacity
    // flips, so tabbing back to narrative lands the user at the top
    // even though the DOM content is preserved. Snap back to bottom.
    if (this._isPrimary && typeof window !== 'undefined') {
      import('../chat/index.js').then(mod => {
        mod.scrollPrimaryToBottom?.();
      }).catch(() => {});
    }
  }

  unmount() {
    if (this._primarySessionListener) {
      document.removeEventListener('augmentum:session-changed', this._primarySessionListener);
      this._primarySessionListener = null;
    }
    if (this._sessionChangedListener) {
      document.removeEventListener('augmentum:session-changed', this._sessionChangedListener);
      this._sessionChangedListener = null;
    }
    if (this._isPrimary) {
      // Re-apply `hidden` when returning singletons to main-area — same reason
      // as ChatSurface.unmount and CoderSurface.unmount: unhidden orphans become
      // flex:1 siblings of #surface-grid and split the layout 50/50.
      const mainArea = document.querySelector('.main-area');
      if (this._container) {
        const chatScroll = this._container.querySelector('#chat-scroll');
        const inputArea = this._container.querySelector('.input-area');
        if (mainArea && chatScroll) {
          chatScroll.classList.add('hidden');
          mainArea.appendChild(chatScroll);
        }
        if (mainArea && inputArea) {
          inputArea.classList.add('hidden');
          mainArea.appendChild(inputArea);
        }
      }
    } else {
      this.stream?.abort();
      this.renderer?.destroy();
      this.input?.destroy();
      this.renderer = null;
      this.input = null;
      this.stream = null;
    }
    super.unmount();
  }

  async _handleSend(text, images, docs) {
    if (!text && (!images || images.length === 0)) return;
    if (this.stream?.isActive()) return;

    const session = sessionStore.get(this._sessionId);
    if (!session) return;

    this.input.setStreaming(true);

    const parentId = session.activeLeafId || null;
    const userNode = tree.addChildNode(session, parentId, 'user', text);
    if (images && images.length > 0) userNode.images = images;
    session.activeLeafId = userNode.id;
    sessionStore.save(session.id);

    this.renderer.renderMessages(session);
    this.renderer.createStreamingMessage(session);

    await this.stream.send(session, {
      model: app.state.currentModel || '',
      mode: 'narrative',
      tools: 'none',
    });
  }

  async _continueMessage(session, nodeId) {
    if (this.stream?.isActive()) return;
    // Continue = extend the trailing assistant message in place (matches chat).
    const lastNode = session.tree?.[session.activeLeafId];
    if (!lastNode || lastNode.role !== 'assistant') return;
    if (lastNode.interrupted) {
      delete lastNode.interrupted;
      delete lastNode.error_message;
      delete lastNode.error_kind;
    }
    sessionStore.save(session.id);

    const resumed = this.renderer.resumeStreamingMessage(session.activeLeafId);
    if (!resumed) this.renderer.createStreamingMessage(session);
    this._continuingNodeId = session.activeLeafId;
    this.input.setStreaming(true);
    await this.stream.send(session, {
      model: app.state.currentModel || '',
      mode: 'narrative',
      tools: 'none',
      continueLastAssistant: true,
    });
  }

  _handleMeta(meta) {
    if (!this.renderer) return;
    if (meta.model_thinking_delta) {
      this.renderer.appendModelThinking(meta.model_thinking_delta);
    }
    if (meta.regex_transformed) {
      this.renderer.replaceStreamedContent(meta.regex_transformed);
    }
    const hasPerfMeta = [
      meta.tokens_per_second,
      meta.context_length,
      meta.context_used,
      meta.prompt_tokens,
      meta.eval_tokens,
      meta.ttft_ms,
      meta.total_duration_ms,
      meta.eval_duration_ms,
    ].some(v => v != null);
    if (hasPerfMeta) {
      this.renderer.updateStreamMetrics({
        tokens_per_second: meta.tokens_per_second,
        context_length: meta.context_length,
        context_used: meta.context_used,
        prompt_tokens: meta.prompt_tokens,
        eval_tokens: meta.eval_tokens,
        ttft_ms: meta.ttft_ms,
        total_duration_ms: meta.total_duration_ms,
        eval_duration_ms: meta.eval_duration_ms,
      });
    }
    if (meta.group_speaker) this.renderer.setStreamSpeaker(meta.group_speaker);
  }

  _handleStreamComplete(result) {
    if (!this.renderer || !this.input) return;
    this.input.setStreaming(false);

    // Continue-in-place: capture + clear the flag up front so it can never
    // leak into the next turn (even on abort / early return).
    const continuingNodeId = this._continuingNodeId;
    this._continuingNodeId = null;

    if (result.aborted) {
      this.renderer.finalizeStreaming();
      return;
    }

    const session = sessionStore.get(this._sessionId);
    if (!session) return;

    const rawContent = this.renderer.getStreamingRawContent();

    // Continuation: streamingEl IS the existing assistant node — write
    // [partial + streamed] back onto it instead of creating a fresh node.
    if (continuingNodeId) {
      const existingNode = session.tree?.[continuingNodeId];
      if (existingNode && existingNode.role === 'assistant') {
        if (rawContent) existingNode.content = rawContent;
        const reasoning = this.renderer.collectReasoningData();
        if (reasoning && reasoning.thinking) {
          const prior = (existingNode.reasoning && existingNode.reasoning.thinking) || '';
          existingNode.reasoning = {
            ...(existingNode.reasoning || {}),
            thinking: prior + reasoning.thinking,
          };
        }
        this.renderer.finalizeStreaming(session);
        sessionStore.save(session.id);
        return;
      }
      // Existing node lost — fall through to fresh-node creation.
    }

    if (rawContent) {
      const parentId = session.activeLeafId;
      const assistantNode = tree.addChildNode(session, parentId, 'assistant', rawContent);
      const metrics = this.renderer._streamMetrics;
      if (metrics.tps > 0) assistantNode.tokens_per_second = metrics.tps;
      if (metrics.promptTokens > 0) assistantNode.prompt_tokens = metrics.promptTokens;
      if (metrics.evalTokens > 0) assistantNode.eval_tokens = metrics.evalTokens;
      if (metrics.contextLen > 0) {
        assistantNode.context_length = metrics.contextLen;
        assistantNode.context_used = metrics.contextUsed;
      }
      if (metrics.ttftMs > 0) assistantNode.ttft_ms = metrics.ttftMs;
      if (metrics.totalDurationMs > 0) assistantNode.total_duration_ms = metrics.totalDurationMs;
      if (metrics.evalDurationMs > 0) assistantNode.eval_duration_ms = metrics.evalDurationMs;
      const speaker = this.renderer.getStreamSpeaker();
      if (speaker) assistantNode.speakerName = speaker;
      // Persist model-native thinking (reasoning_content / <think> tokens)
      // on the node so it survives refresh + server restart. Important for
      // later training-data extraction — otherwise the thinking you saw
      // during streaming is gone as soon as the next renderMessages runs.
      const reasoning = this.renderer.collectReasoningData();
      if (reasoning && ((reasoning.phases && reasoning.phases.length > 0) || reasoning.thinking)) {
        assistantNode.reasoning = reasoning;
      }
      session.activeLeafId = assistantNode.id;
      this.renderer.setStreamingNodeId(assistantNode.id);
    }

    // Pass session so finalizeStreaming can add the branch-swipe indicator
    // when this node has siblings (e.g. after a regenerate).
    this.renderer.finalizeStreaming(session);
    sessionStore.save(session.id);
  }

  _handleAction(action, nodeId, data) {
    const session = sessionStore.get(this._sessionId);
    if (!session) return;

    switch (action) {
      case 'branch':
        tree.switchToSibling(session, nodeId, data);
        this.renderer.renderMessages(session);
        sessionStore.save(session.id);
        break;
      case 'regenerate': {
        const node = session.tree[nodeId];
        if (!node || node.role !== 'assistant') break;
        session.activeLeafId = node.parentId;
        sessionStore.save(session.id);
        this.renderer.renderMessages(session);
        this.renderer.createStreamingMessage(session);
        this.input.setStreaming(true);
        this.stream.send(session, {
          model: app.state.currentModel || '',
          mode: 'narrative',
          tools: 'none',
        });
        break;
      }
      case 'continue': {
        this._continueMessage(session, nodeId);
        break;
      }
      case 'edit': {
        // Shared inline-edit UI, owned by this surface's own renderer.
        const node = session.tree[nodeId];
        if (node) this.renderer.beginInlineEdit(nodeId, node.content);
        break;
      }
      case 'cancel-edit': {
        this.renderer.cancelInlineEdit();
        break;
      }
      case 'save-edit': {
        const node = session.tree[nodeId];
        if (!node) break;
        if (node.role === 'assistant') {
          node.content = data.content;
          sessionStore.save(session.id);
          this.renderer.renderMessages(session);
        } else {
          // User edit → branch and re-stream the story from the edited turn.
          const parentId = node.parentId;
          const newUserNode = tree.addChildNode(session, parentId, 'user', data.content);
          session.activeLeafId = newUserNode.id;
          sessionStore.save(session.id);
          this.renderer.renderMessages(session);
          this.renderer.createStreamingMessage(session);
          this.input.setStreaming(true);
          this.stream.send(session, {
            model: app.state.currentModel || '',
            mode: 'narrative',
            tools: 'none',
          });
        }
        break;
      }
      case 'delete': {
        const descendants = tree.countDescendants(session, nodeId);
        if (descendants > 0 && !confirm(`Delete this message and ${descendants} replies?`)) return;
        const node = session.tree[nodeId];
        const parentId = node?.parentId;
        tree.removeNodeAndDescendants(session, nodeId);
        session.activeLeafId = parentId ? tree.getDeepestLeaf(session, parentId) : null;
        sessionStore.save(session.id);
        this.renderer.renderMessages(session);
        break;
      }
      case 'send':
        if (data?.text) this._handleSend(data.text, [], []);
        break;
      case 'tts': {
        const node = nodeId && session ? session.tree?.[nodeId] : null;
        import('../chat.js').then(m => {
          m.ttsPlayMessage(data?.text, data?.button, { speakerName: node?.speakerName || '' });
        });
        break;
      }
      case 'impersonate': {
        this._impersonateMessage(session);
        break;
      }
    }
  }

  async _impersonateMessage(session) {
    if (this.stream?.isActive()) return;
    const persona = await fetchActivePersona();
    await runImpersonate(session, {
      mode: 'narrative',
      sessionId: this._sessionId,
      userName: persona?.name || 'User',
      charName: this.characterName || session.title || 'the character',
      personaDesc: persona?.description || '',
      onStart: () => {
        this.input.setStreaming(true);
        this.input.setValue('');
      },
      onText: (acc) => this.input.setValue(acc),
      onEnd: () => {
        this.input.setStreaming(false);
        this.input.focus();
      },
    });
  }

  getTitle() { return this.characterName || 'Story'; }
  getIcon() { return 'narrative'; }

  // See ChatSurface.isCloseable — same singleton-adoption landmine.
  isCloseable() { return !this._isPrimary; }

  getContext() {
    return {
      type: 'narrative',
      id: this.id,
      mode: 'narrative',
      sessionId: this._sessionId,
      summary: this.characterName ? `Story with ${this.characterName}` : 'Narrative session',
      character: { name: this.characterName, id: this.characterId },
      capabilities: ['narrative', 'scene-image', 'tts-character', 'illustrate'],
    };
  }

  getState() {
    return {
      ...super.getState(),
      characterId: this.characterId,
      characterName: this.characterName,
      sessionId: this._sessionId,
      primary: this._isPrimary,
    };
  }

  restoreState(state) {
    super.restoreState(state);
    if (state.characterId) this.characterId = state.characterId;
    if (state.characterName) this.characterName = state.characterName;
    if (state.sessionId) this._sessionId = state.sessionId;
    if (state.primary !== undefined) this._isPrimary = state.primary;
  }
}

SurfaceRegistry.register('narrative', NarrativeSurface);
