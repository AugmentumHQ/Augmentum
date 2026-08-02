/* connect/thread-panel.js — Connect text-messaging UI.
 *
 * Surfaces a docked panel (right side) with two columns:
 *
 *   ┌────────────┬──────────────────────────┐
 *   │ Threads    │ Active thread            │
 *   │ ── list    │  • message bubbles       │
 *   │            │  • composer at bottom    │
 *   └────────────┴──────────────────────────┘
 *
 * The panel is created on-demand the first time it's opened (via
 * command palette: "Connect: Open messages") and hidden between
 * uses. It listens for WS-driven events from connect/messages.js
 * so inbound messages bump unread counts + append to the active
 * view in real time.
 *
 * Phase 1 affordances:
 *
 *   - Thread list with peer DID + last message preview + unread badge
 *   - "+ New" action: prompts for a peer DID, opens a fresh thread
 *   - Active-thread view: scrollable message log + composer
 *   - Enter to send; Shift+Enter for newline; auto-marks-as-read on view
 *   - Peer-side messages render on the left, ours on the right
 *
 * Deferred (Phase 2+): edit / delete UI, reply threading, voice-note
 * recording, Augmentum-object embed rendering, infinite-scroll
 * pagination beyond the first 100.
 */

import { escapeHtml, showToast } from '../app.js';
import { getSettings } from '../settings.js';
import { registerCommand } from '../command-palette.js';
import {
  broadcastThreadChanged,
  initBroadcast,
  onBroadcast,
} from './broadcast.js';
import { getOnlinePeers, getPeerStatus, getWelcome } from './client.js';
import { icon } from './icons.js';
import {
  addContact,
  attachmentUrl,
  catchUpThread,
  clearThreadMessages,
  ensureConnectReady,
  initConnectMessaging,
  listCalls,
  listContacts,
  listDirectory,
  listMessages,
  listThreads,
  markThreadRead,
  onOutboxChange,
  outboxDiscard,
  outboxFind,
  outboxFindFailed,
  peerSubtitle,
  reactToMessage,
  removeContact,
  resolvePeerName,
  retryFailedSend,
  searchPeers,
  sendMessage,
  sendTyping,
  setPeerBlocked,
  setThreadCursor,
  setThreadFlags,
  uploadAttachment,
} from './messages.js';

const TYPING_DEBOUNCE_MS = 800;
const TYPING_INDICATOR_STALE_MS = 4000;

let _panel = null;
let _activeThreadId = '';
let _threadCache = new Map();     // thread_id → thread object
let _messageCache = new Map();    // thread_id → messages array (newest-first)
let _initialized = false;
let _replyContext = null;         // active reply: { message_id, sender_did, body }
let _typingState = new Map();     // thread_id → { peerDid, expiresAt }
let _typingTimer = null;          // setTimeout id for sending typing_stop
// Merged recents (Phase 1b): calls fold into the same list/timeline as
// messages so each person is ONE conversation. _recentCalls is the raw
// newest-first listCalls window; _callsByPeer indexes the newest call
// per peer for the recents-row preview + sort key.
let _recentCalls = [];            // newest-first call rows (all peers)
let _callsByPeer = new Map();     // peer_did → newest call row
let _activePeerDid = '';          // peer_did of the open thread (for timeline merge)
let _lightbox = null;             // active fullscreen image viewer overlay
let _lightboxKeyHandler = null;
let _jumpNewCount = 0;            // unseen messages while scrolled up
let _renderedMsgCount = 0;        // message count at last render (delta detect)
let _unreadAnchorId = '';         // message_id the "new messages" divider sits above
let _voiceRecHandle = null;       // active voice-recording handle (or null)
let _voiceElapsed = 0;            // seconds recorded so far
let _stagedFiles = [];            // attachments queued for send: {id,file,url,kind}
let _stagedSeq = 0;               // monotonic id for staged entries
let _updateSendState = null;      // ref to the composer send-button morph fn
let _isTypingActive = false;      // whether we currently signal "typing"
// rAF-coalesced message render: many WS events can fire in a single
// tick (catch-up burst, sender + receipt in the same flush, etc.),
// and the old code did an innerHTML rebuild for each. Coalesce them
// into a single render per animation frame.
let _pendingRenderTid = '';
let _pendingRenderRaf = 0;

// ── Init ────────────────────────────────────────────────────────

let _deferredRetryArmed = false;

export function initConnectMessagingUI() {
  if (_initialized) return;
  if (!_isEnabled()) {
    // Server-side config + modal toggle can flip connectEnabled true
    // after boot's synchronous init pass already returned early. Arm
    // the retry listeners once and bail.
    if (!_deferredRetryArmed) {
      _deferredRetryArmed = true;
      const retry = () => {
        if (_initialized || !_isEnabled()) return;
        try { initConnectMessagingUI(); }
        catch (e) { console.warn('[connect-msg] deferred init failed', e); }
      };
      window.addEventListener('augmentum:settings-loaded', retry);
      window.addEventListener('augmentum:connect-enabled', retry);
    }
    return;
  }
  _initialized = true;

  // Wire WS events from messages.js to the panel's renderer.
  initConnectMessaging();

  registerCommand({
    id: 'connect.openMessages',
    label: 'Connect: Open messages',
    hint: 'Show the Connect messaging panel',
    group: 'Connect',
    keywords: 'connect chat dm text messages thread inbox',
    run: () => import('./home.js').then((m) => m.openConnectHome('chats')),
    when: () => _isEnabled(),
  });

  // Inbound message stream.
  window.addEventListener('augmentum:connect-message-received', _onReceived);
  window.addEventListener('augmentum:connect-message-edit', _onEdited);
  window.addEventListener('augmentum:connect-message-delete', _onDeleted);
  window.addEventListener('augmentum:connect-message-read', _onReadReceipt);
  window.addEventListener('augmentum:connect-message-delivered', _onDeliveredReceipt);
  window.addEventListener('augmentum:connect-message-react', _onReactionEvent);
  window.addEventListener('augmentum:connect-typing-start', _onTypingStart);
  window.addEventListener('augmentum:connect-typing-stop', _onTypingStop);
  // A call ending changes the merged recents (new/updated call row) and
  // the open conversation's timeline. Refresh both when the panel is live.
  window.addEventListener('augmentum:connect-call-ended', _onCallEndedRefresh);
  // Catch-up on reconnect — for every thread we've already opened
  // in this tab, pull messages newer than our persisted cursor.
  window.addEventListener('augmentum:connect-reconnected', _onReconnected);
  // Live tick updates: when the outbox state mutates (enqueue, ack,
  // mark-failed, retry, discard), re-render the active thread so
  // 'pending' → 'sent' → 'delivered' transitions are visible without
  // waiting for the next WS event.
  onOutboxChange(_onOutboxChanged);

  // Cross-tab sync: when a sibling tab on the same account mutates a
  // thread (send/edit/delete/react/read), the WS path won't echo the
  // event back to us — we don't receive our own outbound traffic. The
  // broadcast bus carries an invalidation hint; we react by pulling
  // any messages newer than our local cursor for the affected thread.
  initBroadcast();
  onBroadcast((msg) => {
    if (msg?.type !== 'thread-changed') return;
    const tid = String(msg.thread_id || '');
    if (!tid) return;
    _onSiblingThreadChange(tid).catch((err) => {
      console.warn('connect: sibling-thread refresh failed', err);
    });
  });

  // Notification 'open_thread' action → open the panel + thread.
  window.addEventListener('augmentum:notification-action', (evt) => {
    const detail = evt.detail || {};
    const n = detail.notification || {};
    if (!n.channel_id || !n.channel_id.startsWith('connect.message')) return;
    if (detail.actionId !== 'open_thread') return;
    const tid = n.payload?.thread_id;
    if (!tid) return;
    openMessagingPanel(tid);
  });

  // Expose for console / debug.
  window.augmentumConnectMessages = {
    open: openMessagingPanel,
    send: sendMessage,
  };
}

// ── Public-ish surface ──────────────────────────────────────────

export async function openMessagingPanel(threadId = '') {
  if (!_isEnabled()) {
    showToast('Connect is disabled', 'warning');
    return;
  }
  if (!_panel) _ensurePanel();
  _panel.classList.remove('hidden');
  // Refresh thread list each time the panel opens. Cheap (one fetch)
  // and avoids stale data after a previous session.
  try {
    await ensureConnectReady().catch(() => {});  // best-effort
    await _loadThreadList();
    if (threadId) {
      await _openThread(threadId);
    } else if (_activeThreadId) {
      await _openThread(_activeThreadId);
    }
  } catch (err) {
    console.warn('connect: open messages failed', err);
  }
}

/**
 * Open (or start) a message thread with a specific peer. Used by the
 * Connect picker's per-contact chat icon — clicking the icon should
 * land the user inside the conversation, not at the panel's empty
 * state. If no thread exists yet a placeholder is created lazily by
 * `_openOrCreateThreadForPeer`; the first send mints the real
 * thread_id server-side.
 */
export async function openMessagingPanelForPeer(peerDid) {
  if (!peerDid) return;
  if (!_isEnabled()) {
    showToast('Connect is disabled', 'warning');
    return;
  }
  if (!_panel) _ensurePanel();
  _panel.classList.remove('hidden');
  try {
    await ensureConnectReady().catch(() => {});
    await _loadThreadList();
    await _openOrCreateThreadForPeer(peerDid);
  } catch (err) {
    console.warn('connect: open thread for peer failed', err);
  }
}

export function closeMessagingPanel() {
  if (_panel) _panel.classList.add('hidden');
}

/**
 * Embed the messaging master-detail inside a host container (the
 * Connect home's Chats section) rather than floating it on <body>.
 *
 * The whole `.connect-thread-panel` subtree is relocated into `host`
 * and tagged `.is-embedded`; because every internal lookup goes
 * through `_panel.querySelector(...)`, the subtree stays intact and
 * all composer / edit / voice-note logic keeps working unchanged. CSS
 * (`.connect-thread-panel.is-embedded`) strips the floating-card
 * chrome so it fills the home content region.
 *
 * Idempotent: re-mounting into the same host just refreshes the list.
 */
/**
 * Open (or create) a thread for a peer inside the already-embedded
 * Chats panel. Used by the home's People section after switching to
 * Chats. No-op if the panel isn't mounted yet.
 */
export async function openThreadForPeer(peerDid) {
  if (!peerDid || !_panel) return;
  try {
    await _openOrCreateThreadForPeer(peerDid);
  } catch (err) {
    console.warn('connect: openThreadForPeer failed', err);
  }
}

export async function mountMessagingInto(host, { threadId = '' } = {}) {
  if (!host) return;
  if (!_isEnabled()) {
    showToast('Connect is disabled', 'warning');
    return;
  }
  if (!_panel) _ensurePanel();
  _panel.classList.add('is-embedded');
  _panel.classList.remove('hidden');
  if (_panel.parentElement !== host) host.appendChild(_panel);
  try {
    await ensureConnectReady().catch(() => {});
    await _loadThreadList();
    if (threadId) {
      await _openThread(threadId);
    } else if (_activeThreadId) {
      await _openThread(_activeThreadId);
    }
  } catch (err) {
    console.warn('connect: mount messages failed', err);
  }
}

// ── DOM construction ────────────────────────────────────────────

function _ensurePanel() {
  if (_panel) return _panel;
  const el = document.createElement('div');
  el.className = 'connect-thread-panel hidden';
  el.setAttribute('role', 'dialog');
  el.setAttribute('aria-label', 'Connect messages');
  el.innerHTML = `
    <div class="connect-thread-panel-card">
      <div class="connect-thread-panel-header">
        <div class="connect-thread-panel-title">Messages</div>
        <div class="connect-thread-panel-actions">
          <button class="connect-thread-panel-new" type="button" title="New conversation">+ New</button>
          <button class="connect-thread-panel-close" type="button" aria-label="Close">&#x2715;</button>
        </div>
      </div>
      <div class="connect-thread-panel-body">
        <aside class="connect-thread-list" aria-label="Threads"></aside>
        <section class="connect-thread-active">
          <div class="connect-thread-active-header" hidden></div>
          <div class="connect-thread-messages" hidden></div>
          <button class="connect-thread-jump" type="button" hidden
                  aria-label="Jump to latest messages">
            <span class="connect-thread-jump-count" hidden></span>
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <polyline points="6 9 12 15 18 9"/>
            </svg>
          </button>
          <div class="connect-thread-typing" role="status" aria-live="polite" hidden></div>
          <div class="connect-thread-reply" hidden>
            <div class="connect-thread-reply-bar">
              <div class="connect-thread-reply-text"></div>
              <button class="connect-thread-reply-clear" type="button" aria-label="Clear reply">&#x2715;</button>
            </div>
          </div>
          <div class="connect-thread-staged" hidden aria-label="Attachments to send"></div>
          <div class="connect-thread-composer" hidden>
            <button class="connect-thread-composer-aux" type="button"
                    data-action="attach" title="Attach file" aria-label="Attach file">
              ${icon('plus', { size: 18 })}
            </button>
            <button class="connect-thread-composer-aux" type="button"
                    data-action="emoji" title="Emoji" aria-label="Insert emoji">
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
                   stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/>
                <line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/>
              </svg>
            </button>
            <div class="connect-thread-composer-input-wrap">
              <textarea class="connect-thread-composer-input"
                        rows="1"
                        placeholder="Message…"
                        aria-label="Message"></textarea>
              <div class="connect-thread-composer-hint">Enter to send · Shift+Enter for newline</div>
            </div>
            <button class="connect-thread-composer-send" type="button"
                    data-empty="1" title="Record voice note" aria-label="Record voice note">
              <span class="connect-thread-composer-send-icon mic">${icon('mic', { size: 18 })}</span>
              <span class="connect-thread-composer-send-icon plane">${icon('send', { size: 18 })}</span>
            </button>
            <div class="connect-thread-recording" hidden>
              <button class="connect-thread-rec-btn cancel" type="button" data-rec="cancel"
                      title="Cancel" aria-label="Cancel recording">${icon('trash', { size: 18 })}</button>
              <span class="connect-thread-rec-dot" aria-hidden="true"></span>
              <span class="connect-thread-rec-time">0:00</span>
              <div class="connect-thread-rec-wave" aria-hidden="true">
                <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
              </div>
              <button class="connect-thread-rec-btn send" type="button" data-rec="send"
                      title="Send voice message" aria-label="Send voice message">${icon('send', { size: 18 })}</button>
            </div>
          </div>
        </section>
        <aside class="connect-thread-info-panel" aria-label="Contact info" hidden></aside>
      </div>
    </div>
    <div class="connect-contact-picker hidden" role="dialog" aria-label="New conversation">
      <div class="connect-contact-picker-card">
        <div class="connect-contact-picker-title">New conversation</div>
        <div class="connect-contact-picker-sub">Search anyone on this machine, or paste their address.</div>
        <div class="connect-contact-picker-list"></div>
        <div class="connect-contact-picker-row">
          <input type="text" class="connect-contact-picker-input"
                 autocomplete="off" spellcheck="false"
                 placeholder="user@instance.host" />
          <button class="connect-contact-picker-add" type="button">Open thread</button>
        </div>
        <div class="connect-contact-picker-actions">
          <button class="connect-contact-picker-cancel" type="button">Cancel</button>
        </div>
      </div>
    </div>
  `;
  document.body.appendChild(el);
  _panel = el;

  el.querySelector('.connect-thread-panel-close')
    .addEventListener('click', closeMessagingPanel);

  el.querySelector('.connect-thread-panel-new')
    .addEventListener('click', _openContactPicker);
  el.querySelector('.connect-thread-reply-clear')
    .addEventListener('click', _clearReply);

  // Jump-to-latest: show the button when scrolled up; clicking returns
  // to the live edge and clears the new-message count.
  const msgsScroll = el.querySelector('.connect-thread-messages');
  const jumpBtn = el.querySelector('.connect-thread-jump');
  if (msgsScroll && jumpBtn) {
    msgsScroll.addEventListener('scroll', () => {
      const dist = msgsScroll.scrollHeight - msgsScroll.scrollTop - msgsScroll.clientHeight;
      if (dist <= 80) { _jumpNewCount = 0; }
      _updateJumpButton(dist);
    }, { passive: true });
    jumpBtn.addEventListener('click', () => {
      _jumpNewCount = 0;
      _scrollMessagesToBottom();
      _updateJumpButton(0);
    });
  }

  const input = el.querySelector('.connect-thread-composer-input');
  const sendBtn = el.querySelector('.connect-thread-composer-send');

  // Morphing send slot — when the input is empty the button acts as
  // the voice-record affordance (mic icon); the instant any text is
  // present it morphs to the paper-plane send. WhatsApp pattern.
  sendBtn.addEventListener('click', () => {
    if (sendBtn.dataset.empty === '1') {
      // Mic mode — start recording a voice message.
      _startVoiceRecording();
      return;
    }
    _onSendClick();
  });

  // Recording-bar controls (cancel / stop-and-send).
  for (const rb of el.querySelectorAll('.connect-thread-rec-btn')) {
    rb.addEventListener('click', () => {
      _finishVoiceRecording(rb.dataset.rec === 'send');
    });
  }
  input.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' && !ev.shiftKey) {
      ev.preventDefault();
      // Enter-to-send only when there's actual content. Empty Enter
      // press shouldn't trigger the voice-note stub toast.
      if (input.value.trim()) _onSendClick();
    }
  });
  // Typing indicator — debounced. Each keystroke schedules a "stop"
  // for TYPING_DEBOUNCE_MS later; if we're not currently signalling
  // typing, we send a "start" first.
  input.addEventListener('input', _onComposerInput);
  input.addEventListener('input', _autosizeComposer);
  input.addEventListener('blur', _flushTypingStop);
  // Update morph state + accessibility labels as the user types.
  const updateSendState = () => {
    // "Has content" includes staged attachments, so the button morphs to
    // Send (not the mic) the moment something is queued.
    const hasContent = !!input.value.trim() || _stagedFiles.length > 0;
    sendBtn.dataset.empty = hasContent ? '0' : '1';
    if (hasContent) {
      sendBtn.setAttribute('title', 'Send');
      sendBtn.setAttribute('aria-label', 'Send');
    } else {
      sendBtn.setAttribute('title', 'Record voice note');
      sendBtn.setAttribute('aria-label', 'Record voice note');
    }
  };
  _updateSendState = updateSendState;
  input.addEventListener('input', updateSendState);
  // Paste an image straight from the clipboard → stage it.
  input.addEventListener('paste', _onComposerPaste);
  updateSendState();

  // Attach button — opens a hidden file picker; the change handler
  // uploads + sends a message with attachment_ref set.
  const filePicker = document.createElement('input');
  filePicker.type = 'file';
  filePicker.style.display = 'none';
  // Loose accept list — the server's upload pipeline does the real
  // size/type validation. We keep the picker open to anything common.
  filePicker.accept = 'image/*,audio/*,video/*,application/pdf,text/*';
  filePicker.multiple = true;  // stage several at once
  filePicker.className = 'connect-thread-filepicker';
  el.appendChild(filePicker);
  filePicker.addEventListener('change', _onFilePicked);
  for (const aux of el.querySelectorAll('.connect-thread-composer-aux')) {
    aux.addEventListener('click', () => {
      const action = aux.dataset.action;
      if (action === 'attach') {
        // The "+" opens an attachment menu (Photo/File · Location ·
        // Contact) — the latter two lean on the device's own pickers.
        _openAttachMenu(aux);
      } else if (action === 'emoji') {
        _openComposerEmojiPicker(aux);
      }
    });
  }

  // Wire contact picker controls.
  const pickerInput = el.querySelector('.connect-contact-picker-input');
  const pickerAdd = el.querySelector('.connect-contact-picker-add');
  const pickerCancel = el.querySelector('.connect-contact-picker-cancel');
  pickerAdd.addEventListener('click', _onPickerAddClick);
  pickerInput.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') { ev.preventDefault(); _onPickerAddClick(); }
    else if (ev.key === 'Escape') { ev.preventDefault(); _closeContactPicker(); }
  });
  pickerCancel.addEventListener('click', _closeContactPicker);

  el.addEventListener('click', (ev) => {
    // Backdrop clicks (overlay area) close the panel.
    if (ev.target === el) closeMessagingPanel();
  });

  // Full-canvas drop zone on the active conversation. Dragging a
  // file over the conversation area shows a translucent overlay
  // with "Drop to attach"; releasing the file triggers the same
  // upload+send pipeline as the attach button. Restricted to the
  // active-thread area (no thread = nothing to attach to).
  const activeArea = el.querySelector('.connect-thread-active');
  if (activeArea) {
    let dragCount = 0;  // dragenter/leave fire on each child cross; count up/down
    const showDrop = () => activeArea.classList.add('drop-active');
    const hideDrop = () => activeArea.classList.remove('drop-active');
    activeArea.addEventListener('dragenter', (ev) => {
      if (!_activeThreadId) return;
      if (!ev.dataTransfer || !Array.from(ev.dataTransfer.types || []).includes('Files')) return;
      ev.preventDefault();
      dragCount += 1;
      if (dragCount === 1) showDrop();
    });
    activeArea.addEventListener('dragover', (ev) => {
      if (!_activeThreadId) return;
      if (!ev.dataTransfer) return;
      ev.preventDefault();  // required to allow drop
      ev.dataTransfer.dropEffect = 'copy';
    });
    activeArea.addEventListener('dragleave', () => {
      dragCount = Math.max(0, dragCount - 1);
      if (dragCount === 0) hideDrop();
    });
    activeArea.addEventListener('drop', (ev) => {
      ev.preventDefault();
      dragCount = 0;
      hideDrop();
      if (!_activeThreadId) return;
      const files = ev.dataTransfer?.files;
      if (files && files.length) _stageFiles(files);
    });
  }

  return el;
}

// ── Thread list ────────────────────────────────────────────────

async function _loadThreadList() {
  // Fetch threads + recent calls together so the recents list can merge
  // them. Calls are best-effort — a failed listCalls must not blank the
  // conversation list.
  const [threads] = await Promise.all([
    listThreads({ limit: 100 }),
    _loadRecentCalls(),
  ]);
  _threadCache.clear();
  for (const t of threads) _threadCache.set(t.thread_id, t);
  _renderThreadList();
}

// Pull the recent-calls window and index the newest call per peer. Used
// both for the merged recents preview/sort and the in-thread timeline.
async function _loadRecentCalls() {
  try {
    const calls = await listCalls({ limit: 100 });
    _recentCalls = Array.isArray(calls) ? calls : [];
  } catch (err) {
    console.warn('connect: listCalls (recents merge) failed', err);
    _recentCalls = [];
  }
  _callsByPeer = new Map();
  for (const c of _recentCalls) {
    const did = c.peer_did || '';
    if (!did) continue;
    const prev = _callsByPeer.get(did);
    // _recentCalls is newest-first, so the first one we see per peer is
    // the newest; keep it.
    if (!prev) _callsByPeer.set(did, c);
  }
}

// A call ended: re-pull the calls window so the recents list + the open
// timeline pick up the new/updated call row. No-op when the panel isn't
// mounted (cheap; just refreshes the cache on next open otherwise).
async function _onCallEndedRefresh() {
  if (!_panel) return;
  await _loadRecentCalls();
  _renderThreadList();
  if (_activeThreadId) {
    _renderMessages(_messageCache.get(_activeThreadId) || []);
  }
}

// Build the unified recents list: one entry per person, merging the
// thread (if any) with that peer's newest call (if any). The newer of
// the two drives the preview + sort; call-only peers get a thread-less
// entry the row click resolves via _openOrCreateThreadForPeer.
function _buildRecentsEntries() {
  const entries = [];
  const seenPeers = new Set();
  for (const t of _threadCache.values()) {
    const did = t.peer_did || '';
    const call = did ? _callsByPeer.get(did) : null;
    const msgTs = t.last_message_at || '';
    const callTs = call?.initiated_at || '';
    const callNewer = !!callTs && callTs.localeCompare(msgTs) > 0;
    entries.push({
      thread: t,
      peerDid: did,
      name: (t.peer_display_name || '').trim() || resolvePeerName(did),
      pinned: !!t.pinned,
      sortTs: (callNewer ? callTs : msgTs) || msgTs || callTs,
      showCall: callNewer,
      call,
    });
    if (did) seenPeers.add(did);
  }
  // Peers we've only ever called (no thread row yet).
  for (const [did, call] of _callsByPeer) {
    if (seenPeers.has(did)) continue;
    entries.push({
      thread: null,
      peerDid: did,
      name: (call.peer_display_name || '').trim() || resolvePeerName(did),
      pinned: false,
      sortTs: call.initiated_at || '',
      showCall: true,
      call,
    });
  }
  entries.sort((a, b) => {
    if (a.pinned !== b.pinned) return a.pinned ? -1 : 1;
    return (b.sortTs || '').localeCompare(a.sortTs || '');
  });
  return entries;
}

// Short human label for a call used in the recents preview line.
function _callPreviewText(call) {
  if (!call) return '';
  const outgoing = call.direction === 'outgoing';
  switch (call.state) {
    case 'missed':   return outgoing ? 'No answer' : 'Missed call';
    case 'declined': return outgoing ? 'Call declined' : 'You declined';
    case 'failed':   return 'Call failed';
    default: break;
  }
  const dir = outgoing ? 'Outgoing' : 'Incoming';
  if (typeof call.duration_seconds === 'number' && call.duration_seconds >= 0) {
    return `${dir} call · ${_humaniseCallDuration(call.duration_seconds)}`;
  }
  if (call.state === 'connected' || call.state === 'ringing' || call.state === 'invited') {
    return 'In progress';
  }
  return `${dir} call`;
}

// Recents preview HTML for a call row: a direction glyph (missed in
// red) + the short label.
function _callPreviewHtml(call) {
  const missed = call.state === 'missed';
  const glyph = missed
    ? icon('phone-missed', { size: 12 })
    : (call.direction === 'outgoing'
        ? icon('arrow-up-right', { size: 12 })
        : icon('arrow-down-left', { size: 12 }));
  return `<span class="connect-thread-list-callglyph${missed ? ' missed' : ''}" aria-hidden="true">${glyph}</span> ${escapeHtml(_callPreviewText(call))}`;
}

function _humaniseCallDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '';
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${r}s`;
  return `${r}s`;
}

function _renderThreadList() {
  if (!_panel) return;
  const list = _panel.querySelector('.connect-thread-list');
  if (!list) return;
  // ── Merged recents (Phase 1b) ──────────────────────────────────
  // One row per PERSON: a thread, a call-only peer, or a peer with
  // both. The newer of {last message, last call} drives the preview +
  // sort key, so a missed call surfaces a contact even with no texts.
  const entries = _buildRecentsEntries();

  if (entries.length === 0) {
    list.innerHTML = `
      <div class="connect-thread-list-empty">
        <div class="connect-thread-list-empty-glyph">${icon('message', { size: 32 })}</div>
        <div class="connect-thread-list-empty-text">No conversations yet</div>
        <div class="connect-thread-list-empty-sub">Tap <strong>+ New</strong> to start one.</div>
      </div>
    `;
    return;
  }

  const myDid = (getWelcome() || {}).user_did || '';
  list.innerHTML = entries.map((e) => {
    const t = e.thread;
    const name = e.name;
    const peer = escapeHtml(name);
    // Preview: a call wins when it's newer than the last message (or
    // there's no message at all); otherwise the message preview, with
    // "You:" prefixing our own last line for scan-friendliness.
    let preview;
    if (e.showCall && e.call) {
      preview = _callPreviewHtml(e.call);
    } else {
      const rawPreview = (t?.last_message_preview || '').slice(0, 80);
      const lastSender = (t?.last_message_sender_did || '').trim();
      const isOurs = lastSender && myDid && lastSender === myDid;
      const previewBody = escapeHtml(rawPreview);
      preview = previewBody
        ? (isOurs
            ? `<span class="connect-thread-list-preview-prefix">You:</span> ${previewBody}`
            : previewBody)
        : '<em class="connect-thread-list-preview-empty">No messages yet</em>';
    }
    const stamp = escapeHtml(_humaniseRelativeShort(e.sortTs));
    const unread = t?.unread_count && t.unread_count > 0 ? t.unread_count : 0;
    const badge = unread
      ? `<span class="connect-thread-list-badge">${unread > 99 ? '99+' : unread}</span>`
      : '';
    const pinned = t?.pinned
      ? `<span class="connect-thread-list-pin" title="Pinned">${icon('star', { size: 12 })}</span>`
      : '';
    const muted = t?.muted
      ? `<span class="connect-thread-list-muted-glyph" title="Muted">${icon('mic-off', { size: 12 })}</span>`
      : '';
    const initial = _initialFor(name || '?');
    // Live presence (WS-driven _peerStatus, falls back to 'offline'
    // when we haven't received an EVENT_PRESENCE_UPDATE for them yet).
    const presence = e.peerDid ? getPeerStatus(e.peerDid) : 'offline';
    const isActive = (t && t.thread_id === _activeThreadId)
      || (!t && e.peerDid && e.peerDid === _activePeerDid);
    const activeCls = isActive ? ' active' : '';
    const unreadCls = unread ? ' unread' : '';
    const mutedCls = t?.muted ? ' muted' : '';
    // Call-only rows have no thread yet — the click handler opens/creates
    // a thread for the peer instead of looking up a thread_id.
    return `
      <div class="connect-thread-list-row${activeCls}${unreadCls}${mutedCls}"
           data-thread-id="${escapeHtml(t?.thread_id || '')}"
           data-peer-did="${escapeHtml(e.peerDid || '')}"
           role="button" tabindex="0">
        <span class="connect-thread-list-avatar" aria-hidden="true">
          ${escapeHtml(initial)}
          <span class="connect-thread-list-presence"
                data-presence="${escapeHtml(presence)}"
                title="${escapeHtml(presence)}"></span>
        </span>
        <div class="connect-thread-list-body">
          <div class="connect-thread-list-toprow">
            <span class="connect-thread-list-peer">${peer}</span>
            <span class="connect-thread-list-time">${stamp}</span>
          </div>
          <div class="connect-thread-list-bottomrow">
            <span class="connect-thread-list-preview">${preview}</span>
            <span class="connect-thread-list-meta">${pinned}${muted}${badge}</span>
          </div>
        </div>
        <button class="connect-thread-list-more" type="button"
                aria-label="Thread actions" data-action="thread-menu">${icon('settings', { size: 14 })}</button>
      </div>
    `;
  }).join('');

  for (const row of list.querySelectorAll('.connect-thread-list-row')) {
    const open = async () => {
      const tid = row.dataset.threadId;
      const peerDid = row.dataset.peerDid;
      if (tid) await _openThread(tid);
      else if (peerDid) await _openOrCreateThreadForPeer(peerDid);
    };
    row.addEventListener('click', (ev) => {
      // Don't open the thread when the user clicked the kebab menu.
      if (ev.target.closest('.connect-thread-list-more')) return;
      open();
    });
    row.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' || ev.key === ' ') {
        ev.preventDefault();
        open();
      }
    });
    const menuBtn = row.querySelector('.connect-thread-list-more');
    if (menuBtn) {
      menuBtn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        _openThreadRowMenu(row, menuBtn);
      });
    }
    row.addEventListener('contextmenu', (ev) => {
      ev.preventDefault();
      _openThreadRowMenu(row, menuBtn || row);
    });
  }
}

// ── Active thread header ────────────────────────────────────────
//
// Replaces the old single-line peer-name with a proper header:
// avatar + name + presence/last-seen + 3 icon-button quick actions
// (Voice call / Video call / Info). The Info button currently
// surfaces a contact preview toast — full contact panel ships in a
// follow-up but the affordance lands now so the chrome feels complete.

function _renderActiveHeader(thread) {
  if (!_panel || !thread) return;
  const headerEl = _panel.querySelector('.connect-thread-active-header');
  if (!headerEl) return;
  const peerDid = thread.peer_did || '';
  const name = (thread.peer_display_name || '').trim()
    || resolvePeerName(peerDid)
    || '(unknown peer)';
  const initial = _initialFor(name);
  const presenceKey = getPeerStatus(peerDid);  // 'online' | 'away' | 'dnd' | 'offline'
  const PRESENCE_LABELS = {
    online: 'Online',
    away: 'Away',
    dnd: 'Do not disturb',
    offline: 'Offline',
  };
  // Typing state takes precedence over presence in the header line —
  // it's the most actionable live signal (peer is about to send), and
  // collapses back to plain presence the moment typing stops/expires.
  const typingEntry = _typingState.get(thread.thread_id);
  const typingActive = !!(typingEntry && typingEntry.expiresAt > Date.now());
  const status = typingActive
    ? 'Typing…'
    : (PRESENCE_LABELS[presenceKey] || 'Offline');
  const statusCls = typingActive
    ? 'connect-thread-header-status typing'
    : 'connect-thread-header-status';

  headerEl.innerHTML = `
    <button class="connect-thread-mobile-back" type="button" data-action="back"
            aria-label="Back to conversations" title="Back">
      <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <polyline points="15 18 9 12 15 6"/>
      </svg>
    </button>
    <div class="connect-thread-header-identity" data-action="open-info" tabindex="0" role="button">
      <span class="connect-thread-header-avatar" aria-hidden="true">
        ${escapeHtml(initial)}
        <span class="connect-thread-header-presence presence-${escapeHtml(presenceKey)}" aria-hidden="true"></span>
      </span>
      <span class="connect-thread-header-text">
        <span class="connect-thread-header-name">${escapeHtml(name)}</span>
        <span class="${statusCls}">${escapeHtml(status)}</span>
      </span>
    </div>
    <div class="connect-thread-header-actions">
      <button class="connect-thread-header-action" type="button"
              data-action="voice-call" title="Voice call" aria-label="Voice call">
        ${icon('phone', { size: 16 })}
      </button>
      <button class="connect-thread-header-action" type="button"
              data-action="video-call" title="Video call" aria-label="Video call">
        ${icon('video', { size: 16 })}
      </button>
      <button class="connect-thread-header-action" type="button"
              data-action="info" title="Contact info" aria-label="Contact info">
        ${icon('user', { size: 16 })}
      </button>
    </div>
  `;

  const backEl = headerEl.querySelector('.connect-thread-mobile-back');
  if (backEl) backEl.addEventListener('click', (ev) => { ev.stopPropagation(); _backToList(); });

  const identityEl = headerEl.querySelector('.connect-thread-header-identity');
  if (identityEl) {
    identityEl.addEventListener('click', () => _showContactInfo(thread));
    identityEl.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' || ev.key === ' ') {
        ev.preventDefault();
        _showContactInfo(thread);
      }
    });
  }
  for (const btn of headerEl.querySelectorAll('.connect-thread-header-action')) {
    btn.addEventListener('click', (ev) => {
      ev.stopPropagation();
      const action = btn.dataset.action;
      if (action === 'voice-call' || action === 'video-call') {
        _placeCallFromHeader(peerDid, action === 'video-call');
      } else if (action === 'info') {
        _showContactInfo(thread);
      }
    });
  }

  // Dynamic composer placeholder. Mirrors the active peer's name
  // so the empty composer reads "Message bench" / "Message shadow"
  // instead of the generic "Message…". One write per header render
  // — cheap and DOM-local.
  const composerInput = _panel.querySelector('.connect-thread-composer-input');
  if (composerInput) {
    composerInput.setAttribute('placeholder', `Message ${name}`);
  }
}

async function _placeCallFromHeader(peerDid, withVideo) {
  if (!peerDid) {
    showToast('No peer to call', 'warning');
    return;
  }
  try {
    const mod = await import('./ui.js');
    await mod.startCall?.(peerDid, { withVideo });
  } catch (err) {
    showToast(`Call failed: ${err?.message || 'unknown error'}`, 'error');
  }
}

/**
 * Show the slide-in contact-info panel on the right side of the
 * thread. Renders peer identity (avatar + name + DID + presence),
 * quick actions (voice / video call), recent calls with this peer
 * (filtered client-side from the global recents API), and thread
 * settings (mute / clear / block — Phase 1 mute is local-only,
 * clear + block surface friendly toasts since the routes ship
 * with the next backend pass).
 *
 * Idempotent: closes the panel if it's already open for the same
 * thread (tap-to-toggle behaviour matches iMessage's info button).
 */
async function _showContactInfo(thread) {
  if (!_panel || !thread) return;
  const info = _panel.querySelector('.connect-thread-info-panel');
  if (!info) return;
  // Toggle off if it's already open for this thread.
  if (!info.hidden && info.dataset.threadId === thread.thread_id) {
    _closeContactInfo();
    return;
  }
  info.dataset.threadId = thread.thread_id;
  await _renderContactInfo(info, thread);
  info.hidden = false;
  _panel.querySelector('.connect-thread-panel-card')
    ?.classList.add('info-open');
}

function _closeContactInfo() {
  if (!_panel) return;
  const info = _panel.querySelector('.connect-thread-info-panel');
  if (!info) return;
  info.hidden = true;
  info.dataset.threadId = '';
  _panel.querySelector('.connect-thread-panel-card')
    ?.classList.remove('info-open');
}

async function _renderContactInfo(info, thread) {
  const peerDid = thread.peer_did || '';
  const name = (thread.peer_display_name || '').trim()
    || _prettyPeerForInfo(peerDid);
  const initial = _initialFor(name || peerDid || '?');
  const subtitle = _peerSubtitleForInfo(peerDid);
  // Hydrate thread.blocked from the contact store on first render so
  // the Block / Unblock label is correct on page reload (the thread
  // row itself doesn't carry the flag).
  if (thread.blocked === undefined && peerDid) {
    try {
      const { listContacts: listC } = await import('./messages.js');
      const contacts = await listC({ includeBlocked: true });
      const match = contacts.find((c) => c.peer_did === peerDid);
      thread.blocked = !!(match && match.blocked);
    } catch (err) {
      console.warn('connect: hydrate blocked failed', err);
      thread.blocked = false;
    }
  }

  // Presence — same 4-state vocabulary as the header.
  let presence = 'offline';
  try {
    const { getPeerStatus } = await import('./client.js');
    presence = getPeerStatus(peerDid);
  } catch (_) { /* keep default */ }
  const PRESENCE_LABELS = {
    online: 'Online', away: 'Away', dnd: 'Do not disturb', offline: 'Offline',
  };

  // Render skeleton first so the panel feels snappy; recent calls
  // load asynchronously and slot in below.
  info.innerHTML = `
    <div class="connect-thread-info-head">
      <button class="connect-thread-info-close" type="button" aria-label="Close contact info">
        ${icon('x', { size: 14 })}
      </button>
    </div>
    <div class="connect-thread-info-identity">
      <span class="connect-thread-info-avatar" aria-hidden="true">${escapeHtml(initial)}</span>
      <div class="connect-thread-info-name">${escapeHtml(name)}</div>
      ${subtitle ? `<div class="connect-thread-info-subtitle">${escapeHtml(subtitle)}</div>` : ''}
      <div class="connect-thread-info-presence presence-${escapeHtml(presence)}">
        <span class="connect-thread-info-presence-dot" aria-hidden="true"></span>
        <span>${escapeHtml(PRESENCE_LABELS[presence] || 'Offline')}</span>
      </div>
    </div>
    <div class="connect-thread-info-actions">
      <button class="connect-thread-info-action" type="button" data-action="voice-call">
        <span class="connect-thread-info-action-glyph">${icon('phone', { size: 18 })}</span>
        <span>Voice</span>
      </button>
      <button class="connect-thread-info-action" type="button" data-action="video-call">
        <span class="connect-thread-info-action-glyph">${icon('video', { size: 18 })}</span>
        <span>Video</span>
      </button>
      <button class="connect-thread-info-action" type="button" data-action="mute">
        <span class="connect-thread-info-action-glyph">${icon(thread.muted ? 'mic-off' : 'mic', { size: 18 })}</span>
        <span>${thread.muted ? 'Unmute' : 'Mute'}</span>
      </button>
    </div>
    <div class="connect-thread-info-section connect-thread-info-recents">
      <div class="connect-thread-info-section-head">Recent calls</div>
      <div class="connect-thread-info-recents-list">
        <div class="connect-thread-info-skeleton">Loading…</div>
      </div>
    </div>
    <div class="connect-thread-info-section">
      <div class="connect-thread-info-section-head">Thread</div>
      <button class="connect-thread-info-row" type="button" data-action="clear">
        ${icon('x', { size: 14 })}<span>Clear chat history</span>
      </button>
      <button class="connect-thread-info-row${thread.blocked ? '' : ' destructive'}" type="button" data-action="block">
        ${icon('mic-off', { size: 14 })}<span>${thread.blocked ? 'Unblock contact' : 'Block contact'}</span>
      </button>
      <button class="connect-thread-info-row destructive" type="button" data-action="remove-contact">
        ${icon('trash', { size: 14 })}<span>Remove contact</span>
      </button>
    </div>
  `;

  // Wire interactions.
  info.querySelector('.connect-thread-info-close')
    ?.addEventListener('click', _closeContactInfo);

  for (const btn of info.querySelectorAll('.connect-thread-info-action, .connect-thread-info-row')) {
    btn.addEventListener('click', () => _handleInfoAction(btn.dataset.action, thread, info));
  }

  // Fetch + render recent calls with this peer (client-side filter
  // because the API doesn't take a peer parameter today).
  try {
    const { listCalls } = await import('./messages.js');
    const calls = await listCalls({ limit: 50 });
    const mine = calls.filter((c) => c.peer_did === peerDid).slice(0, 5);
    const host = info.querySelector('.connect-thread-info-recents-list');
    if (!host) return;
    if (mine.length === 0) {
      host.innerHTML = '<div class="connect-thread-info-empty">No calls with this contact yet.</div>';
      return;
    }
    host.innerHTML = mine.map((c) => {
      const arrowIcon = c.state === 'missed'
        ? icon('phone-missed', { size: 12 })
        : c.direction === 'outgoing'
          ? icon('phone-outgoing', { size: 12 })
          : icon('phone-incoming', { size: 12 });
      const when = _humaniseRelativeShort(c.initiated_at);
      const dur = c.duration_seconds
        ? ` · ${_formatDurationShort(c.duration_seconds)}`
        : '';
      const kind = (c.modalities || '').includes('video') ? 'Video' : 'Voice';
      const stateClass = c.state === 'missed' ? ' missed' : '';
      return `
        <div class="connect-thread-info-call-row${stateClass}">
          <span class="connect-thread-info-call-arrow">${arrowIcon}</span>
          <div class="connect-thread-info-call-body">
            <div class="connect-thread-info-call-kind">${escapeHtml(kind)}${escapeHtml(dur)}</div>
            <div class="connect-thread-info-call-when">${escapeHtml(when)}</div>
          </div>
        </div>
      `;
    }).join('');
  } catch (err) {
    console.warn('connect: contact-info recents fetch failed', err);
  }
}

async function _handleInfoAction(action, thread, info) {
  if (action === 'voice-call' || action === 'video-call') {
    _placeCallFromHeader(thread.peer_did || '', action === 'video-call');
    _closeContactInfo();
    return;
  }
  if (action === 'mute') {
    // Optimistic toggle + persist; revert on failure so the toast never lies.
    const prev = !!thread.muted;
    const next = !prev;
    thread.muted = next;
    _renderThreadList();
    _renderContactInfo(info, thread);
    try {
      await setThreadFlags(thread.thread_id, { muted: next });
      showToast(next ? 'Thread muted' : 'Thread unmuted', 'info');
    } catch (err) {
      thread.muted = prev;
      _renderThreadList();
      _renderContactInfo(info, thread);
      console.warn('[connect] mute save failed', err);
      showToast("Couldn't save that — try again.", 'error');
    }
    return;
  }
  if (action === 'clear') {
    const peerName = resolvePeerName(thread.peer_did || '') || 'this thread';
    const ok = window.confirm(
      `Clear chat history with ${peerName}?\n\n` +
      `This wipes every message in this thread on this device. ` +
      `${peerName} keeps their copy on their device. There is no undo.`,
    );
    if (!ok) return;
    try {
      const res = await clearThreadMessages(thread.thread_id);
      const removed = (res && typeof res.removed === 'number') ? res.removed : 0;
      // Drop the local cache + reset the thread tail snapshot so the
      // sidebar preview clears and the active pane shows the empty
      // starter state without a refresh.
      _messageCache.set(thread.thread_id, []);
      thread.last_message_at = null;
      thread.last_message_preview = '';
      thread.last_message_sender_did = '';
      thread.unread_count = 0;
      _scheduleMessageRender();
      _renderThreadList();
      _closeContactInfo();
      showToast(
        removed > 0
          ? `Cleared ${removed} message${removed === 1 ? '' : 's'}`
          : 'Thread already empty',
        'success',
      );
    } catch (err) {
      console.warn('connect: clear-history failed', err);
      showToast(`Could not clear history: ${err?.message || 'unknown error'}`, 'error');
    }
    return;
  }
  if (action === 'block') {
    const peerName = resolvePeerName(thread.peer_did || '') || 'this contact';
    const alreadyBlocked = !!thread.blocked;
    const ok = window.confirm(
      alreadyBlocked
        ? `Unblock ${peerName}?\n\n` +
          `Their messages and calls will start reaching you again.`
        : `Block ${peerName}?\n\n` +
          `They won't be able to reach you with messages or calls. ` +
          `They won't see that they've been blocked — sends look ` +
          `successful to them, but you stop receiving anything.`,
    );
    if (!ok) return;
    try {
      await setPeerBlocked(thread.peer_did || '', !alreadyBlocked);
      thread.blocked = !alreadyBlocked;
      _renderContactInfo(info, thread);
      _renderThreadList();
      showToast(
        thread.blocked
          ? `Blocked ${peerName}`
          : `Unblocked ${peerName}`,
        'success',
      );
    } catch (err) {
      console.warn('connect: block toggle failed', err);
      showToast(`Could not update block: ${err?.message || 'unknown error'}`, 'error');
    }
    return;
  }
  if (action === 'remove-contact') {
    const peerDid = thread.peer_did || '';
    const peerName = resolvePeerName(peerDid) || 'this contact';
    const ok = window.confirm(
      `Remove ${peerName} from your contacts?\n\n` +
      `This deletes the saved contact. Your chat history stays on this ` +
      `device (clear it separately), and they can still reach you unless ` +
      `you also block them.`,
    );
    if (!ok) return;
    try {
      // The contact row is keyed by contact_id; resolve it from the peer DID.
      const contacts = await listContacts({ includeBlocked: true });
      const match = (contacts || []).find((c) => c.peer_did === peerDid);
      if (!match || !match.contact_id) {
        showToast('This person isn’t in your contacts.', 'info');
        return;
      }
      await removeContact(match.contact_id);
      _closeContactInfo();
      showToast(`Removed ${peerName}`, 'success');
    } catch (err) {
      console.warn('connect: remove contact failed', err);
      showToast(`Could not remove contact: ${err?.message || 'unknown error'}`, 'error');
    }
    return;
  }
}

/**
 * True when the trimmed body is composed entirely of emoji glyphs
 * (Extended_Pictographic) plus optional joiners + ZWJs + spaces, and
 * the visible glyph count is small (≤6). Used to switch a message
 * bubble into "burst" mode — no background, larger font — matching
 * iMessage / Telegram / WhatsApp conventions for one-emoji replies.
 *
 * Stays cheap: bail early on any plain-text character.
 */
// Escape a message body for HTML AND turn bare URLs into tappable links.
// Tokenises the RAW text by URL match so escaping never corrupts an href
// (e.g. `&` in a query string). Trailing sentence punctuation is excluded
// from the match so "see https://x.com." doesn't swallow the period.
const _URL_RE = /(https?:\/\/[^\s<]+|www\.[^\s<]+)/gi;
function _linkifyBody(text) {
  const raw = String(text || '');
  if (!raw) return '';
  let out = '';
  let last = 0;
  let m;
  _URL_RE.lastIndex = 0;
  while ((m = _URL_RE.exec(raw)) !== null) {
    let url = m[0];
    // Don't capture trailing punctuation / closing brackets.
    const trail = url.match(/[.,:;!?)\]}'"]+$/);
    if (trail) url = url.slice(0, url.length - trail[0].length);
    if (!url) { _URL_RE.lastIndex = m.index + 1; continue; }
    out += escapeHtml(raw.slice(last, m.index));
    const href = url.toLowerCase().startsWith('www.') ? `https://${url}` : url;
    out += `<a class="connect-message-link" href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer nofollow">${escapeHtml(url)}</a>`;
    last = m.index + url.length;
    _URL_RE.lastIndex = last;
  }
  out += escapeHtml(raw.slice(last));
  return out;
}

function _isEmojiBurst(body) {
  const s = String(body || '').trim();
  if (!s) return false;
  // Strip variation selectors, ZWJ, skin-tone modifiers, spaces.
  const stripped = s.replace(
    /[\s‍️\u{1f3fb}-\u{1f3ff}]/gu,
    '',
  );
  if (!stripped) return false;
  // Count grapheme-ish characters by iterating code points.
  let count = 0;
  for (const ch of stripped) {
    // Quick allow-list: most emoji live in the supplementary plane
    // (>0xFFFF) or specific BMP blocks (Misc Symbols + Pictographs,
    // Dingbats, etc.). Anything else aborts the burst.
    const cp = ch.codePointAt(0);
    const isEmojiRange =
      cp >= 0x1F300 && cp <= 0x1FAFF
      || cp >= 0x2600 && cp <= 0x27BF
      || cp === 0x231A || cp === 0x231B
      || cp >= 0x2300 && cp <= 0x23FF;
    if (!isEmojiRange) return false;
    count += 1;
    if (count > 6) return false;
  }
  return count > 0;
}

function _prettyPeerForInfo(did) {
  // Delegate to the directory/contacts cache so a peer with a real
  // display_name on the server doesn't surface as a Title-Cased dump
  // of their auto-generated user_id ("Usr A8377d20e22188ab"). The
  // resolver itself falls back to the Title-Case heuristic when the
  // cache has no hit, so the previous worst-case behavior is preserved
  // for unknown peers.
  return resolvePeerName(did);
}

function _peerSubtitleForInfo(did) {
  const raw = String(did || '').trim();
  if (!raw) return '';
  const [, instance] = raw.split('@');
  if (!instance || instance === 'this-instance') return '';
  return `@${instance}`;
}

function _formatDurationShort(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  const r = s % 60;
  if (m === 0) return `${r}s`;
  return `${m}m ${r}s`;
}

// ── Thread-row context menu (pin / mute / archive) ──────────────

let _threadRowMenu = null;

function _openThreadRowMenu(row, anchorEl) {
  _closeThreadRowMenu();
  const tid = row.dataset.threadId;
  const thread = _threadCache.get(tid);
  if (!thread) return;

  const menu = document.createElement('div');
  menu.className = 'connect-thread-row-menu';
  menu.setAttribute('role', 'menu');
  menu.innerHTML = `
    <button class="connect-thread-row-menu-item" role="menuitem" data-action="pin">
      ${icon('star', { size: 14 })}
      <span>${thread.pinned ? 'Unpin' : 'Pin to top'}</span>
    </button>
    <button class="connect-thread-row-menu-item" role="menuitem" data-action="mute">
      ${icon(thread.muted ? 'mic' : 'mic-off', { size: 14 })}
      <span>${thread.muted ? 'Unmute' : 'Mute notifications'}</span>
    </button>
    <button class="connect-thread-row-menu-item destructive" role="menuitem" data-action="archive">
      ${icon('x', { size: 14 })}
      <span>${thread.archived ? 'Unarchive' : 'Archive'}</span>
    </button>
  `;
  document.body.appendChild(menu);
  _threadRowMenu = menu;

  // Position next to the anchor, clamped to viewport.
  const rect = (anchorEl || row).getBoundingClientRect();
  const top = Math.min(rect.bottom + 6, window.innerHeight - 160);
  const right = Math.max(8, window.innerWidth - rect.right);
  menu.style.top = `${top}px`;
  menu.style.right = `${right}px`;
  requestAnimationFrame(() => menu.classList.add('open'));

  const handle = async (action) => {
    _closeThreadRowMenu();
    // Map the menu action to the flag + its new value, apply optimistically,
    // then persist via PATCH /threads/{id}. On failure, revert + toast so the
    // UI never diverges from the server (pin/mute/archive now survive reload).
    const flagFor = { pin: 'pinned', mute: 'muted', archive: 'archived' };
    const flag = flagFor[action];
    if (!flag) return;
    const prev = !!thread[flag];
    const next = !prev;
    thread[flag] = next;
    if (flag === 'archived' && next) _threadCache.delete(tid);  // drops out of the list
    _renderThreadList();
    try {
      await setThreadFlags(tid, { [flag]: next });
    } catch (err) {
      // Revert the optimistic change; re-seat the row if we removed it.
      thread[flag] = prev;
      if (flag === 'archived' && next) _threadCache.set(tid, thread);
      _renderThreadList();
      console.warn('[connect] thread flag save failed', action, err);
      showToast("Couldn't save that — try again.", 'error');
    }
  };
  for (const item of menu.querySelectorAll('.connect-thread-row-menu-item')) {
    item.addEventListener('click', () => handle(item.dataset.action));
  }

  const onOutside = (ev) => {
    if (!_threadRowMenu) return;
    if (_threadRowMenu.contains(ev.target)) return;
    _closeThreadRowMenu();
  };
  const onKey = (ev) => {
    if (ev.key === 'Escape') _closeThreadRowMenu();
  };
  setTimeout(() => {
    document.addEventListener('pointerdown', onOutside, true);
    document.addEventListener('keydown', onKey);
  }, 0);
  menu._cleanup = () => {
    document.removeEventListener('pointerdown', onOutside, true);
    document.removeEventListener('keydown', onKey);
  };
}

function _closeThreadRowMenu() {
  if (!_threadRowMenu) return;
  try { _threadRowMenu._cleanup?.(); } catch (_) {}
  const m = _threadRowMenu;
  m.classList.remove('open');
  _threadRowMenu = null;
  setTimeout(() => { try { m.remove(); } catch (_) {} }, 140);
}

function _humaniseRelativeShort(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return '';
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffSec = Math.floor(diffMs / 1000);
    if (diffSec < 60) return 'now';
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m`;
    if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h`;
    const sameYear = d.getFullYear() === now.getFullYear();
    const yesterday = new Date(now);
    yesterday.setDate(now.getDate() - 1);
    const sameDay = (a, b) =>
      a.getFullYear() === b.getFullYear()
      && a.getMonth() === b.getMonth()
      && a.getDate() === b.getDate();
    if (sameDay(d, yesterday)) return 'Yesterday';
    if (diffSec < 86400 * 7) {
      return d.toLocaleDateString(undefined, { weekday: 'short' });
    }
    return sameYear
      ? d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
      : d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: '2-digit' });
  } catch (_) { return ''; }
}

// ── Active thread ──────────────────────────────────────────────

// Mobile single-pane: return from a conversation to the recents list.
// Desktop ignores `.show-conversation` (both panes always visible), so
// this only changes what's shown on narrow screens.
function _backToList() {
  _cancelVoiceRecordingIfAny();
  _clearStaged();
  _activeThreadId = '';
  _activePeerDid = '';
  if (_panel) {
    _panel.classList.remove('show-conversation');
    const headerEl = _panel.querySelector('.connect-thread-active-header');
    const msgsEl = _panel.querySelector('.connect-thread-messages');
    const composerEl = _panel.querySelector('.connect-thread-composer');
    if (headerEl) headerEl.hidden = true;
    if (msgsEl) msgsEl.hidden = true;
    if (composerEl) composerEl.hidden = true;
  }
  _renderThreadList();
}

async function _openThread(threadId) {
  if (!_panel) _ensurePanel();
  _cancelVoiceRecordingIfAny();  // don't carry a recording across threads
  if (threadId !== _activeThreadId) _clearStaged();  // nor staged attachments
  _activeThreadId = threadId;
  // Fresh conversation view → reset the jump-to-latest accounting so it
  // opens at the live edge with no stale unseen count.
  _jumpNewCount = 0;
  _renderedMsgCount = 0;
  _unreadAnchorId = '';
  if (_panel) {
    const jb = _panel.querySelector('.connect-thread-jump');
    if (jb) jb.hidden = true;
  }
  // Mobile single-pane: reveal the conversation pane (CSS hides the list
  // at ≤700px while this class is present). No-op visually on desktop.
  _panel.classList.add('show-conversation');
  _clearReply();
  _flushTypingStop();
  _renderTypingIndicator();  // refresh in case switching reveals one
  _renderThreadList();
  const headerEl = _panel.querySelector('.connect-thread-active-header');
  const msgsEl = _panel.querySelector('.connect-thread-messages');
  const composerEl = _panel.querySelector('.connect-thread-composer');

  headerEl.hidden = false;
  msgsEl.hidden = false;
  composerEl.hidden = false;

  // Render whatever we already have cached for snappy open.
  const cached = _messageCache.get(threadId) || [];
  const thread = _threadCache.get(threadId);
  _activePeerDid = thread?.peer_did || '';
  if (thread) _renderActiveHeader(thread);

  _renderMessages(cached);

  // Fetch fresh + replace cache.
  try {
    const data = await listMessages(threadId, { limit: 100 });
    if (data?.thread) {
      _threadCache.set(threadId, data.thread);
      _activePeerDid = data.thread.peer_did || _activePeerDid;
      _renderActiveHeader(data.thread);
    }
    const msgs = Array.isArray(data?.messages) ? data.messages : [];
    _messageCache.set(threadId, msgs);
    // Place the "new messages" divider above the oldest unread inbound
    // message (the most-recent `unread_count` peer messages are unread).
    // Uses the pre-read count captured before markThreadRead zeroes it.
    _unreadAnchorId = '';
    const unread = thread?.unread_count || 0;
    if (unread > 0) {
      const myDid = _ownDid();
      let need = unread;
      for (const m of msgs) {  // newest-first
        if (m.sender_did !== myDid && !m.deleted_at) {
          need -= 1;
          if (need === 0) { _unreadAnchorId = m.message_id; break; }
        }
      }
    }
    // Open path: render immediately, not next frame — user just
    // clicked the thread and the 16ms delay reads as input lag.
    _renderMessages(msgs);

    // Seed the catch-up cursor at the newest message we just loaded
    // so a reconnect won't refetch the whole window.
    if (msgs[0]?.sent_at) setThreadCursor(threadId, msgs[0].sent_at);

    // Mark thread read on view + bump unread badge to 0 locally.
    if (thread && thread.unread_count) {
      try {
        await markThreadRead(threadId, msgs[0]?.message_id || '');
      } catch (_) { /* best-effort */ }
      const cur = _threadCache.get(threadId);
      if (cur) cur.unread_count = 0;
      _renderThreadList();
    }
  } catch (err) {
    console.warn('connect: openThread fetch failed', err);
  }
}

/**
 * Schedule a message render coalesced to the next animation frame.
 * Many WS events land in one tick (catch-up burst, sender row +
 * delivery receipt in the same flush). Without coalescing the old
 * code rebuilt the entire <div class="connect-thread-messages">
 * innerHTML once per event — visibly laggy on a thread with even a
 * modest history. One render per frame is enough for typing speed.
 */
function _scheduleMessageRender(threadId) {
  const tid = threadId || _activeThreadId;
  if (!tid) return;
  // If the scheduled render is for a different thread (e.g. a thread
  // switch happened mid-coalesce), drop the prior frame and re-schedule.
  if (_pendingRenderRaf && _pendingRenderTid !== tid) {
    cancelAnimationFrame(_pendingRenderRaf);
    _pendingRenderRaf = 0;
  }
  if (_pendingRenderRaf) return;
  _pendingRenderTid = tid;
  _pendingRenderRaf = requestAnimationFrame(() => {
    _pendingRenderRaf = 0;
    if (_pendingRenderTid !== _activeThreadId) {
      _pendingRenderTid = '';
      return;
    }
    _pendingRenderTid = '';
    const msgs = _messageCache.get(_activeThreadId) || [];
    _renderMessages(msgs);
  });
}

function _renderMessages(messages) {
  if (!_panel) return;
  const msgsEl = _panel.querySelector('.connect-thread-messages');
  if (!msgsEl) return;
  const ordered = messages.slice().reverse();
  // Calls for this peer interleave into the same timeline (Phase 1b).
  const callItems = _callItemsForActivePeer();
  if (ordered.length === 0 && callItems.length === 0) {
    const thread = _threadCache.get(_activeThreadId);
    const peerName = (thread?.peer_display_name || '').trim()
      || (thread?.peer_did && resolvePeerName(thread.peer_did))
      || 'them';
    msgsEl.innerHTML = `
      <div class="connect-thread-messages-empty">
        <div class="connect-thread-empty-glyph">${icon('message', { size: 36 })}</div>
        <div class="connect-thread-empty-text">Say hi to ${escapeHtml(peerName)}</div>
        <div class="connect-thread-empty-sub">This thread is empty. Send a starter or write your own.</div>
        <div class="connect-thread-starter-row" role="group" aria-label="Quick starters">
          <button class="connect-thread-starter" type="button" data-starter="wave" data-send="true">
            <span class="connect-thread-starter-glyph">\u{1F44B}</span>
            <span class="connect-thread-starter-label">Wave</span>
          </button>
          <button class="connect-thread-starter" type="button" data-starter="free">
            <span class="connect-thread-starter-glyph">\u{1F4AC}</span>
            <span class="connect-thread-starter-label">You free?</span>
          </button>
          <button class="connect-thread-starter" type="button" data-starter="voice">
            <span class="connect-thread-starter-glyph">${icon('mic', { size: 16 })}</span>
            <span class="connect-thread-starter-label">Voice note</span>
          </button>
          <button class="connect-thread-starter" type="button" data-starter="share">
            <span class="connect-thread-starter-glyph">${icon('plus', { size: 16 })}</span>
            <span class="connect-thread-starter-label">Share</span>
          </button>
        </div>
      </div>
    `;
    // Wire starter handlers.
    for (const btn of msgsEl.querySelectorAll('.connect-thread-starter')) {
      btn.addEventListener('click', () => _onStarterClick(btn.dataset.starter, btn.dataset.send === 'true'));
    }
    return;
  }
  const myDid = _ownDid();
  const byId = new Map(messages.map((m) => [m.message_id, m]));

  // Build one time-sorted timeline of messages + call events, then walk
  // it: a day-divider before any item whose day differs from the
  // previous, and group consecutive messages from the same sender so
  // only the LAST in a group shows an avatar (a call event in between
  // breaks the group, which is what we want).
  const timeline = [
    ...ordered.map((m) => ({ kind: 'message', ts: m.sent_at || '', m })),
    ...callItems,
  ].sort((a, b) => (a.ts || '').localeCompare(b.ts || ''));

  const TIME_GAP_MS = 20 * 60 * 1000;  // insert a time label after a 20-min lull
  const rows = [];
  let lastDay = '';
  let lastTs = '';
  for (let i = 0; i < timeline.length; i++) {
    const it = timeline[i];
    const day = _dayKey(it.ts);
    if (day && day !== lastDay) {
      rows.push({ type: 'divider', label: _humaniseDayLabel(it.ts), id: day });
      lastDay = day;
      lastTs = '';  // day divider already gives context; no time-sep right after
    } else if (it.ts && lastTs) {
      // iMessage-style: a centered time label when messages are spaced out.
      const gap = new Date(it.ts).getTime() - new Date(lastTs).getTime();
      if (Number.isFinite(gap) && gap > TIME_GAP_MS) {
        rows.push({ type: 'timestamp', label: _humaniseClockTime(it.ts) });
      }
    }
    lastTs = it.ts || lastTs;
    // "New messages" divider sits just above the oldest unread message.
    if (it.kind === 'message' && _unreadAnchorId && it.m.message_id === _unreadAnchorId) {
      rows.push({ type: 'unread' });
    }
    if (it.kind === 'call') {
      rows.push({ type: 'call', call: it.call });
      continue;
    }
    const m = it.m;
    const next = timeline[i + 1];
    const nextSameGroup = next && next.kind === 'message'
      && next.m.sender_did === m.sender_did
      && _dayKey(next.ts) === day;
    rows.push({ type: 'message', m, showAvatar: !nextSameGroup });
  }

  // Capture whether the user is pinned near the bottom BEFORE we replace the
  // DOM. If they'd scrolled up to read history, a re-render (e.g. an inbound
  // message) must NOT yank them back down — only auto-scroll when they were
  // already at the live edge. 120px tolerance covers a short trailing line +
  // sub-pixel rounding; a fresh-opened thread reads as near-bottom (empty →
  // scrollHeight ≈ clientHeight) so first paint still lands at the bottom.
  const NEAR_BOTTOM_PX = 120;
  const wasNearBottom =
    (msgsEl.scrollHeight - msgsEl.scrollTop - msgsEl.clientHeight) <= NEAR_BOTTOM_PX;

  msgsEl.innerHTML = rows.map((row) => {
    if (row.type === 'divider') {
      return `
        <div class="connect-thread-divider" role="separator" aria-label="${escapeHtml(row.label)}">
          <span class="connect-thread-divider-label">${escapeHtml(row.label)}</span>
        </div>
      `;
    }
    if (row.type === 'timestamp') {
      return `
        <div class="connect-thread-timesep" role="separator">
          <span class="connect-thread-timesep-label">${escapeHtml(row.label)}</span>
        </div>
      `;
    }
    if (row.type === 'unread') {
      return `
        <div class="connect-thread-unread-divider" role="separator" aria-label="New messages">
          <span class="connect-thread-unread-label">New messages</span>
        </div>
      `;
    }
    if (row.type === 'call') {
      return _callTimelineRowHtml(row.call);
    }
    const m = row.m;
    const outgoing = m.sender_did === myDid;
    const body = m.deleted_at
      ? '<em class="connect-message-deleted">message deleted</em>'
      : _linkifyBody(m.body || '');
    const edited = m.edited_at
      ? '<span class="connect-message-edited">(edited)</span>'
      : '';

    let replyHtml = '';
    if (m.reply_to) {
      const target = byId.get(m.reply_to);
      const targetPreview = target
        ? (target.body || '').slice(0, 80)
        : 'an earlier message';
      const targetSender = target
        ? (target.sender_did === myDid ? 'you' : escapeHtml(resolvePeerName(target.sender_did)))
        : '';
      replyHtml = `
        <div class="connect-message-reply-quote">
          ${targetSender ? `<span class="connect-message-reply-author">${targetSender}</span>` : ''}
          <span class="connect-message-reply-body">${escapeHtml(targetPreview)}</span>
        </div>
      `;
    }
    let receiptHtml = '';
    if (outgoing && !m.deleted_at) {
      const state = _receiptStateFor(m);
      // Glyph picker — distinct shapes for every state so peripheral
      // vision is enough to read the ticks:
      //   pending   — clock (still queued)
      //   sent      — single check (server has it; peer doesn't yet)
      //   delivered — double check at full opacity
      //   read      — double check tinted accent
      //   failed    — warning triangle
      let glyph;
      let title = '';
      if (state === 'pending') {
        glyph = icon('clock', { size: 12 });
        title = 'Waiting to send — will retry on reconnect';
      } else if (state === 'failed') {
        glyph = icon('alert-triangle', { size: 12 });
        const failed = outboxFindFailed(m.message_id);
        title = failed && failed.last_error
          ? `Send failed: ${failed.last_error}`
          : 'Send failed';
      } else if (state === 'sent') {
        glyph = icon('check', { size: 12 });
        title = 'Sent';
      } else {
        // delivered + read share the double-check; CSS colour is
        // what differentiates them.
        glyph = icon('check-double', { size: 12 });
        title = state === 'read' ? 'Read' : 'Delivered';
      }
      receiptHtml = `<span class="connect-message-receipt receipt-${state}" title="${escapeHtml(title)}">${glyph}</span>`;
    }

    // Avatar slot — only rendered for non-outgoing rows (peer side).
    // Outgoing rows are right-aligned and don't show our own avatar
    // (that's the iMessage / Telegram convention).
    const avatarHtml = (!outgoing && row.showAvatar)
      ? `<span class="connect-message-avatar" aria-hidden="true">${escapeHtml(_initialFor(m.sender_did))}</span>`
      : '<span class="connect-message-avatar-spacer" aria-hidden="true"></span>';

    const reactionsHtml = _renderReactionStack(m, myDid);
    const attachmentHtml = m.attachment_ref && !m.deleted_at
      ? _renderAttachment(m)
      : '';
    // When a message has an attachment but no body text we still want
    // a bubble (the attachment widget lives inside it); when there's
    // text too, attachment renders above the body.
    const bodyHtml = m.body || m.deleted_at
      ? `<span class="connect-message-body">${body}${edited}</span>`
      : '';
    // Failed-message action row sits BELOW the bubble (not inside)
    // so it doesn't compete with the body for the bubble's max-width
    // budget. Only rendered for the failed state — every other state
    // gets a normal bubble.
    const failedActionsHtml = outgoing && !m.deleted_at
      && _receiptStateFor(m) === 'failed'
      ? _renderFailedActions(m)
      : '';
    // Emoji-burst: pure-emoji messages (up to ~6 glyphs, no other
    // visible text, no attachment, no reply context) render without
    // a bubble background + bigger font, matching iMessage/Telegram.
    // Skip when there's an attachment or reply chrome.
    const isBurst = !attachmentHtml && !replyHtml && !m.deleted_at
      && _isEmojiBurst(m.body || '');
    return `
      <div class="connect-message-row ${outgoing ? 'out' : 'in'} ${row.showAvatar ? 'tail' : ''}"
           data-message-id="${escapeHtml(m.message_id)}"
           title="${escapeHtml(_humaniseFullTimestamp(m.sent_at))}">
        ${avatarHtml}
        <div class="connect-message-stack">
          <div class="connect-message-bubble${isBurst ? ' emoji-burst' : ''}">
            ${replyHtml}
            ${attachmentHtml}
            ${bodyHtml}
            ${receiptHtml}
          </div>
          ${failedActionsHtml}
          ${reactionsHtml}
        </div>
      </div>
    `;
  }).join('');

  // Only follow to the bottom if the user was already there — otherwise
  // respect their scroll position so reading history isn't interrupted.
  // Runs after the current paint so heights are settled.
  if (wasNearBottom) requestAnimationFrame(_scrollMessagesToBottom);

  // Bind interactions on every row:
  //   - tap (click): set as reply
  //   - long-press (touch, 450ms): open emoji tray
  //   - right-click (desktop): open emoji tray
  //   - tap reaction pill: toggle own reaction
  for (const row of msgsEl.querySelectorAll('.connect-message-row')) {
    const mid = row.dataset.messageId;
    const bubble = row.querySelector('.connect-message-bubble');

    let tapTimer = null;
    row.addEventListener('click', (ev) => {
      // Tappable links open in a new tab — never treat as reply/react.
      if (ev.target.closest('a.connect-message-link')) return;
      // Image attachment → open the in-app lightbox (preventing the
      // anchor's new-tab navigation; right-click "open" still works).
      const imgLink = ev.target.closest('.connect-attachment-image');
      if (imgLink) {
        ev.preventDefault();
        if (!imgLink.classList.contains('is-broken')) {
          _openLightbox(imgLink.getAttribute('href') || '', {
            name: imgLink.dataset.imgName || '',
            downloadUrl: imgLink.dataset.dlUrl || '',
          });
        }
        return;
      }
      // Failed-action buttons take precedence over reply + reactions.
      const failedAction = ev.target.closest('[data-action]');
      if (failedAction && failedAction.closest('.connect-message-failed-actions')) {
        ev.stopPropagation();
        const action = failedAction.dataset.action;
        if (action === 'retry') _onFailedRetryClick(mid);
        else if (action === 'discard') _onFailedDiscardClick(mid);
        return;
      }
      // Reaction pill toggles that reaction.
      const pill = ev.target.closest('.connect-reaction-pill');
      if (pill) {
        ev.stopPropagation();
        _toggleReaction(pill.dataset.messageId, pill.dataset.emoji);
        return;
      }
      // Double-tap → open the tapback bar (iMessage-style reaction row),
      // on either side. A pending single-tap reply is cancelled here.
      if (tapTimer) {
        clearTimeout(tapTimer);
        tapTimer = null;
        ev.stopPropagation();
        if (bubble) _openTapbackBar(bubble, mid);
        return;
      }
      // Single tap: own bubbles do nothing; peer bubbles set a reply,
      // deferred briefly so a double-tap can pre-empt it. (The long-press
      // / right-click menu also offers an explicit Reply.)
      if (row.classList.contains('out')) {
        tapTimer = setTimeout(() => { tapTimer = null; }, 300);
        return;
      }
      tapTimer = setTimeout(() => {
        tapTimer = null;
        const m = byId.get(mid);
        if (m) _setReply(m);
      }, 280);
    });

    if (bubble) {
      bubble.addEventListener('contextmenu', (ev) => {
        ev.preventDefault();
        _openMessageMenu(bubble, mid);
      });
      let pressTimer = null;
      let pressFired = false;
      bubble.addEventListener('pointerdown', (ev) => {
        if (ev.pointerType !== 'touch') return;
        pressFired = false;
        pressTimer = setTimeout(() => {
          pressTimer = null;
          pressFired = true;
          _openMessageMenu(bubble, mid);
        }, 450);
      });
      const clearPress = () => {
        if (pressTimer) { clearTimeout(pressTimer); pressTimer = null; }
      };
      bubble.addEventListener('pointerup', clearPress);
      bubble.addEventListener('pointercancel', clearPress);
      bubble.addEventListener('pointerleave', clearPress);
      // Swallow the synthetic click that follows a touch long-press
      // so it doesn't fall through to set-reply.
      bubble.addEventListener('click', (ev) => {
        if (pressFired) {
          pressFired = false;
          ev.stopImmediatePropagation();
          ev.preventDefault();
        }
      }, { capture: true });
    }
  }

  // Call rows: click / Enter calls the peer back. In-progress calls
  // aren't callable (data-callable unset) so they're inert.
  for (const row of msgsEl.querySelectorAll('.connect-thread-call-row')) {
    if (row.dataset.callable !== '1') continue;
    const peer = row.dataset.peerDid;
    const video = row.dataset.video === '1';
    const callBack = async () => {
      if (!peer) return;
      try {
        const { startCall } = await import('./ui.js');
        await startCall(peer, { withVideo: video });
      } catch (err) {
        console.warn('connect: call back from timeline failed', err);
      }
    };
    row.addEventListener('click', callBack);
    row.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); callBack(); }
    });
  }

  // New-message accounting for the jump-to-latest pill: if the visible
  // count grew while the user was scrolled up, bump the unseen counter.
  // (Replaces an unconditional scroll-to-bottom that defeated the
  // scrolled-up reading position and this affordance.)
  const total = messages.length;
  const delta = total - _renderedMsgCount;
  if (delta > 0 && !wasNearBottom) _jumpNewCount += delta;
  _renderedMsgCount = total;
  _updateJumpButton();
}

function _updateJumpButton(dist) {
  if (!_panel) return;
  const btn = _panel.querySelector('.connect-thread-jump');
  if (!btn) return;
  if (dist === undefined) {
    const msgsEl = _panel.querySelector('.connect-thread-messages');
    dist = msgsEl ? (msgsEl.scrollHeight - msgsEl.scrollTop - msgsEl.clientHeight) : 0;
  }
  btn.hidden = !(dist > 200);
  const countEl = btn.querySelector('.connect-thread-jump-count');
  if (countEl) {
    if (_jumpNewCount > 0) {
      countEl.hidden = false;
      countEl.textContent = _jumpNewCount > 99 ? '99+' : String(_jumpNewCount);
      btn.classList.add('has-new');
    } else {
      countEl.hidden = true;
      btn.classList.remove('has-new');
    }
  }
}

// Call items for the open peer, shaped for the unified timeline.
function _callItemsForActivePeer() {
  if (!_activePeerDid) return [];
  return _recentCalls
    .filter((c) => (c.peer_did || '') === _activePeerDid && c.initiated_at)
    .map((c) => ({ kind: 'call', ts: c.initiated_at, call: c }));
}

// One inline call row for the message timeline (centered system row,
// distinct from message bubbles). Tap calls back when callable.
function _callTimelineRowHtml(call) {
  const missed = call.state === 'missed';
  const outgoing = call.direction === 'outgoing';
  const glyph = missed
    ? icon('phone-missed', { size: 16 })
    : (outgoing ? icon('arrow-up-right', { size: 16 }) : icon('arrow-down-left', { size: 16 }));
  const label = _callPreviewText(call);
  const time = escapeHtml(_humaniseClockTime(call.initiated_at));
  const callable = !!call.peer_did
    && !['connected', 'ringing', 'invited'].includes(call.state);
  return `
    <div class="connect-thread-call-row${missed ? ' missed' : ''}"
         data-call-id="${escapeHtml(call.call_id || '')}"
         data-peer-did="${escapeHtml(call.peer_did || '')}"
         data-video="${(call.modalities || '').includes('video') ? '1' : '0'}"
         data-callable="${callable ? '1' : '0'}"
         ${callable ? 'role="button" tabindex="0" title="Call back"' : ''}>
      <span class="connect-thread-call-glyph" aria-hidden="true">${glyph}</span>
      <span class="connect-thread-call-label">${escapeHtml(label)}</span>
      <span class="connect-thread-call-time">${time}</span>
    </div>
  `;
}

function _humaniseClockTime(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
  } catch (_) {
    return iso;
  }
}

// Full, human date+time for the message hover-title (replaces the raw
// ISO string that used to show on hover).
function _humaniseFullTimestamp(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return '';
    return d.toLocaleString(undefined, {
      weekday: 'short', month: 'short', day: 'numeric',
      hour: 'numeric', minute: '2-digit',
    });
  } catch (_) {
    return '';
  }
}

// Per-thread tracker: how far the peer has read in this thread.
// Keyed by thread_id, value = sent_at of the highest message they've
// read. We update this from inbound EVENT_TEXT_READ events.
const _peerReadAt = new Map();

function _receiptStateFor(message) {
  // 5-state machine, checked in priority order:
  //
  //   failed    — gave up after MAX_ATTEMPTS retries; sits in the
  //               outbox 'failed' bucket until user retry/discards
  //   pending   — still in the outbox queue (WS/HTTP didn't ack yet)
  //   read      — peer has read up through this message's sent_at
  //   delivered — server confirmed peer's WS got the message
  //               (delivered_at stamped via MSG_TEXT_DELIVERED ack
  //                or catch-up fetch side-effect)
  //   sent      — fallback: server stored the row, no peer ack yet
  //
  // 'failed' and 'pending' shadow the message body's own state
  // because outbox lifecycle is the truth for outbound delivery
  // — the message's delivered_at can't have fired if the send
  // didn't actually reach the server.
  if (outboxFindFailed(message.message_id)) return 'failed';
  if (outboxFind(message.message_id)) return 'pending';
  const peerCursor = _peerReadAt.get(message.thread_id);
  if (peerCursor && message.sent_at && message.sent_at <= peerCursor) {
    return 'read';
  }
  if (message.delivered_at) return 'delivered';
  return 'sent';
}

// ── Composer ───────────────────────────────────────────────────

async function _onStarterClick(kind, shouldSend) {
  if (!_panel) return;
  const input = _panel.querySelector('.connect-thread-composer-input');
  if (!input) return;
  const STARTERS = {
    wave:  '\u{1F44B}',
    free:  'You free?',
    voice: '',  // placeholder until voice notes are real
    share: '',
  };
  if (kind === 'voice') {
    _startVoiceRecording();
    return;
  }
  if (kind === 'share') {
    _panel.querySelector('.connect-thread-composer-aux[data-action="attach"]')?.click();
    return;
  }
  const text = STARTERS[kind] || '';
  if (!text) return;
  if (shouldSend) {
    input.value = text;
    input.dispatchEvent(new Event('input'));
    await _onSendClick();
  } else {
    input.value = text;
    input.dispatchEvent(new Event('input'));
    input.focus();
    // Position caret at end so user can append.
    try { input.setSelectionRange(text.length, text.length); } catch (_) {}
  }
}

// File-input change handler — STAGE the chosen files into the preview
// tray (they send when the user hits Send), rather than firing instantly.
function _onFilePicked(ev) {
  const input = ev.currentTarget || ev.target;
  const files = input && input.files;
  if (files && files.length) _stageFiles(files);
  if (input) input.value = '';  // allow re-picking the same file
}

// Clipboard paste of an image → stage it (e.g. screenshot → Ctrl/Cmd+V).
function _onComposerPaste(ev) {
  if (!_activeThreadId) return;
  const items = ev.clipboardData && ev.clipboardData.items;
  if (!items) return;
  const files = [];
  for (const it of items) {
    if (it.kind === 'file' && (it.type || '').startsWith('image/')) {
      const f = it.getAsFile();
      if (f) files.push(f);
    }
  }
  if (files.length) {
    ev.preventDefault();  // don't also paste a data-URL into the textarea
    _stageFiles(files);
  }
}

const _MAX_STAGED = 10;

// Add files to the staging tray (deduped-ish by size+name), capped.
function _stageFiles(fileList) {
  if (!_activeThreadId) {
    showToast('Open a conversation first', 'warning');
    return;
  }
  const incoming = Array.from(fileList || []);
  for (const file of incoming) {
    if (_stagedFiles.length >= _MAX_STAGED) {
      showToast(`You can attach up to ${_MAX_STAGED} at once`, 'info');
      break;
    }
    const mime = (file.type || '').toLowerCase();
    const kind = mime.startsWith('image/') ? (mime === 'image/gif' ? 'gif' : 'image')
      : mime.startsWith('video/') ? 'video'
      : mime.startsWith('audio/') ? 'audio' : 'file';
    let url = '';
    if (kind === 'image' || kind === 'gif' || kind === 'video') {
      try { url = URL.createObjectURL(file); } catch (_) { url = ''; }
    }
    _stagedFiles.push({ id: ++_stagedSeq, file, url, kind });
  }
  _renderStagedTray();
  _updateSendState?.();
  const input = _panel?.querySelector('.connect-thread-composer-input');
  if (input) input.focus();
}

function _renderStagedTray() {
  if (!_panel) return;
  const tray = _panel.querySelector('.connect-thread-staged');
  if (!tray) return;
  if (!_stagedFiles.length) { tray.hidden = true; tray.innerHTML = ''; return; }
  tray.hidden = false;
  tray.innerHTML = _stagedFiles.map((s) => {
    const thumb = (s.kind === 'image' || s.kind === 'gif')
      ? `<img class="connect-staged-thumb" src="${escapeHtml(s.url)}" alt="">`
      : (s.kind === 'video'
        ? `<video class="connect-staged-thumb" src="${escapeHtml(s.url)}" muted preload="metadata"></video>
           <span class="connect-staged-play" aria-hidden="true">&#9654;</span>`
        : `<span class="connect-staged-fileicon">${icon('file', { size: 20 })}</span>`);
    const badge = s.kind === 'gif' ? '<span class="connect-staged-badge">GIF</span>' : '';
    const nameLabel = (s.kind === 'file' || s.kind === 'audio')
      ? `<span class="connect-staged-name">${escapeHtml(s.file.name || 'file')}</span>` : '';
    return `
      <div class="connect-staged-item kind-${s.kind}" data-staged-id="${s.id}">
        ${thumb}${badge}${nameLabel}
        <button class="connect-staged-remove" type="button" data-staged-id="${s.id}"
                aria-label="Remove attachment" title="Remove">&#x2715;</button>
      </div>`;
  }).join('');
  for (const btn of tray.querySelectorAll('.connect-staged-remove')) {
    btn.addEventListener('click', () => _removeStaged(parseInt(btn.dataset.stagedId, 10)));
  }
}

function _removeStaged(id) {
  const idx = _stagedFiles.findIndex((s) => s.id === id);
  if (idx === -1) return;
  const [removed] = _stagedFiles.splice(idx, 1);
  if (removed && removed.url) { try { URL.revokeObjectURL(removed.url); } catch (_) {} }
  _renderStagedTray();
  _updateSendState?.();
}

function _clearStaged() {
  for (const s of _stagedFiles) { if (s.url) { try { URL.revokeObjectURL(s.url); } catch (_) {} } }
  _stagedFiles = [];
  _renderStagedTray();
  _updateSendState?.();
}

// Upload one file and send it as an attachment message (optionally with a
// caption as the body). Shared by the staged-send flow and voice messages.
async function _uploadAndSendFile(file, { body = '' } = {}) {
  const thread = _threadCache.get(_activeThreadId);
  if (!thread) { showToast('Open a thread first', 'warning'); return false; }
  const name = file.name || 'file';
  let lastPct = 0;
  const pctToast = showToast(`Uploading ${name}…`, 'info');
  try {
    const upload = await uploadAttachment(file, {
      onProgress: (frac) => {
        const pct = Math.floor(frac * 100);
        if (pct >= lastPct + 10) {
          lastPct = pct;
          if (pctToast && pctToast.textContent !== undefined) {
            pctToast.textContent = `Uploading ${name}… ${pct}%`;
          }
        }
      },
    });
    const result = await sendMessage({
      peerDid: thread.peer_did,
      threadId: thread.thread_id,
      body,
      format: 'plain',
      attachmentRef: upload.upload_id,
      attachmentName: upload.filename,
      attachmentMime: upload.mime,
      attachmentSize: upload.size,
    });
    const msgs = _messageCache.get(_activeThreadId) || [];
    msgs.unshift({
      message_id: result.message_id,
      thread_id: result.thread_id || _activeThreadId,
      user_id: '',
      sender_did: _ownDid(),
      body,
      format: 'plain',
      attachment_ref: upload.upload_id,
      attachment_name: upload.filename,
      attachment_mime: upload.mime,
      attachment_size: upload.size,
      reply_to: '',
      sent_at: new Date().toISOString(),
      received_at: new Date().toISOString(),
      delivered_at: null,
      read_at: null,
      edited_at: null,
      deleted_at: null,
      transcript: '',
    });
    _messageCache.set(_activeThreadId, msgs);
    _scheduleMessageRender();
    return true;
  } catch (err) {
    showToast(_friendlyMessageError(err, "Couldn't attach that file. Try again."), 'error');
    return false;
  }
}

async function _onSendClick() {
  if (!_panel) return;
  const thread = _threadCache.get(_activeThreadId);
  if (!thread) {
    showToast('Pick a conversation first', 'info');
    return;
  }
  const input = _panel.querySelector('.connect-thread-composer-input');
  const sendBtn = _panel.querySelector('.connect-thread-composer-send');
  const body = (input.value || '').trim();

  // Staged attachments → send each (the typed text rides the FIRST as a
  // caption); a text-only send falls through to the normal path below.
  if (_stagedFiles.length > 0) {
    const items = _stagedFiles.slice();
    _clearStaged();
    input.value = '';
    input.dispatchEvent(new Event('input'));
    _flushTypingStop();
    sendBtn.disabled = true;
    try {
      for (let i = 0; i < items.length; i++) {
        await _uploadAndSendFile(items[i].file, { body: i === 0 ? body : '' });
        if (items[i].url) { try { URL.revokeObjectURL(items[i].url); } catch (_) {} }
      }
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
    return;
  }

  if (!body) return;
  // Stop signalling "typing" — we're about to send.
  _flushTypingStop();
  const replyTo = _replyContext ? _replyContext.message_id : '';
  sendBtn.disabled = true;
  try {
    const result = await sendMessage({
      peerDid: thread.peer_did,
      threadId: thread.thread_id,
      body,
      format: 'plain',
      replyTo,
    });
    // Optimistic local insert at head (newest-first cache order).
    const msgs = _messageCache.get(_activeThreadId) || [];
    msgs.unshift({
      message_id: result.message_id,
      thread_id: result.thread_id || _activeThreadId,
      user_id: '',
      sender_did: _ownDid(),
      body,
      format: 'plain',
      attachment_ref: '',
      reply_to: replyTo,
      sent_at: new Date().toISOString(),
      received_at: new Date().toISOString(),
      delivered_at: result.routed > 0 ? new Date().toISOString() : null,
      read_at: null,
      edited_at: null,
      deleted_at: null,
      transcript: '',
    });
    _messageCache.set(_activeThreadId, msgs);
    _scheduleMessageRender();
    _clearReply();
    input.value = '';
    input.dispatchEvent(new Event('input'));  // reset autosize + send-state
  } catch (err) {
    showToast(_friendlyMessageError(err, "Couldn't send your message. Try again."), 'error');
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

// (_startNewThread was replaced by the contact picker dialog;
// _openContactPicker is wired to the "+ New" button now.)

// ── Event handlers (WS) ────────────────────────────────────────

function _onReceived(evt) {
  const data = evt.detail || {};
  const tid = data.thread_id;
  if (!tid) return;
  // Cache + bump unread if not the active thread.
  if (!_threadCache.has(tid)) {
    _threadCache.set(tid, {
      thread_id: tid,
      peer_did: data.sender_did || '',
      // Seed with the resolver so the thread row in the list shows a
      // human name immediately on first message, not the raw DID.
      // The next listThreads() refresh replaces this with the
      // canonical peer_display_name from the server row.
      peer_display_name: resolvePeerName(data.sender_did || ''),
      last_message_at: data.sent_at || new Date().toISOString(),
      last_message_preview: (data.body || '').slice(0, 200),
      unread_count: 1,
      muted: false, pinned: false, archived: false,
      created_at: new Date().toISOString(),
    });
  } else {
    const t = _threadCache.get(tid);
    t.last_message_at = data.sent_at || new Date().toISOString();
    t.last_message_preview = (data.body || '').slice(0, 200);
    if (tid !== _activeThreadId) t.unread_count = (t.unread_count || 0) + 1;
  }
  // Append to message cache for the affected thread.
  const msgs = _messageCache.get(tid) || [];
  msgs.unshift({
    message_id: data.message_id,
    thread_id: tid,
    user_id: '',
    sender_did: data.sender_did,
    body: data.body || '',
    format: data.format || 'plain',
    attachment_ref: data.attachment_ref || '',
    // Sender's metadata is hint-only; canonical lives on the
    // sender's uploads row. Catch-up path can HEAD the attachment
    // route to refresh these for rows that lacked the live event.
    attachment_name: data.attachment_name || '',
    attachment_mime: data.attachment_mime || '',
    attachment_size: data.attachment_size || 0,
    reply_to: data.reply_to || '',
    sent_at: data.sent_at || new Date().toISOString(),
    received_at: new Date().toISOString(),
    delivered_at: null,
    read_at: null,
    edited_at: null,
    deleted_at: null,
    transcript: data.transcript || '',
  });
  _messageCache.set(tid, msgs);

  // Keep the catch-up cursor current so a future reconnect doesn't
  // refetch this message. (messages.js also bumps the cursor before
  // firing the DOM event — this is the belt-and-suspenders write.)
  if (data.sent_at) setThreadCursor(tid, data.sent_at);

  if (_panel && !_panel.classList.contains('hidden')) {
    _renderThreadList();
    if (tid === _activeThreadId) _scheduleMessageRender();
  }
}

function _onEdited(evt) {
  const data = evt.detail || {};
  const msgs = _messageCache.get(data.thread_id);
  if (!msgs) return;
  const target = msgs.find((m) => m.message_id === data.message_id);
  if (!target) return;
  target.body = data.body || '';
  target.edited_at = new Date().toISOString();
  if (_panel && _activeThreadId === data.thread_id) _scheduleMessageRender();
}

function _onDeleted(evt) {
  const data = evt.detail || {};
  const msgs = _messageCache.get(data.thread_id);
  if (!msgs) return;
  const target = msgs.find((m) => m.message_id === data.message_id);
  if (!target) return;
  target.body = '';
  target.deleted_at = new Date().toISOString();
  if (_panel && _activeThreadId === data.thread_id) _scheduleMessageRender();
}

function _onReadReceipt(evt) {
  const data = evt.detail || {};
  const tid = data.thread_id;
  if (!tid) return;
  // We don't get a precise per-message read_at from the receipt — we
  // get "the peer read up to last_read_message_id". Look it up in
  // our cache and store the corresponding sent_at as the read cursor.
  const lastReadId = data.last_read_message_id || '';
  const msgs = _messageCache.get(tid);
  let cursor = '';
  if (lastReadId && msgs) {
    const target = msgs.find((m) => m.message_id === lastReadId);
    if (target) cursor = target.sent_at || '';
  }
  if (cursor) {
    _peerReadAt.set(tid, cursor);
    if (_panel && _activeThreadId === tid) _scheduleMessageRender();
  }
}

function _onDeliveredReceipt(evt) {
  const data = evt.detail || {};
  const tid = data.thread_id;
  const ids = Array.isArray(data.message_ids) ? data.message_ids : [];
  if (!tid || !ids.length) return;
  const msgs = _messageCache.get(tid);
  if (!msgs) return;
  let changed = false;
  const now = new Date().toISOString();
  for (const m of msgs) {
    if (!ids.includes(m.message_id)) continue;
    if (m.delivered_at) continue;
    m.delivered_at = now;
    changed = true;
  }
  if (changed && _panel && _activeThreadId === tid) {
    _scheduleMessageRender();
  }
}

let _outboxRenderScheduled = false;

function _onOutboxChanged() {
  // Coalesce: a flush that resolves 12 queued sends fires onChange
  // 12 times. We only need one re-render per frame.
  if (_outboxRenderScheduled) return;
  if (!_panel || _panel.classList.contains('hidden')) return;
  if (!_activeThreadId) return;
  _outboxRenderScheduled = true;
  requestAnimationFrame(() => {
    _outboxRenderScheduled = false;
    if (!_panel || _panel.classList.contains('hidden')) return;
    const msgs = _messageCache.get(_activeThreadId);
    if (msgs) _scheduleMessageRender();
  });
}

let _reconnectBannerTimer = null;

function _showReconnectBanner(label = 'Reconnected — catching up…', { state = 'busy' } = {}) {
  if (!_panel) return;
  let banner = _panel.querySelector('.connect-thread-reconnect-banner');
  if (!banner) {
    banner = document.createElement('div');
    banner.className = 'connect-thread-reconnect-banner';
    banner.setAttribute('role', 'status');
    banner.setAttribute('aria-live', 'polite');
    const active = _panel.querySelector('.connect-thread-active');
    if (active) active.prepend(banner);
  }
  // ``state`` swaps the leading glyph between an animated spinner
  // ('busy') and a static check ('done'). Anything else falls back
  // to plain text so callers can pass arbitrary labels safely.
  const glyph = state === 'done'
    ? icon('check', { size: 14 })
    : state === 'busy'
      ? `<span class="connect-thread-reconnect-spinner" aria-hidden="true"></span>`
      : '';
  banner.classList.toggle('state-done', state === 'done');
  banner.classList.toggle('state-busy', state === 'busy');
  banner.innerHTML = `
    ${glyph ? `<span class="connect-thread-reconnect-glyph">${glyph}</span>` : ''}
    <span class="connect-thread-reconnect-text">${escapeHtml(label)}</span>
  `;
  banner.classList.remove('hidden');
  banner.classList.add('visible');
}

function _hideReconnectBanner({ delayMs = 0 } = {}) {
  if (!_panel) return;
  if (_reconnectBannerTimer) clearTimeout(_reconnectBannerTimer);
  const finish = () => {
    const banner = _panel?.querySelector('.connect-thread-reconnect-banner');
    if (!banner) return;
    banner.classList.remove('visible');
    // Wait for the fade-out before removing from layout — keep DOM
    // around so a follow-up reconnect doesn't re-create churn.
    setTimeout(() => banner.classList.add('hidden'), 220);
  };
  if (delayMs > 0) {
    _reconnectBannerTimer = setTimeout(finish, delayMs);
  } else {
    finish();
  }
}

async function _onReconnected() {
  _showReconnectBanner();
  // For every thread we already have cached state for, ask the
  // server for anything newer than our persisted cursor. The fetch
  // itself triggers server-side delivery acks for the messages
  // it returns, so a peer who sent while we were offline sees
  // their "sent" tick flip to "delivered" right after our reconnect.
  const threadIds = Array.from(_threadCache.keys());
  let anyNew = 0;
  for (const tid of threadIds) {
    try {
      const fresh = await catchUpThread(tid);
      if (fresh.length) anyNew += fresh.length;
      if (!fresh.length) continue;
      const existing = _messageCache.get(tid) || [];
      // catchUpThread returns oldest-first; our cache is newest-first.
      // Prepend in reverse so cache stays newest-first.
      const seen = new Set(existing.map((m) => m.message_id));
      for (let i = fresh.length - 1; i >= 0; i -= 1) {
        const m = fresh[i];
        if (seen.has(m.message_id)) continue;
        existing.unshift(m);
      }
      _messageCache.set(tid, existing);
      // Refresh thread tail preview locally so the list reorders.
      const newest = fresh[fresh.length - 1];
      const t = _threadCache.get(tid);
      if (t && newest) {
        t.last_message_at = newest.sent_at || t.last_message_at;
        t.last_message_preview = (newest.body || '').slice(0, 200);
        if (tid !== _activeThreadId) {
          t.unread_count = (t.unread_count || 0) + fresh.length;
        }
      }
      if (_panel && !_panel.classList.contains('hidden')) {
        _renderThreadList();
        if (tid === _activeThreadId) _renderMessages(existing);
      }
    } catch (err) {
      console.warn('connect: catch-up failed for', tid, err);
    }
  }
  // Catch-up done — flip the banner to a 1.2s success state then
  // fade out. If no new messages came in, we still show a brief
  // "caught up" tick so the user knows the reconnect resolved.
  _showReconnectBanner(
    anyNew > 0
      ? `Caught up — ${anyNew} new message${anyNew === 1 ? '' : 's'}`
      : 'Caught up',
    { state: 'done' },
  );
  _hideReconnectBanner({ delayMs: 1500 });
}

// ── Cross-tab sync ─────────────────────────────────────────────

/**
 * A sibling tab on the same account just mutated ``threadId``. Pull
 * anything we don't already have for that thread so our UI matches.
 *
 * Reuses ``catchUpThread`` (the same machinery the WS-reconnect path
 * uses) so we get delivery-ack side effects for free if the sibling
 * tab was the one that sent a new outbound message.
 *
 * Cheap when nothing changed — the server returns 0 rows past our
 * cursor and the loop body is a no-op.
 */
async function _onSiblingThreadChange(threadId) {
  // Threads we've never opened locally aren't in the cache; skip
  // them — opening the panel will fetch them fresh on demand.
  if (!_threadCache.has(threadId)) return;

  let fresh = [];
  try {
    fresh = await catchUpThread(threadId);
  } catch (err) {
    // Network blips are non-fatal — next sibling change re-triggers.
    console.warn('connect: sibling catch-up failed for', threadId, err);
    return;
  }
  if (!fresh.length) {
    // Even with no new rows, edits / deletes / reactions wouldn't
    // change message_count — but they also wouldn't bump sent_at,
    // so the catch-up returns nothing. For those, a full thread
    // refresh would be needed; deferred until we surface a wider
    // sibling-event taxonomy. The send/receipt fast path (the most
    // common cross-tab gap) is fully covered by the catch-up.
    return;
  }
  const existing = _messageCache.get(threadId) || [];
  const seen = new Set(existing.map((m) => m.message_id));
  for (let i = fresh.length - 1; i >= 0; i -= 1) {
    const m = fresh[i];
    if (seen.has(m.message_id)) continue;
    existing.unshift(m);
  }
  _messageCache.set(threadId, existing);
  const newest = fresh[fresh.length - 1];
  const t = _threadCache.get(threadId);
  if (t && newest) {
    t.last_message_at = newest.sent_at || t.last_message_at;
    t.last_message_preview = (newest.body || '').slice(0, 200);
  }
  if (_panel && !_panel.classList.contains('hidden')) {
    _renderThreadList();
    if (threadId === _activeThreadId) _renderMessages(existing);
  }
}


// ── Reply context ──────────────────────────────────────────────

function _setReply(message) {
  if (!message || message.deleted_at) return;
  _replyContext = {
    message_id: message.message_id,
    sender_did: message.sender_did,
    body: (message.body || '').slice(0, 200),
  };
  _renderReplyBar();
  const input = _panel?.querySelector('.connect-thread-composer-input');
  if (input) input.focus();
}

function _clearReply() {
  _replyContext = null;
  _renderReplyBar();
}

function _renderReplyBar() {
  if (!_panel) return;
  const bar = _panel.querySelector('.connect-thread-reply');
  if (!bar) return;
  if (!_replyContext) {
    bar.hidden = true;
    return;
  }
  bar.hidden = false;
  const myDid = _ownDid();
  const who = _replyContext.sender_did === myDid
    ? 'Replying to yourself'
    : `Replying to ${escapeHtml(resolvePeerName(_replyContext.sender_did))}`;
  bar.querySelector('.connect-thread-reply-text').innerHTML = `
    <span class="connect-thread-reply-who">${who}</span>
    <span class="connect-thread-reply-body">${escapeHtml(_replyContext.body)}</span>
  `;
}

// ── Typing indicators ──────────────────────────────────────────

function _onComposerInput() {
  const thread = _threadCache.get(_activeThreadId);
  if (!thread || !thread.peer_did) return;
  if (!_isTypingActive) {
    sendTyping(thread.peer_did, thread.thread_id, true);
    _isTypingActive = true;
  }
  if (_typingTimer) clearTimeout(_typingTimer);
  _typingTimer = setTimeout(_flushTypingStop, TYPING_DEBOUNCE_MS);
}

function _flushTypingStop() {
  if (!_isTypingActive) return;
  const thread = _threadCache.get(_activeThreadId);
  if (thread && thread.peer_did) {
    sendTyping(thread.peer_did, thread.thread_id, false);
  }
  _isTypingActive = false;
  if (_typingTimer) { clearTimeout(_typingTimer); _typingTimer = null; }
}

function _onTypingStart(evt) {
  const data = evt.detail || {};
  const tid = data.thread_id;
  if (!tid) return;
  _typingState.set(tid, {
    peerDid: data.sender_did || '',
    expiresAt: Date.now() + TYPING_INDICATOR_STALE_MS,
  });
  _renderTypingIndicator();
  // Header status line piggybacks on typing — render once so the
  // 'Online' label flips to 'Typing…' immediately. Bail cheaply if
  // the typing event is for a thread we're not viewing.
  if (tid === _activeThreadId) {
    const t = _threadCache.get(_activeThreadId);
    if (t) _renderActiveHeader(t);
  }
  // Auto-clear if no MSG_TYPING_STOP arrives — typing indicators are
  // a "presence" signal; stale entries should auto-decay.
  setTimeout(() => {
    const entry = _typingState.get(tid);
    if (entry && entry.expiresAt <= Date.now()) {
      _typingState.delete(tid);
      _renderTypingIndicator();
    }
  }, TYPING_INDICATOR_STALE_MS + 50);
}

function _onTypingStop(evt) {
  const data = evt.detail || {};
  const tid = data.thread_id;
  if (!tid) return;
  _typingState.delete(tid);
  _renderTypingIndicator();
  if (tid === _activeThreadId) {
    const t = _threadCache.get(_activeThreadId);
    if (t) _renderActiveHeader(t);
  }
}

function _renderTypingIndicator() {
  if (!_panel) return;
  const el = _panel.querySelector('.connect-thread-typing');
  if (!el) return;
  const entry = _typingState.get(_activeThreadId);
  if (!entry || entry.expiresAt <= Date.now()) {
    el.hidden = true;
    el.innerHTML = '';
    return;
  }
  el.hidden = false;
  // Bubble-pill indicator: avatar circle + bubble with 3 animated dots.
  // The animation is pure CSS (`@keyframes connect-typing-bounce`). The
  // dots are decorative (aria-hidden); a visually-hidden line carries the
  // meaningful "<name> is typing" text for the role=status live region.
  const initial = _initialFor(entry.peerDid || 'peer');
  const typerName = (entry.peerDid && resolvePeerName(entry.peerDid)) || 'They';
  el.innerHTML = `
    <span class="connect-typing-avatar" aria-hidden="true">${escapeHtml(initial)}</span>
    <span class="connect-typing-bubble" aria-hidden="true">
      <span class="connect-typing-dot"></span>
      <span class="connect-typing-dot"></span>
      <span class="connect-typing-dot"></span>
    </span>
    <span class="connect-sr-only">${escapeHtml(typerName)} is typing…</span>
  `;
}

// ── Contact picker ─────────────────────────────────────────────

function _openContactPicker() {
  if (!_panel) return;
  const picker = _panel.querySelector('.connect-contact-picker');
  if (!picker) return;
  picker.classList.remove('hidden');
  const input = picker.querySelector('.connect-contact-picker-input');
  if (input) { input.value = ''; setTimeout(() => input.focus(), 0); }
  _wireContactPickerSearch();
  _populateContactPickerList().catch((err) => {
    console.warn('connect: contact list fetch failed', err);
  });
}

function _closeContactPicker() {
  if (!_panel) return;
  const picker = _panel.querySelector('.connect-contact-picker');
  if (picker) picker.classList.add('hidden');
}

let _pickerSearchTimer = null;

// Merge everyone reachable on this machine (the directory) with the user's
// saved contacts, deduped by DID. People on this machine come first so any
// account here is one tap away — no DID pasting needed.
async function _populateContactPickerList(query = '') {
  if (!_panel) return;
  const listEl = _panel.querySelector('.connect-contact-picker-list');
  if (!listEl) return;

  const q = (query || '').trim();
  let people = [];
  let contacts = [];
  try {
    if (q) {
      // Server-side search finds anyone on the machine, not just what's cached.
      people = (await searchPeers(q)).people || [];
    } else {
      const [dir, cts] = await Promise.all([
        listDirectory().catch(() => ({ people: [] })),
        listContacts({ includeBlocked: false }).catch(() => []),
      ]);
      people = dir.people || [];
      contacts = cts || [];
    }
  } catch {
    listEl.innerHTML = `<div class="connect-contact-picker-empty">Couldn't load people.</div>`;
    return;
  }

  const seen = new Set();
  const rows = [];
  for (const p of people) {
    if (!p.peer_did || seen.has(p.peer_did)) continue;
    seen.add(p.peer_did);
    rows.push({ did: p.peer_did, name: p.display_name || resolvePeerName(p.peer_did), online: !!p.online });
  }
  for (const c of contacts) {
    if (!c.peer_did || seen.has(c.peer_did)) continue;
    seen.add(c.peer_did);
    rows.push({ did: c.peer_did, name: (c.peer_display_name || '').trim() || resolvePeerName(c.peer_did), online: false });
  }

  if (!rows.length) {
    listEl.innerHTML = `<div class="connect-contact-picker-empty">${q ? 'No one matches.' : 'No one to message yet.'}</div>`;
    return;
  }
  listEl.innerHTML = rows.map((r) => `
    <div class="connect-contact-picker-row-item" data-peer-did="${escapeHtml(r.did)}">
      <span class="connect-contact-presence presence-${r.online ? 'online' : 'offline'}"></span>
      <div class="connect-contact-picker-name">${escapeHtml(r.name)}</div>
      ${peerSubtitle(r.did) ? `<div class="connect-contact-picker-did">${escapeHtml(peerSubtitle(r.did))}</div>` : ''}
    </div>
  `).join('');
  for (const row of listEl.querySelectorAll('.connect-contact-picker-row-item')) {
    row.addEventListener('click', async () => {
      const peerDid = row.dataset.peerDid;
      _closeContactPicker();
      await _openOrCreateThreadForPeer(peerDid);
    });
  }
}

// Wire the picker's input as a live search-as-you-type box (debounced).
function _wireContactPickerSearch() {
  if (!_panel) return;
  const input = _panel.querySelector('.connect-contact-picker-input');
  if (!input || input.dataset.searchWired === '1') return;
  input.dataset.searchWired = '1';
  input.setAttribute('placeholder', 'Search people, or paste a DID');
  input.addEventListener('input', () => {
    const v = input.value || '';
    clearTimeout(_pickerSearchTimer);
    // A DID (has @) is for the Add button; a plain term searches the directory.
    _pickerSearchTimer = setTimeout(() => {
      _populateContactPickerList(v.includes('@') ? '' : v).catch(() => {});
    }, 220);
  });
}

async function _onPickerAddClick() {
  if (!_panel) return;
  const input = _panel.querySelector('.connect-contact-picker-input');
  const peerDid = (input.value || '').trim();
  if (!peerDid || !peerDid.includes('@')) {
    showToast('Enter an address like alice@home.alice.dev', 'warning');
    return;
  }
  try {
    // Materialise a contact row so the peer shows up next time.
    await addContact({ peerDid });
  } catch (err) {
    console.warn('connect: addContact failed', err);
  }
  _closeContactPicker();
  await _openOrCreateThreadForPeer(peerDid);
}

async function _openOrCreateThreadForPeer(peerDid) {
  // If we already have a thread for this peer (canonical or temp),
  // open it. Otherwise mint a temp entry and route through the
  // existing _openThread path.
  for (const t of _threadCache.values()) {
    if (t.peer_did === peerDid) {
      _activeThreadId = t.thread_id;
      await _openThread(t.thread_id);
      return;
    }
  }
  const placeholder = {
    thread_id: `tmp:${peerDid}`,
    peer_did: peerDid,
    // Seed the placeholder with a resolved name so the active-thread
    // header doesn't flash the raw DID before the first message
    // round-trips. The server-issued thread row that replaces this
    // placeholder will carry the canonical peer_display_name.
    peer_display_name: resolvePeerName(peerDid),
    last_message_at: '',
    last_message_preview: '',
    unread_count: 0,
    muted: false, pinned: false, archived: false,
    created_at: new Date().toISOString(),
  };
  _threadCache.set(placeholder.thread_id, placeholder);
  _activeThreadId = placeholder.thread_id;
  _renderThreadList();
  await _openThread(placeholder.thread_id);
}

// ── Helpers ────────────────────────────────────────────────────

function _initialFor(s) {
  // Prefer the first letter of the *display name* the contact carries
  // so avatar pills read as B / M / J instead of the U from every
  // synthetic usr_... DID.
  const raw = String(s || '').trim();
  if (!raw) return '?';
  let name = '';
  try { name = String(resolvePeerName(raw) || '').trim(); } catch (_) { name = ''; }
  if (!name || name === raw) {
    const beforeAt = raw.split('@')[0] || raw;
    name = beforeAt.replace(/^usr[_-]/i, '');
  }
  // First Unicode-aware codepoint (handles surrogate pairs).
  const first = Array.from(name)[0] || '?';
  return first.toUpperCase();
}

function _dayKey(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso.slice(0, 10);
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  } catch (_) { return iso.slice(0, 10); }
}

function _humaniseDayLabel(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return '';
    const now = new Date();
    const todayKey = _dayKey(now.toISOString());
    const yesterday = new Date(now);
    yesterday.setDate(now.getDate() - 1);
    const yKey = _dayKey(yesterday.toISOString());
    const dKey = _dayKey(iso);
    if (dKey === todayKey) return 'Today';
    if (dKey === yKey) return 'Yesterday';
    // Within last 7 days: weekday name. Otherwise full date.
    const diffDays = Math.floor((now.getTime() - d.getTime()) / (86400 * 1000));
    if (diffDays < 7) {
      return d.toLocaleDateString(undefined, { weekday: 'long' });
    }
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: d.getFullYear() !== now.getFullYear() ? 'numeric' : undefined });
  } catch (_) { return ''; }
}

// ── Reactions ──────────────────────────────────────────────────
//
// Pill stack rendered under each bubble. Each pill shows the emoji
// + count; tapping toggles your own reaction on that emoji (add if
// you haven't, remove if you have).

// The iMessage-style tapback lineup (thumbs-up, heart, thumbs-down,
// haha, ‼️, star). Shown by the double-tap tapback bar and the
// long-press menu tray, both with a "+" for any keyboard emoji.
const QUICK_REACTIONS = ['\u{1F44D}', '\u{2764}\u{FE0F}', '\u{1F44E}', '\u{1F602}', '\u{203C}\u{FE0F}', '\u{2B50}'];

// Emojis the current user has already applied to a message (for
// highlighting the active tapbacks).
function _ownReactionSet(m) {
  const out = new Set();
  const myDid = _ownDid();
  for (const r of (Array.isArray(m.reactions) ? m.reactions : [])) {
    if (Array.isArray(r.reactor_dids) && r.reactor_dids.includes(myDid)) out.add(r.emoji);
  }
  return out;
}

let _tapbackBar = null;

// The quick reaction bar (iMessage tapback). Opened by a double-tap on
// a bubble: a floating pill row of the standard reactions + a "+" that
// opens the full emoji picker for any keyboard emoji.
function _openTapbackBar(bubbleEl, messageId) {
  _closeTapbackBar();
  const msgs = _messageCache.get(_activeThreadId) || [];
  const m = msgs.find((x) => x.message_id === messageId);
  if (!m || m.deleted_at) return;
  const mine = _ownReactionSet(m);

  const bar = document.createElement('div');
  bar.className = 'connect-tapback-bar';
  bar.setAttribute('role', 'menu');
  bar.setAttribute('aria-label', 'React');
  bar.innerHTML = `
    ${QUICK_REACTIONS.map((e) => `
      <button class="connect-tapback-emoji${mine.has(e) ? ' active' : ''}" type="button"
              data-emoji="${escapeHtml(e)}" aria-label="React ${escapeHtml(e)}">${escapeHtml(e)}</button>
    `).join('')}
    <button class="connect-tapback-more" type="button" data-action="more"
            title="More emojis" aria-label="More emojis">${icon('plus', { size: 16 })}</button>
  `;
  document.body.appendChild(bar);
  _tapbackBar = bar;

  // Anchor centered above the bubble, flipped below if no headroom.
  const rect = bubbleEl.getBoundingClientRect();
  const bw = bar.offsetWidth || 280;
  const bh = bar.offsetHeight || 46;
  let top = rect.top - bh - 8;
  if (top < 8) top = Math.min(rect.bottom + 8, window.innerHeight - bh - 8);
  let left = rect.left + rect.width / 2 - bw / 2;
  left = Math.max(8, Math.min(left, window.innerWidth - bw - 8));
  bar.style.top = `${top}px`;
  bar.style.left = `${left}px`;
  requestAnimationFrame(() => bar.classList.add('open'));

  bar.addEventListener('click', (ev) => {
    const more = ev.target.closest('.connect-tapback-more');
    if (more) {
      _closeTapbackBar();
      import('./emoji-picker.js')
        .then((mod) => mod.openEmojiPicker(bubbleEl, (emoji) => _toggleReaction(messageId, emoji)))
        .catch(() => {});
      return;
    }
    const cell = ev.target.closest('.connect-tapback-emoji');
    if (cell) { const emoji = cell.dataset.emoji; _closeTapbackBar(); _toggleReaction(messageId, emoji); }
  });

  const onOutside = (e) => { if (!_tapbackBar) return; if (_tapbackBar.contains(e.target)) return; _closeTapbackBar(); };
  const onKey = (e) => { if (e.key === 'Escape') _closeTapbackBar(); };
  setTimeout(() => {
    document.addEventListener('pointerdown', onOutside, true);
    document.addEventListener('keydown', onKey);
  }, 0);
  bar._cleanup = () => {
    document.removeEventListener('pointerdown', onOutside, true);
    document.removeEventListener('keydown', onKey);
  };
}

function _closeTapbackBar() {
  if (!_tapbackBar) return;
  try { _tapbackBar._cleanup?.(); } catch (_) {}
  const b = _tapbackBar;
  b.classList.remove('open');
  _tapbackBar = null;
  setTimeout(() => { try { b.remove(); } catch (_) {} }, 140);
}

async function _onFailedRetryClick(messageId) {
  try {
    const it = await retryFailedSend(messageId);
    if (!it) {
      showToast('Nothing to retry', 'warning');
      return;
    }
    showToast('Retrying…', 'info');
    // The outbox onChange listener (wired in initConnectMessagingUI)
    // will re-render the bubble to 'pending' / 'sent' / 'failed'
    // as the flush resolves — no explicit render here.
  } catch (err) {
    showToast(`Retry failed: ${err?.message || err}`, 'error');
  }
}

function _onFailedDiscardClick(messageId) {
  const removed = outboxDiscard(messageId);
  if (!removed) return;
  // Also remove the bubble from the local cache. The server-side
  // row was never created (the send never succeeded) so there's
  // nothing to soft-delete on the server.
  const msgs = _messageCache.get(_activeThreadId);
  if (msgs) {
    const idx = msgs.findIndex((m) => m.message_id === messageId);
    if (idx !== -1) {
      msgs.splice(idx, 1);
      _scheduleMessageRender();
    }
  }
}

function _renderFailedActions(m) {
  // Two-button row: Retry re-enqueues + flushes; Discard removes the
  // failed entry permanently (the message bubble stays but the tick
  // flips back to 'sent' since there's no outbox record).
  const errMsg = (outboxFindFailed(m.message_id) || {}).last_error || '';
  return `
    <div class="connect-message-failed-actions" role="group"
         data-message-id="${escapeHtml(m.message_id)}"
         aria-label="Failed message actions">
      ${errMsg ? `<span class="connect-message-failed-reason"
        title="${escapeHtml(errMsg)}">${escapeHtml(_summariseError(errMsg))}</span>` : ''}
      <button type="button" class="connect-message-failed-retry"
              data-action="retry"
              aria-label="Retry send">
        <span class="connect-message-failed-glyph">${icon('refresh', { size: 12 })}</span>
        Retry
      </button>
      <button type="button" class="connect-message-failed-discard"
              data-action="discard"
              aria-label="Discard message">
        <span class="connect-message-failed-glyph">${icon('trash', { size: 12 })}</span>
        Discard
      </button>
    </div>
  `;
}

function _summariseError(raw) {
  // The full error often includes a stack trace fragment or HTTP
  // status code. Surface just the first line — the full text lives
  // in the title attribute for hover.
  const first = (raw || '').split(/\n|:/)[0].trim();
  return first.length > 60 ? `${first.slice(0, 57)}…` : first;
}

// ── Voice messages ─────────────────────────────────────────────
// Tap the mic (empty composer) → record → tap send to ship it through
// the normal attachment pipeline (renders as a playable audio bubble),
// or tap the trash to discard.
async function _startVoiceRecording() {
  if (_voiceRecHandle) return;  // already recording
  const thread = _threadCache.get(_activeThreadId);
  if (!thread) { showToast('Open a conversation first', 'warning'); return; }
  let mod;
  try {
    mod = await import('./voice-recorder.js');
  } catch (_) { showToast('Voice recording is unavailable', 'warning'); return; }
  if (!mod.isVoiceRecordingSupported()) {
    showToast("Voice recording isn't supported on this device", 'warning');
    return;
  }
  const timeEl = _panel.querySelector('.connect-thread-rec-time');
  _voiceElapsed = 0;
  try {
    _voiceRecHandle = await mod.startVoiceRecording({
      onTick: (s) => { _voiceElapsed = s; if (timeEl) timeEl.textContent = _fmtRecTime(s); },
    });
  } catch (_) {
    showToast('Microphone access is needed to record a voice message', 'warning');
    return;
  }
  const composer = _panel.querySelector('.connect-thread-composer');
  if (composer) composer.classList.add('recording');
}

async function _finishVoiceRecording(send) {
  if (!_voiceRecHandle) return;
  const handle = _voiceRecHandle;
  _voiceRecHandle = null;
  const composer = _panel.querySelector('.connect-thread-composer');
  if (composer) composer.classList.remove('recording');
  let result = null;
  try { result = await handle.stop(!send); } catch (_) { /* released */ }
  if (!send || !result) return;
  if (!result.blob || result.blob.size < 800 || _voiceElapsed < 0.6) {
    showToast('Tap and speak — that was too short', 'info');
    return;
  }
  const ext = result.mime.includes('ogg') ? 'ogg'
    : (result.mime.includes('mp4') || result.mime.includes('aac')) ? 'm4a' : 'webm';
  let file;
  try {
    file = new File([result.blob], `Voice message.${ext}`, { type: result.blob.type });
  } catch (_) {
    // Older engines without the File constructor: fall back to the Blob
    // (uploadAttachment reads name/type defensively).
    file = result.blob;
    file.name = `Voice message.${ext}`;
  }
  // Send straight away through the shared upload+send path.
  _uploadAndSendFile(file);
}

function _cancelVoiceRecordingIfAny() {
  if (!_voiceRecHandle) return;
  const handle = _voiceRecHandle;
  _voiceRecHandle = null;
  const composer = _panel?.querySelector('.connect-thread-composer');
  if (composer) composer.classList.remove('recording');
  try { handle.stop(true); } catch (_) {}
}

function _fmtRecTime(s) {
  const t = Math.floor(s || 0);
  const m = Math.floor(t / 60);
  const r = t % 60;
  return `${m}:${String(r).padStart(2, '0')}`;
}

// ── Attach menu ("+") ──────────────────────────────────────────
// Photo/File goes through the normal picker; Location and Contact lean
// on the DEVICE's own pickers (geolocation + Contact Picker API) rather
// than any server-side service — so they only appear where supported.
let _attachMenu = null;
function _openAttachMenu(anchorBtn) {
  if (_attachMenu) { _closeAttachMenu(); return; }
  const hasGeo = !!navigator.geolocation;
  const hasContacts = !!(navigator.contacts && navigator.contacts.select);
  const menu = document.createElement('div');
  menu.className = 'connect-attach-menu';
  menu.setAttribute('role', 'menu');
  menu.innerHTML = `
    <button class="connect-attach-menu-row" role="menuitem" data-attach="file">
      <span class="connect-attach-menu-icon">${icon('plus', { size: 18 })}</span>
      <span class="connect-attach-menu-text">Photo, video or file</span>
    </button>
    ${hasGeo ? `<button class="connect-attach-menu-row" role="menuitem" data-attach="location">
      <span class="connect-attach-menu-icon">${icon('pin', { size: 18 })}</span>
      <span class="connect-attach-menu-text">Location</span>
    </button>` : ''}
    ${hasContacts ? `<button class="connect-attach-menu-row" role="menuitem" data-attach="contact">
      <span class="connect-attach-menu-icon">${icon('user', { size: 18 })}</span>
      <span class="connect-attach-menu-text">Contact</span>
    </button>` : ''}
  `;
  document.body.appendChild(menu);
  _attachMenu = menu;
  const r = anchorBtn.getBoundingClientRect();
  menu.style.left = `${Math.max(8, r.left)}px`;
  menu.style.bottom = `${Math.max(8, window.innerHeight - r.top + 8)}px`;
  requestAnimationFrame(() => menu.classList.add('open'));

  menu.addEventListener('click', (ev) => {
    const row = ev.target.closest('.connect-attach-menu-row');
    if (!row) return;
    const kind = row.dataset.attach;
    _closeAttachMenu();
    if (kind === 'file') {
      const fp = _panel?.querySelector('.connect-thread-filepicker');
      if (fp) { fp.value = ''; fp.click(); }
    } else if (kind === 'location') { _shareLocation(); }
    else if (kind === 'contact') { _shareContact(); }
  });

  const onOutside = (e) => {
    if (!_attachMenu) return;
    if (_attachMenu.contains(e.target) || anchorBtn.contains(e.target)) return;
    _closeAttachMenu();
  };
  const onKey = (e) => { if (e.key === 'Escape') _closeAttachMenu(); };
  setTimeout(() => {
    document.addEventListener('pointerdown', onOutside, true);
    document.addEventListener('keydown', onKey);
  }, 0);
  menu._cleanup = () => {
    document.removeEventListener('pointerdown', onOutside, true);
    document.removeEventListener('keydown', onKey);
  };
}

function _closeAttachMenu() {
  if (!_attachMenu) return;
  try { _attachMenu._cleanup?.(); } catch (_) {}
  const m = _attachMenu;
  m.classList.remove('open');
  _attachMenu = null;
  setTimeout(() => { try { m.remove(); } catch (_) {} }, 140);
}

// Send a quick text message reusing the composer send path.
function _sendQuickText(text) {
  const input = _panel?.querySelector('.connect-thread-composer-input');
  if (!input) return;
  input.value = text;
  input.dispatchEvent(new Event('input'));
  _onSendClick();
}

// Device GPS → a tappable maps link (opens the recipient's maps app).
function _shareLocation() {
  if (!navigator.geolocation) { showToast("Location isn't available here", 'warning'); return; }
  if (!_threadCache.get(_activeThreadId)) { showToast('Open a conversation first', 'warning'); return; }
  showToast('Getting your location…', 'info');
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      const lat = pos.coords.latitude.toFixed(6);
      const lon = pos.coords.longitude.toFixed(6);
      _sendQuickText(`📍 My location: https://maps.google.com/?q=${lat},${lon}`);
    },
    (err) => {
      showToast(err && err.code === 1 ? 'Location permission denied' : "Couldn't get your location", 'warning');
    },
    { enableHighAccuracy: true, timeout: 12000, maximumAge: 60000 },
  );
}

// Device contact sheet (Contact Picker API) → a contact-card message.
async function _shareContact() {
  if (!(navigator.contacts && navigator.contacts.select)) {
    showToast("Your device doesn't offer a contact picker here", 'warning');
    return;
  }
  if (!_threadCache.get(_activeThreadId)) { showToast('Open a conversation first', 'warning'); return; }
  try {
    let props = ['name', 'tel', 'email'];
    if (navigator.contacts.getProperties) {
      const sup = await navigator.contacts.getProperties();
      props = props.filter((p) => sup.includes(p));
      if (!props.length) { showToast("Contacts aren't shareable on this device", 'warning'); return; }
    }
    const sel = await navigator.contacts.select(props, { multiple: false });
    if (!sel || !sel.length) return;  // user cancelled
    const c = sel[0];
    const name = (c.name && c.name[0]) || 'Contact';
    const tel = (c.tel && c.tel[0]) || '';
    const email = (c.email && c.email[0]) || '';
    let txt = `👤 ${name}`;
    if (tel) txt += `\n${tel}`;
    if (email) txt += `\n${email}`;
    _sendQuickText(txt);
  } catch (_) { /* user dismissed the picker */ }
}

// Composer emoji picker: lazy-load the popover and insert the chosen
// emoji at the textarea caret.
function _openComposerEmojiPicker(anchor) {
  import('./emoji-picker.js')
    .then((m) => m.openEmojiPicker(anchor, (emoji) => _insertEmojiAtCursor(emoji)))
    .catch((err) => console.warn('connect: emoji picker failed', err));
}

function _insertEmojiAtCursor(emoji) {
  if (!_panel || !emoji) return;
  const input = _panel.querySelector('.connect-thread-composer-input');
  if (!input) return;
  const start = typeof input.selectionStart === 'number' ? input.selectionStart : input.value.length;
  const end = typeof input.selectionEnd === 'number' ? input.selectionEnd : input.value.length;
  input.value = input.value.slice(0, start) + emoji + input.value.slice(end);
  const pos = start + emoji.length;
  try { input.setSelectionRange(pos, pos); } catch (_) {}
  input.focus();
  // Drives autosize + send-button morph + typing signal (all wired to 'input').
  input.dispatchEvent(new Event('input'));
}

// Fullscreen image viewer. Tap the image to toggle zoom, tap the
// backdrop / Close / Escape to dismiss, and an optional download link.
function _openLightbox(src, { name = '', downloadUrl = '' } = {}) {
  if (!src) return;
  _closeLightbox();
  const overlay = document.createElement('div');
  overlay.className = 'connect-lightbox';
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-modal', 'true');
  overlay.setAttribute('aria-label', name ? `Image: ${name}` : 'Image viewer');
  const dl = downloadUrl
    ? `<a class="connect-lightbox-btn" href="${escapeHtml(downloadUrl)}" download
          title="Download" aria-label="Download">
         <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
              stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
           <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
         </svg>
       </a>`
    : '';
  overlay.innerHTML = `
    <div class="connect-lightbox-bar">
      <span class="connect-lightbox-name">${escapeHtml(name)}</span>
      <div class="connect-lightbox-actions">
        ${dl}
        <button class="connect-lightbox-btn connect-lightbox-close" type="button"
                title="Close" aria-label="Close">&#x2715;</button>
      </div>
    </div>
    <div class="connect-lightbox-stage">
      <img class="connect-lightbox-img" src="${escapeHtml(src)}" alt="${escapeHtml(name)}" />
    </div>
  `;
  document.body.appendChild(overlay);
  _lightbox = overlay;

  const img = overlay.querySelector('.connect-lightbox-img');
  const stage = overlay.querySelector('.connect-lightbox-stage');
  img.addEventListener('click', (e) => { e.stopPropagation(); img.classList.toggle('zoomed'); });
  stage.addEventListener('click', () => _closeLightbox());
  overlay.querySelector('.connect-lightbox-close').addEventListener('click', _closeLightbox);
  _lightboxKeyHandler = (e) => { if (e.key === 'Escape') { e.stopPropagation(); _closeLightbox(); } };
  document.addEventListener('keydown', _lightboxKeyHandler, true);
  requestAnimationFrame(() => overlay.classList.add('open'));
}

function _closeLightbox() {
  if (_lightboxKeyHandler) {
    document.removeEventListener('keydown', _lightboxKeyHandler, true);
    _lightboxKeyHandler = null;
  }
  if (_lightbox) { _lightbox.remove(); _lightbox = null; }
}

function _renderAttachment(m) {
  // m.attachment_ref is the upload_id. The route gates access by
  // participation in the message, so the URL works for both sender
  // and recipient without auth gymnastics.
  // Fabric-delivered messages carry attachment_fetch_url +
  // attachment_fetch_token pointing at the sender's instance; pass
  // them through so attachmentUrl() builds a cross-instance fetch
  // when present, local-route fetch otherwise.
  const fetchUrl = m.attachment_fetch_url || '';
  const fetchToken = m.attachment_fetch_token || '';
  const url = attachmentUrl(
    m.thread_id, m.message_id,
    { download: false, fetchUrl, fetchToken },
  );
  const dlUrl = attachmentUrl(
    m.thread_id, m.message_id,
    { download: true, fetchUrl, fetchToken },
  );
  const mime = (m.attachment_mime || '').toLowerCase();
  const name = m.attachment_name || 'attachment';
  const size = m.attachment_size || 0;

  if (mime.startsWith('image/')) {
    // Three-state image: skeleton shimmer → loaded image OR error
    // fallback. The wrapper holds the placeholder; the <img> swaps
    // it via onload/onerror handlers attached after render. Animated
    // GIFs play in the <img> and get a corner badge.
    const isGif = mime === 'image/gif';
    return `
      <a class="connect-attachment-image is-loading${isGif ? ' is-gif' : ''}"
         href="${escapeHtml(url)}"
         data-img-name="${escapeHtml(name)}"
         data-dl-url="${escapeHtml(dlUrl)}"
         target="_blank" rel="noopener noreferrer"
         aria-label="Open image ${escapeHtml(name)}">
        <div class="connect-attachment-image-skeleton" aria-hidden="true"></div>
        <img loading="lazy" alt="${escapeHtml(name)}"
             src="${escapeHtml(url)}"
             onload="this.parentElement.classList.remove('is-loading');this.parentElement.classList.add('is-loaded')"
             onerror="this.parentElement.classList.remove('is-loading');this.parentElement.classList.add('is-broken')" />
        ${isGif ? '<span class="connect-attachment-gif-badge" aria-hidden="true">GIF</span>' : ''}
        <div class="connect-attachment-image-broken" aria-hidden="true">
          ${icon('alert-triangle', { size: 18 })}
          <span>Image unavailable</span>
        </div>
      </a>
    `;
  }
  if (mime.startsWith('audio/')) {
    return `
      <audio class="connect-attachment-audio" controls preload="metadata"
             src="${escapeHtml(url)}"></audio>
    `;
  }
  if (mime.startsWith('video/')) {
    return `
      <video class="connect-attachment-video" controls playsinline preload="metadata"
             src="${escapeHtml(url)}"></video>
    `;
  }
  // Generic file pill — filename + size + family-aware glyph.
  const glyph = _attachmentGlyphFor(mime, name);
  const sizeText = size > 0 ? _formatBytes(size) : '';
  const familyClass = _attachmentFamilyClass(mime, name);
  return `
    <a class="connect-attachment-file ${familyClass}" href="${escapeHtml(dlUrl)}"
       download="${escapeHtml(name)}"
       aria-label="Download ${escapeHtml(name)}">
      <span class="connect-attachment-file-icon">${glyph}</span>
      <span class="connect-attachment-file-meta">
        <span class="connect-attachment-file-name">${escapeHtml(name)}</span>
        ${sizeText ? `<span class="connect-attachment-file-size">${escapeHtml(sizeText)}</span>` : ''}
      </span>
    </a>
  `;
}

function _attachmentFamilyClass(mime, name) {
  const lower = (name || '').toLowerCase();
  if (mime.includes('pdf') || lower.endsWith('.pdf')) return 'family-doc';
  if (mime.startsWith('text/') || /\.(md|txt|json|csv|log)$/.test(lower)) return 'family-doc';
  if (mime.startsWith('application/zip') || /\.(zip|tar|gz|7z|rar)$/.test(lower)) return 'family-archive';
  if (mime.startsWith('application/')) return 'family-binary';
  return '';
}

function _attachmentGlyphFor(mime, name) {
  // The icon set ships a generic file glyph; until we have richer
  // family icons we use the same shape but tint via CSS family
  // classes. Returns a paperclip fallback if the icon set drops the
  // 'file' helper for some reason.
  try {
    return icon('file', { size: 18 });
  } catch (_) {
    return '\u{1F4CE}';
  }
}

function _formatBytes(n) {
  if (!n || n < 1024) return `${n} B`;
  const kb = n / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  if (mb < 1024) return `${mb.toFixed(1)} MB`;
  return `${(mb / 1024).toFixed(2)} GB`;
}

function _renderReactionStack(m, myDid) {
  const reactions = Array.isArray(m.reactions) ? m.reactions : [];
  if (!reactions.length) return '';
  const pills = reactions.map((r) => {
    const own = Array.isArray(r.reactor_dids) && r.reactor_dids.includes(myDid);
    return `
      <button class="connect-reaction-pill${own ? ' own' : ''}" type="button"
              data-emoji="${escapeHtml(r.emoji)}" data-message-id="${escapeHtml(m.message_id)}"
              title="${escapeHtml((r.reactor_dids || []).join(', '))}">
        <span class="connect-reaction-emoji">${escapeHtml(r.emoji)}</span>
        <span class="connect-reaction-count">${r.count || (r.reactor_dids?.length ?? 1)}</span>
      </button>
    `;
  }).join('');
  return `<div class="connect-reaction-stack">${pills}</div>`;
}

async function _toggleReaction(messageId, emoji) {
  const thread = _threadCache.get(_activeThreadId);
  if (!thread || !thread.peer_did) return;
  const msgs = _messageCache.get(_activeThreadId) || [];
  const m = msgs.find((x) => x.message_id === messageId);
  if (!m) return;
  const myDid = _ownDid();
  const existing = (m.reactions || []).find((r) => r.emoji === emoji);
  const own = existing && Array.isArray(existing.reactor_dids) && existing.reactor_dids.includes(myDid);
  const action = own ? 'remove' : 'add';

  // Optimistic local update so the pill responds instantly.
  _applyReactionLocally(messageId, emoji, myDid, action);
  _scheduleMessageRender();

  try {
    await reactToMessage({
      peerDid: thread.peer_did,
      threadId: _activeThreadId,
      messageId,
      emoji,
      action,
    });
  } catch (err) {
    // Roll back the optimistic update on failure.
    _applyReactionLocally(messageId, emoji, myDid, own ? 'add' : 'remove');
    _scheduleMessageRender();
    showToast(`Reaction failed: ${err?.message || 'unknown'}`, 'error');
  }
}

function _applyReactionLocally(messageId, emoji, reactorDid, action) {
  const msgs = _messageCache.get(_activeThreadId) || [];
  const m = msgs.find((x) => x.message_id === messageId);
  if (!m) return;
  m.reactions = Array.isArray(m.reactions) ? m.reactions : [];
  const idx = m.reactions.findIndex((r) => r.emoji === emoji);
  if (action === 'add') {
    if (idx === -1) {
      m.reactions.push({ emoji, reactor_dids: [reactorDid], count: 1 });
    } else {
      const r = m.reactions[idx];
      r.reactor_dids = Array.isArray(r.reactor_dids) ? r.reactor_dids : [];
      if (!r.reactor_dids.includes(reactorDid)) r.reactor_dids.push(reactorDid);
      r.count = r.reactor_dids.length;
    }
  } else {
    if (idx !== -1) {
      const r = m.reactions[idx];
      r.reactor_dids = (r.reactor_dids || []).filter((d) => d !== reactorDid);
      r.count = r.reactor_dids.length;
      if (r.count === 0) m.reactions.splice(idx, 1);
    }
  }
}

function _onReactionEvent(evt) {
  const data = evt.detail || {};
  const tid = data.thread_id;
  if (!tid || tid !== _activeThreadId) return;
  const messageId = data.message_id;
  const emoji = data.emoji;
  const reactor = data.reactor_did;
  const action = String(data.action || 'add').toLowerCase();
  if (!messageId || !emoji || !reactor) return;
  _applyReactionLocally(messageId, emoji, reactor, action);
  _renderMessages(_messageCache.get(tid) || []);
}

// ── Long-press / right-click message context menu ───────────────
//
// Surface for both touch (long-press) and desktop (right-click).
// Renders a two-tier popover:
//   • Top strip: 6-emoji quick-reaction tray
//   • Below: vertical action list — Reply / Copy / Edit (own) /
//     Delete (own). Edit and Delete wire MSG_TEXT_EDIT / MSG_TEXT_DELETE
//     that have always been protocol-supported but had no UI hook.

let _messageMenu = null;

function _openMessageMenu(bubbleEl, messageId) {
  _closeMessageMenu();
  const msgs = _messageCache.get(_activeThreadId) || [];
  const m = msgs.find((x) => x.message_id === messageId);
  if (!m) return;
  const myDid = _ownDid();
  const own = m.sender_did === myDid;
  const deleted = !!m.deleted_at;

  const rect = bubbleEl.getBoundingClientRect();
  const menu = document.createElement('div');
  menu.className = 'connect-message-menu';
  menu.setAttribute('role', 'menu');
  menu.innerHTML = `
    <div class="connect-message-menu-tray">
      ${QUICK_REACTIONS.map((e) => `
        <button class="connect-message-menu-emoji" type="button" data-emoji="${escapeHtml(e)}">
          ${escapeHtml(e)}
        </button>
      `).join('')}
      <button class="connect-message-menu-emoji connect-message-menu-more" type="button"
              data-action="more-emoji" title="More emojis" aria-label="More emojis">${icon('plus', { size: 16 })}</button>
    </div>
    <div class="connect-message-menu-items">
      ${!deleted ? `
        <button class="connect-message-menu-item" role="menuitem" data-action="reply">
          ${icon('corner-up-left', { size: 14 })}<span>Reply</span>
        </button>
        <button class="connect-message-menu-item" role="menuitem" data-action="copy">
          ${icon('plus', { size: 14 })}<span>Copy text</span>
        </button>
      ` : ''}
      ${own && !deleted ? `
        <button class="connect-message-menu-item" role="menuitem" data-action="edit">
          ${icon('settings', { size: 14 })}<span>Edit</span>
        </button>
        <button class="connect-message-menu-item destructive" role="menuitem" data-action="delete">
          ${icon('x', { size: 14 })}<span>Delete for everyone</span>
        </button>
      ` : ''}
    </div>
  `;
  document.body.appendChild(menu);
  _messageMenu = menu;

  // Position above the bubble preferentially; if no headroom, flip below.
  const menuHeightEstimate = own && !deleted ? 220 : (deleted ? 60 : 160);
  let top = rect.top - menuHeightEstimate - 8;
  if (top < 8) top = Math.min(rect.bottom + 8, window.innerHeight - menuHeightEstimate - 8);
  const left = Math.min(
    Math.max(8, rect.left + rect.width / 2 - 130),
    window.innerWidth - 268,
  );
  menu.style.top = `${top}px`;
  menu.style.left = `${left}px`;
  requestAnimationFrame(() => menu.classList.add('open'));

  for (const btn of menu.querySelectorAll('.connect-message-menu-emoji')) {
    btn.addEventListener('click', () => {
      if (btn.dataset.action === 'more-emoji') {
        _closeMessageMenu();
        import('./emoji-picker.js')
          .then((mod) => mod.openEmojiPicker(bubbleEl, (emoji) => _toggleReaction(messageId, emoji)))
          .catch(() => {});
        return;
      }
      const emoji = btn.dataset.emoji;
      _closeMessageMenu();
      _toggleReaction(messageId, emoji);
    });
  }
  for (const item of menu.querySelectorAll('.connect-message-menu-item')) {
    item.addEventListener('click', () => {
      const action = item.dataset.action;
      _closeMessageMenu();
      _handleMessageMenuAction(action, m);
    });
  }

  const onOutside = (ev) => {
    if (!_messageMenu) return;
    if (_messageMenu.contains(ev.target)) return;
    _closeMessageMenu();
  };
  const onKey = (ev) => {
    if (ev.key === 'Escape') _closeMessageMenu();
  };
  setTimeout(() => {
    document.addEventListener('pointerdown', onOutside, true);
    document.addEventListener('keydown', onKey);
  }, 0);
  menu._cleanup = () => {
    document.removeEventListener('pointerdown', onOutside, true);
    document.removeEventListener('keydown', onKey);
  };
}

function _closeMessageMenu() {
  if (!_messageMenu) return;
  try { _messageMenu._cleanup?.(); } catch (_) {}
  const m = _messageMenu;
  m.classList.remove('open');
  _messageMenu = null;
  setTimeout(() => { try { m.remove(); } catch (_) {} }, 140);
}

async function _handleMessageMenuAction(action, m) {
  if (action === 'reply') {
    _setReply(m);
    return;
  }
  if (action === 'copy') {
    try {
      await navigator.clipboard.writeText(m.body || '');
      showToast('Copied to clipboard', 'info');
    } catch (err) {
      showToast('Copy failed', 'error');
    }
    return;
  }
  if (action === 'edit') {
    _openEditDialog(m);
    return;
  }
  if (action === 'delete') {
    _confirmDelete(m);
    return;
  }
}

// Map a raw send/attach/edit error to a calm, actionable line. The common
// recoverable cases (rate limit, oversized attachment, offline) get specific
// guidance; everything else falls back to the caller's generic message — we
// never surface a raw "TypeError"/HTTP status to the user.
function _friendlyMessageError(err, fallback) {
  const msg = String(err?.message || err || '').toLowerCase();
  if (msg.includes('429') || msg.includes('rate')) {
    return "You're sending too fast — wait a moment and try again.";
  }
  if (msg.includes('413') || msg.includes('too large') || msg.includes('size')) {
    return 'That attachment is too large to send.';
  }
  if (typeof navigator !== 'undefined' && navigator.onLine === false) {
    return "You're offline — it'll send automatically when you reconnect.";
  }
  return fallback;
}

function _openEditDialog(m) {
  // In-theme edit modal — a native window.prompt() here looked jarring
  // (system chrome, no theme, tiny single line) against the polished thread.
  // Self-contained: overlay + card with a pre-filled textarea, Save/Cancel,
  // Enter-to-save / Shift+Enter newline / Esc-to-cancel, focus managed.
  const existing = document.querySelector('.connect-edit-overlay');
  if (existing) existing.remove();

  const overlay = document.createElement('div');
  overlay.className = 'connect-edit-overlay';
  overlay.innerHTML = `
    <div class="connect-edit-card" role="dialog" aria-modal="true" aria-label="Edit message">
      <div class="connect-edit-title">Edit message</div>
      <textarea class="connect-edit-input" rows="3" aria-label="Message text"></textarea>
      <div class="connect-edit-hint">Enter to save · Shift+Enter for a new line · Esc to cancel</div>
      <div class="connect-edit-actions">
        <button type="button" class="connect-edit-cancel">Cancel</button>
        <button type="button" class="connect-edit-save">Save</button>
      </div>
    </div>`;
  document.body.appendChild(overlay);

  const input = overlay.querySelector('.connect-edit-input');
  input.value = m.body || '';

  const close = () => {
    document.removeEventListener('keydown', onKey, true);
    overlay.remove();
  };
  const save = () => {
    const trimmed = input.value.trim();
    if (trimmed && trimmed !== m.body) _sendEditFor(m.message_id, trimmed);
    close();
  };
  const onKey = (ev) => {
    if (ev.key === 'Escape') { ev.preventDefault(); close(); }
    else if (ev.key === 'Enter' && !ev.shiftKey) { ev.preventDefault(); save(); }
  };

  overlay.addEventListener('mousedown', (ev) => { if (ev.target === overlay) close(); });
  overlay.querySelector('.connect-edit-cancel').addEventListener('click', close);
  overlay.querySelector('.connect-edit-save').addEventListener('click', save);
  document.addEventListener('keydown', onKey, true);

  // Focus + place the caret at the end so editing starts naturally.
  requestAnimationFrame(() => {
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  });
}

async function _sendEditFor(messageId, body) {
  const thread = _threadCache.get(_activeThreadId);
  if (!thread || !thread.peer_did) return;
  // We use the WS path when available; HTTP fallback is wired in
  // messages.js. For now we issue the WS verb directly via the client.
  const { send } = await import('./client.js');
  try {
    send({
      verb: 'text_edit',
      peer: thread.peer_did,
      data: {
        thread_id: _activeThreadId,
        message_id: messageId,
        body,
      },
    });
    broadcastThreadChanged(_activeThreadId, 'edit');
    // Optimistic local update — the message row's body becomes `body`
    // and the edited_at marker shows in the bubble.
    const msgs = _messageCache.get(_activeThreadId) || [];
    const m = msgs.find((x) => x.message_id === messageId);
    if (m) {
      m.body = body;
      m.edited_at = new Date().toISOString();
      _scheduleMessageRender();
    }
  } catch (err) {
    showToast(_friendlyMessageError(err, "Couldn't save your edit. Try again."), 'error');
  }
}

async function _confirmDelete(m) {
  const ok = window.confirm('Delete this message for everyone? This can’t be undone.');
  if (!ok) return;
  const thread = _threadCache.get(_activeThreadId);
  if (!thread || !thread.peer_did) return;
  const { send } = await import('./client.js');
  try {
    send({
      verb: 'text_delete',
      peer: thread.peer_did,
      data: {
        thread_id: _activeThreadId,
        message_id: m.message_id,
      },
    });
    broadcastThreadChanged(_activeThreadId, 'delete');
    // Optimistic tombstone — bubble re-renders with the "message deleted"
    // italic placeholder courtesy of the existing _renderMessages path.
    const msgs = _messageCache.get(_activeThreadId) || [];
    const target = msgs.find((x) => x.message_id === m.message_id);
    if (target) {
      target.deleted_at = new Date().toISOString();
      target.body = '';
      _scheduleMessageRender();
    }
  } catch (err) {
    showToast(`Delete failed: ${err?.message || 'unknown'}`, 'error');
  }
}

// Autoresize composer textarea: 1 row resting, grow up to 6 rows as
// the user types or pastes multi-line content. Resets when value
// clears (after a send). We measure scrollHeight, cap at the 6-line
// max derived from the line-height token, and let the textarea
// scroll internally past that point.
function _autosizeComposer() {
  if (!_panel) return;
  const input = _panel.querySelector('.connect-thread-composer-input');
  if (!input) return;
  // Reset so scrollHeight reflects current content, not the previous
  // (taller) measurement.
  input.style.height = 'auto';
  const max = 24 * 6 + 16;  // ~6 lines + vertical padding
  const next = Math.min(input.scrollHeight, max);
  input.style.height = `${next}px`;
}

function _scrollMessagesToBottom() {
  if (!_panel) return;
  const msgsEl = _panel.querySelector('.connect-thread-messages');
  if (!msgsEl) return;
  // Smooth in modern browsers; instant fallback for the first paint
  // so the user doesn't see a scroll animation just from opening.
  msgsEl.scrollTo({ top: msgsEl.scrollHeight, behavior: 'smooth' });
}

function _ownDid() {
  // Cached welcome envelope — populated once the signaling WS has
  // authenticated. Empty during a brief boot window; outgoing
  // messages just render on the left (peer side) until the welcome
  // lands, which is benign.
  const w = getWelcome();
  return (w && w.user_did) || '';
}

function _isEnabled() {
  const s = getSettings?.();
  return !!(s && s.connectEnabled);
}
