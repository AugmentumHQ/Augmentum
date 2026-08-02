/* ==========================================================================
   Chat Module — STT (Speech-to-Text) Recording
   Browser SpeechRecognition with record-upload fallback
   ========================================================================== */

import { showToast } from '../app.js';
import { icons } from './constants.js';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let mediaRecorder = null;
let recordingChunks = [];
let _pendingVoiceInput = false;
let _chatSttRecognition = null;
const _ChatSpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Create and insert mic button into the given input area, before the send button.
 * @param {HTMLElement} inputArea - The .input-area element to insert the mic button into.
 */
export function initMicButton(inputArea) {
  if (!inputArea) return;

  const sendBtn = inputArea.querySelector('#send-btn');
  if (!sendBtn) return;

  // Check if mic button already exists
  if (inputArea.querySelector('#mic-btn')) return;

  const micBtn = document.createElement('button');
  micBtn.id = 'mic-btn';
  micBtn.innerHTML = icons.mic;

  sendBtn.parentNode.insertBefore(micBtn, sendBtn);

  // Android on-device dictation (Moonshine) takes over this same button when the
  // native STT bridge is present — it's offline + cert-free, strictly better than
  // the web path inside the WebView. Moonshine has no VAD, so the press IS the
  // segment: press-and-hold, not click-toggle. The native side delivers the
  // transcript via window.__augReceiveTranscript (see app.js). No second button.
  const bridge = window.AugmentumAndroid;
  let sttBridgeOk = false;
  if (bridge && typeof bridge.startDictation === 'function') {
    try { sttBridgeOk = typeof bridge.sttAvailable !== 'function' || !!bridge.sttAvailable(); }
    catch (_) { sttBridgeOk = true; /* assume available */ }
  }

  if (sttBridgeOk) {
    micBtn.title = 'Hold to dictate';
    micBtn.setAttribute('aria-label', 'Hold to dictate');
    let active = false;
    const begin = (e) => {
      if (e && e.preventDefault) e.preventDefault();
      if (active) return;
      active = true;
      micBtn.classList.add('recording');
      try { bridge.startDictation(); } catch (_) { /* ignore */ }
    };
    const end = () => {
      if (!active) return;
      active = false;
      micBtn.classList.remove('recording');
      try { bridge.stopDictation(); } catch (_) { /* ignore */ }
    };
    micBtn.addEventListener('pointerdown', begin);
    micBtn.addEventListener('pointerup', end);
    micBtn.addEventListener('pointerleave', end);
    micBtn.addEventListener('pointercancel', () => {
      active = false; micBtn.classList.remove('recording');
      try { bridge.cancelDictation(); } catch (_) { /* ignore */ }
    });
    return;
  }

  // Web path — browser SpeechRecognition with record-upload fallback.
  micBtn.title = 'Voice input (STT)';
  micBtn.addEventListener('click', () => {
    const chatInput = inputArea.querySelector('#chat-input') || document.getElementById('chat-input');
    toggleRecording(chatInput);
  });
}

/**
 * Toggle browser SpeechRecognition or record-upload fallback.
 * @param {HTMLElement} [inputEl] - The input/textarea element for inserting transcribed text.
 */
export async function toggleRecording(inputEl) {
  const micBtn = document.querySelector('#mic-btn');
  if (!micBtn) return;

  const input = inputEl || document.getElementById('chat-input');

  // --- Browser SpeechRecognition path (live dictation) ---
  if (_chatSttRecognition) {
    _chatSttRecognition.stop();
    _chatSttRecognition = null;
    micBtn.classList.remove('recording');
    micBtn.innerHTML = icons.mic;
    return;
  }

  if (_ChatSpeechRecognition) {
    _chatSttRecognition = new _ChatSpeechRecognition();
    _chatSttRecognition.continuous = true;
    _chatSttRecognition.interimResults = false;
    _chatSttRecognition.lang = navigator.language || 'en-US';

    _chatSttRecognition.onresult = (event) => {
      if (!input) return;
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          const text = event.results[i][0].transcript.trim();
          if (text) {
            const existing = input.value.trim();
            input.value = existing ? `${existing} ${text}` : text;
            _pendingVoiceInput = true;
            input.dispatchEvent(new Event('input'));
          }
        }
      }
    };

    _chatSttRecognition.onerror = (event) => {
      if (event.error === 'no-speech' || event.error === 'aborted') return;
      if (event.error === 'not-allowed') showToast('Microphone permission denied', 'error');
      if (event.error === 'service-not-available' || event.error === 'language-not-supported') {
        _chatSttRecognition = null;
        micBtn.classList.remove('recording');
        micBtn.innerHTML = icons.mic;
        _startRecordUploadFallback(micBtn, input);
      }
    };

    _chatSttRecognition.onend = () => {
      if (_chatSttRecognition) {
        _chatSttRecognition = null;
        micBtn.classList.remove('recording');
        micBtn.innerHTML = icons.mic;
        if (input) input.focus();
      }
    };

    try {
      _chatSttRecognition.start();
      micBtn.classList.add('recording');
      if (navigator.vibrate) navigator.vibrate(10);
      return;
    } catch {
      _chatSttRecognition = null;
    }
  }

  // --- Record->Upload fallback (server-side batch STT) ---
  _startRecordUploadFallback(micBtn, input);
}

/** Returns true when the input box contains STT text that hasn't been sent yet. */
export function isPendingVoiceInput() {
  return _pendingVoiceInput;
}

/** Clear the pending voice input flag (call after sending). */
export function clearPendingVoiceInput() {
  _pendingVoiceInput = false;
}

// ---------------------------------------------------------------------------
// Internal — record-upload fallback
// ---------------------------------------------------------------------------

async function _startRecordUploadFallback(micBtn, inputEl) {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    micBtn.classList.remove('recording');
    micBtn.innerHTML = icons.mic;
    return;
  }

  try {
    const { acquireMic } = await import('../voice/mic-device.js');
    const stream = await acquireMic({ usage: 'snapshot' });
    recordingChunks = [];

    mediaRecorder = new MediaRecorder(stream, {
      mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus' : 'audio/webm',
    });

    mediaRecorder.addEventListener('dataavailable', (e) => {
      if (e.data.size > 0) recordingChunks.push(e.data);
    });

    mediaRecorder.addEventListener('stop', async () => {
      stream.getTracks().forEach(t => t.stop());

      if (recordingChunks.length === 0) {
        micBtn.classList.remove('recording');
        micBtn.innerHTML = icons.mic;
        return;
      }

      const blob = new Blob(recordingChunks, { type: 'audio/webm' });
      recordingChunks = [];

      micBtn.classList.remove('recording');
      micBtn.classList.add('transcribing');
      micBtn.innerHTML = icons.mic;

      try {
        const formData = new FormData();
        formData.append('file', blob, 'recording.webm');

        const resp = await fetch('/v1/audio/transcriptions', {
          method: 'POST',
          body: formData,
        });

        if (!resp.ok) {
          const err = await resp.text();
          showToast(`STT failed: ${err.slice(0, 100)}`, 'error');
          micBtn.classList.remove('transcribing');
          micBtn.innerHTML = icons.mic;
          return;
        }

        const data = await resp.json();
        const transcript = data.text || '';

        if (transcript) {
          const input = inputEl || document.getElementById('chat-input');
          if (input) {
            const existing = input.value.trim();
            input.value = existing ? `${existing} ${transcript}` : transcript;
            _pendingVoiceInput = true;
            input.focus();
            input.dispatchEvent(new Event('input'));
          }
        } else {
          showToast('No speech detected', 'info');
        }
      } catch {
        showToast('STT request failed', 'error');
      }

      micBtn.classList.remove('transcribing');
      micBtn.innerHTML = icons.mic;
    });

    mediaRecorder.start();
    micBtn.classList.add('recording');
  } catch (err) {
    const reason = err.name === 'NotFoundError' ? 'No microphone found'
      : err.name === 'NotAllowedError' ? 'Microphone permission denied'
      : err.name === 'NotReadableError' ? 'Microphone in use by another app'
      : `Mic error: ${err.name || err.message}`;
    showToast(reason, 'error');
  }
}
