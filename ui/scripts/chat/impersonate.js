/* ==========================================================================
   Chat Module — Impersonate
   Shared impersonation logic used by primary chat/index.js, ChatSurface,
   and NarrativeSurface. Builds a voice-mining prompt from the user's own
   prior turns and streams the completion into a target input widget.
   ========================================================================== */

import { app, showToast } from '../app.js';
import * as tree from './tree.js';

const MAX_STYLE_SAMPLES = 5;
const MAX_SAMPLE_CHARS = 900;
const MAX_SCENE_CHARS = 600;
const DEFAULT_AVG_WORDS = 60;

function textOf(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .filter(p => p && p.type === 'text')
      .map(p => p.text || '')
      .join(' ');
  }
  return '';
}

function detectStyleHints(userTexts) {
  const blob = userTexts.join('\n');
  if (!blob) return [];
  const hints = [];
  if (/\*[^*\n]{2,}\*/.test(blob)) hints.push('uses *action asterisks*');
  if (/\b(I|I'm|I've|me|my|mine)\b/.test(blob)) hints.push('first-person POV');
  const hasEd = /\b\w{3,}ed\b/i.test(blob);
  const hasPresent = /\b(is|are|am|walk|walks|say|says|look|looks)\b/i.test(blob);
  if (hasPresent && !hasEd) hints.push('present tense');
  else if (hasEd && !hasPresent) hints.push('past tense');
  if (/—|--/.test(blob)) hints.push('em-dashes');
  return hints;
}

function avgWordCount(userTexts) {
  if (userTexts.length === 0) return DEFAULT_AVG_WORDS;
  const total = userTexts.reduce(
    (s, t) => s + t.split(/\s+/).filter(Boolean).length,
    0,
  );
  return Math.max(20, Math.round(total / userTexts.length));
}

/**
 * Build the OOC directive message appended to the chat history.
 * Mines the user's own prior turns for voice/style so the model continues
 * the pattern already on screen instead of inventing one.
 */
export function buildImpersonateInstruction(session, opts = {}) {
  const userName = (opts.userName || '').trim() || 'the user';
  const charName = (opts.charName || '').trim() || 'the character';
  const personaDesc = (opts.personaDesc || '').trim();

  const msgs = tree.buildMessagesForAPI(session);
  const userTexts = msgs
    .filter(m => m.role === 'user')
    .map(m => textOf(m.content))
    .filter(Boolean);
  const samples = userTexts.slice(-MAX_STYLE_SAMPLES).map(t =>
    t.length > MAX_SAMPLE_CHARS ? t.slice(-MAX_SAMPLE_CHARS) : t,
  );

  const lastAssistant = [...msgs].reverse().find(m => m.role === 'assistant');
  const sceneBeatRaw = lastAssistant ? textOf(lastAssistant.content) : '';
  const sceneBeat = sceneBeatRaw.length > MAX_SCENE_CHARS
    ? '…' + sceneBeatRaw.slice(-MAX_SCENE_CHARS)
    : sceneBeatRaw;

  const hints = detectStyleHints(userTexts);
  const avg = avgWordCount(userTexts);

  const parts = [
    '[OOC directive — do not narrate or acknowledge this block; just follow it.]',
    '',
    `Write the next message as ${userName}. Do NOT write as ${charName}, and do not narrate ${charName}'s thoughts, speech, or actions.`,
  ];

  if (personaDesc) {
    parts.push('', `${userName}'s persona: ${personaDesc}`);
  }

  if (sceneBeat) {
    parts.push('', `Current scene — ${charName} just said/did:`, sceneBeat);
  }

  if (samples.length > 0) {
    parts.push(
      '',
      `Style reference — these are ${userName}'s OWN prior messages in this session. Match diction, length, POV, tense, and formatting (action asterisks, em-dashes, paragraph breaks, punctuation habits) exactly:`,
      '<<<',
      ...samples,
      '>>>',
    );
  }

  if (hints.length > 0) {
    parts.push('', `Detected style: ${hints.join(', ')}.`);
  }

  parts.push(
    '',
    `Target length: similar to ${userName}'s typical turn (~${avg} words). Write only ${userName}'s next message — no meta, no OOC commentary, no summaries, no stage directions about ${charName}.`,
  );

  return parts.join('\n');
}

/**
 * Fetch the active (default) persona from the server. Returns null if none
 * set or on error.
 */
export async function fetchActivePersona() {
  try {
    const resp = await fetch('/api/personas/');
    if (!resp.ok) return null;
    const data = await resp.json();
    return (data.personas || []).find(p => p.is_default) || null;
  } catch {
    return null;
  }
}

/**
 * Run the impersonation stream. Accumulated text is passed to onText as
 * a full string each delta, so callers can just setValue() their input.
 */
export async function runImpersonate(session, {
  mode,
  sessionId,
  userName,
  charName,
  personaDesc,
  onText,
  onStart,
  onEnd,
  signal,
}) {
  onStart?.();
  try {
    const messages = tree.buildMessagesForAPI(session);
    const instruction = buildImpersonateInstruction(session, {
      userName, charName, personaDesc,
    });
    messages.push({ role: 'user', content: instruction });

    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Augmentum-Mode': mode || 'passthrough',
        'X-Augmentum-Session': sessionId || '',
        'X-Augmentum-Tools': 'none',
      },
      body: JSON.stringify({
        model: app.state.currentModel || '',
        messages,
        stream: true,
      }),
      signal,
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let acc = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const lines = decoder.decode(value, { stream: true }).split('\n');
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const chunk = JSON.parse(line);
          const delta = chunk?.message?.content || '';
          if (delta) {
            acc += delta;
            onText?.(acc);
          }
        } catch { /* skip non-JSON frames */ }
      }
    }

    showToast('Draft ready \u2014 review and send.', 'info');
  } catch (err) {
    if (err.name !== 'AbortError') {
      showToast(`Couldn't impersonate \u2014 ${err.message}`, 'error');
    }
  } finally {
    onEnd?.();
  }
}
