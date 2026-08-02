/*
 * XR Surface Adapters
 *
 * A thin translation layer between desktop Augmentum surfaces and headset-safe
 * XR panels. These adapters deliberately do not mount DOM or iframes in XR.
 * They describe the useful app capabilities that can be driven from a stereo
 * WebGL surface, then avatar-xr.js routes selections back through existing
 * voice/app events.
 */

import { XR_MEDIA_SECTIONS } from './xr-media-library.js';

const ADAPTERS = Object.freeze({
  chat: {
    title: 'Conversation',
    summary: 'Live voice transcript, follow-ups, pins, and recap work.',
    actions: ['reply', 'summarize', 'pin'],
    lines: ['Ask naturally; voice remains the primary input.', 'Pinned context stays attached to the current call.'],
    next: ['Transcript view', 'Pinned answer cards'],
  },
  analytical: {
    title: 'Analyze',
    summary: 'Research, compare evidence, and explain reasoning without leaving XR.',
    actions: ['search', 'compare', 'explain'],
    lines: ['Use voice to provide the query or source.', 'Results should become source cards on this panel.'],
    next: ['Source cards', 'Evidence comparison board'],
  },
  agentic: {
    title: 'Build',
    summary: 'Plan, execute, and check agentic work from a headset-safe task board.',
    actions: ['plan', 'execute', 'check_status'],
    lines: ['Voice drives task intent; progress should stream back as steps.', 'Approvals must remain explicit.'],
    next: ['Live task list', 'Approval cards'],
  },
  narrative: {
    title: 'Story',
    summary: 'Character, scene, speaker, and recap controls for narrative sessions.',
    actions: ['continue_scene', 'switch_speaker', 'summarize_scene'],
    lines: ['Speak the next beat or choose a speaker by voice.', 'Scene state should render here instead of opening desktop panels.'],
    next: ['Character strip', 'Scene state cards'],
  },
  browse: {
    title: 'Browse',
    summary: 'Search, read, save sources, and hand videos to XR media playback.',
    actions: ['search', 'summarize_page', 'save_source', 'play_media'],
    lines: ['Use voice for search and page questions.', 'External web pages need a browser-stream adapter; first-party results can render as cards.'],
    next: ['Search results', 'Readable page cards', 'Video handoff'],
  },
  files: {
    title: 'Files',
    summary: 'Open, attach, and compare documents as source cards in XR.',
    actions: ['open', 'attach', 'compare'],
    lines: ['Use voice to name a file or filter.', 'Document previews should render as pages or summaries.'],
    next: ['Recent files rail', 'Document preview'],
  },
  coder: {
    title: 'Coder',
    summary: 'Plan, diff review, checks, and approval flow while talking with the VRM.',
    actions: ['show_plan', 'review_diff', 'run_checks', 'approve'],
    lines: ['Commands stay explicit; approvals require a selected card.', 'Diff/test output should stream into this panel.'],
    next: ['Plan/checklist cards', 'Diff summary', 'Command approvals'],
  },
  notes: {
    title: 'Notes',
    summary: 'Dictate, clip, and organize notes from voice in XR.',
    actions: ['dictate', 'clip', 'organize'],
    lines: ['Voice creates or appends note content.', 'Clips should attach to the active Browse/Files context.'],
    next: ['Active note editor', 'Recent notes'],
  },
  studio: {
    title: 'Studio',
    summary: 'Generate, vary, edit, and review visual artifacts.',
    actions: ['generate', 'variant', 'edit', 'save'],
    lines: ['Image results should become inspectable XR cards.', 'Editing should use explicit selection handles.'],
    next: ['Generation queue', 'Image gallery'],
  },
  media: {
    title: 'Media',
    summary: 'A headset media center for watching, reading, listening, and playing together.',
    actions: ['continue', 'shows_movies', 'comics', 'audiobooks', 'images', 'local_files', 'games'],
    lines: ['Voice selects content; panels become theater, reader, gallery, or listening surfaces.', 'Controllers and paired gamepads stay reserved for media and games when focused.'],
    next: XR_MEDIA_SECTIONS.map((section) => section.label),
  },
  devices: {
    title: 'Devices',
    summary: 'Casting, pairing, volume, and session status.',
    actions: ['cast', 'volume', 'pair', 'stop'],
    lines: ['Device operations should be confirm-first in XR.', 'Status cards can show active sessions.'],
    next: ['Connected device list', 'Cast controls'],
  },
  games: {
    title: 'Games',
    summary: 'Launch/resume games and pass Quest or paired gamepad input when supported.',
    actions: ['launch', 'resume', 'controller_mode', 'stop_stream'],
    lines: ['Canvas/native game surfaces can receive mapped controller input.', 'Iframe games need a stream/focus adapter before they are full XR-playable.'],
    next: ['Game library rail', 'Controller mapping status', 'Stream surface'],
  },
});

function _formatLabel(value) {
  return String(value || '')
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (m) => m.toUpperCase());
}

export function getXrSurfaceAdapter(action) {
  return ADAPTERS[String(action || '').trim()] || null;
}

export function describeXrSurface(surface = {}, context = {}) {
  const action = String(surface.action || surface.id || '').trim();
  const adapter = getXrSurfaceAdapter(action);
  const actions = adapter?.actions?.length
    ? adapter.actions
    : (Array.isArray(surface.primaryActions) ? surface.primaryActions : ['focus']);
  return {
    title: adapter?.title || surface.label || _formatLabel(action || 'Surface'),
    summary: adapter?.summary || surface.voiceCue || surface.hint || 'Voice stays attached to this surface.',
    lines: adapter?.lines || [],
    next: adapter?.next || [],
    actions,
    activeAction: context.selectedAction || '',
  };
}

export function formatXrActionLabel(action) {
  if (action === 'audiobooks') return 'Audiobooks';
  if (action === 'check_status') return 'Check Status';
  if (action === 'continue') return 'Continue';
  if (action === 'local_files') return 'Local Files';
  if (action === 'summarize_page') return 'Summarize Page';
  if (action === 'continue_scene') return 'Continue Scene';
  if (action === 'switch_speaker') return 'Switch Speaker';
  if (action === 'summarize_scene') return 'Summarize Scene';
  if (action === 'shows_movies') return 'Shows + Movies';
  if (action === 'review_diff') return 'Review Diff';
  if (action === 'run_checks') return 'Run Checks';
  if (action === 'show_plan') return 'Show Plan';
  if (action === 'save_source') return 'Save Source';
  if (action === 'stop_stream') return 'Stop Stream';
  if (action === 'play_media') return 'Play Media';
  if (action === 'controller_mode') return 'Controller Mode';
  return _formatLabel(action);
}
