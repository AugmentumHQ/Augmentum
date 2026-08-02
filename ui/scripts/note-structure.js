/**
 * note-structure.js — Format-Aware Structure Markers
 *
 * Observes HRs in the ProseMirror editor and renders them as
 * format-specific structural markers (chapters, page breaks, slides).
 */

let _container = null;
let _format = 'note';
let _observer = null;

/* ── public API ── */

export function init(editorContainer) {
  _container = editorContainer;
  _format = 'note';
  _setupObserver();
  _renderMarkers();
}

export function setFormat(fmt) {
  _format = fmt || 'note';
  _renderMarkers();
}

export function destroy() {
  if (_observer) { _observer.disconnect(); _observer = null; }
  _unwrapAll();
  _container = null;
}

/* ── observer ── */

function _setupObserver() {
  const pm = _getPM();
  if (!pm) return;

  _observer = new MutationObserver(() => _renderMarkers());
  _observer.observe(pm, { childList: true, subtree: true });
}

function _getPM() {
  if (!_container) return null;
  return _container.querySelector('#note-editor-body .ProseMirror')
      || document.querySelector('#note-editor-body .ProseMirror');
}

/* ── rendering ── */

function _renderMarkers() {
  const pm = _getPM();
  if (!pm) return;

  // unwrap any previous wrappers first
  _unwrapAll();

  if (_format === 'note') return; // plain HR, no markers

  const hrs = Array.from(pm.querySelectorAll('hr'));
  let counter = 0;

  for (const hr of hrs) {
    counter++;
    const { label, color } = _markerInfo(hr, counter);

    const wrapper = document.createElement('div');
    wrapper.className = 'note-structure-marker';
    wrapper.setAttribute('data-structure', '1');

    const labelSpan = document.createElement('span');
    labelSpan.className = 'note-structure-label';
    labelSpan.setAttribute('data-color', color);
    labelSpan.textContent = label;

    hr.parentNode.insertBefore(wrapper, hr);
    hr.style.display = 'none';
    wrapper.appendChild(hr);
    wrapper.insertBefore(labelSpan, hr);
  }
}

function _markerInfo(hr, index) {
  switch (_format) {
    case 'epub': {
      const name = _nextHeadingText(hr);
      const label = name
        ? `\u00A7 Chapter ${index}: ${name}`
        : `\u00A7 Chapter ${index}`;
      return { label, color: 'indigo' };
    }
    case 'pdf':
    case 'docx':
      return { label: 'Page Break', color: 'amber' };
    case 'pptx':
      return { label: `Slide ${index}`, color: 'green' };
    default:
      return { label: '', color: '' };
  }
}

/**
 * Look for the next H1/H2/H3 sibling after the HR (or its wrapper)
 * to extract chapter name text.
 */
function _nextHeadingText(hr) {
  let el = hr.parentNode;
  // if wrapped, start from wrapper
  if (el && el.classList?.contains('note-structure-marker')) {
    el = el.nextElementSibling;
  } else {
    el = hr.nextElementSibling;
  }
  while (el) {
    const tag = el.tagName;
    if (tag === 'H1' || tag === 'H2' || tag === 'H3') {
      return el.textContent.trim();
    }
    el = el.nextElementSibling;
  }
  return '';
}

/* ── cleanup ── */

function _unwrapAll() {
  const pm = _getPM();
  if (!pm) return;

  const wrappers = pm.querySelectorAll('[data-structure]');
  for (const w of wrappers) {
    const hr = w.querySelector('hr');
    if (hr) {
      hr.style.display = '';
      w.parentNode.insertBefore(hr, w);
    }
    w.remove();
  }
}
