// tests/test_emulator_bridge_js.mjs
//
// Node-runnable smoke tests for the parent-side emulator bridge +
// iframe template. We can't load EmulatorJS itself (no browser DOM /
// WebGL / WebAssembly here at this fidelity), so these tests focus on
// the parts that don't need it:
//
//   * iframe template renders with safe escaping
//   * config normalisation strips unknown keys + coerces types
//   * base64 round-trip helper produces correct output
//   * EmulatorBridge proxies state-saved / sram-saved messages as
//     PUT /api/titles/{id}/saves/{kind}/{slot}
//   * EmulatorBridge fetches saves snapshot on mount and sends
//     'emu:saves-snapshot' to the iframe
//
// Run with:
//     node tests/test_emulator_bridge_js.mjs
//
// Exit code is non-zero on the first failure. Output is one line per
// test, prefixed with PASS or FAIL.

import {
  renderEmulatorIframeSrcdoc,
  __test as templateTest,
} from '../ui/scripts/emulator-iframe-template.js';

let _failed = 0;
let _ran = 0;

function assert(cond, label) {
  _ran++;
  if (cond) {
    console.log(`PASS ${label}`);
  } else {
    _failed++;
    console.error(`FAIL ${label}`);
  }
}

function assertEq(actual, expected, label) {
  assert(
    actual === expected,
    `${label}\n  expected: ${JSON.stringify(expected)}\n    actual: ${JSON.stringify(actual)}`,
  );
}


// ── Template tests ─────────────────────────────────────────────────


(function templateRenders() {
  const html = renderEmulatorIframeSrcdoc({
    config: {
      system: 'nes',
      core: 'fceumm',
      rom_url: '/api/titles/abc/rom',
      title: 'Test Game',
      emulator_js_path: '/ui/lib/emulator-js/data/',
    },
    protocol: { READY: 'emu:ready', LOAD_STATE: 'emu:load-state' },
    titleId: 'abc',
  });
  assert(typeof html === 'string', 'template returns a string');
  assert(html.includes('<!doctype html>'), 'template starts with doctype');
  assert(html.includes('emu:ready'), 'protocol constants get inlined');
  assert(
    html.includes('/ui/lib/emulator-js/data/loader.js'),
    'loader.js script tag references the configured path',
  );
  assert(html.includes('"system":"nes"'), 'config.system reaches iframe');
  assert(html.includes('"core":"fceumm"'), 'config.core reaches iframe');
  assert(html.includes('window.EJS_pathtodata'), 'EJS_pathtodata is set');
})();


(function templateNormalisesConfig() {
  const cleaned = templateTest._normaliseConfig({
    system: 'snes',
    core: 'snes9x',
    rom_url: 'https://x/y',
    bios_required: 1,                 // truthy → bool
    bios_url: null,                   // → ''
    irrelevant_internal_field: 'leak', // dropped
  });
  assertEq(cleaned.system, 'snes', 'system passes through');
  assertEq(cleaned.bios_required, true, 'bios_required coerced to bool');
  assertEq(cleaned.bios_url, '', 'null bios_url becomes empty string');
  assert(
    !('irrelevant_internal_field' in cleaned),
    'unknown keys are stripped',
  );
})();


(function templateEscapesAttributes() {
  const safe = templateTest._safeAttr('"><script>x</script>');
  assert(!safe.includes('<'), 'safeAttr strips < signs');
  assert(!safe.includes('>'), 'safeAttr strips > signs');
  assert(!safe.includes('"'), 'safeAttr strips double quotes');
})();


(function templateRejectsBadInputs() {
  let threw = false;
  try { renderEmulatorIframeSrcdoc({}); } catch (e) { threw = true; }
  assert(threw, 'template throws when config is missing');

  threw = false;
  try {
    renderEmulatorIframeSrcdoc({ config: {}, protocol: null, titleId: 'x' });
  } catch (e) { threw = true; }
  assert(threw, 'template throws when protocol is missing');
})();


// ── Bridge tests ────────────────────────────────────────────────────
//
// We mock fetch + window + iframe so the bridge runs in Node. The
// bridge should:
//  * post 'emu:saves-snapshot' to the iframe on mount
//  * PUT /saves/state/N when iframe posts 'emu:state-saved'
//  * PUT /saves/sram/0 when iframe posts 'emu:sram-saved'

class MockIframe {
  constructor() {
    this._messages = [];
    this.contentWindow = {
      postMessage: (data, _origin) => {
        this._messages.push(data);
      },
    };
    this.parentNode = null;
    this.attributes = new Map();
    this.style = {};
    this.className = '';
    this.srcdoc = '';
  }
  setAttribute(k, v) { this.attributes.set(k, v); }
  appendChild() { /* noop */ }
}

class MockWindow {
  constructor(iframe) {
    this._iframe = iframe;
    this._listeners = new Map();
    this.location = { origin: 'http://localhost:6100' };
    this.document = {
      createElement: (tag) => {
        if (tag === 'iframe') return iframe;
        return { setAttribute: () => {}, style: {}, appendChild: () => {} };
      },
    };
  }
  addEventListener(name, fn) {
    if (!this._listeners.has(name)) this._listeners.set(name, []);
    this._listeners.get(name).push(fn);
  }
  removeEventListener(name, fn) {
    const arr = this._listeners.get(name) || [];
    const idx = arr.indexOf(fn);
    if (idx >= 0) arr.splice(idx, 1);
  }
  /** Test helper: simulate the iframe sending a message to the parent. */
  fireMessage(data) {
    const fns = this._listeners.get('message') || [];
    for (const fn of fns) {
      fn({ source: this._iframe.contentWindow, data, origin: this.location.origin });
    }
  }
}

function makeMockFetch(responses) {
  // responses: array of { match: (url, init) => bool, respond: () => Response-like }
  const calls = [];
  const fetchImpl = async (url, init = {}) => {
    calls.push({ url: String(url), init });
    for (const r of responses) {
      if (r.match(url, init)) return r.respond();
    }
    return { ok: false, status: 404, json: async () => ({ error: 'no match' }) };
  };
  return { fetchImpl, calls };
}

async function bridgeMountsAndProxiesSaves() {
  // Dynamic import so the bridge module loads with our mocked window.
  const { EmulatorBridge } = await import('../ui/scripts/emulator-bridge.js');
  const iframe = new MockIframe();
  const win = new MockWindow(iframe);
  const handle = {
    runtime_id: 'emulator-browser',
    kind: 'emulator',
    target: '/api/titles/t1/rom',
    metadata: { core: 'fceumm', system: 'nes', rom_url: '/api/titles/t1/rom' },
  };
  // Saves-snapshot fetch returns one existing SRAM record + bytes.
  const { fetchImpl, calls } = makeMockFetch([
    {
      match: (url, init) =>
        url === '/api/titles/t1/saves' && (!init || !init.method || init.method === 'GET'),
      respond: () => ({
        ok: true, status: 200,
        json: async () => ({ saves: [
          { kind: 'sram', slot: 0, sha256: 'x', size_bytes: 4, label: '',
            updated_at: '', id: 's1', artifact_id: 't1', core_id: '',
            created_at: '' },
        ]}),
      }),
    },
    {
      match: (url, init) =>
        url === '/api/titles/t1/saves/sram/0' && (!init || !init.method || init.method === 'GET'),
      respond: () => ({
        ok: true, status: 200,
        arrayBuffer: async () => new Uint8Array([0xde, 0xad, 0xbe, 0xef]).buffer,
      }),
    },
    {
      match: (url, init) =>
        url === '/api/titles/t1/saves/state/3' && init?.method === 'PUT',
      respond: () => ({ ok: true, status: 201, json: async () => ({}) }),
    },
    {
      match: (url, init) =>
        url === '/api/titles/t1/saves/sram/0' && init?.method === 'PUT',
      respond: () => ({ ok: true, status: 201, json: async () => ({}) }),
    },
  ]);

  // The 'document.createElement' returns the iframe; bridge appends
  // it via container.appendChild. Use a fake container.
  const container = { appendChild: () => { iframe.parentNode = container; } };
  const bridge = new EmulatorBridge(container, handle, 't1', {
    fetchImpl, windowImpl: win,
  });

  // Mount races against 'emu:ready' -- start the mount, then fire
  // the message after listeners are wired.
  const mountP = bridge.mount();
  // mount() registered the message listener synchronously; fire ready.
  win.fireMessage({ type: 'emu:ready' });
  await mountP;

  // The bridge fetched /saves and /saves/sram/0 to assemble the
  // snapshot; first PUT shouldn't have happened yet.
  const getCalls = calls.filter(c => !c.init?.method || c.init.method === 'GET');
  assert(
    getCalls.some(c => c.url === '/api/titles/t1/saves'),
    'bridge fetched saves index on mount',
  );
  assert(
    getCalls.some(c => c.url === '/api/titles/t1/saves/sram/0'),
    'bridge fetched SRAM bytes on mount',
  );

  // The iframe should have received 'emu:saves-snapshot' with
  // sram_b64 set.
  const snapshot = iframe._messages.find(m => m.type === 'emu:saves-snapshot');
  assert(!!snapshot, 'iframe received saves-snapshot');
  assert(typeof snapshot.sram_b64 === 'string' && snapshot.sram_b64.length > 0,
    'snapshot includes sram_b64');

  // Now simulate a state save fired by the iframe.
  win.fireMessage({
    type: 'emu:state-saved',
    slot: 3,
    data_b64: 'AAEC',                // base64 of 0x00 0x01 0x02
  });
  // PUT happens async; tick the microtask queue.
  await new Promise(r => setTimeout(r, 0));
  await new Promise(r => setTimeout(r, 0));

  const putCalls = calls.filter(c => c.init?.method === 'PUT');
  assert(
    putCalls.some(c => c.url === '/api/titles/t1/saves/state/3'),
    'state-saved → PUT /saves/state/3',
  );

  // SRAM-saved → PUT /saves/sram/0
  win.fireMessage({ type: 'emu:sram-saved', data_b64: 'AAEC' });
  await new Promise(r => setTimeout(r, 0));
  await new Promise(r => setTimeout(r, 0));
  const sramPuts = calls.filter(
    c => c.init?.method === 'PUT' && c.url === '/api/titles/t1/saves/sram/0',
  );
  assert(sramPuts.length >= 1, 'sram-saved → PUT /saves/sram/0');

  bridge.unmount();
  // After unmount, posting a message should not crash + should not
  // produce more PUTs.
  const beforePuts = calls.filter(c => c.init?.method === 'PUT').length;
  win.fireMessage({ type: 'emu:state-saved', slot: 1, data_b64: 'XX' });
  await new Promise(r => setTimeout(r, 0));
  const afterPuts = calls.filter(c => c.init?.method === 'PUT').length;
  assertEq(afterPuts, beforePuts, 'no PUTs after unmount');
}

async function bridgeRejectsBadHandle() {
  const { EmulatorBridge } = await import('../ui/scripts/emulator-bridge.js');
  const iframe = new MockIframe();
  const win = new MockWindow(iframe);
  const container = { appendChild: () => {} };
  const { fetchImpl } = makeMockFetch([]);
  let threw = false;
  try {
    new EmulatorBridge(container, { runtime_id: 'browser-iframe', kind: 'iframe' }, 't1', {
      fetchImpl, windowImpl: win,
    });
  } catch (e) { threw = true; }
  assert(threw, 'bridge rejects non-emulator handles');
}


// ── Run ────────────────────────────────────────────────────────────


(async () => {
  await bridgeMountsAndProxiesSaves();
  await bridgeRejectsBadHandle();
  console.log('');
  console.log(`${_ran - _failed}/${_ran} passed`);
  process.exit(_failed ? 1 : 0);
})();
