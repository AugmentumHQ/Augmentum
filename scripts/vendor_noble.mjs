/* Vendor the audited @noble crypto for the client E2E port — recursively.
 *
 *   node scripts/vendor_noble.mjs
 *
 * Crawls the esm.sh ESM module graph from four entry points (ed25519+x25519,
 * chacha20poly1305, hkdf, sha2), fetching every transitive module and
 * flattening it into ui/lib/noble/ with imports rewritten to local files —
 * so the vendored set is fully self-contained (no network at runtime),
 * browser- and Node-loadable, and auditable. Pinned versions; SHA-256
 * recorded in ui/lib/VENDORED.md. Re-run to refresh.
 */
import { writeFileSync, appendFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, '..', 'ui', 'lib', 'noble');
const V = '2.2.0';
const BASE = 'https://esm.sh';

// Stable facade names the four entry modules MUST land at (index.js imports these).
const ENTRIES = {
  [`${BASE}/@noble/curves@${V}/es2022/ed25519.mjs`]: 'curves-ed25519.mjs',
  [`${BASE}/@noble/ciphers@${V}/es2022/chacha.mjs`]: 'ciphers-chacha.mjs',
  [`${BASE}/@noble/hashes@${V}/es2022/hkdf.mjs`]: 'hashes-hkdf.mjs',
  [`${BASE}/@noble/hashes@${V}/es2022/sha2.mjs`]: 'hashes-sha2.mjs',
};

async function get(url) {
  for (let i = 0; i < 5; i++) {
    try { const r = await fetch(url, { signal: AbortSignal.timeout(30000) }); if (r.ok) return await r.text(); }
    catch { /* retry */ }
    await new Promise((s) => setTimeout(s, 1200));
  }
  return null;
}

const IMPORT_RE = /((?:from|import)\s*["'])([^"']+)(["'])/g;

function resolve(spec, baseUrl) {
  if (spec.startsWith('http')) return spec;
  if (spec.startsWith('/')) return BASE + spec;
  if (spec.startsWith('.')) return new URL(spec, baseUrl).href;
  return null; // bare specifier — shouldn't occur inside the esm.sh tree
}

function depName(url) {
  // /@noble/curves@2.2.0/es2022/abstract/edwards.mjs -> curves_abstract_edwards.mjs
  const p = new URL(url).pathname.replace(/^\/@noble\//, '').replace(`@${V}/es2022/`, '/');
  return 'dep_' + p.replace(/[@/]/g, '_');
}

const assigned = new Map(Object.entries(ENTRIES));       // url -> localName
const queue = [...Object.keys(ENTRIES)];
const seen = new Set();
let fetched = 0, failed = 0;

while (queue.length) {
  const url = queue.shift();
  if (seen.has(url)) continue;
  seen.add(url);
  const body = await get(url);
  if (body === null) { console.error('FAILED:', url); failed++; continue; }

  const rewritten = body.replace(IMPORT_RE, (m, pre, spec, post) => {
    const abs = resolve(spec, url);
    if (!abs) { console.error('  bare import (skipped):', spec); return m; }
    if (!assigned.has(abs)) { assigned.set(abs, depName(abs)); queue.push(abs); }
    return pre + './' + assigned.get(abs) + post;
  });
  writeFileSync(join(out, assigned.get(url)), rewritten);
  fetched++;
}

// Record provenance.
let manifest = `\n## @noble E2E crypto (vendored ${new Date().toISOString().slice(0, 10)} by scripts/vendor_noble.mjs)\n`;
manifest += `Pinned @noble/{curves,ciphers,hashes}@${V}, flattened from esm.sh es2022 ESM.\n`;
for (const [url, name] of assigned) {
  const body = await get(url);
  if (body) manifest += `- ${name}  sha256:${createHash('sha256').update(body).digest('hex').slice(0, 32)}\n`;
}
appendFileSync(join(here, '..', 'ui', 'lib', 'VENDORED.md'), manifest);

console.log(`\nvendored ${fetched} modules into ui/lib/noble/ (${failed} failed)`);
process.exit(failed === 0 ? 0 : 1);
