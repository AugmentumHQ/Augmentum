import { mkdir, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import {
  CALL_ACTION_ORDER,
  getCallActionContract,
} from '../ui/scripts/avatar-call-actions.js';

const DEFAULT_AVATARS = [
  'vance.vrm',
  'Becca.vrm',
];

const DEFAULT_VIEWPORTS = [
  { name: 'desktop', width: 1280, height: 900, camera: 'upper' },
  { name: 'half', width: 900, height: 760, camera: 'upper' },
  { name: 'mobile', width: 390, height: 844, camera: 'upper', mobile: true },
];

const chromePath = process.env.CHROME_PATH || 'chrome';
const debugPort = Number(process.env.AVATAR_AUDIT_DEBUG_PORT || 9455);
const baseUrl = process.env.AVATAR_AUDIT_BASE_URL
  || 'http://127.0.0.1:8777/ui/avatar-normalization-lab.html';
const outputDir = process.env.AVATAR_AUDIT_OUT
  || `./out/avatar-action-audit-${timestamp()}`;
const profileDir = process.env.AVATAR_AUDIT_PROFILE
  || './out/chrome-avatar-action-audit-profile';
const avatars = parseList(process.env.AVATAR_AUDIT_AVATARS, DEFAULT_AVATARS);
const actions = parseList(process.env.AVATAR_AUDIT_ACTIONS, CALL_ACTION_ORDER);
const viewports = parseViewports(process.env.AVATAR_AUDIT_VIEWPORTS, DEFAULT_VIEWPORTS);
const frameOffsetsMs = parseNumberList(process.env.AVATAR_AUDIT_OFFSETS_MS, [350, 800, 1250]);
const requestedState = process.env.AVATAR_AUDIT_STATE || 'auto';
const emotion = process.env.AVATAR_AUDIT_EMOTION || 'neutral';
const retryCount = Number(process.env.AVATAR_AUDIT_RETRIES || 1);

async function fetchJson(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
  return await resp.json();
}

async function waitForDebugUrl() {
  const deadline = Date.now() + 15000;
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      const targets = await fetchJson(`http://127.0.0.1:${debugPort}/json/list`);
      const page = targets.find((target) => target.type === 'page' && target.webSocketDebuggerUrl);
      if (page) return page.webSocketDebuggerUrl;
    } catch (error) {
      lastError = error;
    }
    await delay(250);
  }
  throw new Error(`Chrome DevTools did not open: ${lastError?.message || 'timeout'}`);
}

function connectCdp(wsUrl) {
  const ws = new WebSocket(wsUrl);
  let nextId = 1;
  const pending = new Map();

  ws.addEventListener('message', (event) => {
    const msg = JSON.parse(event.data);
    if (!msg.id) return;
    const entry = pending.get(msg.id);
    if (!entry) return;
    pending.delete(msg.id);
    if (msg.error) {
      entry.reject(new Error(msg.error.message || JSON.stringify(msg.error)));
    } else {
      entry.resolve(msg.result || {});
    }
  });

  const ready = new Promise((resolve, reject) => {
    ws.addEventListener('open', resolve, { once: true });
    ws.addEventListener('error', reject, { once: true });
  });

  function send(method, params = {}) {
    const id = nextId++;
    ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
    });
  }

  return { ws, ready, send };
}

async function waitForHarness(cdp, expectedUrl, expectedCase) {
  const deadline = Date.now() + 45000;
  while (Date.now() < deadline) {
    const result = await evaluate(cdp, `
      (() => {
        if (location.href !== ${JSON.stringify(expectedUrl)}) return false;
        const harness = window.__avatarPoseHarness;
        if (!harness) return false;
        if (harness.error) return String(harness.error);
        if (harness.ready !== true) return false;
        const stats = harness.stats || {};
        return stats.avatarFile === ${JSON.stringify(expectedCase.avatar)}
          && stats.action === ${JSON.stringify(expectedCase.action)}
          && stats.state === ${JSON.stringify(expectedCase.state)};
      })()
    `);
    if (result === true) return;
    if (typeof result === 'string') throw new Error(result);
    await delay(400);
  }
  throw new Error('Avatar lab did not become ready');
}

async function evaluate(cdp, expression) {
  const result = await cdp.send('Runtime.evaluate', {
    expression,
    returnByValue: true,
    awaitPromise: true,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.text || 'Runtime evaluation failed');
  }
  return result.result?.value;
}

async function getCanvasRect(cdp) {
  return await evaluate(cdp, `
    (() => {
      const rect = document.getElementById('normalized-canvas').getBoundingClientRect();
      return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
    })()
  `);
}

async function getStats(cdp) {
  return await evaluate(cdp, 'window.__avatarPoseHarness?.stats || {}');
}

async function getVisualMetrics(cdp) {
  return await evaluate(cdp, `
    (() => {
      const canvas = document.getElementById('normalized-canvas');
      if (!canvas) return { ok: false, error: 'missing normalized canvas' };
      const sample = 160;
      const scratch = document.createElement('canvas');
      scratch.width = sample;
      scratch.height = sample;
      const ctx = scratch.getContext('2d', { willReadFrequently: true });
      try {
        ctx.clearRect(0, 0, sample, sample);
        ctx.drawImage(canvas, 0, 0, sample, sample);
        const data = ctx.getImageData(0, 0, sample, sample).data;
        let minX = sample;
        let minY = sample;
        let maxX = -1;
        let maxY = -1;
        let pixels = 0;
        let alphaSum = 0;
        for (let y = 0; y < sample; y += 1) {
          for (let x = 0; x < sample; x += 1) {
            const alpha = data[((y * sample + x) * 4) + 3];
            if (alpha <= 12) continue;
            pixels += 1;
            alphaSum += alpha;
            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
          }
        }
        if (!pixels) {
          return { ok: false, error: 'empty canvas', nonTransparentRatio: 0 };
        }
        const width = maxX - minX + 1;
        const height = maxY - minY + 1;
        const margins = {
          left: minX / sample,
          right: (sample - 1 - maxX) / sample,
          top: minY / sample,
          bottom: (sample - 1 - maxY) / sample,
        };
        return {
          ok: true,
          sample,
          nonTransparentRatio: pixels / (sample * sample),
          averageAlpha: alphaSum / pixels,
          bbox: {
            x: minX / sample,
            y: minY / sample,
            width: width / sample,
            height: height / sample,
            centerX: (minX + maxX) / (2 * sample),
            centerY: (minY + maxY) / (2 * sample),
          },
          margins,
          touchesEdge: Math.min(margins.left, margins.right, margins.top, margins.bottom) < 0.0125,
          centerOffsetX: Math.abs(((minX + maxX) / (2 * sample)) - 0.5),
        };
      } catch (error) {
        return { ok: false, error: String(error?.message || error) };
      }
    })()
  `);
}

async function configureViewport(cdp, viewport) {
  await cdp.send('Emulation.setDeviceMetricsOverride', {
    width: viewport.width,
    height: viewport.height,
    deviceScaleFactor: 1,
    mobile: !!viewport.mobile,
  });
}

async function captureCase(cdp, auditCase, caseDir) {
  const url = buildLabUrl(auditCase);
  await cdp.send('Page.navigate', { url });
  await waitForHarness(cdp, url, auditCase);

  const frames = [];
  const start = Date.now();
  for (let i = 0; i < frameOffsetsMs.length; i += 1) {
    const target = start + frameOffsetsMs[i];
    const waitMs = target - Date.now();
    if (waitMs > 0) await delay(waitMs);

    const [stats, metrics, rect] = await Promise.all([
      getStats(cdp),
      getVisualMetrics(cdp),
      getCanvasRect(cdp),
    ]);
    const shot = await cdp.send('Page.captureScreenshot', {
      format: 'png',
      fromSurface: true,
      captureBeyondViewport: false,
      clip: {
        x: Math.max(0, rect.x),
        y: Math.max(0, rect.y),
        width: Math.max(1, rect.width),
        height: Math.max(1, rect.height),
        scale: 1,
      },
    });
    const screenshot = `${caseId(auditCase)}-f${String(i + 1).padStart(2, '0')}.png`;
    await writeFile(path.join(caseDir, screenshot), Buffer.from(shot.data, 'base64'));
    frames.push({
      index: i + 1,
      offsetMs: frameOffsetsMs[i],
      screenshot,
      stats,
      metrics,
      findings: scoreFrame(stats, metrics),
    });
  }

  return {
    ...auditCase,
    url,
    frames,
    summary: summarizeCase(frames),
  };
}

function scoreFrame(stats, metrics) {
  const findings = [];
  const coverage = stats?.curatedActionCoverage;
  const trigger = stats?.actionTrigger;
  if (trigger && trigger.accepted === false) findings.push(`trigger rejected: ${trigger.reason}`);
  if (trigger?.fallbackUsed) findings.push(`fallback used: ${trigger.fallbackChain?.join(' -> ')}`);
  if (coverage?.canPlay === false) findings.push('action cannot play on this rig');
  if (coverage?.status === 'fallback') findings.push('compatibility requests fallback');
  if (coverage?.warnings?.length) findings.push(...coverage.warnings);

  if (!metrics?.ok) {
    findings.push(metrics?.error || 'visual metrics unavailable');
  } else {
    if (metrics.nonTransparentRatio < 0.018) findings.push('avatar silhouette is very small or blank');
    if (metrics.bbox?.height > 0.96) findings.push('avatar nearly fills canvas vertically');
    if (metrics.bbox?.width > 0.9) findings.push('pose is very wide in frame');
    else if (metrics.bbox?.width > 0.74) findings.push('pose is broad in frame');
    if (metrics.centerOffsetX > 0.18) findings.push('pose is visually off-center');
    if (Math.min(metrics.margins?.left ?? 1, metrics.margins?.right ?? 1) < 0.03) {
      findings.push('gesture near side edge');
    }
    if (metrics.margins?.top < 0.02) findings.push('head/gesture near top edge');
  }
  return [...new Set(findings)];
}

function summarizeCase(frames) {
  const findingCounts = new Map();
  let worstWidth = 0;
  let worstCenterOffset = 0;
  let minMargin = 1;
  for (const frame of frames) {
    for (const finding of frame.findings) {
      findingCounts.set(finding, (findingCounts.get(finding) || 0) + 1);
    }
    if (frame.metrics?.ok) {
      worstWidth = Math.max(worstWidth, frame.metrics.bbox.width || 0);
      worstCenterOffset = Math.max(worstCenterOffset, frame.metrics.centerOffsetX || 0);
      minMargin = Math.min(minMargin, ...Object.values(frame.metrics.margins || { all: 1 }));
    }
  }
  const findings = [...findingCounts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([finding, count]) => ({ finding, count }));
  return {
    status: findings.length ? 'review' : 'pass',
    findings,
    worstWidth,
    worstCenterOffset,
    minMargin,
  };
}

function buildLabUrl({ avatar, action, state: actionState, viewport }) {
  const url = new URL(baseUrl);
  url.searchParams.set('avatar', avatar);
  url.searchParams.set('action', action);
  url.searchParams.set('state', actionState || stateForAction(action));
  url.searchParams.set('emotion', emotion);
  url.searchParams.set('camera', viewport.camera || 'upper');
  url.searchParams.set('autoplay', '1');
  return url.toString();
}

function stateForAction(action) {
  if (requestedState !== 'auto') return requestedState;
  const phases = getCallActionContract(action)?.phases || [];
  return phases[0] || 'speaking';
}

function caseId({ avatar, action, viewport }) {
  return `${stripExt(avatar)}__${action}__${viewport.name}`;
}

function stripExt(file) {
  return file.replace(/\.[^.]+$/, '');
}

function parseList(value, fallback) {
  if (!value) return fallback;
  return value.split(',').map((item) => item.trim()).filter(Boolean);
}

function parseNumberList(value, fallback) {
  if (!value) return fallback;
  const parsed = value.split(',').map((item) => Number(item.trim())).filter(Number.isFinite);
  return parsed.length ? parsed : fallback;
}

function parseViewports(value, fallback) {
  if (!value) return fallback;
  const parsed = value.split(',').map((part) => {
    const [namePart, sizePart] = part.split(':');
    const [w, h] = (sizePart || '').split('x').map(Number);
    if (!namePart || !Number.isFinite(w) || !Number.isFinite(h)) return null;
    return { name: namePart.trim(), width: w, height: h, camera: 'upper', mobile: w <= 480 };
  }).filter(Boolean);
  return parsed.length ? parsed : fallback;
}

function timestamp() {
  return new Date().toISOString().replace(/[:.]/g, '-').replace('T', '-').slice(0, 19);
}

async function writeReport(results, outDir) {
  const summary = {
    generatedAt: new Date().toISOString(),
    baseUrl,
    state: requestedState,
    emotion,
    avatars,
    actions,
    viewports,
    frameOffsetsMs,
    totalCases: results.length,
    reviewCases: results.filter((result) => result.summary.status === 'review').length,
    passCases: results.filter((result) => result.summary.status === 'pass').length,
    results,
  };
  await writeFile(path.join(outDir, 'summary.json'), JSON.stringify(summary, null, 2));
  await writeFile(path.join(outDir, 'report.html'), renderReport(summary));
  return summary;
}

function renderReport(summary) {
  const rows = summary.results.map((result) => {
    const firstFrame = result.frames[0];
    const findings = result.summary.findings.length
      ? result.summary.findings.map((item) => `${escapeHtml(item.finding)} (${item.count})`).join('<br>')
      : '<span class="ok">pass</span>';
    const thumbs = result.frames.map((frame) => (
      `<a href="${escapeAttr(frame.screenshot)}"><img src="${escapeAttr(frame.screenshot)}" alt="${escapeAttr(`${result.action} frame ${frame.index}`)}"></a>`
    )).join('');
    const coverage = firstFrame?.stats?.curatedActionCoverage;
    const readiness = coverage
      ? `${escapeHtml(coverage.status || 'unknown')} ${Math.round((coverage.score || coverage.coverage || 0) * 100)}%`
      : 'unknown';
    return `
      <tr class="${result.summary.status}">
        <td>${escapeHtml(result.avatar)}</td>
        <td>${escapeHtml(result.action)}</td>
        <td>${escapeHtml(result.viewport.name)}<br><span>${result.viewport.width}x${result.viewport.height}</span></td>
        <td>${escapeHtml(result.state || stateForAction(result.action))}</td>
        <td>${readiness}</td>
        <td>${findings}</td>
        <td class="thumbs">${thumbs}</td>
      </tr>
    `;
  }).join('');

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Avatar Action Audit</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, Segoe UI, sans-serif; background: #080b10; color: #eef3ff; }
    body { margin: 0; padding: 24px; background: #080b10; }
    h1 { margin: 0 0 6px; font-size: 22px; }
    .meta { color: #9fb0c8; margin-bottom: 18px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 10px; border-top: 1px solid rgba(255,255,255,.1); vertical-align: top; text-align: left; }
    th { color: #b9c7dc; font-weight: 650; position: sticky; top: 0; background: #080b10; }
    tr.review { background: rgba(255, 195, 89, .055); }
    tr.pass { background: rgba(108, 220, 161, .035); }
    td span, .meta { font-size: 12px; }
    .ok { color: #86efac; }
    .thumbs { display: flex; gap: 8px; min-width: 260px; }
    img { width: 82px; height: 116px; object-fit: contain; background: #101722; border: 1px solid rgba(255,255,255,.12); border-radius: 6px; }
  </style>
</head>
<body>
  <h1>Avatar Action Audit</h1>
  <div class="meta">${summary.generatedAt} · ${summary.totalCases} cases · ${summary.reviewCases} review · ${summary.passCases} pass</div>
  <table>
    <thead>
      <tr><th>Avatar</th><th>Action</th><th>Viewport</th><th>Phase</th><th>Readiness</th><th>Findings</th><th>Frames</th></tr>
    </thead>
    <tbody>${rows}</tbody>
  </table>
</body>
</html>`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[ch]));
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, '&#96;');
}

async function main() {
  await mkdir(outputDir, { recursive: true });

  const chrome = spawn(chromePath, [
    '--headless=new',
    '--disable-gpu',
    '--hide-scrollbars',
    '--no-sandbox',
    '--force-device-scale-factor=1',
    `--remote-debugging-port=${debugPort}`,
    `--user-data-dir=${profileDir}`,
    `--window-size=${viewports[0].width},${viewports[0].height}`,
    'about:blank',
  ], {
    stdio: ['ignore', 'ignore', 'pipe'],
    windowsHide: true,
  });

  let stderr = '';
  chrome.stderr.on('data', (chunk) => {
    stderr += chunk.toString();
  });

  const results = [];
  try {
    const wsUrl = await waitForDebugUrl();
    const cdp = connectCdp(wsUrl);
    await cdp.ready;
    await cdp.send('Page.enable');
    await cdp.send('Network.enable');
    await cdp.send('Network.setCacheDisabled', { cacheDisabled: true });
    await cdp.send('Runtime.enable');

    for (const viewport of viewports) {
      await configureViewport(cdp, viewport);
      for (const avatar of avatars) {
        for (const action of actions) {
          const auditCase = { avatar, action, state: stateForAction(action), viewport };
          const id = caseId(auditCase);
          console.log(`audit ${id}`);
          let captured = null;
          let lastError = null;
          for (let attempt = 0; attempt <= retryCount; attempt += 1) {
            try {
              captured = await captureCase(cdp, auditCase, outputDir);
              break;
            } catch (error) {
              lastError = error;
              if (attempt < retryCount) {
                console.warn(`retry ${id}: ${error?.message || error}`);
                await cdp.send('Page.navigate', { url: 'about:blank' }).catch(() => {});
                await delay(450);
              }
            }
          }
          if (captured) {
            results.push(captured);
          } else {
            results.push({
              ...auditCase,
              url: buildLabUrl(auditCase),
              frames: [],
              summary: {
                status: 'review',
                findings: [{ finding: `capture failed: ${lastError?.message || lastError}`, count: 1 }],
                worstWidth: 0,
                worstCenterOffset: 0,
                minMargin: 0,
              },
            });
          }
        }
      }
    }

    const summary = await writeReport(results, outputDir);
    await cdp.send('Browser.close').catch(() => {});
    cdp.ws.close();
    console.log(`done: ${outputDir}`);
    console.log(`cases: ${summary.totalCases}, review: ${summary.reviewCases}, pass: ${summary.passCases}`);
  } finally {
    if (!chrome.killed) chrome.kill();
    if (stderr.trim()) {
      console.error(stderr.trim().split('\n').slice(-8).join('\n'));
    }
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
