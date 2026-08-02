import { mkdir, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';

const chromePath = process.env.CHROME_PATH || 'chrome';
const debugPort = Number(process.env.AVATAR_CAPTURE_DEBUG_PORT || 9444);
const appUrl = process.env.AVATAR_CAPTURE_URL
  || 'http://127.0.0.1:8777/ui/avatar-pose-harness.html?avatar=vance.vrm';
const outputDir = process.env.AVATAR_CAPTURE_OUT
  || './out/avatar-pose-frames';
const profileDir = process.env.AVATAR_CAPTURE_PROFILE
  || './out/chrome-avatar-pose-profile';
const frameCount = Number(process.env.AVATAR_CAPTURE_FRAMES || 30);
const intervalMs = Number(process.env.AVATAR_CAPTURE_INTERVAL_MS || 1000);
const viewport = {
  width: Number(process.env.AVATAR_CAPTURE_WIDTH || 1280),
  height: Number(process.env.AVATAR_CAPTURE_HEIGHT || 900),
};

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

async function waitForHarness(cdp) {
  const deadline = Date.now() + 45000;
  while (Date.now() < deadline) {
    const result = await cdp.send('Runtime.evaluate', {
      expression: 'window.__avatarPoseHarness && (window.__avatarPoseHarness.ready || window.__avatarPoseHarness.error || false)',
      returnByValue: true,
    });
    if (result.result?.value === true) return;
    if (typeof result.result?.value === 'string') {
      throw new Error(result.result.value);
    }
    await delay(500);
  }
  throw new Error('Avatar harness did not become ready');
}

async function getStats(cdp) {
  const result = await cdp.send('Runtime.evaluate', {
    expression: 'window.__avatarPoseHarness?.stats || {}',
    returnByValue: true,
  });
  return result.result?.value || {};
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
    `--window-size=${viewport.width},${viewport.height}`,
    'about:blank',
  ], {
    stdio: ['ignore', 'ignore', 'pipe'],
    windowsHide: true,
  });

  let stderr = '';
  chrome.stderr.on('data', (chunk) => {
    stderr += chunk.toString();
  });

  try {
    const wsUrl = await waitForDebugUrl();
    const cdp = connectCdp(wsUrl);
    await cdp.ready;
    await cdp.send('Page.enable');
    await cdp.send('Network.enable');
    await cdp.send('Network.setCacheDisabled', { cacheDisabled: true });
    await cdp.send('Runtime.enable');
    await cdp.send('Emulation.setDeviceMetricsOverride', {
      width: viewport.width,
      height: viewport.height,
      deviceScaleFactor: 1,
      mobile: false,
    });
    await cdp.send('Page.navigate', { url: appUrl });
    await waitForHarness(cdp);

    const stats = [];
    const captureStart = Date.now();
    for (let i = 1; i <= frameCount; i += 1) {
      const targetTime = captureStart + (i * intervalMs);
      const waitMs = targetTime - Date.now();
      if (waitMs > 0) await delay(waitMs);
      const shot = await cdp.send('Page.captureScreenshot', {
        format: 'png',
        fromSurface: true,
        captureBeyondViewport: false,
      });
      const file = `${outputDir}\\avatar-pose-${String(i).padStart(2, '0')}.png`;
      await writeFile(file, Buffer.from(shot.data, 'base64'));
      stats.push({ frame: i, ...(await getStats(cdp)) });
      console.log(`captured ${file}`);
    }

    await writeFile(
      `${outputDir}\\avatar-pose-stats.json`,
      JSON.stringify({ appUrl, viewport, frameCount, intervalMs, stats }, null, 2),
    );
    await cdp.send('Browser.close').catch(() => {});
    cdp.ws.close();
    console.log(`done: ${outputDir}`);
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
