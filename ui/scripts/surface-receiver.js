const params = new URLSearchParams(window.location.search);
let token = params.get("token") || "";
const stage = document.getElementById("readerStage");
const titleEl = document.getElementById("surfaceTitle");
const metaEl = document.getElementById("surfaceMeta");

const participantId = (() => {
  const key = "augmentum_surface_receiver_id";
  const existing = window.localStorage.getItem(key);
  if (existing) return existing;
  const next = `display_${crypto.getRandomValues(new Uint32Array(2)).join("")}`;
  window.localStorage.setItem(key, next);
  return next;
})();

let session = null;
let revision = -1;
let pageCount = 0;
let renderedPageCount = 0;
let firstScroll = true;
let activeRun = 0;

function api(path) {
  return `/api/surface-public/${encodeURIComponent(token)}${path}`;
}

function setStatus(title, meta) {
  titleEl.textContent = title || "Surface";
  metaEl.textContent = meta || "";
}

function renderEmpty(message) {
  stage.innerHTML = "";
  const empty = document.createElement("div");
  empty.className = "reader-empty";
  empty.textContent = message;
  stage.append(empty);
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    cache: "no-store",
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function join() {
  await fetchJson(api("/join"), {
    method: "POST",
    body: JSON.stringify({
      participant_id: participantId,
      role: "display",
      label: navigator.userAgent.includes("Android") ? "Android TV Browser" : "Browser Display",
      capabilities: ["surface.follow_state@1", "display.comic_read@1"],
      transport: "public_long_poll",
    }),
  });
}

async function loadSession() {
  const data = await fetchJson(api("/session"));
  session = data.session;
  revision = Number(session?.revision ?? revision);
  return session;
}

async function loadManifest() {
  try {
    const manifest = await fetchJson(api("/comic/manifest"));
    pageCount = Number(manifest.page_count || pageCount || 0);
    const reader = session?.state?.reader || {};
    if (!reader.page_count && pageCount > 0) {
      session.state = {
        ...(session.state || {}),
        reader: { ...reader, page_count: pageCount },
      };
    }
  } catch {
    pageCount = Number(session?.state?.reader?.page_count || pageCount || 0);
  }
}

function ensureComicPages(count) {
  const bounded = Math.max(0, Math.min(500, Number(count || 0)));
  if (bounded <= 0) {
    renderEmpty("Waiting for pages");
    renderedPageCount = 0;
    return;
  }
  if (renderedPageCount === bounded) return;

  stage.innerHTML = "";
  const fragment = document.createDocumentFragment();
  for (let page = 1; page <= bounded; page += 1) {
    const frame = document.createElement("article");
    frame.className = "reader-page";
    frame.dataset.page = String(page);

    const img = document.createElement("img");
    img.alt = "";
    img.decoding = "async";
    img.loading = page <= 3 ? "eager" : "lazy";
    img.src = api(`/comic/page?page=${page}`);
    frame.append(img);
    fragment.append(frame);
  }
  stage.append(fragment);
  renderedPageCount = bounded;
}

function clamp(value, min, max) {
  const n = Number(value);
  if (!Number.isFinite(n)) return min;
  return Math.min(max, Math.max(min, n));
}

function applyReaderState() {
  if (!session) return;
  const reader = session.state?.reader || {};
  const count = Number(reader.page_count || pageCount || 0);
  ensureComicPages(count);
  if (!renderedPageCount) return;

  const page = Math.round(clamp(reader.page || 1, 1, renderedPageCount));
  const ratio = clamp(reader.scroll_ratio || 0, 0, 1);
  for (const el of stage.querySelectorAll(".reader-page")) {
    el.dataset.active = el.dataset.page === String(page) ? "true" : "false";
  }

  requestAnimationFrame(() => {
    const target = stage.querySelector(`.reader-page[data-page="${page}"]`);
    if (!target) return;
    const overflow = Math.max(0, target.offsetHeight - window.innerHeight);
    const y = target.offsetTop + overflow * ratio;
    window.scrollTo({
      top: Math.max(0, y),
      behavior: firstScroll ? "auto" : "smooth",
    });
    firstScroll = false;
  });
}

function renderSession() {
  if (!session) {
    renderEmpty("Waiting for session");
    return;
  }
  const title = session.title || session.content_ref?.title || "Augmentum Surface";
  const reader = session.state?.reader || {};
  const page = Number(reader.page || 1);
  const count = Number(reader.page_count || pageCount || 0);
  setStatus(title, count ? `Page ${page} / ${count}` : `Revision ${session.revision}`);
  if (session.kind === "comic.reader.webtoon" || session.content_ref?.kind === "comic") {
    applyReaderState();
  } else {
    renderEmpty("Surface connected");
  }
}

async function poll() {
  const run = activeRun;
  while (token && run === activeRun) {
    try {
      const data = await fetchJson(api(`/events?after_revision=${revision}&timeout_ms=25000`));
      if (data.session) {
        session = data.session;
        revision = Number(session.revision ?? revision);
        renderSession();
      }
    } catch (error) {
      setStatus(session?.title || "Surface", "Reconnecting");
      await new Promise((resolve) => setTimeout(resolve, 1800));
    }
  }
}

function _resetForToken(nextToken) {
  token = String(nextToken || "").trim();
  activeRun += 1;
  session = null;
  revision = -1;
  pageCount = 0;
  renderedPageCount = 0;
  firstScroll = true;
}

function _tokenFromHandoff(handoff) {
  const payload = handoff?.ble_payload || handoff?.handoff?.ble_payload || handoff || {};
  const access = handoff?.ip || handoff?.handoff?.ip || {};
  return String(payload.token || access.token || "").trim();
}

export async function startWithSurfaceToken(nextToken, { source = "receiver" } = {}) {
  _resetForToken(nextToken);
  if (!token) {
    setStatus("Surface", source === "cast" ? "Waiting for Cast" : "Missing token");
    renderEmpty(source === "cast" ? "Waiting for Cast handoff" : "Missing surface token");
    return;
  }
  try {
    setStatus("Surface", "Pairing");
    await join();
    await loadSession();
    await loadManifest();
    renderSession();
    poll();
  } catch (error) {
    setStatus("Surface", "Unavailable");
    renderEmpty("Surface unavailable");
  }
}

export async function startWithSurfaceHandoff(handoff, { source = "cast" } = {}) {
  const nextToken = _tokenFromHandoff(handoff);
  if (!nextToken) {
    setStatus("Surface", "Missing token");
    renderEmpty("Surface handoff did not include a receiver token");
    return;
  }
  try {
    const url = new URL(window.location.href);
    url.searchParams.set("token", nextToken);
    window.history.replaceState(null, "", url.toString());
  } catch { /* malformed URL or history blocked — token swap still proceeds below */ }
  await startWithSurfaceToken(nextToken, { source });
}

export function getSurfaceReceiverState() {
  return {
    token,
    session,
    revision,
    pageCount,
    renderedPageCount,
  };
}

window.AugmentumSurfaceReceiver = {
  getState: getSurfaceReceiverState,
  startWithHandoff: startWithSurfaceHandoff,
  startWithToken: startWithSurfaceToken,
};

startWithSurfaceToken(token, {
  source: window.AUGMENTUM_SURFACE_RECEIVER_SOURCE || params.get("transport") || "receiver",
});
