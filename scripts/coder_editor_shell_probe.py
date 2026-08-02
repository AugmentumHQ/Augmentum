from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path

from augmentum.tools.application_cdp import BrowserVerifier


async def _eval(bv: BrowserVerifier, expression: str, *, await_promise: bool = True):
    result = await bv._send(  # noqa: SLF001
        "Runtime.evaluate",
        {
            "expression": expression,
            "awaitPromise": await_promise,
            "returnByValue": True,
        },
        timeout=bv.page_timeout,
    )
    return result.get("result", {}).get("value")


async def _wait(bv: BrowserVerifier, expression: str, *, timeout: float = 30.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        value = await _eval(bv, expression)
        if value:
            return value
        await asyncio.sleep(0.2)
    raise TimeoutError(expression)


async def _navigate(bv: BrowserVerifier, url: str):
    await bv._send("Page.navigate", {"url": url})
    load = await bv._await_event("Page.loadEventFired", timeout=bv.page_timeout)  # noqa: SLF001
    if not load:
        raise RuntimeError(f"loadEventFired did not arrive for {url}")
    await asyncio.sleep(0.5)


async def _set_viewport(bv: BrowserVerifier, *, width: int, height: int, mobile: bool):
    await bv._send(
        "Emulation.setDeviceMetricsOverride",
        {
            "width": width,
            "height": height,
            "deviceScaleFactor": 1,
            "mobile": mobile,
            "screenWidth": width,
            "screenHeight": height,
        },
    )
    await bv._send(
        "Emulation.setTouchEmulationEnabled",
        {"enabled": mobile, "maxTouchPoints": 5 if mobile else 1},
    )
    await asyncio.sleep(0.45)


async def _login_if_needed(bv: BrowserVerifier, *, username: str, password: str):
    await _wait(
        bv,
        """
        !!document.querySelector('#login-submit') ||
        !!document.querySelector('.app')
        """,
        timeout=30.0,
    )
    if not await _eval(bv, "!!document.querySelector('#login-submit')"):
        return
    await _eval(
        bv,
        f"""
        (() => {{
          const user = document.querySelector('#login-username');
          const pass = document.querySelector('#login-password');
          const btn = document.querySelector('#login-submit');
          if (!user || !pass || !btn) return false;
          user.value = {json.dumps(username)};
          pass.value = {json.dumps(password)};
          user.dispatchEvent(new Event('input', {{ bubbles: true }}));
          pass.dispatchEvent(new Event('input', {{ bubbles: true }}));
          btn.click();
          return true;
        }})()
        """,
    )
    await _wait(bv, "!!document.querySelector('.app')", timeout=30.0)


async def _inject_shell(bv: BrowserVerifier):
    await _eval(
        bv,
        """
        (() => {
          document.querySelector('#viewport-probe-editor')?.remove();

          const split = document.createElement('div');
          split.id = 'viewport-probe-editor';
          split.className = 'coder-editor-split visible';
          split.style.position = 'fixed';
          split.style.top = '48px';
          split.style.right = '0';
          split.style.bottom = '0';
          split.style.zIndex = '2000';

          const close = document.createElement('button');
          close.className = 'coder-editor-close-mobile';
          close.type = 'button';
          close.setAttribute('aria-label', 'Close editor');
          close.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 6l12 12M18 6L6 18"></path></svg>';

          const tabs = document.createElement('div');
          tabs.className = 'coder-editor-tabs';
          tabs.innerHTML = `
            <button class="coder-tab active" type="button">
              viewport_probe.py
              <span class="coder-tab-close" title="Close">×</span>
            </button>
            <button class="coder-tab" type="button">
              viewport_notes.md
              <span class="coder-tab-close" title="Close">×</span>
            </button>
          `;

          const pane = document.createElement('div');
          pane.className = 'coder-editor-pane';

          const header = document.createElement('div');
          header.className = 'coder-editor-header';
          header.innerHTML = `
            <div class="coder-breadcrumb">
              <span class="coder-breadcrumb-part">workspace</span>
              <span class="coder-breadcrumb-sep">/</span>
              <span class="coder-breadcrumb-active">viewport_probe.py</span>
            </div>
            <button class="coder-save-btn" type="button">Save</button>
          `;

          const body = document.createElement('div');
          body.className = 'coder-editor-body';
          body.innerHTML = `
            <div class="cm-editor">
              <div style="
                height: 100%;
                overflow: auto;
                padding: 18px 20px 28px;
                background: linear-gradient(180deg, color-mix(in srgb, var(--bg-elevated) 55%, var(--bg)) 0%, var(--bg) 14%);
                color: var(--text-primary);
                font-family: var(--font-mono);
                font-size: 13px;
                line-height: 1.7;
                white-space: pre;
              ">def add(a: int, b: int) -> int:
    return a + b


def render_summary() -> str:
    return "Responsive editor probe"


if __name__ == "__main__":
    print(add(2, 3))
    print(render_summary())

# Additional lines to verify bottom breathing room and scroll behavior.
# The final line should remain visible without clipping in the panel.
              </div>
            </div>
          `;

          pane.appendChild(header);
          pane.appendChild(body);
          split.appendChild(close);
          split.appendChild(tabs);
          split.appendChild(pane);
          document.body.appendChild(split);
          return true;
        })()
        """,
    )
    await _wait(
        bv,
        "document.querySelector('#viewport-probe-editor')?.classList.contains('visible') && !!document.querySelector('#viewport-probe-editor .coder-editor-body .cm-editor')",
        timeout=15.0,
    )


async def _capture(bv: BrowserVerifier, *, width: int, height: int, mobile: bool, label: str, output_dir: Path):
    metrics = await _eval(
        bv,
        """
        (() => {
          const split = document.querySelector('#viewport-probe-editor');
          const tabs = split?.querySelector('.coder-editor-tabs');
          const header = split?.querySelector('.coder-editor-header');
          const body = split?.querySelector('.coder-editor-body');
          const editor = split?.querySelector('.cm-editor');
          const closeBtn = split?.querySelector('.coder-editor-close-mobile');
          const box = (node) => {
            if (!node) return null;
            const r = node.getBoundingClientRect();
            return {
              x: Math.round(r.x),
              y: Math.round(r.y),
              width: Math.round(r.width),
              height: Math.round(r.height),
              top: Math.round(r.top),
              right: Math.round(r.right),
              bottom: Math.round(r.bottom),
              left: Math.round(r.left),
            };
          };
          return {
            viewport: { innerWidth: window.innerWidth, innerHeight: window.innerHeight },
            splitVisible: !!split && split.classList.contains('visible'),
            split: box(split),
            tabs: box(tabs),
            header: box(header),
            body: box(body),
            editor: box(editor),
            close: box(closeBtn),
            cutoffDelta: split && editor ? Math.round(split.getBoundingClientRect().bottom - editor.getBoundingClientRect().bottom) : null,
          };
        })()
        """,
    )
    shot = await bv._send("Page.captureScreenshot", {"format": "png"})
    image_path = output_dir / f"{label}.png"
    image_path.write_bytes(base64.b64decode(shot["data"]))
    return {"label": label, "width": width, "height": height, "mobile": mobile, "screenshot": str(image_path), "metrics": metrics}


async def amain():
    parser = argparse.ArgumentParser(description="Capture real coder editor shell screenshots across viewports.")
    parser.add_argument("--ui-url", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--chromium-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    async with BrowserVerifier(port=9228, connect_timeout=30.0, chromium_path=args.chromium_path) as bv:
        await _navigate(bv, args.ui_url)
        await _login_if_needed(bv, username=args.username, password=args.password)
        await _set_viewport(bv, width=1366, height=900, mobile=False)
        await _inject_shell(bv)
        results.append(await _capture(bv, width=1366, height=900, mobile=False, label="desktop", output_dir=output_dir))
        await _set_viewport(bv, width=1024, height=1366, mobile=False)
        await _inject_shell(bv)
        results.append(await _capture(bv, width=1024, height=1366, mobile=False, label="tablet", output_dir=output_dir))
        await _set_viewport(bv, width=390, height=844, mobile=True)
        await _inject_shell(bv)
        results.append(await _capture(bv, width=390, height=844, mobile=True, label="mobile", output_dir=output_dir))

    report = {"results": results}
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(amain())
