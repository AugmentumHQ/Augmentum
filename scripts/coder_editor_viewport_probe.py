from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import httpx

from augmentum.tools.application_cdp import BrowserVerifier


PROBE_FILE = "/workspace/viewport_probe.py"
NOTES_FILE = "/workspace/viewport_notes.md"
PROBE_CONTENT = """def add(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    print(add(2, 3))
"""

NOTES_CONTENT = """# Viewport Probe

This workspace exists only to open the coder editor for responsive checks.
"""


async def _login_and_prepare_workspace(
    base_url: str,
    username: str,
    password: str,
    workspace_name: str,
) -> str:
    async with httpx.AsyncClient(base_url=base_url, follow_redirects=True, timeout=30.0) as client:
        resp = await client.post("/api/auth/login", json={"username": username, "password": password})
        resp.raise_for_status()

        workspaces_resp = await client.get("/api/coder/workspaces")
        workspaces_resp.raise_for_status()
        workspaces = workspaces_resp.json().get("workspaces", [])
        workspace = next((w for w in workspaces if w.get("name") == workspace_name), None)

        if workspace is None:
            create_resp = await client.post(
                "/api/coder/workspaces",
                json={"name": workspace_name, "publish_ports": False},
            )
            create_resp.raise_for_status()
            workspace = create_resp.json()

        workspace_id = workspace["id"]

        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            ready_resp = await client.get(f"/api/coder/workspaces/{workspace_id}/ready")
            ready_resp.raise_for_status()
            if ready_resp.json().get("ready"):
                break
            await asyncio.sleep(1.0)
        else:
            raise RuntimeError(f"workspace {workspace_id} never became ready")

        for path, content in (
            (PROBE_FILE, PROBE_CONTENT),
            (NOTES_FILE, NOTES_CONTENT),
        ):
            write_resp = await client.put(
                f"/api/coder/files/{workspace_id}/write",
                json={"path": path, "content": content, "checkpoint": False},
            )
            write_resp.raise_for_status()

        return workspace_id


async def _send_eval(bv: BrowserVerifier, expression: str, *, await_promise: bool = True):
    result = await bv._send(  # noqa: SLF001 - intentional probe against local CDP wrapper
        "Runtime.evaluate",
        {
            "expression": expression,
            "awaitPromise": await_promise,
            "returnByValue": True,
        },
        timeout=bv.page_timeout,
    )
    return result.get("result", {}).get("value")


async def _wait_for_expr(bv: BrowserVerifier, expression: str, *, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = await _send_eval(bv, expression)
        if value:
            return value
        await asyncio.sleep(0.2)
    raise TimeoutError(f"expression did not become truthy: {expression}")


async def _navigate(bv: BrowserVerifier, url: str):
    await bv._send("Page.navigate", {"url": url})
    load_fired = await bv._await_event("Page.loadEventFired", timeout=bv.page_timeout)  # noqa: SLF001
    if not load_fired:
        raise RuntimeError(f"page never fired loadEventFired for {url}")
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
        {
            "enabled": mobile,
            "maxTouchPoints": 5 if mobile else 1,
        },
    )
    await asyncio.sleep(0.45)


async def _capture_state(
    bv: BrowserVerifier,
    *,
    label: str,
    width: int,
    height: int,
    mobile: bool,
    output_dir: Path,
):
    await _set_viewport(bv, width=width, height=height, mobile=mobile)
    metrics = await _send_eval(
        bv,
        """
        (() => {
          const split = document.querySelector('#coder-editor-split');
          const tabs = document.querySelector('#coder-editor-tabs');
          const header = document.querySelector('.coder-editor-header');
          const body = document.querySelector('.coder-editor-body');
          const editor = document.querySelector('.coder-editor-body .cm-editor');
          const closeBtn = document.querySelector('#coder-editor-close-mobile');
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
            viewport: {
              innerWidth: window.innerWidth,
              innerHeight: window.innerHeight,
            },
            splitVisible: !!split && split.classList.contains('visible'),
            split: box(split),
            tabs: box(tabs),
            header: box(header),
            body: box(body),
            editor: box(editor),
            close: box(closeBtn),
            bodyScrollHeight: body ? body.scrollHeight : null,
            editorScrollHeight: editor ? editor.scrollHeight : null,
            cutoffDelta: split && editor ? Math.round(split.getBoundingClientRect().bottom - editor.getBoundingClientRect().bottom) : null,
          };
        })()
        """,
    )
    shot = await bv._send("Page.captureScreenshot", {"format": "png"})
    image_bytes = base64.b64decode(shot["data"])
    image_path = output_dir / f"{label}.png"
    image_path.write_bytes(image_bytes)
    return {"label": label, "mobile": mobile, "width": width, "height": height, "screenshot": str(image_path), "metrics": metrics}


async def _drive_ui(
    bv: BrowserVerifier,
    *,
    ui_url: str,
    username: str,
    password: str,
    workspace_id: str | None,
):
    await _navigate(bv, ui_url)
    await _wait_for_expr(
        bv,
        """
        !!document.querySelector('#login-submit') ||
        !!document.querySelector('.panel-mode-option[data-mode="coder"]')
        """,
        timeout=30.0,
    )
    has_login = await _send_eval(bv, "!!document.querySelector('#login-submit')")
    if has_login:
        await _send_eval(
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
        await _wait_for_expr(bv, "!!document.querySelector('.panel-mode-option[data-mode=\"coder\"]')", timeout=30.0)
    await _send_eval(
        bv,
        """
        (() => {
          const btn = document.querySelector('.panel-mode-option[data-mode="coder"]');
          if (!btn) return false;
          btn.click();
          return true;
        })()
        """,
    )
    await _wait_for_expr(
        bv,
        "!!document.querySelector('#coder-editor-split') && !!document.querySelector('#coder-editor-pane') && !!document.querySelector('#coder-editor-tabs')",
        timeout=30.0,
    )
    await _send_eval(
        bv,
        """
        (() => {
          document.querySelector('.app')?.setAttribute('data-mode', 'coder');
          document.querySelector('#coder-layout')?.classList.remove('hidden');
          document.querySelector('#coder-editor-split')?.classList.remove('hidden');

          const split = document.querySelector('#coder-editor-split');
          const tabs = document.querySelector('#coder-editor-tabs');
          const pane = document.querySelector('#coder-editor-pane');
          if (!split || !tabs || !pane) return false;

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

          pane.innerHTML = '';
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
          split.classList.add('visible');
          return true;
        })()
        """,
    )
    await _wait_for_expr(
        bv,
        "document.querySelector('#coder-editor-split')?.classList.contains('visible') && !!document.querySelector('.coder-editor-body .cm-editor')",
        timeout=30.0,
    )
    await asyncio.sleep(0.5)


async def amain():
    parser = argparse.ArgumentParser(description="Capture real coder editor screenshots across viewports.")
    parser.add_argument("--base-url", default="http://127.0.0.1:6100")
    parser.add_argument("--ui-url", default=None)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--chromium-path", default=None)
    parser.add_argument("--workspace-name", default="viewport-probe")
    parser.add_argument("--workspace-id", default=None)
    parser.add_argument(
        "--output-dir",
        default="artifacts/ui-probes/coder-editor",
        help="Path relative to repo root inside the container",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ui_url = args.ui_url or f"{args.base_url.rstrip('/')}/ui/index.html"

    workspace_id = args.workspace_id
    if not workspace_id:
        workspace_id = await _login_and_prepare_workspace(
            args.base_url,
            args.username,
            args.password,
            args.workspace_name,
        )

    results: list[dict] = []
    async with BrowserVerifier(port=9228, connect_timeout=30.0, chromium_path=args.chromium_path) as bv:
        await _set_viewport(bv, width=1366, height=900, mobile=False)
        await _drive_ui(
            bv,
            ui_url=ui_url,
            username=args.username,
            password=args.password,
            workspace_id=workspace_id,
        )
        results.append(await _capture_state(bv, label="desktop", width=1366, height=900, mobile=False, output_dir=output_dir))
        results.append(await _capture_state(bv, label="tablet", width=1024, height=1366, mobile=False, output_dir=output_dir))
        results.append(await _capture_state(bv, label="mobile", width=390, height=844, mobile=True, output_dir=output_dir))

    report = {
        "workspace_id": workspace_id,
        "probe_file": PROBE_FILE,
        "results": results,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(amain())
