"""Regression guard: every headless-Chromium launch we drive for capture
must carry the WebGL-safe flag set.

The symptom this fixes: a Vite / Three.js / WebGL preview timed out on
`page.screenshot()` in the coder loop because the workspace Chromium was
launched with no GPU flags. In a GPU-less container, modern Chromium gates
software WebGL behind `--enable-unsafe-swiftshader` and blocklists the
software GPU, so the canvas layer never composites a captureable frame.

`augmentum/utils/chromium.py::HEADLESS_WEBGL_ARGS` is the single source of
truth. These tests assert it is spliced into every in-package launcher —
so adding a new browser tool without the flags, or dropping a flag, fails
here instead of silently timing out on the next WebGL page.
"""
from __future__ import annotations

import shlex

import pytest

from augmentum.utils.chromium import HEADLESS_WEBGL_ARGS, headless_webgl_args

# The flags that actually matter for WebGL in a GPU-less container.
_CRITICAL = (
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
    "--enable-webgl",
)


def test_constant_has_critical_webgl_flags_and_no_disable_gpu():
    for flag in _CRITICAL:
        assert flag in HEADLESS_WEBGL_ARGS, f"missing {flag}"
    # --disable-gpu kills the GPU process and defeats software WebGL capture.
    assert "--disable-gpu" not in HEADLESS_WEBGL_ARGS
    # Container hygiene flags ride along so every launcher gets them too.
    assert "--no-sandbox" in HEADLESS_WEBGL_ARGS
    assert "--disable-dev-shm-usage" in HEADLESS_WEBGL_ARGS


def test_headless_webgl_args_appends_extras():
    got = headless_webgl_args("--foo", "--bar")
    assert got[: len(HEADLESS_WEBGL_ARGS)] == list(HEADLESS_WEBGL_ARGS)
    assert got[-2:] == ["--foo", "--bar"]


class _StubCM:
    """Container manager stub that records the python3 scripts the coder
    browser helpers hand to `run_command`, and looks like a successful
    Playwright run so no HTTP fallback fires."""

    def __init__(self):
        self.scripts: list[str] = []

    async def run_command(self, ws, cmd, timeout=None):
        joined = cmd[2] if len(cmd) > 2 else ""
        if joined.startswith("python3 -c"):
            self.scripts.append(shlex.split(joined)[2])
        return '{"ok": true, "playwright": true, "path": "/x.png"}'

    async def file_write(self, *a, **k):
        return None

    async def file_read(self, *a, **k):
        return "{}"

    async def list_ports(self, *a, **k):
        return []


@pytest.mark.asyncio
async def test_every_coder_browser_launch_carries_webgl_args():
    from augmentum.coder import browser as B

    cm = _StubCM()
    await B.playwright_action(cm, "ws", url="http://x:5173", action="open")
    await B.playwright_screenshot(cm, "ws", url="http://x:5173")
    await B.playwright_evaluate(cm, "ws", url="http://x:5173", expression="document.title")
    await B.playwright_wait(cm, "ws", url="http://x:5173")
    await B.playwright_fill_form(cm, "ws", url="http://x:5173", fields={"#a": "b"})

    assert len(cm.scripts) == 5, f"expected 5 launch scripts, got {len(cm.scripts)}"
    for i, script in enumerate(cm.scripts):
        # Must be runnable Python (the args are injected into a string).
        compile(script, f"<coder-browser-{i}>", "exec")
        assert "args=chrome_args" in script, f"script {i} launches without args=chrome_args"
        assert "chrome_args=" in script, f"script {i} never defines chrome_args"
        for flag in _CRITICAL:
            assert flag in script, f"script {i} missing {flag}"


def test_screenshot_budgets_bounded_and_goto_capped():
    """Every internal budget must stay bounded so the whole thing degrades
    gracefully instead of blowing the subprocess wall and hard-timing-out with
    NO result (the cranked-timeout_ms failure): goto is capped independent of
    timeout_ms, the screenshot budget is capped at 35s, and the subprocess wall
    stays under the tool's 100s timeout so a timeout yields a structured result."""
    import inspect

    from augmentum.coder import browser as B

    src = inspect.getsource(B.playwright_screenshot)
    assert "goto_timeout_ms = min(timeout_ms, 20_000)" in src, "goto not capped -> can blow the wall"
    assert "screenshot_timeout_ms = min(35_000, max(25_000, timeout_ms))" in src
    assert "fallback_timeout_ms = 12_000" in src
    assert "92.0," in src, "subprocess wall must stay under the tool's 100s timeout"
    # a heavy-WebGL timeout must hand back an actionable message, not a bare
    # Playwright TimeoutError, and point at the live GPU preview.
    assert "live GPU preview" in src


def test_interaction_gotos_use_load_not_networkidle():
    """browser_evaluate / _click / _fill_form must not goto(wait_until=
    'networkidle') — a live Vite/HMR dev server never goes idle, which hung
    those tools to timeout. 'load' terminates reliably."""
    import inspect

    from augmentum.coder import browser as B

    for fn in (B.playwright_evaluate, B.playwright_action, B.playwright_fill_form):
        src = inspect.getsource(fn)
        # the initial navigation must not block on network idle
        assert "page.goto(url, wait_until='networkidle'" not in src, (
            f"{fn.__name__} still navigates with wait_until='networkidle'"
        )
        assert "wait_until='load'" in src, f"{fn.__name__} should navigate with wait_until='load'"


def test_build_verify_gate_script_substitutes_webgl_args():
    from augmentum.builds import verify as V

    gate = V._GATE_SCRIPT.replace("__ASSERTIONS_PATH__", "/tmp/a.json").replace(
        "__CHROME_ARGS__", repr(list(HEADLESS_WEBGL_ARGS))
    )
    compile(gate, "<gate>", "exec")
    assert "__CHROME_ARGS__" not in gate
    assert "--enable-unsafe-swiftshader" in gate


def test_game_probe_script_substitutes_webgl_args():
    from augmentum.cast.games.probe import playwright_probe as P

    probe = P._PROBE_SCRIPT.replace("__CHROME_ARGS__", repr(list(HEADLESS_WEBGL_ARGS)))
    compile(probe, "<probe>", "exec")
    assert "__CHROME_ARGS__" not in probe
    assert "--enable-unsafe-swiftshader" in probe


def test_host_cdp_launchers_use_shared_constant_not_disable_gpu():
    """application_cdp + html_renderer build their flag list in start(); assert
    the module source splices the shared constant and dropped --disable-gpu."""
    import inspect

    from augmentum.cast import html_renderer
    from augmentum.tools import application_cdp

    for mod in (application_cdp, html_renderer):
        src = inspect.getsource(mod)
        assert "*HEADLESS_WEBGL_ARGS" in src, f"{mod.__name__} does not splice HEADLESS_WEBGL_ARGS"
        # The old hardcoded --disable-gpu line must be gone from the flags block.
        assert '"--disable-gpu"' not in src, f"{mod.__name__} still hardcodes --disable-gpu"
