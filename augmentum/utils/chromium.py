"""Shared headless-Chromium launch flags — one source of truth.

Every place we drive a headless Chromium for verification or capture — the
coder browser tools, the game screenshot probe, the app-builder CDP, the cast
HTML renderer, the XR panel pool, and the standalone driver scripts — needs
the same GPU/WebGL launch profile. Historically each site hand-rolled its own
list (or none), so a Vite/Three.js/WebGL page would render on one surface and
time out on another.

The failure mode this fixes: in a GPU-less container, modern Chromium (M110+)
disables the SwiftShader software-GL backend for WebGL *unless* explicitly
allowed, and blocklists the software GPU for WebGL by default. Without the
flags below, a WebGL canvas layer never produces a captureable composited
frame, so ``page.screenshot()`` blocks until its timeout. With them, WebGL
initializes on ANGLE-over-SwiftShader and the frame is captureable (slower
than hardware, but it renders).

This exact set is the one proven against real EmulatorJS / Three.js WebGL
pages by the standalone ``scripts/drive_*`` / ``probe_verify_map`` drivers.
Those scripts keep their own inline copy on purpose — they shell out to a
running stack via ``docker exec`` and must stay importable without the
``augmentum`` package on ``sys.path`` — so this module is the source of truth
for every *in-package* launcher (coder browser, app-builder CDP, cast HTML
renderer, XR panel, build-verify gate, game probe). Keep the two in sync if
the flag set ever changes.

Notes:
- Deliberately **no** ``--disable-gpu``. That switch kills the GPU process
  entirely, which defeats forcing ANGLE→SwiftShader and leaves WebGL without
  a compositor to capture. Software rasterization of ordinary HTML still works
  fine with the GPU process backed by SwiftShader.
- Chromium ignores command-line switches it doesn't recognize, so the newer
  ``--enable-unsafe-swiftshader`` gate is harmless on older binaries.
- Playwright's ``launch(headless=True)`` manages the headless switch itself, so
  these args omit ``--headless``; raw ``subprocess.Popen`` launchers that talk
  CDP directly must prepend ``--headless=new`` themselves.
"""

from __future__ import annotations

# WebGL-safe headless Chromium flags. Order mirrors the proven driver scripts.
HEADLESS_WEBGL_ARGS: tuple[str, ...] = (
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
    "--enable-webgl",
    "--no-sandbox",
    "--disable-dev-shm-usage",
)


def headless_webgl_args(*extra: str) -> list[str]:
    """Return the WebGL-safe headless Chromium arg list, plus any ``extra``.

    Use for Playwright ``launch(headless=True, args=headless_webgl_args())``.
    Raw-subprocess CDP launchers that manage their own headless switch should
    prepend ``--headless=new`` before splatting these.
    """
    return [*HEADLESS_WEBGL_ARGS, *extra]
