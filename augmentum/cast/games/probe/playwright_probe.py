"""Headless probe driver — load a game, fingerprint its input, confirm it
reacts, and hand back a classified :class:`CastProfile`.

Runs a self-contained ``sync_playwright`` script as a subprocess (mirrors
``builds/verify.py``'s in-container gate). The subprocess:

  1. installs the :data:`fingerprint.INSTRUMENTATION_JS` trap at
     document-start,
  2. navigates to the embed URL + lets it boot,
  3. reads back the observed input wiring,
  4. screenshots before/after firing synthetic input,

and prints a JSON blob. THIS module then decides ``responded`` via
:mod:`pixel_diff` (one source of truth) and maps the observed events via
:mod:`fingerprint` — so the decision logic stays pure + tested while the
browser run is the only un-unit-testable part.

Graceful degradation is the contract: if Playwright / Chromium isn't
available (the augmentum image may not ship it), or the URL is unsafe, or
the run errors, :meth:`PlaywrightProbe.probe` returns ``None`` and the
caller falls back to the static classifier + telemetry demotion. The
probe is strictly additive — it can make the FIRST cast right, never
break a cast.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from augmentum.cast.games.models import (
    CLASSIFIED_PROBE,
    STRATEGY_SHIM,
    CastProfile,
)
from augmentum.cast.games.probe.fingerprint import (
    INSTRUMENTATION_JS,
    classify_input_style,
)
from augmentum.cast.games.probe.pixel_diff import DEFAULT_DIFF_THRESHOLD, responded
from augmentum.cast.games.proxy.fetcher import is_url_safe
from augmentum.utils.chromium import HEADLESS_WEBGL_ARGS
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


DEFAULT_PROBE_TIMEOUT_S = 45.0
# Keys we fire to try to provoke a reaction — covers d-pad games, action
# games (space/enter), and WASD movers. Best-effort; a game that ignores
# all of them simply reads as "no response" (we still keep the fingerprint).
_DEFAULT_KEYS = (
    "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
    " ", "Enter", "w", "a", "s", "d", "z", "x",
)


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of one probe run (before it's turned into a profile)."""

    ok: bool
    observed: tuple[str, ...] = ()
    responded: bool = False
    error: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


# The subprocess script. Self-contained (no augmentum imports) so it runs
# under any interpreter that has playwright. Emits before/after PNGs as
# base64 for THIS process to diff via pixel_diff (one source of truth).
_PROBE_SCRIPT = r'''
import base64, json, sys
try:
    from playwright.sync_api import sync_playwright
except Exception as e:
    print(json.dumps({"ok": False, "error": "playwright unavailable: " + str(e)}))
    sys.exit(0)

cfg = json.load(open(sys.argv[1]))
url = cfg["url"]
instrumentation = cfg["instrumentation"]
keys = cfg.get("keys", [])
boot_ms = int(cfg.get("boot_ms", 1500))
react_ms = int(cfg.get("react_ms", 600))

out = {"ok": True, "observed": [], "before_b64": "", "after_b64": "", "error": ""}
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=__CHROME_ARGS__)
        page = browser.new_page(viewport={"width": 640, "height": 480})
        page.add_init_script(instrumentation)
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
        except Exception as e:
            out = {"ok": False, "error": "navigation failed: " + str(e)[:160]}
            browser.close()
            print(json.dumps(out)); sys.exit(0)
        page.wait_for_timeout(boot_ms)
        try:
            obs = page.evaluate("window.__augProbeObserved ? window.__augProbeObserved() : []")
            out["observed"] = list(obs or [])
        except Exception:
            out["observed"] = []
        try:
            out["before_b64"] = base64.b64encode(page.screenshot()).decode("ascii")
            for k in keys:
                try: page.keyboard.press(k)
                except Exception: pass
            try: page.mouse.click(320, 240)
            except Exception: pass
            page.wait_for_timeout(react_ms)
            out["after_b64"] = base64.b64encode(page.screenshot()).decode("ascii")
        except Exception as e:
            out["error"] = "capture failed: " + str(e)[:120]
        browser.close()
except Exception as e:
    out = {"ok": False, "error": "probe run failed: " + str(e)[:200]}
print(json.dumps(out))
'''


# A runner takes the config dict + timeout, returns the subprocess stdout
# (JSON string). Injectable so tests can stub the browser run entirely.
ScriptRunner = Callable[[dict[str, Any], float], Awaitable[str]]


class PlaywrightProbe:
    """Drives the headless probe + turns its output into a CastProfile."""

    def __init__(
        self,
        *,
        python_executable: str = sys.executable,
        timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
        diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
        runner: ScriptRunner | None = None,
    ) -> None:
        self._python = python_executable
        self._timeout = float(timeout_s)
        self._threshold = float(diff_threshold)
        # Default runner spawns the subprocess; tests inject a stub.
        self._runner = runner or self._subprocess_runner

    async def probe(self, embed_url: str) -> ProbeResult | None:
        """Run the probe. Returns None when not runnable (unsafe URL,
        Playwright unavailable, subprocess error) so the caller falls back
        to the static classifier."""
        if not embed_url or not is_url_safe(embed_url):
            log.info("cast_probe_skipped_unsafe_url", url=embed_url)
            return None

        cfg = {
            "url": embed_url,
            "instrumentation": INSTRUMENTATION_JS,
            "keys": list(_DEFAULT_KEYS),
            "boot_ms": 1500,
            "react_ms": 600,
        }
        try:
            raw = await self._runner(cfg, self._timeout)
        except Exception as exc:
            log.warning("cast_probe_runner_failed", url=embed_url, error=str(exc)[:160])
            return None
        return self._parse(raw)

    def _parse(self, raw: str) -> ProbeResult | None:
        try:
            obj = json.loads(raw)
        except (TypeError, ValueError):
            log.warning("cast_probe_unparseable_output", raw=(raw or "")[:200])
            return None
        if not isinstance(obj, dict) or not obj.get("ok"):
            reason = (obj.get("error") if isinstance(obj, dict) else "") or "probe not ok"
            log.info("cast_probe_not_ok", reason=str(reason)[:160])
            return None
        observed = tuple(str(e) for e in (obj.get("observed") or []))
        did_respond = responded(
            obj.get("before_b64") or "",
            obj.get("after_b64") or "",
            threshold=self._threshold,
        )
        return ProbeResult(
            ok=True,
            observed=observed,
            responded=did_respond,
            evidence={"observed_count": len(observed)},
        )

    async def _subprocess_runner(self, cfg: dict[str, Any], timeout_s: float) -> str:
        """Default runner — write the script + config to temp files, run
        the probe under our interpreter, return stdout. Cleans up after."""
        with tempfile.TemporaryDirectory(prefix="aug-cast-probe-") as tmp:
            script_path = Path(tmp) / "probe.py"
            cfg_path = Path(tmp) / "cfg.json"
            script_path.write_text(
                _PROBE_SCRIPT.replace("__CHROME_ARGS__", repr(list(HEADLESS_WEBGL_ARGS))),
                encoding="utf-8",
            )
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
            proc = await asyncio.create_subprocess_exec(
                self._python, str(script_path), str(cfg_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_s,
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                raise
            if stderr:
                log.debug("cast_probe_stderr", stderr=stderr.decode("utf-8", "replace")[:300])
            return (stdout or b"").decode("utf-8", "replace").strip()


def build_probe_profile(
    *,
    title_id: str,
    user_id: str,
    embed_url: str,
    result: ProbeResult,
    strategy: str = STRATEGY_SHIM,
    now: float | None = None,
) -> CastProfile:
    """Turn a successful ProbeResult into a persistable CastProfile.

    ``strategy`` is decided by the caller (origin comparison) — the probe
    only determines the input_chain. A probe that ran but saw the game NOT
    react still yields a profile (the fingerprint is useful); we record the
    non-response in ``notes`` so a later telemetry demotion / manual review
    has the context.
    """
    fp = classify_input_style(list(result.observed))
    note = "probe: input reached game" if result.responded else (
        "probe: fingerprinted but no visible reaction to synthetic input"
    )
    return CastProfile(
        title_id=title_id,
        user_id=user_id,
        strategy=strategy,
        embed_url=embed_url,
        input_chain=fp.input_chain,
        classified_by=CLASSIFIED_PROBE,
        classified_at=now if now is not None else time.time(),
        notes=f"{note}; styles={','.join(fp.styles) or 'none'}",
    )
