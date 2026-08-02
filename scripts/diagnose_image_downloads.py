#!/usr/bin/env python
"""Detached diagnostic for the image-model download pipeline.

Verifies the *whole* download path end-to-end — detect → pull → register —
across the different on-disk model structures we support, and asserts the
things that have actually broken in the field:

  * detect returns variants with **non-zero sizes** (the "Auto — 0 KB" bug:
    a missing ``files_metadata=True`` made every variant report 0 bytes).
  * the download streams **intra-file byte progress** instead of sitting at
    0% and then lurching to 100% (the Xet single-file progress bug — HF's
    Xet backend stages bytes in bursts the disk-size monitor couldn't see).
  * the finished directory is a **valid, correctly-classified** model
    (single-file checkpoint vs diffusers component layout).

This is intentionally NOT a pytest — it does real network downloads and is
meant to be run by hand against a live install when a download "doesn't
work":

    docker exec -w /app augmentum-augmentum-1 \
        python scripts/diagnose_image_downloads.py

Flags:
    --case NAME       run only the named case(s) (repeatable)
    --keep            don't delete downloaded models afterwards
    --list            list available cases and exit

Cases marked ``needs_key`` are skipped automatically unless the relevant
API key is configured (CivitAI), so the default run is safe and offline-ish.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import time
from dataclasses import dataclass, field

# Allow running from the repo root or anywhere on PATH.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmentum.config import settings  # noqa: E402
from augmentum.image.model_manager import ModelManager  # noqa: E402
from augmentum.proxy import image_routes as IR  # noqa: E402


@dataclass
class Case:
    name: str
    source: str
    source_type: str               # "huggingface" | "civitai"
    expect_structure: str          # "single_file" | "diffusers" | "any"
    expect_size_nonzero: bool = True
    needs_key: str = ""            # "" | "civitai"
    max_seconds: int = 600
    notes: str = ""


CASES: list[Case] = [
    Case(
        name="hf-diffusers-tiny",
        source="hf-internal-testing/tiny-sd-pipe",
        source_type="huggingface",
        expect_structure="diffusers",
        notes="Multi-file diffusers layout (model_index.json + components), ~8MB. Fast.",
        max_seconds=120,
    ),
    Case(
        name="hf-single-file",
        source="ShoukanLabs/OpenNiji-V2",
        source_type="huggingface",
        expect_structure="single_file",
        notes="Single .safetensors checkpoint, no diffusers config (~2GB). Xet-backed.",
        max_seconds=600,
    ),
    Case(
        name="civitai",
        source="",  # fill in a model/version URL to exercise; needs API key for gated
        source_type="civitai",
        expect_structure="single_file",
        needs_key="civitai",
        notes="CivitAI streams per-chunk progress already; mainly checks auth + structure.",
    ),
]


@dataclass
class Result:
    name: str
    ok: bool = False
    skipped: str = ""
    detect_variants: int = 0
    detect_min_size_gb: float = 0.0
    progress_samples: list = field(default_factory=list)
    intermediate_progress: bool = False
    final_status: str = ""
    structure: str = ""
    valid_dir: bool = False
    failures: list = field(default_factory=list)
    seconds: float = 0.0


def _dest_for(mgr: ModelManager, case: Case) -> str:
    if case.source_type == "huggingface":
        repo = case.source
        if "huggingface.co" in repo:
            repo = repo.split("huggingface.co/")[-1].strip("/")
        return os.path.join(mgr._model_dir, repo.replace("/", "--"))
    # civitai dest name is derived inside pull; best-effort cleanup by prefix
    return ""


async def _detect(case: Case) -> dict:
    if case.source_type == "huggingface":
        return await IR._detect_huggingface(case.source, _FakeRequest())
    return await IR._detect_civitai(case.source, _FakeRequest())


class _FakeRequest:
    """Minimal stand-in: _detect_* only touch request.app.state."""

    class _App:
        class state:  # noqa: N801
            image_model_manager = None

    app = _App()


async def run_case(case: Case, keep: bool) -> Result:
    res = Result(name=case.name)
    t0 = time.time()

    if case.needs_key == "civitai" and not settings.image_civitai_api_key:
        res.skipped = "no CivitAI API key configured"
        return res
    if not case.source:
        res.skipped = "no source configured for this case"
        return res

    mgr = ModelManager(settings.image_model_dir or f"{settings.data_dir}/image_models")

    # --- 1. detect ---
    try:
        det = await _detect(case)
    except Exception as exc:  # noqa: BLE001
        res.failures.append(f"detect raised: {exc!r}")
        res.seconds = time.time() - t0
        return res
    if det.get("error"):
        res.failures.append(f"detect error: {det['error']}")
        res.seconds = time.time() - t0
        return res

    variants = det.get("variants") or []
    res.detect_variants = len(variants)
    sizes = [v.get("size_gb", 0) for v in variants]
    res.detect_min_size_gb = min(sizes) if sizes else 0.0
    if not variants:
        res.failures.append("detect returned no variants")
    if case.expect_size_nonzero and variants and max(sizes) <= 0:
        res.failures.append("detect reported 0-byte sizes for all variants (files_metadata bug?)")

    # --- 2. download via the REAL orchestration (_run_pull_task) ---
    dest = _dest_for(mgr, case)
    if dest and os.path.exists(dest):
        shutil.rmtree(dest, ignore_errors=True)

    tid = f"diag-{case.name}"
    IR._pull_tasks[tid] = {
        "status": "running", "source": case.source, "progress": {},
        "last_event": {}, "result": None, "error": None,
    }
    task = asyncio.create_task(IR._run_pull_task(
        tid, mgr, None, case.source, "", None, "", asset_type="", ctx=None,
    ))

    deadline = t0 + case.max_seconds
    while not task.done():
        await asyncio.sleep(2)
        pct = IR._pull_tasks[tid].get("progress", {}).get("percent")
        res.progress_samples.append(pct)
        if time.time() > deadline:
            task.cancel()
            res.failures.append(f"download exceeded {case.max_seconds}s")
            break
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as exc:  # noqa: BLE001
        res.failures.append(f"download raised: {exc!r}")

    res.final_status = IR._pull_tasks[tid].get("status", "?")

    # Did progress move through an intermediate value (not just 0 then 100)?
    mids = [p for p in res.progress_samples if isinstance(p, (int, float)) and 1.0 <= p <= 95.0]
    res.intermediate_progress = len(mids) >= 1

    # --- 3. structure + validity ---
    if dest and os.path.isdir(dest):
        res.valid_dir = mgr._is_valid_model_dir(dest)
        from augmentum.image.pipeline_v2 import _find_single_safetensors
        single = _find_single_safetensors(dest)
        has_index = os.path.exists(os.path.join(dest, "model_index.json"))
        res.structure = "single_file" if single else ("diffusers" if has_index else "other")

    # --- assertions ---
    if res.final_status not in ("complete", "exists"):
        res.failures.append(f"final status was '{res.final_status}', expected complete/exists")
    if res.final_status == "complete" and not res.intermediate_progress:
        res.failures.append(
            "progress never showed an intermediate value (stuck-at-0%-then-100% regression)"
        )
    if dest and not res.valid_dir:
        res.failures.append("downloaded dir is not a valid model dir")
    if case.expect_structure != "any" and res.structure and res.structure != case.expect_structure:
        res.failures.append(
            f"structure was '{res.structure}', expected '{case.expect_structure}'"
        )

    res.ok = not res.failures
    res.seconds = time.time() - t0

    if dest and os.path.exists(dest) and not keep:
        shutil.rmtree(dest, ignore_errors=True)
    IR._pull_tasks.pop(tid, None)
    return res


def _print_report(results: list[Result]) -> bool:
    print("\n" + "=" * 78)
    print("IMAGE DOWNLOAD DIAGNOSTIC REPORT")
    print("=" * 78)
    all_ok = True
    for r in results:
        if r.skipped:
            print(f"\n[SKIP] {r.name}: {r.skipped}")
            continue
        tag = "PASS" if r.ok else "FAIL"
        if not r.ok:
            all_ok = False
        print(f"\n[{tag}] {r.name}  ({r.seconds:.0f}s)")
        print(f"    detect: {r.detect_variants} variants, min size {r.detect_min_size_gb} GB")
        nums = [p for p in r.progress_samples if isinstance(p, (int, float))]
        curve = " ".join(str(p) for p in nums[:: max(1, len(nums) // 12 or 1)]) if nums else "(none)"
        print(f"    progress: intermediate={r.intermediate_progress}  curve: {curve}")
        print(f"    final: status={r.final_status}  structure={r.structure}  valid_dir={r.valid_dir}")
        for f in r.failures:
            print(f"    ✗ {f}")
    print("\n" + "=" * 78)
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    print("=" * 78)
    return all_ok


async def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", action="append", default=[], help="run only named case(s)")
    ap.add_argument("--keep", action="store_true", help="keep downloaded models")
    ap.add_argument("--list", action="store_true", help="list cases and exit")
    args = ap.parse_args()

    if args.list:
        for c in CASES:
            extra = f" [needs {c.needs_key} key]" if c.needs_key else ""
            print(f"  {c.name:20s} {c.source_type:12s} {c.expect_structure:12s}{extra}")
            print(f"  {'':20s} {c.notes}")
        return 0

    cases = [c for c in CASES if not args.case or c.name in args.case]
    if not cases:
        print("No matching cases. Use --list to see available cases.")
        return 2

    results = []
    for c in cases:
        print(f"\n>>> running case: {c.name} ({c.source or '<unset>'})")
        results.append(await run_case(c, args.keep))
    return 0 if _print_report(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
