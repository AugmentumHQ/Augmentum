#!/usr/bin/env python
"""Verify the *initial pull* works for every model in the image catalog.

For each entry in ``RECOMMENDED_MODELS`` this resolves the exact file
manifest the real download path would fetch — the first step of
``pull_from_huggingface`` (``repo_info(files_metadata=True)`` +
``_filter_inference_files`` with the catalog's ``allow_patterns``) — WITHOUT
streaming the weights. That proves:

  * the backend/source is correctly classified (HF vs CivitAI),
  * the pull resolves a **non-empty, sized** file set (no bad repo_id / no
    allow_patterns that match nothing),
  * the on-disk structure is what we expect (diffusers / single-file / gguf),
  * for GGUF entries: the ``gguf_base_repo`` component files ALSO resolve and
    the ``gguf_pipeline_class`` / ``gguf_transformer_class`` are populated
    (so first inference has its text encoder / VAE / tokenizer).

It does NOT complete any download. Gated repos (need an HF token) are
reported distinctly rather than failed.

    docker exec -w /app augmentum-augmentum-1 python scripts/verify_catalog_pulls.py

Flags:
    --token TOKEN   use an HF token for gated repos (else uses env/configured)
    --only SUBSTR   only check catalog entries whose repo_id contains SUBSTR
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmentum.config import settings  # noqa: E402
from augmentum.image.hardware import RECOMMENDED_LORAS, RECOMMENDED_MODELS  # noqa: E402
from augmentum.image.model_manager import ModelManager  # noqa: E402
from augmentum.proxy import image_routes as IR  # noqa: E402


def _fmt(nbytes: int) -> str:
    if nbytes >= 1_073_741_824:
        return f"{nbytes / 1_073_741_824:.2f} GB"
    if nbytes >= 1_048_576:
        return f"{nbytes / 1_048_576:.0f} MB"
    return f"{nbytes / 1024:.0f} KB"


def _classify(files: list[str]) -> str:
    if any(f == "model_index.json" for f in files):
        return "diffusers"
    if any(f.endswith(".gguf") for f in files):
        return "gguf"
    st = [f for f in files if f.endswith(".safetensors")]
    if len(st) == 1 and "/" not in st[0]:
        return "single_file"
    if st:
        return "safetensors-set"
    return "other"


def _is_gated_error(msg: str) -> bool:
    m = msg.lower()
    return any(k in m for k in ("gated", "401", "403", "awaiting", "access to model", "restricted"))


async def _resolve_hf(mgr: ModelManager, repo_id: str, allow_patterns, token):
    """Mirror pull_from_huggingface's manifest step. Returns (files, sizes, err)."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    try:
        info = await asyncio.to_thread(lambda: api.repo_info(repo_id, files_metadata=True))
    except Exception as exc:  # noqa: BLE001
        return None, None, str(exc)
    siblings = info.siblings or []
    filtered = mgr._filter_inference_files(siblings, allow_patterns=allow_patterns)
    files = [s.rfilename for s in filtered]
    total = sum(s.size or 0 for s in filtered)
    return files, total, None


async def _resolve_gguf_base(repo_id: str, token):
    """Resolve base-repo component files (text encoder/VAE/tokenizer/scheduler)."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    try:
        info = await asyncio.to_thread(api.repo_info, repo_id)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    siblings = info.siblings or []
    skip_pref = {"transformer/"}
    skip_ext = {".ckpt", ".bin", ".pt", ".onnx", ".pb", ".tflite", ".png", ".jpg", ".jpeg", ".gif", ".md"}
    out = []
    for s in siblings:
        fn = s.rfilename
        if any(fn.startswith(p) for p in skip_pref):
            continue
        ext = "." + fn.rsplit(".", 1)[-1].lower() if "." in fn else ""
        if ext in skip_ext:
            continue
        out.append(fn)
    return out, None


async def check_loras() -> int:
    """Verify the LoRA catalog (RECOMMENDED_LORAS) — all CivitAI by id.

    LoRAs install via the same detect→pull path but with asset_type=lora,
    so the file lands in loras/ with a trigger-word/base-model sidecar. This
    checks the CivitAI detect resolves proper backend info for each: a 'lora'
    model_type (so the frontend sets asset_type=lora), a downloadable variant
    with a real URL+size, and a base_model that matches the catalog's claim.
    """
    print(f"Checking {len(RECOMMENDED_LORAS)} catalog LoRAs (CivitAI)\n")
    n_pass = n_fail = 0
    fails = []
    for cl in RECOMMENDED_LORAS:
        det = await IR._detect_civitai(cl.civitai_id, _Req())
        problems = []
        mt = det.get("model_type", "")
        variants = det.get("variants") or []
        det_base = det.get("base_model", "")
        sizes = [v.get("size_gb", 0) for v in variants]
        has_url = any(v.get("download_url") for v in variants)

        if det.get("error"):
            problems.append(det["error"][:50])
        else:
            if mt != "lora":
                problems.append(f"model_type='{mt}' (not lora → frontend won't route to loras/)")
            if not variants:
                problems.append("no downloadable variants")
            if variants and max(sizes) <= 0:
                problems.append("all variants 0 bytes")
            if variants and not has_url:
                problems.append("no download_url on any variant")
            if det_base and cl.base_model and det_base != cl.base_model:
                problems.append(f"base_model {det_base} != catalog {cl.base_model}")

        status = "PASS" if not problems else "FAIL"
        if problems:
            n_fail += 1
            fails.append((cl.name, cl.civitai_id, "; ".join(problems)))
        else:
            n_pass += 1
        trig = det.get("trigger_words") or []
        size_mb = round(max(sizes) * 1024) if sizes else 0
        print(f"[{status:6s}] {cl.name[:30]:30s} id={cl.civitai_id:8s} type={mt or '-':6s} "
              f"base={det_base or '-':5s} variants={len(variants):<2} {size_mb:>5}MB "
              f"triggers={len(trig)}")

    print("\n" + "=" * 80)
    print(f"LoRA SUMMARY: {n_pass} PASS, {n_fail} FAIL  of {len(RECOMMENDED_LORAS)}")
    print("=" * 80)
    if fails:
        print("\nFAILURES:")
        for name, cid, why in fails:
            print(f"  {name} ({cid}): {why}")
    print("\nNote: HuggingFace LoRA repos are NOT auto-detected as LoRAs "
          "(_detect_huggingface returns no model_type), so they install as "
          "regular models. The catalog is CivitAI-only, so this is unaffected.")
    return 1 if n_fail else 0


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--token", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--loras", action="store_true", help="verify the LoRA catalog instead of base models")
    args = ap.parse_args()

    if args.loras:
        return await check_loras()

    token = args.token or ModelManager("")._resolve_hf_token(settings.image_huggingface_token)
    mgr = ModelManager(settings.image_model_dir or f"{settings.data_dir}/image_models")

    models = [m for m in RECOMMENDED_MODELS if not args.only or args.only in m.repo_id]
    print(f"Checking {len(models)} catalog models  (HF token: {'yes' if token else 'no'})\n")

    rows = []
    n_pass = n_gated = n_fail = 0

    for cm in models:
        src_type = IR._detect_source_type(cm.repo_id)
        row = {"name": cm.name, "repo": cm.repo_id, "src": src_type, "status": "?",
               "files": 0, "size": 0, "struct": "", "extra": ""}

        if src_type == "civitai":
            det = await IR._detect_civitai(cm.repo_id, _Req())
            variants = det.get("variants") or []
            if det.get("error"):
                row["status"], row["extra"] = "FAIL", det["error"][:60]
                n_fail += 1
            elif not variants:
                row["status"], row["extra"] = "FAIL", "no variants"
                n_fail += 1
            else:
                row["status"], row["files"] = "PASS", len(variants)
                n_pass += 1
            rows.append(row)
            print(_line(row)); continue

        # HuggingFace
        files, total, err = await _resolve_hf(mgr, cm.repo_id, cm.allow_patterns, token)
        if err is not None:
            if _is_gated_error(err):
                row["status"], row["extra"] = "GATED", "needs HF token"
                n_gated += 1
            else:
                row["status"], row["extra"] = "FAIL", err[:60]
                n_fail += 1
            rows.append(row); print(_line(row)); continue

        row["files"], row["size"], row["struct"] = len(files), total, _classify(files)
        problems = []
        if not files:
            problems.append("0 files resolved")
        if total <= 0:
            problems.append("0 bytes")
        if cm.allow_patterns and not any(any(_match(f, p) for p in cm.allow_patterns) for f in files):
            problems.append(f"allow_patterns {cm.allow_patterns} matched nothing")

        # GGUF-specific completeness
        if cm.gguf_base_repo:
            if row["struct"] != "gguf":
                problems.append(f"expected gguf, got {row['struct']}")
            if not (cm.gguf_pipeline_class and cm.gguf_transformer_class):
                problems.append("missing gguf pipeline/transformer class")
            base_files, base_err = await _resolve_gguf_base(cm.gguf_base_repo, token)
            if base_err is not None:
                tag = "GATED-base" if _is_gated_error(base_err) else "base-FAIL"
                problems.append(f"{tag}: {base_err[:40]}")
            elif not base_files:
                problems.append(f"base repo {cm.gguf_base_repo} resolved 0 components")
            else:
                has_core = any(("text_encoder" in f or "vae" in f or "tokenizer" in f) for f in base_files)
                row["extra"] = f"+{len(base_files)} base files"
                if not has_core:
                    problems.append("base repo missing text_encoder/vae/tokenizer")

        if problems:
            row["status"] = "FAIL"
            row["extra"] = (row["extra"] + " | " if row["extra"] else "") + "; ".join(problems)
            n_fail += 1
        else:
            row["status"] = "PASS"
            n_pass += 1
        rows.append(row); print(_line(row))

    print("\n" + "=" * 96)
    print(f"SUMMARY: {n_pass} PASS, {n_gated} GATED (need token), {n_fail} FAIL  of {len(models)}")
    print("=" * 96)
    if n_fail:
        print("\nFAILURES:")
        for r in rows:
            if r["status"] == "FAIL":
                print(f"  {r['name']} ({r['repo']}): {r['extra']}")
    return 1 if n_fail else 0


def _match(fname: str, pat: str) -> bool:
    import fnmatch
    return fnmatch.fnmatch(fname, pat) or fnmatch.fnmatch(fname.split("/")[-1], pat)


def _line(r: dict) -> str:
    return (f"[{r['status']:10s}] {r['name'][:34]:34s} {r['src']:11s} "
            f"{r['struct']:14s} files={r['files']:<3} {_fmt(r['size']):>9}  {r['extra']}")


class _Req:
    class _App:
        class state:  # noqa: N801
            image_model_manager = None
    app = _App()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
