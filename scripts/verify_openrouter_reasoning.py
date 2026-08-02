"""Live, ZERO-COST verification of the OpenRouter unified ``reasoning`` object
(CORRECTIONS #4). Run it yourself so your key stays in your shell:

    OPENROUTER_API_KEY=sk-or-... .venv/Scripts/python.exe scripts/verify_openrouter_reasoning.py

It:
  1. reads the key from the environment (NEVER prints it),
  2. auto-discovers a FREE ($0/$0) model that advertises ``reasoning`` support,
  3. builds each request through the REAL ``_build_openai_payload`` (openrouter
     profile, with the new ``supports_openrouter_reasoning`` emit), and
  4. POSTs the three cases and reports status + whether reasoning came back.

A 400 surfaces before any billing, and free models cost nothing — so this is
safe on a zero-credit account. Nothing here writes to your account or the repo.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import httpx

from augmentum.models.base import InternalChatRequest, Message
from augmentum.models.openai_compat import OpenAIBackend
from augmentum.models.provider_profiles import PROFILES


def _load_key() -> str | None:
    """Resolve the key WITHOUT it ever touching the command line.

    Order: env var, then a gitignored ``scripts/.openrouter_key`` file (put
    your key there once with your own editor; it's in .gitignore). This means
    you can run the script via ``!`` with no secret in the command, and the
    output is key-redacted regardless.
    """
    for var in ("OPENROUTER_API_KEY", "OPENROUTER_KEY", "OR_API_KEY"):
        if os.environ.get(var):
            return os.environ[var].strip()
    keyfile = Path(__file__).with_name(".openrouter_key")
    if keyfile.exists():
        return keyfile.read_text(encoding="utf-8").strip()
    return None


KEY = _load_key()


def _redact(text: str) -> str:
    """Belt-and-suspenders: never let the key appear in printed output."""
    return text.replace(KEY, "***") if KEY else text


async def _find_free_reasoning_model(client: httpx.AsyncClient) -> str | None:
    r = await client.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {KEY}"},
    )
    r.raise_for_status()
    for m in r.json().get("data", []):
        pricing = m.get("pricing", {}) or {}
        try:
            free = float(pricing.get("prompt", "1") or 1) == 0 and \
                float(pricing.get("completion", "1") or 1) == 0
        except (TypeError, ValueError):
            free = False
        params = m.get("supported_parameters") or []
        if free and "reasoning" in params:
            return m["id"]
    return None


async def _run_case(client, backend, model, label, **req_kw):
    payload = backend._build_openai_payload(
        InternalChatRequest(
            model=model,
            messages=[Message(role="user", content="What is 17 * 23? Think step by step.")],
            max_tokens=512,
            **req_kw,
        )
    )
    payload["stream"] = False
    sent_reasoning = payload.get("reasoning")
    headers = {"Authorization": f"Bearer {KEY}", **(backend._profile.extra_headers or {})}
    try:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload, headers=headers, timeout=120.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [{label}] sent reasoning={sent_reasoning!r} -> REQUEST ERROR: {_redact(str(exc))}")
        return
    ok = r.status_code < 400
    note = ""
    if ok:
        msg = (r.json().get("choices") or [{}])[0].get("message", {})
        got = msg.get("reasoning") or msg.get("reasoning_content")
        content = (msg.get("content") or "").strip()
        note = f"reasoning_returned={'yes' if got else 'no'} content_len={len(content)}"
    else:
        note = "BODY: " + _redact(r.text[:300])
    flag = "OK " if ok else "400" if r.status_code == 400 else str(r.status_code)
    print(f"  [{label}] sent reasoning={sent_reasoning!r} -> {flag}  {note}")


async def main() -> int:
    if not KEY:
        print("Set OPENROUTER_API_KEY (or OPENROUTER_KEY / OR_API_KEY) in env. "
              "It is read from the environment and never printed.")
        return 1
    profile = PROFILES["openrouter"]
    async with httpx.AsyncClient() as client:
        backend = OpenAIBackend(client, profile.base_url, KEY, profile=profile)
        model = os.environ.get("OR_MODEL") or await _find_free_reasoning_model(client)
        if not model:
            print("No FREE model advertising `reasoning` support found. "
                  "Set OR_MODEL=<id> to force one.")
            return 2
        print(f"Model under test (free): {model}\n")
        await _run_case(client, backend, model, "think=True effort=high", think=True, reasoning_effort="high")
        await _run_case(client, backend, model, "think=True default   ", think=True)
        await _run_case(client, backend, model, "think=False disable   ", think=False)
    print("\nInterpretation: every line should be OK. '400' on the disable line "
          "means enabled:false isn't accepted for that model → tell Claude to "
          "switch the disable path to omit-on-False instead.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
