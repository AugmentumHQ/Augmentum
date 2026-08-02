"""Live route integration test — verifies the full Augmentum pipeline against running Docker services.

Tests real production paths: LLM chat, TTS generation, STT transcription,
voice auto-routing, image generation with model load/unload, CRUD operations,
settings toggles, and cross-service round-trips.

    python tests/live_routes_test.py
    python tests/live_routes_test.py --url http://localhost:6100 --verbose
    python tests/live_routes_test.py --section voice
    python tests/live_routes_test.py --section image

Sections: health, config, chat, character, persona, browse, narrative, image, voice, stt, tools, settings
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import httpx
except ImportError:
    import site
    sys.path.insert(0, site.getusersitepackages())
    import httpx

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_pass = 0
_fail = 0
_skip = 0
_cleanup: list = []  # (method, path) tuples to clean up on exit


def _ok(name: str, detail: str = ""):
    global _pass
    _pass += 1
    d = f"  ({detail})" if detail else ""
    print(f"  \033[92mPASS\033[0m  {name}{d}")


def _fail_msg(name: str, reason: str):
    global _fail
    _fail += 1
    print(f"  \033[91mFAIL\033[0m  {name}: {reason}")


def _skipped(name: str, reason: str = ""):
    global _skip
    _skip += 1
    r = f": {reason}" if reason else ""
    print(f"  \033[2mSKIP\033[0m  {name}{r}")


async def _check(client: httpx.AsyncClient, method: str, path: str,
                 *, json_body=None, expected: int = 200,
                 name: str = "", check_key: str = "",
                 timeout: float = 30.0,
                 verbose: bool = False) -> dict | None:
    """Make a request and assert status code."""
    label = name or f"{method.upper()} {path}"
    try:
        kwargs: dict = {"timeout": timeout}
        if json_body is not None:
            kwargs["json"] = json_body
        resp = await getattr(client, method)(path, **kwargs)
        if resp.status_code != expected:
            _fail_msg(label, f"expected {expected}, got {resp.status_code}")
            if verbose:
                print(f"         Body: {resp.text[:200]}")
            return None

        data = None
        ct = resp.headers.get("content-type", "")
        if ct.startswith("application/json"):
            data = resp.json()
        elif ct.startswith(("audio/", "image/")):
            # Binary response — return size info
            data = {"_binary": True, "_size": len(resp.content), "_content_type": ct}

        detail = ""
        if isinstance(data, list):
            detail = f"{len(data)} items"
        elif check_key and isinstance(data, dict) and check_key in data:
            val = data[check_key]
            if isinstance(val, list):
                detail = f"{len(val)} items"
            elif isinstance(val, (str, int, float, bool)):
                detail = str(val)[:50]
        elif isinstance(data, dict) and data.get("_binary"):
            detail = f"{data['_size']} bytes {data['_content_type']}"

        _ok(label, detail)
        if verbose and data and not data.get("_binary"):
            keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
            print(f"         Keys: {keys}")
        return data

    except httpx.TimeoutException:
        _fail_msg(label, f"timeout ({timeout}s)")
        return None
    except Exception as e:
        _fail_msg(label, str(e)[:100])
        return None


async def _cleanup_all(client: httpx.AsyncClient):
    """Best-effort cleanup of test data."""
    for method, path in _cleanup:
        try:
            await getattr(client, method)(path, timeout=5.0)
        except Exception:
            pass
    _cleanup.clear()

# ---------------------------------------------------------------------------
# Test sections
# ---------------------------------------------------------------------------

async def test_health(client: httpx.AsyncClient, verbose: bool):
    print("\n\033[1m--- Health & System ---\033[0m")
    await _check(client, "get", "/", name="Root health", verbose=verbose)
    await _check(client, "get", "/api/health", name="Deep health", verbose=verbose)
    await _check(client, "get", "/api/capabilities", name="Capabilities", verbose=verbose)
    await _check(client, "get", "/api/tools", name="Tool registry", check_key="tools", verbose=verbose)
    await _check(client, "get", "/api/ui/status", name="System status", verbose=verbose)


async def test_config(client: httpx.AsyncClient, verbose: bool):
    print("\n\033[1m--- Config ---\033[0m")
    await _check(client, "get", "/api/config/", name="Get config", verbose=verbose)
    data = await _check(client, "get", "/api/config/tools", name="Get tool settings", verbose=verbose)
    await _check(client, "get", "/api/config/ui", name="Get UI settings", verbose=verbose)

    # Verify key redaction
    if data:
        raw = json.dumps(data)
        if "api_key" in raw.lower() and any(v for k, v in data.items() if "key" in k.lower() and v):
            _fail_msg("Config key redaction", "API keys visible in config response")
        else:
            _ok("Config key redaction", "no keys exposed")


async def test_chat(client: httpx.AsyncClient, verbose: bool):
    print("\n\033[1m--- Chat Sessions ---\033[0m")
    await _check(client, "get", "/api/chats/?meta=1", name="List sessions (meta)", verbose=verbose)

    # Create via sync
    await _check(client, "post", "/api/chats/sync", json_body={
        "sessions": {
            "live-test-session": {
                "id": "live-test-session", "title": "Live Test", "mode": "passthrough",
                "tree": {"root": {"id": "root", "role": "system", "content": "test", "children": []}},
                "currentNodeId": "root", "createdAt": 1700000000000, "updatedAt": 1700000000000,
            }
        }
    }, name="Sync session", verbose=verbose)
    _cleanup.append(("delete", "/api/chats/live-test-session"))

    # Read back
    data = await _check(client, "get", "/api/chats/live-test-session", name="Get session", verbose=verbose)
    if data and data.get("title") != "Live Test":
        _fail_msg("Session roundtrip", f"title mismatch: {data.get('title')}")

    # Discover available model
    model_name = ""
    try:
        r = await client.get("/api/tags", timeout=10.0)
        if r.status_code == 200:
            models = r.json().get("models", [])
            if models:
                model_name = models[0].get("name", "")
    except Exception:
        pass

    # Non-streaming chat
    if model_name:
        chat_resp = await _check(client, "post", "/api/chat", json_body={
            "model": model_name,
            "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
            "stream": False,
        }, name=f"Chat non-streaming ({model_name})", timeout=60.0, verbose=verbose)
        if chat_resp and "message" in chat_resp:
            _ok("Chat response content", chat_resp["message"].get("content", "")[:60])

        # Streaming chat
        try:
            r = await client.post("/api/chat", json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Say 'world' and nothing else."}],
                "stream": True,
            }, timeout=60.0)
            if r.status_code == 200:
                lines = r.text.strip().split("\n")
                last = json.loads(lines[-1]) if lines else {}
                if last.get("done"):
                    content = "".join(
                        json.loads(l).get("message", {}).get("content", "")
                        for l in lines if l.strip()
                    )
                    _ok("Chat streaming", content.strip()[:60])
                else:
                    _fail_msg("Chat streaming", "stream didn't end with done=true")
            else:
                _fail_msg("Chat streaming", f"status {r.status_code}")
        except Exception as e:
            _fail_msg("Chat streaming", str(e)[:80])
    else:
        _skipped("Chat non-streaming", "no models loaded")
        _skipped("Chat streaming", "no models loaded")

    # Cleanup
    await _check(client, "delete", "/api/chats/live-test-session", name="Delete session", verbose=verbose)
    await _check(client, "get", "/api/chats/live-test-session", expected=404, name="Verify deleted", verbose=verbose)


async def test_character(client: httpx.AsyncClient, verbose: bool):
    print("\n\033[1m--- Characters ---\033[0m")
    await _check(client, "get", "/api/characters/", name="List characters", verbose=verbose)

    await _check(client, "put", "/api/characters/live-test-char", json_body={
        "name": "Test Character", "description": "A test character",
        "personality": "helpful", "greeting": "Hello from the live test!",
    }, name="Create character", verbose=verbose)
    _cleanup.append(("delete", "/api/characters/live-test-char"))

    data = await _check(client, "get", "/api/characters/live-test-char", name="Get character", verbose=verbose)
    if data and data.get("name") != "Test Character":
        _fail_msg("Character roundtrip", f"name mismatch: {data.get('name')}")

    await _check(client, "delete", "/api/characters/live-test-char", name="Delete character", verbose=verbose)
    await _check(client, "get", "/api/characters/live-test-char", expected=404, name="Verify deleted", verbose=verbose)


async def test_persona(client: httpx.AsyncClient, verbose: bool):
    print("\n\033[1m--- Personas ---\033[0m")
    await _check(client, "get", "/api/personas/", name="List personas", verbose=verbose)

    data = await _check(client, "post", "/api/personas/", json_body={
        "name": "Test Persona", "description": "Live test persona",
    }, expected=201, name="Create persona", verbose=verbose)
    pid = data.get("id") if data else None

    if pid:
        _cleanup.append(("delete", f"/api/personas/{pid}"))
        await _check(client, "get", f"/api/personas/{pid}", name="Get persona", verbose=verbose)
        await _check(client, "delete", f"/api/personas/{pid}", name="Delete persona", verbose=verbose)


async def test_browse(client: httpx.AsyncClient, verbose: bool):
    print("\n\033[1m--- Browse & Notes ---\033[0m")

    # Notes CRUD
    create = await _check(client, "post", "/api/browse/notes", json_body={
        "title": "Live Test Note", "content": "Created by live integration test.", "tags": ["test"],
    }, expected=201, name="Create note", verbose=verbose)
    note_id = create.get("id") if create else None

    if note_id:
        _cleanup.append(("delete", f"/api/browse/notes/{note_id}"))
        data = await _check(client, "get", f"/api/browse/notes/{note_id}", name="Get note", verbose=verbose)
        if data and data.get("title") != "Live Test Note":
            _fail_msg("Note roundtrip", "title mismatch")

        await _check(client, "put", f"/api/browse/notes/{note_id}", json_body={
            "title": "Updated Note", "content": "Updated.", "tags": ["updated"],
        }, name="Update note", verbose=verbose)

        await _check(client, "delete", f"/api/browse/notes/{note_id}", name="Delete note", verbose=verbose)

    # Web search
    try:
        resp = await client.get("/api/browse/search", params={"q": "python programming"}, timeout=15.0)
        if resp.status_code == 200:
            count = len(resp.json().get("results", []))
            _ok("Web search", f"{count} results")
        else:
            _skipped("Web search", f"status {resp.status_code}")
    except Exception:
        _skipped("Web search", "SearXNG not available")


async def test_narrative(client: httpx.AsyncClient, verbose: bool):
    print("\n\033[1m--- Narrative ---\033[0m")
    await _check(client, "get", "/api/narrative/presets", name="List presets", check_key="presets", verbose=verbose)
    await _check(client, "get", "/api/narrative/regex", name="List regex scripts", verbose=verbose)
    await _check(client, "get", "/api/narrative/groups", name="List groups", check_key="groups", verbose=verbose)
    await _check(client, "get", "/api/narrative/lorebook/global", name="Global lorebook", check_key="collections", verbose=verbose)


async def test_image(client: httpx.AsyncClient, verbose: bool):
    print("\n\033[1m--- Image ---\033[0m")

    # Hardware check
    hw = await _check(client, "get", "/api/image/hardware", name="Hardware detection", verbose=verbose)
    vram_free = hw.get("vram_free_mb", 0) if hw else 0
    vram_total = hw.get("vram_total_mb", 0) if hw else 0
    gpu = hw.get("device_name", "none") if hw else "none"
    if vram_total:
        _ok("GPU detected", f"{gpu} — {vram_free}MB free / {vram_total}MB")
    else:
        _skipped("GPU", "no GPU available")

    # Model list
    r = await client.get("/api/image/models", timeout=10.0)
    models = r.json() if r.status_code == 200 and isinstance(r.json(), list) else []
    _ok("Image models", f"{len(models)} available") if models else _skipped("Image models", "none")

    if not models or not vram_total:
        _skipped("Image generation", "no GPU or models")
        return

    # Unload anything currently loaded
    await client.post("/api/image/unload", timeout=30.0)
    await asyncio.sleep(1)

    # Refresh VRAM
    hw2 = await client.get("/api/image/hardware", timeout=5.0)
    vram_free = hw2.json().get("vram_free_mb", 0) if hw2.status_code == 200 else 0

    # Sort by size, test smallest that fits
    models.sort(key=lambda m: m.get("size_bytes", 0))
    tested = 0
    for m in models:
        name = m.get("name", "?")
        size_mb = m.get("size_bytes", 0) / (1024**2)

        if size_mb > vram_free * 0.85:
            if verbose:
                _skipped(f"Generate ({name})", f"{size_mb/1024:.1f}GB > {vram_free}MB free")
            continue

        start = time.time()
        try:
            r = await client.post("/api/image/generate", json={
                "prompt": "A sunset over mountains",
                "model": name, "width": 512, "height": 512, "steps": 4,
            }, timeout=300.0)
            elapsed = time.time() - start

            if r.status_code == 200:
                img_id = r.json().get("image_id", "")
                img_r = await client.get(f"/api/image/{img_id}", timeout=10.0)
                kb = len(img_r.content) / 1024 if img_r.status_code == 200 else 0
                _ok(f"Generate ({name})", f"{elapsed:.1f}s | {kb:.0f}KB")
                tested += 1
            else:
                _fail_msg(f"Generate ({name})", f"{elapsed:.1f}s | {r.status_code}")
        except httpx.ReadTimeout:
            _fail_msg(f"Generate ({name})", f"timeout ({time.time()-start:.0f}s)")
        except Exception as e:
            _fail_msg(f"Generate ({name})", str(e)[:60])

        # Unload before next model
        await client.post("/api/image/unload", timeout=30.0)
        await asyncio.sleep(1)

        # Refresh VRAM for next iteration
        hw3 = await client.get("/api/image/hardware", timeout=5.0)
        vram_free = hw3.json().get("vram_free_mb", 0) if hw3.status_code == 200 else 0

        if tested >= 3:
            break  # Don't test every model, top 3 that fit is enough

    # Prompt enhancement
    enh = await _check(client, "post", "/api/image/enhance-prompt", json_body={
        "prompt": "a wizard"
    }, name="Prompt enhancement", timeout=30.0, verbose=verbose)
    if enh and enh.get("prompt"):
        _ok("Enhanced prompt", enh["prompt"][:60])


async def test_voice(client: httpx.AsyncClient, verbose: bool):
    print("\n\033[1m--- Voice & Audio ---\033[0m")

    # Provider & voice list (primes the auto-routing cache)
    await _check(client, "get", "/api/audio/providers", name="Audio providers", verbose=verbose)
    voices_data = await client.get("/api/audio/voices", timeout=15.0)
    voices = voices_data.json() if voices_data.status_code == 200 else []
    by_provider: dict[str, list[str]] = {}
    for v in voices:
        pid = v.get("provider_id", "?")
        by_provider.setdefault(pid, []).append(v.get("name", "?"))
    _ok("Voice list", f"{len(voices)} voices across {len(by_provider)} providers")
    if verbose:
        for pid, names in by_provider.items():
            print(f"         {pid}: {', '.join(names[:3])}{'...' if len(names) > 3 else ''}")

    # Test auto-routing: one voice per provider
    for pid, names in by_provider.items():
        voice = names[0]
        start = time.time()
        try:
            r = await client.post("/v1/audio/speech", json={
                "input": f"Testing {pid}.", "voice": voice, "response_format": "wav",
            }, timeout=30.0)
            elapsed = time.time() - start
            if r.status_code == 200 and len(r.content) > 100:
                _ok(f"TTS auto-route: {voice}", f"{pid} | {len(r.content)/1024:.0f}KB | {elapsed:.1f}s")
            elif r.status_code == 200:
                _fail_msg(f"TTS auto-route: {voice}", f"empty audio ({len(r.content)} bytes)")
            else:
                _fail_msg(f"TTS auto-route: {voice}", f"{r.status_code}")
        except Exception as e:
            _fail_msg(f"TTS auto-route: {voice}", str(e)[:60])

    # Kokoro voice blending
    kokoro_voices = by_provider.get("kokoro-builtin", [])
    if len(kokoro_voices) >= 2:
        blend = f"{kokoro_voices[0]}+{kokoro_voices[1]}"
        try:
            r = await client.post("/v1/audio/speech", json={
                "input": "Voice blending test.", "voice": blend, "response_format": "wav",
            }, timeout=30.0)
            if r.status_code == 200 and len(r.content) > 100:
                _ok(f"Kokoro blend: {blend}", f"{len(r.content)/1024:.0f}KB")
            else:
                _fail_msg("Kokoro blend", f"{r.status_code}")
        except Exception as e:
            _fail_msg("Kokoro blend", str(e)[:60])


async def test_stt(client: httpx.AsyncClient, verbose: bool):
    print("\n\033[1m--- Speech-to-Text ---\033[0m")

    # Generate TTS audio as STT input
    tts_r = await client.post("/v1/audio/speech", json={
        "input": "Testing speech recognition.", "voice": "af_heart", "response_format": "wav",
    }, timeout=30.0)
    if tts_r.status_code != 200 or len(tts_r.content) < 100:
        _skipped("STT roundtrip", "TTS generation failed")
        return

    audio = tts_r.content
    _ok("TTS for STT input", f"{len(audio)/1024:.0f}KB WAV")

    # Moonshine STT
    try:
        start = time.time()
        r = await client.post("/v1/audio/transcriptions",
            files={"file": ("test.wav", audio, "audio/wav")}, timeout=30.0)
        elapsed = time.time() - start
        if r.status_code == 200:
            text = r.json().get("text", "")
            _ok("Moonshine STT", f'"{text}" ({elapsed:.1f}s)')
            # Check accuracy
            expected_words = {"testing", "speech", "recognition"}
            actual_words = set(text.lower().split())
            overlap = len(expected_words & actual_words)
            if overlap >= 2:
                _ok("STT accuracy", f"{overlap}/{len(expected_words)} key words matched")
            else:
                _fail_msg("STT accuracy", f"only {overlap}/{len(expected_words)} words: got '{text}'")
        elif r.status_code == 503:
            _skipped("Moonshine STT", r.json().get("detail", "not available")[:60])
        else:
            _fail_msg("Moonshine STT", f"{r.status_code}: {r.text[:100]}")
    except Exception as e:
        _fail_msg("Moonshine STT", str(e)[:80])


async def test_tools(client: httpx.AsyncClient, verbose: bool):
    print("\n\033[1m--- Tools & Infrastructure ---\033[0m")

    data = await _check(client, "get", "/api/tools", name="Tool registry", check_key="tools", verbose=verbose)
    if data and data.get("tools"):
        _ok("Tools registered", f"{len(data['tools'])} tools")

    await _check(client, "get", "/api/reasoning/flows", name="List flows", verbose=verbose)
    await _check(client, "get", "/api/models/status", name="Model status", verbose=verbose)
    await _check(client, "get", "/api/providers/", name="LLM providers", verbose=verbose)
    await _check(client, "get", "/api/cache/stats", name="Cache stats", verbose=verbose)
    await _check(client, "get", "/v1/memory/facts?limit=5", name="Memory facts", verbose=verbose)


async def test_settings(client: httpx.AsyncClient, verbose: bool):
    print("\n\033[1m--- Settings Toggles ---\033[0m")

    # Read current tool settings
    r = await client.get("/api/config/tools", timeout=10.0)
    if r.status_code != 200:
        _fail_msg("Read tool settings", f"status {r.status_code}")
        return
    original = r.json()

    # Test toggling a boolean setting (auto_search)
    key = "uarf_auto_search"
    orig_val = original.get(key, True)
    new_val = not orig_val

    r = await client.put("/api/config/tools", json={key: new_val}, timeout=10.0)
    if r.status_code == 200:
        _ok(f"Toggle {key}", f"{orig_val} -> {new_val}")
    else:
        _fail_msg(f"Toggle {key}", f"status {r.status_code}")

    # Verify it persisted
    r = await client.get("/api/config/tools", timeout=10.0)
    if r.status_code == 200:
        actual = r.json().get(key)
        if actual == new_val:
            _ok(f"Verify {key} persisted", str(new_val))
        else:
            _fail_msg(f"Verify {key}", f"expected {new_val}, got {actual}")

    # Restore original
    await client.put("/api/config/tools", json={key: orig_val}, timeout=10.0)

    # Test a numeric setting with bounds
    key2 = "uarf_auto_search_queries"
    r = await client.put("/api/config/tools", json={key2: 3}, timeout=10.0)
    if r.status_code == 200:
        _ok(f"Set {key2}=3", "within bounds")

    # Test out-of-bounds (should clamp or reject)
    r = await client.put("/api/config/tools", json={key2: 999}, timeout=10.0)
    if r.status_code == 200:
        check = await client.get("/api/config/tools", timeout=10.0)
        val = check.json().get(key2, 0) if check.status_code == 200 else 0
        if val <= 10:
            _ok(f"Bounds check {key2}", f"999 clamped to {val}")
        else:
            _fail_msg(f"Bounds check {key2}", f"accepted 999 without clamping: {val}")
    else:
        _ok(f"Bounds reject {key2}=999", f"status {r.status_code}")

    # Restore
    await client.put("/api/config/tools", json={key2: original.get(key2, 5)}, timeout=10.0)

    # Test string setting
    key3 = "tts_voice_style"
    r = await client.put("/api/config/tools", json={key3: "speak warmly"}, timeout=10.0)
    if r.status_code == 200:
        check = await client.get("/api/config/tools", timeout=10.0)
        val = check.json().get(key3, "") if check.status_code == 200 else ""
        if val == "speak warmly":
            _ok(f"String setting {key3}", "roundtrip OK")
        else:
            _fail_msg(f"String setting {key3}", f"got '{val}'")
    # Restore
    await client.put("/api/config/tools", json={key3: original.get(key3, "")}, timeout=10.0)

    # Test UI settings
    r = await client.get("/api/config/ui", timeout=10.0)
    if r.status_code == 200:
        _ok("Read UI settings", f"{len(r.json())} keys")

    # Test unknown key rejection
    r = await client.put("/api/config/tools", json={"nonexistent_setting_xyz": True}, timeout=10.0)
    _ok("Unknown key handling", f"status {r.status_code}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SECTIONS = {
    "health": test_health,
    "config": test_config,
    "chat": test_chat,
    "character": test_character,
    "persona": test_persona,
    "browse": test_browse,
    "narrative": test_narrative,
    "image": test_image,
    "voice": test_voice,
    "stt": test_stt,
    "tools": test_tools,
    "settings": test_settings,
}


async def main():
    parser = argparse.ArgumentParser(description="Live route integration test")
    parser.add_argument("--url", default="http://localhost:6100", help="Augmentum base URL")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show response details")
    parser.add_argument("--section", "-s", choices=list(SECTIONS.keys()), help="Run only one section")
    args = parser.parse_args()

    print("\n\033[1mAugmentum Live Integration Test\033[0m")
    print(f"Target: {args.url}")

    async with httpx.AsyncClient(base_url=args.url) as client:
        try:
            resp = await client.get("/", timeout=5.0)
            if resp.status_code != 200:
                print(f"\n\033[91mERROR: Server returned {resp.status_code}\033[0m")
                sys.exit(1)
        except Exception as e:
            print(f"\n\033[91mERROR: Cannot connect to {args.url}: {e}\033[0m")
            print("Make sure Docker services are running: start.bat")
            sys.exit(1)

        start = time.time()

        try:
            if args.section:
                await SECTIONS[args.section](client, args.verbose)
            else:
                for section_fn in SECTIONS.values():
                    await section_fn(client, args.verbose)
        finally:
            await _cleanup_all(client)

        elapsed = time.time() - start

    print(f"\n\033[1m{'=' * 50}\033[0m")
    print(f"\033[1m  Results: {_pass} passed, {_fail} failed, {_skip} skipped ({elapsed:.1f}s)\033[0m")
    if _fail:
        print(f"\033[91m  {_fail} FAILURE(S)\033[0m")
    else:
        print("\033[92m  ALL PASSED\033[0m")
    print()
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    asyncio.run(main())
