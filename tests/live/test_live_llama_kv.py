"""Live managed llama-server and KV-session checks.

These tests call a real Augmentum instance and a real local llama.cpp
server through Augmentum's managed engine routes. They are deliberately
opt-in:

    python -m pytest tests/live/test_live_llama_kv.py --run-live -q

Useful environment variables:

    AUGMENTUM_LIVE_URL=http://localhost:6100
    AUGMENTUM_LIVE_TOKEN=<session-token>
    AUGMENTUM_USERNAME=<username>
    AUGMENTUM_PASSWORD=<password>
    AUGMENTUM_LIVE_LLAMA_MODEL=<gguf path, filename, or model id>
    AUGMENTUM_LIVE_LLAMA_MODELS=<comma-separated model refs>
    AUGMENTUM_LIVE_LLAMA_MAX_MODEL_GB=16
    AUGMENTUM_LIVE_TIMEOUT=180
    AUGMENTUM_LIVE_LLAMA_LOAD_TIMEOUT=900
    AUGMENTUM_LIVE_LLAMA_RELOAD=1
    AUGMENTUM_LIVE_TRACE=1
    AUGMENTUM_LIVE_TRACE_DIR=tests/live/artifacts/llama-kv
    AUGMENTUM_LIVE_TRACE_CONTENT=redacted

The optional AUGMENTUM_LIVE_LLAMA_LOAD_OPTIONS_JSON value is merged into
/api/engine/v2/models/load. Current application behavior persists those
options as model defaults, so only set it in disposable/contained test
environments.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.live.trace_recorder import LiveTraceRecorder

pytestmark = [pytest.mark.live, pytest.mark.asyncio]

MODE_HEADER = "X-Augmentum-Mode"
SESSION_HEADER = "X-Augmentum-Session"
_TRACE_BY_CLIENT: dict[int, LiveTraceRecorder] = {}


@dataclass(frozen=True)
class ModelTarget:
    ref: str
    source: str
    size: int | None = None


@dataclass(frozen=True)
class LiveEngineModel:
    model_id: str
    chat_model: str
    load_ref: str
    target_source: str


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        pytest.fail(f"{name} must be a number, got {raw!r}")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        pytest.fail(f"{name} must be an integer, got {raw!r}")


def _live_timeout() -> float:
    return _env_float("AUGMENTUM_LIVE_TIMEOUT", 180.0)


def _load_timeout() -> float:
    return _env_float("AUGMENTUM_LIVE_LLAMA_LOAD_TIMEOUT", 900.0)


def _http_timeout(read_timeout: float | None = None) -> httpx.Timeout:
    read = read_timeout if read_timeout is not None else _live_timeout()
    return httpx.Timeout(connect=10.0, read=read, write=30.0, pool=10.0)


def _trace_for(client: httpx.AsyncClient) -> LiveTraceRecorder | None:
    return _TRACE_BY_CLIENT.get(id(client))


def _trace_label(method: str, path: str) -> str:
    cleaned = path.strip("/").replace("/", "_").replace("-", "_") or "root"
    return f"{method.lower()}_{cleaned}"


def _short_body(resp: httpx.Response, limit: int = 1200) -> str:
    try:
        text = resp.text
    except Exception:
        text = "<unavailable>"
    return text[:limit]


def _coerce_list_payload(data: Any, *keys: str) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


async def _request_json(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    timeout: float | None = None,
    trace_label: str = "",
    **kwargs: Any,
) -> dict[str, Any] | list[Any]:
    trace = _trace_for(client)
    label = trace_label or _trace_label(method, path)
    if trace:
        trace.request_start(
            label=label,
            method=method,
            path=path,
            json_body=kwargs.get("json"),
            headers=kwargs.get("headers") or {},
        )
    started = time.monotonic()
    try:
        resp = await client.request(
            method,
            path,
            timeout=_http_timeout(timeout),
            **kwargs,
        )
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        pytest.skip(f"Augmentum is not reachable at {client.base_url}: {exc}")
    if resp.status_code == 401:
        pytest.skip(
            "Live Augmentum requires auth. Set AUGMENTUM_LIVE_TOKEN or "
            "AUGMENTUM_USERNAME/AUGMENTUM_PASSWORD."
        )
    duration_ms = int((time.monotonic() - started) * 1000)
    if trace and resp.status_code >= 400:
        trace.request_end(
            label=label,
            method=method,
            path=path,
            status_code=resp.status_code,
            duration_ms=duration_ms,
            response=_short_body(resp),
        )
    assert resp.status_code < 400, (
        f"{method} {path} returned {resp.status_code}: {_short_body(resp)}"
    )
    try:
        data = resp.json()
    except ValueError as exc:
        if trace:
            trace.request_end(
                label=label,
                method=method,
                path=path,
                status_code=resp.status_code,
                duration_ms=duration_ms,
                response=_short_body(resp),
            )
        pytest.fail(
            f"{method} {path} returned non-JSON body: {_short_body(resp)}"
        )
        raise AssertionError from exc
    if trace:
        trace.request_end(
            label=label,
            method=method,
            path=path,
            status_code=resp.status_code,
            duration_ms=duration_ms,
            response=data,
        )
    return data


async def _probe_and_authenticate(client: httpx.AsyncClient) -> None:
    token = (
        os.getenv("AUGMENTUM_LIVE_TOKEN", "").strip()
        or os.getenv("AUGMENTUM_TOKEN", "").strip()
    )
    if token:
        client.headers["Authorization"] = f"Bearer {token}"

    try:
        status_resp = await client.get("/api/auth/status", timeout=_http_timeout(10.0))
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        pytest.skip(f"Augmentum is not reachable at {client.base_url}: {exc}")

    assert status_resp.status_code < 500, (
        f"GET /api/auth/status returned {status_resp.status_code}: "
        f"{_short_body(status_resp)}"
    )
    if status_resp.status_code == 404:
        return
    assert status_resp.status_code < 400, (
        f"GET /api/auth/status returned {status_resp.status_code}: "
        f"{_short_body(status_resp)}"
    )
    status = status_resp.json()
    if status.get("setup_required"):
        pytest.skip("Live Augmentum has not completed first-run auth setup.")
    if status.get("authenticated"):
        return

    username = os.getenv("AUGMENTUM_USERNAME", "").strip()
    password = os.getenv("AUGMENTUM_PASSWORD", "")
    if not username or not password:
        return

    login = await client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
        timeout=_http_timeout(20.0),
    )
    if login.status_code == 401:
        pytest.skip("AUGMENTUM_USERNAME/AUGMENTUM_PASSWORD were rejected.")
    assert login.status_code < 400, (
        f"POST /api/auth/login returned {login.status_code}: {_short_body(login)}"
    )


@pytest.fixture(scope="module")
def live_trace() -> LiveTraceRecorder:
    recorder = LiveTraceRecorder.from_env()
    try:
        yield recorder
    finally:
        recorder.finish()


@pytest.fixture
async def live_client(live_trace: LiveTraceRecorder) -> httpx.AsyncClient:
    base_url = os.getenv("AUGMENTUM_LIVE_URL", "http://localhost:6100").rstrip("/")
    live_trace.set_context(base_url=base_url)
    async with httpx.AsyncClient(
        base_url=base_url,
        follow_redirects=True,
        timeout=_http_timeout(),
        headers={"User-Agent": "augmentum-live-llama-kv-tests"},
    ) as client:
        _TRACE_BY_CLIENT[id(client)] = live_trace
        await _probe_and_authenticate(client)
        try:
            yield client
        finally:
            _TRACE_BY_CLIENT.pop(id(client), None)


async def _engine_status_or_skip(client: httpx.AsyncClient) -> dict[str, Any]:
    trace = _trace_for(client)
    label = "engine_status"
    if trace:
        trace.request_start(label=label, method="GET", path="/api/engine/v2/status")
    started = time.monotonic()
    resp = await client.get("/api/engine/v2/status", timeout=_http_timeout(20.0))
    duration_ms = int((time.monotonic() - started) * 1000)
    if resp.status_code == 401:
        pytest.skip(
            "Live Augmentum requires auth. Set AUGMENTUM_LIVE_TOKEN or "
            "AUGMENTUM_USERNAME/AUGMENTUM_PASSWORD."
        )
    if resp.status_code == 404:
        pytest.skip("Managed llama-server engine v2 is not enabled on this instance.")
    assert resp.status_code < 400, (
        f"GET /api/engine/v2/status returned {resp.status_code}: "
        f"{_short_body(resp)}"
    )
    data = resp.json()
    if trace:
        trace.request_end(
            label=label,
            method="GET",
            path="/api/engine/v2/status",
            status_code=resp.status_code,
            duration_ms=duration_ms,
            response=data,
        )
    return data


def _ready_state(data: dict[str, Any]) -> bool:
    state = str(data.get("state") or data.get("status") or "").lower()
    return state in {"ready", "running", "ok"}


def _model_id_from_ref(ref: str) -> str:
    cleaned = ref.rstrip("/\\")
    stem = Path(cleaned).stem
    return stem or cleaned


def _explicit_model_targets() -> list[ModelTarget]:
    raw = (
        os.getenv("AUGMENTUM_LIVE_LLAMA_MODELS", "").strip()
        or os.getenv("AUGMENTUM_LIVE_LLAMA_MODEL", "").strip()
    )
    if not raw:
        return []
    refs = [part.strip() for part in raw.split(",") if part.strip()]
    max_models = max(1, _env_int("AUGMENTUM_LIVE_LLAMA_MAX_MODELS", len(refs)))
    return [ModelTarget(ref=ref, source="env") for ref in refs[:max_models]]


def _model_ref_from_entry(entry: dict[str, Any]) -> str:
    for key in ("path", "model_path", "absolute_path", "filename", "model", "id", "name"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


async def _discover_model_targets(client: httpx.AsyncClient) -> list[ModelTarget]:
    data = await _request_json(
        client,
        "GET",
        "/api/engine/v2/models",
        timeout=60.0,
        trace_label="engine_models",
    )
    models = _coerce_list_payload(data, "models", "data", "files")
    max_bytes = int(_env_float("AUGMENTUM_LIVE_LLAMA_MAX_MODEL_GB", 16.0) * 1024**3)
    max_models = max(1, _env_int("AUGMENTUM_LIVE_LLAMA_MAX_MODELS", 1))
    candidates: list[ModelTarget] = []
    for entry in models:
        ref = _model_ref_from_entry(entry)
        if not ref:
            continue
        lower = ref.lower()
        if "mmproj" in lower or "clip-" in lower:
            continue
        raw_size = entry.get("size")
        size = raw_size if isinstance(raw_size, int) else None
        if size is not None and size > max_bytes:
            continue
        candidates.append(ModelTarget(ref=ref, source="discovered", size=size))

    candidates.sort(key=lambda target: target.size if target.size is not None else 10**18)
    return candidates[:max_models]


async def _select_model_targets(client: httpx.AsyncClient) -> list[ModelTarget]:
    explicit = _explicit_model_targets()
    if explicit:
        return explicit

    status = await _engine_status_or_skip(client)
    model_id = str(status.get("model_id") or "").strip()
    if _ready_state(status) and model_id:
        return [ModelTarget(ref=model_id, source="already-loaded")]

    discovered = await _discover_model_targets(client)
    if discovered:
        return discovered

    pytest.skip(
        "No local GGUF model discovered. Set AUGMENTUM_LIVE_LLAMA_MODEL "
        "or add a model directory before running this live suite."
    )
    return []


def _load_options_from_env() -> dict[str, Any]:
    raw = os.getenv("AUGMENTUM_LIVE_LLAMA_LOAD_OPTIONS_JSON", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        pytest.fail(f"AUGMENTUM_LIVE_LLAMA_LOAD_OPTIONS_JSON is invalid JSON: {exc}")
    if not isinstance(data, dict):
        pytest.fail("AUGMENTUM_LIVE_LLAMA_LOAD_OPTIONS_JSON must decode to an object.")
    return data


async def _wait_for_ready(
    client: httpx.AsyncClient,
    *,
    expected_model_id: str = "",
    timeout: float | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + (timeout if timeout is not None else _load_timeout())
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = await _engine_status_or_skip(client)
        if _ready_state(last):
            loaded = str(last.get("model_id") or "").strip()
            if not expected_model_id or loaded == expected_model_id:
                return last
        await asyncio.sleep(2.0)
    pytest.fail(f"Engine did not become ready in time. Last status: {last!r}")
    raise AssertionError("unreachable")


async def _ensure_model_loaded(
    client: httpx.AsyncClient,
    target: ModelTarget,
) -> LiveEngineModel:
    status = await _engine_status_or_skip(client)
    current_model_id = str(status.get("model_id") or "").strip()
    expected_from_ref = _model_id_from_ref(target.ref)
    if _ready_state(status) and current_model_id and (
        target.source == "already-loaded" or current_model_id == expected_from_ref
    ):
        model_id = current_model_id
        trace = _trace_for(client)
        if trace:
            trace.set_context(
                model_id=model_id,
                chat_model=f"{model_id}@engine",
                model_source=target.source,
            )
        return LiveEngineModel(
            model_id=model_id,
            chat_model=f"{model_id}@engine",
            load_ref=target.ref,
            target_source=target.source,
        )

    body: dict[str, Any] = {"model": target.ref}
    body.update(_load_options_from_env())
    loaded = await _request_json(
        client,
        "POST",
        "/api/engine/v2/models/load",
        json=body,
        timeout=_load_timeout(),
        trace_label="engine_model_load",
    )
    assert isinstance(loaded, dict)
    model_id = str(loaded.get("model_id") or expected_from_ref).strip()
    ready = await _wait_for_ready(
        client,
        expected_model_id=model_id,
        timeout=_load_timeout(),
    )
    model_id = str(ready.get("model_id") or model_id).strip()
    assert model_id, f"Model load did not report a model_id: {loaded!r}"
    trace = _trace_for(client)
    if trace:
        trace.set_context(
            model_id=model_id,
            chat_model=f"{model_id}@engine",
            model_source=target.source,
        )
    return LiveEngineModel(
        model_id=model_id,
        chat_model=f"{model_id}@engine",
        load_ref=target.ref,
        target_source=target.source,
    )


@pytest.fixture
async def live_engine_model(live_client: httpx.AsyncClient) -> LiveEngineModel:
    targets = await _select_model_targets(live_client)
    return await _ensure_model_loaded(live_client, targets[0])


def _content_from_completion(data: dict[str, Any] | list[Any]) -> str:
    if isinstance(data, dict):
        for key in ("content", "response", "text"):
            value = data.get(key)
            if isinstance(value, str):
                return value
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                text = first.get("text")
                if isinstance(text, str):
                    return text
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
    return ""


async def _llama_slots(
    client: httpx.AsyncClient,
    *,
    trace_label: str = "",
) -> list[dict[str, Any]]:
    data = await _request_json(
        client,
        "GET",
        "/api/llamacpp/slots",
        timeout=30.0,
        trace_label=trace_label or "llamacpp_slots",
    )
    slots = _coerce_list_payload(data, "slots")
    trace = _trace_for(client)
    if trace:
        trace.set_context(slot_count=len(slots))
        trace.slot_snapshot(trace_label or "llamacpp_slots", slots)
    return slots


def _slot_has_activity(slot: dict[str, Any]) -> bool:
    for key in (
        "id_task",
        "n_past",
        "n_prompt_tokens_processed",
        "n_tokens",
        "n_tokens_predicted",
        "n_decoded",
        "prompt_tokens",
        "tokens_cached",
        "tokens_predicted",
    ):
        value = slot.get(key)
        if isinstance(value, int | float) and value > 0:
            return True
    next_token = slot.get("next_token")
    if isinstance(next_token, list):
        for item in next_token:
            if not isinstance(item, dict):
                continue
            for key in ("n_decoded", "n_remain"):
                value = item.get(key)
                if isinstance(value, int | float) and value > 0:
                    return True
    return any(bool(slot.get(key)) for key in ("prompt", "cache_prompt", "params", "task_id"))


def _stage_names(chunks: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for chunk in chunks:
        aug = chunk.get("augmentum")
        if not isinstance(aug, dict):
            continue
        for key in ("stage_start", "stage_complete"):
            event = aug.get(key)
            if isinstance(event, dict):
                stage = event.get("stage")
                if isinstance(stage, str):
                    names.append(stage)
    return names


def _has_backend_error(chunks: list[dict[str, Any]]) -> bool:
    for chunk in chunks:
        aug = chunk.get("augmentum")
        if isinstance(aug, dict) and aug.get("backend_error"):
            return True
    return False


async def _stream_ollama_chat(
    client: httpx.AsyncClient,
    *,
    model: str,
    session_id: str,
    messages: list[dict[str, str]],
    max_tokens: int = 32,
    trace_label: str = "",
    ui_action: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0,
            "seed": 11,
        },
    }
    headers = {
        MODE_HEADER: "passthrough",
        SESSION_HEADER: session_id,
        "X-Augmentum-Memory-Query": "none",
    }
    label = trace_label or f"chat_{session_id}"
    trace = _trace_for(client)
    if trace:
        trace.stream_start(
            label=label,
            path="/api/chat",
            model=model,
            session_id=session_id,
            messages=messages,
            headers=headers,
            ui_action=ui_action,
        )
    chunks: list[dict[str, Any]] = []
    started = time.monotonic()
    first_content_ms = 0
    async with client.stream(
        "POST",
        "/api/chat",
        json=payload,
        headers=headers,
        timeout=_http_timeout(_live_timeout()),
    ) as resp:
        if resp.status_code == 401:
            pytest.skip(
                "Live Augmentum requires auth. Set AUGMENTUM_LIVE_TOKEN or "
                "AUGMENTUM_USERNAME/AUGMENTUM_PASSWORD."
            )
        body = b""
        if resp.status_code >= 400:
            body = await resp.aread()
            if trace:
                trace.request_end(
                    label=label,
                    method="POST",
                    path="/api/chat",
                    status_code=resp.status_code,
                    duration_ms=int((time.monotonic() - started) * 1000),
                    response=body.decode("utf-8", errors="replace")[:1200],
                )
        assert resp.status_code < 400, (
            f"POST /api/chat returned {resp.status_code}: "
            f"{body.decode('utf-8', errors='replace')[:1200]}"
        )
        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                pytest.fail(f"Non-JSON chat stream line: {line[:500]!r} ({exc})")
            content_delta = str((chunk.get("message") or {}).get("content") or "")
            if content_delta and first_content_ms == 0:
                first_content_ms = int((time.monotonic() - started) * 1000)
            chunks.append(chunk)
            if trace:
                trace.stream_chunk(
                    label=label,
                    chunk=chunk,
                    ordinal=len(chunks),
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                )
    assert chunks, "Streaming chat returned no chunks."
    assert chunks[-1].get("done") is True, f"Final chunk was not done: {chunks[-1]!r}"
    content = "".join(
        str((chunk.get("message") or {}).get("content") or "")
        for chunk in chunks
    )
    if trace:
        trace.stream_end(
            label=label,
            status_code=200,
            duration_ms=int((time.monotonic() - started) * 1000),
            harness_ttft_ms=first_content_ms,
            chunks=chunks,
            content=content,
        )
    assert not _has_backend_error(chunks), f"Backend error chunks: {chunks!r}"
    assert content.strip(), f"Streaming chat produced no visible content: {chunks!r}"
    return content, chunks


def _normalize_json_candidate(text: str) -> str:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.I)
    if fenced:
        stripped = fenced.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def _openai_message_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    assert isinstance(choices, list) and choices, f"Missing choices: {data!r}"
    first = choices[0]
    assert isinstance(first, dict), f"Malformed first choice: {first!r}"
    message = first.get("message")
    assert isinstance(message, dict), f"Missing choice message: {first!r}"
    content = message.get("content")
    assert isinstance(content, str), f"Missing message content: {message!r}"
    return content


async def test_engine_v2_discovers_or_loads_local_model(
    live_client: httpx.AsyncClient,
    live_engine_model: LiveEngineModel,
) -> None:
    status = await _engine_status_or_skip(live_client)
    assert _ready_state(status), f"Engine is not ready: {status!r}"
    assert str(status.get("model_id") or "") == live_engine_model.model_id

    llama_status = await _request_json(
        live_client,
        "GET",
        "/api/llamacpp/status",
        trace_label="llamacpp_status",
    )
    assert isinstance(llama_status, dict)
    assert "slots" in llama_status or "health" in llama_status or "props" in llama_status

    slots = await _llama_slots(live_client, trace_label="llamacpp_slots_initial")
    assert isinstance(slots, list)


async def test_llamacpp_primitives_and_raw_completion(
    live_client: httpx.AsyncClient,
    live_engine_model: LiveEngineModel,
) -> None:
    tokenized = await _request_json(
        live_client,
        "POST",
        "/api/llamacpp/tokenize",
        json={"content": "Live llama KV test tokenization sample."},
        trace_label="llamacpp_tokenize",
    )
    assert isinstance(tokenized, dict)
    tokens = tokenized.get("tokens")
    assert isinstance(tokens, list) and tokens, f"Unexpected tokenization result: {tokenized!r}"

    detokenized = await _request_json(
        live_client,
        "POST",
        "/api/llamacpp/detokenize",
        json={"tokens": tokens[: min(len(tokens), 12)]},
        trace_label="llamacpp_detokenize",
    )
    assert isinstance(detokenized, dict)
    assert isinstance(detokenized.get("content"), str)

    completion = await _request_json(
        live_client,
        "POST",
        "/api/engine/v2/generate",
        json={
            "prompt": "Write one short word that means operational readiness:",
            "stream": False,
            "n_predict": 12,
            "temperature": 0,
            "cache_prompt": True,
        },
        timeout=_live_timeout(),
        trace_label="raw_completion",
    )
    assert isinstance(completion, dict)
    generated = _content_from_completion(completion).strip()
    assert generated, f"Raw completion returned no content for {live_engine_model.model_id}: {completion!r}"


async def test_streaming_chat_session_kv_affinity_round_trip(
    live_client: httpx.AsyncClient,
    live_engine_model: LiveEngineModel,
) -> None:
    namespace = f"live-llama-kv-{uuid.uuid4().hex[:10]}"
    session_a = f"{namespace}-a"
    session_b = f"{namespace}-b"
    system = {
        "role": "system",
        "content": "You are a concise live infrastructure test assistant. Answer briefly.",
    }

    before_slots = await _llama_slots(live_client, trace_label="session_round_trip.before")
    first_a, chunks_a1 = await _stream_ollama_chat(
        live_client,
        model=live_engine_model.chat_model,
        session_id=session_a,
        trace_label="session_a_first",
        ui_action="first_send",
        messages=[
            system,
            {
                "role": "user",
                "content": "Remember the exact live test key ALPHA-731. Reply with only OK.",
            },
        ],
    )
    _, chunks_b = await _stream_ollama_chat(
        live_client,
        model=live_engine_model.chat_model,
        session_id=session_b,
        trace_label="session_b",
        ui_action="first_send",
        messages=[
            system,
            {"role": "user", "content": "Reply with only the word OK."},
        ],
    )
    second_a, chunks_a2 = await _stream_ollama_chat(
        live_client,
        model=live_engine_model.chat_model,
        session_id=session_a,
        trace_label="session_a_return",
        ui_action="send",
        messages=[
            system,
            {
                "role": "user",
                "content": "Remember the exact live test key ALPHA-731. Reply with only OK.",
            },
            {"role": "assistant", "content": first_a},
            {
                "role": "user",
                "content": "What exact live test key did I ask you to remember? Reply with only the key.",
            },
        ],
    )

    assert "ALPHA" in second_a.upper() or "731" in second_a, (
        "The local model did not echo the simple conversation key; "
        f"response was {second_a!r}"
    )
    assert "prefill" in _stage_names(chunks_a1)
    assert "prefill" in _stage_names(chunks_b)
    assert "prefill" in _stage_names(chunks_a2)

    after_slots = await _llama_slots(live_client, trace_label="session_round_trip.after")
    assert after_slots, "llama.cpp /slots returned no slots after live chat traffic."
    assert any(_slot_has_activity(slot) for slot in after_slots), (
        f"No slot showed token/prompt activity. Before={before_slots!r}; after={after_slots!r}"
    )

    if len(after_slots) == 1:
        assert "slot_restore" in _stage_names(chunks_a2), (
            "Single-slot session round trip did not emit a slot_restore stage "
            f"when returning to session A. Stages: {_stage_names(chunks_a2)!r}"
        )


async def test_openai_json_fallback_path_preserves_engine_session(
    live_client: httpx.AsyncClient,
    live_engine_model: LiveEngineModel,
) -> None:
    session_id = f"live-llama-json-{uuid.uuid4().hex[:10]}"
    payload = {
        "model": live_engine_model.chat_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return only valid minified JSON. Do not include markdown, "
                    "commentary, or extra text."
                ),
            },
            {
                "role": "user",
                "content": 'Return exactly this JSON shape with values: {"ok":true,"path":"json-fallback"}',
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": 64,
        "stream": False,
        "think": False,
    }
    data = await _request_json(
        live_client,
        "POST",
        "/v1/chat/completions",
        json=payload,
        headers={
            MODE_HEADER: "passthrough",
            SESSION_HEADER: session_id,
            "X-Augmentum-Memory-Query": "none",
        },
        timeout=_live_timeout(),
        trace_label="openai_json_fallback",
    )
    assert isinstance(data, dict)
    content = _openai_message_content(data)
    assert content.strip(), f"OpenAI JSON fallback path returned empty content: {data!r}"
    parsed = json.loads(_normalize_json_candidate(content))
    assert isinstance(parsed, dict), f"JSON response was not an object: {content!r}"
    assert parsed.get("ok") is True


@pytest.mark.skipif(
    os.getenv("AUGMENTUM_LIVE_LLAMA_RELOAD", "").strip().lower()
    not in {"1", "true", "yes", "on"},
    reason="set AUGMENTUM_LIVE_LLAMA_RELOAD=1 to unload/reload the live engine",
)
async def test_reload_reuses_persisted_session_without_error(
    live_client: httpx.AsyncClient,
    live_engine_model: LiveEngineModel,
) -> None:
    session_id = f"live-llama-reload-{uuid.uuid4().hex[:10]}"
    messages = [
        {
            "role": "system",
            "content": "You are a concise live reload test assistant.",
        },
        {
            "role": "user",
            "content": "Remember the exact reload key BRAVO-294 and reply OK.",
        },
    ]
    await _stream_ollama_chat(
        live_client,
        model=live_engine_model.chat_model,
        session_id=session_id,
        messages=messages,
        trace_label="reload_before_unload",
        ui_action="first_send",
    )

    await _request_json(
        live_client,
        "POST",
        "/api/engine/v2/models/unload",
        timeout=_load_timeout(),
        trace_label="engine_model_unload",
    )
    reloaded = await _ensure_model_loaded(
        live_client,
        ModelTarget(ref=live_engine_model.load_ref, source=live_engine_model.target_source),
    )
    content, chunks = await _stream_ollama_chat(
        live_client,
        model=reloaded.chat_model,
        session_id=session_id,
        trace_label="reload_after_load_return",
        ui_action="send",
        messages=[
            *messages,
            {"role": "assistant", "content": "OK"},
            {
                "role": "user",
                "content": "What exact reload key did I ask you to remember? Reply with only the key.",
            },
        ],
    )
    assert "BRAVO" in content.upper() or "294" in content
    assert not _has_backend_error(chunks)
