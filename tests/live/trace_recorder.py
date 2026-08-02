"""Opt-in redacted trace artifacts for live Augmentum tests.

The recorder is deliberately small and dependency-free. It is built for local
dogfood/live-test runs where we want an audit trail of model routing, slots,
TTFT, token rates, and stage timing without storing prompt text by default.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TRUTHY = {"1", "true", "yes", "on"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    cleaned = cleaned.strip("._-")
    return cleaned[:96] or "event"


def _sha(value: Any, *, prefix: str = "sha") -> str:
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        raw = value.encode("utf-8", errors="replace")
    else:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:16]}"


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUTHY


class LiveTraceRecorder:
    """Write redacted JSONL + summary artifacts for live engine testing."""

    schema_version = 1

    def __init__(
        self,
        *,
        enabled: bool,
        trace_dir: Path | None = None,
        content_mode: str = "redacted",
    ) -> None:
        self.enabled = enabled
        self.content_mode = content_mode if content_mode in {"redacted", "sample", "full"} else "redacted"
        self.run_id = self._make_run_id()
        self.root = trace_dir / self.run_id if enabled and trace_dir else None
        self.requests_dir = self.root / "requests" if self.root else None
        self._trace_file = None
        self._started_at = _utc_now()
        self._finished = False
        self._context: dict[str, Any] = {}
        self._scenarios: dict[str, dict[str, Any]] = {}
        self._warnings: list[str] = []
        self._request_counter = 0
        self._request_paths: dict[str, Path] = {}
        if self.enabled and self.root and self.requests_dir:
            self.requests_dir.mkdir(parents=True, exist_ok=True)
            self._trace_file = (self.root / "trace.jsonl").open("a", encoding="utf-8")
            self.event("run_start")

    @classmethod
    def from_env(cls) -> LiveTraceRecorder:
        enabled = _truthy_env("AUGMENTUM_LIVE_TRACE")
        base = Path(os.getenv("AUGMENTUM_LIVE_TRACE_DIR", "tests/live/artifacts/llama-kv"))
        content_mode = os.getenv("AUGMENTUM_LIVE_TRACE_CONTENT", "redacted").strip().lower()
        return cls(enabled=enabled, trace_dir=base, content_mode=content_mode)

    @staticmethod
    def _make_run_id() -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        return f"llama-kv-{stamp}-{hashlib.sha1(str(time.time_ns()).encode()).hexdigest()[:6]}"

    def set_context(self, **kwargs: Any) -> None:
        if not self.enabled:
            return
        self._context.update({k: v for k, v in kwargs.items() if v not in (None, "")})

    def event(self, event_type: str, **payload: Any) -> None:
        if not self.enabled or self._trace_file is None:
            return
        record = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "type": event_type,
            "at": _utc_now(),
            **payload,
        }
        self._trace_file.write(json.dumps(record, sort_keys=True, ensure_ascii=True, default=str) + "\n")
        self._trace_file.flush()

    def request_start(
        self,
        *,
        label: str,
        method: str,
        path: str,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.event(
            "http_request",
            label=label,
            method=method,
            path=path,
            request=self._redact_body(json_body),
            headers=self._redact_headers(headers or {}),
        )

    def request_end(
        self,
        *,
        label: str,
        method: str,
        path: str,
        status_code: int,
        duration_ms: int,
        response: Any = None,
    ) -> None:
        response_meta = self._redact_body(response)
        self.event(
            "http_response",
            label=label,
            method=method,
            path=path,
            status=status_code,
            duration_ms=duration_ms,
            response=response_meta,
        )
        scenario = self._scenario(label)
        scenario.update(
            {
                "name": label,
                "route": path,
                "success": status_code < 400,
                "status_code": status_code,
                "total_duration_ms": duration_ms,
            }
        )
        self._merge_response_metrics(scenario, response_meta)

    def stream_start(
        self,
        *,
        label: str,
        path: str,
        model: str,
        session_id: str,
        messages: list[dict[str, Any]],
        headers: dict[str, str],
        ui_action: str = "",
    ) -> None:
        session_hash = _sha(session_id, prefix="ses") if session_id else ""
        request = {
            "model": model,
            "session_hash": session_hash,
            "message_count": len(messages),
            "messages_hash": self.messages_hash(messages),
            "messages": [self._redact_message(m) for m in messages],
        }
        if ui_action:
            request["ui_action"] = ui_action
        self.event(
            "http_request",
            label=label,
            method="POST",
            path=path,
            request=request,
            headers=self._redact_headers(headers),
        )
        scenario = self._scenario(label)
        scenario.update(
            {
                "name": label,
                "route": path,
                "model": model,
                "session_hash": session_hash,
                "ui_action": ui_action,
                "request_prompt_fingerprint": request["messages_hash"],
                "message_count": len(messages),
            }
        )

    def stream_chunk(
        self,
        *,
        label: str,
        chunk: dict[str, Any],
        ordinal: int,
        elapsed_ms: int,
    ) -> None:
        self._append_request_chunk(label, self._redact_stream_chunk(chunk, ordinal, elapsed_ms))
        aug = chunk.get("augmentum")
        if not isinstance(aug, dict):
            return
        for key in ("stage_start", "stage_complete", "stage_progress"):
            event = aug.get(key)
            if isinstance(event, dict):
                self.event(
                    key,
                    label=label,
                    elapsed_ms=elapsed_ms,
                    **self._redact_stage(event),
                )
        if aug.get("backend_error"):
            self.event(
                "backend_error",
                label=label,
                elapsed_ms=elapsed_ms,
                error_kind=aug.get("error_kind", ""),
                retryable=bool(aug.get("retryable")),
            )

    def stream_end(
        self,
        *,
        label: str,
        status_code: int,
        duration_ms: int,
        harness_ttft_ms: int,
        chunks: list[dict[str, Any]],
        content: str,
    ) -> None:
        final = chunks[-1] if chunks else {}
        metrics = self._stream_metrics(final, harness_ttft_ms)
        stage_events = self.stage_events(chunks)
        kv_verdict = self.kv_verdict(chunks)
        token_path = "completion" if any("prompt_eval_count" in c or "eval_count" in c for c in chunks) else "unknown"
        response = {
            "content_len": len(content),
            "content_hash": _sha(content, prefix="txt") if content else "",
            "chunk_count": len(chunks),
            "stage_events": stage_events,
            "kv_verdict": kv_verdict,
            "token_path": token_path,
            **metrics,
        }
        if self.content_mode == "sample" and content:
            response["content_sample"] = content[:160]
        elif self.content_mode == "full" and content:
            response["content"] = content
        self.event(
            "http_response",
            label=label,
            method="POST",
            path="/api/chat",
            status=status_code,
            duration_ms=duration_ms,
            response=response,
        )
        scenario = self._scenario(label)
        scenario.update(
            {
                "name": label,
                "route": "/api/chat",
                "success": status_code < 400 and not self.has_backend_error(chunks),
                "status_code": status_code,
                "total_duration_ms": duration_ms,
                "backend_error": self.backend_error(chunks),
                **response,
            }
        )
        self._add_slow_warnings(label, scenario)

    def slot_snapshot(self, label: str, slots: list[dict[str, Any]]) -> None:
        normalized = [self.normalize_slot(slot) for slot in slots]
        self.event("slot_snapshot", label=label, slots=normalized)

    def warning(self, message: str) -> None:
        if not self.enabled:
            return
        self._warnings.append(message)
        self.event("warning", message=message)

    def finish(self, status: str = "complete") -> None:
        if not self.enabled or self._finished:
            return
        self._finished = True
        summary = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "started_at": self._started_at,
            "finished_at": _utc_now(),
            "status": status,
            **self._context,
            "scenarios": list(self._scenarios.values()),
            "warnings": self._warnings,
        }
        if self.root:
            (self.root / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True, default=str),
                encoding="utf-8",
            )
        self.event("run_end", status=status)
        if self._trace_file is not None:
            self._trace_file.close()
            self._trace_file = None

    def _scenario(self, label: str) -> dict[str, Any]:
        return self._scenarios.setdefault(label, {"name": label})

    def _append_request_chunk(self, label: str, chunk: dict[str, Any]) -> None:
        if not self.enabled or not self.requests_dir:
            return
        path = self._request_paths.get(label)
        if path is None:
            self._request_counter += 1
            path = self.requests_dir / f"{self._request_counter:04d}_{_slug(label)}.ndjson"
            self._request_paths[label] = path
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(chunk, sort_keys=True, ensure_ascii=True, default=str) + "\n")

    def _redact_headers(self, headers: dict[str, str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in headers.items():
            lower = key.lower()
            if lower in {"authorization", "cookie"}:
                result[key] = "<redacted>"
            elif lower == "x-augmentum-session":
                result[key] = _sha(value, prefix="ses") if value else ""
            else:
                result[key] = value
        return result

    def _redact_body(self, body: Any) -> Any:
        if body is None:
            return None
        if isinstance(body, list):
            return {
                "type": "list",
                "count": len(body),
                "items_hash": _sha(body, prefix="list"),
            }
        if not isinstance(body, dict):
            if isinstance(body, str):
                return self._redact_text(body)
            return body
        result: dict[str, Any] = {"keys": sorted(str(k) for k in body.keys())}
        for key in ("model", "stream", "response_format", "temperature", "max_tokens", "n_predict"):
            if key in body:
                result[key] = body[key]
        messages = body.get("messages")
        if isinstance(messages, list):
            msg_dicts = [m for m in messages if isinstance(m, dict)]
            result["message_count"] = len(msg_dicts)
            result["messages_hash"] = self.messages_hash(msg_dicts)
            result["messages"] = [self._redact_message(m) for m in msg_dicts]
        content = self._extract_content(body)
        if content:
            result["content"] = self._redact_text(content)
        choices = body.get("choices")
        if isinstance(choices, list):
            result["choice_count"] = len(choices)
            first_content = ""
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    first_content = msg["content"]
            if first_content:
                result["first_choice_content"] = self._redact_text(first_content)
        usage = body.get("usage")
        if isinstance(usage, dict):
            result["usage"] = {
                key: value
                for key, value in usage.items()
                if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
            }
        for key in ("prompt_eval_count", "eval_count", "total_duration_ms", "ttft_ms"):
            if key in body:
                result[key] = body[key]
        return result

    def _redact_message(self, message: dict[str, Any]) -> dict[str, Any]:
        content = message.get("content")
        text = content if isinstance(content, str) else json.dumps(content, sort_keys=True, default=str)
        result = {
            "role": message.get("role", ""),
            "content_len": len(text),
            "content_hash": _sha(text, prefix="msg"),
        }
        images = message.get("images")
        if isinstance(images, list):
            result["image_count"] = len(images)
        if self.content_mode == "sample" and text:
            result["content_sample"] = text[:120]
        elif self.content_mode == "full" and text:
            result["content"] = text
        return result

    def _redact_text(self, text: str) -> dict[str, Any]:
        result = {"len": len(text), "hash": _sha(text, prefix="txt")}
        if self.content_mode == "sample":
            result["sample"] = text[:160]
        elif self.content_mode == "full":
            result["text"] = text
        return result

    @staticmethod
    def _extract_content(body: dict[str, Any]) -> str:
        for key in ("content", "response", "text"):
            value = body.get(key)
            if isinstance(value, str):
                return value
        return ""

    def _redact_stream_chunk(self, chunk: dict[str, Any], ordinal: int, elapsed_ms: int) -> dict[str, Any]:
        msg = chunk.get("message")
        content = ""
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            content = msg["content"]
        response = chunk.get("response")
        if not content and isinstance(response, str):
            content = response
        result: dict[str, Any] = {
            "ordinal": ordinal,
            "elapsed_ms": elapsed_ms,
            "done": bool(chunk.get("done")),
            "content_len": len(content),
        }
        if content:
            result["content_hash"] = _sha(content, prefix="tok")
            if self.content_mode == "full":
                result["content"] = content
        aug = chunk.get("augmentum")
        if isinstance(aug, dict):
            result["augmentum"] = self._redact_augmentum(aug)
        for key in ("prompt_eval_count", "eval_count", "total_duration", "eval_duration"):
            if key in chunk:
                result[key] = chunk[key]
        return result

    def _redact_augmentum(self, aug: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in aug.items():
            if key in {
                "stage_start",
                "stage_complete",
                "stage_progress",
                "status",
                "heartbeat",
                "phase",
                "phase_status",
                "tokens_per_second",
                "prompt_tokens",
                "eval_tokens",
                "context_length",
                "context_used",
                "ttft_ms",
                "total_duration_ms",
                "eval_duration_ms",
                "backend_error",
                "error_kind",
                "retryable",
            }:
                if key.startswith("stage_") and isinstance(value, dict):
                    safe[key] = self._redact_stage(value)
                elif key == "backend_error":
                    safe[key] = "<redacted>"
                else:
                    safe[key] = value
        return safe

    @staticmethod
    def _redact_stage(event: dict[str, Any]) -> dict[str, Any]:
        keep = {
            "id",
            "stage",
            "label",
            "detail",
            "started_at",
            "success",
            "duration_ms",
            "error",
            "request_id",
            "percent",
            "message",
        }
        return {key: value for key, value in event.items() if key in keep}

    @staticmethod
    def messages_hash(messages: list[dict[str, Any]]) -> str:
        compact = []
        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, sort_keys=True, ensure_ascii=True, default=str)
            compact.append(
                {
                    "role": message.get("role", ""),
                    "content_hash": _sha(content, prefix="msg"),
                    "content_len": len(content),
                    "image_count": len(message.get("images") or []),
                }
            )
        return _sha(compact, prefix="prompt")

    @staticmethod
    def normalize_slot(slot: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key in ("id", "id_slot", "is_processing", "id_task", "n_ctx", "n_past"):
            if key in slot:
                result[key] = slot[key]
        prompt = slot.get("prompt")
        if isinstance(prompt, list):
            result["prompt_token_count"] = len(prompt)
            result["prompt_hash"] = _sha(prompt, prefix="slot_prompt")
        elif isinstance(prompt, str):
            result["prompt_text_len"] = len(prompt)
            result["prompt_hash"] = _sha(prompt, prefix="slot_prompt")
        params = slot.get("params")
        if isinstance(params, dict):
            result["params"] = {
                key: params.get(key)
                for key in ("stream", "n_predict", "cache_prompt", "id_slot")
                if key in params
            }
        next_token = slot.get("next_token")
        if isinstance(next_token, list):
            next_meta = []
            for item in next_token:
                if isinstance(item, dict):
                    next_meta.append(
                        {
                            key: item.get(key)
                            for key in ("n_decoded", "n_remain")
                            if key in item
                        }
                    )
            result["next_token"] = next_meta
        return result

    @staticmethod
    def stage_events(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for chunk in chunks:
            aug = chunk.get("augmentum")
            if not isinstance(aug, dict):
                continue
            for key in ("stage_start", "stage_complete"):
                event = aug.get(key)
                if isinstance(event, dict):
                    events.append({"type": key, **LiveTraceRecorder._redact_stage(event)})
        return events

    @staticmethod
    def kv_verdict(chunks: list[dict[str, Any]]) -> str:
        saw_restore_start = False
        for chunk in chunks:
            aug = chunk.get("augmentum")
            if not isinstance(aug, dict):
                continue
            tier = aug.get("kv_tier")
            if tier == "hot":
                return "hot"
            if tier == "cold_with_checkpoint":
                return "restored"
            if tier == "cold_no_checkpoint":
                return "cold"
            start = aug.get("stage_start")
            if isinstance(start, dict) and start.get("stage") == "slot_restore":
                saw_restore_start = True
            complete = aug.get("stage_complete")
            if (
                isinstance(complete, dict)
                and complete.get("stage") == "slot_restore"
                and complete.get("success") is not False
            ):
                return "restored"
        return "restore_attempted" if saw_restore_start else "unknown"

    @staticmethod
    def has_backend_error(chunks: list[dict[str, Any]]) -> bool:
        return bool(LiveTraceRecorder.backend_error(chunks))

    @staticmethod
    def backend_error(chunks: list[dict[str, Any]]) -> str:
        for chunk in chunks:
            aug = chunk.get("augmentum")
            if isinstance(aug, dict) and aug.get("backend_error"):
                return str(aug.get("error_kind") or "backend_error")
        return ""

    @staticmethod
    def _stream_metrics(final_chunk: dict[str, Any], harness_ttft_ms: int) -> dict[str, Any]:
        aug = final_chunk.get("augmentum")
        if not isinstance(aug, dict):
            aug = {}
        metrics: dict[str, Any] = {
            "ttft_ms": int(aug.get("ttft_ms") or harness_ttft_ms or 0),
            "harness_ttft_ms": int(harness_ttft_ms or 0),
        }
        for key in (
            "tokens_per_second",
            "prompt_tokens",
            "eval_tokens",
            "context_length",
            "context_used",
            "total_duration_ms",
            "eval_duration_ms",
        ):
            if key in aug:
                metrics[key] = aug[key]
        if "prompt_tokens" not in metrics and "prompt_eval_count" in final_chunk:
            metrics["prompt_tokens"] = final_chunk["prompt_eval_count"]
        if "eval_tokens" not in metrics and "eval_count" in final_chunk:
            metrics["eval_tokens"] = final_chunk["eval_count"]
        return metrics

    @staticmethod
    def _merge_response_metrics(scenario: dict[str, Any], response_meta: Any) -> None:
        if not isinstance(response_meta, dict):
            return
        for key in ("prompt_eval_count", "eval_count", "total_duration_ms", "ttft_ms"):
            if key in response_meta:
                scenario[key] = response_meta[key]
        usage = response_meta.get("usage")
        if isinstance(usage, dict):
            if "prompt_tokens" in usage:
                scenario["prompt_tokens"] = usage["prompt_tokens"]
            if "completion_tokens" in usage:
                scenario["eval_tokens"] = usage["completion_tokens"]

    def _add_slow_warnings(self, label: str, scenario: dict[str, Any]) -> None:
        ttft = int(scenario.get("ttft_ms") or 0)
        kv_verdict = str(scenario.get("kv_verdict") or "")
        prompt_tokens = int(scenario.get("prompt_tokens") or 0)
        if kv_verdict == "cold" and prompt_tokens > 512:
            self.warning(f"{label}: established/large request appears cold ({prompt_tokens} prompt tokens)")
        if ttft > 5000 and kv_verdict in {"unknown", "cold"}:
            self.warning(f"{label}: TTFT {ttft}ms with kv_verdict={kv_verdict}")
