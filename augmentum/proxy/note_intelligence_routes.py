from __future__ import annotations

import json
import re

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from augmentum.config import settings
from augmentum.models.base import InternalChatRequest, Message

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/notes", tags=["note-intelligence"])


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ScanRequest(BaseModel):
    content: str
    session_id: str = ""


class AIActionRequest(BaseModel):
    action: str
    selected_text: str
    context: str = ""
    option: str = ""


# ---------------------------------------------------------------------------
# Action prompt map
# ---------------------------------------------------------------------------

_ACTION_PROMPTS: dict[str, str] = {
    "rewrite": "Rewrite the following text, improving clarity and flow while preserving meaning:\n\n",
    "expand": "Expand the following text with more detail, examples, and depth:\n\n",
    "compress": "Condense the following text to be more concise while preserving all key information:\n\n",
    "research": "Research and provide factual information about:\n\n",
    "define": "Define and explain:\n\n",
}


# ---------------------------------------------------------------------------
# POST /{note_id}/scan — margin intelligence annotations
# ---------------------------------------------------------------------------

@router.post("/{note_id}/scan")
async def scan_note(note_id: str, body: ScanRequest, request: Request):
    """Scan note content for margin intelligence annotations."""
    annotations: list[dict] = []
    content = body.content

    # Source 1: Lorebook entries
    try:
        lore_engine = getattr(request.app.state, "lore_engine_global", None)
        if lore_engine is not None:
            triggered = lore_engine.scan_and_trigger([content])
            count = 0
            for entry in triggered:
                if count >= 5:
                    break
                keywords = entry.get("keys", [])
                for kw in keywords:
                    if count >= 5:
                        break
                    pattern = re.compile(re.escape(kw), re.IGNORECASE)
                    match = pattern.search(content)
                    if match:
                        annotations.append({
                            "term": kw,
                            "start": match.start(),
                            "end": match.end(),
                            "source": "lorebook",
                            "type": "lorebook",
                            "content": entry.get("content", "")[:300],
                            "entry_id": entry.get("id", ""),
                        })
                        count += 1
    except Exception as exc:
        log.warning("note_scan_lorebook_failed", error=str(exc))

    # Source 2: Related notes (title appears in content)
    try:
        notes_store = getattr(request.app.state, "notes_store", None)
        if notes_store is not None:
            stubs = await notes_store.list_stubs(user_id=_user_id(request))
            count = 0
            for stub in stubs:
                if count >= 5:
                    break
                # Skip self
                if str(stub.get("id", "")) == note_id:
                    continue
                title = stub.get("title", "")
                if not title or len(title) < 3:
                    continue
                pattern = re.compile(re.escape(title), re.IGNORECASE)
                match = pattern.search(content)
                if match:
                    annotations.append({
                        "term": title,
                        "start": match.start(),
                        "end": match.end(),
                        "source": "related_note",
                        "type": "related_note",
                        "content": stub.get("preview", ""),
                        "entry_id": str(stub.get("id", "")),
                    })
                    count += 1
    except Exception as exc:
        log.warning("note_scan_related_notes_failed", error=str(exc))

    return {"annotations": annotations}


# ---------------------------------------------------------------------------
# GET /tags — unique tag autocomplete list
# ---------------------------------------------------------------------------

@router.get("/tags")
async def list_tags(request: Request):
    """List all unique tags across all notes for autocomplete."""
    all_tags: set[str] = set()
    try:
        notes_store = getattr(request.app.state, "notes_store", None)
        if notes_store is not None:
            stubs = await notes_store.list_stubs(user_id=_user_id(request))
            for stub in stubs:
                raw_tags = stub.get("tags", [])
                # Handle tags stored as JSON string or list
                if isinstance(raw_tags, str):
                    try:
                        raw_tags = json.loads(raw_tags)
                    except (json.JSONDecodeError, TypeError):
                        raw_tags = []
                if isinstance(raw_tags, list):
                    for tag in raw_tags:
                        if isinstance(tag, str) and tag.strip():
                            all_tags.add(tag.strip())
    except Exception as exc:
        log.warning("note_tags_list_failed", error=str(exc))

    return {"tags": sorted(all_tags)}


# ---------------------------------------------------------------------------
# POST /{note_id}/ai — AI action on selected text
# ---------------------------------------------------------------------------

@router.post("/{note_id}/ai")
async def ai_action(note_id: str, body: AIActionRequest, request: Request):
    """Execute an AI action (rewrite, expand, etc.) on selected text."""
    action = body.action
    if action not in _ACTION_PROMPTS:
        return JSONResponse(
            {"error": f"Unknown action: {action}. Valid: {', '.join(_ACTION_PROMPTS)}"},
            status_code=400,
        )

    # Resolve backend
    registry = getattr(request.app.state, "provider_registry", None)
    if registry is None:
        return JSONResponse({"error": "No provider registry available"}, status_code=503)

    try:
        backend, resolved_model = await registry.resolve_model_for_role("utility", settings=settings)
    except Exception:
        backend, resolved_model = None, ""
    if backend is None:
        return JSONResponse({"error": "No LLM backend available"}, status_code=503)

    prompt_prefix = _ACTION_PROMPTS[action]
    user_content = prompt_prefix + body.selected_text
    if body.context:
        user_content += f"\n\nContext:\n{body.context}"
    if body.option:
        user_content += f"\n\nAdditional instruction: {body.option}"

    chat_request = InternalChatRequest(
        model=resolved_model,
        messages=[
            Message(role="system", content="You are a writing assistant. Respond only with the requested text — no preamble, no explanation."),
            Message(role="user", content=user_content),
        ],
        stream=False,
        temperature=0.4,
        max_tokens=2048,
    )

    try:
        resp = await backend.chat(chat_request)
        result = resp.message.content.strip()
        return {"result": result}
    except Exception as exc:
        log.warning("note_ai_action_failed", action=action, error=str(exc))
        return JSONResponse({"error": f"AI action failed: {exc}"}, status_code=502)
